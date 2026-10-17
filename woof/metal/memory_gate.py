"""Minimal device-budget preflight for Metal forecasts.

The engine and resident bytes per column-level are measured from process
footprints of two HRRR cuts (101x81 and 151x113, 50 levels). The legacy
radiation workspace follows its device-sized chunk. These are workload
coefficients, not hardware memory limits; admission uses the device's
recommended working set. WOOF_METAL_MEMORY_FRACTION overrides the share
available to one forecast. --no-memory-gate bypasses admission on the
regional forecast paths; the Hex door always applies its own admission.
Selected RUC slabs use their packed allocation inventory. Ensembles include
their retained host product planes because Metal shares physical memory.
The Hex door prices its own cell/level and column-batch inventory.
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from functools import lru_cache

FIXED_ENGINE_BYTES = 1385 * 2**20
PER_COLUMN_LEVEL_BYTES = 880
# SPPT/SKEBS act on host copies of the four rates: the copies, the SKEBS and
# SPPT results, the coupled mass factors and their device faces peak at
# twenty FP32 volumes.  An SPP PBL pattern adds its host and device levels
# plus the previous step's bound copy.
STOCHASTIC_RATE_COLUMN_LEVEL_BYTES = 20 * 4
SPP_PBL_COLUMN_LEVEL_BYTES = 3 * 4
DEFAULT_FRACTION = 0.65
FRACTION_ENV = "WOOF_METAL_MEMORY_FRACTION"


@dataclass(frozen=True)
class MetalMemoryEstimate:
    peak_bytes: int
    limit_bytes: int
    working_set_bytes: int
    fraction: float
    columns: int

    @property
    def fits_device(self) -> bool:
        return self.peak_bytes <= self.limit_bytes


def _fraction() -> float:
    raw = os.environ.get(FRACTION_ENV, "").strip()
    if not raw:
        return DEFAULT_FRACTION
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{FRACTION_ENV} must be a number in (0, 1]") from None
    if not 0.0 < value <= 1.0:
        raise ValueError(f"{FRACTION_ENV} must be in (0, 1], got {raw!r}")
    return value


@lru_cache(maxsize=1)
def _ruc_table_inventory():
    from woof.core.ruc_runtime import RucRuntimeParameters
    from woof.metal.ruc_metal_runtime import _parameter_tables

    params = RucRuntimeParameters()
    _, tables = _parameter_tables(params)
    return (sum(array.size for array in tables.values()),
            max(len(section.rows) for section in params.bundle.vegetation.values()),
            len(params.bundle.soil.rows))


def estimate(domains, context, *, cumulus=None, spp_conv=None,
             land_surface=None, legacy_radiation=None, mosaic_lu=None,
             mosaic_soil=None, spp_pbl=None,
             stochastic_rates: bool = False) -> MetalMemoryEstimate:
    """Price residents, selected radiation/RUC storage and bounded GF scratch."""
    from woof.core.rrtmg_legacy import (
        metal_rrtmg_bytes_per_column, metal_rrtmg_column_chunk)

    domains = list(domains)
    cumulus = [0] * len(domains) if cumulus is None else list(cumulus)
    spp_conv = [0] * len(domains) if spp_conv is None else list(spp_conv)
    land_surface = [0] * len(domains) if land_surface is None else list(land_surface)
    legacy_radiation = ([True] * len(domains) if legacy_radiation is None
                        else list(legacy_radiation))
    mosaic_lu = [False] * len(domains) if mosaic_lu is None else list(mosaic_lu)
    mosaic_soil = [False] * len(domains) if mosaic_soil is None else list(mosaic_soil)
    spp_pbl = [0] * len(domains) if spp_pbl is None else list(spp_pbl)
    if any(len(values) != len(domains) for values in
           (cumulus, spp_conv, land_surface, legacy_radiation, mosaic_lu, mosaic_soil,
            spp_pbl)):
        raise ValueError("Metal memory physics selectors must match the domain count")
    working_set = int(context.recommended_working_set_size)
    resident = radiation = columns = gf_scratch = ruc_scratch = 0
    for (nx, ny, nz), scheme, stochastic, lsm, legacy, lu, soil, pbl in zip(
            domains, cumulus, spp_conv, land_surface, legacy_radiation,
            mosaic_lu, mosaic_soil, spp_pbl):
        ncol = int(nx) * int(ny)
        columns += ncol
        resident += PER_COLUMN_LEVEL_BYTES * int(nz) * ncol
        resident += (STOCHASTIC_RATE_COLUMN_LEVEL_BYTES * bool(stochastic_rates)
                     + SPP_PBL_COLUMN_LEVEL_BYTES * bool(int(pbl))) * int(nz) * ncol
        if legacy:
            chunk = metal_rrtmg_column_chunk(context, int(nz), ncol)
            radiation = max(radiation, chunk * metal_rrtmg_bytes_per_column(nz))
        if int(lsm) == 3:
            from woof.metal.ruc_metal_workspace import ruc_workspace_bytes
            table_words, land_categories, soil_categories = _ruc_table_inventory()
            table_words += ncol * (land_categories * bool(lu) + soil_categories * bool(soil))
            ruc_scratch += ruc_workspace_bytes(ncol, table_words=table_words)
        if int(scheme) == 3:
            from woof.metal.gf_family import gf_metal_scratch_bytes
            gf_scratch = max(gf_scratch, gf_metal_scratch_bytes(
                int(nz), ncol, spp_enabled=bool(stochastic)))
    fraction = _fraction()
    return MetalMemoryEstimate(
        peak_bytes=FIXED_ENGINE_BYTES + resident + radiation + gf_scratch + ruc_scratch,
        limit_bytes=int(fraction * working_set), working_set_bytes=working_set,
        fraction=fraction, columns=columns)


def estimate_runs(runs, context, *, stochastic_rates: bool = False) -> MetalMemoryEstimate:
    """Resolve the selected physics for each live regional domain.

    ``stochastic_rates`` prices SPPT/SKEBS, which no RunConfig field names.
    """
    runs = list(runs)
    return estimate([(int(run.nx), int(run.ny), int(run.nz)) for run in runs], context,
        stochastic_rates=stochastic_rates,
        spp_pbl=[getattr(run, "spp_pbl", 0) for run in runs],
        cumulus=[run.cu_physics for run in runs], spp_conv=[run.spp_conv for run in runs],
        land_surface=[run.sf_surface_physics for run in runs],
        legacy_radiation=[run.ra_rrtmg_variant == "rrtmg_legacy"
                          and (run.ra_lw_physics == 4 or run.ra_sw_physics == 4)
                          for run in runs],
        mosaic_lu=[run.mosaic_lu for run in runs],
        mosaic_soil=[run.mosaic_soil for run in runs])


def estimate_ensemble(experiments, context) -> MetalMemoryEstimate:
    """Serial members plus the collector's retained unified-memory planes."""
    from dataclasses import replace

    experiments = list(experiments)
    if not experiments:
        raise ValueError("Metal ensemble memory admission needs its member experiments")
    forecasts = [estimate_runs([domain.run for domain in exp.domains], context)
                 for exp in experiments]
    retained = coordinates = product_columns = 0
    for exp in experiments:
        per_member_coords = 0
        for domain in exp.domains:
            columns = int(domain.run.nx) * int(domain.run.ny)
            end = min(float(exp.run_seconds), float(domain.history_end_s)
                      if domain.history_end_s is not None else float(exp.run_seconds))
            begin = float(domain.history_begin_s)
            frames = max(0, math.floor((end - begin) / float(domain.history_interval_s)) + 1)
            # Three product planes plus an initial-rain plane per episode.
            # A moving nest may start a new episode at every saved frame.
            retained += columns * 4 * frames * 4
            per_member_coords += columns * frames * 2 * 4
            product_columns = max(product_columns, columns)
        coordinates = max(coordinates, per_member_coords)
    # finish_run stacks one frame and computes float64 mean/spread. It
    # runs after the last member, with only one forecast's live residency.
    reduction = product_columns * (12 * len(experiments) + 24)
    largest = max(forecasts, key=lambda row: row.peak_bytes)
    return replace(largest, peak_bytes=largest.peak_bytes + retained + coordinates + reduction)


# Hex (MPAS v8.4.1) physics door. The resident share grows with cells and
# levels (HEX_CELL_LEVEL_BYTES per cell-level). The column-batched physics
# transient is HEX_BATCH_COLUMN_BYTES_55 per column at 55 levels, scaled
# linearly with level count, and saturates at HEX_BATCH_COLUMNS columns.
# HEX_FIXED_BYTES covers the fixed engine allocations.
HEX_CELL_LEVEL_BYTES = 1289
HEX_BATCH_COLUMN_BYTES_55 = 494_400
HEX_BATCH_COLUMNS = 2562
HEX_FIXED_BYTES = 64 * 2**20


def estimate_hex(n_cells: int, n_levels: int, context) -> MetalMemoryEstimate:
    """Price one Metal Hex physics forecast before it allocates."""
    n_cells, n_levels = int(n_cells), int(n_levels)
    working_set = int(context.recommended_working_set_size)
    transient = (HEX_BATCH_COLUMN_BYTES_55 * n_levels * min(n_cells, HEX_BATCH_COLUMNS)
                 // 55)
    fraction = _fraction()
    return MetalMemoryEstimate(
        peak_bytes=HEX_FIXED_BYTES + HEX_CELL_LEVEL_BYTES * n_cells * n_levels + transient,
        limit_bytes=int(fraction * working_set), working_set_bytes=working_set,
        fraction=fraction, columns=n_cells)


def refusal_text(result: MetalMemoryEstimate, domains, *, ensemble_members=None) -> str | None:
    if result.fits_device:
        return None
    shape = ", ".join(f"{nx}x{ny}x{nz}" for nx, ny, nz in domains)
    subject = ("Metal forecast" if ensemble_members is None
               else f"Metal ensemble with {ensemble_members} serial members")
    remedy = ("Use a smaller domain or a device with more memory."
              if ensemble_members is None else
              "Use smaller domains, fewer members or fewer history frames.")
    return (
        f"{subject} ({shape}) needs an estimated {result.peak_bytes / 2**30:.1f} GiB; "
        f"the device budget is {result.limit_bytes / 2**30:.1f} GiB "
        f"({result.fraction:.0%} of its {result.working_set_bytes / 2**30:.1f} GiB "
        f"recommended working set). {remedy} "
        f"Override with {FRACTION_ENV} or --no-memory-gate.")
