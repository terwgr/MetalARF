"""Minimal device-budget preflight for Metal forecasts.

The engine and resident bytes per column-level are measured from process
footprints of two HRRR cuts (101x81 and 151x113, 50 levels). The legacy
radiation workspace follows its device-sized chunk. These are workload
coefficients, not hardware memory limits; admission uses the device's
recommended working set. WOOF_METAL_MEMORY_FRACTION overrides the share
available to one forecast; --no-memory-gate bypasses admission.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

FIXED_ENGINE_BYTES = 1385 * 2**20
PER_COLUMN_LEVEL_BYTES = 880
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


def estimate(domains, context, *, cumulus=None, spp_conv=None) -> MetalMemoryEstimate:
    """Add resident costs, radiation, and the selected bounded GF scratch."""
    from woof.core.rrtmg_legacy import (
        metal_rrtmg_bytes_per_column, metal_rrtmg_column_chunk)

    domains = list(domains)
    cumulus = [0] * len(domains) if cumulus is None else list(cumulus)
    spp_conv = [0] * len(domains) if spp_conv is None else list(spp_conv)
    if len(cumulus) != len(domains) or len(spp_conv) != len(domains):
        raise ValueError("Metal memory physics selectors must match the domain count")
    working_set = int(context.recommended_working_set_size)
    resident = radiation = columns = gf_scratch = 0
    for (nx, ny, nz), scheme, stochastic in zip(domains, cumulus, spp_conv):
        ncol = int(nx) * int(ny)
        columns += ncol
        resident += PER_COLUMN_LEVEL_BYTES * int(nz) * ncol
        chunk = metal_rrtmg_column_chunk(context, int(nz), ncol)
        radiation = max(radiation, chunk * metal_rrtmg_bytes_per_column(nz))
        if int(scheme) == 3:
            from woof.metal.gf_family import gf_metal_scratch_bytes
            gf_scratch = max(gf_scratch, gf_metal_scratch_bytes(
                int(nz), ncol, spp_enabled=bool(stochastic)))
    fraction = _fraction()
    return MetalMemoryEstimate(
        peak_bytes=FIXED_ENGINE_BYTES + resident + radiation + gf_scratch,
        limit_bytes=int(fraction * working_set), working_set_bytes=working_set,
        fraction=fraction, columns=columns)


def refusal_text(result: MetalMemoryEstimate, domains) -> str | None:
    if result.fits_device:
        return None
    shape = ", ".join(f"{nx}x{ny}x{nz}" for nx, ny, nz in domains)
    return (
        f"Metal forecast {shape} needs an estimated {result.peak_bytes / 2**30:.1f} GiB; "
        f"the device budget is {result.limit_bytes / 2**30:.1f} GiB "
        f"({result.fraction:.0%} of its {result.working_set_bytes / 2**30:.1f} GiB "
        "recommended working set). Use a smaller domain or a device with "
        f"more memory. Override with {FRACTION_ENV} or --no-memory-gate.")
