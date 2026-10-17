"""Experiment runner, gates, checkpoints, and failure receipts for WOOF global."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import struct
import math
from pathlib import Path
import time

from woof.globe.spectral.diffusion import ExponentialHyperdiffusion
from woof.globe.spectral.transform import SphericalHarmonicTransform

from .checkpoint import (
    TRACKER_KEYS,
    bundle_arrays,
    checkpoint_metadata,
    normalize_trackers,
    read_checkpoint,
    state_from_checkpoint,
    trajectory_from_checkpoint,
    write_checkpoint,
    write_checkpoint_arrays,
)
from .config import (
    DEFAULT_EULERIAN_STEP_RULE_FRACTION,
    ArwenGlobalConfig,
    default_eulerian_step_s,
    eulerian_wind_ceiling_m_s,
)
from .constants import EARTH_RADIUS_M
from .device_memory import (
    device_memory_receipt,
    disable_fft_plan_cache,
    select_device_allocator,
    start_device_peak_tracking,
)
from .dynamics import MoistHybridModel
from .initial_conditions import analytic_initial_state
from .insitu.ledger import InsituLedger, insitu_owned_files
from .physics import NativeArwenPhysicsBridge, ReferencePhysics
from .pins import pins_hash, pins_receipt
from .profile import PROFILE_NAME, StepProfiler, attach_profiler, calibrate
from .receipt import write_receipt
from .semi_implicit import build_semi_implicit
from .semilag.step import SEMILAG_INTEGRATORS


RECEIPT_NAME = "arwen-global-receipt.json"
DIAGNOSTICS_NAME = "diagnostics.jsonl"
CHECKPOINT_PREFIX = "arwen_global_step"

# The mass/water drift gates compare post-fix means against the very targets
# the fixers reset every step, so they cannot see a conservation leak while
# the fixers are on. These budgets gate what the fixers and the physics local
# water repair ABSORBED, per step, so an unbounded leak fails the run even
# while the fixers hold the mean. The limits are per-precision because the
# healthy ceiling is arithmetic roundoff and roundoff scales with the float:
#
# float64, measured on the six-hour T7 reference campaign (360 steps,
# conservative physics, exchange-clamp closure): mass fixer |log offset|
# <= 1.64e-7, water fixer <= 3.9e-8 relative, physics local repair
# <= 1.14e-13 kg/m2.
#
# float32, measured on the six-hour T63 GDAS canary on the RTX 5090
# (72 steps, 2026-08-31): 4.14e-5, 1.51e-6 relative, and 1.22e-4 kg/m2 -
# that last is exactly 2^-13, the fp32 ulp of the ~500 kg/m2 ledger.
#
# The leaks these exist to catch measured 5e-3 (0.5%/step surface-pressure
# sink), 2.4e-4 relative/step (1%-per-call qv deletion), and 8.2e-4
# kg/m2/step (the unfixed diffusion operator) - every limit sits >=4x above
# its precision's measured healthy ceiling and remains below the leak scale.
#
# These envelopes were measured on the REFERENCE suite, whose exchange-clamp
# closure has no kernel-internal error channel.  (They were measured under
# the v4 uniform reservoir levy; under the v5 atmosphere-internal fixer the
# clamps no longer move water between reservoir and atmosphere, so the water
# fixer absorbs only transport/projection residuals - audit 2026-09-01
# VTW-1.)  The arwen-native suite does:
# Morrison's rainncv is credited 1:1 to the reservoir while the kernel's own
# fp32 sedimentation/process arithmetic sets the actual column debit, and the
# post-kernel negative-species clamp silently creates up to ~1e-7 kg/kg per
# level-species.  Both are genuine fp32 accumulation, not booking routes
# (CPU repro water-repair-262, 2026-09-01: every Noah/PBL booking route
# settles to <=2x2^-14 by construction), so a native run config may carry its
# own MEASURED gates.physics_water_repair_max_step_kg_m2 - the config-level
# override below - with the measurement and sizing arithmetic quoted in the
# config.  Reference configs keep this envelope.
_FIXER_ABSORPTION_LIMITS = {
    "float64": {
        "mass_log_offset": 1.0e-6,
        "water_relative": 1.0e-6,
        "physics_repair_kg_m2": 1.0e-4,
    },
    "float32": {
        "mass_log_offset": 2.0e-4,
        "water_relative": 1.0e-5,
        "physics_repair_kg_m2": 5.0e-4,
    },
}


# The vapor fixer (dynamics._fill_column_holes) closes the vapor clamps
# inside each column: the global-mean column water the clips create per
# step, divided by the global-mean atmospheric water column, is the
# fraction of the atmosphere's water the clips created and the columns
# paid back from their own vapor that step.  Under the v5 per-level global
# rescale (retired 2026-09-02 with the spectral condensate) the same ratio
# was 7.6e-5..1.2e-4 sustained on the T63 GDAS reference config and 9.0e-6
# on the T3 smoke, most of it condensate ringing; with vapor the only
# spectral water field it measures 3.4e-7 on the 10-step T255 native case
# (largest column fraction 0.06) and exactly zero on the T3 smoke, so the
# limit now sits four orders above the healthy figure.  The breakage the
# gate names is unchanged: a transport or physics defect that drives whole
# columns negative every step (the class of leak
# test_receipt_fails_when_the_positivity_fixer_carries_a_sign_defect
# installs, a quarter of the vapor) leaves those columns nothing to pay
# with, the clip creates the water (counted as unfillable) and the drift
# gates read only the projection residual because no reservoir moved.
POSITIVITY_FIXER_RELATIVE_LIMIT = 5.0e-3
WIND_MAGNITUDE_RELATIVE_LIMIT = 1.0e-3


def fixer_absorption_limits(precision: str) -> dict[str, float]:
    try:
        return _FIXER_ABSORPTION_LIMITS[precision]
    except KeyError:
        raise ValueError(
            f"no measured fixer-absorption envelope for precision "
            f"{precision!r}; measure one before running it"
        ) from None


def vertical_grid_receipt(cfg: ArwenGlobalConfig) -> dict[str, object]:
    """The vertical grid a run states about itself in its receipt."""
    return {"coordinate": cfg.vertical_coordinate, **cfg.vertical.describe()}


def vertical_grid_sentence(cfg: ArwenGlobalConfig) -> str:
    grid = vertical_grid_receipt(cfg)
    return (
        f"vertical {grid['coordinate']} nlev={grid['nlev']} "
        f"p_top={grid['p_top_pa'] / 100.0:g} hPa: first full level "
        f"{grid['first_full_level_height_m']:.1f} m AGL at "
        f"{grid['reference_surface_pa'] / 100.0:g} hPa, bottom layer "
        f"{grid['bottom_layer_thickness_pa'] / 100.0:.2f} hPa, "
        f"{grid['full_levels_above_50hpa']} full levels above 50 hPa"
    )


def build_transform(cfg: ArwenGlobalConfig) -> SphericalHarmonicTransform:
    # legendre_band and streaming are the transform's own memory levers.
    # Until this call carried them a config could name either and the
    # transform was built on the defaults regardless, which is the flag
    # parsed and silently ignored, not a lever.
    return SphericalHarmonicTransform.create(
        cfg.truncation,
        nlat=cfg.nlat,
        nlon=cfg.nlon,
        dealias_factor=cfg.dealias_factor,
        backend=cfg.backend,
        precision=cfg.precision,
        tensor_core_contractions=cfg.tensor_core_contractions,
        legendre_band=cfg.legendre_band,
        streaming=cfg.streaming,
    )


def build_diffusion(cfg: ArwenGlobalConfig):
    """The drain the config names: ``None`` when disabled, the shipped
    fixed-shape hyperdiffusion, or the spectral eddy viscosity of closure
    theory (``[diffusion] closure = "spectral_eddy_viscosity"``,
    woof.globe.spectral.eddy_viscosity) with the constants the config
    carries."""
    if not cfg.diffusion_enabled:
        return None
    if cfg.diffusion_closure == "spectral_eddy_viscosity":
        from woof.globe.spectral.eddy_viscosity import SpectralEddyViscosity

        return SpectralEddyViscosity(
            plateau=cfg.closure_plateau,
            cusp_amplitude=cfg.closure_cusp_amplitude,
            cusp_decay=cfg.closure_cusp_decay,
            eddy_prandtl=cfg.closure_eddy_prandtl,
            tail_degrees=cfg.closure_tail_degrees,
            preserve_degree=cfg.diffusion_preserve_degree,
        )
    return ExponentialHyperdiffusion(
        order=cfg.diffusion_order,
        e_folding_time_s_at_truncation=cfg.diffusion_efold_s,
        preserve_degree=cfg.diffusion_preserve_degree,
    )


def build_physics(cfg: ArwenGlobalConfig, backend):
    if cfg.physics_mode == "none":
        return None
    if cfg.physics_mode == "reference":
        return ReferencePhysics(backend, cfg.reference_physics)
    if cfg.physics_mode == "arwen-native":
        # FOUR SIGNATURE REFUSALS STOOD HERE UNTIL 2026-09-09, and they are
        # retired rather than weakened.  Each named one engine callable this
        # suite hands an argument a published 2.7.0 does not accept: the
        # surface layer's vegetation fraction, radiation's effective-size
        # bounding, and the two cumulus constructors' column chunk.  Against
        # such an engine the config was accepted, the card priced and
        # allocated, the first steps integrated, and the argument vector
        # refused inside the physics -- so the refusal was moved forward to
        # here, the first point where the answer is "a native forecast is
        # being built".
        #
        # All four callables are now carried in `woof.globe.core`, cut from
        # the tree this model was graded in.  There is no installed engine
        # that can present a different signature for them, so the question
        # the guard asked has no second answer, and a guard whose defect is
        # fixed teaches a reader to expect a failure that cannot happen.
        # What CAN still move is the engine underneath the carried code: that
        # is the seam manifest's job, and the doctor reports it by file.
        return NativeArwenPhysicsBridge(
            cfg.native_adapter_name or "", cfg.native_adapter_options
        )
    raise AssertionError(cfg.physics_mode)


def _card_gather_of(model, transform):
    """The checkpoint's whole-globe assembler, or None on one card."""
    exchange = getattr(model.pipeline, "exchange", None)
    if exchange is None or int(getattr(exchange, "world", 1)) <= 1:
        return None
    from .cards import gather_named_arrays

    grid = transform.grid

    def gather(arrays):
        return gather_named_arrays(
            exchange.session, arrays, grid.nlat, grid.nlon)

    return gather


def _row_exchange(session):
    from .cards import RowExchange

    return RowExchange(session)


def resolve_latitude_bands(cfg: ArwenGlobalConfig) -> int:
    """The band count this run will use: the config's, or the sizer's.

    ``latitude_bands = 0`` (the shipped default) hands the choice to
    :func:`sizing.auto_latitude_bands`, which reads the card's free bytes
    and returns one when the resident run fits and the smallest count
    that fits when it does not.  That is what makes a BARE run at a
    truncation the card cannot hold resident start rather than die in the
    allocator, which is the whole point of the streaming.
    """
    from .sizing import auto_latitude_bands

    declared = int(getattr(cfg, "latitude_bands", 0) or 0)
    if declared > 0:
        return declared
    return auto_latitude_bands(cfg)


def latitude_bands_receipt(model, cfg, sized_by_door: bool) -> dict[str, object]:
    """The band schedule grid space streamed through, for a receipt.

    The count, the rows it cut them into, and who chose it.  It is NOT
    part of any identity (it changes no arithmetic), so it is recorded
    where a reader can see what the run did rather than where a hash
    would refuse a restart for it.

    Every door that sizes a run writes this, because a door that priced a
    band count and a receipt that does not name one is a run nobody can
    check afterwards.
    """
    return {
        **model.pipeline.receipt(),
        # The door writes the count it priced into the config it hands
        # the run, so the config carrying one does not say who chose it;
        # the door says so instead.
        "chosen_by": "sizer" if sized_by_door
        else ("config" if int(getattr(cfg, "latitude_bands", 0) or 0)
              else "sizer"),
    }


def open_card_session(cfg: ArwenGlobalConfig, transform, bands: int):
    """The card session this rank runs behind, or the single-card one.

    ONE CARD IS THE DEFAULT and it still gets a real session rather than a
    ``None``: the single-card transport is the multi-card one with a world
    of one, so the step executes one code path and there is nothing that
    only a two-card run exercises.

    The band assignment follows the cards' MEASURED speeds when a per-band
    profile is available and splits the rows evenly when it is not; either
    way it changes who computes a row and never what the row is (gate
    BIT-6).  ``NCCL_IB_DISABLE=1`` is set here as well as by the launcher,
    so a rank started by hand carries it too and the receipt never has to
    guess which transport a rate belongs to.
    """
    from . import cards as cards_module

    world = int(getattr(cfg, "cards", 1) or 1)
    if world <= 1:
        return cards_module.single_card_session(transform.grid.nlat, bands)
    axis = str(getattr(cfg, "card_axis", "band"))
    if axis == "order":
        raise ValueError(
            "memory.card_axis='order' is not built for a model RUN.  The "
            "order axis is the CAPACITY axis: it partitions the Legendre "
            "orders so each card holds a fraction of the table, which is the "
            "only path past the whole-table wall (design section 10), and it "
            "is proven bit-identical to one card at the transform "
            "(SphericalHarmonicTransform.forward_orders / inverse_orders, "
            "gate ORDER-BIT).  Wiring it end-to-end through the model step "
            "means running grid space, physics and the FFT whole on every "
            "card, and it buys nothing until a truncation whose table does "
            "not fit one card is wanted -- which no card in the building "
            "runs.  Use the 'band' axis, which serves every truncation the "
            "hardware fits and is the speed axis"
        )
    for key, value in cards_module.launch_environment().items():
        os.environ.setdefault(key, value)
    transport = cards_module.open_transport(
        int(getattr(cfg, "card_rank", 0) or 0), world,
        tuple(getattr(cfg, "card_addresses", ()) or ()),
        prefer=str(getattr(cfg, "card_transport", "auto")),
    )
    declared = tuple(getattr(cfg, "card_weights", ()) or ())
    band_ms = None
    weights = None
    if declared:
        if len(declared) != world:
            raise ValueError(
                f"{len(declared)} card weights for {world} cards: one weight "
                "per rank, in rank order, or none to measure them"
            )
        weights = tuple(float(v) for v in declared)
    else:
        # The per-band cost profile: every rank times the same band-shaped
        # work on its own card and the ranks all-gather the milliseconds,
        # so the assignment follows what the cards MEASURE rather than a
        # table with no tense.
        rows = max(1, transform.grid.nlat // max(1, bands))
        mine = cards_module.measure_band_cost_ms(transform, rows)
        pieces = transport.all_gather(
            "band-profile", struct.pack("!d", float(mine)))
        band_ms = tuple(
            struct.unpack("!d", piece)[0] for piece in pieces
        )
    session = cards_module.CardSession(
        transport, transform.grid.nlat, bands,
        exchange=str(getattr(cfg, "card_exchange", "gather")),
        weights=weights, band_ms=band_ms,
        agreement=str(getattr(cfg, "card_agreement", "refuse")),
    )
    # A gather run assembles a Fourier waist out of latitude rows computed
    # on both cards, so it reproduces the single-card answer only where the
    # cards return the same bits (lane 6's finding).  The check rides the
    # run's own contractions (cards.ContractionAgreement, called by the
    # transform once per distinct shape) rather than a probe at fixed
    # widths here: the fixed-width probe refused this pair at an
    # eight-plane synthesis the T255 L40 step never presents, while every
    # contraction the step does present agrees (MEASURED 2026-09-07).
    return session


def build_model_and_cold_state(cfg: ArwenGlobalConfig, transform=None, *,
                               scratch_destination: str | Path | None = None,
                               card_session=None,
                               metal_diffusion_context=None):
    """The model and its cold state.  ``scratch_destination`` is the output
    directory of the work being built (a run's outdir, an export's
    output): the analysis route's GRIB decode stages its frame stream
    beside it rather than in the system temp."""
    transform = transform or build_transform(cfg)
    bands = resolve_latitude_bands(cfg)
    session = card_session
    if session is None and int(getattr(cfg, "cards", 1) or 1) > 1:
        session = open_card_session(cfg, transform, bands)
    if cfg.initial_mode == "analysis":
        from .analysis_initial import analysis_initial_state

        cold, surface_geopotential, initial_provenance = analysis_initial_state(
            cfg, transform, scratch_destination=scratch_destination
        )
    elif cfg.initial_mode == "baroclinic_wave":
        from .testcases import baroclinic_wave_initial_state

        cold, surface_geopotential, initial_provenance = (
            baroclinic_wave_initial_state(cfg, transform)
        )
    else:
        cold, surface_geopotential, initial_provenance = analytic_initial_state(
            cfg, transform
        )
    diffusion = build_diffusion(cfg)
    model = MoistHybridModel(
        transform=transform,
        vertical=cfg.vertical,
        surface_geopotential=surface_geopotential,
        physics=build_physics(cfg, transform.backend),
        physics_split=cfg.physics_split,
        integrator=cfg.integrator,
        diffusion=diffusion,
        metal_diffusion_context=metal_diffusion_context,
        divergence_diffusion_strength=cfg.divergence_diffusion_strength,
        pressure_diffusion_strength=cfg.pressure_diffusion_strength,
        water_diffusion_strength=cfg.water_diffusion_strength,
        semi_implicit=build_semi_implicit(
            cfg.semi_implicit_scheme,
            enabled=cfg.semi_implicit_enabled,
            weight=cfg.semi_implicit_weight,
            external_wave_speed_m_s=cfg.external_wave_speed_m_s,
            reference_temperature_k=cfg.semi_implicit_reference_temperature_k,
            reference_surface_pressure_pa=cfg.semi_implicit_reference_surface_pressure_pa,
            off_centring_weight=cfg.semi_implicit_off_centring_weight,
        ),
        mass_fixer=cfg.mass_fixer,
        water_fixer=cfg.water_fixer,
        positivity_repair=cfg.positivity_repair,
        maximum_cfl=cfg.maximum_cfl,
        maximum_lipschitz=cfg.maximum_lipschitz,
        semilag=cfg.semilag,
        sponge_base_pa=cfg.sponge_base_pa,
        sponge_lid_relaxation_time_s=cfg.sponge_lid_relaxation_time_s,
        initial_provenance=initial_provenance,
        spectral_chunk=cfg.spectral_chunk,
        synthesis_memo=cfg.synthesis_memo,
        latitude_bands=bands,
        cards=None if session is None else _row_exchange(session),
        card_halo_rows=int(getattr(cfg, "card_halo_rows", 16)),
    )
    cold, _, _, _ = model._repair_positivity(cold)
    model.initialize_mass_target(cold.atmosphere)
    model.initialize_water_target(cold)
    model.enforce(cold)
    cold = attach_host_tier(cfg, model, cold)
    return model, cold


def attach_host_tier(cfg: ArwenGlobalConfig, model, bundle):
    """Give ``model`` its pinned host tier and park what it holds.

    The slices are chosen by the MINIMUM-SPILL rule against the peak the
    run will actually take -- the band count's peak, not the resident one
    -- and the census is read off the state's own arrays rather than
    fitted (``sizing.spill_census``).  With nothing to park this returns
    the bundle unchanged and the model keeps no tier, so a run that does
    not need the tier is the run that shipped before it existed.
    """
    from .sizing import SPILL_SLICES, plan_run_memory, spill_census
    from .spill import HostTier

    mode = str(getattr(cfg, "host_spill", "auto"))
    if mode == "off":
        return bundle
    if getattr(model, "host_tier", None) is not None:
        # A second call (the restart path) parks into the tier the builder
        # already made, so the slots the cold state opened are the slots
        # the restart writes rather than a second tier's.
        return model.park_persistent(bundle)
    census = spill_census(bundle, cfg)
    if mode == "on":
        slices = tuple(n for n in SPILL_SLICES if int(census.get(n, 0)) > 0)
    else:
        # THE SAME FUNCTION THE DOOR USED, with a better census: the door
        # priced the three slices from the estimate's terms before an
        # array existed, and this reads them off the arrays themselves.
        # Two policies here and at the door is how a run gets sized for
        # one plan and built for another.
        free = _free_device_bytes_for_spill(cfg)
        plan = plan_run_memory(cfg, free, census=census)
        slices = plan.spill_slices
    if not slices:
        return bundle
    model.host_tier = HostTier(model.transform.backend.xp)
    model.host_tier.slices = list(slices)
    model.spill_slices = tuple(slices)
    return model.park_persistent(bundle)


def _host_spill_receipt(cfg, model, *, wall_steps=0, wall_seconds=0.0,
                        plan=None):
    """What the pinned host tier held, moved, and cost: gate OVERLAP-1.

    ``transfer_share`` is the time the tier's own streams spent copying,
    measured with CUDA events around every transfer, over the wall the
    steps took.  A spill configuration whose transfers are NOT hidden
    reports it here rather than running a campaign fifteen percent slow
    with nothing in the receipt to say why.
    """
    # WHO CHOSE IT, and the plan is asked first.  The door writes the mode
    # it priced into the config it hands the run, so a config reading
    # "off" after the door has spoken does NOT mean a person asked for
    # off; the plan is the only record of which it was.
    chosen_by = (getattr(plan, "spill_chosen_by", None)
                 or ("config"
                     if str(getattr(cfg, "host_spill", "auto")) != "auto"
                     else "sizer"))
    tier = getattr(model, "host_tier", None)
    if tier is None:
        return {
            "mode": str(getattr(cfg, "host_spill", "auto")),
            "chosen_by": chosen_by,
            "slices": [],
            "parked_gib": 0.0,
            "reason": "nothing parked: the run fits the card without the tier",
        }
    receipt = dict(tier.receipt())
    receipt["mode"] = str(getattr(cfg, "host_spill", "auto"))
    receipt["chosen_by"] = chosen_by
    steps = int(wall_steps or receipt.get("steps", 0) or 0)
    seconds = float(wall_seconds or 0.0)
    receipt["step_wall_seconds"] = round(seconds, 6)
    receipt["steps_timed"] = steps
    if steps and seconds > 0.0:
        step_s = seconds / steps
        receipt["step_s"] = round(step_s, 6)
        receipt["transfer_share"] = round(
            (receipt.get("transfer_s_per_step", 0.0)) / step_s, 6
        ) if step_s > 0 else None
    return receipt


def _step_wall_receipt(steps: int, seconds: float) -> dict:
    """The steps' own wall: the count timed, their seconds, one step's."""
    steps = int(steps or 0)
    seconds = float(seconds or 0.0)
    return {
        "steps_timed": steps,
        "step_wall_seconds": round(seconds, 6),
        "step_s": round(seconds / steps, 6) if steps > 0 else None,
        "measures": ("perf_counter around model.step only: the build, the "
                     "cold start, the diagnostics and the checkpoint "
                     "writes are outside it"),
    }


def _free_device_bytes_for_spill(cfg) -> int | None:
    if getattr(cfg, "backend", "numpy") != "cupy":
        return None
    from .sizing import _free_device_bytes

    return _free_device_bytes()


def _owned_files(outdir: Path) -> list[Path]:
    return [
        outdir / RECEIPT_NAME,
        outdir / DIAGNOSTICS_NAME,
        *sorted(outdir.glob(f"{CHECKPOINT_PREFIX}*.npz")),
        *insitu_owned_files(outdir),
    ]


def _prepare_outdir(
    outdir: Path, overwrite: bool, *, keep: frozenset[Path] = frozenset()
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    existing = [path for path in _owned_files(outdir) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"WOOF global output exists in {outdir}; pass --overwrite to replace owned files"
        )
    if overwrite:
        for path in existing:
            if path.resolve() in keep:
                continue
            path.unlink()


def _checkpoint_path(outdir: Path, step: int) -> Path:
    return outdir / f"{CHECKPOINT_PREFIX}{int(step):08d}.npz"


def _append_diagnostics(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


# Receipt-only trackers: the checkpoint tracker schema is pinned to eight
# keys, so these accumulate per run segment and are not inherited on restart.
SUPPLEMENTARY_TRACKER_KEYS = (
    "maximum_repaired_negative_number_per_kg",
    "maximum_positivity_fixer_relative",
    "maximum_positivity_fixer_water_kg_m2",
    "maximum_positivity_fixer_rescale",
    # The semi-Lagrangian core's three (zero on every other path).  They
    # are receipt-only for the reason the four above are: the checkpoint
    # tracker schema is pinned to eight keys and normalize_trackers
    # REFUSES any dict whose key set differs, so a ninth entry would make
    # every checkpoint ever written unreadable.
    "maximum_semilag_lipschitz",
    "maximum_semilag_trajectory_move_cells",
    "maximum_semilag_tracer_mass_fixer_relative",
    "maximum_semilag_tracer_mass_fixer_water_relative",
)


#: The per-species rows the semi-Lagrangian tracer fixer writes each step.
_SPECIES_FIXER_PREFIXES = (
    "semilag_tracer_mass_fixer_relative__",
    "semilag_tracer_mass_fixer_kg_m2__",
    "semilag_tracer_clip_share__",
    # The mass the positivity floor created, which is where the fixer's
    # correction comes from on a species that is zero nearly everywhere.
    # Without this row a receipt says how much was corrected and not what
    # made the correction necessary, and the two arms that separate the
    # limiter from the floor cannot be told apart at all.
    "semilag_tracer_positivity_clamp_kg_m2__",
)


def _update_trackers(
    trackers: dict[str, float],
    supplementary: dict[str, float],
    metrics: dict[str, object],
    species: dict[str, float] | None = None,
) -> None:
    trackers["maximum_spectral_cfl"] = max(
        trackers["maximum_spectral_cfl"], float(metrics["spectral_cfl"])
    )
    trackers["maximum_mass_fixer_log_offset"] = max(
        trackers["maximum_mass_fixer_log_offset"],
        abs(float(metrics["mass_fixer_log_offset"])),
    )
    trackers["maximum_global_water_fixer_kg_m2"] = max(
        trackers["maximum_global_water_fixer_kg_m2"],
        abs(float(metrics["global_water_fixer_kg_m2"])),
    )
    trackers["maximum_repaired_negative_mixing_ratio"] = max(
        trackers["maximum_repaired_negative_mixing_ratio"],
        abs(float(metrics["maximum_repaired_negative_mixing_ratio"])),
    )
    supplementary["maximum_repaired_negative_number_per_kg"] = max(
        supplementary["maximum_repaired_negative_number_per_kg"],
        abs(float(metrics["maximum_repaired_negative_number_per_kg"])),
    )
    for key, metric in (
        ("maximum_positivity_fixer_relative", "positivity_fixer_relative"),
        ("maximum_positivity_fixer_water_kg_m2", "positivity_fixer_water_kg_m2"),
        ("maximum_positivity_fixer_rescale", "positivity_fixer_max_rescale"),
        ("maximum_semilag_lipschitz", "semilag_lipschitz"),
        ("maximum_semilag_trajectory_move_cells",
         "semilag_trajectory_move_max_cells"),
        ("maximum_semilag_tracer_mass_fixer_relative",
         "semilag_tracer_mass_fixer_relative"),
        ("maximum_semilag_tracer_mass_fixer_water_relative",
         "semilag_tracer_mass_fixer_water_relative"),
    ):
        supplementary[key] = max(
            supplementary[key], abs(float(metrics.get(metric, 0.0)))
        )
    # Per species, so a receipt that reports a large correction says which
    # species carried it.  A run that moves four percent of the graupel and
    # nothing else per step is a different finding from one that moves four
    # percent of the cloud water, and the single maximum cannot tell them
    # apart.  Receipt-only, for the reason in SUPPLEMENTARY_TRACKER_KEYS.
    if species is not None:
        for name, value in metrics.items():
            if name.startswith(_SPECIES_FIXER_PREFIXES):
                species[name] = max(species.get(name, 0.0), abs(float(value)))
    trackers["maximum_semi_implicit_divergence_increment_s1"] = max(
        trackers["maximum_semi_implicit_divergence_increment_s1"],
        abs(float(metrics["semi_implicit_max_divergence_increment_s1"])),
    )
    repairs = []
    for half in ("first_half_physics", "second_half_physics"):
        value = metrics.get(half, {})
        if isinstance(value, dict):
            physics = value.get("physics", {})
            if isinstance(physics, dict):
                repairs.append(float(physics.get("maximum_local_water_repair_kg_m2", 0.0)))
    trackers["maximum_physics_water_repair_kg_m2"] = max(
        trackers["maximum_physics_water_repair_kg_m2"],
        *(repairs or [0.0]),
    )
    native_water = []
    native_energy = []
    for half in ("first_half_physics", "second_half_physics"):
        value = metrics.get(half, {})
        if isinstance(value, dict):
            physics = value.get("physics", {})
            if isinstance(physics, dict):
                native_water.append(float(
                    physics.get("maximum_native_water_residual_kg_m2", 0.0)
                ))
                native_energy.append(float(
                    physics.get("maximum_native_energy_change_j_m2", 0.0)
                ))
    trackers["maximum_native_water_residual_kg_m2"] = max(
        trackers["maximum_native_water_residual_kg_m2"],
        *(native_water or [0.0]),
    )
    trackers["maximum_native_energy_residual_j_m2"] = max(
        trackers["maximum_native_energy_residual_j_m2"],
        *(native_energy or [0.0]),
    )


#: Per-step correction the semi-Lagrangian tracer fixer may make, as a
#: fraction of the atmosphere's OWN column water.
#:
#: This is the number the breakage is about.  A tracer scheme that is not
#: conservative silently loses or gains water over a forecast day, which
#: is the class of defect the grid tracers were introduced to retire, and
#: what says whether that is happening is how much water the fixer has to
#: put back, not how large that is beside one trace species' own mass.
#:
#: Priced from measurement, not from an estimate.  MEASURED 2026-09-06,
#: the 16 GB host RTX 5070 Ti, T255 L40 with the full native suite at dt = 300 s:
#: 3.44e-5 over a whole forecast day (288 steps) from the GDAS 2026-09-01
#: 00Z analysis, and 2.17e-5 over the six-hour arms, which run the OTHER
#: case this tree carries (GDAS 2026-08-30 18Z,
#: configs/verify/arwen_global_gdas_t255_native_sl_si_6h.toml) -- two
#: initial states, named because a reading of six hours and a reading of a
#: day here are not two points on one run.  The six-hour number is the same
#: 2.17e-5 whichever form the fixer runs, because it is the ADVECTION's
#: error and not the fixer's.  The limit is 1e-4, three times the day's
#: reading.
#:
#: It interlocks with the two water gates rather than duplicating them.  A
#: run sitting at this limit with every correction the same sign would move
#: 2.9 percent of the column over 288 steps, which
#: ``total_water_relative_drift`` (1e-3) would refuse; a run that hid it in
#: the surface reservoir instead would be refused by
#: ``water_fixer_max_step_relative``.  What this row adds is the reading
#: BEFORE either of those absorbs it, so a scheme whose water error is
#: growing is visible while it is still being closed.
SEMILAG_TRACER_FIXER_WATER_RELATIVE_LIMIT = 1.0e-4

#: Per-step relative mass the fixer may move within ONE species.  This is
#: an accuracy diagnostic of the advection, not a conservation gate: the
#: fixer closes the mass either way, and what a large value says is that
#: the species' field is being shaped by the fixer's weighting as much as
#: by the flow.
#:
#: Read what the number is before reading the limit.  ``fix_mass``
#: computes it as ``|mass(before) - mass(advected)| / mass(before)`` on
#: the field the gather produced and BEFORE any correction is applied,
#: and it computes it for every scheme including ``"none"``.  So it
#: measures the TRANSPORT's mass error and no choice of fixer can move
#: it: the multiplicative form this module ships is, on a non-negative
#: field, algebraically the additive Bermejo-Conde form with
#: mass-proportional weights (``q (1 + d/W) == q + d q/W``), so "take the
#: additive form instead" is not a remedy for this number, it is the same
#: operator.
#:
#: Priced from measurement.  The 1e-4 of the specification of record was an
#: estimate written before the scheme ran; it is off by three orders of
#: magnitude and this is what the arms read.
#:
#: MEASURED 2026-09-06, the 16 GB host, T255 L40 native: 1.01e-1 for graupel over
#: the forecast day (GDAS 2026-09-01 00Z) and 5.96e-2 to 6.35e-2 over the
#: six-hour arms (GDAS 2026-08-30 18Z, the other case), the six-hour
#: reading the same to two figures under every fixer form AND with the
#: quasi-monotone limiter turned off (6.43e-2).  What the number measures
#: is what POSITIVITY costs, not what the limiter costs: a cubic through a
#: field that is zero at three of its four stencil points undershoots
#: below zero on the shoulder of every maximum, something has to lift it
#: back, and turning the limiter off only moves that lift from the
#: gather's clip to the fixer's own floor.  Turning both off stops it and
#: the run is refused at its first step, on a cloud water of -1.49e-5
#: against a maximum of 1.02e-3.
#:
#: The limit is 0.25.  Named breakage: a species whose per-step correction
#: approaches its own mass is a species the fixer is placing rather than
#: the flow transporting, and a quarter of it per step is where that
#: begins to be true over the few steps a convective species lives in one
#: column.  2.5 times the day's worst reading.
#:
#: What the number is a function of, so the headroom is not read as an
#: invariant.  MEASURED 2026-09-06 on the numpy specification, a synthetic
#: condensate blob displaced by a third of a cell in each direction on a
#: 40-level stack: the correction falls from 1.35e-1 for a species ONE
#: model layer deep to 1.9e-2 for one ten layers deep, and from 1.16e-1
#: for a blob two cells across to 2.6e-2 for one ten cells across, while
#: the UNLIMITED gather's own mass error stays at 4.6e-3 throughout.  So
#: the reading is what positivity costs FOR A SPECIES A FEW CELLS ACROSS
#: AND ONE TO THREE LAYERS DEEP, which is what a condensate species is at
#: T255 on forty levels; it is not a property of the scheme alone, and a
#: run at a sharper truncation or on a thinner ladder reads higher.  The
#: 2.5x headroom above the T255 L40 day is headroom AT THAT OPERATING
#: POINT and the limit owes a re-reading at T533 and at T799.
SEMILAG_TRACER_FIXER_RELATIVE_LIMIT = 0.25


def cfl_headroom_receipt(cfg: ArwenGlobalConfig, trackers: dict[str, float]) -> dict[str, object]:
    """What the day's strongest wind left of the step, written into every
    receipt so a refusal on a stronger day is explained by the numbers.

    The gate is the spectral CFL dt |V|max sqrt(N(N+1))/a against
    ``[time] maximum_cfl``; on the Eulerian path it refuses, on the
    semi-Lagrangian path it is measured and the Lipschitz gate refuses
    instead.  The block carries the day's maximum, its fraction of the
    gate, the wind it implies, the largest step this day would have
    admitted at the gate and at the shipped rule, and the shipped step of
    this truncation so a reader sees whether the run took it.
    """
    maximum = float(trackers.get("maximum_spectral_cfl", 0.0))
    dt = float(cfg.dt_s)
    gate = float(cfg.maximum_cfl)
    truncation = int(cfg.truncation)
    factor = math.sqrt(truncation * (truncation + 1.0)) / EARTH_RADIUS_M
    rate = maximum / dt if dt > 0.0 else 0.0
    semi_lagrangian = cfg.integrator in SEMILAG_INTEGRATORS
    rule = DEFAULT_EULERIAN_STEP_RULE_FRACTION
    try:
        shipped = default_eulerian_step_s(truncation, gate)
    except ValueError:
        # A gate no candidate step satisfies (a test's near-zero gate, or a
        # truncation past the rule's reach): the receipt still carries the
        # day's own numbers, and the shipped step reads as none.
        shipped = None
    block: dict[str, object] = {
        "role": (
            "measurement: the semi-Lagrangian path has no advective bound, "
            "the Lipschitz gate refuses instead"
            if semi_lagrangian else "refusal: the Eulerian path refuses above the gate"
        ),
        "dt_s": dt,
        "truncation": truncation,
        "gate": gate,
        "maximum_spectral_cfl": maximum,
        "fraction_of_gate": maximum / gate if gate > 0.0 else None,
        "headroom_fraction_of_gate": 1.0 - maximum / gate if gate > 0.0 else None,
        "implied_maximum_wind_m_s": rate / factor if factor > 0.0 else None,
        "largest_step_at_gate_s": gate / rate if rate > 0.0 else None,
        "largest_step_at_shipped_rule_s": rule * gate / rate if rate > 0.0 else None,
        "shipped_rule": {
            "fraction_of_gate": rule,
            "wind_ceiling_m_s": eulerian_wind_ceiling_m_s(truncation),
            "shipped_eulerian_step_s": shipped,
            "this_run_took_the_shipped_step": (
                shipped is not None and not semi_lagrangian and abs(dt - shipped) < 1.0e-9
            ),
        },
    }
    if semi_lagrangian:
        block["sentence"] = (
            f"spectral CFL reached {maximum:.3f} at dt {dt:g} s on the semi-Lagrangian "
            f"path (no advective bound; the Lipschitz gate refuses instead); the "
            f"Eulerian core would have admitted dt <= {block['largest_step_at_gate_s']:.1f} s "
            f"on this day" if rate > 0.0 else "no CFL reading"
        )
    else:
        block["sentence"] = (
            f"spectral CFL reached {maximum:.3f} of the {gate:g} gate at dt {dt:g} s "
            f"({100.0 * maximum / gate:.0f} percent, implied maximum wind "
            f"{block['implied_maximum_wind_m_s']:.1f} m/s); this day admits dt <= "
            f"{block['largest_step_at_gate_s']:.1f} s at the gate and <= "
            f"{block['largest_step_at_shipped_rule_s']:.1f} s at the shipped {rule:g}-of-gate "
            f"rule, whose step at T{truncation} is "
            + ("none (no candidate step satisfies this gate)" if shipped is None else f"{shipped:g} s")
            if rate > 0.0 else "no CFL reading"
        )
    return block


def _relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-30)


class _CheckpointWriter:
    """Checkpoints published off the model thread.

    The device-to-host copy of the state (``bundle_arrays``) runs on the
    caller's thread, so the arrays are the step's own; the hashing,
    compression, fsync and rename (``write_checkpoint_arrays``, the same
    function and the same bytes as a synchronous write) run on one
    worker while the next steps integrate.  At most one write is in
    flight: the next submission first joins the previous one, so a
    failed write raises on the model thread at the next checkpoint or
    at :meth:`close`, never silently.  Why: a checkpoint's zlib and
    SHA-256 passes held the card idle on the model thread (the step
    profiler's ``checkpoint`` section, 2026-09-04).
    """

    def __init__(self, cfg, to_numpy, trajectory=None, card_gather=None):
        self.cfg = cfg
        self.to_numpy = to_numpy
        # A callable, read at submit time: the second time level of a
        # two-time-level integrator is replaced every step, and the
        # archive must carry the one that belongs to the state it holds.
        self.trajectory = trajectory
        # The multi-card row gather runs on the CALLER's thread with the
        # device-to-host copy, never on the worker: it is a collective and
        # both cards have to reach it in the same order as every other
        # exchange of the step.
        self.card_gather = card_gather
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending = None

    def submit(self, path: Path, state, trackers: dict, *, want_metadata: bool = False):
        """Publish ``state`` to ``path`` off the model thread; returns the
        path, or ``(path, metadata)`` with the record the archive will
        carry (``checkpoint.checkpoint_metadata``, computed here on the
        caller's thread from the same arrays) when ``want_metadata`` is
        set: the cycle door names a background by its ``self_sha256``
        before the write has finished."""
        self.join()
        arrays = bundle_arrays(
            state, self.to_numpy,
            None if self.trajectory is None else self.trajectory(),
            card_gather=self.card_gather,
        )
        cfg = self.cfg
        record = dict(
            step=int(state.step), time_s=float(state.time_s),
            physics_state_schema=state.physics_state.schema,
            physics_metadata=json.loads(json.dumps(
                state.physics_state.metadata, sort_keys=True, allow_nan=False,
            )),
            config_hash=cfg.config_hash, trackers=dict(trackers),
            semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator,
        )
        metadata = checkpoint_metadata(arrays, **record) if want_metadata else None
        self._pending = self._executor.submit(
            write_checkpoint_arrays, path, arrays, metadata=metadata, **record,
        )
        return (path, metadata) if want_metadata else path

    def join(self) -> None:
        pending, self._pending = self._pending, None
        if pending is not None:
            pending.result()

    def close(self) -> None:
        try:
            self.join()
        finally:
            self._executor.shutdown(wait=True)


def _section(profiler, name: str):
    """A profiler section, or a no-op when no profiler is attached."""
    if profiler is None:
        from .profile import NULL_PROFILER

        return NULL_PROFILER.section(name)
    return profiler.section(name)


def _finish_profile(profiler, output: Path, progress) -> None:
    """Write the profile once its last step has been read, and print it."""
    if getattr(profiler, "_written", False):
        return
    path = profiler.write(output / PROFILE_NAME)
    profiler._written = True
    if progress is not None:
        print(
            f"profile: {profiler.profiled_steps} steps after "
            f"{profiler.warmup} warm-up steps, written to {path}"
        )
        print(profiler.table())


def _sizing_model_peak_bytes(cfg: ArwenGlobalConfig) -> int | None:
    """The pre-run prediction, for the receipt's device_memory row and the
    slab's first reservation.

    Priced at the PLAN the run will take, band count and pinned tier
    together: a prediction made for a resident run would size the slab's
    arena for a peak the banded run never reaches, and one that ignored
    the tier would size it for bytes the card is never asked for.
    MEASURED 2026-09-06 at T383 L40 on an RTX 5070 Ti, the bare run: the
    run's live peak read 12,470,658,560 B, and the plan on the tree that
    ships prices that shape at 12,691,484,204 -- 1.8 percent above it,
    which is the direction a figure the slab is sized from has to err in.
    (The 12,467,386,320 B the receipt of that leg carries was priced
    before the census scaling and the fragmentation classes landed, so it
    is not the figure this function returns today.)
    """
    from .sizing import (banded_device_peak_bytes, estimate_global_memory,
                         plan_run_memory, _free_device_bytes)

    try:
        estimate = estimate_global_memory(cfg)
    except Exception:  # noqa: BLE001 - a prediction that cannot be made is not a run failure
        return None
    if estimate.backend != "cupy":
        return None
    try:
        plan = plan_run_memory(cfg, _free_device_bytes(), estimate)
        if plan.bands is not None:
            return int(plan.live_peak_bytes)
    except Exception:  # noqa: BLE001 - the run's own refusal, not the prediction's
        pass
    try:
        bands = resolve_latitude_bands(cfg)
    except Exception:  # noqa: BLE001 - the run's own refusal, not the prediction's
        bands = 1
    return int(banded_device_peak_bytes(estimate, bands))



def metal_refusal(cfg: ArwenGlobalConfig) -> str | None:
    """Why ``--metal`` cannot run ``cfg``, or ``None`` when it can.

    Asked by every door that takes ``--metal`` before it writes anything.
    """

    if cfg.backend != "numpy":
        return (f"--metal runs backend = 'numpy' configs; this one sets "
                f"backend = {cfg.backend!r}. Next: drop --metal")
    if str(cfg.precision).lower() != "float64":
        return (f"--metal runs float64 configs only; this one sets precision = "
                f"{cfg.precision!r}. Next: set precision = 'float64', or "
                "drop --metal")
    if cfg.diffusion_closure == "spectral_eddy_viscosity":
        return ("--metal has no spectral_eddy_viscosity diffusion closure. "
                "Next: choose another closure, or drop --metal")
    if cfg.physics_mode == "arwen-native":
        return ("--metal has no real-data physics: physics mode "
                "'arwen-native' runs on CUDA only. Next: drop --metal")
    bands = resolve_latitude_bands(cfg)
    if bands > 1:
        return (f"--metal runs the whole grid as one latitude band; this one "
                f"sets latitude_bands = {bands}. Next: set "
                "[memory] latitude_bands = 1, or drop --metal")
    return None

def run(
    cfg: ArwenGlobalConfig,
    outdir: str | Path,
    *,
    restart: str | Path | None = None,
    overwrite: bool = False,
    progress=None,
    until_s: float | None = None,
    profile_steps: int = 0,
    profile_warmup: int = 2,
    sized_by_door: bool = False,
    door_plan=None,
    metal: bool = False,
    metal_diffusion: bool | None = None,
) -> dict[str, object]:
    """Integrate ``cfg`` into ``outdir``.

    ``until_s`` ends the integration at that model time (seconds from the
    run's start) instead of ``cfg.duration_s``: a cycling segment, whose
    last checkpoint the next segment restarts from.  It is outside the
    config identity (the receipt records ``segment_until_s``), so every
    segment of a cycle and the forecast from its final analysis share one
    config hash and one checkpoint lineage.

    ``profile_steps`` > 0 attaches the step profiler
    (woof.globe.profile) for that many steps after
    ``profile_warmup`` unprofiled ones, calibrates it first on the run's
    backend, writes ``profile.json`` beside the receipt and prints the
    per-operator table; the profiled run is bit-identical to an
    unprofiled one (the hooks add no arithmetic).
    """
    # The run's allocator is chosen and installed before the first device
    # byte, because the slab's arena has to be the first thing on the card
    # for the run's bytes to come out of it rather than beside it.
    allocator = select_device_allocator(
        cfg.device_allocator, cfg.backend,
        predicted_peak_bytes=_sizing_model_peak_bytes(cfg),
    )
    # The allocator hook goes in before the first device byte (the
    # Legendre tables are the first and, at T533, among the largest) and
    # comes out in the finally below, so the receipt's device peak is the
    # whole run's and the process's allocator is left as it was found.
    tracker = start_device_peak_tracking(cfg.backend)
    # The cuFFT plan cache is off for the run (memory only; see
    # device_memory.disable_fft_plan_cache) and restored with the
    # allocator afterwards.
    restore_plan_cache = disable_fft_plan_cache(cfg.backend)
    if metal_diffusion is not None:
        if metal and not metal_diffusion:
            raise ValueError("metal and metal_diffusion flags disagree")
        metal = bool(metal_diffusion)
    metal_context = None
    try:
        if metal:
            refusal = metal_refusal(cfg)
            if refusal is not None:
                raise ValueError(refusal)
            from woof.metal.runtime import MetalContext
            metal_context = MetalContext()
        return _run_tracked(
            cfg, outdir, restart=restart, overwrite=overwrite,
            progress=progress, tracker=tracker, until_s=until_s,
            profile_steps=int(profile_steps), profile_warmup=int(profile_warmup),
            allocator=allocator,
            sized_by_door=bool(sized_by_door) or (
                door_plan is not None
                and getattr(door_plan, "bands_chosen_by", None) == "sizer"),
            door_plan=door_plan,
            metal_diffusion_context=metal_context,
        )
    finally:
        if metal_context is not None:
            metal_context.close()
        restore_plan_cache()
        if tracker is not None:
            tracker.uninstall()
        if allocator is not None:
            allocator.uninstall()
            close = getattr(allocator, "close", None)
            if close is not None:
                # The arena's bytes go back when the last array that sits
                # in it goes away, not here; this drops the allocator's
                # own hold.
                close()


def _carries_assimilation_chain(metadata: dict) -> bool:
    """Whether a checkpoint's physics metadata holds an assimilation chain
    with at least one cycle (the key the door writes,
    ``assimilate.ASSIMILATION_HISTORY_KEY``; named here by value so the
    runner does not import the door)."""
    physics = metadata.get("physics_metadata") or {}
    chain = physics.get("assimilation_history")
    if not isinstance(chain, dict):
        return False
    cycles = chain.get("cycles")
    return isinstance(cycles, list) and len(cycles) > 0


def _release_cached_device_blocks(backend) -> None:
    """Hand the pool's cached free blocks back to the device once.

    Memory only.  Initialisation (analysis ingest, the cold-state repair,
    the first diagnostics) leaves the pool holding blocks in its own
    shapes that the step never re-requests exactly: measured 19 GiB of
    cached free blocks beside 7 GiB live after a T533 forty-level init
    (2026-09-02).  Returning them lets the step take its working set
    from the device in the shapes it actually asks for instead of
    fragments of the ingest's.
    """
    if backend.name == "cupy":
        # The pool the run's allocator actually spends, not whichever pool
        # the process started with: a run under the async pool or the slab
        # left its cached blocks where they were when this read the default
        # pool by name.
        from .device_memory import installed_pool

        pool = installed_pool(backend.xp)
        release = getattr(pool, "free_all_blocks", None)
        if release is not None:
            release()


def _run_tracked(
    cfg: ArwenGlobalConfig,
    outdir: str | Path,
    *,
    restart,
    overwrite: bool,
    progress,
    tracker,
    until_s: float | None = None,
    profile_steps: int = 0,
    profile_warmup: int = 2,
    allocator=None,
    sized_by_door: bool = False,
    door_plan=None,
    metal_diffusion_context=None,
) -> dict[str, object]:
    output = Path(outdir)
    if until_s is not None:
        until_s = float(until_s)
        if not math.isfinite(until_s) or until_s <= 0.0:
            raise ValueError("--until-s must be a positive, finite model time")
        if until_s > cfg.duration_s + 1.0e-9:
            raise ValueError(
                f"--until-s {until_s:g} lies beyond the config's duration_s "
                f"{cfg.duration_s:g}; a segment cannot outrun the run it "
                "belongs to"
            )
        if abs(until_s / cfg.dt_s - round(until_s / cfg.dt_s)) > 1.0e-6:
            raise ValueError(
                f"--until-s {until_s:g} is not a whole number of {cfg.dt_s:g} s "
                "steps; the segment must end on a step boundary the next "
                "segment can restart from"
            )
    restart_metadata = None
    restart_arrays = None
    if restart is not None:
        # Read the restart source before the overwrite sweep: the sweep owns
        # every arwen_global_step*.npz in outdir, so a resume-in-place would
        # otherwise delete the checkpoint it is about to read. The source is
        # also excluded from the sweep so the receipt's restart row keeps
        # pointing at an existing file.
        # The pin is checked against THIS run's scheme, not the build's
        # default: an archive of the external era resumes under
        # scheme = "external" (its pin is that era's, and the arithmetic is
        # shipped bit-identical), and a checkpoint of either scheme never
        # resumes under the other.
        restart_metadata, restart_arrays = read_checkpoint(
            restart,
            expected_config_hash=cfg.config_hash,
            semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator,
        )
    _prepare_outdir(
        output,
        overwrite,
        keep=(
            frozenset()
            if restart is None
            else frozenset({Path(restart).resolve()})
        ),
    )
    diagnostics_path = output / DIAGNOSTICS_NAME
    receipt_path = output / RECEIPT_NAME
    transform = build_transform(cfg)
    model, cold = build_model_and_cold_state(
        cfg, transform, scratch_destination=output,
        metal_diffusion_context=metal_diffusion_context)
    cold_diag = model.diagnostics(cold)
    target_mass = cold_diag["global_mean_surface_pressure_pa"]
    target_water = cold_diag["global_mean_total_water_kg_m2"]
    trackers = normalize_trackers()
    supplementary = {name: 0.0 for name in SUPPLEMENTARY_TRACKER_KEYS}
    species_fixer: dict[str, float] = {}
    restart_targets = None
    if restart is None:
        state = cold
    else:
        state = state_from_checkpoint(
            restart_metadata, restart_arrays, transform.backend
        )
        # The second time level rides with the state.  Without it the
        # resumed run would take a non-extrapolated start-up step in the
        # middle of a forecast, which is not the continuation of the
        # uninterrupted run, and the device-qualification pin asserts
        # exactly that it is.
        model.set_trajectory_state(
            trajectory_from_checkpoint(
                restart_metadata, restart_arrays, transform.backend
            )
        )
        trackers = normalize_trackers(restart_metadata["run_trackers"])
        if state.time_s >= cfg.duration_s - 1.0e-9:
            raise ValueError(
                "restart checkpoint is already at or beyond configured duration"
            )
        if until_s is not None and state.time_s >= until_s - 1.0e-9:
            raise ValueError(
                f"restart checkpoint at {state.time_s:g} s is already at or "
                f"beyond the segment end --until-s {until_s:g}"
            )
        model.enforce(state)
        if _carries_assimilation_chain(restart_metadata):
            # A state with an assimilation chain opens a new conservation
            # epoch: its analysis increments are deliberate external
            # sources of mass and water, not drift, and the fixers would
            # otherwise remove them on the first step and trip their
            # absorption gates (measured on the T255 hourly cycle: the
            # moisture update's 0.15 kg/m2 global-mean water read 2.1e-4
            # of the column against the water fixer's 1e-5 step gate on
            # the first restarted step).  The targets become the restart
            # state's own global means; a restart without a chain keeps
            # the cold start's, bit for bit as before.
            model.initialize_mass_target(state.atmosphere)
            model.initialize_water_target(state)
            epoch = model.diagnostics(state)
            restart_targets = {
                "reason": (
                    "the restart checkpoint carries an assimilation chain; "
                    "its increments are deliberate sources, so the mass and "
                    "water targets are the restart state's own global means"
                ),
                "cold_mass_target_pa": target_mass,
                "cold_total_water_target_kg_m2": target_water,
                "mass_target_pa": epoch["global_mean_surface_pressure_pa"],
                "total_water_target_kg_m2": epoch["global_mean_total_water_kg_m2"],
            }
            target_mass = epoch["global_mean_surface_pressure_pa"]
            target_water = epoch["global_mean_total_water_kg_m2"]
    # The cold state's only readers above are done; on a cold start it IS
    # ``state`` and on a restart it is dead weight (one spectral state,
    # 1.19 GiB at T533 with 40 levels, held for the whole run otherwise).
    del cold, restart_arrays
    if restart is not None:
        # The cold state was parked by the builder; a restart arrives from
        # the checkpoint reader on the card and is parked here, into the
        # same named slots.
        state = attach_host_tier(cfg, model, state)
    _release_cached_device_blocks(transform.backend)

    # The in-situ ledger watches every step from here on (default on; the
    # [insitu] table is outside the config identity).
    ledger = (
        InsituLedger(cfg, model, output).attach() if cfg.insitu.enabled else None
    )
    # The step profiler, calibrated on this backend before it reads
    # anything (profile.calibrate refuses an instrument that fails).
    profiler = None
    if profile_steps > 0:
        profiler = StepProfiler(
            transform.backend, steps=profile_steps, warmup=profile_warmup,
        )
        profiler.calibration = calibrate(transform.backend)
        attach_profiler(model, profiler)
    transform_check = transform.transform_check(seed=19)
    card_gather = _card_gather_of(model, transform)
    writer = _CheckpointWriter(
        cfg, transform.backend.to_numpy,
        trajectory=model.trajectory_state if model.semi_lagrangian else None,
        card_gather=card_gather,
    )
    output_every = int(round(cfg.output_interval_s / cfg.dt_s))
    total_steps = int(round(cfg.duration_s / cfg.dt_s))
    if until_s is not None:
        total_steps = int(round(until_s / cfg.dt_s))
    checkpoints: list[str] = []
    start_wall = time.perf_counter()
    initial_segment_diag = model.diagnostics(state)
    _append_diagnostics(diagnostics_path, initial_segment_diag)
    if restart is None:
        path = write_checkpoint(
            _checkpoint_path(output, state.step),
            state,
            config_hash=cfg.config_hash,
            to_numpy=transform.backend.to_numpy,
            trackers=trackers,
            semi_implicit_scheme=cfg.semi_implicit_scheme,
            integrator=cfg.integrator,
            trajectory=model.trajectory_state(),
            card_gather=card_gather,
        )
        checkpoints.append(str(path))

    # The wall the STEPS took, separate from the run's: gate OVERLAP-1
    # divides the tier's measured transfer time by it, and a build, a
    # cold start and a first diagnostics call are not steps.
    step_wall_seconds = 0.0
    step_count = 0
    try:
        while state.step < total_steps:
            if profiler is not None:
                # Numbered by the step it produces, like the checkpoints.
                profiler.begin_step(state.step + 1)
            step_started = time.perf_counter()
            state, metrics = model.step(state, cfg.dt_s)
            step_wall_seconds += time.perf_counter() - step_started
            step_count += 1
            if model.pipeline.exchange is not None:
                model.pipeline.exchange.session.mark_step()
            _update_trackers(trackers, supplementary, metrics,
                             species_fixer)
            due = state.step % output_every == 0 or state.step == total_steps
            if due:
                with _section(profiler, "diagnostics"):
                    diag = model.diagnostics(state)
                diag["step_metrics"] = metrics
                _append_diagnostics(diagnostics_path, diag)
                with _section(profiler, "checkpoint"):
                    checkpoint = writer.submit(
                        _checkpoint_path(output, state.step), state, trackers,
                    )
                checkpoints.append(str(checkpoint))
                if progress is not None:
                    progress(diag)
            if profiler is not None:
                profiler.end_step()
                if profiler.profiled_steps == profiler.steps and not profiler.active:
                    _finish_profile(profiler, output, progress)
    except Exception as exc:
        # A checkpoint still in flight is joined so the failure receipt
        # lists only files that exist; a join that fails itself is the
        # same class of failure and rides as this exception's context.
        try:
            writer.close()
        except Exception as writer_exc:  # noqa: BLE001 - recorded beside the model's own failure
            exc.__context__ = writer_exc
        failure = {
            "name": cfg.name,
            "status": "error",
            "config_hash": cfg.config_hash,
            "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
            "pins": pins_receipt(cfg.semi_implicit_scheme, cfg.integrator),
            "model": "moist-hybrid-spectral",
            "staged_metal_stages": (["global_spectral_diffusion",
                                      "global_tracer_mass_flux",
                                      "global_longitude_transform",
                                      "global_momentum_bernoulli",
                                      "global_scalar_tendency",
                                      "global_spectral_mass_offset",
                                      "global_column_positivity",
                                      "global_tracer_zonal_transport",
                                      "global_tracer_walled_transport",
                                      "global_stratospheric_floor",
                                      "global_convective_adjust",
                                      "global_surface_flux"]
                                     if metal_diffusion_context is not None else []),
            "physics_mode": cfg.physics_mode,
            "physics_split": cfg.physics_split,
            "vertical": vertical_grid_receipt(cfg),
            # The schedule the failing run was on.  An out-of-memory
            # death is exactly when a reader needs to know which band
            # count was streaming and who chose it, and the successful
            # receipt carried it while this one did not.
            "latitude_bands": latitude_bands_receipt(model, cfg, sized_by_door),
            # And the other half of the same decision.  A run that died
            # recorded which band count was streaming and neither what the
            # pinned host tier was holding nor what the sizer priced, so
            # the one artifact a reader has of a dead run could not say
            # whether the tier was even on.  MEASURED 2026-09-07: the
            # T533 forecast day raised on a parked slot in its first step
            # and its receipt read host_spill null and sizer null, while
            # the plan was on stdout only.  Both are descriptive, in no
            # identity, and the successful receipt already carried them.
            "host_spill": _host_spill_receipt(cfg, model, plan=door_plan),
            "sizer": None if door_plan is None else door_plan.receipt(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "completed_step": int(state.step),
            "completed_time_s": float(state.time_s),
            "run_trackers": trackers,
            "supplementary_trackers": supplementary,
            # The step's headroom up to the failure: a CFL refusal on a
            # day stronger than the shipped step was set against is
            # explained here, in the step this day would have admitted.
            "cfl": cfl_headroom_receipt(cfg, trackers),
            **({"semilag_tracer_fixer_by_species": species_fixer}
               if species_fixer else {}),
            "checkpoints": checkpoints,
            # The peak up to the failure: an out-of-memory death is
            # exactly when this number is wanted.
            "device_memory": device_memory_receipt(
                tracker, cfg.backend,
                sizing_model_peak_bytes=_sizing_model_peak_bytes(cfg),
                allocator=allocator,
            ),
            "restart": None if restart_metadata is None else {
                "path": str(restart),
                "checkpoint_self_sha256": restart_metadata["self_sha256"],
            },
            "wall_seconds": float(time.perf_counter() - start_wall),
            "step_wall": _step_wall_receipt(step_count, step_wall_seconds),
        }
        if ledger is not None:
            # The failure receipt is the post-mortem; a ledger fault here
            # must not cost the receipt of the model's own failure.
            try:
                failure["insitu"] = ledger.on_failure(state)
            except Exception as ledger_exc:  # noqa: BLE001 - recorded, not raised
                failure["insitu"] = {
                    "error_type": type(ledger_exc).__name__,
                    "error_message": str(ledger_exc),
                }
        write_receipt(receipt_path, failure)
        raise

    # Every checkpoint is on disk before the receipt that lists it.
    writer.close()
    final_diag = model.diagnostics(state)
    # The cards block is built BEFORE the gates, because gate WIRE-1 is
    # one of them and it reads the achieved rate out of this block.
    cards_receipt = (
        model.pipeline.exchange.session.receipt()
        if getattr(model.pipeline, "exchange", None) is not None
        else {"cards": 1, "transport": "single", "card_exchange": "gather"}
    )
    gates = run_gates(
        cfg, transform_check, final_diag, target_mass, target_water,
        trackers, supplementary, cards=cards_receipt,
    )
    status = "pass" if all(row["passed"] for row in gates.values()) else "fail"
    receipt = {
        "name": cfg.name,
        "status": status,
        "config_hash": cfg.config_hash,
        "config": cfg.config_identity,
        # The pin of the arithmetic this run integrated: the scheme's
        # (pins.SEMI_IMPLICIT_PINS), the same pin its checkpoints carry.
        "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
        "pins": pins_receipt(cfg.semi_implicit_scheme, cfg.integrator),
        "model": "moist-hybrid-spectral",
        "staged_metal_stages": (["global_spectral_diffusion",
                                  "global_tracer_mass_flux",
                                  "global_longitude_transform",
                                  "global_momentum_bernoulli",
                                  "global_scalar_tendency",
                                  "global_spectral_mass_offset",
                                  "global_column_positivity",
                                  "global_tracer_zonal_transport",
                                  "global_tracer_walled_transport",
                                  "global_stratospheric_floor",
                                  "global_convective_adjust",
                                  "global_surface_flux"]
                                 if metal_diffusion_context is not None else []),
        "physics_mode": cfg.physics_mode,
        "physics_split": cfg.physics_split,
        "vertical": vertical_grid_receipt(cfg),
        # The scheme and, under vertical_modes, the reference operator's
        # phase speeds and equivalent depths the run integrated with.
        "semi_implicit": model.semi_implicit.describe(model.vertical),
        "physics_identity": None if model.physics is None else model.physics.identity,
        "initial": {
            "mode": cfg.initial_mode,
            "provenance": model.initial_provenance,
        },
        # The planet the land surface ran on, named at the top level so a
        # reader never has to know the synthetic arm is a provenance leaf.
        "statics": (
            None if not isinstance(model.initial_provenance, dict)
            else model.initial_provenance.get("statics")
        ),
        "transform": transform.identity,
        # The four memory levers as this run was built, DESCRIPTIVE and in
        # no hash: two of them are deliberately absent from every identity
        # (synthesis_memo, streaming) and a run that names them left no
        # trace at all -- a streamed run, whose analyses cost 15,144x a
        # resident-table call, read back identical to a resident one.
        "memory": {
            "spectral_chunk": int(cfg.spectral_chunk),
            "synthesis_memo": bool(cfg.synthesis_memo),
            "legendre_band": int(cfg.legendre_band),
            "streaming": bool(cfg.streaming),
        },
        # The band schedule grid space streamed through: the count, the
        # rows it cut them into, and whether the sizer chose it.  It is
        # NOT part of the identity (it changes no arithmetic), so it is
        # recorded where a reader can see what the run did rather than
        # where a hash would refuse a restart for it.
        "latitude_bands": latitude_bands_receipt(model, cfg, sized_by_door),
        "host_spill": _host_spill_receipt(cfg, model, wall_steps=step_count,
                                          wall_seconds=step_wall_seconds,
                                          plan=door_plan),
        # THE WALL THE STEPS TOOK, on every receipt.  The tier's block
        # above carries a per-step figure only when something is parked;
        # a resident run had none, and its step could only be read off
        # the whole run's wall with the build, the cold start and the
        # checkpoint writes inside it.  The capacity rows and the two-card
        # ratio are defined on a step, so every receipt carries one
        # (2026-09-07, the cards lane's T383 pair measurement).
        "step_wall": _step_wall_receipt(step_count, step_wall_seconds),
        # WHAT THE SIZER CHOSE, AND AGAINST WHAT.  The band count and the
        # host tier are one decision (sizing.plan_run_memory), and this is
        # the door's own record of it: the free bytes it read before a
        # byte was allocated, the quarter-free budget it weighed against,
        # the fragmentation and out-of-pool terms it charged, the card
        # figure that came out, and one sentence saying what it took and
        # why.  A bare run's receipt says what a bare run did, which is
        # what a reader who set no flag needs and could not get before.
        # Absent only when the run was not started through a door.
        "sizer": None if door_plan is None else door_plan.receipt(),
        # The cards this rank shared the band schedule with, the transport
        # it used, and what crossed the wire.  Gate WIRE-1 reads
        # "achieved_gb_s" and refuses a run below the floor: every
        # interconnect figure the design was priced on was taken on IDLE
        # cards, and this is the same quantity taken while both cards
        # compute.  A single-card run records the same block with a world
        # of one and zero bytes, so a reader never has to ask whether the
        # field was simply absent.
        "cards": cards_receipt,
        "transform_check": transform_check,
        "cold_start_diagnostics": cold_diag,
        "segment_start_diagnostics": initial_segment_diag,
        "final_diagnostics": final_diag,
        "mass_target_pa": target_mass,
        "total_water_target_kg_m2": target_water,
        "gates": gates,
        "run_trackers": trackers,
        "supplementary_trackers": supplementary,
        # The day's CFL against the gate, the wind it implies and the step
        # it would have admitted at the gate and at the shipped rule.
        "cfl": cfl_headroom_receipt(cfg, trackers),
        # The spectral eddy viscosity's last reading (nu at the plateau per
        # level, the energy at the truncation it was read from); None under
        # the hyperdiffusion.
        "closure": getattr(model, "closure_record", None),
        **({"semilag_tracer_fixer_by_species": species_fixer}
           if species_fixer else {}),
        "checkpoints": checkpoints,
        "segment_until_s": until_s,
        "restart_targets": restart_targets,
        "restart": None if restart_metadata is None else {
            "path": str(restart),
            "checkpoint_self_sha256": restart_metadata["self_sha256"],
            "inherited_run_trackers": restart_metadata["run_trackers"],
        },
        "wall_seconds": float(time.perf_counter() - start_wall),
        "insitu": None if ledger is None else ledger.close(),
        # MEASURED at the allocator over the whole run (tables, initial
        # state, every step); the sizing model's pre-run prediction rides
        # beside it under its own name.  See device_memory.py.
        "device_memory": device_memory_receipt(
            tracker, cfg.backend,
            sizing_model_peak_bytes=_sizing_model_peak_bytes(cfg),
            allocator=allocator,
        ),
    }
    write_receipt(receipt_path, receipt)
    checked = json.loads(receipt_path.read_text(encoding="utf-8"))
    checked["receipt_path"] = str(receipt_path)
    return checked


def run_gates(
    cfg: ArwenGlobalConfig,
    transform_check: dict,
    final_diag: dict,
    target_mass: float,
    target_water: float,
    trackers: dict[str, float],
    supplementary: dict[str, float],
    cards: dict | None = None,
) -> dict[str, dict[str, object]]:
    """The receipt's gates, each with its value, limit and verdict: the
    transform controls, the mass and water drift against the epoch's
    targets, what the fixers and the physics repair absorbed per step,
    and on a multi-card run the rate the wire actually achieved.  One
    function so a run and a cycle judge themselves alike.

    Every row but one is a CEILING: the value must not exceed the limit.
    Gate WIRE-1 is a FLOOR and carries ``direction`` to say so."""
    mass_drift = _relative(
        final_diag["global_mean_surface_pressure_pa"], target_mass
    )
    water_drift = _relative(
        final_diag["global_mean_total_water_kg_m2"], target_water
    )
    gates = {
        "transform_roundtrip_relative_linf": {
            "value": transform_check["roundtrip_relative_linf"],
            "limit": cfg.gate_transform_roundtrip,
        },
        "transform_parseval_relative_error": {
            "value": transform_check["parseval_relative_error"],
            "limit": cfg.gate_transform_parseval,
        },
        "mass_relative_drift": {
            "value": mass_drift,
            "limit": cfg.gate_mass_relative_drift,
        },
        "total_water_relative_drift": {
            "value": water_drift,
            "limit": cfg.gate_total_water_relative_drift,
        },
        "mass_fixer_max_step_log_offset": {
            "value": trackers["maximum_mass_fixer_log_offset"],
            "limit": fixer_absorption_limits(cfg.precision)["mass_log_offset"],
        },
        "water_fixer_max_step_relative": {
            "value": trackers["maximum_global_water_fixer_kg_m2"]
            / max(abs(target_water), 1.0e-30),
            "limit": fixer_absorption_limits(cfg.precision)["water_relative"],
        },
        "physics_water_repair_max_step_kg_m2": {
            "value": trackers["maximum_physics_water_repair_kg_m2"],
            # The config-level override exists for suites with a measured
            # kernel-internal closure floor (see _FIXER_ABSORPTION_LIMITS);
            # unset, the precision envelope stands.
            "limit": (
                fixer_absorption_limits(cfg.precision)["physics_repair_kg_m2"]
                if cfg.gate_physics_water_repair_max_step_kg_m2 is None
                else cfg.gate_physics_water_repair_max_step_kg_m2
            ),
        },
        "positivity_fixer_max_step_relative": {
            "value": supplementary["maximum_positivity_fixer_relative"],
            "limit": (
                POSITIVITY_FIXER_RELATIVE_LIMIT
                if cfg.gate_positivity_fixer_max_step_relative is None
                else cfg.gate_positivity_fixer_max_step_relative
            ),
        },
    }
    if "wind_magnitude_relative_error" in final_diag:
        gates["wind_magnitude_relative_error"] = {
            "value": final_diag["wind_magnitude_relative_error"],
            "limit": WIND_MAGNITUDE_RELATIVE_LIMIT,
        }
    if cfg.integrator in SEMILAG_INTEGRATORS:
        # Only a run that HAS a semi-Lagrangian tracer fixer carries these
        # rows, so every receipt written by every other integrator keeps
        # the gate inventory it had.  Two rows, because the fixer has two
        # separable failures and one number cannot see both.  The water
        # row is the conservation gate and names the breakage: a tracer
        # scheme that is not conservative silently loses or gains water
        # over a forecast day, the class of defect the grid tracers were
        # introduced to retire.  The per-species row is an accuracy gate
        # and names a different one: a species whose field owes more to
        # the fixer's weighting than to the flow forecasts a plausible
        # amount of the wrong thing in the wrong place.
        gates["semilag_tracer_mass_fixer_max_step_water_relative"] = {
            "value": supplementary[
                "maximum_semilag_tracer_mass_fixer_water_relative"
            ],
            "limit": SEMILAG_TRACER_FIXER_WATER_RELATIVE_LIMIT,
        }
        gates["semilag_tracer_mass_fixer_max_step_relative"] = {
            "value": supplementary["maximum_semilag_tracer_mass_fixer_relative"],
            "limit": SEMILAG_TRACER_FIXER_RELATIVE_LIMIT,
        }
        gates["semilag_trajectory_convergence_cells"] = {
            "value": supplementary["maximum_semilag_trajectory_move_cells"],
            "limit": float(cfg.semilag.trajectory_convergence_cells),
        }
        gates["semilag_lipschitz"] = {
            "value": supplementary["maximum_semilag_lipschitz"],
            "limit": float(cfg.maximum_lipschitz),
        }
    for row in gates.values():
        row["direction"] = "ceiling"
        row["passed"] = bool(float(row["value"]) <= float(row["limit"]))
    # Gate WIRE-1.  It exists on a MULTI-CARD run only, because a single
    # card moves no bytes and a floor on zero would fail every run that
    # never opened a socket.
    #
    # The breakage it prevents, named: every interconnect figure this
    # design was priced on was taken on IDLE cards (3.09 GB/s MEASURED),
    # and the whole two-card case rests on the wire keeping up with the
    # compute.  MEASURED 2026-09-06 on the T255 ten-step probe, the rate
    # ACHIEVED while both cards compute fell to 1.82 to 2.31 GB/s across
    # four rank-legs -- a quarter below the priced rate, and three of the
    # four below this floor.  Without this row a run reports a speedup
    # priced on a link it did not get, and the receipt says "pass".
    wire = dict((cards or {}).get("wire") or {})
    if int((cards or {}).get("cards", 1) or 1) > 1 and wire:
        achieved = float(wire.get("achieved_gb_s", 0.0))
        floor = float(wire.get("wire_floor_gb_s", 0.0))
        gates["two_card_wire_achieved_gb_s"] = {
            "value": achieved,
            "limit": floor,
            "direction": "floor",
            "passed": bool(achieved >= floor),
        }
    # A gather run reproduces one card only where its cards return the same
    # bits for every contraction the step presents.  Under
    # card_agreement="record" a disagreeing pair carries on and this row is
    # what says the answer is neither card's: the count of disagreeing
    # shapes, ceiling zero.  Under "refuse" the run never reaches a receipt
    # with a disagreement in it, so the row reads zero and passes.
    agreement = dict((cards or {}).get("card_agreement") or {})
    if int((cards or {}).get("cards", 1) or 1) > 1 and agreement:
        disagreeing = len(agreement.get("disagreements") or [])
        gates["two_card_contractions_agree"] = {
            "value": disagreeing,
            "limit": 0,
            "direction": "ceiling",
            "passed": bool(disagreeing == 0),
        }
    return gates


__all__ = [
    "build_model_and_cold_state", "build_physics", "build_transform",
    "run", "run_gates",
]
