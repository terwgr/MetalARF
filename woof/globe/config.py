"""Strict TOML configuration for the moist/hybrid WOOF global prototype."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import tomllib

import numpy as np

from woof.globe.spectral.legendre import DEFAULT_BAND

from .device_memory import DEVICE_ALLOCATORS

from .constants import (
    DEFAULT_EULERIAN_STEP_RULE_FRACTION,
    EARTH_RADIUS_M,
    RESEARCH_ACKNOWLEDGEMENT,
    RESEARCH_ACKNOWLEDGEMENTS,
    RUN_SCHEMA,
)
from .insitu.options import InsituOptions, insitu_options_from_table
from .physics.reference import ReferencePhysicsOptions
from .imex import IMEX_INTEGRATORS
from .semi_implicit import SEMI_IMPLICIT_SCHEMES, build_semi_implicit
from .semilag.options import (
    SemiLagrangianOptions,
    semilag_options_from_table,
)
from .semilag.step import SEMILAG_INTEGRATORS

#: [physics] split names (dynamics.MoistHybridModel.physics_split).
PHYSICS_SPLITS = ("strang", "merged")
#: [initial] mode names: the smooth analytic planet, a decoded analysis,
#: and the Jablonowski and Williamson 2006 baroclinic wave (testcases.py).
INITIAL_MODES = ("analytic", "analysis", "baroclinic_wave")

#: [memory] whether the persistent grid state lives in the pinned host
#: tier: "auto" parks the minimum the card needs, "on" parks all three
#: slices, "off" parks none (spill.py).
HOST_SPILL_MODES = ("auto", "on", "off")
#: [memory] the widest field stack one transform call may carry
#: (dynamics.MoistHybridModel.spectral_chunk).  Six is the shipped value
#: the T533 archives were written under.
DEFAULT_SPECTRAL_CHUNK = 6

#: The allocator a bare ``woof global run`` spends its device bytes
#: through.  MEASURED 2026-09-06 on an RTX 5070 Ti, the CuPy pool the
#: process already carried held the least over what the run had live and
#: cost the least wall on both shapes measured (T85 reference 36 steps:
#: 1.195 held over live and 5.88 s, against the slab's 1.424 and 6.85 s
#: and the driver async pool's 1.53 and 6.33 s; T255 native ten steps:
#: 1.222, against the slab's 1.367 and the async pool's 1.197), so it
#: stays the default and the other two are selectable.  See
#: ArwenGlobalConfig.device_allocator and
#: woof.globe.bands.SlabAllocator.
DEFAULT_DEVICE_ALLOCATOR = "default"
#: The shipped card exchange.  "gather" ships waist ROWS and keeps the
#: Legendre contraction whole, so two cards return one card's bits;
#: "partial" ships partial Legendre sums, is a change of arithmetic and
#: carries its own pin.
DEFAULT_CARD_EXCHANGE = "gather"
#: The shipped multi-card axis.  "band" partitions grid space by latitude
#: (the speed axis); "order" partitions the Legendre orders (the capacity
#: axis for a truncation past the whole-table wall).
DEFAULT_CARD_AXIS = "band"
CARD_AXES = ("band", "order")
#: What a gather run above one card does when its cards return different
#: bits for a contraction the step presents.  "refuse" (the default) stops
#: the run by name at that contraction, before the waist it feeds is
#: assembled from both cards' rows.  "record" carries on, lists every
#: disagreeing shape in the receipt and FAILS the run's
#: two_card_contractions_agree gate row: the run's answer is then neither
#: card's and the receipt says so.  It exists for one purpose, a timing of
#: a pair of unlike cards whose bits do not agree (the RTX 5070 Ti plus RTX
#: 5090 pair on the shipped core, 2026-09-07), and it is a workaround, not
#: a result: no run under it is a two-card run of record.
DEFAULT_CARD_AGREEMENT = "refuse"
CARD_AGREEMENTS = ("refuse", "record")
#: The deep halo a card boundary exchanges for the meridional sweep.
DEFAULT_CARD_HALO_ROWS = 16

#: Shipped hyperdiffusion: exponent ``order`` on n(n+1)/N(N+1) and the
#: e-folding time of the coefficients at the truncation
#: (woof.globe.spectral.diffusion; kinetic energy decays at twice the
#: rate, divergence carries strength 1.5).  Order 8 at 36 min (2026-09-04)
#: replaced order 4 at 4 h after the energy ledger read the control's
#: 250 hPa tail (T255, hours 12-24): the resolved cascade fed the last
#: fifteen degrees at 11 to 16 times their energy per day against a
#: diffusion drain of 9 to 15 and a physics drain of 0.7, so the tail sat
#: flat at 0.22 to 0.27 of the observed spectrum and the northern
#: effective resolution could not be read, while at n = 161 (250 km) the
#: diffusion took 0.31 per day beside the physics' 1.1 and the cascade
#: fed nothing.  Unit-strength energy drain per day, shipped against the
#: former shape: n = 255 80 against 12, n = 245 42 against 8.7, n = 235
#: 22 against 6.3, n = 220 7.6 against 3.7, n = 210 3.6 against 2.6,
#: n = 202 equal, n = 200 1.65 against 1.73, n = 161 0.052 against 0.31
#: (e-folding 462 h against 78 h at 250 km).
DEFAULT_DIFFUSION_ORDER = 8
DEFAULT_DIFFUSION_EFOLD_S = 2160.0
#: The semi-Lagrangian core's own [diffusion] defaults, read when a config
#: under integrator = "sl_si" omits the key.  The Eulerian defaults above
#: were tuned on 2026-09-04 at a 60 s step against the energy ledger; a
#: semi-Lagrangian gather is itself a filter applied once per step whatever
#: the step is, so the same explicit drain on top of it over-damps.  Set
#: by the coupling lane's dry ladder (MEASURED 2026-09-07, T255 L40, one
#: dry forecast day from the GDAS 2026-09-01 00Z analysis, every arm at
#: dt = 300 s against the Eulerian core at 30 s, tools/semilag_coupling_arms
#: .py and tools/semilag_scale_diagnostics.py): the 500 hPa vorticity power
#: kept in the bands 61-120 / 121-180 / 181-230 / 231-255 was 0.835 / 0.576
#: / 0.474 / 0.502 on the cubic gather at order 8 and 2,160 s, 0.969 /
#: 0.863 / 0.766 / 0.700 on the quintic gather at the same drain, and
#: 0.973 / 0.915 / 1.025 / 0.965 on the quintic gather at order 16 and
#: 720 s, where two Eulerian steps differ from each other by 0.001 /
#: 0.001 / 0.003 / 0.014; the half-variance degree reads 36.1 against the
#: reference's 36.6 (31.3 on the cubic gather).  Order 16 bites only at
#: the truncation (equal to order 8 at 2,160 s at n = 0.87 T, weaker
#: below it, three times stronger at n = T), which is where the quintic
#: gather leaves power the Eulerian core's own drain removes.
SEMILAG_DIFFUSION_ORDER = 16
SEMILAG_DIFFUSION_EFOLD_S = 720.0
#: [diffusion] closure: which operator drains the truncation.
#: "hyperdiffusion" is the shipped fixed-shape drain above; the selectable
#: "spectral_eddy_viscosity" (woof.globe.spectral.eddy_viscosity,
#: 2026-09-08) reads the model's own kinetic energy at the truncation
#: every step and drains at the eddy viscosity closure theory (EDQNM,
#: Chollet-Lesieur) assigns to that energy: no e-folding time is chosen.
#: Its constants (plateau 0.267, cusp 9.21 at decay 3.03, eddy Prandtl
#: 0.6, five tail degrees) are the theory's and join the config identity
#: only when the closure is selected, so every hyperdiffusion config hash
#: is unchanged.
DEFAULT_DIFFUSION_CLOSURE = "hyperdiffusion"
DIFFUSION_CLOSURES = ("hyperdiffusion", "spectral_eddy_viscosity")
#: The semi-Lagrangian core's [semi_implicit] off_centring_weight when the
#: key is omitted: the 0.55 every shipped sl_si config carries and every
#: graded arm ran at (a two-time-level trajectory scheme needs the
#: decentring; the ladder at 0.50, 0.53 and 0.60 is in
#: configs/verify/..._sl_si_24h_alpha*.toml, written and not graded).
#: Every other integrator keeps the 0.5 it always read.
SEMILAG_OFF_CENTRING_WEIGHT = 0.55
#: [time] integrator when the key is omitted (the ruling of 2026-09-06):
#: the semi-Lagrangian core is the core every lane builds on, runs, tests
#: and grades with, and the shipped default of `woof global run` and
#: `woof global da` at every truncation.  MEASURED 2026-09-06 (the grade
#: lane, equal card-time on the RTX 5090, the same stations and sites):
#: sl_si at T383 against imex_ssp3 at T255 at 120 s grades 5 rows better,
#: 4 worse and 9 level, and a T255 forecast day costs 4.3 min on the RTX
#: 5070 Ti against 18.3 at the Eulerian core's 60 s.  imex_ssp3 stays
#: selectable by name at its rule step (default_eulerian_step_s: 90 s at
#: T255, 60 s at T383, 40 s at T533) with its ten-step identity pinned
#: (pins.DEFAULT_INTEGRATOR keeps naming it: that is the arithmetic a
#: checkpoint written before the key existed is read under, not the
#: door's default).
DEFAULT_TIME_INTEGRATOR = "sl_si"
#: [time] dt_s when the key is omitted under the semi-Lagrangian core:
#: 300 s is the step the core exists to take at T255 and T383 (its
#: Lipschitz number reads 0.23 of its 0.75 gate on the day of record),
#: chosen on the observation scorecard over its dt ladder (the grade of
#: 2026-09-06).  An omitted step under an Eulerian integrator reads the
#: CFL rule below (default_eulerian_step_s), not a fixed number.
DEFAULT_SEMILAG_STEP_S = 300.0
#: [time] integrator names: the two explicit steppers of the split era,
#: the IMEX tableaux of imex.py, and the two-time-level semi-Lagrangian
#: semi-implicit core of semilag/.
TIME_INTEGRATORS = ("ssprk3", "rk4", *IMEX_INTEGRATORS, *SEMILAG_INTEGRATORS)
#: The core a config that omits [time] integrator runs: the shipped default
#: of `woof global run` and `woof global da` at every truncation since
#: 2026-09-06, the two-time-level semi-Lagrangian semi-implicit core at its
#: 300 s step (DEFAULT_SEMILAG_STEP_S).  Graded at equal cost against the
#: Eulerian core on the observation scorecard (MEASURED 2026-09-06: sl_si at
#: T383 and 300 s against imex_ssp3 at T255 and 120 s, one forecast day from
#: the GDAS 2026-09-01 00Z analysis, eighteen rows, paired bootstrap at the
#: 0.03 bar) it reads five rows better, four worse and nine level at about a
#: quarter of the Eulerian wall per forecast day; the rows it loses (18Z
#: sea-level pressure, 00Z 2 m dewpoint, 12Z 500 hPa height, 00Z 850 hPa
#: temperature) are named on the door page.  The Eulerian core stays
#: selectable by name at its rule step (default_eulerian_step_s: 90 s at
#: T255, 60 s at T383, 40 s at T533) with its ten-step identity pinned.
#: imex.DEFAULT_INTEGRATOR is a different thing and does not move: the
#: Eulerian pair's name as the library default of the pins, checkpoint and
#: migration signatures, under which every checkpoint hash written before
#: the semi-Lagrangian core existed was computed.  SHIPPED_INTEGRATOR and
#: DEFAULT_TIME_INTEGRATOR are one value under two names (the two lanes
#: that made the core the default named it from each side); both stay.
SHIPPED_INTEGRATOR = DEFAULT_TIME_INTEGRATOR
#: [time] maximum_lipschitz default.  The trajectory map folds above one:
#: two arrival points share a departure point, the fixed-point search
#: stops contracting and the interpolation samples a folded field, and
#: the model does not blow up when this happens, it silently mislocates.
#: 0.75 keeps 25 percent of margin to folding and about 2.7x to the loss
#: of contraction of the trapezoidal iteration, whose derivative is
#: dt |grad V| / 2.  MEASURED on a real T255 state (2026-09-06, forecast
#: time 1500 s): Frobenius 0.33 at dt = 300 s and 0.99 at 900 s.
DEFAULT_MAXIMUM_LIPSCHITZ = 0.75
#: The Eulerian core's shipped step is chosen against its own refusal with a
#: margin.  The refusal is the spectral CFL dt |V|max sqrt(N(N+1))/a against
#: [time] maximum_cfl (0.75); the shipped step is the largest whole step (a
#: multiple of 5 s that divides the hour, so the six-hour output cadence and
#: the radiation bucket land on whole steps) whose CFL on the STRONGEST
#: analysis day on disk stays at or under DEFAULT_EULERIAN_STEP_RULE_FRACTION
#: (constants.py, 0.70) of the gate.  A day stronger than any on disk is
#: refused by the gate with the receipt's ``cfl`` block saying what step it
#: would have admitted.
#:
#: The strongest implied maximum wind on disk, m/s, per truncation: the day's
#: maximum spectral CFL read back through the gate's own formula, so it is
#: the wind the gate saw and not a grid-point maximum.  MEASURED 2026-09-06
#: (tools/arwen_global_cfl_sweep.py): nine analyses, every GDAS cycle on the
#: nodes from 2026-08-30 18Z to 2026-09-02 00Z and the 2026-09-21 18Z
#: medicane start, each run for a whole forecast day at T255 on the Eulerian
#: core with the whole native suite; on every day the maximum is the
#: analysis's own jet at step 1 and the flow relaxes to 0.37 to 0.46 of it by
#: hour 6, so the T383 and T533 rows are one-step probes of the same nine
#: analyses (checked against a whole day at T383 on 2026-09-01 00Z, whose
#: maximum is also step 1).  The strongest day at T255 is 2026-08-31 00Z.
#: A truncation that is not listed takes the nearest listed wind, because
#: the wind is the flow's and the grid changes it by percents (the measured
#: spread across the listed rows is in the sweep table).
EULERIAN_STEP_WIND_CEILING_M_S: dict[int, float] = {
    255: 136.9,
    383: 139.6,
    533: 141.3,
}
#: The candidate steps the rule chooses among.
EULERIAN_STEP_CANDIDATES_S = tuple(
    float(s) for s in range(5, 3601, 5) if 3600 % s == 0
)


def eulerian_wind_ceiling_m_s(truncation: int) -> float:
    """The strongest wind on disk the shipped step is set against, for
    this truncation: the listed row, or the nearest listed one."""
    if truncation in EULERIAN_STEP_WIND_CEILING_M_S:
        return EULERIAN_STEP_WIND_CEILING_M_S[truncation]
    nearest = min(EULERIAN_STEP_WIND_CEILING_M_S, key=lambda n: abs(n - truncation))
    return EULERIAN_STEP_WIND_CEILING_M_S[nearest]


def default_eulerian_step_s(truncation: int, maximum_cfl: float = 0.75,
                            rule_fraction: float = DEFAULT_EULERIAN_STEP_RULE_FRACTION,
                            wind_m_s: float | None = None) -> float:
    """The largest candidate step whose spectral CFL at the wind ceiling
    stays at or under ``rule_fraction * maximum_cfl`` at this truncation.

    The rule the shipped configs are set by, in one place, so a config that
    omits ``[time] dt_s`` on an Eulerian integrator gets the same step the
    shipped config of its truncation carries, and a reader can recompute
    both from the wind ceiling."""
    if wind_m_s is None:
        wind_m_s = eulerian_wind_ceiling_m_s(truncation)
    rate = wind_m_s * math.sqrt(truncation * (truncation + 1)) / EARTH_RADIUS_M
    bound = rule_fraction * maximum_cfl
    admitted = [s for s in EULERIAN_STEP_CANDIDATES_S if s * rate <= bound + 1.0e-12]
    if not admitted:
        raise ValueError(
            f"no candidate step keeps T{truncation} under {bound:.3f} at "
            f"{wind_m_s:.1f} m/s; set [time] dt_s explicitly"
        )
    return max(admitted)
from .statics import STATICS_SOURCES, StaticsOptions, parse_utc
from .vertical import HybridCoordinate

#: [vertical] coordinate names and the constructor each selects.
VERTICAL_COORDINATES = {
    "surface_stretched": HybridCoordinate.surface_stretched,
    "pressure_blend": HybridCoordinate.pressure_blend,
    # The surface_stretched stack with its 120 to 400 hPa band re-laid at
    # 23.4 hPa per layer (48 levels): selectable, graded against the
    # 40-level default on the T255 control case (2026-09-05).
    "jet_refined": HybridCoordinate.jet_refined,
}
#: The level count a [vertical] table gets when it names a coordinate and
#: no nlev: the count each layout was designed and graded at.
VERTICAL_DEFAULT_NLEV = {
    "surface_stretched": 40,
    "pressure_blend": 6,
    "jet_refined": 48,
}


@dataclass(frozen=True)
class ArwenGlobalConfig:
    name: str
    acknowledgement: str
    backend: str
    precision: str
    initial_mode: str
    analysis_grib: str | None
    analysis_mapping: str | None
    truncation: int
    nlat: int | None
    nlon: int | None
    dealias_factor: float
    dt_s: float
    duration_s: float
    output_interval_s: float
    maximum_cfl: float
    integrator: str
    a_half_pa: tuple[float, ...]
    b_half: tuple[float, ...]
    surface_pressure_pa: float
    surface_temperature_k: float
    top_temperature_k: float
    qv_surface: float
    zonal_wind_m_s: float
    perturbation_amplitude: float
    zonal_wavenumber: int
    terrain_amplitude_m: float
    surface_water_kg_m2: float
    physics_mode: str
    native_adapter_name: str | None
    native_adapter_options: dict[str, object]
    reference_physics: ReferencePhysicsOptions
    # [sponge] Top-of-model graded wave absorber, a dycore option read in
    # EVERY physics mode (the suite-v7 reference-physics copy protected
    # only reference mode; the arwen-native T255 run died at hour 6.55 on
    # the 140 K research bound without it).  base_pa=0 disables.
    sponge_base_pa: float
    sponge_lid_relaxation_time_s: float
    diffusion_enabled: bool
    diffusion_order: int
    diffusion_efold_s: float
    diffusion_preserve_degree: int
    divergence_diffusion_strength: float
    pressure_diffusion_strength: float
    water_diffusion_strength: float
    moment_diffusion_strength: float
    semi_implicit_enabled: bool
    external_wave_speed_m_s: float
    semi_implicit_weight: float
    mass_fixer: bool
    water_fixer: bool
    positivity_repair: bool
    gate_transform_roundtrip: float
    gate_transform_parseval: float
    gate_mass_relative_drift: float
    gate_total_water_relative_drift: float
    # Opt-in TF32 tensor-core compute for the Legendre contractions on the
    # cupy backend.  TF32 truncates the float32 mantissa to 10 bits inside the
    # GEMM, so enabling it CHANGES NUMERICS and needs its own transform gate
    # re-measurement on the GPU before any production default flips.
    tensor_core_contractions: bool = False
    # Per-config override for the physics_water_repair_max_step_kg_m2 gate.
    # None keeps the precision-wide reference-derived envelope
    # (runner.fixer_absorption_limits).  A run config sets a value only when
    # its physics suite carries a MEASURED closure floor that envelope was
    # never sized for -- the arwen-native suite's kernel-internal channels
    # (Morrison's rainncv-vs-column-debit closure and the negative-species
    # clamp) are genuine fp32 accumulation the reference suite does not have
    # -- and the config comment must quote the measurement and the sizing
    # arithmetic (the gate law: thresholds cite measurement).
    gate_physics_water_repair_max_step_kg_m2: float | None = None
    # Per-config override for the positivity_fixer_max_step_relative gate
    # (the v5 atmosphere-internal water fixer's per-step magnitude relative
    # to the global-mean atmospheric water column).  None keeps the
    # measured default in runner.POSITIVITY_FIXER_RELATIVE_LIMIT; a config
    # that sets a value must quote the measurement it is sized against.
    gate_positivity_fixer_max_step_relative: float | None = None
    # [vertical] coordinate: which constructor laid out a_half_pa/b_half
    # ("surface_stretched", "pressure_blend", or "explicit" for arrays
    # given in the TOML).  A label only: the A/B arrays are the identity,
    # so it is dropped from config_identity and every archive written
    # under pressure_blend keeps its hash.
    vertical_coordinate: str = "surface_stretched"
    # [semi_implicit] scheme: "vertical_modes" (default: every vertical
    # gravity-wave mode implicit through the reference-state operator,
    # audit 2026-09-01 DN-1) or "external" (the barotropic proxy of the
    # earlier era).  The scheme and its three parameters join the identity
    # only under vertical_modes: a config that selects "external" keeps
    # the hash it had when that was the only scheme, so archives written
    # under it stay readable, while the two schemes never share a hash.
    semi_implicit_scheme: str = "vertical_modes"
    semi_implicit_reference_temperature_k: float = 320.0
    semi_implicit_reference_surface_pressure_pa: float = 100_000.0
    semi_implicit_off_centring_weight: float = 0.5
    # [physics] split: how the physics suite is composed with the dynamics
    # over one step.  "strang" (default): the suite over dt/2 before and
    # after the dynamics, the symmetric split every archive so far was
    # written under.  "merged": the suite once per step, over the full dt,
    # after the dynamics (dynamics.MoistHybridModel.physics_split).  Joins
    # the identity only when merged, so every pre-existing config hash
    # (and every written checkpoint and receipt) stays byte-identical.
    physics_split: str = "strang"
    # [insitu] ledger cadences.  Diagnostic only: dropped from the identity
    # so a run watched and a run unwatched share one config hash (and one
    # checkpoint lineage), which is exactly the bit-identity the ledger
    # proves about itself.
    insitu: InsituOptions = InsituOptions()
    # [initial] analysis_fill_grib / analysis_fill_mapping: a second
    # analysis of the SAME valid time that supplies the surface groups the
    # primary product lacks (analysis_initial.FILL_GROUPS: soil, snow, sea
    # ice), each group taken whole from one source and every field's
    # source named in the receipt.  The IFS open-data product carries no
    # sea-ice concentration and no snow depth; the GDAS analysis of the
    # same hour does.  Both join the identity only when set, so every
    # config hash written without them (and every checkpoint and receipt)
    # stays byte-identical.
    analysis_fill_grib: str | None = None
    analysis_fill_mapping: str | None = None
    # [statics] surface static fields.  The real arm changes results (the
    # planet Noah runs on) and enters the identity as its source, dataset
    # tokens and, for the analytic planet, its date; the synthetic arm is
    # the pre-existing constant planet and enters nothing, so every config
    # hash written before the table existed is unchanged.  The DATA hash
    # of a real cache is not part of the identity: the fields are copied
    # into the surface state and travel with every checkpoint, so a restart
    # carries the statics it was started with rather than re-reading a
    # cache; the receipt records the cache's own hashes beside the run.
    statics: StaticsOptions = StaticsOptions()
    # [time] maximum_lipschitz and the [semilag] table: read only under
    # the semi-Lagrangian integrator, appended at the END of the
    # dataclass with defaults so no existing positional index moves (the
    # tree's own precedent: a field inserted mid-dataclass in another
    # package shifted 150 positional indices), and deleted from the
    # identity under every other integrator so every config hash written
    # before this lane -- and every checkpoint and receipt bound to one --
    # stays byte-identical.
    maximum_lipschitz: float = DEFAULT_MAXIMUM_LIPSCHITZ
    semilag: SemiLagrangianOptions = SemiLagrangianOptions()
    # [memory] the four engine memory levers.  Each one exists in the
    # engine and had no door until now: runner.build_transform passed
    # neither transform lever, and build_model_and_cold_state passed
    # neither model lever, so a run could not reach any of them.
    #
    # spectral_chunk: the widest field stack one transform call carries
    # (dynamics.MoistHybridModel._chunked).  It is the Legendre GEMM's M
    # dimension and BLAS blocks on M, so whether a width moves bits
    # depends on the library and the vertical ladder.  MEASURED
    # 2026-09-06: on the card (RTX 5070 Ti, cupy float32) every width from
    # 1 to 12 is bit-exact at forty levels at T255 and T533; on numpy
    # float64 at T21 the SYNTHESIS moves 1 ulp (2.22e-16 against values of
    # order 1) at two, five and ten levels under numpy 2.2.6 and at five
    # levels under 2.5.2, and holds at forty on both.  A config can ask
    # for a short ladder -- pressure_blend ships with six levels -- so the
    # chunk is carried in the identity at every value other than the
    # shipped six rather than resting on one host's BLAS.
    spectral_chunk: int = DEFAULT_SPECTRAL_CHUNK
    # synthesis_memo: serve repeated syntheses of one state within a step
    # from a memo instead of recomputing them.  Same bits either way
    # (dynamics.MoistHybridModel.synthesis_memo, and the shipped
    # test_the_synthesis_memo_is_bit_neutral gate), so it is dropped from
    # the identity at every value: a watched-memory run and an unwatched
    # one share a config hash and therefore a checkpoint lineage.
    synthesis_memo: bool = True
    # legendre_band: orders per packed Legendre band.  ARITHMETIC on the
    # cupy backend: MEASURED 2026-09-06 on an RTX 5070 Ti at T255 float32,
    # the analysis of a single plane differs by 6.2e-06 between band 32
    # and bands 8 and 16 and its synthesis by 1.4e-05, and a ten-step T255
    # native run at band 16 left 66 of the 125 checkpoint state arrays
    # differing from the band-32 run, entering at step 0 through
    # forward(log(ps)),
    # which is the one field the dycore analyses as a single plane.  The
    # band is carried whenever it is not the shipped 32, and the
    # transform's own identity carries it on the same rule.
    legendre_band: int = DEFAULT_BAND
    # streaming: hold no Legendre table and regenerate the basis one band
    # of orders at a time.  MEASURED bit-neutral 2026-09-06 (RTX 5070 Ti,
    # T255 float32: the streamed analysis and synthesis of a single plane
    # and of a forty-level stack are byte-identical to the resident-table
    # calls), so absent from both identities at every value.  It is the
    # entry point for a truncation whose tables do not fit, and it is
    # slow: a streamed synthesis costs 37.7x a resident-table call at T383
    # and a streamed analysis 15,144x (MEASURED 2026-09-06).
    streaming: bool = False
    # device_allocator: which allocator the run spends its device bytes
    # through -- "default" (the shipped default) leaves CuPy's own memory
    # pool as the process found it, which MEASURED 2026-09-06 held the
    # least over what the run had live and cost the least wall; "slab"
    # takes one contiguous arena before the first model byte and cuts it
    # with an exact-fit coalescing free list; "async" installs the
    # driver's cudaMallocAsync pool.  Memory only: an
    # address does not change a floating-point result, and the ten-step
    # T255 checkpoint gate is run under each, so it is dropped from the
    # identity at every value and a run that moves it shares a config
    # hash, a checkpoint lineage and a receipt with one that does not.
    device_allocator: str = DEFAULT_DEVICE_ALLOCATOR
    # latitude_bands: how many latitude bands grid space is streamed
    # through.  Zero means the auto-sizer chooses (1 when the resident run
    # fits the card, otherwise the smallest count that does), which is
    # what makes a bare run at a truncation too large for the card start
    # rather than die in the allocator.
    #
    # It is a STREAMING GRANULARITY, NOT A DECOMPOSITION.  Spectral space
    # stays whole and the Legendre contraction keeps K = N = nlat at every
    # band count, so the band count changes no arithmetic and enters no
    # identity: a banded run shares a config hash, a checkpoint lineage
    # and a receipt with the resident run and reproduces its checkpoints
    # byte for byte (gates BIT-1 to BIT-4).  The precedent is
    # legendre_band and streaming, absent from transform.identity for the
    # same reason.
    latitude_bands: int = 0
    # host_spill: whether the persistent grid state -- the ten grid
    # tracers, the surface reservoirs and the native physics namespace --
    # lives in PINNED HOST MEMORY instead of on the card, reaching the
    # card only as the copy its consumer builds anyway (spill.py).
    #
    # "auto" (the default) parks the minimum: nothing while the card holds
    # the run, and then one slice at a time, coldest first, until the
    # predicted peak fits.  "on" parks all three; "off" parks none and is
    # the pre-tier run exactly.
    #
    # Memory only, and therefore outside the identity at every value: a
    # parked array holds the same bytes in host memory that it held on the
    # card, a staged copy is a memcpy, and every kernel reads the same
    # values.  The ten-step T255 checkpoint gate is run with it forced on
    # against the resident run (gate BIT-1, spill on), so a spilled run
    # shares a config hash, a checkpoint lineage and a receipt with the
    # run that does not spill.
    host_spill: str = "auto"
    # cards: how many GPUs share the band schedule.  ONE is the shipped
    # default and the only value that needs no launcher.  Above one, grid
    # space is partitioned by band and spectral space is replicated, and
    # the two meet at the Fourier waist (cards.py).
    #
    # In the default 'gather' exchange the waist ROWS cross the wire and
    # the Legendre contraction still runs at K = N = nlat on every card,
    # so the answer is the single-card answer bit for bit and the card
    # count enters no identity (gates BIT-5 and BIT-6).  In the
    # 'partial' exchange each card contracts its own rows and the partial
    # sums are added in rank order, which IS a change of arithmetic: it
    # carries its own pin, and both it and the card count join the
    # identity whenever it is selected.
    cards: int = 1
    card_exchange: str = DEFAULT_CARD_EXCHANGE
    # The multi-card AXIS: which of the two decompositions a run above one
    # card uses (cards.py).  "band" (the default) partitions GRID space by
    # latitude and gathers waist ROWS, keeping the Legendre contraction
    # whole -- the SPEED axis, and the only one the model step runs
    # end-to-end today.  "order" partitions the Legendre ORDERS and gathers
    # coefficient COLUMNS, holding one rank's fraction of the table -- the
    # CAPACITY axis for a truncation whose whole table no longer fits one
    # card.  Both are bit-identical to one card (BIT-5); the axis changes
    # who computes an order or a row, never what it is, so it enters no
    # identity.  The order axis is proven at the transform
    # (SphericalHarmonicTransform.forward_orders / inverse_orders) and is
    # refused for a model RUN by name until a truncation that needs it has a
    # card that fits it (runner).
    card_axis: str = DEFAULT_CARD_AXIS
    # What a disagreeing pair does: refuse (default) or record and fail the
    # gate row.  A measurement device, absent from the identity always.
    card_agreement: str = DEFAULT_CARD_AGREEMENT
    # This rank, and the rendezvous address of every rank in rank order.
    # Launcher wiring, not arithmetic: absent from the identity always.
    card_rank: int = 0
    card_addresses: tuple = ()
    card_transport: str = "auto"
    # One relative speed per rank, in rank order, or empty to MEASURE them
    # at run start.  Assignment, not arithmetic: no identity.
    card_weights: tuple = ()
    # Latitude rows exchanged across a card boundary for the meridional
    # sweep's deep halo.  It must cover 2n for the step's sub-step count n
    # and a run that needs more is refused by name rather than swept from
    # stale neighbour rows.  Layout, not arithmetic: no identity.
    card_halo_rows: int = DEFAULT_CARD_HALO_ROWS
    # [diffusion] closure and the constants of the spectral eddy viscosity
    # (DEFAULT_DIFFUSION_CLOSURE).  At the default closure none of the six
    # joins the identity; a selected closure carries all six.
    diffusion_closure: str = DEFAULT_DIFFUSION_CLOSURE
    closure_plateau: float = 0.267
    closure_cusp_amplitude: float = 9.21
    closure_cusp_decay: float = 3.03
    closure_eddy_prandtl: float = 0.6
    closure_tail_degrees: int = 5

    @property
    def config_identity(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reference_physics"] = asdict(self.reference_physics)
        # native_adapter_options holds every normalized option (the suite
        # is built from it); the hash carries the adapter's identity of
        # them, which drops an option wherever it changes no arithmetic or
        # spells the arithmetic every earlier checkpoint was written under
        # (woof.globe.physics.registry, options_identity).
        if self.physics_mode == "arwen-native" and self.native_adapter_name:
            from woof.globe.physics.registry import (
                identity_of_global_physics_options,
            )
            from .physics.builtin_adapters import ensure_builtin_global_physics_adapters

            ensure_builtin_global_physics_adapters()
            payload["native_adapter_options"] = identity_of_global_physics_options(
                self.native_adapter_name, self.native_adapter_options
            )
        del payload["vertical_coordinate"]
        del payload["insitu"]
        statics_identity = self.statics.identity
        if statics_identity is None:
            del payload["statics"]
        else:
            payload["statics"] = statics_identity
        # tensor_core_contractions joins the identity only when enabled: TF32
        # changes the arithmetic, so an enabled run must not share a hash
        # with the fp32 run it diverges from, while the disabled default
        # keeps every pre-existing config hash (and therefore every written
        # checkpoint and receipt) byte-identical.
        if not payload["tensor_core_contractions"]:
            del payload["tensor_core_contractions"]
        # The spectral eddy viscosity joins the identity only when selected
        # (it replaces the drain, so a run under it must not share a hash
        # with the hyperdiffusion run); the shipped closure keeps every
        # pre-existing config hash byte-identical.
        if payload["diffusion_closure"] == DEFAULT_DIFFUSION_CLOSURE:
            for key in (
                "diffusion_closure", "closure_plateau", "closure_cusp_amplitude",
                "closure_cusp_decay", "closure_eddy_prandtl", "closure_tail_degrees",
            ):
                del payload[key]
        # Same hash-stability contract: only a config that SETS the physics
        # repair gate override carries it in its identity, so every
        # pre-existing config hash (and every written checkpoint and
        # receipt) stays byte-identical while unset.
        if payload["gate_physics_water_repair_max_step_kg_m2"] is None:
            del payload["gate_physics_water_repair_max_step_kg_m2"]
        if payload["gate_positivity_fixer_max_step_relative"] is None:
            del payload["gate_positivity_fixer_max_step_relative"]
        if payload["physics_split"] == "strang":
            del payload["physics_split"]
        # The two memory levers MEASURED bit-neutral leave the identity at
        # EVERY value, so a run that moves them shares a config hash, a
        # checkpoint lineage and a receipt with the run that does not:
        # synthesis_memo by the shipped bit-neutrality gate and by a
        # ten-step T255 native A/B whose 126 checkpoint arrays, metadata
        # included, are byte-identical; streaming by the streamed-against-
        # resident compare of 2026-09-06.
        #
        # The other two move the Legendre GEMM's M dimension or its batch
        # count and are MEASURED arithmetic, so each is carried whenever
        # it is not its shipped value and every config hash written before
        # this table existed stays byte-identical.
        del payload["synthesis_memo"]
        del payload["streaming"]
        del payload["device_allocator"]
        # The band count is a streaming granularity: the Legendre
        # contraction runs at K = N = nlat whatever it is, so it leaves
        # the identity at EVERY value and every config hash written
        # before this field existed stays byte-identical.
        del payload["latitude_bands"]
        # The host tier is a place, not an arithmetic: same bytes, same
        # kernels, same values.  Out of the identity at every value, so
        # every config hash written before this field existed stays
        # byte-identical and a spilled run resumes a resident lineage.
        del payload["host_spill"]
        # The card count and the wiring are outside the arithmetic in the
        # bit-identical exchange, and gate BIT-6 is the measurement that
        # says so: three band-to-card assignments at one band count return
        # one answer.  The APPROXIMATE exchange is not, so it and the card
        # count it was taken on stay in the hash whenever it is selected,
        # and a checkpoint written under it refuses to resume under the
        # other by its own pin.
        del payload["card_rank"]
        del payload["card_addresses"]
        del payload["card_transport"]
        del payload["card_weights"]
        del payload["card_halo_rows"]
        # The axis is layout, not arithmetic: both axes are bit-identical to
        # one card, so a run switching axes must keep every hash it had.
        del payload["card_axis"]
        del payload["card_agreement"]
        if payload["card_exchange"] == DEFAULT_CARD_EXCHANGE:
            del payload["card_exchange"]
            del payload["cards"]
        if payload["spectral_chunk"] == DEFAULT_SPECTRAL_CHUNK:
            del payload["spectral_chunk"]
        if payload["legendre_band"] == DEFAULT_BAND:
            del payload["legendre_band"]
        if payload["analysis_fill_grib"] is None:
            del payload["analysis_fill_grib"]
            del payload["analysis_fill_mapping"]
        if payload["integrator"] not in SEMILAG_INTEGRATORS:
            del payload["maximum_lipschitz"]
            del payload["semilag"]
        else:
            payload["semilag"] = self.semilag.identity
        if payload["semi_implicit_scheme"] == "external":
            for key in (
                "semi_implicit_scheme",
                "semi_implicit_reference_temperature_k",
                "semi_implicit_reference_surface_pressure_pa",
                "semi_implicit_off_centring_weight",
            ):
                del payload[key]
        return payload

    @property
    def config_hash(self) -> str:
        raw = json.dumps(
            self.config_identity, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    @property
    def vertical(self) -> HybridCoordinate:
        return HybridCoordinate(
            np.asarray(self.a_half_pa, dtype=np.float64),
            np.asarray(self.b_half, dtype=np.float64),
        )


def _table(raw: dict, name: str) -> dict:
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"[{name}] must be a TOML table")
    return value


def _unknown(table: dict, allowed: set[str], name: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in [{name}]: {', '.join(unknown)}")


def _finite(value, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _integer(value, name: str) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _boolean(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _multiple(total: float, step: float, name: str) -> None:
    ratio = total / step
    if abs(ratio - round(ratio)) > 1.0e-10 * max(1.0, abs(ratio)):
        raise ValueError(f"{name} must be a whole multiple of dt_s")


def load_config(path: str | Path) -> ArwenGlobalConfig:
    source = Path(path)
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    allowed_tables = {
        "arwen_global", "grid", "time", "vertical", "initial", "physics",
        "reference_physics", "sponge", "diffusion", "semi_implicit",
        "repair", "gates", "insitu", "statics", "semilag", "memory",
    }
    unknown_tables = sorted(set(raw) - allowed_tables)
    if unknown_tables:
        raise ValueError(f"unknown top-level tables: {', '.join(unknown_tables)}")

    top = _table(raw, "arwen_global")
    _unknown(top, {
        "schema", "name", "acknowledgement", "backend", "precision",
        "tensor_core_contractions",
    }, "arwen_global")
    if top.get("schema") != RUN_SCHEMA:
        raise ValueError(f"arwen_global.schema must be {RUN_SCHEMA!r}")
    name = str(top.get("name", "arwen-global-run")).strip()
    if not name:
        raise ValueError("arwen_global.name may not be empty")
    acknowledgement = str(top.get("acknowledgement", ""))
    if acknowledgement not in RESEARCH_ACKNOWLEDGEMENTS:
        raise ValueError(
            "arwen_global.acknowledgement must be exactly "
            f"{RESEARCH_ACKNOWLEDGEMENTS[1]!r} (or the earlier {RESEARCH_ACKNOWLEDGEMENTS[0]!r})"
        )
    backend = str(top.get("backend", "numpy")).lower()
    precision = str(top.get("precision", "float64")).lower()
    if backend not in {"numpy", "cupy"}:
        raise ValueError("backend must be 'numpy' or 'cupy'")
    if precision not in {"float32", "float64"}:
        raise ValueError("precision must be 'float32' or 'float64'")
    tensor_core = _boolean(
        top.get("tensor_core_contractions", False),
        "arwen_global.tensor_core_contractions",
    )
    if tensor_core and backend != "cupy":
        raise ValueError(
            "arwen_global.tensor_core_contractions=true requires "
            "backend='cupy': the numpy backend has no tensor-core path, so "
            "the flag would stamp a TF32 config identity onto arithmetic "
            "that never changes"
        )
    if tensor_core and precision != "float32":
        raise ValueError(
            "arwen_global.tensor_core_contractions=true requires "
            "precision='float32': TF32 acts on float32 GEMMs only, so under "
            "float64 the flag would move the config hash while every "
            "contraction stays fp64"
        )

    grid = _table(raw, "grid")
    _unknown(grid, {"truncation", "nlat", "nlon", "dealias_factor"}, "grid")
    truncation = _integer(grid.get("truncation", 7), "grid.truncation")
    # The old 255 ceiling guarded the (2m-1)!! Legendre overflow, retired by
    # the normalized recurrence (bounded coefficients by construction;
    # roundtrip measured 3.2e-7 fp32 at T533 on the RTX 5090, 2026-08-31).
    # The per-run gate_transform_roundtrip check is the instrument that
    # admits a truncation; this range only rejects nonsense input.
    if not 3 <= truncation <= 2047:
        raise ValueError("grid.truncation must lie in 3..2047")
    nlat = None if grid.get("nlat") is None else _integer(grid["nlat"], "grid.nlat")
    nlon = None if grid.get("nlon") is None else _integer(grid["nlon"], "grid.nlon")
    dealias = _finite(grid.get("dealias_factor", 1.5), "grid.dealias_factor")
    if dealias < 1.0:
        raise ValueError("grid.dealias_factor must be >= 1")

    time = _table(raw, "time")
    _unknown(time, {
        "dt_s", "duration_s", "output_interval_s", "maximum_cfl",
        "integrator", "maximum_lipschitz",
    }, "time")
    # The semi-Lagrangian core is the default (DEFAULT_TIME_INTEGRATOR,
    # the ruling of 2026-09-06).  The IMEX pair of imex.py is the Eulerian
    # core, selectable by name: a balanced state is an exact fixed point
    # of its step (the split steppers drift 8e-3 m/s and 0.25 K per hour
    # at rest over a 2 km mountain at dt = 60 s) and its linearized rest
    # ceilings and Doppler growth measured better than the split's on
    # every instrument (imex.py).  "ssprk3" and "rk4" keep the split of
    # their era, bit-identical.
    integrator = str(time.get("integrator", DEFAULT_TIME_INTEGRATOR)).lower()
    if integrator not in TIME_INTEGRATORS:
        raise ValueError(
            "time.integrator must be one of "
            + ", ".join(repr(name) for name in TIME_INTEGRATORS)
        )
    semilagrangian = integrator in SEMILAG_INTEGRATORS
    maximum_cfl = _finite(time.get("maximum_cfl", 0.75), "time.maximum_cfl")
    # A config that omits dt_s gets the shipped step of its integrator and
    # its truncation: the semi-Lagrangian core's 300 s, the Eulerian core's
    # from the CFL rule (default_eulerian_step_s: the largest whole step
    # keeping the strongest analysis day on disk under 0.70 of the gate,
    # 90 s at T255, 60 s at T383, 40 s at T533).
    if "dt_s" in time:
        dt = _finite(time["dt_s"], "time.dt_s")
    elif semilagrangian:
        dt = DEFAULT_SEMILAG_STEP_S
    elif 0.0 < maximum_cfl <= 1.0:
        dt = default_eulerian_step_s(truncation, maximum_cfl)
    else:
        dt = 60.0  # the gate is refused just below; the step is moot
    duration = _finite(time.get("duration_s", 1800.0), "time.duration_s")
    output = _finite(time.get("output_interval_s", duration), "time.output_interval_s")
    if dt <= 0.0 or duration <= 0.0 or output <= 0.0:
        raise ValueError("time values must be positive")
    _multiple(duration, dt, "time.duration_s")
    _multiple(output, dt, "time.output_interval_s")
    if output > duration:
        raise ValueError("output_interval_s cannot exceed duration_s")
    if not 0.0 < maximum_cfl <= 1.0:
        raise ValueError("time.maximum_cfl must lie in (0,1]")
    maximum_lipschitz = _finite(
        time.get("maximum_lipschitz", DEFAULT_MAXIMUM_LIPSCHITZ),
        "time.maximum_lipschitz",
    )
    if "maximum_lipschitz" in time and not semilagrangian:
        raise ValueError(
            "time.maximum_lipschitz is read only by the semi-Lagrangian "
            f"integrator ({', '.join(SEMILAG_INTEGRATORS)}); this run "
            f"integrates {integrator!r}, whose step is bounded by the "
            "advective CFL and not by the flow deformation.  A key that is "
            "parsed and silently ignored produces the same run whether it "
            "is set or not, and the operator has no way to learn which "
            "happened, so it is refused rather than dropped"
        )
    if not 0.0 < maximum_lipschitz <= 1.0:
        raise ValueError(
            "time.maximum_lipschitz must lie in (0, 1]: above one the "
            "trajectory map is not invertible at all, so two arrival "
            "points share a departure point and the interpolation samples "
            "a folded field"
        )

    vertical = _table(raw, "vertical")
    _unknown(
        vertical, {"a_half_pa", "b_half", "nlev", "p_top_pa", "coordinate"}, "vertical"
    )
    have_arrays = "a_half_pa" in vertical or "b_half" in vertical
    if have_arrays:
        if "a_half_pa" not in vertical or "b_half" not in vertical:
            raise ValueError("vertical.a_half_pa and vertical.b_half must be supplied together")
        if "nlev" in vertical or "p_top_pa" in vertical or "coordinate" in vertical:
            raise ValueError(
                "explicit hybrid arrays cannot be combined with nlev/p_top_pa/coordinate"
            )
        if not isinstance(vertical["a_half_pa"], list) or not isinstance(vertical["b_half"], list):
            raise ValueError("hybrid A/B values must be TOML arrays")
        coordinate = HybridCoordinate(
            np.asarray([_finite(v, "vertical.a_half_pa[]") for v in vertical["a_half_pa"]]),
            np.asarray([_finite(v, "vertical.b_half[]") for v in vertical["b_half"]]),
        )
        coordinate_name = "explicit"
    else:
        # Default surface_stretched at 40 levels (fixed means default): the
        # pressure_blend grid put the lowest full level ~378 m AGL at 20
        # levels, and every surface scheme assumes tens of metres.
        # pressure_blend stays selectable so archives hashed under it stay
        # readable; its nlev default is unchanged for the same reason.
        coordinate_name = str(vertical.get("coordinate", "surface_stretched")).lower()
        if coordinate_name not in VERTICAL_COORDINATES:
            raise ValueError(
                "vertical.coordinate must be one of "
                + ", ".join(repr(name) for name in VERTICAL_COORDINATES)
            )
        default_nlev = VERTICAL_DEFAULT_NLEV[coordinate_name]
        nlev = _integer(vertical.get("nlev", default_nlev), "vertical.nlev")
        p_top = _finite(vertical.get("p_top_pa", 100.0), "vertical.p_top_pa")
        if not 2 <= nlev <= 64:
            raise ValueError("vertical.nlev must lie in 2..64")
        coordinate = VERTICAL_COORDINATES[coordinate_name](nlev, p_top)

    initial = _table(raw, "initial")
    initial_mode = str(initial.get("mode", "analytic")).lower()
    # "baroclinic_wave" is the Jablonowski and Williamson 2006 state
    # (testcases.py): an analytically specified, balanced, steady flow
    # with a published answer, which is what makes it a gate rather than
    # a demonstration.  Its only shape parameter is the amplitude of the
    # 1 m/s wind bump, so the case's steady arm and its perturbed arm are
    # the same config with that one number at zero.
    if initial_mode not in INITIAL_MODES:
        raise ValueError(
            "initial.mode must be one of "
            + ", ".join(repr(name) for name in INITIAL_MODES)
        )
    analysis_grib = None
    analysis_mapping = None
    analysis_fill_grib = None
    analysis_fill_mapping = None
    if initial_mode == "analysis":
        # Analytic shape parameters are refused here rather than ignored, so
        # a config cannot claim a jet or a wavenumber the run never builds.
        _unknown(initial, {
            "mode", "analysis_grib", "analysis_mapping", "surface_water_kg_m2",
            "analysis_fill_grib", "analysis_fill_mapping",
        }, "initial")
        analysis_grib = str(initial.get("analysis_grib", "")).strip()
        if not analysis_grib:
            raise ValueError(
                "initial.mode='analysis' requires initial.analysis_grib"
            )
        fill_grib = str(initial.get("analysis_fill_grib", "") or "").strip()
        fill_mapping = str(initial.get("analysis_fill_mapping", "") or "").strip()
        if bool(fill_grib) != bool(fill_mapping):
            raise ValueError(
                "initial.analysis_fill_grib and initial.analysis_fill_mapping "
                "name one secondary analysis together: the fill reads the "
                "second product through its own mapping and a GRIB without a "
                "mapping (or a mapping without a GRIB) cannot be decoded"
            )
        analysis_fill_grib = fill_grib or None
        analysis_fill_mapping = fill_mapping or None
        # The default is the id of the ONE packaged mapping whose target is
        # this model's initial condition, not the source family.  "gdas" was
        # unique when it was written and now globs three authority mappings
        # (the global analysis, the regional pgrb2 profile and its donor), so
        # every config that left this key out refused before step zero --
        # `analysis mapping id 'gdas' must match exactly one authority
        # mapping, found 3`.  A default that cannot resolve is not a default.
        analysis_mapping = str(
            initial.get("analysis_mapping", "gdas-global")).strip()
        if not analysis_mapping:
            raise ValueError("initial.analysis_mapping may not be empty")
    elif initial_mode == "baroclinic_wave":
        # Every other analytic shape parameter is refused rather than
        # ignored: the case's temperature, jet and orography are the
        # case's, and a config that set them would read as describing a
        # state the run never builds.
        _unknown(initial, {"mode", "perturbation_amplitude"}, "initial")
    else:
        _unknown(initial, {
            "mode",
            "surface_pressure_pa", "surface_temperature_k", "top_temperature_k",
            "qv_surface", "zonal_wind_m_s", "perturbation_amplitude",
            "zonal_wavenumber", "terrain_amplitude_m", "surface_water_kg_m2",
        }, "initial")
    ps = _finite(initial.get("surface_pressure_pa", 100_000.0), "initial.surface_pressure_pa")
    ts = _finite(initial.get("surface_temperature_k", 288.0), "initial.surface_temperature_k")
    tt = _finite(initial.get("top_temperature_k", 215.0), "initial.top_temperature_k")
    qv = _finite(initial.get("qv_surface", 0.010), "initial.qv_surface")
    wind = _finite(initial.get("zonal_wind_m_s", 8.0), "initial.zonal_wind_m_s")
    perturb = _finite(initial.get("perturbation_amplitude", 0.001), "initial.perturbation_amplitude")
    wave = _integer(initial.get("zonal_wavenumber", 3), "initial.zonal_wavenumber")
    terrain = _finite(initial.get("terrain_amplitude_m", 500.0), "initial.terrain_amplitude_m")
    surface_water = _finite(initial.get("surface_water_kg_m2", 200.0), "initial.surface_water_kg_m2")
    if not 50_000.0 <= ps <= 120_000.0:
        raise ValueError("initial.surface_pressure_pa must lie in 50000..120000")
    if not 150.0 <= ts <= 350.0 or not 150.0 <= tt <= 350.0:
        raise ValueError("initial temperatures must lie in 150..350 K")
    if not 0.0 <= qv <= 0.05:
        raise ValueError("initial.qv_surface must lie in 0..0.05")
    if not 0 <= wave <= truncation:
        raise ValueError("initial.zonal_wavenumber must lie in 0..truncation")
    if terrain < 0.0 or surface_water < 0.0:
        raise ValueError("terrain amplitude and surface water must be nonnegative")

    physics = _table(raw, "physics")
    _unknown(physics, {"mode", "native_adapter_name", "native_adapter_options", "split"}, "physics")
    physics_mode = str(physics.get("mode", "reference")).lower()
    if physics_mode not in {"reference", "none", "arwen-native"}:
        raise ValueError("physics.mode must be reference, none, or arwen-native")
    physics_split = str(physics.get("split", "strang")).lower()
    if physics_split not in PHYSICS_SPLITS:
        raise ValueError(
            f"physics.split must be one of {', '.join(PHYSICS_SPLITS)}, got "
            f"{physics_split!r}"
        )
    adapter_name = physics.get("native_adapter_name")
    if adapter_name is not None:
        adapter_name = str(adapter_name).strip().lower()
    options = physics.get("native_adapter_options", {})
    if not isinstance(options, dict):
        raise ValueError("physics.native_adapter_options must be an inline table")
    if physics_mode == "arwen-native" and not adapter_name:
        raise ValueError("arwen-native physics requires physics.native_adapter_name")
    if physics_mode != "arwen-native" and (adapter_name or options):
        raise ValueError("native adapter settings are read only in arwen-native mode")
    if physics_mode == "arwen-native":
        # Registration and option validation are deliberately pure and happen
        # before a transform or CuPy device is touched. Unknown adapters win
        # the refusal order even if backend/precision are also malformed.
        from .physics.builtin_adapters import ensure_builtin_global_physics_adapters
        from woof.globe.physics.registry import validate_global_physics_options

        ensure_builtin_global_physics_adapters()
        normalized = validate_global_physics_options(adapter_name, dict(options))
        # The validator returns every normalized option (the build payload);
        # the hash payload is the adapter's identity of them, computed in
        # config_identity, and an identity may drop a key whose value is
        # the state older checkpoints were written in
        # (gf_updraft_only_when_downdraft_dry = false, YSU's "wrf-layer"
        # length).  The runtime must read the value the operator wrote: a
        # config that says false and runs true is a different model than
        # its own text, and the first dissection arm of the timing lane ran
        # exactly that (bit for bit the switched-on arm), as a config
        # spelling "wrf-layer" once ran "fixed".  So the config carries the
        # normalized defaults with the operator's own keys laid over them.
        options = {**normalized, **dict(options)}
        if backend != "cupy" or precision != "float32":
            raise ValueError(
                "WOOF native physics requires backend='cupy' and precision='float32'"
            )

    reference = _table(raw, "reference_physics")
    # Retired in the v7 -> v8 suite bump: the top-of-model absorber moved
    # to the dycore [sponge] section so it acts in every physics mode.  A
    # config still steering it through [reference_physics] must refuse by
    # name, not fall into the generic unknown-key message: an operator who
    # believes these keys still configure the absorber would otherwise
    # retune a control that no longer exists, and a config that silently
    # dropped them would run the very unsponged configuration that killed
    # both T533 runs at the 140 K research gate (hours 3.0-3.5, twice).
    retired_sponge = sorted(
        {"sponge_base_pa", "sponge_lid_relaxation_time_s"} & set(reference)
    )
    if retired_sponge:
        raise ValueError(
            "[reference_physics] keys "
            + ", ".join(retired_sponge)
            + " are retired: the top-of-model absorber is dycore-owned "
            "since suite v8 and is configured as [sponge] base_pa / "
            "lid_relaxation_time_s (it now acts in every physics mode, "
            "not only reference)"
        )
    allowed_reference = set(ReferencePhysicsOptions.__dataclass_fields__)
    _unknown(reference, allowed_reference, "reference_physics")
    reference_options = ReferencePhysicsOptions(**reference)
    if physics_mode != "reference" and reference:
        raise ValueError("[reference_physics] is read only when physics.mode='reference'")

    sponge = _table(raw, "sponge")
    _unknown(sponge, {"base_pa", "lid_relaxation_time_s"}, "sponge")
    sponge_base = _finite(sponge.get("base_pa", 5000.0), "sponge.base_pa")
    sponge_lid_tau = _finite(
        sponge.get("lid_relaxation_time_s", 900.0),
        "sponge.lid_relaxation_time_s",
    )
    # The v7 option semantics, unchanged by the move: zero is the off
    # switch; a negative base selects no ring while reading as a
    # configured absorber - a run believed sponged then dies exactly like
    # the unsponged T533 runs (139.6-140.0 K at hours 3.0-3.5, twice).
    if sponge_base < 0.0:
        raise ValueError(
            "sponge.base_pa must be nonnegative: a negative base selects "
            "no ring while reading as a configured absorber (zero is the "
            "off switch)"
        )
    # A zero lid time divides the Rayleigh rate by zero; a negative one
    # turns the per-step factor exp(-dt/tau) into growth, so the absorber
    # would amplify the near-truncation wave train it exists to remove.
    if sponge_lid_tau <= 0.0:
        raise ValueError(
            "sponge.lid_relaxation_time_s must be positive: a nonpositive "
            "lid time makes exp(-dt/tau) amplify the wave anomalies "
            "instead of damping them"
        )

    diffusion = _table(raw, "diffusion")
    _unknown(diffusion, {
        "enabled", "order", "e_folding_time_s_at_truncation", "preserve_degree",
        "divergence_strength", "pressure_strength", "water_strength",
        "moment_strength", "closure", "plateau", "cusp_amplitude", "cusp_decay",
        "eddy_prandtl", "tail_degrees",
    }, "diffusion")
    diff_enabled = _boolean(diffusion.get("enabled", True), "diffusion.enabled")
    diff_closure = str(diffusion.get("closure", DEFAULT_DIFFUSION_CLOSURE)).lower()
    if diff_closure not in DIFFUSION_CLOSURES:
        raise ValueError(
            f"diffusion.closure must be one of {DIFFUSION_CLOSURES}, got {diff_closure!r}"
        )
    closure_plateau = _finite(diffusion.get("plateau", 0.267), "diffusion.plateau")
    closure_cusp_amplitude = _finite(diffusion.get("cusp_amplitude", 9.21), "diffusion.cusp_amplitude")
    closure_cusp_decay = _finite(diffusion.get("cusp_decay", 3.03), "diffusion.cusp_decay")
    closure_eddy_prandtl = _finite(diffusion.get("eddy_prandtl", 0.6), "diffusion.eddy_prandtl")
    closure_tail_degrees = _integer(diffusion.get("tail_degrees", 5), "diffusion.tail_degrees")
    if diff_closure == DEFAULT_DIFFUSION_CLOSURE:
        for key in ("plateau", "cusp_amplitude", "cusp_decay", "eddy_prandtl", "tail_degrees"):
            if key in diffusion:
                raise ValueError(
                    f"diffusion.{key} belongs to closure = \"spectral_eddy_viscosity\"; "
                    "the hyperdiffusion reads order and e_folding_time_s_at_truncation"
                )
    else:
        if closure_plateau <= 0.0 or closure_cusp_decay <= 0.0 or closure_eddy_prandtl <= 0.0:
            raise ValueError("diffusion.plateau, cusp_decay and eddy_prandtl must be positive")
        if closure_cusp_amplitude < 0.0:
            raise ValueError("diffusion.cusp_amplitude must be nonnegative")
        if not 1 <= closure_tail_degrees <= truncation:
            raise ValueError("diffusion.tail_degrees must lie in 1..truncation")
        for key in ("order", "e_folding_time_s_at_truncation"):
            if key in diffusion:
                raise ValueError(
                    f"diffusion.{key} has no effect under closure = \"spectral_eddy_viscosity\": "
                    "the drain is set by the resolved energy at the truncation; remove the key"
                )
    # The defaults follow the integrator: an omitted key under sl_si reads
    # the semi-Lagrangian core's own values (SEMILAG_DIFFUSION_*), and under
    # every other integrator exactly what it always read.
    default_order = SEMILAG_DIFFUSION_ORDER if semilagrangian else DEFAULT_DIFFUSION_ORDER
    default_efold = SEMILAG_DIFFUSION_EFOLD_S if semilagrangian else DEFAULT_DIFFUSION_EFOLD_S
    diff_order = _integer(diffusion.get("order", default_order), "diffusion.order")
    diff_efold = _finite(diffusion.get("e_folding_time_s_at_truncation", default_efold), "diffusion.e_folding_time_s_at_truncation")
    diff_preserve = _integer(diffusion.get("preserve_degree", 1), "diffusion.preserve_degree")
    div_strength = _finite(diffusion.get("divergence_strength", 1.5), "diffusion.divergence_strength")
    p_strength = _finite(diffusion.get("pressure_strength", 0.25), "diffusion.pressure_strength")
    water_strength = _finite(diffusion.get("water_strength", 0.5), "diffusion.water_strength")
    moment_strength = _finite(diffusion.get("moment_strength", 0.75), "diffusion.moment_strength")
    # The number moments are grid tracers (2026-09-02) with no spectral
    # coefficients to diffuse; the key stays in the identity at its
    # default so every pre-existing config hash is unchanged, and a
    # non-default value is refused because it would read as a tuned
    # dissipation that nothing applies.
    if "moment_strength" in diffusion and moment_strength != 0.75:
        raise ValueError(
            "diffusion.moment_strength has no effect: the number moments "
            "are grid-point tracers carried by the positive-definite "
            "transport, whose limiter is their only dissipation; remove the "
            "key (its default 0.75 is kept only for config-hash stability)"
        )
    if diff_order < 1 or diff_efold <= 0.0 or not 0 <= diff_preserve <= truncation:
        raise ValueError("invalid diffusion order/e-fold/preserve settings")
    if min(div_strength, p_strength, water_strength, moment_strength) < 0.0:
        raise ValueError("diffusion strengths must be nonnegative")

    implicit = _table(raw, "semi_implicit")
    _unknown(implicit, {
        "enabled", "scheme", "external_wave_speed_m_s", "weight",
        "reference_temperature_k", "reference_surface_pressure_pa",
        "off_centring_weight",
    }, "semi_implicit")
    implicit_enabled = _boolean(implicit.get("enabled", True), "semi_implicit.enabled")
    implicit_scheme = str(implicit.get("scheme", "vertical_modes")).lower()
    if implicit_scheme not in SEMI_IMPLICIT_SCHEMES:
        raise ValueError(
            "semi_implicit.scheme must be one of "
            + ", ".join(repr(name) for name in SEMI_IMPLICIT_SCHEMES)
        )
    wave_speed = _finite(implicit.get("external_wave_speed_m_s", 300.0), "semi_implicit.external_wave_speed_m_s")
    implicit_weight = _finite(implicit.get("weight", 1.0), "semi_implicit.weight")
    if wave_speed <= 0.0 or not 0.0 <= implicit_weight <= 1.0:
        raise ValueError("invalid semi-implicit wave speed or weight")
    implicit_t_ref = _finite(
        implicit.get("reference_temperature_k", 320.0),
        "semi_implicit.reference_temperature_k",
    )
    implicit_ps_ref = _finite(
        implicit.get("reference_surface_pressure_pa", 100_000.0),
        "semi_implicit.reference_surface_pressure_pa",
    )
    # An omitted off_centring_weight follows the integrator too: the
    # semi-Lagrangian core reads the 0.55 its graded arms ran at
    # (SEMILAG_OFF_CENTRING_WEIGHT), every other integrator the centred
    # 0.5 it always read.
    implicit_alpha = _finite(
        implicit.get("off_centring_weight", SEMILAG_OFF_CENTRING_WEIGHT if semilagrangian else 0.5),
        "semi_implicit.off_centring_weight",
    )
    # The scheme object owns the refusals (a weight below 0.5 amplifies
    # every implicit mode; the reference bounds are the research bounds);
    # constructing it here surfaces them at load time with their reasons.
    build_semi_implicit(
        implicit_scheme, enabled=implicit_enabled, weight=implicit_weight,
        external_wave_speed_m_s=wave_speed,
        reference_temperature_k=implicit_t_ref,
        reference_surface_pressure_pa=implicit_ps_ref,
        off_centring_weight=implicit_alpha,
    )
    if semilagrangian:
        # Three pairings the semi-Lagrangian core refuses by name, each
        # because the trajectory would be stable and the gravity waves
        # would not.  A 300 s step has no explicit budget for a 350 m/s
        # mode: at T255 its Courant number is 6.7 and at T533 it is 9.7.
        if implicit_scheme != "vertical_modes":
            raise ValueError(
                f"[semi_implicit] scheme = {implicit_scheme!r} cannot run "
                f"under integrator = {integrator!r}: the barotropic proxy "
                "leaves every INTERNAL vertical mode explicit (its measured "
                "linearized rest ceiling was 104.7 s at T533 whatever its "
                "reference speed), and the semi-Lagrangian step this "
                "integrator exists to take is 300 s"
            )
        if not implicit_enabled:
            raise ValueError(
                f"[semi_implicit] enabled = false cannot run under "
                f"integrator = {integrator!r}: with the linear operator "
                "off, the two-time-level step is an explicit trapezoidal "
                "one and every gravity-wave mode rides the explicit "
                "budget, which a semi-Lagrangian step is chosen precisely "
                "to leave"
            )
        if implicit_weight != 1.0:
            raise ValueError(
                f"[semi_implicit] weight = {implicit_weight:g} cannot run "
                f"under integrator = {integrator!r}: the fraction left "
                "explicit is a gravity wave of speed "
                f"sqrt(1 - {implicit_weight:g}) times the fastest mode, and "
                "at a 300 s step there is no explicit budget for any part "
                "of it"
            )

    semilag_table = _table(raw, "semilag")
    if semilag_table and not semilagrangian:
        raise ValueError(
            "the [semilag] table is read only by the semi-Lagrangian "
            f"integrator ({', '.join(SEMILAG_INTEGRATORS)}); this run "
            f"integrates {integrator!r} and would silently ignore every "
            "key in it, so it is refused rather than dropped"
        )
    semilag = semilag_options_from_table(semilag_table)

    repair = _table(raw, "repair")
    _unknown(repair, {"mass_fixer", "water_fixer", "positivity_repair"}, "repair")
    mass_fixer = _boolean(repair.get("mass_fixer", True), "repair.mass_fixer")
    water_fixer = _boolean(repair.get("water_fixer", True), "repair.water_fixer")
    positivity = _boolean(repair.get("positivity_repair", True), "repair.positivity_repair")

    gates = _table(raw, "gates")
    _unknown(gates, {
        "transform_roundtrip_relative_linf", "transform_parseval_relative_error",
        "mass_relative_drift", "total_water_relative_drift",
        "physics_water_repair_max_step_kg_m2",
        "positivity_fixer_max_step_relative",
    }, "gates")
    default_transform = 5.0e-11 if precision == "float64" else 5.0e-5
    gate_roundtrip = _finite(gates.get("transform_roundtrip_relative_linf", default_transform), "gates.transform_roundtrip_relative_linf")
    gate_parseval = _finite(gates.get("transform_parseval_relative_error", default_transform), "gates.transform_parseval_relative_error")
    gate_mass = _finite(gates.get("mass_relative_drift", 1.0e-9), "gates.mass_relative_drift")
    gate_water = _finite(gates.get("total_water_relative_drift", 1.0e-8), "gates.total_water_relative_drift")
    gate_physics_repair = gates.get("physics_water_repair_max_step_kg_m2")
    if gate_physics_repair is not None:
        gate_physics_repair = _finite(
            gate_physics_repair, "gates.physics_water_repair_max_step_kg_m2"
        )
    gate_positivity_fixer = gates.get("positivity_fixer_max_step_relative")
    if gate_positivity_fixer is not None:
        gate_positivity_fixer = _finite(
            gate_positivity_fixer, "gates.positivity_fixer_max_step_relative"
        )
    if min(
        gate_roundtrip, gate_parseval, gate_mass, gate_water,
        *(() if gate_physics_repair is None else (gate_physics_repair,)),
        *(() if gate_positivity_fixer is None else (gate_positivity_fixer,)),
    ) <= 0.0:
        raise ValueError("all gates must be positive")

    insitu = insitu_options_from_table(_table(raw, "insitu"))
    statics = _statics_options(_table(raw, "statics"), initial_mode)

    memory = _table(raw, "memory")
    _unknown(memory, {
        "spectral_chunk", "synthesis_memo", "legendre_band", "streaming",
        "device_allocator", "latitude_bands", "host_spill",
        "cards", "card_exchange", "card_axis", "card_halo_rows",
        "card_transport", "card_agreement",
    }, "memory")
    spectral_chunk = _integer(
        memory.get("spectral_chunk", DEFAULT_SPECTRAL_CHUNK),
        "memory.spectral_chunk",
    )
    if spectral_chunk < 1:
        raise ValueError(
            "memory.spectral_chunk must be >= 1: a chunk of zero or less "
            "would present the transform an empty field stack and the "
            "chunk loop would return nothing rather than the state"
        )
    synthesis_memo = _boolean(
        memory.get("synthesis_memo", True), "memory.synthesis_memo")
    legendre_band = _integer(
        memory.get("legendre_band", DEFAULT_BAND), "memory.legendre_band")
    if legendre_band < 1:
        raise ValueError("memory.legendre_band must be >= 1")
    streaming = _boolean(memory.get("streaming", False), "memory.streaming")
    device_allocator = str(
        memory.get("device_allocator", DEFAULT_DEVICE_ALLOCATOR)
    ).strip().lower()
    if device_allocator not in DEVICE_ALLOCATORS:
        raise ValueError(
            f"memory.device_allocator must be one of {DEVICE_ALLOCATORS}, got "
            f"{memory.get('device_allocator')!r}: an unknown name would leave "
            "the run on whatever allocator the process happened to carry, and "
            "the receipt would name an allocator the run did not use"
        )

    host_spill = str(memory.get("host_spill", "auto"))
    if host_spill not in HOST_SPILL_MODES:
        raise ValueError(
            f"memory.host_spill must be one of {sorted(HOST_SPILL_MODES)}, got "
            f"{memory.get('host_spill')!r}: an unknown name would leave the "
            "run guessing whether its persistent grid state is on the card"
        )
    cards = _integer(memory.get("cards", 1), "memory.cards")
    if cards < 1:
        raise ValueError(
            "memory.cards must be >= 1: a run with no card has nothing to "
            "run on, and one card is the shipped default"
        )
    card_exchange = str(memory.get("card_exchange", DEFAULT_CARD_EXCHANGE)).strip().lower()
    if card_exchange == "partial":
        raise ValueError(
            "memory.card_exchange='partial' is not built.  It is the right "
            "next lever on the WIRE and the wrong one to pull first: "
            "MEASURED 2026-09-06 on a two-card T127 run, the Fourier waist "
            "is 219.9 of 333 MiB posted per card per step (66 percent), so "
            "trading whole waist rows for partial Legendre sums would take "
            "roughly a third off the largest item.  What it does NOT touch "
            "is the ceiling: the physics half-step runs whole on both cards "
            "(banded, but duplicated rather than split, because every "
            "consumer of the surface and the namespace reads a whole plane; "
            "dynamics.whole_globe_slices), which caps a second card near "
            "1.3x however cheap the wire becomes.  Split the physics across "
            "the cards first, then build this with its own pin, and the "
            "refusal comes out"
        )
    if card_exchange != "gather":
        raise ValueError(
            "memory.card_exchange must be 'gather': the waist rows cross the "
            "wire, the Legendre contraction stays whole at K = N = nlat, and "
            f"two cards return the single-card answer bit for bit; got "
            f"{card_exchange!r}"
        )
    card_transport = str(memory.get("card_transport", "auto")).strip().lower()
    if card_transport not in ("auto", "tcp", "nccl"):
        raise ValueError(
            "memory.card_transport must be 'auto', 'tcp' or 'nccl', got "
            f"{card_transport!r}"
        )
    card_halo_rows = _integer(
        memory.get("card_halo_rows", DEFAULT_CARD_HALO_ROWS),
        "memory.card_halo_rows")
    if card_halo_rows < 1:
        raise ValueError(
            "memory.card_halo_rows must be >= 1: the meridional sweep reads "
            "at least two rows past its block, so a card boundary with no "
            "halo would sweep from stale neighbour values"
        )
    card_axis = str(memory.get("card_axis", DEFAULT_CARD_AXIS)).strip().lower()
    if card_axis not in CARD_AXES:
        raise ValueError(
            f"memory.card_axis must be one of {list(CARD_AXES)}, got "
            f"{card_axis!r}: 'band' partitions grid space by latitude and is "
            "the speed axis; 'order' partitions the Legendre orders and is "
            "the capacity axis past the whole-table wall"
        )
    card_agreement = str(
        memory.get("card_agreement", DEFAULT_CARD_AGREEMENT)).strip().lower()
    if card_agreement not in CARD_AGREEMENTS:
        raise ValueError(
            f"memory.card_agreement must be one of {list(CARD_AGREEMENTS)}, "
            f"got {card_agreement!r}: 'refuse' stops a gather run at the first "
            "contraction its cards disagree on; 'record' carries on, lists the "
            "disagreeing shapes in the receipt and fails the run's "
            "two_card_contractions_agree gate row (a timing device for unlike "
            "cards, never a run of record)"
        )
    latitude_bands = _integer(
        memory.get("latitude_bands", 0), "memory.latitude_bands")
    if latitude_bands < 0:
        raise ValueError(
            "memory.latitude_bands must be >= 0: zero asks the sizer to "
            "choose the count, one is the resident run, and a negative "
            "count is not a streaming granularity"
        )

    return ArwenGlobalConfig(
        name=name, acknowledgement=acknowledgement, backend=backend,
        precision=precision, initial_mode=initial_mode,
        analysis_grib=analysis_grib, analysis_mapping=analysis_mapping,
        truncation=truncation, nlat=nlat, nlon=nlon,
        dealias_factor=dealias, dt_s=dt, duration_s=duration,
        output_interval_s=output, maximum_cfl=maximum_cfl,
        integrator=integrator, a_half_pa=tuple(coordinate.a_half_pa),
        b_half=tuple(coordinate.b_half), vertical_coordinate=coordinate_name,
        surface_pressure_pa=ps,
        surface_temperature_k=ts, top_temperature_k=tt, qv_surface=qv,
        zonal_wind_m_s=wind, perturbation_amplitude=perturb,
        zonal_wavenumber=wave, terrain_amplitude_m=terrain,
        surface_water_kg_m2=surface_water, physics_mode=physics_mode,
        physics_split=physics_split,
        native_adapter_name=adapter_name, native_adapter_options=dict(options),
        reference_physics=reference_options,
        sponge_base_pa=sponge_base,
        sponge_lid_relaxation_time_s=sponge_lid_tau,
        diffusion_enabled=diff_enabled,
        diffusion_order=diff_order, diffusion_efold_s=diff_efold,
        diffusion_preserve_degree=diff_preserve,
        divergence_diffusion_strength=div_strength,
        pressure_diffusion_strength=p_strength,
        water_diffusion_strength=water_strength,
        moment_diffusion_strength=moment_strength,
        semi_implicit_enabled=implicit_enabled,
        external_wave_speed_m_s=wave_speed,
        semi_implicit_weight=implicit_weight,
        semi_implicit_scheme=implicit_scheme,
        semi_implicit_reference_temperature_k=implicit_t_ref,
        semi_implicit_reference_surface_pressure_pa=implicit_ps_ref,
        semi_implicit_off_centring_weight=implicit_alpha,
        mass_fixer=mass_fixer,
        water_fixer=water_fixer, positivity_repair=positivity,
        gate_transform_roundtrip=gate_roundtrip,
        gate_transform_parseval=gate_parseval,
        gate_mass_relative_drift=gate_mass,
        gate_total_water_relative_drift=gate_water,
        tensor_core_contractions=tensor_core,
        gate_physics_water_repair_max_step_kg_m2=gate_physics_repair,
        gate_positivity_fixer_max_step_relative=gate_positivity_fixer,
        analysis_fill_grib=analysis_fill_grib,
        analysis_fill_mapping=analysis_fill_mapping,
        insitu=insitu,
        statics=statics,
        maximum_lipschitz=maximum_lipschitz,
        semilag=semilag,
        spectral_chunk=spectral_chunk,
        synthesis_memo=synthesis_memo,
        legendre_band=legendre_band,
        streaming=streaming,
        device_allocator=device_allocator,
        latitude_bands=latitude_bands,
        host_spill=host_spill,
        cards=cards,
        card_exchange=card_exchange,
        card_axis=card_axis,
        card_agreement=card_agreement,
        card_transport=card_transport,
        card_halo_rows=card_halo_rows,
        diffusion_closure=diff_closure,
        closure_plateau=closure_plateau,
        closure_cusp_amplitude=closure_cusp_amplitude,
        closure_cusp_decay=closure_cusp_decay,
        closure_eddy_prandtl=closure_eddy_prandtl,
        closure_tail_degrees=closure_tail_degrees,
    )


def _statics_options(table: dict, initial_mode: str) -> StaticsOptions:
    """The ``[statics]`` table.

    Real statics are the default for an analysis-initialized run (fixed
    means default: the synthetic planet is the +4.2 K / -4.1 K 2 m bias).
    The analytic planet's land mask is a formula with no geography to
    look up, so its default is the synthetic arm; either mode may name
    the other source explicitly, and a synthetic choice is printed at the
    door and recorded in the receipt as synthetic.
    """
    _unknown(table, {
        "source", "geog_root", "geog_data_res", "cache_dir", "valid_time",
    }, "statics")
    declared = "source" in table
    default_source = "real" if initial_mode == "analysis" else "synthetic"
    source = str(table.get("source", default_source)).strip().lower()
    if source not in STATICS_SOURCES:
        raise ValueError(
            "statics.source must be one of "
            + ", ".join(repr(name) for name in STATICS_SOURCES))
    geog_root = table.get("geog_root")
    if geog_root is not None:
        geog_root = str(geog_root).strip()
        if not geog_root:
            raise ValueError("statics.geog_root may not be empty")
    cache_dir = table.get("cache_dir")
    if cache_dir is not None:
        cache_dir = str(cache_dir).strip()
        if not cache_dir:
            raise ValueError("statics.cache_dir may not be empty")
    tokens = str(table.get("geog_data_res", "default")).strip()
    if not tokens:
        raise ValueError("statics.geog_data_res may not be empty")
    from woof.static.build import GeogSelection

    GeogSelection.from_tokens(Path("."), tokens)  # token validation only
    valid_time = table.get("valid_time")
    if valid_time is not None:
        valid_time = str(valid_time).strip()
        parse_utc(valid_time, "statics.valid_time")
        if initial_mode == "analysis":
            raise ValueError(
                "statics.valid_time is read only with initial.mode="
                "'analytic': an analysis-initialized run resolves its "
                "monthly statics at the analysis valid time, so a second "
                "date here would be silently ignored")
    if source == "real" and initial_mode != "analysis" and valid_time is None:
        raise ValueError(
            "statics.source='real' on the analytic planet needs "
            "statics.valid_time (ISO-8601, UTC): the monthly vegetation, "
            "leaf-area and albedo climatologies resolve to a date and the "
            "analytic initial state carries none")
    if source != "real":
        for key in ("geog_root", "geog_data_res", "cache_dir", "valid_time"):
            if key in table:
                raise ValueError(
                    f"statics.{key} is read only with statics.source='real'; "
                    "a synthetic planet reads no archive")
    return StaticsOptions(
        source=source, geog_root=geog_root, geog_data_res=tokens,
        cache_dir=cache_dir, valid_time=valid_time, declared=declared,
    )
