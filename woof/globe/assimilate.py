"""Minutes-fresh point-observation assimilation for WOOF global (v1).

Research-grade v1 of the fast-cycle door: a data-density-normalised
successive correction in grid space against a single deterministic
background.  Per column the increment is ``sum(w g d) / (1 + sum(w g))``
with ``w`` the localization weight and ``g`` the background/observation
error variance ratio; the single-observation limit ``g d / (1 + g)`` is
the optimal-interpolation gain, and co-located stacks saturate toward the
observed value as OI does, but there is no ``[H B H^T + R]^-1`` solve: the
observation-observation coupling is the per-gridpoint scalar denominator.
Where neighbouring innovations disagree the scheme therefore under-fits
them (two reports of opposite sign one length scale apart: 0.200 of the
innovation at the report against OI's 0.522; at half a length scale 0.052
against 0.246, audit 2026-09-01 DA-1), which is the successive-correction
trade: no matrix, any report count.
Surface pressure (via ln ps), temperature (via theta), winds (via
vorticity/divergence) and, with the moisture update selected, water vapor
(via the dewpoint) are updated; condensate and number moments are
untouched.
The moisture update (``MOISTURE_UPDATE_DIVERGENCE``, selectable): a 2 m dewpoint
report is compared against the dewpoint of the model's lowest-level vapor
at the station pressure, the dewpoint innovation is spread by the same
successive correction with its own vertical localization
(``humidity_decay_height_m`` above the column's surface), and at every
level the background dewpoint moves by the spread increment and the vapor
is re-derived through Bolton's saturation vapor pressure, so the specific
humidity increment is the dewpoint increment through the local
Clausius-Clapeyron slope.  The vapor is bounded by saturation at the
analysed temperature and by zero before the increment enters the spectral
basis, and the model's own positivity repair closes what the triangular
truncation rings below zero inside its column.
The wind increment is spread as u and v, analysed into vorticity and
divergence, and applied as its rotational (streamfunction) part only: the
divergent half that independent scalar spreading manufactures is
gravity-wave energy, not an analysis of anything a wind report measured
(``WIND_BALANCE_BREAKAGE``).
Surface reports are compared at their own height: pressure reduced to the
station, temperature to 2 m, wind to the 10 m anemometer by the model's
Monin-Obukhov similarity diagnostic from the lowest full level and the
surface state.
Every increment field passes through the model transform's triangular
truncation on the way in, which is the increment smoothing.

The gate of record is cross-validation on a withheld set: for every
variable with at least ``gate_minimum_count`` accepted reports, a
deterministic, seeded ``withheld_fraction`` of them is held out of the
analysis, and the analysis must fit the WITHHELD reports better than the
background does (O-A rms < O-B rms on the withheld rows).  The fit to the
assimilated rows is reported as a diagnostic only: evaluated on the rows
that built the increment, O-A < O-B holds for any positive gain and says
nothing about over-fitting (``GATE_BREAKAGE``).  An analysis that predicts
reports it never saw worse than the background did is not an analysis,
and the door reports failure.

Every assimilated report's identity (:meth:`ObsRow.identity_hash`) is
written into the receipt and into the analysis checkpoint's physics-state
metadata (``ASSIMILATION_HISTORY_KEY``), so a cycled background carries
its own assimilation chain and a later cycle refuses a report already in
it (``REJECTION_BREAKAGE["already_assimilated"]``).

The global mean of surface pressure is deliberately preserved: the model's
mass fixer owns that mean and would re-absorb any shift on the first restart
step, tripping the fixer-absorption gate; the pressure stream constrains
gradients, not total mass.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from woof.globe.spectral.sampling import sample_scalar, sample_wind

from .checkpoint import read_checkpoint, state_from_checkpoint, write_checkpoint
from .config import ArwenGlobalConfig
from .da_scorecard import Departures, scorecard as da_scorecard
from .constants import (
    ASSIMILATION_SCHEMA,
    DRY_AIR_GAS_CONSTANT,
    EARTH_RADIUS_M,
    GRAVITY_M_S2,
    KAPPA,
    REFERENCE_PRESSURE_PA,
    RESEARCH_ACKNOWLEDGEMENT,
)
from .obs_table import ObsRow, VARIABLE_TABLE, load_obs, parse_valid_time
from .physics.surface_diagnostics import (
    ANEMOMETER_HEIGHT_M,
    effective_surface_humidity,
    similarity_surface_diagnostics,
)
from .pins import pins_hash
from .receipt import canonical
from .runner import build_model_and_cold_state, build_transform
from .state import ArwenGlobalState
from .surface_energy import (
    _BOLTON_A, _BOLTON_B, _BOLTON_C, _EPSILON, dewpoint_from_specific_humidity,
)

# Converts a ln-pressure separation into metres for localization geometry
# only (roughly R_d * 260 K / g, rounded); no thermodynamic quantity is
# derived from it.
LOCALIZATION_SCALE_HEIGHT_M = 7600.0
# Vertical influence of an aloft (aircraft) report: the ADAS/Bratseth
# height-separation correlation model exp(-(dz / zrange)^2) with the
# separation measured in ln p and converted through the scale height
# above.  zrange is ADAS's "Range (m) for vertical correlation model",
# typical 500 < zrange < 1800 m (CAPS-ARPS & ADAS Version 5.0, Supplement
# 1: ADAS, &adas_zrange); 1000 m is the middle of that band and the value
# the audit asked for.  The predecessor was a Gaussian in PRESSURE with
# sigma 15 kPa (audit 2026-09-01, DA-2): a 250 hPa temperature innovation
# landed at 0.616 of itself on the 267 Pa lid of the 20-level stack, 0.62-
# 0.68 across the three sponge levels, because 15 kPa spans the whole
# stratosphere when the stratosphere is 25 kPa deep.  With zrange = 1 km
# the same report reaches 177 hPa upward and 353 hPa downward and nothing
# beyond: the weight at 100/50/10/1 hPa is exactly zero (see
# VERTICAL_WEIGHT_FLOOR).
DEFAULT_AIRCRAFT_LEVEL_SCALE_M = 1000.0
# ADAS zwlim: the smallest vertical correlation allowed to contribute; a
# weight below it is zero, which is the hard vertical cut.  At 1e-3 the
# cut sits at sqrt(ln 1000) = 2.63 zrange = 2.63 km-equivalent = 0.346 in
# ln p.
VERTICAL_WEIGHT_FLOOR = 1.0e-3
# Lapse rate used to move the lowest-model-level temperature to 2 m above
# the station, and the model surface pressure to station elevation.
SURFACE_LAPSE_K_M = 0.0065

# How the analysed wind increment enters the state.  "rotational" keeps
# the vorticity (streamfunction) part and discards the divergence the
# scalar u/v spreading produced; "unconstrained" applies both and is the
# predecessor, kept so the two can be measured against each other.
DEFAULT_WIND_BALANCE = "rotational"
WIND_BALANCE_MODES = ("rotational", "unconstrained")
WIND_BALANCE_BREAKAGE = (
    "spreading u and v as two independent isotropic scalars puts 40-51 % "
    "of the increment's kinetic energy into divergence (T63, audit "
    "2026-09-01 DA-3: 50 random wind reports 0.403, 400 reports 0.512, one "
    "u-only report 0.499 - the Helmholtz split of an isotropic bump in one "
    "component is half and half by symmetry).  That divergence is not "
    "something a wind report measured; it is gravity-wave energy the dycore "
    "keeps: integrated one hour from a wind-only analysis of 80 synthetic "
    "reports, the unconstrained increment's excess divergent KE over the "
    "control was 1.34 J/kg at t=0 and 1.17 J/kg at 1 h on the T3 smoke "
    "config (1.78 and 1.62 J/kg at T21, 8 levels, dt 300 s), the divergence "
    "hyperdiffusion having removed a tenth of it; the rotational projection "
    "of the same increment left 0.043 J/kg (T21: 0.074) after the hour, and "
    "the hour-mean excess rms surface-pressure tendency fell from 1.7e-2 to "
    "8.5e-4 Pa/s (T21: 5.1e-3 to 1.7e-4)."
)

DEFAULT_BACKGROUND_ERRORS = (
    ("surface_pressure_pa", 300.0),
    ("temperature_k", 2.5),
    ("wind_u_m_s", 4.0),
    ("wind_v_m_s", 4.0),
    # The 2 m dewpoint's background error is larger than temperature's:
    # the GDAS-started background read -2.8 K dry against every CONUS
    # station at 18Z and 00Z where the temperature read +0.8 / +1.1 K
    # (2026-09-05 observation scorecard), so the moisture field is the
    # less trusted of the two.
    ("dewpoint_k", 3.0),
    # Refractivity has no operator in this door (the ensemble filter's
    # point operators carry it, woof.globe.da.operators); the
    # entry keeps the vocabulary covered and the rows are refused by name
    # (REJECTION_BREAKAGE "no_operator") before any of them could reach
    # the spreading sums.
    ("refractivity_n", 3.0),
    # A brightness temperature has no operator in this door either: the
    # radiance streams hand the ensemble filter their own batches; a row
    # of it in a table is refused by name before the spreading sums.
    ("brightness_temperature_k", 2.0),
)

#: Variables the neutral table carries that the SUCCESSIVE CORRECTION has
#: no operator for.  Its rows are refused by name at quality control; the
#: ensemble filter's operators evaluate them (the letkf door passes its
#: own operator vocabulary to :func:`_table_quality_control`, so nothing
#: it can evaluate is refused; a radiance stream hands the filter its own
#: batches, so a brightness temperature never rides a neutral table).
VARIABLES_WITHOUT_OPERATOR = frozenset({"refractivity_n", "brightness_temperature_k"})

#: The cross-stream duplicate rule: two streams reporting the same
#: instrument (the METAR archive and the Aviation Weather Center cache
#: carrying one station's report; a WIS2 SYNOP and the same station's
#: METAR) are one report.  Two rows of one variable at one level are the
#: same instrument when they share a station id (case and whitespace
#: apart) or sit within a neighbouring position cell of this size (the
#: cell and the eight around it, so a cell edge never splits one station
#: whose two streams round its coordinates differently) and were measured
#: within this many seconds of each other (a SYNOP at the hour and the
#: METAR at :53 are one report of one sensor; the two are joined through
#: two half-shifted time partitions of twice this width, so a pair inside
#: the tolerance always shares a bin and no bin edge splits it); the row
#: with the smaller assigned error is kept, ties by stream name, so the
#: choice is deterministic and stated.
CROSS_STREAM_CELL_DEG = 0.01
CROSS_STREAM_TIME_S = 600.0

# Vertical influence of a 2 m dewpoint report, metres above the column's
# surface: e-folding of exp(-z / h).  1500 m is the depth of the mixed
# layer whose moisture the 2 m dewpoint samples in the afternoon; the
# temperature report's 2000 m is not reused because the surface humidity
# gradient sits inside the boundary layer where the temperature's reaches
# into the free troposphere.
DEFAULT_HUMIDITY_DECAY_HEIGHT_M = 1500.0

MOISTURE_UPDATE_DIVERGENCE = (
    "v1 of this door left water untouched: surface pressure, temperature "
    "and rotational wind only.  The moisture update analyses the 2 m "
    "dewpoint (and any aloft dewpoint report) into specific humidity: the "
    "dewpoint innovation is spread by the same successive correction with "
    "its own vertical localization, and at every level the background "
    "dewpoint (Bolton, from the model vapor at that level's pressure) moves "
    "by the spread increment and the vapor is re-derived, so the increment "
    "is the dewpoint change through the local Clausius-Clapeyron slope "
    "rather than a uniform mass of water; it is capped at saturation "
    "against the analysed temperature, floored at zero, and the model's "
    "positivity repair closes the ringing the truncation leaves below zero "
    "inside each column.  The reason: against the CONUS ASOS stations the "
    "GDAS-started forecast read -2.84 / -2.75 K dry at 18Z / 00Z while the "
    "GFS forecast from the same analysis read -2.93 / -2.93 K and the IFS "
    "-0.57 / -0.25 K (observation scorecard, 2026-09-05), so the dry bias "
    "is inherited from the initial state and the stations that measure it "
    "are the way out.  Graded whole on the 24 h hourly T255 cycle "
    "(2026-09-05, arms c and d of the initial-state lane, the same cycle "
    "with and without the update, scored at the CONUS ASOS stations): the "
    "update took the 18 h dewpoint from -2.76 / 5.68 to -0.97 / 3.59 K bias "
    "/ rmse and the 24 h from -2.88 / 5.58 to -1.89 / 4.32, and it took the "
    "sea-level pressure from 2.70 to 4.19 hPa rmse at 18 h and 3.08 to 4.21 "
    "at 24 h (bias -1.6 to -3.3 hPa) with the 24 h forecast raining twice "
    "the cold start's total, so it ships selectable (--moisture-update on) "
    "and off by default: the admission rule allowed no more than 0.03 hPa "
    "of pressure for the dewpoint win."
)

# Each rejection gate names the concrete breakage it prevents; the table is
# embedded in every report so a rejection count is never a bare number.
REJECTION_BREAKAGE = {
    "gross_bounds": (
        "a sentinel or truncated value would paint a physically impossible "
        "innovation across every gridpoint inside the localization radius"
    ),
    "age_window": (
        "a stale report describes an atmosphere the background has already "
        "moved past; blending it in would drag the analysis backward in time"
    ),
    "future_time": (
        "a report timestamped after the analysis time is a clock defect and "
        "its position/value pairing cannot be trusted"
    ),
    "duplicate_superseded": (
        "assimilating a station's superseded report alongside its latest "
        "double-counts one instrument and overweights it against neighbours"
    ),
    "elevation_mismatch": (
        "where station and model terrain disagree beyond the limit, the "
        "reduction between them fabricates an innovation from the height "
        "difference instead of the weather"
    ),
    "above_model_top": (
        "the observation operator would hold the top-level value out to the "
        "report's level, manufacturing agreement the model state cannot see"
    ),
    "below_model_surface": (
        "a level under the local surface pressure has no model equivalent; "
        "clamping to the lowest level would misplace the innovation"
    ),
    "already_assimilated": (
        "a report already blended into the background's assimilation chain "
        "is not independent of that background: the chain holds g/(1+g) of "
        "it already, and offering it again at full gain treats the analysis "
        "that contains it as a fresh forecast - under 15-minute cycling the "
        "90-minute age window re-accepted every hourly report up to seven "
        "times and the temperature O-B rms against reports that never "
        "changed fell 1.211 -> 0.508 -> 0.420 -> 0.411 K over the first "
        "three re-offers on the smoke configuration (audit 2026-09-01, "
        "DA-6: 1.211 -> 0.030 K over seven cycles on the predecessor's "
        "vertical localization)"
    ),
    "no_operator": (
        "a variable the successive correction has no forward operator for "
        "(refractivity) would enter its spreading sums as a NaN innovation and "
        "poison every gridpoint inside its localization radius; the rows are "
        "refused by name here and evaluated by the ensemble filter's operators, "
        "which carry the refractivity operator (--filter letkf)"
    ),
    "duplicate_cross_stream": (
        "the same instrument reported through two streams (the METAR archive "
        "and the Aviation Weather Center cache, a WIS2 SYNOP and the station's "
        "METAR) is one measurement; analysed twice it would enter the local "
        "solve as two independent rows of one error and count double against "
        "its neighbours, so one row is kept (the smaller assigned error, ties "
        "by stream name) and the other counted"
    ),
    "pole_singularity": (
        "every surface row is evaluated through the lowest level's wind, "
        "which is synthesised from streamfunction and velocity potential as "
        "a lat-lon vector, and a lat-lon vector has no direction at the "
        "exact pole: the spectral sampler refuses there (its own guard is "
        "|cos(latitude)| < 1e-12).  Without this gate ONE such row refuses "
        "the whole analysis instead of losing one report, and the public "
        "surface stream carries one: a station reporting at latitude "
        "-90.0000 stopped the first cycle of a global run (measured "
        "2026-09-07, 26,388 rows offered, one at the pole)"
    ),
}

#: The spectral sampler's own domain: it refuses a vector or gradient where
#: ``|cos(latitude)|`` falls under this (``spectral.sampling``).  The gate
#: above rejects exactly what the operator cannot evaluate and no more, so a
#: report the sampler can take is never thrown away for being far north.
POLE_SINGULARITY_COS = 1.0e-12

# The gate of record and the breakage the withheld set prevents.
GATE_BREAKAGE = (
    "evaluated on the rows that built the increment, O-A rms < O-B rms "
    "holds for any positive gain and cannot see over-fitting: reports made "
    "of white noise at ten times the table errors (15 K, 1 kPa, 25 m/s), "
    "with nothing real in them to fit, passed the assimilated-row gate on "
    "every variable while writing 3.1 K and 10.4 m/s of that noise into "
    "the state (measured on the smoke configuration; audit 2026-09-01 "
    "DA-5: 3.37 K, 3.90/4.96 m/s).  The same reports, judged on a "
    "withheld tenth the analysis never saw, fit worse than the background "
    "and fail."
)

# Where the analysis chain lives in the checkpoint: physics_state.metadata
# is the one JSON-valued store every physics suite copies forward
# unchanged step after step, so the chain survives a restart and a run.
ASSIMILATION_HISTORY_KEY = "assimilation_history"
ASSIMILATION_HISTORY_SCHEMA = "gpuwm.arwen-global-assimilation-history/v1"

# The name this module's analysis writes into the lineage: the
# deterministic successive correction of v1.  The ensemble filter writes
# its own (da_filter.FILTERS).
FILTER_NAME = "successive-correction"

# What the global-mean preservation of the surface-pressure increment
# does, measured (audit 2026-09-06 on the smoke configuration, 64
# stations covering the globe, 4000 km length scale): a network-wide
# innovation of +500 Pa spread to 494 Pa everywhere and the offset took
# it back out (log offset -4.94e-3, net increment 0.31 to -0.44 Pa, O-A
# rms 499.99 against O-B 500.00 Pa); -500 Pa the same the other way.  The
# analysis step conserves the global mean of surface pressure to 6e-11 Pa
# and the dry-air mass to 1.4e-5 Pa in both directions.  So a pressure
# bias that every station shares cannot be corrected by this door: the
# model's mass fixer owns the global mean and the door defers to it; the
# report carries the uniform component it removed so a reader can see
# how much of the network's innovation was mass rather than gradient.
MASS_PRESERVATION_DIVERGENCE = (
    "the global mean of surface pressure is preserved across the analysis "
    "(the model's mass fixer owns that mean and would re-absorb any shift on "
    "the first restart step), so the uniform part of the pressure "
    "innovation over the whole globe is removed and only gradients are "
    "analysed; measured on the smoke configuration a network-wide +500 Pa "
    "innovation left 0.4 Pa in the state.  uniform_increment_removed_pa is "
    "the global-mean surface-pressure change the spread increment asked for "
    "and the preservation took out"
)

# The sponge rule is not a rejection - the report keeps its row - but it
# names its breakage the same way: no increment from a report below the
# absorber base may enter a ring the absorber owns unless a report sits
# in that region itself.
SPONGE_EXCLUSION_BREAKAGE = (
    "the dycore's top absorber relaxes every ring whose ring-mean pressure "
    "sits below the sponge base toward its zonal mean and the stratospheric "
    "floor acts there; an increment painted into those rings from a "
    "tropospheric report is not an analysis of anything observed, it is a "
    "lid perturbation the absorber then spends its budget erasing (audit "
    "2026-09-01, DA-2: a 250 hPa report wrote 0.62-0.68 of its innovation "
    "into the three sponged levels of the 20-level stack)"
)


@dataclass(frozen=True)
class AssimilationOptions:
    length_scale_km: float = 300.0
    horizontal_cutoff_scales: float = 3.5
    surface_decay_height_m: float = 2000.0
    aircraft_level_scale_m: float = DEFAULT_AIRCRAFT_LEVEL_SCALE_M
    vertical_weight_floor: float = VERTICAL_WEIGHT_FLOOR
    maximum_age_s: float = 5400.0
    future_tolerance_s: float = 600.0
    elevation_limit_m: float = 500.0
    gate_minimum_count: int = 50
    column_chunk: int = 1024
    background_errors: tuple[tuple[str, float], ...] = DEFAULT_BACKGROUND_ERRORS
    wind_balance: str = DEFAULT_WIND_BALANCE
    # Cross-validation: the fraction of each gated variable's accepted
    # reports held out of the analysis and judged instead, chosen by a
    # seeded permutation of the reports' identity hashes (so the same
    # reports give the same withheld set whatever order they arrived in).
    withheld_fraction: float = 0.1
    withheld_seed: int = 0
    # The moisture update (MOISTURE_UPDATE_DIVERGENCE): dewpoint reports
    # analysed into specific humidity with their own vertical localization.
    # Selectable, not the default: on the graded 24 h T255 cycle it won the
    # 2 m dewpoint and lost sea-level pressure by more than the admission
    # rule allows (the verdict quoted in MOISTURE_UPDATE_DIVERGENCE).
    moisture_update: bool = False
    humidity_decay_height_m: float = DEFAULT_HUMIDITY_DECAY_HEIGHT_M

    def __post_init__(self) -> None:
        if self.wind_balance not in WIND_BALANCE_MODES:
            raise ValueError(
                f"wind_balance must be one of {WIND_BALANCE_MODES}, "
                f"not {self.wind_balance!r}"
            )
        if not 0.0 < self.withheld_fraction < 0.5:
            raise ValueError(
                "withheld_fraction must lie in (0, 0.5): at 0 the gate of "
                "record has no withheld rows to judge and falls back to the "
                "assimilated rows, which pass for any positive gain "
                "(GATE_BREAKAGE); at 0.5 and above more reports are judged "
                "than analysed"
            )
        if isinstance(self.withheld_seed, bool) or not isinstance(
            self.withheld_seed, int
        ) or self.withheld_seed < 0:
            raise ValueError("withheld_seed must be a nonnegative integer")
        for name in (
            "length_scale_km", "horizontal_cutoff_scales",
            "surface_decay_height_m", "aircraft_level_scale_m",
            "maximum_age_s", "future_tolerance_s", "elevation_limit_m",
            "humidity_decay_height_m",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0.0 < self.vertical_weight_floor < 1.0:
            raise ValueError(
                "vertical_weight_floor must lie in (0, 1): it is the smallest "
                "vertical weight allowed to contribute, and the hard cut it "
                "defines vanishes at 0 and removes every level at 1"
            )
        if self.gate_minimum_count < 1 or self.column_chunk < 1:
            raise ValueError("gate_minimum_count and column_chunk must be >= 1")
        errors = dict(self.background_errors)
        if set(errors) != set(VARIABLE_TABLE):
            raise ValueError(
                "background_errors must cover exactly the neutral variable "
                f"vocabulary {sorted(VARIABLE_TABLE)}"
            )
        if any(not math.isfinite(v) or v <= 0.0 for v in errors.values()):
            raise ValueError("background errors must be finite and positive")

    def background_error(self, variable: str) -> float:
        return dict(self.background_errors)[variable]

    def identity(self) -> dict[str, object]:
        return {
            "length_scale_km": self.length_scale_km,
            "horizontal_cutoff_scales": self.horizontal_cutoff_scales,
            "surface_decay_height_m": self.surface_decay_height_m,
            "aircraft_level_scale_m": self.aircraft_level_scale_m,
            "vertical_weight_floor": self.vertical_weight_floor,
            "maximum_age_s": self.maximum_age_s,
            "future_tolerance_s": self.future_tolerance_s,
            "elevation_limit_m": self.elevation_limit_m,
            "gate_minimum_count": self.gate_minimum_count,
            "background_errors": dict(self.background_errors),
            "wind_balance": self.wind_balance,
            "withheld_fraction": self.withheld_fraction,
            "withheld_seed": self.withheld_seed,
            "moisture_update": self.moisture_update,
            "humidity_decay_height_m": self.humidity_decay_height_m,
        }


@dataclass
class _Family:
    """Accepted rows of one neutral variable, columnized."""

    rows: list[ObsRow]

    def __post_init__(self) -> None:
        self.latitude = np.array([r.latitude_deg for r in self.rows])
        self.longitude = np.array([r.longitude_deg for r in self.rows])
        self.elevation = np.array([r.elevation_m for r in self.rows])
        self.level_pa = np.array([
            math.nan if r.level_pa is None else r.level_pa for r in self.rows
        ])
        self.surface = np.array(
            [r.level_pa is None for r in self.rows], dtype=bool
        )
        self.value = np.array([r.value for r in self.rows])
        self.error = np.array([r.error for r in self.rows])

    @property
    def count(self) -> int:
        return len(self.rows)


def _to_numpy_spectral(backend, coeff) -> np.ndarray:
    return np.asarray(backend.to_numpy(coeff)).astype(np.complex128, copy=False)


def _interp_ln_pressure(
    profile: np.ndarray, ln_p: np.ndarray, ln_target: np.ndarray
) -> np.ndarray:
    """Linear-in-ln(p) interpolation of per-point profiles to per-point
    targets.  ``profile``/``ln_p`` are (nlev, n) ordered top to bottom;
    values are held outside the profile span."""
    nlev = profile.shape[0]
    position = np.sum(ln_p <= ln_target[None, :], axis=0)
    upper = np.clip(position - 1, 0, nlev - 1)
    lower = np.clip(position, 0, nlev - 1)
    take = np.take_along_axis
    lp_upper = take(ln_p, upper[None, :], 0)[0]
    lp_lower = take(ln_p, lower[None, :], 0)[0]
    v_upper = take(profile, upper[None, :], 0)[0]
    v_lower = take(profile, lower[None, :], 0)[0]
    span = np.where(lower == upper, 1.0, lp_lower - lp_upper)
    weight = np.clip((ln_target - lp_upper) / span, 0.0, 1.0)
    return v_upper * (1.0 - weight) + v_lower * weight


def specific_humidity_from_dewpoint(dewpoint_k, p_pa) -> np.ndarray:
    """Specific humidity (kg/kg) whose dewpoint at ``p_pa`` is
    ``dewpoint_k``: the inverse of ``surface_energy.
    dewpoint_from_specific_humidity`` (Bolton 1980, the same constants),
    so the round trip q -> Td -> q is the identity above the 1e-9 floor.
    With ``dewpoint_k`` the air temperature it is the saturation specific
    humidity."""
    t = np.asarray(dewpoint_k, dtype=np.float64) - 273.15
    ln = _BOLTON_B * t / (t + _BOLTON_C)
    e = _BOLTON_A * np.exp(ln)
    p = np.asarray(p_pa, dtype=np.float64)
    e = np.minimum(e, 0.999 * p)
    return _EPSILON * e / (p - (1.0 - _EPSILON) * e)


def bounded_vapor_increment(q_bg, t_an, p_an, delta_td):
    """The moisture update's specific-humidity increment from a spread
    dewpoint increment ``delta_td`` (K) on the grid: at every point the
    background dewpoint (from ``q_bg`` at ``p_an``) moves by ``delta_td``
    and the vapor is re-derived, both sides through one relation so a zero
    change is exactly zero; a moistening is capped at saturation against
    the analysed temperature ``t_an`` (a point already above saturation is
    not moistened, and a drying there is left alone), a drying is floored
    at zero vapor.  Returns ``(delta_q, capped, floored)`` with the two
    boolean masks of the bounds that acted."""
    q_bg = np.asarray(q_bg, dtype=np.float64)
    td_bg = dewpoint_from_specific_humidity(q_bg, p_an)
    delta_q = (
        specific_humidity_from_dewpoint(td_bg + delta_td, p_an)
        - specific_humidity_from_dewpoint(td_bg, p_an)
    )
    q_sat = specific_humidity_from_dewpoint(t_an, p_an)
    capped = (delta_q > 0.0) & (q_bg + delta_q > q_sat)
    delta_q = np.where(capped, np.maximum(q_sat - q_bg, 0.0), delta_q)
    floored = (q_bg + delta_q) < 0.0
    delta_q = np.where(floored, -np.maximum(q_bg, 0.0), delta_q)
    return delta_q, capped, floored


def _sample_grid(field: np.ndarray, grid, lat_deg, lon_deg) -> np.ndarray:
    """Bilinear sample of a ``(nlat, nlon)`` grid field at points; latitude
    by fractional index over the grid's (monotone) latitudes, longitude
    periodic at the grid's uniform spacing."""
    field = np.asarray(field, dtype=np.float64)
    grid_lat = np.asarray(grid.latitude_deg, dtype=np.float64)
    grid_lon = np.asarray(grid.longitude_deg, dtype=np.float64)
    order = np.argsort(grid_lat)
    fy = np.interp(lat_deg, grid_lat[order], np.arange(grid_lat.size, dtype=np.float64))
    y0 = np.minimum(fy.astype(np.int64), grid_lat.size - 2)
    wy = np.clip(fy - y0, 0.0, 1.0)
    dlon = 360.0 / grid_lon.size
    fx = np.mod(np.asarray(lon_deg, dtype=np.float64) - grid_lon[0], 360.0) / dlon
    x0 = np.mod(fx.astype(np.int64), grid_lon.size)
    wx = fx - np.floor(fx)
    x1 = (x0 + 1) % grid_lon.size
    rows = field[order]
    return (
        rows[y0, x0] * (1.0 - wy) * (1.0 - wx)
        + rows[y0, x1] * (1.0 - wy) * wx
        + rows[y0 + 1, x0] * wy * (1.0 - wx)
        + rows[y0 + 1, x1] * wy * wx
    )


def _anemometer_wind(
    u_low, v_low, t_low, qv_low, p_full_low, ps, skin_k, land_fraction,
    soil_wetness, roughness_m,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest-full-level wind reduced to 10 m by the model's own
    Monin-Obukhov similarity diagnostic (``physics.surface_diagnostics``,
    the formulation that writes U10/V10 to the render tapes).

    Neutral limit: u10 = u_low ln(10 / z0) / ln(z1 / z0), the log law.
    A 5 m/s anemometer report against a lowest level 23 m up (367 m on
    the 20-level stack) was compared raw before this: an innovation of
    fixed sign, larger than the report's own error (audit 2026-09-01,
    DA-4).
    """
    humidity = effective_surface_humidity(
        skin_k, ps, land_fraction, soil_wetness, np.maximum(qv_low, 0.0), np
    )
    out = similarity_surface_diagnostics(
        u_lowest=np.asarray(u_low, dtype=np.float64),
        v_lowest=np.asarray(v_low, dtype=np.float64),
        temperature_lowest=t_low, qv_lowest=np.maximum(qv_low, 0.0),
        p_full_lowest_pa=p_full_low, p_surface_pa=ps,
        skin_temperature_k=skin_k, surface_humidity=humidity,
        roughness_m=roughness_m, xp=np,
    )
    return out["u10"], out["v10"]


class _ModelSpace:
    """Point observation operators H(x) against one spectral atmosphere.

    ``surface`` is the checkpoint's surface state (skin temperature, land
    fraction, soil water, roughness): the 10 m wind operator needs it, so
    the analysis and the background are reduced by the same profile.
    The physics state's own U10/V10 are not used as H(x): they are frozen
    at the background and would make O-A equal O-B by construction.

    Every operator is evaluated by :meth:`evaluate` on one point set at a
    time: the surface rows from one synthesis of the lowest full level
    (surface pressure, potential temperature, vapor, terrain, and the
    lowest level's wind), the aloft rows from the profiles at their own
    points.  :meth:`hx` puts every family's rows into one point set, so
    the five variables of a report set cost one evaluation, not five
    (measured 2026-09-05 on the T255 hourly cycle: the wind operator
    alone synthesized forty levels of streamfunction and velocity
    potential at two thousand stations eight times per analysis, 125 of
    the door's 211 seconds, for the lowest level of each).
    """

    def __init__(
        self, transform, vertical, terrain_spectral: np.ndarray, surface,
        soil_wetness_capacity: float,
    ):
        self.transform = transform
        self.vertical = vertical
        self.terrain = terrain_spectral  # geopotential, spectral
        self.a = np.asarray(vertical.a_half_pa)
        self.b = np.asarray(vertical.b_half)
        to_numpy = transform.backend.to_numpy
        self.skin_k = np.asarray(to_numpy(surface.temperature_k), dtype=np.float64)
        self.land_fraction = np.asarray(to_numpy(surface.land_fraction), dtype=np.float64)
        self.soil_wetness = np.clip(
            np.asarray(to_numpy(surface.soil_water_fraction), dtype=np.float64)[0]
            / float(soil_wetness_capacity),
            0.0, 1.0,
        )
        self.roughness_m = np.asarray(to_numpy(surface.roughness_m), dtype=np.float64)
        # Memo of host copies and point syntheses, keyed by the atmosphere
        # object (held in the value so its id cannot be recycled while the
        # entry lives) and the sampled points.  The operators are
        # evaluated several times per analysis against the same atmosphere
        # at the same points; repeating a synthesis repeats its bits, so
        # the memo changes nothing but the wall.  clear_cache() at the end
        # of an analysis releases the copies.
        self._cache: dict[tuple, tuple[object, object]] = {}

    def clear_cache(self) -> None:
        self._cache.clear()

    def _memo(self, key: tuple, holder, build):
        entry = self._cache.get(key)
        if entry is not None and entry[0] is holder:
            return entry[1]
        value = build()
        self._cache[key] = (holder, value)
        return value

    def host_spectral(self, atmosphere, name: str) -> np.ndarray:
        """The atmosphere's spectral field ``name`` on the host, complex128."""
        return self._memo(
            ("host", id(atmosphere), name), atmosphere,
            lambda: _to_numpy_spectral(
                self.transform.backend, getattr(atmosphere, name)
            ),
        )

    def _point_pressures(self, ps: np.ndarray) -> np.ndarray:
        p_half = self.a[:, None] + self.b[:, None] * ps[None, :]
        return np.sqrt(p_half[:-1] * p_half[1:])

    def _stack_sample(self, coeffs: list[np.ndarray], lat, lon) -> np.ndarray:
        return sample_scalar(self.transform, np.stack(coeffs), lat, lon)

    @staticmethod
    def _point_key(*arrays) -> tuple:
        return tuple(np.asarray(a, dtype=np.float64).tobytes() for a in arrays)

    def surface_context(self, atmosphere, lat, lon) -> dict[str, np.ndarray]:
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        return self._memo(
            ("surface_context", id(atmosphere), *self._point_key(lat, lon)),
            atmosphere, lambda: self._surface_context(atmosphere, lat, lon),
        )

    def _surface_context(self, atmosphere, lat, lon) -> dict[str, np.ndarray]:
        theta = self.host_spectral(atmosphere, "theta")
        qv = self.host_spectral(atmosphere, "qv")
        lnps = self.host_spectral(atmosphere, "log_surface_pressure")
        sampled = self._stack_sample(
            [lnps, theta[-1], qv[-1], self.terrain], lat, lon
        )
        ps = np.exp(sampled[0])
        z_model = sampled[3] / GRAVITY_M_S2
        p_full_low = self._point_pressures(ps)[-1]
        t_low = sampled[1] * (p_full_low / REFERENCE_PRESSURE_PA) ** KAPPA
        qv_low = np.maximum(sampled[2], 0.0)
        tv_low = t_low * (1.0 + 0.61 * qv_low)
        z_low_msl = z_model + (
            DRY_AIR_GAS_CONSTANT * tv_low / GRAVITY_M_S2
        ) * np.log(ps / p_full_low)
        return {
            "ps": ps, "z_model": z_model, "t_low": t_low, "qv_low": qv_low,
            "p_full_low": p_full_low, "tv_low": tv_low, "z_low_msl": z_low_msl,
        }

    def lowest_wind(self, atmosphere, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """``(u, v)`` of the lowest full level at points, from one gradient
        synthesis of that level's streamfunction and velocity potential
        (the inverse Laplacian acts per coefficient, so one level's wind
        is the same whether or not the other levels ride along)."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)

        def build():
            u, v = sample_wind(
                self.transform,
                self.host_spectral(atmosphere, "vorticity")[-1:],
                self.host_spectral(atmosphere, "divergence")[-1:],
                lat, lon,
            )
            return u[0], v[0]

        return self._memo(
            ("lowest_wind", id(atmosphere), *self._point_key(lat, lon)),
            atmosphere, build,
        )

    def wind_profiles(self, atmosphere, lat, lon):
        """``(u, v)`` profiles ``(nlev, count)`` at points from one wind
        synthesis of every level; the aloft rows read them."""
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        return self._memo(
            ("wind_profiles", id(atmosphere), *self._point_key(lat, lon)),
            atmosphere, lambda: sample_wind(
                self.transform,
                self.host_spectral(atmosphere, "vorticity"),
                self.host_spectral(atmosphere, "divergence"),
                lat, lon,
            ),
        )

    def anemometer_wind(
        self, atmosphere, u_low, v_low, lat, lon
    ) -> tuple[np.ndarray, np.ndarray]:
        """10 m wind at points from the sampled lowest-level wind and the
        column's surface state (``_anemometer_wind``)."""
        ctx = self.surface_context(atmosphere, lat, lon)
        grid = self.transform.grid
        return _anemometer_wind(
            u_low, v_low, ctx["t_low"], ctx["qv_low"], ctx["p_full_low"],
            ctx["ps"],
            _sample_grid(self.skin_k, grid, lat, lon),
            np.clip(_sample_grid(self.land_fraction, grid, lat, lon), 0.0, 1.0),
            _sample_grid(self.soil_wetness, grid, lat, lon),
            _sample_grid(self.roughness_m, grid, lat, lon),
        )

    def evaluate(self, atmosphere, family: _Family) -> dict[str, np.ndarray]:
        """Every variable's operator at the family's points, ``(count,)``
        each.  A surface row is compared at its own height: pressure
        reduced to the station, temperature to 2 m, the dewpoint of the
        lowest level's vapor at the station pressure, wind to the 10 m
        anemometer by the model's similarity diagnostic.  An aloft row
        reads the profile interpolated in ln p to its level (surface
        pressure has no aloft form and reads NaN there)."""
        key = (
            "evaluate", id(atmosphere),
            *self._point_key(
                family.latitude, family.longitude, family.elevation,
                family.level_pa,
            ),
        )
        return self._memo(key, atmosphere, lambda: self._evaluate(atmosphere, family))

    @staticmethod
    def _unique_points(lat, lon):
        """``(ulat, ulon, inverse)``: the distinct (latitude, longitude)
        pairs of a row set and each row's index into them.  A station
        reports up to five variables and a sounding site seven levels of
        each, so the rows repeat their points five to thirty times; the
        syntheses run once per point and the rows gather.  Per-point
        arithmetic is independent of the chunk a point rides in (the
        Legendre recurrence and the einsum over degrees are per point),
        so the gathered values are the same bits the repeated synthesis
        would have produced."""
        pairs = np.stack([
            np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64),
        ], axis=1)
        unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
        return unique[:, 0].copy(), unique[:, 1].copy(), np.asarray(inverse).ravel()

    def _evaluate(self, atmosphere, family: _Family) -> dict[str, np.ndarray]:
        out = {name: np.full(family.count, np.nan) for name in VARIABLE_TABLE}
        surface = family.surface
        if surface.any():
            lat = family.latitude[surface]
            lon = family.longitude[surface]
            elevation = family.elevation[surface]
            ulat, ulon, at = self._unique_points(lat, lon)
            uctx = self.surface_context(atmosphere, ulat, ulon)
            ctx = {name: value[at] for name, value in uctx.items()}
            p_station = ctx["ps"] * np.exp(
                -GRAVITY_M_S2 * (elevation - ctx["z_model"])
                / (DRY_AIR_GAS_CONSTANT * ctx["tv_low"])
            )
            out["surface_pressure_pa"][surface] = p_station
            out["temperature_k"][surface] = ctx["t_low"] + SURFACE_LAPSE_K_M * (
                ctx["z_low_msl"] - (elevation + 2.0)
            )
            # The lowest full level's vapor (23 m up on the default grid;
            # the surface humidity gradient over that height is inside the
            # report's error) at the model surface pressure reduced to the
            # station, the same column the pressure operator uses.
            out["dewpoint_k"][surface] = dewpoint_from_specific_humidity(
                ctx["qv_low"], p_station
            )
            # A surface report is an anemometer at 10 m: reduce the lowest
            # full level to it, never compare it raw (DA-4).
            u_low, v_low = self.lowest_wind(atmosphere, ulat, ulon)
            u10, v10 = self.anemometer_wind(atmosphere, u_low, v_low, ulat, ulon)
            out["wind_u_m_s"][surface] = u10[at]
            out["wind_v_m_s"][surface] = v10[at]
        aloft = ~surface
        if aloft.any():
            lat = family.latitude[aloft]
            lon = family.longitude[aloft]
            level = family.level_pa[aloft]
            ulat, ulon, at = self._unique_points(lat, lon)
            theta = self.host_spectral(atmosphere, "theta")
            qv = self.host_spectral(atmosphere, "qv")
            lnps = self.host_spectral(atmosphere, "log_surface_pressure")
            nlev = theta.shape[0]
            sampled = self._memo(
                ("aloft_profiles", id(atmosphere), *self._point_key(ulat, ulon)),
                atmosphere,
                lambda: self._stack_sample(
                    [*(theta[k] for k in range(nlev)),
                     *(qv[k] for k in range(nlev)), lnps],
                    ulat, ulon,
                ),
            )
            ps = np.exp(sampled[-1])
            p_full = self._point_pressures(ps)[:, at]
            ln_p = np.log(p_full)
            ln_level = np.log(level)
            t_profile = sampled[:nlev][:, at] * (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
            out["temperature_k"][aloft] = _interp_ln_pressure(t_profile, ln_p, ln_level)
            q_profile = np.maximum(sampled[nlev:2 * nlev][:, at], 0.0)
            q_level = _interp_ln_pressure(q_profile, ln_p, ln_level)
            out["dewpoint_k"][aloft] = dewpoint_from_specific_humidity(q_level, level)
            u, v = self.wind_profiles(atmosphere, ulat, ulon)
            out["wind_u_m_s"][aloft] = _interp_ln_pressure(u[:, at], ln_p, ln_level)
            out["wind_v_m_s"][aloft] = _interp_ln_pressure(v[:, at], ln_p, ln_level)
        return out

    def hx_surface_pressure(self, atmosphere, family: _Family) -> np.ndarray:
        return self.evaluate(atmosphere, family)["surface_pressure_pa"]

    def hx_temperature(self, atmosphere, family: _Family) -> np.ndarray:
        return self.evaluate(atmosphere, family)["temperature_k"]

    def hx_dewpoint(self, atmosphere, family: _Family) -> np.ndarray:
        return self.evaluate(atmosphere, family)["dewpoint_k"]

    def hx_wind(self, atmosphere, family: _Family, component: str) -> np.ndarray:
        name = "wind_u_m_s" if component == "u" else "wind_v_m_s"
        return self.evaluate(atmosphere, family)[name]

    def hx(
        self, atmosphere, *groups: dict[str, _Family]
    ) -> tuple[dict[str, np.ndarray], ...]:
        """H(x) for one or more groups of families (the assimilated and
        the withheld rows, say) from ONE evaluation on the union of every
        row, each variable then read off its own rows.  Returns one
        ``{variable: values}`` per group, in order."""
        rows = [
            row for group in groups for family in group.values()
            for row in family.rows
        ]
        values = self.evaluate(atmosphere, _Family(rows)) if rows else None
        results = []
        start = 0
        for group in groups:
            out: dict[str, np.ndarray] = {}
            for variable, family in group.items():
                stop = start + family.count
                out[variable] = (
                    np.zeros(0) if family.count == 0
                    else np.array(values[variable][start:stop])
                )
                start = stop
            results.append(out)
        return tuple(results)


def _resolve_analysis_time(analysis_time, rows: list[ObsRow]) -> dt.datetime:
    if analysis_time is None:
        return max(row.valid_time for row in rows)
    if isinstance(analysis_time, dt.datetime):
        moment = analysis_time
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=dt.timezone.utc)
        return moment.astimezone(dt.timezone.utc)
    parsed = parse_valid_time(str(analysis_time))
    if parsed is None:
        raise ValueError(
            f"analysis time {analysis_time!r} is not an ISO-8601 instant"
        )
    return parsed


def cross_stream_duplicates(rows: list[ObsRow]) -> tuple[list[ObsRow], dict[str, int]]:
    """One row per instrument across streams (:data:`REJECTION_BREAKAGE`
    ``duplicate_cross_stream``): rows of one variable and level that share
    a station id, or a position cell (:data:`CROSS_STREAM_CELL_DEG`), and
    were measured within :data:`CROSS_STREAM_TIME_S` of each other
    (pairwise, whatever the time-bin edges; a group chains through rows
    that are each within the tolerance of another), across DIFFERENT
    sources collapse to the row with the smallest assigned error (ties by
    source name).  Returns ``(kept rows in input order, dropped count per
    source)``; rows of one source never collapse here (the in-stream rules
    own that)."""
    def level_key(row):
        return "sfc" if row.level_pa is None else int(round(row.level_pa / 100.0))

    def seconds_of(row):
        when = row.valid_time if row.valid_time.tzinfo else row.valid_time.replace(tzinfo=dt.timezone.utc)
        return when.timestamp()

    def time_bins(seconds):
        # Two partitions of width 2 * tolerance, the second shifted by the
        # tolerance: two instants within the tolerance share a bin in at
        # least one of them.  The bins only LIMIT the candidates; the pair
        # itself is joined on its measured separation below, because a bin
        # 20 minutes wide also holds pairs 11 to 19 minutes apart, which
        # the rule (ten minutes) must keep as two reports.
        width = 2.0 * CROSS_STREAM_TIME_S
        return ((0, int(math.floor(seconds / width))),
                (1, int(math.floor((seconds + CROSS_STREAM_TIME_S) / width))))

    def cell(row):
        return (int(math.floor(row.latitude_deg / CROSS_STREAM_CELL_DEG)),
                int(math.floor((row.longitude_deg % 360.0) / CROSS_STREAM_CELL_DEG)))

    lon_cells = int(round(360.0 / CROSS_STREAM_CELL_DEG))
    tolerance = float(CROSS_STREAM_TIME_S)
    when_s = [seconds_of(row) for row in rows]

    # Union-find over the rows: two keys (station, cell) join rows into one
    # instrument group; a group with more than one source keeps one row.
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    def join_within_tolerance(candidates, index):
        for held in candidates:
            if abs(when_s[held] - when_s[index]) <= tolerance:
                union(held, index)

    by_station: dict[tuple, list[int]] = {}
    by_cell: dict[tuple, list[int]] = {}
    for index, row in enumerate(rows):
        station = row.station_id.strip().upper()
        i_lat, i_lon = cell(row)
        for time_bin in time_bins(when_s[index]):
            base = (row.variable, level_key(row), time_bin)
            if station:
                key = (*base, station)
                held = by_station.get(key)
                if held is None:
                    by_station[key] = [index]
                else:
                    join_within_tolerance(held, index)
                    held.append(index)
            for d_lat in (-1, 0, 1):
                for d_lon in (-1, 0, 1):
                    held = by_cell.get((*base, i_lat + d_lat, (i_lon + d_lon) % lon_cells))
                    if held is not None:
                        join_within_tolerance(held, index)
            by_cell.setdefault((*base, i_lat, i_lon), []).append(index)
    groups: dict[int, list[int]] = {}
    for index in range(len(rows)):
        groups.setdefault(find(index), []).append(index)
    dropped: set[int] = set()
    dropped_by_source: dict[str, int] = {}
    for members in groups.values():
        sources = {rows[i].source for i in members}
        if len(sources) < 2:
            continue
        keep = min(members, key=lambda i: (rows[i].error, rows[i].source, i))
        for i in members:
            if i != keep and rows[i].source != rows[keep].source:
                dropped.add(i)
                dropped_by_source[rows[i].source] = dropped_by_source.get(rows[i].source, 0) + 1
    kept = [row for i, row in enumerate(rows) if i not in dropped]
    return kept, dropped_by_source


def _table_quality_control(
    rows: list[ObsRow], analysis_time: dt.datetime, options: AssimilationOptions,
    *, operator_variables=None,
) -> tuple[list[ObsRow], dict[str, int]]:
    """The gates that need no model state: bounds, age, duplicates within
    a stream and across streams.  ``operator_variables`` names what the
    caller's operators evaluate (``None``: the successive correction's
    vocabulary, so :data:`VARIABLES_WITHOUT_OPERATOR` is refused by
    name); a caller whose operators carry a variable passes it and the
    refusal does not apply."""
    rejections = {name: 0 for name in REJECTION_BREAKAGE}
    without_operator = (
        VARIABLES_WITHOUT_OPERATOR if operator_variables is None
        else VARIABLES_WITHOUT_OPERATOR - frozenset(operator_variables)
    )
    survivors: list[ObsRow] = []
    for row in rows:
        if row.variable in without_operator:
            rejections["no_operator"] += 1
            continue
        low, high = VARIABLE_TABLE[row.variable]["gross_bounds"]
        if not low <= row.value <= high:
            rejections["gross_bounds"] += 1
            continue
        if abs(math.cos(math.radians(row.latitude_deg))) < POLE_SINGULARITY_COS:
            rejections["pole_singularity"] += 1
            continue
        age = (analysis_time - row.valid_time).total_seconds()
        if age > options.maximum_age_s:
            rejections["age_window"] += 1
            continue
        if age < -options.future_tolerance_s:
            rejections["future_time"] += 1
            continue
        survivors.append(row)
    # Duplicate collapse.  A surface station's latest report supersedes its
    # earlier ones; an aloft platform moves, so only exact repeats (same
    # instant, position, and level) collapse - a flight track is data.
    latest: dict[tuple, tuple] = {}
    seen_aloft: set[tuple] = set()
    deduplicated: list[ObsRow] = []
    for order, row in enumerate(survivors):
        if row.level_pa is None:
            key = (row.source, row.station_id, row.variable)
            held = latest.get(key)
            if held is None or (row.valid_time, order) > (held[0], held[1]):
                latest[key] = (row.valid_time, order, row)
        else:
            key = (
                row.source, row.station_id, row.variable, row.valid_time,
                round(row.latitude_deg, 3), round(row.longitude_deg, 3),
                round(row.level_pa),
            )
            if key in seen_aloft:
                rejections["duplicate_superseded"] += 1
                continue
            seen_aloft.add(key)
            deduplicated.append(row)
    surface_kept = len(latest)
    surface_total = sum(1 for row in survivors if row.level_pa is None)
    rejections["duplicate_superseded"] += surface_total - surface_kept
    deduplicated.extend(entry[2] for entry in latest.values())
    deduplicated, dropped_by_source = cross_stream_duplicates(deduplicated)
    rejections["duplicate_cross_stream"] += sum(dropped_by_source.values())
    return deduplicated, rejections


def _model_quality_control(
    rows: list[ObsRow],
    space: _ModelSpace,
    atmosphere,
    options: AssimilationOptions,
    rejections: dict[str, int],
) -> list[ObsRow]:
    """The gates that need the model: terrain agreement, column span."""
    if not rows:
        return rows
    lat = np.array([r.latitude_deg for r in rows])
    lon = np.array([r.longitude_deg for r in rows])
    lnps = space.host_spectral(atmosphere, "log_surface_pressure")
    sampled = space._stack_sample([space.terrain, lnps], lat, lon)
    z_model = sampled[0] / GRAVITY_M_S2
    ps = np.exp(sampled[1])
    p_top_full = space._point_pressures(ps)[0]
    survivors: list[ObsRow] = []
    for index, row in enumerate(rows):
        if row.level_pa is None:
            if abs(row.elevation_m - z_model[index]) > options.elevation_limit_m:
                rejections["elevation_mismatch"] += 1
                continue
        else:
            if row.level_pa < p_top_full[index]:
                rejections["above_model_top"] += 1
                continue
            if row.level_pa > ps[index]:
                rejections["below_model_surface"] += 1
                continue
        survivors.append(row)
    return survivors


def _spread_column(
    family: _Family,
    innovation: np.ndarray,
    gains: np.ndarray,
    grid_lat_rad: np.ndarray,
    grid_lon_rad: np.ndarray,
    options: AssimilationOptions,
    *,
    ps_columns: np.ndarray | None = None,
    p_full_columns: np.ndarray | None = None,
    sponge_columns: np.ndarray | None = None,
    sponge_base_pa: float = 0.0,
    xp=np,
) -> np.ndarray:
    """Data-density-normalised successive correction onto grid columns.

    Returns ``(ncol,)`` for a two-dimensional family (``p_full_columns``
    None) or ``(nlev, ncol)``.  Per column the increment is
    ``sum_i w_i g_i d_i / (sum_i w_i g_i + 1)`` with ``g_i`` the
    background/observation error variance ratio.  The single-observation
    limit ``g d / (1 + g)`` is the optimal-interpolation gain and stacked
    duplicates saturate toward the observed value instead of overshooting
    it; beyond those two cases this is not OI - there is no
    observation-observation solve, the scalar denominator stands in for
    it, and disagreeing neighbours are under-fitted (module docstring).

    Vertical weights: a surface report decays as exp(-z / decay height)
    above the column's surface; an aloft report carries the ADAS
    height-separation model exp(-(dz / zrange)^2) with dz the ln p
    separation times ``LOCALIZATION_SCALE_HEIGHT_M``, zero below
    ``vertical_weight_floor`` (the hard vertical cut).  ``sponge_columns``
    (``(nlev, ncol)`` bool) marks the levels the dycore's top absorber
    owns; a report whose level is not itself below ``sponge_base_pa``
    contributes nothing there (``SPONGE_EXCLUSION_BREAKAGE``).

    ``xp`` is the array module the sums run on: numpy, or the run's
    device module when the caller holds one (the same expressions, chunk
    by chunk; the 2,000-station T255 analysis spent 45 of its 211 seconds
    here on the host, 2026-09-05).  The result is always a numpy array.
    """
    length_m = options.length_scale_km * 1000.0
    cutoff_m = options.horizontal_cutoff_scales * length_m
    obs_lat = xp.asarray(np.deg2rad(family.latitude), dtype=xp.float64)
    obs_lon = xp.asarray(np.deg2rad(family.longitude), dtype=xp.float64)
    sin_o, cos_o = xp.sin(obs_lat), xp.cos(obs_lat)
    grid_lat = xp.asarray(grid_lat_rad, dtype=xp.float64)
    grid_lon = xp.asarray(grid_lon_rad, dtype=xp.float64)
    ncol = grid_lat_rad.size
    volume = p_full_columns is not None
    nlev = 0 if not volume else p_full_columns.shape[0]
    out = xp.zeros((nlev, ncol), dtype=xp.float64) if volume else xp.zeros(ncol, dtype=xp.float64)
    surface = xp.asarray(family.surface)
    aloft = ~surface
    any_surface = bool(family.surface.any())
    any_aloft = bool((~family.surface).any())
    gains_x = xp.asarray(np.broadcast_to(np.asarray(gains, dtype=np.float64), family.latitude.shape), dtype=xp.float64)
    weighted = gains_x * xp.asarray(np.asarray(innovation, dtype=np.float64), dtype=xp.float64)
    if volume:
        ps_x = xp.asarray(ps_columns, dtype=xp.float64)
        p_full_x = xp.asarray(p_full_columns, dtype=xp.float64)
        ln_levels = xp.log(xp.asarray(family.level_pa[~family.surface], dtype=xp.float64))
        # Reports inside the absorber region may write there; every other
        # report is masked out of the sponged levels.
        aloft_in_sponge = xp.asarray(family.level_pa[~family.surface] < sponge_base_pa)
        vertical_cut = options.aircraft_level_scale_m * math.sqrt(
            -math.log(options.vertical_weight_floor)
        )
        if sponge_columns is None:
            sponge_x = xp.zeros((nlev, ncol), dtype=bool)
        else:
            sponge_x = xp.asarray(sponge_columns, dtype=bool)
    for start in range(0, ncol, options.column_chunk):
        stop = min(ncol, start + options.column_chunk)
        sin_c = xp.sin(grid_lat[start:stop])
        cos_c = xp.cos(grid_lat[start:stop])
        cos_arc = xp.clip(
            sin_o[:, None] * sin_c[None, :]
            + cos_o[:, None] * cos_c[None, :]
            * xp.cos(obs_lon[:, None] - grid_lon[None, start:stop]),
            -1.0, 1.0,
        )
        distance = EARTH_RADIUS_M * xp.arccos(cos_arc)
        horizontal = xp.where(
            distance <= cutoff_m,
            xp.exp(-0.5 * (distance / length_m) ** 2),
            0.0,
        )
        if not volume:
            numerator = horizontal.T @ weighted
            denominator = horizontal.T @ gains_x
            out[start:stop] = numerator / (denominator + 1.0)
            continue
        p_full_c = p_full_x[:, start:stop]
        ln_p_c = xp.log(p_full_c)
        open_c = ~sponge_x[:, start:stop]
        numerator = xp.zeros((nlev, stop - start), dtype=xp.float64)
        denominator = xp.zeros((nlev, stop - start), dtype=xp.float64)
        if any_surface:
            # The vertical factor of a surface observation depends only on
            # the column, so its sums factorize.  A surface report is never
            # inside the absorber region, so the sponge mask applies whole.
            z_above = LOCALIZATION_SCALE_HEIGHT_M * (
                xp.log(ps_x[start:stop])[None, :] - ln_p_c
            )
            vertical = xp.exp(-z_above / options.surface_decay_height_m)
            vertical = xp.where(open_c, vertical, 0.0)
            numerator += vertical * (horizontal[surface].T @ weighted[surface])
            denominator += vertical * (horizontal[surface].T @ gains_x[surface])
        if any_aloft:
            h_aloft = horizontal[aloft]
            w_aloft = weighted[aloft]
            g_aloft = gains_x[aloft]
            for k in range(nlev):
                dz = LOCALIZATION_SCALE_HEIGHT_M * (
                    ln_p_c[k][None, :] - ln_levels[:, None]
                )
                vertical = xp.where(
                    xp.abs(dz) <= vertical_cut,
                    xp.exp(-(dz / options.aircraft_level_scale_m) ** 2),
                    0.0,
                )
                # Sponged columns at this level accept only reports that
                # sit in the absorber region themselves.
                vertical = xp.where(
                    open_c[k][None, :] | aloft_in_sponge[:, None],
                    vertical, 0.0,
                )
                weights = h_aloft * vertical
                numerator[k] += weights.T @ w_aloft
                denominator[k] += weights.T @ g_aloft
        out[:, start:stop] = numerator / (denominator + 1.0)
    if xp is np:
        return out
    return np.asarray(xp.asnumpy(out), dtype=np.float64)


def _sponge_columns(p_full_grid: np.ndarray, sponge_base_pa: float) -> np.ndarray:
    """``(nlev, nlat*nlon)`` mask of the rings the top absorber owns: the
    same ring-mean test the dycore applies (``MoistHybridModel._top_sponge``),
    so the analysis and the absorber agree on which levels are its."""
    nlev = p_full_grid.shape[0]
    ring_mean = np.mean(p_full_grid, axis=-1, keepdims=True)
    in_sponge = np.broadcast_to(
        ring_mean < sponge_base_pa, p_full_grid.shape
    )
    return np.ascontiguousarray(in_sponge.reshape(nlev, -1))


def _increment_kinetic_energy(coeff: np.ndarray, radius_m: float) -> float:
    """Global-mean kinetic energy, J/kg, of the wind that a vorticity OR
    divergence coefficient stack ``(..., n, m)`` carries, summed over
    degrees and over every leading (level) axis:
    ``a^2 / (8 pi n(n+1)) sum_m mult |c_nm|^2`` with mult 1 at m = 0 and
    2 for the mirrored m > 0 (Parseval against grid-mean 0.5 |v|^2)."""
    c = np.asarray(coeff)
    n = np.arange(c.shape[-2], dtype=np.float64)
    lam = n * (n + 1.0)
    lam[0] = np.inf
    mult = np.where(np.arange(c.shape[-1]) == 0, 1.0, 2.0)
    power = np.sum(mult * (c.real * c.real + c.imag * c.imag), axis=-1)
    scale = radius_m * radius_m / (8.0 * math.pi * lam)
    return float(np.sum(power * scale))


def _balanced_wind_increment(vector, transform, delta_u, delta_v, mode: str):
    """Analyse the grid wind increment into vorticity/divergence and apply
    the wind balance ``mode``.

    Returns ``(zeta_inc, div_inc, receipt)``.  The receipt carries the
    rotational and divergent kinetic energy of the analysed increment
    (level-summed global means, J/kg), the divergent fraction the scalar
    spreading produced and the fraction actually applied; under
    ``"rotational"`` the applied divergence is exactly zero.
    """
    backend = transform.backend
    xp = backend.xp
    zeta, div = vector.vordiv_from_wind(
        backend.asarray(delta_u, dtype=backend.float_dtype),
        backend.asarray(delta_v, dtype=backend.float_dtype),
    )
    radius = transform.grid.radius_m
    e_rot = _increment_kinetic_energy(backend.to_numpy(zeta), radius)
    e_div = _increment_kinetic_energy(backend.to_numpy(div), radius)
    total = e_rot + e_div
    analysed = e_div / total if total > 0.0 else 0.0
    if mode == "rotational":
        div = xp.zeros_like(div)
        applied = 0.0
    elif mode == "unconstrained":
        applied = analysed
    else:
        raise ValueError(f"unknown wind balance {mode!r}")
    receipt = {
        "mode": mode,
        "rotational_ke_j_kg": e_rot,
        "divergent_ke_j_kg": e_div,
        "divergent_fraction_analysed": analysed,
        "divergent_fraction_applied": applied,
        "breakage": WIND_BALANCE_BREAKAGE,
    }
    return zeta, div, receipt


def _chain_from_background(physics_metadata: dict) -> dict[str, object]:
    """The assimilation chain the background carries, or an empty one.
    ``reports`` maps identity hash -> valid instant (ISO-8601 UTC);
    ``cycles`` lists the analyses that built the chain."""
    held = physics_metadata.get(ASSIMILATION_HISTORY_KEY)
    if not isinstance(held, dict) or held.get("schema") != ASSIMILATION_HISTORY_SCHEMA:
        return {"schema": ASSIMILATION_HISTORY_SCHEMA, "reports": {}, "cycles": []}
    reports = held.get("reports")
    cycles = held.get("cycles")
    if not isinstance(reports, dict) or not isinstance(cycles, list):
        raise ValueError(
            "the background's assimilation history is malformed: "
            "'reports' must be an object and 'cycles' a list"
        )
    return {
        "schema": ASSIMILATION_HISTORY_SCHEMA,
        "reports": dict(reports),
        "cycles": list(cycles),
    }


def _refuse_chain_rows(
    rows: list[ObsRow], chain: dict[str, object], rejections: dict[str, int]
) -> list[ObsRow]:
    """Drop every row whose identity the background chain already holds."""
    known = chain["reports"]
    survivors = []
    for row in rows:
        if row.identity_hash() in known:
            rejections["already_assimilated"] += 1
        else:
            survivors.append(row)
    return survivors


def _withhold(
    rows: list[ObsRow], variable_index: int, options: AssimilationOptions
) -> tuple[list[ObsRow], list[ObsRow]]:
    """Split one variable's accepted rows into (assimilated, withheld).

    Only a gated variable (count >= ``gate_minimum_count``) gives up rows;
    below that there is no gate and every report is used.  The withheld
    rows are the first ``round(withheld_fraction * count)`` of a seeded
    permutation of the rows ordered by identity hash, so the choice
    depends on which reports arrived and on nothing else.
    """
    count = len(rows)
    if count < options.gate_minimum_count:
        return list(rows), []
    withheld_count = int(round(options.withheld_fraction * count))
    ordered = sorted(rows, key=lambda row: row.identity_hash())
    rng = np.random.default_rng([options.withheld_seed, variable_index])
    order = rng.permutation(count)
    chosen = set(order[:withheld_count].tolist())
    assimilated = [row for k, row in enumerate(ordered) if k not in chosen]
    withheld = [row for k, row in enumerate(ordered) if k in chosen]
    return assimilated, withheld


def _chain_for_analysis(
    chain: dict[str, object],
    assimilated: list[ObsRow],
    moment: dt.datetime,
    options: AssimilationOptions,
    *,
    background_sha256: str,
    withheld_count: int,
    background_step: int | None = None,
    filter_name: str = FILTER_NAME,
    streams: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """The chain the analysis checkpoint carries: the background's chain
    plus this cycle's assimilated reports, pruned to what a later cycle
    could still be offered.  A report older than ``maximum_age_s`` at
    this analysis time fails the age window at this and every later
    time, so its entry is dead weight; a cycle whose reports are all
    that old is likewise dropped (``future_tolerance_s`` keeps the entry
    for the reports it accepted from slightly ahead of itself).

    Each cycle entry is one link of the analysis LINEAGE: the instant,
    the background's checkpoint identity, the filter that formed the
    increment and the streams (obs-table sources) that fed it, so a
    checkpoint says on its face which analyses made it and from what."""
    horizon = moment - dt.timedelta(seconds=options.maximum_age_s)
    if streams is None:
        streams = tuple(row.source for row in assimilated)
    reports = {}
    for key, instant in chain["reports"].items():
        held = parse_valid_time(str(instant))
        if held is not None and held >= horizon:
            reports[key] = instant
    for row in assimilated:
        reports[row.identity_hash()] = row.valid_time.astimezone(
            dt.timezone.utc
        ).isoformat(timespec="seconds")
    cycle_horizon = horizon - dt.timedelta(seconds=options.future_tolerance_s)
    cycles = []
    for entry in chain["cycles"]:
        when = parse_valid_time(str(entry.get("analysis_time_utc", "")))
        if when is not None and when >= cycle_horizon:
            cycles.append(entry)
    entry = {
        "analysis_time_utc": moment.isoformat(timespec="seconds"),
        "background_self_sha256": background_sha256,
        "assimilated": len(assimilated),
        "withheld": withheld_count,
        "filter": str(filter_name),
        "streams": sorted(set(streams)),
    }
    if background_step is not None:
        entry["step"] = int(background_step)
    cycles.append(entry)
    return {
        "schema": ASSIMILATION_HISTORY_SCHEMA,
        "reports": reports,
        "cycles": cycles,
    }


def _statistics(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "rms": 0.0}
    return {
        "mean": float(np.mean(values)),
        "rms": float(math.sqrt(np.mean(values ** 2))),
    }


def _write_report(path: Path, payload: dict[str, object]) -> dict[str, object]:
    result = dict(payload)
    result["schema"] = ASSIMILATION_SCHEMA
    result.pop("self_sha256", None)
    result["self_sha256"] = hashlib.sha256(canonical(result)).hexdigest()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.partial-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return result


def load_observations(obs_locations) -> tuple[list[dict], list[ObsRow]]:
    """Every source decoded once: ``(provenance records, rows)``."""
    if not obs_locations:
        raise ValueError("assimilate requires at least one --obs source")
    sources = []
    rows: list[ObsRow] = []
    for location in obs_locations:
        _, decoded, provenance = load_obs(str(location))
        sources.append(provenance)
        rows.extend(decoded)
    if not rows:
        raise ValueError(
            "no observation rows decoded from any source; nothing to assimilate"
        )
    return sources, rows


def _spread_module(transform):
    """The array module the spreading sums run on: the run's device module
    when the transform lives on one, numpy otherwise."""
    backend = transform.backend
    return backend.xp if getattr(backend, "name", "numpy") == "cupy" else np


class _Stopwatch:
    """Wall seconds of the named phases of one analysis, for the report."""

    def __init__(self) -> None:
        self.readings: dict[str, float] = {}
        self._start = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.readings[name] = self.readings.get(name, 0.0) + (now - self._start)
        self._start = now


def analyse(
    cfg: ArwenGlobalConfig,
    model,
    transform,
    state: ArwenGlobalState,
    rows: list[ObsRow],
    *,
    sources: list[dict],
    background: dict[str, object],
    analysis_time=None,
    options: AssimilationOptions | None = None,
) -> tuple[ArwenGlobalState, dict[str, object], dict[str, float]]:
    """Analyse the resident ``state`` against ``rows``: the door's whole
    arithmetic without a disk in it.

    ``model`` and ``transform`` are the run's own (the surface geopotential
    and the vertical coordinate come from them); ``background`` is the
    identity the state's checkpoint carries or would carry
    (``checkpoint.checkpoint_metadata``): ``self_sha256``, ``step``,
    ``time_s`` and, when it was read from disk, ``path``.  The in-process
    cycle door hands its resident forecast here between two steps; the
    file door :func:`assimilate` reads a checkpoint, calls this and
    writes the result, and the two produce the same analysis bit for bit
    because this function is the only place the increments are formed.

    Returns ``(analysis, report, timings)``: the analysed state (the
    background's own object when nothing was applied is never returned;
    the analysis is always a new state carrying its chain), the report
    without its ``analysis`` entry (the writer adds the path and hash of
    what it wrote), and the wall seconds of each phase.
    """
    options = options or AssimilationOptions()
    clock = _Stopwatch()
    moment = _resolve_analysis_time(analysis_time, rows)
    decoded_total = len(rows)
    rows, rejections = _table_quality_control(rows, moment, options)
    # The background's own assimilation chain: a report it already holds
    # is refused before anything is spread (DA-6).
    chain = _chain_from_background(state.physics_state.metadata)
    rows = _refuse_chain_rows(rows, chain, rejections)
    backend = transform.backend
    terrain_spectral = _to_numpy_spectral(
        backend, transform.forward(model.surface_geopotential)
    )
    space = _ModelSpace(
        transform, cfg.vertical, terrain_spectral, state.surface,
        cfg.reference_physics.soil_wetness_capacity,
    )
    rows = _model_quality_control(
        rows, space, state.atmosphere, options, rejections
    )
    if not rows:
        refused = rejections["already_assimilated"]
        if refused:
            raise ValueError(
                f"nothing new to analyse: of {decoded_total} decoded reports "
                f"{refused} are already in the background's assimilation "
                "chain and the rest failed quality control, so the "
                "background stands - analysing the same reports again would "
                "treat the state that contains them as an independent "
                "forecast"
            )
        raise ValueError(
            "every decoded observation was rejected by quality control; "
            "an empty analysis would be the background wearing a new hash"
        )
    # Cross-validation split (DA-5): a seeded tenth of each gated
    # variable's reports is judged, never analysed.
    families: dict[str, _Family] = {}
    withheld: dict[str, _Family] = {}
    for index, variable in enumerate(VARIABLE_TABLE):
        used, held = _withhold(
            [row for row in rows if row.variable == variable], index, options
        )
        families[variable] = _Family(used)
        withheld[variable] = _Family(held)
    clock.lap("quality_control_s")

    background_hx, background_hx_withheld = space.hx(
        state.atmosphere, families, withheld
    )
    clock.lap("background_operators_s")

    # Grid-space spreading geometry.
    grid = transform.grid
    lon_mesh, lat_mesh = np.meshgrid(grid.longitude_deg, grid.latitude_deg)
    grid_lat_rad = np.deg2rad(lat_mesh.ravel())
    grid_lon_rad = np.deg2rad(lon_mesh.ravel())
    g = model.grid_state(state.atmosphere, only=("ps", "p_full"))
    ps_grid = np.asarray(backend.to_numpy(g["ps"]), dtype=np.float64)
    p_full_grid = np.asarray(backend.to_numpy(g["p_full"]), dtype=np.float64)
    nlev, nlat, nlon = p_full_grid.shape
    ps_columns = ps_grid.reshape(-1)
    p_full_columns = p_full_grid.reshape(nlev, -1)
    sponge_base_pa = float(getattr(model, "sponge_base_pa", 0.0))
    sponge_columns = _sponge_columns(p_full_grid, sponge_base_pa)
    reports_in_sponge = int(sum(
        1 for family in families.values() for row in family.rows
        if row.level_pa is not None and row.level_pa < sponge_base_pa
    ))
    xp = _spread_module(transform)

    def spread(variable: str, innovation: np.ndarray, volume: bool):
        family = families[variable]
        ratio = options.background_error(variable) / family.error
        gains = ratio * ratio
        return _spread_column(
            family, innovation, gains, grid_lat_rad, grid_lon_rad, options,
            ps_columns=ps_columns if volume else None,
            p_full_columns=p_full_columns if volume else None,
            sponge_columns=sponge_columns if volume else None,
            sponge_base_pa=sponge_base_pa,
            xp=xp,
        )

    increments_maxabs: dict[str, float] = {}
    atmosphere = state.atmosphere
    fields = list(atmosphere.fields())

    def add_spectral(index: int, grid_increment):
        spectral = transform.project(transform.forward(
            backend.asarray(grid_increment, dtype=backend.float_dtype)
        ))
        fields[index] = fields[index] + spectral

    if families["surface_pressure_pa"].count:
        family = families["surface_pressure_pa"]
        d_ln = np.log(family.value) - np.log(background_hx["surface_pressure_pa"])
        delta = spread("surface_pressure_pa", d_ln, volume=False)
        increments_maxabs["log_surface_pressure"] = float(np.max(np.abs(delta)))
        add_spectral(3, delta.reshape(nlat, nlon))

    if families["temperature_k"].count:
        family = families["temperature_k"]
        d_t = family.value - background_hx["temperature_k"]
        delta_t = spread("temperature_k", d_t, volume=True)
        increments_maxabs["temperature_k"] = float(np.max(np.abs(delta_t)))
        exner = (p_full_grid / REFERENCE_PRESSURE_PA) ** KAPPA
        add_spectral(2, delta_t.reshape(nlev, nlat, nlon) / exner)

    wind_counts = (families["wind_u_m_s"].count, families["wind_v_m_s"].count)
    wind_balance: dict[str, object] = {
        "mode": options.wind_balance,
        "breakage": WIND_BALANCE_BREAKAGE,
    }
    if any(wind_counts):
        delta_u = np.zeros((nlev, nlat, nlon))
        delta_v = np.zeros((nlev, nlat, nlon))
        for variable, target in (("wind_u_m_s", delta_u), ("wind_v_m_s", delta_v)):
            family = families[variable]
            if family.count:
                d = family.value - background_hx[variable]
                target[:] = spread(variable, d, volume=True).reshape(
                    nlev, nlat, nlon
                )
                increments_maxabs[variable] = float(np.max(np.abs(target)))
        zeta_inc, div_inc, wind_balance = _balanced_wind_increment(
            model.vector, transform, delta_u, delta_v, options.wind_balance
        )
        fields[0] = fields[0] + zeta_inc
        fields[1] = fields[1] + div_inc
    clock.lap("spread_s")

    # The moisture update: dewpoint reports into specific humidity
    # (MOISTURE_UPDATE_DIVERGENCE), after the temperature increment so the
    # saturation cap reads the analysed temperature.
    moisture_report: dict[str, object] = {
        "enabled": bool(options.moisture_update),
        "reports": families["dewpoint_k"].count,
        "vertical_localization": (
            f"exp(-z / {options.humidity_decay_height_m:g} m) above the "
            "column's surface for a surface dewpoint report; an aloft "
            "dewpoint report carries the ADAS height-separation model of "
            "the temperature operator"
        ),
        "divergence_from_v1": MOISTURE_UPDATE_DIVERGENCE,
    }
    positivity_record = None
    q_sat = None
    if options.moisture_update and families["dewpoint_k"].count:
        family = families["dewpoint_k"]
        d_td = family.value - background_hx["dewpoint_k"]
        moisture_options = dataclasses.replace(
            options, surface_decay_height_m=options.humidity_decay_height_m
        )
        ratio = options.background_error("dewpoint_k") / family.error
        delta_td = _spread_column(
            family, d_td, ratio * ratio, grid_lat_rad, grid_lon_rad,
            moisture_options,
            ps_columns=ps_columns, p_full_columns=p_full_columns,
            sponge_columns=sponge_columns, sponge_base_pa=sponge_base_pa,
            xp=xp,
        ).reshape(nlev, nlat, nlon)
        interim = model.grid_state(
            atmosphere.with_fields(fields), only=("qv", "temperature", "p_full")
        )
        q_bg = np.asarray(backend.to_numpy(interim["qv"]), dtype=np.float64)
        t_an = np.asarray(backend.to_numpy(interim["temperature"]), dtype=np.float64)
        p_an = np.asarray(backend.to_numpy(interim["p_full"]), dtype=np.float64)
        del interim
        # The background dewpoint at every level from the vapor there; the
        # increment is the vapor of (that dewpoint plus the spread change)
        # minus the vapor of that dewpoint, bounded (bounded_vapor_increment).
        delta_q, over, under = bounded_vapor_increment(q_bg, t_an, p_an, delta_td)
        q_sat = specific_humidity_from_dewpoint(t_an, p_an)
        increments_maxabs["dewpoint_k"] = float(np.max(np.abs(delta_td)))
        increments_maxabs["specific_humidity_kg_kg"] = float(np.max(np.abs(delta_q)))
        # A report set that agrees with the background moves nothing: no
        # projection and no repair touch the vapor when the increment is
        # exactly zero everywhere.
        applied = increments_maxabs["specific_humidity_kg_kg"] > 0.0
        if applied:
            add_spectral(4, delta_q)
        moisture_report["applied"] = applied
        moisture_report["bounds"] = {
            "saturation_capped_points": int(np.count_nonzero(over)),
            "zero_floored_points": int(np.count_nonzero(under)),
            "saturation": "specific humidity at the analysed temperature and pressure, Bolton",
        }
        moisture_report["increment"] = {
            "dewpoint_k_maxabs": increments_maxabs["dewpoint_k"],
            "specific_humidity_kg_kg_maxabs": increments_maxabs["specific_humidity_kg_kg"],
            "column_water_kg_m2_global_mean": float(np.mean(
                np.sum(delta_q * np.diff(
                    np.asarray(backend.to_numpy(
                        model.grid_state(atmosphere, only=("p_half",))["p_half"]
                    ), dtype=np.float64), axis=0,
                ), axis=0) / GRAVITY_M_S2
            )),
        }
        # After the truncation: how many points the ringing put above
        # saturation that were not, and how much the positivity repair
        # (below) had to close.
        analysed_q = np.asarray(backend.to_numpy(model.grid_state(
            atmosphere.with_fields(fields), only=("qv",)
        )["qv"]), dtype=np.float64)
        moisture_report["after_truncation"] = {
            "newly_supersaturated_points": int(np.count_nonzero(
                (analysed_q > q_sat) & (q_bg <= q_sat)
            )),
            "negative_points": int(np.count_nonzero(analysed_q < 0.0)),
            "min_specific_humidity_kg_kg": float(np.min(analysed_q)),
        }
        del q_bg, t_an, p_an, delta_q, delta_td, analysed_q
    clock.lap("moisture_s")

    # Preserve global-mean surface pressure: the mass fixer owns that mean
    # and would re-absorb any shift on the first restart step, failing the
    # fixer-absorption gate; the pressure stream constrains gradients.
    xp_backend = backend.xp
    analysis_mean_pa = grid.global_mean(
        np.asarray(backend.to_numpy(xp_backend.exp(transform.inverse(fields[3]))))
    )
    mass_offset = math.log(
        grid.global_mean(ps_grid) / float(analysis_mean_pa)
    )
    fields[3] = transform.add_grid_constant(fields[3], mass_offset)

    analysis_atmosphere = atmosphere.with_fields(fields)
    # The analysis carries the chain: the background's plus this cycle's
    # assimilated reports (never the withheld ones - the state does not
    # contain them), on a copy so the background object is untouched.
    assimilated_rows = [row for family in families.values() for row in family.rows]
    withheld_total = int(sum(family.count for family in withheld.values()))
    physics = state.physics_state.copy()
    physics.metadata[ASSIMILATION_HISTORY_KEY] = _chain_for_analysis(
        chain, assimilated_rows, moment, options,
        background_sha256=str(background["self_sha256"]),
        withheld_count=withheld_total,
        background_step=int(background["step"]),
        filter_name=FILTER_NAME,
        streams=tuple(row.source for row in assimilated_rows),
    )
    analysis = ArwenGlobalState(analysis_atmosphere, state.surface, physics)
    if moisture_report.get("applied"):
        # The positivity contract of the vapor: the same repair the cold
        # start and every step apply closes the truncation's ringing
        # inside its column and books what it moved.
        analysis, negative_vapor, _tracer_floor, positivity_record = (
            model._repair_positivity(analysis)
        )
        positivity_record = dict(positivity_record)
        positivity_record["largest_negative_vapor_kg_kg"] = float(negative_vapor)
        moisture_report["positivity_repair"] = positivity_record
        analysis_atmosphere = analysis.atmosphere
        # What the analysis carries after the repair: the vapor the model
        # will step from, read against the same saturation.
        repaired_q = np.asarray(backend.to_numpy(model.grid_state(
            analysis_atmosphere, only=("qv",)
        )["qv"]), dtype=np.float64)
        moisture_report["after_repair"] = {
            "supersaturated_points": int(np.count_nonzero(repaired_q > q_sat)),
            "negative_points": int(np.count_nonzero(repaired_q < 0.0)),
            "min_specific_humidity_kg_kg": float(np.min(repaired_q)),
            "max_specific_humidity_kg_kg": float(np.max(repaired_q)),
        }
        del repaired_q
    del q_sat
    model.enforce(analysis)
    clock.lap("mass_and_positivity_s")

    analysis_hx, analysis_hx_withheld = space.hx(
        analysis_atmosphere, families, withheld
    )
    space.clear_cache()
    clock.lap("analysis_operators_s")

    # The DA scorecard: every row's departure before and after, by
    # stream, variable and region (da_scorecard).  Built from the same
    # operator values the gate below reads, so the card and the gate
    # cannot disagree about a number.
    departures = Departures.concatenate([
        part
        for variable in VARIABLE_TABLE
        for part in (
            Departures.from_rows(
                families[variable].rows, background_hx[variable],
                analysis_hx[variable], withheld=False,
            ),
            Departures.from_rows(
                withheld[variable].rows, background_hx_withheld[variable],
                analysis_hx_withheld[variable], withheld=True,
            ),
        )
    ])
    card = da_scorecard(departures, label=moment.isoformat(timespec="seconds"))

    variables_report: dict[str, object] = {}
    failed: list[str] = []
    for variable, family in families.items():
        held = withheld[variable]
        if family.count + held.count == 0:
            # No report of this variable was offered: nothing to judge and
            # nothing to report (the METAR cache carries no dewpoint).
            continue
        analysed = variable != "dewpoint_k" or bool(options.moisture_update)
        gated = analysed and (
            family.count + held.count >= options.gate_minimum_count
        )
        entry = {
            "analysed": analysed,
            "count": family.count,
            "units": VARIABLE_TABLE[variable]["units"],
            "o_minus_b": _statistics(family.value - background_hx[variable]),
            "o_minus_a": _statistics(family.value - analysis_hx[variable]),
            "withheld": {
                "count": held.count,
                "o_minus_b": _statistics(
                    held.value - background_hx_withheld[variable]
                ),
                "o_minus_a": _statistics(
                    held.value - analysis_hx_withheld[variable]
                ),
                "ids": [row.identity_hash() for row in held.rows],
            },
            "gated": gated,
        }
        if gated:
            judged = entry["withheld"]
            passed = judged["o_minus_a"]["rms"] < judged["o_minus_b"]["rms"]
            entry["gate_passed"] = passed
            if not passed:
                failed.append(variable)
        variables_report[variable] = entry

    ages_s = np.array([
        (moment - row.valid_time).total_seconds()
        for family in families.values() for row in family.rows
    ])
    per_source_counts: dict[str, dict[str, int]] = {}
    for family in families.values():
        for row in family.rows:
            per_source = per_source_counts.setdefault(row.source, {})
            per_source[row.variable] = per_source.get(row.variable, 0) + 1

    report = {
        "acknowledgement": RESEARCH_ACKNOWLEDGEMENT,
        "grade": "research-v1",
        "name": cfg.name,
        "config_hash": cfg.config_hash,
        "pins_hash": pins_hash(cfg.semi_implicit_scheme, cfg.integrator),
        "analysis_time_utc": moment.isoformat(timespec="seconds"),
        "background": {
            "path": None if background.get("path") is None else str(background["path"]),
            "self_sha256": str(background["self_sha256"]),
            "step": int(background["step"]),
            "time_s": float(background["time_s"]),
        },
        "obs_sources": sources,
        "options": options.identity(),
        "rejections": rejections,
        "rejection_breakage": REJECTION_BREAKAGE,
        "assimilated_by_source": per_source_counts,
        "assimilated_total": len(assimilated_rows),
        "withheld_total": withheld_total,
        "assimilated_report_ids": {
            variable: [row.identity_hash() for row in family.rows]
            for variable, family in families.items()
        },
        "assimilation_history": {
            "key": ASSIMILATION_HISTORY_KEY,
            "schema": ASSIMILATION_HISTORY_SCHEMA,
            "reports_in_background_chain": len(chain["reports"]),
            "refused_from_chain": rejections["already_assimilated"],
            "reports_in_analysis_chain": len(
                physics.metadata[ASSIMILATION_HISTORY_KEY]["reports"]
            ),
            "cycles_in_analysis_chain": len(
                physics.metadata[ASSIMILATION_HISTORY_KEY]["cycles"]
            ),
            "pruning": (
                "entries whose valid instant is older than maximum_age_s "
                "at this analysis time are dropped: they fail the age "
                "window at this and every later cycle"
            ),
        },
        "obs_age_minutes": {
            "mean": float(np.mean(ages_s)) / 60.0 if ages_s.size else 0.0,
            "max": float(np.max(ages_s)) / 60.0 if ages_s.size else 0.0,
        },
        "increment_maxabs": increments_maxabs,
        "sponge_exclusion": {
            "base_pa": sponge_base_pa,
            "sponged_levels": int(np.count_nonzero(
                np.any(sponge_columns, axis=1)
            )),
            "reports_inside": reports_in_sponge,
            "breakage": SPONGE_EXCLUSION_BREAKAGE,
        },
        "mass_preserving_log_offset": mass_offset,
        "mass_preservation": {
            "log_offset": mass_offset,
            "uniform_increment_removed_pa": float(analysis_mean_pa) - float(
                grid.global_mean(ps_grid)
            ),
            "global_mean_surface_pressure_pa": float(grid.global_mean(ps_grid)),
            "divergence": MASS_PRESERVATION_DIVERGENCE,
        },
        "lineage": {
            "filter": FILTER_NAME,
            "streams": sorted({row.source for row in assimilated_rows}),
            "background_self_sha256": str(background["self_sha256"]),
            "analysis_time_utc": moment.isoformat(timespec="seconds"),
            "chain_length": len(physics.metadata[ASSIMILATION_HISTORY_KEY]["cycles"]),
        },
        "scorecard": card,
        "wind_balance": wind_balance,
        "moisture_update": moisture_report,
        "spreading_array_module": "numpy" if xp is np else str(getattr(xp, "__name__", xp)),
        "variables": variables_report,
        "gate_of_record": {
            "rule": (
                "for every variable with at least "
                f"{options.gate_minimum_count} accepted reports, a seeded "
                f"{options.withheld_fraction:g} of them is withheld from the "
                "analysis and O-A rms must be smaller than O-B rms on those "
                "withheld reports; the assimilated-row statistics are "
                "diagnostics"
            ),
            "breakage": GATE_BREAKAGE,
            "passed": not failed,
            "failed_variables": failed,
        },
        "status": "pass" if not failed else "fail",
    }
    clock.lap("report_s")
    return analysis, report, dict(clock.readings)


def assimilate(
    cfg: ArwenGlobalConfig,
    checkpoint: str | Path,
    obs_locations: list[str],
    outdir: str | Path,
    *,
    analysis_time=None,
    options: AssimilationOptions | None = None,
    overwrite: bool = False,
    door_plan=None,
) -> dict[str, object]:
    """The file door: read a background checkpoint, :func:`analyse` it
    against the observation sources, write the analysis checkpoint and
    the report beside it.  ``timings_s`` in the report carries the wall
    of every phase, the door's own (model build, checkpoint read and
    write) beside the analysis's.

    ``door_plan`` is the memory plan the door priced this analysis at
    (``sizing.plan_run_memory``), and it goes into the report under
    ``sizer``.  An analysis builds the same model on the same card a
    forecast does and is now sized by the same door, so the report says
    what the sizer chose for it and against what; without it the one door
    of the three that leaves no record of its own plan is the one that
    used to leave no record because it never asked."""
    options = options or AssimilationOptions()
    clock = _Stopwatch()
    if not obs_locations:
        raise ValueError("assimilate requires at least one --obs source")
    transform = build_transform(cfg)
    model, _ = build_model_and_cold_state(
        cfg, transform, scratch_destination=Path(outdir))
    clock.lap("model_build_s")
    metadata, arrays = read_checkpoint(
        checkpoint,
        expected_config_hash=cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme,
        integrator=cfg.integrator,
    )
    state = state_from_checkpoint(metadata, arrays, transform.backend)
    model.enforce(state)
    clock.lap("checkpoint_read_s")

    output = Path(outdir)
    analysis_path = output / (
        f"arwen_global_analysis_step{int(metadata['step']):08d}.npz"
    )
    report_path = output / "assimilation-report.json"
    for owned in (analysis_path, report_path):
        if owned.exists() and not overwrite:
            raise FileExistsError(
                f"output {owned} exists; pass --overwrite to replace it"
            )

    sources, rows = load_observations(obs_locations)
    clock.lap("observations_read_s")
    analysis, report, timings = analyse(
        cfg, model, transform, state, rows,
        sources=sources,
        background={
            "path": str(checkpoint),
            "self_sha256": metadata["self_sha256"],
            "step": int(metadata["step"]),
            "time_s": float(metadata["time_s"]),
        },
        analysis_time=analysis_time,
        options=options,
    )
    clock.lap("analysis_s")

    write_checkpoint(
        analysis_path,
        analysis,
        config_hash=cfg.config_hash,
        to_numpy=transform.backend.to_numpy,
        trackers=metadata["run_trackers"],
        semi_implicit_scheme=cfg.semi_implicit_scheme,
        integrator=cfg.integrator,
    )
    written_metadata, _ = read_checkpoint(
        analysis_path,
        expected_config_hash=cfg.config_hash,
        semi_implicit_scheme=cfg.semi_implicit_scheme,
        integrator=cfg.integrator,
    )
    clock.lap("checkpoint_write_s")
    report["analysis"] = {
        "path": str(analysis_path),
        "self_sha256": written_metadata["self_sha256"],
    }
    report["timings_s"] = {**clock.readings, "analysis_phases": timings}
    report["sizer"] = None if door_plan is None else door_plan.receipt()
    finalized = _write_report(report_path, report)
    finalized["report_path"] = str(report_path)
    return finalized


__all__ = [
    "ASSIMILATION_HISTORY_KEY",
    "ASSIMILATION_HISTORY_SCHEMA",
    "FILTER_NAME",
    "MASS_PRESERVATION_DIVERGENCE",
    "AssimilationOptions",
    "DEFAULT_AIRCRAFT_LEVEL_SCALE_M",
    "DEFAULT_HUMIDITY_DECAY_HEIGHT_M",
    "MOISTURE_UPDATE_DIVERGENCE",
    "analyse",
    "bounded_vapor_increment",
    "load_observations",
    "specific_humidity_from_dewpoint",
    "GATE_BREAKAGE",
    "DEFAULT_BACKGROUND_ERRORS",
    "DEFAULT_WIND_BALANCE",
    "REJECTION_BREAKAGE",
    "VARIABLES_WITHOUT_OPERATOR",
    "CROSS_STREAM_CELL_DEG",
    "CROSS_STREAM_TIME_S",
    "cross_stream_duplicates",
    "SPONGE_EXCLUSION_BREAKAGE",
    "VERTICAL_WEIGHT_FLOOR",
    "WIND_BALANCE_BREAKAGE",
    "WIND_BALANCE_MODES",
    "assimilate",
]
