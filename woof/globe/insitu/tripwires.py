"""Named tripwires over the in-situ ledger.

A tripwire RECORDS - it never refuses.  The research-bound refusals in
``dynamics.enforce`` keep that role; a tripwire's job is to name the term
that moved and to get the state at onset onto disk while it is still a
state and not a post-mortem.

Every threshold cites the measurement that set it (the gate law,
2026-08-16) or is ``None`` = to be measured, which disables the wire.
Thresholds are held in ``THRESHOLDS`` so the citation and the number sit
together; ``measurement`` is the sentence a reader checks.

Sizing rule for the envelope wires (KE change, fixer jumps, negative
water, band growth): the trip sits at 2x the healthy maximum measured on
the reference runs named in each entry - a value twice anything a healthy
run produced is a change of regime, one exactly at the healthy maximum is
a wire that fires on the reference run itself.  Sizing rule for the
pre-warnings: halfway, in the metric, between the healthy extreme of the
reference run and the refusal bound, so the warning has the same lead the
healthy run had margin.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class TripwireSpec:
    name: str
    breakage: str
    #: "max": trips when value > threshold; "min": trips when value < threshold.
    direction: str
    threshold: float | None
    measurement: str

    @property
    def enabled(self) -> bool:
        return self.threshold is not None

    def trips(self, value: float) -> bool:
        if self.threshold is None:
            return False
        if self.direction == "max":
            return value > self.threshold
        return value < self.threshold


# Trailing windows.  The fixer means use 20 steps: the mass-fixer spin-up
# transient on the T63 GDAS reference decays over ~30 steps (audit
# 2026-09-01, VTW-2: 1.9e-6 -> 4.2e-5 at step 3 -> 3e-7 by step 36), so a
# 20-step trailing mean follows the decay instead of tripping on it.  The
# spectral window is 6 samples: one hour at the default 10-step cadence
# and the T63 dt of 300 s, the same horizon the pile-up that killed both
# T533 runs took to reach the 140 K gate (hours 3.0-3.5 from a healthy
# start), so the trailing mean still describes the pre-onset state when
# the band starts to grow.
FIXER_WINDOW_STEPS = 20
SPECTRAL_WINDOW_SAMPLES = 6

# Precision floors for the fixer-jump ratios: the healthy per-step ceilings
# runner.py measured (float64: T7 reference campaign, 360 steps; float32:
# T63 GDAS canary on the RTX 5090, 72 steps, 2026-08-31).  A ratio against
# a trailing mean below the floor is roundoff against roundoff.
FIXER_FLOORS = {
    "float64": {"mass_log_offset": 1.64e-7, "water_relative": 3.9e-8},
    "float32": {"mass_log_offset": 4.14e-5, "water_relative": 1.51e-6},
}

# Research bounds the refusals in dynamics.enforce apply.
TEMPERATURE_BOUNDS_K = (140.0, 380.0)
SURFACE_PRESSURE_BOUNDS_PA = (30_000.0, 120_000.0)

#: name -> (direction, threshold, measurement).  None = to be measured.
#
# Measurement runs of record (2026-09-01, numpy float64, one CPU core,
# on the tree that recorded them):
#   T63  = configs/verify/arwen_global_gdas_t63_48h.toml (GDAS 2026-08-30
#          18Z analysis, 40 levels, dt 300 s), ledger to hour 43.3 (step
#          520; the process then died of a host MemoryError inside the
#          transform, a machine event, not a model one - every maximum
#          below was already set by hour 24).
#   T21  = configs/verify/arwen_global_t21_baroclinic_ten_day.toml (analytic
#          baroclinic wave, 10 levels, dt 360 s), all 2400 steps, receipt
#          pass.
#   T3   = configs/verify/arwen_global_moist_smoke.toml, 24 steps.
# Envelope wires: 2x the larger of the T63 and T21 healthy maxima, taken
# over the armed period (after the 20-row window).  Pre-warnings: halfway
# between the worst healthy extreme and the refusal bound.
THRESHOLDS: dict[str, tuple[str, float | None, str]] = {
    "spectral_top_decile_growth": (
        "max", 2.1,
        "max over levels of top-decile band KE (rot+div) against its "
        "6-sample trailing mean: T63 healthy maximum 1.055 (52 samples, "
        "the band stays flat on a real analysis); 2x = 2.1.  KNOWN TRIP "
        "CLASS: the T21 analytic baroclinic wave fills an initially empty "
        "band as the wave grows, ratio first over 2.1 at step 80 (hour 8), "
        "peak 19.8 at hours 15-17 - a regime change in the band by "
        "construction, recorded, never refused; replaying the T63 and "
        "smoke ledgers through the live wires trips nothing",
    ),
    "mass_fixer_jump": (
        "max", 4.5,
        "|mass_fixer_log_offset| against max(20-step trailing mean, "
        "precision floor), armed period: T63 healthy maximum 1.53 (step "
        "41), T21 2.24 (step 49); 2x the larger = 4.5",
    ),
    "water_fixer_jump": (
        "max", 4.8,
        "|global_water_fixer_kg_m2| against max(20-step trailing mean, "
        "precision floor x water_total), armed period: T63 healthy maximum "
        "1.87 (step 40), T21 2.38 (step 2280); 2x the larger = 4.8",
    ),
    "levy_jump": (
        "max", 7.2,
        "positivity-repair + exchange-clamp levy against max(20-step "
        "trailing mean, the water fixer's floor), armed period: T63 healthy "
        "maximum 1.13 (step 100), T21 3.60 (step 931, the onset of a clamp "
        "episode from a roundoff-level history); 2x the larger = 7.2",
    ),
    "kinetic_energy_step_change": (
        "max", 3.6e-2,
        "|KE_t - KE_t-1| / KE_t-1 per step, armed period: T63 healthy "
        "maximum 1.6e-3 (step 22), T21 1.8e-2 (steps 260-262, the "
        "baroclinic wave's growth phase); 2x the larger = 3.6e-2.  STATED "
        "LIMIT: audit DN-1's slow amplification (rho 1.0094/step in "
        "divergence, ~1.9%/step in KE) sits under this wire; the T21 "
        "dt=4000 s blow-up (max|D| x1200 in 17 steps) is far above it",
    ),
    "temperature_floor_pre_warning": (
        "min", 160.0,
        "grid minimum temperature: T63 healthy minimum 180.8 K (step 16), "
        "T21 194.0 K, against the 140 K refusal; halfway from 180.8 = "
        "160.4 K, held at 160 K",
    ),
    "temperature_ceiling_pre_warning": (
        "max", 349.0,
        "grid maximum temperature: T63 healthy maximum 317.9 K (step 2), "
        "T21 301.3 K, against the 380 K refusal; halfway from 317.9 = "
        "349.0 K",
    ),
    "surface_pressure_floor_pre_warning": (
        "min", 38_000.0,
        "grid minimum surface pressure: T63 healthy minimum 46,310 Pa "
        "(step 3, the plateau columns after the hypsometric re-derivation "
        "onto T63 terrain, audit VTW-3), T21 90,854 Pa, against the "
        "30,000 Pa refusal; halfway from 46,310 = 38,155 Pa, held at 38,000",
    ),
    "surface_pressure_ceiling_pre_warning": (
        "max", 117_000.0,
        "grid maximum surface pressure: T63 healthy maximum 114,093 Pa "
        "(step 3), T21 106,995 Pa, against the 120,000 Pa refusal; halfway "
        "from 114,093 = 117,046 Pa, held at 117,000",
    ),
    "cfl_pre_warning": (
        "max", 0.74,
        "spectral_cfl / maximum_cfl: T63 healthy maximum 0.484 of the gate "
        "(step 1), T21 0.064; halfway from 0.484 to the refusal at 1.0 = "
        "0.742, held at 0.74.  Audit DN-1 shows the gate itself lags a "
        "blow-up already in progress, so this lead is all the wire can give",
    ),
    "negative_water_before_clamp": (
        "max", 8.0e-3,
        "-(grid minimum over the six species) after the positivity repair, "
        "i.e. the negative the next consumer clamps: T63 (40 levels) "
        "healthy maximum 1.5e-4 kg/kg (step 1), T21 2.5e-6, and the first "
        "real-data T63 run on the 20-level stack rang qv to ~4e-3 at sharp "
        "fronts every step (dynamics._repair_positivity, 2026-08-31); 2x "
        "the largest cited = 8e-3",
    ),
    "surface_reservoir_pre_warning": (
        "min", 0.1,
        "minimum column reservoir / initial global-mean reservoir: the "
        "T255 native run drained a 500 kg/m2 column to the floor in 37.7 h "
        "(13.3 kg/m2/h, dynamics._close_created_water), so a warning at "
        "10% of the initial mean gives ~3.8 h of notice at that measured "
        "drain rate before the reservoir refusal.  Healthy references stay "
        "far above it: T63 minimum 0.899 at hour 43, T21 0.897 at day 10",
    ),
}

BREAKAGES = {
    "spectral_top_decile_growth": (
        "energy piling up at the truncation scale: the terrain-locked "
        "near-truncation wave train that killed both T533 runs at the 140 K "
        "gate within 3.5 h (2026-08-31) grew there first"
    ),
    "mass_fixer_jump": (
        "the mass fixer absorbing a new surface-pressure sink: the fixer "
        "re-pins the global mean every step, so a leak that appears mid-run "
        "is invisible to the drift gate and shows only as a jump in what the "
        "fixer absorbs (runner.py: the 0.5%/step sink measured 5e-3 offset)"
    ),
    "water_fixer_jump": (
        "the global water fixer manufacturing water to hide a transport or "
        "physics leak that appears mid-run (runner.py: 1%-per-call qv "
        "deletion measured 2.4e-4 relative/step, invisible to the drift gate)"
    ),
    "levy_jump": (
        "the spectral-clamp levy (2.5 mm/day on the T63 reference, audit "
        "VTW-1) suddenly growing: the T255 ring-column drain that killed the "
        "run at hour 37.7 was this closure concentrated on a few columns"
    ),
    "kinetic_energy_step_change": (
        "a per-step kinetic-energy change outside the healthy envelope: "
        "audit DN-1 shows the CFL gate is a lagging detector of a blow-up "
        "already in progress (dt=4000 s at T21 passed cfl() at 0.068 and died "
        "17 steps later), so the KE step change is the earlier instrument"
    ),
    "temperature_floor_pre_warning": (
        "the 140 K research refusal that ended the T533 runs at hours 3.0-3.5 "
        "and the native baseline at hour 83.4 (polar sag) - this fires first"
    ),
    "temperature_ceiling_pre_warning": "the 380 K research refusal",
    "surface_pressure_floor_pre_warning": "the 30,000 Pa research refusal",
    "surface_pressure_ceiling_pre_warning": "the 120,000 Pa research refusal",
    "cfl_pre_warning": (
        "the spectral CFL refusal (a ValueError with no receipt of the state "
        "that produced it)"
    ),
    "negative_water_before_clamp": (
        "representation ringing of the spectral vapor field deep enough "
        "that a column cannot pay its own clip (the five condensate species "
        "are grid tracers since 2026-09-02 and never go below zero; the "
        "reservoir-floor death this wire was built for is impossible by "
        "construction and the wire now reads vapor alone)"
    ),
    "surface_reservoir_pre_warning": (
        "'requires more surface water than available': the reservoir refusal "
        "that killed the T255 run at hour 37.7 with no earlier signal"
    ),
}


def tripwire_specs(overrides: dict[str, float | None] | None = None):
    specs = {}
    for name, (direction, threshold, measurement) in THRESHOLDS.items():
        if overrides is not None and name in overrides:
            threshold = overrides[name]
        specs[name] = TripwireSpec(
            name=name, breakage=BREAKAGES[name], direction=direction,
            threshold=threshold, measurement=measurement,
        )
    return specs


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


class TripwireSet:
    """Stateful evaluator: feed rows and spectra samples in order."""

    def __init__(
        self,
        *,
        precision: str,
        maximum_cfl: float,
        enabled: bool = True,
        overrides: dict[str, float | None] | None = None,
        advective_cfl_refusal: bool = True,
    ):
        self.specs = tripwire_specs(overrides)
        self.enabled = bool(enabled)
        self.maximum_cfl = float(maximum_cfl)
        # The CFL wire warns of ONE breakage: the spectral CFL refusal,
        # which is a ValueError with no receipt of the state that produced
        # it.  An integrator that has no such refusal has nothing for this
        # wire to warn about, and on the semi-Lagrangian core the advective
        # Courant number is 1.5 at dt = 300 s and 2.2 at 450 s BY DESIGN,
        # so the wire fired on every step and wrote a 3.5 MB snapshot each
        # time (MEASURED 2026-09-06: four trips in forty T255 steps, and
        # the ledger observer at 125.9 ms of a 270.6 ms step).  A wire
        # whose breakage cannot happen is not a wire, it is a cost.
        self.advective_cfl_refusal = bool(advective_cfl_refusal)
        floors = FIXER_FLOORS.get(precision)
        if floors is None:
            raise ValueError(
                f"no measured fixer floor for precision {precision!r}"
            )
        self.floors = floors
        self._mass_history: deque[float] = deque(maxlen=FIXER_WINDOW_STEPS)
        self._water_history: deque[float] = deque(maxlen=FIXER_WINDOW_STEPS)
        self._levy_history: deque[float] = deque(maxlen=FIXER_WINDOW_STEPS)
        self._band_history: deque[list[float]] = deque(
            maxlen=SPECTRAL_WINDOW_SAMPLES
        )
        self._previous_kinetic: float | None = None
        self._initial_reservoir: float | None = None
        self._rows_seen = 0

    @property
    def armed(self) -> bool:
        """The envelope wires (fixer jumps, KE change) evaluate only once a
        full trailing window of rows exists: a ratio against a one-row
        history is not a trailing mean, and the T63 GDAS reference's
        spin-up (mass-fixer offset 1.9e-6 at step 1 -> 3.1e-5 at step 2, a
        15x 'jump' against its own first row, measured 2026-09-01) would
        trip every run at step 2.  The pre-warnings are absolute and act
        from the first row."""
        return self._rows_seen >= FIXER_WINDOW_STEPS

    def describe(self) -> dict[str, dict[str, object]]:
        return {
            name: {
                "direction": spec.direction,
                "threshold": spec.threshold,
                "enabled": bool(self.enabled and spec.enabled),
                "measurement": spec.measurement,
                "breakage": spec.breakage,
            }
            for name, spec in self.specs.items()
        }

    def _trip(self, name, row, term, value) -> dict | None:
        spec = self.specs[name]
        if not (self.enabled and spec.enabled) or not spec.trips(value):
            return None
        return {
            "kind": "trip",
            "tripwire": name,
            "step": int(row["step"]),
            "time_s": float(row["time_s"]),
            "term": term,
            "value": float(value),
            "threshold": float(spec.threshold),
            "direction": spec.direction,
            "breakage": spec.breakage,
        }

    def evaluate_row(self, row: dict) -> list[dict]:
        """Evaluate one flushed step row against the history before it."""
        terms = row["terms"]
        metrics = row["metrics"]
        trips: list[dict] = []

        def finite(value):
            return value is not None and value == value and abs(value) != float("inf")

        armed = self.armed
        self._rows_seen += 1

        # Fixer jumps: ratio against the trailing mean, floored.
        mass = abs(float(metrics.get("mass_fixer_log_offset", 0.0)))
        mass_floor = self.floors["mass_log_offset"]
        if armed:
            ratio = mass / max(_mean(self._mass_history), mass_floor)
            trips.append(self._trip("mass_fixer_jump", row, "mass_fixer_log_offset", ratio))
        self._mass_history.append(mass)

        water = abs(float(metrics.get("global_water_fixer_kg_m2", 0.0)))
        total = terms.get("water_total_kg_m2")
        water_floor = self.floors["water_relative"] * (
            abs(total) if finite(total) else 1.0
        )
        # A DRY atmosphere holds no water, so the relative floor built
        # from its total is exactly zero and so is the trailing history:
        # the two ratios below then divided by zero and killed the run at
        # its first ledger flush (MEASURED 2026-09-06 on the Jablonowski
        # and Williamson case, which carries no water by construction, on
        # both integrators).  A field that is identically zero cannot jump,
        # so the floor is the smallest positive number instead, and every
        # run that holds any water at all reads exactly what it read
        # before: max() with a value below the floor it already had
        # returns that floor.
        if not (water_floor > 0.0):
            water_floor = 5.0e-324
        if armed:
            ratio = water / max(_mean(self._water_history), water_floor)
            trips.append(self._trip("water_fixer_jump", row, "global_water_fixer_kg_m2", ratio))
        self._water_history.append(water)

        # The levy shares the water fixer's floor: on the T21 baroclinic
        # ten-day reference the levy sits at roundoff for hundreds of steps
        # and one 1e-6 kg/m2 clamp then read as a 1.7e8x "jump" against a
        # zero history (step 768, measured 2026-09-01).
        levy = abs(float(metrics.get("levy_kg_m2", 0.0)))
        if armed:
            ratio = levy / max(_mean(self._levy_history), water_floor)
            trips.append(self._trip("levy_jump", row, "levy_kg_m2", ratio))
        self._levy_history.append(levy)

        kinetic = terms.get("kinetic_energy_j_m2")
        if finite(kinetic):
            if armed and self._previous_kinetic is not None and self._previous_kinetic > 0.0:
                change = abs(kinetic - self._previous_kinetic) / self._previous_kinetic
                trips.append(self._trip(
                    "kinetic_energy_step_change", row, "kinetic_energy_j_m2", change
                ))
            self._previous_kinetic = kinetic

        for name, term in (
            ("temperature_floor_pre_warning", "temperature_min_k"),
            ("temperature_ceiling_pre_warning", "temperature_max_k"),
            ("surface_pressure_floor_pre_warning", "surface_pressure_min_pa"),
            ("surface_pressure_ceiling_pre_warning", "surface_pressure_max_pa"),
        ):
            value = terms.get(term)
            if finite(value):
                trips.append(self._trip(name, row, term, value))

        cfl = metrics.get("spectral_cfl")
        if (finite(cfl) and self.maximum_cfl > 0.0
                and self.advective_cfl_refusal):
            trips.append(self._trip("cfl_pre_warning", row, "spectral_cfl", cfl / self.maximum_cfl))

        worst_name = None
        worst = 0.0
        for species in ("qv", "qc", "qr", "qi", "qs", "qg"):
            value = terms.get(f"min_{species}_kg_kg")
            if finite(value) and -value > worst:
                worst = -value
                worst_name = f"min_{species}_kg_kg"
        if worst_name is not None:
            trips.append(self._trip("negative_water_before_clamp", row, worst_name, worst))

        reservoir_min = terms.get("surface_water_min_kg_m2")
        reservoir_mean = terms.get("water_surface_kg_m2")
        if self._initial_reservoir is None and finite(reservoir_mean):
            self._initial_reservoir = float(reservoir_mean)
        if finite(reservoir_min) and self._initial_reservoir and self._initial_reservoir > 0.0:
            trips.append(self._trip(
                "surface_reservoir_pre_warning", row, "surface_water_min_kg_m2",
                reservoir_min / self._initial_reservoir,
            ))
        return [trip for trip in trips if trip is not None]

    def evaluate_spectra(self, sample: dict) -> list[dict]:
        band = [
            r + d
            for r, d in zip(
                sample["rot_top_decile_by_level"], sample["div_top_decile_by_level"]
            )
        ]
        trips: list[dict] = []
        if len(self._band_history) >= 2:
            worst = 0.0
            worst_level = None
            for level, value in enumerate(band):
                trailing = _mean(history[level] for history in self._band_history)
                if trailing > 0.0 and value / trailing > worst:
                    worst = value / trailing
                    worst_level = level
            if worst_level is not None:
                trip = self._trip(
                    "spectral_top_decile_growth", sample,
                    f"top_decile_kinetic_energy_level_{worst_level}", worst,
                )
                if trip is not None:
                    trips.append(trip)
        self._band_history.append(band)
        return trips


__all__ = [
    "BREAKAGES",
    "FIXER_FLOORS",
    "FIXER_WINDOW_STEPS",
    "SPECTRAL_WINDOW_SAMPLES",
    "THRESHOLDS",
    "TripwireSet",
    "TripwireSpec",
    "tripwire_specs",
]
