"""The resident ensemble of WOOF global: N members, one model, one card.

Design (ruling 2026-09-06, item 1): an N-member ensemble at T127 with the
40-level stack, every member resident on one card in ONE process sharing
the transform tables, the statics, the kernels and the physics workspace.
Member state at T127 is the spectral atmosphere, its grid tracers, its
surface and its physics namespace; the 4.66 GiB a single T127 process
holds is workspace and tables, so members step SEQUENTIALLY through the
shared :class:`~woof.globe.dynamics.MoistHybridModel` and the
resident footprint grows by one member's arrays per member.  The receipt
measures both: resident bytes per member (:meth:`GlobalEnsemble.resident_bytes`)
and wall per member-step (:class:`StepTiming`).

What is shared and why it is safe to share: the model object (transform,
Legendre tables, vertical coordinate, surface geopotential, the physics
bridge whose lazily built modules, the RRTMGP solver, Noah's tables, the
cumulus scheme, are workspace), because every step-to-step quantity of the
native suite lives in the member's own ``physics_state`` (restart is bit
for bit, so nothing else carries).  The native runtime caches ONE thing
from the first batch it sees, the frozen-column mask derived from the
sea-ice fraction and the land fraction; the members share the surface
statics and the analysed sea ice (the filter does not perturb them), and
:meth:`GlobalEnsemble.from_state` refuses a member set whose sea-ice or
land planes differ, by name.  The conservation targets (the mass fixer's
global-mean surface pressure, the water fixer's total water) are per
member and are swapped onto the model before each member's step
(``MoistHybridModel.set_conservation_targets``); the synthesis memo is
released between members so one member's grid fields never serve another
and the peak stays one member's.

The checkpoint set is one hash-bound checkpoint per member (the run
checkpoint format, ``checkpoint.write_checkpoint``) plus a manifest
(:data:`ENSEMBLE_MANIFEST_NAME`) naming every member file, its
``self_sha256``, the seeds, the options and the spread.
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..checkpoint import (
    normalize_trackers, read_checkpoint, state_from_checkpoint, trajectory_from_checkpoint, write_checkpoint,
)
from ..constants import SPECTRAL_FIELDS
from ..state import ArwenGlobalState
from ..semilag.step import SEMILAG_INTEGRATORS
from .options import EnsembleOptions
from .perturbations import (
    draw_perturbation,
    lagged_differences,
    member_rng,
    perturbed_state,
    scale_difference,
)

#: Manifest the ensemble writes beside its member checkpoints.
ENSEMBLE_MANIFEST_NAME = "arwen-global-ensemble.json"
ENSEMBLE_SCHEMA = "gpuwm.arwen-global-ensemble/v1"
MEMBER_PREFIX = "arwen_global_member"

#: Why two members with different frozen surfaces cannot share one model.
SHARED_SURFACE_BREAKAGE = (
    "the native runtime derives its frozen-column mask (sea ice and land) "
    "from the first batch it sees and caches it for the process; a member "
    "whose sea-ice fraction or land fraction differs from the others would "
    "run the other members' frozen columns and skip its own"
)


class MemberStepError(FloatingPointError):
    """A member's step failed in the model (a non-finite prognostic in
    the physics, a refused gate): the member's index, the ensemble time,
    the state that entered the step (after its IAU portion) and the pending
    increment ride on the exception, so the filter can write them beside
    the ensemble and the death is reproducible.  Before this the message
    named the field alone and the cycle's death left nothing on disk."""

    def __init__(self, message: str, *, member: int, time_s: float, dt_s: float, state=None,
                 pending=None, pending_steps_left: int = 0, pending_steps_total: int = 0, trajectory=None):
        super().__init__(message)
        self.member = int(member)
        self.time_s = float(time_s)
        self.dt_s = float(dt_s)
        self.state = state
        self.pending = pending
        self.trajectory = trajectory
        self.pending_steps_left = int(pending_steps_left)
        self.pending_steps_total = int(pending_steps_total)


def member_checkpoint_path(outdir: Path, member: int, step: int) -> Path:
    return Path(outdir) / f"{MEMBER_PREFIX}{int(member):03d}_step{int(step):08d}.npz"


@dataclass
class MemberTargets:
    """The conservation epoch of one member: what its fixers hold."""

    mass_pa: float
    total_water_kg_m2: float


@dataclass
class StepTiming:
    """Wall of one ensemble step, for the receipt."""

    members: int
    wall_s: float
    per_member_s: list[float] = field(default_factory=list)

    @property
    def mean_per_member_s(self) -> float:
        return float(np.mean(self.per_member_s)) if self.per_member_s else 0.0

    def as_record(self) -> dict[str, object]:
        return {
            "members": int(self.members),
            "wall_s": float(self.wall_s),
            "mean_per_member_s": self.mean_per_member_s,
            "min_per_member_s": float(min(self.per_member_s)) if self.per_member_s else 0.0,
            "max_per_member_s": float(max(self.per_member_s)) if self.per_member_s else 0.0,
        }


def _array_bytes(array) -> int:
    return int(getattr(array, "nbytes", 0))


def state_bytes(state: ArwenGlobalState) -> dict[str, int]:
    """Bytes of one member's arrays by namespace (device or host, where
    the arrays live)."""
    spectral = sum(_array_bytes(v) for v in state.atmosphere.fields())
    tracers = sum(_array_bytes(v) for v in state.atmosphere.grid_tracers().values() if v is not None)
    surface = sum(_array_bytes(v) for v in state.surface.arrays().values())
    physics = sum(_array_bytes(v) for v in state.physics_state.arrays.values())
    return {
        "spectral": spectral,
        "grid_tracers": tracers,
        "surface": surface,
        "physics": physics,
        "total": spectral + tracers + surface + physics,
    }


def _same_planes(a, b, xp) -> bool:
    return bool(xp.array_equal(xp.asarray(a), xp.asarray(b)))


@dataclass
class GlobalEnsemble:
    """N resident members sharing one model.

    Build one with :meth:`from_state` (a base state at the ensemble's
    truncation, perturbed into N members) or :meth:`read` (a member
    checkpoint set written by :meth:`write`).  ``model`` and ``transform``
    are the ensemble's own (``runner.build_model_and_cold_state`` on the
    ensemble config, see :func:`ensemble_config`), ``cfg`` that config.
    """

    cfg: object
    model: object
    transform: object
    options: EnsembleOptions
    members: list[ArwenGlobalState]
    targets: list[MemberTargets]
    #: Per-member seeds the initial perturbations were drawn from.
    seeds: list[int]
    #: How the ensemble was made, for the receipt.
    provenance: dict = field(default_factory=dict)
    #: Analyses applied so far (the additive-inflation draws are keyed on it).
    cycles: int = 0
    #: Every step's timing since construction or read.
    timings: list[StepTiming] = field(default_factory=list)
    #: Incremental analysis update (amendment D): one spectral increment
    #: per member awaiting application, an equal portion added before
    #: each of ``pending_steps_total`` steps; None when nothing is pending.
    pending_increments: list[dict] | None = None
    pending_steps_total: int = 0
    pending_steps_left: int = 0
    #: The semi-Lagrangian core's second time level, PER MEMBER (None
    #: before a member's first step: the start-up step).  The core keeps
    #: it on the model (``model.trajectory_state``), which is one object
    #: the members share, so the ensemble swaps each member's own level
    #: onto the model before its step and takes it back after, as it does
    #: the conservation targets.  Before this every member extrapolated
    #: its departure points and its nonlinear terms from the level the
    #: PREVIOUS member left on the model: the grade of record's member 15
    #: died on a non-finite theta in its second hour, and the same state
    #: stepped alone survived (2026-09-07).
    trajectories: list = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def step(self) -> int:
        return int(self.members[0].step)

    @property
    def time_s(self) -> float:
        return float(self.members[0].time_s)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_state(
        cls, cfg, model, transform, base: ArwenGlobalState,
        options: EnsembleOptions, *, lagged_states=(), progress=None,
    ) -> "GlobalEnsemble":
        """N members around ``base`` (a state at the ensemble truncation):
        each is ``base`` plus a spectral perturbation drawn from the
        member's own stream (:mod:`woof.globe.da.perturbations`)
        plus, with ``lagged_states`` given, a time-lagged difference of the
        analyses the caller supplies (states at the same truncation; each
        consecutive difference, scaled to the perturbation amplitude, joins
        a member with a random sign).  The surface, the physics namespace
        and the grid tracers are ``base``'s own copies.  The provenance
        states every amplitude, the lagged pairs and the seeds."""
        if int(transform.truncation) != int(options.truncation):
            raise ValueError(
                f"the ensemble transform is T{transform.truncation} but the options say "
                f"T{options.truncation}; build the model from ensemble_config(cfg, options)"
            )
        model.enforce(base)
        differences = []
        if lagged_states:
            if len(lagged_states) < 2:
                raise ValueError("lagged_states needs at least two states to difference")
            for diff in lagged_differences(list(lagged_states), transform):
                scaled, factor = scale_difference(
                    model, transform, base.atmosphere, diff,
                    float(options.perturbation_temperature_k),
                )
                differences.append((scaled, factor))
        members: list[ArwenGlobalState] = []
        targets: list[MemberTargets] = []
        seeds: list[int] = []
        draws: list[dict] = []
        lagged_record: list[dict] = []
        for k in range(int(options.members)):
            rng = member_rng(options.seed, k, "initial")
            seed_k = int(rng.integers(0, 2 ** 31 - 1))
            seeds.append(seed_k)
            increments, record = draw_perturbation(model, transform, base.atmosphere, options, rng)
            if differences and options.lagged_difference_weight > 0.0:
                index = k % len(differences)
                sign = 1.0 if rng.random() < 0.5 else -1.0
                scaled, factor = differences[index]
                weight = float(options.lagged_difference_weight) * sign
                for name in SPECTRAL_FIELDS:
                    increments[name] = increments[name] + weight * scaled[name]
                lagged_record.append({"member": k, "pair": index, "sign": sign, "scale_factor": factor})
            state = perturbed_state(model, transform, base, increments)
            state.atmosphere.time_s = base.atmosphere.time_s
            state.atmosphere.step = base.atmosphere.step
            targets.append(MemberTargets(
                mass_pa=model.initialize_mass_target(state.atmosphere),
                total_water_kg_m2=model.initialize_water_target(state),
            ))
            members.append(state)
            draws.append(record)
            model.release_syntheses()
            if progress is not None:
                progress(k, state)
        ensemble = cls(
            cfg=cfg, model=model, transform=transform, options=options,
            members=members, targets=targets, seeds=seeds,
            provenance={
                "schema": ENSEMBLE_SCHEMA,
                "made_by": "GlobalEnsemble.from_state",
                "base_step": int(base.step),
                "base_time_s": float(base.time_s),
                "options": options.identity(),
                "initial_perturbations": {
                    "family": "spectral (perturbations.draw_perturbation)",
                    "first_member_record": draws[0] if draws else None,
                    "lagged_pairs": len(differences),
                    "lagged_members": lagged_record,
                },
            },
        )
        ensemble._check_shared_surface()
        ensemble.provenance["spread_at_construction"] = ensemble.spread()
        return ensemble

    def _check_shared_surface(self) -> None:
        xp = self.transform.backend.xp
        first = self.members[0].surface
        for k, member in enumerate(self.members[1:], start=1):
            if not _same_planes(first.sea_ice_fraction, member.surface.sea_ice_fraction, xp) or \
                    not _same_planes(first.land_fraction, member.surface.land_fraction, xp):
                raise ValueError(
                    f"member {k} does not share member 0's sea-ice and land planes, "
                    f"so the members cannot share one physics bridge: {SHARED_SURFACE_BREAKAGE}"
                )

    @classmethod
    def read(cls, cfg, model, transform, outdir: str | Path, options: EnsembleOptions | None = None) -> "GlobalEnsemble":
        """The ensemble a manifest and its member checkpoints hold."""
        output = Path(outdir)
        manifest = json.loads((output / ENSEMBLE_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("schema") != ENSEMBLE_SCHEMA:
            raise ValueError(f"{output / ENSEMBLE_MANIFEST_NAME} is not an ensemble manifest")
        if options is None:
            options = EnsembleOptions(**{
                k: v for k, v in manifest["options"].items()
                if k in {f.name for f in dataclasses.fields(EnsembleOptions)}
            })
        members = []
        targets = []
        trajectories = []
        for entry in manifest["members"]:
            path = output / entry["file"]
            metadata, arrays = read_checkpoint(
                path, expected_config_hash=cfg.config_hash,
                semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
            )
            if metadata["self_sha256"] != entry["self_sha256"]:
                raise ValueError(f"member checkpoint {path} does not carry the manifest's hash")
            state = state_from_checkpoint(metadata, arrays, transform.backend)
            model.enforce(state)
            members.append(state)
            # The member's own second time level, so a resumed ensemble
            # continues each member as the uninterrupted one would.
            trajectories.append(trajectory_from_checkpoint(metadata, arrays, transform.backend))
            targets.append(MemberTargets(
                mass_pa=float(entry["mass_target_pa"]),
                total_water_kg_m2=float(entry["total_water_target_kg_m2"]),
            ))
        ensemble = cls(
            cfg=cfg, model=model, transform=transform, options=options,
            members=members, targets=targets,
            seeds=[int(s) for s in manifest["seeds"]],
            provenance={**manifest.get("provenance", {}), "read_from": str(output)},
            cycles=int(manifest.get("cycles", 0)),
            trajectories=trajectories,
        )
        ensemble._check_shared_surface()
        return ensemble

    def write(self, outdir: str | Path, *, trackers=None, label: str | None = None) -> dict:
        """One hash-bound checkpoint per member
        (``arwen_global_member{k:03d}_step{step:08d}.npz``) plus the
        manifest (:data:`ENSEMBLE_MANIFEST_NAME`: members, seeds, options,
        spread summary, per-member checkpoint sha256).  Returns the
        manifest."""
        output = Path(outdir)
        output.mkdir(parents=True, exist_ok=True)
        entries = []
        trackers = normalize_trackers(trackers)
        for k, (state, target) in enumerate(zip(self.members, self.targets)):
            path = member_checkpoint_path(output, k, state.step)
            write_checkpoint(
                path, state, config_hash=self.cfg.config_hash,
                to_numpy=self.transform.backend.to_numpy, trackers=trackers,
                semi_implicit_scheme=self.cfg.semi_implicit_scheme,
                integrator=self.cfg.integrator,
                trajectory=self.trajectory_of(k),
            )
            metadata, _ = read_checkpoint(path)
            entries.append({
                "member": k, "file": path.name, "self_sha256": metadata["self_sha256"],
                "mass_target_pa": target.mass_pa,
                "total_water_target_kg_m2": target.total_water_kg_m2,
            })
        manifest = {
            "schema": ENSEMBLE_SCHEMA,
            "label": label,
            "config_hash": self.cfg.config_hash,
            "truncation": int(self.transform.truncation),
            "step": self.step,
            "time_s": self.time_s,
            "members": entries,
            "seeds": list(self.seeds),
            "options": self.options.identity(),
            "cycles": int(self.cycles),
            "spread": self.spread(),
            "resident_bytes": self.resident_bytes(),
            "provenance": self.provenance,
        }
        write_manifest(output / ENSEMBLE_MANIFEST_NAME, manifest)
        return manifest

    # -- integration ------------------------------------------------------

    def _sync(self) -> None:
        backend = self.transform.backend
        if getattr(backend, "name", "numpy") == "cupy":
            backend.xp.cuda.runtime.deviceSynchronize()

    def schedule_increments(self, increments: list[dict], window_s: float) -> None:
        """Store one spectral increment per member for incremental
        application over ``window_s`` of integration: ``step_all`` adds
        ``increment / n`` before each of the next ``n = window_s / dt``
        steps (``n`` is fixed at the first step from its ``dt``).  A second
        schedule while one is pending is refused: two windows' increments
        would blend into one and neither would be the analysis."""
        if len(increments) != self.size:
            raise ValueError("schedule_increments needs one increment per member")
        if self.pending_increments is not None:
            raise ValueError(
                "an incremental analysis update is still pending "
                f"({self.pending_steps_left} of {self.pending_steps_total} steps left); a second "
                "schedule would blend two analyses into one increment"
            )
        self.pending_increments = [dict(inc) for inc in increments]
        self.pending_window_s = float(window_s)
        self.pending_steps_total = 0
        self.pending_steps_left = 0

    def _apply_pending_portion(self, k: int, state: ArwenGlobalState, dt_s: float) -> ArwenGlobalState:
        if self.pending_increments is None:
            return state
        if self.pending_steps_total == 0:
            n = self.pending_window_s / float(dt_s)
            if abs(n - round(n)) > 1.0e-6 or round(n) < 1:
                raise ValueError(
                    f"the IAU window {self.pending_window_s:g} s is not a whole number of {dt_s:g} s steps"
                )
            self.pending_steps_total = int(round(n))
            self.pending_steps_left = self.pending_steps_total
        portion = 1.0 / float(self.pending_steps_total)
        inc = self.pending_increments[k]
        fields = [getattr(state.atmosphere, name) + inc[name] * portion for name in SPECTRAL_FIELDS]
        new = ArwenGlobalState(state.atmosphere.with_fields(fields), state.surface, state.physics_state)
        new, _n, _t, _f = self.model._repair_positivity(new)
        self.model.enforce(new)
        return new

    def step_all(self, dt_s: float) -> StepTiming:
        """Advance every member one step through the shared model, in
        member order, each under its own conservation targets; the memo of
        syntheses is released between members; a pending IAU portion is
        added to each member before its step.  Returns the timing."""
        model = self.model
        per_member = []
        self._sync()
        t0 = time.perf_counter()
        pending = self.pending_increments is not None
        for k, state in enumerate(self.members):
            target = self.targets[k]
            model.set_conservation_targets(target.mass_pa, target.total_water_kg_m2)
            model.set_trajectory_state(self.trajectory_of(k))
            t_member = time.perf_counter()
            try:
                if pending:
                    state = self._apply_pending_portion(k, state, float(dt_s))
                new_state, _metrics = model.step(state, float(dt_s))
            except FloatingPointError as exc:
                portion = ""
                if pending:
                    done = self.pending_steps_total - self.pending_steps_left
                    portion = f", IAU portion {done + 1} of {self.pending_steps_total}"
                raise MemberStepError(
                    f"member {k} of {self.size} at t={self.time_s:g} s (step {int(state.step)}, "
                    f"dt {float(dt_s):g} s{portion}): {exc}",
                    member=k, time_s=self.time_s, dt_s=float(dt_s), state=state,
                    pending=(self.pending_increments[k] if pending else None),
                    pending_steps_left=self.pending_steps_left,
                    pending_steps_total=self.pending_steps_total,
                    trajectory=self.trajectory_of(k),
                ) from exc
            self.store_trajectory(k, model.trajectory_state())
            model.set_trajectory_state(None)
            model.release_syntheses()
            self._sync()
            per_member.append(time.perf_counter() - t_member)
            self.members[k] = new_state
        if pending:
            self.pending_steps_left -= 1
            if self.pending_steps_left <= 0:
                self.pending_increments = None
                self.pending_steps_total = 0
                self.pending_steps_left = 0
                self.open_epochs()
        timing = StepTiming(members=self.size, wall_s=time.perf_counter() - t0, per_member_s=per_member)
        self.timings.append(timing)
        return timing

    def advance_to(self, time_s: float, dt_s: float, *, progress=None, observer=None) -> list[StepTiming]:
        """Steps of ``dt_s`` until every member's time is ``time_s``;
        ``observer(ensemble, time_s)`` is called after every step (an
        ObservationWindow evaluating the reports due at that time)."""
        target = float(time_s)
        if target < self.time_s - 1.0e-9:
            raise ValueError(f"advance_to {target:g} s lies before the ensemble's {self.time_s:g} s")
        steps = (target - self.time_s) / float(dt_s)
        if abs(steps - round(steps)) > 1.0e-6:
            raise ValueError(
                f"{target - self.time_s:g} s is not a whole number of {dt_s:g} s steps"
            )
        timings = []
        for _ in range(int(round(steps))):
            timings.append(self.step_all(dt_s))
            if observer is not None:
                observer(self, self.time_s)
            if progress is not None:
                progress(self, timings[-1])
        return timings

    def trajectory_of(self, k: int):
        """Member ``k``'s second time level, or None before its first step."""
        return self.trajectories[k] if k < len(self.trajectories) else None

    def store_trajectory(self, k: int, trajectory) -> None:
        while len(self.trajectories) <= k:
            self.trajectories.append(None)
        self.trajectories[k] = trajectory

    def open_epochs(self) -> None:
        """A new conservation epoch for every member from its current
        state (after an analysis, as the cycle door opens one)."""
        model = self.model
        for k, state in enumerate(self.members):
            self.targets[k] = MemberTargets(
                mass_pa=model.initialize_mass_target(state.atmosphere),
                total_water_kg_m2=model.initialize_water_target(state),
            )

    # -- diagnostics ------------------------------------------------------

    def mean_spectral(self, *, include_pending: bool = False) -> dict[str, object]:
        """The ensemble mean of the five spectral fields, by name; with
        ``include_pending`` the pending IAU increments (the part not yet
        added) are counted in."""
        out = {}
        r = float(self.size)
        for name in SPECTRAL_FIELDS:
            total = None
            for k, member in enumerate(self.members):
                value = getattr(member.atmosphere, name)
                if include_pending and self.pending_increments is not None:
                    remaining = (self.pending_steps_left / self.pending_steps_total
                                 if self.pending_steps_total else 1.0)
                    value = value + self.pending_increments[k][name] * remaining
                total = value.copy() if total is None else total + value
            out[name] = total / r
        return out

    def grid_fields(self, names=("u", "v", "theta", "qv", "lnps")):
        """Stacked grid fields of every member: ``{name: (R, ...)}`` on the
        model's namespace plus the ensemble-mean ``ln_p_full``
        ``(nlev, nlat, nlon)`` and ``ln_ps`` ``(nlat, nlon)``.  ``psi``
        and ``chi`` are the streamfunction and velocity potential
        synthesised from the member's vorticity and divergence through
        the inverse Laplacian (their global means are zero)."""
        xp = self.transform.backend.xp
        potentials = {"psi": "vorticity", "chi": "divergence"}
        keys = set()
        for n in names:
            if n in potentials:
                continue
            keys.add({"lnps": "logps"}.get(n, n))
        keys |= {"p_full", "logps"}
        stacks: dict[str, list] = {n: [] for n in names}
        ln_p_sum = None
        ln_ps_sum = None
        for member in self.members:
            g = self.model.grid_state(member.atmosphere, only=tuple(sorted(keys)))
            for n in names:
                if n in potentials:
                    stacks[n].append(self.transform.inverse(
                        self.transform.inverse_laplacian(getattr(member.atmosphere, potentials[n]))))
                else:
                    stacks[n].append(g["logps"] if n == "lnps" else g[n])
            lnp = xp.log(g["p_full"])
            ln_p_sum = lnp if ln_p_sum is None else ln_p_sum + lnp
            ln_ps_sum = g["logps"] if ln_ps_sum is None else ln_ps_sum + g["logps"]
            self.model.release_syntheses()
        fields = {n: xp.stack(stacks[n]) for n in names}
        return fields, ln_p_sum / self.size, ln_ps_sum / self.size

    def spread(self, states=None, *, area_weighted: bool = True) -> dict[str, float]:
        """Grid-space ensemble spread per analysis field (the area-weighted
        global rms of the pointwise spread, levels averaged): ``u``, ``v``,
        ``temperature_k``, ``qv``, ``surface_pressure_pa``; of ``states``
        when given (a member list that is not the ensemble's own, the IAU
        receipt's analysed copies).  ``area_weighted`` False takes the
        equal-weight mean over the Gaussian gridpoints instead (the form
        the first measurements used; it over-weights the polar rings by
        1/cos(latitude) against the area-weighted rmse it was set beside,
        so the two are not one ratio's numerator and denominator)."""
        xp = self.transform.backend.xp
        to_numpy = self.transform.backend.to_numpy
        grid = self.transform.grid
        acc: dict[str, list] = {"u": [], "v": [], "temperature_k": [], "qv": [], "surface_pressure_pa": []}
        for member in (self.members if states is None else states):
            g = self.model.grid_state(member.atmosphere, only=("u", "v", "temperature", "qv", "ps"))
            acc["u"].append(g["u"])
            acc["v"].append(g["v"])
            acc["temperature_k"].append(g["temperature"])
            acc["qv"].append(g["qv"])
            acc["surface_pressure_pa"].append(g["ps"])
            self.model.release_syntheses()
        out = {}
        r = len(acc["u"])
        for name, values in acc.items():
            stack = xp.stack(values)
            mean = stack.mean(axis=0, keepdims=True)
            var = np.asarray(to_numpy(((stack - mean) ** 2).sum(axis=0) / max(r - 1, 1)), dtype=np.float64)
            if area_weighted:
                if var.ndim == 3:
                    mean_square = float(np.mean([grid.global_mean(level) for level in var]))
                else:
                    mean_square = float(grid.global_mean(var))
            else:
                mean_square = float(np.mean(var))
            out[name] = float(math.sqrt(max(0.0, mean_square)))
            del stack
        return out

    def resident_bytes(self) -> dict[str, object]:
        """Measured bytes of the member arrays: per member by namespace,
        the sum, and on a device backend the pool's live and held bytes
        beside them (the shared workspace is the difference)."""
        per_member = [state_bytes(m) for m in self.members]
        total = int(sum(b["total"] for b in per_member))
        row: dict[str, object] = {
            "members": self.size,
            "per_member": per_member[0] if per_member else {},
            "members_total": total,
            "members_total_gib": round(total / 2 ** 30, 3),
        }
        backend = self.transform.backend
        if getattr(backend, "name", "numpy") == "cupy":
            pool = backend.xp.get_default_memory_pool()
            row["pool_used_bytes"] = int(pool.used_bytes())
            row["pool_total_bytes"] = int(pool.total_bytes())
            row["shared_estimate_bytes"] = int(pool.used_bytes()) - total
            row["measures"] = (
                "member arrays' nbytes summed per namespace; pool_used_bytes is the "
                "CuPy default pool's live bytes at the call, and shared_estimate is "
                "that minus the members' arrays (tables, statics, physics workspace)"
            )
        else:
            row["measures"] = "member arrays' nbytes summed per namespace (host arrays)"
        return row


def ensemble_config(cfg, options: EnsembleOptions, *, dt_s: float | None = None, name: str | None = None):
    """The deterministic run's config re-cut at the ensemble truncation:
    the same physics, vertical layout, statics and analysis source, the
    truncation ``options.truncation`` (nlat and nlon derived), the time
    step ``dt_s`` or the largest multiple of the deterministic step at or
    below the deterministic step times the truncation ratio that divides
    the output interval.  The door builds the ensemble model and its cold
    state from this with ``runner.build_model_and_cold_state``."""
    det_t = int(cfg.truncation)
    ens_t = int(options.truncation)
    if ens_t > det_t:
        raise ValueError(
            f"the ensemble truncation T{ens_t} exceeds the deterministic T{det_t}; "
            "the dual-resolution design puts the ensemble at or below the deterministic member"
        )
    return recut_config(cfg, ens_t, dt_s=dt_s, name=name or f"{cfg.name}-ens-t{ens_t}")


def recut_config(cfg, truncation: int, *, dt_s: float | None = None, name: str | None = None):
    """``cfg`` re-cut at ``truncation`` in either direction (the OSSE's
    nature run and control sit ABOVE the config's truncation): nlat and
    nlon derived, the time step ``dt_s`` or the config's step scaled by
    the truncation ratio to the nearest value that divides the output
    interval (a coarser grid takes a longer step, a finer one a shorter;
    under a semi-Lagrangian integrator the step is kept, because the
    deformation bound it runs under does not scale with the grid),
    the native suite's ``dx_m`` scaled by the same ratio so the
    scale-aware closures see the grid they run on."""
    src_t = int(cfg.truncation)
    dst_t = int(truncation)
    if dt_s is None and str(cfg.integrator).lower() in SEMILAG_INTEGRATORS:
        # The semi-Lagrangian step is bounded by the flow deformation (the
        # Lipschitz number dt |grad v|, semilag.trajectory), not by the grid
        # spacing, so a coarser re-cut keeps the config's step: the T127
        # members of a T255 control at 300 s were re-cut to 600 s by the
        # Eulerian rule below and the third hour of the case's real cycle
        # refused at a Lipschitz number of 0.7689 against 0.75 (2026-09-07).
        dt_s = float(cfg.dt_s)
    if dt_s is None:
        ratio = (src_t + 1) / (dst_t + 1)
        if ratio >= 1.0:
            multiple = max(1, int(math.floor(ratio)))
            chosen = None
            while multiple >= 1:
                candidate = float(cfg.dt_s) * multiple
                if abs(cfg.output_interval_s / candidate - round(cfg.output_interval_s / candidate)) < 1.0e-9:
                    chosen = candidate
                    break
                multiple -= 1
            dt_s = chosen if chosen is not None else float(cfg.dt_s)
        else:
            divisor = max(1, int(math.ceil(1.0 / ratio)))
            dt_s = float(cfg.dt_s) / divisor
    dt_s = float(dt_s)
    for total, label in ((cfg.duration_s, "duration_s"), (cfg.output_interval_s, "output_interval_s")):
        if abs(total / dt_s - round(total / dt_s)) > 1.0e-9:
            raise ValueError(f"the re-cut dt {dt_s:g} s does not divide the config's {label} {total:g} s")
    native_options = dict(cfg.native_adapter_options or {})
    if "dx_m" in native_options and native_options["dx_m"] is not None:
        native_options["dx_m"] = float(native_options["dx_m"]) * (src_t + 1) / (dst_t + 1)
    return dataclasses.replace(
        cfg,
        name=name or f"{cfg.name}-t{dst_t}",
        truncation=dst_t, nlat=None, nlon=None, dt_s=dt_s,
        native_adapter_options=native_options,
    )


def truncate_spectral(coeff, truncation: int):
    """A spectral field ``(..., n, m)`` truncated to ``truncation``: the
    degrees and orders above it dropped, exactly (spectral downscaling of
    the deterministic member onto the ensemble)."""
    c = coeff
    n = int(truncation) + 1
    if c.shape[-1] < n:
        raise ValueError(
            f"cannot truncate a T{c.shape[-1] - 1} field to T{truncation}: "
            "that is an embedding, not a truncation"
        )
    return c[..., :n, :n].copy()


def embed_spectral(coeff, truncation: int, xp=np):
    """A spectral field ``(..., n, m)`` embedded in the triangle of a
    higher ``truncation``: its coefficients kept bit for bit, the degrees
    above it exactly zero (spectral upscaling of the ensemble-mean
    increment onto the deterministic member)."""
    c = coeff
    n = int(truncation) + 1
    if c.shape[-1] > n:
        raise ValueError(
            f"cannot embed a T{c.shape[-1] - 1} field in T{truncation}: "
            "that is a truncation, not an embedding"
        )
    out = xp.zeros((*c.shape[:-2], n, n), dtype=c.dtype)
    out[..., : c.shape[-2], : c.shape[-1]] = c
    return out


def write_manifest(path: Path, manifest: dict) -> None:
    Path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "ENSEMBLE_MANIFEST_NAME",
    "ENSEMBLE_SCHEMA",
    "MEMBER_PREFIX",
    "SHARED_SURFACE_BREAKAGE",
    "GlobalEnsemble",
    "MemberTargets",
    "StepTiming",
    "embed_spectral",
    "ensemble_config",
    "member_checkpoint_path",
    "recut_config",
    "state_bytes",
    "truncate_spectral",
]
