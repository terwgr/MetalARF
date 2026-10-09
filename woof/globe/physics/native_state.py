"""Persistent state construction for the built-in native physics suite."""
from __future__ import annotations

from types import SimpleNamespace

from ..constants import CONVECTIVE_RAIN_ACCUMULATOR
from ..spill import resident, spilled
from ..state import PhysicsState
from ..water import SOIL_LAYER_THICKNESS_M

#: The dynamics' own theta (K/s) and vapor (kg/kg/s) tendencies over the
#: last dynamics interval, per level and column: the advective forcing
#: lanes the cumulus schemes read (WRF RTHFTEN/RQVFTEN), and the theta and
#: vapor the previous call left behind with its time_s (metadata
#: CUMULUS_EXIT_TIME_KEY) that the next call at a later time_s measures
#: them from.  All four persistent so a restart resumes bit for bit
#: whatever the call cadence; the lanes zero until the first dynamics
#: interval has been measured; absent without a cumulus scheme.
CUMULUS_DYNAMICS_THETA_LANE = "cumulus_dyn_dtheta"
CUMULUS_DYNAMICS_QV_LANE = "cumulus_dyn_dqv"
CUMULUS_DYNAMICS_LANES = (CUMULUS_DYNAMICS_THETA_LANE, CUMULUS_DYNAMICS_QV_LANE)
CUMULUS_EXIT_THETA = "cumulus_exit_theta"
CUMULUS_EXIT_QV = "cumulus_exit_qv"
CUMULUS_EXIT_ARRAYS = (CUMULUS_EXIT_THETA, CUMULUS_EXIT_QV)
CUMULUS_EXIT_TIME_KEY = "cumulus_exit_time_s"


#: Persistent time integrals of the held radiation planes (J/m2; the
#: cover integral is in seconds), keyed by the held plane each one
#: integrates.  Names are the WRF accumulator names (ACSWDNB and
#: friends) spelled out.
RADIATION_ACCUMULATOR_SWDNB = "acc_sw_down_surface_j_m2"
RADIATION_ACCUMULATOR_SWUPB = "acc_sw_up_surface_j_m2"
RADIATION_ACCUMULATOR_LWDNB = "acc_lw_down_surface_j_m2"
RADIATION_ACCUMULATOR_LWUPB = "acc_lw_up_surface_j_m2"
RADIATION_ACCUMULATOR_SWUPT = "acc_sw_up_top_j_m2"
RADIATION_ACCUMULATOR_SWDNT = "acc_sw_down_top_j_m2"
RADIATION_ACCUMULATOR_LWUPT = "acc_lw_up_top_j_m2"
RADIATION_ACCUMULATOR_CLOUD_COVER = "acc_total_cloud_cover_s"
RADIATION_ACCUMULATORS = {
    RADIATION_ACCUMULATOR_SWDNB: "swdown",
    RADIATION_ACCUMULATOR_SWUPB: "swupb",
    RADIATION_ACCUMULATOR_LWDNB: "glw",
    RADIATION_ACCUMULATOR_LWUPB: "lwupb",
    RADIATION_ACCUMULATOR_SWUPT: "swupt",
    RADIATION_ACCUMULATOR_SWDNT: "swdnt",
    RADIATION_ACCUMULATOR_LWUPT: "olr",
    RADIATION_ACCUMULATOR_CLOUD_COVER: "cldfra_total",
}


class PersistentNativeState:
    def __init__(self, batch, options):
        self.batch = batch
        self.xp = batch.xp
        self.options = options
        self.state = batch.physics_state.copy()
        self.state.validate()
        self.metadata = dict(self.state.metadata)
        self.arrays = self.state.arrays
        self._initialize()
        # Every array the initialiser did not name reaches the card here.
        # A namespace entry a scheme writes but the initialiser does not
        # seed (an adapter's own diagnostic plane, a restart's extra
        # field) would otherwise stay a host handle and the kernel handed
        # it would refuse -- loudly, but at the kernel rather than here.
        for name, value in list(self.arrays.items()):
            if spilled(value):
                self.arrays[name] = resident(self.xp, value)

    def _array(self, name, shape, fill=0.0, *, dtype=None):
        dtype = self.xp.float32 if dtype is None else dtype
        value = self.arrays.get(name)
        if spilled(value):
            # The pinned host tier holds this one.  Staging it here is
            # the SAME copy ``_array`` makes below out of the batch's
            # namespace, so the run's device namespace is one copy either
            # way and the tier keeps the original off the card.
            value = resident(self.xp, value)
        if value is None:
            if hasattr(fill, "shape"):
                if tuple(fill.shape) != tuple(shape):
                    raise ValueError(
                        f"persistent native field {name} seed shape "
                        f"{tuple(fill.shape)} != {tuple(shape)}"
                    )
                value = self.xp.ascontiguousarray(
                    self.xp.asarray(fill, dtype=dtype)
                ).copy()
            else:
                value = self.xp.full(shape, fill, dtype=dtype)
            self.arrays[name] = value
        else:
            value = self.xp.ascontiguousarray(self.xp.asarray(value, dtype=dtype))
            if tuple(value.shape) != tuple(shape):
                raise ValueError(
                    f"persistent native field {name} shape {value.shape} != {shape}"
                )
            # ``self.state`` is already this object's own deep copy of the
            # batch namespace (the constructor), so the array is private
            # here: a second .copy() per field doubled every namespace
            # (1.5 GiB at T533 with 40 levels) for the life of the call.
            self.arrays[name] = value
        return self.arrays[name]

    def _surface_seed(self, name, shape):
        value = getattr(self.batch.surface, name, None)
        if value is None or tuple(getattr(value, "shape", ())) != tuple(shape):
            return None
        return value

    def _initialize(self):
        nz, ny, nx = self.batch.shape
        s = (ny, nx)
        v = (nz, ny, nx)
        soil = (len(SOIL_LAYER_THICKNESS_M), ny, nx)
        # Held radiation and surface carriers.
        for name, fill in (
            ("rad_rthratenlw", 0.0), ("rad_rthratensw", 0.0),
        ):
            self._array(name, v, fill)
        if self.options.cumulus_enabled:
            for name in (*CUMULUS_DYNAMICS_LANES, *CUMULUS_EXIT_ARRAYS):
                self._array(name, v, 0.0)
            self.metadata.setdefault(CUMULUS_EXIT_TIME_KEY, None)
        for name, fill in (
            ("swdown", 0.0), ("glw", 300.0), ("olr", 0.0),
            ("gsw", 0.0), ("coszen", 0.0),
            # The rest of the radiation budget's carriers, held between
            # radiation buckets exactly like the four above: upward and
            # downward shortwave at the top of the radiation column,
            # upward longwave at the surface, and the column cloud cover
            # the scheme's overlap implies (native_runtime
            # _radiation_step).
            ("swupt", 0.0), ("swdnt", 0.0), ("lwupb", 0.0),
            ("cldfra_total", 0.0),
            # Time integrals of the held planes, J/m2 (cover: seconds of
            # cover), advanced by flux * dt on EVERY physics call, so the
            # difference of two checkpoints divided by the accumulated
            # seconds is the exact interval-mean flux the model applied,
            # not a sample of the held plane at checkpoint time.  The
            # radiation scorecard reads them; the accumulated seconds
            # live in the metadata (radiation_accumulated_s).
            (RADIATION_ACCUMULATOR_SWDNB, 0.0),
            (RADIATION_ACCUMULATOR_SWUPB, 0.0),
            (RADIATION_ACCUMULATOR_LWDNB, 0.0),
            (RADIATION_ACCUMULATOR_LWUPB, 0.0),
            (RADIATION_ACCUMULATOR_SWUPT, 0.0),
            (RADIATION_ACCUMULATOR_SWDNT, 0.0),
            (RADIATION_ACCUMULATOR_LWUPT, 0.0),
            (RADIATION_ACCUMULATOR_CLOUD_COVER, 0.0),
            ("znt", 0.05), ("ust", 0.1), ("mol", 0.0),
            ("hfx", 0.0), ("qfx", 0.0), ("qsfc", 0.0),
            ("zol", 0.0), ("pblh", self.options.initial_pbl_height_m),
            # The partial pack's two surface-layer tiles (native_runtime
            # _surface_layer_step): the open water of the leads keeps its
            # own inout state across calls (Charnock roughness from the
            # water start, friction velocity, Monin-Obukhov scale, its
            # saturation humidity and fluxes, its 2 m exchange
            # coefficients), and the ice tile's fluxes are kept beside the
            # composite the atmosphere receives, for the frozen column.
            # Zero everywhere on a planet without a partial pack.
            ("lead_znt", 1.0e-4), ("lead_ust", 0.1), ("lead_mol", 0.0),
            ("lead_zol", 0.0), ("lead_qsfc", 0.0), ("lead_hfx", 0.0),
            ("lead_qfx", 0.0), ("lead_chs2", 0.0), ("lead_cqs2", 0.0),
            ("ice_hfx", 0.0), ("ice_qfx", 0.0),
            ("wspd", 0.1), ("br", 0.0), ("fm", 1.0), ("fh", 1.0),
            ("u10", 0.0), ("v10", 0.0),
            ("chs", 0.0), ("chs2", 0.0), ("cqs2", 0.0), ("qgh", 0.0),
            ("rainnc", 0.0), ("rainncv", 0.0),
            ("snownc", 0.0), ("snowncv", 0.0),
            ("graupelnc", 0.0), ("graupelncv", 0.0), ("sr", 0.0),
            # Precipitation buckets: filled every microphysics call, emptied
            # only by the land step that consumes them.  rainncv and friends
            # are per-call outputs and cannot carry precipitation across the
            # physics calls between two due land steps.
            ("land_rainbl", 0.0), ("land_snowbl", 0.0),
            ("land_graupelbl", 0.0),
            # Accumulated convective precipitation (WRF RAINC, kg/m2): the
            # cumulus step adds each call's RAINCV and the render tape's
            # RAINC reads it (wrfout_export).  Zero forever with
            # cumulus="none".
            (CONVECTIVE_RAIN_ACCUMULATOR, 0.0),
            ("effc", 2.5), ("effr", 10.0),
            ("effi", 5.0), ("effs", 10.0),
        ):
            self._array(name, v if name.startswith("eff") else s, fill)
        # Screen-level diagnostics (sfclay t2/th2/q2): seeded from the
        # lowest full level so a state exported before the first surface-
        # layer call still carries a finite value; the first call overwrites
        # them and stamps the source in the metadata.
        for name, source in (("t2", "temperature"), ("th2", "theta"), ("q2", "qv")):
            self._array(name, s, self.batch.arrays[source][0])
        # Noah four-layer state and complete driver field namespace.  The soil
        # columns are seeded from the exchange's own surface state: Noah's
        # first due call overwrites surface.soil_water_fraction with its copy,
        # so any disagreement between the two initializations is a step-one
        # jump of (seed - surface) * 2.0 m * 1000 kg/m3 * land_fraction that
        # the suite's water closure must drain from the surface reservoir.
        # The option constants remain the fallback for a surface with no soil.
        soil_water = self._surface_seed("soil_water_fraction", soil)
        soil_temperature = self._surface_seed("soil_temperature_k", soil)
        if soil_water is None:
            soil_water = self.options.initial_soil_water_fraction
        if soil_temperature is None:
            soil_temperature = self.options.initial_soil_temperature_k
        self._array("noah_smois", soil, soil_water)
        self._array("noah_sh2o", soil, soil_water)
        self._array("noah_tslb", soil, soil_temperature)
        self._array("noah_smcrel", soil, soil_water)
        for name, fill in (
            ("noah_canwat", 0.0), ("noah_snow", 0.0),
            ("noah_snowc", 0.0), ("noah_snowh", 0.0),
            ("noah_lh", 0.0), ("noah_grdflx", 0.0),
            # The land surface's own hfx/qfx/qsfc from its last integration:
            # on calls where Noah is not due, land columns hand YSU these
            # instead of sfclay's bulk fluxes (native_runtime
            # _hold_land_fluxes; audit 2026-09-01 NB-2).  Checkpointed with
            # the namespace so a restart's first non-due call sees them.
            ("noah_hfx", 0.0), ("noah_qfx", 0.0), ("noah_qsfc", 0.0),
            ("noah_sfcrunoff", 0.0), ("noah_udrunoff", 0.0),
            # Cumulative booked water exits: the land step adds each priced
            # runoff-store increment here at the moment it debits the
            # reservoir for it.  Counted by both water ledgers as an exit
            # term (water.py water_outflow_column), checkpointed with the
            # rest of the namespace so restart continuation stays bit-exact.
            ("water_outflow_kg_m2", 0.0),
        ):
            self._array(name, s, fill)
        self.metadata.setdefault("last_radiation_bucket", -1)
        self.metadata.setdefault("radiation_calls", 0)
        self.metadata.setdefault("radiation_accumulated_s", 0.0)
        # Where the held surface upward longwave came from: the scheme's
        # own solver plane, or the emissivity formula the runtime applies
        # when a scheme publishes none (never a silent substitution).
        self.metadata.setdefault("lwupb_source", None)
        # Particle-size bounding of the radiation coupling
        # (woof.globe.core.rrtmgp.SizeBounding): the last call's record and
        # the running sum over calls.
        self.metadata.setdefault("radiation_size_bounding_last", None)
        self.metadata.setdefault("radiation_size_bounding_sum", None)
        self.metadata.setdefault("last_land_bucket", -1)
        self.metadata.setdefault("last_land_time_s", None)
        self.metadata.setdefault("microphysics_updates", 0)
        self.metadata.setdefault("cumulus_updates", 0)
        self.metadata.setdefault("native_calls", 0)

    def export(self) -> PhysicsState:
        """Hand the namespace over as the result's physics state.

        The arrays are handed over, not copied: this object is discarded
        by the suite right after the export and the next call copies the
        namespace afresh before writing (``_array``), so a copy here was
        one whole namespace (1.5 GiB at T533 with 40 levels) live twice at
        the end of every physics call.  The dict is new, so a later
        ``_array`` on this object cannot reach the exported state.
        """
        return PhysicsState(
            arrays=dict(self.arrays),
            metadata=dict(self.metadata),
        )

    def fake_radiation_state(self):
        batch = self.batch
        # The model-top pressure the radiation's above-model column is
        # built on.  The exchange reads it ONCE over the whole grid
        # (dynamics.apply_physics: the float32 mean of the top half-level
        # plane, the same expression as the fallback below over the whole
        # grid) so that a run banded eight ways and a run banded once hand
        # the radiation the same float; a band's own mean of a plane of
        # equal values is not the globe's mean of it once the sum exceeds
        # 2^24 of the top pressure's binade.  The fallback is the batch's
        # own mean, for a hand-built whole-grid exchange.
        if batch.model_top_pa is not None:
            p_top = float(batch.model_top_pa)
        else:
            p_top = float(
                (batch.xp.asnumpy(batch.arrays["p_half"][-1])
                 if hasattr(batch.xp, "asnumpy")
                 else batch.arrays["p_half"][-1]).mean()
            )
        return SimpleNamespace(
            p_top=p_top,
            elapsed_seconds=float(batch.time_s),
            qc=batch.arrays["qc"], qr=batch.arrays["qr"],
            qi=batch.arrays["qi"], qs=batch.arrays["qs"],
            nc=batch.arrays["nc"], nr=batch.arrays["nr"],
            ni=batch.arrays["ni"], ns=batch.arrays["ns"],
            effc=self.arrays["effc"], effr=self.arrays["effr"],
            effi=self.arrays["effi"], effs=self.arrays["effs"],
            physics=SimpleNamespace(
                microphysics_updates=int(self.metadata["microphysics_updates"])
            ),
        )


__all__ = ["PersistentNativeState", "RADIATION_ACCUMULATORS"]
