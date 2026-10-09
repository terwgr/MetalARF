"""Transactional built-in adapter for existing Arwen CUDA column physics."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ..constants import (
    DRY_AIR_CP,
    GRAVITY_M_S2,
    NUMBER_MOMENTS,
    STEFAN_BOLTZMANN,
    WATER_SPECIES,
)
from ..state import PhysicsState
from ..water import SOIL_LAYER_THICKNESS_M
from .banding import (
    COLUMN_MEAN, COUNT, MAX, SAME, SKIP, finish_alone, merge_band_metadata,
    merge_band_scalars, to_host,
)
from .exchange import PhysicsExchange, PhysicsResult
from .native_batch import NativeColumnBatch
from .native_options import NativePhysicsOptions
from .native_runtime import NativePhysicsRuntime
from .native_state import PersistentNativeState
from ..profile import profiler_of


# WRF's physics order: radiation, surface layer, land surface, PBL, cumulus,
# microphysics.  Grell-Freitas is the sixth native component (default on);
# cumulus="ntiedtke" puts New Tiedtke in the same slot, cumulus="own" puts
# arwen-massflux-v1 there, and cumulus="none" drops the slot from the
# order a suite reports.
NATIVE_COMPONENT_ORDER = ("rrtmgp", "sfclay", "noah", "ysu", "gf", "morrison")
CUMULUS_SLOT = NATIVE_COMPONENT_ORDER.index("gf")


def native_component_order(options: NativePhysicsOptions) -> tuple[str, ...]:
    """The components this options set actually runs, in call order."""
    component = options.cumulus_component
    order = list(NATIVE_COMPONENT_ORDER)
    if component is None:
        del order[CUMULUS_SLOT]
    else:
        order[CUMULUS_SLOT] = component
    return tuple(order)


class ArwenCudaColumnSuite:
    """Copy-run-validate-return adapter; caller state is never mutated in place."""

    # In-situ component capture slot, forwarded to the lazily built runtime.
    observer = None
    # Step profiler slot (woof.globe.profile), forwarded the same
    # way; None is the no-op.
    profiler = None

    #: How each of a call's scalar readings is formed from the bands'
    #: (physics.banding).  A reading not named here is the same on every
    #: band and is refused by name when it is not.  The grid means a
    #: scheme reports of its own call (arwen-massflux-v1's branch
    #: fractions, mass flux, rain and CAPE) are merged as a column-count
    #: weighted mean of the bands' means, which is not the whole grid's
    #: mean bit for bit: they are the readings of a selectable scheme,
    #: they enter no checkpoint, and the merge is stated here.
    DIAGNOSTIC_MERGE = {
        "frozen_surface_columns": COUNT,
        "lake_surface_columns": COUNT,
        "maximum_native_water_residual_kg_m2": MAX,
        "native_water_residual_exceeds_tolerance": MAX,
        "maximum_native_energy_change_j_m2": MAX,
        "native_energy_change_exceeds_advisory": MAX,
        "maximum_local_water_repair_kg_m2": MAX,
    }
    #: Namespace metadata keys that are NOT the same on every band: the
    #: three call counters a band without the columns they count does not
    #: bump (a band with no frozen column, no partial pack, no lake), and the
    #: radiation size-bounding record, which ``finish`` assembles from the
    #: bands' counts and the whole-grid path-sum planes.
    METADATA_MERGE = {
        "frozen_surface_calls": MAX,
        "lead_tile_calls": MAX,
        "lake_surface_calls": MAX,
        "radiation_size_bounding_last": SKIP,
        "radiation_size_bounding_sum": SKIP,
    }
    #: The record's six fractions, each ``part / whole`` of two of the
    #: ten path sums (woof.globe.core.rrtmgp SIZE_BOUNDING_SUM_NAMES), in the
    #: one-call form's expressions.
    _SIZE_BOUNDING_FRACTIONS = {
        "liquid_above_path_fraction": ("liquid_carried", "total_liquid"),
        "ice_above_path_fraction": ("ice_carried", "total_ice"),
        "liquid_above_radiative_fraction": (
            "liquid_carried_radiative", "liquid_radiative"),
        "ice_above_radiative_fraction": ("ice_carried_radiative", "ice_radiative"),
        "liquid_sentinel_radiative_fraction": (
            "liquid_sentinel_radiative", "liquid_radiative"),
        "ice_sentinel_radiative_fraction": (
            "ice_sentinel_radiative", "ice_radiative"),
    }
    _SIZE_BOUNDING_PLANE = "radiation_size_bounding__"
    _FLOOR_PLANE = "stratospheric_floor_heating_j_m2"

    def __init__(
        self,
        options: dict[str, object] | NativePhysicsOptions,
        *,
        array_module=None,
        modules: dict[str, object] | None = None,
    ):
        self.options = (
            options
            if isinstance(options, NativePhysicsOptions)
            else NativePhysicsOptions.from_mapping(dict(options))
        )
        self._xp = array_module
        self._modules = modules
        self._runtime = None

    @property
    def identity(self) -> dict[str, object]:
        return {
            "mode": "arwen-native",
            "adapter": "arwen-cuda-column-suite-v1",
            "admission_status": "device-pending",
            "component_order": list(native_component_order(self.options)),
            "options": self.options.identity,
        }

    def _array_module(self):
        if self._xp is not None:
            return self._xp
        from woof.local_gpu import NO_LOCAL_GPU_ENV, no_local_gpu

        if no_local_gpu():
            raise RuntimeError(
                f"arwen-cuda-column-suite-v1 refused: {NO_LOCAL_GPU_ENV} is "
                "set, so this process may not open the local CUDA device; run "
                "the native suite on a node where the variable is unset."
            )
        try:
            import cupy as cp
        except Exception as exc:
            raise RuntimeError(
                "arwen-cuda-column-suite-v1 requires a working CuPy CUDA runtime"
            ) from exc
        # A CuPy import alone is not device evidence, but this early operation
        # provides a useful immediate refusal on a mismatched/no-device wheel.
        try:
            cp.cuda.runtime.getDevice()
        except Exception as exc:
            raise RuntimeError("CuPy imported but no usable CUDA device is available") from exc
        self._xp = cp
        return cp

    @staticmethod
    def _host(xp, value):
        return np.asarray(xp.asnumpy(value) if hasattr(xp, "asnumpy") else value)

    def _native_cfg(self, dt_s: float):
        return SimpleNamespace(
            mp_physics=10,
            bl_pbl_physics=1,
            icloud_bl=0,
            wrf_rrtmg_compatibility=self.options.wrf_rrtmg_compatibility,
            dt=float(dt_s),
            clock_dt=float(dt_s),
            radt=self.options.radiation_interval_s / 60.0,
            radt_minutes=self.options.radiation_interval_s / 60.0,
            ysu_topdown_pblmix=self.options.ysu_topdown_pblmix,
            ysu_free_atmosphere_mixing_length=(
                self.options.ysu_free_atmosphere_mixing_length),
            # The cumulus schemes read these exactly as the regional
            # RunConfig carries them (woof.globe.core.gf: cfg.dx, cfg.ishallow,
            # cfg.clos_choice; woof.globe.core.ntiedtke: cfg.dx and
            # cfg.ntiedtke_tiedtke_closure; both the model-clock dt above).
            # The scalar dx is the options' grid spacing (Grell-Freitas
            # reads it; New Tiedtke is handed the per-column Gaussian
            # spacing by the runtime instead); clos_choice 0 is the
            # 16-member ensemble, the only oracle-covered closure.
            cu_physics={"gf": 3, "ntiedtke": 16}.get(self.options.cumulus_scheme, 0),
            dx=float(self.options.dx_m),
            dy=float(self.options.dx_m),
            ishallow=int(self.options.gf_ishallow),
            clos_choice=0,
            ntiedtke_tiedtke_closure=bool(self.options.ntiedtke_tiedtke_closure),
        )

    def _water_column(self, batch, persistent: PersistentNativeState | None = None):
        xp = batch.xp
        # The batch holds kernel-side dry mixing ratios; the ledger prices
        # the atmosphere in the model's specific-humidity metric so before
        # and after compare like with like AND like what the dycore's own
        # water ledger sees on the returned fields (native_batch docstring;
        # audit 2026-09-01 NB-3).
        atmosphere = batch.atmospheric_water_kg_m2()
        total = atmosphere + batch.surface.water_kg_m2
        total += (
            batch.surface.soil_water_fraction
            * xp.asarray(SOIL_LAYER_THICKNESS_M, dtype=xp.float32)[:, None, None]
            * xp.float32(1000.0)
            * batch.surface.land_fraction[None]
        ).sum(axis=0)
        arrays = (
            batch.physics_state.arrays
            if persistent is None else persistent.arrays
        )
        for name in ("noah_canwat", "noah_snow"):
            if name in arrays:
                total += xp.maximum(arrays[name], 0.0)
        # This ledger is the CONSERVATION total, held water plus booked
        # exits, so the closure equation carries the outflow term: dW_atm +
        # dW_reservoir + dW_soil + dW_canopy + dW_snow + dW_outflow = 0 per
        # call.  Noah's runoff accumulators themselves are counted at zero:
        # SSTEP clips saturated soil into udrunoff and infiltration excess
        # lands in sfcrunoff (noah.cu:548-564, 1518-1519), nothing in the
        # column ever reads them back, and the land step books each priced
        # increment (max(store, 0) * land_fraction, the v3 weights) to the
        # cumulative outflow account at the moment it debits the reservoir
        # for the exit.  Counting the stores as held water -- the v3 ledger,
        # installed when leaving them out entirely priced the first land
        # call on saturated ice-sheet columns as a 1072.0001220703125 kg/m2
        # repair against the 5e-4 gate (2026-08-31) -- kept every call
        # closed but sequestered a monotone-growing store inside the pinned
        # conservation total, squeezing the global atmosphere+reservoir mean
        # through the fixer target by every kg the accumulators grew.
        # water.py total_water_column counts the same account; the two
        # ledgers must stay at matching terms or the global water fixer
        # sees whatever this closure stops seeing.
        if "water_outflow_kg_m2" in arrays:
            total += arrays["water_outflow_kg_m2"]
        return total

    def _energy_column(self, batch):
        xp = batch.xp
        atmosphere = (
            DRY_AIR_CP * batch.arrays["temperature"]
            * batch.arrays["dp"] / GRAVITY_M_S2
        ).sum(axis=0)
        surface = (
            batch.surface.heat_capacity_j_m2_k
            * batch.surface.temperature_k
        )
        return atmosphere + surface

    def _radiation_diagnostics(self, surface, arrays, metadata):
        """Grid-mean radiation readings under the reference suite's names.

        Diagnostics only; nothing here feeds back into the physics.  The
        harness radiation instrument reads ``mean_outgoing_longwave_w_m2``
        from the result contract, so the names match the reference suite's
        exactly.  ``olr`` is the held RRTMGP top-of-atmosphere upward
        longwave plane, persisted between radiation buckets, so the reading
        is defined on every call, not only on calls where radiation was
        due.  Net surface radiation mirrors the reference formula: absorbed
        shortwave (``gsw`` is sw_down - sw_up at the surface) plus absorbed
        downward longwave minus the surface's own emission.

        Read over the WHOLE surface and namespace, after the bands of a
        call have all been written into them (:meth:`finish`): every
        reading is a flat mean over the globe's plane, which is the same
        plane whatever the band count.
        """

        def plane(value):
            return np.asarray(to_host(value), dtype=np.float64)

        olr = plane(arrays["olr"])
        gsw = plane(arrays["gsw"])
        glw = plane(arrays["glw"])
        emissivity = plane(surface.emissivity)
        skin = plane(surface.temperature_k)
        net_surface = (
            gsw + emissivity * glw
            - emissivity * STEFAN_BOLTZMANN * skin**4
        )
        readings = {
            "mean_outgoing_longwave_w_m2": float(np.mean(olr)),
            "mean_net_surface_radiation_w_m2": float(np.mean(net_surface)),
            # The held budget carriers and the coupling's size bounding
            # (native_state, native_runtime _radiation_step): grid means
            # of the persisted planes, the call count and the last
            # radiation call's bounding record flattened under its field
            # names, so the run's diagnostics say how often the cloud
            # optics ran outside its tables.
            "mean_toa_upward_shortwave_w_m2": float(
                np.mean(plane(arrays["swupt"]))),
            "mean_toa_downward_shortwave_w_m2": float(
                np.mean(plane(arrays["swdnt"]))),
            "mean_surface_upward_longwave_w_m2": float(
                np.mean(plane(arrays["lwupb"]))),
            "mean_total_cloud_cover": float(
                np.mean(plane(arrays["cldfra_total"]))),
            "radiation_calls": float(metadata.get("radiation_calls", 0)),
        }
        bounding = metadata.get("radiation_size_bounding_last")
        if isinstance(bounding, dict):
            readings.update({
                f"radiation_size_bounding_{key}": float(value)
                for key, value in bounding.items()
            })
        return readings

    def _size_bounding_record(self, band_metadata, planes, metadata_in):
        """The radiation size-bounding record of a call whose radiation
        ran, assembled over the globe.

        The counts are exact integers and add across the bands as they
        are.  The six fractions are ratios of whole-grid path sums, so
        they are formed once here from the per-column sums the radiation
        handed back (rrtmgp.size_bounding_column_sums), assembled into the
        globe's planes by the caller: the plane's flat sum is the same
        number whatever the band count.  This two-stage sum (layers per
        column, then the plane) is not the one-call flat sum over every
        cell bit for bit, which is the one seam this form has against the
        record the pre-banded tree wrote; the record enters the
        checkpoint's metadata, not its arrays.
        """
        records = [m.get("radiation_size_bounding_last") for m in band_metadata]
        if not all(isinstance(r, dict) for r in records):
            return None, metadata_in.get("radiation_size_bounding_sum")
        last: dict[str, object] = {}
        for name in records[0]:
            if name.endswith("fraction"):
                continue
            last[name] = int(sum(int(r[name]) for r in records))
        sums = {}
        for part, whole in self._SIZE_BOUNDING_FRACTIONS.values():
            for name in (part, whole):
                if name in sums:
                    continue
                plane = planes.get(self._SIZE_BOUNDING_PLANE + name)
                if plane is None:
                    raise ValueError(
                        f"the radiation ran and no {name!r} path-sum plane "
                        "came back with the bands; the size-bounding record "
                        "cannot be formed over the globe without it"
                    )
                sums[name] = np.float32(np.sum(
                    np.asarray(to_host(plane), dtype=np.float32),
                    dtype=np.float32))
        for name, (part, whole) in self._SIZE_BOUNDING_FRACTIONS.items():
            if float(sums[whole]) > 0.0:
                value = sums[part] / np.maximum(sums[whole], np.float32(1.0e-30))
            else:
                value = np.float32(0.0)
            last[name] = float(value)
        running = metadata_in.get("radiation_size_bounding_sum")
        total = {
            key: (0 if running is None else running.get(key, 0)) + value
            for key, value in last.items()
        }
        return last, total

    def _stratospheric_floor_pull(self, batch, dt_s: float) -> float:
        """One-sided pull toward the stratospheric floor, in theta.

        Same scaffold as reference.py _stratospheric_floor_pull, standing in
        for the residual-circulation warming a 20-level hydrostatic top
        cannot supply (defect citation on the options: the 384 h native
        baseline arm's hour-83.4 polar-sag death through the 140 K research
        bound).  Points above stratospheric_floor_pa colder than the floor
        gain max(floor - T, 0) * dt / tau per call; nothing is ever cooled,
        nothing at or above the floor is touched.  The added enthalpy is
        measured and returned as the column plane (J/m2 this call, one
        value per column) so every run carries its own bill in the
        receipts: the grid mean is taken once over the whole plane in
        :meth:`finish`.  None with the floor off.
        """
        floor_k = float(self.options.stratospheric_floor_k)
        if floor_k <= 0.0:
            return None
        xp = batch.xp
        theta = batch.arrays["theta"]
        exner = batch.arrays["exner"]
        cold = xp.maximum(floor_k - theta * exner, 0.0)
        pull = xp.where(
            batch.arrays["p_full"] < float(self.options.stratospheric_floor_pa),
            cold * (float(dt_s) / float(self.options.stratospheric_relaxation_time_s)),
            xp.float32(0.0),
        )
        batch.arrays["theta"] = xp.ascontiguousarray(theta + pull / exner)
        # Keep the runtime's temperature = theta * exner invariant, so the
        # energy advisory measures the floor's enthalpy like any other term.
        batch.arrays["temperature"] = xp.ascontiguousarray(
            batch.arrays["theta"] * exner
        )
        heating = (
            pull * batch.arrays["dp"] * (DRY_AIR_CP / GRAVITY_M_S2)
        ).sum(axis=0)
        return heating

    def _validate_result(self, batch):
        xp = batch.xp
        prognostics = ("u", "v", "theta", *WATER_SPECIES, *NUMBER_MOMENTS)
        bounded = (*WATER_SPECIES, *NUMBER_MOMENTS)
        surface = batch.surface.arrays()
        # Every reduction runs where the arrays live and the scalars cross
        # to the host in ONE read: the finiteness flags of the thirteen
        # prognostic volumes and the surface fields, and the minima of
        # the eleven bounded volumes.  Reducing the minima on the host
        # copied every volume back first (eleven volumes, 47 MB each at
        # T255, 611 MB per physics call), measured as the largest
        # device-to-host traffic of the step, and the per-array reads
        # were twenty-four synchronisations per call (profile
        # 2026-09-04).  Same values (a minimum and an all-finite flag are
        # exact in either place), same checks in the same order.
        finite = [xp.all(xp.isfinite(batch.arrays[name])) for name in prognostics]
        finite += [xp.all(xp.isfinite(value)) for value in surface.values()]
        minima = [xp.min(batch.arrays[name]) for name in bounded]
        readings = self._host(xp, xp.concatenate([
            xp.stack(finite).astype(xp.float64),
            xp.stack(minima).astype(xp.float64),
        ]))
        finite_host = readings[: len(finite)] != 0.0
        minima_host = dict(zip(bounded, readings[len(finite):]))
        for index, name in enumerate(prognostics):
            if not bool(finite_host[index]):
                raise FloatingPointError(
                    f"native physics produced non-finite {name}: {self._non_finite_column(batch, name)}"
                )
            if name in bounded:
                minimum = float(minima_host[name])
                tolerance = -1.0e-7 if name in WATER_SPECIES else -1.0e-2
                if minimum < tolerance:
                    raise FloatingPointError(
                        f"native physics produced negative {name}: {minimum:.9g}"
                    )
                batch.arrays[name] = xp.maximum(batch.arrays[name], 0.0)
        for index, name in enumerate(surface):
            if not bool(finite_host[len(prognostics) + index]):
                raise FloatingPointError(
                    f"native physics produced non-finite surface field {name}"
                )

    def _non_finite_column(self, batch, name: str) -> str:
        """Where a non-finite prognostic sits, for the refusal's message (the
        failure path only: a state that dies in the physics of one member of
        a cycle left nothing on disk that said which column, so the death was
        not reproducible).  The count of non-finite values and columns, the
        first such column (index, latitude, longitude, the levels affected)
        and the finite range of the same volume in that column."""
        try:
            xp = batch.xp
            arr = batch.arrays[name]
            bad = ~xp.isfinite(arr)
            total = int(self._host(xp, bad.sum()))
            if arr.ndim < 2:
                return f"{total} non-finite values"
            # (nlev, ...) whatever the trailing layout: one column per trailing index.
            arr = arr.reshape(arr.shape[0], -1)
            bad = bad.reshape(bad.shape[0], -1)
            bad_cols = bad.any(axis=0)
            ncols = int(self._host(xp, bad_cols.sum()))
            col = int(self._host(xp, xp.argmax(bad_cols)))
            levels = np.nonzero(np.asarray(self._host(xp, bad[:, col])))[0]
            where = ""
            lat = batch.arrays.get("latitude_deg")
            lon = batch.arrays.get("longitude_deg")
            if lat is not None and lon is not None:
                where = (f", {float(self._host(xp, lat.reshape(-1)[col])):.2f} deg latitude "
                         f"{float(self._host(xp, lon.reshape(-1)[col])):.2f} deg longitude")
            column = np.asarray(self._host(xp, arr[:, col]), dtype=np.float64)
            finite = column[np.isfinite(column)]
            span = (f"; the finite levels of that column read {finite.min():.6g} to {finite.max():.6g}"
                    if finite.size else "; no level of that column is finite")
            ps = ""
            p_half = batch.arrays.get("p_half")
            if p_half is not None:
                ps = f"; surface pressure there {float(self._host(xp, p_half.reshape(p_half.shape[0], -1)[-1, col])):.1f} Pa"
            return (f"{total} values in {ncols} of {int(arr.shape[1])} columns, the first at column {col}{where}, "
                    f"levels {levels.min()} to {levels.max()} of {int(arr.shape[0])}{span}{ps}")
        except Exception as exc:  # noqa: BLE001 - the diagnosis must never hide the refusal
            return f"(the column could not be named: {type(exc).__name__}: {exc})"

    def step(self, exchange: PhysicsExchange) -> PhysicsResult:
        prof = profiler_of(self)
        exchange.validate()
        xp = self._array_module()
        with prof.section("batch_in"):
            batch = NativeColumnBatch.from_exchange(exchange, xp)
        # All subsequent writes are to this copy and to a copied PhysicsState.
        # ONE persistent copy per call: the runtime works on the same
        # object the before-water ledger read, which is the untouched
        # initialisation the runtime would have rebuilt from the same
        # batch (same seeds, same arrays), so the reading is the same and
        # a second whole namespace (1.5 GiB at T533 with 40 levels) no
        # longer sits beside the working one through the call.
        with prof.section("ledger_before"):
            persistent = PersistentNativeState(batch, self.options)
            before_water = self._water_column(batch, persistent)
            before_energy = self._energy_column(batch)
        if self._runtime is None:
            self._runtime = NativePhysicsRuntime(
                self.options, modules=self._modules
            )
        self._runtime.observer = self.observer
        self._runtime.profiler = self.profiler
        persistent, schedule = self._runtime.run(
            batch, self._native_cfg(exchange.dt_s), persistent=persistent
        )
        with prof.section("validate"):
            self._validate_result(batch)
        closure = prof.section("ledger_after")
        closure.__enter__()
        floor_heating = self._stratospheric_floor_pull(batch, exchange.dt_s)
        after_water = self._water_column(batch, persistent)
        water_residual = after_water - before_water
        maximum_water_residual = float(
            np.max(np.abs(self._host(xp, water_residual)))
        )
        # Close only the measured local accounting residual.  This is explicit
        # and finite; it is not a silent atmospheric clip or an infinite store.
        batch.surface.water_kg_m2 -= xp.asarray(
            water_residual, dtype=batch.surface.water_kg_m2.dtype
        )
        if bool(xp.any(batch.surface.water_kg_m2 < -1.0e-7)):
            raise FloatingPointError(
                "native physics water closure exceeds the explicit surface reservoir"
            )
        batch.surface.water_kg_m2 = xp.maximum(
            batch.surface.water_kg_m2, 0.0
        )
        after_energy = self._energy_column(batch)
        energy_change = after_energy - before_energy
        maximum_energy_change = float(
            np.max(np.abs(self._host(xp, energy_change)))
        )
        del before_water, after_water, water_residual, before_energy
        del after_energy, energy_change
        closure.__exit__(None, None, None)
        out = prof.section("batch_out")
        out.__enter__()
        # Nothing below reads the batch's diagnostic volumes (the radiation
        # readings come from the persistent planes and the surface), so
        # they are released before the prognostics are copied back out and
        # the prognostics themselves are released one by one as they go:
        # memory only, the end of a physics call otherwise held the whole
        # batch beside its whole return (4.2 + 2.7 GiB at T533 float32).
        for name in (
            "p_full", "dp", "exner", "temperature", "virtual_temperature",
            "p_half", "omega_half", "terrain_height_m", "latitude_deg",
            "longitude_deg",
        ):
            batch.arrays.pop(name, None)
        prognostics = batch.global_prognostics(consume=True)
        # The readings that are a flat reduction over the globe's plane
        # (the radiation means, the floor's grid-mean heating, the
        # size-bounding fractions) are NOT formed here, on this batch's
        # rows: the planes go back unreduced and :meth:`finish` reduces
        # each once over the assembled globe.
        planes: dict[str, object] = {}
        if floor_heating is not None:
            planes[self._FLOOR_PLANE] = floor_heating
        bounding_columns = self._runtime.last_radiation_bounding_columns
        if bounding_columns is not None:
            for name, plane in bounding_columns.items():
                planes[self._SIZE_BOUNDING_PLANE + name] = plane
        diagnostics = {
            **schedule,
            "maximum_native_water_residual_kg_m2": maximum_water_residual,
            "native_water_residual_exceeds_tolerance": float(
                maximum_water_residual > self.options.water_fix_tolerance_kg_m2
            ),
            "maximum_native_energy_change_j_m2": maximum_energy_change,
            "native_energy_change_exceeds_advisory": float(
                maximum_energy_change > self.options.energy_change_advisory_j_m2
            ),
            "maximum_local_water_repair_kg_m2": maximum_water_residual,
        }
        if floor_heating is None:
            diagnostics["mean_stratospheric_floor_heating_j_m2"] = 0.0
        out.__exit__(None, None, None)
        result = PhysicsResult(
            **prognostics,
            surface=batch.surface.copy(),
            physics_state=persistent.export(),
            diagnostics=diagnostics,
            adapter_receipt=self.identity,
            planes=planes,
        )
        if exchange.band is None:
            # The whole grid, handed over by a harness or a test rather
            # than by the model's band loop: the one result is the call.
            return finish_alone(self, exchange, result)
        return result

    def finish(self, band_diagnostics, band_metadata, planes, surface,
               physics_state, *, metadata_in, columns=None, dt_s=None):
        """The call's diagnostics and namespace metadata from its bands'.

        ``band_diagnostics`` and ``band_metadata`` are each band's
        ``PhysicsResult.diagnostics`` and ``.physics_state.metadata`` in
        band order; ``planes`` the globe's assembled planes under the names
        the bands returned them; ``surface`` and ``physics_state`` the
        whole surface and namespace the bands were written into;
        ``metadata_in`` the namespace metadata the call started from.
        One band is the whole grid and takes the same path.
        """
        rules = dict(self.DIAGNOSTIC_MERGE)
        for values in band_diagnostics:
            for name in values:
                if name.startswith("cumulus_") and name != "cumulus_active":
                    rules.setdefault(name, COLUMN_MEAN)
        diagnostics = merge_band_scalars(
            band_diagnostics, rules, columns=columns,
            what="native physics diagnostic")
        metadata = merge_band_metadata(band_metadata, self.METADATA_MERGE)
        if bool(diagnostics.get("radiation_due", False)):
            last, total = self._size_bounding_record(
                band_metadata, planes, metadata_in)
        else:
            last = metadata_in.get("radiation_size_bounding_last")
            total = metadata_in.get("radiation_size_bounding_sum")
        metadata["radiation_size_bounding_last"] = (
            None if last is None else dict(last))
        metadata["radiation_size_bounding_sum"] = (
            None if total is None else dict(total))
        diagnostics.update(self._radiation_diagnostics(
            surface, physics_state.arrays, metadata))
        heating = planes.get(self._FLOOR_PLANE)
        if heating is not None:
            # The same reading the one-call form took: the float32 plane's
            # own mean, over the whole globe.
            diagnostics["mean_stratospheric_floor_heating_j_m2"] = float(
                np.asarray(to_host(heating)).mean())
        if self.observer is not None and hasattr(self.observer, "close_call"):
            self.observer.close_call(
                self._xp if self._xp is not None else self._array_module())
        return diagnostics, metadata


__all__ = [
    "ArwenCudaColumnSuite", "CUMULUS_SLOT", "NATIVE_COMPONENT_ORDER",
    "native_component_order",
]
