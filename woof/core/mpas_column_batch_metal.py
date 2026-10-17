"""Metal sibling of the MPAS column-batch physics seam.

``woof/core/mpas_column_batch.py`` is the frozen CuPy seam; this module is
its Metal mirror, with the same constructor inputs, the same two-phase
contract and the same restart identity, on resident ``MetalArray`` columns
shaped ``[level, column]`` (interfaces ``[level + 1, column]``, ``k = 0``
the lowest level).

The orchestration is the shared one, not a rewrite: phase 1 calls
``PhysicsDriver.compute`` on a persistent driver built by
``initialize_physics``; phase 2 calls ``woof.core.microphysics.apply``
followed by ``PhysicsDriver.accept_microphysics``.  The column state
carries ``prepared_physics_atmosphere`` and sets ``_metal_setup_state`` and
``_array_module``, so the regional Metal lanes of ``PhysicsDriver`` take
it, with identity mass coupling (``c1h = 1``, ``c2h = 0``, total dry mass
exactly 1, unit map factors) exactly as the frozen seam builds it.

The seam's own arithmetic (the ``max(qv, 0)`` clamp, ``exner``,
``T = theta*exner``, ``dz``, ``rho = (1 + qv)*rho_dry``, ``alt = 1/rho_dry``,
``php = z_interface*g``) runs in the small ``hexv841_seam_*`` kernels with
the frozen formulas and operation order.  Plain copies use the
``MetalArray`` copy and view operations.

Only the WSM6 scheme row is carried.  The P3 and aerosol-aware Thompson
rows of the frozen seam refuse by name at construction.
"""

from __future__ import annotations

import numbers
from datetime import datetime

import numpy as np

from woof.core import constants as _constants
from woof.core.microphysics import apply as _arw_microphysics_apply
from woof.core.mpas_column_batch import (
    RESTART_SCHEMA, MpasColumnPhysicsTendencies, _GRAUPEL_SCHEMES,
    _LAZY_RESTART_KEYS, _MP_PHYSICS_BY_SCHEME, _OUTPUT_BUFFERS,
    _PHASE1, _PHASE2, _SPECIES, _SPECIES_BY_SCHEME,
    _STATE_RESTART_FIELDS_BY_SCHEME, _interface_weights,
    _require_positive, resolve_column_batch_cadences)
from woof.core.physics import PhysicsDriver, initialize_physics

__all__ = [
    "MetalMpasColumnBatchPhysics",
    "MpasColumnPhysicsTendencies",
    "resolve_column_batch_cadences",
    "run_mpas_column_batch_metal",
]

_F32 = np.float32


def _launch(context, kernel, count, arrays, params):
    from woof.hex.metal_v841_launch import launch
    launch(context, f"hexv841_seam_{kernel}", int(count), arrays, params)


def _flat(a):
    return a.view((a.size,))


class _MetalColumnBatchState:
    """The minimal ARW-shaped state the physics chain consumes, on Metal.

    Mirrors ``_ColumnBatchState`` field for field.  Grid
    ``(nz, ny=1, nx=ncol)``; ``thb`` is zero so total theta is ``thp`` and
    total dry mass is the constant one.
    """

    def __init__(self, cfg, fnm, fnp, p_top, terrain_height, xp,
                 species=_SPECIES):
        nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
        shape3 = (nz, ny, nx)
        self._metal_setup_state = True
        self._host_setup_state = False
        self._array_module = xp

        def zeros(*shape):
            return xp.zeros(shape, dtype=_F32)

        def full(shape, value):
            return xp.asarray(np.full(shape, _F32(value), dtype=_F32))

        self.p = zeros(*shape3)
        self.thp = zeros(*shape3)
        self.thb = zeros(nz)
        self.phb = zeros(nz + 1)
        self.php = zeros(nz + 1, ny, nx)
        self.alt = zeros(*shape3)
        self.w = zeros(nz + 1, ny, nx)
        for name in species:
            setattr(self, name, zeros(*shape3))
        # WSM6 scheme-native radii (microns), the mp=6 cold start.
        self.effc = full(shape3, 2.49)
        self.effi = full(shape3, 4.99)
        self.effs = full(shape3, 9.99)
        self.h_diabatic = zeros(*shape3)

        # Identity mass coupling: chm = c1h*(mub2d + mup) + c2h == 1.0.
        self.mup = zeros(ny, nx)
        self.mub2d = xp.ones((ny, nx), dtype=_F32)
        self.c1h = xp.ones(nz, dtype=_F32)
        self.c2h = zeros(nz)
        self.msft = xp.ones((ny, nx), dtype=_F32)
        self.msfu = xp.ones((ny, nx + 1), dtype=_F32)
        self.msfv = xp.ones((ny + 1, nx), dtype=_F32)
        self.has_msf = False

        self.ht = xp.asarray(
            np.asarray(terrain_height, dtype=_F32).reshape(ny, nx))
        self.fnm = xp.asarray(np.asarray(fnm, dtype=_F32))
        self.fnp = xp.asarray(np.asarray(fnp, dtype=_F32))
        self.p_top = float(p_top)

        self.elapsed_seconds = 0.0
        self.physics = None
        self.prepared_physics_atmosphere = None
        self._scratch: dict = {}
        self._xp = xp

    # -- DomainState-compatible surface ---------------------------------
    def scratch(self, shape, slot: str, dtype=None):
        shape = tuple(shape) if isinstance(shape, (tuple, list)) else (shape,)
        requested = np.dtype(np.float32 if dtype is None else dtype)
        buf = self._scratch.get(slot)
        if buf is None:
            buf = self._xp.zeros(shape, dtype=requested)
            self._scratch[slot] = buf
        elif buf.shape != shape:
            raise ValueError(f"scratch slot {slot!r} has shape {buf.shape}, "
                             f"requested {shape}")
        elif buf.dtype != requested:
            raise ValueError(f"scratch slot {slot!r} has dtype {buf.dtype}, "
                             f"requested {requested}")
        return buf

    def existing_scratch(self, slot: str):
        return self._scratch.get(slot)

    def total_theta(self):
        # thb == 0: total theta is thp (0 + x == x).
        return self.thp

    def total_mu(self):
        # mub2d + mup == 1.0 + 0.0, held constant.
        return self.mub2d


class MetalMpasColumnBatchPhysics:
    """Persistent two-phase physics seam for one MPAS column batch (Metal).

    Construct once (``run_mpas_column_batch_metal``), then per model step
    call ``run_phase1`` and ``run_phase2`` strictly alternating.
    """

    _PHASE1_ORCHESTRATION = staticmethod(PhysicsDriver.compute)
    _PHASE2_MICROPHYSICS = staticmethod(_arw_microphysics_apply)

    def __init__(self, *, n_levels, n_columns, dt,
                 radiation_seconds, surface_pbl_seconds,
                 cumulus_seconds=None, cumulus_scheme=None,
                 microphysics_scheme="wsm6",
                 start_time, latitude_deg, longitude_deg,
                 terrain_height_m, z_interface_nominal_m,
                 p_top_pa, dx_m, gf_ishallow=None, dx_column_m=None,
                 landmask=1.0, xland=None, ivgtyp=10, isltyp=6,
                 vegfra=50.0, tsk=300.0, tmn=285.0, xice=0.0, snow=0.0,
                 snow_depth=0.0, soil_temperature=285.0,
                 soil_moisture=0.30, xice_threshold=0.5,
                 wsm6_hail_opt=0, array_module=None):
        if not isinstance(n_levels, numbers.Integral) or n_levels < 3:
            raise ValueError(
                f"n_levels must be an integer >= 3, got {n_levels!r}")
        if not isinstance(n_columns, numbers.Integral) or n_columns < 1:
            raise ValueError(
                f"n_columns must be a positive integer, got {n_columns!r}")
        if not isinstance(start_time, datetime):
            raise TypeError("start_time must be a datetime (UTC)")
        if microphysics_scheme not in _SPECIES_BY_SCHEME:
            raise ValueError(
                f"microphysics_scheme must be one of "
                f"{sorted(_SPECIES_BY_SCHEME)}, got "
                f"{microphysics_scheme!r}: each row names the exact "
                "transported species set, and a scheme without a row "
                "would run on substituted state")
        if microphysics_scheme != "wsm6":
            raise NotImplementedError(
                f"the Metal column-batch seam carries only "
                f"microphysics_scheme='wsm6', got {microphysics_scheme!r}")
        cadences = resolve_column_batch_cadences(
            dt=dt, radiation_seconds=radiation_seconds,
            surface_pbl_seconds=surface_pbl_seconds,
            cumulus_seconds=cumulus_seconds,
            cumulus_scheme=cumulus_scheme)
        p_top = _require_positive("p_top_pa", p_top_pa)
        dx = _require_positive("dx_m", dx_m)
        if int(wsm6_hail_opt) not in (0, 1):
            raise ValueError(
                f"wsm6_hail_opt must be 0 or 1, got {wsm6_hail_opt!r}")
        threshold = float(xice_threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "xice_threshold must be a finite fraction in [0, 1] "
                f"(WRF Registry default 0.5), got {xice_threshold!r}")
        if gf_ishallow is None:
            gf_ishallow = 1 if cumulus_scheme == "gf" else 0
        if gf_ishallow not in (0, 1):
            raise ValueError(
                f"gf_ishallow must be 0 or 1, got {gf_ishallow!r}")
        if gf_ishallow and cumulus_scheme != "gf":
            raise ValueError(
                "gf_ishallow=1 requires cumulus_scheme='gf'")
        nz, ncol = int(n_levels), int(n_columns)
        fnm, fnp = _interface_weights(z_interface_nominal_m)
        if fnm.shape[0] != nz:
            raise ValueError(
                f"z_interface_nominal_m has {fnm.shape[0] + 1} interfaces "
                f"but n_levels={nz} needs {nz + 1}")
        lat = np.asarray(latitude_deg, dtype=np.float32).reshape(-1)
        lon = np.asarray(longitude_deg, dtype=np.float32).reshape(-1)
        if lat.shape != (ncol,) or lon.shape != (ncol,):
            raise ValueError(
                "latitude_deg/longitude_deg must have one value per "
                f"column ({ncol}), got {lat.shape}/{lon.shape}")
        terrain = np.asarray(terrain_height_m,
                             dtype=np.float32).reshape(-1)
        if terrain.shape != (ncol,):
            raise ValueError(
                f"terrain_height_m must have shape ({ncol},), got "
                f"{terrain.shape}")
        dx_column = None
        if dx_column_m is not None:
            dx_column = np.asarray(dx_column_m,
                                   dtype=np.float32).reshape(-1)
            if dx_column.shape != (ncol,):
                raise ValueError(
                    f"dx_column_m must have one value per column "
                    f"({ncol}), got {dx_column.shape}")
            if not np.all(np.isfinite(dx_column)) or np.any(
                    dx_column <= 0.0):
                raise ValueError(
                    "dx_column_m must be finite and positive")
        top_nominal = float(np.asarray(
            z_interface_nominal_m, dtype=np.float64)[-1])

        from woof.config import RunConfig
        from woof.physics_compat import RRTMG_VARIANT_LEGACY
        from woof.metal.array import MetalArrayModule
        mp_physics = _MP_PHYSICS_BY_SCHEME[microphysics_scheme]
        cfg = RunConfig(
            nx=ncol, ny=1, nz=nz, dx=dx, dy=dx,
            ztop=top_nominal, dt=float(cadences["dt"]),
            run_seconds=0.0, moist=True, mp_physics=mp_physics,
            wsm6_hail_opt=int(wsm6_hail_opt),
            ra_physics=4, ra_rrtmg_variant=RRTMG_VARIANT_LEGACY,
            sf_sfclay_physics=1, sf_surface_physics=4,
            bl_pbl_physics=1,
            cu_physics=cadences["cu_physics"],
            ishallow=int(gf_ishallow),
            radt=cadences["radt_minutes"],
            radt_minutes=cadences["radt_minutes"],
            bldt=cadences["bldt_minutes"],
            cudt_minutes=cadences["cudt_minutes"])
        self._cfg = cfg
        self._cadences = cadences
        self._dt = float(cadences["dt"])
        self._nz, self._ncol = nz, ncol
        self._start_time = start_time
        self._microphysics_scheme = microphysics_scheme
        self._has_graupel = microphysics_scheme in _GRAUPEL_SCHEMES
        self._species = _SPECIES_BY_SCHEME[microphysics_scheme]
        self._state_restart_fields = (
            _STATE_RESTART_FIELDS_BY_SCHEME[microphysics_scheme])
        self._identity = {
            "schema": RESTART_SCHEMA,
            "n_levels": nz, "n_columns": ncol, "dt": self._dt,
            "stepra": cadences["stepra"], "stepbl": cadences["stepbl"],
            "stepcu": cadences["stepcu"],
            "cumulus_scheme": cumulus_scheme,
            "start_time": start_time.isoformat(),
            "p_top_pa": p_top, "dx_m": dx,
            "gf_ishallow": int(gf_ishallow),
            "gf_dx_column": ("per-column" if dx_column is not None
                             else "scalar"),
            "wsm6_hail_opt": int(wsm6_hail_opt),
            "xice_threshold": threshold,
            "xland_source": ("native" if xland is not None
                             else "derived-from-landmask"),
        }

        # ---- device construction from here on --------------------------
        xp = array_module if array_module is not None else MetalArrayModule()
        if not getattr(xp, "__woof_metal_array_module__", False):
            raise TypeError("array_module must be a MetalArrayModule")
        self._xp = xp
        self._context = xp.context

        state = _MetalColumnBatchState(cfg, fnm, fnp, p_top, terrain, xp,
                                       species=self._species)
        self._state = state
        self._species_buffers = {
            name: getattr(state, name) for name in self._species}

        def host(values):
            if hasattr(values, "to_numpy"):
                return values.to_numpy()
            return values

        def surface(values):
            values = host(values)
            if np.ndim(values) == 0:
                return values
            return np.asarray(values).reshape(1, ncol)

        def soil(values):
            values = host(values)
            if np.ndim(values) == 0:
                return values
            values = np.asarray(values)
            return values.reshape(values.shape[0], 1, ncol)

        self._driver = initialize_physics(
            state, cfg,
            landmask=surface(landmask),
            xland=None if xland is None else surface(xland),
            xice_threshold=threshold,
            tsk=surface(tsk),
            soil_temperature=soil(soil_temperature),
            soil_moisture=soil(soil_moisture),
            ivgtyp=surface(ivgtyp), isltyp=surface(isltyp),
            vegfra=surface(vegfra), tmn=surface(tmn),
            xice=surface(xice), snow=surface(snow),
            snow_depth=surface(snow_depth),
            glw=None,
            radiation_start_time=start_time,
            radiation_latitude=lat.reshape(1, ncol),
            radiation_longitude=lon.reshape(1, ncol))

        self._aerosol_init = None
        shape3 = (nz, 1, ncol)
        iface3 = (nz + 1, 1, ncol)
        self._gf_rthdynten = xp.zeros(shape3, dtype=_F32)
        self._gf_rqvdynten = xp.zeros(shape3, dtype=_F32)
        self._driver.gf_rthdynten = self._gf_rthdynten
        self._driver.gf_rqvdynten = self._gf_rqvdynten
        if dx_column is not None:
            self._driver.gf_dx_column = xp.asarray(
                dx_column.reshape(1, ncol))
        self._in = {name: xp.zeros(shape3, dtype=_F32)
                    for name in ("u", "v", "theta", "temperature",
                                 "pressure", "exner", "dz", "rho")}
        self._in.update({name: xp.zeros(iface3, dtype=_F32)
                         for name in ("p_interface", "z_interface")})
        self._atmosphere = None
        state.prepared_physics_atmosphere = self._supplied_atmosphere

        self._out = {name: xp.zeros((nz, ncol), dtype=_F32)
                     for name in _OUTPUT_BUFFERS}

        self._step_index = 0
        self._pending_phase = _PHASE1
        state.elapsed_seconds = 0.0

    # ------------------------------------------------------------------
    def _supplied_atmosphere(self):
        if self._atmosphere is None:
            raise RuntimeError(
                "the column-batch atmosphere is built by run_phase1; the "
                "driver must not run outside it")
        return self._atmosphere

    def _require_input(self, name, value, shape):
        from woof.metal.array import MetalArray
        if not isinstance(value, MetalArray):
            raise TypeError(f"{name} must be a MetalArray")
        if value.dtype != np.float32:
            raise TypeError(f"{name} must be float32, got {value.dtype}")
        if value.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape} [level, column], got "
                f"{value.shape}")
        if value.buffer.context is not self._context:
            raise ValueError(f"{name} belongs to a different Metal context")
        return value

    def _require_species(self, phase: str, **candidates) -> dict:
        scheme = self._microphysics_scheme
        missing = [name for name in self._species
                   if candidates.get(name) is None]
        if missing:
            raise ValueError(
                f"{phase}: microphysics_scheme={scheme!r} transports "
                f"{self._species} and {missing} were not supplied; a "
                "zero-defaulted species is substituted state, not an "
                "optional input")
        extra = sorted(name for name, value in candidates.items()
                       if value is not None and name not in self._species)
        if extra:
            raise ValueError(
                f"{phase}: microphysics_scheme={scheme!r} transports "
                f"{self._species}; {extra} are not state of this scheme "
                "and accepting them would silently drop caller fields")
        return {name: candidates[name] for name in self._species}

    def _rebind_species_buffers(self):
        state = self._state
        for name, buffer in self._species_buffers.items():
            setattr(state, name, buffer)
        state.thp = self._in["theta"]
        state.p = self._in["pressure"]

    def _v3(self, a):
        return a.view((a.shape[0], 1, self._ncol))

    # ------------------------------------------------------------------
    def run_phase1(self, *, dt, u, v, theta, pressure, pressure_interface,
                   z_interface, w, rho_dry, qv, qc, qr, qi, qs=None,
                   qg=None, ni=None, nr=None, qir=None, qib=None,
                   nc=None, nwfa=None, nifa=None,
                   exner=None, rthdynten=None,
                   rqvdynten=None) -> MpasColumnPhysicsTendencies:
        """Run the due phase-1 chain; return the held raw tendency set."""
        if self._pending_phase != _PHASE1:
            raise RuntimeError(
                "run_phase1 called out of order: the previous step's "
                "run_phase2 has not run (strict phase1 -> phase2 "
                "alternation)")
        if float(dt) != self._dt:
            raise ValueError(
                f"phase-1 dt={dt} does not match the constructor model "
                f"step {self._dt}; the seam owns the sub-cadences")
        nz, ncol = self._nz, self._ncol
        ctx = self._context
        level, iface = (nz, ncol), (nz + 1, ncol)
        inputs = {"u": u, "v": v, "theta": theta, "pressure": pressure,
                  "rho_dry": rho_dry}
        inputs.update(self._require_species(
            "run_phase1", qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg,
            ni=ni, nr=nr, qir=qir, qib=qib, nc=nc, nwfa=nwfa,
            nifa=nifa))
        for name in ("u", "v", "theta", "pressure",
                     "rho_dry") + self._species:
            self._require_input(name, inputs[name], level)
        for name, value in (("pressure_interface", pressure_interface),
                            ("z_interface", z_interface), ("w", w)):
            self._require_input(name, value, iface)
        if exner is not None:
            self._require_input("exner", exner, level)
        for name, value in (("rthdynten", rthdynten),
                            ("rqvdynten", rqvdynten)):
            if value is not None:
                self._require_input(name, value, level)

        state, buf = self._state, self._in
        self._rebind_species_buffers()
        v3 = self._v3
        n = nz * ncol

        buf["u"][...] = v3(u)
        buf["v"][...] = v3(v)
        buf["theta"][...] = v3(theta)
        buf["pressure"][...] = v3(pressure)
        buf["p_interface"][...] = v3(pressure_interface)
        buf["z_interface"][...] = v3(z_interface)
        state.w[...] = v3(w)
        # NEGATIVE-QV LAW: clamp in seam scratch, never in the caller array.
        _launch(ctx, "clamp_qv", n,
                {"qv": _flat(qv), "out": _flat(state.qv)}, {"n": n})
        for name in self._species[1:]:
            getattr(state, name)[...] = v3(inputs[name])
        if exner is None:
            _launch(ctx, "exner", n,
                    {"pressure": _flat(buf["pressure"]),
                     "exner": _flat(buf["exner"])},
                    {"n": n, "p0": _F32(_constants.P0),
                     "rcp": _F32(_constants.RCP)})
        else:
            buf["exner"][...] = v3(exner)
        _launch(ctx, "temperature", n,
                {"theta": _flat(buf["theta"]), "exner": _flat(buf["exner"]),
                 "temperature": _flat(buf["temperature"])}, {"n": n})
        _launch(ctx, "thickness", n,
                {"z_interface": _flat(buf["z_interface"]),
                 "dz": _flat(buf["dz"])}, {"n": n, "ncol": ncol})
        _launch(ctx, "density", n,
                {"qv": _flat(state.qv), "rho_dry": _flat(rho_dry),
                 "rho": _flat(buf["rho"])}, {"n": n})
        if rthdynten is None:
            self._gf_rthdynten[...] = _F32(0.0)
        else:
            self._gf_rthdynten[...] = v3(rthdynten)
        if rqvdynten is None:
            self._gf_rqvdynten[...] = _F32(0.0)
        else:
            self._gf_rqvdynten[...] = v3(rqvdynten)

        atmosphere_qs = state.qs
        self._atmosphere = {
            "theta": buf["theta"], "temperature": buf["temperature"],
            "pressure": buf["pressure"],
            "p_interface": buf["p_interface"], "exner": buf["exner"],
            "u": buf["u"], "v": buf["v"],
            "z_interface": buf["z_interface"], "dz": buf["dz"],
            "qv": state.qv, "qc": state.qc, "qi": state.qi,
            "qs": atmosphere_qs, "rho": buf["rho"],
        }
        state.elapsed_seconds = float(
            np.float64(self._step_index) * np.float64(self._dt))
        driver, cfg = self._driver, self._cfg
        before = dict(driver.call_counts)
        try:
            tendencies = type(self)._PHASE1_ORCHESTRATION(
                driver, state, cfg)
        finally:
            self._atmosphere = None
        out = self._out

        def flat2(value):
            return value.view((nz, ncol))

        out["dtheta"][...] = flat2(tendencies.rtheta)
        out["dqv"][...] = flat2(tendencies.rqv)
        out["dqc"][...] = flat2(tendencies.rqc)
        for name, value in (("dqr", tendencies.rqr),
                            ("dqi", tendencies.rqi),
                            ("dqs", tendencies.rqs)):
            if value is None:
                out[name][...] = _F32(0.0)
            else:
                out[name][...] = flat2(value)
        last_ysu = driver.last_ysu
        if last_ysu is not None:
            out["du"][...] = flat2(last_ysu["du"])
            out["dv"][...] = flat2(last_ysu["dv"])
        out["h_diabatic"][...] = flat2(state.h_diabatic)

        after = driver.call_counts
        result = MpasColumnPhysicsTendencies(
            du=out["du"], dv=out["dv"], dtheta=out["dtheta"],
            dqv=out["dqv"], dqc=out["dqc"], dqr=out["dqr"],
            dqi=out["dqi"], dqs=out["dqs"], dqg=out["dqg"],
            h_diabatic=out["h_diabatic"],
            step_index=self._step_index,
            elapsed_seconds=state.elapsed_seconds,
            radiation_ran=after["radiation"] > before["radiation"],
            surface_pbl_ran=after["ysu"] > before["ysu"],
            cumulus_ran=after["cumulus"] > before["cumulus"])
        self._pending_phase = _PHASE2
        return result

    # ------------------------------------------------------------------
    def run_phase2(self, *, theta, qv, qc, qr, qi, qs=None, qg=None,
                   ni=None, nr=None, qir=None, qib=None, nc=None,
                   nwfa=None, nifa=None, pressure,
                   rho_dry=None, z_interface,
                   refl_10cm_due: bool = False) -> dict:
        """Run WSM6 in place on the caller's arrays; return the receipt."""
        if self._pending_phase != _PHASE2:
            raise RuntimeError(
                "run_phase2 called out of order: run_phase1 must run "
                "first (strict phase1 -> phase2 alternation)")
        nz, ncol = self._nz, self._ncol
        ctx = self._context
        level, iface = (nz, ncol), (nz + 1, ncol)
        species = self._require_species(
            "run_phase2", qv=qv, qc=qc, qr=qr, qi=qi, qs=qs, qg=qg,
            ni=ni, nr=nr, qir=qir, qib=qib, nc=nc, nwfa=nwfa,
            nifa=nifa)
        if rho_dry is None:
            raise ValueError(
                "run_phase2: microphysics_scheme='wsm6' requires "
                "rho_dry (WSM6's adapter derives rho = 1/alt, "
                "woof/core/wsm6.py:109-112)")
        for name, value in (("theta", theta), ("pressure", pressure),
                            *species.items()):
            self._require_input(name, value, level)
        self._require_input("rho_dry", rho_dry, level)
        self._require_input("z_interface", z_interface, iface)

        state, cfg, driver = self._state, self._cfg, self._driver
        v3 = self._v3

        state.thp = v3(theta)
        for name, value in species.items():
            setattr(state, name, v3(value))
        state.p = v3(pressure)
        alt = state.scratch((nz, 1, ncol), "physics_column_alt")
        _launch(ctx, "alt", nz * ncol,
                {"rho_dry": _flat(rho_dry), "alt": _flat(alt)},
                {"n": nz * ncol})
        state.alt = alt
        php = state.scratch((nz + 1, 1, ncol), "physics_column_php")
        _launch(ctx, "geopotential", (nz + 1) * ncol,
                {"z_interface": _flat(z_interface), "php": _flat(php)},
                {"n": (nz + 1) * ncol, "g": _F32(_constants.G)})
        state.php = php
        try:
            diagnostics = type(self)._PHASE2_MICROPHYSICS(
                state, cfg, self._dt, refl_10cm_due=refl_10cm_due)
            driver.accept_microphysics(diagnostics)

            def col(a):
                return a.view((ncol,)).copy()

            receipt = {
                "rainncv": col(diagnostics.rainncv),
                "snowncv": col(diagnostics.snowncv),
                "sr": col(diagnostics.sr),
            }
            if self._has_graupel:
                receipt["graupelncv"] = col(diagnostics.graupelncv)
            if refl_10cm_due:
                from woof.core.refl import consume_refl_10cm
                receipt["refl_10cm"] = consume_refl_10cm(state).view(
                    (nz, ncol)).copy()
        finally:
            self._rebind_species_buffers()
        self._step_index += 1
        state.elapsed_seconds = float(
            np.float64(self._step_index) * np.float64(self._dt))
        self._pending_phase = _PHASE1
        return receipt

    # ------------------------------------------------------------------
    @property
    def aerosol_init_receipt(self):
        return None

    def accumulated_precipitation(self) -> dict:
        """Device copies: RAINNC/SNOWNC/GRAUPELNC/RAINC (mm)."""
        ncol = self._ncol
        state, driver = self._state, self._driver
        buckets = [("RAINNC", "mp_rainnc"), ("SNOWNC", "mp_snownc")]
        if self._has_graupel:
            buckets.append(("GRAUPELNC", "mp_graupelnc"))
        out = {}
        for name, slot in buckets:
            out[name] = state.scratch((1, ncol), slot).view((ncol,)).copy()
        rainc = driver.rainc
        out["RAINC"] = (self._xp.zeros(ncol, dtype=_F32)
                        if rainc is None else rainc.view((ncol,)).copy())
        return out

    @property
    def call_counts(self) -> dict:
        return dict(self._driver.call_counts)

    @property
    def surface_classification(self) -> dict:
        return dict(self._driver.surface_classification)

    @property
    def last_noahmp_census(self):
        census = getattr(self._driver, "last_noahmp_census", None)
        return None if census is None else dict(census)

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def elapsed_seconds(self) -> float:
        return self._state.elapsed_seconds

    # ------------------------------------------------------------------
    def _restart_manifest(self):
        from woof.io.restart import (_driver_manifest,
                                      classify_scratch_slot)
        driver, state = self._driver, self._state
        manifest = dict(_driver_manifest(driver))
        for slot in sorted(state._scratch):
            if classify_scratch_slot(slot) == "serialize":
                manifest[f"scratch/{slot}"] = state._scratch[slot]
        for name in self._state_restart_fields:
            manifest[f"state/{name}"] = getattr(state, name)
        for name in _OUTPUT_BUFFERS:
            manifest[f"seam/{name}"] = self._out[name]
        return manifest

    def export_state(self) -> dict:
        """Host-side payload of every persisted item (step boundary only)."""
        if self._pending_phase != _PHASE1:
            raise RuntimeError(
                "export_state is legal only at a step boundary (after "
                "run_phase2, before the next run_phase1)")
        driver = self._driver
        arrays = {key: (value.to_numpy() if hasattr(value, "to_numpy")
                        else np.array(value))
                  for key, value in self._restart_manifest().items()}
        kf_time = getattr(driver.cumulus_callable, "_history_time", None)
        scalars = {
            "step_index": self._step_index,
            "elapsed_seconds": self._state.elapsed_seconds,
            "call_counts": dict(driver.call_counts),
            "microphysics_updates":
                int(driver.microphysics_updates),
            "ysu_nan_guard_fires": int(driver.ysu_nan_guard_fires),
            "kf_history_time": (None if kf_time is None
                                else float(kf_time)),
            "carriers": driver.carriers.state(),
        }
        return {
            "identity": dict(self._identity),
            "arrays": arrays,
            "scalars": scalars,
        }

    def restore_state(self, payload: dict) -> None:
        """Restore an ``export_state`` payload in place."""
        if self._pending_phase != _PHASE1:
            raise RuntimeError(
                "restore_state is legal only at a step boundary")
        expected = {"identity", "arrays", "scalars"}
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError(
                "restart payload must carry exactly "
                f"{sorted(expected)}, got "
                f"{sorted(payload) if isinstance(payload, dict) else payload!r}")
        if payload["identity"] != self._identity:
            raise ValueError(
                "restart payload identity does not match this seam's "
                f"configuration: {payload['identity']} != "
                f"{self._identity}")
        driver, state = self._driver, self._state
        live = self._restart_manifest()
        stored = dict(payload["arrays"])
        lazy = {}
        for key in _LAZY_RESTART_KEYS:
            if key in stored and key not in live:
                lazy[key] = stored.pop(key)
            elif key in live and key not in stored:
                raise ValueError(
                    f"restart payload lacks {key!r} but this seam "
                    "already carries live state for it; restore into a "
                    "freshly constructed seam")
        missing = sorted(set(live) - set(stored))
        extra = sorted(set(stored) - set(live))
        if missing or extra:
            raise ValueError(
                "restart payload arrays do not cover this seam "
                f"(missing {missing}, extra {extra})")
        for key, target in live.items():
            source = np.asarray(stored[key])
            if source.shape != target.shape:
                raise ValueError(
                    f"restart array {key!r} has shape {source.shape}, "
                    f"live {target.shape}")
            target[...] = np.ascontiguousarray(source, dtype=target.dtype)
        if "cumulus/w0avg" in lazy:
            adapter = driver.cumulus_callable
            adapter.w0avg = self._xp.asarray(
                np.asarray(lazy["cumulus/w0avg"], dtype=np.float32))
            adapter._history_state = state
        if "radiation/o33d_grid" in lazy:
            driver.radiation_callable._o33d_grid = np.asarray(
                lazy["radiation/o33d_grid"])
        scalars = payload["scalars"]
        self._step_index = int(scalars["step_index"])
        state.elapsed_seconds = float(scalars["elapsed_seconds"])
        driver.call_counts.clear()
        driver.call_counts.update(
            {key: int(value)
             for key, value in scalars["call_counts"].items()})
        driver.microphysics_updates = int(
            scalars["microphysics_updates"])
        driver.ysu_nan_guard_fires = int(scalars["ysu_nan_guard_fires"])
        adapter = driver.cumulus_callable
        if adapter is not None and hasattr(adapter, "_history_time"):
            adapter._history_time = scalars["kf_history_time"]
        driver.carriers.restore(scalars["carriers"])
        self._pending_phase = _PHASE1


def run_mpas_column_batch_metal(**kwargs) -> MetalMpasColumnBatchPhysics:
    """Construct the persistent Metal MPAS column-batch physics seam."""
    return MetalMpasColumnBatchPhysics(**kwargs)
