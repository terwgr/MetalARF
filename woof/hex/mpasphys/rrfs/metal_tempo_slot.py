"""Resident MPAS RRFS TEMPO slot for the Metal framework."""

import numpy as np

from woof.metal.array import MetalArray

from . import metal_tempo_v303 as metal
from .tempo_v303 import (
    CONFIG_ARGUMENTS, CONSTANT_ARGUMENTS, DIAG_ARGUMENTS, FIELD_ARGUMENTS,
    MODEL_ARGUMENTS, PRECIP_ARGUMENTS, PROFILE_ARGUMENTS, RADAR_ARGUMENTS,
    STATE_FIELDS, SURFACE_ARGUMENTS, TEND_ARGUMENTS, _allocated, radii_due,
)


_FIELD_ALIASES = {"p": "pres_p", "pii": "pi_p", "qb": "volg_p"}
_PROFILE_ALIASES = {"re_cloud": "recloud_p", "re_ice": "reice_p", "re_snow": "resnow_p"}
_RATE_FIELDS = ("ave_prate_1min", "ave_prate_5min", "ave_prate_10min")


class MetalMpasTempo:
    """Apply the fork microphysics slot to caller-owned level-two pools.

    The native framework owns initialization, allocation and slot rollback.
    This service owns the TEMPO first-call switch and rolling rain history.
    """

    def __init__(self, flags, *, tables=None, first_call_main=True):
        self.flags = metal._configuration_flags(flags)
        if self.flags[3]:
            raise NotImplementedError("Metal RRFS TEMPO requires non-melting reflectivity")
        self.tables = tables
        self.first_call_main = bool(first_call_main)
        self._table_sha256 = None
        self._table_pin_set = None
        self._rate_dt = None
        self._rolling = None

    @classmethod
    def from_table_witness(cls, context, flags, receipt, *, pin_set, reference_sha256):
        service = cls(flags)
        service.tables = metal.upload_table_witness(context, receipt, flags=service.flags,
                                                    pin_set=pin_set, reference_sha256=reference_sha256)
        service._table_sha256 = reference_sha256
        service._table_pin_set = pin_set
        return service

    def restart_state(self):
        rolling = self._rolling
        if isinstance(rolling, MetalArray):
            rolling = rolling.to_numpy()
        return {"flags": self.flags, "first_call_main": self.first_call_main,
                "table_sha256": self._table_sha256, "rate_dt": self._rate_dt, "rolling_prate": rolling}

    def restore_restart_state(self, state):
        if tuple(state["flags"]) != self.flags or type(state["first_call_main"]) is not bool:
            raise ValueError("RRFS TEMPO checkpoint flags differ from the service")
        if state.get("table_sha256") != self._table_sha256:
            raise ValueError("RRFS TEMPO checkpoint table witness differs from the service")
        rate_dt, rolling = state["rate_dt"], state["rolling_prate"]
        if rate_dt is not None and (not np.isfinite(rate_dt) or rate_dt <= 0):
            raise ValueError("RRFS TEMPO checkpoint has an invalid rate timestep")
        if rolling is not None and (rate_dt is None or not isinstance(rolling, np.ndarray)
                                    or rolling.dtype != np.float32 or rolling.ndim != 2):
            raise ValueError("RRFS TEMPO checkpoint has invalid rolling precipitation words")
        self.first_call_main = state["first_call_main"]
        self._rate_dt = rate_dt
        self._rolling = rolling

    def _rate_windows(self, xp, context, nc, dt):
        rate_dt = np.float32(dt if self._rate_dt is None else self._rate_dt)
        counts = np.asarray((60, 300, 600), np.float32) / rate_dt
        if not np.all(np.isfinite(counts) & (counts < np.float64(np.iinfo(np.int32).max - 1))):
            raise ValueError("RRFS TEMPO rate timestep exceeds the window index range")
        # Fortran FRACTION returns the binary significand, not the fractional part.
        fraction = np.abs(np.frexp(counts)[0]) > np.float32(1.e-6)
        limits = np.where(fraction, np.floor(counts + np.float32(.5)), np.trunc(counts)).astype(np.int32) + 1
        times = np.where(fraction, rate_dt * (limits - 1).astype(np.float32), (60, 300, 600))
        shape = (int(limits.max()), nc)
        if np.any(limits == 1):
            if self._rolling is not None:
                raise ValueError("RRFS TEMPO checkpoint carries an inactive rate window")
        elif self._rolling is None:
            if self._rate_dt is not None:
                raise ValueError("RRFS TEMPO checkpoint is missing its rolling precipitation")
            if shape[0] * shape[1] * 4 > min(context.max_buffer_length, context.recommended_working_set_size):
                raise MemoryError("RRFS TEMPO rate window exceeds the Metal working set")
            self._rolling = xp.zeros(shape, np.float32)
        elif self._rolling.shape != shape:
            raise ValueError("RRFS TEMPO checkpoint rate window differs from the solve extent")
        elif isinstance(self._rolling, np.ndarray):
            self._rolling = xp.asarray(self._rolling)
        self._rate_dt = float(rate_dt)
        return self._rolling, limits, times

    def launch_slot(self, call):
        if (call.slot, call.scheme) != ("microphysics", "mp_tempo"):
            raise ValueError("RRFS TEMPO service requires its microphysics slot")
        pools = call.pools
        if pools.level != 2:
            raise ValueError("RRFS TEMPO requires state time level two")
        flags = tuple(bool(call.options[name]) for name in CONFIG_ARGUMENTS)
        if flags != self.flags:
            raise ValueError("RRFS TEMPO flags differ from initialized table resources")
        aerosol, hail, subgrid, _ = flags
        xp, native = pools.cp, pools.native
        scalar, state = pools["scalar_state"], pools["state"]
        count, step = int(scalar["n_microp"]), np.float32(scalar["dt_microp"])
        if count < 1 or not np.isfinite(step) or step <= 0:
            raise ValueError("RRFS TEMPO requires positive substeps and microphysics timestep")

        scalars = state["scalars"]
        _, nz, nc = scalars.shape
        if nc != pools["nCellsSolve"] or nz < 2:
            raise ValueError("RRFS TEMPO requires compact native solve columns and at least two levels")
        context = scalars.buffer.context
        names = (MODEL_ARGUMENTS[:8] + (MODEL_ARGUMENTS[8:11] if aerosol else ())
                 + (MODEL_ARGUMENTS[11:13] if hail else ()))
        model = {}
        for name in names:
            index = pools.session.layout.indices["state", "index_" + name]
            if not 1 <= index <= scalars.shape[0]:
                raise ValueError(f"RRFS TEMPO scalar index outside the pool for {name}")
            model[name] = scalars[index - 1]
        model.update((name, state[name][:nz]) for name in MODEL_ARGUMENTS[13:])
        if not _allocated(native, "qv_p"):
            raise ValueError("RRFS TEMPO requires the resident native preparation workspace")
        prepared_pool, phys = pools["prepared"], pools["diag_physics"]
        optional = {name for name in FIELD_ARGUMENTS[12:17]
                    if _allocated(native, _FIELD_ALIASES.get(name, name + "_p"))}
        presence = sum(1 << i for i, name in enumerate(FIELD_ARGUMENTS[12:17]) if name in optional)
        fields = {name: prepared_pool[_FIELD_ALIASES.get(name, name + "_p")][:nz]
                  for name in FIELD_ARGUMENTS if name not in FIELD_ARGUMENTS[12:17] or name in optional}
        surface = {name: prepared_pool[name + "_p"]
                   for name in (SURFACE_ARGUMENTS if aerosol else SURFACE_ARGUMENTS[:10])}
        diag = {name: pools["diag"][name] for name in DIAG_ARGUMENTS
                if name not in ("rt_diabatic_tend", "tend_sfc_pressure")}
        diag.update((name, pools["tend"][name]) for name in ("rt_diabatic_tend", "tend_sfc_pressure"))
        diag.update((name, phys[name]) for name in RADAR_ARGUMENTS)
        diag["relhum"] = pools["diag"]["relhum"][:nz]
        tendencies = {name: pools["tend_physics"][name]
                      for name in (TEND_ARGUMENTS if aerosol else TEND_ARGUMENTS[:9])}
        mesh = {"zz": pools["mesh"]["zz"][:nz], "zgrid": pools["mesh"]["zgrid"][:nz + 1]}
        constants = {name: pools["constants"][name] for name in CONSTANT_ARGUMENTS}
        precip = {name: phys[name] for name in (*PRECIP_ARGUMENTS, "i_rainnc")}
        radii = radii_due(pools, call)
        profiles = {name: prepared_pool[_PROFILE_ALIASES.get(name, name + "_p")][:nz]
                    if radii or name == "refl10cm" else xp.empty((nz, nc), np.float32)
                    for name in PROFILE_ARGUMENTS}
        try:
            hourly = phys["do_diag_dbz"]
        except (KeyError, TypeError, ValueError):
            hourly = None
        if not isinstance(hourly, MetalArray) or hourly.dtype != np.int32 or hourly.size != 1:
            hourly = bool(native.get_field("diag_physics", "do_diag_dbz")[()] == 1)
        top, compmax, comp = nz, None, None
        if "config_microphysics_top" in call.options:
            top = int(native.get_field("tend_physics", "mp_top_level"))
            compmax, comp = phys["refl10cm_compref_max"], diag["refl10cm_max"]
        rate = self._rate_windows(xp, context, nc, step) if call.options.get("calc_prate_max", False) else None
        if self.tables is None:
            self.tables = metal.upload_initialized_tables(context, native, flags=flags)

        slab = xp.empty((len(FIELD_ARGUMENTS), nz, nc), np.float32)
        saved = xp.empty((len(TEND_ARGUMENTS), nz, nc), np.float32)
        packed = xp.empty((len(STATE_FIELDS), nz, nc), np.float32)
        # The from operator leaves allocated optional fields alone if their scheme is off.
        for i, name in enumerate(FIELD_ARGUMENTS[12:17], 12):
            if name in optional:
                slab[i][...] = fields[name]
        bucket = call.options["config_bucket_rainnc"]
        accrue = bool(scalar["l_acrain"])
        metal.publish_native_precip(slab, surface, precip, solve_cells=nc,
                                    finish=False, accrue=accrue, bucket=bucket)
        metal.prepare_native_columns(model, mesh, diag, constants, slab, saved,
                                     solve_cells=nc, aerosol=aerosol, hail=hail, presence=presence)
        for name in SURFACE_ARGUMENTS[8:10] + (SURFACE_ARGUMENTS[10:] if aerosol else ()):
            surface[name][...] = phys[name]
        if aerosol:
            metal.apply_native_emissions(slab, surface["nwfa2d"], solve_cells=nc, dt=step)
        pbl = None
        if subgrid and radii:
            pbl = xp.empty((2, nz, nc), np.float32)
            pbl[0][...] = phys["qc_bl"]
            pbl[1][...] = phys["cldfrac_bl"]
        for sub in range(count):
            metal.pack_native_columns(slab, packed, solve_cells=nc, presence=presence)
            out = metal.launch_columns(packed, self.tables, dt=step, flags=flags,
                                       first_call_main=self.first_call_main and sub == 0,
                                       pbl=pbl, presence=presence)
            metal.pack_native_columns(slab, packed, solve_cells=nc, presence=presence, unpack=True)
            metal.accumulate_native_diagnostics(out, surface, profiles, solve_cells=nc)
        for i, name in enumerate(FIELD_ARGUMENTS):
            if name in fields:
                fields[name][...] = slab[i]

        if call.flags.diagnostics or isinstance(hourly, MetalArray) or hourly:
            metal.publish_native_radar(slab, profiles["refl10cm"], diag, solve_cells=nc,
                                       full=call.flags.diagnostics, hourly=hourly)
        metal.publish_native_precip(slab, surface, precip, solve_cells=nc,
                                    finish=True, accrue=accrue, bucket=bucket)
        if rate is not None:
            rolling, limits, times = rate
            metal.publish_native_rate(surface["rainnc"], precip["rainnc"], phys["prate_max"],
                                      {name: phys[name] for name in _RATE_FIELDS}, solve_cells=nc,
                                      dt=step, rolling=rolling, limits=limits, times=times)
        for name in PROFILE_ARGUMENTS:
            if name == "refl10cm" or radii:
                phys[name][...] = profiles[name]
        metal.publish_native_tendencies(model, mesh, diag, tendencies, constants, slab, saved,
                                        solve_cells=nc, aerosol=aerosol, hail=hail, dt=call.dt,
                                        top=top, compmax=compmax, comp=comp)
        for name in SURFACE_ARGUMENTS[8:10]:
            precip[name][...] = surface[name]
        self.first_call_main = False
