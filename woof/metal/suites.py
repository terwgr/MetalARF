"""Tiny idealized regional rows run through the public Metal step path.

Shared by ``woof selftest`` and the combination smoke tooling: each row is a
recognizable physics stack on an 8x8 periodic or bounded domain, built with
the same configuration, initialization and ``run_steps`` call a forecast
uses.  A row passes when every floating state field stays finite.
"""
from __future__ import annotations

from datetime import datetime, timezone
import sys

TURBULENCE_PROFILES = {
    "constant-k": (1, 10.0, 0.0),
    "smag-2d": (4, 0.0, 0.0),
    "smag-3d": (3, 0.0, 0.0),
    "tke-1.5": (2, 0.0, 0.0),
}


def _urban_option(value):
    """Accept the legacy SLUCM boolean or an explicit WRF urban option."""
    if isinstance(value, bool):
        return 1 if value else 0
    option = int(value)
    if option not in (0, 1, 2, 3):
        raise ValueError("urban model must be one of 0, 1, 2, or 3")
    return option


def configuration(combo, *, urban=False, turbulence=None, sfclay_option=None,
                  terrain_opt=None, mp_zero_out=0, mp_zero_out_all=0,
                  sf_lake_physics=0, alb_sol=0, spp_lsm=0, spp_conv=0,
                  h_sca_adv_order=2,
                  bl_mynn_version="wrf_461",
                  thompson_version="wrf_461"):
    from woof.config import RunConfig, validate_run_config
    from woof.physics_compat import WRF_RRTMG_TO_RTE_RRTMGP
    mp, pbl, lsm, cu, ra, boundary = combo
    gfs_kf_profile = (mp, pbl, lsm, cu, ra) == (10, 1, 2, 1, 4)
    sase_profile = pbl == 900
    km_opt, khdif, kvdif = TURBULENCE_PROFILES.get(
        turbulence, (0 if sase_profile else
                     4 if gfs_kf_profile else 1, 0.0, 0.0))
    return validate_run_config(RunConfig(
        nx=8, ny=8, nz=12 if cu == 3 else 10,
        dx=1000., dy=1000., ztop=10000.,
        num_soil_layers=9 if lsm == 3 else 4,
        dt=1., run_seconds=2., moist=(turbulence in (None, "constant-k")),
        mp_physics=mp,
        bl_pbl_physics=pbl, sf_surface_physics=lsm, cu_physics=cu,
        sf_lake_physics=int(sf_lake_physics), alb_sol=int(alb_sol),
        spp_lsm=int(spp_lsm), spp_conv=int(spp_conv),
        sf_urban_physics=_urban_option(urban),
        cudt_minutes=0. if cu in (3, 16) else 5.,
        sf_sfclay_physics=(2 if pbl == 2 else
                           int(sfclay_option) if sfclay_option is not None else
                           91 if any((mp, pbl, lsm, cu, ra)) else 0),
        terrain_opt=(1 if gfs_kf_profile else 0)
        if terrain_opt is None else int(terrain_opt),
        km_opt=km_opt, khdif=khdif, kvdif=kvdif,
        diff_opt=2,
        diff_6th_opt=2 if gfs_kf_profile else 0,
        diff_6th_slopeopt=1 if gfs_kf_profile else 0,
        nwp_diagnostics=1 if gfs_kf_profile else 0,
        # Matrix RAD=1 is Dudhia SW (LW off); RAD=4 is RRTMGP on both
        # streams; RAD=11 encodes the classic RRTM LW + Dudhia SW pair.
        ra_lw_physics=(4 if ra == 40 else
                       1 if ra == 11 else 0 if ra == 1 else ra),
        ra_sw_physics=(0 if ra == 40 else 1 if ra == 11 else ra),
        wrf_rrtmg_compatibility=(WRF_RRTMG_TO_RTE_RRTMGP
                                 if gfs_kf_profile else "none"),
        open_x=boundary == "open", open_y=boundary == "open",
        specified=boundary == "specified", spec_bdy_width=3,
        spec_zone=1, relax_zone=2, h_sca_adv_order=int(h_sca_adv_order),
        surface_radiation_policy="wrf_compat_zero",
        mp_zero_out=int(mp_zero_out),
        mp_zero_out_all=int(mp_zero_out_all),
        thompson_version=thompson_version,
        thompson_fork_snow_fall=(
            "wrf_39_noaa" if thompson_version == "wrf_39_noaa" else "blend"),
        bl_mynn_version=bl_mynn_version,
        bl_mynn_mixlength=(2 if bl_mynn_version == "gsd_41" else 1)))


def initialize(state, cfg, *, gates_only=False, noahmp_glacier_smoke=False):
    import numpy as np
    from woof.core.physics import initialize_physics, physics_driver_required
    if noahmp_glacier_smoke and cfg.sf_surface_physics != 4:
        raise ValueError("Noah-MP glacier smoke requires sf_surface_physics=4")
    if physics_driver_required(cfg):
        options = {}
        if (cfg.ra_lw_physics == 4
                and cfg.ra_sw_physics in (0, 4)):
            from woof.core.rrtmgp import RRTMGPRadiation
            latitude = np.full((cfg.ny, cfg.nx), 35., np.float32)
            longitude = np.full_like(latitude, -100.)
            if gates_only:
                # Gate predicates inspect type/geometry/flags, not table data.
                radiation = object.__new__(RRTMGPRadiation)
                radiation.latitude_deg = latitude
                radiation.longitude_deg = longitude
                radiation.longwave = True
                radiation.shortwave = cfg.ra_sw_physics == 4
                radiation.publishes_olr = True
                radiation.validation_mode = "fused"
            else:
                radiation = RRTMGPRadiation(
                    datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
                    latitude, longitude, longwave=True,
                    shortwave=cfg.ra_sw_physics == 4)
            options["radiation"] = radiation
        elif cfg.ra_lw_physics == 0 and cfg.ra_sw_physics == 1:
            from woof.core.dudhia import DudhiaShortwaveRadiation
            latitude = np.full((cfg.ny, cfg.nx), 35., np.float32)
            longitude = np.full_like(latitude, -100.)
            if gates_only:
                radiation = object.__new__(DudhiaShortwaveRadiation)
                radiation.latitude_deg = latitude
                radiation.longitude_deg = longitude
                radiation.swrad_scat = 1.0
                radiation.icloud = int(cfg.icloud)
            else:
                radiation = DudhiaShortwaveRadiation(
                    datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
                    latitude, longitude, icloud=int(cfg.icloud))
            options["radiation"] = radiation
        elif cfg.ra_lw_physics == 1 and cfg.ra_sw_physics == 1:
            from woof.core.rrtm_lw import (
                RRTMDudhiaRadiation, RRTMLongwaveRadiation)
            from woof.core.dudhia import DudhiaShortwaveRadiation
            latitude = np.full((cfg.ny, cfg.nx), 35., np.float32)
            longitude = np.full_like(latitude, -100.)
            if gates_only:
                radiation = object.__new__(RRTMDudhiaRadiation)
                radiation.latitude_deg = latitude
                radiation.longitude_deg = longitude
                radiation.icloud = int(cfg.icloud)
                radiation.longwave_adapter = object.__new__(RRTMLongwaveRadiation)
                radiation.shortwave_adapter = object.__new__(DudhiaShortwaveRadiation)
            else:
                radiation = RRTMDudhiaRadiation(
                    datetime(2026, 1, 1, 12, tzinfo=timezone.utc),
                    latitude, longitude, icloud=int(cfg.icloud))
            options["radiation"] = radiation
        if cfg.cu_physics == 1:
            from woof.core.kf import KainFritsch
            options["cumulus"] = object.__new__(KainFritsch) if gates_only else KainFritsch()
        elif cfg.cu_physics == 3:
            from woof.core.gf import GrellFreitas
            options["cumulus"] = (object.__new__(GrellFreitas)
                                  if gates_only else GrellFreitas())
        elif cfg.cu_physics == 16 and gates_only:
            from woof.core.ntiedtke import NewTiedtke
            options["cumulus"] = object.__new__(NewTiedtke)
        if cfg.sf_surface_physics == 4:
            # Noah-MP requires explicit solar geometry even in this bounded
            # synthetic combo run. These idealized coordinates are the same
            # style of caller-owned geometry used by the radiation rows.
            latitude = np.full((cfg.ny, cfg.nx), 35., np.float32)
            longitude = np.full_like(latitude, -100.)
            options.update(
                noahmp_start_time=datetime(
                    2026, 1, 1, 12, tzinfo=timezone.utc),
                noahmp_latitude=latitude,
                noahmp_longitude=longitude)
            if noahmp_glacier_smoke:
                from woof.core.noahmp_runtime import NoahmpRuntimeParameters
                identity = NoahmpRuntimeParameters().land_use
                ivgtyp = np.full((cfg.ny, cfg.nx), 10, dtype=np.int32)
                ivgtyp[0, 0] = identity.isice
                options["ivgtyp"] = ivgtyp
        if cfg.sf_lake_physics:
            # Exercise an active, explicitly masked lake column on the RUC
            # Metal path, including its bathymetry and latitude inputs.
            latitude = np.full((cfg.ny, cfg.nx), 45., np.float32)
            options.update(
                lakemask=np.ones((cfg.ny, cfg.nx), np.float32),
                lake_depth=np.full((cfg.ny, cfg.nx), 50., np.float32),
                lake_iswater=16, radiation_latitude=latitude)
        # Explicit idealized constant-sky forcing, never a real-data claim.
        initialize_physics(state, cfg, glw=300., swdown=0., **options)


def execute(combo, *, urban=False, turbulence=None, sfclay_option=None,
            terrain_opt=None, module=None, steps=2, mp_zero_out=0,
            mp_zero_out_all=0, sf_lake_physics=0, alb_sol=0, spp_lsm=0,
            spp_conv=0,
            h_sca_adv_order=2,
            bl_mynn_version="wrf_461",
            thompson_version="wrf_461", noahmp_glacier_smoke=False,
            fields=None):
    """Run one row; ``fields``, when a dict, receives wind and theta ranges."""
    import numpy as np
    from woof.core.grid import make_vertical_coord, make_base_state
    from woof.core.state import init_at_rest
    from woof.core.dycore import run_steps
    from woof.metal.array import MetalArray, MetalArrayModule
    cfg = configuration(combo, urban=urban, turbulence=turbulence,
                        sfclay_option=sfclay_option,
                        terrain_opt=terrain_opt,
                        mp_zero_out=mp_zero_out,
                        mp_zero_out_all=mp_zero_out_all,
                        sf_lake_physics=sf_lake_physics,
                        alb_sol=alb_sol, spp_lsm=spp_lsm,
                        spp_conv=spp_conv,
                        h_sca_adv_order=h_sca_adv_order,
                        bl_mynn_version=bl_mynn_version,
                        thompson_version=thompson_version)
    coord = make_vertical_coord(cfg.nz)
    terrain_z = (np.zeros((cfg.ny, cfg.nx), dtype=np.float64)
                 if cfg.terrain_opt == 1 else None)
    base = make_base_state(coord, lambda z: 300. + .003 * np.asarray(z),
                           cfg.p_surf, cfg.ztop, terrain_z=terrain_z)
    owns_module = module is None
    if module is None:
        module = MetalArrayModule()
    try:
        state = init_at_rest(cfg, coord, base, array_module=module)
        if state.qv is not None:
            state.qv[...] = np.float32(.005)
        if cfg.specified:
            from woof.ingest.lateral_bc import (
                domain_boundary_snapshot, build_lateral_boundaries,
                attach_lateral_boundaries)
            # Constant idealized forcing derived from the initialized state;
            # coupling uses the real Metal public boundary path.
            snapshot = domain_boundary_snapshot(state)
            attach_lateral_boundaries(state, build_lateral_boundaries(
                [snapshot, snapshot], [0., 10.], spec_bdy_width=3,
                spec_zone=1, relax_zone=2))
        initialize(state, cfg, noahmp_glacier_smoke=noahmp_glacier_smoke)
        if cfg.spp_conv:
            pattern = np.empty((4, cfg.ny, cfg.nx), dtype=np.float32)
            pattern[...] = np.asarray((0.1, -0.1, 0.2, -0.2),
                                      dtype=np.float32)[:, None, None]
            state.physics.bind_spp_patterns({
                "conv": module.asarray(pattern, dtype=np.float32)})
        if cfg.spp_lsm:
            pattern = module.asarray(
                np.zeros((cfg.num_soil_layers, cfg.ny, cfg.nx),
                         dtype=np.float32), dtype=np.float32)
            state.physics.bind_spp_patterns({"lsm": pattern})
        if cfg.spp_pbl:
            # A fixed, column-varying perturbation stands in for the
            # CuPy-only pattern generator.
            column = np.linspace(-0.3, 0.3, cfg.ny * cfg.nx, dtype=np.float32)
            pattern = np.broadcast_to(column.reshape(cfg.ny, cfg.nx),
                                      (cfg.nz, cfg.ny, cfg.nx))
            state.physics.bind_spp_patterns({
                "pbl": module.asarray(np.ascontiguousarray(pattern), dtype=np.float32)})
        run_steps(state, cfg, steps)
        arrays = list(vars(state).items())
        driver = getattr(state, "physics", None)
        if driver is not None:
            arrays += list(getattr(driver, "fields", {}).items())
        for name, value in arrays:
            if isinstance(value, MetalArray) and np.issubdtype(value.dtype, np.inexact):
                if not np.isfinite(value.to_numpy()).all():
                    return "NaN", f"nonfinite {name}"
        if (cfg.bl_mynn_version == "gsd_41"
                and any(name == "cupy" or name.startswith("cupy.")
                        for name in sys.modules)):
            raise AssertionError("MYNN GSD 4.1 Metal route imported CuPy")
        if fields is not None:
            u, v = state.u.to_numpy(), state.v.to_numpy()
            thb = np.asarray(base.thb, dtype=np.float64)
            if thb.ndim == 1:
                thb = thb[:, None, None]
            theta = thb + state.thp.to_numpy()
            fields.update(
                max_wind_m_s=float(max(np.abs(u).max(), np.abs(v).max())),
                theta_min_k=float(theta.min()),
                theta_max_k=float(theta.max()))
        return "ok", f"{steps} public run_steps; finite state"
    finally:
        if owns_module:
            module.context.close()


def suite_rows():
    """Small set of recognizable regional forecast physics stacks."""
    return [
        ("CONUS Thompson MYNN Noah-MP Tiedtke RRTMGP",
         (8, 5, 4, 16, 4, "specified"), {"sfclay_option": 5}),
        ("Convection-permitting no-CU",
         (8, 5, 4, 0, 1, "periodic"), {"sfclay_option": 5}),
        ("Tropical KF",
         (8, 5, 4, 1, 4, "periodic"), {"sfclay_option": 5}),
        ("MYJ Eta",
         (8, 2, 0, 0, 1, "periodic"), {"sfclay_option": 2}),
        ("NSSL18",
         (18, 5, 4, 0, 1, "periodic"), {"sfclay_option": 5}),
        ("P3",
         (50, 5, 4, 0, 1, "periodic"), {"sfclay_option": 5}),
        ("RUC nine soil levels",
         (0, 0, 3, 0, 0, "periodic"), {"sfclay_option": 91}),
        ("YSU Noah Morrison KF RRTMGP",
         (10, 1, 2, 1, 4, "specified"), {"sfclay_option": 91}),
        ("Grell-Freitas Dudhia",
         (8, 5, 4, 3, 1, "periodic"), {"sfclay_option": 91}),
        ("Shin-Hong SASE",
         (0, 9, 0, 0, 0, "periodic"), {"sfclay_option": 1}),
        ("MYNN GSD 4.1 | mixlength 2 | EDMF plume",
         (8, 5, 4, 0, 0, "periodic"),
         {"sfclay_option": 5, "bl_mynn_version": "gsd_41"}),
    ]
