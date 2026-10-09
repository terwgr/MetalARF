from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import math
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from conftest import requires_engine_module, requires_netcdf_writer  # noqa: E402

from woof.globe.config import load_config
from woof.globe.constants import (
    NATIVE_PHYSICS_ACKNOWLEDGEMENT,
    NUMBER_MOMENTS,
    WATER_SPECIES,
)
from woof.globe.physics.native_batch import NativeColumnBatch
from woof.globe.physics.native_suite import ArwenCudaColumnSuite
from woof.globe.runner import build_model_and_cold_state


CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
T255_CONFIG = str(_shipped_configs() / "arwen_global_gdas_t255_native_24h.toml")
SFCLAY_CHS = 0.0123
SFCLAY_CHS2 = 0.0456
SFCLAY_CQS2 = 0.0789
SFCLAY_QGH = 0.0111
SFCLAY_U10 = 3.25
SFCLAY_T2_OFFSET = -1.5
FAKE_OLR = 236.0
FAKE_GSW = 132.0
FAKE_GLW = 300.0


def _options():
    return {
        "acknowledgement": NATIVE_PHYSICS_ACKNOWLEDGEMENT,
        "start_time_utc": "2024-05-21T00:00:00Z",
        "radiation_interval_s": 10.0,
        "land_surface_interval_s": 10.0,
    }


def _fake_modules(*, fail_morrison=False, precipitation=None, noah_calls=None,
                  cumulus_calls=None, cumulus=None, radiation_heating=0.0):
    """Fakes shaped like the real modules the runtime drives.

    The surface-layer fake carries every SFClayResult field the runtime
    consumes, so a coefficient the real sfclay computes cannot be missing here
    and substituted in the runtime without this fixture noticing.

    The cumulus fakes are shaped like ``woof.globe.core.gf.GrellFreitas`` and
    ``woof.globe.core.ntiedtke.NewTiedtke``: a ``column_chunk`` keyword
    constructor, the ``bind_driver`` hook the regional driver calls, and
    ``__call__(atmosphere=, fields=, state=, cfg=)`` returning the
    CumulusResult fields (the New Tiedtke double also returns the
    momentum pair, zero unless a test fills it, because the real scheme
    always produces one).  ``cumulus_calls`` records every input the
    runtime handed either; ``cumulus(result, atmosphere)`` fills the rates
    and RAINCV a test wants applied (zero rates and no rain otherwise).
    """
    class Radiation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, *, atmosphere, fields, state, cfg):
            shape = atmosphere["temperature"].shape
            surface = shape[1:]
            return SimpleNamespace(
                rthratenlw=np.full(shape, radiation_heating, np.float32),
                rthratensw=np.zeros(shape, np.float32),
                swdown=np.zeros(surface, np.float32),
                glw=np.full(surface, FAKE_GLW, np.float32),
                olr=np.full(surface, FAKE_OLR, np.float32),
                gsw=np.full(surface, FAKE_GSW, np.float32),
                coszen=np.zeros(surface, np.float32),
            )

    def sfclay(*args, **kwargs):
        shape = args[0].shape
        zero = np.zeros(shape, np.float32)
        one = np.ones(shape, np.float32)
        # Screen-level diagnostics shaped like sfclay's: t2/th2/q2 sit a
        # recognisable offset from the lowest-level inputs (args 2 and 3)
        # so a persisted value can be told from the seed.
        temperature_lowest = np.asarray(args[2], np.float32)
        qv_lowest = np.asarray(args[3], np.float32)
        return SimpleNamespace(
            znt=np.full(shape, 0.05, np.float32),
            ust=np.full(shape, 0.1, np.float32),
            mol=zero.copy(), hfx=zero.copy(), qfx=zero.copy(),
            qsfc=zero.copy(), zol=zero.copy(), wspd=np.full(shape, 0.1, np.float32),
            br=zero.copy(), fm=one.copy(), fh=one.copy(),
            u10=np.full(shape, SFCLAY_U10, np.float32), v10=zero.copy(),
            t2=temperature_lowest + np.float32(SFCLAY_T2_OFFSET),
            th2=temperature_lowest + np.float32(2.0 * SFCLAY_T2_OFFSET),
            q2=qv_lowest * np.float32(0.5),
            pblh=kwargs.get("pblh", np.full(shape, 800.0, np.float32)),
            chs=np.full(shape, SFCLAY_CHS, np.float32),
            chs2=np.full(shape, SFCLAY_CHS2, np.float32),
            cqs2=np.full(shape, SFCLAY_CQS2, np.float32),
            qgh=np.full(shape, SFCLAY_QGH, np.float32),
        )

    # The real launcher's driver field list, so the fake receives every 2-D
    # field the runtime hands the kernel -- including xland/xice.  A
    # hand-kept subset here once omitted the land/sea mask entirely, so the
    # kernel's first-call open-water branch was unreachable from this file
    # while the device run refused its first step.
    from woof.globe.core import noah as kernel_noah
    noah_fields = kernel_noah._F2D

    def launch_noah(dev, params, dt, thickness, **kwargs):
        if noah_calls is not None:
            noah_calls.append({
                "dt": float(dt),
                **{name: np.array(dev[name], copy=True)
                   for name in (*noah_fields, "ivgtyp", "isltyp")},
            })

    def launch_ysu(u, v, theta, qv, qc, qi, *args, **kwargs):
        zeros = np.zeros_like(theta)
        return {
            "du": zeros.copy(), "dv": zeros.copy(), "dtheta": zeros.copy(),
            "dqv": zeros.copy(), "dqc": zeros.copy(), "dqi": zeros.copy(),
            "hpbl": np.full(theta.shape[1:], 800.0, np.float32),
        }

    def launch_morrison(
        theta, qv, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
        rho, exner, p_full, dz,
        rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr, dt,
        **kwargs,
    ):
        theta[0, 0, 0] += np.float32(0.125)
        if precipitation is not None:
            total, frozen_snow, frozen_graupel = precipitation
            # WRF convention, which the Morrison kernel follows: rainncv is
            # the total and snowncv/graupelncv are parts of that total.
            rainncv[...] = np.float32(total)
            snowncv[...] = np.float32(frozen_snow)
            graupelncv[...] = np.float32(frozen_graupel)
            sr[...] = np.float32((frozen_snow + frozen_graupel) / max(total, 1.0e-12))
        if fail_morrison:
            raise RuntimeError("synthetic Morrison failure")

    class GrellFreitas:
        momentum = False
        #: Every construction's keyword arguments, in order.
        constructed = []

        def __init__(self, *, column_chunk=None,
                     updraft_only_when_downdraft_dry=False,
                     resolved_convergence_closure=False):
            self.driver = None
            self.column_chunk = column_chunk
            self.updraft_only_when_downdraft_dry = updraft_only_when_downdraft_dry
            self.resolved_convergence_closure = resolved_convergence_closure
            type(self).constructed.append({
                "column_chunk": column_chunk,
                "updraft_only_when_downdraft_dry": updraft_only_when_downdraft_dry,
                "resolved_convergence_closure": resolved_convergence_closure,
            })

        def bind_driver(self, driver):
            self.driver = driver

        def release(self):
            self.driver = None
            return 0

        def __call__(self, *, atmosphere, fields, state, cfg):
            shape = atmosphere["temperature"].shape
            surface = shape[1:]
            if cumulus_calls is not None:
                lanes = (
                    "rthratenlw", "rthratensw", "gf_rthblten", "gf_rqvblten",
                    "gf_rthdynten", "gf_rqvdynten", "gf_dx_column",
                )
                cumulus_calls.append({
                    "atmosphere": {
                        name: np.array(value, copy=True)
                        for name, value in atmosphere.items()
                    },
                    "fields": {
                        name: np.array(value, copy=True)
                        for name, value in fields.items()
                    },
                    "w": np.array(state.w, copy=True),
                    "ht": np.array(state.ht, copy=True),
                    "qi": None if state.qi is None else np.array(state.qi, copy=True),
                    "cfg": cfg,
                    "driver": {
                        name: (
                            None if getattr(self.driver, name) is None
                            else np.array(getattr(self.driver, name), copy=True)
                        )
                        for name in lanes
                    },
                })
            zeros = np.zeros(shape, np.float32)
            result = SimpleNamespace(
                rthcuten=zeros.copy(), rqvcuten=zeros.copy(),
                rqccuten=zeros.copy(), rqicuten=zeros.copy(),
                rainc=np.zeros(surface, np.float32),
            )
            if self.momentum:
                result.rucuten = zeros.copy()
                result.rvcuten = zeros.copy()
            if cumulus is not None:
                cumulus(result, atmosphere)
            return result

    class NewTiedtke(GrellFreitas):
        momentum = True

    return {
        "woof.globe.core.rrtmgp": SimpleNamespace(RRTMGPRadiation=Radiation),
        "woof.globe.core.sfclay": SimpleNamespace(sfclay=sfclay),
        # The real table loader and packer (CPU parsing of the shipped
        # VEGPARM/SOILPARM/GENPARM): the runtime checks the surface state's
        # category convention against the sections it loads, so the fake
        # must hand it the sections' own names and sizes.
        "woof.globe.core.noah": SimpleNamespace(
            _F2D=noah_fields,
            load_tables=kernel_noah.load_tables,
            pack_params=kernel_noah.pack_params,
            launch_noah=launch_noah,
        ),
        "woof.globe.core.ysu": SimpleNamespace(launch_ysu=launch_ysu),
        "woof.globe.core.gf": SimpleNamespace(GrellFreitas=GrellFreitas),
        "woof.globe.core.ntiedtke": SimpleNamespace(NewTiedtke=NewTiedtke),
        "woof.globe.core.morrison": SimpleNamespace(launch_morrison=launch_morrison),
    }


def _exchange():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    return model._physics_exchange(state, 5.0)


def _noah_first_call_fill_modules(branch_columns, *, fill_side="water"):
    """Fakes whose Noah performs the kernel's first-call diagnostic fill.

    ``fill_side="water"`` reproduces woof/core/kernels/noah.cu:940-948
    verbatim: on itimestep==1 every xland>=1.5 column takes smois=1,
    tslb=273.16, smcrel=1, and the kernel then returns before SFLX
    (noah.cu:968), so those columns are never integrated -- the fill is a
    diagnostic constant, not a flux.  Every other tendency in these fakes is
    zero, so any water-ledger movement the suite's closure sees comes from
    that fill alone.  ``fill_side="land"`` stamps the same values on the
    columns Noah DOES integrate instead, which is genuine created water the
    closure must keep refusing.
    """
    modules = _fake_modules()
    stock = modules["woof.globe.core.noah"]

    def launch_noah(dev, params, dt, dzs, *, itimestep, **kwargs):
        if itimestep != 1:
            return
        water = (np.asarray(dev["xland"]) - 1.5) >= 0.0
        mask = water if fill_side == "water" else ~water
        branch_columns.append(int(mask.sum()))
        dev["smstav"][mask] = 1.0
        dev["smstot"][mask] = 1.0
        dev["smois"][:, mask] = 1.0
        dev["tslb"][:, mask] = 273.16
        dev["smcrel"][:, mask] = 1.0

    modules["woof.globe.core.noah"] = SimpleNamespace(
        _F2D=stock._F2D, load_tables=stock.load_tables,
        pack_params=stock.pack_params, launch_noah=launch_noah,
    )
    return modules


def test_native_batch_reverses_vertical_order_and_uses_contiguous_fp32():
    exchange = _exchange()
    batch = NativeColumnBatch.from_exchange(exchange, np)
    assert batch.arrays["theta"].dtype == np.float32
    assert batch.arrays["theta"].flags.c_contiguous
    assert np.array_equal(batch.arrays["theta"], np.asarray(exchange.theta, np.float32)[::-1])
    assert np.array_equal(batch.arrays["p_half"], np.asarray(exchange.p_half, np.float32)[::-1])


def test_native_suite_is_transactional_and_persists_owned_state():
    exchange = _exchange()
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules()
    )
    original = {
        name: np.array(getattr(exchange, name), copy=True)
        for name in ("theta", *WATER_SPECIES, *NUMBER_MOMENTS)
    }
    result = suite.step(exchange)
    for name, value in original.items():
        assert np.array_equal(getattr(exchange, name), value), name
    assert result.physics_state.metadata["native_calls"] == 1
    assert "noah_smois" in result.physics_state.arrays
    assert result.theta[-1, 0, 0] == pytest.approx(
        float(original["theta"][-1, 0, 0] + 0.125), abs=2.0e-5
    )

    second = replace(
        exchange,
        time_s=10.0,
        theta=result.theta,
        qv=result.qv, qc=result.qc, qr=result.qr,
        qi=result.qi, qs=result.qs, qg=result.qg,
        nc=result.nc, nr=result.nr, ni=result.ni, ns=result.ns, ng=result.ng,
        surface=result.surface,
        physics_state=result.physics_state,
    )
    second_result = suite.step(second)
    assert second_result.physics_state.metadata["native_calls"] == 2


def test_native_suite_failure_cannot_mutate_caller_exchange():
    exchange = _exchange()
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules(fail_morrison=True)
    )
    theta = np.array(exchange.theta, copy=True)
    surface = np.array(exchange.surface.water_kg_m2, copy=True)
    with pytest.raises(RuntimeError, match="synthetic Morrison failure"):
        suite.step(exchange)
    assert np.array_equal(exchange.theta, theta)
    assert np.array_equal(exchange.surface.water_kg_m2, surface)


def _advance(exchange, result, time_s):
    return replace(
        exchange,
        time_s=time_s,
        u=result.u, v=result.v, theta=result.theta,
        qv=result.qv, qc=result.qc, qr=result.qr,
        qi=result.qi, qs=result.qs, qg=result.qg,
        nc=result.nc, nr=result.nr, ni=result.ni, ns=result.ns, ng=result.ng,
        surface=result.surface,
        physics_state=result.physics_state,
    )


def test_noah_is_driven_with_the_surface_layer_exchange_coefficients():
    exchange = _exchange()
    calls = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules(noah_calls=calls)
    )
    suite.step(exchange)
    assert len(calls) == 1
    first = calls[0]
    assert np.allclose(first["chs"], SFCLAY_CHS)
    assert np.allclose(first["chs2"], SFCLAY_CHS2)
    assert np.allclose(first["cqs2"], SFCLAY_CQS2)
    assert np.allclose(first["qgh"], SFCLAY_QGH)


def _seed_planet(exchange):
    """A per-row planet on the smoke exchange's land columns (the water
    columns keep the MODIS water class 17 and STAS water soil 14 the
    runtime's land/water test requires there)."""
    from woof.globe.statics import water_columns

    shape = exchange.surface.land_fraction.shape
    water = water_columns(np.asarray(exchange.surface.land_fraction))
    rows = np.arange(shape[0], dtype=np.float64)[:, None] * np.ones((1, shape[1]))
    seeded = {
        "landuse_category": np.where(water, 17.0, 1.0 + (rows % 14.0)),
        "soil_category_top": np.where(water, 14.0, 2.0 + (rows % 9.0)),
        "vegetation_fraction": 0.05 + 0.01 * rows,
        "vegetation_fraction_min": 0.02 + 0.001 * rows,
        "vegetation_fraction_max": 0.60 + 0.002 * rows,
        "leaf_area_index": 0.5 + 0.1 * rows,
        "background_albedo": 0.11 + 0.001 * rows,
        "snow_albedo": 0.40 + 0.01 * rows,
        "deep_soil_temperature_k": 270.0 + rows,
        "roughness_m": np.where(water, 1.0e-4, 0.05 + 0.01 * rows),
        "albedo": 0.10 + 0.002 * rows,
    }
    for name, value in seeded.items():
        getattr(exchange.surface, name)[...] = value
    return seeded, water


def test_noah_is_driven_with_the_surface_state_statics():
    """The categories, vegetation, LAI, snow albedo and deep-soil
    temperature Noah receives are the surface state's static fields
    (real WPS_GEOG statics or the declared synthetic planet), not
    constants the runtime invents."""
    exchange = _exchange()
    seeded, water = _seed_planet(exchange)
    assert water.any() and (~water).any()
    calls = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules(noah_calls=calls)
    )
    suite.step(exchange)
    (first,) = calls
    assert first["ivgtyp"].dtype == np.int32 and first["isltyp"].dtype == np.int32
    assert np.array_equal(first["ivgtyp"], seeded["landuse_category"].astype(np.int32))
    assert np.array_equal(first["isltyp"], seeded["soil_category_top"].astype(np.int32))
    assert np.allclose(first["vegfra"], 100.0 * seeded["vegetation_fraction"], atol=1e-4)
    assert np.allclose(first["shdmin"], 100.0 * seeded["vegetation_fraction_min"], atol=1e-4)
    assert np.allclose(first["shdmax"], 100.0 * seeded["vegetation_fraction_max"], atol=1e-4)
    assert np.allclose(first["lai"], seeded["leaf_area_index"], atol=1e-5)
    assert np.allclose(first["albbck"], seeded["background_albedo"], atol=1e-6)
    assert np.allclose(first["albedo"], seeded["albedo"], atol=1e-6)
    assert np.allclose(first["snoalb"], seeded["snow_albedo"], atol=1e-6)
    assert np.allclose(first["tmn"], seeded["deep_soil_temperature_k"], atol=1e-3)
    # Land roughness is the statics' until Noah sets its own; water
    # columns carry the surface layer's value from this call (NB-4).
    assert np.allclose(first["znt"][~water], seeded["roughness_m"][~water], atol=1e-7)
    assert np.allclose(first["z0"][~water], seeded["roughness_m"][~water], atol=1e-7)
    # The kernel's land/water flag is the same rule the categories were
    # seeded with: water class exactly on xland >= 1.5.
    assert np.array_equal(first["xland"] >= 1.5, water)
    # The synthetic planet seeds no sea ice: the kernel's xice is the zero
    # plane the surface state materializes (state.SurfaceState), not a
    # hidden constant; an analysis cold start hands the analysed fraction
    # (test_arwen_global_surface_seeding).
    assert np.all(first["xice"] == 0.0)
    assert first["xice"].shape == first["xland"].shape


def _synthetic_planet_arrays(exchange):
    """The former constant planet's kernel inputs, formed from the
    exchange exactly as the pre-statics runtime formed them (float32 of
    the float64 formulas)."""
    lf = np.asarray(exchange.surface.land_fraction, np.float64)
    f32 = lambda value: np.asarray(value, np.float32)  # noqa: E731
    return {
        "vegfra": f32(100.0 * lf), "shdmin": f32(0.0 * lf),
        "shdmax": f32(100.0 + 0.0 * lf), "lai": f32(3.0 + 0.0 * lf),
        "snoalb": f32(0.6 + 0.0 * lf), "albbck": f32(0.08 + 0.12 * lf),
        "albedo": f32(0.08 + 0.12 * lf), "znt": f32(1.0e-4 + 0.08 * lf),
        "tmn": f32(np.asarray(exchange.surface.soil_temperature_k, np.float64)[-1]),
        "xice": f32(0.0 * lf), "embck": f32(0.96 + 0.0 * lf),
    }


def test_synthetic_statics_reach_noah_as_the_former_constants_bit_for_bit():
    """The smoke config declares no [statics] table, so the analytic
    planet runs the synthetic arm: every kernel input Noah integrates is
    the pre-2026-09-01 constant (category 7, soil 8, vegfra = 100 *
    land fraction, LAI 3, snow albedo 0.6, tmn = bottom soil layer,
    albedo and roughness from the land fraction) to the bit, and the
    water columns -- which the kernel never reads a category from --
    carry the MODIS water class."""
    from woof.globe.statics import SURFACE_STATICS_METADATA_KEY

    exchange = _exchange()
    row = exchange.physics_state.metadata[SURFACE_STATICS_METADATA_KEY]
    assert row["source"] == "synthetic"
    assert row["landuse_dataset"] == "MODIFIED_IGBP_MODIS_NOAH"
    calls = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules(noah_calls=calls)
    )
    result = suite.step(exchange)
    (first,) = calls
    water = first["xland"] >= 1.5
    assert water.any() and (~water).any()
    assert np.all(first["ivgtyp"][~water] == 7) and np.all(first["isltyp"][~water] == 8)
    assert np.all(first["ivgtyp"][water] == 17) and np.all(first["isltyp"][water] == 14)
    for name, expected in _synthetic_planet_arrays(exchange).items():
        assert first[name].dtype == np.float32, name
        if name == "znt":
            # water columns carry this call's surface-layer roughness (NB-4)
            assert np.array_equal(first[name][~water], expected[~water]), name
        else:
            assert np.array_equal(first[name], expected), name
    assert result.physics_state.metadata[SURFACE_STATICS_METADATA_KEY] == row


def _refused_step(exchange, match, *, modules=None, **overrides):
    suite = ArwenCudaColumnSuite(
        {**_options(), **overrides}, array_module=np,
        modules=_fake_modules() if modules is None else modules,
    )
    with pytest.raises(ValueError, match=match):
        suite.step(exchange)


def test_statics_the_tables_cannot_index_are_refused_by_name():
    """noah.cu indexes VEGPARM/SOILPARM with the category and no bounds
    check, skips vegtyp == isice and urbanises vegtyp == isurban: a
    state from another dataset, a category outside the loaded section,
    an option from another dataset, a land column carrying the water
    class or a water column carrying a land class is refused before the
    kernel reads it."""
    from woof.globe.statics import SURFACE_STATICS_METADATA_KEY

    exchange = _exchange()
    _seed_planet(exchange)
    row = exchange.physics_state.metadata[SURFACE_STATICS_METADATA_KEY]
    water = np.asarray(exchange.surface.landuse_category) == 17.0
    land_index = tuple(int(v) for v in np.argwhere(~water)[0])
    water_index = tuple(int(v) for v in np.argwhere(water)[0])

    # no convention row at all
    bare = replace(exchange, physics_state=type(exchange.physics_state)())
    _refused_step(bare, "no physics_state.metadata\\['surface_statics'\\]")
    # a convention row naming a dataset the loader does not honour (a
    # loader that ignores the requested section, here by construction)
    from woof.globe.core import noah as kernel_noah

    other = exchange.physics_state.copy()
    other.metadata[SURFACE_STATICS_METADATA_KEY] = {**row, "landuse_dataset": "USGS"}
    deaf = _fake_modules()
    stock = deaf["woof.globe.core.noah"]
    deaf["woof.globe.core.noah"] = SimpleNamespace(
        _F2D=stock._F2D, pack_params=stock.pack_params,
        launch_noah=stock.launch_noah,
        load_tables=lambda **kwargs: kernel_noah.load_tables(),
    )
    _refused_step(replace(exchange, physics_state=other),
                  "index the USGS/STAS table sections but Noah loaded "
                  "MODIFIED_IGBP_MODIS_NOAH/STAS", modules=deaf)
    # a dataset VEGPARM.TBL has no section for
    other = exchange.physics_state.copy()
    other.metadata[SURFACE_STATICS_METADATA_KEY] = {**row, "landuse_dataset": "NOSUCH"}
    _refused_step(replace(exchange, physics_state=other),
                  "landuse dataset 'NOSUCH' not found in VEGPARM.TBL")
    # adapter options from another dataset (USGS isice 24 / isurban 1)
    _refused_step(exchange, "land_ice_category=24 is not the MODIFIED_IGBP_MODIS_NOAH value 15",
                  land_ice_category=24)
    _refused_step(exchange, "urban_category=1 is not the MODIFIED_IGBP_MODIS_NOAH value 13",
                  urban_category=1)
    # a land column carrying the water class
    mixed = replace(exchange, surface=exchange.surface.copy())
    mixed.surface.landuse_category[land_index] = 17.0
    _refused_step(mixed, "1 land column\\(s\\) \\(land_fraction > 0.5\\) carry the water category 17")
    # a water column carrying a land class
    mixed = replace(exchange, surface=exchange.surface.copy())
    mixed.surface.landuse_category[water_index] = 5.0
    _refused_step(mixed, "1 water column\\(s\\) carry a land class")
    # categories outside the loaded sections (VEGPARM MODIS has 20 rows,
    # SOILPARM STAS 19)
    mixed = replace(exchange, surface=exchange.surface.copy())
    mixed.surface.landuse_category[land_index] = 21.0
    _refused_step(mixed, "1 land column\\(s\\) carry a land-use category outside VEGPARM MODIFIED_IGBP_MODIS_NOAH 1..20")
    mixed = replace(exchange, surface=exchange.surface.copy())
    mixed.surface.soil_category_top[land_index] = 20.0
    _refused_step(mixed, "1 a soil category outside SOILPARM STAS 1..19")


def _roughness_recording_modules(sfclay_znt, ysu_znt, *, noah_land_znt):
    """Fakes recording the roughness sfclay is handed (positional arg 8)
    and YSU is handed (keyword znt); Noah stamps ``noah_land_znt`` on the
    columns it integrates, as the kernel writes znt_a (noah.cu:1486)."""
    modules = _fake_modules()
    stock_sfclay = modules["woof.globe.core.sfclay"].sfclay
    stock_ysu = modules["woof.globe.core.ysu"].launch_ysu
    stock_noah = modules["woof.globe.core.noah"]

    def sfclay(*args, **kwargs):
        sfclay_znt.append(np.array(args[8], np.float32, copy=True))
        out = stock_sfclay(*args, **kwargs)
        # WRF inout: unchanged on land, the fake's constant over water.
        xland = np.asarray(args[11])
        out.znt = np.where(xland >= 1.5, out.znt, np.asarray(args[8], np.float32))
        return out

    def launch_ysu(*args, **kwargs):
        ysu_znt.append(np.array(kwargs["znt"], np.float32, copy=True))
        return stock_ysu(*args, **kwargs)

    def launch_noah(dev, params, dt, thickness, **kwargs):
        land = np.asarray(dev["xland"]) < 1.5
        dev["znt"][land] = np.float32(noah_land_znt)

    modules["woof.globe.core.sfclay"] = SimpleNamespace(sfclay=sfclay)
    modules["woof.globe.core.ysu"] = SimpleNamespace(launch_ysu=launch_ysu)
    modules["woof.globe.core.noah"] = SimpleNamespace(
        _F2D=stock_noah._F2D, load_tables=stock_noah.load_tables,
        pack_params=stock_noah.pack_params, launch_noah=launch_noah,
    )
    return modules


def test_surface_layer_and_pbl_see_the_statics_roughness_where_wrf_would():
    """WRF's surface_driver: sfclay reads ZNT (landuse_init's LANDUSE.TBL
    value before the LSM has run), the LSM overwrites it on land, and the
    PBL driver reads the overwritten value on the same step.  Here the
    first sfclay call is handed the statics' roughness on every column,
    YSU on a due land call is handed Noah's, and the next call's sfclay
    starts from Noah's on land."""
    exchange = _exchange()
    seeded, water = _seed_planet(exchange)
    statics_znt = np.asarray(seeded["roughness_m"], np.float32)
    sfclay_znt, ysu_znt = [], []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np,
        modules=_roughness_recording_modules(sfclay_znt, ysu_znt, noah_land_znt=0.123),
    )
    result = suite.step(exchange)
    assert len(sfclay_znt) == len(ysu_znt) == 1
    assert np.array_equal(sfclay_znt[0], statics_znt)
    # Noah was due: YSU sees Noah's roughness on land, sfclay's on water.
    assert np.all(ysu_znt[0][~water] == np.float32(0.123))
    assert np.all(ysu_znt[0][water] == np.float32(0.05))
    roughness = np.asarray(result.surface.roughness_m, np.float32)
    assert np.all(roughness[~water] == np.float32(0.123))
    assert np.all(roughness[water] == np.float32(0.05))
    # A call inside the same land bucket (7.5 s with a 10 s interval):
    # sfclay starts from Noah's roughness on land, the Charnock value on
    # water, and YSU sees the same.
    second = suite.step(_advance(exchange, result, 7.5))
    assert result.physics_state.metadata["last_land_bucket"] == second.physics_state.metadata["last_land_bucket"]
    assert len(sfclay_znt) == 2
    assert np.all(sfclay_znt[1][~water] == np.float32(0.123))
    assert np.all(sfclay_znt[1][water] == np.float32(0.05))
    assert np.all(ysu_znt[1][~water] == np.float32(0.123))


def test_surface_layer_result_without_exchange_coefficients_is_refused():
    exchange = _exchange()
    modules = _fake_modules()
    full = modules["woof.globe.core.sfclay"].sfclay

    def incomplete(*args, **kwargs):
        result = full(*args, **kwargs)
        for name in ("chs", "chs2", "cqs2", "qgh"):
            delattr(result, name)
        return result

    modules["woof.globe.core.sfclay"] = SimpleNamespace(sfclay=incomplete)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=modules)
    with pytest.raises(ValueError, match="chs, chs2, cqs2, qgh"):
        suite.step(exchange)


def test_land_step_receives_every_half_step_of_precipitation():
    # Two physics calls fall between the due land steps at t=0 and t=20, and
    # both of their precipitation must reach the land bucket.
    options = {**_options(), "land_surface_interval_s": 20.0}
    exchange = _exchange()
    calls = []
    suite = ArwenCudaColumnSuite(
        options, array_module=np,
        modules=_fake_modules(precipitation=(0.4, 0.1, 0.05), noah_calls=calls),
    )
    first = suite.step(exchange)
    second = suite.step(_advance(exchange, first, 10.0))
    suite.step(_advance(exchange, second, 20.0))
    assert len(calls) == 2
    # The first land call precedes any microphysics; the second consumes the
    # two calls that ran between them.
    assert np.allclose(calls[0]["rainbl"], 0.0)
    assert np.allclose(calls[1]["rainbl"], 0.8)
    assert np.allclose(calls[1]["sr"], 0.15 / 0.4)


def test_land_step_integrates_the_elapsed_time_not_the_configured_interval():
    options = {**_options(), "land_surface_interval_s": 10.0}
    exchange = _exchange()
    calls = []
    suite = ArwenCudaColumnSuite(
        options, array_module=np, modules=_fake_modules(noah_calls=calls)
    )
    first = suite.step(exchange)
    suite.step(_advance(exchange, first, 60.0))
    assert [row["dt"] for row in calls] == [10.0, 60.0]


def test_precipitation_species_buckets_stay_disjoint():
    exchange = _exchange()
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np,
        modules=_fake_modules(precipitation=(0.4, 0.1, 0.05)),
    )
    before_rain = np.array(exchange.surface.accumulated_rain_kg_m2, copy=True)
    before_snow = np.array(exchange.surface.accumulated_snow_kg_m2, copy=True)
    before_graupel = np.array(
        exchange.surface.accumulated_graupel_kg_m2, copy=True
    )
    result = suite.step(exchange)
    assert np.allclose(
        result.surface.accumulated_rain_kg_m2 - before_rain, 0.25
    )
    assert np.allclose(
        result.surface.accumulated_snow_kg_m2 - before_snow, 0.10
    )
    assert np.allclose(
        result.surface.accumulated_graupel_kg_m2 - before_graupel, 0.05
    )


def test_cold_start_soil_water_is_seeded_from_the_exchange_surface():
    exchange = _exchange()
    surface_soil = np.array(exchange.surface.soil_water_fraction, copy=True)
    surface_soil_temperature = np.array(
        exchange.surface.soil_temperature_k, copy=True
    )
    options = {**_options(), "initial_soil_water_fraction": 0.30}
    suite = ArwenCudaColumnSuite(
        options, array_module=np, modules=_fake_modules()
    )
    assert not np.allclose(surface_soil, 0.30)
    result = suite.step(exchange)
    assert np.allclose(
        result.physics_state.arrays["noah_smois"], surface_soil, atol=1.0e-6
    )
    assert np.allclose(
        result.physics_state.arrays["noah_sh2o"], surface_soil, atol=1.0e-6
    )
    assert np.allclose(
        result.physics_state.arrays["noah_tslb"],
        surface_soil_temperature,
        atol=1.0e-3,
    )


def test_step0_open_water_fill_is_not_priced_against_the_reservoir():
    # The step-0 device conviction, reproduced on the CPU against the REAL
    # global exchange path the unit fakes bypassed: the Noah kernel's first
    # call stamps smois=1/tslb=273.16 on every xland>=1.5 column and returns
    # before SFLX, so the stamp lands on columns the scheme never integrates.
    # Booked into surface.soil_water_fraction it was priced as
    # (1-smois)*2m*1000kg/m3*land_fraction of created water -- ~717 kg/m2 on
    # this grid's worst coastal column against the 500 kg/m2 reservoir -- and
    # suite.step raised 'native physics water closure exceeds the explicit
    # surface reservoir' on the first physics call of the T255 analysis run.
    # The stock fakes hid it twice: an inert launch_noah, and an _F2D
    # without xland.
    exchange = _exchange()
    lf32 = np.asarray(exchange.surface.land_fraction, np.float32)
    water_columns = (1.0 + (1.0 - lf32)) >= 1.5
    smois0 = np.asarray(exchange.surface.soil_water_fraction, np.float64)
    tslb0 = np.asarray(exchange.surface.soil_temperature_k, np.float64)
    reservoir0 = np.array(
        np.asarray(exchange.surface.water_kg_m2, np.float64), copy=True
    )
    # The scenario must contain coastal columns whose stamp out-prices the
    # reservoir, or a pass here proves nothing.
    stamp = (1.0 - smois0.mean(axis=0)) * 2000.0 * np.asarray(lf32, np.float64)
    assert int((water_columns & (stamp > reservoir0)).sum()) >= 1

    branch = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np,
        modules=_noah_first_call_fill_modules(branch),
    )
    result = suite.step(exchange)
    # The fill really ran, on every skipped column.
    assert branch == [int(water_columns.sum())]
    assert branch[0] >= 1
    # Never-integrated columns keep their seeded soil state...
    soil_after = np.asarray(result.surface.soil_water_fraction, np.float64)
    assert np.allclose(
        soil_after[:, water_columns], smois0[:, water_columns], atol=1.0e-6
    )
    soil_t_after = np.asarray(result.surface.soil_temperature_k, np.float64)
    assert np.allclose(
        soil_t_after[:, water_columns], tslb0[:, water_columns], atol=1.0e-3
    )
    # ...and with every tendency zero, nothing beyond fp32 casting noise is
    # booked against the reservoir (the defect booked hundreds of kg/m2).
    assert result.diagnostics["maximum_native_water_residual_kg_m2"] < 1.0e-3
    assert np.allclose(
        np.asarray(result.surface.water_kg_m2, np.float64),
        reservoir0, atol=1.0e-3,
    )


def test_water_closure_still_convicts_creation_on_integrated_columns():
    # Retiring the skipped-column stamp from the water ledger must not blunt
    # the gate: the same smois=1 stamp on columns Noah DID integrate is
    # genuine created water ((1-smois)*2000*land_fraction reaches ~1300
    # kg/m2 where land_fraction is ~0.96) and the finite-reservoir closure
    # keeps refusing it.
    exchange = _exchange()
    lf32 = np.asarray(exchange.surface.land_fraction, np.float32)
    land_columns = (1.0 + (1.0 - lf32)) < 1.5
    smois0 = np.asarray(exchange.surface.soil_water_fraction, np.float64)
    stamp = (1.0 - smois0.mean(axis=0)) * 2000.0 * np.asarray(lf32, np.float64)
    reservoir0 = np.asarray(exchange.surface.water_kg_m2, np.float64)
    assert int((land_columns & (stamp > reservoir0)).sum()) >= 1

    branch = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np,
        modules=_noah_first_call_fill_modules(branch, fill_side="land"),
    )
    with pytest.raises(
        FloatingPointError,
        match="water closure exceeds the explicit surface reservoir",
    ):
        suite.step(exchange)


def _noah_saturation_clip_modules(clipped_columns):
    """Fakes whose Noah carries the kernel's soil saturation clip verbatim.

    Per integrated column (xland < 1.5, xice below threshold), the fake
    performs SSTEP's clip-and-cascade against the REAL SOILPARM SMCMAX for
    the column's isltyp (noah.cu:548-564) and the driver's runoff epilogue
    (noah.cu:1450-1451, 1518-1519): the whole-stack saturation excess leaves
    the soil into udrunoff.  Every other tendency is zero, so any
    water-ledger movement the suite's closure sees comes from that move
    alone.
    """
    modules = _fake_modules()
    stock = modules["woof.globe.core.noah"]
    from woof.globe.core import noah as kernel_noah

    maxsmc = np.asarray(kernel_noah.load_tables().maxsmc, np.float32)

    def launch_noah(dev, params, dt, dzs, *, xice_threshold=0.5, **kwargs):
        xland = np.asarray(dev["xland"], np.float32)
        land = (xland - np.float32(1.5)) < 0.0
        land &= np.asarray(dev["xice"], np.float32) < np.float32(xice_threshold)
        if not land.any():
            return
        smcmax = maxsmc[
            np.asarray(dev["isltyp"], np.int64) - 1
        ].astype(np.float32)
        dzs32 = np.asarray(dzs, np.float32)
        zsoil = np.empty(len(dzs32), np.float32)
        zsoil[0] = -dzs32[0]
        for k in range(1, len(dzs32)):
            zsoil[k] = -dzs32[k] + zsoil[k - 1]
        smc = dev["smois"]
        sh2o = dev["sh2o"]
        sice = (smc - sh2o).astype(np.float32)  # noah.cu:590
        dt32 = np.float32(dt)
        wplus = np.zeros(xland.shape, np.float32)
        for k in range(len(dzs32)):
            ddz = np.float32(-zsoil[0]) if k == 0 else np.float32(
                zsoil[k - 1] - zsoil[k]
            )
            sh2oout = (sh2o[k] + wplus / ddz).astype(np.float32)
            stot = (sh2oout + sice[k]).astype(np.float32)
            wplus = np.where(
                stot > smcmax, ((stot - smcmax) * ddz).astype(np.float32),
                np.float32(0.0),
            )
            smc_new = np.maximum(
                np.minimum(stot, smcmax), np.float32(0.02)
            ).astype(np.float32)
            smc[k] = np.where(land, smc_new, smc[k])
            sh2o[k] = np.where(
                land,
                np.maximum(smc_new - sice[k], np.float32(0.0)).astype(np.float32),
                sh2o[k],
            )
        runoff3 = (wplus / dt32).astype(np.float32)  # noah.cu:564, 1450
        runoff2 = runoff3  # noah.cu:1451, drainage zero here
        dev["udrunoff"][...] = np.where(
            land,
            (dev["udrunoff"] + runoff2 * dt32 * np.float32(1000.0)
             ).astype(np.float32),
            dev["udrunoff"],
        )
        clipped_columns.append(int(land.sum()))

    modules["woof.globe.core.noah"] = SimpleNamespace(
        _F2D=stock._F2D, load_tables=stock.load_tables,
        pack_params=stock.pack_params, launch_noah=launch_noah,
    )
    return modules


def test_saturated_soil_clip_books_runoff_as_moved_not_destroyed_water():
    # The T255 qualification conviction, reproduced on the CPU real exchange
    # path: the analysis init delivers volumetric soil moisture clipped to
    # [0, 1], so ice-sheet columns enter surface.soil_water_fraction at
    # exactly 1.0 and integrate as ordinary uniform-category soil.  Noah's
    # first due call clips every layer to SMCMAX and books the whole-stack
    # excess into udrunoff (noah.cu:548-564, 1450-1451, 1518-1519) --
    # (1.0 - 0.464) * 2 m * 1000 kg/m3 = 1072 kg/m2 in one step at
    # land_fraction 1.  With the runoff stores outside the water ledger the
    # closure priced that move as destroyed water, and both six-hour
    # receipts (continuous and midpoint-restart, 2026-08-31) failed
    # physics_water_repair_max_step_kg_m2 at 1072.0001220703125 against its
    # 5e-4 limit.
    from woof.globe.water import (
        native_store_column,
        soil_water_column,
        water_outflow_column,
    )
    from woof.globe.core import noah as kernel_noah

    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    lf = np.asarray(state.surface.land_fraction, np.float32)
    integrated = (1.0 + (1.0 - lf)) < 1.5
    assert int(integrated.sum()) >= 1
    swf = state.surface.soil_water_fraction
    swf[...] = np.where((lf > 0.0)[None], np.float32(1.0), swf)
    exchange = model._physics_exchange(state, 5.0)

    reservoir0 = np.array(
        np.asarray(exchange.surface.water_kg_m2, np.float64), copy=True
    )
    ledger0 = (
        np.asarray(soil_water_column(exchange.surface, np), np.float64)
        + reservoir0
    )

    clipped = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np,
        modules=_noah_saturation_clip_modules(clipped),
    )
    result = suite.step(exchange)

    # The clip really ran on every integrated column, sheared the whole
    # stack to SMCMAX, and shed the excess into the runoff store -- a pass
    # with none of that proves nothing.
    assert clipped == [int(integrated.sum())]
    # The soil category comes from the surface state now (the smoke config
    # runs the declared synthetic planet), not from an adapter option.
    soil_category = int(np.rint(np.asarray(state.surface.soil_category_top))[integrated][0])
    smcmax = float(np.asarray(kernel_noah.load_tables().maxsmc, np.float32)[
        soil_category - 1
    ])
    assert smcmax < 0.999
    soil_after = np.asarray(result.surface.soil_water_fraction, np.float64)
    assert np.allclose(soil_after[:, integrated], smcmax, atol=1.0e-6)
    udrunoff = np.asarray(
        result.physics_state.arrays["noah_udrunoff"], np.float64
    )
    assert float(udrunoff[integrated].min()) > 1000.0

    # The move is booked as an exit, not destruction: the repair the gate
    # tracker reads stays at fp32 quantization scale (the defect measured
    # ~1072 kg/m2)...
    assert result.diagnostics["maximum_local_water_repair_kg_m2"] < 1.0e-3
    # ...nothing was drained from or dumped into the surface reservoir...
    assert np.allclose(
        np.asarray(result.surface.water_kg_m2, np.float64),
        reservoir0, atol=1.0e-3,
    )
    # ...the whole priced excess was booked to the cumulative outflow
    # account (runoff leaves the column system; the ledgers count the
    # monotone kernel accumulators at zero and the exit account instead)...
    lf64 = np.asarray(lf, np.float64)
    outflow = np.asarray(
        water_outflow_column(
            result.physics_state, result.surface.water_kg_m2, np
        ),
        np.float64,
    )
    assert np.allclose(outflow, udrunoff * lf64, atol=1.0e-3)
    assert float(outflow[integrated].min()) > 500.0
    # ...and the runner-side accounting authority agrees the water still
    # exists as held water plus booked exits: soil + reservoir + native
    # stores + outflow is unchanged column by column, so the global water
    # fixer has no loss to manufacture back.
    ledger1 = (
        np.asarray(soil_water_column(result.surface, np), np.float64)
        + np.asarray(result.surface.water_kg_m2, np.float64)
        + np.asarray(
            native_store_column(
                result.physics_state, result.surface.water_kg_m2, np
            ),
            np.float64,
        )
        + outflow
    )
    assert np.allclose(ledger1, ledger0, atol=1.0e-3)


def _specific_level(kernel_species, level):
    """One kernel-side level (dry mixing ratios, native_batch) as float64
    specific humidities, the model's metric."""
    from woof.globe.physics.native_batch import (
        specific_humidity_from_mixing_ratio,
    )

    return specific_humidity_from_mixing_ratio(
        {name: np.asarray(arr[level], np.float64) for name, arr in kernel_species.items()}
    )


def _write_specific_level(kernel_species, level, specific):
    """Write a level back to the kernel arrays from specific humidities,
    every species' specific humidity preserved exactly."""
    from woof.globe.physics.native_batch import (
        mixing_ratio_from_specific_humidity,
    )

    for name, value in mixing_ratio_from_specific_humidity(specific).items():
        kernel_species[name][level] = np.asarray(value, np.float64).astype(np.float32)


def _mass_consistent_precip_modules(dp_bottom_up, booked, leftovers):
    """Fakes for the precipitation double-entry conviction.

    Morrison removes exactly its reported precipitation total from column
    vapor in the ledger's metric -- specific humidity times dp/g, the
    model's convention, converted at the kernel-side level through the
    bridge's own helpers (the kernel's contract: fallout is water that LEFT
    the atmosphere) -- and Noah consumes rainbl into counted
    stores the way SFLX does -- the liquid part fills the canopy to the
    CMCMAX cap of 0.5 kg/m2 (noah.cu:566-568, 1479) with the excess booked
    into sfcrunoff (noah.cu:1518), and the frozen part accrues into snow
    (noah.cu:1161-1171, 1480).  Every destination is a store both water
    ledgers count; no other tendency moves water, so any closure residual
    is the booking itself.
    """
    modules = _fake_modules()
    stock = modules["woof.globe.core.noah"]
    total, frozen_snow, frozen_graupel = 0.4, 0.1, 0.05

    def launch_morrison(
        theta, qv, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
        rho, exner, p_full, dz,
        rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr, dt,
        **kwargs,
    ):
        rainncv[...] = np.float32(total)
        snowncv[...] = np.float32(frozen_snow)
        graupelncv[...] = np.float32(frozen_graupel)
        sr[...] = np.float32((frozen_snow + frozen_graupel) / total)
        g = np.float64(9.80665)
        species = {"qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg}
        remaining = np.full(qv.shape[1:], np.float64(total))
        for k in range(qv.shape[0]):
            dp_k = np.asarray(dp_bottom_up[k], np.float64)
            specific = _specific_level(species, k)
            take = np.clip(specific["qv"] * dp_k / g, 0.0, remaining)
            specific["qv"] = specific["qv"] - take * g / dp_k
            _write_specific_level(species, k, specific)
            remaining -= take
        leftovers.append(float(remaining.max()))

    def launch_noah(dev, params, dt, dzs, **kwargs):
        xland = np.asarray(dev["xland"], np.float32)
        land = (xland - np.float32(1.5)) < 0.0
        rainbl = np.asarray(dev["rainbl"], np.float32)
        sr = np.asarray(dev["sr"], np.float32)
        frozen = np.where(land, rainbl * sr, np.float32(0.0)).astype(np.float32)
        liquid = np.where(land, rainbl - frozen, np.float32(0.0)).astype(
            np.float32
        )
        cap = np.float32(0.5)  # CMCMAX * 1000
        intercepted = np.minimum(
            liquid, np.maximum(cap - dev["canwat"], np.float32(0.0))
        ).astype(np.float32)
        dev["canwat"][...] = (dev["canwat"] + intercepted).astype(np.float32)
        dev["sfcrunoff"][...] = (
            dev["sfcrunoff"] + (liquid - intercepted)
        ).astype(np.float32)
        dev["snow"][...] = (dev["snow"] + frozen).astype(np.float32)
        booked.append(float(rainbl[land].max()) if land.any() else 0.0)

    modules["woof.globe.core.noah"] = SimpleNamespace(
        _F2D=stock._F2D, load_tables=stock.load_tables,
        pack_params=stock.pack_params, launch_noah=launch_noah,
    )
    modules["woof.globe.core.morrison"] = SimpleNamespace(
        launch_morrison=launch_morrison
    )
    return modules


def test_precipitation_consumed_by_noah_is_debited_from_the_reservoir():
    # The third T255 qualification conviction (after the smois stamp and the
    # runoff store), reproduced on the CPU real exchange path: per physics
    # call the microphysics credits the FULL precipitation total to the
    # surface reservoir AND queues the same water in land_rainbl; one land
    # interval later Noah books the consumed rainbl into
    # canwat/snow/smois/sfcrunoff/udrunoff -- all counted stores -- while the
    # reservoir keeps its copy.  The closure's forced reservoir debit WAS the
    # gate's repair, so physics_water_repair_max_step_kg_m2 read the heaviest
    # one-interval land-column precipitation of the run: 1.0943603515625
    # kg/m2 (a 65.7 mm/h GDAS spin-up burst over one 60 s interval),
    # bit-identical in both six-hour qualification receipts (2026-08-31)
    # against the 5e-4 limit.  Fixed, _land_step debits the reservoir by the
    # priced store delta and the repair stays at fp32 quantization scale.
    from woof.globe.water import (
        atmospheric_water_column,
        native_store_column,
        soil_water_column,
        water_outflow_column,
    )

    exchange = _exchange()
    dp_bottom_up = np.asarray(exchange.dp, np.float32)[::-1].copy()
    lf = np.asarray(exchange.surface.land_fraction, np.float32)
    land = (1.0 + (1.0 - lf)) < 1.5
    assert int(land.sum()) >= 1 and int((~land).sum()) >= 1

    def ledger(surface, fields, physics_state):
        # The conservation total: held water plus the booked outflow exits
        # (runoff overflow past the canopy cap leaves the column system).
        atmosphere = np.asarray(
            atmospheric_water_column(fields), np.float64
        ).sum(axis=0)
        native = np.zeros_like(atmosphere)
        outflow = np.zeros_like(atmosphere)
        if physics_state is not None:
            native = np.asarray(
                native_store_column(
                    physics_state, surface.water_kg_m2, np
                ),
                np.float64,
            )
            outflow = np.asarray(
                water_outflow_column(
                    physics_state, surface.water_kg_m2, np
                ),
                np.float64,
            )
        return (
            atmosphere
            + np.asarray(surface.water_kg_m2, np.float64)
            + np.asarray(soil_water_column(surface, np), np.float64)
            + native
            + outflow
        )

    fields0 = {
        "dp": np.asarray(exchange.dp, np.float64),
        **{
            name: np.asarray(getattr(exchange, name), np.float64)
            for name in WATER_SPECIES
        },
    }
    ledger0 = ledger(exchange.surface, fields0, None)

    booked, leftovers = [], []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np,
        modules=_mass_consistent_precip_modules(dp_bottom_up, booked, leftovers),
    )
    first = suite.step(exchange)
    second = suite.step(_advance(exchange, first, 5.0))
    reservoir2 = np.array(
        np.asarray(second.surface.water_kg_m2, np.float64), copy=True
    )
    third = suite.step(_advance(exchange, second, 10.0))

    # The double-entry really happened: Morrison drained its total from the
    # atmosphere on every call, and the second land call consumed the two
    # queued totals (2 x 0.4) into canwat (0.5, the cap) + snow (0.3).
    assert leftovers == pytest.approx([0.0, 0.0, 0.0], abs=1.0e-9)
    assert booked == pytest.approx([0.0, 0.8], abs=1.0e-6)
    canwat = np.asarray(third.physics_state.arrays["noah_canwat"], np.float64)
    snow = np.asarray(third.physics_state.arrays["noah_snow"], np.float64)
    assert np.allclose(canwat[land], 0.5, atol=1.0e-6)
    assert np.allclose(snow[land], 0.3, atol=1.0e-6)

    # The repair the gate tracker reads stays at fp32 quantization scale
    # (the defect read the whole consumed interval, 0.8 here).
    assert third.diagnostics["maximum_local_water_repair_kg_m2"] < 1.0e-3

    # The reservoir paid for the booked stores: over the consuming step it
    # gains the step's own precipitation (0.4) and pays out the priced store
    # gain (0.8 -- canwat and snow count at full weight), so land columns
    # move -0.4 while never-consuming ocean columns move +0.4.
    reservoir3 = np.asarray(third.surface.water_kg_m2, np.float64)
    assert np.allclose(
        reservoir3 - reservoir2, np.where(land, -0.4, 0.4), atol=1.0e-3
    )

    # And the accounting authority agrees no water was created or destroyed:
    # atmosphere + reservoir + soil + native stores is unchanged column by
    # column across all three steps.
    fields3 = {
        "dp": fields0["dp"],
        **{
            name: np.asarray(getattr(third, name), np.float64)
            for name in WATER_SPECIES
        },
    }
    ledger3 = ledger(third.surface, fields3, third.physics_state)
    assert np.allclose(ledger3, ledger0, atol=2.0e-3)


def _surface_flux_modules(qfx_value):
    """Fakes for the unbooked surface moisture flux conviction.

    sfclay reports a uniform qfx and YSU applies the kernel's vapor bottom
    boundary verbatim (ysu.cu:617: rhs[0] = qv0 + qf*G/delp[0]*dt2), so the
    atmosphere gains exactly qfx*dt kg/m2 per call and nothing else moves
    water.
    """
    modules = _fake_modules()
    stock_sfclay = modules["woof.globe.core.sfclay"].sfclay

    def sfclay(*args, **kwargs):
        result = stock_sfclay(*args, **kwargs)
        result.qfx = np.full(result.qfx.shape, qfx_value, np.float32)
        return result

    def launch_ysu(u, v, theta, qv, qc, qi, p_full, p_half, exner, dz,
                   *args, qfx, **kwargs):
        zeros = np.zeros_like(theta)
        dqv = zeros.copy()
        dp0 = np.asarray(p_half[0] - p_half[1], np.float32)
        dqv[0] = np.asarray(qfx, np.float32) * np.float32(9.80665) / dp0
        return {
            "du": zeros.copy(), "dv": zeros.copy(), "dtheta": zeros.copy(),
            "dqv": dqv, "dqc": zeros.copy(), "dqi": zeros.copy(),
            "hpbl": np.full(theta.shape[1:], 800.0, np.float32),
        }

    modules["woof.globe.core.sfclay"] = SimpleNamespace(sfclay=sfclay)
    modules["woof.globe.core.ysu"] = SimpleNamespace(launch_ysu=launch_ysu)
    return modules


@pytest.mark.parametrize("qfx_value", [2.0e-3, -2.0e-3])
def test_surface_moisture_flux_is_booked_against_the_reservoir(qfx_value):
    # The residual floor beneath the precipitation double-entry, same
    # convicted family (2026-08-31): sfclay/Noah qfx enters the atmosphere
    # through YSU's vapor bottom boundary with no debit of reservoir or
    # soil -- measured 1.5869e-3 kg/m2 per 30 s half-call at 125 W/m2
    # latent, above the physics_water_repair_max_step_kg_m2 limit of 5e-4 on
    # essentially every moist column (dew, qfx < 0, is the same route
    # reversed).  Fixed, _pbl_step books the applied column moisture change
    # against the reservoir; the flux here is 2e-3 * 5 s = 0.01 kg/m2 per
    # call, 10x the repair bound asserted below.
    exchange = _exchange()
    dt_s = float(exchange.dt_s)
    dp = np.asarray(exchange.dp, np.float64)
    atm0 = sum(
        np.asarray(getattr(exchange, name), np.float64) for name in WATER_SPECIES
    )
    atm0 = (atm0 * dp / 9.80665).sum(axis=0)
    reservoir0 = np.array(
        np.asarray(exchange.surface.water_kg_m2, np.float64), copy=True
    )

    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_surface_flux_modules(qfx_value),
    )
    result = suite.step(exchange)

    flux = qfx_value * dt_s
    # The flux really reached the atmosphere...
    atm1 = sum(
        np.asarray(getattr(result, name), np.float64) for name in WATER_SPECIES
    )
    atm1 = (atm1 * dp / 9.80665).sum(axis=0)
    assert np.allclose(atm1 - atm0, flux, atol=1.0e-3)
    # ...the reservoir paid for it (was credited, for dew)...
    reservoir1 = np.asarray(result.surface.water_kg_m2, np.float64)
    assert np.allclose(reservoir1 - reservoir0, -flux, atol=1.0e-3)
    # ...and the repair the gate tracker reads stays at fp32 quantization
    # scale instead of |qfx| * dt.
    assert result.diagnostics["maximum_local_water_repair_kg_m2"] < 1.0e-3


def test_unenforced_water_tolerance_is_refused_instead_of_ignored():
    # The suite reports native_water_residual_exceeds_tolerance and nothing
    # consumes it, so an operator who sets the knob expecting a refusal gets
    # one here rather than a run that continues past their threshold.
    with pytest.raises(ValueError, match="measured and reported, not enforced"):
        ArwenCudaColumnSuite(
            {**_options(), "water_fix_tolerance_kg_m2": 0.0},
            array_module=np, modules=_fake_modules(),
        )


def test_radiation_diagnostics_are_present_finite_and_plane_fed():
    # The harness radiation instrument reads mean_outgoing_longwave_w_m2
    # from the result contract; a NaN or missing reading fails every
    # OLR-fed gate.  The fake radiation writes constant planes, so the
    # diagnostics must equal the means computed from those planes, proving
    # the values come from the persistent olr/gsw/glw arrays and not from
    # a hardcoded zero.
    from woof.globe.constants import STEFAN_BOLTZMANN

    exchange = _exchange()
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules()
    )
    result = suite.step(exchange)
    for name in (
        "mean_outgoing_longwave_w_m2", "mean_net_surface_radiation_w_m2",
    ):
        assert name in result.diagnostics, name
        assert np.isfinite(result.diagnostics[name]), name
    assert result.diagnostics["mean_outgoing_longwave_w_m2"] == pytest.approx(
        FAKE_OLR, rel=1.0e-6
    )
    emissivity = np.asarray(result.surface.emissivity, np.float64)
    skin = np.asarray(result.surface.temperature_k, np.float64)
    expected = float(np.mean(
        FAKE_GSW + emissivity * FAKE_GLW
        - emissivity * STEFAN_BOLTZMANN * skin**4
    ))
    assert result.diagnostics["mean_net_surface_radiation_w_m2"] == pytest.approx(
        expected, rel=1.0e-5
    )


@requires_engine_module("woof.verify.harness", "04")
def test_grade_radiation_reads_the_native_subject_through_the_seam():
    # The instrument's subject contract is step(exchange) -> PhysicsResult,
    # which the native suite (and the bridge that wraps it) satisfies.  A
    # CPU-substituted suite proves the whole path: the report builds and
    # every scenario's OLR reading is the suite's own diagnostic, not NaN.
    from woof.verify.harness.radiation_balance import grade_radiation

    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules()
    )
    report = grade_radiation(CONFIG, physics=suite)
    for scenario in ("isothermal_midnight", "warm_moist_noon", "polar_night"):
        assert report.measurements[scenario]["olr_w_m2"] == pytest.approx(
            FAKE_OLR, rel=1.0e-6
        )


@requires_engine_module("woof.verify.harness", "04")
def test_native_bridge_subject_builds_from_the_t255_config():
    from woof.verify.harness.subjects import native_bridge_subject

    subject = native_bridge_subject(T255_CONFIG)
    assert callable(subject.step)
    assert subject.identity["adapter_name"] == "arwen-cuda-column-suite-v1"
    # Device-pending: the Grell-Freitas component joined the suite after
    # the 2026-09-01 qualification battery, so the evidence that earned
    # 'experimental' measured a stack this registration no longer runs and
    # the battery must be re-run (builtin_adapters).
    assert subject.identity["admission_status"] == "device-pending"
    assert subject.identity["device_evidence_sha256"] == "0" * 64
    assert "Grell-Freitas" in subject.identity["scheme_identity"]["cumulus"]


@requires_engine_module("woof.verify.harness", "04")
def test_native_bridge_subject_refuses_a_reference_mode_config():
    from woof.verify.harness.subjects import native_bridge_subject

    with pytest.raises(ValueError, match="physics.mode"):
        native_bridge_subject(CONFIG)


def test_t255_native_config_loads_with_the_full_five_scheme_block():
    cfg = load_config(T255_CONFIG)
    assert cfg.physics_mode == "arwen-native"
    assert cfg.native_adapter_name == "arwen-cuda-column-suite-v1"
    assert cfg.truncation == 255
    assert cfg.backend == "cupy" and cfg.precision == "float32"
    # the config of record is bare: no integrator and no step in its text,
    # so it reads the shipped default core and that core's step
    assert cfg.integrator == "sl_si" and cfg.dt_s == 300.0
    assert cfg.duration_s == 86400.0
    assert cfg.output_interval_s == 10800.0
    options = cfg.native_adapter_options
    assert options["acknowledgement"] == NATIVE_PHYSICS_ACKNOWLEDGEMENT
    for scheme, name in (
        ("radiation", "rrtmgp"), ("surface_layer", "sfclay"),
        ("land_surface", "noah"), ("pbl", "ysu"),
        ("microphysics", "morrison"),
    ):
        assert options[scheme] == name
    # Radiation buckets land exactly on model steps (the 52 min radt
    # convention at 52 km spacing), and Noah runs every step.
    assert options["radiation_interval_s"] % cfg.dt_s == 0.0
    assert options["land_surface_interval_s"] == cfg.dt_s


def test_t255_config_without_the_acknowledgement_is_refused(tmp_path):
    text = Path(T255_CONFIG).read_text(encoding="utf-8")
    stripped = text.replace(
        'acknowledgement = "device-pending-arwen-native-physics-v1", ', ""
    )
    assert stripped != text
    path = tmp_path / "unacknowledged.toml"
    path.write_text(stripped, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid native adapter options"):
        load_config(path)


def test_t255_config_with_a_wrong_acknowledgement_is_refused(tmp_path):
    text = Path(T255_CONFIG).read_text(encoding="utf-8")
    tampered = text.replace(
        "device-pending-arwen-native-physics-v1", "yes-whatever"
    )
    assert tampered != text
    path = tmp_path / "wrong-acknowledgement.toml"
    path.write_text(tampered, encoding="utf-8")
    with pytest.raises(ValueError, match="acknowledgement must be exactly"):
        load_config(path)


def test_native_mode_step_receives_the_dycore_absorber():
    # The gap that killed arm B: the suite-v7 lid absorber lived inside
    # ReferencePhysics, so the arwen-native five-scheme T255 run had no
    # absorber at all and died at hour 6.55 on the 140 K research bound
    # (polar-top collapse, 242 m/s top winds).  The absorber is dycore-
    # owned now: a CPU-substituted native physics step driven through the
    # model's own apply_physics must receive the damping.
    from woof.globe.state import ArwenGlobalState

    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    g = model.grid_state(
        state.atmosphere, only=("u", "v", "p_half", "p_full")
    )
    lon = np.deg2rad(model.transform.grid.longitude_deg)[None, :]
    u = np.array(g["u"], copy=True)
    v = np.array(g["v"], copy=True)
    u[0] += 5.0 * np.sin(2.0 * lon)
    v[0] += 3.0 * np.cos(3.0 * lon)
    zeta, div = model.vector.vordiv_from_wind(u, v)
    fields = list(state.atmosphere.fields())
    fields[0] = zeta
    fields[1] = div
    bundle = ArwenGlobalState(
        state.atmosphere.with_fields(fields), state.surface,
        state.physics_state,
    )
    dt = 600.0

    def top_anomaly(atmosphere):
        uu, _ = model.vector.wind_from_vordiv(
            atmosphere.vorticity, atmosphere.divergence
        )
        top = np.asarray(uu[0], np.float64)
        return top - np.mean(top, axis=-1, keepdims=True)

    arms = {}
    for arm, base in (("absorbed", None), ("control", 0.0)):
        # A fresh CPU-substituted suite per arm: the runtime carries
        # persistent state, and both arms must step identical physics so
        # the ONLY difference between them is the dycore absorber.
        model.physics = ArwenCudaColumnSuite(
            _options(), array_module=np, modules=_fake_modules()
        )
        if base is not None:
            model.sponge_base_pa = base
        out, info = model.apply_physics(bundle, dt)
        assert info["physics_mode"] == "arwen-native"
        arms[arm] = out

    before = top_anomaly(bundle.atmosphere)
    absorbed = top_anomaly(arms["absorbed"].atmosphere)
    control = top_anomaly(arms["control"].atmosphere)
    # The fakes are wind-neutral, so the control arm keeps the anomaly to
    # fp32 casting noise - positive evidence the physics itself did not
    # damp anything and the absorber is the treatment.
    rms = lambda a: float(np.sqrt(np.mean(a * a)))
    assert rms(before) > 1.0
    assert abs(rms(control) - rms(before)) < 1.0e-3 * rms(before)
    # The absorbed arm removed the ring-decay fraction: at the smoke
    # grid's ~978 Pa top ring the graded rate is ~1.03e-3 1/s, so 600 s
    # keeps ~0.54 of the anomaly.
    ring_p = np.mean(np.asarray(g["p_full"]), axis=-1, keepdims=True)
    ring_lid = np.mean(np.asarray(g["p_half"][0]), axis=-1, keepdims=True)
    ramp = np.clip((ring_p - ring_lid) / (5000.0 - ring_lid), 0.0, 1.0)
    decay = np.exp(
        -dt * np.cos(0.5 * math.pi * ramp) ** 2 / 900.0
    )
    expected = rms(control * decay[0])
    assert abs(rms(absorbed) - expected) < 0.02 * rms(before)
    # Below-base levels are untouched by the absorber: with identical
    # physics in both arms their analyzed winds agree bit-for-bit.
    for name in ("vorticity", "divergence"):
        np.testing.assert_array_equal(
            np.asarray(getattr(arms["absorbed"].atmosphere, name))[1:],
            np.asarray(getattr(arms["control"].atmosphere, name))[1:],
        )


def test_native_options_refuse_before_any_cupy_access(tmp_path):
    text = Path(str(_shipped_configs() / "arwen_global_level5_native_smoke.toml")).read_text(encoding="utf-8")
    text = text.replace(
        'native_adapter_options = { ',
        'native_adapter_options = { made_up_option = 1, ',
    )
    path = tmp_path / "bad.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="unknown native adapter options"):
        load_config(path)


def test_native_config_keeps_the_wrf_faithful_switch_it_was_given_and_the_hash_it_implies(tmp_path):
    """gf_updraft_only_when_downdraft_dry = false must reach the runtime
    as false (the adapter identity drops the key, and the loader used to
    keep only the identity, so the value came back as the default true) and
    must hash as the key's absence, the hash every Grell-Freitas checkpoint
    before the switch was written with."""
    smoke = Path(str(_shipped_configs() / "arwen_global_level5_native_smoke.toml"))
    text = smoke.read_text(encoding="utf-8")
    faithful = tmp_path / "faithful.toml"
    faithful.write_text(text.replace(
        'native_adapter_options = { ',
        'native_adapter_options = { gf_updraft_only_when_downdraft_dry = false, ',
    ), encoding="utf-8")
    on = tmp_path / "switched.toml"
    on.write_text(text.replace(
        'native_adapter_options = { ',
        'native_adapter_options = { gf_updraft_only_when_downdraft_dry = true, ',
    ), encoding="utf-8")
    switched = load_config(on)
    wrf = load_config(faithful)
    assert load_config(smoke).config_hash == wrf.config_hash
    assert switched.native_adapter_options["gf_updraft_only_when_downdraft_dry"] is True
    assert wrf.native_adapter_options["gf_updraft_only_when_downdraft_dry"] is False
    assert switched.config_identity["native_adapter_options"]["gf_updraft_only_when_downdraft_dry"] is True
    assert "gf_updraft_only_when_downdraft_dry" not in wrf.config_identity["native_adapter_options"]
    assert switched.config_hash != wrf.config_hash
    # Every other option of the two configs is one mapping.
    same = {k: v for k, v in wrf.native_adapter_options.items() if k != "gf_updraft_only_when_downdraft_dry"}
    assert same == {k: v for k, v in switched.native_adapter_options.items() if k != "gf_updraft_only_when_downdraft_dry"}


def test_bridge_hands_the_adapter_the_wrf_faithful_switch_and_keeps_it_out_of_the_identity():
    """The bridge used to build its adapter from the validator's identity,
    which drops the switch when false, so the adapter ran switched on under
    options that said off."""
    from woof.globe.physics.arwen_bridge import NativeArwenPhysicsBridge
    from woof.globe.physics.builtin_adapters import ADAPTER_NAME

    off = NativeArwenPhysicsBridge(
        ADAPTER_NAME, {**_options(), "gf_updraft_only_when_downdraft_dry": False}
    )
    assert off.adapter.options.gf_updraft_only_when_downdraft_dry is False
    assert off.adapter_options["gf_updraft_only_when_downdraft_dry"] is False
    assert "gf_updraft_only_when_downdraft_dry" not in off.identity["options"]
    on = NativeArwenPhysicsBridge(
        ADAPTER_NAME, {**_options(), "gf_updraft_only_when_downdraft_dry": True}
    )
    assert on.adapter.options.gf_updraft_only_when_downdraft_dry is True
    assert on.identity["options"]["gf_updraft_only_when_downdraft_dry"] is True
    # Every other word of the two identities is the same.
    same = {k: v for k, v in on.identity["options"].items() if k != "gf_updraft_only_when_downdraft_dry"}
    assert same == dict(off.identity["options"])


def test_native_configs_carry_the_measured_repair_gate():
    # The 0.00262451171875 residual (T255 GDAS day 1, 2026-08-31): both
    # named booking suspects -- canwat/snow full-weight vs lf-weight
    # pricing, and the qfx booking's debit weights -- were acquitted at
    # <= 2 x 2^-14 on the CPU repro (water-repair-262, 217,800
    # column-calls), so the residual is the native suite's kernel-internal
    # fp32 floor (Morrison rainncv-vs-column-debit closure, negative-
    # species clamp), which the reference-derived 5e-4 envelope was never
    # sized for.  Per the measured-thresholds law the NATIVE run configs
    # carry their own measured gate value; reference configs keep the
    # envelope.
    measured_day1_max = 0.00262451171875  # = exactly 43 x 2^-14
    key = "gate_physics_water_repair_max_step_kg_m2"

    t255 = load_config(T255_CONFIG)
    limit = t255.gate_physics_water_repair_max_step_kg_m2
    assert limit == 0.010498046875
    # Sizing arithmetic: 4x the measured day-1 maximum -- the identical
    # 43-quantum excursion class two fp32 binades up (column totals to
    # 4096 kg/m2)...
    assert limit == 4.0 * measured_day1_max == 43.0 * 2.0 ** -12
    # ...while staying 100x below the smallest convicted booking route at
    # run scale (precip double-entry, fixed 7d7b528d0), the unbooked-route
    # breakage this gate exists to catch.
    assert limit * 100.0 <= 1.0943603515625

    smoke = load_config(str(_shipped_configs() / "arwen_global_level5_native_smoke.toml"))
    assert smoke.gate_physics_water_repair_max_step_kg_m2 == limit

    # Reference configs keep the precision envelope, and hash stability
    # holds: only a config that SETS the override carries the key in its
    # identity, so every pre-existing reference config hash (and its
    # checkpoints and receipts) is untouched.
    reference = load_config(CONFIG)
    assert reference.gate_physics_water_repair_max_step_kg_m2 is None
    assert key not in reference.config_identity
    assert t255.config_identity[key] == limit


def test_config_repair_gate_override_is_the_operative_limit(tmp_path):
    # The override must reach the receipt through the real runner, both
    # directions: unset, the precision envelope stands; set, the config
    # value is the comparator, so a repair above it fails the run.
    from woof.globe.runner import fixer_absorption_limits, run

    cfg = load_config(CONFIG)
    result = run(cfg, tmp_path / "envelope")
    row = result["gates"]["physics_water_repair_max_step_kg_m2"]
    assert row["limit"] == fixer_absorption_limits(cfg.precision)[
        "physics_repair_kg_m2"
    ]
    assert row["passed"]

    text = Path(CONFIG).read_text(encoding="utf-8")
    tight = text.replace(
        "[gates]", "[gates]\nphysics_water_repair_max_step_kg_m2 = 1.0e-30"
    )
    assert tight != text
    path = tmp_path / "tight.toml"
    path.write_text(tight, encoding="utf-8")
    tight_cfg = load_config(path)
    result = run(tight_cfg, tmp_path / "tight")
    row = result["gates"]["physics_water_repair_max_step_kg_m2"]
    assert row["limit"] == 1.0e-30
    # The healthy float64 repair (measured 1.14e-13 on the reference
    # campaign) sits above 1e-30, so this run must FAIL: the override is
    # enforced, not merely recorded.
    assert row["value"] > 0.0 and not row["passed"]
    assert result["status"] == "fail"

    # A non-positive override is refused like every other gate.
    zeroed = text.replace(
        "[gates]", "[gates]\nphysics_water_repair_max_step_kg_m2 = 0.0"
    )
    zero_path = tmp_path / "zero.toml"
    zero_path.write_text(zeroed, encoding="utf-8")
    with pytest.raises(ValueError, match="gates must be positive"):
        load_config(zero_path)


def _mirror_morrison_modules():
    """Fakes whose Morrison runs the kernel's float64 transcription
    authority (woof.globe.core.npref.np_morrison_column) on column (0, 0), so
    exchange -> NativeColumnBatch -> runtime ordering -> kernel-shape call
    -> the negative-qv raiser is the real path while the microphysics
    arithmetic is the real Morrison arithmetic."""
    from woof.core import constants as c
    from woof.globe.core.npref import np_morrison_column

    modules = _fake_modules()

    def launch_morrison(theta, qv, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
                        rho, exner, p_full, dz, rainnc, rainncv, snownc,
                        snowncv, graupelnc, graupelncv, sr, dt, **kwargs):
        names = (("theta", theta), ("qv", qv), ("qc", qc), ("qr", qr),
                 ("qi", qi), ("qs", qs), ("qg", qg), ("nc", nc),
                 ("nr", nr), ("ni", ni), ("ns", ns), ("ng", ng))
        column = {name: np.asarray(arr[:, 0, 0], np.float64)
                  for name, arr in names}
        column["pii"] = np.asarray(exner[:, 0, 0], np.float64)
        column["pressure"] = np.asarray(p_full[:, 0, 0], np.float64)
        column["dz"] = np.asarray(dz[:, 0, 0], np.float64)
        # The runtime hands rho as an output scratch; the mirror accepts
        # and ignores it (Morrison recomputes density from p and T).
        column["rho"] = column["pressure"] / (
            c.RD * column["theta"] * column["pii"])
        out = np_morrison_column(**column, dt=float(dt))
        for name, arr in names:
            arr[:, 0, 0] = out[name].astype(np.float32)
        rainncv.fill(0.0)
        snowncv.fill(0.0)
        graupelncv.fill(0.0)
        sr.fill(0.0)

    modules["woof.globe.core.morrison"] = SimpleNamespace(
        launch_morrison=launch_morrison)
    return modules


def _plant_polar_night_cold_trap(exchange):
    """Park the model-top cell of column (0, 0) in the polar-night cold
    trap: T = 156 K with qv at 0.9995 of ice saturation and a small
    pre-existing crystal population, the state the stratosphere-free
    native top cools into.  Returns the planted (qv, ni)."""
    from woof.core import constants as c
    from woof.globe.core.npref import _np_morrison_polysvp

    temp = 156.0
    p = float(np.asarray(exchange.p_full)[0, 0, 0])
    ew = min(0.99 * p, float(_np_morrison_polysvp(temp, False)))
    ei = min(ew, 0.99 * p, float(_np_morrison_polysvp(temp, True)))
    qvi = c.EP2 * ei / (p - ei)
    planted_qv = 0.9995 * qvi
    planted_ni = 1.0e3
    exner = float(np.asarray(exchange.exner)[0, 0, 0])
    np.asarray(exchange.temperature)[0, 0, 0] = temp
    np.asarray(exchange.theta)[0, 0, 0] = temp / exner
    np.asarray(exchange.virtual_temperature)[0, 0, 0] = (
        temp * (1.0 + 0.608 * planted_qv))
    np.asarray(exchange.qv)[0, 0, 0] = planted_qv
    np.asarray(exchange.qi)[0, 0, 0] = 1.0e-9
    np.asarray(exchange.ni)[0, 0, 0] = planted_ni
    return planted_qv, planted_ni


def test_polar_night_cold_trap_cannot_overdraw_vapor_through_the_suite():
    """The hour-49 conviction shape, driven through the real suite path.

    A 384 h T255 native run (2026-09-01) died at hour 49 on this raiser:
    'native physics produced negative qv: -1.6e-4'.  The polar-night model
    top, with no stratospheric floor, cooled through ~159 K where
    POLYSVP's extrapolated liquid curve drops under its ice curve, so
    qvi == qvs and Morrison's 0.999*QVS deposition-nucleation trigger
    fired with qv at or below ice saturation; the one-sided FUDGEF rescale
    skips the dum <= 0 / sum_dep > 0 sign pair, so the Cooper-cap embryo
    mass (500e3/rhoa - moments) * MI0 was withdrawn from ~1e-8 kg/kg of
    vapor.  Pre-fix, this test dies inside suite.step with exactly that
    FloatingPointError.  With MNUCCD bounded at the source by the vapor
    excess over ice saturation, the same driven column steps clean: qv
    stays nonnegative and near its planted value, and the number moment
    scales with the withheld mass, so ni stays at its planted count
    instead of jumping to the ~2e7 kg-1 target.
    """
    exchange = _exchange()
    planted_qv, planted_ni = _plant_polar_night_cold_trap(exchange)
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_mirror_morrison_modules()
    )
    result = suite.step(exchange)

    qv = np.asarray(result.qv)
    assert bool(np.isfinite(qv).all())
    assert float(qv.min()) >= 0.0
    # FUDGEF may relax the planted cell the last 0.05% up to qvi, nothing
    # more; a repeat of the overdraw would sit ~1e4x below zero instead.
    assert qv[0, 0, 0] == pytest.approx(planted_qv, rel=2.0e-3)
    assert float(np.asarray(result.ni)[0, 0, 0]) < 10.0 * planted_ni


def test_stratospheric_floor_holds_the_polar_top_by_default():
    """The native suite carries the same cold-top floor as the reference.

    Defect: the 384 h native baseline arm (T255, GDAS 2026-09-01 00Z) died
    at hour 83.4 -- "temperature outside research bounds: 139.995..298.852 K"
    -- via a slow polar-night top sag (~5e-5 K/s) with winds healthy.  The
    reference suite has held this line since v5 with a one-sided
    Held-Suarez pull; the native suite launched without any equivalent.
    Differential arms through identical fakes isolate the floor: with the
    pull reverted, floored-minus-unfloored is zero everywhere and the
    expected-warming assertion fails.
    """
    exchange = _exchange()
    p_full = np.asarray(exchange.p_full, dtype=np.float64)
    exner = np.asarray(exchange.exner, dtype=np.float64)
    mask = p_full < 5000.0
    assert mask.any(), "smoke config has no levels above the floor pressure"
    temperature = np.asarray(exchange.theta, dtype=np.float64) * exner
    temperature[mask] = 150.0
    chilled = replace(exchange, theta=(temperature / exner))

    floored = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules()
    ).step(chilled)
    unfloored = ArwenCudaColumnSuite(
        {**_options(), "stratospheric_floor_k": 0.0},
        array_module=np,
        modules=_fake_modules(),
    ).step(chilled)

    delta = (
        np.asarray(floored.theta, dtype=np.float64)
        - np.asarray(unfloored.theta, dtype=np.float64)
    ) * exner
    control = np.asarray(unfloored.theta, dtype=np.float64) * exner
    # dt=5 s, tau=1800 s: each cold point gains (195 - T) * 5/1800, computed
    # from the post-physics state both arms share.
    expected = np.maximum(195.0 - control, 0.0) * (5.0 / 1800.0)
    assert np.allclose(delta[mask], expected[mask], atol=1.0e-3)
    assert np.abs(delta[~mask]).max() == pytest.approx(0.0, abs=1.0e-6)
    receipt = floored.diagnostics["mean_stratospheric_floor_heating_j_m2"]
    assert receipt > 0.0
    assert unfloored.diagnostics["mean_stratospheric_floor_heating_j_m2"] == 0.0


def test_stratospheric_floor_options_are_fail_closed():
    with pytest.raises(ValueError, match="stratospheric_floor_pa"):
        ArwenCudaColumnSuite({**_options(), "stratospheric_floor_pa": 0.0})
    with pytest.raises(ValueError, match="stratospheric_relaxation_time_s"):
        ArwenCudaColumnSuite(
            {**_options(), "stratospheric_relaxation_time_s": -1.0}
        )


# ---------------------------------------------------------------------------
# Screen-level diagnostics (audit 2026-09-01, task 1b): sfclay's t2/th2/q2
# persist in the physics state so the render tape exports a 2 m value.
# ---------------------------------------------------------------------------


def test_surface_layer_screen_diagnostics_persist_in_the_physics_state():
    from woof.globe.physics.native_runtime import (
        NATIVE_SURFACE_DIAGNOSTICS_SOURCE,
    )
    from woof.globe.physics.surface_diagnostics import SOURCE_METADATA_KEY

    exchange = _exchange()
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=_fake_modules())
    result = suite.step(exchange)
    arrays = result.physics_state.arrays
    for name in ("t2", "th2", "q2", "u10", "v10"):
        assert name in arrays, name
        assert arrays[name].shape == exchange.grid_shape
    # Water columns persist the surface layer's values, not the lowest-
    # level seed: the fake offsets t2 by SFCLAY_T2_OFFSET from the
    # temperature it was handed and halves q2.  Land columns persist WRF's
    # SFCDIAGS after Noah (native_runtime _screen_level_step): with the
    # fakes' zero HFX that is the skin itself, and with their zero QSFC and
    # QFX the flux inversion gives no mixing ratio at all, so the lowest
    # level's vapor is published (the documented divergence).
    lowest_t = np.asarray(exchange.temperature[-1], np.float32)
    lowest_q = np.asarray(exchange.qv[-1], np.float64)
    land = (1.0 + (1.0 - np.asarray(exchange.surface.land_fraction, np.float64))) < 1.5
    assert land.any() and (~land).any()
    water = ~land
    assert np.allclose(arrays["t2"][water], (lowest_t + np.float32(SFCLAY_T2_OFFSET))[water], atol=1.0e-4)
    assert np.allclose(arrays["th2"][water], (lowest_t + np.float32(2.0 * SFCLAY_T2_OFFSET))[water], atol=1.0e-4)
    skin = np.asarray(result.surface.temperature_k, np.float64)
    assert np.allclose(arrays["t2"][land], skin[land], atol=1.0e-3)
    assert np.allclose(arrays["q2"][land], lowest_q[land], rtol=1.0e-6, atol=0.0)
    # The fake halves the vapor it was handed, which is the kernel-side
    # mixing ratio r = q/(1-q); the persisted q2 is that value converted
    # back to the model's specific humidity (audit 2026-09-01 NB-3).
    half_r = 0.5 * lowest_q / (1.0 - lowest_q)
    assert np.allclose(arrays["q2"][water], (half_r / (1.0 + half_r))[water], rtol=1.0e-6, atol=0.0)
    assert not np.allclose(arrays["q2"][water], (0.5 * lowest_q)[water], rtol=1.0e-6, atol=0.0)
    assert np.allclose(arrays["u10"], SFCLAY_U10)
    assert result.physics_state.metadata[SOURCE_METADATA_KEY] == (
        NATIVE_SURFACE_DIAGNOSTICS_SOURCE
    )


def test_a_surface_layer_without_screen_diagnostics_leaves_none_behind():
    # Nothing in the step needs t2/th2/q2, so their absence is not a
    # refusal; the export must then see no screen-level state (and fall
    # back, labelled) rather than a stale seed dressed as a 2 m value.
    from woof.globe.physics.surface_diagnostics import SOURCE_METADATA_KEY

    modules = _fake_modules()
    full = modules["woof.globe.core.sfclay"].sfclay

    def without_screen(*args, **kwargs):
        result = full(*args, **kwargs)
        for name in ("t2", "th2", "q2"):
            delattr(result, name)
        return result

    modules["woof.globe.core.sfclay"] = SimpleNamespace(sfclay=without_screen)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=modules)
    result = suite.step(_exchange())
    for name in ("t2", "th2", "q2"):
        assert name not in result.physics_state.arrays
    assert SOURCE_METADATA_KEY not in result.physics_state.metadata
    # u10/v10 stay: they are required sfclay fields YSU consumes.
    assert "u10" in result.physics_state.arrays


def _authority_sfclay_modules(calls, *, precipitation=None):
    """Fakes whose surface layer is the kernel's float64 transcription
    authority (woof.globe.core.npref.np_sfclay), so the exchange the runtime
    hands sfclay -- mavail, znt, xland and the inout fields -- reaches the
    real surface-layer arithmetic.  Every call's positional inputs and
    keyword inout values are recorded (host copies) for replay against the
    authority with substituted inputs."""
    from woof.globe.core.npref import np_sfclay

    modules = _fake_modules(precipitation=precipitation)
    names = ("u", "v", "t", "qv", "p", "dz8w", "psfc", "tsk", "znt",
             "pblh", "mavail", "xland")

    def sfclay(*args, **kwargs):
        inputs = {name: np.array(value, np.float64, copy=True)
                  for name, value in zip(names, args)}
        inout = {name: np.array(kwargs[name], np.float64, copy=True)
                 for name in ("qsfc", "zol", "ust", "mol", "hfx", "qfx")}
        options = {name: kwargs[name] for name in ("option", "dx", "lakemask")}
        calls.append({"inputs": inputs, "inout": inout, "options": options})
        out = np_sfclay(*args, **kwargs)
        return SimpleNamespace(
            **{name: np.ascontiguousarray(np.asarray(out[name], np.float32))
               for name in ("znt", "ust", "mol", "hfx", "qfx", "qsfc", "zol",
                            "wspd", "br", "fm", "fh", "u10", "v10", "chs",
                            "chs2", "cqs2", "qgh", "t2", "th2", "q2")},
        )

    modules["woof.globe.core.sfclay"] = SimpleNamespace(sfclay=sfclay)
    return modules


def _replay_authority(call, **substituted):
    from woof.globe.core.npref import np_sfclay

    inputs = {**call["inputs"], **substituted}
    return np_sfclay(
        *(inputs[name] for name in ("u", "v", "t", "qv", "p", "dz8w", "psfc",
                                    "tsk", "znt", "pblh", "mavail", "xland")),
        **call["inout"], **call["options"],
    )


def _warm_ocean(exchange, *, air_theta_k=298.0, skin_k=301.0, qv=0.017,
                wind_m_s=None):
    """The audit's tropical ocean column planted on every water column of
    the smoke exchange: 298 K air over a 301 K sea at 0.017 kg/kg, which
    gives the surface layer a real evaporative signal (the cold-start water
    columns sit at a ~1e-7 kg/m2/s dew).  Planted as POTENTIAL temperature
    (the runtime re-derives temperature = theta * exner after radiation):
    the smoke grid's lowest full level is at ~783 hPa, where a 298 K
    temperature would be a 319 K theta over the 301 K sea, a strongly
    stable layer (bulk Ri 1.8) with a 0.15 m/s friction velocity."""
    lf = np.asarray(exchange.surface.land_fraction)
    water = (1.0 + (1.0 - lf)) >= 1.5
    theta = np.array(exchange.theta, copy=True)
    qv_field = np.array(exchange.qv, copy=True)
    theta[-1][water] = air_theta_k
    qv_field[-1][water] = qv
    fields = {"theta": theta, "qv": qv_field}
    if wind_m_s is not None:
        u = np.array(exchange.u, copy=True)
        v = np.array(exchange.v, copy=True)
        u[-1][water] = wind_m_s
        v[-1][water] = 0.0
        fields.update(u=u, v=v)
    surface = exchange.surface.copy()
    surface.temperature_k[...] = np.where(
        water, surface.temperature_k.dtype.type(skin_k), surface.temperature_k
    )
    # Open-water roughness as the analysis init stamps it at land_fraction
    # 0 (1.0e-4 m); the smoke grid's water columns are fractional coast
    # and would otherwise start at the 1e-4 + 0.08 * lf land blend.
    surface.roughness_m[...] = np.where(
        water, surface.roughness_m.dtype.type(1.0e-4), surface.roughness_m
    )
    return replace(exchange, surface=surface, **fields)


def test_open_water_columns_evaporate_at_full_moisture_availability():
    # Audit 2026-09-01 NB-1: sfclay received surface.soil_water_fraction[0]
    # as MAVAIL on every column; the kernel scales the water-branch qfx by
    # it linearly (sfclay.cu:494,497) and the ocean soil fill is 0.25 for
    # the whole run (Noah never integrates xland >= 1.5), so ocean
    # evaporation ran at a quarter of WRF's (LANDUSE.TBL SLMO = 1.0 for
    # water; np_sfclay: LH 33.82 vs 135.28 W/m2, ratio exactly 0.25).
    exchange = _warm_ocean(_exchange())
    calls = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_authority_sfclay_modules(calls)
    )
    result = suite.step(exchange)
    assert len(calls) == 1
    call = calls[0]
    water = call["inputs"]["xland"] >= 1.5
    assert water.any() and (~water).any()
    soil_top = np.asarray(exchange.surface.soil_water_fraction[0], np.float32)
    assert float(soil_top[water].max()) < 0.5  # the fill the defect used

    # Water columns are handed WRF's water-body availability; land keeps
    # the soil-derived value its slab-style bulk flux is defined against.
    assert np.all(call["inputs"]["mavail"][water] == 1.0)
    assert np.array_equal(
        call["inputs"]["mavail"][~water], soil_top[~water].astype(np.float64)
    )

    # The persisted water-column qfx IS the mavail = 1 authority value, and
    # the starved value the defect produced is soil_top times it.
    full = _replay_authority(call, mavail=np.ones_like(soil_top, np.float64))
    starved = _replay_authority(call, mavail=soil_top.astype(np.float64))
    qfx = np.asarray(result.physics_state.arrays["qfx"], np.float64)
    assert float(full["qfx"][water].min()) > 0.0
    assert np.allclose(qfx[water], full["qfx"][water], rtol=1.0e-6, atol=0.0)
    assert np.allclose(
        starved["qfx"][water], soil_top[water] * full["qfx"][water],
        rtol=1.0e-6, atol=0.0,
    )
    assert float(np.max(starved["qfx"][water] / full["qfx"][water])) < 0.5
    assert np.allclose(
        np.asarray(result.physics_state.arrays["hfx"], np.float64)[water],
        full["hfx"][water], rtol=1.0e-6, atol=1.0e-9,
    )


def test_ocean_roughness_grows_to_the_charnock_value_across_calls():
    # Audit 2026-09-01 NB-4: sfclay's Charnock z0 output (WRF inout znt)
    # was stored only in the persistent 'znt' YSU reads, while the next
    # call's znt input was surface.roughness_m -- written only by Noah on
    # land -- so every ocean call restarted from 1e-4 m.  At gale winds the
    # converged roughness is ~1e-3 m and ust 0.78 vs 1.00 m/s; surface
    # stress ran 13-33% low from 10 to 20 m/s over the whole ocean.
    exchange = _warm_ocean(_exchange(), wind_m_s=25.0)
    lf = np.asarray(exchange.surface.land_fraction)
    water = (1.0 + (1.0 - lf)) >= 1.5
    roughness0 = np.asarray(exchange.surface.roughness_m, np.float64).copy()
    assert float(roughness0[water].max()) < 2.0e-4

    calls = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_authority_sfclay_modules(calls)
    )
    result = suite.step(exchange)
    for n in range(1, 8):
        result = suite.step(_advance(exchange, result, 10.0 * n))
    assert len(calls) == 8

    # Every call after the first starts from the roughness the previous
    # call returned (the persisted 'znt' plane) on every water column...
    for previous, call in zip(calls, calls[1:]):
        handed = call["inputs"]["znt"]
        returned = _replay_authority(previous)["znt"].astype(np.float32)
        assert np.allclose(handed[water], returned[water], rtol=1.0e-6, atol=0.0)
    # ...the defect handed the initial value on every call instead.
    assert not np.allclose(
        calls[-1]["inputs"]["znt"][water], roughness0[water], rtol=0.5
    )

    # The surface state carries the Charnock roughness of the final
    # friction velocity: z0 = min(0.0185 ust^2/g + 0.11 nu/ust, 2.85e-3),
    # an order of magnitude above the 1e-4 m start at 25 m/s.
    final = result.physics_state.arrays
    ust = np.asarray(final["ust"], np.float64)
    charnock = np.minimum(
        0.0185 * ust**2 / 9.81 + 0.11 * 1.5e-5 / ust, 2.85e-3
    )
    roughness = np.asarray(result.surface.roughness_m, np.float64)
    assert np.allclose(roughness[water], charnock[water], rtol=1.0e-5, atol=0.0)
    assert np.allclose(roughness[water], np.asarray(final["znt"], np.float64)[water])
    assert float(roughness[water].min()) > 5.0 * float(roughness0[water].max())
    # Land roughness belongs to Noah (inert here), untouched by sfclay: the
    # float64 exchange comes back through Noah's float32 driver plane.
    assert np.allclose(roughness[~water], roughness0[~water], rtol=1.0e-6, atol=0.0)


SFCLAY_SENTINEL_HFX, SFCLAY_SENTINEL_QFX = 100.0, 1.0e-4
NOAH_SENTINEL_HFX, NOAH_SENTINEL_QFX = 50.0, 5.0e-5


def _flux_alternation_modules(ysu_seen, noah_calls):
    """The audit's land_flux_alternation fakes: sfclay reports one flux
    pair everywhere on every call, Noah (when due) writes a different pair
    on the columns it integrates, and YSU records the land- and water-
    column fluxes it was handed."""
    modules = _fake_modules()
    stock_sfclay = modules["woof.globe.core.sfclay"].sfclay
    stock_noah = modules["woof.globe.core.noah"]
    stock_ysu = modules["woof.globe.core.ysu"].launch_ysu

    def sfclay(*args, **kwargs):
        result = stock_sfclay(*args, **kwargs)
        result.hfx[...] = np.float32(SFCLAY_SENTINEL_HFX)
        result.qfx[...] = np.float32(SFCLAY_SENTINEL_QFX)
        result.qsfc[...] = np.float32(0.011)
        return result

    def launch_noah(dev, params, dt, dzs, **kwargs):
        land = np.asarray(dev["xland"]) < 1.5
        dev["hfx"][land] = np.float32(NOAH_SENTINEL_HFX)
        dev["qfx"][land] = np.float32(NOAH_SENTINEL_QFX)
        dev["qsfc"][land] = np.float32(0.009)
        noah_calls.append(float(dt))

    def launch_ysu(u, v, theta, qv, qc, qi, *args, **kwargs):
        land = np.asarray(kwargs["xland"]) < 1.5
        hfx = np.asarray(kwargs["hfx"], np.float64)
        qfx = np.asarray(kwargs["qfx"], np.float64)
        ysu_seen.append({
            "land_hfx": (float(hfx[land].min()), float(hfx[land].max())),
            "land_qfx": (float(qfx[land].min()), float(qfx[land].max())),
            "water_hfx": (float(hfx[~land].min()), float(hfx[~land].max())),
            "water_qfx": (float(qfx[~land].min()), float(qfx[~land].max())),
        })
        return stock_ysu(u, v, theta, qv, qc, qi, *args, **kwargs)

    modules["woof.globe.core.sfclay"] = SimpleNamespace(sfclay=sfclay)
    modules["woof.globe.core.noah"] = SimpleNamespace(
        _F2D=stock_noah._F2D, load_tables=stock_noah.load_tables,
        pack_params=stock_noah.pack_params, launch_noah=launch_noah,
    )
    modules["woof.globe.core.ysu"] = SimpleNamespace(launch_ysu=launch_ysu)
    return modules


def test_land_columns_hand_ysu_the_land_surface_fluxes_on_every_call():
    # Audit 2026-09-01 NB-2: sfclay overwrote the persistent hfx/qfx/qsfc
    # on every call and YSU read them, while Noah ran only when its bucket
    # changed.  With land_surface_interval_s = dt the Strang halves put two
    # physics calls in every bucket (time_s t and t+dt; the second half of
    # step n and the first half of step n+1 share a time_s), so YSU's land
    # columns received sfclay's bulk fluxes on 3 of 8 calls in the
    # reproduction, asymptotically 1 in 2 -- half of all PBL forcing on
    # land bypassing Noah's surface energy balance.  WRF's surface_driver
    # overwrites HFX/QFX with the LSM's every step before the PBL reads
    # them.  Same call pattern here: the suite is stepped at the exchange
    # times the dycore produces (0, dt, dt, 2dt, 2dt, ...).
    exchange = _exchange()
    dt = float(exchange.dt_s)
    options = {**_options(), "land_surface_interval_s": 2.0 * dt}
    ysu_seen, noah_calls = [], []
    suite = ArwenCudaColumnSuite(
        options, array_module=np,
        modules=_flux_alternation_modules(ysu_seen, noah_calls),
    )
    times = [0.0, 2 * dt, 2 * dt, 4 * dt, 4 * dt, 6 * dt, 6 * dt, 8 * dt]
    result = suite.step(exchange)
    for time_s in times[1:]:
        result = suite.step(_advance(exchange, result, time_s))
    assert len(ysu_seen) == 8
    # Noah ran once per bucket (t = 0 and each bucket's first call), not
    # on the shared-time_s repeats.
    assert len(noah_calls) == 5

    # Every YSU call saw Noah's fluxes on land and sfclay's on water; the
    # defect showed sfclay's sentinel on land on the 3 repeated calls.
    for seen in ysu_seen:
        assert seen["land_hfx"] == (NOAH_SENTINEL_HFX, NOAH_SENTINEL_HFX), seen
        assert seen["land_qfx"] == pytest.approx(
            (NOAH_SENTINEL_QFX, NOAH_SENTINEL_QFX), rel=1.0e-6
        )
        assert seen["water_hfx"] == (SFCLAY_SENTINEL_HFX, SFCLAY_SENTINEL_HFX)
        assert seen["water_qfx"] == pytest.approx(
            (SFCLAY_SENTINEL_QFX, SFCLAY_SENTINEL_QFX), rel=1.0e-6
        )
    # The persisted planes agree after a non-due call (the last one is
    # 8 dt, a due call; the one before it was the repeat at 6 dt).
    arrays = result.physics_state.arrays
    land = (1.0 + (1.0 - np.asarray(exchange.surface.land_fraction))) < 1.5
    assert np.all(np.asarray(arrays["hfx"])[land] == np.float32(NOAH_SENTINEL_HFX))
    assert np.all(np.asarray(arrays["qsfc"])[land] == np.float32(0.009))
    assert np.all(np.asarray(arrays["hfx"])[~land] == np.float32(SFCLAY_SENTINEL_HFX))
    assert np.all(np.asarray(arrays["qsfc"])[~land] == np.float32(0.011))
    for name in ("noah_hfx", "noah_qfx", "noah_qsfc"):
        assert name in arrays


def test_kernels_receive_dry_mixing_ratio_and_the_model_keeps_specific_humidity():
    # Audit 2026-09-01 NB-3: the model's qv is GDAS specific humidity and
    # every WRF-lineage kernel consumes dry mixing ratio (noah.cu:982 forms
    # q2k = qv1/(1+qv1); sfclay/morrison compare qv against EP2*e/(p-e));
    # the bridge copied the field unchanged, so the kernels read every
    # saturation test dry by q^2 -- 2.04% at q = 0.02.  Converted at the
    # bridge boundary, r = q/(1-q_v) in and q = r/(1+r_v) out, the water
    # ledger closes in the model's metric.
    from woof.globe.physics.native_batch import (
        mixing_ratio_from_specific_humidity,
        specific_humidity_from_mixing_ratio,
    )
    from woof.globe.water import atmospheric_water_column

    exchange = _exchange()
    qv = np.array(exchange.qv, copy=True)
    qc = np.array(exchange.qc, copy=True)
    qv[-1, 0, 0] = 0.02      # lowest full level of column (0, 0)
    qc[-1, 0, 0] = 1.0e-3
    exchange = replace(exchange, qv=qv, qc=qc)
    expected_r_v = 0.02 / (1.0 - 0.02)
    expected_r_c = 1.0e-3 / (1.0 - 0.02)

    # The batch boundary itself.
    batch = NativeColumnBatch.from_exchange(exchange, np)
    assert batch.arrays["qv"][0, 0, 0] == pytest.approx(expected_r_v, rel=1.0e-6)
    assert batch.arrays["qc"][0, 0, 0] == pytest.approx(expected_r_c, rel=1.0e-6)
    back = batch.global_prognostics()
    assert back["qv"][-1, 0, 0] == pytest.approx(0.02, rel=1.0e-6)
    assert back["qc"][-1, 0, 0] == pytest.approx(1.0e-3, rel=1.0e-6)
    assert np.allclose(back["qv"], qv, rtol=2.0e-7, atol=1.0e-12)
    # The helpers are exact inverses in float64.
    planted = {"qv": np.float64(0.02), "qc": np.float64(1.0e-3)}
    assert specific_humidity_from_mixing_ratio(
        mixing_ratio_from_specific_humidity(planted)
    ) == pytest.approx(planted, rel=1.0e-15)

    # Through the suite: every kernel that reads vapor sees the mixing
    # ratio, and the returned state carries the model's specific humidity.
    seen = {}
    modules = _fake_modules()
    stock_sfclay = modules["woof.globe.core.sfclay"].sfclay
    stock_ysu = modules["woof.globe.core.ysu"].launch_ysu
    stock_morrison = modules["woof.globe.core.morrison"].launch_morrison

    def sfclay(*args, **kwargs):
        seen["sfclay_qv"] = float(np.asarray(args[3])[0, 0])
        return stock_sfclay(*args, **kwargs)

    def launch_ysu(u, v, theta, qv_k, qc_k, qi_k, *args, **kwargs):
        seen["ysu_qv"] = float(qv_k[0, 0, 0])
        seen["ysu_qc"] = float(qc_k[0, 0, 0])
        return stock_ysu(u, v, theta, qv_k, qc_k, qi_k, *args, **kwargs)

    def launch_morrison(theta, qv_k, qc_k, *args, **kwargs):
        seen["morrison_qv"] = float(qv_k[0, 0, 0])
        seen["morrison_qc"] = float(qc_k[0, 0, 0])
        return stock_morrison(theta, qv_k, qc_k, *args, **kwargs)

    modules["woof.globe.core.sfclay"] = SimpleNamespace(sfclay=sfclay)
    modules["woof.globe.core.ysu"] = SimpleNamespace(launch_ysu=launch_ysu)
    modules["woof.globe.core.morrison"] = SimpleNamespace(launch_morrison=launch_morrison)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=modules)
    result = suite.step(exchange)
    for name in ("sfclay_qv", "ysu_qv", "morrison_qv"):
        assert seen[name] == pytest.approx(expected_r_v, rel=1.0e-6), name
    for name in ("ysu_qc", "morrison_qc"):
        assert seen[name] == pytest.approx(expected_r_c, rel=1.0e-6), name
    assert np.asarray(result.qv)[-1, 0, 0] == pytest.approx(0.02, rel=1.0e-6)
    assert np.asarray(result.qc)[-1, 0, 0] == pytest.approx(1.0e-3, rel=1.0e-6)

    # The ledger compares like with like: with water-inert fakes the
    # model's own column water is unchanged to float32 casting scale and
    # the suite's repair stays there too (the mixing-ratio metric would
    # have read the planted column 2% heavier on one side).
    def column(source):
        fields = {
            "dp": np.asarray(exchange.dp, np.float64),
            **{name: np.asarray(getattr(source, name), np.float64)
               for name in WATER_SPECIES},
        }
        return np.asarray(atmospheric_water_column(fields), np.float64).sum(axis=0)

    assert np.allclose(column(result), column(exchange), atol=1.0e-4)
    assert result.diagnostics["maximum_local_water_repair_kg_m2"] < 1.0e-4
    assert np.allclose(
        np.asarray(result.surface.water_kg_m2, np.float64),
        np.asarray(exchange.surface.water_kg_m2, np.float64), atol=1.0e-4,
    )


# --------------------------------------------------------------------------
# Grell-Freitas, the sixth native component (audit 2026-09-01, task 3).
# --------------------------------------------------------------------------

def _column_water(source, dp):
    from woof.globe.water import atmospheric_water_column

    fields = {
        "dp": np.asarray(dp, np.float64),
        **{name: np.asarray(getattr(source, name), np.float64)
           for name in WATER_SPECIES},
    }
    return np.asarray(atmospheric_water_column(fields), np.float64).sum(axis=0)


def test_cumulus_runs_between_the_pbl_and_microphysics_in_wrf_order():
    from woof.globe.physics.native_suite import (
        NATIVE_COMPONENT_ORDER, native_component_order,
    )

    assert NATIVE_COMPONENT_ORDER == (
        "rrtmgp", "sfclay", "noah", "ysu", "gf", "morrison"
    )
    order = []
    modules = _fake_modules()
    stock_sfclay = modules["woof.globe.core.sfclay"].sfclay
    stock_noah = modules["woof.globe.core.noah"]
    stock_ysu = modules["woof.globe.core.ysu"].launch_ysu
    stock_gf = modules["woof.globe.core.gf"].GrellFreitas
    stock_morrison = modules["woof.globe.core.morrison"].launch_morrison

    def sfclay(*args, **kwargs):
        order.append("sfclay")
        return stock_sfclay(*args, **kwargs)

    def launch_noah(*args, **kwargs):
        order.append("noah")
        return stock_noah.launch_noah(*args, **kwargs)

    def launch_ysu(*args, **kwargs):
        order.append("ysu")
        return stock_ysu(*args, **kwargs)

    class GrellFreitas(stock_gf):
        def __call__(self, **kwargs):
            order.append("gf")
            return super().__call__(**kwargs)

    def launch_morrison(*args, **kwargs):
        order.append("morrison")
        return stock_morrison(*args, **kwargs)

    modules["woof.globe.core.sfclay"] = SimpleNamespace(sfclay=sfclay)
    modules["woof.globe.core.noah"] = SimpleNamespace(
        _F2D=stock_noah._F2D, load_tables=stock_noah.load_tables,
        pack_params=stock_noah.pack_params, launch_noah=launch_noah,
    )
    modules["woof.globe.core.ysu"] = SimpleNamespace(launch_ysu=launch_ysu)
    modules["woof.globe.core.gf"] = SimpleNamespace(GrellFreitas=GrellFreitas)
    modules["woof.globe.core.morrison"] = SimpleNamespace(launch_morrison=launch_morrison)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=modules)
    result = suite.step(_exchange())
    assert order == ["sfclay", "noah", "ysu", "gf", "morrison"]
    assert result.diagnostics["cumulus_active"] is True
    assert result.physics_state.metadata["cumulus_updates"] == 1
    assert suite.identity["component_order"] == list(NATIVE_COMPONENT_ORDER)
    assert native_component_order(suite.options) == NATIVE_COMPONENT_ORDER


def test_cumulus_tendencies_integrate_into_the_column_with_the_exner_invariant():
    from woof.globe.constants import DRY_AIR_CP, GRAVITY_M_S2

    heating = 1.0e-3      # K/s theta at the lowest level
    drying = -2.0e-6      # kg/kg/s dry mixing ratio at the lowest level
    detrain_c = 5.0e-7    # kg/kg/s cloud water one level up
    detrain_i = 2.0e-7    # kg/kg/s cloud ice two levels up

    def convection(result, atmosphere):
        result.rthcuten[0] = heating
        result.rqvcuten[0] = drying
        result.rqccuten[1] = detrain_c
        result.rqicuten[2] = detrain_i

    exchange = _exchange()
    dt = float(exchange.dt_s)
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules(cumulus=convection)
    )
    result = suite.step(exchange)
    # theta: dt * rate, plus the fake Morrison's 0.125 K at column (0, 0).
    dtheta = np.asarray(result.theta[-1], np.float64) - np.asarray(exchange.theta[-1], np.float64)
    expected = np.full(dtheta.shape, dt * heating)
    expected[0, 0] += 0.125
    assert np.allclose(dtheta, expected, atol=2.0e-5)
    # water: the rates act on the kernel-side mixing ratios and come back
    # as the model's specific humidity through the bridge's own inverse.
    batch = NativeColumnBatch.from_exchange(exchange, np)
    r_v = np.asarray(batch.arrays["qv"], np.float64)
    r_c = np.asarray(batch.arrays["qc"], np.float64)
    r_i = np.asarray(batch.arrays["qi"], np.float64)
    r_v[0] += dt * drying
    r_c[1] += dt * detrain_c
    r_i[2] += dt * detrain_i
    moist = 1.0 + r_v
    assert np.allclose(np.asarray(result.qv)[::-1], r_v / moist, rtol=1.0e-6, atol=1.0e-12)
    assert np.allclose(np.asarray(result.qc)[::-1], r_c / moist, rtol=1.0e-6, atol=1.0e-12)
    assert np.allclose(np.asarray(result.qi)[::-1], r_i / moist, rtol=1.0e-6, atol=1.0e-12)
    # temperature = theta * exner held after the cumulus step: the energy
    # advisory measures the convective heating like any other term (it
    # reads batch temperature, which a stale copy would leave at zero).
    dp_bottom = np.asarray(exchange.dp[-1], np.float64)
    exner_bottom = np.asarray(exchange.exner[-1], np.float64)
    energy = DRY_AIR_CP / GRAVITY_M_S2 * dp_bottom * exner_bottom * expected
    assert result.diagnostics["maximum_native_energy_change_j_m2"] == pytest.approx(
        float(np.max(np.abs(energy))), rel=1.0e-3
    )


def test_cumulus_receives_the_wrf_column_state_and_forcing_lanes():
    from woof.globe.constants import (
        DRY_AIR_GAS_CONSTANT, GRAVITY_M_S2, KAPPA, REFERENCE_PRESSURE_PA,
    )
    from woof.globe.physics.native_options import NativePhysicsOptions

    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    exchange = model._physics_exchange(state, 5.0)
    assert exchange.omega_half_pa_s is not None
    calls = []
    pbl_rate = 2.0e-4
    modules = _fake_modules(cumulus_calls=calls, radiation_heating=1.0e-4)
    stock_ysu = modules["woof.globe.core.ysu"].launch_ysu

    def launch_ysu(u, v, theta, qv, qc, qi, *args, **kwargs):
        out = stock_ysu(u, v, theta, qv, qc, qi, *args, **kwargs)
        out["dtheta"][...] = pbl_rate
        return out

    modules["woof.globe.core.ysu"] = SimpleNamespace(launch_ysu=launch_ysu)
    options = {**_options(), "dx_m": 41_000.0}
    suite = ArwenCudaColumnSuite(options, array_module=np, modules=modules)
    suite.step(exchange)
    assert len(calls) == 1
    # The global column's Grell-Freitas is WRF's kernel word for word by
    # default: the coarse-column deep arm (gf_resolved_convergence_closure)
    # and the older updraft-only switch are both opt-ins since their
    # lanes' grades; the seam is built with both switches and the batch's
    # chunk.
    assert modules["woof.globe.core.gf"].GrellFreitas.constructed == [{
        "column_chunk": NativePhysicsOptions.cumulus_column_chunk,
        "updraft_only_when_downdraft_dry": False,
        "resolved_convergence_closure": False,
    }]
    call = calls[0]
    # WRF's t/q: the column BEFORE this call's radiation and PBL forcing,
    # with both rates handed over as the lanes the kernel forms tn from.
    entry_t = np.asarray(exchange.temperature, np.float32)[::-1]
    assert np.array_equal(call["atmosphere"]["temperature"], entry_t)
    batch = NativeColumnBatch.from_exchange(exchange, np)
    assert np.array_equal(call["atmosphere"]["qv"], batch.arrays["qv"])
    assert np.allclose(call["driver"]["rthratenlw"], 1.0e-4)
    assert np.allclose(call["driver"]["rthratensw"], 0.0)
    assert np.allclose(call["driver"]["gf_rthblten"], pbl_rate)
    assert np.allclose(call["driver"]["gf_rqvblten"], 0.0)
    # The dynamics lanes exist from the first call (zero: no dynamics
    # interval has been measured yet) and the GF slot keeps the scalar dx.
    assert call["driver"]["gf_rthdynten"].shape == entry_t.shape
    assert call["driver"]["gf_rqvdynten"].shape == entry_t.shape
    assert not np.any(call["driver"]["gf_rthdynten"])
    assert not np.any(call["driver"]["gf_rqvdynten"])
    assert call["driver"]["gf_dx_column"] is None
    # Vertical velocity from the exchange's omega: w = -omega / (rho g) on
    # the half levels, rho from the ideal gas law on virtual temperature.
    omega = np.asarray(exchange.omega_half_pa_s, np.float64)[::-1]
    rho_full = (
        np.asarray(exchange.p_full, np.float64)
        / (DRY_AIR_GAS_CONSTANT * np.asarray(exchange.virtual_temperature, np.float64))
    )[::-1]
    rho_half = np.empty_like(omega)
    rho_half[0] = rho_full[0]
    rho_half[-1] = rho_full[-1]
    rho_half[1:-1] = 0.5 * (rho_full[:-1] + rho_full[1:])
    expected_w = -omega / np.maximum(rho_half * GRAVITY_M_S2, 1.0e-12)
    assert call["w"].shape == omega.shape
    assert np.allclose(call["w"], expected_w, rtol=1.0e-5, atol=1.0e-9)
    assert float(np.max(np.abs(expected_w))) > 0.0
    # Terrain height recovered from the exchange's own hydrostatic
    # geopotential is the model's surface geopotential over g.
    terrain = np.asarray(model.surface_geopotential, np.float64) / GRAVITY_M_S2
    assert np.allclose(call["ht"], terrain, atol=0.05)
    # The rest of the WRF interface: pressures in Pa bottom-to-top, the
    # land flag, the fluxes YSU saw, the PBL top index rebuilt from pblh
    # when the launcher returns none (the 800 m seed lies below this
    # grid's ~2 km lowest level, so the index is 1 everywhere).
    assert np.array_equal(call["atmosphere"]["pressure"], np.asarray(exchange.p_full, np.float32)[::-1])
    assert np.array_equal(call["atmosphere"]["p_interface"], np.asarray(exchange.p_half, np.float32)[::-1])
    assert np.allclose(
        call["atmosphere"]["exner"],
        (np.asarray(exchange.p_full, np.float32)[::-1] / REFERENCE_PRESSURE_PA) ** KAPPA,
    )
    # The land flag is WRF's integer-valued XLAND on the bridge's own
    # water threshold (the same set sfclay and Noah partition on).
    lf32 = np.asarray(exchange.surface.land_fraction, np.float32)
    expected_xland = np.where((1.0 + (1.0 - lf32)) >= 1.5, 2.0, 1.0)
    assert np.array_equal(call["fields"]["xland"], expected_xland)
    assert {1.0, 2.0} <= set(np.unique(expected_xland).tolist())
    assert call["fields"]["kpbl"].dtype == np.int32
    assert np.all(call["fields"]["kpbl"] == 1)
    assert call["qi"] is not None
    assert call["cfg"].dx == 41_000.0
    assert call["cfg"].ishallow == 1
    assert call["cfg"].clos_choice == 0
    assert call["cfg"].clock_dt == 5.0

    # A launcher that reports its own kpbl is passed through verbatim.
    calls.clear()
    modules = _fake_modules(cumulus_calls=calls)
    stock_ysu = modules["woof.globe.core.ysu"].launch_ysu

    def launch_ysu_with_kpbl(u, v, theta, qv, qc, qi, *args, **kwargs):
        out = stock_ysu(u, v, theta, qv, qc, qi, *args, **kwargs)
        out["kpbl"] = np.full(theta.shape[1:], 3, np.int32)
        return out

    modules["woof.globe.core.ysu"] = SimpleNamespace(launch_ysu=launch_ysu_with_kpbl)
    ArwenCudaColumnSuite(_options(), array_module=np, modules=modules).step(exchange)
    assert np.all(calls[0]["fields"]["kpbl"] == 3)


def test_cumulus_dynamics_lanes_measure_what_ran_between_calls():
    """The advective forcing lanes (WRF RTHFTEN/RQVFTEN) are the change of
    theta and vapor between the previous call's exit and this call's entry
    over the elapsed time: zero on the first call, measured when time_s
    has advanced, held when it has not (the two physics halves of one step
    share a time_s), checkpointed in the physics namespace with the exit
    state it is measured from so a fresh runtime restarted from that
    namespace hands the scheme the same lane at the same time_s and
    measures the same new lane at a later one, and absent without a
    cumulus scheme."""
    from woof.globe.physics.native_state import (
        CUMULUS_DYNAMICS_LANES, CUMULUS_DYNAMICS_QV_LANE, CUMULUS_DYNAMICS_THETA_LANE,
    )

    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    exchange = model._physics_exchange(state, 5.0)
    t0 = float(exchange.time_s)
    calls = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules(cumulus_calls=calls)
    )
    first = suite.step(exchange)
    for name in CUMULUS_DYNAMICS_LANES:
        assert name in first.physics_state.arrays
        assert not np.any(first.physics_state.arrays[name])
    assert not np.any(calls[0]["driver"]["gf_rthdynten"])

    # A later call whose entry theta sits 0.5 K above the previous exit
    # everywhere: the theta lane reads 0.5 K over the elapsed seconds, the
    # vapor lane (vapor unchanged) zero.
    elapsed = 20.0
    second = replace(
        _advance(exchange, first, t0 + elapsed),
        theta=np.asarray(first.theta, np.float32) + np.float32(0.5),
    )
    second_result = suite.step(second)
    lane = calls[1]["driver"]["gf_rthdynten"]
    assert lane.shape == calls[1]["atmosphere"]["temperature"].shape
    assert np.allclose(lane, 0.5 / elapsed, rtol=1.0e-5, atol=0.0)
    # The vapor lane reads only the suite's own post-call water repair
    # (float32 ulps of the column's vapor), nothing the scheme can see.
    assert float(np.max(np.abs(calls[1]["driver"]["gf_rqvdynten"]))) < 1.0e-9
    held = second_result.physics_state.arrays[CUMULUS_DYNAMICS_THETA_LANE]
    assert np.array_equal(held, lane)
    assert float(np.max(np.abs(second_result.physics_state.arrays[CUMULUS_DYNAMICS_QV_LANE]))) < 1.0e-9

    # The same time_s again (the first half of the next step): the entry
    # differs from the exit but no dynamics interval elapsed, so the held
    # lane stands.
    third = replace(
        _advance(second, second_result, t0 + elapsed),
        theta=np.asarray(second_result.theta, np.float32) + np.float32(3.0),
    )
    third_result = suite.step(third)
    assert np.array_equal(calls[2]["driver"]["gf_rthdynten"], lane)

    # A fresh runtime restarted from the checkpointed namespace at that
    # time_s hands the scheme the checkpointed lane, not zero.
    restarted_calls = []
    restarted = ArwenCudaColumnSuite(
        _options(), array_module=np,
        modules=_fake_modules(cumulus_calls=restarted_calls),
    )
    restarted.step(_advance(third, third_result, t0 + elapsed))
    assert np.array_equal(restarted_calls[0]["driver"]["gf_rthdynten"], lane)
    # And at a later time_s the restarted runtime measures the new lane
    # from the checkpointed exit state, the same words the continuous
    # runtime produces.
    fourth = replace(
        _advance(third, third_result, t0 + 2.0 * elapsed),
        theta=np.asarray(third_result.theta, np.float32) - np.float32(1.0),
    )
    suite.step(fourth)
    restarted.step(fourth)
    assert np.array_equal(restarted_calls[1]["driver"]["gf_rthdynten"], calls[3]["driver"]["gf_rthdynten"])
    assert np.allclose(calls[3]["driver"]["gf_rthdynten"], -1.0 / elapsed, rtol=1.0e-5, atol=0.0)

    # Without a cumulus scheme the lanes are not carried.
    bare = ArwenCudaColumnSuite(
        {**_options(), "cumulus": "none"}, array_module=np, modules=_fake_modules()
    )
    bare_result = bare.step(exchange)
    for name in CUMULUS_DYNAMICS_LANES:
        assert name not in bare_result.physics_state.arrays



@pytest.mark.parametrize("excess", [1.10, 0.90])
def test_convective_rain_is_booked_to_the_reservoir_rainc_and_the_land_bucket(excess):
    from woof.globe.constants import (
        CONVECTIVE_RAIN_ACCUMULATOR, GRAVITY_M_S2,
    )

    exchange = _exchange()
    dt = float(exchange.dt_s)
    dp_bottom = np.asarray(exchange.dp[-1], np.float32)
    drying = -2.0e-6
    # The scheme's own RAINCV, deliberately 10% off the column integral of
    # its drying, above it and below it: the reservoir must take the
    # MEASURED loss, and RAINC and the land bucket one number, the smaller
    # of the two (Grell-Freitas's deep arm reports more rain than its rates
    # remove: Noah forced with the excess grew stores out of water the
    # atmosphere never lost, paid by the reservoir, and the maps drew rain
    # that never left the air).
    reported = np.asarray(excess * (-drying) * dt * dp_bottom / GRAVITY_M_S2, np.float32)

    def convection(result, atmosphere):
        result.rqvcuten[0] = drying
        result.rainc[...] = reported

    calls = []
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np,
        modules=_fake_modules(cumulus=convection, noah_calls=calls),
    )
    reservoir0 = np.asarray(exchange.surface.water_kg_m2, np.float64).copy()
    rain0 = np.asarray(exchange.surface.accumulated_rain_kg_m2, np.float64).copy()
    result = suite.step(exchange)
    removed = _column_water(exchange, exchange.dp) - _column_water(result, exchange.dp)
    assert float(removed.min()) > 0.0
    credited = np.asarray(result.surface.water_kg_m2, np.float64) - reservoir0
    # Reservoir: what left the atmosphere, in the model's metric, not the
    # kernel's number (float32 casting of the returned humidities is the
    # ~1e-5 kg/m2 noise floor of this comparison; the 10% disagreement is
    # 4e-3).
    assert np.allclose(credited, removed, atol=2.0e-5)
    assert not np.allclose(credited, reported, atol=1.0e-3)
    assert result.diagnostics["maximum_local_water_repair_kg_m2"] < 1.0e-5
    # RAINC: the reported rain capped at the water that left the column, the
    # same number the land bucket takes; RAINNC untouched.
    landed = np.minimum(np.asarray(reported, np.float64), removed)
    rainc = np.asarray(result.physics_state.arrays[CONVECTIVE_RAIN_ACCUMULATOR])
    assert np.allclose(rainc, landed, rtol=0.0, atol=2.0e-5)
    if excess > 1.0:
        assert np.all(rainc < reported)
    else:
        assert np.allclose(rainc, reported, rtol=1.0e-6)
    assert np.array_equal(
        np.asarray(result.surface.accumulated_rain_kg_m2, np.float64), rain0
    )
    # The call's rain books: what the scheme reported, what it reported
    # beyond the removed water (withheld, never rain), and the removal.
    d = result.diagnostics
    assert np.isclose(d["cumulus_rain_reported_kg_m2"], float(np.mean(reported)), rtol=1.0e-6)
    assert np.isclose(d["cumulus_rain_withheld_kg_m2"],
                      float(np.mean(np.asarray(reported, np.float64) - rainc)), rtol=1.0e-5, atol=1.0e-9)
    assert np.isclose(d["cumulus_water_removed_kg_m2"], float(np.mean(removed)), rtol=1.0e-4)
    if excess > 1.0:
        assert d["cumulus_rain_withheld_kg_m2"] > 0.05 * d["cumulus_rain_reported_kg_m2"]
    else:
        assert d["cumulus_rain_withheld_kg_m2"] == 0.0
    # The land bucket: Noah's next due call is forced with that same number,
    # and the accumulator keeps growing by it.
    second = suite.step(_advance(exchange, result, 10.0))
    assert len(calls) == 2
    assert np.allclose(calls[0]["rainbl"], 0.0)
    assert np.allclose(calls[1]["rainbl"], landed, rtol=0.0, atol=2.0e-5)
    assert np.allclose(calls[1]["rainbl"], rainc, rtol=0.0, atol=1.0e-7)
    if excess > 1.0:
        assert np.all(calls[1]["rainbl"] < reported)
    else:
        assert np.allclose(calls[1]["rainbl"], reported, rtol=1.0e-6)
    assert np.allclose(
        np.asarray(second.physics_state.arrays[CONVECTIVE_RAIN_ACCUMULATOR]),
        2.0 * rainc, rtol=0.0, atol=4.0e-5,
    )
    assert second.diagnostics["maximum_local_water_repair_kg_m2"] < 1.0e-5


def test_water_ledger_closes_through_a_cumulus_call_that_dries_and_detrains():
    # Drying at the bottom, detrainment of liquid and ice aloft, rain to the
    # surface: the suite's before/after column must still close on the
    # atmosphere + reservoir total with no repair beyond float32 casting.
    def convection(result, atmosphere):
        result.rqvcuten[0] = -3.0e-6
        result.rqccuten[1] = 4.0e-7
        result.rqicuten[2] = 1.0e-7
        result.rthcuten[0] = 2.0e-3
        result.rainc[...] = 1.0e-2

    exchange = _exchange()
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules(cumulus=convection)
    )
    total0 = _column_water(exchange, exchange.dp) + np.asarray(
        exchange.surface.water_kg_m2, np.float64
    )
    result = suite.step(exchange)
    total1 = _column_water(result, exchange.dp) + np.asarray(
        result.surface.water_kg_m2, np.float64
    )
    assert np.allclose(total1, total0, atol=1.0e-5)
    assert result.diagnostics["maximum_native_water_residual_kg_m2"] < 1.0e-5
    assert result.diagnostics["native_water_residual_exceeds_tolerance"] == 0.0


def test_cumulus_none_is_inert_and_drops_gf_from_the_order():
    from woof.globe.constants import CONVECTIVE_RAIN_ACCUMULATOR
    from woof.globe.physics.native_suite import native_component_order

    class Refusing:
        def __init__(self):
            raise AssertionError("cumulus='none' must never construct the scheme")

    def convection(result, atmosphere):
        result.rthcuten[...] = 1.0
        result.rainc[...] = 5.0

    exchange = _exchange()
    modules = _fake_modules(cumulus=convection)
    modules["woof.globe.core.gf"] = SimpleNamespace(GrellFreitas=Refusing)
    options = {**_options(), "cumulus": "none"}
    suite = ArwenCudaColumnSuite(options, array_module=np, modules=modules)
    result = suite.step(exchange)
    assert result.diagnostics["cumulus_active"] is False
    assert result.physics_state.metadata["cumulus_updates"] == 0
    assert np.all(np.asarray(result.physics_state.arrays[CONVECTIVE_RAIN_ACCUMULATOR]) == 0.0)
    assert "gf" not in suite.identity["component_order"]
    assert native_component_order(suite.options) == (
        "rrtmgp", "sfclay", "noah", "ysu", "morrison"
    )
    # Bit-for-bit the run a zero-rate scheme produces.
    reference = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules()
    ).step(exchange)
    for name in ("theta", *WATER_SPECIES):
        assert np.array_equal(getattr(result, name), getattr(reference, name)), name
    assert np.array_equal(result.surface.water_kg_m2, reference.surface.water_kg_m2)


def test_the_dry_downdraft_switch_is_read_only_under_grell_freitas():
    """True joins the identity under cumulus="gf" only; False (the
    default) is the identity every earlier Grell-Freitas checkpoint
    carried, and the flag is not part of another scheme's identity at
    all."""
    from woof.globe.physics.native_options import NativePhysicsOptions

    faithful = NativePhysicsOptions.from_mapping(_options())
    assert faithful.gf_updraft_only_when_downdraft_dry is False
    assert "gf_updraft_only_when_downdraft_dry" not in faithful.identity
    default = NativePhysicsOptions.from_mapping(
        {**_options(), "gf_updraft_only_when_downdraft_dry": True}
    )
    assert default.identity["gf_updraft_only_when_downdraft_dry"] is True
    for scheme in ("ntiedtke", "own", "none"):
        other = NativePhysicsOptions.from_mapping({**_options(), "cumulus": scheme})
        assert "gf_updraft_only_when_downdraft_dry" not in other.identity, scheme
    assert {k: v for k, v in default.identity.items() if k != "gf_updraft_only_when_downdraft_dry"} == faithful.identity
    with pytest.raises(ValueError, match="gf_updraft_only_when_downdraft_dry must be true or false"):
        NativePhysicsOptions.from_mapping(
            {**_options(), "gf_updraft_only_when_downdraft_dry": 1}
        )


def test_cumulus_options_are_fail_closed():
    from woof.globe.physics.native_options import NativePhysicsOptions

    with pytest.raises(ValueError, match="cumulus='gf'.*cumulus='none'"):
        NativePhysicsOptions.from_mapping({**_options(), "cumulus": "kf"})
    with pytest.raises(ValueError, match="gf_ishallow must be 0 or 1"):
        NativePhysicsOptions.from_mapping({**_options(), "gf_ishallow": 2})
    with pytest.raises(ValueError, match="gf_ishallow must be 0 or 1"):
        NativePhysicsOptions.from_mapping({**_options(), "gf_ishallow": True})
    options = NativePhysicsOptions.from_mapping(_options())
    assert options.cumulus == "gf" and options.cumulus_enabled
    assert options.gf_ishallow == 1
    assert not NativePhysicsOptions.from_mapping(
        {**_options(), "cumulus": "none"}
    ).cumulus_enabled


def test_cumulus_column_chunk_is_validated_and_reaches_the_scheme():
    from woof.globe.physics.native_options import NativePhysicsOptions

    for bad in (0, -5, True, 2.5):
        with pytest.raises(ValueError, match="cumulus_column_chunk"):
            NativePhysicsOptions.from_mapping(
                {**_options(), "cumulus_column_chunk": bad}
            )
    assert NativePhysicsOptions.from_mapping(_options()).cumulus_column_chunk == 131_072
    suite = ArwenCudaColumnSuite(
        {**_options(), "cumulus_column_chunk": 4_096},
        array_module=np, modules=_fake_modules(),
    )
    suite.step(_exchange())
    assert suite._runtime._cumulus.column_chunk == 4_096


def test_cumulus_refuses_an_exchange_without_vertical_velocity():
    exchange = replace(_exchange(), omega_half_pa_s=None)
    suite = ArwenCudaColumnSuite(
        _options(), array_module=np, modules=_fake_modules()
    )
    with pytest.raises(ValueError, match="w = 0"):
        suite.step(exchange)
    # cumulus='none' has no such need.
    ArwenCudaColumnSuite(
        {**_options(), "cumulus": "none"}, array_module=np,
        modules=_fake_modules(),
    ).step(exchange)


def test_cumulus_result_without_rates_is_refused():
    modules = _fake_modules()

    class NoRates(modules["woof.globe.core.gf"].GrellFreitas):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result.rqvcuten = None
            return result

    modules["woof.globe.core.gf"] = SimpleNamespace(GrellFreitas=NoRates)
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=modules)
    with pytest.raises(ValueError, match="rqvcuten"):
        suite.step(_exchange())


def test_the_registration_names_the_cumulus_scheme_and_the_stale_evidence():
    from woof.globe.physics.arwen_bridge import NativeArwenPhysicsBridge
    from woof.globe.physics.builtin_adapters import ADAPTER_NAME

    bridge = NativeArwenPhysicsBridge(ADAPTER_NAME, _options())
    contract = bridge.registration.contract
    assert "Grell-Freitas" in contract["scheme_identity"]["cumulus"]
    assert contract["admission_status"] == "device-pending"
    assert contract["device_evidence_sha256"] == "0" * 64
    assert any(
        "re-run" in row and "c682dbcc73ea544af04d4892cbeea91af08dc8f7" in row
        for row in contract["limitations"]
    )
    assert not any(row == "no cumulus scheme" for row in contract["limitations"])


@requires_netcdf_writer
def test_export_rainc_carries_the_convective_accumulator(tmp_path):
    from woof.globe.checkpoint import write_checkpoint
    from woof.globe.constants import CONVECTIVE_RAIN_ACCUMULATOR
    from woof.globe.wrfout_export import export_wrfout

    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    state.physics_state.arrays[CONVECTIVE_RAIN_ACCUMULATOR] = np.full(
        model.transform.grid.shape, 2.5, np.float32
    )
    checkpoint = write_checkpoint(
        tmp_path / "run" / "checkpoint.npz", state,
        config_hash=cfg.config_hash, to_numpy=model.transform.backend.to_numpy,
        semi_implicit_scheme=cfg.semi_implicit_scheme, integrator=cfg.integrator,
    )
    tapes = export_wrfout(
        cfg, [checkpoint], tmp_path / "tapes", nlat=18, nlon=36,
        start_date="2026-08-30_18:00:00",
    )
    from netCDF4 import Dataset

    with Dataset(tapes[0]) as ds:
        rainc = np.asarray(ds["RAINC"][0])
        rainnc = np.asarray(ds["RAINNC"][0])
    assert np.allclose(rainc, 2.5)
    assert np.all(rainnc == 0.0)


def test_land_columns_publish_wrf_sfcdiags_after_noah_and_water_keeps_the_surface_layer():
    """native_runtime._screen_level_step: on land T2 = TSK - HFX / (rho cp
    CQS2) and Q2 = QSFC - QFX / (rho CQS2) with rho = PSFC / (R_d TSK)
    (module_sf_sfcdiags.F:45-72, CHS2 = CQS2 per module_sf_noahdrv.F:1275),
    converted to specific humidity on the way out; water columns keep the
    surface layer's own diagnostics; a flux inversion that leaves the
    physical range publishes the lowest level's vapor instead."""
    from woof.core import constants as c

    modules = _fake_modules()
    stock = modules["woof.globe.core.sfclay"].sfclay
    planted = {"hfx": 100.0, "qfx": 5.0e-5, "qsfc": 0.012, "cqs2": 0.02}

    def sfclay(*args, **kwargs):
        out = stock(*args, **kwargs)
        shape = out.hfx.shape
        for name, value in planted.items():
            setattr(out, name, np.full(shape, value, np.float32))
        return out

    modules["woof.globe.core.sfclay"] = SimpleNamespace(sfclay=sfclay)
    exchange = _exchange()
    suite = ArwenCudaColumnSuite(_options(), array_module=np, modules=modules)
    result = suite.step(exchange)
    arrays = result.physics_state.arrays
    land = (1.0 + (1.0 - np.asarray(exchange.surface.land_fraction, np.float64))) < 1.5
    water = ~land
    assert land.any() and water.any()
    tsk = np.asarray(result.surface.temperature_k, np.float64)
    psfc = np.asarray(exchange.p_half[-1], np.float64)
    rho = psfc / (c.RD * tsk)
    t2 = tsk - planted["hfx"] / (rho * c.CP * planted["cqs2"])
    q_mix = planted["qsfc"] - planted["qfx"] / (rho * planted["cqs2"])
    assert np.all(q_mix > 0.0)
    assert np.allclose(arrays["t2"][land], t2[land], rtol=1.0e-5, atol=1.0e-3)
    assert np.allclose(arrays["th2"][land], (t2 * (c.P0 / psfc) ** c.RCP)[land], rtol=1.0e-5, atol=1.0e-3)
    assert np.allclose(arrays["q2"][land], (q_mix / (1.0 + q_mix))[land], rtol=1.0e-5, atol=0.0)
    # The land value is not the surface layer's, and the water value is.
    lowest_t = np.asarray(exchange.temperature[-1], np.float64)
    assert not np.allclose(arrays["t2"][land], (lowest_t + SFCLAY_T2_OFFSET)[land], atol=0.05)
    assert np.allclose(arrays["t2"][water], (lowest_t + SFCLAY_T2_OFFSET)[water], atol=1.0e-4)

    # The divergence: a downward moisture flux large enough to drive the
    # inversion negative publishes the lowest level's vapor on those columns.
    planted.update({"qsfc": 0.001, "qfx": -2.0e-3})
    result = suite.step(exchange)
    arrays = result.physics_state.arrays
    q_mix = planted["qsfc"] - planted["qfx"] / (rho * planted["cqs2"])
    assert np.all(q_mix > 0.0)   # dew ADDS to the surface value: still representable
    planted.update({"qsfc": 0.0, "qfx": 2.0e-3})
    q_mix = planted["qsfc"] - planted["qfx"] / (rho * planted["cqs2"])
    assert np.all(q_mix < 0.0)
    result = suite.step(exchange)
    arrays = result.physics_state.arrays
    lowest_q = np.asarray(exchange.qv[-1], np.float64)
    assert np.allclose(arrays["q2"][land], lowest_q[land], rtol=1.0e-6, atol=0.0)
