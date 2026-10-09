"""Water-budget sustainability regressions for the Level-5 global model.

The h37.7 conviction (T255 five-scheme native suite, 2026-08-31): after
passing its 24 h day and the qualification battery, the 96 h run died at
hour 37.7 on the suite's hard reservoir floor -- 'native physics water
closure exceeds the explicit surface reservoir' -- with the worst column
draining ~13 kg/m2/h.  CPU diagnosis measured the drain 100% in the
DYNAMICS-side spectral clamp closures (the positivity repair's projection
delta and the exchange clamp), which debited each ring column's reservoir
for the water created re-clamping analysis ringing around sharp
hydrometeor features, every step, forever, with no return path; the
suspected runoff no-return debit was falsified (the microphysics
precipitation credit returns every priced-store kg each land interval, so
the sustained-rain drive PASSES under the pre-fix arithmetic).  Two arms
therefore guard two different failure surfaces:

* sustained heavy rain (>= 10 kg/m2/h on an lf=1 land column, > 100
  simulated hours of land-step applications): the reservoir stays bounded,
  the closure holds on every call, and the runoff leaves through the
  cumulative outflow account -- more water than the whole 500 kg/m2
  reservoir passes through it, which is why runoff must be a booked exit
  rather than a held store squeezing the pinned conservation total;

* a continuously-renewed sharp hydrometeor feature (the T255 convection
  signature: species at zero beside a heavy core, re-sharpened every
  step): under the v5 atmosphere-internal fixer (audit 2026-09-01 VTW-1)
  the clamps rang and fired every step and touched NO reservoir column;
  since 2026-09-02 the hydrometeors are grid tracers and do not ring at
  all, the vapor clamps close inside their own columns, and the ring
  column's reservoir still moves only by the global drift fixer's
  uniform correction, so the h37.7 death (a column-local clamp debit)
  and the v4 uniform levy (a 2.5 mm/day reservoir-to-atmosphere source)
  stay structurally impossible and the atmosphere's global water stays
  conserved.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from woof.globe.config import load_config
from woof.globe.constants import (
    NATIVE_PHYSICS_ACKNOWLEDGEMENT,
    WATER_SPECIES,
)
from woof.globe.physics.native_suite import ArwenCudaColumnSuite
from woof.globe.runner import build_model_and_cold_state
from woof.globe.water import (
    atmospheric_water_column,
    native_store_column,
    soil_water_column,
    water_outflow_column,
)

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")
F32 = np.float32


def _inert_radiation():
    class Radiation:
        def __init__(self, **kwargs):
            pass

        def __call__(self, *, atmosphere, fields, state, cfg):
            shape = atmosphere["temperature"].shape
            surface = shape[1:]
            return SimpleNamespace(
                rthratenlw=np.zeros(shape, F32),
                rthratensw=np.zeros(shape, F32),
                swdown=np.zeros(surface, F32),
                glw=np.full(surface, 300.0, F32),
                olr=np.full(surface, 236.0, F32),
                gsw=np.zeros(surface, F32),
                coszen=np.zeros(surface, F32),
            )

    return Radiation


def _inert_sfclay(*args, **kwargs):
    shape = args[0].shape
    zero = np.zeros(shape, F32)
    one = np.ones(shape, F32)
    return SimpleNamespace(
        znt=np.full(shape, 0.05, F32), ust=np.full(shape, 0.1, F32),
        mol=zero.copy(), hfx=zero.copy(), qfx=zero.copy(),
        qsfc=zero.copy(), zol=zero.copy(),
        wspd=np.full(shape, 0.1, F32), br=zero.copy(),
        fm=one.copy(), fh=one.copy(), u10=zero.copy(), v10=zero.copy(),
        pblh=np.full(shape, 800.0, F32),
        chs=zero.copy(), chs2=zero.copy(), cqs2=zero.copy(),
        qgh=zero.copy(),
    )


def _inert_ysu(u, v, theta, qv, qc, qi, *args, **kwargs):
    zeros = np.zeros_like(theta)
    return {
        "du": zeros.copy(), "dv": zeros.copy(), "dtheta": zeros.copy(),
        "dqv": zeros.copy(), "dqc": zeros.copy(), "dqi": zeros.copy(),
        "hpbl": np.full(theta.shape[1:], 800.0, F32),
    }


class _InertGrellFreitas:
    """Shaped like woof.globe.core.gf.GrellFreitas; zero rates, no rain."""

    def __init__(self, *, column_chunk=None,
                 updraft_only_when_downdraft_dry=False,
                 resolved_convergence_closure=False):
        self.column_chunk = column_chunk
        self.updraft_only_when_downdraft_dry = updraft_only_when_downdraft_dry
        self.resolved_convergence_closure = resolved_convergence_closure

    def bind_driver(self, driver):
        pass

    def __call__(self, *, atmosphere, fields, state, cfg):
        shape = atmosphere["temperature"].shape
        zeros = np.zeros(shape, F32)
        return SimpleNamespace(
            rthcuten=zeros.copy(), rqvcuten=zeros.copy(),
            rqccuten=zeros.copy(), rqicuten=zeros.copy(),
            rainc=np.zeros(shape[1:], F32),
        )


def _modules(launch_noah, launch_morrison):
    from woof.globe.core import noah as kernel_noah

    return {
        "woof.globe.core.gf": SimpleNamespace(GrellFreitas=_InertGrellFreitas),
        "woof.globe.core.rrtmgp": SimpleNamespace(RRTMGPRadiation=_inert_radiation()),
        "woof.globe.core.sfclay": SimpleNamespace(sfclay=_inert_sfclay),
        # The real loader and packer: the runtime checks the surface
        # state's category convention against the sections it loads.
        "woof.globe.core.noah": SimpleNamespace(
            _F2D=kernel_noah._F2D,
            load_tables=kernel_noah.load_tables,
            pack_params=kernel_noah.pack_params,
            launch_noah=launch_noah,
        ),
        "woof.globe.core.ysu": SimpleNamespace(launch_ysu=_inert_ysu),
        "woof.globe.core.morrison": SimpleNamespace(launch_morrison=launch_morrison),
    }


# --------------------------------------------------------------------------
# Arm 1: sustained heavy rain through the real suite closure, > 100 h.
# --------------------------------------------------------------------------

RAIN_PER_CALL = 10.0     # kg/m2 per one-hour land interval = 10 kg/m2/h
CALLS = 120              # 120 simulated hours of land-step applications
INTERVAL_S = 3600.0
CANWAT_CAP = 0.5         # CMCMAX * 1000, the kernel's canopy capacity


def _sustained_rain_modules(dp_bottom_up, target):
    """Noah consumes rainbl the way SFLX does at its simplest: canopy fills
    to the CMCMAX cap and everything else leaves as surface runoff -- a
    held destination and an exited destination, both booked.  Morrison
    removes exactly its reported precipitation from column vapor in the
    ledger's metric -- specific humidity times dp/g, the model's
    convention, converted at the kernel-side level through the bridge's
    own helpers (fallout is water that LEFT the atmosphere)."""
    from woof.globe.physics.native_batch import (
        mixing_ratio_from_specific_humidity,
        specific_humidity_from_mixing_ratio,
    )

    def launch_morrison(
        theta, qv, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
        rho, exner, p_full, dz,
        rainnc, rainncv, snownc, snowncv, graupelnc, graupelncv, sr, dt,
        **kwargs,
    ):
        rainncv.fill(0.0)
        snowncv.fill(0.0)
        graupelncv.fill(0.0)
        sr.fill(0.0)
        j, i = target
        g = np.float64(9.80665)
        remaining = np.float64(RAIN_PER_CALL)
        species = {"qv": qv, "qc": qc, "qr": qr, "qi": qi, "qs": qs, "qg": qg}
        for k in range(qv.shape[0]):
            dp_k = np.float64(dp_bottom_up[k, j, i])
            specific = specific_humidity_from_mixing_ratio(
                {name: np.float64(arr[k, j, i]) for name, arr in species.items()}
            )
            take = min(specific["qv"] * dp_k / g, remaining)
            if take <= 0.0:
                continue
            specific["qv"] = specific["qv"] - take * g / dp_k
            for name, value in mixing_ratio_from_specific_humidity(specific).items():
                species[name][k, j, i] = F32(value)
            remaining -= take
            if remaining <= 1.0e-9:
                break
        rainncv[j, i] = F32(np.float64(RAIN_PER_CALL) - remaining)

    def launch_noah(dev, params, dt, dzs, **kwargs):
        xland = np.asarray(dev["xland"], F32)
        land = (xland - F32(1.5)) < 0.0
        rainbl = np.asarray(dev["rainbl"], F32)
        liquid = np.where(land, rainbl, F32(0.0)).astype(F32)
        intercepted = np.minimum(
            liquid, np.maximum(F32(CANWAT_CAP) - dev["canwat"], F32(0.0))
        ).astype(F32)
        dev["canwat"][...] = (dev["canwat"] + intercepted).astype(F32)
        dev["sfcrunoff"][...] = (
            dev["sfcrunoff"] + (liquid - intercepted)
        ).astype(F32)

    return _modules(launch_noah, launch_morrison)


def _conservation_total(fields, surface, physics_state):
    """The v4 conservation statement: held water plus booked exits."""
    atmosphere = np.asarray(
        atmospheric_water_column(fields), np.float64
    ).sum(axis=0)
    like = surface.water_kg_m2
    return (
        atmosphere
        + np.asarray(surface.water_kg_m2, np.float64)
        + np.asarray(soil_water_column(surface, np), np.float64)
        + np.asarray(
            native_store_column(physics_state, like, np), np.float64
        )
        + np.asarray(
            water_outflow_column(physics_state, like, np), np.float64
        )
    )


def _water_fields(source, dp):
    return {
        "dp": np.asarray(dp, np.float64),
        **{
            name: np.asarray(getattr(source, name), np.float64)
            for name in WATER_SPECIES
        },
    }


def test_sustained_heavy_rain_reservoir_bounded_and_runoff_booked_as_outflow():
    cfg = load_config(CONFIG)
    model, state = build_model_and_cold_state(cfg)
    lf = np.asarray(state.surface.land_fraction)
    land = np.argwhere((1.0 + (1.0 - lf)) < 1.5)
    assert len(land) >= 1
    j, i = int(land[0][0]), int(land[0][1])
    # The conviction's kill-column profile is full land: rain bookings are
    # neutral there (no fractional-lf protection credit), so any unbooked
    # drain would show against a flat reservoir.
    lf[j, i] = 1.0

    exchange0 = model._physics_exchange(state, INTERVAL_S)
    dp_bottom_up = np.asarray(exchange0.dp, F32)[::-1].copy()
    suite = ArwenCudaColumnSuite(
        {
            "acknowledgement": NATIVE_PHYSICS_ACKNOWLEDGEMENT,
            "start_time_utc": "2024-05-21T00:00:00Z",
            "radiation_interval_s": INTERVAL_S,
            "land_surface_interval_s": INTERVAL_S,
        },
        array_module=np,
        modules=_sustained_rain_modules(dp_bottom_up, (j, i)),
    )

    reservoir0 = float(np.asarray(exchange0.surface.water_kg_m2)[j, i])
    moist_qv = np.asarray(exchange0.qv, np.float64).copy()
    moist_qv[:, j, i] = 0.02  # ~200 kg/m2 of resupplied column vapor
    exchange = replace(exchange0, qv=moist_qv)

    worst_repair = 0.0
    worst_call_residual = 0.0
    reservoir_series = []
    result = None
    for call in range(CALLS):
        before = _conservation_total(
            _water_fields(exchange, exchange.dp),
            exchange.surface, exchange.physics_state,
        )
        result = suite.step(exchange)
        after = _conservation_total(
            _water_fields(result, exchange.dp),
            result.surface, result.physics_state,
        )
        # The closure holds with the outflow term booked: the suite's own
        # repair AND the runner-side accounting authority agree, every call.
        worst_repair = max(
            worst_repair,
            float(result.diagnostics["maximum_local_water_repair_kg_m2"]),
        )
        worst_call_residual = max(
            worst_call_residual, float(np.max(np.abs(after - before)))
        )
        reservoir_series.append(
            float(np.asarray(result.surface.water_kg_m2)[j, i])
        )
        # Next call: rebuild the exchange from the result and resupply the
        # rained-out vapor (the advection the single-column drive lacks).
        qv = np.asarray(result.qv, np.float64).copy()
        qv[:, j, i] = 0.02
        exchange = replace(
            exchange,
            time_s=(call + 1) * INTERVAL_S,
            u=result.u, v=result.v, theta=result.theta,
            qv=qv, qc=result.qc, qr=result.qr,
            qi=result.qi, qs=result.qs, qg=result.qg,
            nc=result.nc, nr=result.nr, ni=result.ni,
            ns=result.ns, ng=result.ng,
            surface=result.surface,
            physics_state=result.physics_state,
        )

    assert worst_repair < 1.0e-3
    assert worst_call_residual < 2.0e-3
    # 120 h of 10 kg/m2/h on a finite 500 kg/m2 reservoir: bounded means
    # the reservoir CYCLES -- each call ends holding at most one land
    # interval's credit above start (the microphysics credit the next land
    # step consumes) -- and never trends.  The post-fill series is flat to
    # fp32 noise: an unbooked drain of even 1% of the rain rate would move
    # it 12 kg/m2 across the run.
    series = np.asarray(reservoir_series)
    assert float(series.max() - reservoir0) < RAIN_PER_CALL + 1.5
    assert float(reservoir0 - series.min()) < 1.5
    assert abs(float(series[-1] - series[20])) < 1.0
    # The rain went somewhere real: ~1200 kg/m2 -- 2.4x the entire explicit
    # reservoir -- passed through the column into the outflow account, which
    # reconciles against the priced kernel accumulators exactly.  Held as
    # store content (the v3 ledger) this water squeezed the pinned
    # conservation total; booked as exits it leaves the budget open the way
    # rivers do.
    arrays = result.physics_state.arrays
    outflow = float(np.asarray(arrays["water_outflow_kg_m2"])[j, i])
    priced_runoff = float(
        (np.maximum(np.asarray(arrays["noah_sfcrunoff"]), 0.0)
         + np.maximum(np.asarray(arrays["noah_udrunoff"]), 0.0))[j, i]
        * lf[j, i]
    )
    total_rain = RAIN_PER_CALL * (CALLS - 1)  # the last call's rain is queued
    assert outflow == pytest.approx(priced_runoff, abs=1.0e-2)
    assert outflow == pytest.approx(total_rain - CANWAT_CAP, abs=1.0)
    assert outflow > 2.0 * reservoir0
    # Held native stores stay at the canopy cap; the exited water is not
    # double-counted as held content.
    held = float(
        np.asarray(
            native_store_column(
                result.physics_state, result.surface.water_kg_m2, np
            )
        )[j, i]
    )
    assert held == pytest.approx(CANWAT_CAP, abs=1.0e-3)


# --------------------------------------------------------------------------
# Arm 2: continuously-renewed sharp feature vs the clamp-closure payer.
# --------------------------------------------------------------------------

T21_TOML = """
[arwen_global]
schema = "gpuwm.arwen-global-run/v1"
name = "water-sustainability-t21"
acknowledgement = "research-only-arwen-global-v1"
backend = "numpy"
precision = "float32"

[grid]
truncation = 21
dealias_factor = 1.5

[time]
dt_s = 60.0
duration_s = 120.0
output_interval_s = 60.0
maximum_cfl = 0.90
integrator = "ssprk3"

[vertical]
coordinate = "pressure_blend"
nlev = 5
p_top_pa = 100.0

[initial]
surface_pressure_pa = 100000.0
surface_temperature_k = 286.0
top_temperature_k = 220.0
qv_surface = 0.006
zonal_wind_m_s = 2.0
perturbation_amplitude = 0.00001
zonal_wavenumber = 2
terrain_amplitude_m = 25.0
surface_water_kg_m2 = 500.0

[physics]
mode = "reference"

[diffusion]
enabled = true
order = 4
e_folding_time_s_at_truncation = 21600.0
preserve_degree = 1

[semi_implicit]
enabled = true
weight = 0.5

[repair]
mass_fixer = true
water_fixer = true
positivity_repair = true

[gates]
transform_roundtrip_relative_linf = 5.0e-5
transform_parseval_relative_error = 5.0e-5
mass_relative_drift = 1.0e-7
total_water_relative_drift = 1.0e-6
"""

BLOB_AMPLITUDE = 8.0e-3  # kg/kg of rain water at one cell of one level
BLOB_LEVEL = 1


def _sharp_feature_modules(blob):
    """Morrison keeps a one-cell hydrometeor spike eternally fresh: every
    call it returns ALL rain water to vapor in its own cell (per-cell
    conserving, so the suite closure sees residual ~0) and rebuilds the
    spike from the cell's vapor.  The zero-beside-core discontinuity the
    spectral analysis rings against is therefore renewed every step, the
    T255 convection signature; a static blob's ring fee decays once the
    surrounding field saturates."""

    def launch_noah(dev, params, dt, dzs, **kwargs):
        pass

    def launch_morrison(theta, qv, qc, qr, qi, qs, qg, nc, nr, ni, ns, ng,
                        rho, exner, p_full, dz,
                        rainnc, rainncv, snownc, snowncv,
                        graupelnc, graupelncv, sr, dt, **kwargs):
        rainncv.fill(0.0)
        snowncv.fill(0.0)
        graupelncv.fill(0.0)
        sr.fill(0.0)
        k, j, i = BLOB_LEVEL, blob[0], blob[1]
        qv += qr
        qr.fill(0.0)
        move = min(float(BLOB_AMPLITUDE), float(qv[k, j, i]) * 0.5)
        qv[k, j, i] = F32(np.float64(qv[k, j, i]) - move)
        qr[k, j, i] = F32(move)

    return _modules(launch_noah, launch_morrison)


def _blob_model(config_path):
    cfg = load_config(str(config_path))
    model, state = build_model_and_cold_state(cfg)
    lf = np.asarray(state.surface.land_fraction, F32)
    ones = np.argwhere(lf == 1.0)
    assert len(ones) >= 1
    blob = (int(ones[0][0]), int(ones[0][1]))
    model.physics = ArwenCudaColumnSuite(
        {
            "acknowledgement": NATIVE_PHYSICS_ACKNOWLEDGEMENT,
            "start_time_utc": "2026-08-30T18:00:00Z",
            "radiation_interval_s": 3120.0,
            "land_surface_interval_s": 60.0,
            "dx_m": 52100.0,
        },
        array_module=np,
        modules=_sharp_feature_modules(blob),
    )
    return model, state, blob


@pytest.fixture(scope="module")
def t21_config(tmp_path_factory):
    path = tmp_path_factory.mktemp("water-sustainability") / "t21_blob.toml"
    path.write_text(T21_TOML, encoding="utf-8")
    return path


def test_sharp_feature_never_rings_and_the_clamps_never_touch_the_reservoir(t21_config):
    """The h37.7 death and the v4 levy are both impossible by
    construction, and so is the ringing that fed them: across 200 steps
    of the eternally re-sharpened one-cell rain spike the rain field is
    a grid tracer that never goes below zero, the vapor clamps close
    inside their own columns, and the ring column's reservoir moves by
    exactly the global drift fixer's uniform correction and nothing
    else, while the atmosphere's global water is conserved.

    Under the spectral-tracer era the same drive rang the rain field to
    -1.9e-5 kg/kg beside the 8e-3 core every step and the clamps created
    2.2e-2 kg/m2 of global-mean column water over 200 steps (per-level
    rescales up to 0.87), moved out of every rain column on the planet.
    """
    model, state, blob = _blob_model(t21_config)
    nlat, nlon = model.transform.grid.shape
    # The reservoir probe column: one cell away from the spike, where the
    # spectral era's ringing was deepest.
    jw, iw = blob[0], (blob[1] + 1) % nlon
    reservoir = np.asarray(state.surface.water_kg_m2)
    reservoir[jw, iw] = reservoir.dtype.type(1.0)
    model.initialize_water_target(state)
    target = model.diagnostics(state)["global_mean_total_water_kg_m2"]

    created_cumulative = 0.0
    rescale_worst = 0.0
    drift_fixer_worst = 0.0
    rain_min = 0.0
    rain_in_core = 0.0
    unexplained = np.zeros(reservoir.shape, np.float64)
    for _ in range(200):
        res_before = np.asarray(state.surface.water_kg_m2, np.float64).copy()
        state, metrics = model.step(state, 60.0)
        res_after = np.asarray(state.surface.water_kg_m2, np.float64)
        created_cumulative += metrics["positivity_fixer_water_kg_m2"]
        rescale_worst = max(rescale_worst, metrics["positivity_fixer_max_rescale"])
        drift = float(metrics["global_water_fixer_kg_m2"])
        drift_fixer_worst = max(drift_fixer_worst, abs(drift))
        qr = np.asarray(state.atmosphere.qr, np.float64)
        rain_min = min(rain_min, float(qr.min()))
        # BLOB_LEVEL is the kernel's bottom-up index; the model is top-down.
        rain_in_core = max(rain_in_core, float(qr[qr.shape[0] - 1 - BLOB_LEVEL, blob[0], blob[1]]))
        # The inert suite evaporates and rains nothing, so the only
        # legitimate reservoir change is the drift fixer's uniform
        # correction; anything beyond it at a column is a clamp debit.
        unexplained = np.maximum(
            unexplained, np.abs((res_after - res_before) - drift)
        )

    # The drive really re-sharpens the spike every step (vapor-limited to
    # half the cell's vapor, 7e-4 kg/kg here), and the grid rain field
    # never rings: its minimum is exactly zero.
    assert 0.0 < rain_in_core <= BLOB_AMPLITUDE
    assert rain_min == 0.0
    # Whatever the vapor clamps closed, they closed inside a column: no
    # column paid anything beyond the uniform drift correction.  The blob
    # column is excluded: there the fake Morrison's own vapor/rain
    # exchange is booked by the bridge as a measured physics change,
    # which is physics, not a clamp.  Everywhere else the bound is the
    # fp32 rounding of a 500 kg/m2 reservoir (2^-14 = 6.1e-5), and at the
    # 1 kg/m2 probe column the fp32 quantum is 1.2e-7.
    others = np.ones(unexplained.shape, bool)
    others[blob] = False
    assert float(unexplained[others].max()) < 2.0e-4
    assert float(unexplained[jw, iw]) < 1.0e-6
    # The column the v3 debit killed in 13 steps holds its whole preset.
    survivor = float(np.asarray(state.surface.water_kg_m2)[jw, iw])
    assert abs(survivor - 1.0) < 1.0e-2
    # The vapor fixer is bounded: a column can lose at most its own
    # ringing share, and the clamps' global take is a measured number
    # (zero when the vapor does not ring).
    assert 0.0 <= rescale_worst < 1.0
    assert created_cumulative >= 0.0
    # Global conservation: the drift fixer absorbs only transport and
    # projection residuals, and the run-level drift stays at the config's
    # 1e-6 gate scale.
    assert drift_fixer_worst < 1.0e-3
    final = model.diagnostics(state)["global_mean_total_water_kg_m2"]
    assert abs(final - target) / abs(target) < 1.0e-6


def test_water_positivity_repair_conserves_every_column_of_vapor(t21_config):
    """The v6 fixer's contract, checked on the repair: the vapor clip is
    closed inside each column (every column's vapor integral is unchanged
    by clip and fill), the grid tracers pass through bit-identical, and
    the surface is bit-identical."""
    model, state, _blob = _blob_model(t21_config)
    for _ in range(3):
        state, _metrics = model.step(state, 60.0)
    transform = model.transform
    g = model.grid_state(state.atmosphere, only=("qv", "dp"))
    dp = np.asarray(g["dp"], np.float64)
    qv = np.array(g["qv"], np.float64)
    # Ring the vapor for certain: a one-cell hole in the spectral field.
    nlat, nlon = qv.shape[1:]
    qv[2, nlat // 3, nlon // 4] = -6.0 * qv[2, nlat // 3, nlon // 4]
    state.atmosphere.qv = transform.project(
        transform.forward(model.transform.backend.asarray(qv, dtype=F32))
    )
    raw = np.asarray(model.grid_state(state.atmosphere, only=("qv",))["qv"], np.float64)
    assert raw.min() < 0.0
    tracers = {name: np.array(v, copy=True) for name, v in state.atmosphere.grid_tracers().items()}

    filled, created, max_fraction, unfillable = model._fill_column_holes(
        model.transform.backend.asarray(raw, dtype=F32), model.transform.backend.asarray(dp, dtype=F32)
    )
    before = np.sum(raw * dp, axis=0)
    after = np.sum(np.asarray(filled, np.float64) * dp, axis=0)
    fillable = before > 0.0
    assert created > 0.0
    assert unfillable == 0.0
    assert 0.0 < max_fraction < 1.0
    np.testing.assert_allclose(after[fillable], before[fillable], rtol=2.0e-6, atol=0.0)
    assert float(np.min(np.asarray(filled))) >= 0.0

    repaired, negative_water, negative_tracer, fixer = model._repair_positivity(state)
    assert negative_water > 0.0
    assert negative_tracer == 0.0
    assert fixer["water_kg_m2"] == pytest.approx(created, rel=1.0e-6)
    for name, value in tracers.items():
        np.testing.assert_array_equal(np.asarray(getattr(repaired.atmosphere, name)), value)
    assert np.array_equal(
        np.asarray(repaired.surface.water_kg_m2),
        np.asarray(state.surface.water_kg_m2),
    )
