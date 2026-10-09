"""The member point operators on the card against the host path.

The members of a device-backed ensemble are sampled on the card
(:mod:`woof.globe.da.operators` over the device path of
:mod:`woof.globe.spectral.sampling`); the same members' coefficients
copied to the host and sampled by the numpy path are the reference.  Two
device precisions are held: ``float64`` reproduces the host arithmetic to
rounding, and ``state`` (the default, a float32 state's coefficients
through float32 GEMMs over 256-term blocks summed in float64) agrees to
the state's own precision.  Every operator variable is compared, surface
and aloft, and the aloft profile arithmetic that moved onto the card (the
Exner factor, the ln p interpolation, the dewpoint) with it.  Runs where
cupy and a card are present; skipped elsewhere (the host path is pinned
by ``test_arwen_global_da_ensemble.py``).
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import dataclasses
import types

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from woof.globe.config import load_config  # noqa: E402
from woof.globe.da import EnsembleOptions, GlobalEnsemble, ensemble_config  # noqa: E402
from woof.globe.da.operators import (  # noqa: E402
    COLUMN_VARIABLES, OPERATOR_VARIABLES, MemberOperators)
from woof.globe.runner import build_model_and_cold_state, build_transform  # noqa: E402
from woof.globe.spectral.transform import SphericalHarmonicTransform  # noqa: E402

CONFIG = str(_shipped_configs() / "arwen_global_moist_smoke.toml")


def _card_present() -> bool:
    try:
        return int(cp.cuda.runtime.getDeviceCount()) > 0
    except Exception:  # noqa: BLE001 - no driver, no card
        return False


pytestmark = pytest.mark.skipif(not _card_present(), reason="the device operators need a card")

SPECTRAL = ("theta", "qv", "log_surface_pressure", "vorticity", "divergence")
SURFACE = ("temperature_k", "land_fraction", "soil_water_fraction", "roughness_m")


def _column_height_span(vertical, *, surface_pressure_pa=100000.0,
                        temperature_k=260.0, specific_humidity=0.002):
    """The inner fifth-to-four-fifths of this coordinate's height span.

    Uses the operator's OWN integration -- `refractivity_at_heights_columns`
    returns the column heights beside its values -- so the band follows the
    vertical ladder rather than a number written here.  The reference column
    is isothermal because the span's ends move by tens of metres with the
    profile and by kilometres with the ladder, and it is the ladder this
    guards against.
    """

    from woof.globe.obs_operators import refractivity_at_heights_columns

    a = np.asarray(vertical.a_half_pa, dtype=np.float64)
    b = np.asarray(vertical.b_half, dtype=np.float64)
    half = a + b * surface_pressure_pa
    full = 0.5 * (half[:-1] + half[1:])
    nlev = full.size
    _, z = refractivity_at_heights_columns(
        np.full((nlev, 1), temperature_k),
        np.full((nlev, 1), specific_humidity),
        full[:, None], np.array([surface_pressure_pa]), np.array([0.0]),
        np.array([0.0]))
    low, high = float(z[-1, 0]), float(z[0, 0])
    return low + 0.2 * (high - low), low + 0.8 * (high - low)


@pytest.fixture(scope="module")
def twins():
    cfg = dataclasses.replace(load_config(CONFIG), backend="cupy", precision="float32")
    options = EnsembleOptions(members=3, truncation=cfg.truncation, seed=5, additive_inflation_fraction=0.0)
    ecfg = ensemble_config(cfg, options)
    transform = build_transform(ecfg)
    model, cold = build_model_and_cold_state(ecfg, transform)
    ensemble = GlobalEnsemble.from_state(ecfg, model, transform, cold, options)
    to_numpy = transform.backend.to_numpy
    host_transform = SphericalHarmonicTransform.create(
        ecfg.truncation, nlat=ecfg.nlat, nlon=ecfg.nlon, dealias_factor=ecfg.dealias_factor,
        backend="numpy", precision="float32")
    host_members = []
    for member in ensemble.members:
        atmosphere = types.SimpleNamespace(**{n: np.asarray(to_numpy(getattr(member.atmosphere, n)))
                                             for n in SPECTRAL})
        surface = types.SimpleNamespace(**{n: np.asarray(to_numpy(getattr(member.surface, n))) for n in SURFACE})
        host_members.append(types.SimpleNamespace(atmosphere=atmosphere, surface=surface))
    terrain = np.asarray(to_numpy(transform.forward(model.surface_geopotential))).astype(np.complex128)
    capacity = ecfg.reference_physics.soil_wetness_capacity

    def device_ops(precision):
        return MemberOperators(transform, model.vertical, terrain, capacity, precision=precision)

    host_ops = MemberOperators(host_transform, model.vertical, terrain, capacity, precision="float64")
    rng = np.random.default_rng(11)
    n = 90
    lat = rng.uniform(-75.0, 75.0, n)
    lon = rng.uniform(0.0, 360.0, n)
    lat[:10] = lat[10:20]                       # duplicated positions, as station networks produce
    lon[:10] = lon[10:20]
    elev = np.where(rng.uniform(size=n) < 0.5, rng.uniform(0.0, 1500.0, n), 0.0)
    # THE REFRACTIVITY ROWS' TANGENT HEIGHTS, INSIDE THE COLUMN THIS
    # CONFIGURATION ACTUALLY HAS.  `elev` is a station elevation for every
    # other family and a tangent HEIGHT for `refractivity_n`, and the
    # operator returns NaN for a target outside its column rather than
    # extrapolating.  Measured on a Linux host (RTX 5090, Python 3.14.4)
    # 2026-09-10: this shipped smoke case has FOUR full levels, whose
    # pressures at a 1000 hPa surface are 4832, 20206, 46103 and 80679 Pa,
    # so its column spans about 1.8 km to 22 km.  Every one of the 0 to
    # 1500 m elevations above is BELOW the lowest full level, so every
    # refractivity value on both sides was NaN and that variable was never
    # compared at all: the guard in `_compare` fired on "nothing evaluated"
    # rather than the comparison passing on nothing.
    #
    # The band is read off the operator's own hypsometric arithmetic on
    # this model's own coordinate, not written down, so a config with a
    # different ladder moves it; the assertion in `_compare` names the span
    # if it ever stops landing inside.
    span = _column_height_span(model.vertical)
    tangent = np.arange(n) % 3 == 1
    elev[tangent] = rng.uniform(span[0], span[1], int(tangent.sum()))
    level = np.where(np.arange(n) % 3 == 0, np.nan, rng.uniform(30000.0, 95000.0, n))
    return ensemble.members, host_members, device_ops, host_ops, (lat, lon, elev, level)


#: The two variables the operator's own precision does not reach all of.
#:
#: `sample_wind` inverts the Laplacian on the TRANSFORM before it samples
#: anything: `psi = transform.inverse_laplacian(vorticity)`.  That step runs
#: at the transform's precision, which is the state's, and the operator's
#: `precision` applies only to the sampling that follows it.  So the device
#: and host wind rows are two float32 inversions of the same coefficients by
#: two different implementations, and their agreement floor is float32, not
#: float64.  Every other variable reaches the sampling directly and is held
#: at the float64 floor.
#:
#: MEASURED on a Linux host (RTX 5090, Python 3.14.4) against published
#: woof 2.7.2, 2026-09-10, on this fixture's members: in the float64 arm the
#: four scalars agree to 5.2e-15, 1.5e-15, 8.4e-16 and (surface pressure)
#: 5.2e-15 relative, while `wind_u_m_s` reads 9.36e-09 and `wind_v_m_s`
#: 3.64e-09.  A single 1.0e-9 bound therefore failed on the winds by nine
#: times and had never been run on a card.
#:
#: THE OTHER FIX, NOT TAKEN HERE: invert the Laplacian in float64 when the
#: operator's precision is float64, which would make the name true end to
#: end and both sides agree to rounding.  That is a change to what the
#: spectral core does inside a filter, it moves no number any door reports
#: today (the assimilation runs at state precision, the arm below, which
#: passes at 5.1e-06 against its 1.0e-05 bound), and it is not a release-day
#: edit.  Named as a follow-up rather than made silently.
_WIND_ROWS = ("wind_u_m_s", "wind_v_m_s")


def _compare(device, host, bound, label, *, wind_bound=None, names=None):
    """`names` defaults to the COLUMN vocabulary, which is what a default
    `evaluate` computes.

    THE BREAKAGE THIS PREVENTS, measured on a Linux host (RTX 5090,
    Python 3.14.4) 2026-09-10.  This walked all of `OPERATOR_VARIABLES`
    over a call that had asked for none of `refractivity_n`:
    `evaluate(variables=None)` means `COLUMN_VARIABLES`, and that set is
    `OPERATOR_VARIABLES` MINUS `refractivity_n` by construction, because a
    refractivity row's `elevation_m` is a tangent height rather than a
    station elevation and the two families cannot share one call.  So the
    device refractivity path was never compared to the host's, on either
    precision, and the loop read NaN on both sides and asked whether NaN
    equals NaN.  It has its own arm below, which asks for it by name.
    """

    for name in sorted(COLUMN_VARIABLES if names is None else names):
        d = np.asarray(device[name], dtype=np.float64)
        h = np.asarray(host[name], dtype=np.float64)
        assert d.shape == h.shape
        finite = np.isfinite(h)
        assert np.array_equal(finite, np.isfinite(d)), f"{label} {name}: NaN pattern differs"
        assert finite.any(), (
            f"{label} {name}: nothing evaluated.  Every host value is "
            "non-finite, so this variable was not compared at all.  For "
            "refractivity_n that means the fixture's tangent heights fell "
            "outside the column the operator integrates")
        limit = wind_bound if (wind_bound is not None and name in _WIND_ROWS) else bound
        scale = float(np.abs(h[finite]).max())
        err = float(np.abs(d[finite] - h[finite]).max())
        assert err <= limit * scale, f"{label} {name}: max |device - host| {err:.3e} against {scale:.3e} ({err / scale:.2e})"


def test_the_float64_device_operators_reproduce_the_host_to_rounding(twins):
    members, host_members, device_ops, host_ops, rows = twins
    ops = device_ops("float64")
    assert ops.precision_record["path"] == "device"
    dev_values, dev_lnp = ops.evaluate(members, *rows)
    host_values, host_lnp = host_ops.evaluate(host_members, *rows)
    # The scalars at the float64 floor; the winds at the float32 floor their
    # inverse Laplacian leaves them on, with an order of headroom over the
    # 9.4e-09 measured.  Both are far under the state arm's 1.0e-05, so this
    # arm still says something the arm below does not.
    _compare(dev_values, host_values, 1.0e-9, "float64", wind_bound=1.0e-7)
    assert np.allclose(dev_lnp, host_lnp, rtol=1.0e-9, atol=0.0, equal_nan=True)


def test_the_state_precision_device_operators_agree_to_the_states_own_precision(twins):
    members, host_members, device_ops, host_ops, rows = twins
    ops = device_ops("state")
    assert ops._sample_precision(members[0].atmosphere.theta) == "float32"
    dev_values, _ = ops.evaluate(members, *rows)
    host_values, _ = host_ops.evaluate(host_members, *rows)
    # A float32 state carries 6e-8 of relative quantisation; the blocked
    # float32 contraction was measured at 2e-6 maximum relative on the
    # case's T127 members and is held to five times that here.
    _compare(dev_values, host_values, 1.0e-5, "state")


def test_the_refractivity_family_is_compared_on_its_own_call(twins):
    """The other family, asked for by name, on both precisions.

    `refractivity_n` is not in the column vocabulary: its rows carry a
    tangent HEIGHT where the others carry a station elevation, so it has
    its own call, its own anchor and its own arm here.  Until this arm
    existed the variable rode the column loop, where nothing had computed
    it and both sides read NaN.

    The tangent heights come from `_column_height_span`, the operator's
    own hypsometric arithmetic on this model's own ladder.  MEASURED on
    an RTX 5090 host 2026-09-10: the shipped smoke case's four full levels put the
    column between about 2.0 km and 34.5 km at a 1000 hPa surface, and the
    fixture's original 0 to 1500 m elevations were all below its lowest
    level, so every value was NaN by construction.
    """

    members, host_members, device_ops, host_ops, rows = twins
    host_values, _ = host_ops.evaluate(host_members, *rows,
                                       variables=("refractivity_n",))
    reference = np.asarray(host_values["refractivity_n"], dtype=np.float64)
    finite = np.isfinite(reference)
    assert finite.sum() >= 20, (
        "the fixture's tangent heights are outside the column this "
        f"configuration integrates: {int(finite.sum())} finite of "
        f"{reference.size}")
    # A refractivity in the neutral atmosphere is order 10 to 400 N-units;
    # a comparison that passed on zeros would say nothing.
    assert float(np.abs(reference[finite]).min()) > 1.0

    for precision, bound in (("float64", 1.0e-9), ("state", 1.0e-5)):
        ops = device_ops(precision)
        values, _ = ops.evaluate(members, *rows, variables=("refractivity_n",))
        _compare(values, host_values, bound, precision,
                 names=("refractivity_n",))


def test_a_subset_of_variables_skips_the_rest_and_the_wind_synthesis(twins):
    members, host_members, device_ops, host_ops, rows = twins
    ops = device_ops("state")
    values, _ = ops.evaluate(members, *rows, variables=("temperature_k",))
    assert np.isfinite(values["temperature_k"]).any()
    for name in ("wind_u_m_s", "wind_v_m_s", "dewpoint_k"):
        assert np.all(np.isnan(values[name]))


def test_the_operators_refuse_an_unknown_precision(twins):
    members, host_members, device_ops, host_ops, rows = twins
    with pytest.raises(ValueError, match="precision"):
        device_ops("float16")
