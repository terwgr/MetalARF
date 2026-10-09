"""The two-time-level semi-Lagrangian semi-implicit core: the gates.

Every gate here carries its number and its reason.  The ones that matter
most are the two that no other test in the tree can catch:

*   the ADVECTIVE right-hand side is a new evaluation with no counterpart
    in the tree, and a sign error in its curvature or pressure-gradient
    term produces a plausible-looking forecast.  Two instruments answer
    it: a balanced rest state has to be a fixed point (a wrong term shows
    as a wind out of nothing), and the increment of one step has to agree
    with the shipped flux-form core's own as dt goes to zero, where both
    integrate the same continuous tendency and only the discretization
    separates them.

*   the semi-implicit operator's thermodynamic row is the reference
    profile's vertical advection, and this core has to use the operator's
    OWN discretization of it or the difference rides the explicit budget
    as a gravity wave.  The gate rebuilds the operator's gamma matrix from
    the reference interface values this lane reads and compares it, entry
    by entry, with the one the operator built.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = pytest.mark.cpu_only

from woof.globe.checkpoint import (  # noqa: E402
    read_checkpoint,
    schema_for_arrays,
    state_from_checkpoint,
    trajectory_from_checkpoint,
    write_checkpoint,
)
from woof.globe.config import (  # noqa: E402
    DEFAULT_MAXIMUM_LIPSCHITZ, TIME_INTEGRATORS, load_config,
)
from woof.globe.constants import (  # noqa: E402
    GRAVITY_M_S2, KAPPA, REFERENCE_PRESSURE_PA, SEMILAG_CHECKPOINT_SCHEMA,
)
from woof.globe.dynamics import MoistHybridModel  # noqa: E402
from woof.globe import pins  # noqa: E402
from woof.globe.semi_implicit import VerticalModeSemiImplicit  # noqa: E402
from woof.globe.semilag import (  # noqa: E402
    SEMILAG_INTEGRATORS,
    SemiLagrangianOptions,
    TRAJECTORY_FIELDS,
    TrajectoryState,
    reference_theta_faces,
)
from woof.globe.semilag.tables import SphericalGridTables  # noqa: E402
from woof.globe.semilag.trajectory import cartesian_wind  # noqa: E402
from woof.globe.semilag.vectors import transport_to_arrival  # noqa: E402
from woof.globe.semilag.interpolate import zero_stencil  # noqa: E402
from woof.globe.state import (  # noqa: E402
    ArwenGlobalState, MoistHybridState,
)
from woof.globe.vertical import HybridCoordinate  # noqa: E402
from woof.globe.spectral.transform import SphericalHarmonicTransform  # noqa: E402
from woof.globe.spectral.vector import VorticityDivergenceOperator  # noqa: E402

from test_arwen_global_vertical_modes import _surface  # noqa: E402
from test_arwen_global_vertical_numerics import _zero_grid_tracers  # noqa: E402

SL = SEMILAG_INTEGRATORS[0]


def _transform(truncation=21):
    return SphericalHarmonicTransform.create(
        truncation, backend="numpy", precision="float64"
    )


def _model(transform, vertical, integrator=SL, *, alpha=0.55,
           surface_geopotential=None, rotation=7.29212e-5, options=None):
    if surface_geopotential is None:
        surface_geopotential = np.zeros(transform.grid.shape)
    return MoistHybridModel(
        transform=transform, vertical=vertical,
        surface_geopotential=surface_geopotential, physics=None,
        rotation_rate_s=rotation, diffusion=None,
        semi_implicit=VerticalModeSemiImplicit(off_centring_weight=alpha),
        integrator=integrator, mass_fixer=False, water_fixer=False,
        positivity_repair=False, maximum_cfl=1.0e9, sponge_base_pa=0.0,
        semilag=options or SemiLagrangianOptions(),
    )


def _rest(transform, vertical, temperature_k=280.0, terrain_m=0.0):
    """A hydrostatically balanced isothermal rest state, over optional
    terrain, with the surface pressure that terrain implies."""
    lat = transform.grid.lat_rad[:, None]
    lon = transform.grid.lon_rad[None, :]
    height = np.broadcast_to(
        terrain_m * np.exp(-((lat - 0.6) ** 2 + (lon - 2.0) ** 2) / 0.15),
        transform.grid.shape,
    ).copy()
    ps = 1.0e5 * np.exp(-GRAVITY_M_S2 * height / (287.05 * temperature_k))
    ps = transform.inverse(transform.forward(ps))
    p_full = vertical.pressure(ps, transform.backend)["p_full"]
    theta = temperature_k / (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
    zeros = np.zeros(
        (vertical.nlev, *transform.spectral_shape), dtype=np.complex128
    )
    atmosphere = MoistHybridState(
        vorticity=zeros.copy(), divergence=zeros.copy(),
        theta=transform.project(transform.forward(theta)),
        log_surface_pressure=transform.project(transform.forward(np.log(ps))),
        qv=zeros.copy(), **_zero_grid_tracers(transform, vertical),
    )
    return (
        ArwenGlobalState(atmosphere, _surface(transform)),
        GRAVITY_M_S2 * height,
    )


def _baroclinic(transform, vertical, wind_m_s=25.0):
    """A moving, sheared, moist state: what the increment gate reads."""
    lat = transform.grid.lat_rad[:, None]
    lon = transform.grid.lon_rad[None, :]
    ps = 1.0e5 * (1.0 + 0.01 * np.cos(3.0 * lon) * np.cos(lat) ** 2)
    p_full = vertical.pressure(ps, transform.backend)["p_full"]
    normalized = np.log(p_full / p_full[0:1]) / np.log(
        p_full[-1:] / p_full[0:1]
    )
    temperature = (
        220.0 + normalized * (286.0 - 220.0)
        + 4.0 * np.cos(4.0 * lon) * np.cos(lat) ** 2
        * np.sin(math.pi * normalized)
    )
    theta = temperature / (p_full / REFERENCE_PRESSURE_PA) ** KAPPA
    u = np.broadcast_to(
        wind_m_s * np.cos(lat) * (0.4 + 0.6 * np.sin(math.pi * normalized)),
        p_full.shape,
    ).copy()
    v = 0.2 * wind_m_s * np.sin(2.0 * lon) * np.cos(lat) ** 2 + 0.0 * u
    vorticity, divergence = VorticityDivergenceOperator(
        transform
    ).vordiv_from_wind(u, v)
    qv = 0.008 * (p_full / ps[None]) ** 3 * (0.7 + 0.3 * np.cos(lat) ** 2)
    tracers = _zero_grid_tracers(transform, vertical)
    tracers["qc"] = np.asarray(
        1.0e-5 * np.maximum(0.0, np.cos(2.0 * lat) * np.cos(3.0 * lon))
        * np.exp(-((p_full / ps[None] - 0.65) / 0.15) ** 2)
    )
    atmosphere = MoistHybridState(
        vorticity=vorticity, divergence=divergence,
        theta=transform.project(transform.forward(theta)),
        log_surface_pressure=transform.project(transform.forward(np.log(ps))),
        qv=transform.project(transform.forward(qv)), **tracers,
    )
    return ArwenGlobalState(atmosphere, _surface(transform))


# ---------------------------------------------------------------- the core

def test_the_reference_advection_is_the_operators_own_discretization():
    """The gate that keeps the gravity wave out of the explicit budget.

    ``theta'`` is advected and its tendency is the reference profile's
    vertical advection with the sign reversed.  The semi-implicit
    operator's thermodynamic row IS that term at the reference state, and
    this lane reads it through :func:`reference_theta_faces`.  Rebuilding
    the operator's own gamma matrix from those faces has to reproduce it
    exactly: a different discretization would leave the difference of the
    two explicit, and at a 20 percent mismatch that residual is a 156 m/s
    wave whose Courant number at T255 and dt = 300 s is 3.0.
    """
    vertical = HybridCoordinate.surface_stretched(40, 100.0)
    operator = VerticalModeSemiImplicit().operator(vertical)
    face = reference_theta_faces(operator)
    assert face[0] == 0.0 and face[-1] == 0.0

    nlev = operator.nlev
    dp = np.diff(operator.p_half)
    delta_b = np.diff(vertical.b_half)
    theta_ref = operator.theta_ref
    rebuilt = np.zeros((nlev, nlev))
    for j in range(nlev):
        divergence = np.zeros(nlev)
        divergence[j] = 1.0
        flux_divergence = dp * divergence
        ps_t = -np.sum(flux_divergence)
        dp_t = delta_b * ps_t
        omega = np.zeros(nlev + 1)
        for k in range(nlev):
            omega[k + 1] = omega[k] - dp_t[k] - flux_divergence[k]
        omega[-1] = 0.0
        theta_t = -(
            omega[1:] * (face[1:] - theta_ref)
            - omega[:-1] * (face[:-1] - theta_ref)
        ) / dp
        rebuilt[:, j] = -theta_t
    residual = float(np.max(np.abs(rebuilt - operator.gamma)))
    scale = float(np.max(np.abs(operator.gamma)))
    assert residual <= 1.0e-9 * scale, (
        f"the reference advection this lane reads differs from the "
        f"operator's own by {residual:.3e} against a gamma of {scale:.3e}"
    )


def test_a_balanced_rest_state_is_a_fixed_point():
    """R1(a).  A wrong sign anywhere in the advective momentum tendency
    makes a wind out of nothing; at rest the trajectory is the identity,
    so nothing else can."""
    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    bundle, phis = _rest(transform, vertical)
    for dt in (60.0, 600.0):
        model = _model(transform, vertical, surface_geopotential=phis)
        state = bundle
        for _ in range(10):
            state, _metrics = model.step(state, dt)
        g = model.grid_state(state.atmosphere)
        wind = float(np.max(np.hypot(g["u"], g["v"])))
        assert wind < 1.0e-9, f"dt = {dt}: {wind:.3e} m/s out of a rest state"


def test_rest_over_a_mountain_reads_the_same_floor_as_the_shipped_core():
    """R1(b).  Over terrain the rest state is only balanced to the
    hydrostatic and pressure-gradient discretization's own floor.  What
    this gate asserts is that the semi-Lagrangian core reads THAT floor
    and not a larger one, at every step in its ladder: a residual that
    grew with dt would be the integrator's, not the discretization's."""
    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    bundle, phis = _rest(transform, vertical, terrain_m=2000.0)
    reference = None
    for integrator, dt in (
        ("imex_ssp3", 60.0), (SL, 60.0), (SL, 300.0), (SL, 450.0),
    ):
        model = _model(transform, vertical, integrator=integrator,
                       surface_geopotential=phis)
        state = bundle
        for _ in range(int(round(3600.0 / dt))):
            state, _metrics = model.step(state, dt)
        g = model.grid_state(state.atmosphere)
        wind = float(np.max(np.hypot(g["u"], g["v"])))
        if reference is None:
            reference = wind
            continue
        assert wind == pytest.approx(reference, rel=0.05), (
            f"{integrator} at dt = {dt} reads {wind:.4e} m/s against the "
            f"shipped core's {reference:.4e} after one hour at rest over a "
            "2 km mountain"
        )


@pytest.mark.parametrize("dt_s", [4.0, 2.0, 1.0, 0.5])
def test_one_step_converges_on_the_shipped_cores_own_increment(dt_s):
    """R1(c).  As dt goes to zero both integrators integrate the same
    continuous tendency, so their one-step increments have to agree on
    everything except where they genuinely discretize it differently.  A
    sign error in the curvature, the Coriolis or the pressure-gradient
    term of the advective form is an O(1) relative disagreement here and
    nothing else in the battery would see it.

    MEASURED 2026-09-06 on the T21 20-level model, a moving sheared moist
    state, as a fraction of the shipped core's own tendency:

        dt (s)      u        v      temperature    ps
        4.0      1.8e-4   2.1e-4     1.24e-2     8e-5
        2.0      9e-5     1.0e-4     1.19e-2     8e-5
        1.0      5e-5     5e-5       1.17e-2     8e-5
        0.5      3e-5     2e-5       1.16e-2     8e-5

    The momentum rows halve with the step, which is what says the two
    cores are integrating one tendency.  The thermodynamic row does NOT
    go to zero and is not supposed to: it sits at a floor of 1.2 percent
    because the two cores discretize the VERTICAL advection of theta
    differently on purpose.  The shipped core takes a van Leer limited
    flux of the actual profile; this one carries the reference profile's
    advection in the semi-implicit operator's own form and lets the
    interpolation carry the rest, which is the choice that keeps the
    gravity wave out of the explicit budget.  A floor at one percent is
    two discretizations of one term; a floor at one is a sign error.
    """
    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    bundle = _baroclinic(transform, vertical)
    increments = {}
    for integrator in ("imex_ssp3", SL):
        model = _model(transform, vertical, integrator=integrator, alpha=0.5)
        advanced, _metrics = model.step(bundle, dt_s)
        before = model.grid_state(bundle.atmosphere)
        after = model.grid_state(advanced.atmosphere)
        increments[integrator] = {
            name: (np.asarray(after[name]) - np.asarray(before[name])) / dt_s
            for name in ("u", "v", "temperature", "ps")
        }
    worst = {}
    for name in ("u", "v", "temperature", "ps"):
        a = increments["imex_ssp3"][name]
        b = increments[SL][name]
        scale = float(np.max(np.abs(a)))
        worst[name] = float(np.max(np.abs(a - b))) / max(scale, 1.0e-30)
    # The momentum rows converge; a 5x margin on the measured slope.
    for name in ("u", "v"):
        limit = 2.5e-4 * dt_s / 4.0 + 1.0e-4
        assert worst[name] < limit, (
            f"dt = {dt_s}: the {name} tendency of the semi-Lagrangian core "
            f"differs from the shipped core's by {worst[name]:.3e} of it, "
            f"past the {limit:.3e} this step allows"
        )
    # The surface-pressure row is flat at 8e-5 and the thermodynamic one at
    # 1.2 percent; both are held to 3x their measured value, which is far
    # below the 1 or more a sign error reads.
    assert worst["ps"] < 3.0e-4, worst
    assert worst["temperature"] < 0.04, worst


def test_the_parallel_transport_keeps_the_wind_it_was_given():
    """A projection onto the arrival tangent plane would shorten the wind
    by 1 - cos(displacement/a) every step, which at the measured T255
    displacement is 4.2e-3 of the jet per forecast day.  The rotation does
    not, and at zero displacement it is the identity."""
    transform = _transform(31)
    tables = SphericalGridTables.create(
        transform.grid, xp=np, dtype=np.float64
    )
    nlev = 6
    lat = transform.grid.lat_rad[:, None]
    lon = transform.grid.lon_rad[None, :]
    u = np.broadcast_to(
        40.0 * np.cos(lat) + 5.0 * np.sin(3.0 * lon),
        (nlev, *transform.grid.shape),
    ).copy()
    v = np.broadcast_to(
        12.0 * np.sin(2.0 * lon) * np.cos(lat),
        (nlev, *transform.grid.shape),
    ).copy()
    vx, vy, vz = cartesian_wind(u, v, tables)

    identity = zero_stencil(tables, nlev, xp=np, dtype=np.float64)
    back_u, back_v = transport_to_arrival(vx, vy, vz, identity, tables,
                                          fused=False)
    assert np.max(np.abs(back_u - u)) < 1.0e-12
    assert np.max(np.abs(back_v - v)) < 1.0e-12

    # A real displacement: three cells east and one row poleward.  The
    # vector the transport is given is tangent at the DEPARTURE point, so
    # it is built in the departure point's own local basis; that is what
    # the gather hands back, and it is the case in which a projection
    # would shorten the wind and the rotation must not.
    moved = type(identity)(
        xi=identity.xi + 3.0, phi=identity.phi * 0.98,
        level=identity.level, tables=tables,
    )
    lam_d = moved.xi * tables.dlam
    phi_d = moved.phi
    sin_lam, cos_lam = np.sin(lam_d), np.cos(lam_d)
    sin_phi, cos_phi = np.sin(phi_d), np.cos(phi_d)
    dx = -u * sin_lam - v * sin_phi * cos_lam
    dy = u * cos_lam - v * sin_phi * sin_lam
    dz = v * cos_phi + np.zeros_like(u)
    moved_u, moved_v = transport_to_arrival(dx, dy, dz, moved, tables,
                                            fused=False)
    speed_before = np.hypot(u, v)
    speed_after = np.hypot(moved_u, moved_v)
    drift = float(np.max(np.abs(speed_after - speed_before)))
    assert drift < 1.0e-11 * float(np.max(speed_before)), (
        f"the transport changed the wind's magnitude by {drift:.3e} m/s"
    )
    # And what a projection instead of a rotation would have cost, on this
    # displacement, measured rather than argued.
    projected_u = -dx * np.sin(identity.xi * tables.dlam) + dy * np.cos(
        identity.xi * tables.dlam)
    projected_v = (
        -dx * np.sin(identity.phi) * np.cos(identity.xi * tables.dlam)
        - dy * np.sin(identity.phi) * np.sin(identity.xi * tables.dlam)
        + dz * np.cos(identity.phi)
    )
    shortfall = float(np.max(
        speed_before - np.hypot(projected_u, projected_v)
    ))
    assert shortfall > 1.0e-3, (
        "the projection arm must actually shorten the wind, or this gate "
        "is measuring nothing"
    )


# ------------------------------------------------------- state and identity

def test_the_second_time_level_makes_a_restart_bit_exact(tmp_path):
    """The device-qualification pin asserts an uninterrupted run and a
    midpoint restart are bit-exact.  A two-time-level scheme that rebuilt
    its second level with a start-up step would not be, so the level is
    checkpointed and this gate is what says the plumbing works."""
    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    bundle = _baroclinic(transform, vertical)
    dt = 300.0

    model = _model(transform, vertical)
    uninterrupted = bundle
    for _ in range(6):
        uninterrupted, _m = model.step(uninterrupted, dt)

    restarting = _model(transform, vertical)
    midpoint = bundle
    for _ in range(3):
        midpoint, _m = restarting.step(midpoint, dt)
    path = write_checkpoint(
        tmp_path / "midpoint.npz", midpoint,
        config_hash="0" * 64, to_numpy=transform.backend.to_numpy,
        semi_implicit_scheme="vertical_modes", integrator=SL,
        trajectory=restarting.trajectory_state(),
    )
    metadata, arrays = read_checkpoint(
        path, semi_implicit_scheme="vertical_modes", integrator=SL
    )
    assert metadata["schema"] == SEMILAG_CHECKPOINT_SCHEMA
    assert {
        name for name in arrays if name.startswith("trajectory__")
    } == {f"trajectory__{name}" for name in TRAJECTORY_FIELDS}

    resumed_model = _model(transform, vertical)
    resumed = state_from_checkpoint(metadata, arrays, transform.backend)
    resumed_model.set_trajectory_state(
        trajectory_from_checkpoint(metadata, arrays, transform.backend)
    )
    assert resumed_model.trajectory_state() is not None
    for _ in range(3):
        resumed, _m = resumed_model.step(resumed, dt)
    for name in ("vorticity", "divergence", "theta", "log_surface_pressure",
                 "qv", "qc"):
        a = np.asarray(getattr(uninterrupted.atmosphere, name))
        b = np.asarray(getattr(resumed.atmosphere, name))
        assert np.array_equal(a, b), f"{name} differs across the restart"


def test_a_resume_without_the_second_level_is_not_the_same_run():
    """The counter-instrument: the gate above would pass just as well if
    the second level changed nothing.  It changes the run."""
    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    bundle = _baroclinic(transform, vertical)
    dt = 300.0
    model = _model(transform, vertical)
    for _ in range(3):
        bundle, _m = model.step(bundle, dt)
    carried = model.trajectory_state()
    assert carried is not None
    with_level, _m = model.step(bundle, dt)
    model.set_trajectory_state(None)
    without, _m = model.step(bundle, dt)
    difference = float(np.max(np.abs(
        np.asarray(with_level.atmosphere.divergence)
        - np.asarray(without.atmosphere.divergence)
    )))
    assert difference > 0.0
    assert isinstance(carried, TrajectoryState)


def test_the_schema_says_exactly_what_the_arrays_carry():
    assert schema_for_arrays({"atmosphere__theta": np.zeros(1)}) \
        != SEMILAG_CHECKPOINT_SCHEMA
    assert schema_for_arrays({"trajectory__n_u": np.zeros(1)}) \
        == SEMILAG_CHECKPOINT_SCHEMA


def test_a_v3_archive_carrying_a_trajectory_array_is_refused(tmp_path):
    """Both halves of the contract, because an archive that claimed a
    shape it did not have would resume a two-time-level integrator on a
    level it never wrote, which reads as a start-up step in the middle of
    a forecast and shows up nowhere else."""
    from woof.globe.checkpoint import write_checkpoint_arrays
    import json
    import numpy as np

    transform = _transform(7)
    vertical = HybridCoordinate.pressure_blend(4, 100.0)
    bundle = _baroclinic(transform, vertical, wind_m_s=5.0)
    model = _model(transform, vertical)
    model.step(bundle, 60.0)
    path = write_checkpoint(
        tmp_path / "v4.npz", bundle, config_hash="0" * 64,
        to_numpy=transform.backend.to_numpy,
        semi_implicit_scheme="vertical_modes", integrator=SL,
        trajectory=model.trajectory_state(),
    )
    with np.load(path, allow_pickle=False) as archive:
        arrays = {n: np.array(archive[n]) for n in archive.files
                  if n != "__metadata__"}
        metadata = json.loads(str(archive["__metadata__"].item()))
    assert metadata["schema"] == SEMILAG_CHECKPOINT_SCHEMA
    # Drop one of the seven and the archive is refused by name.
    arrays.pop("trajectory__n_lnps")
    broken = tmp_path / "broken.npz"
    np.savez_compressed(
        broken, __metadata__=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )
    with pytest.raises(ValueError, match="trajectory inventory mismatch"):
        read_checkpoint(broken)


# ---------------------------------------------------------------- the door

def test_the_integrator_is_selectable_and_pinned():
    assert SL in TIME_INTEGRATORS
    assert pins.integrator_family(SL) == SL
    assert pins.arithmetic_label("vertical_modes", SL) == f"vertical_modes/{SL}"
    document = pins.pin_document("vertical_modes", SL)
    template = pins.pin_document("vertical_modes", "imex_ssp3")
    assert set(document) == set(template)
    assert {key for key in document if document[key] != template[key]} == {
        "semi_implicit", "momentum", "scalar_transport", "checkpoint",
    }
    # The other four arithmetics did not move.
    assert pins.pins_hash("vertical_modes", "imex_ssp3") == pins.PINS_HASH
    assert len(pins.KNOWN_PINS_HASHES) == 5


def test_the_barotropic_proxy_cannot_pair_with_the_semi_lagrangian_core():
    with pytest.raises(ValueError, match="cannot pair|pinned"):
        pins.pin_document("external", SL)


def _write(tmp_path, text: str):
    path = tmp_path / "cfg.toml"
    path.write_text(text, encoding="utf-8")
    return path


_BASE = """
[arwen_global]
schema = "gpuwm.arwen-global-run/v1"
name = "sl-door"
acknowledgement = "research-only-arwen-global-v1"
backend = "numpy"
precision = "float64"
[grid]
truncation = 10
[time]
dt_s = 300.0
duration_s = 3000.0
integrator = "{integrator}"
{time_extra}
[vertical]
coordinate = "pressure_blend"
nlev = 8
[physics]
mode = "none"
{extra}
"""


def test_the_semilag_table_is_refused_under_another_integrator(tmp_path):
    path = _write(tmp_path, _BASE.format(
        integrator="imex_ssp3", time_extra="",
        extra='[semilag]\ntrajectory_iterations = 4\n'))
    with pytest.raises(ValueError, match=r"\[semilag\] table is read only"):
        load_config(path)


def test_maximum_lipschitz_is_refused_under_another_integrator(tmp_path):
    path = _write(tmp_path, _BASE.format(
        integrator="imex_ssp3", time_extra="maximum_lipschitz = 0.5",
        extra=""))
    with pytest.raises(ValueError, match="maximum_lipschitz is read only"):
        load_config(path)


@pytest.mark.parametrize("table,match", [
    ('[semi_implicit]\nscheme = "external"\n', "barotropic proxy"),
    ("[semi_implicit]\nenabled = false\n", "explicit budget"),
    ("[semi_implicit]\nweight = 0.5\n", "no explicit budget"),
])
def test_the_semi_lagrangian_core_refuses_an_explicit_gravity_wave(
    tmp_path, table, match
):
    path = _write(tmp_path, _BASE.format(
        integrator=SL, time_extra="", extra=table))
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_the_identity_of_every_other_integrator_is_untouched(tmp_path):
    """Hash stability, both directions: the two new fields leave every
    other integrator's config identity exactly as it was, and a semi-
    Lagrangian config carries them."""
    imex = load_config(_write(tmp_path, _BASE.format(
        integrator="imex_ssp3", time_extra="", extra="")))
    assert "semilag" not in imex.config_identity
    assert "maximum_lipschitz" not in imex.config_identity
    assert imex.maximum_lipschitz == DEFAULT_MAXIMUM_LIPSCHITZ

    sl = load_config(_write(tmp_path, _BASE.format(
        integrator=SL, time_extra="maximum_lipschitz = 0.6", extra="")))
    identity = sl.config_identity
    assert identity["maximum_lipschitz"] == 0.6
    assert identity["semilag"]["trajectory_iterations"] == 3
    # The batch size is a memory and launch trade that changes no bits, so
    # it is not in the identity; every other option is.
    assert "gather_batch" not in identity["semilag"]
    assert sl.config_hash != imex.config_hash


def test_the_gate_reads_the_two_folds_and_not_the_mixed_unit_norm():
    """The instrument gate.

    The mixed-unit 3x3 spectral norm converts one model level into metres
    with a single reference length, and its VALUE therefore depends on
    that choice while the flow does not.  MEASURED 2026-09-06 on a real
    T255 L40 forecast day: it reads 3.114e-3 per second where the
    horizontal block reads 1.033e-3, and the whole difference is the
    vertical rate's horizontal gradient multiplied by the 52.1 km
    equatorial spacing on rings whose own spacing is 326 m.  So the gate
    reads the two scale-free folds and the mixed norms are reported
    beside them.
    """
    from woof.globe.semilag.trajectory import (
        CartesianWind, lipschitz,
    )

    transform = _transform(21)
    tables = SphericalGridTables.create(
        transform.grid, xp=np, dtype=np.float64
    )
    nlev = 8
    lat = transform.grid.lat_rad[:, None]
    lon = transform.grid.lon_rad[None, :]
    level = np.arange(nlev)[:, None, None] / (nlev - 1.0)
    u = np.broadcast_to(
        30.0 * np.cos(lat) * (0.3 + level) * (1.0 + 0.2 * np.sin(2.0 * lon)),
        (nlev, *transform.grid.shape),
    ).copy()
    v = np.broadcast_to(
        6.0 * np.sin(3.0 * lon) * np.cos(lat) + 0.0 * level,
        (nlev, *transform.grid.shape),
    ).copy()
    vx, vy, vz = cartesian_wind(u, v, tables)
    # A vertical rate whose HORIZONTAL gradient is what the mixed norm
    # multiplies by the reference length.
    rate = np.broadcast_to(
        2.0e-4 * np.sin(math.pi * level) * (1.0 + 0.8 * np.cos(lon)),
        (nlev, *transform.grid.shape),
    ).copy()
    wind = CartesianWind(
        np.ascontiguousarray(vx), np.ascontiguousarray(vy),
        np.ascontiguousarray(vz), rate,
    )
    diagnostics = lipschitz(wind, tables, 300.0)
    assert diagnostics.lipschitz == pytest.approx(
        max(diagnostics.lipschitz_horizontal, diagnostics.lipschitz_vertical,
            diagnostics.lipschitz_balanced)
    )
    assert diagnostics.lipschitz_mixed == pytest.approx(
        300.0 * diagnostics.jacobian_spectral_s
    )
    # The mixed norm is the LARGER one on this flow, which is the whole
    # reason it is reported and not gated.
    assert diagnostics.lipschitz_mixed > diagnostics.lipschitz
    # The balanced norm dominates both folds, because each of them is a
    # submatrix of the matrix it is the norm of, and it is what the gate
    # reads.
    assert diagnostics.lipschitz_balanced >= max(
        diagnostics.lipschitz_horizontal, diagnostics.lipschitz_vertical
    ) - 1.0e-12
    assert set(diagnostics.as_dict()) >= {
        "semilag_lipschitz", "semilag_lipschitz_horizontal",
        "semilag_lipschitz_vertical", "semilag_lipschitz_mixed",
        "semilag_lipschitz_balanced", "semilag_lipschitz_frobenius",
    }
    # The gated number is SCALE FREE: the mixed norm moves when the length
    # that converts a model level into metres moves and the gated one does
    # not.  That is the whole claim the instrument rests on and nothing
    # else in the battery makes it.
    stretched = lipschitz(wind, tables, 300.0, reference_length_m=4.0 * 52100.0)
    assert stretched.lipschitz == pytest.approx(diagnostics.lipschitz, rel=1e-9)
    assert stretched.lipschitz_mixed != pytest.approx(
        diagnostics.lipschitz_mixed, rel=1e-3
    )


def test_the_tracer_mass_fixer_closes_and_says_by_how_much():
    """MASS-1's tracer row, on a state that actually has condensate.

    Every adiabatic arm of this lane is dry, and a dry run gives the
    fixer nothing to fix: its per-step magnitude reads exactly zero on
    all of them.  A semi-Lagrangian gather reads a MIXING RATIO at a
    point and the layer thickness it is measured against has moved
    underneath it, so without the fixer the species' mass drifts every
    step; with it the mass is restored, and what the receipt carries is
    how much the fixer had to move.
    """
    transform = _transform()
    vertical = HybridCoordinate.pressure_blend(20, 100.0)
    bundle = _baroclinic(transform, vertical)
    weights = np.asarray(transform.grid.quadrature_weights, dtype=np.float64)
    cell = weights[:, None] / (2.0 * transform.grid.nlon)

    def condensate_mass(state):
        dp = np.asarray(
            transform.backend.to_numpy(
                vertical.pressure(
                    np.exp(np.asarray(transform.inverse(
                        state.log_surface_pressure))),
                    transform.backend,
                )["dp"]
            )
        )
        return float(np.sum(np.asarray(state.qc) * dp * cell[None]))

    before = condensate_mass(bundle.atmosphere)
    assert before > 0.0, "the fixture must carry condensate or this gate is blind"

    fixed_model = _model(transform, vertical)
    unfixed_model = _model(
        transform, vertical,
        options=SemiLagrangianOptions(tracer_fixer="none"),
    )
    fixed = bundle
    unfixed = bundle
    magnitudes = []
    for _ in range(8):
        fixed, metrics = fixed_model.step(fixed, 300.0)
        unfixed, _m = unfixed_model.step(unfixed, 300.0)
        magnitudes.append(
            float(metrics["semilag_tracer_mass_fixer_relative__qc"])
        )
    with_fixer = abs(condensate_mass(fixed.atmosphere) - before) / before
    without = abs(condensate_mass(unfixed.atmosphere) - before) / before
    # The fixer restores the mass the gather does not conserve; the
    # counter-arm is what says the gate is measuring something.
    assert with_fixer < 1.0e-12, (
        f"the fixed arm drifted {with_fixer:.3e} of its condensate over "
        "eight steps"
    )
    assert without > 1.0e3 * max(with_fixer, 1.0e-15), (
        f"the unfixed arm drifted only {without:.3e}, so this gate is not "
        "measuring the fixer"
    )
    # And the receipt carries the magnitude, per species, every step.
    assert max(magnitudes) > 0.0
    assert all(m == m for m in magnitudes)


def test_the_reported_fixer_magnitude_is_the_transports_and_not_the_fixers():
    """What the gated number means, pinned.

    ``fix_mass`` computes its magnitude from the field the GATHER
    produced, before any correction, and for every scheme it accepts.  So
    the number is a property of the transport and no choice of fixer can
    move it -- which matters because a receipt that fails this gate can
    be read as asking for a different fixer, and a different fixer would
    change the same number by nothing at all.

    The schemes it is pinned over are the ones the door accepts today.
    ``"proportional"`` was one of them when this gate was written and is
    now refused by name, for the reason this test is about: on a field
    the positivity floor has already made non-negative it weighted the
    correction by the same array ``"bermejo_conde"`` does, so it was a
    door value that changed no bit of any run.  The refusal is pinned
    here beside the equality, because a retired name that stopped being
    refused would put that door value back.
    """
    from woof.globe.semilag.tracers import fix_mass

    transform = _transform()
    shape = (4, *transform.grid.shape)
    rng = np.random.default_rng(11)
    before = {"qc": rng.random(shape) * 1.0e-4}
    advected = {"qc": before["qc"] * 0.93}
    # A non-negative advected field, so the limiter took nothing and the
    # additive stage has nothing to place: it closes multiplicatively and
    # the two conservative forms have to be the same array.
    deficits = {"qc": np.zeros(shape)}
    dp = np.full(shape, 2.0e3)
    magnitudes = {}
    fixed = {}
    for scheme in ("bermejo_conde_additive", "bermejo_conde", "none"):
        out, metrics = fix_mass(
            advected, before, dp, dp, transform, scheme=scheme,
            deficits=deficits,
        )
        magnitudes[scheme] = metrics[
            "semilag_tracer_mass_fixer_relative__qc"
        ]
        fixed[scheme] = np.asarray(out["qc"])
    assert magnitudes["bermejo_conde"] == pytest.approx(
        magnitudes["none"], rel=1e-15
    )
    assert magnitudes["bermejo_conde_additive"] == pytest.approx(
        magnitudes["none"], rel=1e-15
    )
    assert magnitudes["none"] == pytest.approx(0.07, rel=1e-9)
    # "none" leaves the mass wrong by exactly what it reported, and the
    # two conservative arms close it the same way on a field the limiter
    # did not touch.
    assert np.allclose(fixed["bermejo_conde_additive"],
                       fixed["bermejo_conde"], rtol=1e-12, atol=0.0)
    assert not np.allclose(fixed["none"], fixed["bermejo_conde"])
    # And the retired name is refused, not quietly accepted.
    with pytest.raises(ValueError, match="retired"):
        fix_mass(advected, before, dp, dp, transform, scheme="proportional")


def test_the_cfl_pre_warning_is_armed_only_where_its_refusal_exists():
    """The reverse half of the gate law.

    The CFL wire warns of ONE breakage: the spectral CFL refusal, which is
    a ValueError with no receipt of the state that produced it.  The
    semi-Lagrangian core has no such refusal and runs at an advective
    Courant number of 1.5 at dt = 300 s and 2.2 at 450 s BY DESIGN, so the
    wire fired every step and wrote a 3.5 MB snapshot each time (MEASURED
    2026-09-06 on the T255 timing run: four trips in forty steps, with the
    ledger observer at 125.9 ms of a 270.6 ms step).  A wire whose
    breakage cannot happen is not a wire.
    """
    from woof.globe.insitu.tripwires import TripwireSet

    row = {
        "step": 3, "time_s": 900.0, "terms": {},
        "metrics": {"spectral_cfl": 2.2},
    }
    armed = TripwireSet(precision="float32", maximum_cfl=0.75,
                        advective_cfl_refusal=True)
    trips = [t for t in armed.evaluate_row(dict(row))
             if t["tripwire"] == "cfl_pre_warning"]
    assert trips and trips[0]["value"] > 1.0

    disarmed = TripwireSet(precision="float32", maximum_cfl=0.75,
                           advective_cfl_refusal=False)
    assert not [t for t in disarmed.evaluate_row(dict(row))
                if t["tripwire"] == "cfl_pre_warning"]
    # Every other wire is still armed on both.
    assert set(armed.specs) == set(disarmed.specs)
