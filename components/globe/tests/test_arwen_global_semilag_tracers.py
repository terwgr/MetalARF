"""The semi-Lagrangian tracer path: the limiter, the fixer, the coupling.

Three things this file is the only gate on.

*   The clip-deficit gather reports the mass the quasi-monotone limiter
    moved WITHOUT changing the values it returns.  If it changed them,
    every measurement made with the additive fixer would be of a
    different interpolation than the one the default arm runs, and no
    other test in the tree compares the two entry points.

*   The additive mass fixer closes a species' mass exactly and puts the
    correction back where the limiter took it.  The multiplicative form
    closes the same mass by rescaling every point that holds the species,
    which is conservative and wrong in place; the gate that separates
    them is where the correction LANDS, not how large it is.

*   The physics coupling arms agree when there is no physics to couple.
    That is what says the two extra arms are the same integrator with the
    increment moved, and not a second integrator.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import math
import pathlib

import numpy as np
import pytest

# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = pytest.mark.cpu_only

from woof.globe.constants import CONDENSATE_SPECIES  # noqa: E402
from woof.globe.semilag.cases import (  # noqa: E402
    DISTRIBUTIONS, run_deformational_transport,
)
from woof.globe.semilag.interpolate import (  # noqa: E402
    Stencil, gather_batch, zero_stencil,
)
from woof.globe.semilag.options import (  # noqa: E402
    PHYSICS_ARRIVAL_WEIGHT, PHYSICS_COUPLINGS, SemiLagrangianOptions,
    semilag_options_from_table,
)
from woof.globe.semilag.tables import SphericalGridTables  # noqa: E402
from woof.globe.semilag.tracers import (  # noqa: E402
    RETIRED_TRACER_FIXERS, TRACER_FIXERS, area_weights, fix_mass,
)
from woof.globe.spectral.transform import (  # noqa: E402
    SphericalHarmonicTransform,
)


def _transform(truncation=21):
    return SphericalHarmonicTransform.create(
        truncation, backend="numpy", precision="float64"
    )


def _tables(transform):
    backend = transform.backend
    return SphericalGridTables.create(
        transform.grid, xp=backend.xp, dtype=backend.float_dtype
    )


def _shifted_stencil(tables, nlev, *, dx, dy=0.0, dk=0.0):
    """The identity stencil moved by a fixed fraction of a cell."""
    zero = zero_stencil(tables, nlev, xp=np, dtype=tables.dtype)
    nlat = tables.shape[0]
    dphi = float(tables.lat_ext_host[3] - tables.lat_ext_host[2])
    return Stencil(
        xi=np.ascontiguousarray(zero.xi + dx),
        phi=np.ascontiguousarray(zero.phi + dy * dphi),
        level=np.ascontiguousarray(
            np.clip(zero.level + dk, 0.0, nlev - 1.0)
        ),
        tables=tables,
    )


def _spiky(tables, nlev, *, seed=11):
    """A field like a condensate species: zero nearly everywhere, with
    isolated maxima that span orders of magnitude.  A smooth field would
    not exercise the limiter at all, and the limiter is the thing under
    test."""
    rng = np.random.default_rng(seed)
    nlat, nlon = tables.shape
    field = np.zeros((nlev, nlat, nlon), dtype=tables.dtype)
    for _ in range(40):
        k = int(rng.integers(1, max(2, nlev - 1)))
        j = int(rng.integers(2, nlat - 2))
        i = int(rng.integers(0, nlon))
        field[k, j, i] += float(10.0 ** rng.uniform(-6.0, -2.0))
    return field


def _cloudy(tables, nlev, *, seed=7):
    """A field like a condensate species with CLOUDS in it: contiguous
    blobs over a zero background, so an interior cell has a nonzero
    minimum and the two limiter shapes have something to differ about.
    A field of isolated points has a zero in every cell and the two are
    then identical, which is itself worth knowing and is why the sparse
    field above stays in use elsewhere."""
    rng = np.random.default_rng(seed)
    nlat, nlon = tables.shape
    field = np.zeros((nlev, nlat, nlon), dtype=tables.dtype)
    for _ in range(8):
        k = int(rng.integers(1, max(2, nlev - 3)))
        j = int(rng.integers(3, nlat - 6))
        i = int(rng.integers(0, nlon))
        peak = float(10.0 ** rng.uniform(-4.0, -2.0))
        for dk in range(3):
            for dj in range(4):
                for di in range(4):
                    taper = (1.0 - dk / 3.0) * (1.0 - dj / 4.0) * (1.0 - di / 4.0)
                    field[k + dk, j + dj, (i + di) % nlon] += peak * taper
    return field


# ---------------------------------------------------------------------------
# the clip-deficit gather


def test_the_deficit_gather_returns_the_values_the_plain_one_returns():
    transform = _transform()
    tables = _tables(transform)
    nlev = 8
    field = _spiky(tables, nlev)
    stencil = _shifted_stencil(tables, nlev, dx=0.37, dy=0.21, dk=0.13)
    plain = gather_batch([field], stencil, monotone=True)[0]
    pair = gather_batch([field], stencil, monotone=True, deficit=True)[0]
    value, cut = pair
    # Bit for bit, not to a tolerance: the deficit variant computes the
    # same clip and writes the difference it would otherwise discard.
    assert np.array_equal(plain, value)
    # The deficit is raw minus kept, so it is nonzero exactly where the
    # limiter moved the value, and adding it back is the unlimited gather.
    raw = gather_batch([field], stencil, monotone=False)[0]
    assert np.allclose(value + cut, raw, rtol=0, atol=0)
    assert float(np.max(np.abs(cut))) > 0.0


def test_the_deficit_is_refused_without_the_limiter_that_makes_it():
    transform = _transform()
    tables = _tables(transform)
    stencil = _shifted_stencil(tables, 8, dx=0.3)
    with pytest.raises(ValueError, match="an unlimited gather has none"):
        gather_batch([_spiky(tables, 8)], stencil, monotone=False,
                     deficit=True)


def test_a_zero_displacement_gather_clips_nothing():
    transform = _transform()
    tables = _tables(transform)
    nlev = 8
    field = _spiky(tables, nlev)
    stencil = zero_stencil(tables, nlev, xp=np, dtype=tables.dtype)
    value, cut = gather_batch([field], stencil, monotone=True,
                              deficit=True)[0]
    assert np.array_equal(value, field)
    assert float(np.max(np.abs(cut))) == 0.0


# ---------------------------------------------------------------------------
# the mass fixer


def _one_species(transform, tables, nlev, *, dx=0.37, dy=0.23, dk=0.11):
    field = _spiky(tables, nlev)
    stencil = _shifted_stencil(tables, nlev, dx=dx, dy=dy, dk=dk)
    value, cut = gather_batch([field], stencil, monotone=True,
                              deficit=True)[0]
    dp = np.full_like(field, 1000.0)
    return field, value, cut, dp


@pytest.mark.parametrize("scheme", ["bermejo_conde_additive",
                                    "bermejo_conde"])
def test_every_conservative_fixer_closes_the_species_mass(scheme):
    transform = _transform()
    tables = _tables(transform)
    nlev = 8
    before, value, cut, dp = _one_species(transform, tables, nlev)
    cell = area_weights(transform)
    fixed, marks = fix_mass(
        {"qg": value}, {"qg": before}, dp, dp, transform,
        scheme=scheme, deficits={"qg": cut},
    )
    target = float(np.sum(before * dp * cell))
    after = float(np.sum(fixed["qg"] * dp * cell))
    assert abs(after - target) <= 1.0e-12 * target
    # The magnitude the fixer had to move is a property of the ADVECTION
    # and is the same number under every form.
    assert marks["semilag_tracer_mass_fixer_relative__qg"] > 0.0
    assert float(np.min(fixed["qg"])) >= 0.0


def test_the_additive_fixer_restores_a_deficit_where_the_limiter_cut():
    """The ADD side: the limiter took mass off an overshoot and the fixer
    puts it back on the same points, where the multiplicative form would
    have spread it over every point that holds the species."""
    transform = _transform()
    tables = _tables(transform)
    nlev = 8
    _before, value, cut, dp = _one_species(transform, tables, nlev)
    cell = area_weights(transform)
    # A pre-advection field heavier than the advected one by a fraction of
    # what the limiter cut away, so the correction is an ADDITION the clip
    # deficit can carry in full.
    capacity = float(np.sum(np.maximum(cut, 0.0) * dp * cell))
    assert capacity > 0.0
    mass = float(np.sum(value * dp * cell))
    before = value * (1.0 + 0.5 * capacity / mass)
    additive, marks_a = fix_mass(
        {"qg": value}, {"qg": before}, dp, dp, transform,
        scheme="bermejo_conde_additive", deficits={"qg": cut},
    )
    multiplicative, marks_m = fix_mass(
        {"qg": value}, {"qg": before}, dp, dp, transform,
        scheme="bermejo_conde",
    )
    # The same correction to make, by construction: its size is the
    # ADVECTION's error and no fixer form changes it.
    assert (marks_a["semilag_tracer_mass_fixer_kg_m2__qg"]
            == marks_m["semilag_tracer_mass_fixer_kg_m2__qg"])
    # The additive form carried all of it on the clipped points.
    assert marks_a["semilag_tracer_clip_share__qg"] == pytest.approx(1.0)
    assert marks_m["semilag_tracer_clip_share__qg"] == 0.0
    # And changed the field ONLY there, where the multiplicative form
    # changed every point that holds any of the species.
    cutting = cut > 0.0
    assert float(np.max(np.abs(additive["qg"] - value)[~cutting])) == 0.0
    elsewhere = (~cutting) & (value > 0.0)
    assert float(np.max(np.abs(multiplicative["qg"] - value)[elsewhere])) > 0.0


def test_the_additive_fixer_cannot_undo_a_clip_that_lifted_a_field_off_zero():
    """The REMOVE side, and a finding rather than a defect.

    A condensate species is zero nearly everywhere, so a cubic through it
    undershoots below zero on the shoulder of every maximum and the
    limiter lifts those points to the local minimum, which is zero.  That
    CREATES mass, and it cannot be taken back where it was created: the
    point now holds exactly zero and removing anything from it would make
    a negative mixing ratio.  The fixer says so in its own numbers instead
    of quietly closing the mass somewhere else and reporting nothing.
    """
    transform = _transform()
    tables = _tables(transform)
    nlev = 8
    before, value, cut, dp = _one_species(transform, tables, nlev)
    cell = area_weights(transform)
    target = float(np.sum(before * dp * cell))
    assert float(np.sum(value * dp * cell)) > target       # the clip added
    assert float(np.min(cut)) < 0.0                        # by lifting
    fixed, marks = fix_mass(
        {"qg": value}, {"qg": before}, dp, dp, transform,
        scheme="bermejo_conde_additive", deficits={"qg": cut},
    )
    assert marks["semilag_tracer_clip_share__qg"] == 0.0
    assert float(np.sum(fixed["qg"] * dp * cell)) == pytest.approx(
        target, rel=1.0e-12
    )
    assert float(np.min(fixed["qg"])) >= 0.0


def test_the_additive_fixer_refuses_to_run_without_the_deficit():
    transform = _transform()
    tables = _tables(transform)
    before, value, _cut, dp = _one_species(transform, tables, 8)
    with pytest.raises(ValueError, match="cannot run without the limiter"):
        fix_mass({"qg": value}, {"qg": before}, dp, dp, transform,
                 scheme="bermejo_conde_additive")


def test_a_species_that_has_left_the_model_is_refused_by_name():
    transform = _transform()
    tables = _tables(transform)
    nlev = 8
    before = _spiky(tables, nlev)
    gone = np.zeros_like(before)
    dp = np.full_like(before, 1000.0)
    with pytest.raises(ValueError, match="no air to put it back in"):
        fix_mass({"qg": gone}, {"qg": before}, dp, dp, transform,
                 scheme="bermejo_conde", deficits={"qg": gone})


def test_the_fixer_of_record_is_the_multiplicative_one_and_none_is_reachable():
    # The additive stage is selectable and not default: MEASURED
    # 2026-09-06 on the T255 native arms, it carried 1 to 6 percent of the
    # correction under the default physics coupling, which does not pay
    # for the second device array per gathered tracer that reporting the
    # limiter's deficit costs.
    assert SemiLagrangianOptions().tracer_fixer == "bermejo_conde"
    assert set(TRACER_FIXERS) == {
        "bermejo_conde_additive", "bermejo_conde", "none"
    }
    with pytest.raises(ValueError, match="semilag.tracer_fixer must be"):
        SemiLagrangianOptions(tracer_fixer="whatever")
    # The counter-arm reports the drift and closes nothing.
    transform = _transform()
    tables = _tables(transform)
    before, value, cut, dp = _one_species(transform, tables, 8)
    fixed, marks = fix_mass({"qg": value}, {"qg": before}, dp, dp, transform,
                            scheme="none")
    assert np.array_equal(fixed["qg"], value)
    assert marks["semilag_tracer_mass_fixer_relative__qg"] > 0.0


def test_the_proportional_fixer_is_retired_and_refused_by_name():
    """Named breakage: a door knob whose two values produce the SAME run
    under two different config hashes.

    ``proportional`` weighted the correction by the signed advected value
    and ``bermejo_conde`` by its positive part.  Once every conservative
    form floors the field at zero BEFORE it measures the mass -- which it
    must, or the correction is short by exactly the mass the floor
    created -- the two weightings are the same array.  MEASURED
    2026-09-06 on the numpy path: bitwise identical output on a spiky
    field and on a cloudy one, with the quasi-monotone limiter on and
    off.  The arm it was kept for cannot answer anything, so it is
    refused rather than left selectable and reported as an independent
    reading.
    """
    assert "proportional" in RETIRED_TRACER_FIXERS
    assert "proportional" not in TRACER_FIXERS
    with pytest.raises(ValueError, match="is retired"):
        SemiLagrangianOptions(tracer_fixer="proportional")
    with pytest.raises(ValueError, match="is retired"):
        semilag_options_from_table({"tracer_fixer": "proportional"})
    transform = _transform()
    tables = _tables(transform)
    before, value, cut, dp = _one_species(transform, tables, 8)
    with pytest.raises(ValueError, match="is retired"):
        fix_mass({"qg": value}, {"qg": before}, dp, dp, transform,
                 scheme="proportional")


def test_the_weighting_the_retired_form_varied_is_now_one_array():
    """The measurement that retired it, kept so the idea is not rebuilt.

    Weighting by ``max(q, 0)`` and weighting by ``q`` are the same
    weighting once ``q`` has been floored at zero, and the floor is not
    optional.  The two produce the same field to the last bit, on the
    limited gather and on the unlimited one.
    """
    transform = _transform()
    tables = _tables(transform)
    nlev = 8
    cell = area_weights(transform)
    for monotone in (True, False):
        field = _spiky(tables, nlev)
        stencil = _shifted_stencil(tables, nlev, dx=0.37, dy=0.23, dk=0.11)
        value = gather_batch([field], stencil, monotone=monotone)[0]
        dp = np.full_like(field, 1000.0)
        fixed, _ = fix_mass({"qg": value.copy()}, {"qg": field}, dp, dp,
                            transform, scheme="bermejo_conde")
        # The retired form's arithmetic, written out here rather than
        # kept reachable through the door.
        floored = np.maximum(value, 0.0)
        weight = float(np.sum(floored * dp * cell))
        target = float(np.sum(field * dp * cell))
        after = float(np.sum(floored * dp * cell))
        signed = floored * (1.0 + (target - after) / weight)
        assert np.array_equal(fixed["qg"], signed)


def _blob(tables, nlev, *, thickness, width, seed=5, count=12):
    """A condensate-like field with a SHAPE: smooth blobs `thickness`
    model layers deep and `width` cells across, over a zero background.
    The species' own thickness is the variable, because that is what the
    four-point vertical stencil sees."""
    rng = np.random.default_rng(seed)
    nlat, nlon = tables.shape
    field = np.zeros((nlev, nlat, nlon), dtype=tables.dtype)
    for _ in range(count):
        k0 = int(rng.integers(2, max(3, nlev - thickness - 2)))
        j0 = int(rng.integers(4, nlat - width - 4))
        i0 = int(rng.integers(0, nlon))
        peak = float(10.0 ** rng.uniform(-4.0, -2.5))
        for dk in range(thickness):
            wz = math.sin(math.pi * (dk + 0.5) / thickness)
            for dj in range(width):
                wy = math.sin(math.pi * (dj + 0.5) / width)
                for di in range(width):
                    wx = math.sin(math.pi * (di + 0.5) / width)
                    field[k0 + dk, j0 + dj, (i0 + di) % nlon] += peak * wz * wy * wx
    return field


def test_the_fixer_correction_is_priced_by_the_species_shape_not_the_scheme():
    """What ``SEMILAG_TRACER_FIXER_RELATIVE_LIMIT`` is a function of.

    The re-priced gate reads 1.01e-1 on the T255 L40 forecast day and its
    limit is 0.25, which is 2.5x headroom AT THAT OPERATING POINT and not
    an invariant of the scheme.  The correction is what positivity costs
    on a species the four-point stencil cannot resolve, so it falls as
    the species gets thicker and wider and rises as it gets thinner and
    sharper, while the UNLIMITED gather's own mass error does not move at
    all.  A run at a sharper truncation or on a thinner vertical ladder
    therefore reads higher than the day the limit was priced on, and this
    gate is what says so before the limit is read as a margin.
    """
    transform = _transform()
    tables = _tables(transform)
    nlev = 40
    cell = area_weights(transform)
    dp = np.full((nlev, *tables.shape), 1000.0)
    stencil = _shifted_stencil(tables, nlev, dx=0.37, dy=0.23, dk=0.31)

    def correction(field):
        value = gather_batch([field], stencil, monotone=True)[0]
        _, marks = fix_mass({"qg": value}, {"qg": field}, dp, dp, transform,
                            scheme="bermejo_conde")
        return marks["semilag_tracer_mass_fixer_relative__qg"]

    def raw_error(field):
        raw = gather_batch([field], stencil, monotone=False)[0]
        target = float(np.sum(field * dp * cell))
        return abs(float(np.sum(raw * dp * cell)) - target) / target

    thin = _blob(tables, nlev, thickness=1, width=4)
    thick = _blob(tables, nlev, thickness=10, width=4)
    narrow = _blob(tables, nlev, thickness=3, width=2)
    broad = _blob(tables, nlev, thickness=3, width=10)

    # Thinner is dearer, and by an order of magnitude across the range a
    # condensate species actually occupies.
    assert correction(thin) > 5.0 * correction(thick)
    assert correction(narrow) > 3.0 * correction(broad)
    # A one-layer species is already past half the gate's own limit.
    assert correction(thin) > 0.5 * 0.25
    # And none of that is the interpolation's own conservation error,
    # which the unlimited gather holds at a few parts in a thousand
    # whatever shape the species has.
    for field in (thin, thick, narrow, broad):
        assert raw_error(field) < 1.0e-2


def test_the_six_hour_arms_and_the_forecast_day_are_two_cases():
    """Named breakage: a number from the six-hour arms quoted beside a
    number from the forecast day as though they were two points on one run.

    They are not.  The configuration of record for the six-hour arms reads
    the 2026-08-30 18Z analysis and the forecast day reads the 2026-09-01
    00Z one, so "5.96e-2 over six hours and 1.01e-1 over a whole forecast
    day" is two initial states and not an evolution.  An earlier writeup and
    two docstrings in this tree said one case for both; this gate is what
    makes the next writer look.
    """
    import tomllib
    root = _shipped_configs()
    six = tomllib.loads(
        (root / "arwen_global_gdas_t255_native_sl_si_6h.toml").read_text(
            encoding="utf-8")
    )
    day = tomllib.loads(
        (root / "arwen_global_gdas_t255_native_sl_si_24h.toml").read_text(
            encoding="utf-8")
    )
    assert six["initial"]["analysis_grib"] != day["initial"]["analysis_grib"]
    assert (six["physics"]["native_adapter_options"]["start_time_utc"]
            != day["physics"]["native_adapter_options"]["start_time_utc"])
    # And the day is the case the scorecard chain reads, which is the one
    # the grading lane's A/B is of.
    assert "2026-09-01" in day["physics"]["native_adapter_options"][
        "start_time_utc"]


# ---------------------------------------------------------------------------
# the physics coupling door


def test_the_three_couplings_are_the_weights_they_say_they_are():
    assert PHYSICS_COUPLINGS == ("advected", "arrival", "trajectory_average")
    assert PHYSICS_ARRIVAL_WEIGHT["advected"] == 0.0
    assert PHYSICS_ARRIVAL_WEIGHT["arrival"] == 1.0
    assert PHYSICS_ARRIVAL_WEIGHT["trajectory_average"] == 0.5
    assert SemiLagrangianOptions().physics_coupling == "advected"
    with pytest.raises(ValueError, match="semilag.physics_coupling must be"):
        SemiLagrangianOptions(physics_coupling="departure")
    parsed = semilag_options_from_table({"physics_coupling": "ARRIVAL"})
    assert parsed.physics_coupling == "arrival"
    # Each arm has its own identity, so two runs that differ only in where
    # the physics increment lands cannot share a config hash.
    identities = {
        semilag_options_from_table(
            {"physics_coupling": name}
        ).identity["physics_coupling"]
        for name in PHYSICS_COUPLINGS
    }
    assert identities == set(PHYSICS_COUPLINGS)


# ---------------------------------------------------------------------------
# KERN-2 through the tracer path of record, both schemes


@pytest.mark.parametrize("scheme", ["semi_lagrangian", "flux_form"])
def test_the_deformational_case_runs_both_schemes_through_one_driver(scheme):
    transform = _transform(21)
    run = run_deformational_transport(
        transform, case="nondivergent_two_cell", scheme=scheme, steps=24,
        distribution="gaussian_hills", nlev=4,
    )
    out = run.as_dict()
    assert out["scheme"] == scheme
    # The semi-Lagrangian arm's mass is closed by the fixer, exactly.  The
    # flux arm's moves by the pseudo-density mismatch, because the driver
    # re-references the mixing ratio to the same layer thickness every step
    # exactly as dynamics.step does, and the prescribed wind is
    # non-divergent analytically and not on the grid.  MEASURED 8.3e-4 on
    # this case, and flat in the step count, which is what says it is the
    # mismatch and not a Courant effect.
    if scheme == "semi_lagrangian":
        assert abs(out["tracer_mass_relative_change"]) < 1.0e-10
    else:
        assert abs(out["tracer_mass_relative_change"]) < 1.0e-3
        assert out["pseudo_density_relative_drift"] > 0.0
    # Both return a field, not a smear: T21 over a twelve-day deformation
    # is coarse, and what this gate holds is that the driver ran the
    # scheme it says it ran.
    assert 0.0 < out["l2"] < 2.0
    if scheme == "semi_lagrangian":
        assert out["tracer_fixer"] == "bermejo_conde_additive"
        assert out["mass_fixer_max_step_relative"] >= 0.0
    else:
        assert out["substeps_max"] >= 1


def test_the_flux_form_arm_is_refused_on_the_divergent_case():
    transform = _transform(21)
    with pytest.raises(ValueError, match="density assumption"):
        run_deformational_transport(
            transform, case="divergent", scheme="flux_form", steps=4,
        )


def test_the_limiter_keeps_a_slotted_cylinder_inside_its_own_bounds():
    """The limiter's OWN contract, with no fixer on top of it.

    The named breakage is unphysical extrema in a field with a
    discontinuity, which is what condensate looks like at a cloud edge.
    The fixer runs afterwards and is a global rescale, so it moves the
    background of a field that has one; that is measured in the arm
    below, not folded into the limiter's gate.
    """
    transform = _transform(21)
    plain = run_deformational_transport(
        transform, case="nondivergent_two_cell", scheme="semi_lagrangian",
        steps=24, distribution="slotted_cylinders", nlev=4, monotone=True,
        fixer="none",
    ).as_dict()
    assert plain["undershoot"] >= -1.0e-12
    assert plain["overshoot"] <= 1.0e-12
    # Unlimited, the same case rings: that is the measurement that says
    # what the limiter is worth rather than asserting it.
    ringing = run_deformational_transport(
        transform, case="nondivergent_two_cell", scheme="semi_lagrangian",
        steps=24, distribution="slotted_cylinders", nlev=4, monotone=False,
        fixer="none",
    ).as_dict()
    assert ringing["undershoot"] < -1.0e-3
    # With the fixer on, the bound the field keeps is the limiter's own
    # scaled by the correction the fixer made, and the arm reports it.
    fixed = run_deformational_transport(
        transform, case="nondivergent_two_cell", scheme="semi_lagrangian",
        steps=24, distribution="slotted_cylinders", nlev=4, monotone=True,
    ).as_dict()
    assert fixed["minimum"] >= 0.0
    assert abs(fixed["tracer_mass_relative_change"]) < 1.0e-10


def test_the_condensate_species_are_the_ones_the_water_gate_counts():
    # The water-relative gate sums the MASS species only: a number moment
    # is a count per kilogram and adding it to a water budget would be
    # adding a different unit to the same total.
    assert CONDENSATE_SPECIES == ("qc", "qr", "qi", "qs", "qg")
    assert all(name.startswith("q") for name in CONDENSATE_SPECIES)
    assert set(DISTRIBUTIONS) == {
        "cosine_bells", "gaussian_hills", "slotted_cylinders"
    }
    assert math.isfinite(1.0)


# ---------------------------------------------------------------------------
# the physics coupling, through the core


def _coupling_step(coupling, *, increment: float):
    """One semi-Lagrangian step with a synthetic first-half increment.

    The increment is applied to the state the step advances and the state
    it started from is handed to the model, so the core sees exactly what
    ``dynamics.step`` hands it after a real physics half, without a
    physics suite in the loop.  ``increment = 0`` makes the two states the
    same and every coupling has to agree.
    """
    import dataclasses

    from test_arwen_global_semilag_step import _baroclinic, _model

    from woof.globe.vertical import HybridCoordinate

    transform = _transform(21)
    vertical = HybridCoordinate.pressure_blend(12, 100.0)
    options = SemiLagrangianOptions(physics_coupling=coupling)
    model = _model(transform, vertical, options=options)
    bundle = _baroclinic(transform, vertical)
    pre = bundle.atmosphere
    post = dataclasses.replace(
        pre,
        theta=pre.theta * (1.0 + increment),
        qv=pre.qv * (1.0 - increment),
        divergence=pre.divergence * (1.0 + increment),
        qc=pre.qc * (1.0 - 2.0 * increment),
    )
    if coupling != "advected":
        model.hold_pre_physics(pre)
    advanced, metrics = model.integrate_dynamics(post, 900.0)
    return advanced, metrics


def test_the_couplings_agree_when_there_is_no_physics_to_couple():
    reference, _ = _coupling_step("advected", increment=0.0)
    for coupling in ("arrival", "trajectory_average"):
        got, _ = _coupling_step(coupling, increment=0.0)
        for name in ("vorticity", "divergence", "theta", "qv",
                     "log_surface_pressure", "qc"):
            assert np.array_equal(
                getattr(got, name), getattr(reference, name)
            ), name


def test_the_trajectory_average_is_the_midpoint_of_the_other_two():
    advected, _ = _coupling_step("advected", increment=0.02)
    arrival, _ = _coupling_step("arrival", increment=0.02)
    middle, _ = _coupling_step("trajectory_average", increment=0.02)
    for name in ("theta", "qv", "log_surface_pressure"):
        a = np.asarray(getattr(advected, name))
        b = np.asarray(getattr(arrival, name))
        m = np.asarray(getattr(middle, name))
        # The arms differ, and the average arm sits halfway.  The whole
        # path from the correction to the solved field is linear in the
        # correction, so this is an identity and the only thing separating
        # the two orderings of it is roundoff, measured against the field's
        # own scale rather than entry by entry (a spectral field spans
        # fifteen decades and a per-entry relative tolerance would be a
        # gate on the smallest coefficient's roundoff, not on the scheme).
        scale = float(np.max(np.abs(a)))
        assert float(np.max(np.abs(b - a))) > 1.0e-9 * scale
        assert float(np.max(np.abs(m - 0.5 * (a + b)))) <= 1.0e-12 * scale


def test_a_coupling_that_needs_the_pre_physics_state_refuses_without_it():
    from test_arwen_global_semilag_step import _baroclinic, _model

    from woof.globe.vertical import HybridCoordinate

    transform = _transform(21)
    vertical = HybridCoordinate.pressure_blend(12, 100.0)
    model = _model(transform, vertical,
                   options=SemiLagrangianOptions(physics_coupling="arrival"))
    bundle = _baroclinic(transform, vertical)
    with pytest.raises(ValueError, match="did not hand one over"):
        model.integrate_dynamics(bundle.atmosphere, 900.0)


def test_the_reported_fixer_magnitude_is_a_relative_number_only():
    """The step reports ONE maximum for the tracer fixer and it has to be
    the relative one.

    The fixer's per-species record carries three rows per species: the
    relative correction, the correction in the species' own units, and the
    share the clip deficit carried.  A number moment is a count per
    kilogram, so its correction in its own units is a number like 2.6e9,
    and a maximum taken over the whole record reports that as the relative
    magnitude and fails every gate that reads it.  MEASURED: it did, on the
    first six-hour arm of 2026-09-06.
    """
    from woof.globe.semilag.step import semilag_step  # noqa: F401
    from test_arwen_global_semilag_step import _baroclinic, _model

    from woof.globe.vertical import HybridCoordinate

    transform = _transform(21)
    vertical = HybridCoordinate.pressure_blend(12, 100.0)
    model = _model(transform, vertical)
    bundle = _baroclinic(transform, vertical)
    # A number moment three orders of magnitude larger than any mixing
    # ratio, which is what a real one is.
    import dataclasses
    state = dataclasses.replace(bundle.atmosphere,
                                nc=bundle.atmosphere.qc * 1.0e9)
    _advanced, metrics = model.integrate_dynamics(state, 900.0)
    reported = float(metrics["semilag_tracer_mass_fixer_relative"])
    relatives = [
        float(v) for k, v in metrics.items()
        if k.startswith("semilag_tracer_mass_fixer_relative__")
    ]
    assert relatives
    assert reported == max(relatives)
    assert reported <= 1.0
    assert float(metrics["tracer_transport"]["mass_fixer_max_relative"]) == reported


# ---------------------------------------------------------------------------
# the limiter shapes


def test_the_limiter_shape_that_was_measured_and_left_out():
    """Why the gather has ONE limiter shape and not two.

    A positive-definite shape, [0, cell maximum] rather than [cell
    minimum, cell maximum], is the obvious next idea: a species that is
    zero nearly everywhere has no meaningful local minimum, and holding it
    to one is what lifts an undershoot and creates mass.  It was built and
    measured on 2026-09-06 and it is not in the tree, and this gate is the
    measurement that says why, so the idea is not rebuilt from the same
    reasoning next time.

    On a condensate-shaped field the two shapes are the same array at
    every point, because every undershoot the quasi-monotone shape lifts
    is already below zero there.  On a field with a positive floor they
    part, and the positive-definite one is 27 times further from the mass
    it started with, because dropping the lower bound stops the lifted
    undershoots from paying for the cut overshoots.
    """
    transform = _transform()
    tables = _tables(transform)
    nlev = 8
    cell = area_weights(transform)
    stencil = _shifted_stencil(tables, nlev, dx=0.37, dy=0.23, dk=0.11)

    def shapes(field):
        dp = np.full_like(field, 1000.0)
        start_mass = float(np.sum(field * dp * cell))
        out = {}
        for name, limiter in (("raw", "none"), ("qm", "quasi_monotone")):
            value = gather_batch([field], stencil, monotone=limiter)[0]
            out[name] = value
            out[name + "_err"] = abs(
                float(np.sum(value * dp * cell)) - start_mass
            ) / start_mass
        # [0, cell maximum], computed here from the limited and unlimited
        # gathers the tree does ship: the quasi-monotone value wherever it
        # cut an overshoot, the raw value wherever it lifted, floored.
        out["pd"] = np.maximum(
            np.minimum(out["raw"], out["qm"]), 0.0
        )
        out["pd_err"] = abs(
            float(np.sum(out["pd"] * dp * cell)) - start_mass
        ) / start_mass
        return out

    sparse = shapes(_cloudy(tables, nlev))
    assert float(np.min(sparse["raw"])) < 0.0
    assert np.array_equal(sparse["qm"], sparse["pd"])
    assert sparse["qm_err"] > 5.0 * sparse["raw_err"]

    nlat, nlon = tables.shape
    plateau = np.full((nlev, nlat, nlon), 0.5, dtype=tables.dtype)
    plateau[:, nlat // 3: 2 * nlat // 3, nlon // 4: 3 * nlon // 4] = 1.0
    step = shapes(plateau)
    assert float(np.min(step["raw"])) > 0.0
    assert not np.array_equal(step["qm"], step["pd"])
    assert step["pd_err"] > 10.0 * step["qm_err"]


def test_the_deficit_follows_the_limiter_that_produced_it():
    transform = _transform()
    tables = _tables(transform)
    nlev = 8
    field = _spiky(tables, nlev)
    stencil = _shifted_stencil(tables, nlev, dx=0.37, dy=0.23, dk=0.11)
    for limiter in ("quasi_monotone",):
        plain = gather_batch([field], stencil, monotone=limiter)[0]
        value, cut = gather_batch([field], stencil, monotone=limiter,
                                  deficit=True)[0]
        raw = gather_batch([field], stencil, monotone="none")[0]
        assert np.array_equal(plain, value)
        assert np.allclose(value + cut, raw, rtol=0.0, atol=0.0)


def test_the_limiter_door_names_its_shapes():
    from woof.globe.semilag.interpolate import LIMITERS

    assert LIMITERS == ("quasi_monotone", "none")
    assert SemiLagrangianOptions().quasi_monotone is True
    with pytest.raises(ValueError, match="limiter must be one of"):
        gather_batch(
            [np.zeros((8, *_tables(_transform()).shape))],
            _shifted_stencil(_tables(_transform()), 8, dx=0.1),
            monotone="monotone",
        )
    # The identity carries the limiter, so two runs that differ only in
    # whether it is on cannot share a config hash.
    assert "quasi_monotone" in SemiLagrangianOptions().identity
