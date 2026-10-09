"""The refractivity and wind-at-pressure operators: the two synthetic
families both directions, the member-batched operator against the column
functions, and the refusals by name."""
from __future__ import annotations

import math

import numpy as np
import pytest

from woof.globe import obs_operators as ops
from woof.globe.obs_operators import (
    CALIBRATION_BARS,
    REFRACTIVITY_REQUIRED_FIELDS,
    RefractivityOperator,
    calibrate,
    column_heights_m,
    interp_ln_pressure,
    planted_member_operator,
    refractivity_at_heights,
    refractivity_n,
)


def test_refractivity_matches_the_two_term_form_at_hand_values():
    # Sea level, 288.15 K, dry: 77.6 * 1013.25 / 288.15 = 272.86 N-units.
    assert abs(refractivity_n(101325.0, 288.15, 0.0) - 77.6 * 1013.25 / 288.15) < 1e-9
    # 18 g/kg at 1008 hPa and 300 K adds the wet term 3.73e5 e / T^2 with e = q p / (eps + (1-eps) q).
    e_hpa = 0.018 * 1008.0 / (0.622 + 0.378 * 0.018)
    expected = 77.6 * 1008.0 / 300.0 + 3.73e5 * e_hpa / 300.0 ** 2
    assert abs(refractivity_n(100800.0, 300.0, 0.018) - expected) < 1e-9


def test_column_heights_follow_the_isa_column():
    t, q, p, ps, phi = ops._isa_column(400)
    z = column_heights_m(t, q, p, ps, phi)
    assert np.max(np.abs(z - ops._isa_height_m(p))) < 0.2
    # a surface geopotential lifts the whole column by phi / g
    lifted = column_heights_m(t, q, p, ps, 9.80616 * 1500.0)
    assert np.allclose(lifted - z, 1500.0)


def test_calibration_passes_both_families_both_directions():
    report = calibrate()
    assert report["pass"], report
    for name in ("isa_dry", "tropical_moist"):
        family = report["families"][name]
        assert family["read_back_max_fraction"] <= CALIBRATION_BARS["refractivity_read_back_max_fraction"]
        assert family["read_back_median_fraction"] <= CALIBRATION_BARS["refractivity_read_back_median_fraction"]
        assert family["response_vs_fine_median_n_over_t"] <= CALIBRATION_BARS["refractivity_response_median_n_over_t"]
        assert family["response_surface_vs_analytic_fraction"] <= CALIBRATION_BARS["refractivity_surface_response_fraction"]
        assert family["unchanged_column_moves"] == 0.0
        assert family["outside_span_refused"]
        member = report["member_operator"][name]
        assert member["member_minus_column_max"] <= 1e-9
    # the worst read-back sits at a lapse-rate kink, as the module states
    assert abs(report["families"]["isa_dry"]["read_back_worst_target_m"] - 11000.0) < 1500.0
    wind = report["wind_at_pressure"]
    assert wind["linear_in_ln_p_read_back_m_s"] <= CALIBRATION_BARS["wind_linear_read_back_m_s"]
    assert wind["jet_read_back_max_m_s"] <= CALIBRATION_BARS["wind_jet_read_back_m_s"]


def test_targets_outside_the_column_are_refused_not_extrapolated():
    t, q, p, ps, phi = ops._isa_column()
    z = column_heights_m(t, q, p, ps, phi)
    values, _ = refractivity_at_heights(t, q, p, ps, phi, np.array([z.min() - 1.0, z.max() + 1.0, 5000.0]))
    assert np.isnan(values[0]) and np.isnan(values[1]) and np.isfinite(values[2])


def test_wind_at_pressure_reads_a_linear_profile_exactly_and_refuses_outside():
    p = np.exp(np.linspace(math.log(1000.0), math.log(100_000.0), 40))
    profile = 3.0 + 4.0 * np.log(p / 1.0e5)
    targets = np.array([50_000.0, 25_000.0, 500.0, 101_000.0])
    got = interp_ln_pressure(profile, p, targets)
    assert abs(got[0] - (3.0 + 4.0 * math.log(0.5))) < 1e-9
    assert abs(got[1] - (3.0 + 4.0 * math.log(0.25))) < 1e-9
    assert np.isnan(got[2]) and np.isnan(got[3])


def test_the_member_operator_reads_the_column_and_refuses_a_member_missing_a_field():
    t, q, p, ps, phi = ops._tropical_column()
    operator = planted_member_operator(t, q, p, ps, phi)
    z = column_heights_m(t, q, p, ps, phi, gravity=ops.MODEL_GRAVITY_M_S2,
                         gas_constant=ops.MODEL_DRY_AIR_GAS_CONSTANT)
    targets = np.array([2000.0, 5000.0, 12000.0, 2000.0])
    batch = ops._Batch([10.0, 10.0, 10.0, -30.0], [20.0, 20.0, 20.0, 40.0], targets)
    members = [ops._PlantedMember(t, q, p, ps), ops._PlantedMember(t, q, p, ps)]
    values = operator(members, batch)
    assert values.shape == (2, 4)
    column, _ = refractivity_at_heights(t, q, p, ps, phi, targets, gravity=ops.MODEL_GRAVITY_M_S2,
                                        gas_constant=ops.MODEL_DRY_AIR_GAS_CONSTANT)
    assert np.allclose(values[0], column, rtol=0, atol=1e-9)
    # two identical members read identical bits, and two points at one height agree
    assert np.array_equal(values[0], values[1])
    assert values[0, 0] == values[0, 3]
    assert z.min() < 2000.0 < z.max()
    # a member without qv is refused by that field's name
    broken = ops._PlantedMember(t, q, p, ps)
    broken.atmosphere.qv = None
    with pytest.raises(ValueError, match="lacks the spectral field\\(s\\) \\['qv'\\]"):
        operator([members[0], broken], batch)
    assert REFRACTIVITY_REQUIRED_FIELDS == ("log_surface_pressure", "theta", "qv")
    # and a batch of another variable is refused too
    with pytest.raises(ValueError, match="evaluates refractivity_n rows only"):
        operator(members, ops._Batch([0.0], [0.0], [1000.0], variable="temperature_k"))
    # an empty batch is an empty answer, not an error
    assert operator(members, ops._Batch([], [], [])).shape == (2, 0)


def test_the_member_operator_class_takes_the_vertical_coordinate_and_terrain():
    t, q, p, ps, phi = ops._isa_column()
    vertical = ops._PlantedVertical(p)
    assert np.allclose(np.sqrt(vertical.a_half_pa[:-1] * vertical.a_half_pa[1:]), p)
    op = RefractivityOperator(ops._PlantedTransform(), vertical, np.array([0.0]),
                              sampler=lambda tr, c, lat, lon: ops._planted_sampler_stack(c, lat),
                              host=lambda tr, c: np.asarray(c, dtype=np.float64))
    values = op([ops._PlantedMember(t, q, p, ps)], ops._Batch([0.0], [0.0], [3000.0]))
    column, _ = refractivity_at_heights(t, q, p, ps, phi, np.array([3000.0]),
                                        gravity=ops.MODEL_GRAVITY_M_S2,
                                        gas_constant=ops.MODEL_DRY_AIR_GAS_CONSTANT)
    assert abs(values[0, 0] - column[0]) < 1e-9
