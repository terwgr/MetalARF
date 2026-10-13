"""Conservation gates for the actual WSM6 production shader helper bodies."""
import numpy as np
import pytest

from tools.wsm6_remap_check import ShaderCPU, integral_reference, rain_reproducer, water_balance, EPS32


@pytest.fixture(scope='module')
def shader(tmp_path_factory):
    return ShaderCPU(tmp_path_factory.mktemp('wsm6-shader'))


@pytest.mark.parametrize('shift', [0., np.nextafter(np.float32(50), np.float32(0)), 50.,
                                  np.nextafter(np.float32(50), np.float32(np.inf)), 175., 325.])
def test_remap_counts_partial_top_cell_and_surface_outflow(shader, shift):
    zi = np.array([0, 150, 300], np.float32)
    za = np.asarray(zi - np.float32(shift), np.float32)
    dza = np.array([150, 150, max(float(shift), 1.)], np.float32)
    qa = np.array([1e-4, 1e-4, 0], np.float32)
    actual, rain = shader.remap(qa, zi, za, dza)
    expected, expected_rain = integral_reference(qa, zi, za, dza)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-11)
    np.testing.assert_allclose(rain, expected_rain, rtol=2e-6, atol=2e-10)
    assert (actual >= 0).all()
    assert abs(float(np.dot(actual.astype(float), np.diff(zi.astype(float)))) + rain
               - float(np.dot(qa[:-1].astype(float), dza[:-1].astype(float)))) <= 16*EPS32*.03


def test_partial_floor_precipitation_integrates_the_linear_profile(shader):
    zi = np.array([0, 100, 200, 300, 400], np.float32)
    za = zi - np.float32(125)
    dza = np.array([100,100,100,100,125], np.float32)
    qa = np.array([1e-4,2e-4,3e-4,4e-4,0], np.float32)
    actual, rain = shader.remap(qa, zi, za, dza)
    expected, expected_rain = integral_reference(qa, zi, za, dza)
    np.testing.assert_allclose(actual, expected, rtol=4e-6, atol=2e-11)
    assert abs(rain-expected_rain) <= 2e-8


def test_arrival_to_remap_seeded_columns_conserve_and_match_integral(shader):
    rng = np.random.default_rng(0x57534D36)
    for nz in (2,8,64):
        for _ in range(12):
            q = np.asarray(10**rng.uniform(-9,-2,nz), np.float32)
            rho = np.asarray(rng.uniform(.2,1.5,nz), np.float32)
            dz = np.asarray(rng.uniform(10,600,nz), np.float32)
            velocity = np.asarray(rng.uniform(0,20,nz), np.float32)
            zi, za, dza, qa = shader.arrival(q*rho, velocity, dz, rho, 30.)
            actual, rain = shader.remap(qa, zi, za, dza)
            expected, expected_rain = integral_reference(qa, zi, za, dza)
            np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=3e-8)
            np.testing.assert_allclose(rain, expected_rain, rtol=5e-5, atol=3e-7)
            before = float(np.dot((q*rho).astype(float), dz.astype(float)))
            after = float(np.dot(actual.astype(float), np.diff(zi.astype(float)))) + rain
            assert abs(after-before) <= 32*EPS32*max(before,1e-6)
            assert (actual >= 0).all()


@pytest.mark.parametrize('hail_opt', [0,1])
def test_complete_column_neighbors_conserve_water(shader, hail_opt):
    _, inputs = rain_reproducer()
    result = shader.column(inputs, hail_opt=hail_opt)
    mass, residual = water_balance(inputs, result)
    assert np.all(np.abs(residual) <= 32*EPS32*mass + 8*EPS32*.1)
    for key in ('qv','qc','qi','qr','qs','qg'):
        assert (result[key] >= 0).all()
