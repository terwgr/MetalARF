"""The shipped hyperdiffusion shape (woof.globe.config, 2026-09-04).

Order 8 at 2160 s at the truncation replaced order 4 at 14400 s after the
energy ledger read the control's 250 hPa tail fed faster than it was
drained on the last fifteen degrees (a flat tail at 0.22 to 0.27 of the
observed spectrum).  These tests hold the default where every door reads
it, the three shipped configs beside it, and the arithmetic of the shape:
the drain per day at named degrees of T255, and the crossing of the two
shapes at n = 202 (0.79 of the truncation).
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import math
from pathlib import Path

import numpy as np
import pytest

# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = pytest.mark.cpu_only

from woof.globe.config import (  # noqa: E402
    DEFAULT_DIFFUSION_EFOLD_S,
    DEFAULT_DIFFUSION_ORDER,
    SEMILAG_DIFFUSION_EFOLD_S,
    SEMILAG_DIFFUSION_ORDER,
    load_config,
)
from woof.globe.spectral.diffusion import ExponentialHyperdiffusion  # noqa: E402
from woof.globe.spectral.transform import SphericalHarmonicTransform  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
#: The shipped EULERIAN configs carry the Eulerian default (the T255 one is
#: the Eulerian config of record at its rule step of 90 s, the ten-step
#: identity gate's config since 2026-09-06).  The quickstart, the arms of
#: record and the bare configs of record run the semi-Lagrangian core (the
#: shipped default core, 2026-09-06) and read that core's own drain,
#: SEMILAG_DIFFUSION_*, chosen by the coupling lane's dry ladder, whether
#: they write it out (the arms) or say nothing (the bare records);
#: SEMILAG_SHIPPED_CONFIGS below.
SHIPPED_CONFIGS = (
    str(_shipped_configs() / "arwen_global_gdas_t255_native_imex_24h.toml"),
    str(_shipped_configs() / "arwen_global_gdas_t533_24h.toml"),
)
SEMILAG_SHIPPED_CONFIGS = (
    str(_shipped_configs() / "arwen_global_t255_quickstart.toml"),
    str(_shipped_configs() / "arwen_global_gdas_t255_native_sl_si_24h.toml"),
    str(_shipped_configs() / "arwen_global_gdas_t383_native_sl_si_24h.toml"),
    str(_shipped_configs() / "arwen_global_gdas_t255_native_24h.toml"),
    str(_shipped_configs() / "arwen_global_gdas_t255_native_24h_bare.toml"),
    str(_shipped_configs() / "arwen_global_gdas_t383_native_24h.toml"),
    str(_shipped_configs() / "arwen_global_gdas_t533_native_24h.toml"),
    str(_shipped_configs() / "arwen_global_gdas_t255_jet48_24h.toml"),
)


def _drain_per_day(order: int, tau_s: float, n: int, truncation: int) -> float:
    """Kinetic-energy drain per day at degree ``n`` for a unit-strength
    field: the coefficient decays at ratio^order / tau, the energy at
    twice that (the ledger books energy; divergence carries strength 1.5)."""
    ratio = n * (n + 1.0) / (truncation * (truncation + 1.0))
    return 2.0 * 86400.0 / tau_s * ratio ** order


def test_the_default_is_order_8_at_36_minutes():
    assert DEFAULT_DIFFUSION_ORDER == 8
    assert DEFAULT_DIFFUSION_EFOLD_S == 2160.0
    assert ExponentialHyperdiffusion(order=DEFAULT_DIFFUSION_ORDER, e_folding_time_s_at_truncation=DEFAULT_DIFFUSION_EFOLD_S).order == 8


@pytest.mark.parametrize("relative", SHIPPED_CONFIGS)
def test_every_shipped_config_carries_the_default(relative):
    cfg = load_config(ROOT / relative)
    assert cfg.diffusion_enabled
    assert cfg.diffusion_order == DEFAULT_DIFFUSION_ORDER
    assert cfg.diffusion_efold_s == DEFAULT_DIFFUSION_EFOLD_S
    assert cfg.diffusion_preserve_degree == 1


@pytest.mark.parametrize("relative", SEMILAG_SHIPPED_CONFIGS)
def test_every_shipped_semi_lagrangian_config_carries_that_core_s_own_drain(relative):
    cfg = load_config(ROOT / relative)
    assert cfg.integrator == "sl_si"
    assert cfg.diffusion_enabled
    assert cfg.diffusion_order == SEMILAG_DIFFUSION_ORDER == 16
    assert cfg.diffusion_efold_s == SEMILAG_DIFFUSION_EFOLD_S == 720.0
    assert cfg.diffusion_preserve_degree == 1


def test_a_config_without_a_diffusion_table_reads_the_default(tmp_path):
    source = (ROOT / SHIPPED_CONFIGS[0]).read_text(encoding="utf-8")
    head, _, rest = source.partition("[diffusion]")
    tail = rest[rest.index("\n["):]
    (tmp_path / "bare.toml").write_text(head + tail, encoding="utf-8")
    cfg = load_config(tmp_path / "bare.toml")
    assert cfg.diffusion_order == DEFAULT_DIFFUSION_ORDER
    assert cfg.diffusion_efold_s == DEFAULT_DIFFUSION_EFOLD_S


def test_the_shape_drains_the_last_degrees_harder_and_250_km_six_times_less():
    """T255: the kinetic-energy drain per day of both shapes at the
    degrees the ledger reading named (unit strength), and their crossing
    at n = 202: above it the shipped shape drains harder (1.4x at n = 210,
    2.1x at 220, 3.5x at 235, 6.7x at the truncation), below it less
    (0.31 to 0.052 per day at n = 161, 250 km: e-folding 78 h to 462 h)."""
    t = 255
    new = {n: _drain_per_day(DEFAULT_DIFFUSION_ORDER, DEFAULT_DIFFUSION_EFOLD_S, n, t) for n in (161, 200, 210, 220, 235, 245, 255)}
    old = {n: _drain_per_day(4, 14400.0, n, t) for n in (161, 200, 210, 220, 235, 245, 255)}
    assert new[255] == pytest.approx(80.0, rel=1e-12)
    assert new[245] == pytest.approx(42.2, abs=0.1)
    assert new[235] == pytest.approx(21.7, abs=0.1)
    assert new[220] == pytest.approx(7.57, abs=0.02)
    assert new[210] == pytest.approx(3.60, abs=0.02)
    assert new[200] == pytest.approx(1.65, abs=0.01)
    assert new[161] == pytest.approx(0.052, abs=0.001)
    assert old[255] == pytest.approx(12.0, rel=1e-12)
    assert old[245] == pytest.approx(8.72, abs=0.02)
    assert old[235] == pytest.approx(6.25, abs=0.02)
    assert old[220] == pytest.approx(3.69, abs=0.02)
    assert old[210] == pytest.approx(2.55, abs=0.02)
    assert old[200] == pytest.approx(1.73, abs=0.01)
    assert old[161] == pytest.approx(0.306, abs=0.002)
    assert old[161] / new[161] == pytest.approx(5.9, abs=0.05)
    assert new[255] / old[255] == pytest.approx(6.67, abs=0.01)
    assert 24.0 / new[161] == pytest.approx(462.0, abs=1.0)
    assert 24.0 / old[161] == pytest.approx(78.5, abs=0.5)
    crossing = [n for n in range(150, 256) if _drain_per_day(DEFAULT_DIFFUSION_ORDER, DEFAULT_DIFFUSION_EFOLD_S, n, t) >= _drain_per_day(4, 14400.0, n, t)][0]
    assert crossing == 202, crossing
    # The transform's own factors carry the same arithmetic at dt = 50 s.
    transform = SphericalHarmonicTransform.create(t, backend="numpy", precision="float64")
    factors = np.asarray(ExponentialHyperdiffusion(order=DEFAULT_DIFFUSION_ORDER, e_folding_time_s_at_truncation=DEFAULT_DIFFUSION_EFOLD_S, preserve_degree=1).factors(transform, 50.0))
    assert factors[255] == pytest.approx(math.exp(-50.0 / 2160.0), rel=1e-12)
    assert factors[161] == pytest.approx(math.exp(-50.0 / 2160.0 * (161 * 162 / (255 * 256)) ** 8), rel=1e-12)
    assert factors[0] == 1.0 and factors[1] == 1.0
