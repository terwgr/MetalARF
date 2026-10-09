"""The Eulerian core's shipped step is a rule, not a number typed into a
config: the largest whole step (a multiple of 5 s dividing the hour) whose
spectral CFL at the strongest analysis wind on disk stays at or under
DEFAULT_EULERIAN_STEP_RULE_FRACTION of the gate.  These tests pin the
arithmetic of the rule, the config door that applies it when ``[time] dt_s``
is omitted, the refusal that names the step a stronger day would have
admitted, and the receipt block every run carries so a refusal is explained
by numbers rather than a traceback.

The wind ceilings themselves are MEASURED values (config.EULERIAN_STEP_WIND_
CEILING_M_S, tools/arwen_global_cfl_sweep.py) and are not re-derived here;
the arithmetic is tested with injected winds so the test does not move when
a stronger day is added to the table.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import math
import textwrap
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from woof.globe.constants import DEFAULT_EULERIAN_STEP_RULE_FRACTION, EARTH_RADIUS_M
from woof.globe.config import (
    DEFAULT_SEMILAG_STEP_S,
    EULERIAN_STEP_CANDIDATES_S,
    EULERIAN_STEP_WIND_CEILING_M_S,
    SEMILAG_OFF_CENTRING_WEIGHT,
    SHIPPED_INTEGRATOR,
    default_eulerian_step_s,
    eulerian_wind_ceiling_m_s,
    load_config,
)
from woof.globe.semilag.step import SEMILAG_INTEGRATORS
from woof.globe.dynamics import spectral_cfl_refusal
from woof.globe.runner import cfl_headroom_receipt

MOIST_SMOKE = Path(str(_shipped_configs() / "arwen_global_moist_smoke.toml"))
T255_IMEX = Path(str(_shipped_configs() / "arwen_global_gdas_t255_native_imex_24h.toml"))
T383_IMEX = Path(str(_shipped_configs() / "arwen_global_gdas_t383_native_imex_24h.toml"))
T533_IMEX = Path(str(_shipped_configs() / "arwen_global_gdas_t533_native_imex_24h.toml"))
#: The configs of record, one per truncation: bare, naming no integrator and
#: no step, so a run of each with no flag is the shipped default core.
RECORD_CONFIGS = (
    Path(str(_shipped_configs() / "arwen_global_gdas_t255_native_24h.toml")),
    Path(str(_shipped_configs() / "arwen_global_gdas_t383_native_24h.toml")),
    Path(str(_shipped_configs() / "arwen_global_gdas_t533_native_24h.toml")),
)


def _rate(wind_m_s: float, truncation: int) -> float:
    return wind_m_s * math.sqrt(truncation * (truncation + 1.0)) / EARTH_RADIUS_M


def test_candidate_steps_are_multiples_of_five_seconds_dividing_the_hour():
    assert EULERIAN_STEP_CANDIDATES_S
    for step in EULERIAN_STEP_CANDIDATES_S:
        assert step % 5.0 == 0.0
        assert 3600.0 % step == 0.0
    assert 90.0 in EULERIAN_STEP_CANDIDATES_S
    assert 100.0 in EULERIAN_STEP_CANDIDATES_S
    assert 120.0 in EULERIAN_STEP_CANDIDATES_S
    assert 70.0 not in EULERIAN_STEP_CANDIDATES_S  # does not divide the hour


@pytest.mark.parametrize("truncation", [255, 383, 533])
def test_rule_picks_the_largest_candidate_under_the_fraction(truncation):
    wind = 130.0
    step = default_eulerian_step_s(truncation, 0.75, wind_m_s=wind)
    bound = DEFAULT_EULERIAN_STEP_RULE_FRACTION * 0.75
    assert step in EULERIAN_STEP_CANDIDATES_S
    assert step * _rate(wind, truncation) <= bound + 1e-12
    larger = [s for s in EULERIAN_STEP_CANDIDATES_S if s > step]
    assert all(s * _rate(wind, truncation) > bound for s in larger)


def test_rule_is_monotone_in_the_wind_and_the_truncation():
    weak = default_eulerian_step_s(255, 0.75, wind_m_s=60.0)
    strong = default_eulerian_step_s(255, 0.75, wind_m_s=180.0)
    assert weak > strong
    assert default_eulerian_step_s(255, 0.75, wind_m_s=130.0) >= default_eulerian_step_s(533, 0.75, wind_m_s=130.0)


def test_rule_refuses_when_no_candidate_step_satisfies_the_gate():
    with pytest.raises(ValueError, match="no candidate step"):
        default_eulerian_step_s(255, 1.0e-9, wind_m_s=130.0)


def test_wind_ceiling_table_is_measured_per_truncation_and_nearest_elsewhere():
    assert set(EULERIAN_STEP_WIND_CEILING_M_S) >= {255, 383, 533}
    for wind in EULERIAN_STEP_WIND_CEILING_M_S.values():
        assert 100.0 < wind < 200.0  # an analysis jet, not a typo
    assert eulerian_wind_ceiling_m_s(255) == EULERIAN_STEP_WIND_CEILING_M_S[255]
    assert eulerian_wind_ceiling_m_s(799) == EULERIAN_STEP_WIND_CEILING_M_S[533]
    assert eulerian_wind_ceiling_m_s(63) == EULERIAN_STEP_WIND_CEILING_M_S[255]


@pytest.mark.parametrize("path", [T255_IMEX, T383_IMEX, T533_IMEX])
def test_shipped_eulerian_configs_carry_the_rule_step(path):
    """The verify configs are the door: each shipped Eulerian config's dt is
    the rule's step for its truncation, so a reader can recompute it from
    the wind ceiling and the config's own gate."""
    cfg = load_config(path)
    assert cfg.integrator == "imex_ssp3"
    assert cfg.dt_s == default_eulerian_step_s(cfg.truncation, cfg.maximum_cfl)
    # and the step keeps the strongest day on disk under the fraction
    ceiling = eulerian_wind_ceiling_m_s(cfg.truncation)
    assert cfg.dt_s * _rate(ceiling, cfg.truncation) <= DEFAULT_EULERIAN_STEP_RULE_FRACTION * cfg.maximum_cfl + 1e-12
    # the output cadence and the radiation bucket land on whole steps
    assert (cfg.output_interval_s / cfg.dt_s) == int(cfg.output_interval_s / cfg.dt_s)
    rad = float(cfg.native_adapter_options["radiation_interval_s"])
    assert abs(rad / cfg.dt_s - round(rad / cfg.dt_s)) < 1e-9
    assert float(cfg.native_adapter_options["land_surface_interval_s"]) == cfg.dt_s


def test_the_shipped_default_core_is_the_semi_lagrangian_one():
    """A config that names no integrator runs sl_si: the default at every
    truncation since 2026-09-06 (the equal-cost grade, the door page)."""
    assert SHIPPED_INTEGRATOR == "sl_si"
    assert SHIPPED_INTEGRATOR in SEMILAG_INTEGRATORS


@pytest.mark.parametrize("path", RECORD_CONFIGS)
def test_the_configs_of_record_are_bare_and_run_the_default_core(path):
    """Each config of record names no integrator and no step in its text, so
    a bare run of it is the shipped default: the semi-Lagrangian core at
    300 s with the graded arm's off-centring, the radiation bucket a whole
    number of steps, the land bucket the step, the output cadence whole."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    assert "integrator" not in raw["time"] and "dt_s" not in raw["time"]
    cfg = load_config(path)
    assert cfg.integrator == SHIPPED_INTEGRATOR == "sl_si"
    assert cfg.dt_s == DEFAULT_SEMILAG_STEP_S == 300.0
    assert cfg.semi_implicit_off_centring_weight == SEMILAG_OFF_CENTRING_WEIGHT == 0.55
    assert (cfg.output_interval_s / cfg.dt_s) == int(cfg.output_interval_s / cfg.dt_s)
    rad = float(cfg.native_adapter_options["radiation_interval_s"])
    assert abs(rad / cfg.dt_s - round(rad / cfg.dt_s)) < 1e-9
    assert float(cfg.native_adapter_options["land_surface_interval_s"]) == cfg.dt_s


def test_off_centring_default_follows_the_integrator(tmp_path):
    bare_sl = load_config(_write(tmp_path, """
        [time]
        duration_s = 1200.0

        [semi_implicit]
        enabled = true
        weight = 1.0
    """, drop=("[time]", "[semi_implicit]")))
    assert bare_sl.integrator == "sl_si" and bare_sl.dt_s == 300.0
    assert bare_sl.semi_implicit_off_centring_weight == 0.55
    truncation = load_config(MOIST_SMOKE).truncation
    step = default_eulerian_step_s(truncation, 0.75)
    eulerian = load_config(_write(tmp_path, f"""
        [time]
        duration_s = {4 * step}
        integrator = "imex_ssp3"
    """))
    assert eulerian.semi_implicit_off_centring_weight == 0.5


def _write(tmp_path: Path, body: str, drop: tuple[str, ...] = ("[time]",)) -> Path:
    base = MOIST_SMOKE.read_text(encoding="utf-8")
    # drop the smoke config's own tables named in ``drop`` and append ours
    lines = []
    skipping = False
    for line in base.splitlines():
        head = line.strip()
        if head.startswith("[") and head in drop:
            skipping = True
            continue
        if skipping and head.startswith("["):
            skipping = False
        if not skipping:
            lines.append(line)
    out = tmp_path / "cfg.toml"
    out.write_text("\n".join(lines) + "\n" + textwrap.dedent(body), encoding="utf-8")
    return out


def test_config_without_dt_gets_the_rule_step_on_the_eulerian_core(tmp_path):
    # the duration has to be a whole number of the rule's steps, so it is
    # written from the rule for the smoke config's own truncation
    truncation = load_config(MOIST_SMOKE).truncation
    step = default_eulerian_step_s(truncation, 0.75)
    cfg = load_config(_write(tmp_path, f"""
        [time]
        duration_s = {4 * step}
        integrator = "imex_ssp3"
    """))
    assert cfg.dt_s == step == default_eulerian_step_s(cfg.truncation, cfg.maximum_cfl)


def test_config_without_dt_gets_300_s_on_the_semi_lagrangian_core(tmp_path):
    # the semi-Lagrangian core refuses a partial semi-implicit weight by
    # name, so the smoke config's [semi_implicit] table is replaced too
    cfg = load_config(_write(tmp_path, """
        [time]
        duration_s = 1200.0
        integrator = "sl_si"

        [semi_implicit]
        enabled = true
        weight = 1.0
    """, drop=("[time]", "[semi_implicit]")))
    assert cfg.dt_s == DEFAULT_SEMILAG_STEP_S == 300.0


def test_explicit_dt_is_kept_verbatim(tmp_path):
    cfg = load_config(_write(tmp_path, """
        [time]
        dt_s = 120.0
        duration_s = 600.0
        integrator = "imex_ssp3"
    """))
    assert cfg.dt_s == 120.0


def test_refusal_names_the_step_the_flow_admits():
    # a T255 flow reading 0.9 at 120 s: rate 0.0075 per second of step
    text = spectral_cfl_refusal(0.9, 120.0, 0.75, 255, EARTH_RADIUS_M)
    assert "exceeds 0.750" in text
    assert "dt_s <= 100.0 s at the gate" in text
    assert "<= 70.0 s at the shipped 0.7-of-gate rule" in text
    wind = 0.0075 * EARTH_RADIUS_M / math.sqrt(255 * 256)
    assert f"implied maximum wind {wind:.1f} m/s" in text


def test_receipt_block_carries_headroom_and_the_shipped_step():
    cfg = SimpleNamespace(dt_s=120.0, maximum_cfl=0.75, truncation=255, integrator="imex_ssp3")
    block = cfl_headroom_receipt(cfg, {"maximum_spectral_cfl": 0.644})
    assert block["role"].startswith("refusal")
    assert block["fraction_of_gate"] == pytest.approx(0.644 / 0.75)
    assert block["headroom_fraction_of_gate"] == pytest.approx(1.0 - 0.644 / 0.75)
    assert block["largest_step_at_gate_s"] == pytest.approx(0.75 / (0.644 / 120.0))
    assert block["largest_step_at_shipped_rule_s"] == pytest.approx(0.7 * 0.75 / (0.644 / 120.0))
    assert block["shipped_rule"]["shipped_eulerian_step_s"] == default_eulerian_step_s(255, 0.75)
    assert block["shipped_rule"]["this_run_took_the_shipped_step"] is (default_eulerian_step_s(255, 0.75) == 120.0)
    assert "percent" in block["sentence"]


def test_receipt_block_on_the_semi_lagrangian_path_is_a_measurement():
    cfg = SimpleNamespace(dt_s=300.0, maximum_cfl=0.75, truncation=255, integrator="sl_si")
    block = cfl_headroom_receipt(cfg, {"maximum_spectral_cfl": 1.554})
    assert block["role"].startswith("measurement")
    assert block["shipped_rule"]["this_run_took_the_shipped_step"] is False
    assert "Lipschitz" in block["sentence"]
    assert block["largest_step_at_gate_s"] == pytest.approx(0.75 / (1.554 / 300.0))


def test_receipt_block_without_a_reading_says_so():
    cfg = SimpleNamespace(dt_s=60.0, maximum_cfl=0.75, truncation=63, integrator="imex_ssp3")
    block = cfl_headroom_receipt(cfg, {})
    assert block["sentence"] == "no CFL reading"
    assert block["largest_step_at_gate_s"] is None
