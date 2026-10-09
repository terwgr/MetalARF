"""The shipped default core of WOOF global is the semi-Lagrangian one.

The ruling of 2026-09-06: the semi-Lagrangian core is the core every lane
builds on, runs, tests and grades with, and the default of ``woof global
run`` and ``woof global da`` at every truncation, with the Eulerian pair
selectable by name.  These tests pin the door: what a config that says
nothing about the core gets, that the Eulerian config of record's identity
did not move when the default did (its ten-step T255 gate reads the same
config hash), that the bare door config in the tree IS the arm of record
with every key left unsaid, and that every shipped semi-Lagrangian config
carries the core's own drain written out.
"""
from __future__ import annotations

from woof.globe.configs_dir import config_root as _shipped_configs
import dataclasses
from pathlib import Path

import pytest

# CPU only: the device switch is set for each test in this module by
# `conftest._cpu_only_marked_tests` and put back afterwards, because a
# module that set it at import time decided it for the whole session.
pytestmark = pytest.mark.cpu_only

from woof.globe import pins  # noqa: E402
from woof.globe.config import (  # noqa: E402
    DEFAULT_DIFFUSION_EFOLD_S,
    DEFAULT_DIFFUSION_ORDER,
    DEFAULT_SEMILAG_STEP_S,
    DEFAULT_TIME_INTEGRATOR,
    SEMILAG_DIFFUSION_EFOLD_S,
    SEMILAG_DIFFUSION_ORDER,
    SEMILAG_OFF_CENTRING_WEIGHT,
    SHIPPED_INTEGRATOR,
    default_eulerian_step_s,
    load_config,
)
from woof.globe.semilag.options import SemiLagrangianOptions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VERIFY = _shipped_configs()
BARE = VERIFY / "arwen_global_gdas_t255_native_24h_bare.toml"
SL_RECORD = VERIFY / "arwen_global_gdas_t255_native_sl_si_24h.toml"
IMEX_RECORD = VERIFY / "arwen_global_gdas_t255_native_imex_24h.toml"
QUICKSTART = _shipped_configs() / "arwen_global_t255_quickstart.toml"

#: The config hash of the Eulerian config of record, the one the ten-step
#: T255 BIT-IMEX gate runs from here on: the GDAS 2026-09-01 00Z native
#: suite on imex_ssp3 at its rule step of 90 s (MEASURED 2026-09-07: two
#: trees compared on the RTX 5090 and two more on the RTX 5070 Ti,
#: 42 arrays identical at
#: step 0 and 125 at step 10 under this hash; the merge's own gate on the
#: merged tree is in the changelog).  The 60 s content the gate ran on
#: until 2026-09-06 hashed cf964993... and its identity is proven too; the
#: hash moved because the step is part of the config identity, not because
#: the Eulerian arithmetic did.  The door's default moved to the
#: semi-Lagrangian core without moving either, because a config that names
#: its integrator reads exactly what it always read.
IMEX_RECORD_CONFIG_HASH = "ae84e41ebfc767a2aa0afab4368595c2d26a1f4eedb2e88b28698078bd4cb4e1"

MINIMAL = """
[arwen_global]
schema = "gpuwm.arwen-global-run/v1"
name = "default-core-door"
acknowledgement = "research-only-arwen-global-v1"
backend = "numpy"
precision = "float64"

[grid]
truncation = 21

[vertical]
coordinate = "surface_stretched"
nlev = 40
p_top_pa = 100.0
"""

SHIPPED_SEMILAG = sorted(
    path for path in VERIFY.glob("*.toml")
    if 'integrator = "sl_si"' in path.read_text(encoding="utf-8")
)


def _load(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return load_config(path)


def test_the_door_default_is_the_semi_lagrangian_core_at_300_s(tmp_path):
    assert DEFAULT_TIME_INTEGRATOR == SHIPPED_INTEGRATOR == "sl_si"
    cfg = _load(tmp_path, MINIMAL)
    assert cfg.integrator == "sl_si"
    assert cfg.dt_s == DEFAULT_SEMILAG_STEP_S == 300.0
    assert cfg.diffusion_order == SEMILAG_DIFFUSION_ORDER == 16
    assert cfg.diffusion_efold_s == SEMILAG_DIFFUSION_EFOLD_S == 720.0
    assert cfg.semilag.horizontal_interpolation == "quintic_lagrange"
    assert SemiLagrangianOptions().horizontal_interpolation == "quintic_lagrange"
    assert cfg.semi_implicit_off_centring_weight == SEMILAG_OFF_CENTRING_WEIGHT == 0.55
    assert cfg.semi_implicit_scheme == "vertical_modes" and cfg.semi_implicit_weight == 1.0


def test_the_eulerian_core_by_name_reads_what_it_always_read(tmp_path):
    cfg = _load(tmp_path, '[time]\nintegrator = "imex_ssp3"\n' + MINIMAL)
    assert cfg.integrator == "imex_ssp3"
    # an omitted Eulerian step is the CFL rule's for the truncation (T21
    # here; 90 s at T255), not a fixed number: test_arwen_global_eulerian_step_rule
    assert cfg.dt_s == default_eulerian_step_s(21, cfg.maximum_cfl)
    assert cfg.diffusion_order == DEFAULT_DIFFUSION_ORDER == 8
    assert cfg.diffusion_efold_s == DEFAULT_DIFFUSION_EFOLD_S == 2160.0
    assert cfg.semi_implicit_off_centring_weight == 0.5
    # the [semilag] options are not part of an Eulerian run's identity, so
    # the gather default cannot reach an Eulerian config hash
    quintic = _load(tmp_path, '[time]\nintegrator = "imex_ssp3"\n' + MINIMAL)
    assert quintic.config_hash == cfg.config_hash


def test_an_omitted_step_follows_the_integrator(tmp_path):
    rule = default_eulerian_step_s(21, 0.75)
    split = _load(tmp_path, '[time]\nintegrator = "ssprk3"\n' + MINIMAL)
    assert split.dt_s == rule
    assert _load(tmp_path, '[time]\nintegrator = "rk4"\n' + MINIMAL).dt_s == rule
    assert _load(tmp_path, '[time]\nintegrator = "sl_si"\n' + MINIMAL).dt_s == 300.0
    # at the truncations that ship, the Eulerian rule reads 90, 60 and 40 s
    assert [default_eulerian_step_s(t, 0.75) for t in (255, 383, 533)] == [90.0, 60.0, 40.0]
    # a written step is read as written under either core
    assert _load(tmp_path, "[time]\ndt_s = 100.0\n" + MINIMAL).dt_s == 100.0
    assert _load(tmp_path, '[time]\ndt_s = 100.0\nintegrator = "imex_ssp3"\n' + MINIMAL).dt_s == 100.0


def test_the_config_of_record_s_identity_did_not_move_with_the_default():
    cfg = load_config(IMEX_RECORD)
    assert cfg.integrator == "imex_ssp3"
    assert cfg.dt_s == 90.0 == default_eulerian_step_s(255, cfg.maximum_cfl)
    assert cfg.diffusion_order == 8 and cfg.diffusion_efold_s == 2160.0
    assert cfg.semi_implicit_off_centring_weight == 0.5
    assert cfg.config_hash == IMEX_RECORD_CONFIG_HASH


def test_the_bare_door_config_is_the_arm_of_record_with_nothing_said():
    bare = load_config(BARE)
    record = load_config(SL_RECORD)
    text = BARE.read_text(encoding="utf-8")
    for key in ("integrator", "dt_s", "[semilag]", "order =", "e_folding", "off_centring"):
        assert key not in text.replace("# ", "").split("[arwen_global]")[1], key
    differing = [
        field.name for field in dataclasses.fields(bare)
        if getattr(bare, field.name) != getattr(record, field.name)
    ]
    assert differing == ["name"]
    assert bare.integrator == "sl_si" and bare.dt_s == 300.0 and bare.truncation == 255
    assert bare.physics_mode == "arwen-native"


@pytest.mark.parametrize("path", SHIPPED_SEMILAG, ids=lambda p: p.name)
def test_every_shipped_semi_lagrangian_config_carries_the_core_s_drain_written_out(path):
    text = path.read_text(encoding="utf-8")
    assert "\norder = 16\n" in text, path.name
    assert "\ne_folding_time_s_at_truncation = 720.0\n" in text, path.name
    cfg = load_config(path)
    assert cfg.integrator == "sl_si"
    assert cfg.diffusion_order == SEMILAG_DIFFUSION_ORDER
    assert cfg.diffusion_efold_s == SEMILAG_DIFFUSION_EFOLD_S
    assert cfg.semilag.horizontal_interpolation == "quintic_lagrange"


def test_the_quickstart_runs_the_default_core():
    cfg = load_config(QUICKSTART)
    assert cfg.integrator == "sl_si"
    assert cfg.dt_s == 300.0
    assert cfg.duration_s == 86400.0 and cfg.output_interval_s == 10800.0
    assert cfg.diffusion_order == 16 and cfg.diffusion_efold_s == 720.0
    assert cfg.semi_implicit_off_centring_weight == 0.55


def test_the_pins_default_still_names_the_eulerian_arithmetic():
    # A checkpoint or receipt written before the integrator key existed is
    # read under the IMEX pin; that reader default is not the door's
    # default and the two are allowed to differ.
    assert pins.DEFAULT_INTEGRATOR == "imex_ssp3"
    assert DEFAULT_TIME_INTEGRATOR != pins.DEFAULT_INTEGRATOR
