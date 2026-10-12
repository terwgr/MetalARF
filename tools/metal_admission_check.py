#!/usr/bin/env python3
"""CPU-only inspection of the current Metal combo matrix admission."""
from __future__ import annotations

from collections import Counter
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.metal_combo_smoke import GF_PROFILE_ROW, RUC_PROFILE_ROW, configuration


def matrix_rows():
    axes = ((0, 1, 8, 10, 18, 28, 50), (0, 1, 5), (0, 2, 4),
            (0, 1, 16), (0, 1, 4, 11))
    boundaries = ("periodic", "open", "specified")
    rows = [GF_PROFILE_ROW]
    rows.extend(row for row in itertools.product(*axes, boundaries)
                if row not in (GF_PROFILE_ROW, RUC_PROFILE_ROW))
    rows.append(RUC_PROFILE_ROW)
    rows.extend((0, pbl, 0, 0, 0, "periodic")
                for pbl in (0, 11, 9) for _sfclay in (1, 91))
    rows.append((8, 5, 4, 16, 1, "periodic"))
    rows.append((50, 5, 4, 16, 11, "periodic"))
    return rows


def dycore_config_refusals(cfg):
    """Evaluate the dycore's configuration-only Metal gates without a device."""
    from woof.core.physics_inventory import physics_enabled

    refusals = []
    if cfg.open_x and cfg.open_y and cfg.specified:
        refusals.append("combined radiative-open and specified boundaries")
    if (cfg.open_x or cfg.open_y) and cfg.h_sca_adv_order != 2:
        refusals.append("radiative-open boundaries require h_sca_adv_order=2")
    if (cfg.km_opt, cfg.diff_6th_opt) not in {
            (km, diff6) for km in (1, 2, 3, 4) for diff6 in (0, 1, 2)}:
        refusals.append(
            f"Smagorinsky/sixth-order diffusion pair "
            f"(km_opt={cfg.km_opt}, diff_6th_opt={cfg.diff_6th_opt}) "
            "has no Metal kernel pair")
    if int(getattr(cfg, "tke_budget", 0)) != 0:
        refusals.append("tke_budget=1 has no Metal implementation")
    if cfg.zadvect_implicit != 0:
        refusals.append("IEVA vertical advection")
    if cfg.damp_opt not in (0, 3):
        refusals.append(f"damp_opt={cfg.damp_opt} has no Metal path")
    if (cfg.nwp_diagnostics != 0
            and not (cfg.nwp_diagnostics == 1
                     and (cfg.specified or cfg.nested
                          or (cfg.open_x and cfg.open_y)))):
        refusals.append(
            "NWP diagnostics require specified, nested, or two-sided "
            "radiative-open boundaries")
    if physics_enabled(cfg) and (cfg.open_x or cfg.open_y) and cfg.specified:
        refusals.append("physics with combined open and specified boundaries")
    return refusals


def allocation_plan(cfg):
    from woof.core.physics import (
        _composed_optional_tendency_components,
        _cumulus_optional_tendency_components,
        _pbl_optional_tendency_components,
        microphysics_scratch_slots,
    )

    return {
        "pending_rainbl": bool(cfg.sf_surface_physics or cfg.cu_physics),
        "pbl_tendencies": _pbl_optional_tendency_components(cfg),
        "cumulus_tendencies": _cumulus_optional_tendency_components(cfg),
        "composed_tendencies": _composed_optional_tendency_components(cfg),
        "microphysics_scratch": tuple(sorted(
            microphysics_scratch_slots(cfg.mp_physics))),
    }


def main():
    from woof.core.physics import _metal_admission_refusals

    capability_counts = Counter()
    config_counts = Counter()
    dycore_counts = Counter()
    plans = Counter()
    total = 0
    for row in matrix_rows():
        total += 1
        try:
            cfg = configuration(row)
        except (NotImplementedError, ValueError) as exc:
            config_counts[str(exc).splitlines()[0]] += 1
            continue
        for message in _metal_admission_refusals(
                cfg, radiation=None, cumulus=None, cam_ozone=None):
            if message.startswith("radiation adapter") or message.startswith("cumulus adapter"):
                continue
            capability_counts[message] += 1
        for message in dycore_config_refusals(cfg):
            dycore_counts[message] += 1
        plans[repr(allocation_plan(cfg))] += 1

    print(f"rows: {total}")
    for title, counter in (("configuration refusals", config_counts),
                           ("capability refusals", capability_counts),
                           ("dycore admission refusals", dycore_counts),
                           ("consumer allocation plans", plans)):
        print(f"\n{title}:")
        if not counter:
            print("  (none)")
        else:
            for message, count in counter.most_common():
                print(f"  {count:4d}  {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
