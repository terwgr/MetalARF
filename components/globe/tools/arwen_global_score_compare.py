"""Lay one scored WOOF global arm beside the control, number by number.

``python tools/arwen_global_score_compare.py CONTROL=summary-dt50.json ARM=summary-spd24.json
[--rule 0.03] [--md out.md]``

Both files are the ``summary-<arm>.json`` the scoring chain of record
writes (schema arwen-global-arm-score-summary/v1).  The table prints
every headline reading of the control beside the arm's,
the difference, and the lane rule's verdict on the readings the rule
names: a temperature (K), a pressure (hPa) or a ratio may not worsen by
more than ``--rule`` (default 0.03).  "Worsen" is toward a larger error:
a larger rmse, a larger absolute bias, a ratio further from the
control's value in either direction (the rule names ratios without a
sign, so a ratio is held to the band).  Readings in other units (hours,
m/s, mm, mm/day) are printed with their difference and no verdict.  A
reading missing from either file prints as missing, never as passed.

The water-budget residual is printed with the fixer sum beside it: the
instrument's calibrated identity is residual = minus the fixer sum, and
the floor is stated as that identity's own slack.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RULE_ROWS = [
    # (key, unit, kind)  kind: bias (abs), rmse (up), ratio (band), info
    *[(f"bias.{r}.t2_bias_k", "K", "bias") for r in ("conus_land", "nh_midlat_land", "global_land", "nh_extratropics_all", "global_all")],
    *[(f"bias.{r}.t2_rmse_k", "K", "rmse") for r in ("conus_land", "nh_midlat_land", "global_land", "nh_extratropics_all", "global_all")],
    *[(f"bias.{r}.mslp_bias_hpa", "hPa", "bias") for r in ("conus_land", "nh_midlat_land", "global_land", "nh_extratropics_all", "global_all")],
    *[(f"bias.{r}.mslp_rmse_hpa", "hPa", "rmse") for r in ("conus_land", "nh_midlat_land", "global_land", "nh_extratropics_all", "global_all")],
    *[(f"bias.{r}.wspd10_bias_m_s", "m/s", "info") for r in ("conus_land", "nh_midlat_land", "global_all")],
    *[(f"bias.{r}.wspd10_rmse_m_s", "m/s", "info") for r in ("conus_land", "nh_midlat_land", "global_all")],
    *[(f"spectrum.{h}.all.f250", "ratio", "ratio") for h in ("northern", "global", "southern")],
    *[(f"spectrum.{h}.all.ratio_at_truncation", "ratio", "ratio") for h in ("northern", "global", "southern")],
    *[(f"spectrum.{h}.250.f250", "ratio", "ratio") for h in ("northern", "global", "southern")],
    *[(f"spectrum.{h}.250.ratio_at_truncation", "ratio", "ratio") for h in ("northern", "global", "southern")],
    *[(f"spectrum.{h}.250.slope", "", "info") for h in ("northern", "global", "southern")],
    ("spectrum.southern.all.abs_resolved_km", "km", "info"),
    ("spectrum.southern.all.abs_resolved_n", "n", "info"),
    *[(f"spectrum.{h}.250.div_share_201_400", "ratio", "ratio") for h in ("global", "northern", "southern")],
    *[(f"spectrum.{h}.250.div_share_121_200", "ratio", "ratio") for h in ("global", "northern", "southern")],
    ("diurnal.conus_land.total.composite_phase_lst", "h LST", "info"),
    ("diurnal.conus_land.total.composite_amp_mm_h", "mm/h", "info"),
    ("diurnal.conus_land.convective.composite_phase_lst", "h LST", "info"),
    ("diurnal_obs.conus_land.total.composite_delta_h", "h", "info"),
    ("diurnal_obs.conus_land.total.n_paired_cells", "n", "info"),
    ("diurnal.global_land.total.composite_phase_lst", "h LST", "info"),
    ("diurnal.global_land.convective.composite_phase_lst", "h LST", "info"),
    ("budget.global.E_books_mm_day", "mm/day", "info"),
    ("budget.global.E_flux_mm_day", "mm/day", "info"),
    ("budget.global.P_conv_mm_day", "mm/day", "info"),
    ("budget.global.P_grid_mm_day", "mm/day", "info"),
    ("budget.global.dW_vapor_mm_day", "mm/day", "info"),
    ("budget.global.dW_condensate_mm_day", "mm/day", "info"),
    ("budget.global.residual_mm_day", "mm/day", "info"),
    ("budget.global.residual_fraction", "fraction", "info"),
    ("budget.global.fixer_kg_m2", "kg/m2", "info"),
    ("budget.land.P_grid_mm_day", "mm/day", "info"),
    ("budget.ocean.P_grid_mm_day", "mm/day", "info"),
    ("precip.RAINNC.area_mean_mm", "mm", "info"),
    ("precip.RAINNC.max_mm", "mm", "info"),
    ("precip.RAINNC.cells_above_200_mm", "cells", "info"),
    ("precip.RAINC.area_mean_mm", "mm", "info"),
    ("precip.RAINC.max_mm", "mm", "info"),
    ("precip.TOTAL.area_mean_mm", "mm", "info"),
    ("precip.TOTAL.max_mm", "mm", "info"),
    ("receipt.wall_seconds", "s", "info"),
    ("receipt.allocator_peak_gib", "GiB", "info"),
    ("receipt.status", "", "info"),
    ("tracker.maximum_spectral_cfl", "", "info"),
    ("tracker.maximum_positivity_fixer_relative", "", "info"),
    ("tracker.maximum_global_water_fixer_kg_m2", "kg/m2", "info"),
    ("tracker.maximum_physics_water_repair_kg_m2", "kg/m2", "info"),
    ("tracker.maximum_repaired_negative_number_per_kg", "", "info"),
    ("drift.mass_pa.relative_per_hour", "1/h", "info"),
    ("drift.total_dry_energy_j_m2.relative_per_hour", "1/h", "info"),
    ("drift.moist_total_energy_j_m2.relative_per_hour", "1/h", "info"),
    ("final.max_wind_m_s", "m/s", "info"),
]


def load(spec: str) -> tuple[str, dict]:
    label, _, path = spec.partition("=")
    if not path:
        raise SystemExit(f"{spec!r}: expected LABEL=path")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != "arwen-global-arm-score-summary/v1":
        raise SystemExit(f"{path}: not an arm score summary (schema {data.get('schema')!r})")
    return label, data["numbers"]


def fmt(value) -> str:
    if value is None:
        return "missing"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 1.0e-3 or abs(value) >= 1.0e5:
            return f"{value:.3e}"
        return f"{value:.3f}"
    return str(value)


def verdict(kind: str, control, arm, rule: float) -> str:
    if kind == "info":
        return ""
    if control is None or arm is None:
        return "INCOMPLETE"
    try:
        c = float(control)
        a = float(arm)
    except (TypeError, ValueError):
        return "INCOMPLETE"
    if kind == "bias":
        worse = abs(a) - abs(c)
    elif kind == "rmse":
        worse = a - c
    else:  # ratio: held to the band either way
        worse = abs(a - c)
    return "within rule" if worse <= rule else f"WORSE by {worse:.3f}"


def chart(control: dict, arm: dict, labels: tuple[str, str], rule: float, png: str) -> None:
    """The rule readings' differences (arm minus control) as horizontal
    bars inside the rule band: an analysis chart, not a weather field."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, values, colors = [], [], []
    for key, unit, kind in RULE_ROWS:
        if kind == "info":
            continue
        c = control.get(key)
        a = arm.get(key)
        if not isinstance(c, (int, float)) or not isinstance(a, (int, float)):
            continue
        names.append(f"{key} [{unit}]")
        values.append(float(a) - float(c))
        colors.append("#c0392b" if verdict(kind, c, a, rule).startswith("WORSE") else "#2c7fb8")
    fig, ax = plt.subplots(figsize=(11, 0.28 * len(names) + 2.0))
    y = range(len(names))
    ax.barh(list(y), values, color=colors)
    ax.axvline(-rule, color="gray", linestyle="--", linewidth=1)
    ax.axvline(rule, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"{labels[1]} minus {labels[0]} (dashed: the rule band, {rule:g})")
    ax.set_title("WOOF global 24 h grade: every rule reading beside the control")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=130)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("control")
    parser.add_argument("arm")
    parser.add_argument("--rule", type=float, default=0.03)
    parser.add_argument("--md", default=None)
    parser.add_argument("--png", default=None)
    args = parser.parse_args(argv)
    control_label, control = load(args.control)
    arm_label, arm = load(args.arm)
    lines = [
        f"| reading | unit | {control_label} | {arm_label} | arm minus control | rule ({args.rule:g}) |",
        "|---|---|---|---|---|---|",
    ]
    worse = []
    incomplete = []
    for key, unit, kind in RULE_ROWS:
        c = control.get(key)
        a = arm.get(key)
        delta = ""
        if isinstance(c, (int, float)) and isinstance(a, (int, float)) and not isinstance(c, bool):
            delta = fmt(float(a) - float(c))
        v = verdict(kind, c, a, args.rule)
        if v.startswith("WORSE"):
            worse.append((key, v))
        if v == "INCOMPLETE":
            incomplete.append(key)
        lines.append(f"| {key} | {unit} | {fmt(c)} | {fmt(a)} | {delta} | {v} |")
    lines.append("")
    if incomplete:
        lines.append(f"INCOMPLETE: {len(incomplete)} rule readings missing on one side: {', '.join(incomplete)}")
    if worse:
        lines.append(f"RULE: {len(worse)} readings worsen beyond {args.rule:g}: " + "; ".join(f"{k} ({v})" for k, v in worse))
    elif not incomplete:
        lines.append(f"RULE: no temperature, pressure or ratio reading worsens by more than {args.rule:g}.")
    text = "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n", encoding="utf-8")
    if args.png:
        chart(control, arm, (control_label, arm_label), args.rule, args.png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
