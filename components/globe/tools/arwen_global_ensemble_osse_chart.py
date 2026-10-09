"""Chart the global ensemble OSSE's recovery curves (an analysis chart,
matplotlib; no weather field is drawn here).

    python tools/arwen_global_ensemble_osse_chart.py osse-recovery.json --png out.png [--table out.md]
    python tools/arwen_global_ensemble_osse_chart.py osse-sweep-<option>.json --png out.png [--table out.md]

For one twin report: the grid rmse of the CONTROL against the nature run
on the control grid and of the ENSEMBLE MEAN against the nature run on
the ensemble grid, before and after every analysis, with the spread beside
the mean, for the temperature, the wind, the surface pressure and the
vapor; the mean-increment arm when the report carries one (the transfer
family); O-B and O-A rms per stream and variable per cycle for the
control; the setup and the verdict in a panel.  For a sweep report: the
control temperature and wind rmse by cycle, one line per arm.  The table
lists the same numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = (
    ("temperature_k", "temperature (K)", 1.0),
    ("u", "u wind (m/s)", 1.0),
    ("surface_pressure_pa", "surface pressure (hPa)", 0.01),
    ("qv", "vapor (g/kg)", 1000.0),
)


def _series(report):
    cycles = report["cycles"]
    out = {}
    for name, label, scale in FIELDS:
        out[name] = {
            "label": label,
            "control_before": [c["before"]["control"][name]["rmse"] * scale for c in cycles],
            "control_after": [c["after"]["control"][name]["rmse"] * scale for c in cycles],
            "mean_before": [c["before"]["ensemble"][name]["rmse"] * scale for c in cycles],
            "mean_after": [c["after"]["ensemble"][name]["rmse"] * scale for c in cycles],
            "spread": [c["after"]["ensemble"][name]["spread"] * scale for c in cycles],
            "arm_after": [c["mean_increment_arm"]["after"][name]["rmse"] * scale for c in cycles]
            if all("mean_increment_arm" in c for c in cycles) else None,
        }
    return out


def _sawtooth(cycles, before, after):
    x, y = [], []
    for c, b, a in zip(cycles, before, after):
        x += [c - 0.4, c]
        y += [b, a]
    return x, y


def render(report: dict, png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "sweep" in report:
        return _render_sweep(report, png, plt)
    series = _series(report)
    cycles = [c["cycle"] for c in report["cycles"]]
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    axes = axes.ravel()
    for ax, (name, s) in zip(axes[:4], series.items()):
        x, y = _sawtooth(cycles, s["control_before"], s["control_after"])
        ax.plot(x, y, "-", color="#1f77b4", lw=1.8, label=f"control T{report['control_truncation']} rmse vs nature")
        ax.plot(cycles, s["control_after"], "o", color="#1f77b4")
        x, y = _sawtooth(cycles, s["mean_before"], s["mean_after"])
        ax.plot(x, y, "-", color="#2ca02c", lw=1.2, label=f"ensemble mean T{report['ensemble_truncation']} rmse vs nature")
        ax.plot(cycles, s["spread"], "s--", color="#ff7f0e", label="ensemble spread after analysis")
        if s["arm_after"] is not None:
            ax.plot(cycles, s["arm_after"], "^:", color="#d62728", label="mean-increment arm (comparison)")
        ax.set_title(s["label"])
        ax.set_xlabel("analysis cycle")
        ax.grid(alpha=0.3)
        if name == "temperature_k":
            ax.legend(fontsize=7, loc="upper right")
    ax = axes[4]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    k = 0
    for stream in sorted({s for c in report["cycles"] for s in c["streams"]}):
        for variable in sorted({v for c in report["cycles"] for v in c["streams"].get(stream, {})}):
            ob = [c["streams"].get(stream, {}).get(variable, {}).get("control_o_minus_b_rms") for c in report["cycles"]]
            oa = [c["streams"].get(stream, {}).get(variable, {}).get("control_o_minus_a_rms") for c in report["cycles"]]
            if all(v is None for v in ob):
                continue
            color = colors[k % len(colors)]
            k += 1
            ax.plot(cycles, ob, "-", color=color, lw=1, label=f"{stream}/{variable} O-B")
            ax.plot(cycles, oa, "--", color=color, lw=1, label=f"{stream}/{variable} O-A")
    ax.set_yscale("log")
    ax.set_title("control O-B (solid) and O-A (dashed) rms, assimilated rows, own units")
    ax.set_xlabel("analysis cycle")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2)
    ax = axes[5]
    ax.axis("off")
    setup = report["setup"]
    verdict = report.get("verdict", {})
    ratios = verdict.get("spread_over_rmse_last_cycle", {})
    text = [
        f"family {report['family']}: nature and control T{report['control_truncation']}, ensemble T{report['ensemble_truncation']}, L{report['nlev']}",
        f"dt {report['control_dt_s']:g} / {report['ensemble_dt_s']:g} s, members {setup['members']}, cycles {setup['cycles']},"
        f" interval {setup['interval_s']:g} s, bins {report['observation_time_bin_s']:g} s",
        f"network: {report['network']['stations']} stations, {report['network']['soundings']} soundings,"
        f" {report['network']['amv_points_per_cycle']} AMVs per cycle",
        f"localisation {setup['horizontal_cutoff_km']:g} km, ln p {setup['vertical_cutoff_lnp']:g}"
        f" (surface {setup['surface_vertical_cutoff_lnp']:g}), RTPS {setup['rtps_alpha']:g},"
        f" additive {setup['additive_inflation_fraction']:g}",
        f"wind {setup['wind_balance']}, application {setup['increment_application']}, recentring {setup['recentering_fraction']:g}",
        f"start displacement {setup['start_perturbation_scale']:g} x the initial perturbation",
        "control T rmse by cycle: " + ", ".join(f"{v:.3f}" for v in verdict.get("control_temperature_rmse_by_cycle", [])) + " K",
        "control u rmse by cycle: " + ", ".join(f"{v:.3f}" for v in verdict.get("control_wind_rmse_by_cycle", [])) + " m/s",
        "spread / rmse at the last cycle: " + ", ".join(f"{k} {v:.2f}" for k, v in ratios.items() if v is not None),
        f"engineering every cycle {verdict.get('engineering_validity_every_cycle')}, passed {verdict.get('passed')}",
    ]
    if "control_analysis_beats_mean_increment_arm_at_last_cycle" in verdict:
        text.append("control analysis beats the mean-increment arm at the last cycle: "
                    + json.dumps(verdict["control_analysis_beats_mean_increment_arm_at_last_cycle"]))
    ax.text(0.0, 1.0, "\n".join(text), va="top", family="monospace", fontsize=8)
    fig.suptitle("WOOF global ensemble filter: dual-resolution twin against the nature run")
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=130)


def _render_sweep(report: dict, png: Path, plt) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, label, scale) in zip(axes, FIELDS[:3]):
        for value, arm in report["arms"].items():
            cycles = [c["cycle"] for c in arm["cycles"]]
            after = [c["after"]["control"][name]["rmse"] * scale for c in arm["cycles"]]
            ax.plot(cycles, after, "o-", label=f"{report['sweep']} = {value}")
        ax.set_title(f"control rmse vs nature after analysis, {label}")
        ax.set_xlabel("analysis cycle")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"WOOF global ensemble filter twin: sweep of {report['sweep']} ({report['family']} family)")
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=130)


def table(report: dict) -> str:
    if "sweep" in report:
        lines = [f"| {report['sweep']} | cycle | control T rmse after (K) | control u rmse after (m/s) | control ps rmse after (hPa) | mean T rmse after (K) | spread T (K) |",
                 "|---|---|---|---|---|---|---|"]
        for value, arm in report["arms"].items():
            for c in arm["cycles"]:
                ctl, ens = c["after"]["control"], c["after"]["ensemble"]
                lines.append(f"| {value} | {c['cycle']} | {ctl['temperature_k']['rmse']:.3f} | {ctl['u']['rmse']:.3f} "
                             f"| {ctl['surface_pressure_pa']['rmse'] / 100:.3f} | {ens['temperature_k']['rmse']:.3f} "
                             f"| {ens['temperature_k']['spread']:.3f} |")
        return "\n".join(lines) + "\n"
    lines = ["| cycle | control T rmse before / after (K) | control u before / after (m/s) | control ps before / after (hPa) "
             "| mean T before / after / spread (K) | mean u after / spread (m/s) | status | analysis wall (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for c in report["cycles"]:
        cb, ca = c["before"]["control"], c["after"]["control"]
        eb, ea = c["before"]["ensemble"], c["after"]["ensemble"]
        lines.append(
            f"| {c['cycle']} | {cb['temperature_k']['rmse']:.3f} / {ca['temperature_k']['rmse']:.3f} "
            f"| {cb['u']['rmse']:.3f} / {ca['u']['rmse']:.3f} "
            f"| {cb['surface_pressure_pa']['rmse'] / 100:.3f} / {ca['surface_pressure_pa']['rmse'] / 100:.3f} "
            f"| {eb['temperature_k']['rmse']:.3f} / {ea['temperature_k']['rmse']:.3f} / {ea['temperature_k']['spread']:.3f} "
            f"| {ea['u']['rmse']:.3f} / {ea['u']['spread']:.3f} | {c['status']} | {c['analysis_wall_s']:.1f} |")
    if all("mean_increment_arm" in c for c in report["cycles"]):
        lines.append("")
        lines.append("| cycle | control analysis T rmse (K) | mean-increment arm T rmse (K) | control u (m/s) | arm u (m/s) | control ps (hPa) | arm ps (hPa) |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in report["cycles"]:
            ca, arm = c["after"]["control"], c["mean_increment_arm"]["after"]
            lines.append(f"| {c['cycle']} | {ca['temperature_k']['rmse']:.3f} | {arm['temperature_k']['rmse']:.3f} "
                         f"| {ca['u']['rmse']:.3f} | {arm['u']['rmse']:.3f} | {ca['surface_pressure_pa']['rmse'] / 100:.3f} "
                         f"| {arm['surface_pressure_pa']['rmse'] / 100:.3f} |")
    lines.append("")
    lines.append("| cycle | stream / variable | count | ensemble O-B rms | ensemble O-A rms | control O-B rms | control O-A rms "
                 "| Desroziers error ratio | innovation ratio |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in report["cycles"]:
        for stream, variables in c["streams"].items():
            for variable, v in variables.items():
                def f(x):
                    return "" if x is None else f"{x:.4g}"
                d = v.get("desroziers", {})
                lines.append(
                    f"| {c['cycle']} | {stream} / {variable} | {v['count']} | {f(v['o_minus_b_rms'])} | {f(v['o_minus_a_rms'])} "
                    f"| {f(v.get('control_o_minus_b_rms'))} | {f(v.get('control_o_minus_a_rms'))} "
                    f"| {f(d.get('error_variance_ratio'))} | {f(d.get('innovation_ratio'))} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("report", type=Path)
    p.add_argument("--png", type=Path, required=True)
    p.add_argument("--table", type=Path, default=None)
    a = p.parse_args(argv)
    report = json.loads(a.report.read_text(encoding="utf-8"))
    render(report, a.png)
    if a.table is not None:
        a.table.write_text(table(report), encoding="utf-8")
    print(f"wrote {a.png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
