"""Analysis charts of the DA door's control twin: the control's rmse against
the truth before and after every analysis for one or more twin reports
laid beside each other (the control path against the ensemble-mean
transfer, or several truncations), with the ensemble spread beside it, and
a markdown table of the same numbers.

Analysis charts are matplotlib's (the render law reserves the Rust
renderer for weather fields; a recovery curve is not one).

    python tools/arwen_global_control_twin_chart.py REPORT.json [REPORT.json ...] \
        --out chart.png [--table table.md] [--title "..."]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = (("temperature_k", "temperature rmse (K)"), ("u", "zonal wind rmse (m/s)"),
          ("surface_pressure_pa", "surface pressure rmse (Pa)"))


def _curve(report: dict, field: str):
    xs = [0.0]
    ys = [report["initial_score"][field]]
    labels = ["start"]
    for c in report["cycles"]:
        k = c["cycle"] + 1
        xs.extend([k - 0.35, k])
        ys.extend([c["before"][field], c["after"][field]])
        labels.extend([f"b{k}", f"a{k}"])
    return xs, ys, labels


def _label(report: dict) -> str:
    setup = report["setup"]
    return (f"{report['family']} / {setup['increment_source']} "
            f"(T{report['control']['truncation']} over T{report['ensemble']['truncation']}, "
            f"{report['ensemble']['members']} members)")


def chart(reports: list[dict], out: Path, *, title: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(FIELDS), figsize=(5.2 * len(FIELDS), 4.2))
    for ax, (field, ylabel) in zip(axes, FIELDS):
        for report in reports:
            xs, ys, _labels = _curve(report, field)
            ax.plot(xs, ys, marker="o", markersize=3, label=_label(report))
            if field == "temperature_k":
                spread = [c["spread"]["after"]["temperature_k"] for c in report["cycles"]]
                ax.plot([c["cycle"] + 1 for c in report["cycles"]], spread, linestyle=":", marker="x",
                        markersize=3, label=f"{_label(report)} ensemble spread")
        ax.set_xlabel("analysis (b = before, a = after)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle(title or "DA door control twin: the control's rmse against the nature run")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def table(reports: list[dict]) -> str:
    lines = ["| arm | cycle | T before (K) | T after (K) | u before (m/s) | u after (m/s) | ps before (Pa) | ps after (Pa) | spread T after (K) | status |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for report in reports:
        label = _label(report)
        for c in report["cycles"]:
            lines.append(
                f"| {label} | {c['cycle']} | {c['before']['temperature_k']:.4f} | {c['after']['temperature_k']:.4f} | "
                f"{c['before']['u']:.4f} | {c['after']['u']:.4f} | {c['before']['surface_pressure_pa']:.2f} | "
                f"{c['after']['surface_pressure_pa']:.2f} | {c['spread']['after']['temperature_k']:.4f} | {c['status']} |"
            )
        verdict = report["verdict"]
        lines.append(f"| {label} | verdict | | | | | | | | {'passed' if verdict['passed'] else 'FAILED'}: {verdict['bar']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--table", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    reports = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.reports]
    chart(reports, Path(args.out), title=args.title)
    if args.table:
        Path(args.table).write_text(table(reports), encoding="utf-8")
    print(f"wrote {args.out}" + (f" and {args.table}" if args.table else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
