"""Analysis charts of a vertical level set beside the default (matplotlib;
no weather field is drawn here, those come from the Rust renderer).

    python tools/arwen_global_level_set_evidence.py spacing OUT.png
        [--coordinates surface_stretched:40 jet_refined:48]
    python tools/arwen_global_level_set_evidence.py floor-table OUT.png
        --floors A.json B.json --labels "40-level" "48-level"

``spacing`` draws each stack's layer thickness in pressure and in ln p
against the full-level pressure, with the 120 to 400 hPa jet band
shaded and the 250 hPa scorecard target marked.  ``floor-table`` puts
the representation floors of two level sets (``upper_air_scorecard
--floor`` payloads) side by side for every row, target and two regions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PS_REF = 101_325.0


def _coordinate(spec: str):
    from woof.globe.vertical import HybridCoordinate

    name, _, count = spec.partition(":")
    builder = getattr(HybridCoordinate, name)
    return name, builder(int(count)) if count else builder()


def spacing_chart(out: Path, specs: list[str]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True)
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")
    for k, spec in enumerate(specs):
        name, coordinate = _coordinate(spec)
        p_half = coordinate.a_half_pa + coordinate.b_half * PS_REF
        p_full = np.sqrt(p_half[:-1] * p_half[1:]) / 100.0
        dp = np.diff(p_half) / 100.0
        ln_ratio = np.log(p_half[1:] / p_half[:-1])
        described = coordinate.describe()
        label = (f"{name} ({coordinate.nlev} levels): thickest layer in 120-400 hPa "
                 f"{described['jet_band_thickest_layer_pa'] / 100.0:.1f} hPa, "
                 f"{described['full_levels_in_jet_band']} full levels there")
        axes[0].plot(dp, p_full, marker="o", ms=3.5, lw=1.2, color=colors[k % 4], label=label)
        axes[1].plot(ln_ratio, p_full, marker="o", ms=3.5, lw=1.2, color=colors[k % 4], label=label)
    for ax, title, xlabel in (
        (axes[0], "layer thickness in pressure", "layer thickness (hPa)"),
        (axes[1], "layer thickness in ln p", "ln(p_bottom / p_top) of the layer"),
    ):
        ax.axhspan(120.0, 400.0, color="#ffdd88", alpha=0.35, lw=0, label="jet band 120 to 400 hPa" if ax is axes[0] else None)
        ax.axhline(250.0, color="k", ls="--", lw=0.8)
        ax.text(ax.get_xlim()[1] * 0.98 if ax is axes[1] else 60.0, 250.0, " 250 hPa (W250 target)", va="bottom", ha="right", fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(1050.0, 1.0)
        ax.set_yticks([1000, 850, 700, 500, 400, 300, 250, 200, 150, 120, 100, 70, 50, 30, 20, 10, 5, 2, 1])
        ax.set_yticklabels([str(v) for v in [1000, 850, 700, 500, 400, 300, 250, 200, 150, 120, 100, 70, 50, 30, 20, 10, 5, 2, 1]], fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.grid(True, which="both", lw=0.3, alpha=0.5)
    axes[0].set_ylabel("full-level pressure (hPa) at a 1013.25 hPa surface")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", fontsize=8.5, ncol=1, frameon=False)
    fig.suptitle("WOOF global vertical level sets: the 40-level default and the 48-level jet-refined candidate", fontsize=12)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


ROWS = (("vertical", "vertical round trip"), ("spectral", "spectral truncation"), ("cold_start", "cold start read back"))
CELLS = (
    ("z500", "rmse", "Z500 rmse m"),
    ("z500", "bias", "Z500 bias m"),
    ("t850", "rmse", "T850 rmse K"),
    ("w250", "rmsve", "W250 rmsve m/s"),
    ("w250", "speed_bias", "W250 speed bias m/s"),
    ("w850", "rmsve", "W850 rmsve m/s"),
    ("rh700", "rmse", "RH700 rmse %"),
)


def floor_table(out: Path, floors: list[Path], labels: list[str], regions=("global", "nh_extratropics")) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payloads = [json.loads(Path(p).read_text(encoding="utf-8")) for p in floors]
    if len(labels) != len(payloads):
        raise SystemExit("one label per floor payload")
    header = ["row", "reading"] + [f"{label} ({p['nlev']} lev)" for label, p in zip(labels, payloads)]
    if len(payloads) == 2:
        header.append("difference")
    body = []
    for region in regions:
        body.append([f"[{region}]"] + [""] * (len(header) - 1))
        for tag, row_label in ROWS:
            for target, key, cell_label in CELLS:
                values = []
                for payload in payloads:
                    r = payload["rows"][tag][target]["regions"][region]
                    values.append(float(r[key]))
                cells = [row_label, cell_label] + [f"{v:+.3f}" if "bias" in key else f"{v:.3f}" for v in values]
                if len(values) == 2:
                    cells.append(f"{values[1] - values[0]:+.3f}")
                body.append(cells)
    fig_h = 0.28 * (len(body) + 3)
    fig, ax = plt.subplots(figsize=(11.5, fig_h))
    ax.axis("off")
    table = ax.table(cellText=body, colLabels=header, loc="center", cellLoc="left", colLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.15)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e8e8e8")
        elif body[r - 1][0].startswith("["):
            cell.set_facecolor("#f4f4f4")
            cell.set_text_props(weight="bold")
    factors = ", ".join(
        f"{label}: bracket factor at 250 hPa {p.get('kink_read_factor_250hpa', float('nan')):.4f}, "
        f"thickest jet-band layer {(p.get('jet_band_thickest_layer_pa') or 0.0) / 100.0:.1f} hPa"
        for label, p in zip(labels, payloads)
    )
    ax.set_title(
        "Representation floor of the initial analysis on each level set, before any forecast\n"
        f"{payloads[0]['reference'].get('path', '')}\n{factors}",
        fontsize=9,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("spacing")
    s.add_argument("out")
    s.add_argument("--coordinates", nargs="*", default=["surface_stretched:40", "jet_refined:48"])
    f = sub.add_parser("floor-table")
    f.add_argument("out")
    f.add_argument("--floors", nargs="+", required=True)
    f.add_argument("--labels", nargs="+", required=True)
    args = parser.parse_args(argv)
    if args.command == "spacing":
        print(spacing_chart(Path(args.out), args.coordinates))
    else:
        print(floor_table(Path(args.out), [Path(p) for p in args.floors], args.labels))
    return 0


if __name__ == "__main__":
    sys.exit(main())
