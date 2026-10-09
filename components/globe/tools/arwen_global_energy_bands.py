"""Per-operator, per-band kinetic-energy budget of a WOOF global run.

Reads the in-situ ledger (``insitu.ndjson``) of a run whose energy marks
were sampled (``[insitu] energy_every``) and prints, for every operator
of the step and every spectral band the ledger books
(``woof.globe.insitu.energy``: n 1-20, 21-60, 61-120, 121-200,
201-truncation), the column kinetic energy the operator netted, summed
over the sampled steps, as

* the net in J/m2 over the sampled span (northern, southern, global);
* the rate in W/m2 (net / (sampled steps x dt));
* the e-folding time the rate implies against the band's mean energy
  over the span (hours; positive = the operator drains the band,
  negative = it feeds it; blank when the rate is below the reading's
  floor);

plus the same for the reference-level (250 hPa) booking in m2/s2, the
level's rotational / divergent split by Parseval (which part of a band
each operator moves), and the cross-band closure (unfiltered total minus
the sum of the bands) so the reader sees what the band partition does
not hold.  It is a reader: nothing here recomputes a mark.

    python tools/arwen_global_energy_bands.py <run-dir or insitu.ndjson>
        [--hours LO HI] [--out budget.json] [--table budget.md]
        [--png budget.png] [--label NAME]

``--hours`` keeps the sampled steps whose time (hours since the run's
origin, from the rows' ``time_s``) lies in [LO, HI); the default is
every sampled step.  The PNG is an analysis chart (bars of the rate per
operator per band), not a weather field.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

OPERATOR_ORDER = (
    "physics_first",
    "positivity_first",
    "semi_implicit_pre",
    "explicit_dynamics",
    "semi_implicit_post",
    "dynamics_imex",
    "dynamics_explicit",
    "dynamics_implicit",
    "diffusion",
    "mass_fixer",
    "tracer_transport",
    "physics_second",
    "positivity_second",
)


def read_energy_rows(path: Path) -> tuple[dict, list[dict]]:
    """The ledger header and every step row carrying energy marks."""
    path = Path(path)
    if path.is_dir():
        path = path / "insitu.ndjson"
    header = None
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "header":
                header = row
            elif row.get("kind") == "step" and row.get("energy"):
                rows.append(row)
    if header is None:
        raise ValueError(f"{path}: no ledger header")
    if not rows:
        raise ValueError(f"{path}: no step row carries energy marks (energy_every larger than the run?)")
    return header, rows


def aggregate(header: dict, rows: list[dict], hours: tuple[float, float] | None = None) -> dict:
    dt = float(header["dt_s"])
    if hours is not None:
        lo, hi = float(hours[0]) * 3600.0, float(hours[1]) * 3600.0
        rows = [row for row in rows if lo <= float(row["time_s"]) - dt < hi]
        if not rows:
            raise ValueError(f"no sampled step lies in hours [{hours[0]}, {hours[1]})")
    bands = rows[0]["energy"]["bands"]
    labels = list(bands["labels"])
    operators = list(rows[0]["energy"]["operators"])
    nb = len(labels)
    steps = len(rows)
    span_s = steps * dt
    # Sums over the sampled steps: net per operator per band [N, S], the
    # unfiltered total per operator [N, S], and the opening measures.
    column_net = {op: np.zeros((nb, 2)) for op in operators}
    column_total_net = {op: np.zeros(2) for op in operators}
    level_net = {op: np.zeros((nb, 2)) for op in operators}
    level_total_net = {op: np.zeros(2) for op in operators}
    column_start = np.zeros((nb, 2))
    column_total_start = np.zeros(2)
    level_start = np.zeros((nb, 2))
    level_total_start = np.zeros(2)
    # Reference-level [rotational, divergent] by Parseval (sphere means).
    spectral_net = {op: np.zeros((nb, 2)) for op in operators}
    spectral_start = np.zeros((nb, 2))
    # The same by total degree (2026-09-04), when the ledger wrote it.
    t1 = int(header["truncation"]) + 1
    degree_key = "level_spectral_by_degree_m2_s2"
    has_degrees = degree_key in rows[0]["energy"]["band_start"]
    degree_net = {op: np.zeros((2, t1)) for op in operators}
    degree_start = np.zeros((2, t1))
    for row in rows:
        energy = row["energy"]
        if energy["operators"] != operators:
            raise ValueError(f"step {row['step']}: operators {energy['operators']} differ from {operators}")
        start = energy["band_start"]
        column_start += np.asarray(start["column_kinetic_j_m2"]["bands"], dtype=np.float64)
        column_total_start += np.asarray(start["column_kinetic_j_m2"]["total"], dtype=np.float64)
        level_start += np.asarray(start["level_kinetic_m2_s2"]["bands"], dtype=np.float64)
        level_total_start += np.asarray(start["level_kinetic_m2_s2"]["total"], dtype=np.float64)
        if "level_spectral_m2_s2" in start:
            spectral_start += np.asarray(start["level_spectral_m2_s2"]["bands"], dtype=np.float64)
        if has_degrees:
            degree_start[0] += np.asarray(start[degree_key]["rotational"], dtype=np.float64)
            degree_start[1] += np.asarray(start[degree_key]["divergent"], dtype=np.float64)
        for op in operators:
            entry = energy["band_net"][op]
            column_net[op] += np.asarray(entry["column_kinetic_j_m2"]["bands"], dtype=np.float64)
            column_total_net[op] += np.asarray(entry["column_kinetic_j_m2"]["total"], dtype=np.float64)
            level_net[op] += np.asarray(entry["level_kinetic_m2_s2"]["bands"], dtype=np.float64)
            level_total_net[op] += np.asarray(entry["level_kinetic_m2_s2"]["total"], dtype=np.float64)
            if "level_spectral_m2_s2" in entry:
                spectral_net[op] += np.asarray(entry["level_spectral_m2_s2"]["bands"], dtype=np.float64)
            if has_degrees:
                degree_net[op][0] += np.asarray(entry[degree_key]["rotational"], dtype=np.float64)
                degree_net[op][1] += np.asarray(entry[degree_key]["divergent"], dtype=np.float64)
    column_mean = column_start / steps
    level_mean = level_start / steps
    spectral_mean = spectral_start / steps

    def hemispheres(pair):
        return {"northern": float(pair[0]), "southern": float(pair[1]), "global": float(pair[0] + pair[1])}

    def rotdiv(pair):
        return {"rotational": float(pair[0]), "divergent": float(pair[1]), "total": float(pair[0] + pair[1])}

    def rotdiv_efold_hours(net, mean):
        out = {}
        for name, n_value, m_value in (
            ("rotational", net[0], mean[0]), ("divergent", net[1], mean[1]),
            ("total", net[0] + net[1], mean[0] + mean[1]),
        ):
            rate = n_value / span_s
            out[name] = None if rate == 0.0 or m_value <= 0.0 else float(-m_value / rate / 3600.0)
        return out

    def efold_hours(net, mean):
        out = {}
        for name, n_value, m_value in (
            ("northern", net[0], mean[0]), ("southern", net[1], mean[1]),
            ("global", net[0] + net[1], mean[0] + mean[1]),
        ):
            rate = n_value / span_s
            out[name] = None if rate == 0.0 or m_value <= 0.0 else float(-m_value / rate / 3600.0)
        return out

    result = {
        "source_steps": [int(rows[0]["step"]), int(rows[-1]["step"])],
        "source_hours": [float(rows[0]["time_s"]) / 3600.0 - dt / 3600.0, float(rows[-1]["time_s"]) / 3600.0],
        "sampled_steps": steps,
        "dt_s": dt,
        "span_s": span_s,
        "truncation": int(header["truncation"]),
        "nlev": int(header["nlev"]),
        "bands": bands,
        "operators": operators,
        "column": {
            "units": "J/m2 (net over the span), W/m2 (rate), hours (e-folding of the band's mean energy)",
            "mean_band_energy_j_m2": {label: hemispheres(column_mean[i]) for i, label in enumerate(labels)},
            "mean_total_energy_j_m2": hemispheres(column_total_start / steps),
            "cross_band_mean_j_m2": hemispheres(column_total_start / steps - column_mean.sum(axis=0)),
            "net_j_m2": {op: {label: hemispheres(column_net[op][i]) for i, label in enumerate(labels)} for op in operators},
            "total_net_j_m2": {op: hemispheres(column_total_net[op]) for op in operators},
            "cross_band_net_j_m2": {op: hemispheres(column_total_net[op] - column_net[op].sum(axis=0)) for op in operators},
            "rate_w_m2": {op: {label: hemispheres(column_net[op][i] / span_s) for i, label in enumerate(labels)} for op in operators},
            "efold_hours": {op: {label: efold_hours(column_net[op][i], column_mean[i]) for i, label in enumerate(labels)} for op in operators},
        },
        "level": {
            "units": "m2/s2 (net over the span), m2/s3 (rate), hours (e-folding)",
            "level_index": int(bands["level_index"]),
            "level_reference_pressure_pa": float(bands["level_reference_pressure_pa"]),
            "mean_band_energy_m2_s2": {label: hemispheres(level_mean[i]) for i, label in enumerate(labels)},
            "mean_total_energy_m2_s2": hemispheres(level_total_start / steps),
            "net_m2_s2": {op: {label: hemispheres(level_net[op][i]) for i, label in enumerate(labels)} for op in operators},
            "total_net_m2_s2": {op: hemispheres(level_total_net[op]) for op in operators},
            "rate_m2_s3": {op: {label: hemispheres(level_net[op][i] / span_s) for i, label in enumerate(labels)} for op in operators},
            "efold_hours": {op: {label: efold_hours(level_net[op][i], level_mean[i]) for i, label in enumerate(labels)} for op in operators},
        },
        "level_spectral": {
            "units": "m2/s2 sphere mean on the reference level by Parseval: [rotational, divergent]; hours (e-folding)",
            "mean_band_energy_m2_s2": {label: rotdiv(spectral_mean[i]) for i, label in enumerate(labels)},
            "net_m2_s2": {op: {label: rotdiv(spectral_net[op][i]) for i, label in enumerate(labels)} for op in operators},
            "rate_m2_s3": {op: {label: rotdiv(spectral_net[op][i] / span_s) for i, label in enumerate(labels)} for op in operators},
            "efold_hours": {op: {label: rotdiv_efold_hours(spectral_net[op][i], spectral_mean[i]) for i, label in enumerate(labels)} for op in operators},
            "sum_over_operators_m2_s2": {label: rotdiv(sum(spectral_net[op][i] for op in operators)) for i, label in enumerate(labels)},
        },
    }
    # The sum over operators per band is the band's own change over the
    # span (start of the first sampled step to the end of the last, when
    # every step is sampled): the closure of the booking.
    result["column"]["sum_over_operators_j_m2"] = {
        label: hemispheres(sum(column_net[op][i] for op in operators)) for i, label in enumerate(labels)
    }
    result["level"]["sum_over_operators_m2_s2"] = {
        label: hemispheres(sum(level_net[op][i] for op in operators)) for i, label in enumerate(labels)
    }
    if has_degrees:
        # Per total degree on the reference level: the mean energy over
        # the span, each operator's net, and the net as a fraction of the
        # mean per day (the reading that shows a shelf's builder).
        mean = degree_start / steps
        total_mean = mean.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            per_day = {
                op: np.where(total_mean > 0.0, 86400.0 * degree_net[op].sum(axis=0) / span_s / total_mean, 0.0)
                for op in operators
            }
        result["level_by_degree"] = {
            "units": "m2/s2 sphere mean on the reference level by Parseval per total degree: [rotational, divergent]",
            "degrees": list(range(t1)),
            "mean_energy_m2_s2": {"rotational": mean[0].tolist(), "divergent": mean[1].tolist(), "total": total_mean.tolist()},
            "net_m2_s2": {
                op: {"rotational": degree_net[op][0].tolist(), "divergent": degree_net[op][1].tolist(),
                     "total": degree_net[op].sum(axis=0).tolist()}
                for op in operators
            },
            "net_fraction_of_mean_per_day": {op: per_day[op].tolist() for op in operators},
            "sum_over_operators_m2_s2": sum(degree_net[op].sum(axis=0) for op in operators).tolist(),
        }
    return result


def _fmt(value, width=10, digits=3):
    if value is None:
        return " " * (width - 1) + "-"
    if abs(value) >= 1.0e5 or (abs(value) < 1.0e-3 and value != 0.0):
        return f"{value:{width}.{digits}e}"
    return f"{value:{width}.{digits}f}"


def render_table(result: dict, label: str, hemisphere: str = "global") -> str:
    labels = result["bands"]["labels"]
    ops = [op for op in OPERATOR_ORDER if op in result["operators"]] + [
        op for op in result["operators"] if op not in OPERATOR_ORDER
    ]
    lines = []
    lines.append(f"# Kinetic-energy budget by operator and band: {label}")
    lines.append("")
    lines.append(
        f"Sampled steps {result['source_steps'][0]}..{result['source_steps'][1]} "
        f"(hours {result['source_hours'][0]:.2f} to {result['source_hours'][1]:.2f}), "
        f"{result['sampled_steps']} steps of dt {result['dt_s']:.0f} s, span {result['span_s'] / 3600.0:.2f} h, "
        f"T{result['truncation']}, {result['nlev']} levels; hemisphere: {hemisphere}."
    )
    lines.append("")
    col = result["column"]
    lines.append("## Column (mass-weighted) kinetic energy, J/m2 of the sphere mean")
    lines.append("")
    header = "| band | mean energy J/m2 | " + " | ".join(ops) + " | sum over operators |"
    lines.append(header)
    lines.append("|" + "---|" * (len(ops) + 3))
    for band in labels:
        cells = [_fmt(col["net_j_m2"][op][band][hemisphere]) for op in ops]
        lines.append(
            f"| {band} | {_fmt(col['mean_band_energy_j_m2'][band][hemisphere])} | "
            + " | ".join(cells) + f" | {_fmt(col['sum_over_operators_j_m2'][band][hemisphere])} |"
        )
    lines.append("")
    lines.append("Net over the span (J/m2).  Rate (W/m2) and e-folding hours (positive drains the band, negative feeds it):")
    lines.append("")
    lines.append("| band | " + " | ".join(f"{op} W/m2 | {op} h" for op in ops) + " |")
    lines.append("|" + "---|" * (2 * len(ops) + 1))
    for band in labels:
        cells = []
        for op in ops:
            cells.append(_fmt(col["rate_w_m2"][op][band][hemisphere], 10, 4))
            cells.append(_fmt(col["efold_hours"][op][band][hemisphere], 9, 1))
        lines.append(f"| {band} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        f"Unfiltered total mean {_fmt(col['mean_total_energy_j_m2'][hemisphere])} J/m2; "
        f"cross-band term of the mean {_fmt(col['cross_band_mean_j_m2'][hemisphere])} J/m2."
    )
    lines.append("")
    lev = result["level"]
    lines.append(
        f"## Reference level (index {lev['level_index']}, {lev['level_reference_pressure_pa'] / 100.0:.0f} hPa "
        "on the reference column), kinetic energy per unit mass, m2/s2"
    )
    lines.append("")
    lines.append("| band | mean m2/s2 | " + " | ".join(f"{op} net | {op} h" for op in ops) + " |")
    lines.append("|" + "---|" * (2 * len(ops) + 2))
    for band in labels:
        cells = []
        for op in ops:
            cells.append(_fmt(lev["net_m2_s2"][op][band][hemisphere], 10, 4))
            cells.append(_fmt(lev["efold_hours"][op][band][hemisphere], 9, 1))
        lines.append(f"| {band} | {_fmt(lev['mean_band_energy_m2_s2'][band][hemisphere], 10, 4)} | " + " | ".join(cells) + " |")
    lines.append("")
    spec = result["level_spectral"]
    lines.append(
        "## Reference level by Parseval (sphere mean): rotational | divergent kinetic energy per unit mass, m2/s2"
    )
    lines.append("")
    lines.append("| band | mean rot | mean div | " + " | ".join(f"{op} rot | {op} div" for op in ops) + " | sum rot | sum div |")
    lines.append("|" + "---|" * (2 * len(ops) + 5))
    for band in labels:
        cells = []
        for op in ops:
            cells.append(_fmt(spec["net_m2_s2"][op][band]["rotational"], 10, 4))
            cells.append(_fmt(spec["net_m2_s2"][op][band]["divergent"], 10, 4))
        mean = spec["mean_band_energy_m2_s2"][band]
        total = spec["sum_over_operators_m2_s2"][band]
        lines.append(
            f"| {band} | {_fmt(mean['rotational'], 10, 4)} | {_fmt(mean['divergent'], 10, 4)} | "
            + " | ".join(cells) + f" | {_fmt(total['rotational'], 10, 4)} | {_fmt(total['divergent'], 10, 4)} |"
        )
    lines.append("")
    lines.append("E-folding hours of the rotational / divergent part per operator (positive drains, negative feeds):")
    lines.append("")
    lines.append("| band | " + " | ".join(f"{op} rot h | {op} div h" for op in ops) + " |")
    lines.append("|" + "---|" * (2 * len(ops) + 1))
    for band in labels:
        cells = []
        for op in ops:
            cells.append(_fmt(spec["efold_hours"][op][band]["rotational"], 9, 1))
            cells.append(_fmt(spec["efold_hours"][op][band]["divergent"], 9, 1))
        lines.append(f"| {band} | " + " | ".join(cells) + " |")
    lines.append("")
    if "level_by_degree" in result:
        deg = result["level_by_degree"]
        t = len(deg["degrees"]) - 1
        picks = sorted({n for n in (20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 210, 220, 230, 240, 245, 250, 253, t) if 0 < n <= t})
        lines.append(
            "## Reference level by total degree (sphere mean, Parseval): each operator's net as a fraction of "
            "the degree's mean energy per day (positive feeds the degree, negative drains it)"
        )
        lines.append("")
        lines.append("| degree | mean m2/s2 | div share | " + " | ".join(ops) + " | sum /day |")
        lines.append("|" + "---|" * (len(ops) + 4))
        for n in picks:
            mean_total = deg["mean_energy_m2_s2"]["total"][n]
            share = deg["mean_energy_m2_s2"]["divergent"][n] / mean_total if mean_total > 0.0 else 0.0
            cells = [_fmt(deg["net_fraction_of_mean_per_day"][op][n], 9, 3) for op in ops]
            total = sum(deg["net_fraction_of_mean_per_day"][op][n] for op in ops)
            lines.append(f"| {n} | {_fmt(mean_total, 10, 4)} | {share:.2f} | " + " | ".join(cells) + f" | {_fmt(total, 9, 3)} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_degree_png(result: dict, label: str, path: Path, lowest_degree: int = 20) -> None:
    """Per-degree rates on the reference level: each operator's net as a
    fraction of the degree's mean energy per day, under the mean spectrum."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    deg = result["level_by_degree"]
    n = np.asarray(deg["degrees"])
    keep = n >= int(lowest_degree)
    ops = [op for op in OPERATOR_ORDER if op in result["operators"]] + [
        op for op in result["operators"] if op not in OPERATOR_ORDER
    ]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1.6]})
    mean = deg["mean_energy_m2_s2"]
    top.loglog(n[keep], np.asarray(mean["total"])[keep], color="black", label="total")
    top.loglog(n[keep], np.asarray(mean["rotational"])[keep], color="tab:blue", label="rotational")
    top.loglog(n[keep], np.asarray(mean["divergent"])[keep], color="tab:red", label="divergent")
    top.set_ylabel("mean KE per unit mass, m2/s2")
    top.legend(fontsize=8)
    top.grid(alpha=0.3, which="both")
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(ops)))
    total = np.zeros(n.size)
    for i, op in enumerate(ops):
        rate = np.asarray(deg["net_fraction_of_mean_per_day"][op])
        if not np.any(rate != 0.0):
            continue
        total += rate
        bottom.semilogx(n[keep], rate[keep], color=colors[i], label=op, linewidth=1.4)
    bottom.semilogx(n[keep], total[keep], color="black", linestyle="--", label="sum", linewidth=1.0)
    bottom.axhline(0.0, color="black", linewidth=0.8)
    bottom.set_xlabel("total spherical-harmonic degree n")
    bottom.set_ylabel("net rate, fraction of the degree's mean energy per day")
    bottom.legend(fontsize=8, ncol=3)
    bottom.grid(alpha=0.3, which="both")
    level = result["level"]
    fig.suptitle(f"{label}: per-operator kinetic-energy traffic by degree, {level['level_reference_pressure_pa'] / 100.0:.0f} hPa (sphere mean)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def render_png(result: dict, label: str, path: Path, hemisphere: str = "global", kind: str = "column") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = result["bands"]["labels"]
    ops = [op for op in OPERATOR_ORDER if op in result["operators"]] + [
        op for op in result["operators"] if op not in OPERATOR_ORDER
    ]
    block = result[kind]
    rate_key = "rate_w_m2" if kind == "column" else "rate_m2_s3"
    mean_key = "mean_band_energy_j_m2" if kind == "column" else "mean_band_energy_m2_s2"
    # Rate as a fraction of the band's mean energy per day: comparable
    # across bands whose energies differ by orders of magnitude.
    data = np.zeros((len(ops), len(labels)))
    for i, op in enumerate(ops):
        for j, band in enumerate(labels):
            mean = block[mean_key][band][hemisphere]
            rate = block[rate_key][op][band][hemisphere]
            data[i, j] = 0.0 if mean <= 0.0 else 86400.0 * rate / mean
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.8 / len(ops)
    x = np.arange(len(labels))
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, len(ops)))
    for i, op in enumerate(ops):
        ax.bar(x + (i - 0.5 * (len(ops) - 1)) * width, data[i], width, label=op, color=colors[i])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("net kinetic-energy rate, fraction of the band's mean energy per day")
    unit = "column, mass-weighted" if kind == "column" else f"{block['level_reference_pressure_pa'] / 100.0:.0f} hPa level"
    ax.set_title(f"{label}: per-operator kinetic-energy traffic by spectral band ({unit}, {hemisphere})")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="run directory or insitu.ndjson")
    parser.add_argument("--hours", type=float, nargs=2, default=None, metavar=("LO", "HI"))
    parser.add_argument("--out", type=Path, default=None, help="budget JSON")
    parser.add_argument("--table", type=Path, default=None, help="markdown table")
    parser.add_argument("--png", type=Path, default=None, help="bar chart of the column rates")
    parser.add_argument("--png-level", type=Path, default=None, help="bar chart of the level rates")
    parser.add_argument("--png-degree", type=Path, default=None, help="per-degree rates on the reference level (needs a ledger that wrote them)")
    parser.add_argument("--hemisphere", default="global", choices=("global", "northern", "southern"))
    parser.add_argument("--label", default=None)
    args = parser.parse_args(argv)
    header, rows = read_energy_rows(args.source)
    result = aggregate(header, rows, None if args.hours is None else tuple(args.hours))
    label = args.label or str(args.source)
    result["label"] = label
    result["hemisphere_shown"] = args.hemisphere
    table = render_table(result, label, args.hemisphere)
    sys.stdout.write(table)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    if args.table is not None:
        args.table.parent.mkdir(parents=True, exist_ok=True)
        args.table.write_text(table, encoding="utf-8")
    if args.png is not None:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        render_png(result, label, args.png, args.hemisphere, "column")
    if args.png_level is not None:
        args.png_level.parent.mkdir(parents=True, exist_ok=True)
        render_png(result, label, args.png_level, args.hemisphere, "level")
    if args.png_degree is not None:
        if "level_by_degree" not in result:
            raise ValueError("this ledger carries no per-degree reading (written since 2026-09-04)")
        args.png_degree.parent.mkdir(parents=True, exist_ok=True)
        render_degree_png(result, label, args.png_degree)
    return 0


if __name__ == "__main__":
    sys.exit(main())
