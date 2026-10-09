"""Analysis charts of the YSU free-atmosphere lane (matplotlib on
instrument JSON; no weather fields).

    python tools/arwen_global_ysu_lane_charts.py spectrum --out PNG
        --arm LABEL=spectrum.json [--arm ...] [--region northern]
    python tools/arwen_global_ysu_lane_charts.py kprofile --out PNG
        --probe LABEL=probe.json [--probe ...] [--wrf LABEL=compare.json]
        [--region north_20_70]
    python tools/arwen_global_ysu_lane_charts.py bands --out PNG
        --bands LABEL=bands.json [--bands ...] [--hemisphere northern]
    python tools/arwen_global_ysu_lane_charts.py degrees --out PNG
        --bands LABEL=bands.json [--bands ...] [--hemisphere northern]

``spectrum`` draws, for one region, the 250 hPa kinetic-energy spectrum
of every arm averaged over its hours-12-to-24 samples beside the
Lindborg (1999) fit the instrument anchors on, and the model-over-
observed ratio by wavelength with the 0.5 and 0.2 lines the effective-
resolution reading uses; the legend carries the instrument's own
retained fraction at 250 km and its grid-limit ratio (the score chain's
run_fractions arithmetic: means over the region's 250 hPa samples).
``kprofile`` draws the probe's interface-mean momentum diffusivity by
pressure for each probed arm (the kernel's word), the control's formula
under both length rules, the WRF Fortran reference's mean on the
exported cells, the asymptotic length both rules give, and the
free-atmosphere kinetic tendency per level.  ``bands`` draws the ledger
reader's per-operator, per-band column rates for one hemisphere, one
panel per arm; ``degrees`` the per-degree physics fraction per day at
the reference level for every arm on one axis.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

EARTH_RADIUS_M = 6_371_220.0


def _pairs(values):
    out = []
    for item in values or ():
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"expected LABEL=path, got {item!r}")
        out.append((label, Path(path)))
    return out


def lindborg_per_degree(n: np.ndarray, d1: float, d2: float, a: float = EARTH_RADIUS_M) -> np.ndarray:
    """E_n = E(k_n) dk/dn with k_n = sqrt(n(n+1))/a and E(k) = d1 k^-5/3 + d2 k^-3."""
    n = np.asarray(n, dtype=np.float64)
    k = np.sqrt(n * (n + 1.0)) / a
    dk_dn = (2.0 * n + 1.0) / (2.0 * a * np.sqrt(n * (n + 1.0)))
    return (d1 * k ** (-5.0 / 3.0) + d2 * k ** (-3.0)) * dk_dn


def wavelength_km(n: np.ndarray, a: float = EARTH_RADIUS_M) -> np.ndarray:
    n = np.asarray(n, dtype=np.float64)
    return 2.0 * np.pi * a / np.sqrt(n * (n + 1.0)) / 1000.0


def spectrum_curves(path: Path, region: str, level_pa: float = 25_000.0, hours: tuple[float, float] = (12.0, 24.0)):
    d = json.loads(path.read_text(encoding="utf-8"))
    m = d["measurements"]
    samples = [s for s in m["samples"] if s["region"] == region and abs(s["level_pa"] - level_pa) < 5000.0
               and hours[0] <= s["hours"] <= hours[1]]
    if not samples:
        raise SystemExit(f"{path}: no {region} samples near {level_pa} Pa in hours {hours}")
    spectra = np.asarray([s["spectrum_total"] for s in samples], dtype=np.float64)
    mean = spectra.mean(axis=0)
    ref = d["provenance"]["absolute_reference"]
    n = np.arange(mean.size)
    observed = np.full(mean.size, np.nan)
    observed[1:] = lindborg_per_degree(n[1:], ref["d1_m43_s2"], ref["d2_m2_s2"])
    f250 = [s["absolute"]["retained_fraction"]["250_km"] for s in samples if s["absolute"].get("retained_fraction")]
    grid = [s["absolute"]["ratio_at_smallest_wavelength"] for s in samples if s["absolute"].get("ratio_at_smallest_wavelength") is not None]
    resolved = [s["absolute"]["wavelength_km"] for s in samples if s["absolute"].get("status") == "resolved"]
    return {
        "n": n, "mean": mean, "observed": observed, "samples": len(samples),
        "ratio_250km": float(np.mean(f250)) if f250 else float("nan"),
        "grid_limit_ratio": float(np.mean(grid)) if grid else float("nan"),
        "resolved_km": float(np.mean(resolved)) if resolved else float("nan"),
        "resolved_n": len(resolved),
        "truncation_km": float(m["truncation_wavelength_km"]),
    }


def chart_spectrum(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = _pairs(args.arm)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    summary = {}
    for (label, path), color in zip(arms, colors):
        c = spectrum_curves(path, args.region)
        n = c["n"][1:]
        lam = wavelength_km(n)
        ax1.loglog(lam, c["mean"][1:], color=color, lw=1.6, label=f"{label} ({c['samples']} samples)")
        ratio = c["mean"][1:] / c["observed"][1:]
        ax2.semilogx(lam, ratio, color=color, lw=1.6,
                     label=f"{label}: 250 km ratio {c['ratio_250km']:.3f}, grid limit {c['grid_limit_ratio']:.3f}"
                           + (f", resolved {c['resolved_km']:.0f} km ({c['resolved_n']})" if c["resolved_n"] else ", unresolved"))
        summary[label] = {k: v for k, v in c.items() if k not in ("n", "mean", "observed")}
        observed = c["observed"]
        truncation_km = c["truncation_km"]
    ax1.loglog(wavelength_km(c["n"][1:]), observed[1:], "k--", lw=1.2, label="Lindborg (1999) fit, aircraft")
    for ax in (ax1, ax2):
        ax.invert_xaxis()
        ax.set_xlabel("wavelength, km")
        ax.axvline(250.0, color="0.6", lw=0.8, ls=":")
        ax.axvline(truncation_km, color="0.3", lw=0.8, ls="-.")
        ax.grid(True, which="both", alpha=0.25)
    ax1.set_ylabel("kinetic energy per degree, m2/s2")
    ax1.set_ylim(3.0e-5, 40.0)
    ax1.set_title(f"{args.region} 250 hPa spectrum, hours 12 to 24")
    ax1.set_xlim(4000.0, 140.0)
    ax2.axhline(1.0, color="k", lw=0.8)
    ax2.axhline(0.5, color="0.4", lw=0.8, ls="--")
    ax2.axhline(0.2, color="0.4", lw=0.8, ls=":")
    ax2.set_ylabel("model / observed")
    ax2.set_ylim(0.0, 1.6)
    ax2.set_xlim(4000.0, 140.0)
    ax2.set_title("ratio to the observed spectrum (0.5: effective resolution; 0.2: confirmation)")
    ax1.legend(fontsize=8, loc="lower left")
    ax2.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def chart_kprofile(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    probes = _pairs(args.probe)
    wrf = _pairs(args.wrf) if args.wrf else []
    region = args.region
    fig, axes = plt.subplots(1, 4, figsize=(19, 6.2), sharey=True)
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    first = None
    for (label, path), color in zip(probes, colors):
        r = json.loads(path.read_text(encoding="utf-8"))
        rows = r["per_interface"]
        p = np.asarray([row["p_interface_pa"] for row in rows]) / 100.0
        e = [row["regions"][region] for row in rows]
        km = np.asarray([x["exch_m_mean"] for x in e])
        p99 = np.asarray([x["exch_m_percentiles"]["p99"] for x in e])
        mode = r["mixing_length_mode"]
        axes[0].plot(km, p, "-o", color=color, ms=3, lw=1.5, label=f"{label}: kernel exch_m mean ({mode})")
        axes[0].plot(p99, p, ":", color=color, lw=1.0, label=f"{label}: 99th percentile")
        if first is None:
            first = (r, e, p)
            axes[0].plot([x["xkzm_fixed_mean"] for x in e], p, "--", color="0.2", lw=1.2, label=f"{label}: formula, fixed 30 m length")
            axes[0].plot([x["xkzm_wrf-layer_mean"] for x in e], p, "-.", color="0.5", lw=1.0, label=f"{label}: formula, WRF 0.1 dz length")
            axes[1].plot([x["rlamdz_wrf-layer_mean"] for x in e], p, "-", color="0.5", lw=1.5, label="WRF rule: clip(0.1 dz, 30, 300) m")
            axes[1].plot([x["rlamdz_fixed_mean"] for x in e], p, "--", color="0.2", lw=1.5, label="fixed: 30 m")
            axes[1].plot([x["dza_mean"] for x in e], p, "-", color="#ff7f0e", lw=1.0, label="layer thickness dz, m")
            axes[2].plot([x["shear_mean"] * 1000.0 for x in e], p, "-", color="#8c564b", lw=1.2, label=f"{label}: shear, 1e-3 1/s")
            axes[2].plot([x["ri_median"] for x in e], p, "--", color="#17becf", lw=1.2, label=f"{label}: median Ri")
        tend = r["per_level_kinetic_tendency_w_m2"]
        ridx = tend["regions"].index(region)
        pf = np.asarray(tend["p_full_pa"]) / 100.0
        free = np.asarray(tend["free_atmosphere"])[:, ridx] * 1000.0
        axes[3].plot(free, pf, "-o", color=color, ms=3, lw=1.5, label=f"{label}: free-atmosphere KE tendency, mW/m2 per level")
    for label, path in wrf:
        c = json.loads(path.read_text(encoding="utf-8"))
        pl = c["per_level"]
        # exch_m[k] sits on the interface below level k: plot at the interface pressure of the probe rows.
        p_int = np.asarray([row["p_interface_pa"] for row in first[0]["per_interface"]]) / 100.0
        wrf_mean = np.asarray([row["wrf_exch_m_mean"] for row in pl])[1:]
        ker_mean = np.asarray([row["kernel_exch_m_mean"] for row in pl])[1:]
        axes[0].plot(wrf_mean, p_int, "s", color="k", ms=4, mfc="none", label=f"{label}: WRF Fortran on {c['cells']} cells")
        lo_k, hi_k = c["levels_of_interest"]
        axes[0].plot(ker_mean, p_int, "x", color="k", ms=4,
                     label=f"{label}: kernel on the same cells (max {c['summary']['exch_m_max_ulp_levels_of_interest']} ULP at levels {lo_k}..{hi_k})")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("momentum diffusivity K_m, m2/s")
    axes[0].set_title(f"K profile above the boundary layer, {region}")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("m")
    axes[1].set_title("asymptotic mixing length and layer thickness")
    axes[2].set_xlabel("shear (1e-3 1/s), median Ri")
    axes[2].set_xlim(-0.5, 8.0)
    axes[2].set_title("resolved shear and Richardson number")
    axes[3].axvline(0.0, color="k", lw=0.8)
    axes[3].set_xlabel("mW/m2 per level (negative drains)")
    axes[3].set_title("YSU kinetic-energy tendency per level")
    axes[0].set_ylabel("pressure, hPa")
    axes[0].set_ylim(1000.0, 50.0)
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


def chart_bands(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = _pairs(args.bands)
    fig, axes = plt.subplots(1, len(arms), figsize=(6.2 * len(arms), 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    operators = ("dynamics", "diffusion", "physics")
    colors = {"dynamics": "#2ca02c", "diffusion": "#d62728", "physics": "#9467bd"}
    hemi = args.hemisphere
    table = {}
    for ax, (label, path) in zip(axes, arms):
        d = json.loads(path.read_text(encoding="utf-8"))
        labels = d["bands"]["labels"]
        ops = d["operators"]
        grouped = {name: np.zeros(len(labels)) for name in operators}
        if args.level:
            # The reference level (237 hPa): each operator's rate as a
            # fraction of the band's mean energy per day, the reading the
            # per-degree chart gives, band by band.
            rate = d["level"]["rate_m2_s3"]
            energy = np.asarray([d["level"]["mean_band_energy_m2_s2"][b][hemi] for b in labels])
            scale = 86400.0 / energy
        else:
            rate = d["column"]["rate_w_m2"]
            scale = np.full(len(labels), 1000.0)
        for op in ops:
            group = ("physics" if op.startswith("physics") else "diffusion" if op == "diffusion"
                     else "dynamics" if op.startswith("dynamics") or op.startswith("explicit") or op.startswith("semi") else None)
            if group is None:
                continue
            grouped[group] += np.asarray([rate[op][b][hemi] for b in labels]) * scale
        x = np.arange(len(labels))
        width = 0.27
        for i, name in enumerate(operators):
            ax.bar(x + (i - 1) * width, grouped[name], width, color=colors[name], label=name)
        ax.set_yscale("symlog", linthresh=0.1 if args.level else 0.01)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, fontsize=8)
        ax.axhline(0.0, color="k", lw=0.8)
        where = f"{d['level']['level_reference_pressure_pa'] / 100.0:.0f} hPa" if args.level else "column"
        ax.set_title(f"{label}: hours {d['source_hours'][0]:.0f} to {d['source_hours'][1]:.0f}, {hemi}, {where}", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        table[label] = {name: {b: float(v) for b, v in zip(labels, grouped[name])} for name in operators}
        table[label]["physics_efold_hours_level"] = {
            b: d["level"]["efold_hours"]["physics_first"][b][hemi] for b in labels
        }
        table[label]["level_mean_band_energy_m2_s2"] = {
            b: d["level"]["mean_band_energy_m2_s2"][b][hemi] for b in labels
        }
    axes[0].set_ylabel("fraction of the band's energy per day (negative drains)" if args.level
                       else "column kinetic-energy rate, mW/m2 (negative drains)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    Path(args.out).with_suffix(".json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    print(json.dumps(table, indent=1))


def chart_degrees(args) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = _pairs(args.bands)
    fig, axes = plt.subplots(1, len(arms), figsize=(6.4 * len(arms), 5.4), sharey=True)
    axes = np.atleast_1d(axes)
    groups = {"dynamics": ("dynamics_explicit", "dynamics_implicit", "explicit_dynamics", "semi_implicit_pre",
                           "semi_implicit_post", "dynamics_imex"),
              "diffusion": ("diffusion",), "physics": ("physics_first", "physics_second")}
    colors = {"dynamics": "#2ca02c", "diffusion": "#d62728", "physics": "#9467bd"}
    summary = {}
    for ax, (label, path) in zip(axes, arms):
        d = json.loads(path.read_text(encoding="utf-8"))
        lb = d.get("level_by_degree")
        if not lb:
            ax.set_title(f"{label}: no per-degree block")
            continue
        fractions = lb["net_fraction_of_mean_per_day"]
        n = np.asarray(lb["degrees"], dtype=np.float64)
        total = np.zeros(n.size)
        entry = {}
        for name, ops in groups.items():
            series = np.zeros(n.size)
            for op in ops:
                if op in fractions:
                    series += np.asarray(fractions[op], dtype=np.float64)
            total += series
            ax.plot(n[args.from_degree:], series[args.from_degree:], color=colors[name], lw=1.4, label=name)
            entry[name] = {str(k): float(series[k]) for k in (120, 160, 200, 230, 250, int(n[-1]))}
        ax.plot(n[args.from_degree:], total[args.from_degree:], "k--", lw=1.0, label="sum (the degree's own growth)")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.axvline(160.0, color="0.6", lw=0.8, ls=":")
        ax.set_yscale("symlog", linthresh=0.5)
        ax.set_xlabel("total degree n (160 = 250 km)")
        ax.set_title(f"{label}: hours {d['source_hours'][0]:.0f} to {d['source_hours'][1]:.0f}", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")
        summary[label] = entry
    axes[0].set_ylabel("fraction of the degree's energy per day (negative drains)")
    fig.suptitle("each operator's kinetic-energy net at the reference level (237 hPa), per total degree, as a fraction of the degree's mean energy per day", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("spectrum")
    s.add_argument("--out", required=True)
    s.add_argument("--arm", action="append", required=True)
    s.add_argument("--region", default="northern")
    k = sub.add_parser("kprofile")
    k.add_argument("--out", required=True)
    k.add_argument("--probe", action="append", required=True)
    k.add_argument("--wrf", action="append")
    k.add_argument("--region", default="north_20_70")
    b = sub.add_parser("bands")
    b.add_argument("--out", required=True)
    b.add_argument("--bands", action="append", required=True)
    b.add_argument("--hemisphere", default="northern")
    b.add_argument("--level", action="store_true", help="the reference level's rates as fractions of the band energy per day instead of the column's W/m2")
    g = sub.add_parser("degrees")
    g.add_argument("--out", required=True)
    g.add_argument("--bands", action="append", required=True)
    g.add_argument("--from-degree", type=int, default=100)
    args = parser.parse_args(argv)
    {"spectrum": chart_spectrum, "kprofile": chart_kprofile, "bands": chart_bands, "degrees": chart_degrees}[args.command](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
