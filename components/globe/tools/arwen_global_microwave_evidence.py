"""Evidence charts for the microwave leg (analysis charts, not weather
fields, so matplotlib is the right tool under the render law).

Reads the ``score`` door's outputs (``microwave-scorecard.json``,
``microwave-cells.npz``, ``microwave-calibration.json``,
``microwave-operator-entry.json``) and writes:

* ``01-ob-per-channel.png``: bias and rmse per channel before and after
  the bias corrections (constant, background, geometry, geometry plus
  wind), the noise floor and the 1 K bar drawn.
* ``02-weighting-functions.png``: the sounding channels' weighting
  functions on the calibration column, centroid pressures labelled.
* ``03-ob-map-chNN.png``: the clear-sky over-ocean cells of the day with
  O-B of one channel as colour.
* ``04-ob-by-scan-angle.png``: raw O-B against satellite zenith for the
  sounding channels.
* ``05-screen-funnel.png``: the screening counts.
* ``06-ob-by-wind.png``: raw O-B against the analysis 10 m wind for the
  surface-sensitive channels (the roughness term).
* ``07-ob-vs-background-ch14-15.png``: O-B of the two highest channels
  against their background brightness temperature (the upper term).
* ``CAPTIONS.md``: plain-language captions.

Usage: python tools/arwen_global_microwave_evidence.py SCORE_DIR OUT_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from woof.globe.microwave import calibrate  # noqa: E402
from woof.globe.microwave.channels import CHANNELS, TEMPERATURE_SOUNDING_CHANNELS  # noqa: E402
from woof.globe.microwave.entry import channel_vertical  # noqa: E402
from woof.globe.microwave.rte import weighting_function  # noqa: E402


def main(score_dir: str, out_dir: str) -> int:
    score_dir = Path(score_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    card = json.loads((score_dir / "microwave-scorecard.json").read_text(encoding="utf-8"))
    cells = np.load(score_dir / "microwave-cells.npz")
    entry_path = score_dir / "microwave-operator-entry.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8")) if entry_path.exists() else None
    channels = [c["channel"] for c in card["channels"]]
    verdict = card["verdict"]
    bar = verdict["bar_k"]
    n_scored = card["provenance"]["cells_scored"]
    admitted = verdict.get("admitted_channels", verdict.get("within_bar", []))
    outside = verdict["outside_bar"]
    terms = verdict.get("nearest_term", {})
    captions = [
        "# Microwave leg evidence (ATMS clear-sky over ocean against GDAS columns)\n",
        f"NOAA-21 ATMS, 2026-09-01, {n_scored:,} clear-sky ocean cells on the 0.25 degree grid in hourly bins, "
        "scored against GDAS analysis columns interpolated to each cell's time. Every rmse after a correction "
        "is out of sample (fitted on half the cells in time order, scored on the other half). "
        f"Admitted sounding channels (geometry correction, rmse at or under {bar} K): {admitted}. "
        f"Outside the bar: {outside}" + (
            " (" + "; ".join(f"channel {k}: {v}" for k, v in terms.items()) + ")." if terms else "."
        ) + "\n",
    ]

    # 1. per-channel O-B
    bias = np.array([c["raw"]["bias"] for c in card["channels"]])
    rmse = np.array([c["raw"]["rmse"] for c in card["channels"]])
    rmse_const = np.array([c["constant_corrected"]["rmse"] for c in card["channels"]])
    rmse_lin = np.array([c["linear_corrected"]["rmse"] for c in card["channels"]])
    rmse_geo = np.array([c["geometry_corrected"]["rmse"] for c in card["channels"]])
    rmse_wind = np.array([c["wind_corrected"]["rmse"] for c in card["channels"]])
    floor = np.array([c["noise_floor_k"] if c["noise_floor_k"] is not None else np.nan for c in card["channels"]])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True)
    x = np.arange(len(channels))
    axes[0].bar(x, bias, color=["#1f77b4" if c in TEMPERATURE_SOUNDING_CHANNELS else "#bbbbbb" for c in channels])
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_ylabel("O-B bias (K)")
    axes[0].set_title(f"ATMS NOAA-21 2026-09-01, {n_scored:,} clear-sky ocean cells; "
                      "blue = temperature-sounding channels 4 to 15")
    axes[1].plot(x, rmse, "o-", label="raw rmse")
    axes[1].plot(x, rmse_const, "s--", label="after constant correction")
    axes[1].plot(x, rmse_lin, "^-", label="after correction linear in background")
    axes[1].plot(x, rmse_geo, "D-", color="k", label="after geometry correction (the entry's model, the record)")
    axes[1].plot(x, rmse_wind, "v:", label="geometry plus analysis 10 m wind (diagnostic)")
    axes[1].plot(x, floor, "_", color="gray", markersize=12, label="noise floor left in the cell mean")
    axes[1].axhline(bar, color="r", lw=1, ls=":", label=f"{bar} K bar")
    axes[1].set_ylabel("O-B rmse (K)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{c}\n{CHANNELS[c - 1].centre_ghz:.1f}" for c in channels], fontsize=8)
    axes[1].set_xlabel("ATMS channel / centre GHz")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "01-ob-per-channel.png", dpi=130)
    plt.close(fig)
    captions.append(
        "## 01-ob-per-channel.png\n"
        "Top: mean observation-minus-model difference per channel (the bias) before any correction. "
        "Bottom: the scatter (rmse) before and after removing a per-channel bias: a constant, a line in the "
        "modelled brightness temperature, the geometry model (constant, background and path length) the "
        "ensemble operator carries, and that model plus the analysis 10 m wind as a diagnostic. The gray "
        "ticks are the instrument noise left in a cell mean (within-cell spread over the square root of the "
        "beam count); an rmse cannot fall below them. "
        f"Sounding channels inside the {bar} K bar after the geometry correction: {admitted}; outside: {outside}.\n"
    )

    # 2. weighting functions
    column = calibrate.standard_column()
    fig, ax = plt.subplots(figsize=(7, 8))
    for number in TEMPERATURE_SOUNDING_CHANNELS:
        p, w, t_s = weighting_function(column, number, 30.0)
        dlnp = np.gradient(np.log(p[:, 0]))
        density = w[:, 0] / np.abs(dlnp)
        centre_pa, _ = channel_vertical(number, 30.0)
        ax.plot(density / density.max(), p[:, 0],
                label=f"ch{number} centroid {centre_pa / 100.0:.1f} hPa, surface {t_s[0]:.2f}")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_ylim(1100, 0.5)
    ax.set_xlabel("temperature weight per unit ln p (normalised)")
    ax.set_ylabel("pressure (hPa)")
    ax.set_title("ATMS temperature-sounding weighting functions, 30 degree zenith, tropical column")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "02-weighting-functions.png", dpi=130)
    plt.close(fig)
    captions.append(
        "## 02-weighting-functions.png\n"
        "Where each sounding channel gets its signal from: the layer whose temperature a channel reads, "
        "with the centroid pressure the operator entry places the channel at and the fraction of the "
        "signal that comes from the surface. Channel 4 sees the lower troposphere and a third of the "
        "surface, channel 9 the tropopause, channels 11 to 15 the stratosphere up to about 2 hPa.\n"
    )

    # 3. map of O-B for a mid-troposphere channel
    for number in (6, 9):
        k = channels.index(number)
        omb = cells["observed"][:, k] - cells["background"][:, k]
        fig, ax = plt.subplots(figsize=(12, 5.5))
        sc = ax.scatter(cells["longitude"], cells["latitude"], c=omb - np.nanmean(omb), s=2,
                        cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xlim(0, 360)
        ax.set_ylim(-62, 62)
        ax.set_xlabel("longitude (deg E)")
        ax.set_ylabel("latitude")
        ax.set_title(f"ATMS channel {number} ({CHANNELS[number - 1].centre_ghz:.2f} GHz) O-B minus its mean, "
                     f"clear-sky ocean cells, 2026-09-01 (mean {np.nanmean(omb):+.2f} K)")
        fig.colorbar(sc, ax=ax, label="O-B anomaly (K)")
        fig.tight_layout()
        fig.savefig(out / f"03-ob-map-ch{number:02d}.png", dpi=130)
        plt.close(fig)
        captions.append(
            f"## 03-ob-map-ch{number:02d}.png\n"
            f"Every clear-sky ocean cell scored on the day, coloured by channel {number}'s difference from "
            "the model after removing the day's mean. Structure that follows the satellite track or the "
            "scan edges is instrument or operator geometry; structure that follows weather is the analysis.\n"
        )

    # 4. O-B by zenith angle
    fig, ax = plt.subplots(figsize=(10, 5))
    zen = cells["zenith"]
    edges = np.arange(0, 61, 5)
    for number in (4, 6, 8, 10, 12, 14):
        k = channels.index(number)
        omb = cells["observed"][:, k] - cells["background"][:, k]
        means = [np.nanmean(omb[(zen >= a) & (zen < b)]) if np.any((zen >= a) & (zen < b)) else np.nan
                 for a, b in zip(edges[:-1], edges[1:])]
        ax.plot(0.5 * (edges[:-1] + edges[1:]), means, "o-", label=f"ch{number}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("satellite zenith angle (deg)")
    ax.set_ylabel("mean O-B (K)")
    ax.set_title("Raw O-B against viewing angle: a slope is a path-length term of the operator")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "04-ob-by-scan-angle.png", dpi=130)
    plt.close(fig)
    captions.append(
        "## 04-ob-by-scan-angle.png\n"
        "Mean O-B in 5 degree bins of viewing angle. A channel whose bias changes with angle has an "
        "operator term that depends on path length (absorption strength, the polarization mix, or the "
        "plane-parallel geometry); a flat line means the geometry is handled. The geometry correction "
        "removes the linear part of this slope.\n"
    )

    # 5. screening funnel
    stages = card["screen"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = list(stages.keys())
    values = [stages[n] for n in names]
    ax.barh(range(len(names)), values, color="#4c72b0")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("cells surviving")
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    ax.set_title("Clear-sky over-ocean screening, one day of NOAA-21 ATMS on the 0.25 degree grid")
    fig.tight_layout()
    fig.savefig(out / "05-screen-funnel.png", dpi=130)
    plt.close(fig)
    captions.append(
        "## 05-screen-funnel.png\n"
        "How many 0.25 degree hourly cells the day produced and how many survive each screen: the "
        "geometry candidates (inside 60 degrees latitude, three beams, zenith under 60), open ocean, the "
        "analysis cloud water, the window-channel cloud retrieval, inside the analysis time span, finite "
        "radiances.\n"
    )

    # 6. O-B by wind for the surface-sensitive channels
    wind = cells["wind_speed"]
    edges = np.arange(0, 16.1, 2.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    for number in (1, 2, 3, 4, 5, 16):
        k = channels.index(number)
        omb = cells["observed"][:, k] - cells["background"][:, k]
        omb = omb - np.nanmean(omb)
        means = [np.nanmean(omb[(wind >= a) & (wind < b)]) if np.any((wind >= a) & (wind < b)) else np.nan
                 for a, b in zip(edges[:-1], edges[1:])]
        slope = card["channels"][k]["wind_coefficients"]["d"]
        ax.plot(0.5 * (edges[:-1] + edges[1:]), means, "o-", label=f"ch{number} ({slope:+.2f} K per m/s)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("analysis 10 m wind speed (m/s)")
    ax.set_ylabel("mean O-B minus the day's mean (K)")
    ax.set_title("Raw O-B against wind: the specular ocean emissivity omits roughness and foam")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "06-ob-by-wind.png", dpi=130)
    plt.close(fig)
    captions.append(
        "## 06-ob-by-wind.png\n"
        "Mean O-B (the day's mean removed) in 2 m/s bins of the analysis 10 m wind for the channels that "
        "see the surface. The window channels 1, 2, 3 and 16 read warmer than the model by a few tenths of "
        "a kelvin per m/s, the signature of the wind-roughened, foam-covered sea the specular emissivity "
        "model does not carry; channel 4 inherits a third of it through its surface transmittance and is "
        "the channel this term holds at the bar (admitted 0.03 K inside it). Channels 5 and above barely see it.\n"
    )

    # 7. O-B against background for channels 14 and 15
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, number in zip(axes, (14, 15)):
        k = channels.index(number)
        o = cells["observed"][:, k]
        b = cells["background"][:, k]
        good = np.isfinite(o) & np.isfinite(b)
        ax.scatter(b[good], (o - b)[good], s=2, alpha=0.4)
        coef = card["channels"][k]["geometry_coefficients"]
        grid = np.linspace(np.nanmin(b[good]), np.nanmax(b[good]), 50)
        ax.plot(grid, coef["a"] + coef["b"] * (grid - coef["mean_background_k"]), "r-",
                label=f"fit a + b (B - mean): b = {coef['b']:+.3f}")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel(f"channel {number} background brightness temperature (K)")
        ax.set_title(f"ch{number}: rmse after geometry correction {card['channels'][k]['geometry_corrected']['rmse']:.2f} K, "
                     f"noise floor {card['channels'][k]['noise_floor_k']:.2f} K", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("O-B (K)")
    fig.tight_layout()
    fig.savefig(out / "07-ob-vs-background-ch14-15.png", dpi=130)
    plt.close(fig)
    captions.append(
        "## 07-ob-vs-background-ch14-15.png\n"
        "The two highest channels (weighting functions near 5 and 2 hPa) against their own modelled "
        "brightness temperature. A slope means the operator or the analysis scales the upper stratosphere "
        "wrongly; the spread that remains after the fit is the instrument noise plus what the analysis and "
        "the absorption above 5 hPa get wrong (the Zeeman splitting of the 57 GHz lines is not modelled). "
        "Channel 15 is the sounding channel the bar refuses.\n"
    )

    if entry is not None and entry.get("operator_entry_ships"):
        captions.append(
            "## Operator entry\n"
            f"`{entry['name']}` admits channels {entry['admitted_channels']} with per-channel errors "
            + ", ".join(f"ch{c['channel']} {c['error_k']:.2f} K" for c in entry["channels"])
            + ". " + entry["contract"]["assimilation_status"] + "\n"
        )

    (out / "CAPTIONS.md").write_text("\n".join(captions), encoding="utf-8")
    print(json.dumps({"out": str(out), "files": sorted(p.name for p in out.iterdir())}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
