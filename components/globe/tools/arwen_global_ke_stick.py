"""Plot WOOF global run checkpoints through the dynamics.spectrum instrument.

This is GLUE, not arithmetic: the spectra and readings come from the
official harness instrument
(``woof.verify.harness.dynamics_spectrum.measure_run``) - kinetic energy
by spherical-harmonic total degree from the checkpoint's own
vorticity/divergence coefficients, global and per hemisphere, read
against the run's own synoptic power law (self-relative) and against the
Lindborg (1999) aircraft reference (absolute, 250 hPa only).

    python tools/arwen_global_ke_stick.py <config.toml> <checkpoint.npz> \
        [more checkpoints ...] --out <plot.png> [--minimum-hours 12] \
        [--region northern] [--level-pa 25000]

The retired 10-row FFT band arithmetic this tool used to carry survives
only as ``dynamics_spectrum.band_fft_diagnostic`` with its audited
defects listed on it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from woof.verify.harness.dynamics_spectrum import (  # noqa: E402
    isotropic_wavelength_km,
    lindborg_reference_by_degree,
    measure_run,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-hours", type=float, default=12.0)
    parser.add_argument("--region", default="northern")
    parser.add_argument("--level-pa", type=float, default=25_000.0)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = measure_run(args.config, args.checkpoints, minimum_hours=args.minimum_hours)
    payload = report.to_payload()
    samples = [
        s for s in payload["measurements"]["samples"]
        if s["region"] == args.region and s["target_level_pa"] == args.level_pa
    ]
    if not samples:
        raise SystemExit(f"no samples for region {args.region!r} at {args.level_pa} Pa")
    truncation = len(samples[0]["spectrum_total"]) - 1
    radius_m = 6_371_220.0
    degrees = np.arange(1, truncation + 1)
    wavelength = isotropic_wavelength_km(degrees, radius_m)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    colors = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    for index, sample in enumerate(samples):
        color = colors[min(index, len(colors) - 1)]
        ax.loglog(
            wavelength, np.asarray(sample["spectrum_total"])[1:], color=color, lw=2,
            label=f"t = {sample['hours']:.0f} h",
        )
        reading = sample["self_relative"]
        if reading["status"] == "resolved":
            ax.axvline(reading["wavelength_km"], color=color, lw=1.0, ls=":")
    ax.loglog(
        wavelength, lindborg_reference_by_degree(truncation, radius_m)[1:],
        color="#52514e", lw=1.2, ls="--", label="Lindborg (1999) aircraft reference",
    )
    ax.axvline(
        payload["measurements"]["truncation_wavelength_km"], color="#eb6834",
        lw=1.0, ls="-.", label="truncation wavelength",
    )
    ax.invert_xaxis()
    ax.set_xlabel("isotropic wavelength 2*pi*a/sqrt(n(n+1)) [km]")
    ax.set_ylabel("KE per degree [m^2 s^-2]")
    mean = payload["measurements"]["self_relative_km_mean"]
    headline = (
        f"self-relative {mean:.0f} km ({payload['measurements']['self_relative_over_truncation']:.2f}x truncation)"
        if np.isfinite(mean) else
        f"self-relative not formed ({payload['measurements']['self_relative_resolved_fraction']:.0%} of samples resolved)"
    )
    ax.set_title(f"{payload['subject']}: {headline}")
    ax.legend(frameon=False)
    ax.grid(True, which="both", color="#e8e7e4", lw=0.6)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    args.out.with_suffix(".json").write_text(report.to_json() + "\n", encoding="utf-8")
    summary = {
        "subject": payload["subject"],
        "status": payload["status"],
        "gates": {g["name"]: (g["value"], g["passed"]) for g in payload["gates"]},
        "self_relative_km_mean": mean,
        "self_relative_over_truncation": payload["measurements"]["self_relative_over_truncation"],
        "absolute_km_mean": payload["measurements"]["absolute_km_mean"],
        "absolute_statuses": payload["measurements"]["absolute_statuses"],
        "fitted_synoptic_slope_mean": payload["measurements"]["fitted_synoptic_slope_mean"],
    }
    print(json.dumps(summary, indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
