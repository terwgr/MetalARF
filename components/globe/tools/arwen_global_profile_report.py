"""Read WOOF global step profiles and lay their operators side by side.

``python tools/arwen_global_profile_report.py LABEL=profile.json [LABEL=profile.json ...]
[--depth N] [--png out.png] [--json out.json]``

Each profile is the ``profile.json`` a run wrote with ``--profile-steps``
(woof.globe.profile).  The report prints, per section path to
``--depth``, the mean device and host milliseconds per profiled step of
every profile, the steps on which a section ran, and a split of the
step mean into the steps that carried a radiation call and those that
did not (the radiation cadence is many steps, so a window's mean hides
it).  The PNG is an analysis chart (horizontal bars of device time per
top-level section, one bar group per profile), not a weather field.

What the numbers are: the profiler's own columns, unchanged.
``device_ms`` is stream time from a section's entry event to its exit
event and ``host_ms`` the host wall inside it (profile.py, WHAT IT
MEASURES); the calibration receipt of every profile is checked to have
passed before its numbers are printed, and a profile whose calibration
failed is refused by name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    profile = json.loads(Path(path).read_text(encoding="utf-8"))
    calibration = profile.get("calibration") or {}
    if not calibration.get("passed"):
        raise SystemExit(f"{path}: calibration did not pass; its readings are not cited")
    return profile


def section_rows(profile: dict) -> dict[str, dict]:
    return {row["section"]: row for row in profile["summary"]}


def per_step(profile: dict, section: str, key: str) -> list[float]:
    values = []
    for row in profile["steps"]:
        reading = row["sections"].get(section)
        values.append(float(reading[key]) if reading else 0.0)
    return values


def radiation_split(profile: dict) -> tuple[list[int], list[int]] | None:
    """Indices of profiled steps with and without a radiation call: a
    step whose rrtmgp sections read more than five times the window's
    median rrtmgp reading carried the call (every other step pays only
    the stored heating rates).  None when the profile carries no rrtmgp
    section (taken before the suite sections were attached): the split
    is then not measured, not guessed from the step total."""
    rad = [
        sum(
            float(reading["device_ms"])
            for path, reading in row["sections"].items()
            if path.endswith(".rrtmgp")
        )
        for row in profile["steps"]
    ]
    if not rad or max(rad) <= 0.0:
        return None
    ordered = sorted(rad)
    median = ordered[len(ordered) // 2]
    with_rad = [i for i, v in enumerate(rad) if v > 5.0 * median]
    without = [i for i in range(len(rad)) if i not in with_rad]
    return with_rad, without


def mean(values, indices) -> float:
    picked = [values[i] for i in indices]
    return sum(picked) / len(picked) if picked else 0.0


def report(profiles: dict[str, dict], depth: int) -> tuple[str, dict]:
    labels = list(profiles)
    paths: list[str] = []
    for profile in profiles.values():
        for row in profile["summary"]:
            if row["section"] not in paths and row["depth"] <= depth:
                paths.append(row["section"])
    # Keep the first profile's ordering (device time within depth), then
    # the paths only later profiles carry.
    lines = []
    header = "| section | " + " | ".join(
        f"{label} device ms | {label} host ms | {label} d2h calls" for label in labels
    ) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + 3 * len(labels)))
    table: dict[str, dict] = {}
    for path in paths:
        cells = []
        table[path] = {}
        for label in labels:
            rows = section_rows(profiles[label])
            row = rows.get(path)
            if row is None:
                cells.append("n/a | n/a | n/a")
                table[path][label] = None
                continue
            cells.append(
                f"{row['device_ms']:.1f} | {row['host_ms']:.1f} | {row['transfer_calls']:.0f}"
            )
            table[path][label] = {
                "device_ms": row["device_ms"], "host_ms": row["host_ms"],
                "transfer_calls": row["transfer_calls"], "transfer_mb": row["transfer_mb"],
            }
        indent = "&nbsp;&nbsp;" * path.count(".")
        lines.append(f"| {indent}{path.split('.')[-1]} | " + " | ".join(cells) + " |")
    split_lines = ["", "| profile | profiled steps | steps with a radiation call | step device ms, with radiation | without | step host ms, with | without |", "|---|---|---|---|---|---|---|"]
    splits = {}
    for label, profile in profiles.items():
        dev = per_step(profile, "step", "device_ms")
        host = per_step(profile, "step", "host_ms")
        split = radiation_split(profile)
        splits[label] = {
            "profiled_steps": len(profile["steps"]),
            "device_ms_mean": sum(dev) / max(1, len(dev)),
            "device_ms_min": min(dev) if dev else 0.0,
            "host_ms_mean": sum(host) / max(1, len(host)),
        }
        s = splits[label]
        if split is None:
            s["radiation_steps"] = None
            split_lines.append(
                f"| {label} | {s['profiled_steps']} | not measured (no rrtmgp section) | n/a | "
                f"{s['device_ms_mean']:.1f} (mean of all) | n/a | {s['host_ms_mean']:.1f} (mean of all) |"
            )
            continue
        with_rad, without = split
        s.update({
            "radiation_steps": len(with_rad),
            "device_ms_with_radiation": mean(dev, with_rad),
            "device_ms_without_radiation": mean(dev, without),
            "host_ms_with_radiation": mean(host, with_rad),
            "host_ms_without_radiation": mean(host, without),
        })
        split_lines.append(
            f"| {label} | {s['profiled_steps']} | {s['radiation_steps']} | "
            f"{s['device_ms_with_radiation']:.1f} | {s['device_ms_without_radiation']:.1f} | "
            f"{s['host_ms_with_radiation']:.1f} | {s['host_ms_without_radiation']:.1f} |"
        )
    cal_lines = ["", "| profile | backend | calibration | device fma read / reference ms | device reduce read / reference ms | sleep read ms | device memory free / total GiB |", "|---|---|---|---|---|---|---|"]
    for label, profile in profiles.items():
        c = profile["calibration"]
        checks = c["checks"]
        memory = c.get("device_memory") or {}
        free = memory.get("free_bytes")
        total = memory.get("total_bytes")
        mem = f"{free / 2**30:.1f} / {total / 2**30:.1f}" if free is not None else "n/a"
        cal_lines.append(
            f"| {label} | {profile['backend']} | {'passed' if c['passed'] else 'FAILED'} | "
            f"{checks['device_fma']['read_ms']:.2f} / {checks['device_fma']['reference_ms']:.2f} | "
            f"{checks['device_reduce']['read_ms']:.2f} / {checks['device_reduce']['reference_ms']:.2f} | "
            f"{checks['sleep_device_ms']['read_ms']:.1f} | {mem} |"
        )
    return "\n".join(lines + split_lines + cal_lines), {"sections": table, "steps": splits}


def section_minima(profile: dict) -> dict[str, float]:
    """Each section's smallest reading over the profiled steps: on a
    shared card the least-contended step is the reading that survives
    the sharing (the mean carries the co-tenant's phases)."""
    out: dict[str, float] = {}
    for row in profile["steps"]:
        for path, reading in row["sections"].items():
            value = float(reading["device_ms"])
            if path not in out or value < out[path]:
                out[path] = value
    return out


def chart(profiles: dict[str, dict], png: str, depth: int = 1, *, minima: bool = False, exclude: tuple[str, ...] = ()) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(profiles)
    first = profiles[labels[0]]
    paths = [
        row["section"] for row in first["summary"]
        if row["depth"] == depth and row["section"].startswith("step.")
    ]
    for profile in profiles.values():
        for row in profile["summary"]:
            if row["depth"] == depth and row["section"].startswith("step.") and row["section"] not in paths:
                paths.append(row["section"])
    paths = [p for p in paths if p.split(".")[-1] not in exclude]
    height = 0.8 / max(1, len(labels))
    fig, ax = plt.subplots(figsize=(11, 0.45 * len(paths) + 2.5))
    for k, label in enumerate(labels):
        if minima:
            rows = section_minima(profiles[label])
            values = [rows.get(p, 0.0) for p in paths]
        else:
            rows = section_rows(profiles[label])
            values = [rows[p]["device_ms"] if p in rows else 0.0 for p in paths]
        y = [i + (k - (len(labels) - 1) / 2.0) * height for i in range(len(paths))]
        bars = ax.barh(y, values, height=height, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:.0f}", va="center", fontsize=7)
    ax.set_yticks(range(len(paths)))
    ax.set_yticklabels([p.split(".", 1)[1] for p in paths])
    ax.invert_yaxis()
    if minima:
        ax.set_xlabel("device ms per step: each section's smallest reading over the profiled steps (the least-contended step on a shared card)")
        ax.set_title("WOOF global: one step by operator, least-contended readings")
    else:
        ax.set_xlabel("device ms per step (stream time from entry event to exit event, mean over the profiled window)")
        ax.set_title("WOOF global: one step by operator")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=130)


def steps_chart(profiles: dict[str, dict], png: str) -> None:
    """Every profiled step's device time, one line per profile: the
    minimum over the window is the least-contended reading on a shared
    card and the spikes are the radiation calls and checkpoints."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for label, profile in profiles.items():
        steps = [int(row["step"]) for row in profile["steps"]]
        values = per_step(profile, "step", "device_ms")
        low = min(values)
        ax.plot(steps, values, marker=".", label=f"{label}: min {low:.0f} ms, mean {sum(values) / len(values):.0f} ms")
    ax.set_xlabel("model step")
    ax.set_ylabel("step device ms")
    ax.set_title("WOOF global: device time per step over the profiled window")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png, dpi=130)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profiles", nargs="+", help="LABEL=path/to/profile.json")
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--png", default=None)
    parser.add_argument("--steps-png", default=None)
    parser.add_argument("--minima-png", default=None, help="the per-section minima chart")
    parser.add_argument("--exclude", default="", help="comma-separated section names left off the charts (e.g. diagnostics,checkpoint: the runner's per-output-interval work, not a step operator)")
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    profiles = {}
    for item in args.profiles:
        label, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"{item!r}: expected LABEL=path")
        profiles[label] = load(path)
    text, data = report(profiles, args.depth)
    print(text)
    exclude = tuple(name for name in args.exclude.split(",") if name)
    if args.png:
        chart(profiles, args.png, exclude=exclude)
    if args.steps_png:
        steps_chart(profiles, args.steps_png)
    if args.minima_png:
        chart(profiles, args.minima_png, minima=True, exclude=exclude)
    if args.json:
        Path(args.json).write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
