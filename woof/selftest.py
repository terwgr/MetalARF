"""``woof selftest``: one tiny step of each model through its public path.

A contributor's first question after an install or a change is "does the
GPU route still run?".  This answers it in well under a minute: a regional
physics row (the same row the combination smoke runs), the 12-cell Hex dry
step and the shipped Global moist smoke, each printed as PASS, FAIL or SKIP
with the fields a reader would check by hand -- the largest wind and the
potential-temperature (regional, Hex) or temperature (Global) range.
"""
from __future__ import annotations

import json
from pathlib import Path
import signal
import tempfile
import time

#: The regional suite row the self-test runs (see woof.metal.suites).
REGIONAL_ROW = "Convection-permitting no-CU"
#: The shipped Global experiment the self-test runs to completion.
GLOBAL_CONFIG = "arwen_global_moist_smoke"
#: Wall-clock bound for the whole self-test, in seconds.
DEADLINE_S = 240


class SelftestDeadline(Exception):
    pass


def metal_available() -> bool:
    try:
        from woof.metal.runtime import MetalContext

        with MetalContext():
            return True
    except Exception:
        return False


def resolve_backend(requested: str) -> str:
    if requested == "auto":
        return "metal" if metal_available() else "numpy"
    if requested == "metal" and not metal_available():
        raise RuntimeError("no Metal device is available; "
                           "use --backend numpy or --backend auto")
    return requested


def regional_check(backend: str) -> dict:
    if backend != "metal":
        return {"status": "SKIP",
                "detail": "the regional engine runs on a GPU backend only"}
    from woof.metal.suites import execute, suite_rows

    _label, combo, options = next(
        row for row in suite_rows() if row[0] == REGIONAL_ROW)
    fields: dict = {}
    status, detail = execute(combo, steps=1, fields=fields, **options)
    return {"status": "PASS" if status == "ok" else "FAIL",
            "detail": f"{REGIONAL_ROW}, 1 step" if status == "ok" else
            f"{status}: {detail}", **fields}


#: Largest relative Metal-vs-NumPy difference the Hex check accepts.
HEX_NUMPY_TOLERANCE = 1e-4


def hex_check(backend: str) -> dict:
    import numpy as np

    from woof.hex.metal_dry_door import tiny_dry_steps

    mesh, state, _ = tiny_dry_steps(1)
    agreement = {}
    if backend == "metal":
        from woof.metal.runtime import MetalContext

        reference = state
        with MetalContext() as context:
            mesh, state, _ = tiny_dry_steps(1, metal_context=context)
        # The Metal step must reproduce the NumPy step, not just stay finite.
        worst = 0.0
        for name in ("rho", "rho_theta", "rho_u", "rho_w", "scalars"):
            metal = np.asarray(getattr(state, name), dtype=np.float64)
            numpy_ = np.asarray(getattr(reference, name), dtype=np.float64)
            scale = max(float(np.abs(numpy_).max()), 1e-30)
            worst = max(worst, float(np.abs(metal - numpy_).max()) / scale)
        agreement = {"numpy_max_rel_diff": worst}
        if worst > HEX_NUMPY_TOLERANCE:
            return {"status": "FAIL",
                    "detail": f"12-cell dry mesh: Metal differs from NumPy by "
                              f"{worst:.2e} (limit {HEX_NUMPY_TOLERANCE:.0e})",
                    **agreement}
    rho = np.asarray(state.rho, dtype=np.float64)
    cells = np.asarray(mesh.arrays["cellsOnEdge"])
    rho_edge = 0.5 * (rho[:, cells[:, 0]] + rho[:, cells[:, 1]])
    theta = np.asarray(state.rho_theta, dtype=np.float64) / rho
    return {"status": "PASS", "detail": "12-cell dry mesh, 1 step",
            "max_wind_m_s": float(np.abs(
                np.asarray(state.rho_u, dtype=np.float64) / rho_edge).max()),
            "theta_min_k": float(theta.min()),
            "theta_max_k": float(theta.max()),
            **agreement}


def global_check(backend: str) -> dict:
    from woof.globe.config import load_config
    from woof.globe.configs_dir import resolve_config
    from woof.globe.runner import RECEIPT_NAME, run

    cfg = load_config(resolve_config(GLOBAL_CONFIG))
    with tempfile.TemporaryDirectory(prefix="woof-selftest-") as outdir:
        result = run(cfg, outdir, metal=backend == "metal")
        receipt = json.loads((Path(outdir) / RECEIPT_NAME).read_text(
            encoding="utf-8"))
    failed = sorted(name for name, gate in receipt["gates"].items()
                    if not gate.get("passed", False))
    final = receipt["final_diagnostics"]
    passed = result["status"] == "pass" and not failed
    return {"status": "PASS" if passed else "FAIL",
            "detail": (f"{GLOBAL_CONFIG}, {len(receipt['gates'])} gates"
                       if passed else "failed gates: "
                       + (", ".join(failed) or result["status"])),
            "max_wind_m_s": float(final["maximum_wind_m_s"]),
            "temperature_min_k": float(final["minimum_temperature_k"]),
            "temperature_max_k": float(final["maximum_temperature_k"])}


CHECKS = (("regional", regional_check), ("hex", hex_check),
          ("global", global_check))


def _line(name: str, row: dict) -> str:
    parts = [f"{row['status']:<4}", f"{name:<8}", row["detail"]]
    if "max_wind_m_s" in row:
        parts.append(f"max wind {row['max_wind_m_s']:.3f} m/s")
    for label, key in (("theta", "theta"), ("T", "temperature")):
        if f"{key}_min_k" in row:
            parts.append(f"{label} {row[f'{key}_min_k']:.2f}"
                         f"..{row[f'{key}_max_k']:.2f} K")
    parts.append(f"{row['seconds']:.1f} s")
    return "  ".join(parts)


def run_selftest(backend: str = "auto", *, as_json: bool = False) -> int:
    backend = resolve_backend(backend)
    if not as_json:
        print(f"woof selftest: backend {backend}")

    def deadline(_signum, _frame):
        raise SelftestDeadline(f"exceeded {DEADLINE_S} s")

    previous = signal.signal(signal.SIGALRM, deadline)
    signal.alarm(DEADLINE_S)
    rows = {}
    try:
        for name, check in CHECKS:
            started = time.monotonic()
            try:
                row = check(backend)
            except SelftestDeadline:
                raise
            except Exception as error:
                row = {"status": "FAIL",
                       "detail": f"{type(error).__name__}: {error}"[:300]}
            row["seconds"] = time.monotonic() - started
            rows[name] = row
            if not as_json:
                print(_line(name, row), flush=True)
    except SelftestDeadline as error:
        rows[name] = {"status": "FAIL", "detail": str(error),
                      "seconds": time.monotonic() - started}
        if not as_json:
            print(_line(name, rows[name]), flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
    ok = all(row["status"] != "FAIL" for row in rows.values())
    if as_json:
        print(json.dumps({"backend": backend, "passed": ok, "checks": rows},
                         indent=2, sort_keys=True))
    else:
        print("woof selftest: " + ("passed" if ok else "FAILED"))
    return 0 if ok else 1


def selftest_main(args) -> int:
    return run_selftest(args.backend, as_json=args.json)


def register_cli(subparsers) -> None:
    parser = subparsers.add_parser(
        "selftest",
        help="run one tiny regional, Hex and Global step and report pass/fail",
        description=(
            "Run one tiny step of each model through its public path and "
            "print PASS/FAIL/SKIP with the largest wind and the theta or "
            "temperature range. Exits nonzero when any check fails."))
    parser.add_argument(
        "--backend", choices=("auto", "metal", "numpy"), default="auto",
        help="auto picks Metal when a Metal device opens, else NumPy "
             "(the regional check needs a GPU backend and is skipped on NumPy)")
    parser.add_argument("--json", action="store_true",
                        help="print the results as one JSON document")
    parser.set_defaults(func=selftest_main)
