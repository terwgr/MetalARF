#!/usr/bin/env python3
"""Compare complete Metal mp_physics=28 calls with their CUDA source.

``host`` runs the production adapter on the host build of the CUDA units
(tools/thompson_real_column_parity/host_backend.py) and saves every state
word after each call; ``metal`` runs the same calls on Metal and compares.
The start state is a prepared case's cached HRRR initial condition.  Run the
two modes as separate processes; run ``metal`` through the GPU queue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/thompson_real_column_parity"))

FIELDS = ("qv", "qc", "qi", "ni", "qs", "qg", "qr", "nr", "nc", "nwfa", "nifa",
          "thp", "h_diabatic", "effc", "effi", "effs")
SURFACE = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv", "sr")


class State:
    """The DomainState surface the mp=28 adapter reads."""

    def __init__(self, xp, arrays, metal):
        self._xp = xp
        self._scratch = {}
        self._metal_setup_state = metal
        self._array_module = xp
        self.elapsed_seconds = 0.0
        for name, value in arrays.items():
            setattr(self, name, xp.asarray(value))

    def scratch(self, shape, name):
        slot = self._scratch.get(name)
        if slot is None:
            slot = self._xp.zeros(shape, dtype=np.float32)
            self._scratch[name] = slot
        return slot

    def existing_scratch(self, name):
        return self._scratch.get(name)

    def total_theta(self):
        return self.thb + self.thp


def load_case(prepared, restart, columns, *, seed=True):
    cache = Path(prepared) / "native/prepared-cache"
    header = json.loads((cache / "header.json").read_text())
    saved = np.load(restart)
    j0, j1, i0, i1 = columns

    def cut(value):
        value = value[..., j0:j1, i0:i1] if value.ndim >= 2 else value
        return np.ascontiguousarray(value, np.float32)

    out = {name: cut(saved[f"state/{name}"]) for name in (
        "qv", "qc", "qi", "ni", "qs", "qg", "qr", "nr", "nc", "nwfa", "nifa",
        "thp", "h_diabatic", "effc", "effi", "effs", "p", "php", "w",
        "nwfa2d", "nifa2d")}
    for name in ("thb", "phb"):
        out[name] = cut(np.load(cache / header["arrays"][f"base/{name}"]["file"]))
    if seed:
        seed_condensate(out)
    return out


def seed_condensate(out):
    """Place cloud, rain, ice, snow and graupel where each can exist, and a
    supersaturated layer, so every source network and fallout runs."""
    rng = np.random.default_rng(2804)
    shape = out["qv"].shape
    thb = out["thb"] if out["thb"].ndim == 3 else out["thb"][:, None, None]
    t = (thb + out["thp"]) * (out["p"] / 1.0e5) ** (287.0 / 1004.0)
    tc = t - 273.15
    es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    qsat = 0.622 * es / (out["p"] - es)
    active = rng.random(shape[1:]) < 0.7
    def place(where, low, high):
        value = rng.uniform(low, high, shape) * (where & active)
        value[rng.random(shape) < 0.2] = 0.0
        return value.astype(np.float32)
    layer = (tc > -25) & (tc < 15)
    out["qv"] = np.where(layer & active, qsat * rng.uniform(0.97, 1.04, shape),
                         out["qv"]).astype(np.float32)
    out["qc"] = place((tc > -30) & (tc < 20), 1e-6, 1e-3)
    out["nc"] = (out["qc"] > 0) * rng.uniform(2e7, 4e8, shape).astype(np.float32)
    out["qr"] = place(tc > -10, 1e-6, 2e-3)
    out["nr"] = (out["qr"] > 0) * rng.uniform(1e2, 1e5, shape).astype(np.float32)
    out["qi"] = place(tc < 0, 1e-7, 2e-4)
    out["ni"] = (out["qi"] > 0) * rng.uniform(1e3, 1e6, shape).astype(np.float32)
    out["qs"] = place(tc < 5, 1e-6, 1.5e-3)
    out["qg"] = place(tc < 10, 1e-6, 3e-3)
    for name in ("nc", "nr", "ni"):
        out[name] = out[name].astype(np.float32)


def config(version):
    return SimpleNamespace(mp_physics=28, no_mp_heating=0, mp_tend_lim=10.0,
                           thompson_version=version,
                           thompson_cloud_surface_booking=False,
                           thompson_fork_snow_fall="blend")


TRACED = ("woof.core.thompson_aerosol_cold", "woof.core.thompson_aerosol_warm",
          "woof.core.thompson_aerosol_sat", "woof.core.thompson_aerosol_sed",
          "woof.core.thompson_aerosol_state", "woof.core.thompson")


def trace_stages(state, metal, sink, reference=None):
    """Snapshot every state field and scratch slot after each launcher.

    With ``reference`` (the host trace), the Metal state is reset to the host
    words after every stage, so each stage's differences are its own.
    """
    import importlib
    seen = {}
    host = (lambda v: v.to_numpy()) if metal else np.asarray
    slots = lambda: [(f"state/{k}", getattr(state, k)) for k in FIELDS] + [
        (f"scratch/{k}", v) for k, v in state._scratch.items()]
    def snapshot(label):
        if not metal and "launch_tau1_density" not in seen and label in (
                "launch_hydrometeor_column_mask", "launch_graupel_fallout_column_mask"):
            # The host's two entry masks are Metal's one adapter_masks launch.
            if label == "launch_hydrometeor_column_mask":
                return
            label = "launch_adapter_masks"
        seen[label] = seen.get(label, 0) + 1
        tag = f"{label}#{seen[label]}"
        sink.append((tag, {k: host(v).copy() for k, v in slots()}))
        if reference is not None:
            for key, array in slots():
                ref_key = f"{tag}|{key}"
                if ref_key in reference.files and reference[ref_key].shape == array.shape:
                    array[...] = reference[ref_key]
    for name in TRACED:
        module = importlib.import_module(name)
        for attr in [a for a in vars(module) if a.startswith("launch_")
                     and a != "launch_grid"]:
            original = getattr(module, attr)
            def wrapped(*a, __original=original, __label=attr, **k):
                result = __original(*a, **k)
                snapshot(__label)
                return result
            setattr(module, attr, wrapped)


def run(xp, arrays, version, calls, dt, metal, trace=None, reference=None):
    from woof.core.microphysics_aerosol import _apply_thompson_aerosol_call
    from woof.core.thompson_aerosol_launch import thompson_version_scope
    state = State(xp, arrays, metal)
    cfg = config(version)
    if trace is not None:
        trace_stages(state, metal, trace, reference)
        calls = 1
    frames = []
    with thompson_version_scope(version):
        for _ in range(calls):
            diagnostics = _apply_thompson_aerosol_call(
                state, cfg, dt, refl_10cm_due=False, version=version)
            frame = {f"state/{k}": np.asarray(getattr(state, k) if not metal
                     else getattr(state, k).to_numpy()) for k in FIELDS}
            for k in SURFACE:
                value = getattr(diagnostics, k)
                frame[f"surface/{k}"] = np.asarray(value if not metal else value.to_numpy())
            frames.append({k: v.copy() for k, v in frame.items()})
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("host", "metal"))
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--restart", required=True, type=Path)
    parser.add_argument("--version", default="wrf_39_noaa")
    parser.add_argument("--columns", default="0,48,0,60")
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--dt", type=float, default=20.0)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--resync", action="store_true")
    args = parser.parse_args()
    columns = tuple(int(v) for v in args.columns.split(","))
    if args.mode == "host":
        import host_backend
        host_backend.install()
        arrays = load_case(args.prepared, args.restart, columns)
        if args.trace:
            trace = []
            run(np, arrays, args.version, 1, args.dt, False, trace)
            np.savez(args.reference, **{f"{label}|{k}": v for label, f in trace
                                        for k, v in f.items()})
            print(json.dumps({"stages": [label for label, _ in trace]}))
            return 0
        frames = run(np, arrays, args.version, args.calls, args.dt, False)
        np.savez(args.reference, **{f"{n}:{k}": v for n, f in enumerate(frames)
                                    for k, v in f.items()})
        print(json.dumps({"saved": str(args.reference), "calls": len(frames)}))
        return 0
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    arrays = load_case(args.prepared, args.restart, columns)
    reference = np.load(args.reference)
    if args.trace:
        trace = []
        failure = None
        with MetalContext() as context:
            try:
                run(MetalArrayModule(context), arrays, args.version, 1, args.dt,
                    True, trace, reference if args.resync else None)
            except Exception as error:  # report the stages that did run
                failure = f"{type(error).__name__}: {error}"
        report = []
        for label, frame in trace:
            diffs = {}
            for key, actual in frame.items():
                ref_key = f"{label}|{key}"
                if ref_key not in reference.files:
                    continue
                expected = reference[ref_key]
                if expected.shape != actual.shape:
                    diffs[key] = "shape"
                    continue
                n = int(np.count_nonzero(actual.astype(np.float32).view(np.uint32)
                                         != expected.astype(np.float32).view(np.uint32)))
                if n:
                    diffs[key] = n
            report.append({"stage": label, "diffs": diffs})
            if diffs and args.out is None:
                break
        if failure:
            report.append({"stage": "stopped", "error": failure})
        text = json.dumps(report, indent=1)
        if args.out:
            args.out.write_text(text + "\n")
        print(text[-4000:])
        return 0
    with MetalContext() as context:
        frames = run(MetalArrayModule(context), arrays, args.version,
                     args.calls, args.dt, True)
    rows, total, bad = [], 0, 0
    for n, frame in enumerate(frames):
        for key, actual in frame.items():
            expected = reference[f"{n}:{key}"]
            a = actual.astype(np.float32).view(np.uint32).reshape(-1)
            e = expected.astype(np.float32).view(np.uint32).reshape(-1)
            diff = np.flatnonzero(a != e)
            total += a.size
            bad += diff.size
            if diff.size:
                af = a.view(np.float32)[diff].astype(np.float64)
                ef = e.view(np.float32)[diff].astype(np.float64)
                rows.append({"call": n, "field": key, "mismatches": int(diff.size),
                             "first_index": int(diff[0]),
                             "max_abs": float(np.nanmax(np.abs(af - ef))),
                             "max_rel": float(np.nanmax(np.abs(af - ef)
                                              / np.maximum(np.abs(ef), 1e-30)))})
    receipt = {"status": "PASS" if not bad else "FAIL", "version": args.version,
               "words": total, "mismatches": bad, "fields": rows}
    if args.out:
        args.out.write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
