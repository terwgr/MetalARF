#!/usr/bin/env python3
"""Replay rain_peak_capture inputs through unchanged CUDA source on the CPU.

Uses the existing host shim with contraction disabled and host libm. This
isolates translation/driver defects; it does not execute or emulate NVIDIA
CUDA arithmetic. Input columns are independent in these three operators.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import csv
import json
import re
import sys
import time

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--capture-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--wait", action="store_true")
parser.add_argument("--max-calls", type=int, default=0)
parser.add_argument("--operators", default="sfclay,noah,ysu")
parser.add_argument("--contract", choices=("off", "fast"), default="off",
                    help="CPU contraction counterfactual; fast is not NVIDIA emulation")
options = parser.parse_args()
options.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "thompson_real_column_parity"))
import host_backend
if options.contract == "fast":
    host_backend.CXX_FLAGS = tuple("-ffp-contract=fast" if flag == "-ffp-contract=off"
                                  else flag for flag in host_backend.CXX_FLAGS)
host_backend.install()
# CUDA occupancy hints have no CPU meaning and the generic host parser
# expects __global__ immediately before void. Remove only that annotation.
def noah_host_declarations(text):
    text = re.sub(r"__launch_bounds__\([^)]*\)", "", text)
    if options.contract == "fast":
        # CUDA explicit RN intrinsics prohibit contraction across their
        # result. Preserve those boundaries in the CPU counterfactual too.
        functions = []
        for name, operator in (("__fadd_rn", "+"), ("__fsub_rn", "-"),
                               ("__fmul_rn", "*"), ("__fdiv_rn", "/")):
            replacement = "host_separate" + name
            text = re.sub(r"\b" + name + r"\b", replacement, text)
            functions.append(f"static __attribute__((noinline)) float {replacement}"
                             f"(float a, float b) {{ return a {operator} b; }}")
        text = "\n".join(functions) + "\n" + text
    return text


host_backend.set_transform("noah", noah_host_declarations)
from woof.core import sfclay, noah, ysu
ysu.ysu_tile_columns = lambda _function, ncol: ncol

root = options.capture_root
table_arrays = dict(np.load(root / "noah-tables.npz"))
parameters = SimpleNamespace(
    veg=table_arrays["veg"], soil=table_arrays["soil"], gen=table_arrays["gen"],
    lucats=table_arrays["veg"].shape[0], slcats=table_arrays["soil"].shape[0])
rows = []
done = set()
deadline = time.monotonic() + 1800
while True:
    try:
        events = json.loads((root / "events.json").read_text())
    except json.JSONDecodeError:
        time.sleep(0.2)
        continue
    for event in events:
        key = (event["call"], event["operator"])
        if (key in done or event["operator"] not in options.operators.split(",")
                or (options.max_calls and event["call"] >= options.max_calls)):
            continue
        before = {name: value.copy() for name, value in np.load(root / event["before"]).items()}
        actual = dict(np.load(root / event["after"]))
        if event["operator"] == "sfclay":
            result = sfclay.sfclay(**before, **event["scalars"])
            expected = {name: getattr(result, name) for name in sfclay.SFCLAY_OUTPUTS}
        elif event["operator"] == "noah":
            noah.launch_noah(before, parameters, dzs=table_arrays["dzs"],
                             **event["scalars"])
            expected = before
        elif event["operator"] == "ysu":
            expected = ysu.launch_ysu(**before, **event["scalars"])
        else:
            raise ValueError(event["operator"])
        name = f"{event['call']:05d}-{event['operator']}-host.npz"
        np.savez(options.output / name, **expected)
        for field, want in expected.items():
            got = actual[field]
            delta = got.astype(float) - want.astype(float)
            index = np.unravel_index(np.abs(delta).argmax(), delta.shape)
            rows.append(dict(call=event["call"], elapsed_seconds=event["elapsed_seconds"],
                             operator=event["operator"], field=field,
                             rmse=float(np.sqrt(np.mean(delta * delta))),
                             max_abs=float(np.abs(delta).max()),
                             reference_max=float(np.abs(want).max()),
                             differing_words=int(np.count_nonzero(got.view(np.uint32) != want.view(np.uint32))),
                             point=int(index[-1]), k=int(index[0]) if delta.ndim == 3 else -1,
                             metal=float(got[index]), host=float(want[index])))
        done.add(key)
        if len(done) % 150 == 0:
            print(f"Replayed {len(done)} calls through {event['elapsed_seconds']:g} seconds", flush=True)
    with (options.output / "comparison.csv").open("w") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if (not options.wait or (root / "capture.json").exists()
            or (options.max_calls and any(e["call"] >= options.max_calls - 1 and e["operator"] == "ysu" for e in events))):
        break
    if time.monotonic() > deadline:
        raise TimeoutError("Waiting for complete native peak capture")
    time.sleep(2)
summary = dict(events=len(done), rows=len(rows),
               oracle=f"Unchanged CUDA source compiled on CPU, contraction={options.contract}, host libm; not NVIDIA execution",
               worst={operator: sorted((r for r in rows if r["operator"] == operator),
                                       key=lambda r: r["max_abs"], reverse=True)[:8]
                      for operator in ("sfclay", "noah", "ysu")})
(options.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
