#!/usr/bin/env python3
"""Replay selected peak-capture calls on native Metal three times."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--capture-root", type=Path, required=True)
parser.add_argument("--reference-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--calls", default="0,17,161,230,366,382")
parser.add_argument("--operators", default="sfclay,ysu")
options = parser.parse_args()
options.output.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from woof.core import sfclay, noah, ysu
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

root = options.capture_root
calls = set(map(int, options.calls.split(",")))
events = [e for e in json.loads((root / "events.json").read_text())
          if e["call"] in calls and e["operator"] in options.operators.split(",")]
tables = dict(np.load(root / "noah-tables.npz"))
parameters = SimpleNamespace(veg=tables["veg"], soil=tables["soil"],
                             gen=tables["gen"], lucats=tables["veg"].shape[0],
                             slcats=tables["soil"].shape[0])
rows = []
with MetalContext() as context:
    xp = MetalArrayModule(context)
    for event in events:
        before = dict(np.load(root / event["before"]))
        old = dict(np.load(root / event["after"]))
        reference = dict(np.load(options.reference_root /
                                f"{event['call']:05d}-{event['operator']}-host.npz"))
        runs = []
        for _ in range(3):
            inputs = {name: xp.asarray(value, copy=True)
                      for name, value in before.items()}
            if event["operator"] == "sfclay":
                result = sfclay.sfclay(**inputs, **event["scalars"])
                got = {name: getattr(result, name).to_numpy()
                       for name in sfclay.SFCLAY_OUTPUTS}
            elif event["operator"] == "ysu":
                result = ysu.launch_ysu(**inputs, **event["scalars"])
                got = {name: value.to_numpy() for name, value in result.items()}
            else:
                noah.launch_noah(inputs, parameters, dzs=tables["dzs"],
                                 **event["scalars"])
                got = {name: value.to_numpy() for name, value in inputs.items()}
            runs.append(got)
        for name, value in runs[0].items():
            expected = reference[name].astype(float)
            delta = value.astype(float) - expected
            old_delta = old[name].astype(float) - expected
            identical = all(np.array_equal(value, run[name]) for run in runs[1:])
            assert identical, (event["call"], event["operator"], name)
            rows.append(dict(call=event["call"], elapsed_seconds=event["elapsed_seconds"],
                             operator=event["operator"], field=name,
                             before_rmse=float(np.sqrt(np.mean(old_delta * old_delta))),
                             after_rmse=float(np.sqrt(np.mean(delta * delta))),
                             before_max_abs=float(np.abs(old_delta).max()),
                             after_max_abs=float(np.abs(delta).max()),
                             repeat_identical=identical))
        np.savez(options.output / f"{event['call']:05d}-{event['operator']}-metal.npz", **runs[0])
(options.output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps([r for r in rows if r["field"] in
                  ("hpbl", "kpbl", "br", "v10", "hfx", "dv", "tsk", "qfx")], indent=2))
