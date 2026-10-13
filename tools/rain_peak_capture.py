#!/usr/bin/env python3
"""Capture selected columns at every surface/PBL call through a rainy peak.

Use --capture-root DIR --points 'i,j;i,j' --stop-seconds N followed by
ordinary woof go arguments. This observes native execution without replacing
any operator or changing its inputs. It stops after the last captured YSU call.
"""
from pathlib import Path
import argparse
import inspect
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--capture-root", type=Path, required=True)
parser.add_argument("--points", required=True)
parser.add_argument("--stop-seconds", type=float, required=True)
options, cli = parser.parse_known_args()
options.capture_root.mkdir(parents=True, exist_ok=True)
points = [tuple(map(int, pair.split(",")))
          for pair in options.points.split(";")]
ii, jj = np.array(points).T
from woof.core import physics, sfclay, noah, ysu

elapsed = 0.0
call = -1
events = []


class Complete(BaseException):
    pass


def host(value):
    return value.to_numpy() if hasattr(value, "to_numpy") else np.asarray(value)


def columns(value):
    array = host(value)
    selected = array[..., jj, ii]
    return selected.reshape((*selected.shape[:-1], 1, len(points))).copy()


def save(operator, side, arrays):
    name = f"{call:05d}-{operator}-{side}.npz"
    np.savez(options.capture_root / name, **arrays)
    return name


def record(operator, before, after, scalars):
    events.append(dict(call=call, elapsed_seconds=elapsed, operator=operator,
                       before=before, after=after, scalars=scalars))
    (options.capture_root / "events.json").write_text(
        json.dumps(events, indent=2) + "\n")


def replace(original, replacement):
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "__name__", "").startswith("woof."):
            for name, value in list(vars(module).items()):
                if value is original:
                    setattr(module, name, replacement)


original_sf = sfclay.launch_sfclay
signature_sf = inspect.signature(original_sf)


def capture_sf(*args, **kwargs):
    bound = signature_sf.bind(*args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    result = values["result"]
    inputs = {name: columns(values[name]) for name in
              ("u", "v", "t", "qv", "p", "dz8w", "psfc", "tsk",
               "pblh", "mavail", "xland", "lakemask")}
    inputs.update({name: columns(getattr(result, name)) for name in
                   ("znt", "ust", "ustm", "mol", "hfx", "qfx", "qsfc", "zol")})
    before = save("sfclay", "before", inputs)
    original_sf(*args, **kwargs)
    after = save("sfclay", "after", {name: columns(getattr(result, name))
                                     for name in sfclay.SFCLAY_OUTPUTS})
    record("sfclay", before, after, {name: values[name] for name in
                                    ("option", "dx", "isfflx", "isftcflx", "iz0tlnd")})


replace(original_sf, capture_sf)
original_no = noah.launch_noah
signature_no = inspect.signature(original_no)


def capture_no(*args, **kwargs):
    bound = signature_no.bind(*args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    fields = values["dev"]
    names = noah._F2D + noah._F3D + ("ivgtyp", "isltyp", "ebal")
    before = save("noah", "before", {name: columns(fields[name]) for name in names})
    parameters = values["params"]
    if call == 0:
        np.savez(options.capture_root / "noah-tables.npz",
                 veg=host(parameters.veg), soil=host(parameters.soil),
                 gen=host(parameters.gen), dzs=np.asarray(values["dzs"]))
    original_no(*args, **kwargs)
    after = save("noah", "after", {name: columns(fields[name]) for name in names})
    record("noah", before, after, {name: values[name] for name in
                                   ("dt", "isurban", "isice", "xice_threshold",
                                    "frpcpn", "usemonalb", "rdlai2d", "opt_thcnd",
                                    "itimestep")})


replace(original_no, capture_no)
original_ys = ysu.launch_ysu
signature_ys = inspect.signature(original_ys)


def capture_ys(*args, **kwargs):
    bound = signature_ys.bind(*args, **kwargs)
    bound.apply_defaults()
    values = bound.arguments
    names = ("u", "v", "theta", "qv", "qc", "qi", "p", "p_interface",
             "exner", "dz", "rthraten", "psfc", "znt", "ust", "hfx", "qfx",
             "wspd", "br", "psim", "psih", "xland", "u10", "v10")
    before = save("ysu", "before", {name: columns(values[name]) for name in names})
    result = original_ys(*args, **kwargs)
    after = save("ysu", "after", {name: columns(value)
                                  for name, value in result.items()})
    record("ysu", before, after, {name: values[name] for name in
                                  ("dt", "ysu_topdown_pblmix")})
    if call % 50 == 0:
        print(f"Captured surface call {call} at {elapsed:g} seconds", flush=True)
    if elapsed >= options.stop_seconds:
        (options.capture_root / "capture.json").write_text(json.dumps(dict(
            status="COMPLETE", elapsed_seconds=elapsed, calls=call + 1,
            points=points, execution="synchronous diagnostic readbacks only",
            source_inputs_unmodified=True), indent=2) + "\n")
        raise Complete()
    return result


replace(original_ys, capture_ys)
original_timing = physics.PhysicsDriver._run_sfclay


def timing(self, *args, **kwargs):
    global elapsed, call
    elapsed = float(self.state.elapsed_seconds)
    call += 1
    return original_timing(self, *args, **kwargs)


physics.PhysicsDriver._run_sfclay = timing
from woof.cli import main

try:
    main(cli)
except Complete:
    raise SystemExit(0)
if (options.capture_root / "capture.json").exists():
    raise SystemExit(0)
raise RuntimeError("Forecast ended before the requested peak capture")
