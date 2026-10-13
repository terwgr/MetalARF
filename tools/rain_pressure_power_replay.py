#!/usr/bin/env python3
"""Check FP32 pressure powers against a rounded mathematical reference.

Uses pressures from dump_backend_step1's first physics input. The reference
is FP32 normalization, binary64 NumPy power, then FP32 rounding. It measures
mathematical accuracy, not actual NVIDIA powf bits. All computations under
test run on Metal; the temporary inverse/forward probe uses the production
helper and production safe/precise compiler flags.
"""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--capture-root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--sample-stride", type=int, default=10)
options = parser.parse_args()
if options.sample_stride < 1:
    parser.error("--sample-stride must be positive")
options.output.mkdir(parents=True, exist_ok=True)
source = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(source))
from woof.core import constants
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
from woof.metal.thompson_family import submit_exner_power

root = options.capture_root
events = (json.loads(line) for line in (root / "events.jsonl").read_text().splitlines())
event = next(e for e in events if e["point"] == "physics_tendencies_complete"
             and e["side"] == "before")
pressure = np.load(root / event["arrays"]["state.p"]["blob"])
p0, exponent = np.float32(constants.P0), np.float32(constants.RCP)
normalized = pressure / p0
reference = np.power(normalized.astype(float), float(exponent)).astype(np.float32)
rows = []


def metric(label, got, expected, repeat):
    delta = got.astype(float) - expected.astype(float)
    ulp = got.view(np.int32).astype(np.int64) - expected.view(np.int32).astype(np.int64)
    rows.append(dict(mode=label, repeat=repeat, count=got.size,
                     differing_words=int(np.count_nonzero(ulp)),
                     max_ulp=int(np.abs(ulp).max()),
                     bias=float(delta.mean()), max_abs=float(np.abs(delta).max())))


with MetalContext() as context:
    xp = MetalArrayModule(context)
    inputs = xp.asarray(pressure)
    output = xp.empty(pressure.shape, np.float32)
    for repeat in range(3):
        submit_exner_power(inputs, output, reference_pressure=p0, exponent=exponent)
        got = output.to_numpy()
        metric("production_exner_full", got, reference, repeat)
        if repeat == 0:
            first = got.copy()
        else:
            assert np.array_equal(first, got), "native Exner repeat differs"

probe = r'''
#include <metal_stdlib>
#include "thompson_exner_power_bindings.metalh"
#include "pressure_power.metalh"
using namespace metal;
#pragma clang fp contract(off)
kernel void thompson_exner_power(
    constant ThompsonExnerPowerResources &r [[buffer(0)]],
    constant ThompsonExnerPowerParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if(idx >= uint(p.size)) return;
    const uint mode = idx / (uint(p.size) / 4u);
    const float pressure = r.pressure[idx];
    if(mode == 0u) r.exner[idx] = pow(pressure / p.reference_pressure, p.exponent);
    else if(mode == 1u) r.exner[idx] = fp32_pressure_power(pressure, p.reference_pressure, p.exponent);
    else if(mode == 2u) r.exner[idx] = pow(p.reference_pressure / pressure, p.exponent);
    else r.exner[idx] = fp32_pressure_power(p.reference_pressure, pressure, p.exponent);
}
// Required context pipeline; never launched by this bounded diagnostic.
kernel void calc_coefs() {}
'''
sample = np.concatenate((pressure[0].ravel(), pressure.ravel()[::options.sample_stride]))
forward = np.power((sample / p0).astype(float), float(exponent)).astype(np.float32)
inverse = np.power((p0 / sample).astype(float), float(exponent)).astype(np.float32)
labels = ("native_forward", "production_forward", "native_inverse", "production_inverse")
references = (forward, forward, inverse, inverse)
with tempfile.TemporaryDirectory(prefix="woof-pressure-power-") as directory:
    directory = Path(directory)
    shader = directory / "probe.metal"
    shader.write_text(probe)
    common = ("xcrun", "-sdk", "macosx", "metal", "-c", "-std=macos-metal2.4",
              "-target", "air64-apple-macos14.0", "-fmetal-math-mode=safe",
              "-fmetal-math-fp32-functions=precise", "-ffp-contract=off")
    subprocess.run((*common, "-I", str(source / "woof/metal/generated"),
                    "-I", str(source / "woof/metal/shaders"), str(shader),
                    "-o", str(directory / "probe.air")), check=True)
    subprocess.run((*common, str(source / "woof/metal/shaders/rk_bookkeeping.metal"),
                    "-o", str(directory / "rk.air")), check=True)
    library = directory / "probe.metallib"
    subprocess.run(("xcrun", "-sdk", "macosx", "metallib", str(directory / "probe.air"),
                    str(directory / "rk.air"), "-o", str(library)), check=True)
    host_inputs = np.tile(sample, 4).reshape(1, 4, -1)
    with MetalContext(metallib=library) as context:
        xp = MetalArrayModule(context)
        inputs = xp.asarray(host_inputs)
        output = xp.empty(host_inputs.shape, np.float32)
        for repeat in range(3):
            submit_exner_power(inputs, output, reference_pressure=p0, exponent=exponent)
            got = output.to_numpy().reshape(4, -1)
            for label, actual, expected in zip(labels, got, references):
                metric(label, actual, expected, repeat)
            if repeat == 0:
                first = got.copy()
            else:
                assert np.array_equal(first, got), "native pressure-power repeat differs"

result = dict(reference="Rounded mathematical reference; not NVIDIA execution",
              source_normalization="FP32 division before power; FP32 output",
              metrics=rows)
(options.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
