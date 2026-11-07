#!/usr/bin/env python3
"""Compare the Metal WRF 3.9 fork fallout passes with their CUDA source.

The reference is thompson_aerosol_sed.cu under THOMPSON_AA_WRF39, built for
the host by tools/thompson_real_column_parity/host_backend.py.  Metal work
runs through the production launchers; run it through the GPU queue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/thompson_real_column_parity"))

NY, NX = 3, 4


def case(nz, seed):
    rng = np.random.default_rng(seed)
    shape = (nz, NY, NX)
    level = np.linspace(0.0, 1.0, nz, dtype=np.float32)[:, None, None]

    def uniform(low, high, sparse=0.0):
        value = rng.uniform(low, high, shape).astype(np.float32)
        if sparse:
            value[rng.random(shape) < sparse] = 0.0
        return value

    temperature = (300.0 - 75.0 * level + uniform(-3, 3)).astype(np.float32)
    pressure = (100000.0 - 80000.0 * level + uniform(-200, 200)).astype(np.float32)
    density = uniform(0.4, 1.25)
    carried = density.copy()
    carried[rng.random(shape) < 0.25] = 0.0
    flip = rng.random(shape) < 0.2
    carried[flip] = -carried[flip]
    surface = lambda: rng.uniform(0, 2, (NY, NX)).astype(np.float32)
    fields = {
        "temperature": temperature, "pressure": pressure,
        "qv": uniform(1e-5, 0.015), "dz": uniform(40, 600),
        "qr": uniform(0, 2e-3, 0.3), "nr": uniform(1e2, 1e5),
        "qi": uniform(0, 2e-4, 0.4), "ni": uniform(1e2, 1e6),
        "qs": uniform(0, 2e-3, 0.3), "qg": uniform(0, 3e-3, 0.4),
        "graupel_intercept": uniform(1e4, 5e6),
        "reference_density": density, "melt_rain_density": carried,
        "reference_temperature": (temperature + uniform(-0.5, 0.5)).astype(np.float32),
        "snow_melt_marker": (rng.random(shape) < 0.3).astype(np.float32),
        "velocity_boost": np.where(rng.random(shape) < 0.3, 1.5, 1.0).astype(np.float32),
        "rainnc": surface(), "rainncv": surface(), "snownc": surface(),
        "snowncv": surface(), "graupelnc": surface(), "graupelncv": surface(),
        "active_columns": (rng.random((NY, NX)) < 0.8).astype(np.float32),
    }
    fields["qr"][:, 0, 0] = 0.0  # one rain-free column
    fields["melt_rain_qr"] = fields["qr"].copy()
    fields["melt_rain_nr"] = fields["nr"].copy()
    return fields


PASSES = {
    "ice": (("qi", "ni", "temperature", "pressure", "qv", "reference_density",
             "dz", "rainnc", "rainncv", "snownc", "snowncv"),
            ("qi", "ni", "rainnc", "rainncv", "snownc", "snowncv")),
    "graupel": (("qg", "graupel_intercept", "temperature", "pressure", "qv",
                 "reference_density", "melt_rain_qr", "melt_rain_nr",
                 "melt_rain_density", "dz", "rainnc", "rainncv", "graupelnc",
                 "graupelncv", "active_columns"),
                ("qg", "rainnc", "rainncv", "graupelnc", "graupelncv")),
    "snow": (("qs", "snow_melt_marker", "melt_rain_qr", "melt_rain_nr",
              "temperature", "pressure", "qv", "reference_density",
              "reference_temperature", "velocity_boost", "melt_rain_density",
              "dz", "rainnc", "rainncv", "snownc", "snowncv"),
             ("qs", "rainnc", "rainncv", "snownc", "snowncv")),
    "rain": (("qr", "nr", "temperature", "pressure", "qv", "reference_density",
              "dz", "rainnc", "rainncv"),
             ("qr", "nr", "rainnc", "rainncv")),
}


def host_run(module, name, fields, nz, dt, singular):
    arrays = {k: v.copy() for k, v in fields.items()}
    inputs, _ = PASSES[name]
    args = [arrays[k] for k in inputs]
    if name == "snow":
        args.append(np.int32(singular))
    args += [np.float32(dt), np.int32(nz), np.int32(NY), np.int32(NX)]
    kernel = f"thompson_aa_wrf39_{name}_sediment_{'64' if nz <= 64 else '256'}"
    module.get_function(kernel)(((NY * NX + 31) // 32,), (32,), tuple(args))
    return arrays


def metal_run(xp, name, fields, dt, singular):
    from woof.core import thompson_aerosol_sed as sed
    d = {k: xp.asarray(v) for k, v in fields.items()}
    if name == "ice":
        sed.launch_wrf39_ice_sedimentation(
            d["qi"], d["ni"], d["temperature"], d["pressure"], d["qv"], d["dz"],
            d["rainnc"], d["rainncv"], d["snownc"], d["snowncv"], dt,
            reference_density=d["reference_density"])
    elif name == "graupel":
        sed.launch_wrf39_graupel_sedimentation(
            d["qg"], d["graupel_intercept"], d["temperature"], d["pressure"],
            d["qv"], d["dz"], d["rainnc"], d["rainncv"], d["graupelnc"],
            d["graupelncv"], dt, reference_density=d["reference_density"],
            melt_rain_qr=d["melt_rain_qr"], melt_rain_nr=d["melt_rain_nr"],
            melt_rain_density=d["melt_rain_density"],
            active_columns=d["active_columns"])
    elif name == "snow":
        sed.launch_wrf39_snow_sedimentation(
            d["qs"], d["temperature"], d["pressure"], d["qv"], d["dz"],
            d["rainnc"], d["rainncv"], d["snownc"], d["snowncv"], dt,
            reference_density=d["reference_density"],
            reference_temperature=d["reference_temperature"],
            snow_melt_marker=d["snow_melt_marker"],
            melt_rain_qr=d["melt_rain_qr"], melt_rain_nr=d["melt_rain_nr"],
            melt_rain_density=d["melt_rain_density"],
            velocity_boost=d["velocity_boost"], singular_fall=singular)
    else:
        sed.launch_wrf39_rain_sedimentation(
            d["qr"], d["nr"], d["temperature"], d["pressure"], d["qv"], d["dz"],
            d["rainnc"], d["rainncv"], dt,
            reference_density=d["melt_rain_density"])
    return {k: d[k].to_numpy() for k in PASSES[name][1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(tempfile.gettempdir())
                        / "woof-wrf39-fallout-word-probe-result.json")
    args = parser.parse_args()
    from host_backend import HostModule
    from woof.core.kernels import module_source_int_defines
    from woof.core.thompson_aerosol_launch import (
        WRF39_DEFINES, thompson_version_scope)
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    module = HostModule("thompson_aerosol_sed_wrf39", module_source_int_defines(
        "thompson_aerosol_sed", WRF39_DEFINES))
    records = []
    with MetalContext() as context, thompson_version_scope("wrf_39_noaa"):
        xp = MetalArrayModule(context)
        for nz in (4, 49, 65, 129):
            fields = case(nz, 7300 + nz)
            for name in PASSES:
                for singular in ((0, 1) if name == "snow" else (0,)):
                    if name == "rain":
                        fields_used = dict(fields, reference_density=fields["melt_rain_density"])
                    else:
                        fields_used = fields
                    expected = host_run(module, name, fields_used, nz, 20.0, singular)
                    actual = metal_run(xp, name, fields, 20.0, bool(singular))
                    for field in PASSES[name][1]:
                        a = actual[field].view(np.uint32).reshape(-1)
                        e = expected[field].view(np.uint32).reshape(-1)
                        bad = np.flatnonzero(a != e)
                        record = {"pass": name, "nz": nz, "singular": singular,
                                  "field": field, "words": int(a.size),
                                  "mismatches": int(bad.size)}
                        if bad.size:
                            i = int(bad[0])
                            record["first"] = {"index": i,
                                "expected": float(e[i:i+1].view(np.float32)[0]),
                                "actual": float(a[i:i+1].view(np.float32)[0])}
                        records.append(record)
    receipt = {"status": "FAIL" if any(r["mismatches"] for r in records) else "PASS",
               "output_words": sum(r["words"] for r in records),
               "mismatches": sum(r["mismatches"] for r in records),
               "failing": [r for r in records if r["mismatches"]]}
    args.out.write_text(json.dumps(dict(receipt, cases=records), indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
