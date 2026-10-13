"""Bounded verification of Metal's HRRR physical-boundary diffusion path.

Run from the source checkout after rebuilding the Metal bundle:
python tools/metal_diff6_edge_experiment.py --out edge-check.json
The float64 reference is the existing CUDA diffusion transcription, evaluated
on NumPy edge-padded inputs. This verifies padding/cropping and all staggerings;
it does not establish complete forecast agreement with CUDA.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from tests.physics_reference.npref import np_diff6
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.core.dycore import launch_diff6_to_edge
from woof.core.diff6_edge_workspace import field_values, planes_values, phb_values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(240710)
    nz, ny, nx = 4, 7, 9
    rows = []
    with MetalContext() as context:
        arrays = MetalArrayModule(context)
        upload = lambda x: arrays.asarray(np.ascontiguousarray(x, dtype=np.float32))
        mass = (95000 + 1000 * rng.random((ny, nx))).astype(np.float32)
        slots = {}
        def scratch(shape, name):
            if name not in slots:
                slots[name] = arrays.zeros(shape, dtype=np.float32)
            return slots[name]
        state = SimpleNamespace(_metal_setup_state=True, _array_module=arrays,
                                mup=upload(np.zeros((ny, nx))),
                                mup0=upload(np.zeros((ny, nx))),
                                mub2d=upload(mass), scratch=scratch)
        work = {"field": arrays.empty((field_values(nz, ny, nx),)),
                "tend": arrays.empty((field_values(nz, ny, nx),)),
                "planes": arrays.empty((planes_values(ny, nx),)),
                "phb": arrays.empty((phb_values(nz, ny, nx),))}
        phb = np.broadcast_to(np.arange(nz+1)[:,None,None]*10000.,
                              (nz+1,ny,nx)).copy()
        phb += np.arange(ny)[None,:,None]*11 + np.arange(nx)[None,None,:]*23
        pad = lambda a: np.pad(a, [(0,0)]*(a.ndim-2)+[(3,4),(3,4)], mode="edge")
        for stag in ("", "x", "y", "z"):
            shape = (nz+int(stag=="z"), ny+int(stag=="y"), nx+int(stag=="x"))
            f = rng.normal(size=shape).astype(np.float32)
            c1 = np.linspace(.2,1,shape[0],dtype=np.float32)
            c2 = np.linspace(30,70,shape[0],dtype=np.float32)
            for slope in (0,1):
                ref = np_diff6(pad(f),pad(mass),c1,c2,.12,15.,2,stagger=stag,
                               phb=pad(phb),slopeopt=slope,dx=3000.,dy=3000.)
                ref = (ref[:,3:3+shape[1],3:3+shape[2]]+.25).astype(np.float32)
                previous = None
                for repeat in range(3):
                    target = upload(np.full(shape,.25,dtype=np.float32))
                    launch_diff6_to_edge(
                        upload(f),target,None,upload(c1),upload(c2),.12,15.,2,
                        stagger=stag,phb=upload(phb),
                        msfu=upload(np.ones((ny,nx+1))),
                        msfv=upload(np.ones((ny+1,nx))),
                        msft=upload(np.ones((ny,nx))),slopeopt=slope,
                        dx=3000.,dy=3000.,work=work,state=state)
                    actual = target.to_numpy()  # synchronized readback
                    np.testing.assert_allclose(actual,ref,rtol=3e-6,atol=5e-5)
                    if previous is not None:
                        assert actual.tobytes()==previous.tobytes()
                    previous = actual
                rows.append({"stagger":stag or "mass","slope":slope,
                             "repeats":3,"max_abs_error":float(np.max(np.abs(actual-ref))),
                             "finite":bool(np.isfinite(actual).all()),"repeat_bitwise_equal":True})
    result={"status":"PASS","reference":"existing np_diff6 float64 mirror on edge-padded inputs",
            "rtol":3e-6,"atol":5e-5,"cases":rows}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
