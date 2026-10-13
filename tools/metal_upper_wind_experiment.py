"""Verify the Metal saved-wind limiter against the existing Fortran fixture."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from woof.core.acoustic import prepare_upper_wind_limiter
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    rows=[]
    with np.load(ROOT/'tests/data/upper_wind_fork.npz') as golden, MetalContext() as context:
        arrays=MetalArrayModule(context)
        for case in range(int(golden['cases'])):
            prefix=f'{case}_'
            cfg=SimpleNamespace(nx=7,ny=5,nz=6,damp_opt=3,
                zdamp=float(golden[prefix+'zdamp']),
                specified=bool(golden[prefix+'zone']),nested=False,
                upper_wind_limiter_form='noaa_wrf39')
            for repeat in range(3):
                state=SimpleNamespace(_metal_setup_state=True,_array_module=arrays,
                    **{n:arrays.asarray(np.ascontiguousarray(golden[prefix+n]))
                       for n in ('u','v','php','phb')})
                state.thb=state.phb
                launch=prepare_upper_wind_limiter(state,cfg,float(golden[prefix+'dts']))
                for _ in range(int(golden[prefix+'substeps'])):
                    launch()
                for name in ('u','v'):
                    np.testing.assert_array_equal(state.__dict__[name].to_numpy().view(np.uint32),
                        golden[prefix+name+'_out'].view(np.uint32),err_msg=prefix+name)
            rows.append({'case':case,'repeats':3,'u_v_bitwise_equal_to_fortran':True})
    result={'status':'PASS','reference':'tests/data/upper_wind_fork.npz compiled Fortran fixture','cases':rows}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
