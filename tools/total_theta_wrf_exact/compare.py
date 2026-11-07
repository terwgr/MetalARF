"""Compare the actual total-theta caller with selected host-compiled CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.mp28_metal.parse import closure, split_top
from tools.smag_wrf_exact.compare import compare_words


def source():
    from woof.core.kernels import module_source
    text='#define GPUWM_WRF_EXACT 1\n'+module_source('bandwidth_glue')
    expanded=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],
                            input=text,text=True,capture_output=True,check=True).stdout
    return '\n'.join(closure(split_top(expanded),('glue_total_theta',))[0])


def run(thb,thp,xp,metal):
    from woof.core.bandwidth_glue import total_theta
    state=SimpleNamespace(thb=xp.asarray(thb.copy()),thp=xp.asarray(thp.copy()),
                          _metal_setup_state=metal)
    if metal:
        state._array_module=xp
        state.scratch=lambda shape,slot:xp.asarray(np.zeros(shape,np.float32))
    out=total_theta(state)
    return out.to_numpy() if metal else np.asarray(out).copy()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    from woof.wrf_exact import ENABLED
    if not ENABLED: raise ValueError('GPUWM_WRF_EXACT must be enabled before import')
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    retained=source()
    module=host_backend.HostModule('wrf_exact_total_theta',
        'struct alignas(16) float4 { float x,y,z,w; };\n'+retained)
    from woof.core import bandwidth_glue
    original=bandwidth_glue.get_kernel
    bandwidth_glue.get_kernel=lambda unit,name,*args: (module.get_function(name)
        if unit=='bandwidth_glue' and name=='glue_total_theta' else original(unit,name,*args))
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    report={'source_sha256':hashlib.sha256(retained.encode()).hexdigest(),'cases':{}}
    pattern=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,
                      0x3f800000,0xbf800000,0x3f000000,0x33800000],np.uint32)
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        for shape in ((1,1,1),(3,5,7),(5,8,9)):
            for base3d in (False,True):
                for offset in (1,3,7):
                    thp=np.resize(pattern,np.prod(shape)).view(np.float32).reshape(shape).copy()
                    bshape=shape if base3d else (shape[0],)
                    thb=np.resize(np.roll(pattern,offset),np.prod(bshape)).view(np.float32).reshape(bshape).copy()
                    expected=run(thb,thp,np,False)
                    actual=run(thb,thp,xp,True)
                    name=f'{shape}_base3d{int(base3d)}_offset{offset}'
                    report['cases'][name]=compare_words(actual,expected)
    report['words']=sum(x['words'] for x in report['cases'].values())
    report['different_words']=sum(x['different_words'] for x in report['cases'].values())
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    if report['different_words']: raise SystemExit(1)


if __name__=='__main__': main()
