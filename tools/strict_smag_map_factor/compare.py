"""Compare the actual dry map-coupling caller with NumPy binary32 operations."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.smag_wrf_exact.compare import compare_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    from woof.wrf_exact import ENABLED, BIGSTEP_ENABLED
    if not ENABLED:
        raise ValueError('strict root must be enabled')
    from woof.core.dycore import _couple_dry_mixing_map_factor
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    nz, ny, nx = 3, 5, 7
    shape = {'': (nz,ny,nx), 'x': (nz,ny,nx+1),
             'y': (nz,ny+1,nx), 'z': (nz+1,ny,nx)}
    map_shape = {'': (ny,nx), 'x': (ny,nx+1), 'y': (ny+1,nx), 'z': (ny,nx)}
    words = np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,
                      0x3f800001,0xbf123456,0x40490fdb],dtype=np.uint32)
    with MetalContext() as context:
        context.batch_kernel_calls = False
        submit = context._submit_kernel_batch
        def sync(*a,**kw):
            result=submit(*a,**kw);result.wait();return result
        context._submit_kernel_batch = sync
        xp = MetalArrayModule(context)
        cases = {}
        for stag in ('','x','y','z'):
            original=np.resize(words,np.prod(shape[stag])).view(np.float32).reshape(shape[stag]).copy()
            factor=np.resize(np.array([0.5,0.75,1.0,1.25,2.0],np.float32),map_shape[stag])
            def state(module, metal):
                tend=module.asarray(original.copy());m=module.asarray(factor.copy())
                return SimpleNamespace(has_msf=True,_metal_setup_state=metal,p=np.empty((nz,ny,nx)),
                    msft=m,msfu=m,msfv=m,msfv_inv=module.asarray(np.nextafter(np.float32(1.0)/factor, np.float32(np.inf))),scratch=lambda shape,slot:tend),tend
            h,ht=state(np,False);d,dt=state(xp,True)
            spec=[(ht,ht,None,None,None,'probe',stag)]
            _couple_dry_mixing_map_factor(h,spec)
            _couple_dry_mixing_map_factor(d,[(dt,dt,None,None,None,'probe',stag)])
            cases[stag or 'scalar']=compare_words(dt.to_numpy(),ht)
    result={'bigstep':bool(BIGSTEP_ENABLED),'cases':cases,
            'words':sum(v['words'] for v in cases.values()),
            'different_words':sum(v['different_words'] for v in cases.values())}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if result['different_words']: raise SystemExit(1)


if __name__=='__main__': main()
