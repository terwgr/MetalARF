"""Compare selected sixth-order CUDA source with actual Metal callers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.diff6_wrf_exact.generate import retained_source
from tools.smag_wrf_exact.compare import compare_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--unlimited', action='store_true')
    args = ap.parse_args()
    from woof.wrf_exact import ENABLED
    if not ENABLED: raise ValueError('strict mode required')
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    from woof.core import dycore
    from woof.core.diff6_edge_workspace import field_values, planes_values, phb_values
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    source = 'typedef float real;\n' + '\n'.join(retained_source())
    module = host_backend.HostModule('strict_diff6_source', source)
    previous = dycore.get_kernel
    dycore.get_kernel = lambda unit, name, *rest: module.get_function(name) if unit in ('diff6', 'diff6_seam') else previous(unit, name, *rest)
    rng = np.random.default_rng(7823)
    cases = {}
    with MetalContext() as context:
        xp = MetalArrayModule(context)
        nz, ny, nx = 5, 9, 10
        mass = rng.uniform(.9, 1.1, (ny,nx)).astype(np.float32)
        c1, c2 = np.ones(nz+1,np.float32), np.full(nz+1,.1,np.float32)
        msfu = rng.uniform(.9,1.1,(ny,nx+1)).astype(np.float32)
        msfv = rng.uniform(.9,1.1,(ny+1,nx)).astype(np.float32)
        msft = rng.uniform(.9,1.1,(ny,nx)).astype(np.float32)
        phb = rng.uniform(-5,5,(nz+1,ny,nx)).astype(np.float32)
        def put(value): return xp.asarray(value)
        state = SimpleNamespace(_metal_setup_state=True,_array_module=xp,
            p=put(np.zeros((nz,ny,nx),np.float32)), mup=put(mass),
            msfu=put(msfu),msfv=put(msfv),msft=put(msft))
        work={'field':xp.zeros((field_values(nz,ny,nx),),np.float32),
              'tend':xp.zeros((field_values(nz,ny,nx),),np.float32),
              'planes':xp.zeros((planes_values(ny,nx),),np.float32),
              'phb':xp.zeros((phb_values(nz,ny,nx),),np.float32)}
        for scale_name, scale in (('normal', np.float32(1)), ('subnormal', np.float32(2**-128))):
            for stagger in ('', 'x', 'y', 'z'):
                shape=(nz+int(stagger=='z'), ny+int(stagger=='y'), nx+int(stagger=='x'))
                field=rng.uniform(-1,1,shape).astype(np.float32)
                factor=np.float32(2**-120) if scale_name == 'subnormal' else np.float32(.12)
                initial=(rng.uniform(-1,1,shape).astype(np.float32)*scale)
                for mode in ('periodic','specified','edge'):
                    for slope in (0,1):
                        expected=initial.copy(); actual=put(initial)
                        kw=dict(stagger=stagger,phb=phb,msfu=msfu,msfv=msfv,msft=msft,
                                slopeopt=slope,thresh=.1,dx=1000.,dy=1000.)
                        if mode == 'edge':
                            dycore.launch_diff6_to_edge(field,expected,mass,c1[:shape[0]],c2[:shape[0]],factor,1.,1 if args.unlimited else 2,**kw)
                            dycore.launch_diff6_to_edge(put(field),actual,put(mass),put(c1[:shape[0]]),put(c2[:shape[0]]),factor,1.,1 if args.unlimited else 2,
                                **{**kw,'phb':put(phb),'msfu':state.msfu,'msfv':state.msfv,'msft':state.msft},state=state,work=work)
                        else:
                            forced=mode=='specified'
                            dycore.launch_diff6(field,expected,mass,c1[:shape[0]],c2[:shape[0]],factor,1.,1 if args.unlimited else 2,**kw,bnd_x=forced,bnd_y=forced)
                            dycore.launch_diff6(put(field),actual,put(mass),put(c1[:shape[0]]),put(c2[:shape[0]]),factor,1.,1 if args.unlimited else 2,
                                **{**kw,'phb':put(phb),'msfu':state.msfu,'msfv':state.msfv,'msft':state.msft},bnd_x=forced,bnd_y=forced,state=state)
                        if not np.any(expected.view(np.uint32) != initial.view(np.uint32)):
                            raise AssertionError(f'inactive filter: {scale_name}, {stagger}, {mode}, {slope}')
                        key=f'{scale_name}_{stagger or "mass"}_{mode}_slope{slope}'
                        cases[key]=compare_words(actual.to_numpy(),expected)
                        print(key,json.dumps(cases[key]),flush=True)
    report={'cases':cases,'words':sum(c['words'] for c in cases.values()),
            'different_words':sum(c['different_words'] for c in cases.values())}
    if not any(c['subnormal_words'] for k,c in cases.items() if k.startswith('subnormal_')):
        raise AssertionError('subnormal output not exercised')
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('words','different_words')}))
    if report['different_words']: raise SystemExit(1)

if __name__ == '__main__': main()
