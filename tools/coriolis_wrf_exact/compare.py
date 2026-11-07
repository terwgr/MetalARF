"""Compare actual Coriolis state caller with selected host-compiled source."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.coriolis_wrf_exact.generate import retained_source
from tools.smag_wrf_exact.compare import compare_words


def inputs(subnormal, mapped):
    nz,ny,nx=5,6,7
    rng=np.random.default_rng(1034)
    shapes={'u':(nz,ny,nx+1),'v':(nz,ny+1,nx),'w':(nz+1,ny,nx),
            'ru':(nz,ny,nx+1),'rv':(nz,ny+1,nx),
            'ru_t':(nz,ny,nx+1),'rv_t':(nz,ny+1,nx),'rw_t':(nz+1,ny,nx),
            'f':(ny,nx),'e':(ny,nx),'sina':(ny,nx),'cosa':(ny,nx)}
    a={key:rng.uniform(-1,1,shape).astype(np.float32) for key,shape in shapes.items()}
    if subnormal:
        for key in ('u','v','w','ru','rv','ru_t','rv_t','rw_t'):
            a[key] *= np.float32(2**-128)
    a.update(mub2d=rng.uniform(.9,1.1,(ny,nx)).astype(np.float32),
             mup=rng.uniform(-.05,.05,(ny,nx)).astype(np.float32),
             c1f=np.linspace(.8,1.,nz+1,dtype=np.float32),
             c2f=np.linspace(.1,.3,nz+1,dtype=np.float32),
             fnm=np.full(nz,.4,np.float32),fnp=np.full(nz,.6,np.float32))
    for key,shape in [('msft',(ny,nx)),('msfu',(ny,nx+1)),('msfv',(ny+1,nx))]:
        a[key]=rng.uniform(.8,1.2,shape).astype(np.float32) if mapped else np.ones(shape,np.float32)
    return a


def run(arrays,xp,metal,bx,by):
    from woof.core import dycore
    class State(SimpleNamespace):
        def scratch(self, shape, slot):
            if slot not in self._scratch: self._scratch[slot]=xp.asarray(np.zeros(shape,np.float32))
            return self._scratch[slot]
        def total_mu(self):
            if metal:
                from woof.metal.dycore import submit_total_mu
                return submit_total_mu(self)
            return np.add(self.mub2d,self.mup,dtype=np.float32)
    values={k:xp.asarray(v.copy()) for k,v in arrays.items()}
    state=State(**values,_metal_setup_state=metal,_scratch={})
    if metal: state._array_module=xp
    cfg=SimpleNamespace(dx=3.,dy=7.,open_x=bx,open_y=by,specified=False,nested=False)
    dycore.add_coriolis_curvature(state,cfg,state.ru,state.rv)
    return {key:(getattr(state,key).to_numpy() if metal else np.asarray(getattr(state,key)).copy())
            for key in ('ru_t','rv_t','rw_t')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    from woof.wrf_exact import ENABLED,BIGSTEP_ENABLED
    if not ENABLED: raise ValueError('GPUWM_WRF_EXACT must be enabled before import')
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    source='\n'.join(retained_source(BIGSTEP_ENABLED))
    module=host_backend.HostModule('wrf_exact_coriolis',source)
    from woof.core import dycore
    original=dycore.get_kernel
    dycore.get_kernel=lambda unit,name,*args: (module.get_function(name)
        if unit == 'coriolis_map' and name == 'coriolis_curvature' else original(unit,name,*args))
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    report={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'bigstep':bool(BIGSTEP_ENABLED),'cases':{}}
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        for sub,mapped,bx,by in product((False,True),repeat=4):
            arrays=inputs(sub,mapped)
            expected=run(arrays,np,False,bx,by)
            actual=run(arrays,xp,True,bx,by)
            for key in expected:
                name=f'{key}_sub{int(sub)}_map{int(mapped)}_x{int(bx)}y{int(by)}'
                report['cases'][name]=compare_words(actual[key],expected[key])
        # Exercise the actual total-mass producer on subnormal operands too.
        from woof.metal.dycore import submit_total_mu
        a=np.array([[2**-128,-2**-128,2**-126]],np.float32)
        b=np.array([[2**-128,2**-129,-2**-128]],np.float32)
        state=SimpleNamespace(_metal_setup_state=True,_array_module=xp,
            mub2d=xp.asarray(a),mup=xp.asarray(b),scratch=lambda shape,slot:xp.asarray(np.zeros(shape,np.float32)))
        out=submit_total_mu(state).to_numpy()
        report['cases']['total_mu_subnormal']=compare_words(out,np.add(a,b,dtype=np.float32))
    report['words']=sum(x['words'] for x in report['cases'].values())
    report['different_words']=sum(x['different_words'] for x in report['cases'].values())
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if report['different_words']: raise SystemExit(1)


if __name__=='__main__': main()
