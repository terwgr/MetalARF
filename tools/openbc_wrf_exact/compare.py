"""Compare open boundaries with selected CUDA radiative and shared host copies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.openbc_wrf_exact.generate import retained_source
from tools.smag_wrf_exact.compare import compare_words


def initial(subnormal):
    nz,ny,nx=5,7,8
    rng=np.random.default_rng(401)
    def values(shape): return rng.uniform(-2,2,shape).astype(np.float32)
    u,v=values((nz,ny,nx+1)),values((nz,ny+1,nx))
    if subnormal:
        u *= np.float32(2**-128);v *= np.float32(2**-128)
    return dict(u=u,v=v,w=values((nz+1,ny,nx)),thp=values((nz,ny,nx)),
        qv=values((nz,ny,nx)),qc=values((nz,ny,nx)),qr=values((nz,ny,nx)),
        ru_t=values((nz,ny,nx+1)),rv_t=values((nz,ny+1,nx)),
        mup=values((ny,nx)),mub2d=np.full((ny,nx),10000,np.float32),
        c1h=np.linspace(.5,1,nz,dtype=np.float32),c2h=np.ones(nz,np.float32),
        msfu=np.full((ny,nx+1),1.1,np.float32),msfv=np.full((ny+1,nx),1.2,np.float32))


def run(source,xp,metal,mapped):
    from woof.core import dycore
    a={name:xp.asarray(value.copy()) for name,value in source.items()}
    s=SimpleNamespace(**a,_metal_setup_state=metal,has_msf=mapped)
    if metal:s._array_module=xp
    cfg=SimpleNamespace(nx=8,ny=7,nz=5,dx=1000.,dy=1300.,open_x=True,open_y=True,specified=False,nested=False)
    dycore.apply_open_radiative_bc(s,cfg)
    dycore.apply_open_zero_gradient(s,cfg)
    return {name:(getattr(s,name).to_numpy() if metal else np.asarray(getattr(s,name)).copy())
            for name in ('ru_t','rv_t','u','v','w','thp','qv','qc','qr')}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    from woof.wrf_exact import ENABLED
    if not ENABLED:raise ValueError('strict mode required')
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    source='\n'.join(retained_source()); module=host_backend.HostModule('strict_openbc',source)
    from woof.core import dycore
    old=dycore.get_kernel
    dycore.get_kernel=lambda unit,name,*a: module.get_function(name) if unit=='openbc' and name in ('open_u_radiative','open_v_radiative') else old(unit,name,*a)
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    report={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'cases':{}}
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        for sub in (False,True):
            data=initial(sub)
            for mapped in (False,True):
                host=run(data,np,False,mapped);metal=run(data,xp,True,mapped)
                assert np.any(host['thp'] != data['thp'])
                for name in host:report['cases'][f'sub{int(sub)}_map{int(mapped)}_{name}']=compare_words(metal[name],host[name])
    assert report['cases']['sub1_map0_u']['subnormal_words'] > 0
    assert report['cases']['sub1_map0_v']['subnormal_words'] > 0
    report['words']=sum(v['words'] for v in report['cases'].values())
    report['different_words']=sum(v['different_words'] for v in report['cases'].values())
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    if report['different_words']:raise SystemExit(1)
if __name__=='__main__':main()
