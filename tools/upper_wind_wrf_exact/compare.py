"""Compare the actual upper-wind caller with selected WOOF CUDA source."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.upper_wind_wrf_exact.generate import retained_source
from tools.smag_wrf_exact.compare import compare_words


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    from woof.wrf_exact import ENABLED
    if not ENABLED: raise ValueError('strict mode required')
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    source='\n'.join(retained_source())
    module=host_backend.HostModule('strict_upper_wind',source)
    from woof.core import acoustic
    original=acoustic.get_kernel
    acoustic.get_kernel=lambda unit,name,*args: module.get_function(name) if unit=='upper_wind_limiter' else original(unit,name,*args)
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    report={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'cases':{}}
    nz,ny,nx=9,7,8
    rng=np.random.default_rng(734)
    with MetalContext() as ctx:
        xp=MetalArrayModule(ctx)
        for b3 in (0,1):
            for zone in (0,1):
                for sub in (0,1):
                    u=rng.uniform(-2,2,(nz,ny,nx+1)).astype(np.float32)
                    v=rng.uniform(-2,2,(nz,ny+1,nx)).astype(np.float32)
                    speeds=np.array([109.,110.,np.nextafter(np.float32(110),np.float32(111)),111.,125.,160.,200.,250.],np.float32)
                    v[-3:,:ny,:nx]=np.broadcast_to(speeds[None,None,:],(3,ny,nx))
                    if sub: u[-3:,:,:nx]*=np.float32(2**-128)
                    phb=np.arange(nz+1,dtype=np.float32)*np.float32(9810.)
                    if b3: phb=np.broadcast_to(phb[:,None,None],(nz+1,ny,nx)).copy()
                    php=np.zeros((nz+1,ny,nx),np.float32)
                    thb=np.zeros((nz,ny,nx) if b3 else (nz,),np.float32)
                    h=SimpleNamespace(u=u.copy(),v=v.copy(),php=php,phb=phb,thb=thb,_metal_setup_state=False)
                    m=SimpleNamespace(u=xp.asarray(u),v=xp.asarray(v),php=xp.asarray(php),phb=xp.asarray(phb),thb=xp.asarray(thb),_metal_setup_state=True,_array_module=xp)
                    cfg=SimpleNamespace(nz=nz,ny=ny,nx=nx,damp_opt=3,upper_wind_limiter_form='noaa_wrf39',
                        zdamp=1800.,specified=bool(zone),nested=False,open_x=False,open_y=False,spec_zone=zone)
                    acoustic.prepare_upper_wind_limiter(h,cfg,1.)()
                    acoustic.prepare_upper_wind_limiter(m,cfg,1.)()
                    if not np.any(h.v.view(np.uint32)!=v.view(np.uint32)): raise AssertionError('limiter inactive')
                    for name in ('u','v'):
                        key=f'base{b3}_zone{zone}_sub{sub}_{name}'
                        report['cases'][key]=compare_words(getattr(m,name).to_numpy(),getattr(h,name))
    report['words']=sum(c['words'] for c in report['cases'].values())
    report['different_words']=sum(c['different_words'] for c in report['cases'].values())
    if not any(v['subnormal_words'] for k,v in report['cases'].items() if '_sub1_u' in k):
        raise AssertionError('subnormal output not exercised')
    args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('words','different_words')}))
    if report['different_words']: raise SystemExit(1)

if __name__=='__main__': main()
