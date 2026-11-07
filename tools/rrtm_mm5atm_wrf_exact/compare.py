"""Compare strict classic RRTM MM5ATM with the shared array source."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.smag_wrf_exact.compare import compare_words


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    from woof.wrf_exact import ENABLED
    if not ENABLED: raise ValueError('strict mode required')
    from woof.core.rrtm_lw import mm5atm_columns,_o3_climatology
    from woof.core.rrtm_tables import load_rrtm_lw_statics
    from woof.metal.rrtm_lw_family import submit_mm5atm
    from woof.metal.runtime import MetalContext
    from woof.metal.array import MetalArrayModule
    tables=SimpleNamespace(statics=load_rrtm_lw_statics())
    ozone,ppwrkh=_o3_climatology()
    pprof=tables.statics['MM5ATM__PPROF'].astype(np.float32)
    tprof=tables.statics['MM5ATM__TPROF'].astype(np.float32)
    rng=np.random.default_rng(348)
    report={'cases':{}}
    with MetalContext() as context:
        context.batch_kernel_calls=False
        submit=context._submit_kernel_batch
        def sync(*a,**kw):
            result=submit(*a,**kw);result.wait();return result
        context._submit_kernel_batch=sync
        xp=MetalArrayModule(context)
        for nlay in (5,8):
            ncol,nz=7,5
            ptop = 100.0 if nlay == nz else 1200.0
            pw=np.broadcast_to(np.linspace(ptop,100000,nz+1,dtype=np.float32),(ncol,nz+1)).copy()
            pw[:,1:]+=rng.uniform(-100,100,(ncol,nz)).astype(np.float32)
            p=(np.float32(.5)*(pw[:,:-1]+pw[:,1:])).astype(np.float32)
            t=rng.uniform(230,305,(ncol,nz)).astype(np.float32)
            dz=rng.uniform(200,1100,(ncol,nz)).astype(np.float32)
            tw=rng.uniform(225,305,(ncol,nz+1)).astype(np.float32)
            qv=rng.uniform(0,.012,(ncol,nz)).astype(np.float32)
            hyd={k:rng.uniform(0,.002,(ncol,nz)).astype(np.float32) for k in ('qc','qr','qi','qs','qg')}
            for value in hyd.values(): value[0,0]=0
            hyd['qr'][0,0]=np.float32(2**-128);hyd['qs'][1,1]=np.float32(0)
            cldfra=rng.uniform(0,1,(ncol,nz)).astype(np.float32)
            tsfc=np.array([200,250,280,300,339,340,341],np.float32)
            emiss=rng.uniform(.8,1,ncol).astype(np.float32)
            co2,n2o,ch4=0.000417000010,1.4902620386954864e-07,2.696003660708664e-06
            ref=mm5atm_columns(np,tables,t=t,p_mb=p/np.float32(100),dz=dz,qv=qv,
                **hyd,cldfra=cldfra,tw=tw,pw_mb=pw/np.float32(100),tsfc=tsfc,
                emiss=emiss,r=287.0,g=9.81,co2vmr=co2,n2ovmr=n2o,ch4vmr=ch4,nlayers=nlay)
            assert ref['taucloud'][0,nz-1] > 0, 'subnormal rain must affect optical depth'
            vals=dict(t=t,p_pa=p,dz=dz,qv=qv,**hyd,cldfra=cldfra,tw=tw,pw_pa=pw,
                      tsfc=tsfc,emiss=emiss,o3wrk=ozone,ppwrkh=ppwrkh,pprof=pprof,tprof=tprof)
            device={k:xp.asarray(v) for k,v in vals.items()}
            shapes={'pavel':(ncol,nlay),'tavel':(ncol,nlay),'pz':(ncol,nlay+1),
                    'tz':(ncol,nlay+1),'cldfra_out':(ncol,nlay),'taucloud':(ncol,nlay),
                    'coldry':(ncol,nlay),'wkl':(ncol,nlay,35),'wx':(ncol,nlay,4),
                    'tbound':(ncol,),'semiss':(ncol,16)}
            out={k:xp.zeros(shape,np.float32) for k,shape in shapes.items()}
            submit_mm5atm(**device,**out,co2vmr=co2,n2ovmr=n2o,ch4vmr=ch4,
                          r=287.,g=9.81,nlayers=nlay)
            for name,value in out.items():
                report['cases'][f'nlay{nlay}_{name}']=compare_words(value.to_numpy(),ref['cldfrac' if name=='cldfra_out' else name])
            if nlay == nz:
                before={name:value.to_numpy() for name,value in out.items()}
                bad=qv.copy();bad[0,0]=np.nan
                invalid={**device, 'qv':xp.asarray(bad)}
                try:
                    submit_mm5atm(**invalid,**out,co2vmr=co2,n2ovmr=n2o,ch4vmr=ch4,
                                  r=287.,g=9.81,nlayers=nlay)
                except ValueError as error:
                    assert 'finite qv' in str(error)
                else:
                    raise AssertionError('nonfinite water was not refused')
                assert all(np.array_equal(out[name].to_numpy().view(np.uint32),v.view(np.uint32))
                           for name,v in before.items())
                report['nonfinite_water_refused_before_output_mutation']=True
    report['words']=sum(x['words'] for x in report['cases'].values())
    report['different_words']=sum(x['different_words'] for x in report['cases'].values())
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    if report['different_words']:raise SystemExit(1)
if __name__=='__main__':main()
