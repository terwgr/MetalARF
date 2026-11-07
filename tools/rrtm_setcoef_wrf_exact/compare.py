"""Compare strict classic RRTM SETCOEF to the shared source."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.smag_wrf_exact.compare import compare_words

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    from woof.wrf_exact import ENABLED
    if not ENABLED: raise ValueError('strict mode required')
    from woof.core.rrtm_lw import setcoef_columns, mm5atm_columns, _o3_climatology
    from woof.core.rrtm_tables import load_rrtm_lw_statics
    from woof.metal.rrtm_lw_family import submit_setcoef, submit_mm5atm
    from woof.metal.runtime import MetalContext
    from woof.metal.array import MetalArrayModule
    statics=load_rrtm_lw_statics();tables=SimpleNamespace(statics=statics)
    rng=np.random.default_rng(802);ncol,nlay=5,8
    base=np.array([1000,750,500,320,250,95,12,1],np.float32)
    pavel=np.broadcast_to(base,(ncol,nlay)).copy()
    pavel*=rng.uniform(.97,1.03,(ncol,nlay)).astype(np.float32)
    from woof.core.noahmp_libm import expf, logf_array
    for threshold,index in ((4.56,5),(5.76,3),(6.62,1)):
        center=expf(np.float32(threshold)); low=high=center
        for _ in range(12):
            low=np.nextafter(low,np.float32(0));high=np.nextafter(high,np.float32(np.inf))
        pavel[:3,index]=[low,center,high]
        logarithms=logf_array(pavel[:3,index])
        assert logarithms[0] < np.float32(threshold) and logarithms[2] > np.float32(threshold)
    tavel=rng.uniform(190,310,(ncol,nlay)).astype(np.float32)
    coldry=rng.uniform(2e23,5e24,(ncol,nlay)).astype(np.float32)
    wkl=np.zeros((ncol,nlay,35),np.float32)
    for i,v in ((0,.01),(1,.00042),(2,.000003),(3,.0000003),(5,.0000017)):
        wkl[:,:,i]=coldry*np.float32(v)
    wkl[0,0,1]=0;wkl[1,0,3]=0;wkl[2,0,5]=0
    reference=setcoef_columns(np,tables,pavel=pavel,tavel=tavel,coldry=coldry,wkl=wkl)
    report={'cases':{}}
    with MetalContext() as context:
        context.batch_kernel_calls=False;submit=context._submit_kernel_batch
        def sync(*a,**kw):
            result=submit(*a,**kw);result.wait();return result
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        result=submit_setcoef(pavel=xp.asarray(pavel),tavel=xp.asarray(tavel),
                              coldry=xp.asarray(coldry),wkl=xp.asarray(wkl),
                              preflog=xp.asarray(statics['PREFLOG']),tref=xp.asarray(statics['TREF']))
        for name,value in result.items():report['cases'][name]=compare_words(value.to_numpy(),reference[name])
        # Feed SETCOEF from the actual strict MM5ATM outputs, with the
        # corresponding shared source stages evaluated in the same order.
        nc,nz,nrad=2,5,8
        pw=np.broadcast_to(np.linspace(1200,100000,nz+1,dtype=np.float32),(nc,nz+1)).copy()
        pressure=(np.float32(.5)*(pw[:,:-1]+pw[:,1:])).astype(np.float32)
        temp=rng.uniform(230,300,(nc,nz)).astype(np.float32)
        dz=np.full((nc,nz),500,np.float32);water=np.full((nc,nz),.001,np.float32)
        cloud=np.full((nc,nz),.4,np.float32);tw=np.full((nc,nz+1),260,np.float32)
        skin=np.full(nc,290,np.float32);emiss=np.full(nc,.95,np.float32)
        ozone,ppwrkh=_o3_climatology();co2,n2o,ch4=.0004,3.19e-7,1.774e-6
        host=mm5atm_columns(np,tables,t=temp,p_mb=pressure/np.float32(100),dz=dz,
             qv=water,qc=water,qr=water,qi=water,qs=water,qg=water,
             cldfra=cloud,tw=tw,pw_mb=pw/np.float32(100),tsfc=skin,emiss=emiss,
             r=287.,g=9.81,co2vmr=co2,n2ovmr=n2o,ch4vmr=ch4,nlayers=nrad)
        ref_chain=setcoef_columns(np,tables,pavel=host['pavel'],tavel=host['tavel'],
                                  coldry=host['coldry'],wkl=host['wkl'])
        shapes={'pavel':(nc,nrad),'tavel':(nc,nrad),'pz':(nc,nrad+1),
                'tz':(nc,nrad+1),'cldfra_out':(nc,nrad),'taucloud':(nc,nrad),
                'coldry':(nc,nrad),'wkl':(nc,nrad,35),'wx':(nc,nrad,4),
                'tbound':(nc,),'semiss':(nc,16)}
        outputs={name:xp.zeros(shape,np.float32) for name,shape in shapes.items()}
        vals={'t':temp,'p_pa':pressure,'dz':dz,'qv':water,'qc':water,'qr':water,
              'qi':water,'qs':water,'qg':water,'cldfra':cloud,'tw':tw,'pw_pa':pw,
              'tsfc':skin,'emiss':emiss,'o3wrk':ozone,'ppwrkh':ppwrkh,
              'pprof':statics['MM5ATM__PPROF'],'tprof':statics['MM5ATM__TPROF']}
        inputs={name:xp.asarray(value,dtype=np.float32) for name,value in vals.items()}
        submit_mm5atm(**inputs,**outputs,co2vmr=co2,n2ovmr=n2o,ch4vmr=ch4,
                      r=287.,g=9.81,nlayers=nrad)
        chain=submit_setcoef(pavel=outputs['pavel'],tavel=outputs['tavel'],
                             coldry=outputs['coldry'],wkl=outputs['wkl'],
                             preflog=xp.asarray(statics['PREFLOG']),tref=xp.asarray(statics['TREF']))
        for name,value in chain.items():
            report['cases']['mm5atm_setcoef_'+name]=compare_words(value.to_numpy(),ref_chain[name])
        original_submit=context.submit_kernel
        called=[]
        def record_submit(*a,**kw):
            called.append(True)
            return original_submit(*a,**kw)
        context.submit_kernel=record_submit
        invalid=wkl.copy();invalid[0,0,0]=np.nan
        try:
            submit_setcoef(pavel=xp.asarray(pavel),tavel=xp.asarray(tavel),
                           coldry=xp.asarray(coldry),wkl=xp.asarray(invalid),
                           preflog=xp.asarray(statics['PREFLOG']),tref=xp.asarray(statics['TREF']))
        except ValueError as error:
            assert 'finite wkl' in str(error)
        else:
            raise AssertionError('nonfinite gas column was not refused')
        assert not called, 'nonfinite input submitted a kernel'
        report['nonfinite_refused_before_submission']=True
        empty=xp.zeros((2,0),np.float32)
        try:
            submit_setcoef(pavel=empty,tavel=empty,coldry=empty,
                           wkl=xp.zeros((2,0,35),np.float32),
                           preflog=xp.asarray(statics['PREFLOG']),tref=xp.asarray(statics['TREF']))
        except ValueError as error:
            assert 'at least one layer' in str(error)
        else:
            raise AssertionError('empty layer input was not refused')
        assert not called, 'empty layer input submitted a kernel'
        report['empty_layers_refused_before_submission']=True
    report['words']=sum(x['words'] for x in report['cases'].values())
    report['different_words']=sum(x['different_words'] for x in report['cases'].values())
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    if report['different_words']:raise SystemExit(1)
if __name__=='__main__':main()
