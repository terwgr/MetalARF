"""Compare the classic RRTM GASABS GASABS Pade indices with shared source."""
import argparse,inspect,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.smag_wrf_exact.compare import compare_words

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    from woof.wrf_exact import ENABLED
    if not ENABLED:raise ValueError('strict mode required')
    from woof.core.rrtm_lw import mm5atm_columns,setcoef_columns,_o3_climatology
    from woof.core.rrtm_taumol import rrtm_taumol
    from woof.core.rrtm_tables import load_rrtm_lw_tables
    from woof.metal.rrtm_lw_family import submit_mm5atm,submit_setcoef,submit_taumol_01_02,submit_taumol_03,submit_taumol_04_05,submit_taumol_06,submit_taumol_07,submit_taumol_08,submit_taumol_09,submit_taumol_10_11,submit_taumol_12_16,submit_taumol_all,submit_make_itr
    from woof.metal.runtime import MetalContext
    from woof.metal.array import MetalArrayModule
    tables=load_rrtm_lw_tables();ncol,nz,nlay=3,5,8
    rng=np.random.default_rng(413);pw=np.broadcast_to(np.linspace(1200,100000,nz+1,dtype=np.float32),(ncol,nz+1)).copy()
    p=(np.float32(.5)*(pw[:,:-1]+pw[:,1:])).astype(np.float32)
    t=rng.uniform(235,300,(ncol,nz)).astype(np.float32);dz=np.full((ncol,nz),600,np.float32)
    qv=rng.uniform(.001,.02,(ncol,nz)).astype(np.float32)
    hydro=np.full((ncol,nz),.0002,np.float32);cloud=np.full((ncol,nz),.4,np.float32)
    tw=np.full((ncol,nz+1),260,np.float32);skin=np.full(ncol,290,np.float32);emiss=np.full(ncol,.95,np.float32)
    co2,n2o,ch4=.0004,3.19e-7,1.774e-6
    mm=mm5atm_columns(np,tables,t=t,p_mb=p/np.float32(100),dz=dz,qv=qv,qc=hydro,qr=hydro,qi=hydro,qs=hydro,qg=hydro,cldfra=cloud,tw=tw,pw_mb=pw/np.float32(100),tsfc=skin,emiss=emiss,r=287.,g=9.81,co2vmr=co2,n2ovmr=n2o,ch4vmr=ch4,nlayers=nlay)
    coef=setcoef_columns(np,tables,pavel=mm['pavel'],tavel=mm['tavel'],coldry=mm['coldry'],wkl=mm['wkl']);coef['coldry']=mm['coldry']
    assert np.all((coef['laytrop']>0)&(coef['laytrop']<nlay))
    def source(c):
        return rrtm_taumol(**c,wx=mm['wx'],tables=tables)
    with np.errstate(all='ignore'):want_tau,want_frac=source(coef)
    report={'cases':{}}
    with MetalContext() as context:
        context.batch_kernel_calls=False;submit=context._submit_kernel_batch
        def sync(*a,**kw):
            result=submit(*a,**kw);result.wait();return result
        context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
        ozone,ppwrkh=_o3_climatology();stat=tables.statics
        vals={'t':t,'p_pa':p,'dz':dz,'qv':qv,'qc':hydro,'qr':hydro,'qi':hydro,'qs':hydro,'qg':hydro,'cldfra':cloud,'tw':tw,'pw_pa':pw,'tsfc':skin,'emiss':emiss,'o3wrk':ozone,'ppwrkh':ppwrkh,'pprof':stat['MM5ATM__PPROF'],'tprof':stat['MM5ATM__TPROF']}
        inputs={name:xp.asarray(value,dtype=np.float32) for name,value in vals.items()}
        shapes={'pavel':(ncol,nlay),'tavel':(ncol,nlay),'pz':(ncol,nlay+1),'tz':(ncol,nlay+1),'cldfra_out':(ncol,nlay),'taucloud':(ncol,nlay),'coldry':(ncol,nlay),'wkl':(ncol,nlay,35),'wx':(ncol,nlay,4),'tbound':(ncol,),'semiss':(ncol,16)}
        mmout={name:xp.zeros(shape,np.float32) for name,shape in shapes.items()}
        submit_mm5atm(**inputs,**mmout,co2vmr=co2,n2ovmr=n2o,ch4vmr=ch4,r=287.,g=9.81,nlayers=nlay)
        dev=submit_setcoef(pavel=mmout['pavel'],tavel=mmout['tavel'],coldry=mmout['coldry'],wkl=mmout['wkl'],preflog=xp.asarray(stat['PREFLOG']),tref=xp.asarray(stat['TREF']));dev['coldry']=mmout['coldry']
        out=submit_taumol_all(coef=dev,wx=mmout['wx'],tables=tables)
        def host_index(tau,bpade=(np.float32(1)/np.float32(.278))):
            od=np.float32(1.66)*tau
            with np.errstate(all='ignore'): frac=od/(np.float32(bpade)+od)
            frac=np.where(od<=np.float32(0),np.float32(0),frac)
            return np.clip(np.trunc(np.float32(5000)*frac+np.float32(.5)).astype(np.int32),0,5000)
        want_index=host_index(want_tau);got_index=submit_make_itr(out['taug']).to_numpy()
        report['cases']['chain_index']=compare_words(got_index,want_index)
        # Source-level tiny carrier exercises both multiplication and gradual underflow.
        small={k:v.copy() for k,v in coef.items()};small['colh2o'][0,0]=np.float32(2**-145);small['colco2'][0,0]=np.float32(2**-145);small['co2mult'][0,0]=np.float32(2**-145);small['colch4'][0,0]=np.float32(2**-145);small['coln2o'][0,0]=np.float32(2**-145)
        with np.errstate(all='ignore'):tiny_tau,tiny_frac=source(small)
        tiny_dev={k:xp.asarray(v) for k,v in small.items()};tiny=submit_taumol_all(coef=tiny_dev,wx=mmout['wx'],tables=tables)
        report['cases']['tiny_index']=compare_words(submit_make_itr(tiny['taug']).to_numpy(),host_index(tiny_tau))
        points=np.array([0, -0.0, -1.0, 0.0010840126, 1e-40, .1, 1., 100.],np.float32)
        values=np.resize(points,(1,1,140));want=host_index(values)
        assert want[0,0,3]==2
        report['cases']['threshold_index']=compare_words(submit_make_itr(xp.asarray(values)).to_numpy(),want)
        report['cases']['custom_index']=compare_words(submit_make_itr(xp.asarray(values),bpade=1.25).to_numpy(),host_index(values,1.25))
        for bad in (np.nan,np.inf):
            invalid=values.copy();invalid[0,0,0]=bad
            try:
                submit_make_itr(xp.asarray(invalid))
                raise AssertionError('nonfinite optical depth accepted')
            except ValueError:
                pass
    report['words']=sum(x['words'] for x in report['cases'].values());report['different_words']=sum(x['different_words'] for x in report['cases'].values())
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    if report['different_words']:raise SystemExit(1)
if __name__=='__main__':main()
