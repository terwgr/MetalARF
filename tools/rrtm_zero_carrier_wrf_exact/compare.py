"""Compare the classic RRTM GASABS band 7 with shared source."""
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
    from woof.metal.rrtm_lw_family import submit_mm5atm,submit_setcoef,submit_taumol_01_02,submit_taumol_03,submit_taumol_04_05,submit_taumol_06,submit_taumol_07,submit_taumol_all
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
        out=submit_taumol_01_02(coef=dev,tables=tables)
        submit_taumol_03(coef=dev,tables=tables,taug=out['taug'],pfrac=out['pfrac'])
        submit_taumol_04_05(coef=dev,wx=mmout['wx'],tables=tables,taug=out['taug'],pfrac=out['pfrac'])
        submit_taumol_06(coef=dev,wx=mmout['wx'],tables=tables,taug=out['taug'],pfrac=out['pfrac'])
        submit_taumol_07(coef=dev,tables=tables,taug=out['taug'],pfrac=out['pfrac'])
        for name,want in (('taug',want_tau),('pfrac',want_frac)):
            report['cases']['chain_'+name]=compare_words(out[name].to_numpy()[:,:,76:88].copy(),want[:,:,76:88].copy())
        # Source-level tiny carrier exercises both multiplication and gradual underflow.
        small={k:v.copy() for k,v in coef.items()};small['colh2o'][0,0]=np.float32(2**-145);small['colco2'][0,0]=np.float32(2**-145);small['co2mult'][0,0]=np.float32(2**-145);small['colo3'][0,0]=np.float32(2**-145)
        with np.errstate(all='ignore'):tiny_tau,tiny_frac=source(small)
        tiny_dev={k:xp.asarray(v) for k,v in small.items()};tiny=submit_taumol_01_02(coef=tiny_dev,tables=tables)
        submit_taumol_03(coef=tiny_dev,tables=tables,taug=tiny['taug'],pfrac=tiny['pfrac'])
        submit_taumol_04_05(coef=tiny_dev,wx=mmout['wx'],tables=tables,taug=tiny['taug'],pfrac=tiny['pfrac'])
        submit_taumol_06(coef=tiny_dev,wx=mmout['wx'],tables=tables,taug=tiny['taug'],pfrac=tiny['pfrac'])
        submit_taumol_07(coef=tiny_dev,tables=tables,taug=tiny['taug'],pfrac=tiny['pfrac'])
        for name,want in (('taug',tiny_tau),('pfrac',tiny_frac)):
            report['cases']['tiny_'+name]=compare_words(tiny[name].to_numpy()[:,:,76:88].copy(),want[:,:,76:88].copy())
        assert report['cases']['tiny_taug']['subnormal_words']>0
        # Each zero endpoint is compared independently with the selected source.
        for name in ('colh2o','colco2','colo3'):
            zero={k:v.copy() for k,v in coef.items()};zero[name][0,0]=np.float32(0);zero[name][0,-1]=np.float32(0)
            with np.errstate(all='ignore'): want_tau,want_frac=source(zero)
            zdev={k:xp.asarray(v) for k,v in zero.items()};out=submit_taumol_01_02(coef=zdev,tables=tables)
            submit_taumol_03(coef=zdev,tables=tables,taug=out['taug'],pfrac=out['pfrac'])
            submit_taumol_04_05(coef=zdev,wx=mmout['wx'],tables=tables,taug=out['taug'],pfrac=out['pfrac'])
            submit_taumol_07(coef=zdev,tables=tables,taug=out['taug'],pfrac=out['pfrac'])
            for band,sl in (('3',slice(22,38)),('45',slice(38,68)),('7',slice(76,88))):
                for key,want in (('taug',want_tau),('pfrac',want_frac)):
                    got=out[key].to_numpy()[0,[0,-1],sl].copy();expected=want[0,[0,-1],sl].copy()
                    assert np.all(np.isfinite(expected))
                    report['cases'][name+'_'+band+'_'+key]=compare_words(got,expected)
        for fn,pair in ((submit_taumol_03,('colh2o','colco2')),
                        (submit_taumol_04_05,('colh2o','colco2')),
                        (submit_taumol_04_05,('colo3','colco2')),
                        (submit_taumol_07,('colh2o','colo3'))):
            invalid=dict(dev)
            for name in pair:
                bad=coef[name].copy();bad[0,0]=0;invalid[name]=xp.asarray(bad)
            before=[out[k].to_numpy().copy() for k in ('taug','pfrac')]
            kw={'coef':invalid,'tables':tables,'taug':out['taug'],'pfrac':out['pfrac']}
            if fn is submit_taumol_04_05:kw['wx']=mmout['wx']
            try:
                fn(**kw)
                raise AssertionError('zero combined carrier accepted')
            except ValueError:
                pass
            for key,old in zip(('taug','pfrac'),before):
                assert np.array_equal(out[key].to_numpy().view(np.uint32),old.view(np.uint32))
    report['words']=sum(x['words'] for x in report['cases'].values());report['different_words']=sum(x['different_words'] for x in report['cases'].values())
    args.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
    if report['different_words']:raise SystemExit(1)
if __name__=='__main__':main()
