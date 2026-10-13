"""Bounded scalar-advection/PD replay on captured inputs and SL branch controls."""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
from woof.core.advection import launch_flux_div_scalar
from woof.core.moist import launch_pd_fluxes,launch_pd_renorm_apply
from tests.physics_reference.npref import np_flux_div_scalar,np_pd_fluxes,np_pd_renorm_apply

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
with np.load(a.inputs) as z:base={n:z[n].copy() for n in z.files}
rows=[]
with MetalContext() as context:
    xp=MetalArrayModule(context)
    coord=SimpleNamespace(_metal_setup_state=True,_array_module=xp,**{n:xp.asarray(base[n]) for n in ['c1h','c2h','rdnw','fnm','fnp']})
    coord.msft=xp.asarray(base['msf']);refcoord=SimpleNamespace(**{n:base[n] for n in ['c1h','c2h','rdnw','fnm','fnp']})
    q=xp.asarray(base['q']);ru=xp.asarray(base['ru']);rv=xp.asarray(base['rv']);rw=xp.asarray(base['rw'])
    for order in [3,5]:
        got=xp.zeros(q.shape,np.float32)
        launch_flux_div_scalar(q,ru,rv,rw,got,coord,3000.,3000.,msf=coord.msft,has_msf=True,open_x=True,open_y=True,spec=True,vorder=order)
        ref=np_flux_div_scalar(base['q'],base['ru'],base['rv'],base['rw'],refcoord,3000.,3000.,msf=base['msf'],open_x=True,open_y=True,spec=True,vorder=order)
        delta=got.to_numpy().astype(float)-ref
        bound=float(64*np.finfo(np.float32).eps*max(float(np.abs(ref).max()),1e-7))
        assert float(np.abs(delta).max())<bound,(order,float(np.abs(delta).max()),bound)
        rows.append({'case':'captured_unlimited','vorder':order,'max_abs':float(np.abs(delta).max()),'rmse':float(np.sqrt(np.mean(delta*delta))),'fp32_bound':bound})
    nz,ny,nx=q.shape
    for order,target_cr in [(3,None),(5,None),(5,2.25),(5,-2.25),(5,3.75),(5,-3.75)]:
        raw=base['rw'].copy()
        if target_cr is not None:
            for k in range(2,nz-1):
                dz=np.float32(2)/(base['rdnw'][k]+base['rdnw'][k-1]);mass=base['c1h'][k]*base['mu']+base['c2h'][k]
                raw[k]=np.float32(target_cr)*dz*mass/np.float32(15)
            raw[1]=raw[nz-1]=0
        device_rw=xp.asarray(raw);q0=xp.asarray(base['q']);mu=xp.asarray(base['mu']);mu_old=xp.asarray(base.get('mu_old',base['mu']))
        shapes=[(nz,ny,nx+1)]*2+[(nz,ny+1,nx)]*2+[(nz+1,ny,nx)]*2
        flux=[xp.zeros(s,np.float32) for s in shapes]
        launch_pd_fluxes(q,q0,ru,rv,device_rw,mu,coord,3000.,3000.,15.,*flux,msft=coord.msft,has_msf=True,open_x=True,open_y=True,vorder=order)
        refs=np_pd_fluxes(base['q'],base['q'],base['ru'],base['rv'],raw,base['mu'],refcoord,3000.,3000.,15.,msft=base['msf'],open_x=True,open_y=True,vorder=order)
        flux_metrics=[]
        for name,dev,ref in zip(['fxl','fxc','fyl','fyc','fzl','fzc'],flux,refs):
            diff=dev.to_numpy().astype(float)-ref
            pair=['fxl','fxc','fyl','fyc','fzl','fzc'].index(name)//2
            scale=max(float(np.abs(refs[2*pair]).max()),float(np.abs(refs[2*pair]+refs[2*pair+1]).max()),1e-10)
            bound=float(64*np.finfo(np.float32).eps*scale)
            assert float(np.abs(diff).max())<bound,(name,order,target_cr,float(np.abs(diff).max()),bound)
            flux_metrics.append({'field':name,'max_abs':float(np.abs(diff).max()),'rmse':float(np.sqrt(np.mean(diff*diff))),'fp32_bound':bound})
        tend=xp.zeros(q.shape,np.float32)
        launch_pd_renorm_apply(q0,mu_old,*flux,tend=tend,coord=coord,dx=3000.,dy=3000.,dt=15.,msft=coord.msft,has_msf=True,open_x=True,open_y=True)
        ref=np_pd_renorm_apply(base['q'],base.get('mu_old',base['mu']),*refs,refcoord,3000.,3000.,15.,msft=base['msf'],open_x=True,open_y=True)
        diff=tend.to_numpy().astype(float)-ref
        bound=float(128*np.finfo(np.float32).eps*max(float(np.abs(ref).max()),1e-7))
        assert float(np.abs(diff).max())<bound,(order,target_cr,float(np.abs(diff).max()),bound)
        rows.append({'case':'captured_pd' if target_cr is None else 'sl_signed_branch_control','vorder':order,'target_vertical_courant':target_cr,'flux_metrics':flux_metrics,'limited_tendency_max_abs':float(np.abs(diff).max()),'limited_tendency_rmse':float(np.sqrt(np.mean(diff*diff))),'fp32_bound':bound})
result={'status':'PASS','method':'Identical full-domain captured step1 inputs, existing float64 unlimited/PD flux and donor-budget renormalization mirrors; order3/5, specified map-weighted geometry. Signed SL controls replace only rw on interior eta faces with prescribed |Courant|>1; they are branch controls, not physically integrated forecasts. Bounds scale from FP32 epsilon and the parent flux/tendency magnitude.','cases':rows}
a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
