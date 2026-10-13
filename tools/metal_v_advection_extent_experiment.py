#!/usr/bin/env python3
"""Replay captured V advection on identical FP32 inputs against float64.

Consumes a step-1 capture exported as q,ru,rv,rw,tend,rdnw,fnm,fnp,msf,
and baseline. Uses the public Metal launcher at orders 3 and 5; no forecast.
The float64 mirror is a CUDA source transcription, not an executed CUDA GPU.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from woof.core.advection import launch_flux_div_v
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from tests.physics_reference.npref import np_flux_div_v

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 with np.load(a.inputs) as f:inputs={n:f[n].copy() for n in f.files}
 nz,nyp1,nx=inputs['q'].shape;ny=nyp1-1
 coord=SimpleNamespace(**{n:inputs[n] for n in ['rdnw','fnm','fnp']})
 results=[]
 with MetalContext() as context:
  xp=MetalArrayModule(context)
  dev={n:xp.asarray(inputs[n]) for n in ['q','ru','rv','rw','rdnw','fnm','fnp','msf']}
  dc=SimpleNamespace(**{n:dev[n] for n in ['rdnw','fnm','fnp']})
  for order in [5,3]:
   reference=np_flux_div_v(inputs['q'],inputs['ru'],inputs['rv'],inputs['rw'],coord,3000.,3000.,open_x=True,open_y=True,msf=inputs['msf'],spec=True,vorder=order)
   target=xp.asarray(inputs['tend'])
   launch_flux_div_v(dev['q'],dev['ru'],dev['rv'],dev['rw'],target,dc,3000.,3000.,open_x=True,open_y=True,msf=dev['msf'],has_msf=True,spec=True,vorder=order)
   got=target.to_numpy().astype(float)-inputs['tend'].astype(float)
   d=got-reference
   gross=float(np.max(np.abs(inputs['q']))*(np.max(np.abs(inputs['ru']))+np.max(np.abs(inputs['rv'])))/3000 + np.max(np.abs(inputs['q']))*np.max(np.abs(inputs['rw']))*np.max(np.abs(inputs['rdnw'])))
   bound=64*np.finfo(np.float32).eps*gross
   assert np.max(np.abs(d))<=bound,(order,np.max(np.abs(d)),bound)
   # Entire last level must be dispatched. Excluded normal faces remain zero.
   assert np.all(got[-1,[0,-1],:]==0)
   top_difference=got[-1]-reference[-1]
   row={'vorder':order,'shape':[nz,nyp1,nx],'grid_size_requested':inputs['q'].size,'max_abs':float(np.abs(d).max()),'rmse':float(np.sqrt(np.mean(d*d))),'fp32_parent_bound':float(bound),'top_level_max_abs':float(np.abs(top_difference).max()),'top_level_rmse':float(np.sqrt(np.mean(top_difference**2))),'top_level_nonzero':int(np.count_nonzero(got[-1]))}
   if order==5:
    original=inputs['baseline'].astype(float)-inputs['tend'].astype(float);old=original-reference
    row['baseline_max_abs']=float(np.abs(old).max());row['baseline_rmse']=float(np.sqrt(np.mean(old*old)));row['baseline_top_level_nonzero']=int(np.count_nonzero(original[-1]))
    row['rms_reduction_factor']=row['baseline_rmse']/row['rmse']
   results.append(row)
 result={'status':'PASS','reference':'Existing float64 np_flux_div_v CUDA source transcription; actual CUDA execution unavailable','same_FP32_inputs':True,'cases':results,'full_forecasts':0}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
