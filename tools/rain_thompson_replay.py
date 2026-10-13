#!/usr/bin/env python3
"""Replay captured Thompson component inputs on Metal or the CUDA-source host shim.

The host shim executes unchanged CUDA source with CPU libm, so it diagnoses
translation defects rather than certifying NVIDIA floating-point equivalence.
Run host first, then Metal against its saved component outputs.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--capture-root',type=Path,required=True)
p.add_argument('--tables',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--backend',choices=['host','metal'],required=True)
p.add_argument('--reference-root',type=Path)
p.add_argument('--stages',default='cold,sources,condensation,rain_evaporation,sedimentation,cleanup,finalize')
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
if a.backend=='host':
 sys.path.insert(0,str(ROOT/'tools/thompson_real_column_parity'))
 import host_backend
 host_backend.install()
 import cupy as xp
 context=None
else:
 from woof.metal.runtime import MetalContext
 from woof.metal.array import MetalArrayModule
 context=MetalContext();xp=MetalArrayModule(context)
from woof.core import thompson
from woof.core.thompson_runtime import load_classic_device_tables
owner=load_classic_device_tables(a.tables,backend=xp,cache=False)
records=[]
def host(v):return v.to_numpy() if hasattr(v,'to_numpy') else np.asarray(v)
try:
 for stage in a.stages.split(','):
  path=a.capture_root/(stage+'-parameters.json')
  if not path.exists():continue
  meta=json.loads(path.read_text());raw=dict(np.load(a.capture_root/(stage+'-inputs.npz')))
  runs=[]
  for repeat in range(3 if context else 1):
   dev={n:xp.asarray(v.copy()) for n,v in raw.items()}
   objects={n:owner for n in meta['objects'] if n=='table_owner'}
   if len(objects)!=len(meta['objects']):raise ValueError(meta['objects'])
   getattr(thompson,meta['function'])(**dev,**meta['scalars'],**objects)
   runs.append({n:host(v).copy() for n,v in dev.items()})
  np.savez(a.output/(stage+'-outputs.npz'),**runs[0])
  if a.reference_root:
   reference=dict(np.load(a.reference_root/(stage+'-outputs.npz')))
   previous=dict(np.load(a.capture_root/(stage+'-actual-outputs.npz')))
   metrics=[]
   for n,v in runs[0].items():
    d=v.astype(float)-reference[n];old=previous[n].astype(float)-reference[n]
    metrics.append({'field':n,'old_rmse':float(np.sqrt(np.mean(old*old))),'new_rmse':float(np.sqrt(np.mean(d*d))),'old_max_abs':float(np.abs(old).max()),'new_max_abs':float(np.abs(d).max()),'repeat_identical':all(np.array_equal(v,r[n]) for r in runs[1:])})
   records.append({'operator':stage,'metrics':metrics})
   print(stage,json.dumps([r for r in metrics if r['field'] in ['qc','qs','qg','qv','temperature']]),flush=True)
  (a.output/'comparison.json').write_text(json.dumps(records,indent=2)+'\n')
finally:
 if context:context.close()
