"""Compare repeated typed Noah column calls on a finite fixture subset."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/thompson_real_column_parity'))
import host_backend
from tests.physics_reference.noah_oracle import load_noah_oracle,FIXTURE_DT,FIXTURE_DZS
from woof.core import kernels
from woof.core.noah import _F2D,_F3D,load_tables,pack_params
from woof.metal.noah_family import submit_noah_column
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED
assert ENABLED
source='#define __launch_bounds__(...)\n'+kernels.module_source('noah')
k=host_backend.HostModule('strict_noah_land',source).get_function('noah_column')
f=load_noah_oracle('noah-lsm.csv');indices=[0,42,55,67]
ref={'ivgtyp':np.ascontiguousarray(f.ivgtyp[:,indices]),'isltyp':np.ascontiguousarray(f.isltyp[:,indices])}
for n,v in f.inputs.items():ref[n]=np.ascontiguousarray(v[...,indices])
shape=(1,len(indices))
for n in ('noahres','reslin','chklowq'):ref[n]=np.zeros(shape,np.float32)
ref['smcrel']=np.zeros((4,*shape),np.float32);ref['ebal']=np.zeros(shape,np.int32)
p=pack_params(load_tables());tables=[np.asarray(p.veg,dtype=np.float32).reshape(p.lucats,15),np.asarray(p.soil,dtype=np.float32).reshape(p.slcats,10),np.asarray(p.gen,dtype=np.float32),np.asarray(FIXTURE_DZS,dtype=np.float32)]
words=diff=0
with MetalContext() as context:
    xp=MetalArrayModule(context);dev={name:xp.asarray(value.copy()) for name,value in ref.items()}
    for step in range(2,5):
        args=[ref['ivgtyp'],ref['isltyp']]+[ref[name] for name in _F2D]+[ref[name] for name in _F3D]+[ref['ebal']]+tables
        args += [np.float32(FIXTURE_DT),np.int32(p.lucats),np.int32(p.slcats),np.int32(13),np.int32(15),np.float32(.5),np.int32(step),np.int32(0),np.int32(0),np.int32(0),np.int32(1),np.int32(shape[0]),np.int32(shape[1])]
        k((1,),(64,),args)
        submit_noah_column(dev,p,FIXTURE_DT,FIXTURE_DZS,itimestep=step)
        for name in (*_F2D,*_F3D,'ebal'):
            a=dev[name].to_numpy();b=ref[name]
            if a.dtype==np.float32 and (not np.isfinite(a).all() or not np.isfinite(b).all()):raise AssertionError(f'nonfinite at step {step}: {name}')
            ab=a.view(np.uint32).ravel();bb=b.view(np.uint32).ravel();bad=np.flatnonzero(ab!=bb);words+=ab.size;diff+=len(bad)
            if len(bad):print(f'step={step} {name} differences={len(bad)} first={bad[0]} host=0x{int(bb[bad[0]]):08x} metal=0x{int(ab[bad[0]]):08x}')
        print(f'step={step} cases={[f.cases[i] for i in indices]}')
print(f'words={words} different_words={diff} finite=true reference=selected-CUDA-source-compiled-on-host')
if diff:raise SystemExit(1)
