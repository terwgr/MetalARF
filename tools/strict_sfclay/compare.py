"""Compare revised MM5 caller with host-compiled selected CUDA source."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity import host_backend
host_backend.install()
from woof.core import kernels
from woof.core.sfclay import SFCLAY_OUTPUTS, SFClayResult, launch_sfclay
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED, effective_options

assert ENABLED
source=kernels.module_source('sfclay')
reference=host_backend.HostModule('sfclay_selected',source).get_function('sfclay_column')
N=24
x=np.arange(N,dtype=np.float32)
shape=(3,8)
def arr(v): return np.broadcast_to(np.asarray(v,dtype=np.float32),shape).copy()
inputs=[arr(v) for v in (3.0+0.08*x.reshape(shape), 1.5, 278.+.5*x.reshape(shape),.005,
                         92500., 30., 99000., 275.+.75*x.reshape(shape),700.,.7,
                         np.where((x.reshape(shape)%2)==0,1.,2.),0.)]
initial={name:arr(0.) for name in SFCLAY_OUTPUTS}
initial.update(znt=arr(.05),ust=arr(.2),ustm=arr(.15),mol=arr(.01),hfx=arr(20.),
               qfx=arr(.00001),qsfc=arr(.005),zol=arr(.02),chs=arr(.13),chs2=arr(.12),cqs2=arr(.11))
with MetalContext() as context:
    context.batch_kernel_calls=False
    submit=context._submit_kernel_batch
    def sync(*a,**kw):
        r=submit(*a,**kw);r.wait();return r
    context._submit_kernel_batch=sync
    xp=MetalArrayModule(context)
    total=bad=0; cases={}
    for isfflx in (0,1):
      for isftcflx,iz0tlnd in ((0,0),(1,1),(2,2)):
        host={k:v.copy() for k,v in initial.items()}
        device={k:xp.asarray(v) for k,v in initial.items()}
        din=[xp.asarray(v) for v in inputs]
        result=SFClayResult(**device)
        for step in range(3):
            reference(((N+127)//128,),(128,),tuple(inputs)+tuple(host[k] for k in SFCLAY_OUTPUTS)+
                      (np.float32(1000.),np.int32(1),np.int32(isfflx),np.int32(isftcflx),np.int32(iz0tlnd),np.int32(N)))
            launch_sfclay(*din,result,option=1,dx=1000.,isfflx=bool(isfflx),isftcflx=isftcflx,iz0tlnd=iz0tlnd)
            for name in SFCLAY_OUTPUTS:
                want=host[name].view(np.uint32); got=device[name].to_numpy().view(np.uint32)
                count=int(np.count_nonzero(want!=got)); total+=want.size;bad+=count
                if count: cases[f'{isfflx}/{isftcflx}/{iz0tlnd}/{step}/{name}']={'count':count,'first':int(np.flatnonzero(want!=got)[0]),'want':int(want.flat[np.flatnonzero(want!=got)[0]]),'got':int(got.flat[np.flatnonzero(want!=got)[0]])}
    # A source-defined rescue input must refuse before changing any in/out field.
    for roughness in (np.float32(0),np.float32(1e-40)):
        seeded={k:xp.asarray(v) for k,v in initial.items()};value=initial['znt'].copy();value[0,0]=roughness
        seeded['znt']=xp.asarray(value); before={k:v.to_numpy().view(np.uint32).copy() for k,v in seeded.items()}
        try: launch_sfclay(*(xp.asarray(v) for v in inputs),SFClayResult(**seeded),option=1,dx=1000.)
        except NotImplementedError: pass
        else: raise AssertionError('roughness rescue was not refused')
        assert all(np.array_equal(before[k],v.to_numpy().view(np.uint32)) for k,v in seeded.items())
    # Exercise the real PhysicsDriver surface caller with its actual level-zero
    # views and shared fields. The surrounding strict physics gate stays closed.
    from woof.core.physics import PhysicsDriver
    cfg=SimpleNamespace(sf_sfclay_physics=1,dx=1000.,isfflx=1,isftcflx=0,iz0tlnd=0)
    fields={k:xp.asarray(v) for k,v in initial.items()}
    fields.update({name:xp.asarray(inputs[i]) for name,i in
                   (('tsk',7),('pblh',8),('mavail',9),('xland',10),('lakemask',11))})
    atm={name:xp.asarray(np.stack([inputs[i]]*(3 if name=='p_interface' else 2)))
         for name,i in (('u',0),('v',1),('temperature',2),('qv',3),('pressure',4),('dz',5),('p_interface',6))}
    state=SimpleNamespace(_metal_setup_state=True,mup=xp.asarray(arr(0)),p=atm['pressure'])
    driver=PhysicsDriver.__new__(PhysicsDriver)
    driver.state=state;driver.fields=fields;driver.ruc_params=None
    driver.sfclay_result=SFClayResult(**{name:fields[name] for name in SFCLAY_OUTPUTS})
    host={k:v.copy() for k,v in initial.items()}
    for step in range(3):
        reference(((N+127)//128,),(128,),tuple(inputs)+tuple(host[k] for k in SFCLAY_OUTPUTS)+
                  (np.float32(1000.),np.int32(1),np.int32(1),np.int32(0),np.int32(0),np.int32(N)))
        driver._run_sfclay(atm,cfg)
        for name in SFCLAY_OUTPUTS:
            want=host[name].view(np.uint32);got=fields[name].to_numpy().view(np.uint32)
            count=int(np.count_nonzero(want!=got));total+=want.size;bad+=count
            if count: cases[f'driver/{step}/{name}']={'count':count}
    assert all(getattr(driver.sfclay_result,k) is fields[k] for k in SFCLAY_OUTPUTS)
    before={k:fields[k].to_numpy().view(np.uint32).copy() for k in SFCLAY_OUTPUTS}
    driver.ruc_params=object()
    try: driver._run_sfclay(atm,cfg)
    except NotImplementedError: pass
    else: raise AssertionError('strict RUC wrapper was not refused')
    assert all(np.array_equal(before[k],fields[k].to_numpy().view(np.uint32)) for k in SFCLAY_OUTPUTS)
report={'reference':'selected CUDA source compiled for host; not NVIDIA device','source_sha256':hashlib.sha256(source.encode()).hexdigest(),'cuda_options':list(effective_options(kernels.module_options('sfclay'))),'words':total,'different_words':bad,'cases':cases,'roughness_rescue_outputs_unchanged':True,'ruc_wrapper_outputs_unchanged':True}
report_json = json.dumps(report, indent=2)
print(report_json)
report_path = os.environ.get('SFCLAY_REPORT')
if report_path:
    Path(report_path).write_text(report_json + '\n')
if bad: raise SystemExit(1)
