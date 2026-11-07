"""Run stored WRF lake columns through production Metal init and step."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.lake_wrf461_oracle.compare import metric
from woof.core.lake_schema import LAKE_OUTPUT_LAYOUT, LAKE_STATE_LAYOUT
from woof.metal.array import MetalArrayModule
from woof.metal.lake_family import initialize_lake_metal
from woof.metal.runtime import MetalContext

p=argparse.ArgumentParser();p.add_argument('--fixture',type=Path,required=True);p.add_argument('--host-reference',type=Path);p.add_argument('--steps',type=int,default=1);p.add_argument('--out',type=Path,required=True);p.add_argument('--zero-precip',action='store_true');p.add_argument('--oracle-reseed',action='store_true');p.add_argument('--snapshot',type=Path);a=p.parse_args()
path=a.fixture
assert hashlib.sha256(path.read_bytes()).hexdigest()=='5f9bbb0c9b69f3b1a61690f4924ba1bb93f9ea84065771e1e7302a882d792d7a'
o=np.load(path);seed=o['seed'];forcing=o['forcing'];n=seed.shape[1];shape=(1,n)
with MetalContext() as context:
    context.batch_kernel_calls=False
    submit=context._submit_kernel_batch
    def sync(*args,**kw):
        result=submit(*args,**kw);result.wait();return result
    context._submit_kernel_batch=sync
    xp=MetalArrayModule(context)
    arr=lambda v: xp.asarray(np.asarray(v,dtype=np.float32).reshape(shape))
    fields={'lakemask':arr(np.ones(n)), 'isltyp':xp.asarray(seed[0].astype(np.int32).reshape(shape)),
            'tsk':arr(seed[2]),'snow':arr(seed[3]),'xice':arr(seed[4]),
            'ivgtyp':xp.asarray(np.full(shape,17,np.int32)),'xland':arr(np.full(n,2.))}
    for name,_ in LAKE_OUTPUT_LAYOUT:
        if name not in fields: fields[name]=arr(np.zeros(n))
    fields.update(glw=arr(forcing[7]),emiss=arr(forcing[8]),swdown=arr(forcing[10]))
    lake=initialize_lake_metal(fields,latitude=forcing[12].reshape(shape),lake_depth=seed[1].reshape(shape))
    initial=fields['lake_columns'].to_numpy().reshape(131,n);static=fields['lake_static'].to_numpy().reshape(71,n)
    result={'fixture_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'dt_seconds':30.,
            'forcing': 'zero-precipitation control' if a.zero_precip else 'stored fixture',
            'oracle_reseed': a.oracle_reseed,
            'initial_state':metric(o['initial_state'],initial),'static':metric(o['reference_static'],static),'steps':[]}
    atmosphere={'temperature':xp.asarray(forcing[0].reshape(1,*shape)),
       'p_interface':xp.asarray(forcing[1:3].reshape(2,*shape)), 'dz':xp.asarray(forcing[3].reshape(1,*shape)),
       'qv':xp.asarray(forcing[4].reshape(1,*shape)), 'u':xp.asarray(forcing[5].reshape(1,*shape)),
       'v':xp.asarray(forcing[6].reshape(1,*shape))}
    # Two equal halves reconstruct the exact stored accumulated rainbl words.
    pulse=np.zeros(n,np.float32) if a.zero_precip else forcing[9]
    precipitation=arr(pulse*np.float32(.5)); pending=arr(pulse*np.float32(.5))
    if a.host_reference:
        result['host_reference_sha256']=hashlib.sha256(a.host_reference.read_bytes()).hexdigest()
        from tools.lake_wrf461_oracle.compare import ptr
        dll=ctypes.CDLL(str(a.host_reference))
        fp=ctypes.POINTER(ctypes.c_float);ip=ctypes.POINTER(ctypes.c_int)
        dll.lake_step_columns.argtypes=[ctypes.c_int,fp,fp,fp,fp,ctypes.c_float,ctypes.c_float,ip,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    for i in range(a.steps):
        if a.oracle_reseed and i:
            fields['lake_columns'][...] = o['trace_state'][i-1].reshape(131,*shape)
        if a.host_reference:
            host_state=fields['lake_columns'].to_numpy().reshape(131,n)
            host_static=static.copy();host_out=np.zeros((9,n),np.float32);host_errors=np.zeros(n,np.int32)
            dll.lake_step_columns(n,ptr(forcing),ptr(host_state),ptr(host_static),ptr(host_out),np.float32(30),np.float32(.5),ptr(host_errors),0,n,None)
            assert not host_errors.any()
        fields['albedo'][...] = forcing[11].reshape(shape)
        lake.step(fields,atmosphere,dt=30.,precipitation=precipitation,pending_rainbl=pending)
        state=fields['lake_columns'].to_numpy().reshape(131,n)
        output=np.stack([fields[name].to_numpy().reshape(n) for name,_ in LAKE_OUTPUT_LAYOUT])
        offset=0;state_fields={}
        for name,count in LAKE_STATE_LAYOUT:
            state_fields[name]=metric(o['trace_state'][i,offset:offset+count],state[offset:offset+count]);offset+=count
        if a.host_reference:
            result.setdefault('host_source',[]).append({'state':metric(host_state,state),'output':metric(host_out,output)})
        result['steps'].append({'step':i+1,'state_fields':state_fields,'state':metric(o['trace_state'][i],state),
          'output':metric(o['trace_output'][i],output),
          'output_fields':{name:metric(o['trace_output'][i,j],output[j]) for j,(name,_) in enumerate(LAKE_OUTPUT_LAYOUT)}})
if a.snapshot: np.savez(a.snapshot,state=state,output=output)
a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
