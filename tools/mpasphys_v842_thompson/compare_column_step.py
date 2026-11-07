"""Compare one native Thompson column with the selected source and certified tables."""
from pathlib import Path
import sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys import thompson_v842 as native
from woof.hex.mpasphys.resources_build import _vendored_maths,thompson_maths
import os
from woof.hex.mpasphys.metal_thompson_tables_v842 import load_raw_tables
from woof.hex.mpasphys.metal_thompson_column_v842 import step
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
resources={'table_directory':sys.argv[1]};nz=int(sys.argv[2]) if len(sys.argv)>2 else 4;nc=int(sys.argv[3]) if len(sys.argv)>3 else 1
if not 1 <= nz <= 256 or nc<1: raise ValueError('depth must be between 1 and 256 with nonempty columns')
raw=load_raw_tables(resources)
tables={n:v.view(np.float32 if n=='tnccn_act' else np.float64) for n,v in raw.items()}
declarations=','.join(f"const {'float' if n=='tnccn_act' else 'double'}* input_{n}" for n in native.TABLE_ARGUMENTS)
assignments='\n'.join(f'm.{n}=input_{n};' for n in native.TABLE_ARGUMENTS)
source=f'#define THOMPSON_COLUMN_KMAX {nz}\n'+thompson_maths(_vendored_maths())+'\n'+native._SUPPORT+native._COLUMN+native._DRIVER.replace('TABLE_DECLARATIONS',declarations).replace('TABLE_ASSIGNMENTS',assignments)
host=HostModule('native_thompson_step',source).get_function('mpas_thompson_step')
fields={name:np.zeros((nz,nc),np.float32) for name in native.FIELD_ARGUMENTS}
values={'qv':[.006,.004,.002,.001],'qc':[.0002,.0001,0,0],'qi':[0,.00001,.00002,.00001],
        'qr':[.0001,.00008,0,0],'qs':[0,.00001,.00002,.00001],'qg':[0,0,.00001,.00001],
        'ni':[500,900,1200,500],'nr':[2000,2000,1000,500],'th':[292,285,275,265],
        'pii':[.98,.95,.91,.87],'p':[99000,90000,80000,70000],'w':[.1,.2,-.1,0],'dz':[400,500,600,700]}
for name,v in values.items():
    base=np.interp(np.linspace(0,3,nz),np.arange(4),v)
    for col in range(nc):
        fields[name][:,col]=base*(1+0.1*col if name in ('qv','qc','qi','qr','qs','qg') else 1)
surface={name:np.zeros((nc,),np.float32) for name in native.SURFACE_ARGUMENTS};surface['ntc'][:]=100e6;surface['muc'][:]=0
want={name:v.copy() for name,v in fields.items()};want_surface={name:v.copy() for name,v in surface.items()}
host(((nc+31)//32,),(32,),tuple(want[n] for n in native.FIELD_ARGUMENTS)+tuple(want_surface[n] for n in native.SURFACE_ARGUMENTS)+tuple(map(np.int32,(nz,nc)))+(np.float32(1),)+tuple(map(np.int32,(1,1,1)))+tuple(tables[n] for n in native.TABLE_ARGUMENTS))
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        started=time.monotonic();result=submit(*args,**kwargs);result.wait();print('dispatch seconds',time.monotonic()-started,flush=True);return result
    if not os.environ.get("THOMPSON_ASYNC"):context._submit_kernel_batch=sync
    xp=MetalArrayModule(context)
    device_tables={n:xp.asarray(v.view(np.float32) if n=='tnccn_act' else v) for n,v in raw.items()};device_tables['_file_sha256']=native.TABLE_FILE_SHA256.copy()
    bad_total=0
    for attempt in range(2):
        print("attempt",attempt,flush=True)
        f={n:xp.asarray(v) for n,v in fields.items()};s={n:xp.asarray(v) for n,v in surface.items()}
        if os.environ.get('THOMPSON_POISON'):
            from woof.hex import metal_dynamics_v841 as dynamics
            original_arrays=dynamics._arrays
            class PoisonedArrays:
                def __init__(self,ctx):self.xp=original_arrays(ctx)
                def empty(self,shape,dtype):
                    result=self.xp.empty(shape,dtype)
                    result.buffer.write(bytes([0xa5])*(result.size*result.dtype.itemsize),offset=result.byte_offset)
                    return result
            dynamics._arrays=PoisonedArrays
        try:step(f,s,device_tables,dt=1)
        finally:
            if os.environ.get('THOMPSON_POISON'):dynamics._arrays=original_arrays
        bad=words=0
        for group,expected in ((f,want),(s,want_surface)):
            for name,ref in expected.items():
                got=group[name].to_numpy();ix=np.flatnonzero(got.view(np.uint32)!=ref.view(np.uint32));bad+=len(ix);words+=ref.size
                if len(ix):print(name,'differences',len(ix),'first',[(int(i),hex(int(got.view(np.uint32).flat[i])),hex(int(ref.view(np.uint32).flat[i]))) for i in ix[:5]],flush=True)
        print('native Thompson words',words,'differences',bad,flush=True)
        bad_total+=bad
    if bad_total:raise SystemExit(1)
