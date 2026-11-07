"""Compare standalone serial cloud with host-compiled CUDA source.

This does not grade the production shallow combined parallel kernel.
"""
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from woof.wrf_exact import ENABLED,effective_options
if not ENABLED: raise SystemExit('GPUWM_WRF_EXACT=1 is required')
from woof.core.kernels import module_source,module_options
from woof.core.nssl2_driver_support import NSSL2DriverWorkspace, launch_cloud_sedimentation
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
src=module_source('nssl2_driver_support');a=src.index('template <int KMAX>\n__device__ __forceinline__ void nssl2_cloud_sediment_impl(');b=src.index('// One category runs in each block',a)
shim=ROOT/'tools/thompson_real_column_parity/cuda_host_shim.h';glibc=ROOT/'woof/core/kernels/glibc_flt32.cuh';header=(ROOT/'woof/core/kernels/nssl2_wrf_math.cuh').read_text()
macros='\n'.join(line for line in header.splitlines() if line.startswith(('#define NSSL2_WRF_CWMASN ','#define NSSL2_WRF_CWMASX ')))
wrap='''\nextern "C" void run(const float *rho,const float *temperature,float *qc,float *nc,const float *dz,float *exported,float dt,int nz,int ny,int nx){for(int c=0;c<ny*nx;++c){blockIdx.x=c;if(nz<=64)nssl2_cloud_sediment_impl<64>(rho,temperature,qc,nc,dz,exported,dt,nz,ny,nx);else nssl2_cloud_sediment_impl<256>(rho,temperature,qc,nc,dz,exported,dt,nz,ny,nx);}}\n'''
with tempfile.TemporaryDirectory(prefix='woof-nssl-cloud-sed-') as tmp:
 tmp=Path(tmp);(tmp/'source.cpp').write_text(f'#include "{shim}"\n#include "{glibc}"\n'+macros+'\n#define IDX3(k,j,i) ((size_t(k)*ny+(j))*nx+(i))\n'+src[a:b]+wrap)
 subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(tmp/'source.cpp'),'-o',str(tmp/'source.dylib')],check=True)
 f=ctypes.CDLL(str(tmp/'source.dylib')).run;fp=ctypes.POINTER(ctypes.c_float);f.argtypes=[fp]*6+[ctypes.c_float]+[ctypes.c_int]*3;ptr=lambda x:x.ctypes.data_as(fp)
 results=[]
 with MetalContext() as ctx:
  ctx.batch_kernel_calls=False;submit=ctx._submit_kernel_batch
  def sync(*a,**kw):
   result=submit(*a,**kw);result.wait();return result
  ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
  for nz in (4,64,65,128,129,256):
   shape=(nz,1,8);rng=np.random.default_rng(1800+nz)
   rho=rng.uniform(.5,1.4,shape).astype(np.float32);temp=rng.uniform(230,305,shape).astype(np.float32);dz=rng.uniform(.8,2.0,shape).astype(np.float32)
   qc=np.power(10.,rng.uniform(-14,-2,shape)).astype(np.float32);nc=np.power(10.,rng.uniform(-10,9,shape)).astype(np.float32);qc[:,:,0]=0;nc[:,:,1]=0
   qc.flat[2]=np.nextafter(np.float32(0),np.float32(1));nc.flat[2]=np.nextafter(np.float32(0),np.float32(1))
   for dt in (.5,10.,30.):
    refq=qc.copy();refn=nc.copy();refout=np.full(shape[1:],-7,np.float32)
    f(*map(ptr,(rho,temp,refq,refn,dz,refout)),np.float32(dt),nz,1,8)
    dr,dtp,dd=[xp.asarray(x) for x in (rho,temp,dz)];out=xp.asarray(np.full(shape[1:],-7,np.float32))
    packed=np.full((16,*shape),-19,np.float32);packed[1]=qc;packed[7]=nc
    state=xp.asarray(packed);exports=xp.asarray(np.full((5,*shape[1:]),-23,np.float32))
    workspace=NSSL2DriverWorkspace(state=state,category_surface_export=exports,shape=shape,ignored_accumulator=out)
    launch_cloud_sedimentation(workspace,dr,dtp,dd,dt)
    actual_state=state.to_numpy()
    assert np.array_equal(actual_state[[i for i in range(16) if i not in (1,7)]],packed[[i for i in range(16) if i not in (1,7)]])
    assert np.all(exports.to_numpy()==np.float32(-23))
    actual=[actual_state[1],actual_state[7],out.to_numpy()];expected=[refq,refn,refout]
    diff=[int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))) for a,b in zip(expected,actual)]
    results.append({'nz':nz,'dt':dt,'words':2*qc.size+8,'different':diff,'active_exports':int(np.count_nonzero(refout))})
 print(json.dumps({'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'effective_options':effective_options(module_options('nssl2_driver_support')),'cases':results},indent=2))
 if any(v for r in results for v in r['different']):raise SystemExit(1)
