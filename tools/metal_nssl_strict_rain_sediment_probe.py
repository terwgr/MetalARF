"""Compare standalone serial rain with host-compiled selected CUDA source.

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
from woof.core.nssl2_driver_support import NSSL2DriverWorkspace,launch_rain_sedimentation
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
src=module_source('nssl2_driver_support');a=src.index('#define NSSL2_SED_ZFAC');b=src.index('template <bool ACCUMULATE = true>\n__device__ __forceinline__ void nssl2_rain_sediment_parallel_impl(',a)
shim=ROOT/'tools/thompson_real_column_parity/cuda_host_shim.h';glibc=ROOT/'woof/core/kernels/glibc_flt32.cuh';header=(ROOT/'woof/core/kernels/nssl2_wrf_math.cuh').read_text()
macros='\n'.join(line for line in header.splitlines() if line.startswith(('#define NSSL2_WRF_XVRMN ','#define NSSL2_WRF_XVRMX0 ','#define NSSL2_WRF_PI ','#define NSSL2_WRF_PIINV ')))
wrap='''\nextern "C" void run(const float *rho,float *qr,float *nr,const float *dz,float *acc,float *exported,float dt,int nz,int ny,int nx){for(int c=0;c<ny*nx;++c){blockIdx.x=c;if(nz<=64)nssl2_rain_sediment_impl<64>(rho,qr,nr,dz,acc,exported,dt,nz,ny,nx);else nssl2_rain_sediment_impl<256>(rho,qr,nr,dz,acc,exported,dt,nz,ny,nx);}}\n'''
with tempfile.TemporaryDirectory(prefix='woof-nssl-rain-sed-') as tmp:
 tmp=Path(tmp)
 body=src[a:b].replace('if (z > (double)z_advected && z > 0.0 && z_advected > z_initial) {','if (z > (double)z_advected && z > 0.0 && z_advected > z_initial) { ++corrections;')
 body=body.replace('substeps = 1;','++branches[0]; substeps = 1;',1).replace('substeps = max(2,','++branches[1]; substeps = max(2,',1).replace('substeps = 1 + (int)','++branches[2]; substeps = 1 + (int)',1)
 (tmp/'source.cpp').write_text(f'#include "{shim}"\n#include "{glibc}"\n'+macros+'\n#define IDX3(k,j,i) ((size_t(k)*ny+(j))*nx+(i))\nextern "C" { int corrections=0; int branches[3]={0,0,0}; }\n'+body+wrap)
 subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(tmp/'source.cpp'),'-o',str(tmp/'source.dylib')],check=True)
 lib=ctypes.CDLL(str(tmp/'source.dylib'));f=lib.run;corrections=ctypes.c_int.in_dll(lib,'corrections');branches=(ctypes.c_int*3).in_dll(lib,'branches');fp=ctypes.POINTER(ctypes.c_float);f.argtypes=[fp]*6+[ctypes.c_float]+[ctypes.c_int]*3;ptr=lambda x:x.ctypes.data_as(fp)
 results=[]
 with MetalContext() as ctx:
  ctx.batch_kernel_calls=False;submit=ctx._submit_kernel_batch
  def sync(*a,**kw):
   result=submit(*a,**kw);result.wait();return result
  ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
  for nz in (4,64,65,128,129,256):
   shape=(nz,1,8);rng=np.random.default_rng(2900+nz)
   rho=rng.uniform(.5,1.4,shape).astype(np.float32);dz=rng.uniform(20.,80.,shape).astype(np.float32)
   qr=np.power(10.,rng.uniform(-14,-2,shape)).astype(np.float32);nr=np.power(10.,rng.uniform(-10,9,shape)).astype(np.float32);qr[:,:,0]=0;nr[:,:,1]=0
   qr.flat[2]=np.nextafter(np.float32(0),np.float32(1));nr.flat[2]=np.nextafter(np.float32(0),np.float32(1))
   for dt in (.5,10.,30.):
    refq=qr.copy();refn=nr.copy();refacc=np.full(shape[1:],.125,np.float32);refout=np.full(shape[1:],-7,np.float32)
    before=corrections.value; before_branches=list(branches)
    f(*map(ptr,(rho,refq,refn,dz,refacc,refout)),np.float32(dt),nz,1,8)
    dr,dd=[xp.asarray(x) for x in (rho,dz)];acc=xp.asarray(np.full(shape[1:],.125,np.float32))
    packed=np.full((16,*shape),-19,np.float32);packed[2]=qr;packed[8]=nr
    state=xp.asarray(packed);exports=xp.asarray(np.full((5,*shape[1:]),-23,np.float32))
    workspace=NSSL2DriverWorkspace(state=state,category_surface_export=exports,shape=shape,ignored_accumulator=acc)
    launch_rain_sedimentation(workspace,dr,dd,dt)
    actual_state=state.to_numpy();actual_exports=exports.to_numpy()
    assert np.array_equal(actual_state[[i for i in range(16) if i not in (2,8)]],packed[[i for i in range(16) if i not in (2,8)]])
    assert np.all(actual_exports[1:]==np.float32(-23))
    actual=[actual_state[2],actual_state[8],acc.to_numpy(),actual_exports[0]];expected=[refq,refn,refacc,refout]
    diff=[int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))) for a,b in zip(expected,actual)]
    results.append({'nz':nz,'dt':dt,'words':2*qr.size+16,'different':diff,'active_exports':int(np.count_nonzero(refout)),'corrections':corrections.value-before,'substep_branches':[branches[i]-before_branches[i] for i in range(3)]})
 print(json.dumps({'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'effective_options':effective_options(module_options('nssl2_driver_support')),'cases':results},indent=2))
 if any(v for r in results for v in r['different']):raise SystemExit(1)
