"""Host-execute selected parallel CUDA source with 64 barrier-synchronized lanes."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from woof.core.kernels import module_source
from woof.core.nssl2_driver_support import NSSL2DriverWorkspace,launch_shallow_warm_sedimentation,reduce_nssl2_precipitation
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
source=module_source('nssl2_driver_support')
assert '__device__ __forceinline__ double nssl2_gamma_dp(double xx)' in source
source=source.replace('__device__ __forceinline__ double nssl2_gamma_dp(double xx)\n{','__device__ __forceinline__ double nssl2_gamma_dp(double xx)\n{ ++gamma_calls;',1)
shim=(ROOT/'tools/thompson_real_column_parity/cuda_host_shim.h').read_text().replace('static HostUint3 threadIdx =','static thread_local HostUint3 threadIdx =')
preamble='''#include <thread>\n#include <vector>\n#include <atomic>\n#include <mutex>\n#include <condition_variable>\nstruct Gate {std::mutex m; std::condition_variable cv; int count=0, gen=0; void wait(){std::unique_lock<std::mutex> lock(m); int g=gen; if(++count==64){count=0;++gen;cv.notify_all();}else cv.wait(lock,[&]{return gen!=g;});}}; static Gate gate; static std::atomic<int> gamma_calls{0};\n#define __shared__ static\n#define __syncthreads() gate.wait()\n'''
wrap='''\nextern "C" int run(const float*rho,const float*temp,float*state,const float*dz,float*ignored,float*exports,float dt,int nz,int ny,int nx){\n gamma_calls=0;double dummy[240]={};blockDim.x=64;blockIdx.z=0;\n for(int col=0;col<ny*nx;++col)for(int cat=0;cat<6;++cat){\n  blockIdx.x=col;blockIdx.y=cat;std::vector<std::thread> lanes;\n  for(int k=0;k<64;++k)lanes.emplace_back([&,k]{threadIdx.x=k;nssl2_sediment_all_parallel_64(rho,temp,state,dz,ignored,exports,dummy,dt,nz,ny,nx);});\n  for(auto &lane:lanes)lane.join();\n }return gamma_calls.load();}\n'''
wrap += '''\nextern "C" void reduce(const float*ex,const float*rho,float*a,float*b,float*c,float*d,float*e,float*f,float*g,float*h,float*i,float dt,int n){blockDim.x=1; for(int col=0;col<n;++col){blockIdx.x=col;threadIdx.x=0;nssl2_driver_reduce_precipitation(ex,rho,a,b,c,d,e,f,g,h,i,dt,n);}}\n'''
with tempfile.TemporaryDirectory(prefix='nssl-parallel-source-') as tmp:
 tmp=Path(tmp);(tmp/'source.cpp').write_text(preamble+shim+'\n'+source+wrap)
 subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-pthread','-fno-fast-math','-ffp-contract=off',str(tmp/'source.cpp'),'-o',str(tmp/'source.dylib')],check=True)
 f=ctypes.CDLL(str(tmp/'source.dylib')).run;fp=ctypes.POINTER(ctypes.c_float);f.argtypes=[fp]*6+[ctypes.c_float]+[ctypes.c_int]*3;f.restype=ctypes.c_int;red=ctypes.CDLL(str(tmp/'source.dylib')).reduce;red.argtypes=[fp]*11+[ctypes.c_float,ctypes.c_int];ptr=lambda x:x.ctypes.data_as(fp)
 results=[]
 with MetalContext() as ctx:
  ctx.batch_kernel_calls=False;submit=ctx._submit_kernel_batch
  def sync(*a,**kw):
   v=submit(*a,**kw);v.wait();return v
  ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
  for nz in (4,17,64):
   shape=(nz,1,2);rng=np.random.default_rng(876+nz);rho=rng.uniform(.6,1.4,shape).astype(np.float32);temperature=rng.uniform(240,305,shape).astype(np.float32);dz=rng.uniform(5,25,shape).astype(np.float32)
   initial=np.zeros((16,*shape),np.float32);initial[0]=.01
   for i in (1,2,3,4):initial[i]=np.power(10.,rng.uniform(-14,-3,shape)).astype(np.float32)
   for i in (7,8,9,10,11,12,14,15):initial[i]=np.power(10.,rng.uniform(-9,5,shape)).astype(np.float32)
   state=initial.copy();ignored=np.full(shape[1:],-11,np.float32);exports=np.full((5,*shape[1:]),-23,np.float32)
   precip=[rng.uniform(0,1,shape[1:]).astype(np.float32) for _ in range(9)]
   dr,dtp,dd=[xp.asarray(x) for x in (rho,temperature,dz)];ds=xp.asarray(initial);di=xp.asarray(ignored);de=xp.asarray(exports);dp=[xp.asarray(x) for x in precip]
   ws=NSSL2DriverWorkspace(state=ds,shape=shape,category_surface_export=de,ignored_accumulator=di)
   for dt in (10.,30.):
    ignored.fill(0)
    calls=f(ptr(rho),ptr(temperature),ptr(state),ptr(dz),ptr(ignored),ptr(exports),np.float32(dt),nz,1,2)
    assert calls==0 and np.all(exports[3:]==0)
    red(ptr(exports),ptr(rho),*[ptr(x) for x in precip],np.float32(dt),2)
    launch_shallow_warm_sedimentation(ws,dr,dtp,dd,dt)
    reduce_nssl2_precipitation(ws,*dp,air_density=dr,dt_s=dt)
    outputs=[ds.to_numpy(),di.to_numpy(),de.to_numpy(),*[x.to_numpy() for x in dp]];refs=[state,ignored,exports,*precip]
    differences=[int(np.count_nonzero(x.view(np.uint32)!=y.view(np.uint32))) for x,y in zip(refs,outputs)]
    results.append({'nz':nz,'dt':dt,'words':sum(x.size for x in refs),'different':differences,'gamma_calls':calls})
    # Reconstruct the component state between steps, simulating saved workspace carry.
    ds,di,de=[xp.asarray(x) for x in outputs[:3]];dp=[xp.asarray(x) for x in outputs[3:]]
    ws=NSSL2DriverWorkspace(state=ds,shape=shape,category_surface_export=de,ignored_accumulator=di)
   for plane in (5,6):
    bad=initial.copy();bad[plane,0,0,0]=1e-6;badstate=xp.asarray(bad);badex=xp.asarray(exports);badignore=xp.asarray(ignored)
    badws=NSSL2DriverWorkspace(state=badstate,shape=shape,category_surface_export=badex,ignored_accumulator=badignore)
    try: launch_shallow_warm_sedimentation(badws,dr,dtp,dd,10.)
    except NotImplementedError: pass
    else: raise AssertionError('dense mass accepted')
    for before,after in ((bad,badstate),(exports,badex),(ignored,badignore)):
     assert np.array_equal(before.view(np.uint32),after.to_numpy().view(np.uint32))
 print(json.dumps({'method':'host barrier-execution of selected parallel CUDA source and reducer; strict Metal warm caller; workspace reconstruction between steps','results':results},indent=2))
 if any(d for r in results for d in r['different']):raise SystemExit(1)
