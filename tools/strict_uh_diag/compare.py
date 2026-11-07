#!/usr/bin/env python3
"""Compare strict UH Metal columns and folds with the CUDA source built on host."""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRELUDE = r'''
#include <cstddef>
#include <cmath>
#define __device__
#define __global__
#define __forceinline__ inline
#define __restrict__ __restrict
#define G 9.81f
struct D { int x; }; static D blockIdx,blockDim,threadIdx;
static inline float __fadd_rn(float a,float b){return a+b;}
static inline float __fsub_rn(float a,float b){return a-b;}
static inline float __fmul_rn(float a,float b){return a*b;}
static inline float __fdiv_rn(float a,float b){return a/b;}
'''
WRAPPER = r'''
extern "C" void ref(float* u,float* v,float* w,float* ph,float* phb,int phb3d,
 float* msfu,float* msfv,float* ht,float* dn,float* dnw,float* fnm,float* fnp,
 float cf1,float cf2,float cf3,float rdx,float rdy,float* uh,float* use,float* out,
 int nz,int ny,int nx) {
 blockDim.x=1;threadIdx.x=0;
 for(int col=0;col<nx*ny;++col){blockIdx.x=col;uh_columns(u,v,w,ph,phb,phb3d,msfu,msfv,ht,dn,dnw,fnm,fnp,cf1,cf2,cf3,rdx,rdy,uh,use,nz,ny,nx);}
 for(int col=0;col<nx*ny;++col){blockIdx.x=col;uh_smooth_max(uh,use,out,ny,nx);}
 for(int j=0;j<ny;++j)out[j*nx]=out[j*nx+1];
 for(int i=0;i<nx;++i)out[i]=out[nx+i];
}
extern "C" void ref_fold(float* uh,float* use,float* out,int ny,int nx) {
 blockDim.x=1;threadIdx.x=0;
 for(int col=0;col<nx*ny;++col){blockIdx.x=col;uh_smooth_max(uh,use,out,ny,nx);}
 for(int j=0;j<ny;++j)out[j*nx]=out[j*nx+1];
 for(int i=0;i<nx;++i)out[i]=out[nx+i];
}
'''

def host_library(directory):
    source = (ROOT/'woof/core/kernels/common.cuh').read_text() + (ROOT/'woof/core/kernels/uh_diag.cu').read_text()
    file = Path(directory)/'uh.cpp'; file.write_text(PRELUDE+source+WRAPPER)
    lib = Path(directory)/'uh.dylib'
    subprocess.run(['clang++','-std=c++17','-O2','-ffp-contract=off','-shared','-fPIC',str(file),'-o',str(lib)],check=True)
    library=ctypes.CDLL(str(lib));func=library.ref
    ptr=ctypes.POINTER(ctypes.c_float); integer=ctypes.c_int; scalar=ctypes.c_float
    func.argtypes=[ptr]*5+[integer]+[ptr]*7+[scalar]*5+[ptr]*3+[integer]*3
    fold=library.ref_fold;fold.argtypes=[ptr]*3+[integer]*2
    return func,fold

def run():
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    from woof.metal.uh_diag_family import submit_uh_step
    from woof.wrf_exact import ENABLED
    assert ENABLED
    rng=np.random.default_rng(752); nz,ny,nx=12,6,7
    grid=np.arange(nz+1,dtype=np.float32)*np.float32(750)
    x=np.arange(nx,dtype=np.float32)[None,:];y=np.arange(ny,dtype=np.float32)[:,None]
    with tempfile.TemporaryDirectory() as directory, MetalContext() as context:
        ref,fold=host_library(directory);context.batch_kernel_calls=False
        submit=context._submit_kernel_batch
        def sync(*a,**kw):
            result=submit(*a,**kw); result.wait(); return result
        context._submit_kernel_batch=sync; xp=MetalArrayModule(context)
        for phb3d in (False,True):
          for dry in (False,True):
            terrain=np.broadcast_to((x+y)*np.float32(3),(ny,nx)).copy()
            phb=(grid*np.float32(9.81)).astype(np.float32)
            if phb3d: phb=(phb[:,None,None]+terrain[None,:,:]*np.float32(9.81)).astype(np.float32)
            ph=np.broadcast_to((x-y)[None,:,:]*np.float32(0.7),(nz+1,ny,nx)).copy()
            u=np.broadcast_to((np.arange(ny,dtype=np.float32)[None,:,None]*-0.04),(nz,ny,nx+1)).copy()+rng.normal(0,.01,(nz,ny,nx+1)).astype(np.float32)
            v=np.broadcast_to(np.arange(nx,dtype=np.float32)[None,None,:]*0.07,(nz,ny+1,nx)).copy()+rng.normal(0,.01,(nz,ny+1,nx)).astype(np.float32)
            w=np.full((nz+1,ny,nx),4.,np.float32)
            if dry:w[:,2:4,2:4]=-1.
            msfu=rng.uniform(.96,1.04,(ny,nx+1)).astype(np.float32);msfv=rng.uniform(.96,1.04,(ny+1,nx)).astype(np.float32)
            dn=np.ones(nz,np.float32);dnw=np.full(nz,.9,np.float32);fnm=np.full(nz,.55,np.float32);fnp=np.full(nz,.45,np.float32)
            args=[u,v,w,ph,phb,msfu,msfv,terrain,dn,dnw,fnm,fnp]
            vals=[np.float32(x) for x in (.8,.25,-.05,.001,.0012)]
            uh=np.zeros((ny,nx),np.float32);use=np.zeros_like(uh);target=np.zeros_like(uh)
            ptr=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            ref(*[ptr(a) for a in args[:5]],int(phb3d),*[ptr(a) for a in args[5:]],*map(float,vals),ptr(uh),ptr(use),ptr(target),nz,ny,nx)
            dev=[xp.asarray(a) for a in args];du=xp.asarray(np.zeros_like(uh));ds=xp.asarray(np.zeros_like(use));dt=xp.asarray(np.zeros_like(target))
            submit_uh_step(*dev,*vals,du,ds,dt)
            results=[]
            for name,want,got in (('columns',uh,du.to_numpy()),('use',use,ds.to_numpy()),('maximum',target,dt.to_numpy())):
                bad=np.count_nonzero(want.view(np.uint32)!=got.view(np.uint32)); results.append((name,int(bad),int(want.size)))
            print('phb3d',phb3d,'negative_updraft',dry,'nonzero',int(np.count_nonzero(target)),'results',results,flush=True)
            assert all(not row[1] for row in results)
        from woof.metal.uh_diag_family import submit_uh_fold
        words=np.resize(np.array([1, 2, 0x400000, 0x800000, 0x80000001],dtype=np.uint32),ny*nx)
        columns=words.view(np.float32).reshape(ny,nx).copy();use=np.ones((ny,nx),np.float32);want=np.zeros_like(columns)
        ptr=lambda a:a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        fold(ptr(columns),ptr(use),ptr(want),ny,nx)
        dc=xp.asarray(columns);du=xp.asarray(use);dm=xp.asarray(np.zeros_like(want));submit_uh_fold(dc,du,dm)
        got=dm.to_numpy();bad=int(np.count_nonzero(want.view(np.uint32)!=got.view(np.uint32)))
        print('subnormal fold nonzero',int(np.count_nonzero(want)),'different',bad,'words',want.size,flush=True);assert bad==0
if __name__=='__main__':run()
