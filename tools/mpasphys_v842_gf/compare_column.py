"""Compare a native-controlled GF columns with selected pinned host source."""
import ctypes
import hashlib
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import numpy as np
from tools.gf_wrf461_oracle import gf_regime_columns as fixture
from tools.thompson_real_column_parity.host_backend import SHIM
from woof.core import kernels
from woof.hex.mpasphys import gf_mpas_v842 as native
from woof.hex.mpasphys.metal_gf_v842 import run_columns
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule

columns=[fixture.build_column(fixture.REGIMES[i],dx=3000,ishallow=1,ichoice=0,dt=60) for i in (1,5)]
nz=len(columns[0].lev['t']);levels=np.empty((2,15,nz),np.float32);surface=np.empty((2,8),np.float32)
for c,column in enumerate(columns):
    if len(column.lev['t'])!=nz:raise ValueError('GF comparison columns must share a level count')
    for i,name in enumerate(native.LEVEL_INPUTS):
        levels[c,i]=column.p8w[:-1] if name=='p8w' else column.lev['qv' if name=='q' else name]
    surface[c]=[column.sca['ht'],column.sca['hfx'],column.sca['qfx'],column.sca['xland'],200,3000,3000**2,column.dt]
kpbl=np.array([column.kpbl for column in columns],np.int32);initial=(levels.copy(),surface.copy(),kpbl.copy())
ref=(levels.copy(),surface.copy(),kpbl.copy(),np.zeros((2,6,nz),np.float32),np.zeros((2,6),np.float32),np.zeros((2,3),np.int32),np.array([1,0,8,1],np.int32),np.array(native.MPAS_DRIVER_CONSTANTS,np.float32))
size=native._scratch_bytes(nz);assert size%4==0
with tempfile.TemporaryDirectory(prefix='native-gf-column-') as d:
    path=Path(d);src=path/'native.cpp';lib=path/'native.dylib'
    support=native._SUPPORT.replace('#define __global__\n','').replace('#define __device__\n','')
    support=re.sub(r'inline float (__fadd_rn|__fsub_rn|__fmul_rn|__fdiv_rn|__fsqrt_rn|gfk_log|gfk_pow)\([^\n]*\n','',support)
    header=Path(kernels.__file__).with_name('glibc_flt32.cuh').read_text()
    capture='''\nextern "C" float gf_port_pow(float x,float y){return gfk_pow(x,y);}
extern "C" void gf_port_capture(int nz,float* lev,float* surface,int* isurface,float* outlev,float* outsurface,int* outint,int* control,float* constants,char* scratch,unsigned long long size){GFWorkspace ws={scratch,size,0,0,0};gf_column(nz,(GFReal*)lev,(GFReal*)surface,isurface,(GFReal*)outlev,(GFReal*)outsurface,outint,control,(GFReal*)constants,ws);}\n'''
    src.write_text('#include "'+str(SHIM)+'"\n'+header+'\n#define GF_HOST_WITNESS\n'+support+native._SCHEME_SOURCE+native._ENTRY+capture)
    subprocess.run(['clang++','-std=c++17','-O1','-ffp-contract=off','-shared','-fPIC',str(src),'-o',str(lib)],check=True)
    poison=np.uint32(0x7fc00001 if os.environ.get('GF_POISON') else 0)
    scratch=np.full(size//4,poison,np.uint32)
    library=ctypes.CDLL(str(lib)); power=library.gf_port_pow;power.argtypes=[ctypes.c_float,ctypes.c_float];power.restype=ctypes.c_float
    exponent=np.array([0x3fb07b57],np.uint32).view(np.float32)[0];word=np.array([power(10.0,float(exponent))],np.float32).view(np.uint32)[0]
    if word!=0x41bf5c7b:raise AssertionError('GF host witness must use the selected pinned power helper')
    fn=library.gf_port_capture;fn.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*9+[ctypes.c_ulonglong];fn.restype=None
    for c in range(2):
        fn(nz,*(v[c:c+1].ctypes.data for v in ref[:6]),ref[6].ctypes.data,ref[7].ctypes.data,scratch.ctypes.data,size)
    if not (ref[4][0,2]>0 and ref[4][0,3]>0 and np.all(ref[4][1,2:]==0) and np.all(ref[5][1]==0)):
        raise AssertionError('GF source fixture must exercise active and inactive columns')
    print('source surface outputs',ref[4].tolist(),'integer outputs',ref[5].tolist(),flush=True)
    if os.environ.get('GF_HOST_ONLY'):sys.exit(0)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        levels_m,surface_m,kpbl_m=(xp.asarray(x) for x in initial)
        outputs=run_columns(levels_m,surface_m,kpbl_m,itimestep=1,constants=native.MPAS_DRIVER_CONSTANTS)
        bad=total=0
        for name,actual,expected in zip(('levels','surface','kpbl','outlev','outsurface','outint'),(levels_m,surface_m,kpbl_m,*outputs),ref[:6]):
            got=actual.to_numpy().view(np.uint32);want=expected.view(np.uint32);idx=np.flatnonzero(got!=want);bad+=len(idx);total+=got.size
            if len(idx):print(name,'differences',[(int(i),hex(int(want.flat[i])),hex(int(got.flat[i]))) for i in idx[:12]],flush=True)
        print('surface outputs',ref[4].tolist(),flush=True)
        print('output digest',hashlib.sha256(b''.join(x.tobytes() for x in ref[3:6])).hexdigest(),flush=True)
        print('GF native selected-host-source comparison',bad,'/',total,'words',flush=True)
        if bad:raise SystemExit(1)
