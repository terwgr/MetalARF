"""Compare strict radar caller with host-compiled selected CUDA source."""
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
from woof.core.nssl2_diagnostics import launch_radardd02
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
src=module_source('nssl2_diagnostics');a=src.index('namespace {',src.index('WRF v4.6.1 NSSL option-18 default'));b=src.index('// Concentration-space twin',a)
shim=ROOT/'tools/thompson_real_column_parity/cuda_host_shim.h';glibc=ROOT/'woof/core/kernels/glibc_flt32.cuh';header=(ROOT/'woof/core/kernels/nssl2_wrf_math.cuh').read_text()
macros='\n'.join(line for line in header.splitlines() if line.startswith(('#define NSSL2_WRF_XVHMN ','#define NSSL2_WRF_XVHMX0 ','#define NSSL2_WRF_XVHLMN ','#define NSSL2_WRF_XVHLMX ')))
names=('air_density','temperature_k','qr','qi','qs','qg','qh','qnr','qni','qns','qng','qnh','qvolg','qvolh')
args=','.join('const float *'+x for x in names);fields=','.join(names)
wrap='\nextern "C" void run('+args+',float *out,int mode,int n){for(int i=0;i<n;++i){blockIdx.x=i;nssl2_radardd02('+fields+',out,mode,n);}}\n'
with tempfile.TemporaryDirectory(prefix='woof-nssl-radar-') as tmp:
 tmp=Path(tmp);(tmp/'source.cpp').write_text(f'#include "{shim}"\n#include "{glibc}"\n'+macros+'\n'+src[a:b]+wrap)
 subprocess.run(['clang++','-std=c++17','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(tmp/'source.cpp'),'-o',str(tmp/'source.dylib')],check=True)
 f=ctypes.CDLL(str(tmp/'source.dylib')).run;fp=ctypes.POINTER(ctypes.c_float);f.argtypes=[fp]*15+[ctypes.c_int]*2;ptr=lambda x:x.ctypes.data_as(fp)
 shape=(4,1,2048);n=int(np.prod(shape));rng=np.random.default_rng(7912)
 inputs={'air_density':rng.uniform(.3,1.5,shape).astype(np.float32),'temperature_k':rng.uniform(220,310,shape).astype(np.float32)}
 for name in ('qr','qi','qs','qg','qh'):inputs[name]=np.power(10.,rng.uniform(-15,-2,shape)).astype(np.float32)
 for name in ('qnr','qni','qns','qng','qnh'):inputs[name]=np.power(10.,rng.uniform(-9,10,shape)).astype(np.float32)
 for name in ('qvolg','qvolh'):inputs[name]=np.power(10.,rng.uniform(-15,-2,shape)).astype(np.float32)
 categories=('qr','qi','qs','qg','qh');bands=(np.arange(n)%7).reshape(shape)
 for j,key in enumerate(categories):inputs[key][(bands!=0)&(bands!=j+1)]=0.0
 for name in categories: inputs[name].flat[:3]=[0.,-0.,np.nextafter(np.float32(0),np.float32(1))]
 cases={}
 with MetalContext() as ctx:
  ctx.batch_kernel_calls=False;submit=ctx._submit_kernel_batch
  def sync(*a,**kw):
   res=submit(*a,**kw);res.wait();return res
  ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx);dev=[xp.asarray(inputs[x]) for x in names]
  for mode in (False,True):
   expected=np.empty(shape,np.float32);f(*map(ptr,[*inputs.values(),expected]),int(mode),n)
   out=xp.asarray(np.full(shape,-7.,np.float32))
   assert launch_radardd02(*dev,out,output_due=False,concentration_space=mode) is False
   assert np.all(out.to_numpy()==np.float32(-7))
   assert launch_radardd02(*dev,out,output_due=True,concentration_space=mode)
   got=out.to_numpy();counts={str(j):int(np.count_nonzero(expected[bands==j].view(np.uint32)!=got[bands==j].view(np.uint32))) for j in range(7)}
   cases[str(mode)]={'different_by_category':counts,'nonzero':int(np.count_nonzero(expected))}
 print(json.dumps({'source_sha256':hashlib.sha256(src.encode()).hexdigest(),'effective_options':effective_options(module_options('nssl2_diagnostics')),'words':2*n,'cases':cases,'not_due_untouched':True},indent=2))
 if any(v for c in cases.values() for v in c['different_by_category'].values()):raise SystemExit(1)
