"""Compare NSSL temperature publication with gradual FP32 multiplication."""
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from woof.wrf_exact import ENABLED
if not ENABLED:raise SystemExit('GPUWM_WRF_EXACT=1 is required')
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.core.nssl2_runtime import _publish_temperature_metal
with tempfile.TemporaryDirectory(prefix='woof-nssl-temp-') as tmp:
 tmp=Path(tmp);(tmp/'reference.cpp').write_text('extern "C" void run(const float*a,const float*b,float*out,int n){for(int i=0;i<n;i++){volatile float x=a[i]*b[i];out[i]=x;}}')
 subprocess.run(['clang++','-O0','-shared','-fPIC','-fno-fast-math','-ffp-contract=off',str(tmp/'reference.cpp'),'-o',str(tmp/'reference.dylib')],check=True)
 f=ctypes.CDLL(str(tmp/'reference.dylib')).run;fp=ctypes.POINTER(ctypes.c_float);f.argtypes=[fp]*3+[ctypes.c_int];ptr=lambda x:x.ctypes.data_as(fp)
 shape=(4,2,1024);rng=np.random.default_rng(2026)
 theta=rng.uniform(-400,400,shape).astype(np.float32);pii=rng.uniform(-2,2,shape).astype(np.float32)
 theta.flat[:8]=np.array([0x00000001,0x80000001,0x00800000,0x80800000,0x3f800000,0xbf800000,0,0x80000000],np.uint32).view(np.float32)
 pii.flat[:8]=np.array([0x3f800000,0x3f800000,0x3f000000,0x3f000000,0x00000001,0x00000001,0x3f800000,0x3f800000],np.uint32).view(np.float32)
 expected=np.empty(shape,np.float32);f(ptr(theta),ptr(pii),ptr(expected),theta.size)
 with MetalContext() as ctx:
  ctx.batch_kernel_calls=False;submit=ctx._submit_kernel_batch
  def sync(*a,**kw):
   v=submit(*a,**kw);v.wait();return v
  ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
  out=xp.zeros(shape);_publish_temperature_metal(xp.asarray(theta),xp.asarray(pii),out);actual=out.to_numpy()
 differences=int(np.count_nonzero(expected.view(np.uint32)!=actual.view(np.uint32)))
 print(json.dumps({'words':theta.size,'different':differences,'subnormal_discriminators':6,'reference':'host scalar FP32 multiply; CuPy device body not captured'},indent=2))
 if differences:raise SystemExit(1)
