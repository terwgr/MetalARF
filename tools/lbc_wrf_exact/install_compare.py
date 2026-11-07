"""Compare strict MU boundary installation caller with composed CUDA words."""
import ctypes as ct,os,subprocess,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
os.environ['GPUWM_WRF_EXACT']='1'
from woof.ingest.lateral_bc import _launch_mu_boundary_values
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.lbc_state import _perimeter_count
from tools.lbc_wrf_exact.generate_install import retained
from tools.mp28_metal.parse import def_name
from tools.thompson_real_column_parity.host_backend import parse_kernels
orig=MetalContext._submit_kernel_batch
def sync(self,*a,**k):
 r=orig(self,*a,**k)
 if r is not None:r.wait()
 return r
MetalContext._submit_kernel_batch=sync
items=retained();params=parse_kernels('\n'.join(items))['install_mu_boundary'];kernel=next(x for x in items if def_name(x)=='install_mu_boundary');head=kernel[:kernel.index('{')].replace('extern "C" __global__','extern "C"').replace('void install_mu_boundary(', 'void host_install(');names=','.join(p.name for p in params);wrap=head+'{blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)frame_count;++blockIdx.x)install_mu_boundary('+names+');}\n'
pre='''#include <stddef.h>\n#include <algorithm>\nusing std::min;using std::max;\n#define __device__\n#define __global__\n#define __forceinline__ inline\nstruct dims{unsigned x;};dims blockIdx,blockDim,threadIdx;\nstatic float __fadd_rn(float a,float b){volatile float v=a+b;return v;}\nstatic float __fmul_rn(float a,float b){volatile float v=a*b;return v;}\n'''
class State(SimpleNamespace):
 def scratch(self,shape,slot):
  if slot not in self.slots:self.slots[slot]=xp.empty(shape,np.float32)
  return self.slots[slot]
with tempfile.TemporaryDirectory() as tmp:
 src=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';src.write_text(pre+'\n'.join(items)+'\n'+wrap);subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(src),'-o',str(lib)],check=True)
 fun=ct.CDLL(str(lib)).host_install;P=ct.POINTER(ct.c_float);F=ct.c_float;I=ct.c_int;fun.argtypes=[P if p.pointer else F if p.base in ('real','float') else I for p in params]
 xp=MetalArrayModule();rng=np.random.default_rng(643);cases=words=0
 for width in (1,2):
  for dt in (np.float32(.2),np.float32(-.3)):
   ny,nx=7,8;mu=rng.uniform(-2,2,(ny,nx)).astype(np.float32);mu.flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
   h={'mup':mu.copy()};boundary={}
   for name,shape in [('west',(1,ny,width)),('east',(1,ny,width)),('south',(1,width,nx)),('north',(1,width,nx))]:
    h[name]=rng.uniform(-2,2,shape).astype(np.float32);h[name+'_t']=rng.uniform(-2,2,shape).astype(np.float32);h[name].flat[:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
    boundary[name]=SimpleNamespace(value=xp.asarray(h[name]),tendency=xp.asarray(h[name+'_t']))
   frame=_perimeter_count(ny,nx,width);h.update(old_mup_frame=np.empty((frame,),np.float32),dtbc=dt,width=width,spec_zone=width,ny=ny,nx=nx,frame_count=frame)
   args=[h[p.name].ctypes.data_as(P) if p.pointer else F(h[p.name]) if p.base in ('real','float') else I(h[p.name]) for p in params];fun(*args)
   state=State(_metal_setup_state=True,_array_module=xp,mup=xp.asarray(mu),slots={});old=_launch_mu_boundary_values(state,SimpleNamespace(**boundary),float(dt),width)
   np.testing.assert_array_equal(state.mup.to_numpy().view(np.uint32),h['mup'].view(np.uint32));np.testing.assert_array_equal(old.to_numpy().view(np.uint32),h['old_mup_frame'].view(np.uint32))
   words+=mu.size+frame;cases+=1
 print('matched',cases,'strict MU installation callers,',words,'words; includes old frame, signed zeros and subnormals')
