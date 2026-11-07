import ctypes as ct,os,subprocess,tempfile
from pathlib import Path
import numpy as np
os.environ.setdefault('GPUWM_WRF_EXACT','1')
from woof.core.kernels import module_source
from tools.mp28_metal.parse import closure,split_top,def_name
from tools.thompson_real_column_parity.host_backend import parse_kernels
from woof.core.nest_interp import register_nest,sint,bdy_interp1
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
orig=MetalContext._submit_kernel_batch
def sync(self,*a,**kw):
 r=orig(self,*a,**kw)
 if r is not None:r.wait()
 return r
MetalContext._submit_kernel_batch=sync
src=module_source('nest');pre=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=src,text=True,capture_output=True,check=True).stdout
names=('nest_sint','nest_bdy_interp1');items=closure(split_top(pre),names)[0];params=parse_kernels('\n'.join(items));wrappers=[]
for name in names:
 kernel=next(x for x in items if def_name(x)==name);head=kernel[:kernel.index('{')].replace('extern "C" __global__','extern "C"').replace('void '+name+'(', 'void host_'+name+'(');argnames=','.join(p.name for p in params[name]);count='nz*nyc*nxc' if name=='nest_sint' else 'nz*(side<2?nyc:nxc)*sz';wrappers.append(head+'{blockDim.x=1;threadIdx.x=0;for(blockIdx.x=0;blockIdx.x<(unsigned)('+count+');++blockIdx.x)'+name+'('+argnames+');}\n')
pre='''#include <stddef.h>\n#include <cmath>\n#define __device__\n#define __global__\n#define __forceinline__ inline\nstruct D{unsigned x;};D blockIdx,blockDim,threadIdx;\n'''
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'ref.cpp';lib=Path(tmp)/'ref.dylib';p.write_text(pre+'\n'.join(items)+'\n'+''.join(wrappers));subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off','-dynamiclib',str(p),'-o',str(lib)],check=True);dll=ct.CDLL(str(lib));FP=ct.POINTER(ct.c_float);IP=ct.POINTER(ct.c_int32)
 for name in names:getattr(dll,'host_'+name).argtypes=[(IP if p.base=='int' else FP) if p.pointer else ct.c_float if p.base=='float' else ct.c_int for p in params[name]]
 xp=MetalArrayModule();rng=np.random.default_rng(1915);cases=words=0
 for ratio in (2,3):
  for stagger in ('','x','y'):
   for wrapper in ('interp','bdy'):
    reg=register_nest(nri=ratio,nrj=ratio,i_parent_start=4,j_parent_start=4,child_nx=6,child_ny=6,parent_nx=12,parent_ny=12,stagger=stagger,wrapper=wrapper);reg.device_tables(alloc=lambda name,shape,dtype:xp.empty(shape,dtype));nz=2;parent=rng.uniform(-1,1,(nz,reg.nyp,reg.nxp)).astype('f');child=rng.uniform(-1,1,(nz,reg.nyc,reg.nxc)).astype('f');
    if os.environ['GPUWM_WRF_EXACT']=='1':
     parent[:,4,4:8]=np.array([0,0x80000000,1,0x80000001],np.uint32).view('f');child[:,0,0:4]=np.array([0,0x80000000,1,0x80000001],np.uint32).view('f')
    dp=xp.asarray(parent);dc=xp.asarray(child)
    vals=dict(cfld=parent,nfld=child,ci_map=reg.ci,ip_map=reg.ip,cj_map=reg.cj,jp_map=reg.jp,xig=reg.xig,xjg=reg.xjg,nz=nz,nyc=reg.nyc,nxc=reg.nxc,nyp=reg.nyp,nxp=reg.nxp,cdt=np.float32(7.3),sz=2)
    def call(name,v):
     args=[v[p.name].ctypes.data_as(IP if p.base=='int' else FP) if p.pointer else ct.c_float(v[p.name]) if p.base=='float' else ct.c_int(v[p.name]) for p in params[name]];getattr(dll,'host_'+name)(*args)
    if wrapper=='interp':
     ref=np.empty_like(child);call('nest_sint',dict(vals,nfld=ref));got=sint(dp,reg).to_numpy();np.testing.assert_array_equal(got.view('u4'),ref.view('u4'),err_msg=str((ratio,stagger,wrapper)));cases+=1;words+=ref.size
    else:
     tables=bdy_interp1(dp,dc,reg,parent_dt_fp32=7.3,spec_zone=1,relax_zone=1,spec_bdy_width=2)
     for side,label in enumerate(('west','east','south','north')):
      shape=(nz,reg.nyc,2) if side<2 else (nz,2,reg.nxc);rv=np.empty(shape,'f');rt=np.empty(shape,'f');call('nest_bdy_interp1',dict(vals,side=side,bdy_val=rv,bdy_tend=rt));gv,gt=tables[label];np.testing.assert_array_equal(gv.to_numpy().view('u4'),rv.view('u4'));np.testing.assert_array_equal(gt.to_numpy().view('u4'),rt.view('u4'),err_msg=str((ratio,stagger,label)));cases+=1;words+=rv.size+rt.size
 print('matched',cases,'actual SINT/boundary callers,',words,'words')
