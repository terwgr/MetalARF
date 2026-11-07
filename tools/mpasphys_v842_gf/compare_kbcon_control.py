"""Compare source-specific GF cloud-base control restructuring."""
from pathlib import Path
import re
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.mpasphys_v842_gf.lower_control import procedure,lower_kbcon
from woof.core import kernels
from woof.hex.mpasphys import gf_mpas_v842 as native
support=native._SUPPORT.replace('#define __global__\n','').replace('#define __device__\n','')
support=re.sub(r'inline float (__fadd_rn|__fsub_rn|__fmul_rn|__fdiv_rn|__fsqrt_rn|gfk_log|gfk_pow)\([^\n]*\n','',support)
orig=procedure('cup_kbcon').replace('void cup_kbcon (','void kb_orig (',1)
lower,_=lower_kbcon();lower=lower.replace('void cup_kbcon (','void kb_lower (',1)
header=Path(kernels.__file__).with_name('glibc_flt32.cuh').read_text()
source=header+'\n#define GF_HOST_WITNESS\n'+support+orig+lower+'''
extern "C" __global__ void check(const float* he,const float* hes,const float* pressure,const float* hinput,const int* in_k22,const int* in_max,const int* in_error,const int* loops,const int* excess,float* hout,int* k22out,int* kbout,int* errout,char* message,int n,int nz,int lowered){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;char scratch[256];GFWorkspace ws={scratch,256,0,0,0};int one=1,k22=in_k22[i],kb=0,err=in_error[i],kbmax=in_max[i],loop=loops[i],use=excess[i];GFReal capinc=5,capmax=100,xl=2500000,cp=1004.5,ztexec=.01,zqexec=.00001,hkb=hinput[i];
 if(lowered)kb_lower(ws,(GFText*)(message+i*50),&capinc,&loop,&k22,&kb,(GFReal*)(he+i*nz),(GFReal*)(hes+i*nz),&hkb,&err,&kbmax,(GFReal*)(pressure+i*nz),&capmax,&xl,&cp,&ztexec,&zqexec,&use,&one,&one,&nz,&one,&one,&one,&one,&one,&nz,50);
 else kb_orig(ws,(GFText*)(message+i*50),&capinc,&loop,&k22,&kb,(GFReal*)(he+i*nz),(GFReal*)(hes+i*nz),&hkb,&err,&kbmax,(GFReal*)(pressure+i*nz),&capmax,&xl,&cp,&ztexec,&zqexec,&use,&one,&one,&nz,&one,&one,&one,&one,&one,&nz,50);
 hout[i]=(float)hkb;k22out[i]=k22;kbout[i]=kb;errout[i]=err;
}
'''
host=HostModule('native_gf_kbcon_control',source).get_function('check')
n,nz=12,12;he=np.tile(np.linspace(100,111,nz,dtype=np.float32),(n,1));pressure=np.tile(np.linspace(1000,450,nz,dtype=np.float32),(n,1));hes=np.tile(np.linspace(105,94,nz,dtype=np.float32),(n,1))
hinput=np.array([95,100,103,110,95,100,103,110,95,100,103,110],np.float32);k22=np.full(n,2,np.int32);kbmax=np.full(n,7,np.int32);error=np.zeros(n,np.int32);error[-1]=9;loops=np.array([1,2,3,4,5,1,2,3,4,5,1,2],np.int32);excess=np.array([0,0,0,0,0,1,1,1,2,2,2,0],np.int32)
outputs=[]
for lowered in (0,1):
    hout=np.empty(n,np.float32);kout=np.empty(n,np.int32);bout=kout.copy();eout=kout.copy();msg=np.full((n,50),ord('X'),np.int8)
    host((1,),(32,),(he,hes,pressure,hinput,k22,kbmax,error,loops,excess,hout,kout,bout,eout,msg,np.int32(n),np.int32(nz),np.int32(lowered)))
    outputs.append((hout,kout,bout,eout,msg))
words=0
for a,b in zip(*outputs):
    assert np.array_equal(a.view(np.uint32),b.view(np.uint32)) if a.dtype!=np.int8 else np.array_equal(a,b)
    words+=a.size
print('GF cloud-base original vs structured words',words,'differences 0','kbcon',outputs[0][2].tolist(),'errors',outputs[0][3].tolist(),flush=True)
