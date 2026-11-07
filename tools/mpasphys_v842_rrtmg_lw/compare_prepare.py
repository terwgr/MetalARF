"""Compare native longwave extended-atmosphere preparation to selected source."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.rrtmg_lw_v842 import _DRIVER_SOURCE as source
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import prepare
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
constants=[]
for name in ('retab','pprof','tprof','o3sum','ppsum','o3win','ppwin'):
    constants.append(re.search(r'__device__ const float mpas_'+name+r'\[\d+\] = \{.*?\};',source,re.S).group())
macros=source[source.index('#define M(f,k)'):source.index('// MPAS: module_ra_rrtmg_lw.F:11954-11983.')]
def extract(name):
    m=re.search(r'(?:__device__ (?:float|void)|extern "C" __global__ void) '+name+r'\s*\(',source)
    i=source.index('{',m.end());j=i+1;depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    return source[m.start():j]
parts=[extract(n) for n in ('mpas_profile','mpas_ice_radius','mpas_liquid_radius','mpas_vinterp_one','mpas_lw_prepare_one','mpas_lw_prepare')]
header='''#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
'''
host=HostModule('mpas_v842_lw_prepare',header+'\n'.join(constants)+'\n'+macros+'\n'+'\n'.join(parts)).get_function('mpas_lw_prepare')
nc,nm,nl,nz=3,4,9,4
m=np.zeros((nc,12,nm),np.float32)
m[:,0,:]=[90000,65000,35000,11000];m[:,1,:]=[285,270,250,225];m[:,2,:]=[600,1200,2000,3500]
m[:,3,:]=[.01,.005,.002,.0003];m[:,4,:]=.0002;m[:,5,:]=.0001;m[:,6,:]=.00005;m[:,7,:]=[.1,.7,.3,0]
m[:,8,:]=9e-6;m[:,9,:]=30e-6;m[:,10,:]=60e-6;m[:,11,:]=.9
m[1,7,:]=0;m[2,7,:]=[.9,.01,.5,.2]
m[2,7,3]=np.array([1],np.uint32).view(np.float32)[0]
m[2,4,3]=np.array([1],np.uint32).view(np.float32)[0]
m[2,8,3]=0;m[2,9,3]=0
interfaces=np.empty((nc,2,nm+1),np.float32);interfaces[:,0,:]=[100000,80000,50000,20000,2000];interfaces[:,1,:]=[290,278,260,240,220]
surface=np.array([[290,.95,1,0,0,.1],[275,.98,2,0,.2,-.4],[265,.97,1,.7,.1,.5]],np.float32)
pin=np.array([1,100,500,1100],np.float32);ozone=np.tile(np.array([2e-6,1e-6,2e-7,5e-8],np.float32),(nc,1))
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*a,**k):
        result=submit(*a,**k);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    dm,di,ds,dp,do=map(xp.asarray,(m,interfaces,surface,pin,ozone))
    for icloud,reqc,reqi,reqs,o3input,presence in ((1,0,0,0,0,15),(1,1,1,1,2,15),(1,1,0,1,2,11),(0,0,0,0,0,0)):
        a=np.full((nc,12,nl),-7,np.float32);lev=np.full((nc,2,nl+1),-7,np.float32)
        params=[nc,nm,nl,icloud,reqc,reqi,reqs,o3input,nz,presence]
        host((1,),(128,),(*map(np.int32,params),m,interfaces,surface,pin,ozone,a,lev))
        da=xp.empty(a.shape,np.float32);dl=xp.empty(lev.shape,np.float32)
        prepare(dm,di,ds,dp,do,da,dl,icloud=icloud,reqc=reqc,reqi=reqi,reqs=reqs,o3input=o3input,presence=presence)
        differences=0
        for name,actual,expected in [('atmosphere',da,a),('interfaces',dl,lev)]:
            got=actual.to_numpy();bad=np.argwhere(got.view(np.uint32)!=expected.view(np.uint32));differences+=len(bad)
            if len(bad):print(name,'mismatches',len(bad),'first',bad[:8].tolist())
        print('case',params[3:],'words',a.size+lev.size,'differences',differences)
        if differences:raise SystemExit(1)
    try:
        prepare(dm,di,ds,xp.asarray(pin[:1]),xp.asarray(ozone[:,:1].copy()),da,dl,icloud=1,reqc=0,reqi=0,reqs=0,o3input=2,presence=15)
        raise AssertionError('one-level source boundary must be refused')
    except ValueError as e:
        assert 'two defined' in str(e)
