"""Compare native LW cloud-optics and status handling with selected source."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.core.rrtmg_lw import _lw_coalesced_source
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import cloud_optics
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
source=_lw_coalesced_source((ROOT/'woof/core/kernels/rrtmg_lw_chain.cu').read_text())
m=re.search(r'extern "C" __global__ void rlw_cldprmc_coalesced\s*\(',source);i=source.index('{',m.end());j=i+1;depth=1
while depth:
    if source[j]=='{':depth+=1
    elif source[j]=='}':depth-=1
    j+=1
header='''#define NGPTLW 140
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
'''
host=HostModule('mpas_v842_lw_cloud_optics',header+source[m.start():j]).get_function('rlw_cldprmc_coalesced')
nc,nl=2,5;rng=np.random.default_rng(842);shape=(nc,nl,140)
cloud=(rng.random(shape)>.35).astype(np.float32)
ice=rng.uniform(.01,.5,shape).astype(np.float32);liquid=rng.uniform(.01,.5,shape).astype(np.float32);snow=rng.uniform(.01,.5,shape).astype(np.float32)
ice[0,0,0]=np.array([1],np.uint32).view(np.float32)[0]
radii=[np.full((nc,nl),x,np.float32) for x in (30,20,40)]
names=('absice1','absice2','absice3','absice0','absliq1');sizes=(10,688,736,2,928)
tables={n:rng.uniform(.01,.1,size).astype(np.float32) for n,size in zip(names,sizes)}
tables['absliq0']=np.float32(.09);tables['ngb']=(np.arange(140)%16+1).astype(np.int32)
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    dc,di,dl,ds=map(xp.asarray,(cloud,ice,liquid,snow));dtables={n:xp.asarray(v) if isinstance(v,np.ndarray) else v for n,v in tables.items()}
    for inflag,iceflag,liqflag,bad,expected_error in [(0,0,0,None,0),(1,0,0,None,1),*( (2,i,l,None,0) for i in range(6) for l in (0,1)),(2,0,0,('ice',1),2),(2,1,0,('ice',1),3),(2,2,0,('ice',1),4),(2,3,0,('ice',1),5),(2,5,0,('snow',1),6),(2,3,1,('liquid',1),7)]:
        rr=[x.copy() for x in radii]
        if bad:rr[{'ice':0,'liquid':1,'snow':2}[bad[0]]][:]=bad[1]
        tau=np.full(shape,.05,np.float32);ncb=np.zeros(nc,np.int32);err=np.zeros(1,np.int32)
        host(((nc*140+127)//128,),(128,),(*map(np.int32,(nc,nl,inflag,iceflag,liqflag)),cloud,ice,liquid,snow,*rr,*[tables[n] for n in names],tables['absliq0'],tables['ngb'],tau,ncb,err))
        assert err[0]==expected_error,(inflag,iceflag,liqflag,bad,err)
        dout=xp.asarray(np.full(shape,.05,np.float32));dn=xp.zeros(ncb.shape,np.int32);de=xp.zeros(err.shape,np.int32)
        cloud_optics(dc,di,dl,ds,*map(xp.asarray,rr),dtables,dout,dn,de,inflag=inflag,iceflag=iceflag,liqflag=liqflag)
        diff=0
        for name,got,want in (('tau',dout,tau),('ncb',dn,ncb),('err',de,err)):
            actual=got.to_numpy();count=np.count_nonzero(actual.view(np.uint32)!=want.view(np.uint32));diff+=count
            if count:print(name,'differences',count)
        print('case',inflag,iceflag,liqflag,bad,'words',tau.size+ncb.size+err.size,'differences',diff)
        if diff:raise SystemExit(1)
