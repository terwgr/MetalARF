"""Compare native MPAS MYNN surface columns against host-compiled CUDA source."""
from pathlib import Path
import ctypes
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.metal_mynn_word_probe import plain_arithmetic
from woof.hex.mpasphys.sf_mynn_v842 import _cuda_source,FIELDS,CONSTANTS,OUTPUTS
from woof.hex.mpasphys.metal_tendencies_v842 import _launch
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def fixture(n):
    f=np.zeros((len(FIELDS),n),np.float32)
    def put(name,val): f[FIELDS.index(name)]=val
    put('u1d',np.linspace(-8,8,n,dtype=np.float32));put('v1d',np.linspace(2,10,n,dtype=np.float32))
    put('u1d2',np.linspace(-7,9,n,dtype=np.float32));put('v1d2',np.linspace(3,11,n,dtype=np.float32))
    put('t1d',np.linspace(257,303,n,dtype=np.float32));put('qv1d',np.linspace(.002,.018,n,dtype=np.float32))
    put('p1d',np.linspace(90000,100000,n,dtype=np.float32));put('psfcpa',np.linspace(95000,102000,n,dtype=np.float32))
    for name,value in (('dz8w1d',50),('dz2w1d',60),('rho1d',1.15),('chs',.01),('chs2',.01),('cqs2',.01),('cpm',1004),('pblh',600),('znt',.05),('ust',.25),('mavail',.7),('regime',1),('snowh',0),('dx',3000),('qcg',.01),('qsfc',.005)):
        put(name,value)
    put('xland',np.where(np.arange(n)%2,2,1));put('hfx',np.linspace(-40,150,n,dtype=np.float32))
    put('qfx',np.linspace(0,0.0002,n,dtype=np.float32));put('tsk',np.linspace(254,307,n,dtype=np.float32))
    constants=dict(cp=1004.5,g=9.81,rovcp=287.04/1004.5,r=287.04,xlv=2.5e6,svp1=6.112,svp2=17.67,svp3=29.65,svpt0=273.15,ep1=.608,ep2=.622,karman=.4)
    return f,np.array([constants[n] for n in CONSTANTS],np.float32)


def mismatch(name,actual,expected):
    bad=np.flatnonzero(actual.view(np.uint32).reshape(-1)!=expected.view(np.uint32).reshape(-1))
    print(name,'words',actual.size,'different',bad.size,'first',bad[:8].tolist(),flush=True)
    return len(bad)


def main():
    host=HostModule('mpas_v842_sf_mynn',plain_arithmetic(_cuda_source()))
    host.get_function('sf_mynn_tables')((1,),(1,),())
    expected_tables=np.concatenate([np.ctypeslib.as_array((ctypes.c_float*1001).in_dll(host.lib,n)).copy()
        for n in ('psim_stab','psih_stab','psim_unstab','psih_unstab')])
    n=8; data,constants=fixture(n); expected=data.copy(); total=0
    with MetalContext(batch_kernel_calls=False) as context:
        original=context._submit_kernel_batch
        def sync(*a,**k):
            result=original(*a,**k);result.wait();return result
        context._submit_kernel_batch=sync
        xp=MetalArrayModule(context); tables=xp.empty((4004,),np.float32)
        _launch('hexv842_mpas_sf_mynn_tables_f32',1001,context,dict(tables=tables),dict(n=np.int32(1001)))
        total+=mismatch('stability tables',tables.to_numpy(),expected_tables)
        fields=xp.asarray(data.reshape(-1)); const=xp.asarray(constants)
        kernel=host.get_function('sf_mynn_columns')
        for step in (1,0):
            kernel((1,),(128,),(expected,constants,np.int32(step),np.int32(1),np.int32(0),np.int32(0),np.int32(0),np.uint32(127),np.int32(n)))
            _launch('hexv842_mpas_sf_mynn_columns_f32',n,context,dict(fields=fields,constants=const,tables=tables),dict(n=np.int32(n),itimestep=np.int32(step),isfflx=np.int32(1),spp_pbl=np.int32(0),isftcflx=np.int32(0),iz0tlnd=np.int32(0),present=np.int32(127)))
            actual=fields.to_numpy().reshape(data.shape)
            for name in OUTPUTS:
                j=FIELDS.index(name);total+=mismatch(f'step {step} {name}',actual[j],expected[j])
    return int(total!=0)
if __name__=='__main__': raise SystemExit(main())
