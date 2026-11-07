"""Compare the shared mass operator with host-compiled v842 selected source."""
from pathlib import Path
import ctypes
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.resident_v842 import preparation_source
from woof.hex.mpasphys.metal_preparation_v842 import prepare_mass
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def main():
    kernel=HostModule('mpas_v842_preparation',preparation_source()).get_function('prep_mass_v841_f32')
    rng=np.random.default_rng(841842)
    nz,nc,ns=5,7,5
    shape=(nz,nc)
    fields={n:rng.uniform(lo,hi,shape).astype(np.float32) for n,lo,hi in (
        ('rho_zz',.5,1.5),('theta_m',250,320),('exner',.7,1.1),('pressure_base',40000,100000),
        ('pressure_p',-500,500),('zz',.8,1.2))}
    fields['scalars']=rng.uniform(-.03,.05,(ns,nz,nc)).astype(np.float32)
    fields['scalars'].flat[1]=np.float32(-2**-149)
    fields['zgrid']=np.cumsum(rng.uniform(40,200,(nz+1,nc)).astype(np.float32),axis=0)
    expected={n:np.empty(shape,np.float32) for n in ('zz_p','rho_dry','rho_p','th_p','t_p','pi_p','pres_p','zmid_p','dz_p')}
    expected['qout']=np.empty((ns,nz,nc),np.float32); expected['invalid']=np.zeros(1,np.int32)
    ptrs=(ctypes.c_void_p*ns)(*(expected['qout'][s].ctypes.data for s in range(ns)))
    inputs=('rho_zz','theta_m','scalars','exner','pressure_base','pressure_p','zgrid','zz')
    outputs=('zz_p','rho_dry','rho_p','th_p','t_p','pi_p','pres_p','zmid_p','dz_p','invalid')
    args=[nz,nc,ns,np.float32(np.float32(461.6)/np.float32(287.))]+[fields[n] for n in inputs]+[None]+[expected[n] for n in outputs]
    converted=[ctypes.c_void_p(x.ctypes.data) if isinstance(x,np.ndarray) else ctypes.cast(ptrs,ctypes.c_void_p) if x is None else x for x in args]
    kernel._fn(1,1,1,128,1,1,*converted)
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        result=prepare_mass(fields={n:xp.asarray(v) for n,v in fields.items()},scalar_names=('qv','qc','qi','qs','ni'))
        for n,expected_value in expected.items():
            actual=result[n].to_numpy(); count=np.count_nonzero(actual.view(np.uint32)!=expected_value.view(np.uint32))
            if count: raise AssertionError(f'{n}: {count} differing words')
    print('mass preparation matches host-selected source word for word')

if __name__=='__main__': main()
