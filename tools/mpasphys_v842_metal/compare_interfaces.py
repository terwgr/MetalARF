"""Compare v842 interface preparation with the selected host-compiled source."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.mpasphys.resident_v842 import preparation_source
from woof.hex.mpasphys.metal_preparation_v842 import prepare_interfaces
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext


def main():
    source=preparation_source()+'''\nextern "C" __global__ void exp_variants(const float *x, float *fused, float *plain) {
        int i=blockIdx.x*blockDim.x+threadIdx.x;
        if (i<2) {
            fused[i]=gfk_exp_fma(x[i]);
            double z=__dmul_rn(0x1.71547652b82fep+5, (double)x[i]);
            plain[i]=gfk_d2f_rn(gfk_exp2_core(z, GFK_EXP2F_SHIFT,
                GFK_EXP2F_P0/32.0/32.0/32.0, GFK_EXP2F_P1/32.0/32.0,
                GFK_EXP2F_P2/32.0, 0ULL));
        }
    }\n'''
    host=HostModule('mpas_v842_preparation_fma_probe',source)
    kernel=host.get_function('prep_interfaces_v841_f32')
    rng=np.random.default_rng(842)
    nz,nc=5,7
    shape=(nz,nc); interface=(nz+1,nc)
    fields={name:rng.uniform(lo,hi,shape).astype(np.float32) for name,lo,hi in (
        ('rho_zz',.5,1.5),('zz',.8,1.2),('qv_p',0,.03),('pres_p',20000,100000),
        ('t_p',230,310),('rho_p',.3,1.4),('dz_p',30,200),
        ('pressure_p',-300,300),('pressure_base',80000,100000))}
    words=(0x4202422f,0xc27c65d9)
    for col,word in enumerate(words):
        value=np.exp(np.array(word,np.uint32).view(np.float32)).astype(np.float32)
        fields['pres_p'][:,col]=value
        fields['pressure_base'][:,col]=value
        fields['pressure_p'][:,col]=0
    fields['zgrid']=np.cumsum(rng.uniform(50,180,interface).astype(np.float32),axis=0)
    # Top extrapolation weights are exactly 1.75 and -0.75 for both columns.
    fields['zgrid'][-3:,:2]=np.array([512,576,768],np.float32)[:,None]
    fields['w']=rng.uniform(-1,1,interface).astype(np.float32)
    interfaces=('w_p','z_p','t2_p','pres2_p','pres2_hyd_p','pres2_hydd_p','weights_minus','weights_plus')
    mass=('pres_hyd_p','pres_hydd_p','znu_p','znu_hyd_p')
    surface=('psfc_p','psfc_hyd_p','psfc_hydd_p','plrad')
    for n in interfaces: fields[n]=np.full(interface,-99,np.float32)
    for n in mass: fields[n]=np.full(shape,-99,np.float32)
    for n in surface: fields[n]=np.full(nc,-99,np.float32)
    fields['invalid']=np.zeros(1,np.int32)
    reference={n:x.copy() for n,x in fields.items()}
    order=('rho_zz','zz','qv_p','pres_p','t_p','rho_p','dz_p','zgrid','w','w_p','z_p','t2_p','pres2_p','pres_hyd_p','pres2_hyd_p','pres_hydd_p','pres2_hydd_p','psfc_p','psfc_hyd_p','psfc_hydd_p','znu_p','znu_hyd_p','plrad','invalid','pressure_p','pressure_base','weights_minus','weights_plus')
    kernel((1,),(128,),(nz,nc,np.float32(9.81),*[reference[n] for n in order]))
    x=np.array(words,np.uint32).view(np.float32)
    fused=np.empty(2,np.float32); plain=np.empty(2,np.float32)
    host.get_function('exp_variants')((1,),(128,),(x,fused,plain))
    if np.any(fused.view(np.uint32)==plain.view(np.uint32)):
        raise AssertionError('FMA probe inputs do not distinguish source variants')
    if not np.array_equal(reference['pres2_p'][-1,:2].view(np.uint32),fused.view(np.uint32)):
        raise AssertionError('interface pressure did not reach both discriminating FMA inputs')
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        device={n:xp.asarray(x) for n,x in fields.items()}
        prepare_interfaces(fields=device,nlev=nz,ncells=nc,gravity=9.81)
        for n in interfaces+mass+surface+('invalid',):
            actual=device[n].to_numpy(); expected=reference[n]
            count=np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32))
            if count: raise AssertionError(f'{n}: {count} differing words; max abs {np.max(np.abs(actual-expected))}')
    print('interface preparation matches host-selected source word for word')

if __name__=='__main__': main()
