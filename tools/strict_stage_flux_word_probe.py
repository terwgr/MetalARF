"""Check strict Metal stage fluxes against retained source operation trees.

Elementwise CUDA bodies compile for the host; Python face operations use
NumPy's separate binary32 rounds. Neither reference is NVIDIA-device evidence.
"""
import ctypes as ct
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np

os.environ['GPUWM_WRF_EXACT'] = '1'
from woof.core import dycore, ieva
from woof.metal.array import MetalArrayModule

# Capture the production elementwise bodies without executing a CUDA kernel.
saved_cp = dycore.cp
dycore.cp = SimpleNamespace(ElementwiseKernel=lambda *a, **kw: a)
bodies = {}
for mapped in (False, True):
    for reciprocal in (False, True):
        bodies['couple_%d_%d' % (mapped, reciprocal)] = dycore._couple_momentum_kernel(mapped, reciprocal)[2]
    bodies['omega_%d' % mapped] = dycore._omega_column_kernel(mapped)[2]
dycore.cp = saved_cp
dycore._couple_momentum_kernel.cache_clear()
dycore._omega_column_kernel.cache_clear()

prelude = '''#include <stddef.h>
typedef float T;
static float __fadd_rn(float a,float b){volatile float r=a+b;return r;}
static float __fsub_rn(float a,float b){volatile float r=a-b;return r;}
static float __fmul_rn(float a,float b){volatile float r=a*b;return r;}
static float __fdiv_rn(float a,float b){volatile float r=a/b;return r;}
'''
unit = prelude
for name, body in bodies.items():
    if name.startswith('couple'):
        unit += 'extern "C" void '+name+'(const float* winds,const float* c1h,const float* c2h,const float* muface,const float* msf,float* out,int ncol,int count){for(size_t i=0;i<(size_t)count;++i){T wind=winds[i],flux;'+body+'out[i]=flux;}}\n'
    else:
        unit += 'extern "C" void '+name+'(const float* ru,const float* rv,const float* dnw,const float* c1h,const float* msft,float rdx,float rdy,int nz,int ny,int nx,float* ww){for(size_t i=0;i<(size_t)(ny*nx);++i){'+body+'}}\n'

class State(SimpleNamespace):
    def scratch(self, shape, slot):
        if slot not in self.slots:
            self.slots[slot] = self.xp.zeros(shape, dtype=np.float32)
        return self.slots[slot]

    def existing_scratch(self, slot):
        return self.slots.get(slot)

    def total_mu(self):
        if getattr(self, "_metal_setup_state", False):
            from woof.metal.dycore import submit_total_mu
            return submit_total_mu(self)
        return self.mub2d + self.mup

with tempfile.TemporaryDirectory() as tmp:
    cpp = Path(tmp)/'ref.cpp'; library = Path(tmp)/'ref.dylib'
    cpp.write_text(unit)
    subprocess.run(['clang++','-O0','-std=c++17','-ffp-contract=off','-dynamiclib',str(cpp),'-o',str(library)],check=True)
    dll = ct.CDLL(str(library)); ptr = ct.POINTER(ct.c_float)
    for name in bodies:
        if name.startswith('couple'):
            getattr(dll,name).argtypes = [ptr]*6+[ct.c_int]*2
        else:
            getattr(dll,name).argtypes = [ptr]*5+[ct.c_float]*2+[ct.c_int]*3+[ptr]
    def address(a):
        return a.ctypes.data_as(ptr)
    rng = np.random.default_rng(715); xp = MetalArrayModule(); cases = words = old_omega_differences = inverse_differences = 0
    try:
        for mapped in (False, True):
            for boundary in ('periodic','open_x','open_y','specified','nested'):
                for first_stage in (False, True):
                    nz,ny,nx = 7,3,7
                    h = State(xp=np,slots={},has_msf=mapped,p=np.zeros((nz,ny,nx),np.float32))
                    cfg = SimpleNamespace(dx=1200.,dy=1700.,open_x=boundary=='open_x',open_y=boundary=='open_y',specified=boundary=='specified',nested=boundary=='nested')
                    for name,shape in (('mup',(ny,nx)),('mub2d',(ny,nx)),('u',(nz,ny,nx+1)),('v',(nz,ny+1,nx)),('c1h',(nz,)),('c2h',(nz,)),('dnw',(nz,)),('msft',(ny,nx)),('msfu',(ny,nx+1)),('msfv',(ny+1,nx))):
                        a = rng.uniform(.5,2,shape).astype(np.float32)
                        if name in ('mup','mub2d','u','v'):
                            a.flat[:4] = np.array([0,0x80000000,1,0x80000001],np.uint32).view(np.float32)
                        setattr(h,name,a)
                    # Interior face: exactly one unit mass and wind makes
                    # the one-ULP stored inverse difference survive coupling.
                    h.mup[0:2,2] = np.float32(0.)
                    h.mub2d[0:2,2] = np.float32(1.)
                    h.c1h[2] = np.float32(1.)
                    h.c2h[2] = np.float32(0.)
                    h.v[2,1,2] = np.float32(1.)
                    h.msfv[1,2] = np.float32(1.)
                    h.msfv_inv = np.nextafter(np.float32(1.0) / h.msfv, np.float32(np.inf))
                    # A different retained ring makes the first-stage halo branch observable.
                    h.slots[dycore.WRF_MU_HALO_SLOT] = np.nextafter(h.mup,np.float32(np.inf))
                    m = State(xp=xp,slots={},_metal_setup_state=True,_array_module=xp,has_msf=mapped)
                    for name in ('p','mup','mub2d','u','v','c1h','c2h','dnw','msft','msfu','msfv','msfv_inv'):
                        setattr(m,name,xp.asarray(getattr(h,name)))
                    m.slots[dycore.WRF_MU_HALO_SLOT] = xp.asarray(h.slots[dycore.WRF_MU_HALO_SLOT])
                    # Use the actual face source, with only its array module changed.
                    old = ieva.cp; ieva.cp = np
                    mux,muy = ieva.stage_face_masses(h,cfg,h.total_mu())
                    ieva.cp = old
                    expected = {}
                    for axis,wind,face,msf in (('u',h.u,mux,h.msfu),('v',h.v,muy,h.msfv_inv)):
                        result = np.zeros_like(wind)
                        getattr(dll,'couple_%d_%d' % (mapped,axis=='v'))(*map(address,(wind,h.c1h,h.c2h,face,msf,result)),face.size,wind.size)
                        expected['rk_r'+axis] = result
                    if mapped:
                        recomputed = np.zeros_like(h.v)
                        reciprocal = np.float32(1.) / h.msfv
                        dll.couple_1_1(*map(address, (h.v,h.c1h,h.c2h,muy,reciprocal,recomputed)), muy.size, h.v.size)
                        assert expected['rk_rv'][2,1,2].view(np.uint32) != recomputed[2,1,2].view(np.uint32), 'inverse fixture is not discriminating'
                        inverse_differences += int(np.count_nonzero(expected['rk_rv'].view(np.uint32) != recomputed.view(np.uint32)))
                    old = dycore.cp; dycore.cp = np
                    fu,fv = dycore._calc_ww_cp_fluxes(h,cfg,first_stage=first_stage)
                    dycore.cp = old
                    expected.update(rk_mux=mux,rk_muy=muy,rk_ww_ru=fu,rk_ww_rv=fv)
                    ww = np.zeros((nz+1,ny,nx),np.float32)
                    getattr(dll,'omega_%d' % mapped)(*map(address,(fu,fv,h.dnw,h.c1h,h.msft)),np.float32(1)/np.float32(cfg.dx),np.float32(1)/np.float32(cfg.dy),nz,ny,nx,address(ww))
                    expected['rk_ww'] = ww
                    old_ww = np.zeros_like(ww)
                    getattr(dll,'omega_%d' % mapped)(*map(address,(expected['rk_ru'],expected['rk_rv'],h.dnw,h.c1h,h.msft)),np.float32(1)/np.float32(cfg.dx),np.float32(1)/np.float32(cfg.dy),nz,ny,nx,address(old_ww))
                    old_omega_differences += int(np.count_nonzero(old_ww.view(np.uint32) != ww.view(np.uint32)))
                    dycore.stage_fluxes(m,cfg,first_stage=first_stage)
                    for slot,want in expected.items():
                        got = m.slots[slot].to_numpy()
                        np.testing.assert_array_equal(got.view(np.uint32),want.view(np.uint32),err_msg=str((mapped,boundary,first_stage,slot)))
                        words += want.size
                    # Exercise the production refresh, with separate outputs.
                    refresh_ru = xp.zeros(h.u.shape, dtype=np.float32)
                    refresh_rv = xp.zeros(h.v.shape, dtype=np.float32)
                    dycore.refresh_saved_wind_fluxes(m, cfg, refresh_ru, refresh_rv)
                    for name, got in (('rk_ru', refresh_ru), ('rk_rv', refresh_rv)):
                        np.testing.assert_array_equal(got.to_numpy().view(np.uint32), expected[name].view(np.uint32),
                                                      err_msg=str(('refresh',mapped,boundary,first_stage,name)))
                        words += expected[name].size
                    cases += 1
    finally:
        xp.context.close()
    assert inverse_differences > 0, 'stored inverse is not discriminating'
    assert old_omega_differences > 0, 'inputs do not discriminate the discarded shared-flux construction'
    print(dict(cases=cases,words=words,mismatched_words=0,old_shared_flux_omega_differing_words=old_omega_differences,stored_inverse_differing_words=inverse_differences,host_flags='-O0 -ffp-contract=off'))
