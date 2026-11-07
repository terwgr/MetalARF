"""Compare strict Noah forcing and surface diagnostics to selected source bodies."""
from pathlib import Path
import ctypes, json, subprocess, tempfile
import numpy as np
from woof.core import noah_forcing, noah_sfcdiags
from woof.core.physics import PhysicsDriver
from woof.metal.noah_family import submit_noah_column
from woof.metal.noah_mosaic_family import submit_noah_mosaic_column
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED
from types import SimpleNamespace

assert ENABLED
state=SimpleNamespace(_metal_setup_state=True)
for fn,args in ((PhysicsDriver._run_noah,(SimpleNamespace(state=state),{},SimpleNamespace(sf_surface_mosaic=1),1)),
                (PhysicsDriver._run_noah_mosaic,(SimpleNamespace(state=state),{},None,1))):
    try: fn(*args)
    except NotImplementedError: pass
    else: raise AssertionError('unsupported Noah solver not refused')
try:
    submit_noah_column({},None,0,None,urban={})
except NotImplementedError: pass
else: raise AssertionError('strict Noah urban solver not refused')
try:
    submit_noah_mosaic_column({},None,0,None,mosaic_cat=1,categories=None,
        xice_threshold=.5,frpcpn=False,usemonalb=False,rdlai2d=False,opt_thcnd=1,itimestep=1)
except NotImplementedError: pass
else: raise AssertionError('strict Noah mosaic solver not refused')
root=Path(__file__).resolve().parents[2]
shim=root/'tools/glibc_trig_flt32_proof/host_shim.hpp'
with tempfile.TemporaryDirectory() as temp:
    d=Path(temp); source=d/'ref.cpp'; lib=d/'ref.dylib'
    source.write_text(f'#include "{shim}"\n#define __int_as_float __uint_as_float\n#define __global__\n#define __forceinline__ inline\n'
      'struct Idx { int x; }; static Idx blockIdx={0}, threadIdx={0}, blockDim={1};\n'
      + noah_forcing._SOURCE + noah_sfcdiags._source())
    subprocess.run(['clang++','-std=c++17','-O2','-ffp-contract=off','-fno-fast-math','-shared','-fPIC',str(source),'-o',str(lib)],check=True)
    # Source kernels use block/thread index 0; advance each pointer by one float.
    ref=ctypes.CDLL(str(lib)); f=ref.noah_forcing; clear=ref.noah_rain_clear; s=ref.noah_sfcdiags
    ptr=ctypes.c_void_p; flt=ctypes.c_float; integer=ctypes.c_int; lng=ctypes.c_int64
    f.argtypes=[ptr]*7+[integer,flt]+[ptr]*8+[lng]
    clear.argtypes=[ptr,ptr,lng]
    s.argtypes=[ptr]*11+[flt]*5+[lng]
    n=24; shape=(3,8); i=np.arange(n,dtype=np.float32)
    words=diff=0; addr=lambda a,k: int(a.ctypes.data)+4*k
    with MetalContext() as context:
        xp=MetalArrayModule(context)
        for use in (False,True):
            h={name:np.array(v,dtype=np.float32).reshape(shape) for name,v in {
              'p_if0':90000+i*21,'p_if1':89000+i*11,'temperature0':260+i,
              'qv0':.002+i*.0001,'dz0':1+i*.1,'br':-.02+i*.001,
              'sr_scheme':i/24,'pending':.001+i*.0001,'rainbl':.1+i*.0002,
              'sfcprs':i*0,'sfctmp':i*0,'qv1':i*0,'dz8w1':i*0,'rib':i*0,'sr':i*0}.items()}
            dct={k:xp.asarray(v) for k,v in h.items()}
            noah_forcing.forcing(dct,p_if0=dct['p_if0'],p_if1=dct['p_if1'],temperature0=dct['temperature0'],qv0=dct['qv0'],dz0=dct['dz0'],sr_scheme=dct['sr_scheme'] if use else None,pending=dct['pending'],freezing=273.15)
            for k in range(n):
                f(*[addr(h[x],k) for x in ('p_if0','p_if1','temperature0','qv0','dz0','br','sr_scheme')],int(use),273.15,
                  *[addr(h[x],k) for x in ('pending','sfcprs','sfctmp','qv1','dz8w1','rib','sr','rainbl')],1)
            for name in ('sfcprs','sfctmp','qv1','dz8w1','rib','sr','rainbl'):
                actual=dct[name].to_numpy(); words+=n; diff+=int(np.count_nonzero(actual.view(np.uint32)!=h[name].view(np.uint32)))
            noah_forcing.rain_clear(dct['rainbl'],dct['pending'])
            for k in range(n):clear(addr(h['rainbl'],k),addr(h['pending'],k),1)
            for name in ('rainbl','pending'):
                actual=dct[name].to_numpy(); words+=n; diff+=int(np.count_nonzero(actual.view(np.uint32)!=h[name].view(np.uint32)))
        x=np.arange(n,dtype=np.float32)
        h={k:np.array(v,dtype=np.float32).reshape(shape) for k,v in {
          'psfc':80000+x*900,'tsk':260+x*1.2,'cqs2':np.where(x%3==0,0.,.01),'chs2':np.where(x%4==0,0.,.02),
          'qsfc':np.where(x%5==0,-.001,.01),'qfx':.0001+x*.00001,'hfx':2+x,'qv1':.004+x*.0001,
          'q2':x*0,'t2':x*0,'th2':x*0}.items()}
        dct={k:xp.asarray(v) for k,v in h.items()}
        args=dict(rd=287.04,cp_air=1004.5,p0=100000.,rcp=.2857143,active_floor=1.e-8)
        noah_sfcdiags.refresh(dct,dct['qv1'],**args)
        for k in range(n):s(*[addr(h[x],k) for x in ('psfc','tsk','cqs2','chs2','qsfc','qfx','hfx','qv1','q2','t2','th2')],*[args[x] for x in ('rd','cp_air','p0','rcp','active_floor')],1)
        for name in ('q2','t2','th2'):
            actual=dct[name].to_numpy(); words+=n; diff+=int(np.count_nonzero(actual.view(np.uint32)!=h[name].view(np.uint32)))
    print(json.dumps({'words':words,'different_words':diff,'reference':'selected CUDA source compiled for host','unsupported_solvers_refused':True}))
    if diff: raise SystemExit(1)
