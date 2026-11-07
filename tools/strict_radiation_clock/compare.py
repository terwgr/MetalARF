"""Compare the strict Metal radiation-call clock with the selected source body."""
from datetime import datetime, timezone
from pathlib import Path
import ctypes
import json
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
from woof.core.wrf_radiation_clock import radiation_call_scalars, wrf_radiation_coszen
from woof.core.rrtmg_legacy import _DEGRAD, _DPD
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.core.analytic_radiation import AnalyticClearSkyRadiation
import woof.metal.analytic_radiation as metal_analytic
from woof.wrf_exact import ENABLED

assert ENABLED, 'run with GPUWM_WRF_EXACT=1'
adapter=AnalyticClearSkyRadiation(datetime(2024,1,1,tzinfo=timezone.utc),
                                  np.zeros((1,),np.float32),np.zeros((1,),np.float32))
state=SimpleNamespace(_metal_setup_state=True)
fields={'sentinel':object()}
before=fields.copy()
def forbidden(*args,**kwargs):
    raise AssertionError('strict adapter called the ordinary flux kernel')
metal_analytic.clear_sky_fluxes=forbidden
for call in (adapter,adapter._call_metal):
    try:
        call(atmosphere={},fields=fields,state=state,cfg=None)
    except NotImplementedError as error:
        assert 'surface flux arithmetic' in str(error)
    else:
        raise AssertionError('strict analytic adapter was admitted')
    assert adapter.update_count == 0 and fields == before

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    unit = (ROOT / 'woof/core/kernels/swint.cu').read_text()
    body = unit[unit.index('extern "C" __global__ void swint_coszen_loc('):unit.index('// The exponent of one power-law fit')]
    source = tmp / 'source.cpp'
    source.write_text(f'#include "{ROOT / "tools/glibc_trig_flt32_proof/host_shim.hpp"}"\n'
        f'#include "{ROOT / "woof/core/kernels/glibc_trig_flt32.cuh"}"\n'
        '#define __global__\n#define FADD __fadd_rn\n#define FSUB __fsub_rn\n'
        '#define FMUL __fmul_rn\n#define FDIV __fdiv_rn\n'
        'struct Idx { int x; }; static Idx blockIdx={0}, threadIdx={0}, blockDim={1};\n'
        + body + '\nextern "C" void ref(long long n,const float *a,const float *b,float j,float x,float g,float d,float p,float *out) {'
        'for(int i=0;i<n;i++){threadIdx.x=i;swint_coszen_loc(n,a,b,j,x,g,d,p,0,out);}}\n')
    lib = tmp / 'ref.dylib'
    subprocess.run(['clang++','-std=c++17','-O2','-ffp-contract=off','-fno-fast-math','-shared','-fPIC',str(source),'-o',str(lib)],check=True)
    reference = ctypes.CDLL(str(lib)).ref
    reference.argtypes = [ctypes.c_int64,ctypes.c_void_p,ctypes.c_void_p] + [ctypes.c_float]*5 + [ctypes.c_void_p]
    lat=np.linspace(-90,90,257,dtype=np.float32); lon=np.linspace(-180,180,257,dtype=np.float32)
    cases=[(datetime(2024,1,1,tzinfo=timezone.utc),0.,0.),(datetime(2024,3,20,23,59,tzinfo=timezone.utc),61.,7.5),
           (datetime(2024,6,20,12,tzinfo=timezone.utc),3600.,30.),(datetime(2024,12,31,23,tzinfo=timezone.utc),7200.,0.05)]
    words=diff=0
    with MetalContext() as context:
        arrays=MetalArrayModule(context); dlat=arrays.asarray(lat); dlon=arrays.asarray(lon)
        for start,elapsed,radt in cases:
            j,x,g=radiation_call_scalars(start,elapsed,radt)
            expected=np.empty_like(lat)
            reference(lat.size,lat.ctypes.data,lon.ctypes.data,j,x,g,_DEGRAD,_DPD,expected.ctypes.data)
            actual=wrf_radiation_coszen(start,elapsed,radt,dlat,dlon).to_numpy()
            words+=lat.size; diff+=int(np.count_nonzero(expected.view(np.uint32)!=actual.view(np.uint32)))
    print(json.dumps({'words':words,'different_words':diff,'reference':'selected CUDA body compiled on host'}))
    if diff: raise SystemExit(1)
