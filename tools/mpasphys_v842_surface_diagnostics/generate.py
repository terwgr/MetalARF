"""Translate native post-LSM surface diagnostics without changing its operation graph."""
from pathlib import Path
import json,re
from woof.hex.mpasphys.sfclay_mm5_v842 import CUDA_SOURCE
ROOT=Path(__file__).resolve().parents[2]
start=CUDA_SOURCE.index('// MPAS mpas_atmphys_sfc_diagnostics.F:79-91')
end=CUDA_SOURCE.index('// MPAS physics_wrf/module_sf_sfcdiags.F:7-69 SFCDIAGS',start)
source=CUDA_SOURCE[start:end]
helpers,body=source.split('extern "C" __global__ void mm5_sfc_diagnostics(',1)
body=body[body.index('{')+1:body.rindex('}')]
body=body.replace('int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n || lsm==0) return;','if(i>=uint(p.n) || p.lsm==0) return;\nint n=p.n,lsm=p.lsm; float xice_threshold=p.threshold;')
for name in ('a','xice','q2mxy','t2mxy','constants'):
    body=re.sub(r'\b'+name+r'\[', 'r.'+name+'[',body)
for old,new in {'__device__':'static inline','__uint_as_float':'as_type<float>',
                '__float_as_uint':'as_type<uint>','FMUL':'native_diag_mul',
                'FSUB':'native_diag_sub','FDIV':'cu_div','mm5_pow':'gfk_pow'}.items():
    helpers=re.sub(r'\b'+old+r'\b',new,helpers)
    body=re.sub(r'\b'+old+r'\b',new,body)
# Retain IEEE comparisons even when the subtraction produces a subnormal.
body=body.replace('cqs2<1.e-5f','soft32_less(as_type<uint>(cqs2),as_type<uint>(1.e-5f))')
body=body.replace('chs2<1.e-5f','soft32_less(as_type<uint>(chs2),as_type<uint>(1.e-5f))')
expr='mm5_diag_sub(xland,1.5f)'
body=body.replace(expr+'>0.f','soft32_less(0u,as_type<uint>('+expr+'))')
body=body.replace(expr+'<=0.f','soft32_less_equal(as_type<uint>('+expr+'),0u)')
body=body.replace('r.xice[i]>=xice_threshold','soft32_less_equal(as_type<uint>(xice_threshold),as_type<uint>(r.xice[i]))')
struct=re.search(r'struct MM5Constants \{.*?\};',CUDA_SOURCE).group()
shader='''#include <metal_stdlib>
using namespace metal;
#include "cumulus_support.metalh"
#include "hexv842_mpas_surface_diagnostics_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float native_diag_mul(float a,float b) {return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float native_diag_sub(float a,float b) {return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
'''+struct+'\n'+helpers+'''
kernel void hexv842_mpas_surface_diagnostics_f32(
 constant Hexv842MpasSurfaceDiagnosticsF32Resources &r [[buffer(0)]],
 constant Hexv842MpasSurfaceDiagnosticsF32Params &p [[buffer(1)]],
 uint i [[thread_position_in_grid]]) {
'''+body+'\n}\n'
(ROOT/'woof/metal/shaders/hexv842_mpas_surface_diagnostics_f32.metal').write_text(shader)
resources=[['a','float','read_write'],['xice','float','read'],['q2mxy','float','read'],['t2mxy','float','read'],['constants','float','read']]
spec={'kernels':[dict(name='hexv842_mpas_surface_diagnostics_f32',module='hexv842_mpas_surface_diagnostics_f32',source='woof/metal/shaders/hexv842_mpas_surface_diagnostics_f32.metal',cuda_source='woof/hex/mpasphys/sfclay_mm5_v842.py',cuda_symbol='mm5_sfc_diagnostics',resources=resources,shapes={r[0]:[['any_1d']] for r in resources},parameters=[['n','int32'],['lsm','int32'],['threshold','float32']],threads_per_group=128,layout_version=1,arithmetic_policies=['default'])]}
(ROOT/'woof/metal/signature_fragments/hexv842_mpas_surface_diagnostics_f32.json').write_text(json.dumps(spec,indent=2)+'\n')
