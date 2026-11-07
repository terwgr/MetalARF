"""Translate the native MPAS Thompson input/output and precipitation boundaries."""
from pathlib import Path
import json,re
from woof.hex.mpasphys import thompson_v842 as source
ROOT=Path(__file__).resolve().parents[2]
SHADERS=ROOT/'woof/metal/shaders'

def function(name,kind='float'):
    match=re.search(r'__device__ '+kind+' '+name+r'\('+('float a' if name!='tc_nan' else 'float a')+r'[^)]*\) \{',source._SUPPORT)
    assert match,name
    start=source._SUPPORT.index('{',match.start());end=source._body_end(source._SUPPORT,start)
    return source._SUPPORT[match.start():end]
helpers='\n'.join(function(name,'bool' if name=='tc_nan' else 'float') for name in ('tc_nan','tc_quiet','tc_add','tc_sub','tc_mul','tc_div','tc_pow'))
for old,new in {'__device__':'static inline','__float_as_uint':'as_type<uint>','__uint_as_float':'as_type<float>',
                '__fadd_rn':'native_tc_add','__fsub_rn':'native_tc_sub','__fmul_rn':'native_tc_mul',
                '__fdiv_rn':'native_tc_div','mpas_powf':'native_tc_powf'}.items():
    helpers=re.sub(r'\b'+old+r'\b',new,helpers)
header='''#ifndef WOOF_MPAS_THOMPSON_SCALAR_METALH
#define WOOF_MPAS_THOMPSON_SCALAR_METALH
#include "thompson_aerosol_exner_libm.metalh"
static inline float native_tc_add(float a,float b) {return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float native_tc_sub(float a,float b) {return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
static inline float native_tc_mul(float a,float b) {return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float native_tc_div(float a,float b) {return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));}
static inline float native_tc_powf(float a,float b) {return as_type<float>(thompson_exner_powf_bits(as_type<uint>(a),as_type<uint>(b)));}
'''+helpers+'''
static inline int tc_add(int a,int b) {return a+b;}
static inline float tc_max(float a,float b) {return soft32_less(as_type<uint>(b),as_type<uint>(a))?a:b;}
#endif
'''
(SHADERS/'hexv842_mpas_thompson_scalar.metalh').write_text(header)

for phase in ('from','to'):
    text=source._CONVERSIONS
    match=re.search(r'extern "C" __global__ void mpas_thompson_'+phase+r'\(',text)
    start=text.index('{',match.start());end=source._body_end(text,start);body=text[start+1:end-1]
    body=body.replace('ARRAY_ALIASES','\n'.join('device float* '+name+'=r.'+name+';' for name in source._CONVERSION_ARRAYS))
    body=body.replace('int idx=blockIdx.x*blockDim.x+threadIdx.x;','int idx=int(gid);').replace('int col=blockIdx.x*blockDim.x+threadIdx.x;','int col=int(gid);')
    params='int nz=p.nz,columns=p.columns,top=p.top;float dt_dyn=p.dt_dyn;\n'
    params+='\n'.join('float '+name+'=p.'+name+';' for name in source.CONVERSION_CONSTANTS)+'\n'
    entry='hexv842_mpas_thompson_'+phase+'_f32';title=''.join(p.title() for p in entry.split('_'))
    shader='#include <metal_stdlib>\nusing namespace metal;\n#include "hexv842_mpas_thompson_scalar.metalh"\n#include "'+entry+'_bindings.metalh"\n#pragma clang fp contract(off)\n'
    shader+='kernel void '+entry+'(constant '+title+'Resources &r [[buffer(0)]],constant '+title+'Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {\n'+params+body+'\n}\n'
    (SHADERS/(entry+'.metal')).write_text(shader)
    resources=[[name,'float','read_write'] for name in source._CONVERSION_ARRAYS]
    parameters=[['nz','int32'],['columns','int32'],['top','int32'],['dt_dyn','float32']]+[[name,'float32'] for name in source.CONVERSION_CONSTANTS]
    spec={'kernels':[dict(name=entry,module=entry,source='woof/metal/shaders/'+entry+'.metal',cuda_source='woof/hex/mpasphys/thompson_v842.py',cuda_symbol='mpas_thompson_'+phase,resources=resources,shapes={name:[['any_1d']] for name,_,_ in resources},parameters=parameters,threads_per_group=32,layout_version=1,arithmetic_policies=['default'])]}
    (ROOT/'woof/metal/signature_fragments'/(entry+'.json')).write_text(json.dumps(spec,indent=2)+'\n')

text=source._PRECIP;start=text.index('{');end=source._body_end(text,start);body=text[start+1:end-1]
body=body.replace('int col=blockIdx.x*blockDim.x+threadIdx.x;','int col=int(gid);')
body=body.replace('for(int n=0;n<7;n++)work[n][col]=0.0f;','\n'.join('work['+str(n)+'][col]=0.0f;' for n in range(7)))
for n,name in enumerate(source.SURFACE_ARGUMENTS[:7]):body=body.replace('work['+str(n)+'][','r.work_'+name+'[')
for n,name in enumerate(source.PRECIP_ARGUMENTS):body=body.replace('diag['+str(n)+'][','r.diag_'+name+'[')
for name in ('qv','rho','dz','i_rainnc'):body=re.sub(r'\b'+name+r'\[','r.'+name+'[',body)
old=source.CORRECT_MODE_SOURCE['WB3'][0][0];new=source.CORRECT_MODE_SOURCE['WB3'][0][1]
assert body.count(old)==0 # pointer names were already bound
old=old.replace('rho[idx]','r.rho[idx]').replace('qv[idx]','r.qv[idx]')
assert body.count(old)==1
body=body.replace(old,'float rho_a=p.correct_wb3 ? r.rho[idx] : tc_div(r.rho[idx],(tc_add(1.0f,r.qv[idx])));')
body=body.replace('bucket>0.0f','soft32_less(0u,as_type<uint>(bucket))').replace('r.diag_rainnc[col]>bucket','soft32_less(as_type<uint>(bucket),as_type<uint>(r.diag_rainnc[col]))')
entry='hexv842_mpas_thompson_precip_f32';title='Hexv842MpasThompsonPrecipF32'
shader='#include <metal_stdlib>\nusing namespace metal;\n#include "hexv842_mpas_thompson_scalar.metalh"\n#include "'+entry+'_bindings.metalh"\n#pragma clang fp contract(off)\n'
shader+='kernel void '+entry+'(constant '+title+'Resources &r [[buffer(0)]],constant '+title+'Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {\nint nz=p.nz,columns=p.columns,phase=p.phase,l_acrain=p.l_acrain;float bucket=p.bucket;\n'+body+'\n}\n'
(SHADERS/(entry+'.metal')).write_text(shader)
resources=[[name,'float','read'] for name in ('qv','rho','dz')]+[['work_'+name,'float','read_write'] for name in source.SURFACE_ARGUMENTS[:7]]+[['diag_'+name,'float','read_write'] for name in source.PRECIP_ARGUMENTS]+[['i_rainnc','int32','read_write']]
parameters=[['nz','int32'],['columns','int32'],['phase','int32'],['l_acrain','int32'],['correct_wb3','int32'],['bucket','float32']]
spec={'kernels':[dict(name=entry,module=entry,source='woof/metal/shaders/'+entry+'.metal',cuda_source='woof/hex/mpasphys/thompson_v842.py',cuda_symbol='mpas_thompson_precip',resources=resources,shapes={name:[['any_1d']] for name,_,_ in resources},parameters=parameters,threads_per_group=32,layout_version=1,arithmetic_policies=['default'])]}
(ROOT/'woof/metal/signature_fragments'/(entry+'.json')).write_text(json.dumps(spec,indent=2)+'\n')
