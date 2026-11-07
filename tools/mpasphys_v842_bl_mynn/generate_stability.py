"""Generate the native MYNN surface stability and scale-aware factors."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_MODULE_CUDA as source

def func(name):
    m=re.search(r'__device__ __noinline__ (float|void) '+name+r'\s*\(.*?\)\s*\{',source,re.S);assert m
    start=m.start();i=m.end();j=i;depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    text=source[start:j].replace('__device__ __noinline__','static inline').replace('MynnFrame mynn_frame(arena);\n','').replace('float& psig','thread float& psig').replace('mynn_ipow<2>','mynn_square')
    text=text.replace('if (((zet) >= (0.0f)))', 'if (soft32_greater_equal(as_type<uint>(zet),0u))')
    for n in ('psig_bl','psig_shcu'):
        text=text.replace(f'(({n}) > (1.0f))',f'(soft32_greater(as_type<uint>({n}),0x3f800000u))').replace(f'(({n}) < (0.0f))',f'(soft32_less(as_type<uint>({n}),0u))')
    return text
name='hexv842_mpas_bl_mynn_stability_f32';header=(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_level2_f32.metal').read_text().split('\nkernel void ')[0]
header=header.replace('hexv842_mpas_bl_mynn_level2_f32_bindings.metalh',name+'_bindings.metalh')
header+='''
#include "hexv842_mpas_bl_mynn_stability_validate_f32_bindings.metalh"
#include "soft_urban_trig.metalh"
static inline float mynn_pow(float a,float b){
    uint x=as_type<uint>(a),y=as_type<uint>(b);
    if ((x&0x7fffffffu)>0x7f800000u && (y&0x7fffffffu)==0x3f800000u)
        return as_type<float>((x&0x7fffffffu)|0x00400000u);
    bool sx=(x&0x7fc00000u)==0x7f800000u && (x&0x003fffffu)!=0u;
    bool sy=(y&0x7fc00000u)==0x7f800000u && (y&0x003fffffu)!=0u;
    if(sx||sy)return mynn_result(a,b,as_type<float>(0xffc00000u));
    return mynn_result(a,b,gfk_pow(a,b));
}
static inline float mynn_log(float a){return mynn_result(a,a,gfk_log(a));}
static inline float mynn_atan(float a){return mynn_result(a,a,as_type<float>(soft_urban_atanf(as_type<uint>(a))));}
constant float cphm_unst=16.f,cphh_unst=16.f;
'''
header+='\n'.join(func(n) for n in ('f_phim','f_phih','f_scale_aware'))
resources=[['zet','float','read'],['dx','float','read'],['pblh','float','read'],['phim','float','read_write'],['phih','float','read_write'],['psig_bl','float','read_write'],['psig_shcu','float','read_write']]
params=[['count','int32']];typename='Hexv842MpasBlMynnStabilityF32'
header+=f'''\nkernel void {name}(constant {typename}Resources &r [[buffer(0)]],constant {typename}Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]) {{
if(i>=uint(p.count))return;float bl,sh;f_scale_aware(r.dx[i],r.pblh[i],bl,sh);r.phim[i]=f_phim(r.zet[i]);r.phih[i]=f_phih(r.zet[i]);r.psig_bl[i]=bl;r.psig_shcu[i]=sh;
}}\n'''
vname='hexv842_mpas_bl_mynn_stability_validate_f32'
header+=f'''\nkernel void {vname}(constant Hexv842MpasBlMynnStabilityValidateF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnStabilityValidateF32Params &p [[buffer(1)]],uint i [[thread_position_in_grid]]) {{
if(i>=uint(p.count))return;uint z=as_type<uint>(r.zet[i])&0x7fffffffu,d=as_type<uint>(r.dx[i]),h=as_type<uint>(r.pblh[i]);
r.invalid[i]=(z>=0x7f800000u || d<0x00800000u || d>=0x7f800000u || h<0x00800000u || h>=0x7f800000u);
}}\n'''
validate={'name':vname,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_scale_aware','resources':[['zet','float','read'],['dx','float','read'],['pblh','float','read'],['invalid','int32','read_write']],'shapes':{n:[['any_1d']] for n in ('zet','dx','pblh','invalid')},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']}
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'f_phim','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']},validate]}
(root/f'woof/metal/shaders/{name}.metal').write_text(header)
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
