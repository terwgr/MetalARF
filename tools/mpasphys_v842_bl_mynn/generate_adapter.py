"""Generate native MPAS MYNN moisture pre/post and radiation kernels."""
from pathlib import Path
import json,re,sys
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
from woof.hex.mpasphys.bl_mynn_v842 import ADAPTER_CUDA_SOURCE

header='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_adapter_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float mynn_result(float a,float b,float value) {
    uint x=as_type<uint>(a),y=as_type<uint>(b),r=as_type<uint>(value);
    if ((r&0x7fffffffu)<=0x7f800000u) return value;
    if ((x&0x7fffffffu)>0x7f800000u) return as_type<float>(x|0x00400000u);
    if ((y&0x7fffffffu)>0x7f800000u) return as_type<float>(y|0x00400000u);
    return as_type<float>(0xffc00000u);
}
static inline float mynn_add(float a,float b){return mynn_result(a,b,mpas_add(a,b));}
static inline float mynn_sub(float a,float b){return mynn_result(a,b,mpas_sub(a,b));}
static inline float mynn_mul(float a,float b){return mynn_result(a,b,mpas_mul(a,b));}
static inline float mynn_div(float a,float b){return mynn_result(a,b,cu_div(a,b));}
#define mpas_add mynn_add
#define mpas_sub mynn_sub
#define mpas_mul mynn_mul
#define mpas_div mynn_div
'''
kernels=[];text=header
for native in ('mynn_pre','mynn_pre_rp1','mynn_post','mynn_post_rp1','mynn_radiation'):
    pattern=r'extern "C" __global__ void '+native+r'\((.*?)\)\s*\{'
    match=re.search(pattern,ADAPTER_CUDA_SOURCE,re.S);assert match
    i=match.end();j=i;depth=1
    while depth:
        if ADAPTER_CUDA_SOURCE[j]=='{':depth+=1
        elif ADAPTER_CUDA_SOURCE[j]=='}':depth-=1
        j+=1
    body=ADAPTER_CUDA_SOURCE[i:j-1]
    resources=[];params=[];declarations=[]
    for arg in match.group(1).split(','):
        arg=' '.join(arg.strip().split());q=re.fullmatch(r'(const )?(int|float)(\*)?\s*(\w+)',arg);assert q,arg
        const,typ,pointer,name=q.groups()
        if pointer:
            resources.append([name,'float','read' if const else 'read_write'])
            declarations.append('device '+('const ' if const else '')+'float* '+name+'=r.'+name+';')
        else:
            params.append([name,'int32' if typ=='int' else 'float32'])
            declarations.append(typ+' '+name+'=p.'+name+';')
    body=body.replace('blockIdx.x * blockDim.x + threadIdx.x','gid')
    name='hexv842_mpas_bl_'+native+'_f32'
    typename=''.join(part.title() for part in name.split('_'))
    text+=f'\nkernel void {name}(constant {typename}Resources &r [[buffer(0)]],constant {typename}Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{\n'+''.join(declarations)+body+'\n}\n'
    kernels.append({'name':name,'module':'hexv842_mpas_bl_mynn_adapter_f32','source':'woof/metal/shaders/hexv842_mpas_bl_mynn_adapter_f32.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':native,'resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']})
for native,resources,params,body in (
    ('mynn_dx_validate', [['mesh_density','float','read'],['invalid','int32','read_write']], [['ncol','int32']],
     "uint bits=as_type<uint>(r.mesh_density[gid]);r.invalid[gid]=(bits<0x00800000u || bits>=0x7f800000u);"),
    ('mynn_dx', [['mesh_density','float','read'],['dx','float','read_write']], [['ncol','int32'],['len_disp','float32']],
     "float d=r.mesh_density[gid];float power=as_type<float>(soft_gfk_pow_bits(as_type<uint>(d),0x3e800000u));r.dx[gid]=mynn_div(p.len_disp,power);")):
    name='hexv842_mpas_bl_'+native+'_f32';typename=''.join(part.title() for part in name.split('_'))
    text+=f'\nkernel void {name}(constant {typename}Resources &r [[buffer(0)]],constant {typename}Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {{if(gid>=uint(p.ncol))return;{body}}}\n'
    kernels.append({'name':name,'module':'hexv842_mpas_bl_mynn_adapter_f32','source':'woof/metal/shaders/hexv842_mpas_bl_mynn_adapter_f32.metal','cuda_source':'woof/hex/mpasphys/bl_mynn_v842.py','cuda_symbol':'mynn_dx','resources':resources,'shapes':{x[0]:[['any_1d']] for x in resources},'parameters':params,'threads_per_group':128,'layout_version':1,'arithmetic_policies':['default']})
(root/'woof/metal/shaders/hexv842_mpas_bl_mynn_adapter_f32.metal').write_text(text)
(root/'woof/metal/signature_fragments/hexv842_mpas_bl_mynn_adapter_f32.json').write_text(json.dumps({'kernels':kernels},indent=2)+'\n')
