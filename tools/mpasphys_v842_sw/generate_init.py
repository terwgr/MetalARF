"""Generate the native v842 shortwave coefficient initialization kernels."""
from pathlib import Path
import json
from woof.hex.mpasphys.rrtmg_sw_v842 import CUDA_SOURCE
root=Path(__file__).resolve().parents[2]
start=CUDA_SOURCE.index('__device__ const float MPAS_STATIC_TABLES[6936]')
end=CUDA_SOURCE.index('extern "C" __global__ void mpas_sw_weights',start)
constants=CUDA_SOURCE[start:end].replace('__device__ const ', 'constant ')
shader='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_gfk_exp.metalh"
#include "hexv842_sw_weights_f32_bindings.metalh"
#include "hexv842_sw_reduce_f32_bindings.metalh"
#include "hexv842_sw_static_f32_bindings.metalh"
#include "hexv842_sw_exp_f32_bindings.metalh"
#pragma clang fp contract(off)
'''+constants
shader+='''
kernel void hexv842_sw_weights_f32(constant Hexv842SwWeightsF32Resources &r [[buffer(0)]],
    constant Hexv842SwWeightsF32Params &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    int b=int(tid); if(b>=14) return; int g=0;
    for(int j=MPAS_NGS[b];j<MPAS_NGS[b+1];j++) {
        int start=g; float sum=0.f;
        for(int n=0;n<MPAS_NGN[j];n++) sum=mpas_add(sum,MPAS_WT[g++]);
        for(int k=start;k<g;k++) r.rwgt[b*16+k]=mpas_div(MPAS_WT[k],sum);
    }
}
kernel void hexv842_sw_reduce_f32(constant Hexv842SwReduceF32Resources &r [[buffer(0)]],
    constant Hexv842SwReduceF32Params &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    int i=int(tid), b=p.band, rows=p.rows; if(i>=p.count) return;
    int group=p.first_axis?i%MPAS_NGC[b]:i/rows;
    int row=p.first_axis?i/MPAS_NGC[b]:i%rows;
    int start=0;for(int j=0;j<group;j++)start+=MPAS_NGN[MPAS_NGS[b]+j];
    float sum=0.f;for(int n=0;n<MPAS_NGN[MPAS_NGS[b]+group];n++) {
        int g=start+n;float x=r.raw[p.first_axis?g+16*row:row+rows*g];
        sum=mpas_add(sum,p.unweighted?x:mpas_mul(x,r.rwgt[b*16+g]));
    } r.reduced[i]=sum;
}
kernel void hexv842_sw_static_f32(constant Hexv842SwStaticF32Resources &r [[buffer(0)]],
    constant Hexv842SwStaticF32Params &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if(tid<6936) r.out[tid]=MPAS_STATIC_TABLES[tid];
}
kernel void hexv842_sw_exp_f32(constant Hexv842SwExpF32Resources &r [[buffer(0)]],
    constant Hexv842SwExpF32Params &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    int i=int(tid);if(i>10000)return;
    if(i==0){r.out[i]=1.f;return;}if(i==10000){r.out[i]=1.e-20f;return;}
    float tfn=mpas_div(float(i),10000.f),tau=mpas_div(mpas_mul(as_type<float>(1080440641u),tfn),mpas_sub(1.f,tfn));
    r.out[i]=fmax(1.e-20f,as_type<float>(soft_gfk_exp(as_type<uint>(-tau))));
}
'''
(root/'woof/metal/shaders/hexv842_sw_init_f32.metal').write_text(shader)
kernels=[]
for op,res,params,symbol in [('weights',[('rwgt','float','read_write')],[('reserved','int32')],'mpas_sw_weights'),('reduce',[('raw','float','read'),('rwgt','float','read'),('reduced','float','read_write')],[(x,'int32') for x in ('band','rows','first_axis','unweighted','count')],'mpas_sw_reduce_coefficients'),('static',[('out','float','read_write')],[('reserved','int32')],'mpas_sw_static_tables'),('exp',[('out','float','read_write')],[('reserved','int32')],'mpas_sw_exp_table')]:
    name=f'hexv842_sw_{op}_f32'
    kernels.append(dict(name=name,module='hexv842_sw_init_f32',source='woof/metal/shaders/hexv842_sw_init_f32.metal',cuda_source='woof/hex/mpasphys/rrtmg_sw_v842.py',cuda_symbol=symbol,resources=res,shapes={x[0]:[['any_1d']] for x in res},parameters=params,threads_per_group=128,layout_version=1,arithmetic_policies=['default']))
(root/'woof/metal/signature_fragments/hexv842_sw_init_f32.json').write_text(json.dumps({'kernels':kernels},indent=2)+'\n')
