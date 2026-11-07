"""Generate native level/cell to radiation column layout conversion."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
s='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_sw_layout_f32_bindings.metalh"
kernel void hexv842_sw_layout_f32(constant Hexv842SwLayoutF32Resources &r [[buffer(0)]],
 constant Hexv842SwLayoutF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int t=int(tid);if(t>=p.nc*p.nl)return;int k=t/p.nc,c=t%p.nc;
 int column=c*p.nl+k;float v=p.reverse?r.src[column]:r.src[t];
 if(p.divide)v=mpas_div(v,p.divisor);
 r.dst[p.reverse?t:column]=v;
}
'''
(root/'woof/metal/shaders/hexv842_sw_layout_f32.metal').write_text(s)
res=[('src','float','read'),('dst','float','read_write')]
k=dict(name='hexv842_sw_layout_f32',module='hexv842_sw_layout_f32',source='woof/metal/shaders/hexv842_sw_layout_f32.metal',cuda_source='woof/hex/mpasphys/rrtmg_sw_v842.py',cuda_symbol='radiation_sw_from_MPAS',resources=res,shapes={r[0]:[['any_1d']] for r in res},parameters=[('nc','int32'),('nl','int32'),('reverse','int32'),('divide','int32'),('divisor','float32')],threads_per_group=128,layout_version=1,arithmetic_policies=['default'])
(root/'woof/metal/signature_fragments/hexv842_sw_layout_f32.json').write_text(json.dumps({'kernels':[k]},indent=2)+'\n')
