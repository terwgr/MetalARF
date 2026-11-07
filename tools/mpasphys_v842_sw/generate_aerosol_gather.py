"""Gather native aerosol slabs for the sunlit solver batch."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
s='''#include <metal_stdlib>
using namespace metal;
#include "hexv842_sw_aerosol_gather_f32_bindings.metalh"
kernel void hexv842_sw_aerosol_gather_f32(constant Hexv842SwAerosolGatherF32Resources &r [[buffer(0)]],
 constant Hexv842SwAerosolGatherF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int t=int(tid),w=14*p.nl;if(t>=p.nd*w)return;int c=t/w,j=t%w,src=r.indices[c];
 r.tau_out[t]=r.tau_in[src*w+j];r.ssa_out[t]=r.ssa_in[src*w+j];r.asy_out[t]=r.asy_in[src*w+j];
}
'''
(root/'woof/metal/shaders/hexv842_sw_aerosol_gather_f32.metal').write_text(s)
res=[('indices','int32','read')]+[(n,'float','read') for n in ('tau_in','ssa_in','asy_in')]+[(n,'float','read_write') for n in ('tau_out','ssa_out','asy_out')]
k=dict(name='hexv842_sw_aerosol_gather_f32',module='hexv842_sw_aerosol_gather_f32',source='woof/metal/shaders/hexv842_sw_aerosol_gather_f32.metal',cuda_source='woof/hex/mpasphys/rrtmg_sw_v842.py',cuda_symbol='_sunlit_batches',resources=res,shapes={r[0]:[['any_1d']] for r in res},parameters=[(x,'int32') for x in ('nd','nl')],threads_per_group=128,layout_version=1,arithmetic_policies=['default'])
(root/'woof/metal/signature_fragments/hexv842_sw_aerosol_gather_f32.json').write_text(json.dumps({'kernels':[k]},indent=2)+'\n')
