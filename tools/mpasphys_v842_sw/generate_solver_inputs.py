"""Generate the native wrapper's sunlit solver input preparation."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
s='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_sw_solver_inputs_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_sw_solver_inputs_f32(constant Hexv842SwSolverInputsF32Resources &r [[buffer(0)]],
 constant Hexv842SwSolverInputsF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int i=int(tid),n=p.nc;if(i>=n)return;
 float c=r.cosine[i];r.prmu0[i]=soft32_less_equal(as_type<uint>(c),as_type<uint>(1.e-10f))?1.e-10f:c;
 float adj=mpas_mul(1.f,mpas_div(r.controls[3],as_type<float>(1152059146u)));
 for(int b=0;b<14;b++){r.adjflux[i*14+b]=adj;r.albdir[i*14+b]=r.albedo[i];r.albdif[i*14+b]=r.albedo[i];}
}
'''
(root/'woof/metal/shaders/hexv842_sw_solver_inputs_f32.metal').write_text(s)
res=[(n,'float','read') for n in ('cosine','albedo','controls')]+[(n,'float','read_write') for n in ('prmu0','adjflux','albdir','albdif')]
k=dict(name='hexv842_sw_solver_inputs_f32',module='hexv842_sw_solver_inputs_f32',source='woof/metal/shaders/hexv842_sw_solver_inputs_f32.metal',cuda_source='woof/hex/mpasphys/rrtmg_sw_v842.py',cuda_symbol='_sunlit_batches',resources=res,shapes={x[0]:[['any_1d']] for x in res},parameters=[('nc','int32')],threads_per_group=128,layout_version=1,arithmetic_policies=['default'])
(root/'woof/metal/signature_fragments/hexv842_sw_solver_inputs_f32.json').write_text(json.dumps({'kernels':[k]},indent=2)+'\n')
