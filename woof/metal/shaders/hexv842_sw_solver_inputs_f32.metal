#include <metal_stdlib>
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
