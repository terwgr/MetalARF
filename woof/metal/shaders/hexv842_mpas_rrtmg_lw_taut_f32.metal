#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_gfk_exp.metalh"
#pragma clang fp contract(off)
#include "hexv842_mpas_rrtmg_lw_taut_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_taut_f32(constant Hexv842MpasRrtmgLwTautF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwTautF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
ulong total=ulong(p.ncol)*p.nl*140;
if(ulong(tid)>=total)return;
uint ig=tid%140; ulong collay=ulong(tid)/140;
r.taut[tid]=mpas_add(r.gas_tau[tid],r.taua[collay*16+(r.ngb[ig]-1)]);
}
