#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_monthly_surface_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_monthly_surface_f32(
    constant Hexv842MpasMonthlySurfaceF32Resources &r [[buffer(0)]],
    constant Hexv842MpasMonthlySurfaceF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int i=int(tid);
    if (i>=p.nsolve) return;
    r.veg[i]=mpas_div(mpas_add(mpas_mul(r.green[p.m2*p.ncells+i],float(p.w2)),
        mpas_mul(r.green[p.m1*p.ncells+i],float(p.w1))),float(p.span));
    float mn=r.green[i], mx=mn;
    for (int k=1;k<12;k++) { float v=r.green[k*p.ncells+i]; if(v<mn)mn=v; if(v>mx)mx=v; }
    r.lo[i]=mn; r.hi[i]=mx;
    if(p.use_alb) {
        float v=mpas_div(mpas_add(mpas_mul(r.alb[p.m2*p.ncells+i],float(p.w2)),
            mpas_mul(r.alb[p.m1*p.ncells+i],float(p.w1))),float(p.span));
        r.bg[i]=r.land[i]==0 ? 0.08f : mpas_div(v,100.f);
    }
}
