#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexrrfs_mpas_monthly_surface_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexrrfs_mpas_monthly_surface_f32(
    constant HexrrfsMpasMonthlySurfaceF32Resources &r [[buffer(0)]],
    constant HexrrfsMpasMonthlySurfaceF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    int i=int(tid);
    if (i>=p.nsolve) return;
    float v=mpas_div(mpas_add(mpas_mul(r.monthly[p.m2*p.ncells+i],float(p.w2)),
        mpas_mul(r.monthly[p.m1*p.ncells+i],float(p.w1))),float(p.span));
    r.out[i]=p.mode==1 ? (r.land[i]==0 ? 0.08f : mpas_div(v,100.f)) : v;
    if (p.mode==2) {
        float mn=r.monthly[i], mx=mn;
        for (int k=1;k<12;k++) {
            float x=r.monthly[k*p.ncells+i];
            if (soft32_less(as_type<uint>(x),as_type<uint>(mn))) mn=x;
            if (soft32_less(as_type<uint>(mx),as_type<uint>(x))) mx=x;
        }
        r.lo[i]=mn; r.hi[i]=mx;
    }
}
