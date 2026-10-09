#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "zero_sumflux_bindings.metalh"
#include "accumulate_sumflux_bindings.metalh"
#include "finish_sumflux_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

// WRF acoustic.cu spells these operations as explicit div.rn.f32/add.rn.f32
// regardless of the surrounding compiler arithmetic policy. The shared
// integer implementation preserves that contract in both bundle variants.
static inline float sumflux_add_rn(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float sumflux_div_rn(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}
static inline bool sumflux_layout_valid(int nx, int ny, int nz,
                                        int nu, int nv, int nw, int nmax) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || nu <= 0 || nv <= 0 || nw <= 0)
        return false;
    const ulong ex_u = ulong(nz) * ulong(ny) * (ulong(nx) + 1ul);
    const ulong ex_v = ulong(nz) * (ulong(ny) + 1ul) * ulong(nx);
    const ulong ex_w = (ulong(nz) + 1ul) * ulong(ny) * ulong(nx);
    return ex_u == ulong(nu) && ex_v == ulong(nv) && ex_w == ulong(nw)
        && uint(nmax) == max(uint(nu), max(uint(nv), uint(nw)));
}

kernel void zero_sumflux(constant ZeroSumfluxResources &r [[buffer(0)]],
                         constant ZeroSumfluxParams &q [[buffer(1)]],
                         uint tid [[thread_position_in_grid]]) {
    if (!sumflux_layout_valid(q.nx, q.ny, q.nz, q.nu, q.nv, q.nw, q.nmax))
        return;
    if (tid >= uint(q.nmax)) return;
    if (tid < uint(q.nu)) r.ru_m[tid] = 0.0f;
    if (tid < uint(q.nv)) r.rv_m[tid] = 0.0f;
    if (tid < uint(q.nw)) r.ww_m[tid] = 0.0f;
}

kernel void accumulate_sumflux(
        constant AccumulateSumfluxResources &r [[buffer(0)]],
        constant AccumulateSumfluxParams &q [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
    if (!sumflux_layout_valid(q.nx, q.ny, q.nz, q.nu, q.nv, q.nw, q.nmax))
        return;
    if (tid >= uint(q.nmax)) return;
    if (tid < uint(q.nu))
        r.ru_m[tid] = sumflux_add_rn(r.ru_m[tid], r.ru[tid]);
    if (tid < uint(q.nv))
        r.rv_m[tid] = sumflux_add_rn(r.rv_m[tid], r.rv[tid]);
    if (tid < uint(q.nw))
        r.ww_m[tid] = sumflux_add_rn(r.ww_m[tid], r.ww[tid]);
}

kernel void finish_sumflux(
        constant FinishSumfluxResources &r [[buffer(0)]],
        constant FinishSumfluxParams &q [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
    if (!sumflux_layout_valid(q.nx, q.ny, q.nz, q.nu, q.nv, q.nw, q.nmax))
        return;
    if (tid >= uint(q.nmax)) return;
    if (tid < uint(q.nu)) {
        const float mean = sumflux_div_rn(r.ru_m[tid], q.nsub);
        r.ru_m[tid] = sumflux_add_rn(mean, r.ru[tid]);
    }
    if (tid < uint(q.nv)) {
        const float mean = sumflux_div_rn(r.rv_m[tid], q.nsub);
        r.rv_m[tid] = sumflux_add_rn(mean, r.rv[tid]);
    }
    if (tid < uint(q.nw)) {
        const float mean = sumflux_div_rn(r.ww_m[tid], q.nsub);
        r.ww_m[tid] = sumflux_add_rn(mean, r.ww[tid]);
    }
}
