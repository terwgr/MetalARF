#include <metal_stdlib>
#include "ucm_overrides_noah_bindings.metalh"
#include "ucm_overrides_noahmp_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_pow.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ucm_fadd(float a, float b)
{ return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ucm_fsub(float a, float b)
{ return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ucm_fmul(float a, float b)
{ return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ucm_fdiv(float a, float b)
{ return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }

kernel void ucm_overrides_noah(
    constant UcmOverridesNoahResources &r [[buffer(0)]],
    constant UcmOverridesNoahParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    const uint count=uint(p.ny)*uint(p.nx);
    if (idx>=count || r.urban[idx]==0) return;

    // Exact ucm_overrides source assignments for noahmp == 0.
    r.field_u10[idx]=r.state_u10[idx];
    r.field_v10[idx]=r.state_v10[idx];
    r.field_psim[idx]=r.state_psim[idx];
    r.field_psih[idx]=r.state_psih[idx];
    r.field_gz1oz0[idx]=r.state_gz1oz0[idx];
    r.field_akhs[idx]=r.field_chs[idx];
    r.field_akms[idx]=r.state_akms[idx];
}

kernel void ucm_overrides_noahmp(
    constant UcmOverridesNoahmpResources &r [[buffer(0)]],
    constant UcmOverridesNoahmpParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    const uint count = uint(p.ny) * uint(p.nx);
    if (idx >= count || r.urban[idx] == 0) return;

    const float f = r.frc[idx];
    const float omf = ucm_fsub(1.0f, f);
    const float fv = r.fvegxy[idx];
    const float omfv = ucm_fsub(1.0f, fv);
    const float qveg = ucm_fadd(ucm_fmul(fv, r.q2mvxy[idx]),
                                 ucm_fmul(omfv, r.q2mbxy[idx]));
    r.field_q2[idx] = ucm_fadd(ucm_fmul(qveg, omf),
                                ucm_fmul(r.state_q2[idx], f));
    const float tveg = ucm_fadd(ucm_fmul(fv, r.t2mvxy[idx]),
                                 ucm_fmul(omfv, r.t2mbxy[idx]));
    const float t2 = ucm_fadd(ucm_fmul(tveg, omf),
                               ucm_fmul(r.state_th2[idx], f));
    r.field_t2[idx] = t2;
    const float rcp = as_type<float>(0x3e924925u);
    const uint pressure_ratio = as_type<uint>(ucm_fdiv(1.0e5f, r.field_psfc[idx]));
    const uint power = soft_gfk_pow_bits(pressure_ratio, as_type<uint>(rcp));
    r.field_th2[idx] = ucm_fmul(t2, as_type<float>(power));

    r.field_u10[idx] = r.state_u10[idx];
    r.field_v10[idx] = r.state_v10[idx];
    r.field_psim[idx] = r.state_psim[idx];
    r.field_psih[idx] = r.state_psih[idx];
    r.field_gz1oz0[idx] = r.state_gz1oz0[idx];
    r.field_akhs[idx] = r.field_chs[idx];
    r.field_akms[idx] = r.state_akms[idx];
}
