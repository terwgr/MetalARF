#include <metal_stdlib>
using namespace metal;
#include "refl10cm_kessler_cell_bindings.metalh"
#include "refl_temperature_product_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"

// Rain-only Smith (1975) REFL_10CM fallback used by the WOOF Kessler port.
// Keep every intermediate that is binary64 in refl10cm_kessler_cell as a
// software-binary64 value; the diagnosed Ze and final field are FP32.
static inline ulong refl64_mul(ulong a, ulong b) { return soft64_mul(a,b); }
static inline ulong refl64_div(ulong a, ulong b) { return soft64_div(a,b); }
static inline ulong refl64_pow_quarter(ulong x) {
    const ulong quarter=0x3fd0000000000000ul;
    return soft64_plm_exp(refl64_mul(soft64_plm_log(x),quarter));
}
static inline ulong refl64_pow_seven(ulong x) {
    ulong x2=refl64_mul(x,x);
    ulong x4=refl64_mul(x2,x2);
    return refl64_mul(refl64_mul(x4,x2),x);
}
static inline float refl64_to_f32(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}

kernel void refl_temperature_product(
    constant ReflTemperatureProductResources& r [[buffer(0)]],
    constant ReflTemperatureProductParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    uint n=uint(p.nz*p.ny*p.nx);
    if (idx>=n) return;
    r.temperature[idx]=r.theta[idx]*r.pii[idx];
}

kernel void refl10cm_kessler_cell(
    constant Refl10CmKesslerCellResources& r [[buffer(0)]],
    constant Refl10CmKesslerCellParams& p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    uint n=uint(p.nz*p.ny*p.nx);
    if (idx>=n) return;
    const ulong r0622=refl64_div(soft64_from_i32(311),soft64_from_i32(500));
    const ulong r_d=soft64_from_i32(287);
    const ulong qv_min=0x3ddb7cdfd9d7bdbbul; // 1e-10
    const ulong pi=0x400921fb54442d18ul;
    const ulong thousand=soft64_from_i32(1000);
    const ulong n0=soft64_from_i32(8000000);
    const ulong gamma7=soft64_from_i32(720);
    const ulong e18=0x43abc16d674ec800ul; // 1e18

    ulong qv=soft64_from_f32_bits(as_type<uint>(r.qv[idx]));
    if (soft64_is_nan(qv) || soft64_sign(qv) || soft64_less(qv,qv_min))
        qv=qv_min;
    ulong pressure=soft64_from_f32_bits(as_type<uint>(r.p[idx]));
    ulong temperature=soft64_from_f32_bits(as_type<uint>(r.t[idx]));
    ulong rho=refl64_div(refl64_mul(r0622,pressure),
        refl64_mul(refl64_mul(r_d,temperature),soft64_add(qv,r0622)));

    float ze=1.0e-22f;
    float qr=r.qr[idx];
    if (qr>1.0e-9f) {
        ulong rain=soft64_from_f32_bits(as_type<uint>(qr));
        ulong numerator=refl64_mul(refl64_mul(pi,thousand),n0);
        ulong lambda=refl64_pow_quarter(
            refl64_div(numerator,refl64_mul(rho,rain)));
        ulong ze64=refl64_div(refl64_mul(gamma7,n0),
                              refl64_pow_seven(lambda));
        ze=refl64_to_f32(ze64);
    }
    ulong ze64=soft64_from_f32_bits(as_type<uint>(ze));
    ulong dbz=refl64_mul(soft64_from_i32(10),
        refl64_div(soft64_plm_log(refl64_mul(ze64,e18)),
                   0x40026bb1bbb55516ul)); // ln(10)
    float result=refl64_to_f32(dbz);
    r.refl[idx]=isnan(result) ? -35.0f : max(-35.0f,result);
}
