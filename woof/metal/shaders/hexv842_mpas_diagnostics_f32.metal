#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_scalar.metalh"
#include "hexv842_mpas_diagnostics_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float native_diagnostic_pow(float a,float b) {
    return as_type<float>(thompson_exner_powf_bits(as_type<uint>(a),as_type<uint>(b)));
}

kernel void hexv842_mpas_output_diagnostics_v842_f32(constant Hexv842MpasOutputDiagnosticsV842F32Resources &r [[buffer(0)]],constant Hexv842MpasOutputDiagnosticsV842F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.count) return;
    r.theta[i] = tc_div(r.theta_m[i],
        tc_add(1.0f, tc_mul(p.rvord, r.qv[i])));
    r.rho[i] = tc_mul(r.rho_zz[i], r.zz[i]);
    r.pressure[i] = tc_add(r.pressure_base[i], r.pressure_p[i]);
}

kernel void hexv842_mpas_cold_thermodynamics_v842_f32(constant Hexv842MpasColdThermodynamicsV842F32Resources &r [[buffer(0)]],constant Hexv842MpasColdThermodynamicsV842F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.count) return;
    const float coefficient = tc_div(p.rgas, p.p0);
    const float factor = tc_mul(r.zz[i], coefficient);
    r.exner[i] = native_diagnostic_pow(tc_mul(factor,
        tc_add(r.rtheta_p[i], r.rtheta_base[i])), p.rcv);
    r.exner_base[i] = native_diagnostic_pow(tc_mul(factor, r.rtheta_base[i]), p.rcv);
    r.pressure_p[i] = tc_mul(tc_mul(r.zz[i], p.rgas),
        tc_add(tc_mul(r.exner[i], r.rtheta_p[i]),
            tc_mul(r.rtheta_base[i], tc_sub(r.exner[i], r.exner_base[i]))));
    r.pressure_base[i] = tc_mul(tc_mul(tc_mul(r.zz[i], p.rgas),
        r.exner_base[i]), r.rtheta_base[i]);
}
