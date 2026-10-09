#include <metal_stdlib>
#include "noah_sfcdiags_bindings.metalh"

using namespace metal;

// Keep each FP32 operator boundary from the CUDA SFCDIAGS transcription.
#pragma clang fp contract(off)

kernel void noah_sfcdiags(
    constant NoahSfcdiagsResources &r [[buffer(0)]],
    constant NoahSfcdiagsParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const int c = int(flat);
    if (c >= p.ncol) return;

    const float ps = r.psfc_a[c];
    const float ts = r.tsk_a[c];
    const float rho_denominator = p.rd * ts;
    const float rho = ps / rho_denominator;
    const float cq = r.cqs2_a[c];
    const float ch = r.chs2_a[c];
    const bool q_active = cq >= p.active_floor;
    const bool t_active = ch >= p.active_floor;
    const float safe_cqs2 = q_active ? cq : 1.0f;
    const float safe_chs2 = t_active ? ch : 1.0f;

    // CuPy evaluates both where branches before selection. Keep the flux
    // inversion materialized even for inactive transfer coefficients.
    const float qs = r.qsfc_a[c];
    const float q_denominator = rho * safe_cqs2;
    const float q_correction = r.qfx_a[c] / q_denominator;
    const float inverted = qs - q_correction;
    const float diagnosed = q_active ? inverted : qs;
    r.q2_a[c] = diagnosed > 0.0f ? diagnosed : r.qv1_a[c];

    const float t_rho_cp = rho * p.cp_air;
    const float t_denominator = t_rho_cp * safe_chs2;
    const float t_correction = r.hfx_a[c] / t_denominator;
    const float heated = ts - t_correction;
    const float temperature = t_active ? heated : ts;
    r.t2_a[c] = temperature;

    const float exner_base = p.p0 / ps;
    const float exner = pow(exner_base, p.rcp);
    r.th2_a[c] = temperature * exner;
}
