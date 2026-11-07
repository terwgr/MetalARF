#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "hexrrfs_ruc_diagnostics_bindings.metalh"
using namespace ntc;

// Preserve the retained RUC min/max operand ordering, including ties.
inline F diag_min(F a, F b) { return a < b ? a : b; }
inline F diag_max(F a, F b) { return a > b ? a : b; }

constant float ruc_diag_water[9] = {
    .611583699e03f, .444606896e02f, .143177157e01f, .264224321e-1f,
    .299291081e-3f, .203154182e-5f, .702620698e-8f, .379534310e-11f,
    -.321582393e-13f,
};
constant float ruc_diag_ice[9] = {
    .609868993e03f, .499320233e02f, .184672631e01f, .402737184e-1f,
    .565392987e-3f, .521693933e-5f, .307839583e-7f, .105785160e-9f,
    .161444444e-12f,
};

inline F diag_saturation(F pressure, F temperature) {
    constant float *coeff = tc_sub(temperature, F(273.15f)) <= F(0.f)
        ? ruc_diag_ice : ruc_diag_water;
    F x = diag_max(F(-80.f), tc_sub(temperature, F(273.16f)));
    F es = F(coeff[8]);
    for (int k = 7; k >= 0; --k)
        es = tc_add(F(coeff[k]), tc_mul(x, es));
    return tc_div(tc_mul(F(.622f), es), tc_sub(pressure, es));
}

inline F diag_log_factor(F delta, F scale, F dz1) {
    F fh = diag_min(diag_max(tc_sub(F(1.f), tc_div(delta, scale)), F(.01f)), F(1.f));
    F divisor = tc_add(F(.05f), fh);
    return tc_div(gfk_log(tc_div(tc_add(F(2.f), F(.05f)), divisor)),
                  gfk_log(tc_div(tc_add(dz1, F(.05f)), divisor)));
}

kernel void hexrrfs_ruc_diagnostics(constant HexrrfsRucDiagnosticsResources &r [[buffer(0)]],
                                    constant HexrrfsRucDiagnosticsParams &p [[buffer(1)]],
                                    uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.solve))
        return;
    F rho = F(r.rho3d[c]), psfc = F(r.psfc2d[c]);
    F air = F(r.t3d[c]), skin = F(r.tsk[c]), chs2 = F(r.chs2[c]);
    F exner_inv = gfk_pow(tc_div(F(1.e5f), psfc), F(p.rovcp));
    F theta2 = chs2 < F(1.e-5f)
        ? tc_mul(air, exner_inv)
        : tc_sub(tc_mul(skin, exner_inv), tc_div(F(r.hfx[c]), tc_mul(tc_mul(rho, F(p.cp)), chs2)));
    F t2 = tc_mul(theta2, gfk_pow(tc_mul(F(1.e-5f), psfc), F(p.rovcp)));

    F cqs = F(p.mynn ? r.cqs[c] : r.chs[c]);
    if (!p.mynn)
        r.cqs[c] = as_type<float>(cqs.bits);
    F qlev1 = diag_min(diag_saturation(F(r.p3d[c]), air), F(r.qv3d[c]));
    F qfx = F(r.qfx[c]), cqs2 = F(r.cqs2[c]), qsfc = F(r.qsfc[c]);
    F qsfcprox = tc_add(qlev1, tc_div(qfx, tc_mul(rho, cqs)));
    F qsfcmr = tc_div(qsfc, tc_sub(F(1.f), qsfc));
    F q2 = cqs2 < F(1.e-5f) ? qlev1 : tc_sub(qsfcprox, tc_div(qfx, tc_mul(rho, cqs2)));

    F dt = tc_sub(air, skin), dq = tc_sub(qlev1, qsfcmr);
    F dz1 = tc_mul(F(.5f), F(r.dz[c]));
    if (dt > F(0.f))
        t2 = tc_add(skin, tc_mul(diag_log_factor(dt, F(10.f), dz1), tc_sub(air, skin)));
    if (dq > F(0.f))
        q2 = tc_add(qsfcmr, tc_mul(diag_log_factor(dq, F(.003f), dz1), tc_sub(qlev1, qsfcmr)));

    t2 = diag_min(diag_max(skin, air), diag_max(diag_min(skin, air), t2));
    q2 = diag_min(diag_max(qsfcmr, qlev1), diag_max(diag_min(qsfcmr, qlev1), q2));
    q2 = diag_min(diag_saturation(psfc, t2), q2);
    r.t2[c] = as_type<float>(t2.bits);
    r.th2[c] = as_type<float>(tc_mul(t2, exner_inv).bits);
    r.q2[c] = as_type<float>(q2.bits);
}
