// Hex v8.4.1 kernel acoustic_column_solve_lts, transcribed statement for statement from
// woof/hex/cuda_acoustic_lts.py (derived from acoustic_column_solve_v841 with the index-list gather).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_acoustic_column_solve_lts_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ZBS(k,c,s,nc,me) ((((k)*(nc) + (c))*(me)) + (s))

kernel void hexv841_acoustic_column_solve_lts(
    constant Hexv841AcousticColumnSolveLtsResources &rsc [[buffer(0)]],
    constant Hexv841AcousticColumnSolveLtsParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm.nlev;
    const int ncells = prm.ncells;
    const float dts = prm.dts;

    const int lts_slot = int(tid);
    if (lts_slot >= prm.lts_n_active) return;
    const int cell = rsc.lts_active[lts_slot];
    const float dts2 = mpas_mul(dts, dts);
    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        rsc.ww_avg[i] = mpas_add(rsc.ww_avg[i], mpas_mul(rsc.ewm[k], rsc.rw_p[i]));

        const float theta_implicit = mpas_sub(
            mpas_mul(mpas_mul(rsc.etp[k], rsc.zz[i]), rsc.ts[i]),
            mpas_mul(mpas_mul(rsc.etp[k - 1], rsc.zz[im]), rsc.ts[im]));
        const float theta_explicit = mpas_sub(
            mpas_mul(mpas_mul(rsc.etm[k], rsc.zz[i]), rsc.rtheta_pp[i]),
            mpas_mul(mpas_mul(rsc.etm[k - 1], rsc.zz[im]), rsc.rtheta_pp[im]));
        const float density_implicit = mpas_add(
            mpas_mul(rsc.etp[k], rsc.rs[i]), mpas_mul(rsc.etp[k - 1], rsc.rs[im]));
        const float density_explicit = mpas_add(
            mpas_mul(rsc.etm[k], rsc.rho_pp[i]),
            mpas_mul(rsc.etm[k - 1], rsc.rho_pp[im]));
        const float heat_upper = mpas_add(
            mpas_mul(rsc.etp[k], rsc.ts[i]), mpas_mul(rsc.etm[k], rsc.rtheta_pp[i]));
        const float heat_lower = mpas_add(
            mpas_mul(rsc.etp[k - 1], rsc.ts[im]),
            mpas_mul(rsc.etm[k - 1], rsc.rtheta_pp[im]));

        float inner = mpas_sub(rsc.tend_rw[i], mpas_mul(rsc.cofwz[i],
            mpas_add(theta_implicit, theta_explicit)));
        inner = mpas_sub(inner, mpas_mul(rsc.cofwr[i],
            mpas_add(density_implicit, density_explicit)));
        inner = mpas_add(inner, mpas_mul(rsc.cofwt[i], heat_upper));
        inner = mpas_add(inner, mpas_mul(rsc.cofwt[im], heat_lower));
        rsc.rw_p[i] = mpas_add(rsc.rw_p[i], mpas_mul(dts, inner));
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        rsc.rw_p[i] = mpas_mul(mpas_sub(rsc.rw_p[i], mpas_mul(mpas_mul(
            dts2, rsc.a_tri[i]), rsc.rw_p[im])), rsc.alpha_tri[i]);
    }
    for (int k = nlev - 1; k >= 0; --k) {
        const int i = C2(k, cell, ncells);
        rsc.rw_p[i] = mpas_sub(rsc.rw_p[i], mpas_mul(
            rsc.gamma_tri[i], rsc.rw_p[C2(k + 1, cell, ncells)]));
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float delta_saved = mpas_sub(rsc.rw_save[i], rsc.rw[i]);
        const float density_interface = mpas_mul(
            mpas_add(mpas_mul(rsc.fzm[k], rsc.zz[i]), mpas_mul(rsc.fzp[k], rsc.zz[im])),
            mpas_add(mpas_mul(rsc.fzm[k], rsc.rho_zz[i]),
                mpas_mul(rsc.fzp[k], rsc.rho_zz[im])));
        float value = mpas_add(rsc.rw_p[i], delta_saved);
        value = mpas_sub(value, mpas_mul(mpas_mul(mpas_mul(
            dts, rsc.dss[i]), density_interface), rsc.w[i]));
        value = mpas_div(value, mpas_add(1.0f, mpas_mul(dts, rsc.dss[i])));
        rsc.rw_p[i] = mpas_sub(value, delta_saved);
        rsc.ww_avg[i] = mpas_add(rsc.ww_avg[i], mpas_mul(rsc.ewp[k], rsc.rw_p[i]));
    }
    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        rsc.rho_pp[i] = mpas_sub(rsc.rs[i], mpas_mul(mpas_mul(dts, rsc.cofrz[k]),
            mpas_sub(mpas_mul(rsc.ewp[k + 1], rsc.rw_p[C2(k + 1, cell, ncells)]),
                mpas_mul(rsc.ewp[k], rsc.rw_p[i]))));
        rsc.rtheta_pp[i] = mpas_sub(rsc.ts[i], mpas_mul(mpas_mul(dts, rsc.rdzw[k]),
            mpas_sub(mpas_mul(mpas_mul(rsc.ewp[k + 1],
                    rsc.coftz[C2(k + 1, cell, ncells)]),
                rsc.rw_p[C2(k + 1, cell, ncells)]),
                mpas_mul(mpas_mul(rsc.ewp[k], rsc.coftz[i]), rsc.rw_p[i]))));
    }
}
