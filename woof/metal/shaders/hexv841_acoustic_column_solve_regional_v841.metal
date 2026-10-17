// Hex v8.4.1 Metal transcription of CUDA kernel acoustic_column_solve_regional_v841 (woof/hex/cuda_regional_v841.py:700).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_acoustic_column_solve_regional_v841_bindings.metalh"
#pragma clang fp contract(off)

#define RC2(k,c,nc) ((k)*(nc) + (c))
#define RE2(k,e,ne) ((k)*(ne) + (e))
#define RQ3(t,k,c,nlev,nc) (((t)*(nlev) + (k))*(nc) + (c))
#define RQE3(t,k,e,nlev,ne) (((t)*(nlev) + (k))*(ne) + (e))
#define RQI3(t,k,c,nlev,nc) (((t)*((nlev)+1) + (k))*(nc) + (c))
#define RCES(c,s,me) ((c)*(me) + (s))
#define RADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
#define REGIONAL_N_RELAX_ZONE 5
#define REGIONAL_N_SPEC_ZONE 2
#define MPAS_FTZ_FALLBACK_ENABLED 1


kernel void hexv841_acoustic_column_solve_regional_v841(
    constant Hexv841AcousticColumnSolveRegionalV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841AcousticColumnSolveRegionalV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm__.nlev;
    const int ncells = prm__.ncells;
    const float dts = prm__.dts;
    device const float *zz = rsc__.zz;
    device const float *rho_zz = rsc__.rho_zz;
    device const float *fzm = rsc__.fzm;
    device const float *fzp = rsc__.fzp;
    device const float *rdzw = rsc__.rdzw;
    device const float *dss = rsc__.dss;
    device const float *w = rsc__.w;
    device const float *rw = rsc__.rw;
    device const float *rw_save = rsc__.rw_save;
    device const float *tend_rw = rsc__.tend_rw;
    device const float *tend_rho = rsc__.tend_rho;
    device const float *tend_rt = rsc__.tend_rt;
    device const float *rs = rsc__.rs;
    device const float *ts = rsc__.ts;
    device const float *spec_zone_mask_cell = rsc__.spec_zone_mask_cell;
    device const float *cofwr = rsc__.cofwr;
    device const float *cofwz = rsc__.cofwz;
    device const float *coftz = rsc__.coftz;
    device const float *cofwt = rsc__.cofwt;
    device const float *cofrz = rsc__.cofrz;
    device const float *a_tri = rsc__.a_tri;
    device const float *alpha_tri = rsc__.alpha_tri;
    device const float *gamma_tri = rsc__.gamma_tri;
    device const float *etp = rsc__.etp;
    device const float *etm = rsc__.etm;
    device const float *ewp = rsc__.ewp;
    device const float *ewm = rsc__.ewm;
    device float *rw_p = rsc__.rw_p;
    device float *rho_pp = rsc__.rho_pp;
    device float *rtheta_pp = rsc__.rtheta_pp;
    device float *ww_avg = rsc__.ww_avg;

    const int cell = int(tid);
    if (cell >= ncells) return;
    if (spec_zone_mask_cell[cell] != 0.0f) {
        for (int k = 0; k < nlev; ++k) {
            const int i = RC2(k, cell, ncells);
            rho_pp[i] = mpas_add(rho_pp[i], mpas_mul(dts, tend_rho[i]));
            rtheta_pp[i] = mpas_add(rtheta_pp[i], mpas_mul(dts, tend_rt[i]));
            rw_p[i] = mpas_add(rw_p[i], mpas_mul(dts, tend_rw[i]));
            ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(ewp[k], rw_p[i]));
        }
        return;
    }
    const float dts2 = mpas_mul(dts, dts);
    for (int k = 1; k < nlev; ++k) {
        const int i = RC2(k, cell, ncells);
        const int im = RC2(k - 1, cell, ncells);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(ewm[k], rw_p[i]));

        const float theta_implicit = mpas_sub(
            mpas_mul(mpas_mul(etp[k], zz[i]), ts[i]),
            mpas_mul(mpas_mul(etp[k - 1], zz[im]), ts[im]));
        const float theta_explicit = mpas_sub(
            mpas_mul(mpas_mul(etm[k], zz[i]), rtheta_pp[i]),
            mpas_mul(mpas_mul(etm[k - 1], zz[im]), rtheta_pp[im]));
        const float density_implicit = mpas_add(
            mpas_mul(etp[k], rs[i]), mpas_mul(etp[k - 1], rs[im]));
        const float density_explicit = mpas_add(
            mpas_mul(etm[k], rho_pp[i]),
            mpas_mul(etm[k - 1], rho_pp[im]));
        const float heat_upper = mpas_add(
            mpas_mul(etp[k], ts[i]), mpas_mul(etm[k], rtheta_pp[i]));
        const float heat_lower = mpas_add(
            mpas_mul(etp[k - 1], ts[im]),
            mpas_mul(etm[k - 1], rtheta_pp[im]));

        float inner = mpas_sub(tend_rw[i], mpas_mul(cofwz[i],
            mpas_add(theta_implicit, theta_explicit)));
        inner = mpas_sub(inner, mpas_mul(cofwr[i],
            mpas_add(density_implicit, density_explicit)));
        inner = mpas_add(inner, mpas_mul(cofwt[i], heat_upper));
        inner = mpas_add(inner, mpas_mul(cofwt[im], heat_lower));
        rw_p[i] = mpas_add(rw_p[i], mpas_mul(dts, inner));
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = RC2(k, cell, ncells);
        const int im = RC2(k - 1, cell, ncells);
        rw_p[i] = mpas_mul(mpas_sub(rw_p[i], mpas_mul(mpas_mul(
            dts2, a_tri[i]), rw_p[im])), alpha_tri[i]);
    }
    for (int k = nlev - 1; k >= 0; --k) {
        const int i = RC2(k, cell, ncells);
        rw_p[i] = mpas_sub(rw_p[i], mpas_mul(
            gamma_tri[i], rw_p[RC2(k + 1, cell, ncells)]));
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = RC2(k, cell, ncells);
        const int im = RC2(k - 1, cell, ncells);
        const float delta_saved = mpas_sub(rw_save[i], rw[i]);
        const float density_interface = mpas_mul(
            mpas_add(mpas_mul(fzm[k], zz[i]), mpas_mul(fzp[k], zz[im])),
            mpas_add(mpas_mul(fzm[k], rho_zz[i]),
                mpas_mul(fzp[k], rho_zz[im])));
        float value = mpas_add(rw_p[i], delta_saved);
        value = mpas_sub(value, mpas_mul(mpas_mul(mpas_mul(
            dts, dss[i]), density_interface), w[i]));
        value = mpas_div(value, mpas_add(1.0f, mpas_mul(dts, dss[i])));
        rw_p[i] = mpas_sub(value, delta_saved);
        ww_avg[i] = mpas_add(ww_avg[i], mpas_mul(ewp[k], rw_p[i]));
    }
    for (int k = 0; k < nlev; ++k) {
        const int i = RC2(k, cell, ncells);
        rho_pp[i] = mpas_sub(rs[i], mpas_mul(mpas_mul(dts, cofrz[k]),
            mpas_sub(mpas_mul(ewp[k + 1], rw_p[RC2(k + 1, cell, ncells)]),
                mpas_mul(ewp[k], rw_p[i]))));
        rtheta_pp[i] = mpas_sub(ts[i], mpas_mul(mpas_mul(dts, rdzw[k]),
            mpas_sub(mpas_mul(mpas_mul(ewp[k + 1],
                    coftz[RC2(k + 1, cell, ncells)]),
                rw_p[RC2(k + 1, cell, ncells)]),
                mpas_mul(mpas_mul(ewp[k], coftz[i]), rw_p[i]))));
    }
}
