// Hex v8.4.1 kernel acoustic_coefficients_lts, transcribed statement for statement from
// woof/hex/cuda_acoustic_lts.py (derived from acoustic_coefficients_v841 with the index-list gather).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_acoustic_coefficients_lts_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ZBS(k,c,s,nc,me) ((((k)*(nc) + (c))*(me)) + (s))

kernel void hexv841_acoustic_coefficients_lts(
    constant Hexv841AcousticCoefficientsLtsResources &rsc [[buffer(0)]],
    constant Hexv841AcousticCoefficientsLtsParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm.nlev;
    const int ncells = prm.ncells;
    const float dts = prm.dts;
    const float gravity = prm.gravity;
    const float rgas = prm.rgas;
    const float cp = prm.cp;

    const int lts_slot = int(tid);
    if (lts_slot >= prm.lts_n_active) return;
    const int cell = rsc.lts_active[lts_slot];
    const float one = 1.0f;
    const float half_ = 0.5f;
    const float rcv = mpas_div(rgas, mpas_sub(cp, rgas));
    const float csquared = mpas_mul(cp, rcv);
    const float dts2 = mpas_mul(dts, dts);

    for (int k = 0; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        rsc.cofwr[i] = 0.0f;
        rsc.cofwz[i] = 0.0f;
        rsc.coftz[i] = 0.0f;
        float value = mpas_mul(half_, rcv);
        value = mpas_mul(value, rsc.zz[i]);
        value = mpas_mul(value, gravity);
        value = mpas_mul(value, rsc.rho_base[i]);
        value = mpas_div(value, mpas_add(one, rsc.qtot[i]));
        value = mpas_mul(value, rsc.exner[i]);
        value = mpas_div(value, mpas_mul(
            mpas_add(rsc.rtheta_base[i], rsc.rtheta_p[i]), rsc.exner_base[i]));
        rsc.cofwt[i] = value;
        rsc.a_tri[i] = 0.0f;
        rsc.b_tri[i] = 0.0f;
        rsc.c_tri[i] = 0.0f;
        rsc.alpha_tri[i] = 0.0f;
        rsc.gamma_tri[i] = 0.0f;
    }
    rsc.coftz[C2(nlev, cell, ncells)] = 0.0f;
    if (nlev > 0) rsc.b_tri[C2(0, cell, ncells)] = one;

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float interp_zz = mpas_add(mpas_mul(rsc.fzm[k], rsc.zz[i]),
            mpas_mul(rsc.fzp[k], rsc.zz[im]));
        rsc.cofwr[i] = mpas_mul(mpas_mul(half_, gravity), interp_zz);
        float zcoef = mpas_mul(csquared, interp_zz);
        zcoef = mpas_mul(zcoef, rsc.rdzu[k]);
        zcoef = mpas_mul(zcoef, rsc.cqw[i]);
        zcoef = mpas_mul(zcoef, mpas_add(mpas_mul(rsc.fzm[k], rsc.exner[i]),
            mpas_mul(rsc.fzp[k], rsc.exner[im])));
        rsc.cofwz[i] = zcoef;
        rsc.coftz[i] = mpas_add(mpas_mul(rsc.fzm[k], rsc.theta[i]),
            mpas_mul(rsc.fzp[k], rsc.theta[im]));
    }

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const int ip = C2(k + 1, cell, ncells);

        float avalue = mpas_sub(0.0f, mpas_mul(mpas_mul(mpas_mul(
            rsc.cofwz[i], rsc.coftz[im]), rsc.rdzw[k - 1]), rsc.zz[im]));
        avalue = mpas_add(avalue, mpas_mul(rsc.cofwr[i], rsc.cofrz[k - 1]));
        avalue = mpas_sub(avalue, mpas_mul(mpas_mul(
            rsc.cofwt[im], rsc.coftz[im]), rsc.rdzw[k - 1]));
        avalue = mpas_mul(avalue, rsc.etp[k - 1]);
        rsc.a_tri[i] = mpas_mul(avalue, rsc.ewp[k - 1]);

        const float upper_metric = mpas_mul(mpas_mul(
            rsc.etp[k], rsc.rdzw[k]), rsc.zz[i]);
        const float lower_metric = mpas_mul(mpas_mul(
            rsc.etp[k - 1], rsc.rdzw[k - 1]), rsc.zz[im]);
        float bvalue = mpas_mul(mpas_mul(rsc.cofwz[i], rsc.coftz[i]),
            mpas_add(upper_metric, lower_metric));
        const float upper_heat = mpas_mul(mpas_mul(
            rsc.etp[k], rsc.cofwt[i]), rsc.rdzw[k]);
        const float lower_heat = mpas_mul(mpas_mul(
            rsc.etp[k - 1], rsc.cofwt[im]), rsc.rdzw[k - 1]);
        bvalue = mpas_sub(bvalue, mpas_mul(rsc.coftz[i],
            mpas_sub(upper_heat, lower_heat)));
        const float upper_mass = mpas_mul(rsc.etp[k], rsc.cofrz[k]);
        const float lower_mass = mpas_mul(rsc.etp[k - 1], rsc.cofrz[k - 1]);
        bvalue = mpas_add(bvalue, mpas_mul(rsc.cofwr[i],
            mpas_sub(upper_mass, lower_mass)));
        rsc.b_tri[i] = mpas_mul(bvalue, rsc.ewp[k]);

        float cvalue = mpas_sub(0.0f, mpas_mul(mpas_mul(mpas_mul(
            rsc.cofwz[i], rsc.coftz[ip]), rsc.rdzw[k]), rsc.zz[i]));
        cvalue = mpas_sub(cvalue, mpas_mul(rsc.cofwr[i], rsc.cofrz[k]));
        cvalue = mpas_add(cvalue, mpas_mul(mpas_mul(
            rsc.cofwt[i], rsc.coftz[ip]), rsc.rdzw[k]));
        cvalue = mpas_mul(cvalue, rsc.etp[k]);
        rsc.c_tri[i] = mpas_mul(cvalue, rsc.ewp[k + 1]);
    }
    if (nlev > 0) rsc.c_tri[C2(nlev - 1, cell, ncells)] = 0.0f;

    for (int k = 1; k < nlev; ++k) {
        const int i = C2(k, cell, ncells);
        const int im = C2(k - 1, cell, ncells);
        const float denominator = mpas_add(one, mpas_mul(dts2,
            mpas_sub(rsc.b_tri[i], mpas_mul(rsc.a_tri[i], rsc.gamma_tri[im]))));
        if (!(isfinite(denominator) && denominator != 0.0f))
            atomic_exchange_explicit(reinterpret_cast<device atomic_int *>(rsc.singular), 1, memory_order_relaxed);
        rsc.alpha_tri[i] = mpas_div(one, denominator);
        rsc.gamma_tri[i] = mpas_mul(mpas_mul(dts2, rsc.c_tri[i]), rsc.alpha_tri[i]);
    }
}
