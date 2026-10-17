// Hex v8.4.1 kernel acoustic_ru_v841, transcribed statement for statement from
// woof/hex/cuda_acoustic_v841.py.
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_acoustic_ru_v841_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ZBS(k,c,s,nc,me) ((((k)*(nc) + (c))*(me)) + (s))

kernel void hexv841_acoustic_ru_v841(
    constant Hexv841AcousticRuV841Resources &rsc [[buffer(0)]],
    constant Hexv841AcousticRuV841Params &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm.nlev;
    const int nedges = prm.nedges;
    const int ncells = prm.ncells;
    const int small_step = prm.small_step;
    const float dts = prm.dts;
    const float gravity = prm.gravity;
    const float rgas = prm.rgas;
    const float cp = prm.cp;

    const int edge = int(tid);
    if (edge >= nedges) return;
    const int c0 = rsc.cells_on_edge[2 * edge];
    const int c1 = rsc.cells_on_edge[2 * edge + 1];
    const float half_ = 0.5f;
    const float rcv = mpas_div(rgas, mpas_sub(cp, rgas));
    const float csquared = mpas_mul(cp, rcv);
    for (int k = 0; k < nlev; ++k) {
        const int index = E2(k, edge, nedges);
        if (small_step == 1) {
            const float value = mpas_mul(dts, rsc.tend_ru[index]);
            rsc.ru_p[index] = value;
            rsc.ru_avg[index] = value;
        } else {
            const float normal = mpas_mul(mpas_sub(
                rsc.rtheta_pp[C2(k, c1, ncells)],
                rsc.rtheta_pp[C2(k, c0, ncells)]), rsc.inv_dc_edge[edge]);
            const float normalized = mpas_div(normal, mpas_mul(half_, mpas_add(
                rsc.zz[C2(k, c1, ncells)], rsc.zz[C2(k, c0, ncells)])));
            float pgrad = mpas_mul(rsc.cqu[index], half_);
            pgrad = mpas_mul(pgrad, csquared);
            pgrad = mpas_mul(pgrad, mpas_add(rsc.exner[C2(k, c0, ncells)],
                rsc.exner[C2(k, c1, ncells)]));
            pgrad = mpas_mul(pgrad, normalized);
            const float terrain = mpas_mul(mpas_mul(mpas_mul(
                half_, rsc.zxu[index]), gravity), mpas_add(
                    rsc.rho_pp[C2(k, c0, ncells)],
                    rsc.rho_pp[C2(k, c1, ncells)]));
            pgrad = mpas_add(pgrad, terrain);
            rsc.ru_p[index] = mpas_add(rsc.ru_p[index], mpas_mul(
                dts, mpas_sub(rsc.tend_ru[index], pgrad)));
            rsc.ru_avg[index] = mpas_add(rsc.ru_avg[index], rsc.ru_p[index]);
        }
    }
}
