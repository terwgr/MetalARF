// Hex v8.4.1 Metal transcription of CUDA kernel acoustic_ru_regional_v841 (woof/hex/cuda_regional_v841.py:582).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_acoustic_ru_regional_v841_bindings.metalh"
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


kernel void hexv841_acoustic_ru_regional_v841(
    constant Hexv841AcousticRuRegionalV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841AcousticRuRegionalV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm__.nlev;
    const int nedges = prm__.nedges;
    const int ncells = prm__.ncells;
    const int small_step = prm__.small_step;
    const float dts = prm__.dts;
    const float gravity = prm__.gravity;
    const float rgas = prm__.rgas;
    const float cp = prm__.cp;
    device const int *cells_on_edge = rsc__.cells_on_edge;
    device const float *inv_dc_edge = rsc__.inv_dc_edge;
    device const float *spec_zone_mask_edge = rsc__.spec_zone_mask_edge;
    device const float *zz = rsc__.zz;
    device const float *exner = rsc__.exner;
    device const float *cqu = rsc__.cqu;
    device const float *zxu = rsc__.zxu;
    device const float *tend_ru = rsc__.tend_ru;
    device const float *rho_pp = rsc__.rho_pp;
    device const float *rtheta_pp = rsc__.rtheta_pp;
    device float *ru_p = rsc__.ru_p;
    device float *ru_avg = rsc__.ru_avg;
    device int *invalid = rsc__.invalid;

    const int element = int(tid);
    if (element >= (nlev * nedges)) return;
    {
        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const float half_ = 0.5f;
        const float one = 1.0f;
        const float rcv = mpas_div(rgas, mpas_sub(cp, rgas));
        const float csquared = mpas_mul(cp, rcv);
        const float spec = spec_zone_mask_edge[edge];
        const int index = RE2(k, edge, nedges);
        if (small_step == 1) {
            const float value = mpas_mul(dts, tend_ru[index]);
            ru_p[index] = value;
            ru_avg[index] = value;
            return;
        }
        if (c0 < 0 || c1 < 0) {
            if (spec != one) atomicExch(invalid, 1);
            ru_p[index] = mpas_add(ru_p[index],
                mpas_mul(dts, tend_ru[index]));
            ru_avg[index] = mpas_add(ru_avg[index], ru_p[index]);
            return;
        }
        const float normal = mpas_mul(mpas_sub(
            rtheta_pp[RC2(k, c1, ncells)],
            rtheta_pp[RC2(k, c0, ncells)]), inv_dc_edge[edge]);
        const float normalized = mpas_div(normal, mpas_mul(half_, mpas_add(
            zz[RC2(k, c1, ncells)], zz[RC2(k, c0, ncells)])));
        float pgrad = mpas_mul(cqu[index], half_);
        pgrad = mpas_mul(pgrad, csquared);
        pgrad = mpas_mul(pgrad, mpas_add(exner[RC2(k, c0, ncells)],
            exner[RC2(k, c1, ncells)]));
        pgrad = mpas_mul(pgrad, normalized);
        const float terrain = mpas_mul(mpas_mul(mpas_mul(
            half_, zxu[index]), gravity), mpas_add(
                rho_pp[RC2(k, c0, ncells)],
                rho_pp[RC2(k, c1, ncells)]));
        pgrad = mpas_add(pgrad, terrain);
        ru_p[index] = mpas_add(ru_p[index], mpas_mul(dts, mpas_sub(
            tend_ru[index], mpas_mul(mpas_sub(one, spec), pgrad))));
        ru_avg[index] = mpas_add(ru_avg[index], ru_p[index]);
    }
}
