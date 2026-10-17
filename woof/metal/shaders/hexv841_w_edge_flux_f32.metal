// Hex v8.4.1 Metal transcription of CUDA kernel w_edge_flux_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_w_edge_flux_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_w_edge_flux_f32(
    constant Hexv841WEdgeFluxF32Resources &r [[buffer(0)]],
    constant Hexv841WEdgeFluxF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int width = p.width;
    const float coefficient = p.coefficient;
    device const float *w = r.w;
    device const float *ru = r.ru;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device const float *adv = r.adv;
    device const float *adv3 = r.adv3;
    device const int *n_adv = r.n_adv;
    device const int *adv_cells = r.adv_cells;
    device float *flux = r.flux;
    const int element = int(tid);
    if (element >= ((nlev + 1) * nedges)) return;
    {

        const int edge = element % nedges;
        const int k = element / nedges;
        const int index = E2(k, edge, nedges);
        float value = 0.0f;
        if (k > 0 && k < nlev) {
            const float ru_interface = mpas_add(
                mpas_mul(fzm[k], ru[E2(k, edge, nedges)]),
                mpas_mul(fzp[k], ru[E2(k - 1, edge, nedges)]));
            const float sign = mpas_copysign(1.0f, ru_interface);
            float edge_w = 0.0f;
            for (int slot = 0; slot < n_adv[edge]; ++slot) {
                const int cell = adv_cells[ADV(edge, slot, width)];
                edge_w = mpas_add(edge_w, mpas_mul(mpas_add(
                    adv[ADV(edge, slot, width)], mpas_mul(
                        mpas_mul(coefficient, sign),
                        adv3[ADV(edge, slot, width)])),
                    w[C2(k, cell, ncells)]));
            }
            value = mpas_mul(ru_interface, edge_w);
        }
        flux[index] = value;
    }
}
