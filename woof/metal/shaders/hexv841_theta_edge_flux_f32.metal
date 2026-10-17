// Hex v8.4.1 Metal transcription of CUDA kernel theta_edge_flux_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_theta_edge_flux_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_theta_edge_flux_f32(
    constant Hexv841ThetaEdgeFluxF32Resources &r [[buffer(0)]],
    constant Hexv841ThetaEdgeFluxF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int width = p.width;
    const int rk_step = p.rk_step;
    const float coefficient = p.coefficient;
    device const float *theta = r.theta;
    device const float *theta_saved = r.theta_saved;
    device const float *ru = r.ru;
    device const float *ru_saved = r.ru_saved;
    device const float *dv_edge = r.dv_edge;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *adv = r.adv;
    device const float *adv3 = r.adv3;
    device const int *n_adv = r.n_adv;
    device const int *adv_cells = r.adv_cells;
    device float *flux = r.flux;
    const int element = int(tid);
    if (element >= (nlev * nedges)) return;
    {

        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const int index = E2(k, edge, nedges);
        float edge_theta = 0.0f;
        const float sign = mpas_copysign(1.0f, ru[index]);
        for (int slot = 0; slot < n_adv[edge]; ++slot) {
            const int cell = adv_cells[ADV(edge, slot, width)];
            const float weight = mpas_add(adv[ADV(edge, slot, width)],
                mpas_mul(mpas_mul(coefficient, sign),
                    adv3[ADV(edge, slot, width)]));
            edge_theta = mpas_add(edge_theta, mpas_mul(
                weight, theta[C2(k, cell, ncells)]));
        }
        float value = mpas_mul(ru[index], edge_theta);
        if (rk_step > 1) {
            value = mpas_add(value, mpas_mul(mpas_mul(mpas_mul(
                dv_edge[edge], mpas_sub(ru_saved[index], ru[index])), 0.5f),
                mpas_add(theta_saved[C2(k, c0, ncells)],
                    theta_saved[C2(k, c1, ncells)])));
        }
        flux[index] = value;
    }
}
