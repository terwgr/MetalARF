// Hex v8.4.1 Metal transcription of CUDA kernel vertical_u_flux_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_vertical_u_flux_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_vertical_u_flux_f32(
    constant Hexv841VerticalUFluxF32Resources &r [[buffer(0)]],
    constant Hexv841VerticalUFluxF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    device const float *u = r.u;
    device const float *rw = r.rw;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device float *flux = r.flux;
    const int element = int(tid);
    if (element >= ((nlev + 1) * nedges)) return;
    {

        const int edge = element % nedges;
        const int k = element / nedges;
        const int c0 = cells_on_edge[2 * edge];
        const int c1 = cells_on_edge[2 * edge + 1];
        const int index = E2(k, edge, nedges);
        float value = 0.0f;
        if (k > 0 && k < nlev) {
            const float velocity = mpas_mul(0.5f, mpas_add(
                rw[C2(k, c0, ncells)], rw[C2(k, c1, ncells)]));
            if (k == 1 || k == nlev - 1) {
                value = mpas_mul(velocity, mpas_add(
                    mpas_mul(fzm[k], u[E2(k, edge, nedges)]),
                    mpas_mul(fzp[k], u[E2(k - 1, edge, nedges)])));
            } else {
                const float qm2 = u[E2(k - 2, edge, nedges)];
                const float qm1 = u[E2(k - 1, edge, nedges)];
                const float qi = u[E2(k, edge, nedges)];
                const float qp1 = u[E2(k + 1, edge, nedges)];
                value = mpas_add(mpas_div(mpas_mul(velocity, mpas_sub(
                    mpas_mul(7.0f, mpas_add(qi, qm1)),
                    mpas_add(qp1, qm2))),
                    mpas_third_order_denominator), mpas_div(mpas_mul(
                        mpas_abs(velocity), mpas_sub(mpas_sub(qp1, qm2),
                            mpas_mul(3.0f, mpas_sub(qi, qm1)))),
                        mpas_third_order_denominator));
            }
        }
        flux[index] = value;
    }
}
