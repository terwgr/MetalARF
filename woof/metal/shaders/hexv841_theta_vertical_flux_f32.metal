// Hex v8.4.1 Metal transcription of CUDA kernel theta_vertical_flux_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_theta_vertical_flux_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_theta_vertical_flux_f32(
    constant Hexv841ThetaVerticalFluxF32Resources &r [[buffer(0)]],
    constant Hexv841ThetaVerticalFluxF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float coefficient = p.coefficient;
    device const float *theta = r.theta;
    device const float *theta_saved = r.theta_saved;
    device const float *rw = r.rw;
    device const float *rw_saved = r.rw_saved;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device float *flux = r.flux;
    const int element = int(tid);
    if (element >= ((nlev + 1) * ncells)) return;
    {

        const int cell = element % ncells;
        const int k = element / ncells;
        const int index = C2(k, cell, ncells);
        float value = 0.0f;
        if (k > 0 && k < nlev) {
            if (k == 1) {
                value = mpas_mul(rw[index], mpas_add(
                    mpas_mul(fzm[k], theta[C2(k, cell, ncells)]),
                    mpas_mul(fzp[k],
                        theta[C2(k - 1, cell, ncells)])));
            } else if (k == nlev - 1) {
                value = mpas_mul(rw_saved[index], mpas_add(
                    mpas_mul(fzm[k], theta[C2(k, cell, ncells)]),
                    mpas_mul(fzp[k],
                        theta[C2(k - 1, cell, ncells)])));
            } else {
                const float qm2 = theta[C2(k - 2, cell, ncells)];
                const float qm1 = theta[C2(k - 1, cell, ncells)];
                const float qi = theta[C2(k, cell, ncells)];
                const float qp1 = theta[C2(k + 1, cell, ncells)];
                const float velocity = rw[index];
                value = mpas_add(mpas_div(mpas_mul(velocity, mpas_sub(
                    mpas_mul(7.0f, mpas_add(qi, qm1)),
                    mpas_add(qp1, qm2))),
                    mpas_third_order_denominator), mpas_div(mpas_mul(
                        mpas_mul(coefficient, mpas_abs(velocity)),
                        mpas_sub(mpas_sub(qp1, qm2),
                            mpas_mul(3.0f, mpas_sub(qi, qm1)))),
                        mpas_third_order_denominator));
            }
            if (k < nlev - 1) {
                value = mpas_add(value, mpas_mul(
                    mpas_sub(rw_saved[index], rw[index]), mpas_add(
                        mpas_mul(fzm[k],
                            theta_saved[C2(k, cell, ncells)]),
                        mpas_mul(fzp[k],
                            theta_saved[C2(k - 1, cell, ncells)]))));
            }
        }
        flux[index] = value;
    }
}
