// Hex v8.4.1 Metal transcription of CUDA kernel validate_positive_state_v841_f32 (woof/hex/cuda_dynamics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_validate_positive_state_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_validate_positive_state_v841_f32(
    constant Hexv841ValidatePositiveStateV841F32Resources &r [[buffer(0)]],
    constant Hexv841ValidatePositiveStateV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nvalues = p.nvalues;
    device const float *rho = r.rho;
    device const float *rho_theta = r.rho_theta;
    device int *invalid = r.invalid;
    const int index = int(tid);
    if (index >= nvalues) return;


    const float density = rho[index];
    const float theta_mass = rho_theta[index];
    if (!(isfinite(density) && density > 0.0f
            && isfinite(theta_mass) && theta_mass > 0.0f)) {
        atomicExch(invalid, 1);
    }

}
