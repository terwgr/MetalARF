// Hex v8.4.1 Metal transcription of CUDA kernel validate_recovered_v841_f32 (woof/hex/cuda_dynamics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_validate_recovered_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_validate_recovered_v841_f32(
    constant Hexv841ValidateRecoveredV841F32Resources &r [[buffer(0)]],
    constant Hexv841ValidateRecoveredV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    device const float *rho = r.rho;
    device const float *rho_theta = r.rho_theta;
    device const float *rho_u = r.rho_u;
    device const float *rho_w = r.rho_w;
    device const float *theta = r.theta;
    device const float *exner = r.exner;
    device const float *rho_p = r.rho_p;
    device const float *rtheta_p = r.rtheta_p;
    device const float *pressure_p = r.pressure_p;
    device const float *normal_velocity = r.normal_velocity;
    device const float *vertical_velocity = r.vertical_velocity;
    device int *invalid = r.invalid;
    const int owner = int(tid);
    if (owner >= max(ncells, nedges)) return;


    if (owner < ncells) {
        for (int k = 0; k < nlev; ++k) {
            const int index = C2(k, owner, ncells);
            if (!(isfinite(rho[index]) && rho[index] > 0.0f
                    && isfinite(rho_theta[index]) && rho_theta[index] > 0.0f
                    && isfinite(theta[index]) && theta[index] > 0.0f
                    && isfinite(exner[index]) && exner[index] > 0.0f
                    && isfinite(rho_p[index]) && isfinite(rtheta_p[index])
                    && isfinite(pressure_p[index]))) {
                atomicExch(invalid, 1);
            }
        }
        for (int k = 0; k <= nlev; ++k) {
            const int index = C2(k, owner, ncells);
            if (!(isfinite(rho_w[index]) && isfinite(vertical_velocity[index])))
                atomicExch(invalid, 1);
        }
    }
    if (owner < nedges) {
        for (int k = 0; k < nlev; ++k) {
            const int index = E2(k, owner, nedges);
            if (!(isfinite(rho_u[index]) && isfinite(normal_velocity[index])))
                atomicExch(invalid, 1);
        }
    }

}
