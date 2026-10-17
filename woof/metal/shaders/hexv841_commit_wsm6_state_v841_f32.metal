// Hex v8.4.1 Metal transcription of CUDA kernel commit_wsm6_state_v841_f32 (woof/hex/cuda_physics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_commit_wsm6_state_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


kernel void hexv841_commit_wsm6_state_v841_f32(
    constant Hexv841CommitWsm6StateV841F32Resources &r [[buffer(0)]],
    constant Hexv841CommitWsm6StateV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int count = p.count;
    device const float *candidate_rho_theta = r.candidate_rho_theta;
    device const float *candidate_scalars = r.candidate_scalars;
    device float *rho_theta = r.rho_theta;
    device float *scalars = r.scalars;

    const int i = int(tid);
    if (i >= count) return;
    rho_theta[i] = candidate_rho_theta[i];
    // The run's species row (six for WSM6, eight for P3, eleven for mp=28).
    for (int s = 0; s < p.nspecies; ++s) {
        scalars[s * count + i] = candidate_scalars[s * count + i];
    }
}
