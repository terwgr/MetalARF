// Hex v8.4.1 Metal transcription of CUDA kernel validate_wsm6_surface_v841_f32 (woof/hex/cuda_physics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_validate_wsm6_surface_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


kernel void hexv841_validate_wsm6_surface_v841_f32(
    constant Hexv841ValidateWsm6SurfaceV841F32Resources &r [[buffer(0)]],
    constant Hexv841ValidateWsm6SurfaceV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int count = p.count;
    const int nbuckets = p.nbuckets;
    const float sr_upper = p.sr_upper;
    device const float *acc_p = r.acc_p;
    device const float *inc_p = r.inc_p;
    device const float *old_p = r.old_p;
    device const float *sr = r.sr;
    device int *invalid = r.invalid;

    const int i = int(tid);
    if (i >= count) return;
    // One accumulator/increment/previous triple per bucket the ROW declares.
    // WSM6 declares three (rain, snow, graupel); P3 declares two, because
    // its single ice category has no graupel accumulator and reporting a
    // permanent zero would let output claim a field the scheme never has.
    for (int j = 0; j < nbuckets; ++j) {
        const float acc[1] = {acc_p[j * count + i]};
        const float inc[1] = {inc_p[j * count + i]};
        const float old[1] = {old_p[j * count + i]};
        const int k = 0;
        if (!isfinite(acc[k]) || acc[k] < 0.0f || acc[k] < old[k]
                || !isfinite(inc[k]) || inc[k] < 0.0f) atomicExch(invalid, 1);
    }
    if (!isfinite(sr[i]) || sr[i] < 0.0f || sr[i] > sr_upper)
        atomicExch(invalid, 1);
}
