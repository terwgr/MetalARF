// Hex v8.4.1 scalar transport: density validation flag (cuda_transport_v841.py validate_density_v841).
#include <metal_stdlib>
#include "hexv841_validate_density_v841_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void hexv841_validate_density_v841(
    constant Hexv841ValidateDensityV841Resources &r [[buffer(0)]],
    constant Hexv841ValidateDensityV841Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int index = int(tid);
    if (index >= p.nvalues) return;
    const float value = r.density[index];
    if (!(isfinite(value) && value > 0.0f))
        atomic_exchange_explicit(
            reinterpret_cast<device atomic_int *>(r.invalid), 1,
            memory_order_relaxed);
}
