// needs integration: Hex RK tendency assembly.
// Proposed elementwise stage boundary: source fields and saved tendencies
// are supplied by the RK caller; the integrator owns coefficients/order.
#include <metal_stdlib>
using namespace metal;

struct HexRkTendenciesResources {
    device const float *base;
    device const float *tendency;
    device float *updated;
};
struct HexRkTendenciesParams { uint count; float dt; };
kernel void hex_rk_tendencies_fragment(
    constant HexRkTendenciesResources &r [[buffer(0)]],
    constant HexRkTendenciesParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    r.updated[tid] = r.base[tid] + p.dt * r.tendency[tid];
}
