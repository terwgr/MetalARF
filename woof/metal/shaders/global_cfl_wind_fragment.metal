// Resident Global advective-CFL wind magnitude.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_cfl_wind_fragment_bindings.metalh"

using namespace metal;

kernel void global_cfl_wind_fragment(
    constant GlobalCflWindFragmentResources &r [[buffer(0)]],
    constant GlobalCflWindFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const ulong u2 = soft64_mul(r.u[tid], r.u[tid]);
    const ulong v2 = soft64_mul(r.v[tid], r.v[tid]);
    r.wind_speed[tid] = soft64_sqrt(soft64_add(u2, v2));
}
