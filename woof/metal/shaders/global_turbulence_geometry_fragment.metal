// Height above the surface and Kz profile for resident Global turbulence.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "global_turbulence_geometry_fragment_bindings.metalh"
using namespace metal;

kernel void global_turbulence_geometry_fragment(
    constant GlobalTurbulenceGeometryFragmentResources &r [[buffer(0)]],
    constant GlobalTurbulenceGeometryFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlevel) * uint(p.ncolumn);
    if (tid >= count) return;
    const uint column = tid % uint(p.ncolumn);
    const uint surface_index = (uint(p.nlevel) - 1u) * uint(p.ncolumn) + column;
    const ulong z = soft64_div(r.geopotential[tid], r.constants[0]);
    const ulong surface_z = soft64_div(
        r.geopotential[surface_index], r.constants[0]);
    const ulong raw_height = soft64_sub(z, surface_z);
    const ulong height = soft64_greater(raw_height, 0ul) ? raw_height : 0ul;
    const ulong exponent = soft64_negate_bits(
        soft64_div(height, r.constants[3]));
    const ulong attenuation = soft64_plm_exp(exponent);
    r.height[tid] = height;
    r.diffusivity[tid] = soft64_add(
        r.constants[1], soft64_mul(r.constants[2], attenuation));
}
