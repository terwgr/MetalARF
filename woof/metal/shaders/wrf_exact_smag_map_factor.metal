#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "wrf_exact_smag_map_factor_bindings.metalh"
using namespace metal;

// _couple_dry_mixing_map_factor: y uses the supplied inverse.
kernel void wrf_exact_smag_map_factor(
    constant WrfExactSmagMapFactorResources &r [[buffer(0)]],
    constant WrfExactSmagMapFactorParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    ulong nx = ulong(p.nx + (p.stagger == 1));
    ulong ny = ulong(p.ny + (p.stagger == 2));
    ulong nz = ulong(p.nz + (p.stagger == 3));
    if (ulong(gid) >= nx * ny * nz) return;
    uint factor = as_type<uint>(r.map_factor[ulong(gid) % (nx * ny)]);
    uint value = as_type<uint>(r.tendency[gid]);
    uint result = p.multiply_reciprocal
        ? soft32_mul(value, factor)
        : soft32_div(value, factor);
    reinterpret_cast<device uint *>(r.tendency)[gid] = result;
}
