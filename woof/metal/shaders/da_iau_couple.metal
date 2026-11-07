#include <metal_stdlib>
#include "da_iau_couple_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;

kernel void da_iau_couple(
    constant DaIauCoupleResources &r [[buffer(0)]],
    constant DaIauCoupleParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    const uint nx = uint(p.nx), ny = uint(p.ny);
    const uint width = nx + uint(p.stagger == 1);
    const uint height = ny + uint(p.stagger == 2);
    const uint plane = width * height;
    const uint k = gid / plane, cell = gid % plane;
    const uint y = cell / width, x = cell % width;
    uint x0 = min(x, nx - 1u), y0 = min(y, ny - 1u);
    uint mass = soft32_add(as_type<uint>(r.mub[y0 * nx + x0]),
                           as_type<uint>(r.mup[y0 * nx + x0]));
    if ((p.stagger == 1 && x > 0u && x < nx) ||
        (p.stagger == 2 && y > 0u && y < ny)) {
        const uint prev = p.stagger == 1 ? y0 * nx + x0 - 1u : (y0 - 1u) * nx + x0;
        uint other = soft32_add(as_type<uint>(r.mub[prev]), as_type<uint>(r.mup[prev]));
        mass = soft32_mul(soft32_add(mass, other), 0x3f000000u);
    }
    uint coeff = soft32_add(soft32_mul(as_type<uint>(r.c1[k]), mass), as_type<uint>(r.c2[k]));
    uint value = soft32_mul(coeff, as_type<uint>(r.rate[gid]));
    if ((p.mask_x > 0 && (x < uint(p.mask_x) || x >= width - min(width, uint(p.mask_x)))) ||
        (p.mask_y > 0 && (y < uint(p.mask_y) || y >= height - min(height, uint(p.mask_y)))))
        value = 0u;
    // When enabled, map is one stagger-specific horizontal plane; [n] is an inactive dummy.
    if (p.has_map) value = soft32_div(value, as_type<uint>(r.map[cell]));
    r.rate[gid] = as_type<float>(value);
}
