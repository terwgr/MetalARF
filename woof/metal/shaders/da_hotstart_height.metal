#include <metal_stdlib>
#include "da_hotstart_height_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;

kernel void da_hotstart_height(constant DaHotstartHeightResources& r [[buffer(0)]],
                               constant DaHotstartHeightParams& p [[buffer(1)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    uint plane = uint(p.plane), level = i / plane, cell = i % plane;
    uint b0 = p.base_column ? level : i;
    uint b1 = p.base_column ? level + 1 : i + plane;
    uint bs = p.base_column ? 0 : cell;
    uint a = as_type<uint>(r.php[i]), b = as_type<uint>(r.php[i + plane]);
    uint s = as_type<uint>(r.php[cell]);
    uint ba = as_type<uint>(r.phb[b0]), bb = as_type<uint>(r.phb[b1]);
    uint bsurf = as_type<uint>(r.phb[bs]);
    uint g = as_type<uint>(p.gravity);
    uint z0 = soft32_div(soft32_add(ba, a), g);
    uint z1 = soft32_div(soft32_add(bb, b), g);
    uint surface = soft32_div(soft32_add(bsurf, s), g);
    uint height = soft32_sub(soft32_mul(0x3f000000u, soft32_add(z0, z1)), surface);
    r.height[i] = as_type<float>(height);
}
