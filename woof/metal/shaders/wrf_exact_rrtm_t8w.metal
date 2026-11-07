// Generated from the classic RRTM interface preparation.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_rrtm_lw_t8w_bindings.metalh"
using namespace metal;
kernel void wrf_exact_rrtm_lw_t8w(
    constant WrfExactRrtmLwT8WResources &r [[buffer(0)]],
    constant WrfExactRrtmLwT8WParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= uint(p.ncell)) return;
    const int col = int(index) / p.nlev;
    const int lev = int(index) - col * p.nlev;
    const ulong layer_base = ulong(col) * ulong(p.nz);
    const ulong level_base = ulong(col) * ulong(p.nlev);
    if (lev > 0 && lev < p.nz) {
        const int j = p.nz - lev;
        reinterpret_cast<device Exact32*>(r.tw)[level_base + ulong(lev)] = reinterpret_cast<device const Exact32*>(r.fnm)[j] * reinterpret_cast<device const Exact32*>(r.t)[layer_base + ulong(lev - 1)]
            + reinterpret_cast<device const Exact32*>(r.fnp)[j] * reinterpret_cast<device const Exact32*>(r.t)[layer_base + ulong(lev)];
    } else if (lev == p.nz) {
        const Exact32 z0 = 0.5f * (reinterpret_cast<device const Exact32*>(r.z_interface)[level_base + ulong(p.nz)]
                                + reinterpret_cast<device const Exact32*>(r.z_interface)[level_base + ulong(p.nz - 1)]);
        const Exact32 z1 = 0.5f * (reinterpret_cast<device const Exact32*>(r.z_interface)[level_base + ulong(p.nz - 1)]
                                + reinterpret_cast<device const Exact32*>(r.z_interface)[level_base + ulong(p.nz - 2)]);
        const Exact32 w = (reinterpret_cast<device const Exact32*>(r.z_interface)[level_base + ulong(p.nz)] - z1)
                      / (z0 - z1);
        reinterpret_cast<device Exact32*>(r.tw)[level_base + ulong(lev)] = w * reinterpret_cast<device const Exact32*>(r.t)[layer_base + ulong(p.nz - 1)]
                                      + (1.0f - w) * reinterpret_cast<device const Exact32*>(r.t)[layer_base + ulong(p.nz - 2)];
    } else {
        const Exact32 z0 = 0.5f * (reinterpret_cast<device const Exact32*>(r.z_interface)[level_base]
                                + reinterpret_cast<device const Exact32*>(r.z_interface)[level_base + 1ul]);
        const Exact32 z1 = 0.5f * (reinterpret_cast<device const Exact32*>(r.z_interface)[level_base + 1ul]
                                + reinterpret_cast<device const Exact32*>(r.z_interface)[level_base + 2ul]);
        const Exact32 w = (reinterpret_cast<device const Exact32*>(r.z_interface)[level_base] - z1) / (z0 - z1);
        reinterpret_cast<device Exact32*>(r.tw)[level_base] = w * reinterpret_cast<device const Exact32*>(r.t)[layer_base]
                         + (1.0f - w) * reinterpret_cast<device const Exact32*>(r.t)[layer_base + 1ul];
    }
}
