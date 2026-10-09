// FP64 Global gray longwave radiation transfer.
//
// This fragment computes the downward/upward two-stream recurrence and
// layer heating in ReferencePhysics._surface_radiation_fluxes. Emissivity,
// emission, surface longwave emission and absorbed shortwave are prepared by
// the caller.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_gray_radiation_fragment_bindings.metalh"

using namespace metal;

static inline ulong global_gray_down_fragment(
    device const ulong *epsilon, device const ulong *emission,
    uint nlevel, uint column, uint ncolumn, uint interface) {
    ulong flux = 0ul;
    const ulong one = 0x3ff0000000000000ul;
    for (uint level = 0u; level < interface; ++level) {
        const uint index = level * ncolumn + column;
        flux = soft64_add(
            soft64_mul(flux, soft64_sub(one, epsilon[index])),
            soft64_mul(epsilon[index], emission[index]));
    }
    return flux;
}

static inline ulong global_gray_up_fragment(
    device const ulong *epsilon, device const ulong *emission,
    uint nlevel, uint column, uint ncolumn, uint interface,
    ulong surface_emission, ulong surface_emissivity, ulong down_surface) {
    const ulong one = 0x3ff0000000000000ul;
    ulong flux = soft64_add(surface_emission,
        soft64_mul(soft64_sub(one, surface_emissivity), down_surface));
    for (uint level = nlevel; level > interface; --level) {
        const uint k = level - 1u;
        const uint index = k * ncolumn + column;
        flux = soft64_add(
            soft64_mul(flux, soft64_sub(one, epsilon[index])),
            soft64_mul(epsilon[index], emission[index]));
    }
    return flux;
}

kernel void global_gray_radiation_fragment(
    constant GlobalGrayRadiationFragmentResources &r [[buffer(0)]],
    constant GlobalGrayRadiationFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlevel) * uint(p.ncolumn);
    if (tid >= count) return;
    const uint level = tid / uint(p.ncolumn);
    const uint column = tid - level * uint(p.ncolumn);
    const ulong down_top = global_gray_down_fragment(
        r.epsilon, r.emission, uint(p.nlevel), column, uint(p.ncolumn), level);
    const ulong down_bottom = global_gray_down_fragment(
        r.epsilon, r.emission, uint(p.nlevel), column, uint(p.ncolumn), level + 1u);
    const ulong up_bottom = global_gray_up_fragment(
        r.epsilon, r.emission, uint(p.nlevel), column, uint(p.ncolumn), level + 1u,
        r.surface_emission[column], r.surface_emissivity[column],
        global_gray_down_fragment(
            r.epsilon, r.emission, uint(p.nlevel), column, uint(p.ncolumn), uint(p.nlevel)));
    const ulong up_top = global_gray_up_fragment(
        r.epsilon, r.emission, uint(p.nlevel), column, uint(p.ncolumn), level,
        r.surface_emission[column], r.surface_emissivity[column],
        global_gray_down_fragment(
            r.epsilon, r.emission, uint(p.nlevel), column, uint(p.ncolumn), uint(p.nlevel)));
    r.longwave_heating[tid] = soft64_sub(
        soft64_sub(up_bottom, down_bottom), soft64_sub(up_top, down_top));
    if (level == 0u) {
        r.outgoing_longwave[column] = up_top;
        const ulong down_surface = global_gray_down_fragment(
            r.epsilon, r.emission, uint(p.nlevel), column, uint(p.ncolumn), uint(p.nlevel));
        r.net_surface_radiation[column] = soft64_sub(
            soft64_add(r.absorbed_shortwave[column],
                soft64_mul(r.surface_emissivity[column], down_surface)),
            r.surface_emission[column]);
    }
}
