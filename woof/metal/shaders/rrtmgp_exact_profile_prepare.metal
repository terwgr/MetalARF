// Source-ordered RRTMGP pressure-weighted interface temperatures.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "rrtmgp_exact_interface_temperatures_bindings.metalh"

using namespace metal;

kernel void rrtmgp_exact_interface_temperatures(
    constant RrtmgpExactInterfaceTemperaturesResources &r [[buffer(0)]],
    constant RrtmgpExactInterfaceTemperaturesParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const uint nlay = uint(p.nlay), nlev = nlay + 1u;
    if (gid >= uint(p.ncol) * nlev || nlay < 2u) return;
    const uint col = gid / nlev, lev = gid % nlev;
    const uint start = col * nlay;
    const Exact32 face(r.plev[gid]);
    Exact32 temperature;
    if (lev == 0u || lev == nlay) {
        const uint a = lev == 0u ? start : start + nlay - 1u;
        const uint b = lev == 0u ? a + 1u : a - 1u;
        const Exact32 ta(r.tlay[a]), tb(r.tlay[b]);
        const Exact32 pa(r.play[a]), pb(r.play[b]);
        temperature = lev == 0u
            ? ta + (face - pa) * (tb - ta) / (pb - pa)
            : ta + (face - pa) * (ta - tb) / (pa - pb);
    } else {
        const uint lower = start + lev - 1u, upper = lower + 1u;
        const Exact32 pl(r.play[lower]), pu(r.play[upper]);
        const Exact32 tl(r.tlay[lower]), tu(r.tlay[upper]);
        const Exact32 numerator = pl * tl * (face - pu) + pu * tu * (pl - face);
        temperature = numerator / (face * (pl - pu));
    }
    r.tlev[gid] = as_type<float>(temperature.bits);
}

#include "rrtmgp_exact_nonnegative_columns_bindings.metalh"
kernel void rrtmgp_exact_nonnegative_columns(
    constant RrtmgpExactNonnegativeColumnsResources &r [[buffer(0)]],
    constant RrtmgpExactNonnegativeColumnsParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    const uint word = as_type<uint>(r.input[gid]);
    r.output[gid] = as_type<float>(
        (word & 0x80000000u) != 0u && !soft32_is_nan(word) ? 0u : word);
}
