// Strict RRTMGP delta scaling, VMR stores and radiation publication.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "soft_binary64.metalh"
#include "rrtmgp_exact_delta_scale_bindings.metalh"
#include "rrtmgp_exact_gas_vmr_fill_bindings.metalh"
#include "rrtmgp_exact_fluxes_to_radiation_bindings.metalh"

using namespace metal;

kernel void rrtmgp_exact_delta_scale(
    constant RrtmgpExactDeltaScaleResources &r [[buffer(0)]],
    constant RrtmgpExactDeltaScaleParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    const Exact32 eps(3.0f * 1.17549435e-38f);
    const Exact32 g(r.g_in[gid]), ssa(r.ssa_in[gid]), tau(r.tau_in[gid]);
    const Exact32 f = g * g, wf = ssa * f;
    const Exact32 one(1.0f);
    r.tau[gid] = as_type<float>(((one - wf) * tau).bits);
    r.ssa[gid] = as_type<float>(((ssa - wf) / exact_max(eps, one - wf)).bits);
    r.asym[gid] = as_type<float>(((g - f) / exact_max(eps, one - f)).bits);
}

kernel void rrtmgp_exact_gas_vmr_fill(
    constant RrtmgpExactGasVmrFillResources &r [[buffer(0)]],
    constant RrtmgpExactGasVmrFillParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.ncells) * uint(p.nslot)) return;
    const uint cell = gid / uint(p.nslot);
    const int slot = int(gid % uint(p.nslot));
    uint value = 0u;
    if (slot == p.h2o_index) value = (Exact32(r.qv[cell]) * Exact32(p.h2o_scale)).bits;
    if (slot == p.o3_index) {
        const ulong lo = as_type<uint>(r.o3_words[2u * cell]);
        const ulong hi = as_type<uint>(r.o3_words[2u * cell + 1u]);
        value = soft64_to_f32_bits((hi << 32) | lo);
    }
    for (int t = 0; t < p.ntrace; ++t) {
        if (as_type<int>(r.trace_index[t]) == slot) value = as_type<uint>(r.trace_value[t]);
    }
    r.vmr[gid] = as_type<float>(value);
}

static inline Exact32 flux_heating(device const float *up, device const float *down,
                                   uint lower, Exact32 dp, Exact32 exner, Exact32 factor) {
    const Exact32 bottom = Exact32(down[lower]) - Exact32(up[lower]);
    const Exact32 top = Exact32(down[lower + 1u]) - Exact32(up[lower + 1u]);
    return factor * (top - bottom) / dp / exner;
}

kernel void rrtmgp_exact_fluxes_to_radiation(
    constant RrtmgpExactFluxesToRadiationResources &r [[buffer(0)]],
    constant RrtmgpExactFluxesToRadiationParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(max(p.ncell, p.ncol))) return;
    if (gid < uint(p.ncell)) {
        const uint lay = gid / uint(p.ncol), col = gid % uint(p.ncol);
        const uint lower = col * uint(p.nlev) + lay;
        const uint cell = col * uint(p.nlay) + lay;
        Exact32 dp = Exact32(r.plev[lower + 1u]) - Exact32(r.plev[lower]);
        dp.bits &= 0x7fffffffu;
        const Exact32 exner(r.exner[cell]), factor(p.heating_factor);
        r.rthratenlw[gid] = as_type<float>(flux_heating(r.lw_up, r.lw_dn, lower, dp, exner, factor).bits);
        r.rthratensw[gid] = as_type<float>(flux_heating(r.sw_up, r.sw_dn, lower, dp, exner, factor).bits);
    }
    if (gid < uint(p.ncol)) {
        const uint base = gid * uint(p.nlev);
        r.swdown[gid] = r.sw_dn[base];
        r.glw[gid] = r.lw_dn[base];
        r.olr[gid] = r.lw_up[base + uint(p.nlay)];
        r.gsw[gid] = as_type<float>((Exact32(r.sw_dn[base]) - Exact32(r.sw_up[base])).bits);
    }
}
