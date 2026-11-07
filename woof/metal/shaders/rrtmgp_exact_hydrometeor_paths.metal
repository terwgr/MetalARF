#include <metal_stdlib>
#include "wrf_exact_rrtmgp_hydrometeor_paths_bindings.metalh"
#include "exact_binary32.metalh"
using namespace metal;

// NumPy maximum/minimum propagate NaNs and select the second equal operand.
static inline Exact32 hp_max(Exact32 a, Exact32 b) {
    if (soft32_is_nan(a.bits)) return a;
    if (soft32_is_nan(b.bits)) return b;
    return a > b ? a : b;
}
static inline Exact32 hp_min(Exact32 a, Exact32 b) {
    if (soft32_is_nan(a.bits)) return a;
    if (soft32_is_nan(b.bits)) return b;
    return b > a ? a : b;
}

kernel void wrf_exact_rrtmgp_hydrometeor_paths(
    constant WrfExactRrtmgpHydrometeorPathsResources &r [[buffer(0)]],
    constant WrfExactRrtmgpHydrometeorPathsParams &p [[buffer(1)]],
    uint cell [[thread_position_in_grid]]) {
    if (cell >= uint(p.ncell)) return;
    device const Exact32 *plev = reinterpret_cast<device const Exact32 *>(r.plev);
    device const Exact32 *qc = reinterpret_cast<device const Exact32 *>(r.qc);
    device const Exact32 *qi = reinterpret_cast<device const Exact32 *>(r.qi);
    device const Exact32 *qs = reinterpret_cast<device const Exact32 *>(r.qs);
    device const Exact32 *effc = reinterpret_cast<device const Exact32 *>(r.effc);
    device const Exact32 *effi = reinterpret_cast<device const Exact32 *>(r.effi);
    device const Exact32 *effs = reinterpret_cast<device const Exact32 *>(r.effs);
    device const Exact32 *cldfra = reinterpret_cast<device const Exact32 *>(r.cldfra);
    device Exact32 *clwp = reinterpret_cast<device Exact32 *>(r.clwp);
    device Exact32 *ciwp = reinterpret_cast<device Exact32 *>(r.ciwp);
    device Exact32 *reliq = reinterpret_cast<device Exact32 *>(r.reliq);
    device Exact32 *dgice = reinterpret_cast<device Exact32 *>(r.dgice);
    const uint idx = (cell / uint(p.nlay)) * uint(p.nlay + 1) + cell % uint(p.nlay);
    const Exact32 zero=0.0f, tiny=1.0e-20f;
    const Exact32 ice = p.has_qi ? qi[cell] : zero;
    const Exact32 snow = p.has_qs ? qs[cell] : zero;
    const Exact32 mass = exact_abs(plev[idx+1] - plev[idx]) * Exact32(101.97162129779283f);
    const Exact32 incloud = p.has_cldfra ? hp_max(Exact32(0.01f), cldfra[cell]) : Exact32(1.0f);
    Exact32 liquid = qc[cell] * mass;
    Exact32 frozen_path = (ice + snow) * mass;
    if (p.has_cldfra) { liquid = liquid / incloud; frozen_path = frozen_path / incloud; }
    Exact32 rl=10.0f, di=50.0f;
    if (p.scheme == 1 || p.scheme == 2) {
        Exact32 snow_eff = snow, radius = p.scheme == 1 ? effs[cell] : effi[cell];
        if (p.snow_discount || p.scheme == 2) {
            const Exact32 r0 = hp_max(Exact32(10.0f), radius);
            const Exact32 quotient = Exact32(130.0f) / r0;
            const Exact32 factor = r0 > Exact32(130.0f)
                ? hp_min(Exact32(0.99f), quotient * quotient) : Exact32(0.99f);
            radius = hp_min(r0, Exact32(130.0f));
            snow_eff = (p.scheme == 2 ? ice : snow) * factor;
            frozen_path = (p.scheme == 2 ? snow_eff : ice + snow_eff) * mass;
            if (p.has_cldfra) frozen_path = frozen_path / incloud;
        }
        Exact32 ri=25.0f;
        if (p.scheme == 2) {
            if (snow_eff > tiny) ri = radius;
        } else {
            const Exact32 frozen = ice + snow_eff;
            if (frozen > tiny)
                ri = (ice * effi[cell] + snow_eff * radius) / hp_max(frozen, tiny);
        }
        rl = qc[cell] > tiny ? effc[cell] : Exact32(10.0f);
        rl = hp_min(hp_max(rl, Exact32(2.5f)), Exact32(21.5f));
        di = hp_min(hp_max(Exact32(2.0f) * ri, Exact32(10.0f)), Exact32(180.0f));
    }
    clwp[cell] = liquid; ciwp[cell] = frozen_path;
    reliq[cell] = rl; dgice[cell] = di;
}
