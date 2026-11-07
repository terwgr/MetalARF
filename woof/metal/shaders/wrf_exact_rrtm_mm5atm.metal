// Generated from classic RRTM column preparation; do not edit.
#include <metal_stdlib>
#include "exact_gfk_math.metalh"
#include "wrf_exact_rrtm_lw_mm5atm_bindings.metalh"
using namespace metal;
static inline bool operator<(Exact32 a, Exact32 b) { return soft32_less(a.bits, b.bits); }
inline Exact32 rrtm_lw_o3_layer(device const Exact32 *o3wrk,
                              device const Exact32 *ppwrkh,
                              Exact32 bottom, Exact32 top) {
    Exact32 weighted = 0.0f;
    for (int i = 0; i < 31; ++i) {
        const Exact32 pb1 = exact_max(bottom - ppwrkh[i], 0.0f);
        const Exact32 pb2 = exact_max(bottom - ppwrkh[i + 1], 0.0f);
        const Exact32 pt1 = exact_max(top - ppwrkh[i], 0.0f);
        const Exact32 pt2 = exact_max(top - ppwrkh[i + 1], 0.0f);
        weighted += (pb2 - pb1 - pt2 + pt1) * o3wrk[i];
    }
    return weighted / (bottom - top);
}

inline Exact32 rrtm_lw_standard_temperature(device const Exact32 *pprof,
                                          device const Exact32 *tprof,
                                          int nprof, Exact32 pressure) {
    int above = 0;
    for (int i = 0; i < nprof; ++i) above += pprof[i] >= pressure;
    int klev = clamp(above, 1, nprof);
    if (!(pressure > pprof[nprof - 1])) klev = nprof;
    const bool last = klev == nprof;
    const int k0 = clamp(klev - 1, 0, nprof - 1);
    const int k1 = clamp(klev, 0, nprof - 1);
    const Exact32 vark = tprof[k0];
    const Exact32 vark1 = last ? vark : tprof[k1];
    const Exact32 span = last ? 1.0f : pprof[k1] - pprof[k0];
    const Exact32 weight = last ? 0.0f : (pressure - pprof[k0]) / span;
    return weight * (vark1 - vark) + vark;
}

kernel void wrf_exact_rrtm_lw_mm5atm(
    constant WrfExactRrtmLwMm5AtmResources &r [[buffer(0)]],
    constant WrfExactRrtmLwMm5AtmParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong c = ulong(col);
    const ulong nlay = ulong(p.nlay);
    const ulong nz = ulong(p.nz);
    const ulong levbase = c * (nlay + 1ul);
    const ulong laybase = c * nlay;
    const ulong inlay = c * nz;
    const ulong inlev = c * (nz + 1ul);
    for (int lev = 0; lev <= p.nz; ++lev) {
        reinterpret_cast<device Exact32*>(r.pz)[levbase + ulong(lev)] = reinterpret_cast<device const Exact32*>(r.pw_pa)[inlev + ulong(p.nz - lev)] / 100.0f;
        reinterpret_cast<device Exact32*>(r.tz)[levbase + ulong(lev)] = reinterpret_cast<device const Exact32*>(r.tw)[inlev + ulong(p.nz - lev)];
    }
    for (int lay = 0; lay < p.nz; ++lay) {
        const ulong dst = laybase + ulong(lay);
        const ulong src = inlay + ulong(p.nz - 1 - lay);
        reinterpret_cast<device Exact32*>(r.pavel)[dst] = reinterpret_cast<device const Exact32*>(r.p_pa)[src] / 100.0f;
        reinterpret_cast<device Exact32*>(r.tavel)[dst] = reinterpret_cast<device const Exact32*>(r.t)[src];
    }
    for (int lev = p.nz + 1; lev < p.nlay; ++lev) {
        const Exact32 pressure = reinterpret_cast<device Exact32*>(r.pz)[levbase + ulong(lev - 1)] - 4.0f;
        reinterpret_cast<device Exact32*>(r.pz)[levbase + ulong(lev)] = pressure;
        reinterpret_cast<device Exact32*>(r.pavel)[laybase + ulong(lev - 1)] = 0.5f
            * (pressure + reinterpret_cast<device Exact32*>(r.pz)[levbase + ulong(lev - 1)]);
    }
    reinterpret_cast<device Exact32*>(r.pz)[levbase + nlay] = 0.0f;
    reinterpret_cast<device Exact32*>(r.pavel)[laybase + nlay - 1ul] = 0.5f * (Exact32(0.0f) + reinterpret_cast<device Exact32*>(r.pz)[levbase + nlay - 1ul]);
    if (p.nlay > p.nz) {
        const Exact32 offset = reinterpret_cast<device Exact32*>(r.tz)[levbase + nz]
            - rrtm_lw_standard_temperature(
                reinterpret_cast<device const Exact32*>(r.pprof), reinterpret_cast<device const Exact32*>(r.tprof), p.nprof, reinterpret_cast<device Exact32*>(r.pz)[levbase + nz]);
        for (int lev = p.nz + 1; lev <= p.nlay; ++lev) {
            const Exact32 standard = rrtm_lw_standard_temperature(
                reinterpret_cast<device const Exact32*>(r.pprof), reinterpret_cast<device const Exact32*>(r.tprof), p.nprof, reinterpret_cast<device Exact32*>(r.pz)[levbase + ulong(lev)]);
            reinterpret_cast<device Exact32*>(r.tz)[levbase + ulong(lev)] = standard + offset;
            reinterpret_cast<device Exact32*>(r.tavel)[laybase + ulong(lev - 1)] = 0.5f
                * (reinterpret_cast<device Exact32*>(r.tz)[levbase + ulong(lev)]
                   + reinterpret_cast<device Exact32*>(r.tz)[levbase + ulong(lev - 1)]);
        }
    }
    const Exact32 tbound = reinterpret_cast<device const Exact32*>(r.tsfc)[c] > 340.0f ? 339.99f : reinterpret_cast<device const Exact32*>(r.tsfc)[c];
    reinterpret_cast<device Exact32*>(r.tbound)[c] = tbound;
    for (int band = 0; band < 16; ++band)
        reinterpret_cast<device Exact32*>(r.semiss)[c * 16ul + ulong(band)] = reinterpret_cast<device const Exact32*>(r.emiss)[c];

    const Exact32 grav = Exact32(p.g) * 100.0f;
    for (int lay = 0; lay < p.nlay; ++lay) {
        const ulong dst = laybase + ulong(lay);
        Exact32 o3;
        if (p.nlay == p.nz) {
            o3 = rrtm_lw_o3_layer(reinterpret_cast<device const Exact32*>(r.o3wrk), reinterpret_cast<device const Exact32*>(r.ppwrkh),
                reinterpret_cast<device Exact32*>(r.pz)[levbase + ulong(lay)], (p.nlay == p.nz && lay == p.nz - 1 ? reinterpret_cast<device const Exact32*>(r.pw_pa)[inlev] / 100.0f : reinterpret_cast<device Exact32*>(r.pz)[levbase + ulong(lay + 1)]));
        } else if (lay == p.nlay - 1) {
            o3 = 6.135e-6f;
        } else {
            o3 = rrtm_lw_o3_layer(reinterpret_cast<device const Exact32*>(r.o3wrk), reinterpret_cast<device const Exact32*>(r.ppwrkh),
                (p.nlay == p.nz && lay == p.nz - 1 ? reinterpret_cast<device const Exact32*>(r.pw_pa)[inlev] / 100.0f : reinterpret_cast<device Exact32*>(r.pz)[levbase + ulong(lay + 1)]),
                reinterpret_cast<device Exact32*>(r.pz)[levbase + ulong(lay + 2)]);
        }
        const ulong src = inlay + ulong(p.nz - 1 - metal::min(lay, p.nz - 1));
        const Exact32 q = lay < p.nz ? exact_max(reinterpret_cast<device const Exact32*>(r.qv)[src], 1.0e-12f) * 1.607758f : 5.0e-6f;
        const Exact32 co2 = Exact32(p.co2vmr);
        const Exact32 n2o = Exact32(p.n2ovmr);
        const Exact32 ch4 = Exact32(p.ch4vmr);
        const Exact32 amm = (1.0f - q) * 28.9644f
                        + q * 18.0154f;
        const Exact32 coldry = (reinterpret_cast<device Exact32*>(r.pz)[levbase + ulong(lay)]
                             - (p.nlay == p.nz && lay == p.nz - 1 ? reinterpret_cast<device const Exact32*>(r.pw_pa)[inlev] / 100.0f : reinterpret_cast<device Exact32*>(r.pz)[levbase + ulong(lay + 1)]))
                           * 1.0e3f * 6.022e23f
                           / (grav * amm * (1.0f + q));
        reinterpret_cast<device Exact32*>(r.coldry)[dst] = coldry;
        Exact32 cloud_tau = 0.0f;
        if (lay < p.nz) {
            const ulong src = inlay + ulong(p.nz - 1 - lay);
            const Exact32 temp = reinterpret_cast<device const Exact32*>(r.t)[src];
            const Exact32 rho = (reinterpret_cast<device const Exact32*>(r.p_pa)[src] / 100.0f) / (Exact32(p.r) * temp) * 100.0f;
            const Exact32 clwp = rho * exact_max(reinterpret_cast<device const Exact32*>(r.qc)[src], 0.0f) * reinterpret_cast<device const Exact32*>(r.dz)[src] * 1000.0f;
            const Exact32 ciwp = rho * exact_max(reinterpret_cast<device const Exact32*>(r.qi)[src], 0.0f) * reinterpret_cast<device const Exact32*>(r.dz)[src] * 1000.0f;
            const Exact32 plwp = exact_gfk_pow(rho * exact_max(reinterpret_cast<device const Exact32*>(r.qr)[src], 0.0f), 0.75f)
                               * reinterpret_cast<device const Exact32*>(r.dz)[src] * 1000.0f;
            const Exact32 piwp = exact_gfk_pow(rho * exact_max(reinterpret_cast<device const Exact32*>(r.qs)[src], 0.0f), 0.75f)
                               * reinterpret_cast<device const Exact32*>(r.dz)[src] * 1000.0f;
            cloud_tau = 0.144f * clwp + 0.0735f * ciwp
                      + 0.000330f * plwp + 0.00234f * piwp;
            reinterpret_cast<device Exact32*>(r.taucloud)[dst] = cloud_tau;
            reinterpret_cast<device Exact32*>(r.cldfra_out)[dst] = cloud_tau > 0.01f ? 1.0f : reinterpret_cast<device const Exact32*>(r.cldfra)[src];
        } else {
            reinterpret_cast<device Exact32*>(r.taucloud)[dst] = 0.0f;
            reinterpret_cast<device Exact32*>(r.cldfra_out)[dst] = 0.0f;
        }
        for (int m = 0; m < 35; ++m) reinterpret_cast<device Exact32*>(r.wkl)[dst * 35ul + ulong(m)] = 0.0f;
        for (int x = 0; x < 4; ++x) reinterpret_cast<device Exact32*>(r.wx)[dst * 4ul + ulong(x)] = 0.0f;
        reinterpret_cast<device Exact32*>(r.wkl)[dst * 35ul] = q * coldry;
        reinterpret_cast<device Exact32*>(r.wkl)[dst * 35ul + 1ul] = co2 * coldry;
        reinterpret_cast<device Exact32*>(r.wkl)[dst * 35ul + 2ul] = o3 * 0.603461f * coldry;
        reinterpret_cast<device Exact32*>(r.wkl)[dst * 35ul + 3ul] = n2o * coldry;
        reinterpret_cast<device Exact32*>(r.wkl)[dst * 35ul + 5ul] = ch4 * coldry;
    }
}
