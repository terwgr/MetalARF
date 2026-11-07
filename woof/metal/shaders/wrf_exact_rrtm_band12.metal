// Generated from classic RRTM GASABS bands 1 and 2; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_rrtm_lw_taumol_01_02_bindings.metalh"
using namespace metal;
static inline int rrtm_trunc(Exact32 x) { return int(trunc(as_type<float>(x.bits))); }
inline Exact32 rrtm_lw_interp4(device const Exact32 *tab, int nrow, int ng,
                              int g, int i0, int i1,
                              Exact32 f00, Exact32 f01, Exact32 f10, Exact32 f11) {
    i0 = clamp(i0, 0, nrow - 1); i1 = clamp(i1, 0, nrow - 1);
    const int i0n = clamp(i0 + 1, 0, nrow - 1);
    const int i1n = clamp(i1 + 1, 0, nrow - 1);
    return f00 * tab[ulong(i0) * ulong(ng) + ulong(g)]
         + f10 * tab[ulong(i0n) * ulong(ng) + ulong(g)]
         + f01 * tab[ulong(i1) * ulong(ng) + ulong(g)]
         + f11 * tab[ulong(i1n) * ulong(ng) + ulong(g)];
}

kernel void wrf_exact_rrtm_lw_taumol_01_02(
    constant WrfExactRrtmLwTaumol0102Resources &r [[buffer(0)]],
    constant WrfExactRrtmLwTaumol0102Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col = int(tid) / p.nlay;
    const int lay = int(tid) - col * p.nlay;
    const ulong i = ulong(tid);
    const int jp = r.jp[i], jt = r.jt[i], jt1 = r.jt1[i];
    const int lo1 = ((jp - 1) * 5 + (jt - 1)) * p.nspa1;
    const int lo11 = (jp * 5 + (jt1 - 1)) * p.nspa1;
    const int hi1 = ((jp - 13) * 5 + (jt - 1)) * p.nspb1;
    const int hi11 = ((jp - 12) * 5 + (jt1 - 1)) * p.nspb1;
    const int self = clamp(r.indself[i] - 1, 0, p.nself1 - 2);
    const bool trop = lay + 1 <= r.laytrop[col];
    for (int g = 0; g < 8; ++g) {
    const Exact32 selfterm = reinterpret_cast<device const Exact32*>(r.selffac)[i] * (reinterpret_cast<device const Exact32*>(r.self1)[ulong(self) * 8ul + ulong(g)]
        + reinterpret_cast<device const Exact32*>(r.selffrac)[i] * (reinterpret_cast<device const Exact32*>(r.self1)[ulong(self + 1) * 8ul + ulong(g)]
                           - reinterpret_cast<device const Exact32*>(r.self1)[ulong(self) * 8ul + ulong(g)]));
        const Exact32 low = reinterpret_cast<device const Exact32*>(r.colh2o)[i] * (rrtm_lw_interp4(
            reinterpret_cast<device const Exact32*>(r.absa1), p.nabsa1, 8, g, lo1, lo11,
            reinterpret_cast<device const Exact32*>(r.fac00)[i], reinterpret_cast<device const Exact32*>(r.fac01)[i], reinterpret_cast<device const Exact32*>(r.fac10)[i], reinterpret_cast<device const Exact32*>(r.fac11)[i])
            + selfterm + reinterpret_cast<device const Exact32*>(r.forfac)[i] * reinterpret_cast<device const Exact32*>(r.forref1)[g]);
        const Exact32 high = reinterpret_cast<device const Exact32*>(r.colh2o)[i] * (rrtm_lw_interp4(
            reinterpret_cast<device const Exact32*>(r.absb1), p.nabsb1, 8, g, hi1, hi11,
            reinterpret_cast<device const Exact32*>(r.fac00)[i], reinterpret_cast<device const Exact32*>(r.fac01)[i], reinterpret_cast<device const Exact32*>(r.fac10)[i], reinterpret_cast<device const Exact32*>(r.fac11)[i])
            + reinterpret_cast<device const Exact32*>(r.forfac)[i] * reinterpret_cast<device const Exact32*>(r.forref1)[g]);
        const ulong out = i * 140ul + ulong(g);
        reinterpret_cast<device Exact32*>(r.taug)[out] = trop ? low : high;
        reinterpret_cast<device Exact32*>(r.pfrac)[out] = trop ? reinterpret_cast<device const Exact32*>(r.fraca1)[g] : reinterpret_cast<device const Exact32*>(r.fracb1)[g];
    }

    const Exact32 water = 1.0e20f * reinterpret_cast<device const Exact32*>(r.colh2o)[i] / reinterpret_cast<device const Exact32*>(r.coldry)[i];
    const Exact32 h2oparam = water / (water + 0.002f);
    int ifrac = 13;
    for (int candidate = 12; candidate >= 2; --candidate)
        if (h2oparam >= reinterpret_cast<device const Exact32*>(r.refparam2)[candidate - 1]) ifrac = candidate;
    const Exact32 refat = reinterpret_cast<device const Exact32*>(r.refparam2)[ifrac - 1];
    const Exact32 refprev = reinterpret_cast<device const Exact32*>(r.refparam2)[ifrac - 2];
    const Exact32 fracint = (h2oparam - refat) / (refprev - refat);
    const Exact32 fp2 = reinterpret_cast<device const Exact32*>(r.fac11)[i] + reinterpret_cast<device const Exact32*>(r.fac01)[i];
    int ifp = rrtm_trunc(200.0f * fp2 + 0.5f);
    if (ifp <= 0) ifp = 0;
    ifp = clamp(ifp, 0, p.ncorr - 1);
    const Exact32 fc00 = reinterpret_cast<device const Exact32*>(r.fac00)[i] * reinterpret_cast<device const Exact32*>(r.corr2)[ifp];
    const Exact32 fc10 = reinterpret_cast<device const Exact32*>(r.fac10)[i] * reinterpret_cast<device const Exact32*>(r.corr2)[ifp];
    const Exact32 fc01 = reinterpret_cast<device const Exact32*>(r.fac01)[i] * reinterpret_cast<device const Exact32*>(r.corr1)[ifp];
    const Exact32 fc11 = reinterpret_cast<device const Exact32*>(r.fac11)[i] * reinterpret_cast<device const Exact32*>(r.corr1)[ifp];
    const int lo2 = ((jp - 1) * 5 + (jt - 1)) * p.nspa2;
    const int lo21 = (jp * 5 + (jt1 - 1)) * p.nspa2;
    const int hi2 = ((jp - 13) * 5 + (jt - 1)) * p.nspb2;
    const int hi21 = ((jp - 12) * 5 + (jt1 - 1)) * p.nspb2;
    const int self2 = clamp(r.indself[i] - 1, 0, p.nself2 - 2);
    for (int g = 0; g < 14; ++g) {
    const Exact32 selfterm2 = reinterpret_cast<device const Exact32*>(r.selffac)[i] * (reinterpret_cast<device const Exact32*>(r.self2)[ulong(self2) * 14ul + ulong(g)]
        + reinterpret_cast<device const Exact32*>(r.selffrac)[i] * (reinterpret_cast<device const Exact32*>(r.self2)[ulong(self2 + 1) * 14ul + ulong(g)]
                           - reinterpret_cast<device const Exact32*>(r.self2)[ulong(self2) * 14ul + ulong(g)]));
        const Exact32 low = reinterpret_cast<device const Exact32*>(r.colh2o)[i] * (rrtm_lw_interp4(
            reinterpret_cast<device const Exact32*>(r.absa2), p.nabsa2, 14, g, lo2, lo21,
            fc00, fc01, fc10, fc11) + selfterm2 + reinterpret_cast<device const Exact32*>(r.forfac)[i] * reinterpret_cast<device const Exact32*>(r.forref2)[g]);
        const Exact32 high = reinterpret_cast<device const Exact32*>(r.colh2o)[i] * (rrtm_lw_interp4(
            reinterpret_cast<device const Exact32*>(r.absb2), p.nabsb2, 14, g, hi2, hi21,
            fc00, fc01, fc10, fc11) + reinterpret_cast<device const Exact32*>(r.forfac)[i] * reinterpret_cast<device const Exact32*>(r.forref2)[g]);
        const ulong out = i * 140ul + ulong(8 + g);
        const Exact32 f0 = reinterpret_cast<device const Exact32*>(r.fraca2)[ulong(ifrac - 1) * 14ul + ulong(g)];
        const Exact32 f1 = reinterpret_cast<device const Exact32*>(r.fraca2)[ulong(ifrac - 2) * 14ul + ulong(g)];
        reinterpret_cast<device Exact32*>(r.taug)[out] = trop ? low : high;
        reinterpret_cast<device Exact32*>(r.pfrac)[out] = trop ? f0 + fracint * (f1 - f0) : reinterpret_cast<device const Exact32*>(r.fracb2)[g];
    }
}
