// Generated from classic RRTM SETCOEF; do not edit.
#include <metal_stdlib>
#include "exact_gfk_math.metalh"
#include "wrf_exact_rrtm_lw_setcoef_bindings.metalh"
using namespace metal;
static inline bool operator==(Exact32 a, Exact32 b) { return soft32_equal(a.bits,b.bits); }
static inline int rrtm_trunc(Exact32 x) { return int(trunc(as_type<float>(x.bits))); }
kernel void wrf_exact_rrtm_lw_setcoef(
    constant WrfExactRrtmLwSetcoefResources &r [[buffer(0)]],
    constant WrfExactRrtmLwSetcoefParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong base = ulong(col) * ulong(p.nlay);
    int laytrop = 0, layswtch = 0, laylow = 0;
    for (int lay = 0; lay < p.nlay; ++lay) {
        const ulong i = base + ulong(lay);
        const Exact32 plog = exact_gfk_log(reinterpret_cast<device const Exact32*>(r.pavel)[i]);
        const int jp = clamp(rrtm_trunc(36.0f - 5.0f * (plog + 0.04f)), 1, 58);
        const Exact32 fp = 5.0f * (reinterpret_cast<device const Exact32*>(r.preflog)[jp - 1] - plog);
        const Exact32 tdiff = reinterpret_cast<device const Exact32*>(r.tavel)[i] - reinterpret_cast<device const Exact32*>(r.tref)[jp - 1];
        const int jt = clamp(rrtm_trunc(3.0f + tdiff / 15.0f), 1, 4);
        const Exact32 ft = tdiff / 15.0f - Exact32(float(jt - 3));
        const Exact32 tdiff1 = reinterpret_cast<device const Exact32*>(r.tavel)[i] - reinterpret_cast<device const Exact32*>(r.tref)[jp];
        const int jt1 = clamp(rrtm_trunc(3.0f + tdiff1 / 15.0f), 1, 4);
        const Exact32 ft1 = tdiff1 / 15.0f - Exact32(float(jt1 - 3));
        const Exact32 water = reinterpret_cast<device const Exact32*>(r.wkl)[i * 35ul] / reinterpret_cast<device const Exact32*>(r.coldry)[i];
        const Exact32 scalefac = reinterpret_cast<device const Exact32*>(r.pavel)[i] * (Exact32(296.0f) / Exact32(1013.0f)) / reinterpret_cast<device const Exact32*>(r.tavel)[i];
        const Exact32 forfac = scalefac / (1.0f + water);
        const Exact32 selffac = water * forfac;
        const Exact32 factor = (reinterpret_cast<device const Exact32*>(r.tavel)[i] - 188.0f) / 7.2f;
        const int indself = clamp(rrtm_trunc(factor) - 7, 1, 9);
        const Exact32 selffrac = factor - Exact32(float(indself + 7));
        r.jp[i] = jp; r.jt[i] = jt; r.jt1[i] = jt1;
        r.indself[i] = indself;
        reinterpret_cast<device Exact32*>(r.fac00)[i] = (1.0f - fp) * (1.0f - ft);
        reinterpret_cast<device Exact32*>(r.fac01)[i] = fp * (1.0f - ft1);
        reinterpret_cast<device Exact32*>(r.fac10)[i] = (1.0f - fp) * ft;
        reinterpret_cast<device Exact32*>(r.fac11)[i] = fp * ft1;
        reinterpret_cast<device Exact32*>(r.forfac)[i] = forfac;
        reinterpret_cast<device Exact32*>(r.selffac)[i] = selffac;
        reinterpret_cast<device Exact32*>(r.selffrac)[i] = selffrac;
        if (plog > 4.56f) ++laytrop;
        if (plog > 5.76f) ++layswtch;
        if (plog >= 6.62f) ++laylow;
    }
    if (laylow == 0) laylow = 1;
    const int probe = clamp(layswtch, 0, p.nlay - 1);
    if (r.jp[base + ulong(probe)] <= 6) ++layswtch;
    r.laytrop[col] = laytrop;
    r.layswtch[col] = layswtch;
    r.laylow[col] = laylow;
    for (int lay = 0; lay < p.nlay; ++lay) {
        const ulong i = base + ulong(lay);
        const Exact32 coldry = reinterpret_cast<device const Exact32*>(r.coldry)[i];
        const Exact32 floor = 1.0e-32f * coldry;
        reinterpret_cast<device Exact32*>(r.colh2o)[i] = 1.0e-20f * reinterpret_cast<device const Exact32*>(r.wkl)[i * 35ul];
        reinterpret_cast<device Exact32*>(r.colco2)[i] = 1.0e-20f * reinterpret_cast<device const Exact32*>(r.wkl)[i * 35ul + 1ul];
        reinterpret_cast<device Exact32*>(r.colo3)[i] = 1.0e-20f * reinterpret_cast<device const Exact32*>(r.wkl)[i * 35ul + 2ul];
        reinterpret_cast<device Exact32*>(r.coln2o)[i] = 1.0e-20f * reinterpret_cast<device const Exact32*>(r.wkl)[i * 35ul + 3ul];
        reinterpret_cast<device Exact32*>(r.colch4)[i] = 1.0e-20f * reinterpret_cast<device const Exact32*>(r.wkl)[i * 35ul + 5ul];
        reinterpret_cast<device Exact32*>(r.colo2)[i] = 1.0e-20f * reinterpret_cast<device const Exact32*>(r.wkl)[i * 35ul + 6ul];
        if (reinterpret_cast<device Exact32*>(r.colco2)[i] == 0.0f) reinterpret_cast<device Exact32*>(r.colco2)[i] = floor;
        if (reinterpret_cast<device Exact32*>(r.coln2o)[i] == 0.0f) reinterpret_cast<device Exact32*>(r.coln2o)[i] = floor;
        if (reinterpret_cast<device Exact32*>(r.colch4)[i] == 0.0f) reinterpret_cast<device Exact32*>(r.colch4)[i] = floor;
        const Exact32 co2reg = 3.55e-24f * coldry;
        reinterpret_cast<device Exact32*>(r.co2mult)[i] = ((reinterpret_cast<device Exact32*>(r.colco2)[i] - co2reg) * 272.63f
            * exact_gfk_exp(-1919.4f / reinterpret_cast<device const Exact32*>(r.tavel)[i]))
            / (8.7604e-4f * reinterpret_cast<device const Exact32*>(r.tavel)[i]);
    }
}
