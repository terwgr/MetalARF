#include <metal_stdlib>
#include "wrf_exact_cldfra1_bindings.metalh"
#include "exact_gfk_math.metalh"
using namespace metal;

static inline bool cl_less(Exact32 a, Exact32 b) { return soft32_less(a.bits,b.bits); }
static inline Exact32 cl_max(Exact32 a, Exact32 b) {
    // NumPy/CuPy maximum propagates a NaN, unlike fmax.
    if (soft32_is_nan(a.bits)) return a;
    if (soft32_is_nan(b.bits)) return b;
    return cl_less(a,b) ? b : a;
}

kernel void wrf_exact_cldfra1(
    constant WrfExactCldfra1Resources &r [[buffer(0)]],
    constant WrfExactCldfra1Params &p [[buffer(1)]],
    uint cell [[thread_position_in_grid]]) {
    if (cell >= uint(p.ncell)) return;
    device const Exact32 *qv = reinterpret_cast<device const Exact32 *>(r.qv);
    device const Exact32 *qc = reinterpret_cast<device const Exact32 *>(r.qc);
    device const Exact32 *qi = reinterpret_cast<device const Exact32 *>(r.qi);
    device const Exact32 *qs = reinterpret_cast<device const Exact32 *>(r.qs);
    device const Exact32 *tlay = reinterpret_cast<device const Exact32 *>(r.tlay);
    device const Exact32 *play = reinterpret_cast<device const Exact32 *>(r.play);
    device Exact32 *cldfra = reinterpret_cast<device Exact32 *>(r.cldfra);
    const Exact32 zero=0.0f, one=1.0f, qcldmin=1.0e-12f, svpt0=273.15f;
    const Exact32 t=tlay[cell], tc=t-svpt0;
    const Exact32 esw=(Exact32(1000.0f)*Exact32(0.61078f))*exact_gfk_exp(
        (Exact32(17.2693882f)*tc)/(t-Exact32(35.86f)));
    const Exact32 esi=(Exact32(1000.0f)*Exact32(0.61078f))*exact_gfk_exp(
        (Exact32(21.8745584f)*tc)/(t-Exact32(7.66f)));
    const Exact32 ep2=Exact32(287.0f)/Exact32(461.6f);
    const Exact32 qvsw=(ep2*esw)/(play[cell]-esw);
    const Exact32 qvsi=(ep2*esi)/(play[cell]-esi);
    Exact32 qcld, weight;
    if (p.f_qi && p.f_qs) {
        qcld=(qi[cell]+qc[cell])+qs[cell];
        weight=cl_less(qcld,qcldmin) ? zero : (qi[cell]+qs[cell])/cl_max(qcld,qcldmin);
    } else if (p.f_qi) {
        qcld=qi[cell]+qc[cell];
        weight=cl_less(qcld,qcldmin) ? zero : qi[cell]/cl_max(qcld,qcldmin);
        if (p.ferrier && !cl_less(qcld,qcldmin) && cl_less(tc,Exact32(-40.0f))) weight=one;
    } else {
        qcld=qc[cell];
        weight=cl_less(qcld,qcldmin) ? zero : (t>svpt0 ? zero : one);
    }
    const Exact32 qvs_weight=(one-weight)*qvsw+weight*qvsi;
    const Exact32 rhum=qv[cell]/qvs_weight;
    const Exact32 subsat=cl_max(Exact32(1.0e-10f),qvs_weight-qv[cell]);
    const Exact32 arg=cl_max(Exact32(-6.9f),(Exact32(-100.0f)*qcld)/exact_gfk_pow(subsat,Exact32(0.49f)));
    const Exact32 fraction=exact_gfk_pow(cl_max(Exact32(1.0e-10f),rhum),Exact32(0.25f))*(one-exact_gfk_exp(arg));
    const Exact32 clipped=cl_less(fraction,Exact32(0.01f)) ? zero : fraction;
    cldfra[cell]=cl_less(qcld,qcldmin) ? zero : (rhum>=one ? one : clipped);
}
