#include <metal_stdlib>
#include "hexv842_mm5_prepare_bindings.metalh"
#include "hexv842_mm5_blend_bindings.metalh"
#include "soft_word32.metalh"
using namespace metal;
static inline F operator/(F a,F b) { return aa_div_impl(a,b); }
static inline F mm5_max(F a,F b) {
    if (soft32_is_nan(a.bits)) return b;
    if (soft32_is_nan(b.bits)) return a;
    if (((a.bits|b.bits)&0x7fffffffu)==0) return f32_word(a.bits&b.bits);
    return a>b?a:b;
}
kernel void hexv842_mm5_prepare(constant Hexv842Mm5PrepareResources &r [[buffer(0)]],
                                 constant Hexv842Mm5PrepareParams &p [[buffer(1)]],
                                 uint i [[thread_position_in_grid]]) {
    if (i>=uint(p.n)) return;
    int n=p.n;
    r.a[46*n+i]=float(F(p.len_disp)/gfk_pow(F(r.density[i]),F(0.25f)));
    r.a[18*n+i]=0.0f; r.a[17*n+i]=0.0f; r.a[16*n+i]=0.0f;
    r.a[14*n+i]=0.0f; r.a[15*n+i]=0.0f;
    r.a[32*n+i]=0.0f; r.a[33*n+i]=0.0f; r.a[30*n+i]=0.0f; r.a[31*n+i]=0.0f;
    r.a[45*n+i]=0.0f; r.a[47*n+i]=0.0f;
    if (!p.frac) return;
    for (int j=0;j<48;j++) r.sea[j*n+i]=r.a[j*n+i];
    r.sea[6*n+i]=0.0f;
    F ice=F(r.ice[i]);
    if (ice>=F(p.threshold) && ice<=F(1.0f)) {
        r.sea[44*n+i]=2.0f; r.sea[43*n+i]=1.0f; r.sea[0*n+i]=0.0001f;
        F sea_t=mm5_max(F(r.sst[i]),F(271.4f));
        r.sea[41*n+i]=float(sea_t);
        if (p.rp3) {
            F blend=F(r.a[41*n+i]);
            F ice_t=mm5_max((blend-(F(1.0f)-ice)*sea_t)/ice,F(221.4f));
            if (ice<F(0.2f) && blend<F(253.15f)) ice_t=F(253.15f);
            if (ice<F(0.1f) && blend<F(263.15f)) ice_t=F(263.15f);
            r.a[41*n+i]=float(ice_t);
        }
    }
}
kernel void hexv842_mm5_blend(constant Hexv842Mm5BlendResources &r [[buffer(0)]],
                               constant Hexv842Mm5BlendParams &p [[buffer(1)]],
                               uint i [[thread_position_in_grid]]) {
    if (i>=uint(p.n)) return;
    int n=p.n;
    F ice=F(r.ice[i]);
    if (!(ice>=F(p.threshold) && ice<=F(1.0f))) return;
    F ocean=F(1.0f)-ice;
    const int fields[]={27,22,23,28,3,10,9,25,1,26,7,18,17,16,14,15,32,33,30,31,2,12,11};
    for (uint j=0;j<23;j++) {
        int k=fields[j]*n+i;
        r.a[k]=float(F(r.a[k])*ice+ocean*F(r.sea[k]));
    }
    if (ice>=F(0.5f)) r.a[8*n+i]=r.regime_hold[i];
}
