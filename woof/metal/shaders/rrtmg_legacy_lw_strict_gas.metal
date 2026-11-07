#include <metal_stdlib>
using namespace metal;
#include "soft_binary32.metalh"
#include "soft_gfk_pow.metalh"
#pragma clang fp contract(off)
#define NGPTLW 140
#define MAXXSEC 4
#define RLW_AD(a,b) as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)))
#define RLW_SU(a,b) as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)))
#define RLW_MU(a,b) as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)))
#define RLW_DV(a,b) as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)))
#define rlw_pow(x,y) as_type<float>(soft_gfk_pow_bits(as_type<uint>(x),as_type<uint>(y)))
#define fmodf(x,y) as_type<float>(climate_fmod(as_type<uint>(x),as_type<uint>(y)))
static inline float rlw_pow4(float x){float x2=RLW_MU(x,x);return RLW_MU(x2,x2);}
static uint climate_fmod(uint x, uint y) {
    const uint sign=x & 0x80000000u;
    uint ax=x & 0x7fffffffu, ay=y & 0x7fffffffu;
    if (soft32_is_nan(x)) return soft32_quiet_nan(x);
    if (soft32_is_nan(y)) return soft32_quiet_nan(y);
    if (ax==0x7f800000u || ay==0u) return soft32_qnan();
    if (ay==0x7f800000u || ax<ay) return x;
    if (ax==ay) return sign;
    Soft32Decoded a=soft32_decode(x), b=soft32_decode(y);
    uint m=a.significand; int e=a.exponent;
    while(e>b.exponent) {
        if(m>=b.significand) m-=b.significand;
        if(m==0u) return sign;
        m <<= 1; --e;
    }
    if(m>=b.significand) m-=b.significand;
    if(m==0u) return sign;
    while(m<0x800000u) {m<<=1;--e;}
    return soft32_pack_extended(sign!=0u,e,m<<3);
}

#define SL_PAVEL 0
#define SL_COLDRY 1
#define SL_COLH2O 2
#define SL_COLCO2 3
#define SL_COLO3 4
#define SL_COLN2O 5
#define SL_COLCO 6
#define SL_COLCH4 7
#define SL_COLO2 8
#define SL_COLBRD 9
#define SL_FAC00 10
#define SL_FAC01 11
#define SL_FAC10 12
#define SL_FAC11 13
#define SL_RAT_H2OCO2 14
#define SL_RAT_H2OCO2_1 15
#define SL_RAT_H2OO3 16
#define SL_RAT_H2OO3_1 17
#define SL_RAT_H2ON2O 18
#define SL_RAT_H2ON2O_1 19
#define SL_RAT_H2OCH4 20
#define SL_RAT_H2OCH4_1 21
#define SL_RAT_N2OCO2 22
#define SL_RAT_N2OCO2_1 23
#define SL_RAT_O3CO2 24
#define SL_RAT_O3CO2_1 25
#define SL_SELFFAC 26
#define SL_SELFFRAC 27
#define SL_FORFAC 28
#define SL_FORFRAC 29
#define SL_MINORFRAC 30
#define SL_SCALEMINOR 31
#define SL_SCALEMINORN2 32
#define NFSLOT 33

#define SI_JP 0
#define SI_JT 1
#define SI_JT1 2
#define SI_INDSELF 3
#define SI_INDFOR 4
#define SI_INDMINOR 5
#define NISLOT 6

// 1-based lay; col/nl from enclosing kernel scope.
#define FS(slot) fs[((long)(slot) * ncol + col) * nl + ((lay) - 1)]
#define IS(slot) isv[((long)(slot) * ncol + col) * nl + ((lay) - 1)]
#define WXS(ix) wx[((long)col * MAXXSEC + ((ix) - 1)) * nl + ((lay) - 1)]
// Fortran-order 2-D coefficient table with leading dimension ld.
#define A2(tab, ld, i, g) tab[((i) - 1) + ((g) - 1) * (ld)]
// taug/fracs (ncol, nl, NGPTLW) C-order; ig1 is the 1-based GLOBAL g-point.
#define TAUG(ig1) taug[((long)col * nl + ((lay) - 1)) * NGPTLW + ((ig1) - 1)]
#define FRACS(ig1) fracs[((long)col * nl + ((lay) - 1)) * NGPTLW + ((ig1) - 1)]


#define CHI35(s, j) chi_mls[((s) - 1) + 7 * ((j) - 1)]
// 3-D minor tables, Fortran (9,19,ng) / (5,19,ng): element (j,i,g).
#define KA9X19(tab, j, i, g) \
    tab[((j) - 1) + 9 * ((i) - 1) + 171 * ((g) - 1)]
#define KB5X19(tab, j, i, g) \
    tab[((j) - 1) + 5 * ((i) - 1) + 95 * ((g) - 1)]

// speccomb/specparm/js/fs block (Fortran e.g. lines 5297-5302).
// mult = 8. below laytrop, 4. above.
struct RlwSpecB35 {
    float comb;   // speccomb
    float parm;   // specparm (clamped at oneminus)
    int j;        // js (1-based)
    float f;      // fs
};

static inline RlwSpecB35 rlw_b35_spec(float col1, float rat, float col2,
                                   float oneminus, float mult)
{
    RlwSpecB35 s;
    s.comb = RLW_AD(col1, RLW_MU(rat, col2));
    s.parm = RLW_DV(col1, s.comb);
    if (soft32_greater_equal(as_type<uint>(s.parm),as_type<uint>(oneminus))) s.parm = oneminus;
    float specmult = RLW_MU(mult, s.parm);
    s.j = 1 + (int)specmult;
    s.f = fmodf(specmult, 1.0f);
    return s;
}

// The specparm branch triplet factors (Fortran e.g. lines 5343-5372);
// facp/fact are fac00/fac10 for the ind0 half, fac01/fac11 for ind1.
// branch: 0 = specparm < 0.125, 1 = specparm > 0.875, 2 = else.
struct RlwFacB35 {
    int branch;
    float f0p, f1p, f2p, f0t, f1t, f2t;
};

static inline RlwFacB35 rlw_b35_facs(float specparm, float sf,
                                  float facp, float fact)
{
    RlwFacB35 r;
    if (soft32_less(as_type<uint>(specparm),as_type<uint>(0.125f))) {
        float p = RLW_SU(sf, 1.0f);
        float p4 = rlw_pow4(p);
        float fk0 = p4;
        float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
        float fk2 = RLW_AD(p, p4);
        r.branch = 0;
        r.f0p = RLW_MU(fk0, facp);
        r.f1p = RLW_MU(fk1, facp);
        r.f2p = RLW_MU(fk2, facp);
        r.f0t = RLW_MU(fk0, fact);
        r.f1t = RLW_MU(fk1, fact);
        r.f2t = RLW_MU(fk2, fact);
    } else if (soft32_greater(as_type<uint>(specparm),as_type<uint>(0.875f))) {
        float p = as_type<float>(as_type<uint>(sf)^0x80000000u);
        float p4 = rlw_pow4(p);
        float fk0 = p4;
        float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
        float fk2 = RLW_AD(p, p4);
        r.branch = 1;
        r.f0p = RLW_MU(fk0, facp);
        r.f1p = RLW_MU(fk1, facp);
        r.f2p = RLW_MU(fk2, facp);
        r.f0t = RLW_MU(fk0, fact);
        r.f1t = RLW_MU(fk1, fact);
        r.f2t = RLW_MU(fk2, fact);
    } else {
        r.branch = 2;
        r.f0p = RLW_MU(RLW_SU(1.0f, sf), facp);
        r.f0t = RLW_MU(RLW_SU(1.0f, sf), fact);
        r.f1p = RLW_MU(sf, facp);
        r.f1t = RLW_MU(sf, fact);
        r.f2p = 0.0f;
        r.f2t = 0.0f;
    }
    return r;
}

// tau_major for one g-point, absa ld 585 (Fortran e.g. lines 5415-5437).
static inline float rlw_b35_major(const RlwFacB35 f, device const float* absa,
                               int ind, int ig, float comb)
{
    float t;
    if (f.branch == 0) {
        t = RLW_MU(f.f0p, A2(absa, 585, ind, ig));
        t = RLW_AD(t, RLW_MU(f.f1p, A2(absa, 585, ind + 1, ig)));
        t = RLW_AD(t, RLW_MU(f.f2p, A2(absa, 585, ind + 2, ig)));
        t = RLW_AD(t, RLW_MU(f.f0t, A2(absa, 585, ind + 9, ig)));
        t = RLW_AD(t, RLW_MU(f.f1t, A2(absa, 585, ind + 10, ig)));
        t = RLW_AD(t, RLW_MU(f.f2t, A2(absa, 585, ind + 11, ig)));
    } else if (f.branch == 1) {
        t = RLW_MU(f.f2p, A2(absa, 585, ind - 1, ig));
        t = RLW_AD(t, RLW_MU(f.f1p, A2(absa, 585, ind, ig)));
        t = RLW_AD(t, RLW_MU(f.f0p, A2(absa, 585, ind + 1, ig)));
        t = RLW_AD(t, RLW_MU(f.f2t, A2(absa, 585, ind + 8, ig)));
        t = RLW_AD(t, RLW_MU(f.f1t, A2(absa, 585, ind + 9, ig)));
        t = RLW_AD(t, RLW_MU(f.f0t, A2(absa, 585, ind + 10, ig)));
    } else {
        t = RLW_MU(f.f0p, A2(absa, 585, ind, ig));
        t = RLW_AD(t, RLW_MU(f.f1p, A2(absa, 585, ind + 1, ig)));
        t = RLW_AD(t, RLW_MU(f.f0t, A2(absa, 585, ind + 9, ig)));
        t = RLW_AD(t, RLW_MU(f.f1t, A2(absa, 585, ind + 10, ig)));
    }
    return RLW_MU(comb, t);
}

// Upper-atmosphere 4-term absb sum, ld 1175 (Fortran e.g. 5536-5545):
// comb * (f0p*absb(ind) + f1p*absb(ind+1) + f0t*absb(ind+5)
//         + f1t*absb(ind+6)).
static inline float rlw_b35_bmajor(float f0p, float f1p, float f0t, float f1t,
                                device const float* absb, int ind, int ig,
                                float comb)
{
    float t = RLW_MU(f0p, A2(absb, 1175, ind, ig));
    t = RLW_AD(t, RLW_MU(f1p, A2(absb, 1175, ind + 1, ig)));
    t = RLW_AD(t, RLW_MU(f0t, A2(absb, 1175, ind + 5, ig)));
    t = RLW_AD(t, RLW_MU(f1t, A2(absb, 1175, ind + 6, ig)));
    return RLW_MU(comb, t);
}


#define KA3_0609(tab, j, i, g) \
    tab[((j) - 1) + 9 * ((i) - 1) + 171 * ((g) - 1)]

// specparm branch triplet shared by the 9-species bands (7 and 9).
// facp/fact are fac00/fac10 (ind0 half) or fac01/fac11 (ind1 half).
// Fortran e.g. lines 6278-6307 (band 7) / 6674-6703 (band 9).
struct RlwSpecFac0609 {
    float f0p, f1p, f2p, f0t, f1t, f2t;
};

static inline RlwSpecFac0609 rlw_specfac_0609(float specparm, float fsl,
                                           float facp, float fact)
{
    RlwSpecFac0609 r;
    if (soft32_less(as_type<uint>(specparm),as_type<uint>(0.125f))) {
        float p = RLW_SU(fsl, 1.0f);              // p = fs - 1
        float p4 = rlw_pow4(p);                   // p**4
        float fk0 = p4;
        float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
        float fk2 = RLW_AD(p, p4);
        r.f0p = RLW_MU(fk0, facp);
        r.f1p = RLW_MU(fk1, facp);
        r.f2p = RLW_MU(fk2, facp);
        r.f0t = RLW_MU(fk0, fact);
        r.f1t = RLW_MU(fk1, fact);
        r.f2t = RLW_MU(fk2, fact);
    } else if (soft32_greater(as_type<uint>(specparm),as_type<uint>(0.875f))) {
        float p = as_type<float>(as_type<uint>(fsl)^0x80000000u);                           // p = -fs
        float p4 = rlw_pow4(p);
        float fk0 = p4;
        float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
        float fk2 = RLW_AD(p, p4);
        r.f0p = RLW_MU(fk0, facp);
        r.f1p = RLW_MU(fk1, facp);
        r.f2p = RLW_MU(fk2, facp);
        r.f0t = RLW_MU(fk0, fact);
        r.f1t = RLW_MU(fk1, fact);
        r.f2t = RLW_MU(fk2, fact);
    } else {
        r.f0p = RLW_MU(RLW_SU(1.0f, fsl), facp);
        r.f0t = RLW_MU(RLW_SU(1.0f, fsl), fact);
        r.f1p = RLW_MU(fsl, facp);
        r.f1t = RLW_MU(fsl, fact);
        r.f2p = 0.0f;                             // unused in else branch
        r.f2t = 0.0f;
    }
    return r;
}

// tau_major sum for one g-point; absa leading dimension 585 (9-species).
// Term ORDER matches the Fortran statements (e.g. 6350-6372) exactly.
static inline float rlw_ninemajor_0609(float specparm, float speccomb,
                                    RlwSpecFac0609 f,
                                    device const float* absa, int ind, int ig)
{
    float s;
    if (soft32_less(as_type<uint>(specparm),as_type<uint>(0.125f))) {
        s = RLW_MU(f.f0p, A2(absa, 585, ind, ig));
        s = RLW_AD(s, RLW_MU(f.f1p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(f.f2p, A2(absa, 585, ind + 2, ig)));
        s = RLW_AD(s, RLW_MU(f.f0t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(f.f1t, A2(absa, 585, ind + 10, ig)));
        s = RLW_AD(s, RLW_MU(f.f2t, A2(absa, 585, ind + 11, ig)));
    } else if (soft32_greater(as_type<uint>(specparm),as_type<uint>(0.875f))) {
        s = RLW_MU(f.f2p, A2(absa, 585, ind - 1, ig));
        s = RLW_AD(s, RLW_MU(f.f1p, A2(absa, 585, ind, ig)));
        s = RLW_AD(s, RLW_MU(f.f0p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(f.f2t, A2(absa, 585, ind + 8, ig)));
        s = RLW_AD(s, RLW_MU(f.f1t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(f.f0t, A2(absa, 585, ind + 10, ig)));
    } else {
        s = RLW_MU(f.f0p, A2(absa, 585, ind, ig));
        s = RLW_AD(s, RLW_MU(f.f1p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(f.f0t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(f.f1t, A2(absa, 585, ind + 10, ig)));
    }
    return RLW_MU(speccomb, s);
}


#define RLW1316_A3(tab, j, i, g) \
    tab[((j) - 1) + 9 * ((i) - 1) + 171 * ((g) - 1)]

// The specparm branch triplet + tau_major sum shared by the 9-species
// bands in this file (13, 15, 16; absa leading dimension 585 for all).
// Mirrors _nine_major in gpuwm/core/rrtmg_lw.py; facp/fact are
// fac00/fac10 (ind0 half) or fac01/fac11 (ind1 half); ind/ig 1-based.
static inline float rlw_nine_major_b1316(float specparm, float fsv,
                                      float facp, float fact,
                                      device const float* absa, int ind, int ig,
                                      float speccomb)
{
    if (soft32_less(as_type<uint>(specparm),as_type<uint>(0.125f))) {
        float p = RLW_SU(fsv, 1.0f);
        float p4 = rlw_pow4(p);
        float fk0 = p4;
        float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
        float fk2 = RLW_AD(p, p4);
        float fac0p = RLW_MU(fk0, facp);
        float fac1p = RLW_MU(fk1, facp);
        float fac2p = RLW_MU(fk2, facp);
        float fac0t = RLW_MU(fk0, fact);
        float fac1t = RLW_MU(fk1, fact);
        float fac2t = RLW_MU(fk2, fact);
        float s = RLW_MU(fac0p, A2(absa, 585, ind, ig));
        s = RLW_AD(s, RLW_MU(fac1p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(fac2p, A2(absa, 585, ind + 2, ig)));
        s = RLW_AD(s, RLW_MU(fac0t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(fac1t, A2(absa, 585, ind + 10, ig)));
        s = RLW_AD(s, RLW_MU(fac2t, A2(absa, 585, ind + 11, ig)));
        return RLW_MU(speccomb, s);
    } else if (soft32_greater(as_type<uint>(specparm),as_type<uint>(0.875f))) {
        float p = as_type<float>(as_type<uint>(fsv)^0x80000000u);
        float p4 = rlw_pow4(p);
        float fk0 = p4;
        float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
        float fk2 = RLW_AD(p, p4);
        float fac0p = RLW_MU(fk0, facp);
        float fac1p = RLW_MU(fk1, facp);
        float fac2p = RLW_MU(fk2, facp);
        float fac0t = RLW_MU(fk0, fact);
        float fac1t = RLW_MU(fk1, fact);
        float fac2t = RLW_MU(fk2, fact);
        float s = RLW_MU(fac2p, A2(absa, 585, ind - 1, ig));
        s = RLW_AD(s, RLW_MU(fac1p, A2(absa, 585, ind, ig)));
        s = RLW_AD(s, RLW_MU(fac0p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(fac2t, A2(absa, 585, ind + 8, ig)));
        s = RLW_AD(s, RLW_MU(fac1t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(fac0t, A2(absa, 585, ind + 10, ig)));
        return RLW_MU(speccomb, s);
    } else {
        float fac0p = RLW_MU(RLW_SU(1.0f, fsv), facp);
        float fac0t = RLW_MU(RLW_SU(1.0f, fsv), fact);
        float fac1p = RLW_MU(fsv, facp);
        float fac1t = RLW_MU(fsv, fact);
        float s = RLW_MU(fac0p, A2(absa, 585, ind, ig));
        s = RLW_AD(s, RLW_MU(fac1p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(fac0t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(fac1t, A2(absa, 585, ind + 10, ig)));
        return RLW_MU(speccomb, s);
    }
}



#include "rrtmg_legacy_lw_strict_gas_bindings.metalh"

kernel void rrtmg_legacy_lw_strict_band1(constant RrtmgLegacyLwStrictBand1Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand1Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[1] = absa(65,ng), absb(235,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mn2(19,ng), kb_mn2(19,ng),
    //                    fracrefa(ng), fracrefb(ng)
    device const float* absa = r.absa;
    device const float* absb = r.absb;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* ka_mn2 = r.ka_mn2;
    device const float* kb_mn2 = r.kb_mn2;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa1 = 1, nspb1 = 1, ng1 = 10, gs = 0;

    if (lay <= laytrop) {
        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa1 + 1;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa1 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);
        float pp = FS(SL_PAVEL);
        float corradj = 1.0f;
        if (soft32_less(as_type<uint>(pp),as_type<uint>(250.0f)))
            corradj = RLW_SU(1.0f, RLW_DV(RLW_MU(0.15f, RLW_SU(250.0f, pp)),
                                          154.4f));
        float scalen2 = RLW_MU(FS(SL_COLBRD), FS(SL_SCALEMINORN2));
        for (int ig = 1; ig <= ng1; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float taun2 = RLW_MU(scalen2,
                RLW_AD(A2(ka_mn2, 19, indm, ig),
                       RLW_MU(FS(SL_MINORFRAC),
                              RLW_SU(A2(ka_mn2, 19, indm + 1, ig),
                                     A2(ka_mn2, 19, indm, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, taun2);
            TAUG(gs + ig) = RLW_MU(corradj, t);
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb1 + 1;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb1 + 1;
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);
        float pp = FS(SL_PAVEL);
        float corradj = RLW_SU(1.0f, RLW_MU(0.15f, RLW_DV(pp, 95.6f)));
        float scalen2 = RLW_MU(FS(SL_COLBRD), FS(SL_SCALEMINORN2));
        for (int ig = 1; ig <= ng1; ++ig) {
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float taun2 = RLW_MU(scalen2,
                RLW_AD(A2(kb_mn2, 19, indm, ig),
                       RLW_MU(FS(SL_MINORFRAC),
                              RLW_SU(A2(kb_mn2, 19, indm + 1, ig),
                                     A2(kb_mn2, 19, indm, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, taun2);
            TAUG(gs + ig) = RLW_MU(corradj, t);
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band2(constant RrtmgLegacyLwStrictBand2Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand2Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[2] = absa(65,ng), absb(235,ng), selfref(10,ng),
    //                    forref(4,ng), fracrefa(ng), fracrefb(ng)
    device const float* absa = r.absa;
    device const float* absb = r.absb;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa2 = 1, nspb2 = 1, ng2 = 12, gs = 10;

    if (lay <= laytrop) {
        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa2 + 1;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa2 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        float pp = FS(SL_PAVEL);
        // corradj = 1._rb - .05_rb * (pp - 100._rb) / 900._rb
        float corradj = RLW_SU(1.0f,
            RLW_DV(RLW_MU(0.05f, RLW_SU(pp, 100.0f)), 900.0f));
        for (int ig = 1; ig <= ng2; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = RLW_MU(corradj, t);
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb2 + 1;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb2 + 1;
        int indf = IS(SI_INDFOR);
        for (int ig = 1; ig <= ng2; ++ig) {
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band3(constant RrtmgLegacyLwStrictBand3Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand3Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[3] = absa(585,ng), absb(1175,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mn2o(9,19,ng),
    //                    kb_mn2o(5,19,ng), fracrefa(ng,9), fracrefb(ng,5)
    device const float* absa = r.absa;
    device const float* absb = r.absb;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* ka_mn2o = r.ka_mn2o;
    device const float* kb_mn2o = r.kb_mn2o;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa3 = 9, nspb3 = 5, ng3 = 16, gs = 22;

    // P = 212.725 mb                       chi_mls(1,9)/chi_mls(2,9)
    float refrat_planck_a = RLW_DV(CHI35(1, 9), CHI35(2, 9));
    // P = 95.58 mb                         chi_mls(1,13)/chi_mls(2,13)
    float refrat_planck_b = RLW_DV(CHI35(1, 13), CHI35(2, 13));
    // P = 706.270 mb                       chi_mls(1,3)/chi_mls(2,3)
    float refrat_m_a = RLW_DV(CHI35(1, 3), CHI35(2, 3));
    // P = 95.58 mb                         chi_mls(1,13)/chi_mls(2,13)
    float refrat_m_b = RLW_DV(CHI35(1, 13), CHI35(2, 13));

    float colh2o = FS(SL_COLH2O);
    float colco2 = FS(SL_COLCO2);
    float coln2o = FS(SL_COLN2O);
    float coldry = FS(SL_COLDRY);
    int jpv = IS(SI_JP);

    if (lay <= laytrop) {
        RlwSpecB35 s0 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2), colco2,
                                     oneminus, 8.0f);
        RlwSpecB35 s1 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2_1), colco2,
                                     oneminus, 8.0f);
        RlwSpecB35 smn = rlw_b35_spec(colh2o, refrat_m_a, colco2,
                                      oneminus, 8.0f);
        int jmn2o = smn.j;
        float fmn2o = smn.f;
        // fmn2omf = minorfrac*fmn2o is a dead statement in the Fortran
        // (never read anywhere in taugb3); omitted, as in the NumPy port.

        float chi_n2o = RLW_DV(coln2o, coldry);
        float ratn2o = RLW_DV(RLW_MU(1.e20f, chi_n2o), CHI35(4, jpv + 1));
        float adjcoln2o;
        if (soft32_greater(as_type<uint>(ratn2o),as_type<uint>(1.5f))) {
            float adjfac = RLW_AD(0.5f,
                rlw_pow(RLW_SU(ratn2o, 0.5f), 0.65f));
            adjcoln2o = RLW_MU(RLW_MU(RLW_MU(adjfac, CHI35(4, jpv + 1)),
                                      coldry), 1.e-20f);
        } else {
            adjcoln2o = coln2o;
        }

        RlwSpecB35 spl = rlw_b35_spec(colh2o, refrat_planck_a, colco2,
                                      oneminus, 8.0f);
        int jpl = spl.j;
        float fpl = spl.f;

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa3 + s0.j;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa3 + s1.j;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        RlwFacB35 f0 = rlw_b35_facs(s0.parm, s0.f, FS(SL_FAC00),
                                    FS(SL_FAC10));
        RlwFacB35 f1 = rlw_b35_facs(s1.parm, s1.f, FS(SL_FAC01),
                                    FS(SL_FAC11));

        for (int ig = 1; ig <= ng3; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float n2om1 = RLW_AD(KA9X19(ka_mn2o, jmn2o, indm, ig),
                RLW_MU(fmn2o,
                       RLW_SU(KA9X19(ka_mn2o, jmn2o + 1, indm, ig),
                              KA9X19(ka_mn2o, jmn2o, indm, ig))));
            float n2om2 = RLW_AD(KA9X19(ka_mn2o, jmn2o, indm + 1, ig),
                RLW_MU(fmn2o,
                       RLW_SU(KA9X19(ka_mn2o, jmn2o + 1, indm + 1, ig),
                              KA9X19(ka_mn2o, jmn2o, indm + 1, ig))));
            float absn2o = RLW_AD(n2om1, RLW_MU(FS(SL_MINORFRAC),
                                                RLW_SU(n2om2, n2om1)));

            float tau_major = rlw_b35_major(f0, absa, ind0, ig, s0.comb);
            float tau_major1 = rlw_b35_major(f1, absa, ind1, ig, s1.comb);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcoln2o, absn2o));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 16, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 16, ig, jpl + 1),
                                   A2(fracrefa, 16, ig, jpl))));
        }
    } else {
        RlwSpecB35 s0 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2), colco2,
                                     oneminus, 4.0f);
        RlwSpecB35 s1 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2_1), colco2,
                                     oneminus, 4.0f);

        float fac000 = RLW_MU(RLW_SU(1.0f, s0.f), FS(SL_FAC00));
        float fac010 = RLW_MU(RLW_SU(1.0f, s0.f), FS(SL_FAC10));
        float fac100 = RLW_MU(s0.f, FS(SL_FAC00));
        float fac110 = RLW_MU(s0.f, FS(SL_FAC10));
        float fac001 = RLW_MU(RLW_SU(1.0f, s1.f), FS(SL_FAC01));
        float fac011 = RLW_MU(RLW_SU(1.0f, s1.f), FS(SL_FAC11));
        float fac101 = RLW_MU(s1.f, FS(SL_FAC01));
        float fac111 = RLW_MU(s1.f, FS(SL_FAC11));

        RlwSpecB35 smn = rlw_b35_spec(colh2o, refrat_m_b, colco2,
                                      oneminus, 4.0f);
        int jmn2o = smn.j;
        float fmn2o = smn.f;
        // fmn2omf: dead statement in the Fortran here too; omitted.

        float chi_n2o = RLW_DV(coln2o, coldry);
        // Fortran line 5508 writes the literal as 1.e20 (default real,
        // same f32 value as the 1.e20_rb of the lower loop).
        float ratn2o = RLW_DV(RLW_MU(1.e20f, chi_n2o), CHI35(4, jpv + 1));
        float adjcoln2o;
        if (soft32_greater(as_type<uint>(ratn2o),as_type<uint>(1.5f))) {
            float adjfac = RLW_AD(0.5f,
                rlw_pow(RLW_SU(ratn2o, 0.5f), 0.65f));
            adjcoln2o = RLW_MU(RLW_MU(RLW_MU(adjfac, CHI35(4, jpv + 1)),
                                      coldry), 1.e-20f);
        } else {
            adjcoln2o = coln2o;
        }

        RlwSpecB35 spl = rlw_b35_spec(colh2o, refrat_planck_b, colco2,
                                      oneminus, 4.0f);
        int jpl = spl.j;
        float fpl = spl.f;

        int ind0 = ((jpv - 13) * 5 + (IS(SI_JT) - 1)) * nspb3 + s0.j;
        int ind1 = ((jpv - 12) * 5 + (IS(SI_JT1) - 1)) * nspb3 + s1.j;
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng3; ++ig) {
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float n2om1 = RLW_AD(KB5X19(kb_mn2o, jmn2o, indm, ig),
                RLW_MU(fmn2o,
                       RLW_SU(KB5X19(kb_mn2o, jmn2o + 1, indm, ig),
                              KB5X19(kb_mn2o, jmn2o, indm, ig))));
            float n2om2 = RLW_AD(KB5X19(kb_mn2o, jmn2o, indm + 1, ig),
                RLW_MU(fmn2o,
                       RLW_SU(KB5X19(kb_mn2o, jmn2o + 1, indm + 1, ig),
                              KB5X19(kb_mn2o, jmn2o, indm + 1, ig))));
            float absn2o = RLW_AD(n2om1, RLW_MU(FS(SL_MINORFRAC),
                                                RLW_SU(n2om2, n2om1)));

            float t = rlw_b35_bmajor(fac000, fac100, fac010, fac110,
                                     absb, ind0, ig, s0.comb);
            t = RLW_AD(t, rlw_b35_bmajor(fac001, fac101, fac011, fac111,
                                         absb, ind1, ig, s1.comb));
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcoln2o, absn2o));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefb, 16, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefb, 16, ig, jpl + 1),
                                   A2(fracrefb, 16, ig, jpl))));
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band4(constant RrtmgLegacyLwStrictBand4Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand4Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[4] = absa(585,ng), absb(1175,ng), selfref(10,ng),
    //                    forref(4,ng), fracrefa(ng,9), fracrefb(ng,5)
    device const float* absa = r.absa;
    device const float* absb = r.absb;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa4 = 9, nspb4 = 5, ng4 = 14, gs = 38;

    // P = 142.5940 mb                      chi_mls(1,11)/chi_mls(2,11)
    float refrat_planck_a = RLW_DV(CHI35(1, 11), CHI35(2, 11));
    // P = 95.58350 mb                      chi_mls(3,13)/chi_mls(2,13)
    float refrat_planck_b = RLW_DV(CHI35(3, 13), CHI35(2, 13));

    float colco2 = FS(SL_COLCO2);

    if (lay <= laytrop) {
        float colh2o = FS(SL_COLH2O);
        int jpv = IS(SI_JP);

        RlwSpecB35 s0 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2), colco2,
                                     oneminus, 8.0f);
        RlwSpecB35 s1 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2_1), colco2,
                                     oneminus, 8.0f);
        RlwSpecB35 spl = rlw_b35_spec(colh2o, refrat_planck_a, colco2,
                                      oneminus, 8.0f);
        int jpl = spl.j;
        float fpl = spl.f;

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa4 + s0.j;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa4 + s1.j;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);

        RlwFacB35 f0 = rlw_b35_facs(s0.parm, s0.f, FS(SL_FAC00),
                                    FS(SL_FAC10));
        RlwFacB35 f1 = rlw_b35_facs(s1.parm, s1.f, FS(SL_FAC01),
                                    FS(SL_FAC11));

        for (int ig = 1; ig <= ng4; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));

            float tau_major = rlw_b35_major(f0, absa, ind0, ig, s0.comb);
            float tau_major1 = rlw_b35_major(f1, absa, ind1, ig, s1.comb);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 14, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 14, ig, jpl + 1),
                                   A2(fracrefa, 14, ig, jpl))));
        }
    } else {
        float colo3 = FS(SL_COLO3);

        RlwSpecB35 s0 = rlw_b35_spec(colo3, FS(SL_RAT_O3CO2), colco2,
                                     oneminus, 4.0f);
        RlwSpecB35 s1 = rlw_b35_spec(colo3, FS(SL_RAT_O3CO2_1), colco2,
                                     oneminus, 4.0f);

        float fac000 = RLW_MU(RLW_SU(1.0f, s0.f), FS(SL_FAC00));
        float fac010 = RLW_MU(RLW_SU(1.0f, s0.f), FS(SL_FAC10));
        float fac100 = RLW_MU(s0.f, FS(SL_FAC00));
        float fac110 = RLW_MU(s0.f, FS(SL_FAC10));
        float fac001 = RLW_MU(RLW_SU(1.0f, s1.f), FS(SL_FAC01));
        float fac011 = RLW_MU(RLW_SU(1.0f, s1.f), FS(SL_FAC11));
        float fac101 = RLW_MU(s1.f, FS(SL_FAC01));
        float fac111 = RLW_MU(s1.f, FS(SL_FAC11));

        RlwSpecB35 spl = rlw_b35_spec(colo3, refrat_planck_b, colco2,
                                      oneminus, 4.0f);
        int jpl = spl.j;
        float fpl = spl.f;

        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb4 + s0.j;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb4 + s1.j;

        for (int ig = 1; ig <= ng4; ++ig) {
            float t = rlw_b35_bmajor(fac000, fac100, fac010, fac110,
                                     absb, ind0, ig, s0.comb);
            t = RLW_AD(t, rlw_b35_bmajor(fac001, fac101, fac011, fac111,
                                         absb, ind1, ig, s1.comb));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefb, 14, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefb, 14, ig, jpl + 1),
                                   A2(fracrefb, 14, ig, jpl))));
        }

        // Empirical modification to improve stratospheric cooling rates
        // for co2 (Fortran lines 5802-5808; literals are default real).
        TAUG(gs + 8) = RLW_MU(TAUG(gs + 8), 0.92f);
        TAUG(gs + 9) = RLW_MU(TAUG(gs + 9), 0.88f);
        TAUG(gs + 10) = RLW_MU(TAUG(gs + 10), 1.07f);
        TAUG(gs + 11) = RLW_MU(TAUG(gs + 11), 1.1f);
        TAUG(gs + 12) = RLW_MU(TAUG(gs + 12), 0.99f);
        TAUG(gs + 13) = RLW_MU(TAUG(gs + 13), 0.88f);
        TAUG(gs + 14) = RLW_MU(TAUG(gs + 14), 0.943f);
    }

}

kernel void rrtmg_legacy_lw_strict_band5(constant RrtmgLegacyLwStrictBand5Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand5Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[5] = absa(585,ng), absb(1175,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mo3(9,19,ng), ccl4(ng),
    //                    fracrefa(ng,9), fracrefb(ng,5)
    device const float* absa = r.absa;
    device const float* absb = r.absb;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* ka_mo3 = r.ka_mo3;
    device const float* ccl4 = r.ccl4;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa5 = 9, nspb5 = 5, ng5 = 16, gs = 52;

    // P = 473.420 mb                       chi_mls(1,5)/chi_mls(2,5)
    float refrat_planck_a = RLW_DV(CHI35(1, 5), CHI35(2, 5));
    // P = 0.2369 mb                        chi_mls(3,43)/chi_mls(2,43)
    float refrat_planck_b = RLW_DV(CHI35(3, 43), CHI35(2, 43));
    // P = 317.3480 mb                      chi_mls(1,7)/chi_mls(2,7)
    float refrat_m_a = RLW_DV(CHI35(1, 7), CHI35(2, 7));

    float colco2 = FS(SL_COLCO2);

    if (lay <= laytrop) {
        float colh2o = FS(SL_COLH2O);
        int jpv = IS(SI_JP);

        RlwSpecB35 s0 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2), colco2,
                                     oneminus, 8.0f);
        RlwSpecB35 s1 = rlw_b35_spec(colh2o, FS(SL_RAT_H2OCO2_1), colco2,
                                     oneminus, 8.0f);
        RlwSpecB35 smo = rlw_b35_spec(colh2o, refrat_m_a, colco2,
                                      oneminus, 8.0f);
        int jmo3 = smo.j;
        float fmo3 = smo.f;
        RlwSpecB35 spl = rlw_b35_spec(colh2o, refrat_planck_a, colco2,
                                      oneminus, 8.0f);
        int jpl = spl.j;
        float fpl = spl.f;

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa5 + s0.j;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa5 + s1.j;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        RlwFacB35 f0 = rlw_b35_facs(s0.parm, s0.f, FS(SL_FAC00),
                                    FS(SL_FAC10));
        RlwFacB35 f1 = rlw_b35_facs(s1.parm, s1.f, FS(SL_FAC01),
                                    FS(SL_FAC11));

        for (int ig = 1; ig <= ng5; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float o3m1 = RLW_AD(KA9X19(ka_mo3, jmo3, indm, ig),
                RLW_MU(fmo3,
                       RLW_SU(KA9X19(ka_mo3, jmo3 + 1, indm, ig),
                              KA9X19(ka_mo3, jmo3, indm, ig))));
            float o3m2 = RLW_AD(KA9X19(ka_mo3, jmo3, indm + 1, ig),
                RLW_MU(fmo3,
                       RLW_SU(KA9X19(ka_mo3, jmo3 + 1, indm + 1, ig),
                              KA9X19(ka_mo3, jmo3, indm + 1, ig))));
            float abso3 = RLW_AD(o3m1, RLW_MU(FS(SL_MINORFRAC),
                                              RLW_SU(o3m2, o3m1)));

            float tau_major = rlw_b35_major(f0, absa, ind0, ig, s0.comb);
            float tau_major1 = rlw_b35_major(f1, absa, ind1, ig, s1.comb);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(abso3, FS(SL_COLO3)));
            t = RLW_AD(t, RLW_MU(WXS(1), ccl4[ig - 1]));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 16, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 16, ig, jpl + 1),
                                   A2(fracrefa, 16, ig, jpl))));
        }
    } else {
        float colo3 = FS(SL_COLO3);

        RlwSpecB35 s0 = rlw_b35_spec(colo3, FS(SL_RAT_O3CO2), colco2,
                                     oneminus, 4.0f);
        RlwSpecB35 s1 = rlw_b35_spec(colo3, FS(SL_RAT_O3CO2_1), colco2,
                                     oneminus, 4.0f);

        float fac000 = RLW_MU(RLW_SU(1.0f, s0.f), FS(SL_FAC00));
        float fac010 = RLW_MU(RLW_SU(1.0f, s0.f), FS(SL_FAC10));
        float fac100 = RLW_MU(s0.f, FS(SL_FAC00));
        float fac110 = RLW_MU(s0.f, FS(SL_FAC10));
        float fac001 = RLW_MU(RLW_SU(1.0f, s1.f), FS(SL_FAC01));
        float fac011 = RLW_MU(RLW_SU(1.0f, s1.f), FS(SL_FAC11));
        float fac101 = RLW_MU(s1.f, FS(SL_FAC01));
        float fac111 = RLW_MU(s1.f, FS(SL_FAC11));

        RlwSpecB35 spl = rlw_b35_spec(colo3, refrat_planck_b, colco2,
                                      oneminus, 4.0f);
        int jpl = spl.j;
        float fpl = spl.f;

        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb5 + s0.j;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb5 + s1.j;

        for (int ig = 1; ig <= ng5; ++ig) {
            float t = rlw_b35_bmajor(fac000, fac100, fac010, fac110,
                                     absb, ind0, ig, s0.comb);
            t = RLW_AD(t, rlw_b35_bmajor(fac001, fac101, fac011, fac111,
                                         absb, ind1, ig, s1.comb));
            t = RLW_AD(t, RLW_MU(WXS(1), ccl4[ig - 1]));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefb, 16, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefb, 16, ig, jpl + 1),
                                   A2(fracrefb, 16, ig, jpl))));
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band6(constant RrtmgLegacyLwStrictBand6Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand6Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[6] = absa(65,ng), selfref(10,ng), forref(4,ng),
    //                    ka_mco2(19,ng), cfc11adj(ng), cfc12(ng),
    //                    fracrefa(ng)
    device const float* absa = r.absa;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* ka_mco2 = r.ka_mco2;
    device const float* cfc11adj = r.cfc11adj;
    device const float* cfc12 = r.cfc12;
    device const float* fracrefa = r.fracrefa;
    const int nspa6 = 1, ng6 = 8, gs = 68;

    if (lay <= laytrop) {
        int jpv = IS(SI_JP);
        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        // ratco2 = 1.e20_rb*chi_co2/chi_mls(2,jp(lay)+1)
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2),
                              chi_mls[1 + 7 * jpv]);
        float adjcolco2;
        if (soft32_greater(as_type<uint>(ratco2),as_type<uint>(3.0f))) {
            float adjfac = RLW_AD(2.0f,
                                  rlw_pow(RLW_SU(ratco2, 2.0f), 0.77f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[1 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa6 + 1;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa6 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng6; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float absco2 = RLW_AD(A2(ka_mco2, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(ka_mco2, 19, indm + 1, ig),
                              A2(ka_mco2, 19, indm, ig))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10),
                                       A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01),
                                       A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11),
                                       A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
            t = RLW_AD(t, RLW_MU(WXS(2), cfc11adj[ig - 1]));
            t = RLW_AD(t, RLW_MU(WXS(3), cfc12[ig - 1]));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        // Nothing important goes on above laytrop in this band.
        for (int ig = 1; ig <= ng6; ++ig) {
            float t = RLW_AD(0.0f, RLW_MU(WXS(2), cfc11adj[ig - 1]));
            t = RLW_AD(t, RLW_MU(WXS(3), cfc12[ig - 1]));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band7(constant RrtmgLegacyLwStrictBand7Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand7Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[7] = absa(585,ng), absb(235,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mco2(9,19,ng), kb_mco2(19,ng),
    //                    fracrefa(ng,9), fracrefb(ng)
    device const float* absa = r.absa;
    device const float* absb = r.absb;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* ka_mco2 = r.ka_mco2;
    device const float* kb_mco2 = r.kb_mco2;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa7 = 9, nspb7 = 1, ng7 = 12, gs = 76;

    if (lay <= laytrop) {
        // P = 706.2620 mb / 706.2720 mb (identical expressions in this
        // source): refrat_planck_a = refrat_m_a = chi_mls(1,3)/chi_mls(3,3)
        float refrat_planck_a = RLW_DV(chi_mls[0 + 7 * 2],
                                       chi_mls[2 + 7 * 2]);
        float refrat_m_a = RLW_DV(chi_mls[0 + 7 * 2], chi_mls[2 + 7 * 2]);

        int jpv = IS(SI_JP);
        float colh2o = FS(SL_COLH2O);
        float colo3 = FS(SL_COLO3);

        float speccomb = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2OO3), colo3));
        float specparm = RLW_DV(colh2o, speccomb);
        if (soft32_greater_equal(as_type<uint>(specparm),as_type<uint>(oneminus))) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fsv = fmodf(specmult, 1.0f);

        float speccomb1 = RLW_AD(colh2o,
                                 RLW_MU(FS(SL_RAT_H2OO3_1), colo3));
        float specparm1 = RLW_DV(colh2o, speccomb1);
        if (soft32_greater_equal(as_type<uint>(specparm1),as_type<uint>(oneminus))) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1v = fmodf(specmult1, 1.0f);

        float speccomb_mco2 = RLW_AD(colh2o, RLW_MU(refrat_m_a, colo3));
        float specparm_mco2 = RLW_DV(colh2o, speccomb_mco2);
        if (soft32_greater_equal(as_type<uint>(specparm_mco2),as_type<uint>(oneminus))) specparm_mco2 = oneminus;
        float specmult_mco2 = RLW_MU(8.0f, specparm_mco2);
        int jmco2 = 1 + (int)specmult_mco2;
        float fmco2 = fmodf(specmult_mco2, 1.0f);

        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        // ratco2 = 1.e20*chi_co2/chi_mls(2,jp(lay)+1)  (no _rb in source)
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2),
                              chi_mls[1 + 7 * jpv]);
        float adjcolco2;
        if (soft32_greater(as_type<uint>(ratco2),as_type<uint>(3.0f))) {
            float adjfac = RLW_AD(3.0f,
                                  rlw_pow(RLW_SU(ratco2, 3.0f), 0.79f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[1 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        float speccomb_planck = RLW_AD(colh2o,
                                       RLW_MU(refrat_planck_a, colo3));
        float specparm_planck = RLW_DV(colh2o, speccomb_planck);
        if (soft32_greater_equal(as_type<uint>(specparm_planck),as_type<uint>(oneminus))) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, 1.0f);

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa7 + js;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa7 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        RlwSpecFac0609 f0 = rlw_specfac_0609(specparm, fsv,
                                             FS(SL_FAC00), FS(SL_FAC10));
        RlwSpecFac0609 f1 = rlw_specfac_0609(specparm1, fs1v,
                                             FS(SL_FAC01), FS(SL_FAC11));

        for (int ig = 1; ig <= ng7; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float co2m1 = RLW_AD(KA3_0609(ka_mco2, jmco2, indm, ig),
                RLW_MU(fmco2,
                       RLW_SU(KA3_0609(ka_mco2, jmco2 + 1, indm, ig),
                              KA3_0609(ka_mco2, jmco2, indm, ig))));
            float co2m2 = RLW_AD(KA3_0609(ka_mco2, jmco2, indm + 1, ig),
                RLW_MU(fmco2,
                       RLW_SU(KA3_0609(ka_mco2, jmco2 + 1, indm + 1, ig),
                              KA3_0609(ka_mco2, jmco2, indm + 1, ig))));
            float absco2 = RLW_AD(co2m1, RLW_MU(FS(SL_MINORFRAC),
                                                RLW_SU(co2m2, co2m1)));

            float tau_major = rlw_ninemajor_0609(specparm, speccomb, f0,
                                                 absa, ind0, ig);
            float tau_major1 = rlw_ninemajor_0609(specparm1, speccomb1, f1,
                                                  absa, ind1, ig);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 12, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 12, ig, jpl + 1),
                                   A2(fracrefa, 12, ig, jpl))));
        }
    } else {
        int jpv = IS(SI_JP);
        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2),
                              chi_mls[1 + 7 * jpv]);
        float adjcolco2;
        if (soft32_greater(as_type<uint>(ratco2),as_type<uint>(3.0f))) {
            float adjfac = RLW_AD(2.0f,
                                  rlw_pow(RLW_SU(ratco2, 2.0f), 0.79f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[1 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        int ind0 = ((jpv - 13) * 5 + (IS(SI_JT) - 1)) * nspb7 + 1;
        int ind1 = ((jpv - 12) * 5 + (IS(SI_JT1) - 1)) * nspb7 + 1;
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng7; ++ig) {
            float absco2 = RLW_AD(A2(kb_mco2, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(kb_mco2, 19, indm + 1, ig),
                              A2(kb_mco2, 19, indm, ig))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10),
                                       A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01),
                                       A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11),
                                       A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLO3), tmaj);
            t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefb[ig - 1];
        }

        // Empirical modification to improve stratospheric cooling rates
        // for o3 (reduced g-points 6..11 of this band).
        TAUG(gs + 6) = RLW_MU(TAUG(gs + 6), 0.92f);
        TAUG(gs + 7) = RLW_MU(TAUG(gs + 7), 0.88f);
        TAUG(gs + 8) = RLW_MU(TAUG(gs + 8), 1.07f);
        TAUG(gs + 9) = RLW_MU(TAUG(gs + 9), 1.1f);
        TAUG(gs + 10) = RLW_MU(TAUG(gs + 10), 0.99f);
        TAUG(gs + 11) = RLW_MU(TAUG(gs + 11), 0.855f);
    }

}

kernel void rrtmg_legacy_lw_strict_band8(constant RrtmgLegacyLwStrictBand8Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand8Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[8] = absa(65,ng), absb(235,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mco2(19,ng), kb_mco2(19,ng),
    //                    ka_mn2o(19,ng), kb_mn2o(19,ng), ka_mo3(19,ng),
    //                    cfc12(ng), cfc22adj(ng), fracrefa(ng),
    //                    fracrefb(ng)
    device const float* absa = r.absa;
    device const float* absb = r.absb;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* ka_mco2 = r.ka_mco2;
    device const float* kb_mco2 = r.kb_mco2;
    device const float* ka_mn2o = r.ka_mn2o;
    device const float* kb_mn2o = r.kb_mn2o;
    device const float* ka_mo3 = r.ka_mo3;
    device const float* cfc12 = r.cfc12;
    device const float* cfc22adj = r.cfc22adj;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa8 = 1, nspb8 = 1, ng8 = 8, gs = 88;

    if (lay <= laytrop) {
        int jpv = IS(SI_JP);
        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2),
                              chi_mls[1 + 7 * jpv]);
        float adjcolco2;
        if (soft32_greater(as_type<uint>(ratco2),as_type<uint>(3.0f))) {
            float adjfac = RLW_AD(2.0f,
                                  rlw_pow(RLW_SU(ratco2, 2.0f), 0.65f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[1 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa8 + 1;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa8 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng8; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float absco2 = RLW_AD(A2(ka_mco2, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(ka_mco2, 19, indm + 1, ig),
                              A2(ka_mco2, 19, indm, ig))));
            float abso3 = RLW_AD(A2(ka_mo3, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(ka_mo3, 19, indm + 1, ig),
                              A2(ka_mo3, 19, indm, ig))));
            float absn2o = RLW_AD(A2(ka_mn2o, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(ka_mn2o, 19, indm + 1, ig),
                              A2(ka_mn2o, 19, indm, ig))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10),
                                       A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01),
                                       A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11),
                                       A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
            t = RLW_AD(t, RLW_MU(FS(SL_COLO3), abso3));
            t = RLW_AD(t, RLW_MU(FS(SL_COLN2O), absn2o));
            t = RLW_AD(t, RLW_MU(WXS(3), cfc12[ig - 1]));
            t = RLW_AD(t, RLW_MU(WXS(4), cfc22adj[ig - 1]));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        int jpv = IS(SI_JP);
        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2),
                              chi_mls[1 + 7 * jpv]);
        float adjcolco2;
        if (soft32_greater(as_type<uint>(ratco2),as_type<uint>(3.0f))) {
            float adjfac = RLW_AD(2.0f,
                                  rlw_pow(RLW_SU(ratco2, 2.0f), 0.65f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[1 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        int ind0 = ((jpv - 13) * 5 + (IS(SI_JT) - 1)) * nspb8 + 1;
        int ind1 = ((jpv - 12) * 5 + (IS(SI_JT1) - 1)) * nspb8 + 1;
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng8; ++ig) {
            float absco2 = RLW_AD(A2(kb_mco2, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(kb_mco2, 19, indm + 1, ig),
                              A2(kb_mco2, 19, indm, ig))));
            float absn2o = RLW_AD(A2(kb_mn2o, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(kb_mn2o, 19, indm + 1, ig),
                              A2(kb_mn2o, 19, indm, ig))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10),
                                       A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01),
                                       A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11),
                                       A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLO3), tmaj);
            t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
            t = RLW_AD(t, RLW_MU(FS(SL_COLN2O), absn2o));
            t = RLW_AD(t, RLW_MU(WXS(3), cfc12[ig - 1]));
            t = RLW_AD(t, RLW_MU(WXS(4), cfc22adj[ig - 1]));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band9(constant RrtmgLegacyLwStrictBand9Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand9Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[9] = absa(585,ng), absb(235,ng), selfref(10,ng),
    //                    forref(4,ng), ka_mn2o(9,19,ng), kb_mn2o(19,ng),
    //                    fracrefa(ng,9), fracrefb(ng)
    device const float* absa = r.absa;
    device const float* absb = r.absb;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* ka_mn2o = r.ka_mn2o;
    device const float* kb_mn2o = r.kb_mn2o;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa9 = 9, nspb9 = 1, ng9 = 12, gs = 96;

    if (lay <= laytrop) {
        // P = 212 mb: refrat_planck_a = chi_mls(1,9)/chi_mls(6,9)
        float refrat_planck_a = RLW_DV(chi_mls[0 + 7 * 8],
                                       chi_mls[5 + 7 * 8]);
        // P = 706.272 mb: refrat_m_a = chi_mls(1,3)/chi_mls(6,3)
        float refrat_m_a = RLW_DV(chi_mls[0 + 7 * 2], chi_mls[5 + 7 * 2]);

        int jpv = IS(SI_JP);
        float colh2o = FS(SL_COLH2O);
        float colch4 = FS(SL_COLCH4);

        float speccomb = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2OCH4), colch4));
        float specparm = RLW_DV(colh2o, speccomb);
        if (soft32_greater_equal(as_type<uint>(specparm),as_type<uint>(oneminus))) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fsv = fmodf(specmult, 1.0f);

        float speccomb1 = RLW_AD(colh2o,
                                 RLW_MU(FS(SL_RAT_H2OCH4_1), colch4));
        float specparm1 = RLW_DV(colh2o, speccomb1);
        if (soft32_greater_equal(as_type<uint>(specparm1),as_type<uint>(oneminus))) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1v = fmodf(specmult1, 1.0f);

        float speccomb_mn2o = RLW_AD(colh2o, RLW_MU(refrat_m_a, colch4));
        float specparm_mn2o = RLW_DV(colh2o, speccomb_mn2o);
        if (soft32_greater_equal(as_type<uint>(specparm_mn2o),as_type<uint>(oneminus))) specparm_mn2o = oneminus;
        float specmult_mn2o = RLW_MU(8.0f, specparm_mn2o);
        int jmn2o = 1 + (int)specmult_mn2o;
        float fmn2o = fmodf(specmult_mn2o, 1.0f);

        float chi_n2o = RLW_DV(FS(SL_COLN2O), FS(SL_COLDRY));
        // ratn2o = 1.e20_rb*chi_n2o/chi_mls(4,jp(lay)+1)
        float ratn2o = RLW_DV(RLW_MU(1.e20f, chi_n2o),
                              chi_mls[3 + 7 * jpv]);
        float adjcoln2o;
        if (soft32_greater(as_type<uint>(ratn2o),as_type<uint>(1.5f))) {
            float adjfac = RLW_AD(0.5f,
                                  rlw_pow(RLW_SU(ratn2o, 0.5f), 0.65f));
            adjcoln2o = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[3 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcoln2o = FS(SL_COLN2O);
        }

        float speccomb_planck = RLW_AD(colh2o,
                                       RLW_MU(refrat_planck_a, colch4));
        float specparm_planck = RLW_DV(colh2o, speccomb_planck);
        if (soft32_greater_equal(as_type<uint>(specparm_planck),as_type<uint>(oneminus))) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, 1.0f);

        int ind0 = ((jpv - 1) * 5 + (IS(SI_JT) - 1)) * nspa9 + js;
        int ind1 = (jpv * 5 + (IS(SI_JT1) - 1)) * nspa9 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        RlwSpecFac0609 f0 = rlw_specfac_0609(specparm, fsv,
                                             FS(SL_FAC00), FS(SL_FAC10));
        RlwSpecFac0609 f1 = rlw_specfac_0609(specparm1, fs1v,
                                             FS(SL_FAC01), FS(SL_FAC11));

        for (int ig = 1; ig <= ng9; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float n2om1 = RLW_AD(KA3_0609(ka_mn2o, jmn2o, indm, ig),
                RLW_MU(fmn2o,
                       RLW_SU(KA3_0609(ka_mn2o, jmn2o + 1, indm, ig),
                              KA3_0609(ka_mn2o, jmn2o, indm, ig))));
            float n2om2 = RLW_AD(KA3_0609(ka_mn2o, jmn2o, indm + 1, ig),
                RLW_MU(fmn2o,
                       RLW_SU(KA3_0609(ka_mn2o, jmn2o + 1, indm + 1, ig),
                              KA3_0609(ka_mn2o, jmn2o, indm + 1, ig))));
            float absn2o = RLW_AD(n2om1, RLW_MU(FS(SL_MINORFRAC),
                                                RLW_SU(n2om2, n2om1)));

            float tau_major = rlw_ninemajor_0609(specparm, speccomb, f0,
                                                 absa, ind0, ig);
            float tau_major1 = rlw_ninemajor_0609(specparm1, speccomb1, f1,
                                                  absa, ind1, ig);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcoln2o, absn2o));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 12, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 12, ig, jpl + 1),
                                   A2(fracrefa, 12, ig, jpl))));
        }
    } else {
        int jpv = IS(SI_JP);
        float chi_n2o = RLW_DV(FS(SL_COLN2O), FS(SL_COLDRY));
        float ratn2o = RLW_DV(RLW_MU(1.e20f, chi_n2o),
                              chi_mls[3 + 7 * jpv]);
        float adjcoln2o;
        if (soft32_greater(as_type<uint>(ratn2o),as_type<uint>(1.5f))) {
            float adjfac = RLW_AD(0.5f,
                                  rlw_pow(RLW_SU(ratn2o, 0.5f), 0.65f));
            adjcoln2o = RLW_MU(RLW_MU(RLW_MU(adjfac, chi_mls[3 + 7 * jpv]),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcoln2o = FS(SL_COLN2O);
        }

        int ind0 = ((jpv - 13) * 5 + (IS(SI_JT) - 1)) * nspb9 + 1;
        int ind1 = ((jpv - 12) * 5 + (IS(SI_JT1) - 1)) * nspb9 + 1;
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng9; ++ig) {
            float absn2o = RLW_AD(A2(kb_mn2o, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(kb_mn2o, 19, indm + 1, ig),
                              A2(kb_mn2o, 19, indm, ig))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10),
                                       A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01),
                                       A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11),
                                       A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLCH4), tmaj);
            t = RLW_AD(t, RLW_MU(adjcoln2o, absn2o));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band10(constant RrtmgLegacyLwStrictBand10Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand10Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[10] = absa(65,ng), absb(235,ng), selfref(10,ng),
    //                     forref(4,ng), fracrefa(ng), fracrefb(ng)
    device const float* absa = r.absa;
    device const float* absb = r.absb;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa10 = 1, nspb10 = 1, ng10 = 6, gs = 108;

    if (lay <= laytrop) {
        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa10 + 1;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa10 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        for (int ig = 1; ig <= ng10; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb10 + 1;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb10 + 1;
        int indf = IS(SI_INDFOR);
        for (int ig = 1; ig <= ng10; ++ig) {
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band11(constant RrtmgLegacyLwStrictBand11Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand11Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[11] = absa(65,ng), absb(235,ng), selfref(10,ng),
    //                     forref(4,ng), ka_mo2(19,ng), kb_mo2(19,ng),
    //                     fracrefa(ng), fracrefb(ng)
    device const float* absa = r.absa;
    device const float* absb = r.absb;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* ka_mo2 = r.ka_mo2;
    device const float* kb_mo2 = r.kb_mo2;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa11 = 1, nspb11 = 1, ng11 = 8, gs = 114;

    if (lay <= laytrop) {
        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa11 + 1;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa11 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);
        float scaleo2 = RLW_MU(FS(SL_COLO2), FS(SL_SCALEMINOR));
        for (int ig = 1; ig <= ng11; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float tauo2 = RLW_MU(scaleo2,
                RLW_AD(A2(ka_mo2, 19, indm, ig),
                       RLW_MU(FS(SL_MINORFRAC),
                              RLW_SU(A2(ka_mo2, 19, indm + 1, ig),
                                     A2(ka_mo2, 19, indm, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, tauo2);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb11 + 1;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb11 + 1;
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);
        float scaleo2 = RLW_MU(FS(SL_COLO2), FS(SL_SCALEMINOR));
        for (int ig = 1; ig <= ng11; ++ig) {
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float tauo2 = RLW_MU(scaleo2,
                RLW_AD(A2(kb_mo2, 19, indm, ig),
                       RLW_MU(FS(SL_MINORFRAC),
                              RLW_SU(A2(kb_mo2, 19, indm + 1, ig),
                                     A2(kb_mo2, 19, indm, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC10), A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC01), A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj, RLW_MU(FS(SL_FAC11), A2(absb, 235, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLH2O), tmaj);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, tauo2);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band12(constant RrtmgLegacyLwStrictBand12Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand12Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[12] = absa(585,ng), selfref(10,ng), forref(4,ng),
    //                     fracrefa(ng12,9)
    device const float* absa = r.absa;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* fracrefa = r.fracrefa;
    const int nspa12 = 9, ng12 = 8, gs = 122;

    if (lay <= laytrop) {
        // refrat_planck_a = chi_mls(1,10)/chi_mls(2,10)   (P = 174.164 mb)
        float refrat_planck_a = RLW_DV(chi_mls[0 + 7 * 9],
                                       chi_mls[1 + 7 * 9]);
        float colh2o = FS(SL_COLH2O);
        float colco2 = FS(SL_COLCO2);

        float speccomb = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2OCO2), colco2));
        float specparm = RLW_DV(colh2o, speccomb);
        if (soft32_greater_equal(as_type<uint>(specparm),as_type<uint>(oneminus))) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fs0 = fmodf(specmult, 1.0f);

        float speccomb1 = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2OCO2_1), colco2));
        float specparm1 = RLW_DV(colh2o, speccomb1);
        if (soft32_greater_equal(as_type<uint>(specparm1),as_type<uint>(oneminus))) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1 = fmodf(specmult1, 1.0f);

        float speccomb_planck = RLW_AD(colh2o,
                                       RLW_MU(refrat_planck_a, colco2));
        float specparm_planck = RLW_DV(colh2o, speccomb_planck);
        if (soft32_greater_equal(as_type<uint>(specparm_planck),as_type<uint>(oneminus))) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, 1.0f);

        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa12 + js;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa12 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);

        float fac000 = 0.0f, fac100 = 0.0f, fac200 = 0.0f;
        float fac010 = 0.0f, fac110 = 0.0f, fac210 = 0.0f;
        if (soft32_less(as_type<uint>(specparm),as_type<uint>(0.125f))) {
            float p = RLW_SU(fs0, 1.0f);
            float p4 = rlw_pow4(p);
            float fk0 = p4;
            float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
            float fk2 = RLW_AD(p, p4);
            fac000 = RLW_MU(fk0, FS(SL_FAC00));
            fac100 = RLW_MU(fk1, FS(SL_FAC00));
            fac200 = RLW_MU(fk2, FS(SL_FAC00));
            fac010 = RLW_MU(fk0, FS(SL_FAC10));
            fac110 = RLW_MU(fk1, FS(SL_FAC10));
            fac210 = RLW_MU(fk2, FS(SL_FAC10));
        } else if (soft32_greater(as_type<uint>(specparm),as_type<uint>(0.875f))) {
            float p = as_type<float>(as_type<uint>(fs0)^0x80000000u);
            float p4 = rlw_pow4(p);
            float fk0 = p4;
            float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
            float fk2 = RLW_AD(p, p4);
            fac000 = RLW_MU(fk0, FS(SL_FAC00));
            fac100 = RLW_MU(fk1, FS(SL_FAC00));
            fac200 = RLW_MU(fk2, FS(SL_FAC00));
            fac010 = RLW_MU(fk0, FS(SL_FAC10));
            fac110 = RLW_MU(fk1, FS(SL_FAC10));
            fac210 = RLW_MU(fk2, FS(SL_FAC10));
        } else {
            fac000 = RLW_MU(RLW_SU(1.0f, fs0), FS(SL_FAC00));
            fac010 = RLW_MU(RLW_SU(1.0f, fs0), FS(SL_FAC10));
            fac100 = RLW_MU(fs0, FS(SL_FAC00));
            fac110 = RLW_MU(fs0, FS(SL_FAC10));
        }

        float fac001 = 0.0f, fac101 = 0.0f, fac201 = 0.0f;
        float fac011 = 0.0f, fac111 = 0.0f, fac211 = 0.0f;
        if (soft32_less(as_type<uint>(specparm1),as_type<uint>(0.125f))) {
            float p = RLW_SU(fs1, 1.0f);
            float p4 = rlw_pow4(p);
            float fk0 = p4;
            float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
            float fk2 = RLW_AD(p, p4);
            fac001 = RLW_MU(fk0, FS(SL_FAC01));
            fac101 = RLW_MU(fk1, FS(SL_FAC01));
            fac201 = RLW_MU(fk2, FS(SL_FAC01));
            fac011 = RLW_MU(fk0, FS(SL_FAC11));
            fac111 = RLW_MU(fk1, FS(SL_FAC11));
            fac211 = RLW_MU(fk2, FS(SL_FAC11));
        } else if (soft32_greater(as_type<uint>(specparm1),as_type<uint>(0.875f))) {
            float p = as_type<float>(as_type<uint>(fs1)^0x80000000u);
            float p4 = rlw_pow4(p);
            float fk0 = p4;
            float fk1 = RLW_SU(RLW_SU(1.0f, p), RLW_MU(2.0f, p4));
            float fk2 = RLW_AD(p, p4);
            fac001 = RLW_MU(fk0, FS(SL_FAC01));
            fac101 = RLW_MU(fk1, FS(SL_FAC01));
            fac201 = RLW_MU(fk2, FS(SL_FAC01));
            fac011 = RLW_MU(fk0, FS(SL_FAC11));
            fac111 = RLW_MU(fk1, FS(SL_FAC11));
            fac211 = RLW_MU(fk2, FS(SL_FAC11));
        } else {
            fac001 = RLW_MU(RLW_SU(1.0f, fs1), FS(SL_FAC01));
            fac011 = RLW_MU(RLW_SU(1.0f, fs1), FS(SL_FAC11));
            fac101 = RLW_MU(fs1, FS(SL_FAC01));
            fac111 = RLW_MU(fs1, FS(SL_FAC11));
        }

        for (int ig = 1; ig <= ng12; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));

            float tau_major;
            if (soft32_less(as_type<uint>(specparm),as_type<uint>(0.125f))) {
                float t = RLW_MU(fac000, A2(absa, 585, ind0, ig));
                t = RLW_AD(t, RLW_MU(fac100, A2(absa, 585, ind0 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac200, A2(absa, 585, ind0 + 2, ig)));
                t = RLW_AD(t, RLW_MU(fac010, A2(absa, 585, ind0 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac110, A2(absa, 585, ind0 + 10, ig)));
                t = RLW_AD(t, RLW_MU(fac210, A2(absa, 585, ind0 + 11, ig)));
                tau_major = RLW_MU(speccomb, t);
            } else if (soft32_greater(as_type<uint>(specparm),as_type<uint>(0.875f))) {
                float t = RLW_MU(fac200, A2(absa, 585, ind0 - 1, ig));
                t = RLW_AD(t, RLW_MU(fac100, A2(absa, 585, ind0, ig)));
                t = RLW_AD(t, RLW_MU(fac000, A2(absa, 585, ind0 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac210, A2(absa, 585, ind0 + 8, ig)));
                t = RLW_AD(t, RLW_MU(fac110, A2(absa, 585, ind0 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac010, A2(absa, 585, ind0 + 10, ig)));
                tau_major = RLW_MU(speccomb, t);
            } else {
                float t = RLW_MU(fac000, A2(absa, 585, ind0, ig));
                t = RLW_AD(t, RLW_MU(fac100, A2(absa, 585, ind0 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac010, A2(absa, 585, ind0 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac110, A2(absa, 585, ind0 + 10, ig)));
                tau_major = RLW_MU(speccomb, t);
            }

            float tau_major1;
            if (soft32_less(as_type<uint>(specparm1),as_type<uint>(0.125f))) {
                float t = RLW_MU(fac001, A2(absa, 585, ind1, ig));
                t = RLW_AD(t, RLW_MU(fac101, A2(absa, 585, ind1 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac201, A2(absa, 585, ind1 + 2, ig)));
                t = RLW_AD(t, RLW_MU(fac011, A2(absa, 585, ind1 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac111, A2(absa, 585, ind1 + 10, ig)));
                t = RLW_AD(t, RLW_MU(fac211, A2(absa, 585, ind1 + 11, ig)));
                tau_major1 = RLW_MU(speccomb1, t);
            } else if (soft32_greater(as_type<uint>(specparm1),as_type<uint>(0.875f))) {
                float t = RLW_MU(fac201, A2(absa, 585, ind1 - 1, ig));
                t = RLW_AD(t, RLW_MU(fac101, A2(absa, 585, ind1, ig)));
                t = RLW_AD(t, RLW_MU(fac001, A2(absa, 585, ind1 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac211, A2(absa, 585, ind1 + 8, ig)));
                t = RLW_AD(t, RLW_MU(fac111, A2(absa, 585, ind1 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac011, A2(absa, 585, ind1 + 10, ig)));
                tau_major1 = RLW_MU(speccomb1, t);
            } else {
                float t = RLW_MU(fac001, A2(absa, 585, ind1, ig));
                t = RLW_AD(t, RLW_MU(fac101, A2(absa, 585, ind1 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac011, A2(absa, 585, ind1 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac111, A2(absa, 585, ind1 + 10, ig)));
                tau_major1 = RLW_MU(speccomb1, t);
            }

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            // fracs = fracrefa(ig,jpl) + fpl*(fracrefa(ig,jpl+1)-fracrefa(ig,jpl))
            float fr0 = A2(fracrefa, 8, ig, jpl);
            float fr1 = A2(fracrefa, 8, ig, jpl + 1);
            FRACS(gs + ig) = RLW_AD(fr0, RLW_MU(fpl, RLW_SU(fr1, fr0)));
        }
    } else {
        // Upper atmosphere loop: taug = 0.0_rb ; fracs = 0.0_rb, literally.
        for (int ig = 1; ig <= ng12; ++ig) {
            TAUG(gs + ig) = 0.0f;
            FRACS(gs + ig) = 0.0f;
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band13(constant RrtmgLegacyLwStrictBand13Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand13Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[13] = absa(585,ng), selfref(10,ng), forref(4,ng),
    //                     ka_mco2(9,19,ng), ka_mco(9,19,ng),
    //                     kb_mo3(19,ng), fracrefa(ng,9), fracrefb(ng)
    device const float* absa = r.absa;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* ka_mco2 = r.ka_mco2;
    device const float* ka_mco = r.ka_mco;
    device const float* kb_mo3 = r.kb_mo3;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa13 = 9, ng13 = 4, gs = 130;

    if (lay <= laytrop) {
        // P = 473.420 mb (Level 5): chi_mls(1,5)/chi_mls(4,5)
        float refrat_planck_a = RLW_DV(chi_mls[0 + 7 * 4],
                                       chi_mls[3 + 7 * 4]);
        // P = 1053. (Level 1): chi_mls(1,1)/chi_mls(4,1)
        float refrat_m_a = RLW_DV(chi_mls[0 + 7 * 0], chi_mls[3 + 7 * 0]);
        // P = 706. (Level 3): chi_mls(1,3)/chi_mls(4,3)
        float refrat_m_a3 = RLW_DV(chi_mls[0 + 7 * 2], chi_mls[3 + 7 * 2]);

        float colh2o = FS(SL_COLH2O);
        float coln2o = FS(SL_COLN2O);

        float speccomb = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2ON2O), coln2o));
        float specparm = RLW_DV(colh2o, speccomb);
        if (soft32_greater_equal(as_type<uint>(specparm),as_type<uint>(oneminus))) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fsv = fmodf(specmult, 1.0f);

        float speccomb1 = RLW_AD(colh2o,
                                 RLW_MU(FS(SL_RAT_H2ON2O_1), coln2o));
        float specparm1 = RLW_DV(colh2o, speccomb1);
        if (soft32_greater_equal(as_type<uint>(specparm1),as_type<uint>(oneminus))) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1 = fmodf(specmult1, 1.0f);

        float speccomb_mco2 = RLW_AD(colh2o, RLW_MU(refrat_m_a, coln2o));
        float specparm_mco2 = RLW_DV(colh2o, speccomb_mco2);
        if (soft32_greater_equal(as_type<uint>(specparm_mco2),as_type<uint>(oneminus))) specparm_mco2 = oneminus;
        float specmult_mco2 = RLW_MU(8.0f, specparm_mco2);
        int jmco2 = 1 + (int)specmult_mco2;
        float fmco2 = fmodf(specmult_mco2, 1.0f);

        // CO2-too-major empirical adjustment (FIXED 3.55e-4 reference;
        // real exponent 0.68 -> glibc powf transcription).
        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2), 3.55e-4f);
        float adjcolco2;
        if (soft32_greater(as_type<uint>(ratco2),as_type<uint>(3.0f))) {
            float adjfac = RLW_AD(2.0f,
                                  rlw_pow(RLW_SU(ratco2, 2.0f), 0.68f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, 3.55e-4f),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        float speccomb_mco = RLW_AD(colh2o, RLW_MU(refrat_m_a3, coln2o));
        float specparm_mco = RLW_DV(colh2o, speccomb_mco);
        if (soft32_greater_equal(as_type<uint>(specparm_mco),as_type<uint>(oneminus))) specparm_mco = oneminus;
        float specmult_mco = RLW_MU(8.0f, specparm_mco);
        int jmco = 1 + (int)specmult_mco;
        float fmco = fmodf(specmult_mco, 1.0f);

        float speccomb_planck = RLW_AD(colh2o,
                                       RLW_MU(refrat_planck_a, coln2o));
        float specparm_planck = RLW_DV(colh2o, speccomb_planck);
        if (soft32_greater_equal(as_type<uint>(specparm_planck),as_type<uint>(oneminus))) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, 1.0f);

        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa13 + js;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa13 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        for (int ig = 1; ig <= ng13; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float co2m1 = RLW_AD(RLW1316_A3(ka_mco2, jmco2, indm, ig),
                RLW_MU(fmco2,
                       RLW_SU(RLW1316_A3(ka_mco2, jmco2 + 1, indm, ig),
                              RLW1316_A3(ka_mco2, jmco2, indm, ig))));
            float co2m2 = RLW_AD(RLW1316_A3(ka_mco2, jmco2, indm + 1, ig),
                RLW_MU(fmco2,
                       RLW_SU(RLW1316_A3(ka_mco2, jmco2 + 1, indm + 1, ig),
                              RLW1316_A3(ka_mco2, jmco2, indm + 1, ig))));
            float absco2 = RLW_AD(co2m1, RLW_MU(FS(SL_MINORFRAC),
                                                RLW_SU(co2m2, co2m1)));
            float com1 = RLW_AD(RLW1316_A3(ka_mco, jmco, indm, ig),
                RLW_MU(fmco,
                       RLW_SU(RLW1316_A3(ka_mco, jmco + 1, indm, ig),
                              RLW1316_A3(ka_mco, jmco, indm, ig))));
            float com2 = RLW_AD(RLW1316_A3(ka_mco, jmco, indm + 1, ig),
                RLW_MU(fmco,
                       RLW_SU(RLW1316_A3(ka_mco, jmco + 1, indm + 1, ig),
                              RLW1316_A3(ka_mco, jmco, indm + 1, ig))));
            float absco = RLW_AD(com1, RLW_MU(FS(SL_MINORFRAC),
                                              RLW_SU(com2, com1)));

            float tau_major = rlw_nine_major_b1316(
                specparm, fsv, FS(SL_FAC00), FS(SL_FAC10),
                absa, ind0, ig, speccomb);
            float tau_major1 = rlw_nine_major_b1316(
                specparm1, fs1, FS(SL_FAC01), FS(SL_FAC11),
                absa, ind1, ig, speccomb1);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, RLW_MU(adjcolco2, absco2));
            t = RLW_AD(t, RLW_MU(FS(SL_COLCO), absco));
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 4, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 4, ig, jpl + 1),
                                   A2(fracrefa, 4, ig, jpl))));
        }
    } else {
        // Upper atmosphere: o3 minor ONLY (no foreign continuum in
        // this WRF copy).
        int indm = IS(SI_INDMINOR);
        for (int ig = 1; ig <= ng13; ++ig) {
            float abso3 = RLW_AD(A2(kb_mo3, 19, indm, ig),
                RLW_MU(FS(SL_MINORFRAC),
                       RLW_SU(A2(kb_mo3, 19, indm + 1, ig),
                              A2(kb_mo3, 19, indm, ig))));
            TAUG(gs + ig) = RLW_MU(FS(SL_COLO3), abso3);
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band14(constant RrtmgLegacyLwStrictBand14Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand14Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[14] = absa(65,ng), absb(235,ng), selfref(10,ng),
    //                     forref(4,ng), fracrefa(ng), fracrefb(ng)
    device const float* absa = r.absa;
    device const float* absb = r.absb;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa14 = 1, nspb14 = 1, ng14 = 2, gs = 134;

    if (lay <= laytrop) {
        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa14 + 1;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa14 + 1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        for (int ig = 1; ig <= ng14; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absa, 65, ind0, ig));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC10), A2(absa, 65, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC01), A2(absa, 65, ind1, ig)));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC11), A2(absa, 65, ind1 + 1, ig)));
            float t = RLW_MU(FS(SL_COLCO2), tmaj);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = fracrefa[ig - 1];
        }
    } else {
        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb14 + 1;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb14 + 1;
        for (int ig = 1; ig <= ng14; ++ig) {
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC10), A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC01), A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC11), A2(absb, 235, ind1 + 1, ig)));
            TAUG(gs + ig) = RLW_MU(FS(SL_COLCO2), tmaj);
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band15(constant RrtmgLegacyLwStrictBand15Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand15Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[15] = absa(585,ng), selfref(10,ng), forref(4,ng),
    //                     ka_mn2(9,19,ng), fracrefa(ng,9)
    device const float* absa = r.absa;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* ka_mn2 = r.ka_mn2;
    device const float* fracrefa = r.fracrefa;
    const int nspa15 = 9, ng15 = 2, gs = 136;

    if (lay <= laytrop) {
        // P = 1053. mb (Level 1): chi_mls(4,1)/chi_mls(2,1) -- Planck
        // and minor use the same ratio.
        float refrat_planck_a = RLW_DV(chi_mls[3 + 7 * 0],
                                       chi_mls[1 + 7 * 0]);
        float refrat_m_a = RLW_DV(chi_mls[3 + 7 * 0], chi_mls[1 + 7 * 0]);

        float coln2o = FS(SL_COLN2O);
        float colco2 = FS(SL_COLCO2);

        float speccomb = RLW_AD(coln2o, RLW_MU(FS(SL_RAT_N2OCO2), colco2));
        float specparm = RLW_DV(coln2o, speccomb);
        if (soft32_greater_equal(as_type<uint>(specparm),as_type<uint>(oneminus))) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fsv = fmodf(specmult, 1.0f);

        float speccomb1 = RLW_AD(coln2o,
                                 RLW_MU(FS(SL_RAT_N2OCO2_1), colco2));
        float specparm1 = RLW_DV(coln2o, speccomb1);
        if (soft32_greater_equal(as_type<uint>(specparm1),as_type<uint>(oneminus))) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1 = fmodf(specmult1, 1.0f);

        float speccomb_mn2 = RLW_AD(coln2o, RLW_MU(refrat_m_a, colco2));
        float specparm_mn2 = RLW_DV(coln2o, speccomb_mn2);
        if (soft32_greater_equal(as_type<uint>(specparm_mn2),as_type<uint>(oneminus))) specparm_mn2 = oneminus;
        float specmult_mn2 = RLW_MU(8.0f, specparm_mn2);
        int jmn2 = 1 + (int)specmult_mn2;
        float fmn2 = fmodf(specmult_mn2, 1.0f);

        float speccomb_planck = RLW_AD(coln2o,
                                       RLW_MU(refrat_planck_a, colco2));
        float specparm_planck = RLW_DV(coln2o, speccomb_planck);
        if (soft32_greater_equal(as_type<uint>(specparm_planck),as_type<uint>(oneminus))) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, 1.0f);

        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa15 + js;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa15 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);
        int indm = IS(SI_INDMINOR);

        // scalen2 = colbrd*scaleminor -- NOT scaleminorn2 (band 15 is
        // the one lower-minor band that scales by plain scaleminor).
        float scalen2 = RLW_MU(FS(SL_COLBRD), FS(SL_SCALEMINOR));

        for (int ig = 1; ig <= ng15; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));
            float n2m1 = RLW_AD(RLW1316_A3(ka_mn2, jmn2, indm, ig),
                RLW_MU(fmn2,
                       RLW_SU(RLW1316_A3(ka_mn2, jmn2 + 1, indm, ig),
                              RLW1316_A3(ka_mn2, jmn2, indm, ig))));
            float n2m2 = RLW_AD(RLW1316_A3(ka_mn2, jmn2, indm + 1, ig),
                RLW_MU(fmn2,
                       RLW_SU(RLW1316_A3(ka_mn2, jmn2 + 1, indm + 1, ig),
                              RLW1316_A3(ka_mn2, jmn2, indm + 1, ig))));
            float taun2 = RLW_MU(scalen2,
                RLW_AD(n2m1, RLW_MU(FS(SL_MINORFRAC),
                                    RLW_SU(n2m2, n2m1))));

            float tau_major = rlw_nine_major_b1316(
                specparm, fsv, FS(SL_FAC00), FS(SL_FAC10),
                absa, ind0, ig, speccomb);
            float tau_major1 = rlw_nine_major_b1316(
                specparm1, fs1, FS(SL_FAC01), FS(SL_FAC11),
                absa, ind1, ig, speccomb1);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            t = RLW_AD(t, taun2);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 2, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 2, ig, jpl + 1),
                                   A2(fracrefa, 2, ig, jpl))));
        }
    } else {
        // Upper atmosphere: nothing -- taug and fracs BOTH zeroed.
        for (int ig = 1; ig <= ng15; ++ig) {
            TAUG(gs + ig) = 0.0f;
            FRACS(gs + ig) = 0.0f;
        }
    }

}

kernel void rrtmg_legacy_lw_strict_band16(constant RrtmgLegacyLwStrictBand16Resources &r [[buffer(0)]], constant RrtmgLegacyLwStrictBand16Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid); if(tid>=long(ncol)*nl)return;
 int col=int(tid/nl),lay=int(tid%nl)+1,laytrop=laytrop_v[col];
    // GPU_BAND_TABS[16] = absa(585,ng), absb(235,ng), selfref(10,ng),
    //                     forref(4,ng), fracrefa(ng,9), fracrefb(ng)
    device const float* absa = r.absa;
    device const float* absb = r.absb;
    device const float* selfref = r.selfref;
    device const float* forref = r.forref;
    device const float* fracrefa = r.fracrefa;
    device const float* fracrefb = r.fracrefb;
    const int nspa16 = 9, nspb16 = 0, ng16 = 2, gs = 138;

    if (lay <= laytrop) {
        // P = 387. mb (Level 6): chi_mls(1,6)/chi_mls(6,6)
        float refrat_planck_a = RLW_DV(chi_mls[0 + 7 * 5],
                                       chi_mls[5 + 7 * 5]);

        float colh2o = FS(SL_COLH2O);
        float colch4 = FS(SL_COLCH4);

        float speccomb = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2OCH4), colch4));
        float specparm = RLW_DV(colh2o, speccomb);
        if (soft32_greater_equal(as_type<uint>(specparm),as_type<uint>(oneminus))) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fsv = fmodf(specmult, 1.0f);

        float speccomb1 = RLW_AD(colh2o,
                                 RLW_MU(FS(SL_RAT_H2OCH4_1), colch4));
        float specparm1 = RLW_DV(colh2o, speccomb1);
        if (soft32_greater_equal(as_type<uint>(specparm1),as_type<uint>(oneminus))) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1 = fmodf(specmult1, 1.0f);

        float speccomb_planck = RLW_AD(colh2o,
                                       RLW_MU(refrat_planck_a, colch4));
        float specparm_planck = RLW_DV(colh2o, speccomb_planck);
        if (soft32_greater_equal(as_type<uint>(specparm_planck),as_type<uint>(oneminus))) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = fmodf(specmult_planck, 1.0f);

        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa16 + js;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa16 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);

        for (int ig = 1; ig <= ng16; ++ig) {
            float tauself = RLW_MU(FS(SL_SELFFAC),
                RLW_AD(A2(selfref, 10, inds, ig),
                       RLW_MU(FS(SL_SELFFRAC),
                              RLW_SU(A2(selfref, 10, inds + 1, ig),
                                     A2(selfref, 10, inds, ig)))));
            float taufor = RLW_MU(FS(SL_FORFAC),
                RLW_AD(A2(forref, 4, indf, ig),
                       RLW_MU(FS(SL_FORFRAC),
                              RLW_SU(A2(forref, 4, indf + 1, ig),
                                     A2(forref, 4, indf, ig)))));

            float tau_major = rlw_nine_major_b1316(
                specparm, fsv, FS(SL_FAC00), FS(SL_FAC10),
                absa, ind0, ig, speccomb);
            float tau_major1 = rlw_nine_major_b1316(
                specparm1, fs1, FS(SL_FAC01), FS(SL_FAC11),
                absa, ind1, ig, speccomb1);

            float t = RLW_AD(tau_major, tau_major1);
            t = RLW_AD(t, tauself);
            t = RLW_AD(t, taufor);
            TAUG(gs + ig) = t;
            FRACS(gs + ig) = RLW_AD(A2(fracrefa, 2, ig, jpl),
                RLW_MU(fpl, RLW_SU(A2(fracrefa, 2, ig, jpl + 1),
                                   A2(fracrefa, 2, ig, jpl))));
        }
    } else {
        // nspb(16) = 0: both indices collapse to 1 -> absb rows 1 and 2.
        int ind0 = ((IS(SI_JP) - 13) * 5 + (IS(SI_JT) - 1)) * nspb16 + 1;
        int ind1 = ((IS(SI_JP) - 12) * 5 + (IS(SI_JT1) - 1)) * nspb16 + 1;
        for (int ig = 1; ig <= ng16; ++ig) {
            float tmaj = RLW_MU(FS(SL_FAC00), A2(absb, 235, ind0, ig));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC10), A2(absb, 235, ind0 + 1, ig)));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC01), A2(absb, 235, ind1, ig)));
            tmaj = RLW_AD(tmaj,
                          RLW_MU(FS(SL_FAC11), A2(absb, 235, ind1 + 1, ig)));
            TAUG(gs + ig) = RLW_MU(FS(SL_COLCH4), tmaj);
            FRACS(gs + ig) = fracrefb[ig - 1];
        }
    }

}
