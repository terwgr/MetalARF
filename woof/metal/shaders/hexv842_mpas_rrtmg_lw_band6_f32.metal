#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "soft_gfk_pow.metalh"
#pragma clang fp contract(off)
#define NGPTLW 140
#define MAXXSEC 4
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#define rlw_pow(x,y) as_type<float>(soft_gfk_pow_bits(as_type<uint>(x),as_type<uint>(y)))
#define fmodf(x,y) as_type<float>(climate_fmod(as_type<uint>(x),as_type<uint>(y)))
static inline float rlw_pow4(float x){float x2=mpas_mul(x,x);return mpas_mul(x2,x2);}
#include "hexv842_mpas_rrtmg_lw_band6_f32_bindings.metalh"
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
    if (specparm < 0.125f) {
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
    } else if (specparm > 0.875f) {
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
    if (specparm < 0.125f) {
        s = RLW_MU(f.f0p, A2(absa, 585, ind, ig));
        s = RLW_AD(s, RLW_MU(f.f1p, A2(absa, 585, ind + 1, ig)));
        s = RLW_AD(s, RLW_MU(f.f2p, A2(absa, 585, ind + 2, ig)));
        s = RLW_AD(s, RLW_MU(f.f0t, A2(absa, 585, ind + 9, ig)));
        s = RLW_AD(s, RLW_MU(f.f1t, A2(absa, 585, ind + 10, ig)));
        s = RLW_AD(s, RLW_MU(f.f2t, A2(absa, 585, ind + 11, ig)));
    } else if (specparm > 0.875f) {
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


kernel void hexv842_mpas_rrtmg_lw_band6_f32(constant Hexv842MpasRrtmgLwBand6F32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwBand6F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid);if(tid>=long(ncol)*nl)return;
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
        if (ratco2 > 3.0f) {
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
