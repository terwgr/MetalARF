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
#include "hexv842_mpas_rrtmg_lw_band3_f32_bindings.metalh"
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
    if (s.parm >= oneminus) s.parm = oneminus;
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
    if (specparm < 0.125f) {
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
    } else if (specparm > 0.875f) {
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


kernel void hexv842_mpas_rrtmg_lw_band3_f32(constant Hexv842MpasRrtmgLwBand3F32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwBand3F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid);if(tid>=long(ncol)*nl)return;
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
        if (ratn2o > 1.5f) {
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
        if (ratn2o > 1.5f) {
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
