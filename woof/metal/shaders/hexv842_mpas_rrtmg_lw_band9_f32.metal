#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_band9_f32_bindings.metalh"
#pragma clang fp contract(off)
#define NGPTLW 140
#define MAXXSEC 4
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
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


#include "hexv842_mpas_rrtmg_lw_upper_math.metalh"

kernel void hexv842_mpas_rrtmg_lw_band9_f32(constant Hexv842MpasRrtmgLwBand9F32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwBand9F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid);if(tid>=long(ncol)*nl)return;
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
        if (specparm >= oneminus) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fsv = lw_upper_fmod(specmult, 1.0f);

        float speccomb1 = RLW_AD(colh2o,
                                 RLW_MU(FS(SL_RAT_H2OCH4_1), colch4));
        float specparm1 = RLW_DV(colh2o, speccomb1);
        if (specparm1 >= oneminus) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1v = lw_upper_fmod(specmult1, 1.0f);

        float speccomb_mn2o = RLW_AD(colh2o, RLW_MU(refrat_m_a, colch4));
        float specparm_mn2o = RLW_DV(colh2o, speccomb_mn2o);
        if (specparm_mn2o >= oneminus) specparm_mn2o = oneminus;
        float specmult_mn2o = RLW_MU(8.0f, specparm_mn2o);
        int jmn2o = 1 + (int)specmult_mn2o;
        float fmn2o = lw_upper_fmod(specmult_mn2o, 1.0f);

        float chi_n2o = RLW_DV(FS(SL_COLN2O), FS(SL_COLDRY));
        // ratn2o = 1.e20_rb*chi_n2o/chi_mls(4,jp(lay)+1)
        float ratn2o = RLW_DV(RLW_MU(1.e20f, chi_n2o),
                              chi_mls[3 + 7 * jpv]);
        float adjcoln2o;
        if (ratn2o > 1.5f) {
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
        if (specparm_planck >= oneminus) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = lw_upper_fmod(specmult_planck, 1.0f);

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
        if (ratn2o > 1.5f) {
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
