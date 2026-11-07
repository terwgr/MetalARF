#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_band13_f32_bindings.metalh"
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

kernel void hexv842_mpas_rrtmg_lw_band13_f32(constant Hexv842MpasRrtmgLwBand13F32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwBand13F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid);if(tid>=long(ncol)*nl)return;
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
        if (specparm >= oneminus) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fsv = lw_upper_fmod(specmult, 1.0f);

        float speccomb1 = RLW_AD(colh2o,
                                 RLW_MU(FS(SL_RAT_H2ON2O_1), coln2o));
        float specparm1 = RLW_DV(colh2o, speccomb1);
        if (specparm1 >= oneminus) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1 = lw_upper_fmod(specmult1, 1.0f);

        float speccomb_mco2 = RLW_AD(colh2o, RLW_MU(refrat_m_a, coln2o));
        float specparm_mco2 = RLW_DV(colh2o, speccomb_mco2);
        if (specparm_mco2 >= oneminus) specparm_mco2 = oneminus;
        float specmult_mco2 = RLW_MU(8.0f, specparm_mco2);
        int jmco2 = 1 + (int)specmult_mco2;
        float fmco2 = lw_upper_fmod(specmult_mco2, 1.0f);

        // CO2-too-major empirical adjustment (FIXED 3.55e-4 reference;
        // real exponent 0.68 -> glibc powf transcription).
        float chi_co2 = RLW_DV(FS(SL_COLCO2), FS(SL_COLDRY));
        float ratco2 = RLW_DV(RLW_MU(1.e20f, chi_co2), 3.55e-4f);
        float adjcolco2;
        if (ratco2 > 3.0f) {
            float adjfac = RLW_AD(2.0f,
                                  rlw_pow(RLW_SU(ratco2, 2.0f), 0.68f));
            adjcolco2 = RLW_MU(RLW_MU(RLW_MU(adjfac, 3.55e-4f),
                                      FS(SL_COLDRY)), 1.e-20f);
        } else {
            adjcolco2 = FS(SL_COLCO2);
        }

        float speccomb_mco = RLW_AD(colh2o, RLW_MU(refrat_m_a3, coln2o));
        float specparm_mco = RLW_DV(colh2o, speccomb_mco);
        if (specparm_mco >= oneminus) specparm_mco = oneminus;
        float specmult_mco = RLW_MU(8.0f, specparm_mco);
        int jmco = 1 + (int)specmult_mco;
        float fmco = lw_upper_fmod(specmult_mco, 1.0f);

        float speccomb_planck = RLW_AD(colh2o,
                                       RLW_MU(refrat_planck_a, coln2o));
        float specparm_planck = RLW_DV(colh2o, speccomb_planck);
        if (specparm_planck >= oneminus) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = lw_upper_fmod(specmult_planck, 1.0f);

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
