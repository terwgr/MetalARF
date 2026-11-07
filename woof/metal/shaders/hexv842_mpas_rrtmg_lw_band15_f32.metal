#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_band15_f32_bindings.metalh"
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

kernel void hexv842_mpas_rrtmg_lw_band15_f32(constant Hexv842MpasRrtmgLwBand15F32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwBand15F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid);if(tid>=long(ncol)*nl)return;
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
        if (specparm >= oneminus) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fsv = lw_upper_fmod(specmult, 1.0f);

        float speccomb1 = RLW_AD(coln2o,
                                 RLW_MU(FS(SL_RAT_N2OCO2_1), colco2));
        float specparm1 = RLW_DV(coln2o, speccomb1);
        if (specparm1 >= oneminus) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1 = lw_upper_fmod(specmult1, 1.0f);

        float speccomb_mn2 = RLW_AD(coln2o, RLW_MU(refrat_m_a, colco2));
        float specparm_mn2 = RLW_DV(coln2o, speccomb_mn2);
        if (specparm_mn2 >= oneminus) specparm_mn2 = oneminus;
        float specmult_mn2 = RLW_MU(8.0f, specparm_mn2);
        int jmn2 = 1 + (int)specmult_mn2;
        float fmn2 = lw_upper_fmod(specmult_mn2, 1.0f);

        float speccomb_planck = RLW_AD(coln2o,
                                       RLW_MU(refrat_planck_a, colco2));
        float specparm_planck = RLW_DV(coln2o, speccomb_planck);
        if (specparm_planck >= oneminus) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = lw_upper_fmod(specmult_planck, 1.0f);

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
