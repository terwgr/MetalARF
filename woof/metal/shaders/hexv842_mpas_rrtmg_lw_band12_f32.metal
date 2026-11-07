#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_band12_f32_bindings.metalh"
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

kernel void hexv842_mpas_rrtmg_lw_band12_f32(constant Hexv842MpasRrtmgLwBand12F32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwBand12F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid);if(tid>=long(ncol)*nl)return;
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
        if (specparm >= oneminus) specparm = oneminus;
        float specmult = RLW_MU(8.0f, specparm);
        int js = 1 + (int)specmult;
        float fs0 = lw_upper_fmod(specmult, 1.0f);

        float speccomb1 = RLW_AD(colh2o, RLW_MU(FS(SL_RAT_H2OCO2_1), colco2));
        float specparm1 = RLW_DV(colh2o, speccomb1);
        if (specparm1 >= oneminus) specparm1 = oneminus;
        float specmult1 = RLW_MU(8.0f, specparm1);
        int js1 = 1 + (int)specmult1;
        float fs1 = lw_upper_fmod(specmult1, 1.0f);

        float speccomb_planck = RLW_AD(colh2o,
                                       RLW_MU(refrat_planck_a, colco2));
        float specparm_planck = RLW_DV(colh2o, speccomb_planck);
        if (specparm_planck >= oneminus) specparm_planck = oneminus;
        float specmult_planck = RLW_MU(8.0f, specparm_planck);
        int jpl = 1 + (int)specmult_planck;
        float fpl = lw_upper_fmod(specmult_planck, 1.0f);

        int ind0 = ((IS(SI_JP) - 1) * 5 + (IS(SI_JT) - 1)) * nspa12 + js;
        int ind1 = (IS(SI_JP) * 5 + (IS(SI_JT1) - 1)) * nspa12 + js1;
        int inds = IS(SI_INDSELF);
        int indf = IS(SI_INDFOR);

        float fac000 = 0.0f, fac100 = 0.0f, fac200 = 0.0f;
        float fac010 = 0.0f, fac110 = 0.0f, fac210 = 0.0f;
        if (specparm < 0.125f) {
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
        } else if (specparm > 0.875f) {
            float p = -fs0;
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
        if (specparm1 < 0.125f) {
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
        } else if (specparm1 > 0.875f) {
            float p = -fs1;
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
            if (specparm < 0.125f) {
                float t = RLW_MU(fac000, A2(absa, 585, ind0, ig));
                t = RLW_AD(t, RLW_MU(fac100, A2(absa, 585, ind0 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac200, A2(absa, 585, ind0 + 2, ig)));
                t = RLW_AD(t, RLW_MU(fac010, A2(absa, 585, ind0 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac110, A2(absa, 585, ind0 + 10, ig)));
                t = RLW_AD(t, RLW_MU(fac210, A2(absa, 585, ind0 + 11, ig)));
                tau_major = RLW_MU(speccomb, t);
            } else if (specparm > 0.875f) {
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
            if (specparm1 < 0.125f) {
                float t = RLW_MU(fac001, A2(absa, 585, ind1, ig));
                t = RLW_AD(t, RLW_MU(fac101, A2(absa, 585, ind1 + 1, ig)));
                t = RLW_AD(t, RLW_MU(fac201, A2(absa, 585, ind1 + 2, ig)));
                t = RLW_AD(t, RLW_MU(fac011, A2(absa, 585, ind1 + 9, ig)));
                t = RLW_AD(t, RLW_MU(fac111, A2(absa, 585, ind1 + 10, ig)));
                t = RLW_AD(t, RLW_MU(fac211, A2(absa, 585, ind1 + 11, ig)));
                tau_major1 = RLW_MU(speccomb1, t);
            } else if (specparm1 > 0.875f) {
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
