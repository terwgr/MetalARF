#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_band10_f32_bindings.metalh"
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

kernel void hexv842_mpas_rrtmg_lw_band10_f32(constant Hexv842MpasRrtmgLwBand10F32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwBand10F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float oneminus=p.oneminus;
 device const int *laytrop_v=r.laytrop_v,*isv=r.isv;
 device const float *fs=r.fs,*wx=r.wx,*chi_mls=r.chi_mls;
 device float *taug=r.gas_tau,*fracs=r.fracs;

    long tid=long(gid);if(tid>=long(ncol)*nl)return;
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
