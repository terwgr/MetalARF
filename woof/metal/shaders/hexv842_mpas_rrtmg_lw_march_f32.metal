#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_rrtmg_lw_march_f32_bindings.metalh"
#pragma clang fp contract(off)
#define NGPTLW 140
#define NBNDLW 16
#define RLW_MAXLAY 128
#define RLW_REC6 0.166667f
#define RLW_TBLINT 10000.0f
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
kernel void hexv842_mpas_rrtmg_lw_march_f32(constant Hexv842MpasRrtmgLwMarchF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwMarchF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nl=p.nl;float bpade=p.bpade;
 device const float* cldfmc=r.cldfmc;
 device const float* odcld=r.odcld;
 device const float* efclfrac=r.efclfrac;
 device const int* icldlyr=r.icldlyr;
 device const float* secdiff=r.secdiff;
 device const float* semiss=r.semiss;
 device const float* planklay=r.planklay;
 device const float* planklev=r.planklev;
 device const float* plankbnd=r.plankbnd;
 device const float* fracs=r.fracs;
 device const float* taut=r.taut;
 device const float* tau_tbl=r.tau_tbl;
 device const float* exp_tbl=r.exp_tbl;
 device const float* tfn_tbl=r.tfn_tbl;
 device const int* ngb=r.ngb;
 device float* radld_p=r.radld_p;
 device float* radclrd_p=r.radclrd_p;
 device float* radlu_p=r.radlu_p;
 device float* radclru_p=r.radclru_p;
 device int* iclddn_p=r.iclddn_p;
 device float* radlu_sfc=r.radlu_sfc;
 device float* radclru_sfc=r.radclru_sfc;
 device float* scratch=r.scratch;

    long tid = long(gid);
    if (tid >= (long)ncol * NGPTLW) return;
    if (nl > RLW_MAXLAY) return;   // host asserts nl <= RLW_MAXLAY
    int col = (int)(tid / NGPTLW);
    int igc = (int)(tid % NGPTLW) + 1;
    int iband = ngb[igc - 1];
    float sd = secdiff[(long)col * NBNDLW + (iband - 1)];

    device float* atrans = scratch + tid*4*nl;
    device float* atot = atrans + nl;
    device float* bbugas = atot + nl;
    device float* bbutot = bbugas + nl;

#define PRO(a, l) a[((long)col * nl + ((l) - 1)) * NGPTLW + (igc - 1)]
#define PLLAY2(l) planklay[((long)col * nl + ((l) - 1)) * NBNDLW + (iband - 1)]
#define PLLEV2(l) planklev[((long)col * (nl + 1) + (l)) * NBNDLW + (iband - 1)]
#define FR(l) fracs[((long)col * nl + ((l) - 1)) * NGPTLW + (igc - 1)]
#define TT(l) taut[((long)col * nl + ((l) - 1)) * NGPTLW + (igc - 1)]

    float radld = 0.0f;
    float radclrd = 0.0f;
    int iclddn = 0;

    for (int lev = nl; lev >= 1; --lev) {
        float plfrac = FR(lev);
        float blay = PLLAY2(lev);
        float dplankup = RLW_SU(PLLEV2(lev), blay);
        float dplankdn = RLW_SU(PLLEV2(lev - 1), blay);
        float odepth = RLW_MU(sd, TT(lev));
        if (soft32_less(as_type<uint>(odepth),0u)) odepth = 0.0f;
        float bbd;
        if (icldlyr[(long)col * nl + (lev - 1)] == 1) {
            iclddn = 1;
            float oc = PRO(odcld, lev);
            float odtot = RLW_AD(odepth, oc);
            float gassrc, bbdtot, at, ao, bug, but;
            if (odtot < 0.06f) {
                at = RLW_SU(odepth, RLW_MU(0.5f, RLW_MU(odepth, odepth)));
                float od_rec = RLW_MU(RLW_REC6, odepth);
                gassrc = RLW_MU(RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankdn, od_rec))), at);
                ao = RLW_SU(odtot, RLW_MU(0.5f, RLW_MU(odtot, odtot)));
                float odtot_rec = RLW_MU(RLW_REC6, odtot);
                bbdtot = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankdn, odtot_rec)));
                bbd = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankdn, od_rec)));
                bug = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankup, od_rec)));
                but = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankup, odtot_rec)));
            } else if (odepth <= 0.06f) {
                at = RLW_SU(odepth, RLW_MU(0.5f, RLW_MU(odepth, odepth)));
                float od_rec = RLW_MU(RLW_REC6, odepth);
                gassrc = RLW_MU(RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankdn, od_rec))), at);
                float tblind = RLW_DV(odtot, RLW_AD(bpade, odtot));
                int ittot = (int)RLW_AD(RLW_MU(RLW_TBLINT, tblind), 0.5f);
                float tfactot = tfn_tbl[ittot];
                bbdtot = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tfactot, dplankdn)));
                bbd = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankdn, od_rec)));
                ao = RLW_SU(1.0f, exp_tbl[ittot]);
                bug = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(dplankup, od_rec)));
                but = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tfactot, dplankup)));
            } else {
                float tblind = RLW_DV(odepth, RLW_AD(bpade, odepth));
                int itgas = (int)RLW_AD(RLW_MU(RLW_TBLINT, tblind), 0.5f);
                odepth = tau_tbl[itgas];
                at = RLW_SU(1.0f, exp_tbl[itgas]);
                float tfacgas = tfn_tbl[itgas];
                gassrc = RLW_MU(RLW_MU(at, plfrac),
                    RLW_AD(blay, RLW_MU(tfacgas, dplankdn)));
                float odtot2 = RLW_AD(odepth, oc);
                float tblind2 = RLW_DV(odtot2, RLW_AD(bpade, odtot2));
                int ittot = (int)RLW_AD(RLW_MU(RLW_TBLINT, tblind2), 0.5f);
                float tfactot = tfn_tbl[ittot];
                bbdtot = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tfactot, dplankdn)));
                bbd = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tfacgas, dplankdn)));
                ao = RLW_SU(1.0f, exp_tbl[ittot]);
                bug = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tfacgas, dplankup)));
                but = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tfactot, dplankup)));
            }
            float cf = cldfmc[((long)col * nl + (lev - 1)) * NGPTLW + (igc - 1)];
            float ef = PRO(efclfrac, lev);
            radld = RLW_SU(radld,
                RLW_MU(radld, RLW_AD(at, RLW_MU(ef, RLW_SU(1.0f, at)))));
            radld = RLW_AD(radld, gassrc);
            radld = RLW_AD(radld,
                RLW_MU(cf, RLW_SU(RLW_MU(bbdtot, ao), gassrc)));
            atrans[lev - 1] = at;
            atot[lev - 1] = ao;
            bbugas[lev - 1] = bug;
            bbutot[lev - 1] = but;
        } else {
            float at, bug;
            if (odepth <= 0.06f) {
                at = RLW_SU(odepth, RLW_MU(0.5f, RLW_MU(odepth, odepth)));
                float od2 = RLW_MU(RLW_REC6, odepth);
                bbd = RLW_MU(plfrac, RLW_AD(blay, RLW_MU(dplankdn, od2)));
                bug = RLW_MU(plfrac, RLW_AD(blay, RLW_MU(dplankup, od2)));
            } else {
                float tblind = RLW_DV(odepth, RLW_AD(bpade, odepth));
                int itr = (int)RLW_AD(RLW_MU(RLW_TBLINT, tblind), 0.5f);
                float transc = exp_tbl[itr];
                at = RLW_SU(1.0f, transc);
                float tausfac = tfn_tbl[itr];
                bbd = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tausfac, dplankdn)));
                bug = RLW_MU(plfrac,
                    RLW_AD(blay, RLW_MU(tausfac, dplankup)));
            }
            radld = RLW_AD(radld, RLW_MU(RLW_SU(bbd, radld), at));
            atrans[lev - 1] = at;
            bbugas[lev - 1] = bug;
        }
        PRO(radld_p, lev) = radld;
        if (iclddn == 1) {
            radclrd = RLW_AD(radclrd,
                             RLW_MU(RLW_SU(bbd, radclrd), atrans[lev - 1]));
        } else {
            radclrd = radld;
        }
        PRO(radclrd_p, lev) = radclrd;
        iclddn_p[((long)col * nl + (lev - 1)) * NGPTLW + (igc - 1)] =
            (int)iclddn;
    }

    float rad0 = RLW_MU(FR(1),
                        plankbnd[(long)col * NBNDLW + (iband - 1)]);
    float reflect = RLW_SU(1.0f,
                           semiss[(long)col * NBNDLW + (iband - 1)]);
    float radlu = RLW_AD(rad0, RLW_MU(reflect, radld));
    float radclru = RLW_AD(rad0, RLW_MU(reflect, radclrd));
    radlu_sfc[(long)col * NGPTLW + (igc - 1)] = radlu;
    radclru_sfc[(long)col * NGPTLW + (igc - 1)] = radclru;

    for (int lev = 1; lev <= nl; ++lev) {
        if (icldlyr[(long)col * nl + (lev - 1)] == 1) {
            float gassrc = RLW_MU(bbugas[lev - 1], atrans[lev - 1]);
            float cf = cldfmc[((long)col * nl + (lev - 1)) * NGPTLW + (igc - 1)];
            float ef = PRO(efclfrac, lev);
            radlu = RLW_SU(radlu,
                RLW_MU(radlu, RLW_AD(atrans[lev - 1],
                    RLW_MU(ef, RLW_SU(1.0f, atrans[lev - 1])))));
            radlu = RLW_AD(radlu, gassrc);
            radlu = RLW_AD(radlu,
                RLW_MU(cf, RLW_SU(RLW_MU(bbutot[lev - 1], atot[lev - 1]),
                                  gassrc)));
        } else {
            radlu = RLW_AD(radlu,
                RLW_MU(RLW_SU(bbugas[lev - 1], radlu), atrans[lev - 1]));
        }
        PRO(radlu_p, lev) = radlu;
        if (iclddn == 1) {
            radclru = RLW_AD(radclru,
                RLW_MU(RLW_SU(bbugas[lev - 1], radclru), atrans[lev - 1]));
        } else {
            radclru = radlu;
        }
        PRO(radclru_p, lev) = radclru;
    }
#undef PRO
#undef PLLAY2
#undef PLLEV2
#undef FR
#undef TT

}
