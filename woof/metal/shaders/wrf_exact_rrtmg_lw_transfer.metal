#include <metal_stdlib>
using namespace metal;
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "wrf_exact_rrtmg_lw_transfer_bindings.metalh"
#pragma clang fp contract(off)
#define RLW_MAXLAY 128
#define NGPTLW 140
#define NBNDLW 16
#define RLW_TBLINT 10000.0f
#define RLW_REC6 0.166667f
static inline float add(float x,float y){return as_type<float>(soft32_add(as_type<uint>(x),as_type<uint>(y)));}
static inline float sub(float x,float y){return as_type<float>(soft32_sub(as_type<uint>(x),as_type<uint>(y)));}
static inline float mul(float x,float y){return as_type<float>(soft32_mul(as_type<uint>(x),as_type<uint>(y)));}
static inline float divr(float x,float y){return as_type<float>(soft32_div(as_type<uint>(x),as_type<uint>(y)));}
static inline float ex(float x){return as_type<float>(soft_gfk_exp(as_type<uint>(x)));}
#define RLW_AD(x,y) add(x,y)
#define RLW_SU(x,y) sub(x,y)
#define RLW_MU(x,y) mul(x,y)
#define RLW_DV(x,y) divr(x,y)
#define A(x,y) add(x,y)
#define S(x,y) sub(x,y)
#define M(x,y) mul(x,y)
#define D(x,y) divr(x,y)

kernel void wrf_exact_rrtmg_lw_secdiff(constant WrfExactRrtmgLwSecdiffResources &r [[buffer(0)]],constant WrfExactRrtmgLwSecdiffParams &p [[buffer(1)]],uint col [[thread_position_in_grid]]) {
device const float *pw=r.pwvcm;
device const float *a0=r.a0;
device const float *a1=r.a1;
device const float *a2=r.a2;
device float *sdout=r.secdiff;
uint n=uint(p.ncol);

 if(col>=n)return;
 for(uint ib=0;ib<16;++ib){float sd;if(ib==0||ib==3||ib>=9)sd=1.66f;else{sd=add(a0[ib],mul(a1[ib],ex(mul(a2[ib],pw[col]))));if(sd>1.80f)sd=1.80f;if(sd<1.50f)sd=1.50f;}sdout[ulong(col)*16+ib]=sd;}

}

kernel void wrf_exact_rrtmg_lw_prol(constant WrfExactRrtmgLwProlResources &r [[buffer(0)]],constant WrfExactRrtmgLwProlParams &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
device const float *cloud=r.cldfmc;
device const float *tau=r.taucmc;
device const float *sd=r.secdiff;
device const int *ngb=r.ngb;
device float *od=r.odcld;
device float *ef=r.efclfrac;
device int *flags=r.icldlyr;
uint2 size=uint2(p.ncol,p.nl);

 if(ulong(tid)>=ulong(size.x)*size.y)return;
 uint col=tid/size.y,lay=tid%size.y;int flag=0;
 for(uint ig=0;ig<140;++ig){ulong i=(ulong(col)*140+ig)*size.y+lay;float cf=cloud[i];if(cf==1.0f){float x=mul(sd[ulong(col)*16+uint(ngb[ig]-1)],tau[i]);od[i]=x;float trans=ex(as_type<float>(as_type<uint>(x)^0x80000000u));ef[i]=mul(sub(1.0f,trans),cf);flag=1;}else{od[i]=0.0f;ef[i]=0.0f;}}
 flags[tid]=flag;

}

kernel void wrf_exact_rrtmg_lw_march(constant WrfExactRrtmgLwMarchResources &r [[buffer(0)]],constant WrfExactRrtmgLwMarchParams &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
device const float *cldfmc=r.cldfmc;
device const float *odcld=r.odcld;
device const float *efclfrac=r.efclfrac;
device const int *icldlyr=r.icldlyr;
device const float *secdiff=r.secdiff;
device const float *semiss=r.semiss;
device const float *planklay=r.planklay;
device const float *planklev=r.planklev;
device const float *plankbnd=r.plankbnd;
device const float *fracs=r.fracs;
device const float *taut=r.taut;
device const float *tau_tbl=r.tau_tbl;
device const float *exp_tbl=r.exp_tbl;
device const float *tfn_tbl=r.tfn_tbl;
device const int *ngb=r.ngb;
device float *radld_p=r.radld_p;
device float *radclrd_p=r.radclrd_p;
device float *radlu_p=r.radlu_p;
device float *radclru_p=r.radclru_p;
device int *iclddn_p=r.iclddn_p;
device float *radlu_sfc=r.radlu_sfc;
device float *radclru_sfc=r.radclru_sfc;

    int ncol=p.ncol, nl=p.nl; float bpade=p.bpade;
    long tid = (long)gid;
    if (tid >= (long)ncol * NGPTLW) return;
    if (nl > RLW_MAXLAY) return;   // host asserts nl <= RLW_MAXLAY
    int col = (int)(tid / NGPTLW);
    int igc = (int)(tid % NGPTLW) + 1;
    int iband = ngb[igc - 1];
    float sd = secdiff[(long)col * NBNDLW + (iband - 1)];

    float atrans[RLW_MAXLAY], atot[RLW_MAXLAY], bbugas[RLW_MAXLAY],
          bbutot[RLW_MAXLAY];

#define PRO(a, l) a[((long)col * NGPTLW + (igc - 1)) * nl + ((l) - 1)]
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
        if (odepth < 0.0f) odepth = 0.0f;
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
            float cf = cldfmc[((long)col * NGPTLW + (igc - 1)) * nl
                              + (lev - 1)];
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
        iclddn_p[((long)col * NGPTLW + (igc - 1)) * nl + (lev - 1)] =
            iclddn;
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
            float cf = cldfmc[((long)col * NGPTLW + (igc - 1)) * nl
                              + (lev - 1)];
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

kernel void wrf_exact_rrtmg_lw_accum(constant WrfExactRrtmgLwAccumResources &r [[buffer(0)]],constant WrfExactRrtmgLwAccumParams &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
device const float *ld=r.radld_p;
device const float *cd=r.radclrd_p;
device const float *lu=r.radlu_p;
device const float *cu=r.radclru_p;
device const int *flag=r.iclddn_p;
device const float *su=r.radlu_sfc;
device const float *sc=r.radclru_sfc;
device const int *ngs=r.ngs;
device const float *dw=r.delwave;
device float *tu=r.totuflux;
device float *td=r.totdflux;
device float *tc=r.totuclfl;
device float *te=r.totdclfl;

 if(ulong(tid)>=ulong(p.ncol)*(p.nl+1))return;uint col=tid/(p.nl+1),lev=tid%(p.nl+1);float u=0,d=0,uc=0,dc=0;
 for(int b=0;b<16;b++){int lo=b?ngs[b-1]:0,hi=ngs[b];float ur=0,dr=0,cr=0,cl=0;
  for(int g=lo;g<hi;g++){ulong q=(ulong(col)*140+uint(g))*p.nl;
   if(lev<p.nl){dr=A(dr,ld[q+lev]);cl=flag[q+lev]?A(cl,cd[q+lev]):dr;}
   if(lev==0){ur=A(ur,su[ulong(col)*140+g]);cr=flag[q]?A(cr,sc[ulong(col)*140+g]):ur;}
   else{ur=A(ur,lu[q+lev-1]);cr=flag[q]?A(cr,cu[q+lev-1]):ur;}
  }
  u=A(u,M(M(ur,p.wtdiff),dw[b]));d=A(d,M(M(dr,p.wtdiff),dw[b]));uc=A(uc,M(M(cr,p.wtdiff),dw[b]));dc=A(dc,M(M(cl,p.wtdiff),dw[b]));
 }
 tu[tid]=u;td[tid]=d;tc[tid]=uc;te[tid]=dc;

}

kernel void wrf_exact_rrtmg_lw_final(constant WrfExactRrtmgLwFinalResources &r [[buffer(0)]],constant WrfExactRrtmgLwFinalParams &p [[buffer(1)]],uint col [[thread_position_in_grid]]) {
device const float *pz=r.pz;
device float *tu=r.totuflux;
device float *td=r.totdflux;
device float *tc=r.totuclfl;
device float *te=r.totdclfl;
device float *net=r.fnet;
device float *netc=r.fnetc;
device float *h=r.htr;
device float *hc=r.htrc;

 if(col>=p.ncol)return;ulong base=ulong(col)*(p.nl+1);
 for(uint l=0;l<=p.nl;l++){ulong q=base+l;tu[q]=M(tu[q],p.fluxfac);td[q]=M(td[q],p.fluxfac);net[q]=S(tu[q],td[q]);tc[q]=M(tc[q],p.fluxfac);te[q]=M(te[q],p.fluxfac);netc[q]=S(tc[q],te[q]);if(l){ulong prev=q-1;float den=S(pz[prev],pz[q]);h[prev]=D(M(p.heatfac,S(net[prev],net[q])),den);hc[prev]=D(M(p.heatfac,S(netc[prev],netc[q])),den);}}
 h[base+p.nl]=0;hc[base+p.nl]=0;

}

kernel void wrf_exact_rrtmg_lw_taut(constant WrfExactRrtmgLwTautResources &r [[buffer(0)]], constant WrfExactRrtmgLwTautParams &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
 if(ulong(tid)>=ulong(p.ncol)*ulong(p.nl)*ulong(p.ngpt))return;
 uint ig=tid%uint(p.ngpt); ulong cell=ulong(tid)/uint(p.ngpt);
 r.taut[tid]=add(r.taug[tid],r.taua[cell*uint(p.nbnd)+uint(r.ngb[ig]-1)]);
}
