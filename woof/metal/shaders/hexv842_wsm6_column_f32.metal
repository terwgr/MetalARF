#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
using namespace ntc;
// This reproduces the source's explicit operation order for the two powers used by init.
template<int N> static inline F ipow(F x) {
 if constexpr (N==2) return tc_mul(x,x);
 if constexpr (N==3) return tc_mul(tc_mul(x,x),x);
 if constexpr (N==8) {F x2=tc_mul(x,x),x4=tc_mul(x2,x2);return tc_mul(x4,x4);}
}
struct Wsm6Workspace {
 device F *mp_wsm6_run_rh;
 device F *mp_wsm6_run_qsat;
 device F *mp_wsm6_run_rslope;
 device F *mp_wsm6_run_rslope2;
 device F *mp_wsm6_run_rslope3;
 device F *mp_wsm6_run_rslopeb;
 device F *mp_wsm6_run_qrs_tmp;
 device F *mp_wsm6_run_falk;
 device F *mp_wsm6_run_fall;
 device F *mp_wsm6_run_work1;
 device F *mp_wsm6_run_fallc;
 device F *mp_wsm6_run_falkc;
 device F *mp_wsm6_run_work1c;
 device F *mp_wsm6_run_work2c;
 device F *mp_wsm6_run_workr;
 device F *mp_wsm6_run_worka;
 device F *mp_wsm6_run_den_tmp;
 device F *mp_wsm6_run_delz_tmp;
 device F *mp_wsm6_run_pigen;
 device F *mp_wsm6_run_pidep;
 device F *mp_wsm6_run_pcond;
 device F *mp_wsm6_run_prevp;
 device F *mp_wsm6_run_psevp;
 device F *mp_wsm6_run_pgevp;
 device F *mp_wsm6_run_psdep;
 device F *mp_wsm6_run_pgdep;
 device F *mp_wsm6_run_praut;
 device F *mp_wsm6_run_psaut;
 device F *mp_wsm6_run_pgaut;
 device F *mp_wsm6_run_piacr;
 device F *mp_wsm6_run_pracw;
 device F *mp_wsm6_run_praci;
 device F *mp_wsm6_run_pracs;
 device F *mp_wsm6_run_psacw;
 device F *mp_wsm6_run_psaci;
 device F *mp_wsm6_run_psacr;
 device F *mp_wsm6_run_pgacw;
 device F *mp_wsm6_run_pgaci;
 device F *mp_wsm6_run_pgacr;
 device F *mp_wsm6_run_pgacs;
 device F *mp_wsm6_run_paacw;
 device F *mp_wsm6_run_psmlt;
 device F *mp_wsm6_run_pgmlt;
 device F *mp_wsm6_run_pseml;
 device F *mp_wsm6_run_pgeml;
 device F *mp_wsm6_run_qsum;
 device F *mp_wsm6_run_xl;
 device F *mp_wsm6_run_cpm;
 device F *mp_wsm6_run_work2;
 device F *mp_wsm6_run_denfac;
 device F *mp_wsm6_run_xni;
 device F *mp_wsm6_run_denqrs1;
 device F *mp_wsm6_run_denqrs2;
 device F *mp_wsm6_run_denqrs3;
 device F *mp_wsm6_run_denqci;
 device F *mp_wsm6_run_n0sfac;
 device F *slope_wsm6_n0sfac;
 device F *slope_rain_n0sfac;
 device F *slope_snow_n0sfac;
 device F *slope_graup_n0sfac;
 device F *nislfv_rain_plm_dz;
 device F *nislfv_rain_plm_ww;
 device F *nislfv_rain_plm_qq;
 device F *nislfv_rain_plm_wd;
 device F *nislfv_rain_plm_wa;
 device F *nislfv_rain_plm_was;
 device F *nislfv_rain_plm_den;
 device F *nislfv_rain_plm_denfac;
 device F *nislfv_rain_plm_tk;
 device F *nislfv_rain_plm_qn;
 device F *nislfv_rain_plm_qr;
 device F *nislfv_rain_plm_tmp;
 device F *nislfv_rain_plm_tmp1;
 device F *nislfv_rain_plm_tmp2;
 device F *nislfv_rain_plm_tmp3;
 device F *nislfv_rain_plm_wi;
 device F *nislfv_rain_plm_zi;
 device F *nislfv_rain_plm_za;
 device F *nislfv_rain_plm_dza;
 device F *nislfv_rain_plm_qa;
 device F *nislfv_rain_plm_qmi;
 device F *nislfv_rain_plm_qpi;
 device F *nislfv_rain_plm6_dz;
 device F *nislfv_rain_plm6_ww;
 device F *nislfv_rain_plm6_qq;
 device F *nislfv_rain_plm6_qq2;
 device F *nislfv_rain_plm6_wd;
 device F *nislfv_rain_plm6_wa;
 device F *nislfv_rain_plm6_wa2;
 device F *nislfv_rain_plm6_was;
 device F *nislfv_rain_plm6_den;
 device F *nislfv_rain_plm6_denfac;
 device F *nislfv_rain_plm6_tk;
 device F *nislfv_rain_plm6_qn;
 device F *nislfv_rain_plm6_qr;
 device F *nislfv_rain_plm6_qr2;
 device F *nislfv_rain_plm6_tmp;
 device F *nislfv_rain_plm6_tmp1;
 device F *nislfv_rain_plm6_tmp2;
 device F *nislfv_rain_plm6_tmp3;
 device F *nislfv_rain_plm6_wi;
 device F *nislfv_rain_plm6_zi;
 device F *nislfv_rain_plm6_za;
 device F *nislfv_rain_plm6_dza;
 device F *nislfv_rain_plm6_qa;
 device F *nislfv_rain_plm6_qa2;
 device F *nislfv_rain_plm6_qmi;
 device F *nislfv_rain_plm6_qpi;
};
struct Wsm6State {
 thread Wsm6Workspace *workspace;
 F dtcldcr;
 F n0r;
 F avtr;
 F bvtr;
 F r0;
 F peaut;
 F xncr;
 F xmyu;
 F avts;
 F bvts;
 F lamdarmax;
 F lamdasmax;
 F dicon;
 F dimax;
 F pfrz1;
 F pfrz2;
 F qcrmin;
 F eacrc;
 F dens;
 F qs0;
 F n0smax;
 F n0s;
 F alpha;
 F qc0;
 F qck1;
 F bvtr1;
 F bvtr2;
 F bvtr3;
 F bvtr4;
 F g1pbr;
 F g3pbr;
 F g4pbr;
 F g5pbro2;
 F pvtr;
 F eacrr;
 F pacrr;
 F bvtr6;
 F g6pbr;
 F precr1;
 F precr2;
 F roqimax;
 F bvts1;
 F bvts2;
 F bvts3;
 F bvts4;
 F g1pbs;
 F g3pbs;
 F g4pbs;
 F n0g;
 F avtg;
 F bvtg;
 F deng;
 F lamdagmax;
 F g5pbso2;
 F pvts;
 F pacrs;
 F precs1;
 F precs2;
 F pidn0r;
 F xlv1;
 F pacrc;
 F pi;
 F bvtg1;
 F bvtg2;
 F bvtg3;
 F bvtg4;
 F g1pbg;
 F g3pbg;
 F g4pbg;
 F g5pbgo2;
 F pvtg;
 F pacrg;
 F precg1;
 F precg2;
 F pidn0g;
 F rslopermax;
 F rslopesmax;
 F rslopegmax;
 F rsloperbmax;
 F rslopesbmax;
 F rslopegbmax;
 F rsloper2max;
 F rslopes2max;
 F rslopeg2max;
 F rsloper3max;
 F rslopes3max;
 F rslopeg3max;
 F pidn0s;
 F pidnc;
};
static inline void state_parameters(thread Wsm6State &S) {
S.dtcldcr = 120.f;
S.n0r = 8.e6f;
S.avtr = 841.9f;
S.bvtr = 0.8f;
S.r0 = .8e-5f;
S.peaut = .55f;
S.xncr = 3.e8f;
S.xmyu = 1.718e-5f;
S.avts = 11.72f;
S.bvts = .41f;
S.lamdarmax = 8.e4f;
S.lamdasmax = 1.e5f;
S.dicon = 11.9f;
S.dimax = 500.e-6f;
S.pfrz1 = 100.f;
S.pfrz2 = 0.66f;
S.qcrmin = 1.e-9f;
S.eacrc = 1.0f;
S.dens = 100.0f;
S.qs0 = 6.e-4f;
S.n0smax = 1.e11f;
S.n0s = 2.e6f;
S.alpha = .12f;
}
static inline F rgmma(thread Wsm6State &S, F x) {
int i;
F euler = 0.577215664901532f;
F y;
F result;
  if ((x == 1.f)) { // mp_wsm6.F90:1483
  result = 0.f; // mp_wsm6.F90:1484
  } else { // mp_wsm6.F90:1485
  result = tc_mul(x, tc_exp(tc_mul(euler, x))); // mp_wsm6.F90:1486
  for (i=1; i<=10000; i+=1) { // mp_wsm6.F90:1487
  y = F(float(i)); // mp_wsm6.F90:1488
  result = tc_mul(tc_mul(result, tc_add(1.000f, tc_div(x, y))), tc_exp(tc_div((-x), y))); // mp_wsm6.F90:1489
   } // mp_wsm6.F90:1490
  result = tc_div(1.f, result); // mp_wsm6.F90:1491
  } // mp_wsm6.F90:1492
return result;
}
static inline void mp_wsm6_init(thread Wsm6State &S, F den0, F denr, F dens, F cl, F cpv, int hail_opt, thread int &errflg) {
  if ((hail_opt == 1)) { // mp_wsm6.F90:88
  S.n0g = 4.e4f; // mp_wsm6.F90:89
  S.deng = 700.f; // mp_wsm6.F90:90
  S.avtg = 285.0f; // mp_wsm6.F90:91
  S.bvtg = 0.8f; // mp_wsm6.F90:92
  S.lamdagmax = 2.e4f; // mp_wsm6.F90:93
  } else { // mp_wsm6.F90:94
  S.n0g = 4.e6f; // mp_wsm6.F90:95
  S.deng = 500.f; // mp_wsm6.F90:96
  S.avtg = 330.0f; // mp_wsm6.F90:97
  S.bvtg = 0.8f; // mp_wsm6.F90:98
  S.lamdagmax = 6.e4f; // mp_wsm6.F90:99
  } // mp_wsm6.F90:100
  S.pi = tc_mul(4.f, f32_word(0x3f490fdbu)); // mp_wsm6.F90:102
  S.xlv1 = tc_sub(cl, cpv); // mp_wsm6.F90:103
  S.qc0 = tc_div(tc_mul(tc_mul(tc_mul(tc_mul(tc_div(4.f, 3.f), S.pi), denr), ipow<3>(S.r0)), S.xncr), den0); // mp_wsm6.F90:105
  S.qck1 = tc_mul(tc_div(tc_div(tc_mul(tc_mul(.104f, 9.8f), S.peaut), tc_pow(tc_mul(S.xncr, denr), tc_div(1.f, 3.f))), S.xmyu), tc_pow(den0, tc_div(4.f, 3.f))); // mp_wsm6.F90:106
  S.pidnc = tc_div(tc_mul(S.pi, denr), 6.f); // mp_wsm6.F90:107
  S.bvtr1 = tc_add(1.f, S.bvtr); // mp_wsm6.F90:109
  S.bvtr2 = tc_add(2.5f, tc_mul(.5f, S.bvtr)); // mp_wsm6.F90:110
  S.bvtr3 = tc_add(3.f, S.bvtr); // mp_wsm6.F90:111
  S.bvtr4 = tc_add(4.f, S.bvtr); // mp_wsm6.F90:112
  S.bvtr6 = tc_add(6.f, S.bvtr); // mp_wsm6.F90:113
  S.g1pbr = rgmma(S, S.bvtr1); // mp_wsm6.F90:114
  S.g3pbr = rgmma(S, S.bvtr3); // mp_wsm6.F90:115
  S.g4pbr = rgmma(S, S.bvtr4); // mp_wsm6.F90:116
  S.g6pbr = rgmma(S, S.bvtr6); // mp_wsm6.F90:117
  S.g5pbro2 = rgmma(S, S.bvtr2); // mp_wsm6.F90:118
  S.pvtr = tc_div(tc_mul(S.avtr, S.g4pbr), 6.f); // mp_wsm6.F90:119
  S.eacrr = 1.0f; // mp_wsm6.F90:120
  S.pacrr = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(S.pi, S.n0r), S.avtr), S.g3pbr), .25f), S.eacrr); // mp_wsm6.F90:121
  S.precr1 = tc_mul(tc_mul(tc_mul(2.f, S.pi), S.n0r), .78f); // mp_wsm6.F90:122
  S.precr2 = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(2.f, S.pi), S.n0r), .31f), tc_pow(S.avtr, .5f)), S.g5pbro2); // mp_wsm6.F90:123
  S.roqimax = tc_mul(2.08e22f, ipow<8>(S.dimax)); // mp_wsm6.F90:124
  S.bvts1 = tc_add(1.f, S.bvts); // mp_wsm6.F90:126
  S.bvts2 = tc_add(2.5f, tc_mul(.5f, S.bvts)); // mp_wsm6.F90:127
  S.bvts3 = tc_add(3.f, S.bvts); // mp_wsm6.F90:128
  S.bvts4 = tc_add(4.f, S.bvts); // mp_wsm6.F90:129
  S.g1pbs = rgmma(S, S.bvts1); // mp_wsm6.F90:130
  S.g3pbs = rgmma(S, S.bvts3); // mp_wsm6.F90:131
  S.g4pbs = rgmma(S, S.bvts4); // mp_wsm6.F90:132
  S.g5pbso2 = rgmma(S, S.bvts2); // mp_wsm6.F90:133
  S.pvts = tc_div(tc_mul(S.avts, S.g4pbs), 6.f); // mp_wsm6.F90:134
  S.pacrs = tc_mul(tc_mul(tc_mul(tc_mul(S.pi, S.n0s), S.avts), S.g3pbs), .25f); // mp_wsm6.F90:135
  S.precs1 = tc_mul(tc_mul(4.f, S.n0s), .65f); // mp_wsm6.F90:136
  S.precs2 = tc_mul(tc_mul(tc_mul(tc_mul(4.f, S.n0s), .44f), tc_pow(S.avts, .5f)), S.g5pbso2); // mp_wsm6.F90:137
  S.pidn0r = tc_mul(tc_mul(S.pi, denr), S.n0r); // mp_wsm6.F90:138
  S.pidn0s = tc_mul(tc_mul(S.pi, dens), S.n0s); // mp_wsm6.F90:139
  S.pacrc = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(S.pi, S.n0s), S.avts), S.g3pbs), .25f), S.eacrc); // mp_wsm6.F90:141
  S.bvtg1 = tc_add(1.f, S.bvtg); // mp_wsm6.F90:143
  S.bvtg2 = tc_add(2.5f, tc_mul(.5f, S.bvtg)); // mp_wsm6.F90:144
  S.bvtg3 = tc_add(3.f, S.bvtg); // mp_wsm6.F90:145
  S.bvtg4 = tc_add(4.f, S.bvtg); // mp_wsm6.F90:146
  S.g1pbg = rgmma(S, S.bvtg1); // mp_wsm6.F90:147
  S.g3pbg = rgmma(S, S.bvtg3); // mp_wsm6.F90:148
  S.g4pbg = rgmma(S, S.bvtg4); // mp_wsm6.F90:149
  S.pacrg = tc_mul(tc_mul(tc_mul(tc_mul(S.pi, S.n0g), S.avtg), S.g3pbg), .25f); // mp_wsm6.F90:150
  S.g5pbgo2 = rgmma(S, S.bvtg2); // mp_wsm6.F90:151
  S.pvtg = tc_div(tc_mul(S.avtg, S.g4pbg), 6.f); // mp_wsm6.F90:152
  S.precg1 = tc_mul(tc_mul(tc_mul(2.f, S.pi), S.n0g), .78f); // mp_wsm6.F90:153
  S.precg2 = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(2.f, S.pi), S.n0g), .31f), tc_pow(S.avtg, .5f)), S.g5pbgo2); // mp_wsm6.F90:154
  S.pidn0g = tc_mul(tc_mul(S.pi, S.deng), S.n0g); // mp_wsm6.F90:155
  S.rslopermax = tc_div(1.f, S.lamdarmax); // mp_wsm6.F90:157
  S.rslopesmax = tc_div(1.f, S.lamdasmax); // mp_wsm6.F90:158
  S.rslopegmax = tc_div(1.f, S.lamdagmax); // mp_wsm6.F90:159
  S.rsloperbmax = tc_pow(S.rslopermax, S.bvtr); // mp_wsm6.F90:160
  S.rslopesbmax = tc_pow(S.rslopesmax, S.bvts); // mp_wsm6.F90:161
  S.rslopegbmax = tc_pow(S.rslopegmax, S.bvtg); // mp_wsm6.F90:162
  S.rsloper2max = tc_mul(S.rslopermax, S.rslopermax); // mp_wsm6.F90:163
  S.rslopes2max = tc_mul(S.rslopesmax, S.rslopesmax); // mp_wsm6.F90:164
  S.rslopeg2max = tc_mul(S.rslopegmax, S.rslopegmax); // mp_wsm6.F90:165
  S.rsloper3max = tc_mul(S.rsloper2max, S.rslopermax); // mp_wsm6.F90:166
  S.rslopes3max = tc_mul(S.rslopes2max, S.rslopesmax); // mp_wsm6.F90:167
  S.rslopeg3max = tc_mul(S.rslopeg2max, S.rslopegmax); // mp_wsm6.F90:168
  // CCPP status text // mp_wsm6.F90:186
  errflg = 0; // mp_wsm6.F90:187
}
static inline F wsm_lamdar(thread Wsm6State &S,F x, F y) { return tc_sqrt(tc_sqrt(tc_div(S.pidn0r, tc_mul(x, y)))); }
static inline F wsm_lamdas(thread Wsm6State &S,F x, F y, F z) { return tc_sqrt(tc_sqrt(tc_div(tc_mul(S.pidn0s, z), tc_mul(x, y)))); }
static inline F wsm_lamdag(thread Wsm6State &S,F x, F y) { return tc_sqrt(tc_sqrt(tc_div(S.pidn0g, tc_mul(x, y)))); }
static inline F wsm_cpmcal(F cpd,F qmin,F cpv,F x) { return tc_add(tc_mul(cpd, tc_sub(1.f, tc_max(x, qmin))), tc_mul(tc_max(x, qmin), cpv)); }
static inline F wsm_xlcal(thread Wsm6State &S,F xlv0,F t0c,F x) { return tc_sub(xlv0, tc_mul(S.xlv1, tc_sub(x, t0c))); }
static inline F wsm_diffus(F x, F y) { return tc_div(tc_mul(8.794e-5f, tc_exp(tc_mul(tc_log(x), 1.81f))), y); }
static inline F wsm_viscos(F x, F y) { return tc_div(tc_div(tc_mul(1.496e-6f, tc_mul(x, tc_sqrt(x))), tc_add(x, 120.f)), y); }
static inline F wsm_xka(F x, F y) { return tc_mul(tc_mul(1.414e3f, wsm_viscos(x, y)), y); }
static inline F wsm_diffac(F rv,F a, F b, F c, F d, F e) { return tc_add(tc_div(tc_mul(tc_mul(d, a), a), tc_mul(tc_mul(tc_mul(wsm_xka(c, d), rv), c), c)), tc_div(1.f, tc_mul(e, wsm_diffus(c, b)))); }
static inline F wsm_venfac(F den0,F a, F b, F c) { return tc_mul(tc_div(tc_exp(tc_mul(tc_log(tc_div(wsm_viscos(b, c), wsm_diffus(b, a))), .3333333f)), tc_sqrt(wsm_viscos(b, c))), tc_sqrt(tc_sqrt(tc_div(den0, c)))); }
static inline F wsm_conden(F qmin,F rv,F a, F b, F c, F d, F e) { return tc_div(tc_sub(tc_max(b, qmin), c), tc_add(1.f, tc_div(tc_mul(tc_div(tc_mul(d, d), tc_mul(rv, e)), c), tc_mul(a, a)))); }
static inline int wsm_nint(F x){return int(round(as_type<float>(x.bits)));}
static inline F fpvs(thread Wsm6State &S, F t, int ice, F rd, F rv, F cvap, F cliq, F cice, F hvap, F hsub, F psat, F t0c) {
F tr;
F ttp;
F dldt;
F dldti;
F xa;
F xb;
F xai;
F xbi;
F result;
  ttp = tc_add(t0c, 0.01f); // mp_wsm6.F90:1508
  dldt = tc_sub(cvap, cliq); // mp_wsm6.F90:1509
  xa = tc_div((-dldt), rv); // mp_wsm6.F90:1510
  xb = tc_add(xa, tc_div(hvap, tc_mul(rv, ttp))); // mp_wsm6.F90:1511
  dldti = tc_sub(cvap, cice); // mp_wsm6.F90:1512
  xai = tc_div((-dldti), rv); // mp_wsm6.F90:1513
  xbi = tc_add(xai, tc_div(hsub, tc_mul(rv, ttp))); // mp_wsm6.F90:1514
  tr = tc_div(ttp, t); // mp_wsm6.F90:1515
  if (((t < ttp) && (ice == 1))) { // mp_wsm6.F90:1516
  result = tc_mul(tc_mul(psat, tc_pow(tr, xai)), tc_exp(tc_mul(xbi, tc_sub(1.f, tr)))); // mp_wsm6.F90:1517
  } else { // mp_wsm6.F90:1518
  result = tc_mul(tc_mul(psat, tc_pow(tr, xa)), tc_exp(tc_mul(xb, tc_sub(1.f, tr)))); // mp_wsm6.F90:1519
  } // mp_wsm6.F90:1520
return result;
}
template<class P0,class P1,class P2,class P3,class P4,class P5,class P6,class P7,class P8>
static inline void slope_wsm6(thread Wsm6State &S, P0 qrs, P1 den, P2 denfac, P3 t, P4 rslope, P5 rslopeb, P6 rslope2, P7 rslope3, P8 vt, int its, int ite, int kts, int kte) {
int i;
int k;
F t0c = 273.15f;
F x;
F y;
F z;
F supcol;
device F *n0sfac=S.workspace->slope_wsm6_n0sfac;

   // mp_wsm6.F90:1549
   // mp_wsm6.F90:1550
   // mp_wsm6.F90:1551
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1553
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1554
  supcol = tc_sub(t0c, t[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1555
  n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(tc_min(tc_exp(tc_mul(S.alpha, supcol)), tc_div(S.n0smax, S.n0s)), 1.f); // mp_wsm6.F90:1559
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] <= S.qcrmin)) { // mp_wsm6.F90:1560
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopermax; // mp_wsm6.F90:1561
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rsloperbmax; // mp_wsm6.F90:1562
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rsloper2max; // mp_wsm6.F90:1563
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rsloper3max; // mp_wsm6.F90:1564
  } else { // mp_wsm6.F90:1565
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(1.f, wsm_lamdar(S,qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], den[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1566
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_pow(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], S.bvtr); // mp_wsm6.F90:1567
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1568
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1569
  } // mp_wsm6.F90:1570
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] <= S.qcrmin)) { // mp_wsm6.F90:1571
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopesmax; // mp_wsm6.F90:1572
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopesbmax; // mp_wsm6.F90:1573
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopes2max; // mp_wsm6.F90:1574
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopes3max; // mp_wsm6.F90:1575
  } else { // mp_wsm6.F90:1576
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(1.f, wsm_lamdas(S,qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], den[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1577
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_pow(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], S.bvts); // mp_wsm6.F90:1578
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1579
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1580
  } // mp_wsm6.F90:1581
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] <= S.qcrmin)) { // mp_wsm6.F90:1582
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopegmax; // mp_wsm6.F90:1583
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopegbmax; // mp_wsm6.F90:1584
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopeg2max; // mp_wsm6.F90:1585
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = S.rslopeg3max; // mp_wsm6.F90:1586
  } else { // mp_wsm6.F90:1587
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(1.f, wsm_lamdag(S,qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], den[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1588
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_pow(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], S.bvtg); // mp_wsm6.F90:1589
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1590
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1591
  } // mp_wsm6.F90:1592
  vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(tc_mul(S.pvtr, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1593
  vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(tc_mul(S.pvts, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1594
  vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(tc_mul(S.pvtg, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1595
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] <= 0.0f)) { vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.0f; } // mp_wsm6.F90:1596
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] <= 0.0f)) { vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.0f; } // mp_wsm6.F90:1597
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] <= 0.0f)) { vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.0f; } // mp_wsm6.F90:1598
   } // mp_wsm6.F90:1599
   } // mp_wsm6.F90:1600
}
template<class P0,class P1,class P2,class P3,class P4,class P5,class P6,class P7,class P8>
static inline void slope_rain(thread Wsm6State &S, P0 qrs, P1 den, P2 denfac, P3 t, P4 rslope, P5 rslopeb, P6 rslope2, P7 rslope3, P8 vt, int its, int ite, int kts, int kte) {
int i;
int k;
F t0c = 273.15f;
F x;
F y;
device F *n0sfac=S.workspace->slope_rain_n0sfac;

   // mp_wsm6.F90:1628
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1630
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1631
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= S.qcrmin)) { // mp_wsm6.F90:1632
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopermax; // mp_wsm6.F90:1633
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rsloperbmax; // mp_wsm6.F90:1634
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rsloper2max; // mp_wsm6.F90:1635
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rsloper3max; // mp_wsm6.F90:1636
  } else { // mp_wsm6.F90:1637
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(1.f, wsm_lamdar(S,qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1638
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_pow(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], S.bvtr); // mp_wsm6.F90:1639
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1640
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1641
  } // mp_wsm6.F90:1642
  vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(tc_mul(S.pvtr, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1643
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= 0.0f)) { vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.0f; } // mp_wsm6.F90:1644
   } // mp_wsm6.F90:1645
   } // mp_wsm6.F90:1646
}
template<class P0,class P1,class P2,class P3,class P4,class P5,class P6,class P7,class P8>
static inline void slope_snow(thread Wsm6State &S, P0 qrs, P1 den, P2 denfac, P3 t, P4 rslope, P5 rslopeb, P6 rslope2, P7 rslope3, P8 vt, int its, int ite, int kts, int kte) {
int i;
int k;
F t0c = 273.15f;
F x;
F y;
F z;
F supcol;
device F *n0sfac=S.workspace->slope_snow_n0sfac;

   // mp_wsm6.F90:1674
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1676
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1677
  supcol = tc_sub(t0c, t[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1678
  n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(tc_min(tc_exp(tc_mul(S.alpha, supcol)), tc_div(S.n0smax, S.n0s)), 1.f); // mp_wsm6.F90:1682
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= S.qcrmin)) { // mp_wsm6.F90:1683
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopesmax; // mp_wsm6.F90:1684
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopesbmax; // mp_wsm6.F90:1685
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopes2max; // mp_wsm6.F90:1686
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopes3max; // mp_wsm6.F90:1687
  } else { // mp_wsm6.F90:1688
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(1.f, wsm_lamdas(S,qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1689
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_pow(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], S.bvts); // mp_wsm6.F90:1690
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1691
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1692
  } // mp_wsm6.F90:1693
  vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(tc_mul(S.pvts, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1694
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= 0.0f)) { vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.0f; } // mp_wsm6.F90:1695
   } // mp_wsm6.F90:1696
   } // mp_wsm6.F90:1697
}
template<class P0,class P1,class P2,class P3,class P4,class P5,class P6,class P7,class P8>
static inline void slope_graup(thread Wsm6State &S, P0 qrs, P1 den, P2 denfac, P3 t, P4 rslope, P5 rslopeb, P6 rslope2, P7 rslope3, P8 vt, int its, int ite, int kts, int kte) {
int i;
int k;
F t0c = 273.15f;
F x;
F y;
device F *n0sfac=S.workspace->slope_graup_n0sfac;

   // mp_wsm6.F90:1725
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1727
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1728
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= S.qcrmin)) { // mp_wsm6.F90:1732
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopegmax; // mp_wsm6.F90:1733
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopegbmax; // mp_wsm6.F90:1734
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopeg2max; // mp_wsm6.F90:1735
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = S.rslopeg3max; // mp_wsm6.F90:1736
  } else { // mp_wsm6.F90:1737
  rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(1.f, wsm_lamdag(S,qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1738
  rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_pow(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], S.bvtg); // mp_wsm6.F90:1739
  rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1740
  rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1741
  } // mp_wsm6.F90:1742
  vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(tc_mul(S.pvtg, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1743
  if ((qrs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] <= 0.0f)) { vt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.0f; } // mp_wsm6.F90:1744
   } // mp_wsm6.F90:1745
   } // mp_wsm6.F90:1746
}
template<class P0,class P1,class P2,class P3,class P4,class P5,class P6>
static inline void nislfv_rain_plm(thread Wsm6State &S, int im, int km, P0 denl, P1 denfacl, P2 tkl, P3 dzl, P4 wwl, P5 rql, P6 precip, F dt, int id, int iter) {
int i;
int k;
int n;
int m;
int kk;
int kb;
int kt;
F tl;
F tl2;
F qql;
F dql;
F qqd;
F th;
F th2;
F qqh;
F dqh;
F zsum;
F qsum;
F dim;
F dip;
F c1;
F con1;
F fa1;
F fa2;
F allold;
F allnew;
F zz;
F dzamin;
F cflmax;
F decfl;
// Native km/km+1 arrays: mp_wsm6.F90:1789-1793,2036-2041.
device F *dz=S.workspace->nislfv_rain_plm_dz;

device F *ww=S.workspace->nislfv_rain_plm_ww;

device F *qq=S.workspace->nislfv_rain_plm_qq;

device F *wd=S.workspace->nislfv_rain_plm_wd;

device F *wa=S.workspace->nislfv_rain_plm_wa;

device F *was=S.workspace->nislfv_rain_plm_was;

device F *den=S.workspace->nislfv_rain_plm_den;

device F *denfac=S.workspace->nislfv_rain_plm_denfac;

device F *tk=S.workspace->nislfv_rain_plm_tk;

device F *qn=S.workspace->nislfv_rain_plm_qn;

device F *qr=S.workspace->nislfv_rain_plm_qr;

device F *tmp=S.workspace->nislfv_rain_plm_tmp;

device F *tmp1=S.workspace->nislfv_rain_plm_tmp1;

device F *tmp2=S.workspace->nislfv_rain_plm_tmp2;

device F *tmp3=S.workspace->nislfv_rain_plm_tmp3;

device F *wi=S.workspace->nislfv_rain_plm_wi;

device F *zi=S.workspace->nislfv_rain_plm_zi;

device F *za=S.workspace->nislfv_rain_plm_za;

device F *dza=S.workspace->nislfv_rain_plm_dza;

device F *qa=S.workspace->nislfv_rain_plm_qa;

device F *qmi=S.workspace->nislfv_rain_plm_qmi;

device F *qpi=S.workspace->nislfv_rain_plm_qpi;

  for(int slice0=1;slice0<=im;++slice0) precip[((slice0)-(1))*(1)] = 0.0f; // mp_wsm6.F90:1798
  for (i=1; i<=im; i+=1) { // mp_wsm6.F90:1800
  for(int slice0=1;slice0<=km;++slice0) dz[((slice0)-(1))*(1)] = dzl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:1801
  for(int slice0=1;slice0<=km;++slice0) qq[((slice0)-(1))*(1)] = rql[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:1802
  for(int slice0=1;slice0<=km;++slice0) ww[((slice0)-(1))*(1)] = wwl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:1803
  for(int slice0=1;slice0<=km;++slice0) den[((slice0)-(1))*(1)] = denl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:1804
  for(int slice0=1;slice0<=km;++slice0) denfac[((slice0)-(1))*(1)] = denfacl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:1805
  for(int slice0=1;slice0<=km;++slice0) tk[((slice0)-(1))*(1)] = tkl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:1806
  allold = 0.0f; // mp_wsm6.F90:1808
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1809
  allold = tc_add(allold, qq[((k)-(1))*(1)]); // mp_wsm6.F90:1810
   } // mp_wsm6.F90:1811
  if ((allold <= 0.0f)) { // mp_wsm6.F90:1812
  continue; // mp_wsm6.F90:1813
  } // mp_wsm6.F90:1814
  zi[((1)-(1))*(1)] = 0.0f; // mp_wsm6.F90:1817
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1818
  zi[(((k + 1))-(1))*(1)] = tc_add(zi[((k)-(1))*(1)], dz[((k)-(1))*(1)]); // mp_wsm6.F90:1819
   } // mp_wsm6.F90:1820
  for(int slice0=1;slice0<=km;++slice0) wd[((slice0)-(1))*(1)] = ww[((slice0)-(1))*(1)]; // mp_wsm6.F90:1823
  n = 1; // mp_wsm6.F90:1824
  while (true) { // mp_wsm6.F90:1825
  wi[((1)-(1))*(1)] = ww[((1)-(1))*(1)]; // mp_wsm6.F90:1828
  wi[(((km + 1))-(1))*(1)] = ww[((km)-(1))*(1)]; // mp_wsm6.F90:1829
  for (k=2; k<=km; k+=1) { // mp_wsm6.F90:1830
  wi[((k)-(1))*(1)] = tc_div(tc_add(tc_mul(ww[((k)-(1))*(1)], dz[(((k - 1))-(1))*(1)]), tc_mul(ww[(((k - 1))-(1))*(1)], dz[((k)-(1))*(1)])), tc_add(dz[(((k - 1))-(1))*(1)], dz[((k)-(1))*(1)])); // mp_wsm6.F90:1831
   } // mp_wsm6.F90:1832
  fa1 = tc_div(9.f, 16.f); // mp_wsm6.F90:1834
  fa2 = tc_div(1.f, 16.f); // mp_wsm6.F90:1835
  wi[((1)-(1))*(1)] = ww[((1)-(1))*(1)]; // mp_wsm6.F90:1836
  wi[((2)-(1))*(1)] = tc_mul(0.5f, tc_add(ww[((2)-(1))*(1)], ww[((1)-(1))*(1)])); // mp_wsm6.F90:1837
  for (k=3; k<=(km - 1); k+=1) { // mp_wsm6.F90:1838
  wi[((k)-(1))*(1)] = tc_sub(tc_mul(fa1, tc_add(ww[((k)-(1))*(1)], ww[(((k - 1))-(1))*(1)])), tc_mul(fa2, tc_add(ww[(((k + 1))-(1))*(1)], ww[(((k - 2))-(1))*(1)]))); // mp_wsm6.F90:1839
   } // mp_wsm6.F90:1840
  wi[((km)-(1))*(1)] = tc_mul(0.5f, tc_add(ww[((km)-(1))*(1)], ww[(((km - 1))-(1))*(1)])); // mp_wsm6.F90:1841
  wi[(((km + 1))-(1))*(1)] = ww[((km)-(1))*(1)]; // mp_wsm6.F90:1842
  for (k=2; k<=km; k+=1) { // mp_wsm6.F90:1845
  if ((ww[((k)-(1))*(1)] == 0.0f)) { wi[((k)-(1))*(1)] = ww[(((k - 1))-(1))*(1)]; } // mp_wsm6.F90:1846
   } // mp_wsm6.F90:1847
  con1 = 0.05f; // mp_wsm6.F90:1850
  for (k=km; k>=1; k+=(-1)) { // mp_wsm6.F90:1851
  decfl = tc_div(tc_mul(tc_sub(wi[(((k + 1))-(1))*(1)], wi[((k)-(1))*(1)]), dt), dz[((k)-(1))*(1)]); // mp_wsm6.F90:1852
  if ((decfl > con1)) { // mp_wsm6.F90:1853
  wi[((k)-(1))*(1)] = tc_sub(wi[(((k + 1))-(1))*(1)], tc_div(tc_mul(con1, dz[((k)-(1))*(1)]), dt)); // mp_wsm6.F90:1854
  } // mp_wsm6.F90:1855
   } // mp_wsm6.F90:1856
  for (k=1; k<=(km + 1); k+=1) { // mp_wsm6.F90:1858
  za[((k)-(1))*(1)] = tc_sub(zi[((k)-(1))*(1)], tc_mul(wi[((k)-(1))*(1)], dt)); // mp_wsm6.F90:1859
   } // mp_wsm6.F90:1860
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1862
  dza[((k)-(1))*(1)] = tc_sub(za[(((k + 1))-(1))*(1)], za[((k)-(1))*(1)]); // mp_wsm6.F90:1863
   } // mp_wsm6.F90:1864
  dza[(((km + 1))-(1))*(1)] = tc_sub(zi[(((km + 1))-(1))*(1)], za[(((km + 1))-(1))*(1)]); // mp_wsm6.F90:1865
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1868
  qa[((k)-(1))*(1)] = tc_div(tc_mul(qq[((k)-(1))*(1)], dz[((k)-(1))*(1)]), dza[((k)-(1))*(1)]); // mp_wsm6.F90:1869
  qr[((k)-(1))*(1)] = tc_div(qa[((k)-(1))*(1)], den[((k)-(1))*(1)]); // mp_wsm6.F90:1870
   } // mp_wsm6.F90:1871
  qa[(((km + 1))-(1))*(1)] = 0.0f; // mp_wsm6.F90:1872
  if ((n <= iter)) { // mp_wsm6.F90:1877
  slope_rain(S, qr, den, denfac, tk, tmp, tmp1, tmp2, tmp3, wa, 1, 1, 1, km); // mp_wsm6.F90:1878
  if ((n >= 2)) { for(int slice0=1;slice0<=km;++slice0) wa[((slice0)-(1))*(1)] = tc_mul(0.5f, tc_add(wa[((slice0)-(1))*(1)], was[((slice0)-(1))*(1)])); } // mp_wsm6.F90:1879
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1880
  ww[((k)-(1))*(1)] = tc_mul(0.5f, tc_add(wd[((k)-(1))*(1)], wa[((k)-(1))*(1)])); // mp_wsm6.F90:1885
   } // mp_wsm6.F90:1886
  for(int slice0=1;slice0<=km;++slice0) was[((slice0)-(1))*(1)] = wa[((slice0)-(1))*(1)]; // mp_wsm6.F90:1887
  n = (n + 1); // mp_wsm6.F90:1888
  continue; // mp_wsm6.F90:1889
  } // mp_wsm6.F90:1890
  break; }
  for (k=2; k<=km; k+=1) { // mp_wsm6.F90:1893
  dip = tc_div(tc_sub(qa[(((k + 1))-(1))*(1)], qa[((k)-(1))*(1)]), tc_add(dza[(((k + 1))-(1))*(1)], dza[((k)-(1))*(1)])); // mp_wsm6.F90:1894
  dim = tc_div(tc_sub(qa[((k)-(1))*(1)], qa[(((k - 1))-(1))*(1)]), tc_add(dza[(((k - 1))-(1))*(1)], dza[((k)-(1))*(1)])); // mp_wsm6.F90:1895
  if ((tc_mul(dip, dim) <= 0.0f)) { // mp_wsm6.F90:1896
  qmi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:1897
  qpi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:1898
  } else { // mp_wsm6.F90:1899
  qpi[((k)-(1))*(1)] = tc_add(qa[((k)-(1))*(1)], tc_mul(tc_mul(0.5f, tc_add(dip, dim)), dza[((k)-(1))*(1)])); // mp_wsm6.F90:1900
  qmi[((k)-(1))*(1)] = tc_sub(tc_mul(2.0f, qa[((k)-(1))*(1)]), qpi[((k)-(1))*(1)]); // mp_wsm6.F90:1901
  if (((qpi[((k)-(1))*(1)] < 0.0f) || (qmi[((k)-(1))*(1)] < 0.0f))) { // mp_wsm6.F90:1902
  qpi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:1903
  qmi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:1904
  } // mp_wsm6.F90:1905
  } // mp_wsm6.F90:1906
   } // mp_wsm6.F90:1907
  qpi[((1)-(1))*(1)] = qa[((1)-(1))*(1)]; // mp_wsm6.F90:1908
  qmi[((1)-(1))*(1)] = qa[((1)-(1))*(1)]; // mp_wsm6.F90:1909
  qmi[(((km + 1))-(1))*(1)] = qa[(((km + 1))-(1))*(1)]; // mp_wsm6.F90:1910
  qpi[(((km + 1))-(1))*(1)] = qa[(((km + 1))-(1))*(1)]; // mp_wsm6.F90:1911
  for(int slice0=1;slice0<=km;++slice0) qn[((slice0)-(1))*(1)] = 0.0f; // mp_wsm6.F90:1914
  kb = 1; // mp_wsm6.F90:1915
  kt = 1; // mp_wsm6.F90:1916
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1917
  kb = tc_max((kb - 1), 1); // mp_wsm6.F90:1918
  kt = tc_max((kt - 1), 1); // mp_wsm6.F90:1919
  if ((zi[((k)-(1))*(1)] >= za[(((km + 1))-(1))*(1)])) { // mp_wsm6.F90:1921
  break; // mp_wsm6.F90:1922
  } else { // mp_wsm6.F90:1923
  for (kk=kb; kk<=km; kk+=1) { // mp_wsm6.F90:1924
  if ((zi[((k)-(1))*(1)] <= za[(((kk + 1))-(1))*(1)])) { // mp_wsm6.F90:1925
  kb = kk; // mp_wsm6.F90:1926
  break; // mp_wsm6.F90:1927
  } else { // mp_wsm6.F90:1928
  continue; // mp_wsm6.F90:1929
  } // mp_wsm6.F90:1930
   }  // mp_wsm6.F90:1931
  for (kk=kt; kk<=km; kk+=1) { // mp_wsm6.F90:1932
  if ((zi[(((k + 1))-(1))*(1)] <= za[((kk)-(1))*(1)])) { // mp_wsm6.F90:1933
  kt = kk; // mp_wsm6.F90:1934
  break; // mp_wsm6.F90:1935
  } else { // mp_wsm6.F90:1936
  continue; // mp_wsm6.F90:1937
  } // mp_wsm6.F90:1938
   }  // mp_wsm6.F90:1939
  kt = (kt - 1); // mp_wsm6.F90:1940
  if ((kt == kb)) { // mp_wsm6.F90:1942
  tl = tc_div(tc_sub(zi[((k)-(1))*(1)], za[((kb)-(1))*(1)]), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:1943
  th = tc_div(tc_sub(zi[(((k + 1))-(1))*(1)], za[((kb)-(1))*(1)]), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:1944
  tl2 = tc_mul(tl, tl); // mp_wsm6.F90:1945
  th2 = tc_mul(th, th); // mp_wsm6.F90:1946
  qqd = tc_mul(0.5f, tc_sub(qpi[((kb)-(1))*(1)], qmi[((kb)-(1))*(1)])); // mp_wsm6.F90:1947
  qqh = tc_add(tc_mul(qqd, th2), tc_mul(qmi[((kb)-(1))*(1)], th)); // mp_wsm6.F90:1948
  qql = tc_add(tc_mul(qqd, tl2), tc_mul(qmi[((kb)-(1))*(1)], tl)); // mp_wsm6.F90:1949
  qn[((k)-(1))*(1)] = tc_div(tc_sub(qqh, qql), tc_sub(th, tl)); // mp_wsm6.F90:1950
  } else if ((kt > kb)) { // mp_wsm6.F90:1951
  tl = tc_div(tc_sub(zi[((k)-(1))*(1)], za[((kb)-(1))*(1)]), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:1952
  tl2 = tc_mul(tl, tl); // mp_wsm6.F90:1953
  qqd = tc_mul(0.5f, tc_sub(qpi[((kb)-(1))*(1)], qmi[((kb)-(1))*(1)])); // mp_wsm6.F90:1954
  qql = tc_add(tc_mul(qqd, tl2), tc_mul(qmi[((kb)-(1))*(1)], tl)); // mp_wsm6.F90:1955
  dql = tc_sub(qa[((kb)-(1))*(1)], qql); // mp_wsm6.F90:1956
  zsum = tc_mul(tc_sub(1.f, tl), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:1957
  qsum = tc_mul(dql, dza[((kb)-(1))*(1)]); // mp_wsm6.F90:1958
  if (((kt - kb) > 1)) { // mp_wsm6.F90:1959
  for (m=(kb + 1); m<=(kt - 1); m+=1) { // mp_wsm6.F90:1960
  zsum = tc_add(zsum, dza[((m)-(1))*(1)]); // mp_wsm6.F90:1961
  qsum = tc_add(qsum, tc_mul(qa[((m)-(1))*(1)], dza[((m)-(1))*(1)])); // mp_wsm6.F90:1962
   } // mp_wsm6.F90:1963
  } // mp_wsm6.F90:1964
  th = tc_div(tc_sub(zi[(((k + 1))-(1))*(1)], za[((kt)-(1))*(1)]), dza[((kt)-(1))*(1)]); // mp_wsm6.F90:1965
  th2 = tc_mul(th, th); // mp_wsm6.F90:1966
  qqd = tc_mul(0.5f, tc_sub(qpi[((kt)-(1))*(1)], qmi[((kt)-(1))*(1)])); // mp_wsm6.F90:1967
  dqh = tc_add(tc_mul(qqd, th2), tc_mul(qmi[((kt)-(1))*(1)], th)); // mp_wsm6.F90:1968
  zsum = tc_add(zsum, tc_mul(th, dza[((kt)-(1))*(1)])); // mp_wsm6.F90:1969
  qsum = tc_add(qsum, tc_mul(dqh, dza[((kt)-(1))*(1)])); // mp_wsm6.F90:1970
  qn[((k)-(1))*(1)] = tc_div(qsum, zsum); // mp_wsm6.F90:1971
  } // mp_wsm6.F90:1972
  continue; // mp_wsm6.F90:1973
  } // mp_wsm6.F90:1974
   }  // mp_wsm6.F90:1976
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:1979
  if (((za[((k)-(1))*(1)] < 0.0f) && (za[(((k + 1))-(1))*(1)] < 0.0f))) { // mp_wsm6.F90:1980
  precip[((i)-(1))*(1)] = tc_add(precip[((i)-(1))*(1)], tc_mul(qa[((k)-(1))*(1)], dza[((k)-(1))*(1)])); // mp_wsm6.F90:1981
  continue; // mp_wsm6.F90:1982
  } else if (((za[((k)-(1))*(1)] < 0.0f) && (za[(((k + 1))-(1))*(1)] >= 0.0f))) { // mp_wsm6.F90:1983
  precip[((i)-(1))*(1)] = tc_add(precip[((i)-(1))*(1)], tc_mul(qa[((k)-(1))*(1)], tc_sub(0.0f, za[((k)-(1))*(1)]))); // mp_wsm6.F90:1984
  break; // mp_wsm6.F90:1985
  } // mp_wsm6.F90:1986
  break; // mp_wsm6.F90:1987
   }  // mp_wsm6.F90:1988
  for(int slice0=1;slice0<=km;++slice0) rql[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))] = qn[((slice0)-(1))*(1)]; // mp_wsm6.F90:1991
   }  // mp_wsm6.F90:1992
}
template<class P0,class P1,class P2,class P3,class P4,class P5,class P6,class P7,class P8>
static inline void nislfv_rain_plm6(thread Wsm6State &S, int im, int km, P0 denl, P1 denfacl, P2 tkl, P3 dzl, P4 wwl, P5 rql, P6 rql2, P7 precip1, P8 precip2, F dt, int id, int iter) {
int i;
int ist;
int k;
int n;
int m;
int kk;
int kb;
int kt;
F tl;
F tl2;
F qql;
F dql;
F qqd;
F th;
F th2;
F qqh;
F dqh;
F zsum;
F qsum;
F dim;
F dip;
F c1;
F con1;
F fa1;
F fa2;
F allold;
F allnew;
F zz;
F dzamin;
F cflmax;
F decfl;
// Native km/km+1 arrays: mp_wsm6.F90:1789-1793,2036-2041.
device F *dz=S.workspace->nislfv_rain_plm6_dz;

device F *ww=S.workspace->nislfv_rain_plm6_ww;

device F *qq=S.workspace->nislfv_rain_plm6_qq;

device F *qq2=S.workspace->nislfv_rain_plm6_qq2;

device F *wd=S.workspace->nislfv_rain_plm6_wd;

device F *wa=S.workspace->nislfv_rain_plm6_wa;

device F *wa2=S.workspace->nislfv_rain_plm6_wa2;

device F *was=S.workspace->nislfv_rain_plm6_was;

device F *den=S.workspace->nislfv_rain_plm6_den;

device F *denfac=S.workspace->nislfv_rain_plm6_denfac;

device F *tk=S.workspace->nislfv_rain_plm6_tk;

device F *qn=S.workspace->nislfv_rain_plm6_qn;

device F *qr=S.workspace->nislfv_rain_plm6_qr;

device F *qr2=S.workspace->nislfv_rain_plm6_qr2;

device F *tmp=S.workspace->nislfv_rain_plm6_tmp;

device F *tmp1=S.workspace->nislfv_rain_plm6_tmp1;

device F *tmp2=S.workspace->nislfv_rain_plm6_tmp2;

device F *tmp3=S.workspace->nislfv_rain_plm6_tmp3;

device F *wi=S.workspace->nislfv_rain_plm6_wi;

device F *zi=S.workspace->nislfv_rain_plm6_zi;

device F *za=S.workspace->nislfv_rain_plm6_za;

device F *dza=S.workspace->nislfv_rain_plm6_dza;

device F *qa=S.workspace->nislfv_rain_plm6_qa;

device F *qa2=S.workspace->nislfv_rain_plm6_qa2;

device F *qmi=S.workspace->nislfv_rain_plm6_qmi;

device F *qpi=S.workspace->nislfv_rain_plm6_qpi;

F precip[1]; // mp_wsm6.F90:2041; im=1 per column launch
  for(int slice0=1;slice0<=im;++slice0) precip[((slice0)-(1))*(1)] = 0.0f; // mp_wsm6.F90:2045
  for(int slice0=1;slice0<=im;++slice0) precip1[((slice0)-(1))*(1)] = 0.0f; // mp_wsm6.F90:2046
  for(int slice0=1;slice0<=im;++slice0) precip2[((slice0)-(1))*(1)] = 0.0f; // mp_wsm6.F90:2047
  for (i=1; i<=im; i+=1) { // mp_wsm6.F90:2049
  for(int slice0=1;slice0<=km;++slice0) dz[((slice0)-(1))*(1)] = dzl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2050
  for(int slice0=1;slice0<=km;++slice0) qq[((slice0)-(1))*(1)] = rql[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2051
  for(int slice0=1;slice0<=km;++slice0) qq2[((slice0)-(1))*(1)] = rql2[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2052
  for(int slice0=1;slice0<=km;++slice0) ww[((slice0)-(1))*(1)] = wwl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2053
  for(int slice0=1;slice0<=km;++slice0) den[((slice0)-(1))*(1)] = denl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2054
  for(int slice0=1;slice0<=km;++slice0) denfac[((slice0)-(1))*(1)] = denfacl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2055
  for(int slice0=1;slice0<=km;++slice0) tk[((slice0)-(1))*(1)] = tkl[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))]; // mp_wsm6.F90:2056
  allold = 0.0f; // mp_wsm6.F90:2058
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2059
  allold = tc_add(tc_add(allold, qq[((k)-(1))*(1)]), qq2[((k)-(1))*(1)]); // mp_wsm6.F90:2060
   } // mp_wsm6.F90:2061
  if ((allold <= 0.0f)) { // mp_wsm6.F90:2062
  continue; // mp_wsm6.F90:2063
  } // mp_wsm6.F90:2064
  zi[((1)-(1))*(1)] = 0.0f; // mp_wsm6.F90:2067
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2068
  zi[(((k + 1))-(1))*(1)] = tc_add(zi[((k)-(1))*(1)], dz[((k)-(1))*(1)]); // mp_wsm6.F90:2069
   } // mp_wsm6.F90:2070
  for(int slice0=1;slice0<=km;++slice0) wd[((slice0)-(1))*(1)] = ww[((slice0)-(1))*(1)]; // mp_wsm6.F90:2073
  n = 1; // mp_wsm6.F90:2074
  while (true) { // mp_wsm6.F90:2075
  wi[((1)-(1))*(1)] = ww[((1)-(1))*(1)]; // mp_wsm6.F90:2078
  wi[(((km + 1))-(1))*(1)] = ww[((km)-(1))*(1)]; // mp_wsm6.F90:2079
  for (k=2; k<=km; k+=1) { // mp_wsm6.F90:2080
  wi[((k)-(1))*(1)] = tc_div(tc_add(tc_mul(ww[((k)-(1))*(1)], dz[(((k - 1))-(1))*(1)]), tc_mul(ww[(((k - 1))-(1))*(1)], dz[((k)-(1))*(1)])), tc_add(dz[(((k - 1))-(1))*(1)], dz[((k)-(1))*(1)])); // mp_wsm6.F90:2081
   } // mp_wsm6.F90:2082
  fa1 = tc_div(9.f, 16.f); // mp_wsm6.F90:2084
  fa2 = tc_div(1.f, 16.f); // mp_wsm6.F90:2085
  wi[((1)-(1))*(1)] = ww[((1)-(1))*(1)]; // mp_wsm6.F90:2086
  wi[((2)-(1))*(1)] = tc_mul(0.5f, tc_add(ww[((2)-(1))*(1)], ww[((1)-(1))*(1)])); // mp_wsm6.F90:2087
  for (k=3; k<=(km - 1); k+=1) { // mp_wsm6.F90:2088
  wi[((k)-(1))*(1)] = tc_sub(tc_mul(fa1, tc_add(ww[((k)-(1))*(1)], ww[(((k - 1))-(1))*(1)])), tc_mul(fa2, tc_add(ww[(((k + 1))-(1))*(1)], ww[(((k - 2))-(1))*(1)]))); // mp_wsm6.F90:2089
   } // mp_wsm6.F90:2090
  wi[((km)-(1))*(1)] = tc_mul(0.5f, tc_add(ww[((km)-(1))*(1)], ww[(((km - 1))-(1))*(1)])); // mp_wsm6.F90:2091
  wi[(((km + 1))-(1))*(1)] = ww[((km)-(1))*(1)]; // mp_wsm6.F90:2092
  for (k=2; k<=km; k+=1) { // mp_wsm6.F90:2095
  if ((ww[((k)-(1))*(1)] == 0.0f)) { wi[((k)-(1))*(1)] = ww[(((k - 1))-(1))*(1)]; } // mp_wsm6.F90:2096
   } // mp_wsm6.F90:2097
  con1 = 0.05f; // mp_wsm6.F90:2100
  for (k=km; k>=1; k+=(-1)) { // mp_wsm6.F90:2101
  decfl = tc_div(tc_mul(tc_sub(wi[(((k + 1))-(1))*(1)], wi[((k)-(1))*(1)]), dt), dz[((k)-(1))*(1)]); // mp_wsm6.F90:2102
  if ((decfl > con1)) { // mp_wsm6.F90:2103
  wi[((k)-(1))*(1)] = tc_sub(wi[(((k + 1))-(1))*(1)], tc_div(tc_mul(con1, dz[((k)-(1))*(1)]), dt)); // mp_wsm6.F90:2104
  } // mp_wsm6.F90:2105
   } // mp_wsm6.F90:2106
  for (k=1; k<=(km + 1); k+=1) { // mp_wsm6.F90:2108
  za[((k)-(1))*(1)] = tc_sub(zi[((k)-(1))*(1)], tc_mul(wi[((k)-(1))*(1)], dt)); // mp_wsm6.F90:2109
   } // mp_wsm6.F90:2110
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2112
  dza[((k)-(1))*(1)] = tc_sub(za[(((k + 1))-(1))*(1)], za[((k)-(1))*(1)]); // mp_wsm6.F90:2113
   } // mp_wsm6.F90:2114
  dza[(((km + 1))-(1))*(1)] = tc_sub(zi[(((km + 1))-(1))*(1)], za[(((km + 1))-(1))*(1)]); // mp_wsm6.F90:2115
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2118
  qa[((k)-(1))*(1)] = tc_div(tc_mul(qq[((k)-(1))*(1)], dz[((k)-(1))*(1)]), dza[((k)-(1))*(1)]); // mp_wsm6.F90:2119
  qa2[((k)-(1))*(1)] = tc_div(tc_mul(qq2[((k)-(1))*(1)], dz[((k)-(1))*(1)]), dza[((k)-(1))*(1)]); // mp_wsm6.F90:2120
  qr[((k)-(1))*(1)] = tc_div(qa[((k)-(1))*(1)], den[((k)-(1))*(1)]); // mp_wsm6.F90:2121
  qr2[((k)-(1))*(1)] = tc_div(qa2[((k)-(1))*(1)], den[((k)-(1))*(1)]); // mp_wsm6.F90:2122
   } // mp_wsm6.F90:2123
  qa[(((km + 1))-(1))*(1)] = 0.0f; // mp_wsm6.F90:2124
  qa2[(((km + 1))-(1))*(1)] = 0.0f; // mp_wsm6.F90:2125
  if ((n <= iter)) { // mp_wsm6.F90:2130
  slope_snow(S, qr, den, denfac, tk, tmp, tmp1, tmp2, tmp3, wa, 1, 1, 1, km); // mp_wsm6.F90:2131
  slope_graup(S, qr2, den, denfac, tk, tmp, tmp1, tmp2, tmp3, wa2, 1, 1, 1, km); // mp_wsm6.F90:2132
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2133
  tmp[((k)-(1))*(1)] = tc_max(tc_add(qr[((k)-(1))*(1)], qr2[((k)-(1))*(1)]), 1.e-15f); // mp_wsm6.F90:2134
  if ((tmp[((k)-(1))*(1)] > 1.e-15f)) { // mp_wsm6.F90:2135
  wa[((k)-(1))*(1)] = tc_div(tc_add(tc_mul(wa[((k)-(1))*(1)], qr[((k)-(1))*(1)]), tc_mul(wa2[((k)-(1))*(1)], qr2[((k)-(1))*(1)])), tmp[((k)-(1))*(1)]); // mp_wsm6.F90:2136
  } else { // mp_wsm6.F90:2137
  wa[((k)-(1))*(1)] = 0.f; // mp_wsm6.F90:2138
  } // mp_wsm6.F90:2139
   } // mp_wsm6.F90:2140
  if ((n >= 2)) { for(int slice0=1;slice0<=km;++slice0) wa[((slice0)-(1))*(1)] = tc_mul(0.5f, tc_add(wa[((slice0)-(1))*(1)], was[((slice0)-(1))*(1)])); } // mp_wsm6.F90:2141
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2142
  ww[((k)-(1))*(1)] = tc_mul(0.5f, tc_add(wd[((k)-(1))*(1)], wa[((k)-(1))*(1)])); // mp_wsm6.F90:2148
   } // mp_wsm6.F90:2149
  for(int slice0=1;slice0<=km;++slice0) was[((slice0)-(1))*(1)] = wa[((slice0)-(1))*(1)]; // mp_wsm6.F90:2150
  n = (n + 1); // mp_wsm6.F90:2151
  continue; // mp_wsm6.F90:2152
  } // mp_wsm6.F90:2153
  break; }
  for (ist=1; ist<=2; ist+=1) { // mp_wsm6.F90:2155
  if ((ist == 2)) { // mp_wsm6.F90:2156
  for(int slice0=1;slice0<=(km + 1);++slice0) qa[((slice0)-(1))*(1)] = qa2[((slice0)-(1))*(1)]; // mp_wsm6.F90:2157
  } // mp_wsm6.F90:2158
  precip[((i)-(1))*(1)] = 0.f; // mp_wsm6.F90:2160
  for (k=2; k<=km; k+=1) { // mp_wsm6.F90:2163
  dip = tc_div(tc_sub(qa[(((k + 1))-(1))*(1)], qa[((k)-(1))*(1)]), tc_add(dza[(((k + 1))-(1))*(1)], dza[((k)-(1))*(1)])); // mp_wsm6.F90:2164
  dim = tc_div(tc_sub(qa[((k)-(1))*(1)], qa[(((k - 1))-(1))*(1)]), tc_add(dza[(((k - 1))-(1))*(1)], dza[((k)-(1))*(1)])); // mp_wsm6.F90:2165
  if ((tc_mul(dip, dim) <= 0.0f)) { // mp_wsm6.F90:2166
  qmi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:2167
  qpi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:2168
  } else { // mp_wsm6.F90:2169
  qpi[((k)-(1))*(1)] = tc_add(qa[((k)-(1))*(1)], tc_mul(tc_mul(0.5f, tc_add(dip, dim)), dza[((k)-(1))*(1)])); // mp_wsm6.F90:2170
  qmi[((k)-(1))*(1)] = tc_sub(tc_mul(2.0f, qa[((k)-(1))*(1)]), qpi[((k)-(1))*(1)]); // mp_wsm6.F90:2171
  if (((qpi[((k)-(1))*(1)] < 0.0f) || (qmi[((k)-(1))*(1)] < 0.0f))) { // mp_wsm6.F90:2172
  qpi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:2173
  qmi[((k)-(1))*(1)] = qa[((k)-(1))*(1)]; // mp_wsm6.F90:2174
  } // mp_wsm6.F90:2175
  } // mp_wsm6.F90:2176
   } // mp_wsm6.F90:2177
  qpi[((1)-(1))*(1)] = qa[((1)-(1))*(1)]; // mp_wsm6.F90:2178
  qmi[((1)-(1))*(1)] = qa[((1)-(1))*(1)]; // mp_wsm6.F90:2179
  qmi[(((km + 1))-(1))*(1)] = qa[(((km + 1))-(1))*(1)]; // mp_wsm6.F90:2180
  qpi[(((km + 1))-(1))*(1)] = qa[(((km + 1))-(1))*(1)]; // mp_wsm6.F90:2181
  for(int slice0=1;slice0<=km;++slice0) qn[((slice0)-(1))*(1)] = 0.0f; // mp_wsm6.F90:2184
  kb = 1; // mp_wsm6.F90:2185
  kt = 1; // mp_wsm6.F90:2186
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2187
  kb = tc_max((kb - 1), 1); // mp_wsm6.F90:2188
  kt = tc_max((kt - 1), 1); // mp_wsm6.F90:2189
  if ((zi[((k)-(1))*(1)] >= za[(((km + 1))-(1))*(1)])) { // mp_wsm6.F90:2191
  break; // mp_wsm6.F90:2192
  } else { // mp_wsm6.F90:2193
  for (kk=kb; kk<=km; kk+=1) { // mp_wsm6.F90:2194
  if ((zi[((k)-(1))*(1)] <= za[(((kk + 1))-(1))*(1)])) { // mp_wsm6.F90:2195
  kb = kk; // mp_wsm6.F90:2196
  break; // mp_wsm6.F90:2197
  } else { // mp_wsm6.F90:2198
  continue; // mp_wsm6.F90:2199
  } // mp_wsm6.F90:2200
   }  // mp_wsm6.F90:2201
  for (kk=kt; kk<=km; kk+=1) { // mp_wsm6.F90:2202
  if ((zi[(((k + 1))-(1))*(1)] <= za[((kk)-(1))*(1)])) { // mp_wsm6.F90:2203
  kt = kk; // mp_wsm6.F90:2204
  break; // mp_wsm6.F90:2205
  } else { // mp_wsm6.F90:2206
  continue; // mp_wsm6.F90:2207
  } // mp_wsm6.F90:2208
   }  // mp_wsm6.F90:2209
  kt = (kt - 1); // mp_wsm6.F90:2210
  if ((kt == kb)) { // mp_wsm6.F90:2212
  tl = tc_div(tc_sub(zi[((k)-(1))*(1)], za[((kb)-(1))*(1)]), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:2213
  th = tc_div(tc_sub(zi[(((k + 1))-(1))*(1)], za[((kb)-(1))*(1)]), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:2214
  tl2 = tc_mul(tl, tl); // mp_wsm6.F90:2215
  th2 = tc_mul(th, th); // mp_wsm6.F90:2216
  qqd = tc_mul(0.5f, tc_sub(qpi[((kb)-(1))*(1)], qmi[((kb)-(1))*(1)])); // mp_wsm6.F90:2217
  qqh = tc_add(tc_mul(qqd, th2), tc_mul(qmi[((kb)-(1))*(1)], th)); // mp_wsm6.F90:2218
  qql = tc_add(tc_mul(qqd, tl2), tc_mul(qmi[((kb)-(1))*(1)], tl)); // mp_wsm6.F90:2219
  qn[((k)-(1))*(1)] = tc_div(tc_sub(qqh, qql), tc_sub(th, tl)); // mp_wsm6.F90:2220
  } else if ((kt > kb)) { // mp_wsm6.F90:2221
  tl = tc_div(tc_sub(zi[((k)-(1))*(1)], za[((kb)-(1))*(1)]), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:2222
  tl2 = tc_mul(tl, tl); // mp_wsm6.F90:2223
  qqd = tc_mul(0.5f, tc_sub(qpi[((kb)-(1))*(1)], qmi[((kb)-(1))*(1)])); // mp_wsm6.F90:2224
  qql = tc_add(tc_mul(qqd, tl2), tc_mul(qmi[((kb)-(1))*(1)], tl)); // mp_wsm6.F90:2225
  dql = tc_sub(qa[((kb)-(1))*(1)], qql); // mp_wsm6.F90:2226
  zsum = tc_mul(tc_sub(1.f, tl), dza[((kb)-(1))*(1)]); // mp_wsm6.F90:2227
  qsum = tc_mul(dql, dza[((kb)-(1))*(1)]); // mp_wsm6.F90:2228
  if (((kt - kb) > 1)) { // mp_wsm6.F90:2229
  for (m=(kb + 1); m<=(kt - 1); m+=1) { // mp_wsm6.F90:2230
  zsum = tc_add(zsum, dza[((m)-(1))*(1)]); // mp_wsm6.F90:2231
  qsum = tc_add(qsum, tc_mul(qa[((m)-(1))*(1)], dza[((m)-(1))*(1)])); // mp_wsm6.F90:2232
   } // mp_wsm6.F90:2233
  } // mp_wsm6.F90:2234
  th = tc_div(tc_sub(zi[(((k + 1))-(1))*(1)], za[((kt)-(1))*(1)]), dza[((kt)-(1))*(1)]); // mp_wsm6.F90:2235
  th2 = tc_mul(th, th); // mp_wsm6.F90:2236
  qqd = tc_mul(0.5f, tc_sub(qpi[((kt)-(1))*(1)], qmi[((kt)-(1))*(1)])); // mp_wsm6.F90:2237
  dqh = tc_add(tc_mul(qqd, th2), tc_mul(qmi[((kt)-(1))*(1)], th)); // mp_wsm6.F90:2238
  zsum = tc_add(zsum, tc_mul(th, dza[((kt)-(1))*(1)])); // mp_wsm6.F90:2239
  qsum = tc_add(qsum, tc_mul(dqh, dza[((kt)-(1))*(1)])); // mp_wsm6.F90:2240
  qn[((k)-(1))*(1)] = tc_div(qsum, zsum); // mp_wsm6.F90:2241
  } // mp_wsm6.F90:2242
  continue; // mp_wsm6.F90:2243
  } // mp_wsm6.F90:2244
   }  // mp_wsm6.F90:2246
  for (k=1; k<=km; k+=1) { // mp_wsm6.F90:2249
  if (((za[((k)-(1))*(1)] < 0.0f) && (za[(((k + 1))-(1))*(1)] < 0.0f))) { // mp_wsm6.F90:2250
  precip[((i)-(1))*(1)] = tc_add(precip[((i)-(1))*(1)], tc_mul(qa[((k)-(1))*(1)], dza[((k)-(1))*(1)])); // mp_wsm6.F90:2251
  continue; // mp_wsm6.F90:2252
  } else if (((za[((k)-(1))*(1)] < 0.0f) && (za[(((k + 1))-(1))*(1)] >= 0.0f))) { // mp_wsm6.F90:2253
  precip[((i)-(1))*(1)] = tc_add(precip[((i)-(1))*(1)], tc_mul(qa[((k)-(1))*(1)], tc_sub(0.0f, za[((k)-(1))*(1)]))); // mp_wsm6.F90:2254
  break; // mp_wsm6.F90:2255
  } // mp_wsm6.F90:2256
  break; // mp_wsm6.F90:2257
   }  // mp_wsm6.F90:2258
  if ((ist == 1)) { // mp_wsm6.F90:2261
  for(int slice0=1;slice0<=km;++slice0) rql[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))] = qn[((slice0)-(1))*(1)]; // mp_wsm6.F90:2262
  precip1[((i)-(1))*(1)] = precip[((i)-(1))*(1)]; // mp_wsm6.F90:2263
  } else { // mp_wsm6.F90:2264
  for(int slice0=1;slice0<=km;++slice0) rql2[((i)-(1))*(1)+((slice0)-(1))*((1)*((im)-(1)+1))] = qn[((slice0)-(1))*(1)]; // mp_wsm6.F90:2265
  precip2[((i)-(1))*(1)] = precip[((i)-(1))*(1)]; // mp_wsm6.F90:2266
  } // mp_wsm6.F90:2267
   }  // mp_wsm6.F90:2268
   }  // mp_wsm6.F90:2270
}
template<class P0,class P1,class P2,class P3,class P4,class P5,class P6,class P7,class P8,class P9,class P10,class P11,class P12,class P13,class P14,class P15,class P16,class P17,class P18>
static inline void mp_wsm6_run(thread Wsm6State &S, P0 t, P1 q, P2 qc, P3 qi, P4 qr, P5 qs, P6 qg, P7 den, P8 p, P9 delz, F delt, F g, F cpd, F cpv, F rd, F rv, F t0c, F ep1, F ep2, F qmin, F xls, F xlv0, F xlf0, F den0, F denr, F cliq, F cice, F psat, P10 rain, P11 rainncv, P12 sr, P13 snow, P14 snowncv, P15 graupel, P16 graupelncv, P17 rainprod2d, P18 evapprod2d, int its, int ite, int kts, int kte, thread int &errflg) {
device F *rh=S.workspace->mp_wsm6_run_rh;

device F *qsat=S.workspace->mp_wsm6_run_qsat;

device F *rslope=S.workspace->mp_wsm6_run_rslope;

device F *rslope2=S.workspace->mp_wsm6_run_rslope2;

device F *rslope3=S.workspace->mp_wsm6_run_rslope3;

device F *rslopeb=S.workspace->mp_wsm6_run_rslopeb;

device F *qrs_tmp=S.workspace->mp_wsm6_run_qrs_tmp;

device F *falk=S.workspace->mp_wsm6_run_falk;

device F *fall=S.workspace->mp_wsm6_run_fall;

device F *work1=S.workspace->mp_wsm6_run_work1;

// Native column-sized storage: mp_wsm6.F90:331-392.
device F *fallc=S.workspace->mp_wsm6_run_fallc;

device F *falkc=S.workspace->mp_wsm6_run_falkc;

device F *work1c=S.workspace->mp_wsm6_run_work1c;

device F *work2c=S.workspace->mp_wsm6_run_work2c;

device F *workr=S.workspace->mp_wsm6_run_workr;

device F *worka=S.workspace->mp_wsm6_run_worka;

device F *den_tmp=S.workspace->mp_wsm6_run_den_tmp;

device F *delz_tmp=S.workspace->mp_wsm6_run_delz_tmp;

device F *pigen=S.workspace->mp_wsm6_run_pigen;

device F *pidep=S.workspace->mp_wsm6_run_pidep;

device F *pcond=S.workspace->mp_wsm6_run_pcond;

device F *prevp=S.workspace->mp_wsm6_run_prevp;

device F *psevp=S.workspace->mp_wsm6_run_psevp;

device F *pgevp=S.workspace->mp_wsm6_run_pgevp;

device F *psdep=S.workspace->mp_wsm6_run_psdep;

device F *pgdep=S.workspace->mp_wsm6_run_pgdep;

device F *praut=S.workspace->mp_wsm6_run_praut;

device F *psaut=S.workspace->mp_wsm6_run_psaut;

device F *pgaut=S.workspace->mp_wsm6_run_pgaut;

device F *piacr=S.workspace->mp_wsm6_run_piacr;

device F *pracw=S.workspace->mp_wsm6_run_pracw;

device F *praci=S.workspace->mp_wsm6_run_praci;

device F *pracs=S.workspace->mp_wsm6_run_pracs;

device F *psacw=S.workspace->mp_wsm6_run_psacw;

device F *psaci=S.workspace->mp_wsm6_run_psaci;

device F *psacr=S.workspace->mp_wsm6_run_psacr;

device F *pgacw=S.workspace->mp_wsm6_run_pgacw;

device F *pgaci=S.workspace->mp_wsm6_run_pgaci;

device F *pgacr=S.workspace->mp_wsm6_run_pgacr;

device F *pgacs=S.workspace->mp_wsm6_run_pgacs;

device F *paacw=S.workspace->mp_wsm6_run_paacw;

device F *psmlt=S.workspace->mp_wsm6_run_psmlt;

device F *pgmlt=S.workspace->mp_wsm6_run_pgmlt;

device F *pseml=S.workspace->mp_wsm6_run_pseml;

device F *pgeml=S.workspace->mp_wsm6_run_pgeml;

device F *qsum=S.workspace->mp_wsm6_run_qsum;

device F *xl=S.workspace->mp_wsm6_run_xl;

device F *cpm=S.workspace->mp_wsm6_run_cpm;

device F *work2=S.workspace->mp_wsm6_run_work2;

device F *denfac=S.workspace->mp_wsm6_run_denfac;

device F *xni=S.workspace->mp_wsm6_run_xni;

device F *denqrs1=S.workspace->mp_wsm6_run_denqrs1;

device F *denqrs2=S.workspace->mp_wsm6_run_denqrs2;

device F *denqrs3=S.workspace->mp_wsm6_run_denqrs3;

device F *denqci=S.workspace->mp_wsm6_run_denqci;

device F *n0sfac=S.workspace->mp_wsm6_run_n0sfac;

F delqrs1[(1)];
F delqrs2[(1)];
F delqrs3[(1)];
F delqi[(1)];
F tstepsnow[(1)];
F tstepgraup[(1)];
int mstep[(1)];
int numdt[(1)];
bool flgcld[(1)];
F x;
F y;
F z;
F a;
F b;
F c;
F d;
F e;
F qdt;
F holdrr;
F holdrs;
F holdrg;
F supcol;
F supcolt;
F pvt;
F coeres;
F supsat;
F dtcld;
F xmi;
F eacrs;
F satdt;
F qimax;
F diameter;
F xni0;
F roqi0;
F fallsum;
F fallsum_qsi;
F fallsum_qg;
F vt2i;
F vt2r;
F vt2s;
F vt2g;
F acrfac;
F egs;
F egi;
F xlwork2;
F factor;
F source;
F value;
F xlf;
F pfrzdtc;
F pfrzdtr;
F supice;
F alpha2;
F delta2;
F delta3;
F vt2ave;
F holdc;
F holdci;
int i;
int j;
int k;
int mstepmax;
int iprt;
int latd;
int lond;
int loop;
int loops;
int ifsat;
int n;
int idim;
int kdim;
F dldti;
F xb;
F xai;
F tr;
F xbi;
F xa;
F hvap;
F cvap;
F hsub;
F dldt;
F ttp;
F dvec1[(1)];
F tvec1[(1)];
F temp;
   // mp_wsm6.F90:418
   // mp_wsm6.F90:419
   // mp_wsm6.F90:425
   // mp_wsm6.F90:426
   // mp_wsm6.F90:427
   // mp_wsm6.F90:428
   // mp_wsm6.F90:429
   // mp_wsm6.F90:431
  idim = ((ite - its) + 1); // mp_wsm6.F90:434
  kdim = ((kte - kts) + 1); // mp_wsm6.F90:435
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:440
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:441
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.0f); // mp_wsm6.F90:442
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.0f); // mp_wsm6.F90:443
  qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.0f); // mp_wsm6.F90:444
  qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.0f); // mp_wsm6.F90:445
  qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.0f); // mp_wsm6.F90:446
   } // mp_wsm6.F90:447
   } // mp_wsm6.F90:448
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:454
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:455
  cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = wsm_cpmcal(cpd,qmin,cpv,q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:456
  xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = wsm_xlcal(S,xlv0,t0c,t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:457
   } // mp_wsm6.F90:458
   } // mp_wsm6.F90:459
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:460
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:461
  delz_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = delz[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:462
  den_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:463
   } // mp_wsm6.F90:464
   } // mp_wsm6.F90:465
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:470
  rainncv[((i)-(its))*(1)] = 0.f; // mp_wsm6.F90:471
  if (((snowncv != nullptr) && (snow != nullptr))) { snowncv[((i)-(its))*(1)] = 0.f; } // mp_wsm6.F90:472
  if (((graupelncv != nullptr) && (graupel != nullptr))) { graupelncv[((i)-(its))*(1)] = 0.f; } // mp_wsm6.F90:473
  sr[((i)-(its))*(1)] = 0.f; // mp_wsm6.F90:474
  tstepsnow[((i)-(its))*(1)] = 0.f; // mp_wsm6.F90:476
  tstepgraup[((i)-(its))*(1)] = 0.f; // mp_wsm6.F90:477
   } // mp_wsm6.F90:478
  loops = tc_max(wsm_nint(tc_div(delt, S.dtcldcr)), 1); // mp_wsm6.F90:483
  dtcld = tc_div(delt, loops); // mp_wsm6.F90:484
  if ((delt <= S.dtcldcr)) { dtcld = delt; } // mp_wsm6.F90:485
  for (loop=1; loop<=loops; loop+=1) { // mp_wsm6.F90:487
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:492
  mstep[((i)-(its))*(1)] = 1; // mp_wsm6.F90:493
  flgcld[((i)-(its))*(1)] = true; // mp_wsm6.F90:494
   } // mp_wsm6.F90:495
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:502
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:503
  dvec1[((i)-(its))*(1)] = den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:504
   } // mp_wsm6.F90:505
  for(int vi=0; vi<((ite - its) + 1); ++vi) tvec1[vi] = tc_div(1.0f, dvec1[vi]); // mp_wsm6.F90:506
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:507
  tvec1[((i)-(its))*(1)] = tc_mul(tvec1[((i)-(its))*(1)], den0); // mp_wsm6.F90:508
   } // mp_wsm6.F90:509
  for(int vi=0; vi<((ite - its) + 1); ++vi) dvec1[vi] = tc_sqrt(tvec1[vi]); // mp_wsm6.F90:510
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:511
  denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = dvec1[((i)-(its))*(1)]; // mp_wsm6.F90:512
   } // mp_wsm6.F90:513
   } // mp_wsm6.F90:514
  hsub = xls; // mp_wsm6.F90:519
  hvap = xlv0; // mp_wsm6.F90:520
  cvap = cpv; // mp_wsm6.F90:521
  ttp = tc_add(t0c, 0.01f); // mp_wsm6.F90:522
  dldt = tc_sub(cvap, cliq); // mp_wsm6.F90:523
  xa = tc_div((-dldt), rv); // mp_wsm6.F90:524
  xb = tc_add(xa, tc_div(hvap, tc_mul(rv, ttp))); // mp_wsm6.F90:525
  dldti = tc_sub(cvap, cice); // mp_wsm6.F90:526
  xai = tc_div((-dldti), rv); // mp_wsm6.F90:527
  xbi = tc_add(xai, tc_div(hsub, tc_mul(rv, ttp))); // mp_wsm6.F90:528
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:529
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:530
  tr = tc_div(ttp, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:531
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(tc_mul(psat, tc_exp(tc_mul(tc_log(tr), xa))), tc_exp(tc_mul(xb, tc_sub(1.f, tr)))); // mp_wsm6.F90:532
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_min(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], tc_mul(0.99f, p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:533
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(tc_mul(ep2, qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_sub(p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:534
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_max(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], qmin); // mp_wsm6.F90:535
  rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_max(tc_div(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), qmin); // mp_wsm6.F90:536
  tr = tc_div(ttp, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:537
  if ((t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] < ttp)) { // mp_wsm6.F90:538
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(tc_mul(psat, tc_exp(tc_mul(tc_log(tr), xai))), tc_exp(tc_mul(xbi, tc_sub(1.f, tr)))); // mp_wsm6.F90:539
  } else { // mp_wsm6.F90:540
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(tc_mul(psat, tc_exp(tc_mul(tc_log(tr), xa))), tc_exp(tc_mul(xb, tc_sub(1.f, tr)))); // mp_wsm6.F90:541
  } // mp_wsm6.F90:542
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_min(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], tc_mul(0.99f, p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:543
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(tc_mul(ep2, qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_sub(p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:544
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_max(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], qmin); // mp_wsm6.F90:545
  rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_max(tc_div(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), qmin); // mp_wsm6.F90:546
   } // mp_wsm6.F90:547
   } // mp_wsm6.F90:548
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:554
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:555
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:556
  psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:557
  pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:558
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:559
  psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:560
  pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:561
  pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:562
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:563
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:564
  psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:565
  psacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:566
  pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:567
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:568
  pgacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:569
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:570
  pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:571
  pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:572
  pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:573
  pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:574
  pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:575
  pcond[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:576
  psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:577
  pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:578
  pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:579
  pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:580
  psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:581
  pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:582
  falk[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.f; // mp_wsm6.F90:583
  falk[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.f; // mp_wsm6.F90:584
  falk[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.f; // mp_wsm6.F90:585
  fall[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.f; // mp_wsm6.F90:586
  fall[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.f; // mp_wsm6.F90:587
  fall[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = 0.f; // mp_wsm6.F90:588
  fallc[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:589
  falkc[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:590
  xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 1.e3f; // mp_wsm6.F90:591
   } // mp_wsm6.F90:592
   } // mp_wsm6.F90:593
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:597
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:598
  temp = tc_mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_max(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qmin)); // mp_wsm6.F90:599
  temp = tc_sqrt(tc_sqrt(tc_mul(tc_mul(temp, temp), temp))); // mp_wsm6.F90:600
  xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_max(tc_mul(5.38e7f, temp), 1.e3f), 1.e6f); // mp_wsm6.F90:601
   } // mp_wsm6.F90:602
   } // mp_wsm6.F90:603
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:609
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:610
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:611
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:612
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:613
   } // mp_wsm6.F90:614
   } // mp_wsm6.F90:615
  slope_wsm6(S, qrs_tmp, den_tmp, denfac, t, rslope, rslopeb, rslope2, rslope3, work1, its, ite, kts, kte); // mp_wsm6.F90:616
  for (k=kte; k>=kts; k+=(-1)) { // mp_wsm6.F90:619
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:620
  workr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]; // mp_wsm6.F90:621
  qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(tc_add(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 1.e-15f); // mp_wsm6.F90:622
  if ((qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] > 1.e-15f)) { // mp_wsm6.F90:623
  worka[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_add(tc_mul(work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), tc_mul(work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:624
  } else { // mp_wsm6.F90:626
  worka[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:627
  } // mp_wsm6.F90:628
  denqrs1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:629
  denqrs2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:630
  denqrs3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:631
  if ((qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] <= 0.0f)) { workr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.0f; } // mp_wsm6.F90:632
   } // mp_wsm6.F90:633
   } // mp_wsm6.F90:634
  nislfv_rain_plm(S, idim, kdim, den_tmp, denfac, t, delz_tmp, workr, denqrs1, delqrs1, dtcld, 1, 1); // mp_wsm6.F90:635
  nislfv_rain_plm6(S, idim, kdim, den_tmp, denfac, t, delz_tmp, worka, denqrs2, denqrs3, delqrs2, delqrs3, dtcld, 1, 1); // mp_wsm6.F90:637
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:639
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:640
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_div(denqrs1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.f); // mp_wsm6.F90:641
  qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_div(denqrs2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.f); // mp_wsm6.F90:642
  qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_div(denqrs3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.f); // mp_wsm6.F90:643
  fall[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(tc_mul(denqrs1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], workr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), delz[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:644
  fall[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(tc_mul(denqrs2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], worka[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), delz[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:645
  fall[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(tc_mul(denqrs3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], worka[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), delz[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:646
   } // mp_wsm6.F90:647
   } // mp_wsm6.F90:648
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:649
  fall[((i)-(its))*(1)+((1)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(tc_div(delqrs1[((i)-(its))*(1)], delz[((i)-(its))*(1)+((1)-(1))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:650
  fall[((i)-(its))*(1)+((1)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(tc_div(delqrs2[((i)-(its))*(1)], delz[((i)-(its))*(1)+((1)-(1))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:651
  fall[((i)-(its))*(1)+((1)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(tc_div(delqrs3[((i)-(its))*(1)], delz[((i)-(its))*(1)+((1)-(1))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:652
   } // mp_wsm6.F90:653
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:654
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:655
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:656
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:657
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:658
   } // mp_wsm6.F90:659
   } // mp_wsm6.F90:660
  slope_wsm6(S, qrs_tmp, den_tmp, denfac, t, rslope, rslopeb, rslope2, rslope3, work1, its, ite, kts, kte); // mp_wsm6.F90:661
  for (k=kte; k>=kts; k+=(-1)) { // mp_wsm6.F90:664
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:665
  supcol = tc_sub(t0c, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:666
  n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(tc_min(tc_exp(tc_mul(S.alpha, supcol)), tc_div(S.n0smax, S.n0s)), 1.f); // mp_wsm6.F90:667
  if ((t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > t0c)) { // mp_wsm6.F90:668
  xlf = xlf0; // mp_wsm6.F90:673
  work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = wsm_venfac(den0,p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:674
  if ((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { // mp_wsm6.F90:675
  coeres = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], tc_sqrt(tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:676
  psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_mul(tc_mul(tc_div(tc_mul(tc_mul(tc_div(wsm_xka(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), xlf), tc_sub(t0c, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), S.pi), 2.f), n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_add(tc_mul(S.precs1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(S.precs2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:677
  psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_max(tc_div(tc_mul(psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], dtcld), mstep[((i)-(its))*(1)]), tc_div((-qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), mstep[((i)-(its))*(1)])), 0.f); // mp_wsm6.F90:680
  qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:682
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_sub(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:683
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_div(xlf, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:684
  } // mp_wsm6.F90:685
  if ((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { // mp_wsm6.F90:690
  coeres = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], tc_sqrt(tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:691
  pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_mul(tc_mul(tc_div(wsm_xka(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), xlf), tc_sub(t0c, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), tc_add(tc_mul(S.precg1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(S.precg2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:692
  pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_max(tc_div(tc_mul(pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], dtcld), mstep[((i)-(its))*(1)]), tc_div((-qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), mstep[((i)-(its))*(1)])), 0.f); // mp_wsm6.F90:695
  qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:697
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_sub(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:698
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_div(xlf, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgmlt[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:699
  } // mp_wsm6.F90:700
  } // mp_wsm6.F90:701
   } // mp_wsm6.F90:702
   } // mp_wsm6.F90:703
  for (k=kte; k>=kts; k+=(-1)) { // mp_wsm6.F90:707
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:708
  if ((qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] <= 0.f)) { // mp_wsm6.F90:709
  work1c[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:710
  } else { // mp_wsm6.F90:711
  xmi = tc_div(tc_mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:712
  diameter = tc_max(tc_min(tc_mul(S.dicon, tc_sqrt(xmi)), S.dimax), 1.e-25f); // mp_wsm6.F90:713
  work1c[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(1.49e4f, tc_exp(tc_mul(tc_log(diameter), 1.31f))); // mp_wsm6.F90:714
  } // mp_wsm6.F90:715
   } // mp_wsm6.F90:716
   } // mp_wsm6.F90:717
  for (k=kte; k>=kts; k+=(-1)) { // mp_wsm6.F90:721
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:722
  denqci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:723
   } // mp_wsm6.F90:724
   } // mp_wsm6.F90:725
  nislfv_rain_plm(S, idim, kdim, den_tmp, denfac, t, delz_tmp, work1c, denqci, delqi, dtcld, 1, 0); // mp_wsm6.F90:726
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:728
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:729
  qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_div(denqci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.f); // mp_wsm6.F90:730
   } // mp_wsm6.F90:731
   } // mp_wsm6.F90:732
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:733
  fallc[((i)-(its))*(1)+((1)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_div(delqi[((i)-(its))*(1)], delz[((i)-(its))*(1)+((1)-(1))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:734
   } // mp_wsm6.F90:735
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:740
  fallsum = tc_add(tc_add(tc_add(fall[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], fall[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), fall[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), fallc[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:741
  fallsum_qsi = tc_add(fall[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], fallc[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:742
  fallsum_qg = fall[((i)-(its))*(1)+((kts)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]; // mp_wsm6.F90:743
  if ((fallsum > 0.f)) { // mp_wsm6.F90:744
  rainncv[((i)-(its))*(1)] = tc_add(tc_mul(tc_mul(tc_div(tc_mul(fallsum, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), rainncv[((i)-(its))*(1)]); // mp_wsm6.F90:745
  rain[((i)-(its))*(1)] = tc_add(tc_mul(tc_mul(tc_div(tc_mul(fallsum, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), rain[((i)-(its))*(1)]); // mp_wsm6.F90:746
  } // mp_wsm6.F90:747
  if ((fallsum_qsi > 0.f)) { // mp_wsm6.F90:748
  tstepsnow[((i)-(its))*(1)] = tc_add(tc_mul(tc_mul(tc_div(tc_mul(fallsum_qsi, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), tstepsnow[((i)-(its))*(1)]); // mp_wsm6.F90:749
  if (((snowncv != nullptr) && (snow != nullptr))) { // mp_wsm6.F90:751
  snowncv[((i)-(its))*(1)] = tc_add(tc_mul(tc_mul(tc_div(tc_mul(fallsum_qsi, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), snowncv[((i)-(its))*(1)]); // mp_wsm6.F90:752
  snow[((i)-(its))*(1)] = tc_add(tc_mul(tc_mul(tc_div(tc_mul(fallsum_qsi, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), snow[((i)-(its))*(1)]); // mp_wsm6.F90:754
  } // mp_wsm6.F90:755
  } // mp_wsm6.F90:756
  if ((fallsum_qg > 0.f)) { // mp_wsm6.F90:757
  tstepgraup[((i)-(its))*(1)] = tc_add(tc_mul(tc_mul(tc_div(tc_mul(fallsum_qg, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), tstepgraup[((i)-(its))*(1)]); // mp_wsm6.F90:758
  if (((graupelncv != nullptr) && (graupel != nullptr))) { // mp_wsm6.F90:760
  graupelncv[((i)-(its))*(1)] = tc_add(tc_mul(tc_mul(tc_div(tc_mul(fallsum_qg, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), graupelncv[((i)-(its))*(1)]); // mp_wsm6.F90:761
  graupel[((i)-(its))*(1)] = tc_add(tc_mul(tc_mul(tc_div(tc_mul(fallsum_qg, delz[((i)-(its))*(1)+((kts)-(1))*((1)*((ite)-(its)+1))]), denr), dtcld), 1000.f), graupel[((i)-(its))*(1)]); // mp_wsm6.F90:763
  } // mp_wsm6.F90:764
  } // mp_wsm6.F90:765
  if ((snowncv != nullptr)) { // mp_wsm6.F90:766
  if ((fallsum > 0.f)) { sr[((i)-(its))*(1)] = tc_div(tc_add(snowncv[((i)-(its))*(1)], graupelncv[((i)-(its))*(1)]), tc_add(rainncv[((i)-(its))*(1)], 1.e-12f)); } // mp_wsm6.F90:767
  } else { // mp_wsm6.F90:768
  if ((fallsum > 0.f)) { sr[((i)-(its))*(1)] = tc_div(tc_add(tstepsnow[((i)-(its))*(1)], tstepgraup[((i)-(its))*(1)]), tc_add(rainncv[((i)-(its))*(1)], 1.e-12f)); } // mp_wsm6.F90:769
  } // mp_wsm6.F90:770
   } // mp_wsm6.F90:771
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:777
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:778
  supcol = tc_sub(t0c, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:779
  xlf = tc_sub(xls, xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:780
  if ((supcol < 0.f)) { xlf = xlf0; } // mp_wsm6.F90:781
  if (((supcol < 0) && (qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f))) { // mp_wsm6.F90:782
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:783
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_sub(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_div(xlf, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:784
  qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:785
  } // mp_wsm6.F90:786
  if (((supcol > 40.f) && (qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f))) { // mp_wsm6.F90:791
  qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:792
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_div(xlf, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:793
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:794
  } // mp_wsm6.F90:795
  if (((supcol > 0.f) && (qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > qmin))) { // mp_wsm6.F90:800
  supcolt = tc_min(supcol, 50.f); // mp_wsm6.F90:803
  pfrzdtc = tc_min(tc_mul(tc_mul(tc_mul(tc_div(tc_div(tc_mul(tc_mul(S.pfrz1, tc_sub(tc_exp(tc_mul(S.pfrz2, supcolt)), 1.f)), den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), denr), S.xncr), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:804
  qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pfrzdtc); // mp_wsm6.F90:806
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_div(xlf, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pfrzdtc)); // mp_wsm6.F90:807
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_sub(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pfrzdtc); // mp_wsm6.F90:808
  } // mp_wsm6.F90:809
  if (((supcol > 0.f) && (qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f))) { // mp_wsm6.F90:814
  temp = rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]; // mp_wsm6.F90:818
  temp = tc_mul(tc_mul(temp, temp), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:819
  supcolt = tc_min(supcol, 50.f); // mp_wsm6.F90:820
  pfrzdtr = tc_min(tc_mul(tc_mul(tc_mul(tc_div(tc_mul(tc_mul(tc_mul(tc_mul(20.f, tc_mul(S.pi, S.pi)), S.pfrz1), S.n0r), denr), den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), tc_sub(tc_exp(tc_mul(S.pfrz2, supcolt)), 1.f)), temp), dtcld), qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:821
  qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pfrzdtr); // mp_wsm6.F90:824
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_div(xlf, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pfrzdtr)); // mp_wsm6.F90:825
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_sub(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pfrzdtr); // mp_wsm6.F90:826
  } // mp_wsm6.F90:827
   } // mp_wsm6.F90:828
   } // mp_wsm6.F90:829
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:835
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:836
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:837
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:838
  qrs_tmp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]; // mp_wsm6.F90:839
   } // mp_wsm6.F90:840
   } // mp_wsm6.F90:841
  slope_wsm6(S, qrs_tmp, den_tmp, denfac, t, rslope, rslopeb, rslope2, rslope3, work1, its, ite, kts, kte); // mp_wsm6.F90:842
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:850
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:851
  work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = wsm_diffac(rv,xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:852
  work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = wsm_diffac(rv,xls, p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:853
  work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = wsm_venfac(den0,p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:854
   } // mp_wsm6.F90:855
   } // mp_wsm6.F90:856
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:866
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:867
  supsat = tc_sub(tc_max(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qmin), qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:868
  satdt = tc_div(supsat, dtcld); // mp_wsm6.F90:869
  if ((qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qc0)) { // mp_wsm6.F90:874
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(S.qck1, tc_pow(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_div(7.f, 3.f))); // mp_wsm6.F90:875
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:876
  } // mp_wsm6.F90:877
  if (((qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin) && (qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > qmin))) { // mp_wsm6.F90:882
  pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_mul(tc_mul(tc_mul(tc_mul(S.pacrr, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_div(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:883
  } // mp_wsm6.F90:885
  if ((qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { // mp_wsm6.F90:890
  coeres = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], tc_sqrt(tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:891
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_mul(tc_sub(rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], 1.f), tc_add(tc_mul(S.precr1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(S.precr2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:892
  if ((prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] < 0.f)) { // mp_wsm6.F90:894
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div((-qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:895
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(satdt, 2)); // mp_wsm6.F90:896
  } else { // mp_wsm6.F90:897
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(satdt, 2)); // mp_wsm6.F90:898
  } // mp_wsm6.F90:899
  } // mp_wsm6.F90:900
   } // mp_wsm6.F90:901
   } // mp_wsm6.F90:902
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:917
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:918
  supcol = tc_sub(t0c, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:919
  n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(tc_min(tc_exp(tc_mul(S.alpha, supcol)), tc_div(S.n0smax, S.n0s)), 1.f); // mp_wsm6.F90:920
  supsat = tc_sub(tc_max(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qmin), qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:921
  satdt = tc_div(supsat, dtcld); // mp_wsm6.F90:922
  ifsat = 0; // mp_wsm6.F90:923
  temp = tc_mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_max(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qmin)); // mp_wsm6.F90:929
  temp = tc_sqrt(tc_sqrt(tc_mul(tc_mul(temp, temp), temp))); // mp_wsm6.F90:930
  xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_max(tc_mul(5.38e7f, temp), 1.e3f), 1.e6f); // mp_wsm6.F90:931
  eacrs = tc_exp(tc_mul(0.07f, (-supcol))); // mp_wsm6.F90:932
  xmi = tc_div(tc_mul(den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:934
  diameter = tc_min(tc_mul(S.dicon, tc_sqrt(xmi)), S.dimax); // mp_wsm6.F90:935
  vt2i = tc_mul(1.49e4f, tc_pow(diameter, 1.31f)); // mp_wsm6.F90:936
  vt2r = tc_mul(tc_mul(S.pvtr, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:937
  vt2s = tc_mul(tc_mul(S.pvts, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:938
  vt2g = tc_mul(tc_mul(S.pvtg, rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:939
  qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(tc_add(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 1.e-15f); // mp_wsm6.F90:940
  if ((qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] > 1.e-15f)) { // mp_wsm6.F90:941
  vt2ave = tc_div(tc_add(tc_mul(vt2s, qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), tc_mul(vt2g, qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:942
  } else { // mp_wsm6.F90:943
  vt2ave = 0.f; // mp_wsm6.F90:944
  } // mp_wsm6.F90:945
  if (((supcol > 0) && (qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > qmin))) { // mp_wsm6.F90:946
  if ((qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin)) { // mp_wsm6.F90:947
  acrfac = tc_add(tc_add(tc_mul(2.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(2.f, diameter), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])), tc_mul(ipow<2>(diameter), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:952
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_mul(tc_mul(tc_mul(tc_mul(S.pi, qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), S.n0r), tc_abs(tc_sub(vt2r, vt2i))), acrfac), 4.f); // mp_wsm6.F90:954
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], ipow<2>(tc_min(tc_max(tc_div(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))); // mp_wsm6.F90:956
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:957
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_div(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(ipow<2>(S.pi), S.avtr), S.n0r), denr), xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), S.g6pbr), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), 24.f), den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:962
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], ipow<2>(tc_min(tc_max(tc_div(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))); // mp_wsm6.F90:966
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:967
  } // mp_wsm6.F90:968
  if ((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin)) { // mp_wsm6.F90:973
  acrfac = tc_add(tc_add(tc_mul(2.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(2.f, diameter), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])), tc_mul(ipow<2>(diameter), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:974
  psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(S.pi, qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), eacrs), S.n0s), n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_abs(tc_sub(vt2ave, vt2i))), acrfac), 4.f); // mp_wsm6.F90:976
  psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:978
  } // mp_wsm6.F90:979
  if ((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin)) { // mp_wsm6.F90:984
  egi = tc_exp(tc_mul(0.07f, (-supcol))); // mp_wsm6.F90:985
  acrfac = tc_add(tc_add(tc_mul(2.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(2.f, diameter), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])), tc_mul(ipow<2>(diameter), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:986
  pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(S.pi, egi), qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), S.n0g), tc_abs(tc_sub(vt2ave, vt2i))), acrfac), 4.f); // mp_wsm6.F90:988
  pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:989
  } // mp_wsm6.F90:990
  } // mp_wsm6.F90:991
  if (((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin) && (qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > qmin))) { // mp_wsm6.F90:996
  psacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(S.pacrc, n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), ipow<2>(tc_min(tc_max(tc_div(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_div(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:997
  } // mp_wsm6.F90:1001
  if (((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin) && (qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > qmin))) { // mp_wsm6.F90:1006
  pgacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(S.pacrg, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), ipow<2>(tc_min(tc_max(tc_div(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))), qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), denfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_div(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1007
  } // mp_wsm6.F90:1011
  if ((qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] > 1.e-15f)) { // mp_wsm6.F90:1016
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_add(tc_mul(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], psacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_mul(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], pgacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), qsum[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1017
  } // mp_wsm6.F90:1019
  if (((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin) && (qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin))) { // mp_wsm6.F90:1024
  if ((supcol > 0)) { // mp_wsm6.F90:1025
  acrfac = tc_add(tc_add(tc_mul(tc_mul(tc_mul(5.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(tc_mul(2.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])), tc_mul(tc_mul(tc_mul(.5f, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:1026
  pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(ipow<2>(S.pi), S.n0r), S.n0s), n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_abs(tc_sub(vt2r, vt2ave))), tc_div(S.dens, den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), acrfac); // mp_wsm6.F90:1029
  pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], ipow<2>(tc_min(tc_max(tc_div(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))); // mp_wsm6.F90:1032
  pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1033
  } // mp_wsm6.F90:1034
  acrfac = tc_add(tc_add(tc_mul(tc_mul(tc_mul(5.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(tc_mul(2.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])), tc_mul(tc_mul(tc_mul(.5f, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:1039
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(ipow<2>(S.pi), S.n0r), S.n0s), n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_abs(tc_sub(vt2ave, vt2r))), tc_div(denr, den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), acrfac); // mp_wsm6.F90:1042
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], ipow<2>(tc_min(tc_max(tc_div(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))); // mp_wsm6.F90:1045
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1046
  } // mp_wsm6.F90:1047
  if (((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin) && (qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin))) { // mp_wsm6.F90:1052
  acrfac = tc_add(tc_add(tc_mul(tc_mul(tc_mul(5.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(tc_mul(2.f, rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])), tc_mul(tc_mul(tc_mul(.5f, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), rslope3[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:1053
  pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(tc_mul(tc_mul(tc_mul(tc_mul(ipow<2>(S.pi), S.n0r), S.n0g), tc_abs(tc_sub(vt2ave, vt2r))), tc_div(denr, den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), acrfac); // mp_wsm6.F90:1056
  pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], ipow<2>(tc_min(tc_max(tc_div(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), 0.0f), 1.f))); // mp_wsm6.F90:1059
  pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1060
  } // mp_wsm6.F90:1061
  if (((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin) && (qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > S.qcrmin))) { // mp_wsm6.F90:1068
  pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = 0.f; // mp_wsm6.F90:1069
  } // mp_wsm6.F90:1070
  if ((supcol <= 0)) { // mp_wsm6.F90:1071
  xlf = xlf0; // mp_wsm6.F90:1072
  if ((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_max(tc_div(tc_mul(tc_mul(cliq, supcol), tc_add(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), xlf), tc_div((-qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); } // mp_wsm6.F90:1077
  if ((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_max(tc_div(tc_mul(tc_mul(cliq, supcol), tc_add(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), xlf), tc_div((-qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); } // mp_wsm6.F90:1084
  } // mp_wsm6.F90:1087
  if ((supcol > 0)) { // mp_wsm6.F90:1088
  if (((qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0) && (ifsat != 1))) { // mp_wsm6.F90:1093
  pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_mul(tc_mul(tc_mul(4.f, diameter), xni[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_sub(rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], 1.f)), work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1094
  supice = tc_sub(satdt, prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1095
  if ((pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] < 0.f)) { // mp_wsm6.F90:1096
  pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(tc_max(pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(satdt, 2)), supice); // mp_wsm6.F90:1097
  pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div((-qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:1098
  } else { // mp_wsm6.F90:1099
  pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_min(pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(satdt, 2)), supice); // mp_wsm6.F90:1100
  } // mp_wsm6.F90:1101
  if ((tc_abs(tc_add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])) >= tc_abs(satdt))) { ifsat = 1; } // mp_wsm6.F90:1102
  } // mp_wsm6.F90:1103
  if (((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f) && (ifsat != 1))) { // mp_wsm6.F90:1108
  coeres = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], tc_sqrt(tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:1109
  psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_mul(tc_mul(tc_sub(rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], 1.f), n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_add(tc_mul(S.precs1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(S.precs2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1110
  supice = tc_sub(tc_sub(satdt, prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1112
  if ((psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] < 0.f)) { // mp_wsm6.F90:1113
  psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div((-qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:1114
  psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(tc_max(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(satdt, 2)), supice); // mp_wsm6.F90:1115
  } else { // mp_wsm6.F90:1116
  psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_min(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(satdt, 2)), supice); // mp_wsm6.F90:1117
  } // mp_wsm6.F90:1118
  if ((tc_abs(tc_add(tc_add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])) >= tc_abs(satdt))) { ifsat = 1; } // mp_wsm6.F90:1119
  } // mp_wsm6.F90:1121
  if (((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f) && (ifsat != 1))) { // mp_wsm6.F90:1126
  coeres = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], tc_sqrt(tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:1127
  pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_mul(tc_sub(rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], 1.f), tc_add(tc_mul(S.precg1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(S.precg2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1128
  supice = tc_sub(tc_sub(tc_sub(satdt, prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1130
  if ((pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] < 0.f)) { // mp_wsm6.F90:1131
  pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div((-qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:1132
  pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(tc_max(pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(satdt, 2)), supice); // mp_wsm6.F90:1133
  } else { // mp_wsm6.F90:1134
  pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_min(pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div(satdt, 2)), supice); // mp_wsm6.F90:1135
  } // mp_wsm6.F90:1136
  if ((tc_abs(tc_add(tc_add(tc_add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])) >= tc_abs(satdt))) { ifsat = 1; } // mp_wsm6.F90:1137
  } // mp_wsm6.F90:1139
  if (((supsat > 0) && (ifsat != 1))) { // mp_wsm6.F90:1144
  supice = tc_sub(tc_sub(tc_sub(tc_sub(satdt, prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1145
  xni0 = tc_mul(1.e3f, tc_exp(tc_mul(0.1f, supcol))); // mp_wsm6.F90:1146
  roqi0 = tc_mul(4.92e-11f, tc_pow(xni0, 1.33f)); // mp_wsm6.F90:1147
  pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(tc_div(tc_sub(tc_div(roqi0, den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), tc_max(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.f)), dtcld), 0.f); // mp_wsm6.F90:1148
  pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_min(pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], satdt), supice); // mp_wsm6.F90:1149
  } // mp_wsm6.F90:1150
  if ((qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { // mp_wsm6.F90:1156
  qimax = tc_div(S.roqimax, den[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1157
  psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_max(tc_div(tc_sub(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qimax), dtcld), 0.f); // mp_wsm6.F90:1158
  } // mp_wsm6.F90:1159
  if ((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f)) { // mp_wsm6.F90:1165
  alpha2 = tc_mul(1.e-3f, tc_exp(tc_mul(0.09f, (-supcol)))); // mp_wsm6.F90:1166
  pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_max(tc_mul(alpha2, tc_sub(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], S.qs0)), 0.f), tc_div(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1167
  } // mp_wsm6.F90:1168
  } // mp_wsm6.F90:1169
  if ((supcol < 0.f)) { // mp_wsm6.F90:1175
  if (((qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f) && (rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] < 1.f))) { // mp_wsm6.F90:1176
  coeres = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], tc_sqrt(tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:1177
  psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_mul(tc_mul(tc_sub(rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], 1.f), n0sfac[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_add(tc_mul(S.precs1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(S.precs2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1178
  psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_max(psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div((-qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1181
  } // mp_wsm6.F90:1182
  if (((qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f) && (rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] < 1.f))) { // mp_wsm6.F90:1187
  coeres = tc_mul(rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], tc_sqrt(tc_mul(rslope[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], rslopeb[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]))); // mp_wsm6.F90:1188
  pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_mul(tc_sub(rh[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], 1.f), tc_add(tc_mul(S.precg1, rslope2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((3)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_mul(tc_mul(S.precg2, work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), coeres))), work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1189
  pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_max(pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_div((-qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1191
  } // mp_wsm6.F90:1192
  } // mp_wsm6.F90:1193
   } // mp_wsm6.F90:1194
   } // mp_wsm6.F90:1195
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1202
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1203
  delta2 = 0.f; // mp_wsm6.F90:1205
  delta3 = 0.f; // mp_wsm6.F90:1206
  if (((qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] < 1.e-4f) && (qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] < 1.e-4f))) { delta2 = 1.f; } // mp_wsm6.F90:1207
  if ((qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] < 1.e-4f)) { delta3 = 1.f; } // mp_wsm6.F90:1208
  if ((t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] <= t0c)) { // mp_wsm6.F90:1209
  value = tc_max(qmin, qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1213
  source = tc_mul(tc_add(tc_add(tc_add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:1214
  if ((source > value)) { // mp_wsm6.F90:1215
  factor = tc_div(value, source); // mp_wsm6.F90:1216
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1217
  pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1218
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1219
  } // mp_wsm6.F90:1220
  value = tc_max(qmin, qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1224
  source = tc_mul(tc_add(tc_add(tc_add(tc_sub(tc_sub(psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:1225
  if ((source > value)) { // mp_wsm6.F90:1227
  factor = tc_div(value, source); // mp_wsm6.F90:1228
  psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1229
  pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1230
  pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1231
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1232
  psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1233
  pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1234
  } // mp_wsm6.F90:1235
  value = tc_max(qmin, qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1239
  source = tc_mul(tc_add(tc_add(tc_add(tc_sub(tc_sub((-praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:1240
  if ((source > value)) { // mp_wsm6.F90:1242
  factor = tc_div(value, source); // mp_wsm6.F90:1243
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1244
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1245
  pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1246
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1247
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1248
  pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1249
  } // mp_wsm6.F90:1250
  value = tc_max(qmin, qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1254
  source = tc_mul((-tc_sub(tc_add(tc_add(tc_sub(tc_add(tc_add(tc_add(tc_sub(tc_add(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], delta3)), tc_mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], delta3)), tc_mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_sub(1.f, delta2))), tc_mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], delta2)), psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), dtcld); // mp_wsm6.F90:1255
  if ((source > value)) { // mp_wsm6.F90:1258
  factor = tc_div(value, source); // mp_wsm6.F90:1259
  psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1260
  psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1261
  pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1262
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1263
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1264
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1265
  psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1266
  pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1267
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1268
  pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1269
  } // mp_wsm6.F90:1270
  value = tc_max(qmin, qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1274
  source = tc_mul((-tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_sub(1.f, delta3))), tc_mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_sub(1.f, delta3))), tc_mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_sub(1.f, delta2))), tc_mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_sub(1.f, delta2))), pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), dtcld); // mp_wsm6.F90:1275
  if ((source > value)) { // mp_wsm6.F90:1279
  factor = tc_div(value, source); // mp_wsm6.F90:1280
  pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1281
  pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1282
  piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1283
  praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1284
  psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1285
  pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1286
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1287
  pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1288
  pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1289
  pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1290
  } // mp_wsm6.F90:1291
  work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = (-tc_add(tc_add(tc_add(tc_add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1293
  q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1295
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_sub(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_add(tc_add(tc_add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1296
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_add(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_sub(tc_sub(tc_sub(tc_add(tc_add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1298
  qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_sub(qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_sub(tc_sub(tc_add(tc_add(tc_add(psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1301
  qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_add(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_add(tc_sub(tc_sub(tc_add(tc_add(tc_add(tc_sub(tc_add(tc_add(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], delta3)), tc_mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], delta3)), psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_sub(1.f, delta2))), tc_mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], delta2)), dtcld)), 0.f); // mp_wsm6.F90:1304
  qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_add(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_mul(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_sub(1.f, delta3))), tc_mul(praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_sub(1.f, delta3))), tc_mul(psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_sub(1.f, delta2))), tc_mul(pracs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], tc_sub(1.f, delta2))), pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1309
  xlf = tc_sub(xls, xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1314
  xlwork2 = tc_sub(tc_sub(tc_mul((-xls), tc_add(tc_add(tc_add(psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pidep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pigen[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), tc_mul(xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), tc_mul(xlf, tc_add(tc_add(tc_add(tc_add(piacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psacr[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]))); // mp_wsm6.F90:1315
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_sub(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_div(xlwork2, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:1318
  } else { // mp_wsm6.F90:1319
  value = tc_max(qmin, qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1323
  source = tc_mul(tc_add(tc_add(tc_add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:1324
  if ((source > value)) { // mp_wsm6.F90:1325
  factor = tc_div(value, source); // mp_wsm6.F90:1326
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1327
  pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1328
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1329
  } // mp_wsm6.F90:1330
  value = tc_max(qmin, qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1334
  source = tc_mul(tc_sub(tc_sub(tc_sub(tc_add(tc_add(tc_sub((-paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:1335
  if ((source > value)) { // mp_wsm6.F90:1337
  factor = tc_div(value, source); // mp_wsm6.F90:1338
  praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1339
  prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1340
  pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1341
  paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1342
  pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1343
  pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1344
  } // mp_wsm6.F90:1345
  value = tc_max(S.qcrmin, qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1349
  source = tc_mul(tc_sub(tc_sub(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld); // mp_wsm6.F90:1350
  if ((source > value)) { // mp_wsm6.F90:1351
  factor = tc_div(value, source); // mp_wsm6.F90:1352
  pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1353
  psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1354
  pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1355
  } // mp_wsm6.F90:1356
  value = tc_max(S.qcrmin, qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1360
  source = tc_mul((-tc_add(tc_add(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), dtcld); // mp_wsm6.F90:1361
  if ((source > value)) { // mp_wsm6.F90:1362
  factor = tc_div(value, source); // mp_wsm6.F90:1363
  pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1364
  pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1365
  pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_mul(pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], factor); // mp_wsm6.F90:1366
  } // mp_wsm6.F90:1367
  work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = (-tc_add(tc_add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1369
  q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1371
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_sub(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_add(tc_add(tc_add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1372
  qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_add(qr[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_sub(tc_sub(tc_add(tc_add(tc_add(tc_add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1374
  qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_add(qs[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_add(tc_sub(psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1377
  qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_add(qg[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_add(tc_add(pgacs[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)), 0.f); // mp_wsm6.F90:1379
  xlf = tc_sub(xls, xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1381
  xlwork2 = tc_sub(tc_mul((-xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), tc_add(tc_add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])), tc_mul(xlf, tc_add(pseml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pgeml[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]))); // mp_wsm6.F90:1382
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_sub(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_div(xlwork2, cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:1384
  } // mp_wsm6.F90:1385
   } // mp_wsm6.F90:1386
   } // mp_wsm6.F90:1387
  hsub = xls; // mp_wsm6.F90:1392
  hvap = xlv0; // mp_wsm6.F90:1393
  cvap = cpv; // mp_wsm6.F90:1394
  ttp = tc_add(t0c, 0.01f); // mp_wsm6.F90:1395
  dldt = tc_sub(cvap, cliq); // mp_wsm6.F90:1396
  xa = tc_div((-dldt), rv); // mp_wsm6.F90:1397
  xb = tc_add(xa, tc_div(hvap, tc_mul(rv, ttp))); // mp_wsm6.F90:1398
  dldti = tc_sub(cvap, cice); // mp_wsm6.F90:1399
  xai = tc_div((-dldti), rv); // mp_wsm6.F90:1400
  xbi = tc_add(xai, tc_div(hsub, tc_mul(rv, ttp))); // mp_wsm6.F90:1401
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1402
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1403
  tr = tc_div(ttp, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1404
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(tc_mul(psat, tc_exp(tc_mul(tc_log(tr), xa))), tc_exp(tc_mul(xb, tc_sub(1.f, tr)))); // mp_wsm6.F90:1405
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_min(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], tc_mul(0.99f, p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1406
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(tc_mul(ep2, qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_sub(p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:1407
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_max(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], qmin); // mp_wsm6.F90:1408
  tr = tc_div(ttp, t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1409
  if ((t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] < ttp)) { // mp_wsm6.F90:1410
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(tc_mul(psat, tc_exp(tc_mul(tc_log(tr), xai))), tc_exp(tc_mul(xbi, tc_sub(1.f, tr)))); // mp_wsm6.F90:1411
  } else { // mp_wsm6.F90:1412
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_mul(tc_mul(psat, tc_exp(tc_mul(tc_log(tr), xa))), tc_exp(tc_mul(xb, tc_sub(1.f, tr)))); // mp_wsm6.F90:1413
  } // mp_wsm6.F90:1414
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_min(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], tc_mul(0.99f, p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1415
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_div(tc_mul(ep2, qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]), tc_sub(p[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))])); // mp_wsm6.F90:1416
  qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = tc_max(qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((2)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], qmin); // mp_wsm6.F90:1417
   } // mp_wsm6.F90:1418
   } // mp_wsm6.F90:1419
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1426
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1427
  work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] = wsm_conden(qmin,rv,t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], qsat[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1428
  work2[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_add(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))]); // mp_wsm6.F90:1429
  pcond[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_min(tc_div(tc_max(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], 0.f), dtcld), tc_max(tc_div(work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], dtcld), 0.f)); // mp_wsm6.F90:1430
  if (((qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] > 0.f) && (work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))] < 0.f))) { pcond[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))] = tc_div(tc_max(work1[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))+((1)-(1))*(((1)*((ite)-(its)+1))*((kte)-(kts)+1))], (-qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))])), dtcld); } // mp_wsm6.F90:1431
  q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_sub(q[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(pcond[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], dtcld)); // mp_wsm6.F90:1433
  qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_max(tc_add(qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(pcond[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], dtcld)), 0.f); // mp_wsm6.F90:1434
  t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(t[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))], tc_mul(tc_div(tc_mul(pcond[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], xl[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), cpm[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), dtcld)); // mp_wsm6.F90:1435
   } // mp_wsm6.F90:1436
   } // mp_wsm6.F90:1437
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1443
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1444
  if ((qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] <= qmin)) { qc[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = 0.0f; } // mp_wsm6.F90:1445
  if ((qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] <= qmin)) { qi[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = 0.0f; } // mp_wsm6.F90:1446
   } // mp_wsm6.F90:1447
   } // mp_wsm6.F90:1448
   } // mp_wsm6.F90:1449
  if (((rainprod2d != nullptr) && (evapprod2d != nullptr))) { // mp_wsm6.F90:1451
  for (k=kts; k<=kte; k+=1) { // mp_wsm6.F90:1452
  for (i=its; i<=ite; i+=1) { // mp_wsm6.F90:1453
  rainprod2d[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(tc_add(praut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], pracw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), praci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgaci[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), paacw[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psaut[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]); // mp_wsm6.F90:1454
  evapprod2d[((i)-(its))*(1)+((k)-(1))*((1)*((ite)-(its)+1))] = (-tc_add(tc_add(tc_add(tc_add(prevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))], psevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgevp[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), psdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))]), pgdep[((i)-(its))*(1)+((k)-(kts))*((1)*((ite)-(its)+1))])); // mp_wsm6.F90:1456
   } // mp_wsm6.F90:1457
   } // mp_wsm6.F90:1458
  } // mp_wsm6.F90:1459
  // CCPP status text // mp_wsm6.F90:1465
   // mp_wsm6.F90:1465
  errflg = 0; // mp_wsm6.F90:1466
}
#include "hexv842_wsm6_column_f32_bindings.metalh"
kernel void hexv842_wsm6_column_f32(constant Hexv842Wsm6ColumnF32Resources &r [[buffer(0)]],constant Hexv842Wsm6ColumnF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int nz=p.nz,nc=p.nc;if(gid>=uint(p.ns))return;
 Wsm6Workspace ws; device F *cursor=(device F*)r.workspace+ulong(gid)*ulong(p.workspace_words);
 ws.mp_wsm6_run_rh=cursor;cursor+=(nz * 3 + 3);
 ws.mp_wsm6_run_qsat=cursor;cursor+=(nz * 3 + 3);
 ws.mp_wsm6_run_rslope=cursor;cursor+=(nz * 3 + 3);
 ws.mp_wsm6_run_rslope2=cursor;cursor+=(nz * 3 + 3);
 ws.mp_wsm6_run_rslope3=cursor;cursor+=(nz * 3 + 3);
 ws.mp_wsm6_run_rslopeb=cursor;cursor+=(nz * 3 + 3);
 ws.mp_wsm6_run_qrs_tmp=cursor;cursor+=(nz * 3 + 3);
 ws.mp_wsm6_run_falk=cursor;cursor+=(nz * 3 + 3);
 ws.mp_wsm6_run_fall=cursor;cursor+=(nz * 3 + 3);
 ws.mp_wsm6_run_work1=cursor;cursor+=(nz * 3 + 3);
 ws.mp_wsm6_run_fallc=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_falkc=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_work1c=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_work2c=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_workr=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_worka=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_den_tmp=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_delz_tmp=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pigen=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pidep=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pcond=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_prevp=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_psevp=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pgevp=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_psdep=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pgdep=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_praut=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_psaut=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pgaut=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_piacr=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pracw=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_praci=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pracs=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_psacw=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_psaci=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_psacr=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pgacw=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pgaci=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pgacr=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pgacs=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_paacw=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_psmlt=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pgmlt=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pseml=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_pgeml=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_qsum=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_xl=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_cpm=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_work2=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_denfac=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_xni=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_denqrs1=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_denqrs2=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_denqrs3=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_denqci=cursor;cursor+=(nz + 3);
 ws.mp_wsm6_run_n0sfac=cursor;cursor+=(nz + 3);
 ws.slope_wsm6_n0sfac=cursor;cursor+=(nz + 3);
 ws.slope_rain_n0sfac=cursor;cursor+=(nz + 3);
 ws.slope_snow_n0sfac=cursor;cursor+=(nz + 3);
 ws.slope_graup_n0sfac=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_dz=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_ww=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_qq=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_wd=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_wa=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_was=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_den=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_denfac=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_tk=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_qn=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_qr=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_tmp=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_tmp1=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_tmp2=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_tmp3=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_wi=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_zi=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_za=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_dza=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_qa=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_qmi=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm_qpi=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_dz=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_ww=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_qq=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_qq2=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_wd=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_wa=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_wa2=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_was=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_den=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_denfac=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_tk=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_qn=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_qr=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_qr2=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_tmp=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_tmp1=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_tmp2=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_tmp3=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_wi=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_zi=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_za=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_dza=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_qa=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_qa2=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_qmi=cursor;cursor+=(nz + 3);
 ws.nislfv_rain_plm6_qpi=cursor;cursor+=(nz + 3);
 device F *local=cursor;cursor+=10*nz;device F *surface=cursor;
 Wsm6State S{};S.workspace=&ws;
 S.dtcldcr=((device const F*)r.state)[0*nc+gid];
 S.n0r=((device const F*)r.state)[1*nc+gid];
 S.avtr=((device const F*)r.state)[2*nc+gid];
 S.bvtr=((device const F*)r.state)[3*nc+gid];
 S.r0=((device const F*)r.state)[4*nc+gid];
 S.peaut=((device const F*)r.state)[5*nc+gid];
 S.xncr=((device const F*)r.state)[6*nc+gid];
 S.xmyu=((device const F*)r.state)[7*nc+gid];
 S.avts=((device const F*)r.state)[8*nc+gid];
 S.bvts=((device const F*)r.state)[9*nc+gid];
 S.lamdarmax=((device const F*)r.state)[10*nc+gid];
 S.lamdasmax=((device const F*)r.state)[11*nc+gid];
 S.dicon=((device const F*)r.state)[12*nc+gid];
 S.dimax=((device const F*)r.state)[13*nc+gid];
 S.pfrz1=((device const F*)r.state)[14*nc+gid];
 S.pfrz2=((device const F*)r.state)[15*nc+gid];
 S.qcrmin=((device const F*)r.state)[16*nc+gid];
 S.eacrc=((device const F*)r.state)[17*nc+gid];
 S.dens=((device const F*)r.state)[18*nc+gid];
 S.qs0=((device const F*)r.state)[19*nc+gid];
 S.n0smax=((device const F*)r.state)[20*nc+gid];
 S.n0s=((device const F*)r.state)[21*nc+gid];
 S.alpha=((device const F*)r.state)[22*nc+gid];
 S.qc0=((device const F*)r.state)[23*nc+gid];
 S.qck1=((device const F*)r.state)[24*nc+gid];
 S.bvtr1=((device const F*)r.state)[25*nc+gid];
 S.bvtr2=((device const F*)r.state)[26*nc+gid];
 S.bvtr3=((device const F*)r.state)[27*nc+gid];
 S.bvtr4=((device const F*)r.state)[28*nc+gid];
 S.g1pbr=((device const F*)r.state)[29*nc+gid];
 S.g3pbr=((device const F*)r.state)[30*nc+gid];
 S.g4pbr=((device const F*)r.state)[31*nc+gid];
 S.g5pbro2=((device const F*)r.state)[32*nc+gid];
 S.pvtr=((device const F*)r.state)[33*nc+gid];
 S.eacrr=((device const F*)r.state)[34*nc+gid];
 S.pacrr=((device const F*)r.state)[35*nc+gid];
 S.bvtr6=((device const F*)r.state)[36*nc+gid];
 S.g6pbr=((device const F*)r.state)[37*nc+gid];
 S.precr1=((device const F*)r.state)[38*nc+gid];
 S.precr2=((device const F*)r.state)[39*nc+gid];
 S.roqimax=((device const F*)r.state)[40*nc+gid];
 S.bvts1=((device const F*)r.state)[41*nc+gid];
 S.bvts2=((device const F*)r.state)[42*nc+gid];
 S.bvts3=((device const F*)r.state)[43*nc+gid];
 S.bvts4=((device const F*)r.state)[44*nc+gid];
 S.g1pbs=((device const F*)r.state)[45*nc+gid];
 S.g3pbs=((device const F*)r.state)[46*nc+gid];
 S.g4pbs=((device const F*)r.state)[47*nc+gid];
 S.n0g=((device const F*)r.state)[48*nc+gid];
 S.avtg=((device const F*)r.state)[49*nc+gid];
 S.bvtg=((device const F*)r.state)[50*nc+gid];
 S.deng=((device const F*)r.state)[51*nc+gid];
 S.lamdagmax=((device const F*)r.state)[52*nc+gid];
 S.g5pbso2=((device const F*)r.state)[53*nc+gid];
 S.pvts=((device const F*)r.state)[54*nc+gid];
 S.pacrs=((device const F*)r.state)[55*nc+gid];
 S.precs1=((device const F*)r.state)[56*nc+gid];
 S.precs2=((device const F*)r.state)[57*nc+gid];
 S.pidn0r=((device const F*)r.state)[58*nc+gid];
 S.xlv1=((device const F*)r.state)[59*nc+gid];
 S.pacrc=((device const F*)r.state)[60*nc+gid];
 S.pi=((device const F*)r.state)[61*nc+gid];
 S.bvtg1=((device const F*)r.state)[62*nc+gid];
 S.bvtg2=((device const F*)r.state)[63*nc+gid];
 S.bvtg3=((device const F*)r.state)[64*nc+gid];
 S.bvtg4=((device const F*)r.state)[65*nc+gid];
 S.g1pbg=((device const F*)r.state)[66*nc+gid];
 S.g3pbg=((device const F*)r.state)[67*nc+gid];
 S.g4pbg=((device const F*)r.state)[68*nc+gid];
 S.g5pbgo2=((device const F*)r.state)[69*nc+gid];
 S.pvtg=((device const F*)r.state)[70*nc+gid];
 S.pacrg=((device const F*)r.state)[71*nc+gid];
 S.precg1=((device const F*)r.state)[72*nc+gid];
 S.precg2=((device const F*)r.state)[73*nc+gid];
 S.pidn0g=((device const F*)r.state)[74*nc+gid];
 S.rslopermax=((device const F*)r.state)[75*nc+gid];
 S.rslopesmax=((device const F*)r.state)[76*nc+gid];
 S.rslopegmax=((device const F*)r.state)[77*nc+gid];
 S.rsloperbmax=((device const F*)r.state)[78*nc+gid];
 S.rslopesbmax=((device const F*)r.state)[79*nc+gid];
 S.rslopegbmax=((device const F*)r.state)[80*nc+gid];
 S.rsloper2max=((device const F*)r.state)[81*nc+gid];
 S.rslopes2max=((device const F*)r.state)[82*nc+gid];
 S.rslopeg2max=((device const F*)r.state)[83*nc+gid];
 S.rsloper3max=((device const F*)r.state)[84*nc+gid];
 S.rslopes3max=((device const F*)r.state)[85*nc+gid];
 S.rslopeg3max=((device const F*)r.state)[86*nc+gid];
 S.pidn0s=((device const F*)r.state)[87*nc+gid];
 S.pidnc=((device const F*)r.state)[88*nc+gid];
 device F *v=(device F*)r.v,*s=(device F*)r.s;device const F *c=(device const F*)r.constants,*pi=(device const F*)r.pi;
 for(int n=0;n<10;++n)for(int k=0;k<nz;++k)local[n*nz+k]=v[(n*nz+k)*nc+gid];
 for(int n=0;n<7;++n)surface[n]=s[n*nc+gid];
 if(p.wrapper)for(int k=0;k<nz;++k)local[k]=tc_mul(local[k],pi[k*nc+gid]);
 int err=0;
 mp_wsm6_run(S,local,local+nz,local+2*nz,local+3*nz,local+4*nz,local+5*nz,local+6*nz,local+7*nz,local+8*nz,local+9*nz,F(p.dt),
 c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18],c[19],
 surface,surface+1,surface+6,surface+2,surface+3,surface+4,surface+5,(device F*)nullptr,(device F*)nullptr,1,1,1,nz,err);
 if(p.wrapper)for(int k=0;k<nz;++k){F theta=tc_div(local[k],pi[k*nc+gid]);v[k*nc+gid]=theta;local[k]=tc_mul(theta,pi[k*nc+gid]);}
 for(int n=p.wrapper?1:0;n<10;++n)for(int k=0;k<nz;++k)v[(n*nz+k)*nc+gid]=local[n*nz+k];
 for(int n=0;n<7;++n)s[n*nc+gid]=surface[n];r.status[gid]=err;
}
