#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
using namespace ntc;
// This reproduces the source's explicit operation order for the two powers used by init.
template<int N> static inline F ipow(F x) {
 if constexpr (N==3) return tc_mul(tc_mul(x,x),x);
 if constexpr (N==8) {F x2=tc_mul(x,x),x4=tc_mul(x2,x2);return tc_mul(x4,x4);}
}
struct Wsm6State {
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
#include "hexv842_wsm6_state_f32_bindings.metalh"
kernel void hexv842_wsm6_state_f32(constant Hexv842Wsm6StateF32Resources &r [[buffer(0)]],constant Hexv842Wsm6StateF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if (gid >= uint(p.count)) return;
 Wsm6State S{};int err=0;state_parameters(S);
 device const F *c=(device const F *)r.constants;
 mp_wsm6_init(S,c[15],c[16],100.f,c[17],c[2],r.hail[gid],err);
 r.state[0*p.count+gid]=as_type<float>(S.dtcldcr.bits);
 r.state[1*p.count+gid]=as_type<float>(S.n0r.bits);
 r.state[2*p.count+gid]=as_type<float>(S.avtr.bits);
 r.state[3*p.count+gid]=as_type<float>(S.bvtr.bits);
 r.state[4*p.count+gid]=as_type<float>(S.r0.bits);
 r.state[5*p.count+gid]=as_type<float>(S.peaut.bits);
 r.state[6*p.count+gid]=as_type<float>(S.xncr.bits);
 r.state[7*p.count+gid]=as_type<float>(S.xmyu.bits);
 r.state[8*p.count+gid]=as_type<float>(S.avts.bits);
 r.state[9*p.count+gid]=as_type<float>(S.bvts.bits);
 r.state[10*p.count+gid]=as_type<float>(S.lamdarmax.bits);
 r.state[11*p.count+gid]=as_type<float>(S.lamdasmax.bits);
 r.state[12*p.count+gid]=as_type<float>(S.dicon.bits);
 r.state[13*p.count+gid]=as_type<float>(S.dimax.bits);
 r.state[14*p.count+gid]=as_type<float>(S.pfrz1.bits);
 r.state[15*p.count+gid]=as_type<float>(S.pfrz2.bits);
 r.state[16*p.count+gid]=as_type<float>(S.qcrmin.bits);
 r.state[17*p.count+gid]=as_type<float>(S.eacrc.bits);
 r.state[18*p.count+gid]=as_type<float>(S.dens.bits);
 r.state[19*p.count+gid]=as_type<float>(S.qs0.bits);
 r.state[20*p.count+gid]=as_type<float>(S.n0smax.bits);
 r.state[21*p.count+gid]=as_type<float>(S.n0s.bits);
 r.state[22*p.count+gid]=as_type<float>(S.alpha.bits);
 r.state[23*p.count+gid]=as_type<float>(S.qc0.bits);
 r.state[24*p.count+gid]=as_type<float>(S.qck1.bits);
 r.state[25*p.count+gid]=as_type<float>(S.bvtr1.bits);
 r.state[26*p.count+gid]=as_type<float>(S.bvtr2.bits);
 r.state[27*p.count+gid]=as_type<float>(S.bvtr3.bits);
 r.state[28*p.count+gid]=as_type<float>(S.bvtr4.bits);
 r.state[29*p.count+gid]=as_type<float>(S.g1pbr.bits);
 r.state[30*p.count+gid]=as_type<float>(S.g3pbr.bits);
 r.state[31*p.count+gid]=as_type<float>(S.g4pbr.bits);
 r.state[32*p.count+gid]=as_type<float>(S.g5pbro2.bits);
 r.state[33*p.count+gid]=as_type<float>(S.pvtr.bits);
 r.state[34*p.count+gid]=as_type<float>(S.eacrr.bits);
 r.state[35*p.count+gid]=as_type<float>(S.pacrr.bits);
 r.state[36*p.count+gid]=as_type<float>(S.bvtr6.bits);
 r.state[37*p.count+gid]=as_type<float>(S.g6pbr.bits);
 r.state[38*p.count+gid]=as_type<float>(S.precr1.bits);
 r.state[39*p.count+gid]=as_type<float>(S.precr2.bits);
 r.state[40*p.count+gid]=as_type<float>(S.roqimax.bits);
 r.state[41*p.count+gid]=as_type<float>(S.bvts1.bits);
 r.state[42*p.count+gid]=as_type<float>(S.bvts2.bits);
 r.state[43*p.count+gid]=as_type<float>(S.bvts3.bits);
 r.state[44*p.count+gid]=as_type<float>(S.bvts4.bits);
 r.state[45*p.count+gid]=as_type<float>(S.g1pbs.bits);
 r.state[46*p.count+gid]=as_type<float>(S.g3pbs.bits);
 r.state[47*p.count+gid]=as_type<float>(S.g4pbs.bits);
 r.state[48*p.count+gid]=as_type<float>(S.n0g.bits);
 r.state[49*p.count+gid]=as_type<float>(S.avtg.bits);
 r.state[50*p.count+gid]=as_type<float>(S.bvtg.bits);
 r.state[51*p.count+gid]=as_type<float>(S.deng.bits);
 r.state[52*p.count+gid]=as_type<float>(S.lamdagmax.bits);
 r.state[53*p.count+gid]=as_type<float>(S.g5pbso2.bits);
 r.state[54*p.count+gid]=as_type<float>(S.pvts.bits);
 r.state[55*p.count+gid]=as_type<float>(S.pacrs.bits);
 r.state[56*p.count+gid]=as_type<float>(S.precs1.bits);
 r.state[57*p.count+gid]=as_type<float>(S.precs2.bits);
 r.state[58*p.count+gid]=as_type<float>(S.pidn0r.bits);
 r.state[59*p.count+gid]=as_type<float>(S.xlv1.bits);
 r.state[60*p.count+gid]=as_type<float>(S.pacrc.bits);
 r.state[61*p.count+gid]=as_type<float>(S.pi.bits);
 r.state[62*p.count+gid]=as_type<float>(S.bvtg1.bits);
 r.state[63*p.count+gid]=as_type<float>(S.bvtg2.bits);
 r.state[64*p.count+gid]=as_type<float>(S.bvtg3.bits);
 r.state[65*p.count+gid]=as_type<float>(S.bvtg4.bits);
 r.state[66*p.count+gid]=as_type<float>(S.g1pbg.bits);
 r.state[67*p.count+gid]=as_type<float>(S.g3pbg.bits);
 r.state[68*p.count+gid]=as_type<float>(S.g4pbg.bits);
 r.state[69*p.count+gid]=as_type<float>(S.g5pbgo2.bits);
 r.state[70*p.count+gid]=as_type<float>(S.pvtg.bits);
 r.state[71*p.count+gid]=as_type<float>(S.pacrg.bits);
 r.state[72*p.count+gid]=as_type<float>(S.precg1.bits);
 r.state[73*p.count+gid]=as_type<float>(S.precg2.bits);
 r.state[74*p.count+gid]=as_type<float>(S.pidn0g.bits);
 r.state[75*p.count+gid]=as_type<float>(S.rslopermax.bits);
 r.state[76*p.count+gid]=as_type<float>(S.rslopesmax.bits);
 r.state[77*p.count+gid]=as_type<float>(S.rslopegmax.bits);
 r.state[78*p.count+gid]=as_type<float>(S.rsloperbmax.bits);
 r.state[79*p.count+gid]=as_type<float>(S.rslopesbmax.bits);
 r.state[80*p.count+gid]=as_type<float>(S.rslopegbmax.bits);
 r.state[81*p.count+gid]=as_type<float>(S.rsloper2max.bits);
 r.state[82*p.count+gid]=as_type<float>(S.rslopes2max.bits);
 r.state[83*p.count+gid]=as_type<float>(S.rslopeg2max.bits);
 r.state[84*p.count+gid]=as_type<float>(S.rsloper3max.bits);
 r.state[85*p.count+gid]=as_type<float>(S.rslopes3max.bits);
 r.state[86*p.count+gid]=as_type<float>(S.rslopeg3max.bits);
 r.state[87*p.count+gid]=as_type<float>(S.pidn0s.bits);
 r.state[88*p.count+gid]=as_type<float>(S.pidnc.bits);
 r.status[gid]=err;
}
