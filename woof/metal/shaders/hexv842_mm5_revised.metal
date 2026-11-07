#include <metal_stdlib>
#include "hexv842_mm5_tables_bindings.metalh"
#include "hexv842_mm5_revised_bindings.metalh"
#include "soft_word32.metalh"
#include "soft_glibc_solar_trig.metalh"
using namespace metal;
static inline F operator/(F a,F b) { return aa_div_impl(a,b); }
static inline F sf_min(F a,F b) { if (soft32_is_nan(a.bits)) return b; if (soft32_is_nan(b.bits)) return a; if (((a.bits|b.bits)&0x7fffffffu)==0) return f32_word(a.bits|b.bits); return a<b?a:b; }
static inline F sf_max(F a,F b) { if (soft32_is_nan(a.bits)) return b; if (soft32_is_nan(b.bits)) return a; if (((a.bits|b.bits)&0x7fffffffu)==0) return f32_word(a.bits&b.bits); return a>b?a:b; }
static inline F mm5_log(F a) { return gfk_log(a); }
static inline F mm5_exp(F a) { return gfk_exp(a); }
static inline F mm5_pow(F a,F b) { return gfk_pow(a,b); }
static inline F mm5_atan(F a) { return f32_word(soft_glibc_solar_atanf(a.bits)); }
#define FADD(a,b) (F(a)+F(b))
#define FSUB(a,b) (F(a)-F(b))
#define FMUL(a,b) (F(a)*F(b))
#define FDIV(a,b) (F(a)/F(b))
#define FSQRT(a) f32_word(soft32_sqrt(F(a).bits))
struct MM5State { F znt; F ust; F ustm; F mol; F hfx; F qfx; F qsfc; F zol; F regime; F psim; F psih; F fm; F fh; F lh; F u10; F v10; F th2; F t2; F q2; F chs; F chs2; F cqs2; F flhc; F flqc; F qgh; F rmol; F wspd; F br; F gz1oz0; F cpm; F ck; F cka; F cd; F cda; F ux; F vx; F t1d; F qv1d; F p1d; F dz8w1d; F psfcpa; F tsk; F pblh; F mavail; F xland; F lakemask; F dx; F water_depth; };
struct MM5Constants { F cp; F g; F rovcp; F r; F xlv; F svp1; F svp2; F svp3; F svpt0; F ep1; F ep2; F karman; F p1000mb; };
// MPAS physics_mmm/sf_sfclayrev.F90:987-992 psim_stable_full
static __attribute__((noinline)) F mm5_psim_stable_full(F zolf, device const float *tables) {
F result;

result = FMUL(-0x1.8666660000000p+2f, mm5_log(FADD(zolf, mm5_pow(FADD(0x1.0000000000000p+0f, mm5_pow(zolf, 0x1.4000000000000p+1f)), 0x1.99999a0000000p-2f)))); // MPAS:989
return result; // MPAS:991
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:995-1000 psih_stable_full
static __attribute__((noinline)) F mm5_psih_stable_full(F zolf, device const float *tables) {
F result;

result = FMUL(-0x1.5333340000000p+2f, mm5_log(FADD(zolf, mm5_pow(FADD(0x1.0000000000000p+0f, mm5_pow(zolf, 0x1.19999a0000000p+0f)), 0x1.d1745c0000000p-1f)))); // MPAS:997
return result; // MPAS:999
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1003-1015 psim_unstable_full
static __attribute__((noinline)) F mm5_psim_unstable_full(F zolf, device const float *tables) {
F result;
F psimc;
F psimk;
F x;
F y;
F ym;
x = mm5_pow(FSUB(0x1.0000000000000p+0f, FMUL(0x1.0000000000000p+4f, zolf)), 0x1.0000000000000p-2f); // MPAS:1006
psimk = FADD(FSUB(FADD(FMUL(0x1.0000000000000p+1f, mm5_log(FMUL(0x1.0000000000000p-1f, FADD(0x1.0000000000000p+0f, x)))), mm5_log(FMUL(0x1.0000000000000p-1f, FADD(0x1.0000000000000p+0f, FMUL(x, x))))), FMUL(0x1.0000000000000p+1f, mm5_atan(x))), 0x1.921fb60000000p+0f); // MPAS:1007
ym = mm5_pow(FSUB(0x1.0000000000000p+0f, FMUL(0x1.4000000000000p+3f, zolf)), 0x1.51eb860000000p-2f); // MPAS:1009
psimc = FADD(FSUB(FMUL(0x1.8000000000000p+0f, mm5_log(FDIV(FADD(FADD(FMUL(ym, ym), ym), 0x1.0000000000000p+0f), 0x1.8000000000000p+1f))), FMUL(0x1.bb67ae0000000p+0f, mm5_atan(FDIV(FADD(FMUL(0x1.0000000000000p+1f, ym), 0x1.0000000000000p+0f), 0x1.bb67ae0000000p+0f)))), 0x1.d0552a0000000p+0f); // MPAS:1010
result = FDIV(FADD(psimk, FMUL(FMUL(zolf, zolf), psimc)), FADD(0x1.0000000000000p+0f, FMUL(zolf, zolf))); // MPAS:1012
return result; // MPAS:1014
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1018-1030 psih_unstable_full
static __attribute__((noinline)) F mm5_psih_unstable_full(F zolf, device const float *tables) {
F result;
F psihc;
F psihk;
F y;
F yh;
y = mm5_pow(FSUB(0x1.0000000000000p+0f, FMUL(0x1.0000000000000p+4f, zolf)), 0x1.0000000000000p-1f); // MPAS:1021
psihk = FMUL(0x1.0000000000000p+1f, mm5_log(FDIV(FADD(0x1.0000000000000p+0f, y), 0x1.0000000000000p+1f))); // MPAS:1022
yh = mm5_pow(FSUB(0x1.0000000000000p+0f, FMUL(0x1.1000000000000p+5f, zolf)), 0x1.51eb860000000p-2f); // MPAS:1024
psihc = FADD(FSUB(FMUL(0x1.8000000000000p+0f, mm5_log(FDIV(FADD(FADD(FMUL(yh, yh), yh), 0x1.0000000000000p+0f), 0x1.8000000000000p+1f))), FMUL(0x1.bb67ae0000000p+0f, mm5_atan(FDIV(FADD(FMUL(0x1.0000000000000p+1f, yh), 0x1.0000000000000p+0f), 0x1.bb67ae0000000p+0f)))), 0x1.d0552a0000000p+0f); // MPAS:1025
result = FDIV(FADD(psihk, FMUL(FMUL(zolf, zolf), psihc)), FADD(0x1.0000000000000p+0f, FMUL(zolf, zolf))); // MPAS:1027
return result; // MPAS:1029
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1034-1047 psim_stable
static __attribute__((noinline)) F mm5_psim_stable(F zolf, device const float *tables) {
F result;
int nzol;
F rzol;
nzol = int(FMUL(zolf, 0x1.9000000000000p+6f)); // MPAS:1038
rzol = FSUB(FMUL(zolf, 0x1.9000000000000p+6f), nzol); // MPAS:1039
if ((FADD(nzol, 0x1.0000000000000p+0f) < 0x1.f400000000000p+9f)) { // MPAS:1040
result = FADD(tables[0*1001 + (int)(nzol)], FMUL(rzol, FSUB(tables[0*1001 + int(FADD(nzol, 0x1.0000000000000p+0f))], tables[0*1001 + (int)(nzol)]))); // MPAS:1041
} else { // MPAS:1042
result = mm5_psim_stable_full(zolf, tables); // MPAS:1043
} // MPAS:1044
return result; // MPAS:1046
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1050-1063 psih_stable
static __attribute__((noinline)) F mm5_psih_stable(F zolf, device const float *tables) {
F result;
int nzol;
F rzol;
nzol = int(FMUL(zolf, 0x1.9000000000000p+6f)); // MPAS:1054
rzol = FSUB(FMUL(zolf, 0x1.9000000000000p+6f), nzol); // MPAS:1055
if ((FADD(nzol, 0x1.0000000000000p+0f) < 0x1.f400000000000p+9f)) { // MPAS:1056
result = FADD(tables[1*1001 + (int)(nzol)], FMUL(rzol, FSUB(tables[1*1001 + int(FADD(nzol, 0x1.0000000000000p+0f))], tables[1*1001 + (int)(nzol)]))); // MPAS:1057
} else { // MPAS:1058
result = mm5_psih_stable_full(zolf, tables); // MPAS:1059
} // MPAS:1060
return result; // MPAS:1062
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1066-1079 psim_unstable
static __attribute__((noinline)) F mm5_psim_unstable(F zolf, device const float *tables) {
F result;
int nzol;
F rzol;
nzol = int(FMUL(-(zolf), 0x1.9000000000000p+6f)); // MPAS:1070
rzol = FSUB(FMUL(-(zolf), 0x1.9000000000000p+6f), nzol); // MPAS:1071
if ((FADD(nzol, 0x1.0000000000000p+0f) < 0x1.f400000000000p+9f)) { // MPAS:1072
result = FADD(tables[2*1001 + (int)(nzol)], FMUL(rzol, FSUB(tables[2*1001 + int(FADD(nzol, 0x1.0000000000000p+0f))], tables[2*1001 + (int)(nzol)]))); // MPAS:1073
} else { // MPAS:1074
result = mm5_psim_unstable_full(zolf, tables); // MPAS:1075
} // MPAS:1076
return result; // MPAS:1078
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1082-1095 psih_unstable
static __attribute__((noinline)) F mm5_psih_unstable(F zolf, device const float *tables) {
F result;
int nzol;
F rzol;
nzol = int(FMUL(-(zolf), 0x1.9000000000000p+6f)); // MPAS:1086
rzol = FSUB(FMUL(-(zolf), 0x1.9000000000000p+6f), nzol); // MPAS:1087
if ((FADD(nzol, 0x1.0000000000000p+0f) < 0x1.f400000000000p+9f)) { // MPAS:1088
result = FADD(tables[3*1001 + (int)(nzol)], FMUL(rzol, FSUB(tables[3*1001 + int(FADD(nzol, 0x1.0000000000000p+0f))], tables[3*1001 + (int)(nzol)]))); // MPAS:1089
} else { // MPAS:1090
result = mm5_psih_unstable_full(zolf, tables); // MPAS:1091
} // MPAS:1092
return result; // MPAS:1094
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:960-981 zolri2
static __attribute__((noinline)) F mm5_zolri2(thread F &zol2, F ri2, F z, F z0, device const float *tables) {
F result;
F psih2;
F psix2;
F zol20;
F zol3;
if ((FMUL(zol2, ri2) < 0x0.0p+0f)) zol2 = 0x0.0p+0f; // MPAS:965
zol20 = FDIV(FMUL(zol2, z0), z); // MPAS:967
zol3 = FADD(zol2, zol20); // MPAS:968
if ((ri2 < 0x0.0p+0f)) { // MPAS:970
psix2 = FSUB(mm5_log(FDIV(FADD(z, z0), z0)), FSUB(mm5_psim_unstable(zol3, tables), mm5_psim_unstable(zol20, tables))); // MPAS:971
psih2 = FSUB(mm5_log(FDIV(FADD(z, z0), z0)), FSUB(mm5_psih_unstable(zol3, tables), mm5_psih_unstable(zol20, tables))); // MPAS:972
} else { // MPAS:973
psix2 = FSUB(mm5_log(FDIV(FADD(z, z0), z0)), FSUB(mm5_psim_stable(zol3, tables), mm5_psim_stable(zol20, tables))); // MPAS:974
psih2 = FSUB(mm5_log(FDIV(FADD(z, z0), z0)), FSUB(mm5_psih_stable(zol3, tables), mm5_psih_stable(zol20, tables))); // MPAS:975
} // MPAS:976
result = FSUB(FDIV(FMUL(zol2, psih2), FMUL(psix2, psix2)), ri2); // MPAS:978
return result; // MPAS:980
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:922-957 zolri
static __attribute__((noinline)) F mm5_zolri(F ri, F z, F z0, device const float *tables) {
F result;
int iter;
F fx1;
F fx2;
F x1;
F x2;
if ((ri < 0x0.0p+0f)) { // MPAS:929
x1 = -0x1.4000000000000p+2f; // MPAS:930
x2 = 0x0.0p+0f; // MPAS:931
} else { // MPAS:932
x1 = 0x0.0p+0f; // MPAS:933
x2 = 0x1.4000000000000p+2f; // MPAS:934
} // MPAS:935
fx1 = mm5_zolri2(x1, ri, z, z0, tables); // MPAS:937
fx2 = mm5_zolri2(x2, ri, z, z0, tables); // MPAS:938
iter = 0x0.0p+0f; // MPAS:939
while ((fabs(FSUB(x1, x2)) > 0x1.47ae140000000p-7f)) {
if ((iter == 0x1.4000000000000p+3f)) return result; // MPAS:941
if ((fx1 == fx2)) return result; // MPAS:943
if ((fabs(fx2) < fabs(fx1))) { // MPAS:944
x1 = FSUB(x1, FMUL(FDIV(fx1, FSUB(fx2, fx1)), FSUB(x2, x1))); // MPAS:945
fx1 = mm5_zolri2(x1, ri, z, z0, tables); // MPAS:946
result = x1; // MPAS:947
} else { // MPAS:948
x2 = FSUB(x2, FMUL(FDIV(fx2, FSUB(fx2, fx1)), FSUB(x2, x1))); // MPAS:949
fx2 = mm5_zolri2(x2, ri, z, z0, tables); // MPAS:950
result = x2; // MPAS:951
} // MPAS:952
iter = int(FADD(iter, 0x1.0000000000000p+0f)); // MPAS:953
}
return result; // MPAS:956
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:1098-1115 depth_dependent_z0
static __attribute__((noinline)) F mm5_depth_dependent_z0(F water_depth, F z0, F ust, device const float *tables) {
F result;
F depth_b;
F effective_depth;
if ((water_depth < 0x1.4000000000000p+3f)) { // MPAS:1102
effective_depth = 0x1.4000000000000p+3f; // MPAS:1103
} else if ((water_depth > 0x1.9000000000000p+6f)) { // MPAS:1104
effective_depth = 0x1.9000000000000p+6f; // MPAS:1105
} else { // MPAS:1106
effective_depth = water_depth; // MPAS:1107
} // MPAS:1108
depth_b = FMUL(0x1.1111120000000p-5f, mm5_log(FDIV(0x1.3b00000000000p+10f, effective_depth))); // MPAS:1110
result = mm5_exp(FDIV(FSUB(FMUL(0x1.59999a0000000p+1f, ust), FDIV(0x1.cccccc0000000p+0f, depth_b)), FADD(ust, FDIV(0x1.5c28f60000000p-3f, depth_b)))); // MPAS:1111
result = sf_min(result, 0x1.99999a0000000p-4f); // MPAS:1112
return result; // MPAS:1114
return result;
}
// MPAS physics_mmm/sf_sfclayrev.F90:79-920
static __attribute__((noinline)) void mm5_revised(thread MM5State &s, thread const MM5Constants &c, device const float *tables, int isfflx, int isftcflx, int iz0tlnd, int scm_force_flux, int shalwater_z0, unsigned int optional_mask) {
F pl, thcon, tvcon, e1, zl, tskv, dthvdz, dthvm, vconv, rzol, rzol2, rzol10, zol2, zol10, dtg, psix, dtthx, psix10, psit, psit2, psiq, psiq2, psiq10, fluxc, vsgd, z0q, visc, restar, czil, gz0ozq, gz0ozt, zw, zn1, zn2, zolzz, zol0, zl2, zl10, z0t, za, thvx, zqkl, zqklp1, thx, qx, psih2, psim2, psih10, psim10, denomq, denomq2, denomt2, wspdi, gz2oz0, gz10oz0, rhox, govrth, tgdsa, scr3, scr4, thgb, psfc, pq, pq2, pq10, czc;
int nzol, nzol2, nzol10;
psfc = FDIV(s.psfcpa, 0x1.f400000000000p+9f); // MPAS:221
tgdsa = s.tsk; // MPAS:227
thgb = FMUL(s.tsk, mm5_pow(FDIV(c.p1000mb, s.psfcpa), c.rovcp)); // MPAS:230



pl = FDIV(s.p1d, 0x1.f400000000000p+9f); // MPAS:255
scr3 = s.t1d; // MPAS:256
thcon = mm5_pow(FDIV(FMUL(c.p1000mb, 0x1.0624de0000000p-10f), pl), c.rovcp); // MPAS:258
thx = FMUL(scr3, thcon); // MPAS:259
scr4 = scr3; // MPAS:260
thvx = thx; // MPAS:261
qx = 0x0.0p+0f; // MPAS:262

s.qgh = 0x0.0p+0f; // MPAS:266
s.flhc = 0x0.0p+0f; // MPAS:267
s.flqc = 0x0.0p+0f; // MPAS:268
s.cpm = c.cp; // MPAS:269
qx = s.qv1d; // MPAS:274
tvcon = FADD(0x1.0000000000000p+0f, FMUL(c.ep1, qx)); // MPAS:275
thvx = FMUL(thx, tvcon); // MPAS:276
scr4 = FMUL(scr3, tvcon); // MPAS:277

e1 = FMUL(c.svp1, mm5_exp(FDIV(FMUL(c.svp2, FSUB(tgdsa, c.svpt0)), FSUB(tgdsa, c.svp3)))); // MPAS:281
if (((s.xland > 0x1.8000000000000p+0f) && (s.lakemask == 0x0.0p+0f))) e1 = FMUL(e1, 0x1.f5c2900000000p-1f); // MPAS:283
if (((s.xland > 0x1.8000000000000p+0f) || (s.qsfc <= 0x0.0p+0f))) s.qsfc = FDIV(FMUL(c.ep2, e1), FSUB(psfc, e1)); // MPAS:285
e1 = FMUL(c.svp1, mm5_exp(FDIV(FMUL(c.svp2, FSUB(s.t1d, c.svpt0)), FSUB(s.t1d, c.svp3)))); // MPAS:288
pl = FDIV(s.p1d, 0x1.f400000000000p+9f); // MPAS:289
s.qgh = FDIV(FMUL(c.ep2, e1), FSUB(pl, e1)); // MPAS:290
s.cpm = FMUL(c.cp, FADD(0x1.0000000000000p+0f, FMUL(0x1.99999a0000000p-1f, qx))); // MPAS:291


zqklp1 = 0x0.0p+0f; // MPAS:299
rhox = FDIV(FMUL(psfc, 0x1.f400000000000p+9f), FMUL(c.r, scr4)); // MPAS:300

zqkl = FADD(s.dz8w1d, zqklp1); // MPAS:304

za = FMUL(0x1.0000000000000p-1f, FADD(zqkl, zqklp1)); // MPAS:308

govrth = FDIV(c.g, thx); // MPAS:312

s.gz1oz0 = mm5_log(FDIV(FADD(za, s.znt), s.znt)); // MPAS:318
gz2oz0 = mm5_log(FDIV(FADD(0x1.0000000000000p+1f, s.znt), s.znt)); // MPAS:319
gz10oz0 = mm5_log(FDIV(FADD(0x1.4000000000000p+3f, s.znt), s.znt)); // MPAS:320
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:321
zl = s.znt; // MPAS:322
} else { // MPAS:323
zl = 0x1.47ae140000000p-7f; // MPAS:324
} // MPAS:325
s.wspd = FSQRT(FADD(FMUL(s.ux, s.ux), FMUL(s.vx, s.vx))); // MPAS:326
tskv = FMUL(thgb, FADD(0x1.0000000000000p+0f, FMUL(c.ep1, s.qsfc))); // MPAS:328
dthvdz = FSUB(thvx, tskv); // MPAS:329
if ((s.xland < 0x1.8000000000000p+0f)) { // MPAS:336
fluxc = sf_max(FADD(FDIV(FDIV(s.hfx, rhox), c.cp), FDIV(FMUL(FMUL(c.ep1, tskv), s.qfx), rhox)), 0x0.0p+0f); // MPAS:337
vconv = FMUL(0x1.0000000000000p+0f, mm5_pow(FMUL(FMUL(FDIV(c.g, tgdsa), s.pblh), fluxc), 0x1.51eb860000000p-2f)); // MPAS:339
} else { // MPAS:340
if ((-(dthvdz) >= 0x0.0p+0f)) { // MPAS:341
dthvm = -(dthvdz); // MPAS:342
} else { // MPAS:343
dthvm = 0x0.0p+0f; // MPAS:344
} // MPAS:345
vconv = FSQRT(dthvm); // MPAS:348
} // MPAS:349
vsgd = FMUL(0x1.47ae140000000p-2f, mm5_pow(sf_max(FSUB(FDIV(s.dx, 0x1.3880000000000p+12f), 0x1.0000000000000p+0f), 0x0.0p+0f), 0x1.51eb860000000p-2f)); // MPAS:351
s.wspd = FSQRT(FADD(FADD(FMUL(s.wspd, s.wspd), FMUL(vconv, vconv)), FMUL(vsgd, vsgd))); // MPAS:352
s.wspd = sf_max(s.wspd, 0x1.99999a0000000p-4f); // MPAS:353
s.br = FDIV(FMUL(FMUL(govrth, za), dthvdz), FMUL(s.wspd, s.wspd)); // MPAS:354
if ((s.mol < 0x0.0p+0f)) s.br = sf_min(s.br, 0x0.0p+0f); // MPAS:356
s.rmol = FMUL(FMUL(FMUL(-(govrth), dthvdz), za), c.karman); // MPAS:357

if ((s.br > 0x0.0p+0f)) { // MPAS:381
if ((s.br > 0x1.f400000000000p+7f)) { // MPAS:382
s.zol = mm5_zolri(0x1.f400000000000p+7f, za, s.znt, tables); // MPAS:383
} else { // MPAS:384
s.zol = mm5_zolri(s.br, za, s.znt, tables); // MPAS:385
} // MPAS:386
} // MPAS:387
if ((s.br < 0x0.0p+0f)) { // MPAS:389
if ((s.ust < 0x1.0624de0000000p-10f)) { // MPAS:390
s.zol = FMUL(s.br, s.gz1oz0); // MPAS:391
} else { // MPAS:392
if ((s.br < -0x1.f400000000000p+7f)) { // MPAS:393
s.zol = mm5_zolri(-0x1.f400000000000p+7f, za, s.znt, tables); // MPAS:394
} else { // MPAS:395
s.zol = mm5_zolri(s.br, za, s.znt, tables); // MPAS:396
} // MPAS:397
} // MPAS:398
} // MPAS:399
zolzz = FDIV(FMUL(s.zol, FADD(za, s.znt)), za); // MPAS:403
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, s.znt)), za); // MPAS:404
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, s.znt)), za); // MPAS:405
zol0 = FDIV(FMUL(s.zol, s.znt), za); // MPAS:406
zl2 = FMUL(FDIV(0x1.0000000000000p+1f, za), s.zol); // MPAS:407
zl10 = FMUL(FDIV(0x1.4000000000000p+3f, za), s.zol); // MPAS:408
if ((FSUB(s.xland, 0x1.8000000000000p+0f) < 0x0.0p+0f)) { // MPAS:410
zl = FMUL(FDIV(0x1.47ae140000000p-7f, za), s.zol); // MPAS:411
} else { // MPAS:412
zl = zol0; // MPAS:413
} // MPAS:414
if (!(s.br < 0x0.0p+0f) && !(s.br == 0x0.0p+0f)) {
s.regime = 0x1.0000000000000p+0f; // MPAS:421
s.psim = FSUB(mm5_psim_stable(zolzz, tables), mm5_psim_stable(zol0, tables)); // MPAS:425
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:426
psim10 = FSUB(mm5_psim_stable(zol10, tables), mm5_psim_stable(zol0, tables)); // MPAS:428
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:429
psim2 = FSUB(mm5_psim_stable(zol2, tables), mm5_psim_stable(zol0, tables)); // MPAS:431
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:432
pq = FSUB(mm5_psih_stable(s.zol, tables), mm5_psih_stable(zl, tables)); // MPAS:436
pq2 = FSUB(mm5_psih_stable(zl2, tables), mm5_psih_stable(zl, tables)); // MPAS:437
pq10 = FSUB(mm5_psih_stable(zl10, tables), mm5_psih_stable(zl, tables)); // MPAS:438
s.rmol = FDIV(s.zol, za); // MPAS:441
} else if (s.br == 0x0.0p+0f) {
s.regime = 0x1.8000000000000p+1f; // MPAS:447
s.psim = 0x0.0p+0f; // MPAS:448
s.psih = s.psim; // MPAS:449
psim10 = 0x0.0p+0f; // MPAS:450
psih10 = psim10; // MPAS:451
psim2 = 0x0.0p+0f; // MPAS:452
psih2 = psim2; // MPAS:453
pq = s.psih; // MPAS:457
pq2 = psih2; // MPAS:458
pq10 = 0x0.0p+0f; // MPAS:459
s.zol = 0x0.0p+0f; // MPAS:461
s.rmol = FDIV(s.zol, za); // MPAS:462
} else {
s.regime = 0x1.0000000000000p+2f; // MPAS:469
s.psim = FSUB(mm5_psim_unstable(zolzz, tables), mm5_psim_unstable(zol0, tables)); // MPAS:473
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:474
psim10 = FSUB(mm5_psim_unstable(zol10, tables), mm5_psim_unstable(zol0, tables)); // MPAS:476
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:477
psim2 = FSUB(mm5_psim_unstable(zol2, tables), mm5_psim_unstable(zol0, tables)); // MPAS:479
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:480
pq = FSUB(mm5_psih_unstable(s.zol, tables), mm5_psih_unstable(zl, tables)); // MPAS:484
pq2 = FSUB(mm5_psih_unstable(zl2, tables), mm5_psih_unstable(zl, tables)); // MPAS:485
pq10 = FSUB(mm5_psih_unstable(zl10, tables), mm5_psih_unstable(zl, tables)); // MPAS:486
s.psih = sf_min(s.psih, FMUL(0x1.cccccc0000000p-1f, s.gz1oz0)); // MPAS:490
s.psim = sf_min(s.psim, FMUL(0x1.cccccc0000000p-1f, s.gz1oz0)); // MPAS:491
psih2 = sf_min(psih2, FMUL(0x1.cccccc0000000p-1f, gz2oz0)); // MPAS:492
psim10 = sf_min(psim10, FMUL(0x1.cccccc0000000p-1f, gz10oz0)); // MPAS:493
psih10 = sf_min(psih10, FMUL(0x1.cccccc0000000p-1f, gz10oz0)); // MPAS:496
s.rmol = FDIV(s.zol, za); // MPAS:497
}
dtg = FSUB(thx, thgb); // MPAS:505
psix = FSUB(s.gz1oz0, s.psim); // MPAS:506
psix10 = FSUB(gz10oz0, psim10); // MPAS:507
psit = FSUB(s.gz1oz0, s.psih); // MPAS:512
psit2 = FSUB(gz2oz0, psih2); // MPAS:513
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:515
zl = s.znt; // MPAS:516
} else { // MPAS:517
zl = 0x1.47ae140000000p-7f; // MPAS:518
} // MPAS:519
psiq = FSUB(mm5_log(FADD(FDIV(FMUL(FMUL(c.karman, s.ust), za), 0x1.92a7380000000p-16f), FDIV(za, zl))), pq); // MPAS:521
psiq2 = FSUB(mm5_log(FADD(FDIV(FMUL(FMUL(c.karman, s.ust), 0x1.0000000000000p+1f), 0x1.92a7380000000p-16f), FDIV(0x1.0000000000000p+1f, zl))), pq2); // MPAS:522
psiq10 = FSUB(mm5_log(FADD(FDIV(FMUL(FMUL(c.karman, s.ust), 0x1.4000000000000p+3f), 0x1.92a7380000000p-16f), FDIV(0x1.4000000000000p+3f, zl))), pq10); // MPAS:525
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:529
visc = FMUL(FADD(0x1.51eb860000000p+0f, FMUL(0x1.26e9780000000p-7f, FSUB(scr3, 0x1.1126660000000p+8f))), 0x1.4f8b580000000p-17f); // MPAS:530
restar = FDIV(FMUL(s.ust, s.znt), visc); // MPAS:531
z0t = FMUL(0x1.cd5f9a0000000p-15f, mm5_pow(restar, -0x1.3333340000000p-1f)); // MPAS:532
z0t = sf_min(z0t, 0x1.a36e2e0000000p-14f); // MPAS:533
z0t = sf_max(z0t, 0x1.12e0be0000000p-29f); // MPAS:534
z0q = z0t; // MPAS:535
zolzz = FDIV(FMUL(s.zol, FADD(za, z0t)), za); // MPAS:538
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, z0t)), za); // MPAS:539
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, z0t)), za); // MPAS:540
zol0 = FDIV(FMUL(s.zol, z0t), za); // MPAS:541
if ((s.zol > 0x0.0p+0f)) { // MPAS:543
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:544
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:545
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:546
} else { // MPAS:547
if ((s.zol == 0x0.0p+0f)) { // MPAS:548
s.psih = 0x0.0p+0f; // MPAS:549
psih10 = 0x0.0p+0f; // MPAS:550
psih2 = 0x0.0p+0f; // MPAS:551
} else { // MPAS:552
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:553
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:554
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:555
} // MPAS:556
} // MPAS:557
psit = FSUB(mm5_log(FDIV(FADD(za, z0t), z0t)), s.psih); // MPAS:558
psit2 = FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0t), z0t)), psih2); // MPAS:559
zolzz = FDIV(FMUL(s.zol, FADD(za, z0q)), za); // MPAS:561
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, z0q)), za); // MPAS:562
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, z0q)), za); // MPAS:563
zol0 = FDIV(FMUL(s.zol, z0q), za); // MPAS:564
if ((s.zol > 0x0.0p+0f)) { // MPAS:566
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:567
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:568
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:569
} else { // MPAS:570
if ((s.zol == 0x0.0p+0f)) { // MPAS:571
s.psih = 0x0.0p+0f; // MPAS:572
psih10 = 0x0.0p+0f; // MPAS:573
psih2 = 0x0.0p+0f; // MPAS:574
} else { // MPAS:575
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:576
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:577
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:578
} // MPAS:579
} // MPAS:580
psiq = FSUB(mm5_log(FDIV(FADD(za, z0q), z0q)), s.psih); // MPAS:582
psiq2 = FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0q), z0q)), psih2); // MPAS:583
psiq10 = FSUB(mm5_log(FDIV(FADD(0x1.4000000000000p+3f, z0q), z0q)), psih10); // MPAS:584
} // MPAS:585
if ((optional_mask & 1u) == 1u) { // MPAS:587
if (((isftcflx == 0x1.0000000000000p+0f) && (FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f))) { // MPAS:588
z0q = 0x1.a36e2e0000000p-14f; // MPAS:594
zolzz = FDIV(FMUL(s.zol, FADD(za, z0q)), za); // MPAS:598
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, z0q)), za); // MPAS:599
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, z0q)), za); // MPAS:600
zol0 = FDIV(FMUL(s.zol, z0q), za); // MPAS:601
if ((s.zol > 0x0.0p+0f)) { // MPAS:603
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:604
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:605
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:606
} else { // MPAS:607
if ((s.zol == 0x0.0p+0f)) { // MPAS:608
s.psih = 0x0.0p+0f; // MPAS:609
psih10 = 0x0.0p+0f; // MPAS:610
psih2 = 0x0.0p+0f; // MPAS:611
} else { // MPAS:612
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:613
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:614
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:615
} // MPAS:616
} // MPAS:617
psiq = FSUB(mm5_log(FDIV(FADD(za, z0q), z0q)), s.psih); // MPAS:619
psit = psiq; // MPAS:620
psiq2 = FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0q), z0q)), psih2); // MPAS:621
psiq10 = FSUB(mm5_log(FDIV(FADD(0x1.4000000000000p+3f, z0q), z0q)), psih10); // MPAS:622
psit2 = psiq2; // MPAS:623
} // MPAS:624
if (((isftcflx == 0x1.0000000000000p+1f) && (FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f))) { // MPAS:625
visc = FMUL(FADD(0x1.51eb860000000p+0f, FMUL(0x1.26e9780000000p-7f, FSUB(scr3, 0x1.1126660000000p+8f))), 0x1.4f8b580000000p-17f); // MPAS:631
restar = FDIV(FMUL(s.ust, s.znt), visc); // MPAS:633
gz0ozt = FMUL(0x1.99999a0000000p-2f, FSUB(FMUL(FMUL(0x1.d333340000000p+2f, FSQRT(FSQRT(restar))), 0x1.af6b3a0000000p-1f), 0x1.4000000000000p+2f)); // MPAS:634
z0t = FDIV(s.znt, mm5_exp(gz0ozt)); // MPAS:638
zolzz = FDIV(FMUL(s.zol, FADD(za, z0t)), za); // MPAS:640
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, z0t)), za); // MPAS:641
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, z0t)), za); // MPAS:642
zol0 = FDIV(FMUL(s.zol, z0t), za); // MPAS:643
if ((s.zol > 0x0.0p+0f)) { // MPAS:645
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:646
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:647
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:648
} else { // MPAS:649
if ((s.zol == 0x0.0p+0f)) { // MPAS:650
s.psih = 0x0.0p+0f; // MPAS:651
psih10 = 0x0.0p+0f; // MPAS:652
psih2 = 0x0.0p+0f; // MPAS:653
} else { // MPAS:654
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:655
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:656
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:657
} // MPAS:658
} // MPAS:659
psit = FSUB(mm5_log(FDIV(FADD(za, z0t), z0t)), s.psih); // MPAS:663
psit2 = FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0t), z0t)), psih2); // MPAS:664
gz0ozq = FMUL(0x1.99999a0000000p-2f, FSUB(FMUL(FMUL(0x1.d333340000000p+2f, FSQRT(FSQRT(restar))), 0x1.8c97f00000000p-1f), 0x1.4000000000000p+2f)); // MPAS:666
z0q = FDIV(s.znt, mm5_exp(gz0ozq)); // MPAS:667
zolzz = FDIV(FMUL(s.zol, FADD(za, z0q)), za); // MPAS:669
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, z0q)), za); // MPAS:670
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, z0q)), za); // MPAS:671
zol0 = FDIV(FMUL(s.zol, z0q), za); // MPAS:672
if ((s.zol > 0x0.0p+0f)) { // MPAS:674
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:675
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:676
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:677
} else { // MPAS:678
if ((s.zol == 0x0.0p+0f)) { // MPAS:679
s.psih = 0x0.0p+0f; // MPAS:680
psih10 = 0x0.0p+0f; // MPAS:681
psih2 = 0x0.0p+0f; // MPAS:682
} else { // MPAS:683
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:684
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:685
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:686
} // MPAS:687
} // MPAS:688
psiq = FSUB(mm5_log(FDIV(FADD(za, z0q), z0q)), s.psih); // MPAS:690
psiq2 = FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0q), z0q)), psih2); // MPAS:691
psiq10 = FSUB(mm5_log(FDIV(FADD(0x1.4000000000000p+3f, z0q), z0q)), psih10); // MPAS:692
} // MPAS:696
} // MPAS:697
if ((optional_mask & 240u) == 240u) { // MPAS:698
s.ck = FMUL(FDIV(c.karman, psix10), FDIV(c.karman, psiq10)); // MPAS:699
s.cd = FMUL(FDIV(c.karman, psix10), FDIV(c.karman, psix10)); // MPAS:700
s.cka = FMUL(FDIV(c.karman, psix), FDIV(c.karman, psiq)); // MPAS:701
s.cda = FMUL(FDIV(c.karman, psix), FDIV(c.karman, psix)); // MPAS:702
} // MPAS:703
if ((optional_mask & 2u) == 2u) { // MPAS:704
if (((iz0tlnd >= 0x1.0000000000000p+0f) && (FSUB(s.xland, 0x1.8000000000000p+0f) <= 0x0.0p+0f))) { // MPAS:705
zl = s.znt; // MPAS:706
visc = FMUL(FADD(0x1.51eb860000000p+0f, FMUL(0x1.26e9780000000p-7f, FSUB(scr3, 0x1.1126660000000p+8f))), 0x1.4f8b580000000p-17f); // MPAS:708
restar = FDIV(FMUL(s.ust, zl), visc); // MPAS:709
if ((iz0tlnd == 0x1.0000000000000p+0f)) { // MPAS:713
czil = mm5_pow(0x1.4000000000000p+3f, FMUL(-0x1.99999a0000000p-2f, FDIV(zl, 0x1.1eb8520000000p-4f))); // MPAS:714
} else if ((iz0tlnd == 0x1.0000000000000p+1f)) { // MPAS:715
czil = 0x1.99999a0000000p-4f; // MPAS:716
} // MPAS:717
z0t = FDIV(s.znt, mm5_exp(FMUL(FMUL(czil, c.karman), FSQRT(restar)))); // MPAS:721
zolzz = FDIV(FMUL(s.zol, FADD(za, z0t)), za); // MPAS:723
zol10 = FDIV(FMUL(s.zol, FADD(0x1.4000000000000p+3f, z0t)), za); // MPAS:724
zol2 = FDIV(FMUL(s.zol, FADD(0x1.0000000000000p+1f, z0t)), za); // MPAS:725
zol0 = FDIV(FMUL(s.zol, z0t), za); // MPAS:726
if ((s.zol > 0x0.0p+0f)) { // MPAS:728
s.psih = FSUB(mm5_psih_stable(zolzz, tables), mm5_psih_stable(zol0, tables)); // MPAS:729
psih10 = FSUB(mm5_psih_stable(zol10, tables), mm5_psih_stable(zol0, tables)); // MPAS:730
psih2 = FSUB(mm5_psih_stable(zol2, tables), mm5_psih_stable(zol0, tables)); // MPAS:731
} else { // MPAS:732
if ((s.zol == 0x0.0p+0f)) { // MPAS:733
s.psih = 0x0.0p+0f; // MPAS:734
psih10 = 0x0.0p+0f; // MPAS:735
psih2 = 0x0.0p+0f; // MPAS:736
} else { // MPAS:737
s.psih = FSUB(mm5_psih_unstable(zolzz, tables), mm5_psih_unstable(zol0, tables)); // MPAS:738
psih10 = FSUB(mm5_psih_unstable(zol10, tables), mm5_psih_unstable(zol0, tables)); // MPAS:739
psih2 = FSUB(mm5_psih_unstable(zol2, tables), mm5_psih_unstable(zol0, tables)); // MPAS:740
} // MPAS:741
} // MPAS:742
psiq = FSUB(mm5_log(FDIV(FADD(za, z0t), z0t)), s.psih); // MPAS:744
psiq2 = FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0t), z0t)), psih2); // MPAS:745
psit = psiq; // MPAS:746
psit2 = psiq2; // MPAS:747
} // MPAS:753
} // MPAS:754
s.ust = FADD(FMUL(0x1.0000000000000p-1f, s.ust), FDIV(FMUL(FMUL(0x1.0000000000000p-1f, c.karman), s.wspd), psix)); // MPAS:756
wspdi = FSQRT(FADD(FMUL(s.ux, s.ux), FMUL(s.vx, s.vx))); // MPAS:758
if ((optional_mask & 8u) == 8u) { // MPAS:759
s.ustm = FADD(FMUL(0x1.0000000000000p-1f, s.ustm), FDIV(FMUL(FMUL(0x1.0000000000000p-1f, c.karman), wspdi), psix)); // MPAS:760
} // MPAS:761
s.u10 = FDIV(FMUL(s.ux, psix10), psix); // MPAS:763
s.v10 = FDIV(FMUL(s.vx, psix10), psix); // MPAS:764
s.th2 = FADD(thgb, FDIV(FMUL(dtg, psit2), psit)); // MPAS:765
s.q2 = FADD(s.qsfc, FDIV(FMUL(FSUB(qx, s.qsfc), psiq2), psiq)); // MPAS:766
s.t2 = FMUL(s.th2, mm5_pow(FDIV(s.psfcpa, c.p1000mb), c.rovcp)); // MPAS:767
if ((FSUB(s.xland, 0x1.8000000000000p+0f) < 0x0.0p+0f)) { // MPAS:769
s.ust = sf_max(s.ust, 0x1.0624de0000000p-10f); // MPAS:770
} // MPAS:771
s.mol = FDIV(FDIV(FMUL(c.karman, dtg), psit), 0x1.0000000000000p+0f); // MPAS:772
denomq = psiq; // MPAS:773
denomq2 = psiq2; // MPAS:774
denomt2 = psit2; // MPAS:775
s.fm = psix; // MPAS:776
s.fh = psit; // MPAS:777


if (!((optional_mask & 4u) == 4u && scm_force_flux)) {
s.qfx = 0x0.0p+0f; // MPAS:787
s.hfx = 0x0.0p+0f; // MPAS:788
}
if (!isfflx) return;
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:796
if (shalwater_z0) { // MPAS:800
s.znt = mm5_depth_dependent_z0(s.water_depth, s.znt, s.ust, tables); // MPAS:801
} else { // MPAS:802
s.znt = FADD(FDIV(FMUL(FMUL(0x1.2f1aa00000000p-6f, s.ust), s.ust), c.g), FDIV(0x1.baeb220000000p-20f, s.ust)); // MPAS:804
s.znt = sf_min(s.znt, 0x1.758e220000000p-9f); // MPAS:806
} // MPAS:807
if ((optional_mask & 1u) == 1u) { // MPAS:813
if ((isftcflx != 0x0.0p+0f)) { // MPAS:814
zw = sf_min(mm5_pow(FDIV(s.ust, 0x1.0f5c280000000p+0f), 0x1.3333340000000p-2f), 0x1.0000000000000p+0f); // MPAS:821
zn1 = FADD(FDIV(FMUL(FMUL(0x1.6872b00000000p-7f, s.ust), s.ust), c.g), 0x1.0ac1fa0000000p-16f); // MPAS:822
zn2 = FADD(FMUL(0x1.4000000000000p+3f, mm5_exp(FMUL(-0x1.3000000000000p+3f, mm5_pow(s.ust, -0x1.554c980000000p-2f)))), FDIV(0x1.baeb220000000p-20f, sf_max(s.ust, 0x1.47ae140000000p-7f))); // MPAS:823
s.znt = FADD(FMUL(FSUB(0x1.0000000000000p+0f, zw), zn1), FMUL(zw, zn2)); // MPAS:825
s.znt = sf_min(s.znt, 0x1.758e220000000p-9f); // MPAS:826
s.znt = sf_max(s.znt, 0x1.10bafe0000000p-23f); // MPAS:827
} // MPAS:828
} // MPAS:829
zl = s.znt; // MPAS:830
} else { // MPAS:831
zl = 0x1.47ae140000000p-7f; // MPAS:832
} // MPAS:833
s.flqc = FDIV(FMUL(FMUL(FMUL(rhox, s.mavail), s.ust), c.karman), denomq); // MPAS:834
dtthx = fabs(FSUB(thx, thgb)); // MPAS:837
if ((dtthx > 0x1.4f8b580000000p-17f)) { // MPAS:838
s.flhc = FDIV(FMUL(FMUL(FMUL(s.cpm, rhox), s.ust), s.mol), FSUB(thx, thgb)); // MPAS:839
} else { // MPAS:842
s.flhc = 0x0.0p+0f; // MPAS:843
} // MPAS:844

if (!((optional_mask & 4u) == 4u && scm_force_flux)) {
s.qfx = FMUL(s.flqc, FSUB(s.qsfc, qx)); // MPAS:857
s.lh = FMUL(c.xlv, s.qfx); // MPAS:859


if ((FSUB(s.xland, 0x1.8000000000000p+0f) > 0x0.0p+0f)) { // MPAS:866
s.hfx = FMUL(s.flhc, FSUB(thgb, thx)); // MPAS:867
} else if ((FSUB(s.xland, 0x1.8000000000000p+0f) < 0x0.0p+0f)) { // MPAS:874
s.hfx = FMUL(s.flhc, FSUB(thgb, thx)); // MPAS:875
} // MPAS:877

}
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:883
zl = s.znt; // MPAS:884
} else { // MPAS:885
zl = 0x1.47ae140000000p-7f; // MPAS:886
} // MPAS:887
s.chs = FDIV(FMUL(s.ust, c.karman), denomq); // MPAS:891
s.cqs2 = FDIV(FMUL(s.ust, c.karman), denomq2); // MPAS:900
s.chs2 = FDIV(FMUL(s.ust, c.karman), denomt2); // MPAS:901

}
kernel void hexv842_mm5_tables(constant Hexv842Mm5TablesResources &r [[buffer(0)]], constant Hexv842Mm5TablesParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.n)) return;
 F z=FMUL(F(float(i)),0.01f);
 r.tables[i]=float(mm5_psim_stable_full(z,r.tables));
 r.tables[1001+i]=float(mm5_psih_stable_full(z,r.tables));
 z=FMUL(-F(float(i)),0.01f);
 r.tables[2002+i]=float(mm5_psim_unstable_full(z,r.tables));
 r.tables[3003+i]=float(mm5_psih_unstable_full(z,r.tables));
}
kernel void hexv842_mm5_revised(constant Hexv842Mm5RevisedResources &r [[buffer(0)]], constant Hexv842Mm5RevisedParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.n)) return;
 MM5State s; MM5Constants c;
 s.znt=r.a[0*p.n+i];
 s.ust=r.a[1*p.n+i];
 s.ustm=r.a[2*p.n+i];
 s.mol=r.a[3*p.n+i];
 s.hfx=r.a[4*p.n+i];
 s.qfx=r.a[5*p.n+i];
 s.qsfc=r.a[6*p.n+i];
 s.zol=r.a[7*p.n+i];
 s.regime=r.a[8*p.n+i];
 s.psim=r.a[9*p.n+i];
 s.psih=r.a[10*p.n+i];
 s.fm=r.a[11*p.n+i];
 s.fh=r.a[12*p.n+i];
 s.lh=r.a[13*p.n+i];
 s.u10=r.a[14*p.n+i];
 s.v10=r.a[15*p.n+i];
 s.th2=r.a[16*p.n+i];
 s.t2=r.a[17*p.n+i];
 s.q2=r.a[18*p.n+i];
 s.chs=r.a[19*p.n+i];
 s.chs2=r.a[20*p.n+i];
 s.cqs2=r.a[21*p.n+i];
 s.flhc=r.a[22*p.n+i];
 s.flqc=r.a[23*p.n+i];
 s.qgh=r.a[24*p.n+i];
 s.rmol=r.a[25*p.n+i];
 s.wspd=r.a[26*p.n+i];
 s.br=r.a[27*p.n+i];
 s.gz1oz0=r.a[28*p.n+i];
 s.cpm=r.a[29*p.n+i];
 s.ck=r.a[30*p.n+i];
 s.cka=r.a[31*p.n+i];
 s.cd=r.a[32*p.n+i];
 s.cda=r.a[33*p.n+i];
 s.ux=r.a[34*p.n+i];
 s.vx=r.a[35*p.n+i];
 s.t1d=r.a[36*p.n+i];
 s.qv1d=r.a[37*p.n+i];
 s.p1d=r.a[38*p.n+i];
 s.dz8w1d=r.a[39*p.n+i];
 s.psfcpa=r.a[40*p.n+i];
 s.tsk=r.a[41*p.n+i];
 s.pblh=r.a[42*p.n+i];
 s.mavail=r.a[43*p.n+i];
 s.xland=r.a[44*p.n+i];
 s.lakemask=r.a[45*p.n+i];
 s.dx=r.a[46*p.n+i];
 s.water_depth=r.a[47*p.n+i];
 c.cp=r.constants[0];
 c.g=r.constants[1];
 c.rovcp=r.constants[2];
 c.r=r.constants[3];
 c.xlv=r.constants[4];
 c.svp1=r.constants[5];
 c.svp2=r.constants[6];
 c.svp3=r.constants[7];
 c.svpt0=r.constants[8];
 c.ep1=r.constants[9];
 c.ep2=r.constants[10];
 c.karman=r.constants[11];
 c.p1000mb=r.constants[12];
 mm5_revised(s,c,r.tables,p.isfflx,p.isftcflx,p.iz0tlnd,p.scm_force_flux,p.shalwater_z0,uint(p.optional_mask));
 r.a[0*p.n+i]=float(s.znt);
 r.a[1*p.n+i]=float(s.ust);
 r.a[2*p.n+i]=float(s.ustm);
 r.a[3*p.n+i]=float(s.mol);
 r.a[4*p.n+i]=float(s.hfx);
 r.a[5*p.n+i]=float(s.qfx);
 r.a[6*p.n+i]=float(s.qsfc);
 r.a[7*p.n+i]=float(s.zol);
 r.a[8*p.n+i]=float(s.regime);
 r.a[9*p.n+i]=float(s.psim);
 r.a[10*p.n+i]=float(s.psih);
 r.a[11*p.n+i]=float(s.fm);
 r.a[12*p.n+i]=float(s.fh);
 r.a[13*p.n+i]=float(s.lh);
 r.a[14*p.n+i]=float(s.u10);
 r.a[15*p.n+i]=float(s.v10);
 r.a[16*p.n+i]=float(s.th2);
 r.a[17*p.n+i]=float(s.t2);
 r.a[18*p.n+i]=float(s.q2);
 r.a[19*p.n+i]=float(s.chs);
 r.a[20*p.n+i]=float(s.chs2);
 r.a[21*p.n+i]=float(s.cqs2);
 r.a[22*p.n+i]=float(s.flhc);
 r.a[23*p.n+i]=float(s.flqc);
 r.a[24*p.n+i]=float(s.qgh);
 r.a[25*p.n+i]=float(s.rmol);
 r.a[26*p.n+i]=float(s.wspd);
 r.a[27*p.n+i]=float(s.br);
 r.a[28*p.n+i]=float(s.gz1oz0);
 r.a[29*p.n+i]=float(s.cpm);
 r.a[30*p.n+i]=float(s.ck);
 r.a[31*p.n+i]=float(s.cka);
 r.a[32*p.n+i]=float(s.cd);
 r.a[33*p.n+i]=float(s.cda);
}
