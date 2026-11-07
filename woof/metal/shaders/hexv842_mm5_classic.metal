#include <metal_stdlib>
#include "hexv842_mm5_classic_tables_bindings.metalh"
#include "hexv842_mm5_classic_bindings.metalh"
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
// MPAS physics_wrf/module_sf_sfclay.F:265-946
static __attribute__((noinline)) void mm5_classic(thread MM5State &s, thread const MM5Constants &c, device const float *tables, int isfflx, int isftcflx, int iz0tlnd, int scm_force_flux, int shalwater_z0, unsigned int optional_mask) {
F pl, thcon, tvcon, e1, zl, tskv, dthvdz, dthvm, vconv, rzol, rzol2, rzol10, zol2, zol10, dtg, psix, dtthx, psix10, psit, psit2, psiq, psiq2, psiq10, fluxc, vsgd, z0q, visc, restar, czil, gz0ozq, gz0ozt, zw, zn1, zn2, zolzz, zol0, zl2, zl10, z0t, za, thvx, zqkl, zqklp1, thx, qx, psih2, psim2, psih10, psim10, denomq, denomq2, denomt2, wspdi, gz2oz0, gz10oz0, rhox, govrth, tgdsa, scr3, scr4, thgb, psfc, pq, pq2, pq10, czc;
int nzol, nzol2, nzol10;
psfc = FDIV(s.psfcpa, 0x1.f400000000000p+9f); // MPAS:393
tgdsa = s.tsk; // MPAS:399
thgb = FMUL(s.tsk, mm5_pow(FDIV(c.p1000mb, s.psfcpa), c.rovcp)); // MPAS:402



pl = FDIV(s.p1d, 0x1.f400000000000p+9f); // MPAS:427
scr3 = s.t1d; // MPAS:428
thcon = mm5_pow(FDIV(FMUL(c.p1000mb, 0x1.0624de0000000p-10f), pl), c.rovcp); // MPAS:430
thx = FMUL(scr3, thcon); // MPAS:431
scr4 = scr3; // MPAS:432
thvx = thx; // MPAS:433
qx = 0x0.0p+0f; // MPAS:434

s.qgh = 0x0.0p+0f; // MPAS:438
s.flhc = 0x0.0p+0f; // MPAS:439
s.flqc = 0x0.0p+0f; // MPAS:440
s.cpm = c.cp; // MPAS:441
qx = s.qv1d; // MPAS:446
tvcon = FADD(0x1.0000000000000p+0f, FMUL(c.ep1, qx)); // MPAS:447
thvx = FMUL(thx, tvcon); // MPAS:448
scr4 = FMUL(scr3, tvcon); // MPAS:449

e1 = FMUL(c.svp1, mm5_exp(FDIV(FMUL(c.svp2, FSUB(tgdsa, c.svpt0)), FSUB(tgdsa, c.svp3)))); // MPAS:453
if (((s.xland > 0x1.8000000000000p+0f) || (s.qsfc <= 0x0.0p+0f))) s.qsfc = FDIV(FMUL(c.ep2, e1), FSUB(psfc, e1)); // MPAS:455
e1 = FMUL(c.svp1, mm5_exp(FDIV(FMUL(c.svp2, FSUB(s.t1d, c.svpt0)), FSUB(s.t1d, c.svp3)))); // MPAS:458
pl = FDIV(s.p1d, 0x1.f400000000000p+9f); // MPAS:459
s.qgh = FDIV(FMUL(c.ep2, e1), FSUB(pl, e1)); // MPAS:460
s.cpm = FMUL(c.cp, FADD(0x1.0000000000000p+0f, FMUL(0x1.99999a0000000p-1f, qx))); // MPAS:461


zqklp1 = 0x0.0p+0f; // MPAS:469
rhox = FDIV(FMUL(psfc, 0x1.f400000000000p+9f), FMUL(c.r, scr4)); // MPAS:470

zqkl = FADD(s.dz8w1d, zqklp1); // MPAS:474

za = FMUL(0x1.0000000000000p-1f, FADD(zqkl, zqklp1)); // MPAS:478

govrth = FDIV(c.g, thx); // MPAS:482

s.gz1oz0 = mm5_log(FDIV(za, s.znt)); // MPAS:489
gz2oz0 = mm5_log(FDIV(0x1.0000000000000p+1f, s.znt)); // MPAS:490
gz10oz0 = mm5_log(FDIV(0x1.4000000000000p+3f, s.znt)); // MPAS:491
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:492
zl = s.znt; // MPAS:493
} else { // MPAS:494
zl = 0x1.47ae140000000p-7f; // MPAS:495
} // MPAS:496
s.wspd = FSQRT(FADD(FMUL(s.ux, s.ux), FMUL(s.vx, s.vx))); // MPAS:497
tskv = FMUL(thgb, FADD(0x1.0000000000000p+0f, FMUL(c.ep1, s.qsfc))); // MPAS:499
dthvdz = FSUB(thvx, tskv); // MPAS:500
if ((s.xland < 0x1.8000000000000p+0f)) { // MPAS:507
fluxc = sf_max(FADD(FDIV(FDIV(s.hfx, rhox), c.cp), FDIV(FMUL(FMUL(c.ep1, tskv), s.qfx), rhox)), 0x0.0p+0f); // MPAS:508
vconv = FMUL(0x1.0000000000000p+0f, mm5_pow(FMUL(FMUL(FDIV(c.g, tgdsa), s.pblh), fluxc), 0x1.51eb860000000p-2f)); // MPAS:510
} else { // MPAS:511
if ((-(dthvdz) >= 0x0.0p+0f)) { // MPAS:512
dthvm = -(dthvdz); // MPAS:513
} else { // MPAS:514
dthvm = 0x0.0p+0f; // MPAS:515
} // MPAS:516
vconv = FSQRT(dthvm); // MPAS:519
} // MPAS:520
vsgd = FMUL(0x1.47ae140000000p-2f, mm5_pow(sf_max(FSUB(FDIV(s.dx, 0x1.3880000000000p+12f), 0x1.0000000000000p+0f), 0x0.0p+0f), 0x1.51eb860000000p-2f)); // MPAS:522
s.wspd = FSQRT(FADD(FADD(FMUL(s.wspd, s.wspd), FMUL(vconv, vconv)), FMUL(vsgd, vsgd))); // MPAS:523
s.wspd = sf_max(s.wspd, 0x1.99999a0000000p-4f); // MPAS:524
s.br = FDIV(FMUL(FMUL(govrth, za), dthvdz), FMUL(s.wspd, s.wspd)); // MPAS:525
if ((s.mol < 0x0.0p+0f)) s.br = sf_min(s.br, 0x0.0p+0f); // MPAS:527
s.rmol = FMUL(FMUL(FMUL(-(govrth), dthvdz), za), c.karman); // MPAS:529

if (!(s.br < 0x0.0p+0f) && !(s.br < 0x1.99999a0000000p-3f)) {
s.regime = 0x1.0000000000000p+0f; // MPAS:567
s.psim = FMUL(-0x1.4000000000000p+3f, s.gz1oz0); // MPAS:568
s.psim = sf_max(s.psim, -0x1.4000000000000p+3f); // MPAS:570
s.psih = s.psim; // MPAS:571
psim10 = FMUL(FDIV(0x1.4000000000000p+3f, za), s.psim); // MPAS:572
psim10 = sf_max(psim10, -0x1.4000000000000p+3f); // MPAS:573
psih10 = psim10; // MPAS:574
psim2 = FMUL(FDIV(0x1.0000000000000p+1f, za), s.psim); // MPAS:575
psim2 = sf_max(psim2, -0x1.4000000000000p+3f); // MPAS:576
psih2 = psim2; // MPAS:577
if ((s.ust < 0x1.47ae140000000p-7f)) { // MPAS:580
s.rmol = FMUL(s.br, s.gz1oz0); // MPAS:581
} else { // MPAS:582
s.rmol = FDIV(FMUL(FMUL(FMUL(c.karman, govrth), za), s.mol), FMUL(s.ust, s.ust)); // MPAS:583
} // MPAS:584
s.rmol = sf_min(s.rmol, 0x1.3ff7ce0000000p+3f); // MPAS:585
s.rmol = FDIV(s.rmol, za); // MPAS:586
} else if (!(s.br < 0x0.0p+0f) && !(s.br == 0x0.0p+0f)) {
s.regime = 0x1.0000000000000p+1f; // MPAS:593
s.psim = FDIV(FMUL(FMUL(-0x1.4000000000000p+2f, s.br), s.gz1oz0), FSUB(0x1.19999a0000000p+0f, FMUL(0x1.4000000000000p+2f, s.br))); // MPAS:594
s.psim = sf_max(s.psim, -0x1.4000000000000p+3f); // MPAS:596
s.psih = s.psim; // MPAS:598
psim10 = FMUL(FDIV(0x1.4000000000000p+3f, za), s.psim); // MPAS:599
psim10 = sf_max(psim10, -0x1.4000000000000p+3f); // MPAS:600
psih10 = psim10; // MPAS:601
psim2 = FMUL(FDIV(0x1.0000000000000p+1f, za), s.psim); // MPAS:602
psim2 = sf_max(psim2, -0x1.4000000000000p+3f); // MPAS:603
psih2 = psim2; // MPAS:604
s.zol = FDIV(FMUL(s.br, s.gz1oz0), FSUB(0x1.0000a80000000p+0f, FMUL(0x1.4000000000000p+2f, s.br))); // MPAS:610
if ((s.zol > 0x1.0000000000000p-1f)) { // MPAS:612
s.zol = FADD(FMUL(FMUL(FADD(FMUL(0x1.e3d70a0000000p+0f, s.gz1oz0), 0x1.61999a0000000p+5f), s.br), s.br), FMUL(FSUB(FMUL(0x1.2e147a0000000p+0f, s.gz1oz0), 0x1.5eb8520000000p+0f), s.br)); // MPAS:616
s.zol = sf_min(s.zol, 0x1.3ff7ce0000000p+3f); // MPAS:618
} // MPAS:619
s.rmol = FDIV(s.zol, za); // MPAS:622
} else if (!(s.br < 0x0.0p+0f)) {
s.regime = 0x1.8000000000000p+1f; // MPAS:628
s.psim = 0x0.0p+0f; // MPAS:629
s.psih = s.psim; // MPAS:630
psim10 = 0x0.0p+0f; // MPAS:631
psih10 = psim10; // MPAS:632
psim2 = 0x0.0p+0f; // MPAS:633
psih2 = psim2; // MPAS:634
if ((s.ust < 0x1.47ae140000000p-7f)) { // MPAS:637
s.zol = FMUL(s.br, s.gz1oz0); // MPAS:638
} else { // MPAS:639
s.zol = FDIV(FMUL(FMUL(FMUL(c.karman, govrth), za), s.mol), FMUL(s.ust, s.ust)); // MPAS:640
} // MPAS:641
s.rmol = FDIV(s.zol, za); // MPAS:643
} else {
s.regime = 0x1.0000000000000p+2f; // MPAS:650
if ((s.ust < 0x1.47ae140000000p-7f)) { // MPAS:651
s.zol = FMUL(s.br, s.gz1oz0); // MPAS:652
} else { // MPAS:653
s.zol = FDIV(FMUL(FMUL(FMUL(c.karman, govrth), za), s.mol), FMUL(s.ust, s.ust)); // MPAS:654
} // MPAS:655
zol10 = FMUL(FDIV(0x1.4000000000000p+3f, za), s.zol); // MPAS:656
zol2 = FMUL(FDIV(0x1.0000000000000p+1f, za), s.zol); // MPAS:657
s.zol = sf_min(s.zol, 0x0.0p+0f); // MPAS:658
s.zol = sf_max(s.zol, -0x1.3fff2e0000000p+3f); // MPAS:659
zol10 = sf_min(zol10, 0x0.0p+0f); // MPAS:660
zol10 = sf_max(zol10, -0x1.3fff2e0000000p+3f); // MPAS:661
zol2 = sf_min(zol2, 0x0.0p+0f); // MPAS:662
zol2 = sf_max(zol2, -0x1.3fff2e0000000p+3f); // MPAS:663
nzol = int(FMUL(-(s.zol), 0x1.9000000000000p+6f)); // MPAS:664
rzol = FSUB(FMUL(-(s.zol), 0x1.9000000000000p+6f), nzol); // MPAS:665
nzol10 = int(FMUL(-(zol10), 0x1.9000000000000p+6f)); // MPAS:666
rzol10 = FSUB(FMUL(-(zol10), 0x1.9000000000000p+6f), nzol10); // MPAS:667
nzol2 = int(FMUL(-(zol2), 0x1.9000000000000p+6f)); // MPAS:668
rzol2 = FSUB(FMUL(-(zol2), 0x1.9000000000000p+6f), nzol2); // MPAS:669
s.psim = FADD(tables[4*1001 + (int)(nzol)], FMUL(rzol, FSUB(tables[4*1001 + int(FADD(nzol, 0x1.0000000000000p+0f))], tables[4*1001 + (int)(nzol)]))); // MPAS:670
s.psih = FADD(tables[5*1001 + (int)(nzol)], FMUL(rzol, FSUB(tables[5*1001 + int(FADD(nzol, 0x1.0000000000000p+0f))], tables[5*1001 + (int)(nzol)]))); // MPAS:671
psim10 = FADD(tables[4*1001 + (int)(nzol10)], FMUL(rzol10, FSUB(tables[4*1001 + int(FADD(nzol10, 0x1.0000000000000p+0f))], tables[4*1001 + (int)(nzol10)]))); // MPAS:672
psih10 = FADD(tables[5*1001 + (int)(nzol10)], FMUL(rzol10, FSUB(tables[5*1001 + int(FADD(nzol10, 0x1.0000000000000p+0f))], tables[5*1001 + (int)(nzol10)]))); // MPAS:673
psim2 = FADD(tables[4*1001 + (int)(nzol2)], FMUL(rzol2, FSUB(tables[4*1001 + int(FADD(nzol2, 0x1.0000000000000p+0f))], tables[4*1001 + (int)(nzol2)]))); // MPAS:674
psih2 = FADD(tables[5*1001 + (int)(nzol2)], FMUL(rzol2, FSUB(tables[5*1001 + int(FADD(nzol2, 0x1.0000000000000p+0f))], tables[5*1001 + (int)(nzol2)]))); // MPAS:675
s.psih = sf_min(s.psih, FMUL(0x1.cccccc0000000p-1f, s.gz1oz0)); // MPAS:681
s.psim = sf_min(s.psim, FMUL(0x1.cccccc0000000p-1f, s.gz1oz0)); // MPAS:682
psih2 = sf_min(psih2, FMUL(0x1.cccccc0000000p-1f, gz2oz0)); // MPAS:683
psim10 = sf_min(psim10, FMUL(0x1.cccccc0000000p-1f, gz10oz0)); // MPAS:684
psih10 = sf_min(psih10, FMUL(0x1.cccccc0000000p-1f, gz10oz0)); // MPAS:686
s.rmol = FDIV(s.zol, za); // MPAS:688
}
dtg = FSUB(thx, thgb); // MPAS:696
psix = FSUB(s.gz1oz0, s.psim); // MPAS:697
psix10 = FSUB(gz10oz0, psim10); // MPAS:698
psit = sf_max(FSUB(s.gz1oz0, s.psih), 0x1.0000000000000p+1f); // MPAS:701
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:703
zl = s.znt; // MPAS:704
} else { // MPAS:705
zl = 0x1.47ae140000000p-7f; // MPAS:706
} // MPAS:707
psiq = FSUB(mm5_log(FADD(FDIV(FMUL(FMUL(c.karman, s.ust), za), 0x1.92a7380000000p-16f), FDIV(za, zl))), s.psih); // MPAS:708
psit2 = FSUB(gz2oz0, psih2); // MPAS:709
psiq2 = FSUB(mm5_log(FADD(FDIV(FMUL(FMUL(c.karman, s.ust), 0x1.0000000000000p+1f), 0x1.92a7380000000p-16f), FDIV(0x1.0000000000000p+1f, zl))), psih2); // MPAS:710
psiq10 = FSUB(mm5_log(FADD(FDIV(FMUL(FMUL(c.karman, s.ust), 0x1.4000000000000p+3f), 0x1.92a7380000000p-16f), FDIV(0x1.4000000000000p+3f, zl))), psih10); // MPAS:712
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:716
visc = FMUL(FADD(0x1.51eb860000000p+0f, FMUL(0x1.26e9780000000p-7f, FSUB(scr3, 0x1.1126660000000p+8f))), 0x1.4f8b580000000p-17f); // MPAS:717
restar = FDIV(FMUL(s.ust, s.znt), visc); // MPAS:720
z0t = FMUL(0x1.cd5f9a0000000p-15f, mm5_pow(restar, -0x1.3333340000000p-1f)); // MPAS:721
z0t = sf_min(z0t, 0x1.a36e2e0000000p-14f); // MPAS:722
z0t = sf_max(z0t, 0x1.12e0be0000000p-29f); // MPAS:723
z0q = z0t; // MPAS:724
psiq = sf_max(FSUB(mm5_log(FDIV(FADD(za, z0q), z0q)), s.psih), 0x1.0000000000000p+1f); // MPAS:726
psit = sf_max(FSUB(mm5_log(FDIV(FADD(za, z0t), z0t)), s.psih), 0x1.0000000000000p+1f); // MPAS:727
psiq2 = sf_max(FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0q), z0q)), psih2), 0x1.0000000000000p+1f); // MPAS:728
psit2 = sf_max(FSUB(mm5_log(FDIV(FADD(0x1.0000000000000p+1f, z0t), z0t)), psih2), 0x1.0000000000000p+1f); // MPAS:729
psiq10 = sf_max(FSUB(mm5_log(FDIV(FADD(0x1.4000000000000p+3f, z0q), z0q)), psih10), 0x1.0000000000000p+1f); // MPAS:730
} // MPAS:731
if ((optional_mask & 1u) == 1u) { // MPAS:733
if (((isftcflx == 0x1.0000000000000p+0f) && (FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f))) { // MPAS:734
z0q = 0x1.a36e2e0000000p-14f; // MPAS:740
psiq = FSUB(mm5_log(FDIV(za, z0q)), s.psih); // MPAS:741
psit = psiq; // MPAS:742
psiq2 = FSUB(mm5_log(FDIV(0x1.0000000000000p+1f, z0q)), psih2); // MPAS:743
psiq10 = FSUB(mm5_log(FDIV(0x1.4000000000000p+3f, z0q)), psih10); // MPAS:744
psit2 = psiq2; // MPAS:745
} // MPAS:746
if (((isftcflx == 0x1.0000000000000p+1f) && (FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f))) { // MPAS:747
visc = FMUL(FADD(0x1.51eb860000000p+0f, FMUL(0x1.26e9780000000p-7f, FSUB(scr3, 0x1.1126660000000p+8f))), 0x1.4f8b580000000p-17f); // MPAS:753
restar = FDIV(FMUL(s.ust, s.znt), visc); // MPAS:755
gz0ozt = FMUL(0x1.99999a0000000p-2f, FSUB(FMUL(FMUL(0x1.d333340000000p+2f, FSQRT(FSQRT(restar))), 0x1.af6b3a0000000p-1f), 0x1.4000000000000p+2f)); // MPAS:756
gz0ozq = FMUL(0x1.99999a0000000p-2f, FSUB(FMUL(FMUL(0x1.d333340000000p+2f, FSQRT(FSQRT(restar))), 0x1.8c97f00000000p-1f), 0x1.4000000000000p+2f)); // MPAS:757
psit = FADD(FSUB(s.gz1oz0, s.psih), gz0ozt); // MPAS:758
psiq = FADD(FSUB(s.gz1oz0, s.psih), gz0ozq); // MPAS:759
psit2 = FADD(FSUB(gz2oz0, psih2), gz0ozt); // MPAS:760
psiq2 = FADD(FSUB(gz2oz0, psih2), gz0ozq); // MPAS:761
psiq10 = FADD(FSUB(gz10oz0, s.psih), gz0ozq); // MPAS:762
} // MPAS:763
} // MPAS:764
if ((optional_mask & 240u) == 240u) { // MPAS:765
s.ck = FMUL(FDIV(c.karman, psix10), FDIV(c.karman, psiq10)); // MPAS:766
s.cd = FMUL(FDIV(c.karman, psix10), FDIV(c.karman, psix10)); // MPAS:767
s.cka = FMUL(FDIV(c.karman, psix), FDIV(c.karman, psiq)); // MPAS:768
s.cda = FMUL(FDIV(c.karman, psix), FDIV(c.karman, psix)); // MPAS:769
} // MPAS:770
if ((optional_mask & 2u) == 2u) { // MPAS:771
if (((iz0tlnd >= 0x1.0000000000000p+0f) && (FSUB(s.xland, 0x1.8000000000000p+0f) <= 0x0.0p+0f))) { // MPAS:772
zl = s.znt; // MPAS:773
visc = FMUL(FADD(0x1.51eb860000000p+0f, FMUL(0x1.26e9780000000p-7f, FSUB(scr3, 0x1.1126660000000p+8f))), 0x1.4f8b580000000p-17f); // MPAS:775
restar = FDIV(FMUL(s.ust, zl), visc); // MPAS:776
if ((iz0tlnd == 0x1.0000000000000p+0f)) { // MPAS:780
czil = mm5_pow(0x1.4000000000000p+3f, FMUL(-0x1.99999a0000000p-2f, FDIV(zl, 0x1.1eb8520000000p-4f))); // MPAS:781
} else if ((iz0tlnd == 0x1.0000000000000p+1f)) { // MPAS:782
czil = 0x1.99999a0000000p-4f; // MPAS:783
} // MPAS:784
psit = FADD(FSUB(s.gz1oz0, s.psih), FMUL(FMUL(czil, c.karman), FSQRT(restar))); // MPAS:786
psiq = FADD(FSUB(s.gz1oz0, s.psih), FMUL(FMUL(czil, c.karman), FSQRT(restar))); // MPAS:787
psit2 = FADD(FSUB(gz2oz0, psih2), FMUL(FMUL(czil, c.karman), FSQRT(restar))); // MPAS:788
psiq2 = FADD(FSUB(gz2oz0, psih2), FMUL(FMUL(czil, c.karman), FSQRT(restar))); // MPAS:789
} // MPAS:791
} // MPAS:792
s.ust = FADD(FMUL(0x1.0000000000000p-1f, s.ust), FDIV(FMUL(FMUL(0x1.0000000000000p-1f, c.karman), s.wspd), psix)); // MPAS:794
wspdi = FSQRT(FADD(FMUL(s.ux, s.ux), FMUL(s.vx, s.vx))); // MPAS:796
if ((optional_mask & 8u) == 8u) { // MPAS:797
s.ustm = FADD(FMUL(0x1.0000000000000p-1f, s.ustm), FDIV(FMUL(FMUL(0x1.0000000000000p-1f, c.karman), wspdi), psix)); // MPAS:798
} // MPAS:799
s.u10 = FDIV(FMUL(s.ux, psix10), psix); // MPAS:800
s.v10 = FDIV(FMUL(s.vx, psix10), psix); // MPAS:801
s.th2 = FADD(thgb, FDIV(FMUL(dtg, psit2), psit)); // MPAS:802
s.q2 = FADD(s.qsfc, FDIV(FMUL(FSUB(qx, s.qsfc), psiq2), psiq)); // MPAS:803
s.t2 = FMUL(s.th2, mm5_pow(FDIV(s.psfcpa, c.p1000mb), c.rovcp)); // MPAS:805
if ((FSUB(s.xland, 0x1.8000000000000p+0f) < 0x0.0p+0f)) { // MPAS:812
s.ust = sf_max(s.ust, 0x1.99999a0000000p-4f); // MPAS:813
} // MPAS:814
s.mol = FDIV(FDIV(FMUL(c.karman, dtg), psit), 0x1.0000000000000p+0f); // MPAS:815
denomq = psiq; // MPAS:816
denomq2 = psiq2; // MPAS:817
denomt2 = psit2; // MPAS:818
s.fm = psix; // MPAS:819
s.fh = psit; // MPAS:820


if (!((optional_mask & 4u) == 4u && scm_force_flux == 1)) {
s.qfx = 0x0.0p+0f; // MPAS:830
s.hfx = 0x0.0p+0f; // MPAS:831
}
if (!isfflx) return;
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:840
s.znt = FADD(FDIV(FMUL(FMUL(0x1.2f1aa00000000p-6f, s.ust), s.ust), c.g), FDIV(0x1.baeb220000000p-20f, s.ust)); // MPAS:843
s.znt = sf_min(s.znt, 0x1.758e220000000p-9f); // MPAS:845
if ((optional_mask & 1u) == 1u) { // MPAS:851
if ((isftcflx != 0x0.0p+0f)) { // MPAS:852
zw = sf_min(mm5_pow(FDIV(s.ust, 0x1.0f5c280000000p+0f), 0x1.3333340000000p-2f), 0x1.0000000000000p+0f); // MPAS:859
zn1 = FADD(FDIV(FMUL(FMUL(0x1.6872b00000000p-7f, s.ust), s.ust), c.g), 0x1.0ac1fa0000000p-16f); // MPAS:860
zn2 = FADD(FMUL(0x1.4000000000000p+3f, mm5_exp(FMUL(-0x1.3000000000000p+3f, mm5_pow(s.ust, -0x1.554c980000000p-2f)))), FDIV(0x1.baeb220000000p-20f, sf_max(s.ust, 0x1.47ae140000000p-7f))); // MPAS:861
s.znt = FADD(FMUL(FSUB(0x1.0000000000000p+0f, zw), zn1), FMUL(zw, zn2)); // MPAS:863
s.znt = sf_min(s.znt, 0x1.758e220000000p-9f); // MPAS:864
s.znt = sf_max(s.znt, 0x1.10bafe0000000p-23f); // MPAS:865
} // MPAS:866
} // MPAS:867
zl = s.znt; // MPAS:868
} else { // MPAS:869
zl = 0x1.47ae140000000p-7f; // MPAS:870
} // MPAS:871
s.flqc = FDIV(FMUL(FMUL(FMUL(rhox, s.mavail), s.ust), c.karman), denomq); // MPAS:872
dtthx = fabs(FSUB(thx, thgb)); // MPAS:875
if ((dtthx > 0x1.4f8b580000000p-17f)) { // MPAS:876
s.flhc = FDIV(FMUL(FMUL(FMUL(s.cpm, rhox), s.ust), s.mol), FSUB(thx, thgb)); // MPAS:877
} else { // MPAS:880
s.flhc = 0x0.0p+0f; // MPAS:881
} // MPAS:882

if (!((optional_mask & 4u) == 4u && scm_force_flux == 1)) {
s.qfx = FMUL(s.flqc, FSUB(s.qsfc, qx)); // MPAS:894
s.qfx = sf_max(s.qfx, 0x0.0p+0f); // MPAS:895
s.lh = FMUL(c.xlv, s.qfx); // MPAS:896


if ((FSUB(s.xland, 0x1.8000000000000p+0f) > 0x0.0p+0f)) { // MPAS:903
s.hfx = FMUL(s.flhc, FSUB(thgb, thx)); // MPAS:904
} else if ((FSUB(s.xland, 0x1.8000000000000p+0f) < 0x0.0p+0f)) { // MPAS:911
s.hfx = FMUL(s.flhc, FSUB(thgb, thx)); // MPAS:912
s.hfx = sf_max(s.hfx, -0x1.f400000000000p+7f); // MPAS:913
} // MPAS:914

}
if ((FSUB(s.xland, 0x1.8000000000000p+0f) >= 0x0.0p+0f)) { // MPAS:920
zl = s.znt; // MPAS:921
} else { // MPAS:922
zl = 0x1.47ae140000000p-7f; // MPAS:923
} // MPAS:924
s.chs = FDIV(FMUL(s.ust, c.karman), denomq); // MPAS:925
s.cqs2 = FDIV(FMUL(s.ust, c.karman), denomq2); // MPAS:930
s.chs2 = FDIV(FMUL(s.ust, c.karman), denomt2); // MPAS:931

}
kernel void hexv842_mm5_classic_tables(constant Hexv842Mm5ClassicTablesResources &r [[buffer(0)]], constant Hexv842Mm5ClassicTablesParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.n)) return;
 F z=FMUL(-F(float(i)),0.01f);
 F x=mm5_pow(FSUB(1.f,FMUL(16.f,z)),0.25f);
 r.tables[4004+i]=float(FADD(FSUB(FADD(FMUL(2.f,mm5_log(FMUL(0.5f,FADD(1.f,x)))),mm5_log(FMUL(0.5f,FADD(1.f,FMUL(x,x))))),FMUL(2.f,mm5_atan(x))),1.57079637050628662109375f));
 F y=mm5_pow(FSUB(1.f,FMUL(16.f,z)),0.5f);
 r.tables[5005+i]=float(FMUL(2.f,mm5_log(FMUL(0.5f,FADD(1.f,y)))));
}
kernel void hexv842_mm5_classic(constant Hexv842Mm5ClassicResources &r [[buffer(0)]], constant Hexv842Mm5ClassicParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
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
 mm5_classic(s,c,r.tables,p.isfflx,p.isftcflx,p.iz0tlnd,p.scm_force_flux,p.shalwater_z0,uint(p.optional_mask));
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
