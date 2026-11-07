#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "soft_urban_trig.metalh"
#include "hexv842_mpas_sf_mynn_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float native_mpas_atan(float x) { return as_type<float>(soft_urban_atanf(as_type<uint>(x))); }

// MPAS physics_mmm/sf_mynn.F90:130-996 sf_mynn_run
static inline void m_sf_mynn_run(float u1d, float v1d, float t1d, float qv1d, float p1d, float dz8w1d, float rho1d, float u1d2, float v1d2, float dz2w1d, float cp, float g, float rovcp, float r, float xlv, float psfcpa, thread float &chs, thread float &chs2, thread float &cqs2, thread float &cpm, float pblh, thread float &rmol, thread float &znt, thread float &ust, float mavail, thread float &zol, thread float &mol, thread float &regime, thread float &psim, thread float &psih, float xland, thread float &hfx, thread float &qfx, float tsk, thread float &u10, thread float &v10, thread float &th2, thread float &t2, thread float &q2, thread float &flhc, thread float &flqc, float snowh, thread float &qgh, thread float &qsfc, thread float &lh, thread float &gz1oz0, thread float &wspd, thread float &br, int isfflx, float dx, float svp1, float svp2, float svp3, float svpt0, float ep1, float ep2, float karman, thread float &ch, float qcg, int itimestep, thread float &wstar, thread float &qstar, thread float &ustm, thread float &ck, thread float &cka, thread float &cd, thread float &cda, bool spp_pbl, float rstoch1d, int isftcflx, int iz0tlnd, int its, int ite, bool has_isftcflx, bool has_iz0tlnd, bool has_ck, bool has_cka, bool has_cd, bool has_cda, bool has_ustm, device const float *tables);
// MPAS physics_mmm/sf_mynn.F90:999-1073 zilitinkevich_1995
static inline void m_zilitinkevich_1995(float z_0, thread float &zt, thread float &zq, float restar, float ustar, float karman, float landsea, int iz0tlnd2, bool spp_pbl, float rstoch, bool has_iz0tlnd2);
// MPAS physics_mmm/sf_mynn.F90:1076-1109 davis_etal_2008
static inline void m_davis_etal_2008(thread float &z_0, float ustar);
// MPAS physics_mmm/sf_mynn.F90:1112-1141 taylor_yelland_2001
static inline void m_taylor_yelland_2001(thread float &z_0, float ustar, float wsp10);
// MPAS physics_mmm/sf_mynn.F90:1144-1172 charnock_1955
static inline void m_charnock_1955(thread float &z_0, float ustar, float wsp10, float visc, float zu);
// MPAS physics_mmm/sf_mynn.F90:1175-1208 edson_etal_2013
static inline void m_edson_etal_2013(thread float &z_0, float ustar, float wsp10, float visc, float zu);
// MPAS physics_mmm/sf_mynn.F90:1211-1250 garratt_1992
static inline void m_garratt_1992(thread float &zt, thread float &zq, float z_0, float ren, float landsea);
// MPAS physics_mmm/sf_mynn.F90:1253-1302 fairall_etal_2003
static inline void m_fairall_etal_2003(thread float &zt, thread float &zq, float ren, float ustar, float visc, float rstoch, bool spp_pbl);
// MPAS physics_mmm/sf_mynn.F90:1305-1336 fairall_etal_2014
static inline void m_fairall_etal_2014(thread float &zt, thread float &zq, float ren, float ustar, float visc, float rstoch, bool spp_pbl);
// MPAS physics_mmm/sf_mynn.F90:1339-1400 yang_2008
static inline void m_yang_2008(float z_0, thread float &zt, thread float &zq, float ustar, float tstar, float qst, float ren, float visc, float landsea);
// MPAS physics_mmm/sf_mynn.F90:1403-1457 andreas_2002
static inline void m_andreas_2002(float z_0, float bvisc, float ustar, thread float &zt, thread float &zq);
// MPAS physics_mmm/sf_mynn.F90:1460-1499 psi_hogstrom_1996
static inline void m_psi_hogstrom_1996(thread float &psi_m, thread float &psi_h, float zl, float zt, float z_0, float za);
// MPAS physics_mmm/sf_mynn.F90:1502-1544 psi_dyerhicks
static inline void m_psi_dyerhicks(thread float &psi_m, thread float &psi_h, float zl, float zt, float z_0, float za);
// MPAS physics_mmm/sf_mynn.F90:1547-1581 psi_beljaars_holtslag_1991
static inline void m_psi_beljaars_holtslag_1991(thread float &psi_m, thread float &psi_h, float zl);
// MPAS physics_mmm/sf_mynn.F90:1584-1618 psi_zilitinkevich_esau_2007
static inline void m_psi_zilitinkevich_esau_2007(thread float &psi_m, thread float &psi_h, float zl);
// MPAS physics_mmm/sf_mynn.F90:1621-1656 psi_businger_1971
static inline void m_psi_businger_1971(thread float &psi_m, thread float &psi_h, float zl);
// MPAS physics_mmm/sf_mynn.F90:1659-1695 psi_suselj_sood_2010
static inline void m_psi_suselj_sood_2010(thread float &psi_m, thread float &psi_h, float zl);
// MPAS physics_mmm/sf_mynn.F90:1698-1718 psi_cb2005
static inline void m_psi_cb2005(thread float &psim1, thread float &psih1, float zl, float z0l);
// MPAS physics_mmm/sf_mynn.F90:1721-1788 li_etal_2010
static inline void m_li_etal_2010(thread float &zl, float rib, float zaz0, float z0zt);
// MPAS physics_mmm/sf_mynn.F90:1791-1844 zolri
// MPAS physics_mmm/sf_mynn.F90:1847-1891 zolri2
// MPAS physics_mmm/sf_mynn.F90:1894-1964 zolrib
static inline float m_zolrib(float ri, float za, float z0, float zt, float logz0, float logzt, thread float &zol1, int psi_opt, device const float *tables);
// MPAS physics_mmm/sf_mynn.F90:1967-2004 psi_init
// MPAS physics_mmm/sf_mynn.F90:2009-2016 psim_stable_full
static inline float m_psim_stable_full(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2019-2026 psih_stable_full
static inline float m_psih_stable_full(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2029-2043 psim_unstable_full
static inline float m_psim_unstable_full(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2046-2060 psih_unstable_full
static inline float m_psih_unstable_full(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2066-2076 psim_stable_full_gfs
static inline float m_psim_stable_full_gfs(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2079-2089 psih_stable_full_gfs
static inline float m_psih_stable_full_gfs(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2092-2109 psim_unstable_full_gfs
static inline float m_psim_unstable_full_gfs(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2112-2129 psih_unstable_full_gfs
static inline float m_psih_unstable_full_gfs(float zolf);
// MPAS physics_mmm/sf_mynn.F90:2135-2157 psim_stable
static inline float m_psim_stable(float zolf, int psi_opt, device const float *tables);
// MPAS physics_mmm/sf_mynn.F90:2160-2182 psih_stable
static inline float m_psih_stable(float zolf, int psi_opt, device const float *tables);
// MPAS physics_mmm/sf_mynn.F90:2185-2207 psim_unstable
static inline float m_psim_unstable(float zolf, int psi_opt, device const float *tables);
// MPAS physics_mmm/sf_mynn.F90:2210-2232 psih_unstable
static inline float m_psih_unstable(float zolf, int psi_opt, device const float *tables);
// MPAS physics_mmm/sf_mynn.F90:130-996 sf_mynn_run
static inline void m_sf_mynn_run(float u1d, float v1d, float t1d, float qv1d, float p1d, float dz8w1d, float rho1d, float u1d2, float v1d2, float dz2w1d, float cp, float g, float rovcp, float r, float xlv, float psfcpa, thread float &chs, thread float &chs2, thread float &cqs2, thread float &cpm, float pblh, thread float &rmol, thread float &znt, thread float &ust, float mavail, thread float &zol, thread float &mol, thread float &regime, thread float &psim, thread float &psih, float xland, thread float &hfx, thread float &qfx, float tsk, thread float &u10, thread float &v10, thread float &th2, thread float &t2, thread float &q2, thread float &flhc, thread float &flqc, float snowh, thread float &qgh, thread float &qsfc, thread float &lh, thread float &gz1oz0, thread float &wspd, thread float &br, int isfflx, float dx, float svp1, float svp2, float svp3, float svpt0, float ep1, float ep2, float karman, thread float &ch, float qcg, int itimestep, thread float &wstar, thread float &qstar, thread float &ustm, thread float &ck, thread float &cka, thread float &cd, thread float &cda, bool spp_pbl, float rstoch1d, int isftcflx, int iz0tlnd, int its, int ite, bool has_isftcflx, bool has_iz0tlnd, bool has_ck, bool has_cka, bool has_cd, bool has_cda, bool has_ustm, device const float *tables) {
float prt = 0x1.0000000000000p+0f;
float xka = 0x1.92a7380000000p-16f;
int n;
int i;
int k;
int l;
int yesno;
float ep3;
float pl;
float thcon;
float tvcon;
float e1;
float dthvdz;
float dthvm;
float vconv;
float zol2;
float zol10;
float zolza;
float zolz0;
float zolzt;
float dtg;
float psix;
float dtthx;
float dthdz;
float psix10;
float psit;
float psit2;
float psiq;
float psiq2;
float psiq10;
float dzdt;
float fluxc;
float vsgd;
float restar;
float visc;
float dqg;
float oldust;
float oldtst;
float za;
float za2;
float thv1d;
float th1d;
float tc1d;
float tv1d;
float qvsh;
float psih2;
float psim10;
float psih10;
float wspdi;
float z_q;
float z_t;
float zntstoch;
float govrth;
float thgb;
float thvgb;
float psfc;
float qsfcmr;
float gz2oz0;
float gz10oz0;
float gz2ozt;
float gz10ozt;
float gz1ozt;
float zratio;
ep3 = mpas_sub(0x1.0000000000000p+0f,ep2);
{ // independent MPAS column loop
psfc = cu_div(psfcpa,0x1.f400000000000p+9f);
thgb = mpas_mul(tsk,gfk_pow(cu_div(0x1.9000000000000p+6f,psfc),rovcp));
pl = cu_div(p1d,0x1.f400000000000p+9f);
thcon = gfk_pow(cu_div(0x1.9000000000000p+6f,pl),rovcp);
th1d = mpas_mul(t1d,thcon);
tc1d = mpas_sub(t1d,0x1.1126660000000p+8f);
qvsh = cu_div(qv1d,mpas_add(0x1.0000000000000p+0f,qv1d));
tvcon = mpas_add(0x1.0000000000000p+0f,mpas_mul(ep1,qvsh));
thv1d = mpas_mul(th1d,tvcon);
tv1d = mpas_mul(t1d,tvcon);
za = mpas_mul(0x1.0000000000000p-1f,dz8w1d);
za2 = mpas_add(dz8w1d,mpas_mul(0x1.0000000000000p-1f,dz2w1d));
govrth = cu_div(g,th1d);
}
{ // independent MPAS column loop
if ((tsk<0x1.1126660000000p+8f)) {
e1 = mpas_mul(svp1,gfk_exp(mpas_add(mpas_sub(mpas_mul(4648,mpas_sub(0x1.dfda840000000p-9f,cu_div(0x1.0000000000000p+0f,tsk))),mpas_mul(0x1.747ae20000000p+3f,gfk_log(cu_div(0x1.1126660000000p+8f,tsk)))),mpas_mul(0x1.7318fc0000000p-6f,mpas_sub(0x1.1126660000000p+8f,tsk)))));
} else {
e1 = mpas_mul(svp1,gfk_exp(cu_div(mpas_mul(svp2,mpas_sub(tsk,svpt0)),mpas_sub(tsk,svp3))));
}
if (((xland>0x1.8000000000000p+0f)||(qsfc<=0x0.0p+0f))) {
qsfc = cu_div(mpas_mul(ep2,e1),mpas_sub(psfc,mpas_mul(ep3,e1)));
qsfcmr = cu_div(mpas_mul(ep2,e1),mpas_sub(psfc,e1));
} else {
qsfcmr = cu_div(qsfc,mpas_sub(0x1.0000000000000p+0f,qsfc));
}
if ((tsk<0x1.1126660000000p+8f)) {
e1 = mpas_mul(svp1,gfk_exp(mpas_add(mpas_sub(mpas_mul(4648,mpas_sub(0x1.dfda840000000p-9f,cu_div(0x1.0000000000000p+0f,t1d))),mpas_mul(0x1.747ae20000000p+3f,gfk_log(cu_div(0x1.1126660000000p+8f,t1d)))),mpas_mul(0x1.7318fc0000000p-6f,mpas_sub(0x1.1126660000000p+8f,t1d)))));
} else {
e1 = mpas_mul(svp1,gfk_exp(cu_div(mpas_mul(svp2,mpas_sub(t1d,svpt0)),mpas_sub(t1d,svp3))));
}
pl = cu_div(p1d,0x1.f400000000000p+9f);
qgh = cu_div(mpas_mul(ep2,e1),mpas_sub(pl,e1));
cpm = mpas_mul(cp,mpas_add(0x1.0000000000000p+0f,mpas_mul(0x1.ae147a0000000p-1f,qv1d)));
}
{ // independent MPAS column loop
wspd = cu_sqrt(mpas_add(mpas_mul(u1d,u1d),mpas_mul(v1d,v1d)));
thvgb = mpas_mul(thgb,mpas_add(0x1.0000000000000p+0f,mpas_mul(ep1,qsfc)));
dthdz = mpas_sub(th1d,thgb);
dthvdz = mpas_sub(thv1d,thvgb);
fluxc = fmax(mpas_add(cu_div(cu_div(hfx,rho1d),cp),cu_div(mpas_mul(mpas_mul(ep1,thvgb),qfx),rho1d)),0x0.0p+0f);
if (((xland>0x1.8000000000000p+0f)||(qsfc<=0x0.0p+0f))) {
wstar = mpas_mul(0x1.4000000000000p+0f,gfk_pow(mpas_mul(mpas_mul(cu_div(g,tsk),pblh),fluxc),0x1.51eb860000000p-2f));
} else {
wstar = mpas_mul(0x1.4000000000000p+0f,gfk_pow(mpas_mul(mpas_mul(cu_div(g,tsk),fmin(mpas_mul(0x1.8000000000000p+0f,pblh),0x1.f400000000000p+11f)),fluxc),0x1.51eb860000000p-2f));
}
vsgd = mpas_mul(0x1.47ae140000000p-2f,gfk_pow(fmax(mpas_sub(cu_div(dx,0x1.3880000000000p+12f),0x1.0000000000000p+0f),0x0.0p+0f),0x1.51eb860000000p-2f));
wspd = cu_sqrt(mpas_add(mpas_add(mpas_mul(wspd,wspd),mpas_mul(wstar,wstar)),mpas_mul(vsgd,vsgd)));
wspd = fmax(wspd,0x1.99999a0000000p-4f);
br = cu_div(mpas_mul(mpas_mul(govrth,za),dthvdz),mpas_mul(wspd,wspd));
if ((itimestep==1)) {
br = fmax(br,-0x1.0000000000000p+1f);
br = fmin(br,0x1.0000000000000p+1f);
} else {
br = fmax(br,-0x1.0000000000000p+2f);
br = fmin(br,0x1.0000000000000p+2f);
}
}
{ // independent MPAS column loop
visc = mpas_mul(0x1.bcee880000000p-17f,mpas_sub(mpas_add(mpas_add(0x1.0000000000000p+0f,mpas_mul(0x1.acbc8c0000000p-8f,tc1d)),mpas_mul(mpas_mul(0x1.16890c0000000p-17f,tc1d),tc1d)),mpas_mul(mpas_mul(mpas_mul(0x1.4c9a2e0000000p-28f,tc1d),tc1d),tc1d)));
if ((mpas_sub(xland,0x1.8000000000000p+0f)>=0)) {
if (has_isftcflx) {
if ((isftcflx==0)) {
if (true) {
m_charnock_1955(znt,ust,wspd,visc,za);
} else {
m_edson_etal_2013(znt,ust,wspd,visc,za);
}
} else if (((isftcflx==1)||(isftcflx==2))) {
m_davis_etal_2008(znt,ust);
} else if ((isftcflx==3)) {
m_taylor_yelland_2001(znt,ust,wspd);
} else if ((isftcflx==4)) {
if (true) {
m_charnock_1955(znt,ust,wspd,visc,za);
} else {
m_edson_etal_2013(znt,ust,wspd,visc,za);
}
}
} else {
if (true) {
m_charnock_1955(znt,ust,wspd,visc,za);
} else {
m_edson_etal_2013(znt,ust,wspd,visc,za);
}
}
if (spp_pbl) {
zntstoch = fmax(mpas_add(znt,mpas_mul(mpas_mul(znt,0x1.0000000000000p+0f),rstoch1d)),0x1.0c6f7a0000000p-20f);
} else {
zntstoch = znt;
}
restar = fmax(cu_div(mpas_mul(ust,zntstoch),visc),0x1.99999a0000000p-4f);
if (has_isftcflx) {
if ((isftcflx==0)) {
if (true) {
m_fairall_etal_2003(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
} else {
m_fairall_etal_2014(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
}
} else if ((isftcflx==1)) {
if (true) {
m_fairall_etal_2003(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
} else {
m_fairall_etal_2014(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
}
} else if ((isftcflx==2)) {
m_garratt_1992(z_t,z_q,zntstoch,restar,xland);
} else if ((isftcflx==3)) {
if (true) {
m_fairall_etal_2003(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
} else {
m_fairall_etal_2014(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
}
}
} else {
if (true) {
m_fairall_etal_2003(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
} else {
m_fairall_etal_2014(z_t,z_q,restar,ust,visc,rstoch1d,spp_pbl);
}
}
} else {
if (spp_pbl) {
zntstoch = fmax(mpas_add(znt,mpas_mul(mpas_mul(znt,0x1.0000000000000p+0f),rstoch1d)),0x1.0c6f7a0000000p-20f);
} else {
zntstoch = znt;
}
restar = fmax(cu_div(mpas_mul(ust,zntstoch),visc),0x1.99999a0000000p-4f);
if ((snowh>=0x1.99999a0000000p-4f)) {
m_andreas_2002(zntstoch,visc,ust,z_t,z_q);
} else {
if (has_iz0tlnd) {
if ((iz0tlnd<=1)) {
m_zilitinkevich_1995(zntstoch,z_t,z_q,restar,ust,karman,xland,iz0tlnd,spp_pbl,rstoch1d,true);
} else if ((iz0tlnd==2)) {
m_yang_2008(zntstoch,z_t,z_q,ust,mol,qstar,restar,visc,xland);
} else if ((iz0tlnd==3)) {
m_garratt_1992(z_t,z_q,zntstoch,restar,xland);
}
} else {
m_zilitinkevich_1995(zntstoch,z_t,z_q,restar,ust,karman,xland,0,spp_pbl,rstoch1d,true);
}
}
}
zratio = cu_div(zntstoch,z_t);
gz1oz0 = gfk_log(cu_div(mpas_add(za,zntstoch),zntstoch));
gz1ozt = gfk_log(cu_div(mpas_add(za,zntstoch),z_t));
gz2oz0 = gfk_log(cu_div(mpas_add(0x1.0000000000000p+1f,zntstoch),zntstoch));
gz2ozt = gfk_log(cu_div(mpas_add(0x1.0000000000000p+1f,zntstoch),z_t));
gz10oz0 = gfk_log(cu_div(mpas_add(0x1.4000000000000p+3f,zntstoch),zntstoch));
gz10ozt = gfk_log(cu_div(mpas_add(0x1.4000000000000p+3f,zntstoch),z_t));
if ((br>0x0.0p+0f)) {
if ((br>0x1.99999a0000000p-3f)) {
regime = 0x1.0000000000000p+0f;
} else {
regime = 0x1.0000000000000p+1f;
}
if ((itimestep<=1)) {
m_li_etal_2010(zol,br,cu_div(za,zntstoch),zratio);
} else {
zol = cu_div(mpas_mul(mpas_mul(mpas_mul(za,karman),g),mol),mpas_mul(th1d,fmax(mpas_mul(ust,ust),0x1.a36e2e0000000p-14f)));
zol = fmax(zol,0x0.0p+0f);
zol = fmin(zol,0x1.4000000000000p+4f);
}
zol = m_zolrib(br,za,zntstoch,z_t,gz1oz0,gz1ozt,zol,0, tables);
zol = fmax(zol,0x0.0p+0f);
zol = fmin(zol,0x1.4000000000000p+4f);
zolzt = cu_div(mpas_mul(zol,z_t),za);
zolz0 = cu_div(mpas_mul(zol,zntstoch),za);
zolza = cu_div(mpas_mul(zol,mpas_add(za,zntstoch)),za);
zol10 = cu_div(mpas_mul(zol,mpas_add(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = cu_div(mpas_mul(zol,mpas_add(0x1.0000000000000p+1f,zntstoch)),za);
if ((mpas_sub(xland,0x1.8000000000000p+0f)>=0)) {
psim = mpas_sub(m_psim_stable(zolza,0, tables),m_psim_stable(zolz0,0, tables));
psih = mpas_sub(m_psih_stable(zolza,0, tables),m_psih_stable(zolzt,0, tables));
psim10 = mpas_sub(m_psim_stable(zol10,0, tables),m_psim_stable(zolz0,0, tables));
psih10 = mpas_sub(m_psih_stable(zol10,0, tables),m_psih_stable(zolz0,0, tables));
psih2 = mpas_sub(m_psih_stable(zol2,0, tables),m_psih_stable(zolz0,0, tables));
} else {
psim = mpas_sub(m_psim_stable(zolza,0, tables),m_psim_stable(zolz0,0, tables));
psih = mpas_sub(m_psih_stable(zolza,0, tables),m_psih_stable(zolzt,0, tables));
psim10 = mpas_sub(m_psim_stable(zol10,0, tables),m_psim_stable(zolz0,0, tables));
psih10 = mpas_sub(m_psih_stable(zol10,0, tables),m_psih_stable(zolz0,0, tables));
psih2 = mpas_sub(m_psih_stable(zol2,0, tables),m_psih_stable(zolz0,0, tables));
}
rmol = cu_div(zol,za);
} else if ((br==0x0.0p+0f)) {
regime = 0x1.8000000000000p+1f;
psim = 0x0.0p+0f;
psih = psim;
psim10 = 0x0.0p+0f;
psih10 = 0x0.0p+0f;
psih2 = 0x0.0p+0f;
zol = 0x0.0p+0f;
rmol = cu_div(zol,za);
} else if ((br<0x0.0p+0f)) {
regime = 0x1.0000000000000p+2f;
if ((itimestep<=1)) {
m_li_etal_2010(zol,br,cu_div(za,zntstoch),zratio);
} else {
zol = cu_div(mpas_mul(mpas_mul(mpas_mul(za,karman),g),mol),mpas_mul(th1d,fmax(mpas_mul(ust,ust),0x1.0624de0000000p-10f)));
zol = fmax(zol,-0x1.4000000000000p+4f);
zol = fmin(zol,0x0.0p+0f);
}
zol = m_zolrib(br,za,zntstoch,z_t,gz1oz0,gz1ozt,zol,0, tables);
zol = fmax(zol,-0x1.4000000000000p+4f);
zol = fmin(zol,0x0.0p+0f);
zolzt = cu_div(mpas_mul(zol,z_t),za);
zolz0 = cu_div(mpas_mul(zol,zntstoch),za);
zolza = cu_div(mpas_mul(zol,mpas_add(za,zntstoch)),za);
zol10 = cu_div(mpas_mul(zol,mpas_add(0x1.4000000000000p+3f,zntstoch)),za);
zol2 = cu_div(mpas_mul(zol,mpas_add(0x1.0000000000000p+1f,zntstoch)),za);
if ((mpas_sub(xland,0x1.8000000000000p+0f)>=0)) {
psim = mpas_sub(m_psim_unstable(zolza,0, tables),m_psim_unstable(zolz0,0, tables));
psih = mpas_sub(m_psih_unstable(zolza,0, tables),m_psih_unstable(zolzt,0, tables));
psim10 = mpas_sub(m_psim_unstable(zol10,0, tables),m_psim_unstable(zolz0,0, tables));
psih10 = mpas_sub(m_psih_unstable(zol10,0, tables),m_psih_unstable(zolz0,0, tables));
psih2 = mpas_sub(m_psih_unstable(zol2,0, tables),m_psih_unstable(zolz0,0, tables));
} else {
psim = mpas_sub(m_psim_unstable(zolza,0, tables),m_psim_unstable(zolz0,0, tables));
psih = mpas_sub(m_psih_unstable(zolza,0, tables),m_psih_unstable(zolzt,0, tables));
psim10 = mpas_sub(m_psim_unstable(zol10,0, tables),m_psim_unstable(zolz0,0, tables));
psih10 = mpas_sub(m_psih_unstable(zol10,0, tables),m_psih_unstable(zolz0,0, tables));
psih2 = mpas_sub(m_psih_unstable(zol2,0, tables),m_psih_unstable(zolz0,0, tables));
}
psih = fmin(psih,mpas_mul(0x1.cccccc0000000p-1f,gz1ozt));
psim = fmin(psim,mpas_mul(0x1.cccccc0000000p-1f,gz1oz0));
psih2 = fmin(psih2,mpas_mul(0x1.cccccc0000000p-1f,gz2ozt));
psim10 = fmin(psim10,mpas_mul(0x1.cccccc0000000p-1f,gz10oz0));
psih10 = fmin(psih10,mpas_mul(0x1.cccccc0000000p-1f,gz10ozt));
rmol = cu_div(zol,za);
}
psix = mpas_sub(gz1oz0,psim);
psix10 = mpas_sub(gz10oz0,psim10);
oldust = ust;
ust = mpas_add(mpas_mul(0x1.0000000000000p-1f,ust),cu_div(mpas_mul(mpas_mul(0x1.0000000000000p-1f,karman),wspd),psix));
wspdi = fmax(cu_sqrt(mpas_add(mpas_mul(u1d,u1d),mpas_mul(v1d,v1d))),0x1.99999a0000000p-4f);
if (has_ustm) {
ustm = mpas_add(mpas_mul(0x1.0000000000000p-1f,ustm),cu_div(mpas_mul(mpas_mul(0x1.0000000000000p-1f,karman),wspdi),psix));
}
if ((mpas_sub(xland,0x1.8000000000000p+0f)<0x0.0p+0f)) {
ust = fmax(ust,0x1.47ae140000000p-8f);
if (has_ustm) ustm = ust;
}
gz1ozt = gfk_log(cu_div(mpas_add(za,zntstoch),z_t));
gz2ozt = gfk_log(cu_div(mpas_add(0x1.0000000000000p+1f,zntstoch),z_t));
psit = fmax(mpas_sub(gz1ozt,psih),0x1.0000000000000p+0f);
psit2 = fmax(mpas_sub(gz2ozt,psih2),0x1.0000000000000p+0f);
psiq = fmax(mpas_sub(gfk_log(cu_div(mpas_add(za,zntstoch),z_q)),psih),0x1.0000000000000p+0f);
psiq2 = fmax(mpas_sub(gfk_log(cu_div(mpas_add(0x1.0000000000000p+1f,zntstoch),z_q)),psih2),0x1.0000000000000p+0f);
psiq10 = fmax(mpas_sub(gfk_log(cu_div(mpas_add(0x1.4000000000000p+3f,zntstoch),z_q)),psih10),0x1.0000000000000p+0f);
dtg = mpas_sub(thv1d,thvgb);
oldtst = mol;
mol = cu_div(cu_div(mpas_mul(karman,dtg),psit),0x1.0000000000000p+0f);
dqg = mpas_mul(mpas_sub(qvsh,qsfc),0x1.f400000000000p+9f);
qstar = cu_div(cu_div(mpas_mul(karman,dqg),psiq),0x1.0000000000000p+0f);
}
{ // independent MPAS column loop
psix = mpas_sub(gz1oz0,psim);
psix10 = mpas_sub(gz10oz0,psim10);
psit = fmax(mpas_sub(gz1ozt,psih),0x1.0000000000000p+0f);
psit2 = fmax(mpas_sub(gz2ozt,psih2),0x1.0000000000000p+0f);
psiq = fmax(mpas_sub(gfk_log(cu_div(mpas_add(za,z_q),z_q)),psih),0x1.0000000000000p+0f);
psiq2 = fmax(mpas_sub(gfk_log(cu_div(mpas_add(0x1.0000000000000p+1f,z_q),z_q)),psih2),0x1.0000000000000p+0f);
psiq10 = fmax(mpas_sub(gfk_log(cu_div(mpas_add(0x1.4000000000000p+3f,z_q),z_q)),psih10),0x1.0000000000000p+0f);
if ((isfflx<1)) {
qfx = 0x0.0p+0f;
hfx = 0x0.0p+0f;
flhc = 0x0.0p+0f;
flqc = 0x0.0p+0f;
lh = 0x0.0p+0f;
chs = 0x0.0p+0f;
ch = 0x0.0p+0f;
chs2 = 0x0.0p+0f;
cqs2 = 0x0.0p+0f;
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = 0x0.0p+0f;
cd = 0x0.0p+0f;
cka = 0x0.0p+0f;
cda = 0x0.0p+0f;
}
} else {
flqc = cu_div(mpas_mul(mpas_mul(mpas_mul(rho1d,mavail),ust),karman),psiq);
flhc = cu_div(mpas_mul(mpas_mul(mpas_mul(rho1d,cpm),ust),karman),psit);
qfx = mpas_mul(flqc,mpas_sub(qsfcmr,qv1d));
qfx = fmax(qfx,-0x1.47ae140000000p-6f);
lh = mpas_mul(xlv,qfx);
if ((mpas_sub(xland,0x1.8000000000000p+0f)>0x0.0p+0f)) {
hfx = mpas_mul(flhc,mpas_sub(thgb,th1d));
if (has_isftcflx) {
if ((isftcflx!=0)) {
hfx = mpas_add(hfx,mpas_mul(mpas_mul(mpas_mul(rho1d,ustm),ustm),wspdi));
}
}
} else if ((mpas_sub(xland,0x1.8000000000000p+0f)<0x0.0p+0f)) {
hfx = mpas_mul(flhc,mpas_sub(thgb,th1d));
hfx = fmax(hfx,-0x1.f400000000000p+7f);
}
chs = cu_div(mpas_mul(ust,karman),psit);
ch = cu_div(flhc,mpas_mul(cpm,rho1d));
cqs2 = cu_div(mpas_mul(ust,karman),psiq2);
chs2 = cu_div(mpas_mul(ust,karman),psit2);
if ((((has_ck&&has_cd)&&has_cka)&&has_cda)) {
ck = mpas_mul(cu_div(karman,psix10),cu_div(karman,psiq10));
cd = mpas_mul(cu_div(karman,psix10),cu_div(karman,psix10));
cka = mpas_mul(cu_div(karman,psix),cu_div(karman,psiq));
cda = mpas_mul(cu_div(karman,psix),cu_div(karman,psix));
}
}
if ((za<=0x1.c000000000000p+2f)) {
if (((za2>0x1.c000000000000p+2f)&&(za2<0x1.a000000000000p+3f))) {
u10 = u1d2;
v10 = v1d2;
} else {
u10 = cu_div(mpas_mul(u1d,gfk_log(cu_div(0x1.4000000000000p+3f,zntstoch))),gfk_log(cu_div(za,zntstoch)));
v10 = cu_div(mpas_mul(v1d,gfk_log(cu_div(0x1.4000000000000p+3f,zntstoch))),gfk_log(cu_div(za,zntstoch)));
}
} else if (((za>0x1.c000000000000p+2f)&&(za<0x1.a000000000000p+3f))) {
u10 = cu_div(mpas_mul(u1d,gfk_log(cu_div(0x1.4000000000000p+3f,zntstoch))),gfk_log(cu_div(za,zntstoch)));
v10 = cu_div(mpas_mul(v1d,gfk_log(cu_div(0x1.4000000000000p+3f,zntstoch))),gfk_log(cu_div(za,zntstoch)));
} else {
u10 = cu_div(mpas_mul(u1d,psix10),psix);
v10 = cu_div(mpas_mul(v1d,psix10),psix);
}
dtg = mpas_sub(th1d,thgb);
th2 = mpas_add(thgb,cu_div(mpas_mul(dtg,psit2),psit));
if ((((th1d>thgb)&&((th2<thgb)||(th2>th1d)))||((th1d<thgb)&&((th2>thgb)||(th2<th1d))))) {
th2 = mpas_add(thgb,cu_div(mpas_mul(0x1.0000000000000p+1f,mpas_sub(th1d,thgb)),za));
}
t2 = mpas_mul(th2,gfk_pow(cu_div(psfc,0x1.9000000000000p+6f),rovcp));
q2 = mpas_add(qsfcmr,cu_div(mpas_mul(mpas_sub(qv1d,qsfcmr),psiq2),psiq));
q2 = fmax(q2,fmin(qsfcmr,qv1d));
q2 = fmin(q2,mpas_mul(0x1.0ccccc0000000p+0f,qv1d));
if (false) {
yesno = 0;
if (((hfx>0x1.2c00000000000p+10f)||(hfx<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
if (((lh>0x1.2c00000000000p+10f)||(lh<-0x1.5e00000000000p+9f))) {
yesno = 1;
}
if (((ust<0x0.0p+0f)||(ust>0x1.0000000000000p+2f))) {
yesno = 1;
}
if (((wstar<0x0.0p+0f)||(wstar>0x1.8000000000000p+2f))) {
yesno = 1;
}
if (((rho1d<0x0.0p+0f)||(rho1d>0x1.99999a0000000p+0f))) {
yesno = 1;
}
if (((mpas_mul(qsfc,0x1.f400000000000p+9f)<0x0.0p+0f)||(mpas_mul(qsfc,0x1.f400000000000p+9f)>0x1.4000000000000p+5f))) {
yesno = 1;
}
if (((pblh<0x0.0p+0f)||(pblh>0x1.7700000000000p+12f))) {
yesno = 1;
}
if ((yesno==1)) {
}
}
}
}
// MPAS physics_mmm/sf_mynn.F90:999-1073 zilitinkevich_1995
static inline void m_zilitinkevich_1995(float z_0, thread float &zt, thread float &zq, float restar, float ustar, float karman, float landsea, int iz0tlnd2, bool spp_pbl, float rstoch, bool has_iz0tlnd2) {
float czil;
if ((mpas_sub(landsea,0x1.8000000000000p+0f)>0)) {
if ((restar<0x1.99999a0000000p-4f)) {
zt = mpas_mul(z_0,gfk_exp(mpas_mul(karman,0x1.0000000000000p+1f)));
zt = fmin(zt,0x1.f751040000000p-15f);
zt = fmax(zt,0x1.12e0be0000000p-29f);
zq = mpas_mul(z_0,gfk_exp(mpas_mul(karman,0x1.8000000000000p+1f)));
zq = fmin(zq,0x1.f751040000000p-15f);
zq = fmax(zq,0x1.12e0be0000000p-29f);
} else {
zt = mpas_mul(z_0,gfk_exp(mpas_mul((-karman),mpas_sub(mpas_mul(0x1.0000000000000p+2f,cu_sqrt(restar)),0x1.99999a0000000p+1f))));
zt = fmin(zt,0x1.f751040000000p-15f);
zt = fmax(zt,0x1.12e0be0000000p-29f);
zq = mpas_mul(z_0,gfk_exp(mpas_mul((-karman),mpas_sub(mpas_mul(0x1.0000000000000p+2f,cu_sqrt(restar)),0x1.0ccccc0000000p+2f))));
zq = fmin(zt,0x1.f751040000000p-15f);
zq = fmax(zt,0x1.12e0be0000000p-29f);
}
} else {
if ((iz0tlnd2==1)) {
czil = gfk_pow(0x1.4000000000000p+3f,mpas_mul(-0x1.99999a0000000p-2f,cu_div(z_0,0x1.1eb8520000000p-4f)));
} else {
czil = 0x1.5c28f60000000p-4f;
}
zt = mpas_mul(z_0,gfk_exp(mpas_mul(mpas_mul((-karman),czil),cu_sqrt(restar))));
zt = fmin(zt,mpas_mul(0x1.8000000000000p-1f,z_0));
zq = mpas_mul(z_0,gfk_exp(mpas_mul(mpas_mul((-karman),czil),cu_sqrt(restar))));
zq = fmin(zq,mpas_mul(0x1.8000000000000p-1f,z_0));
if (spp_pbl) {
zt = mpas_add(zt,mpas_mul(mpas_mul(zt,0x1.0000000000000p-1f),rstoch));
zt = fmax(zt,0x1.a36e2e0000000p-14f);
zq = zt;
}
}
}
// MPAS physics_mmm/sf_mynn.F90:1076-1109 davis_etal_2008
static inline void m_davis_etal_2008(thread float &z_0, float ustar) {
float zw;
float zn1;
float zn2;
float g = 0x1.39eb860000000p+3f;
float ozo = 0x1.0ac1fa0000000p-16f;
zw = fmin(gfk_pow(cu_div(ustar,0x1.0f5c280000000p+0f),0x1.3333340000000p-2f),0x1.0000000000000p+0f);
zn1 = mpas_add(cu_div(mpas_mul(mpas_mul(0x1.6872b00000000p-7f,ustar),ustar),0x1.39eb860000000p+3f),0x1.0ac1fa0000000p-16f);
zn2 = mpas_add(mpas_mul(0x1.4000000000000p+3f,gfk_exp(mpas_mul(-0x1.3000000000000p+3f,gfk_pow(ustar,-0x1.554c980000000p-2f)))),cu_div(0x1.baeb220000000p-20f,fmax(ustar,0x1.47ae140000000p-7f)));
z_0 = mpas_add(mpas_mul(mpas_sub(0x1.0000000000000p+0f,zw),zn1),mpas_mul(zw,zn2));
z_0 = fmax(z_0,0x1.10bafe0000000p-23f);
z_0 = fmin(z_0,0x1.758e220000000p-9f);
}
// MPAS physics_mmm/sf_mynn.F90:1112-1141 taylor_yelland_2001
static inline void m_taylor_yelland_2001(thread float &z_0, float ustar, float wsp10) {
float g = 0x1.39eb860000000p+3f;
float pi = 0x1.921fb60000000p+1f;
float hs;
float tp;
float lp;
hs = mpas_mul(0x1.9652be0000000p-6f,mpas_mul(wsp10,wsp10));
tp = mpas_mul(0x1.753f7c0000000p-1f,fmax(wsp10,0x1.99999a0000000p-4f));
lp = cu_div(mpas_mul(0x1.39eb860000000p+3f,mpas_mul(tp,tp)),0x1.921fb60000000p+2f);
z_0 = mpas_mul(mpas_mul(0x1.2c00000000000p+10f,hs),gfk_pow(cu_div(hs,lp),0x1.2000000000000p+2f));
z_0 = fmax(z_0,0x1.10bafe0000000p-23f);
z_0 = fmin(z_0,0x1.758e220000000p-9f);
}
// MPAS physics_mmm/sf_mynn.F90:1144-1172 charnock_1955
static inline void m_charnock_1955(thread float &z_0, float ustar, float wsp10, float visc, float zu) {
float g = 0x1.39eb860000000p+3f;
float czo2 = 0x1.6872b00000000p-7f;
float czc;
float wsp10m;
wsp10m = cu_div(mpas_mul(wsp10,0x1.7069e20000000p+3f),gfk_log(cu_div(zu,0x1.a36e2e0000000p-14f)));
czc = mpas_add(0x1.6872b00000000p-7f,mpas_mul(0x1.cac0840000000p-8f,fmin(fmax(cu_div(mpas_sub(wsp10m,0x1.4000000000000p+3f),0x1.0000000000000p+3f),0x0.0p+0f),0x1.0000000000000p+0f)));
z_0 = mpas_add(cu_div(mpas_mul(mpas_mul(czc,ustar),ustar),0x1.39eb860000000p+3f),cu_div(mpas_mul(0x1.c28f5c0000000p-4f,visc),fmax(ustar,0x1.99999a0000000p-5f)));
z_0 = fmax(z_0,0x1.10bafe0000000p-23f);
z_0 = fmin(z_0,0x1.758e220000000p-9f);
}
// MPAS physics_mmm/sf_mynn.F90:1175-1208 edson_etal_2013
static inline void m_edson_etal_2013(thread float &z_0, float ustar, float wsp10, float visc, float zu) {
float g = 0x1.39eb860000000p+3f;
float m = 0x1.bda5120000000p-10f;
float b = -0x1.47ae140000000p-8f;
float czc;
float wsp10m;
wsp10m = cu_div(mpas_mul(wsp10,0x1.7069e20000000p+3f),gfk_log(cu_div(zu,0x1.a36e2e0000000p-14f)));
wsp10m = fmin(0x1.3000000000000p+4f,wsp10m);
czc = mpas_add(mpas_mul(0x1.bda5120000000p-10f,wsp10m),-0x1.47ae140000000p-8f);
czc = fmax(czc,0x0.0p+0f);
z_0 = mpas_add(cu_div(mpas_mul(mpas_mul(czc,ustar),ustar),0x1.39eb860000000p+3f),cu_div(mpas_mul(0x1.c28f5c0000000p-4f,visc),fmax(ustar,0x1.1eb8520000000p-4f)));
z_0 = fmax(z_0,0x1.10bafe0000000p-23f);
z_0 = fmin(z_0,0x1.758e220000000p-9f);
}
// MPAS physics_mmm/sf_mynn.F90:1211-1250 garratt_1992
static inline void m_garratt_1992(thread float &zt, thread float &zq, float z_0, float ren, float landsea) {
float rq;
float e = 0x1.5bf0a80000000p+1f;
if ((mpas_sub(landsea,0x1.8000000000000p+0f)>0)) {
zt = mpas_mul(z_0,gfk_exp(mpas_sub(0x1.0000000000000p+1f,mpas_mul(0x1.3d70a40000000p+1f,gfk_pow(ren,0x1.0000000000000p-2f)))));
zq = mpas_mul(z_0,gfk_exp(mpas_sub(0x1.0000000000000p+1f,mpas_mul(0x1.23d70a0000000p+1f,gfk_pow(ren,0x1.0000000000000p-2f)))));
zq = fmin(zq,0x1.cd5f9a0000000p-15f);
zq = fmax(zq,0x1.12e0be0000000p-29f);
zt = fmin(zt,0x1.cd5f9a0000000p-15f);
zt = fmax(zt,0x1.12e0be0000000p-29f);
} else {
zq = cu_div(z_0,0x1.d8e64a0000000p+2f);
zt = zq;
}
}
// MPAS physics_mmm/sf_mynn.F90:1253-1302 fairall_etal_2003
static inline void m_fairall_etal_2003(thread float &zt, thread float &zq, float ren, float ustar, float visc, float rstoch, bool spp_pbl) {
if ((ren<=0x1.0000000000000p+1f)) {
zt = mpas_mul(0x1.cd5f9a0000000p-15f,gfk_pow(ren,-0x1.3333340000000p-1f));
zq = zt;
} else {
zt = mpas_mul(0x1.cd5f9a0000000p-15f,gfk_pow(ren,-0x1.3333340000000p-1f));
zq = zt;
}
if (spp_pbl) {
zt = mpas_add(zt,mpas_mul(mpas_mul(zt,0x1.0000000000000p-1f),rstoch));
zq = zt;
}
zt = fmin(zt,0x1.a36e2e0000000p-14f);
zt = fmax(zt,0x1.12e0be0000000p-29f);
zq = fmin(zt,0x1.a36e2e0000000p-14f);
zq = fmax(zt,0x1.12e0be0000000p-29f);
}
// MPAS physics_mmm/sf_mynn.F90:1305-1336 fairall_etal_2014
static inline void m_fairall_etal_2014(thread float &zt, thread float &zq, float ren, float ustar, float visc, float rstoch, bool spp_pbl) {
zt = fmin(0x1.4f8b580000000p-13f,cu_div(0x1.e68a0e0000000p-15f,gfk_pow(ren,0x1.70a3d80000000p-1f)));
zq = zt;
if (spp_pbl) {
zt = fmax(mpas_add(zt,mpas_mul(mpas_mul(zt,0x1.0000000000000p-1f),rstoch)),0x1.12e0be0000000p-29f);
zq = fmax(mpas_add(zt,mpas_mul(mpas_mul(zt,0x1.0000000000000p-1f),rstoch)),0x1.12e0be0000000p-29f);
} else {
zt = fmax(zt,0x1.12e0be0000000p-29f);
zq = fmax(zt,0x1.12e0be0000000p-29f);
}
}
// MPAS physics_mmm/sf_mynn.F90:1339-1400 yang_2008
static inline void m_yang_2008(float z_0, thread float &zt, thread float &zq, float ustar, float tstar, float qst, float ren, float visc, float landsea) {
float ht;
float tstar2;
float qstar2;
float z_02;
float renc2;
float renc = 0x1.2c00000000000p+8f;
float beta = 0x1.8000000000000p+0f;
float m = 0x1.5400000000000p+7f;
float b = 0x1.5980000000000p+9f;
z_02 = fmin(z_0,0x1.0000000000000p-1f);
z_02 = fmax(z_02,0x1.47ae140000000p-5f);
renc2 = mpas_add(0x1.5980000000000p+9f,mpas_mul(0x1.5400000000000p+7f,gfk_log(z_02)));
ht = cu_div(mpas_mul(renc2,visc),fmax(ustar,0x1.47ae140000000p-7f));
tstar2 = fmin(tstar,0x0.0p+0f);
qstar2 = fmin(qst,0x0.0p+0f);
zt = mpas_mul(ht,gfk_exp(mpas_mul(mpas_mul(-0x1.8000000000000p+0f,gfk_pow(ustar,0x1.0000000000000p-1f)),fabs(tstar2))));
zq = mpas_mul(ht,gfk_exp(mpas_mul(mpas_mul(-0x1.8000000000000p+0f,gfk_pow(ustar,0x1.0000000000000p-1f)),fabs(qstar2))));
zt = fmin(zt,cu_div(z_0,0x1.0000000000000p+1f));
zq = fmin(zq,cu_div(z_0,0x1.0000000000000p+1f));
}
// MPAS physics_mmm/sf_mynn.F90:1403-1457 andreas_2002
static inline void m_andreas_2002(float z_0, float bvisc, float ustar, thread float &zt, thread float &zq) {
float ren2;
float zntsno;
float bt0_s = 0x1.4000000000000p+0f;
float bt0_t = 0x1.3126ea0000000p-3f;
float bt0_r = 0x1.449ba60000000p-2f;
float bt1_s = 0x0.0p+0f;
float bt1_t = -0x1.19999a0000000p-1f;
float bt1_r = -0x1.2147ae0000000p-1f;
float bt2_s = 0x0.0p+0f;
float bt2_t = 0x0.0p+0f;
float bt2_r = -0x1.76c8b40000000p-3f;
float bq0_s = 0x1.9c28f60000000p+0f;
float bq0_t = 0x1.676c8c0000000p-2f;
float bq0_r = 0x1.9581060000000p-2f;
float bq1_s = 0x0.0p+0f;
float bq1_t = -0x1.4189380000000p-1f;
float bq1_r = -0x1.0624de0000000p-1f;
float bq2_s = 0x0.0p+0f;
float bq2_t = 0x0.0p+0f;
float bq2_r = -0x1.70a3d80000000p-3f;
zntsno = mpas_add(cu_div(mpas_mul(0x1.147ae20000000p-3f,bvisc),ustar),mpas_mul(cu_div(mpas_mul(0x1.1eb8520000000p-5f,mpas_mul(ustar,ustar)),0x1.39999a0000000p+3f),mpas_add(mpas_mul(0x1.4000000000000p+2f,gfk_exp(mpas_mul(-0x1.0000000000000p+0f,mpas_mul(cu_div(mpas_sub(ustar,0x1.70a3d80000000p-3f),0x1.99999a0000000p-4f),cu_div(mpas_sub(ustar,0x1.70a3d80000000p-3f),0x1.99999a0000000p-4f))))),0x1.0000000000000p+0f)));
ren2 = cu_div(mpas_mul(ustar,zntsno),bvisc);
if ((ren2>0x1.f400000000000p+9f)) ren2 = 0x1.f400000000000p+9f;
if ((ren2<=0x1.147ae20000000p-3f)) {
zt = mpas_mul(zntsno,gfk_exp(mpas_add(mpas_add(0x1.4000000000000p+0f,mpas_mul(0x0.0p+0f,gfk_log(ren2))),mpas_mul(0x0.0p+0f,mpas_mul(gfk_log(ren2),gfk_log(ren2))))));
zq = mpas_mul(zntsno,gfk_exp(mpas_add(mpas_add(0x1.9c28f60000000p+0f,mpas_mul(0x0.0p+0f,gfk_log(ren2))),mpas_mul(0x0.0p+0f,mpas_mul(gfk_log(ren2),gfk_log(ren2))))));
} else if (((ren2>0x1.147ae20000000p-3f)&&(ren2<0x1.4000000000000p+1f))) {
zt = mpas_mul(zntsno,gfk_exp(mpas_add(mpas_add(0x1.3126ea0000000p-3f,mpas_mul(-0x1.19999a0000000p-1f,gfk_log(ren2))),mpas_mul(0x0.0p+0f,mpas_mul(gfk_log(ren2),gfk_log(ren2))))));
zq = mpas_mul(zntsno,gfk_exp(mpas_add(mpas_add(0x1.676c8c0000000p-2f,mpas_mul(-0x1.4189380000000p-1f,gfk_log(ren2))),mpas_mul(0x0.0p+0f,mpas_mul(gfk_log(ren2),gfk_log(ren2))))));
} else {
zt = mpas_mul(zntsno,gfk_exp(mpas_add(mpas_add(0x1.449ba60000000p-2f,mpas_mul(-0x1.2147ae0000000p-1f,gfk_log(ren2))),mpas_mul(-0x1.76c8b40000000p-3f,mpas_mul(gfk_log(ren2),gfk_log(ren2))))));
zq = mpas_mul(zntsno,gfk_exp(mpas_add(mpas_add(0x1.9581060000000p-2f,mpas_mul(-0x1.0624de0000000p-1f,gfk_log(ren2))),mpas_mul(-0x1.70a3d80000000p-3f,mpas_mul(gfk_log(ren2),gfk_log(ren2))))));
}
}
// MPAS physics_mmm/sf_mynn.F90:1460-1499 psi_hogstrom_1996
static inline void m_psi_hogstrom_1996(thread float &psi_m, thread float &psi_h, float zl, float zt, float z_0, float za) {
float x;
float x0;
float y;
float y0;
float zml;
float zhl;
zml = cu_div(mpas_mul(z_0,zl),za);
zhl = cu_div(mpas_mul(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = mpas_mul(-0x1.5333340000000p+2f,mpas_sub(zl,zml));
psi_h = mpas_mul(-0x1.0000000000000p+3f,mpas_sub(zl,zhl));
} else {
x = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.3000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.3000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.7333340000000p+3f,zl)),0x1.0000000000000p-1f);
y0 = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.7333340000000p+3f,zhl)),0x1.0000000000000p-1f);
psi_m = mpas_add(mpas_sub(mpas_add(mpas_mul(0x1.0000000000000p+1f,gfk_log(cu_div(mpas_add(0x1.0000000000000p+0f,x),mpas_add(0x1.0000000000000p+0f,x0)))),gfk_log(cu_div(mpas_add(0x1.0000000000000p+0f,mpas_mul(x,x)),mpas_add(0x1.0000000000000p+0f,mpas_mul(x0,x0))))),mpas_mul(0x1.0000000000000p+1f,native_mpas_atan(x))),mpas_mul(0x1.0000000000000p+1f,native_mpas_atan(x0)));
psi_h = mpas_mul(0x1.0000000000000p+1f,gfk_log(cu_div(mpas_add(0x1.0000000000000p+0f,y),mpas_add(0x1.0000000000000p+0f,y0))));
}
}
// MPAS physics_mmm/sf_mynn.F90:1502-1544 psi_dyerhicks
static inline void m_psi_dyerhicks(thread float &psi_m, thread float &psi_h, float zl, float zt, float z_0, float za) {
float x;
float x0;
float y;
float y0;
float zml;
float zhl;
zml = cu_div(mpas_mul(z_0,zl),za);
zhl = cu_div(mpas_mul(zt,zl),za);
if ((zl>0x0.0p+0f)) {
psi_m = mpas_mul(-0x1.4000000000000p+2f,mpas_sub(zl,zml));
psi_h = mpas_mul(-0x1.4000000000000p+2f,mpas_sub(zl,zhl));
} else {
x = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-2f);
x0 = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.0000000000000p+4f,zml)),0x1.0000000000000p-2f);
y = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.0000000000000p+4f,zl)),0x1.0000000000000p-1f);
y0 = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.0000000000000p+4f,zhl)),0x1.0000000000000p-1f);
psi_m = mpas_add(mpas_sub(mpas_add(mpas_mul(0x1.0000000000000p+1f,gfk_log(cu_div(mpas_add(0x1.0000000000000p+0f,x),mpas_add(0x1.0000000000000p+0f,x0)))),gfk_log(cu_div(mpas_add(0x1.0000000000000p+0f,mpas_mul(x,x)),mpas_add(0x1.0000000000000p+0f,mpas_mul(x0,x0))))),mpas_mul(0x1.0000000000000p+1f,native_mpas_atan(x))),mpas_mul(0x1.0000000000000p+1f,native_mpas_atan(x0)));
psi_h = mpas_mul(0x1.0000000000000p+1f,gfk_log(cu_div(mpas_add(0x1.0000000000000p+0f,y),mpas_add(0x1.0000000000000p+0f,y0))));
}
}
// MPAS physics_mmm/sf_mynn.F90:1547-1581 psi_beljaars_holtslag_1991
static inline void m_psi_beljaars_holtslag_1991(thread float &psi_m, thread float &psi_h, float zl) {
float a = 0x1.0000000000000p+0f;
float b = 0x1.54fdf40000000p-1f;
float c = 0x1.4000000000000p+2f;
float d = 0x1.6666660000000p-2f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = (-mpas_add(mpas_add(mpas_mul(0x1.0000000000000p+0f,zl),mpas_mul(mpas_mul(0x1.54fdf40000000p-1f,mpas_sub(zl,0x1.c924920000000p+3f)),gfk_exp(mpas_mul(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f));
psi_h = (-mpas_sub(mpas_add(mpas_add(gfk_pow(mpas_add(0x1.0000000000000p+0f,mpas_mul(0x1.54fdf40000000p-1f,zl)),0x1.8000000000000p+0f),mpas_mul(mpas_mul(0x1.54fdf40000000p-1f,mpas_sub(zl,0x1.c924920000000p+3f)),gfk_exp(mpas_mul(-0x1.6666660000000p-2f,zl)))),0x1.3075080000000p+3f),0x1.0000000000000p+0f));
}
}
// MPAS physics_mmm/sf_mynn.F90:1584-1618 psi_zilitinkevich_esau_2007
static inline void m_psi_zilitinkevich_esau_2007(thread float &psi_m, thread float &psi_h, float zl) {
float cm = 0x1.8000000000000p+1f;
float ct = 0x1.4000000000000p+1f;
if ((zl<0x0.0p+0f)) {
psi_m = 0x0.0p+0f;
psi_h = 0x0.0p+0f;
} else {
psi_m = mpas_mul(-0x1.8000000000000p+1f,gfk_pow(zl,0x1.aaaaaa0000000p-1f));
psi_h = mpas_mul(-0x1.4000000000000p+1f,gfk_pow(zl,0x1.99999a0000000p-1f));
}
}
// MPAS physics_mmm/sf_mynn.F90:1621-1656 psi_businger_1971
static inline void m_psi_businger_1971(thread float &psi_m, thread float &psi_h, float zl) {
float x;
float y;
float pi180 = 0x1.1df46a0000000p-6f;
if ((zl<0x0.0p+0f)) {
x = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.e000000000000p+3f,zl)),0x1.0000000000000p-2f);
y = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.2000000000000p+3f,zl)),0x1.0000000000000p-1f);
psi_m = mpas_add(mpas_sub(mpas_add(gfk_log(mpas_mul(cu_div(mpas_add(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f),cu_div(mpas_add(0x1.0000000000000p+0f,x),0x1.0000000000000p+1f))),gfk_log(cu_div(mpas_add(0x1.0000000000000p+0f,mpas_mul(x,x)),0x1.0000000000000p+1f))),mpas_mul(0x1.0000000000000p+1f,native_mpas_atan(x))),0x1.921fb60000000p+0f);
psi_h = mpas_mul(0x1.0000000000000p+1f,gfk_log(cu_div(mpas_add(0x1.0000000000000p+0f,y),0x1.0000000000000p+1f)));
} else {
psi_m = mpas_mul(-0x1.2ccccc0000000p+2f,zl);
psi_h = mpas_mul(-0x1.967c880000000p+2f,zl);
}
}
// MPAS physics_mmm/sf_mynn.F90:1659-1695 psi_suselj_sood_2010
static inline void m_psi_suselj_sood_2010(thread float &psi_m, thread float &psi_h, float zl) {
float rfc = 0x1.851eb80000000p-3f;
float ric = 0x1.76c8b40000000p-3f;
float phit = 0x1.99999a0000000p-1f;
if ((zl>0x0.0p+0f)) {
psi_m = (-mpas_add(cu_div(zl,0x1.851eb80000000p-3f),mpas_mul(0x1.1f4f0e0000000p+0f,gfk_exp(mpas_sub(0x1.0000000000000p+0f,cu_div(0x1.aaa64c0000000p+0f,zl))))));
psi_h = (-mpas_add(cu_div(mpas_mul(zl,0x1.76c8b40000000p-3f),0x1.71a9fc0000000p-3f),mpas_mul(0x1.c5c2900000000p+2f,gfk_pow(zl,0x1.1bedfa0000000p+0f))));
} else {
psi_m = mpas_mul(0x1.fb15b60000000p-1f,gfk_log(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.c872b00000000p+3f,zl))));
psi_h = mpas_mul(0x1.02a3060000000p+0f,gfk_log(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.04e7d60000000p+4f,zl))));
}
}
// MPAS physics_mmm/sf_mynn.F90:1698-1718 psi_cb2005
static inline void m_psi_cb2005(thread float &psim1, thread float &psih1, float zl, float z0l) {
psim1 = mpas_sub(mpas_mul(-0x1.8666660000000p+2f,gfk_log(mpas_add(zl,gfk_pow(mpas_add(0x1.0000000000000p+0f,gfk_pow(zl,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))),mpas_mul(0x1.8666660000000p+2f,gfk_log(mpas_add(z0l,gfk_pow(mpas_add(0x1.0000000000000p+0f,gfk_pow(z0l,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f)))));
psih1 = mpas_sub(mpas_mul(-0x1.6000000000000p+2f,gfk_log(mpas_add(zl,gfk_pow(mpas_add(0x1.0000000000000p+0f,gfk_pow(zl,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))),mpas_mul(0x1.6000000000000p+2f,gfk_log(mpas_add(z0l,gfk_pow(mpas_add(0x1.0000000000000p+0f,gfk_pow(z0l,0x1.19999a0000000p+0f)),0x1.d1745e0000000p-1f)))));
}
// MPAS physics_mmm/sf_mynn.F90:1721-1788 li_etal_2010
static inline void m_li_etal_2010(thread float &zl, float rib, float zaz0, float z0zt) {
float alfa;
float beta;
float zaz02;
float z0zt2;
float au11 = 0x1.70a3d80000000p-5f;
float bu11 = 0x1.89374c0000000p-9f;
float bu12 = 0x1.82a9940000000p-8f;
float bu21 = -0x1.5326180000000p-4f;
float bu22 = 0x1.c4dd300000000p-1f;
float bu31 = 0x1.6425ae0000000p-3f;
float bu32 = -0x1.d7b4a20000000p-1f;
float bu33 = -0x1.b0f27c0000000p-4f;
float aw11 = 0x1.25c91e0000000p-1f;
float aw12 = -0x1.c275260000000p-2f;
float aw21 = -0x1.39a9fc0000000p+2f;
float aw22 = 0x1.a400000000000p+5f;
float bw11 = -0x1.b98c7e0000000p-5f;
float bw12 = 0x1.8a3d700000000p+0f;
float bw21 = -0x1.56872c0000000p-1f;
float bw22 = -0x1.a418940000000p+1f;
float as11 = 0x1.817c1c0000000p-1f;
float as21 = 0x1.de147a0000000p+3f;
float bs11 = 0x1.4154ca0000000p-3f;
float bs21 = -0x1.3c84b60000000p-2f;
float bs22 = -0x1.4d91680000000p+0f;
zaz02 = zaz0;
if ((zaz0<0x1.9000000000000p+6f)) zaz02 = 0x1.9000000000000p+6f;
if ((zaz0>0x1.86a0000000000p+16f)) zaz02 = 0x1.86a0000000000p+16f;
z0zt2 = z0zt;
if ((z0zt<0x1.0000000000000p-1f)) z0zt2 = 0x1.0000000000000p-1f;
if ((z0zt>0x1.9000000000000p+6f)) z0zt2 = 0x1.9000000000000p+6f;
alfa = gfk_log(zaz02);
beta = gfk_log(z0zt2);
if ((rib<=0x0.0p+0f)) {
zl = mpas_add(mpas_mul(mpas_mul(0x1.70a3d80000000p-5f,alfa),mpas_mul(rib,rib)),mpas_mul(mpas_add(mpas_add(mpas_mul(mpas_add(mpas_mul(0x1.89374c0000000p-9f,beta),0x1.82a9940000000p-8f),mpas_mul(alfa,alfa)),mpas_mul(mpas_add(mpas_mul(-0x1.5326180000000p-4f,beta),0x1.c4dd300000000p-1f),alfa)),mpas_add(mpas_add(mpas_mul(0x1.6425ae0000000p-3f,mpas_mul(beta,beta)),mpas_mul(-0x1.d7b4a20000000p-1f,beta)),-0x1.b0f27c0000000p-4f)),rib));
zl = fmax(zl,-0x1.e000000000000p+3f);
zl = fmin(zl,0x0.0p+0f);
} else if (((rib>0x0.0p+0f)&&(rib<=0x1.99999a0000000p-3f))) {
zl = mpas_add(mpas_mul(mpas_add(mpas_mul(mpas_add(mpas_mul(0x1.25c91e0000000p-1f,beta),-0x1.c275260000000p-2f),alfa),mpas_add(mpas_mul(-0x1.39a9fc0000000p+2f,beta),0x1.a400000000000p+5f)),mpas_mul(rib,rib)),mpas_mul(mpas_add(mpas_mul(mpas_add(mpas_mul(-0x1.b98c7e0000000p-5f,beta),0x1.8a3d700000000p+0f),alfa),mpas_add(mpas_mul(-0x1.56872c0000000p-1f,beta),-0x1.a418940000000p+1f)),rib));
zl = fmin(zl,0x1.0000000000000p+2f);
zl = fmax(zl,0x0.0p+0f);
} else {
zl = mpas_add(mpas_add(mpas_add(mpas_mul(mpas_add(mpas_mul(0x1.817c1c0000000p-1f,alfa),0x1.de147a0000000p+3f),rib),mpas_mul(0x1.4154ca0000000p-3f,alfa)),mpas_mul(-0x1.3c84b60000000p-2f,beta)),-0x1.4d91680000000p+0f);
zl = fmin(zl,0x1.4000000000000p+4f);
zl = fmax(zl,0x1.0000000000000p+0f);
}
}
// MPAS physics_mmm/sf_mynn.F90:1791-1844 zolri

// MPAS physics_mmm/sf_mynn.F90:1847-1891 zolri2

// MPAS physics_mmm/sf_mynn.F90:1894-1964 zolrib
static inline float m_zolrib(float ri, float za, float z0, float zt, float logz0, float logzt, thread float &zol1, int psi_opt, device const float *tables) {
float result;
int n;
int nmax = 20;
float zol20;
float zol3;
float zolt;
float zolold;
float psit2;
float psix2;
if ((mpas_mul(zol1,ri)<0x0.0p+0f)) {
zol1 = 0x0.0p+0f;
}
if ((ri<0x0.0p+0f)) {
zolold = -0x1.869f000000000p+16f;
result = -0x1.046a000000000p+16f;
} else {
zolold = 0x1.869f000000000p+16f;
result = 0x1.046a000000000p+16f;
}
n = 1;
while (((fabs(mpas_sub(zolold,result))>0x1.47ae140000000p-7f)&&(n<20))) {
if ((n==1)) {
zolold = zol1;
} else {
zolold = result;
}
zol20 = cu_div(mpas_mul(zolold,z0),za);
zol3 = mpas_add(zolold,zol20);
zolt = cu_div(mpas_mul(zolold,zt),za);
if ((ri<0)) {
psit2 = fmax(mpas_sub(logzt,mpas_sub(m_psih_unstable(zol3,psi_opt, tables),m_psih_unstable(zolt,psi_opt, tables))),0x1.0000000000000p+0f);
psix2 = fmax(mpas_sub(logz0,mpas_sub(m_psim_unstable(zol3,psi_opt, tables),m_psim_unstable(zol20,psi_opt, tables))),0x1.0000000000000p+0f);
} else {
psit2 = fmax(mpas_sub(logzt,mpas_sub(m_psih_stable(zol3,psi_opt, tables),m_psih_stable(zolt,psi_opt, tables))),0x1.0000000000000p+0f);
psix2 = fmax(mpas_sub(logz0,mpas_sub(m_psim_stable(zol3,psi_opt, tables),m_psim_stable(zol20,psi_opt, tables))),0x1.0000000000000p+0f);
}
result = cu_div(mpas_mul(ri,mpas_mul(psix2,psix2)),psit2);
n = (n+1);
}
if (((n==20)&&(fabs(mpas_sub(zolold,result))>0x1.47ae140000000p-7f))) {
m_li_etal_2010(result,ri,cu_div(za,z0),cu_div(z0,zt));
} else {
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:1967-2004 psi_init

// MPAS physics_mmm/sf_mynn.F90:2009-2016 psim_stable_full
static inline float m_psim_stable_full(float zolf) {
float result;
result = mpas_mul(-0x1.8666660000000p+2f,gfk_log(mpas_add(zolf,gfk_pow(mpas_add(1,gfk_pow(zolf,0x1.4000000000000p+1f)),0x1.99999a0000000p-2f))));
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2019-2026 psih_stable_full
static inline float m_psih_stable_full(float zolf) {
float result;
result = mpas_mul(-0x1.5333340000000p+2f,gfk_log(mpas_add(zolf,gfk_pow(mpas_add(1,gfk_pow(zolf,0x1.19999a0000000p+0f)),0x1.d1745c0000000p-1f))));
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2029-2043 psim_unstable_full
static inline float m_psim_unstable_full(float zolf) {
float result;
float x;
float ym;
float psimc;
float psimk;
x = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-2f);
psimk = mpas_add(mpas_sub(mpas_add(mpas_mul(2,gfk_log(mpas_mul(0x1.0000000000000p-1f,mpas_add(1,x)))),gfk_log(mpas_mul(0x1.0000000000000p-1f,mpas_add(1,mpas_mul(x,x))))),mpas_mul(0x1.0000000000000p+1f,native_mpas_atan(x))),0x1.921fb60000000p+0f);
ym = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.4000000000000p+3f,zolf)),0x1.51eb860000000p-2f);
psimc = mpas_add(mpas_sub(mpas_mul(0x1.8000000000000p+0f,gfk_log(cu_div(mpas_add(mpas_add(mpas_mul(ym,ym),ym),0x1.0000000000000p+0f),0x1.8000000000000p+1f))),mpas_mul(0x1.bb67ae0000000p+0f,native_mpas_atan(cu_div(mpas_add(mpas_mul(0x1.0000000000000p+1f,ym),1),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = cu_div(mpas_add(psimk,mpas_mul(mpas_mul(zolf,zolf),psimc)),mpas_add(1,mpas_mul(zolf,zolf)));
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2046-2060 psih_unstable_full
static inline float m_psih_unstable_full(float zolf) {
float result;
float y;
float yh;
float psihc;
float psihk;
y = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.0000000000000p+4f,zolf)),0x1.0000000000000p-1f);
psihk = mpas_mul(0x1.0000000000000p+1f,gfk_log(cu_div(mpas_add(1,y),0x1.0000000000000p+1f)));
yh = gfk_pow(mpas_sub(0x1.0000000000000p+0f,mpas_mul(0x1.1000000000000p+5f,zolf)),0x1.51eb860000000p-2f);
psihc = mpas_add(mpas_sub(mpas_mul(0x1.8000000000000p+0f,gfk_log(cu_div(mpas_add(mpas_add(mpas_mul(yh,yh),yh),0x1.0000000000000p+0f),0x1.8000000000000p+1f))),mpas_mul(0x1.bb67ae0000000p+0f,native_mpas_atan(cu_div(mpas_add(mpas_mul(0x1.0000000000000p+1f,yh),1),0x1.bb67ae0000000p+0f)))),0x1.d0552a0000000p+0f);
result = cu_div(mpas_add(psihk,mpas_mul(mpas_mul(zolf,zolf),psihc)),mpas_add(1,mpas_mul(zolf,zolf)));
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2066-2076 psim_stable_full_gfs
static inline float m_psim_stable_full_gfs(float zolf) {
float result;
float aa;
float alpha4 = 0x1.4000000000000p+4f;
aa = cu_sqrt(mpas_add(0x1.0000000000000p+0f,mpas_mul(0x1.4000000000000p+4f,zolf)));
result = mpas_add(mpas_mul(-0x1.0000000000000p+0f,aa),gfk_log(mpas_add(aa,0x1.0000000000000p+0f)));
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2079-2089 psih_stable_full_gfs
static inline float m_psih_stable_full_gfs(float zolf) {
float result;
float bb;
float alpha4 = 0x1.4000000000000p+4f;
bb = cu_sqrt(mpas_add(0x1.0000000000000p+0f,mpas_mul(0x1.4000000000000p+4f,zolf)));
result = mpas_add(mpas_mul(-0x1.0000000000000p+0f,bb),gfk_log(mpas_add(bb,0x1.0000000000000p+0f)));
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2092-2109 psim_unstable_full_gfs
static inline float m_psim_unstable_full_gfs(float zolf) {
float result;
float hl1;
float tem1;
float a0 = -0x1.fccccc0000000p+1f;
float a1 = 0x1.8a3d700000000p+3f;
float b1 = -0x1.f051ec0000000p+2f;
float b2 = 0x1.829fbe0000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = cu_div(mpas_mul(mpas_add(-0x1.fccccc0000000p+1f,mpas_mul(0x1.8a3d700000000p+3f,hl1)),hl1),mpas_add(0x1.0000000000000p+0f,mpas_mul(mpas_add(-0x1.f051ec0000000p+2f,mpas_mul(0x1.829fbe0000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = cu_div(0x1.0000000000000p+0f,cu_sqrt(hl1));
result = mpas_sub(mpas_add(gfk_log(hl1),mpas_mul(0x1.0000000000000p+1f,cu_sqrt(tem1))),0x1.c154ca0000000p-1f);
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2112-2129 psih_unstable_full_gfs
static inline float m_psih_unstable_full_gfs(float zolf) {
float result;
float hl1;
float tem1;
float a0p = -0x1.fc39580000000p+2f;
float a1p = 0x1.8c00000000000p+4f;
float b1p = -0x1.168f5c0000000p+3f;
float b2p = 0x1.f989380000000p+2f;
if ((zolf>=-0x1.0000000000000p-1f)) {
hl1 = zolf;
result = cu_div(mpas_mul(mpas_add(-0x1.fc39580000000p+2f,mpas_mul(0x1.8c00000000000p+4f,hl1)),hl1),mpas_add(0x1.0000000000000p+0f,mpas_mul(mpas_add(-0x1.168f5c0000000p+3f,mpas_mul(0x1.f989380000000p+2f,hl1)),hl1)));
} else {
hl1 = (-zolf);
tem1 = cu_div(0x1.0000000000000p+0f,cu_sqrt(hl1));
result = mpas_add(mpas_add(gfk_log(hl1),mpas_mul(0x1.0000000000000p-1f,tem1)),0x1.62d0e60000000p+0f);
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2135-2157 psim_stable
static inline float m_psim_stable(float zolf, int psi_opt, device const float *tables) {
float result;
int nzol;
float rzol;
nzol = (int)(mpas_mul(zolf,0x1.9000000000000p+6f));
rzol = mpas_sub(mpas_mul(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<=1000)) {
result = mpas_add(tables[0 + (nzol)],mpas_mul(rzol,mpas_sub(tables[0 + ((nzol+1))],tables[0 + (nzol)])));
} else {
if ((psi_opt==0)) {
result = m_psim_stable_full(zolf);
} else {
result = m_psim_stable_full_gfs(zolf);
}
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2160-2182 psih_stable
static inline float m_psih_stable(float zolf, int psi_opt, device const float *tables) {
float result;
int nzol;
float rzol;
nzol = (int)(mpas_mul(zolf,0x1.9000000000000p+6f));
rzol = mpas_sub(mpas_mul(zolf,0x1.9000000000000p+6f),nzol);
if (((nzol+1)<=1000)) {
result = mpas_add(tables[1001 + (nzol)],mpas_mul(rzol,mpas_sub(tables[1001 + ((nzol+1))],tables[1001 + (nzol)])));
} else {
if ((psi_opt==0)) {
result = m_psih_stable_full(zolf);
} else {
result = m_psih_stable_full_gfs(zolf);
}
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2185-2207 psim_unstable
static inline float m_psim_unstable(float zolf, int psi_opt, device const float *tables) {
float result;
int nzol;
float rzol;
nzol = (int)(mpas_mul((-zolf),0x1.9000000000000p+6f));
rzol = mpas_sub(mpas_mul((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<=1000)) {
result = mpas_add(tables[2002 + (nzol)],mpas_mul(rzol,mpas_sub(tables[2002 + ((nzol+1))],tables[2002 + (nzol)])));
} else {
if ((psi_opt==0)) {
result = m_psim_unstable_full(zolf);
} else {
result = m_psim_unstable_full_gfs(zolf);
}
}
return result;
}
// MPAS physics_mmm/sf_mynn.F90:2210-2232 psih_unstable
static inline float m_psih_unstable(float zolf, int psi_opt, device const float *tables) {
float result;
int nzol;
float rzol;
nzol = (int)(mpas_mul((-zolf),0x1.9000000000000p+6f));
rzol = mpas_sub(mpas_mul((-zolf),0x1.9000000000000p+6f),nzol);
if (((nzol+1)<=1000)) {
result = mpas_add(tables[3003 + (nzol)],mpas_mul(rzol,mpas_sub(tables[3003 + ((nzol+1))],tables[3003 + (nzol)])));
} else {
if ((psi_opt==0)) {
result = m_psih_unstable_full(zolf);
} else {
result = m_psih_unstable_full_gfs(zolf);
}
}
return result;
}

kernel void hexv842_mpas_sf_mynn_tables_f32(constant Hexv842MpasSfMynnTablesF32Resources &r [[buffer(0)]], constant Hexv842MpasSfMynnTablesF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i>=1001) return;
    float z=mpas_mul(float(i),0x1.47ae140000000p-7f);
    r.tables[i]=m_psim_stable_full(z); r.tables[1001+i]=m_psih_stable_full(z);
    z=mpas_mul(-float(i),0x1.47ae140000000p-7f);
    r.tables[2002+i]=m_psim_unstable_full(z); r.tables[3003+i]=m_psih_unstable_full(z);
}
kernel void hexv842_mpas_sf_mynn_columns_f32(constant Hexv842MpasSfMynnColumnsF32Resources &r [[buffer(0)]], constant Hexv842MpasSfMynnColumnsF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i>=uint(p.n)) return;
    float v0=r.fields[0*p.n+i];
    float v1=r.fields[1*p.n+i];
    float v2=r.fields[2*p.n+i];
    float v3=r.fields[3*p.n+i];
    float v4=r.fields[4*p.n+i];
    float v5=r.fields[5*p.n+i];
    float v6=r.fields[6*p.n+i];
    float v7=r.fields[7*p.n+i];
    float v8=r.fields[8*p.n+i];
    float v9=r.fields[9*p.n+i];
    float v10=r.fields[10*p.n+i];
    float v11=r.fields[11*p.n+i];
    float v12=r.fields[12*p.n+i];
    float v13=r.fields[13*p.n+i];
    float v14=r.fields[14*p.n+i];
    float v15=r.fields[15*p.n+i];
    float v16=r.fields[16*p.n+i];
    float v17=r.fields[17*p.n+i];
    float v18=r.fields[18*p.n+i];
    float v19=r.fields[19*p.n+i];
    float v20=r.fields[20*p.n+i];
    float v21=r.fields[21*p.n+i];
    float v22=r.fields[22*p.n+i];
    float v23=r.fields[23*p.n+i];
    float v24=r.fields[24*p.n+i];
    float v25=r.fields[25*p.n+i];
    float v26=r.fields[26*p.n+i];
    float v27=r.fields[27*p.n+i];
    float v28=r.fields[28*p.n+i];
    float v29=r.fields[29*p.n+i];
    float v30=r.fields[30*p.n+i];
    float v31=r.fields[31*p.n+i];
    float v32=r.fields[32*p.n+i];
    float v33=r.fields[33*p.n+i];
    float v34=r.fields[34*p.n+i];
    float v35=r.fields[35*p.n+i];
    float v36=r.fields[36*p.n+i];
    float v37=r.fields[37*p.n+i];
    float v38=r.fields[38*p.n+i];
    float v39=r.fields[39*p.n+i];
    float v40=r.fields[40*p.n+i];
    float v41=r.fields[41*p.n+i];
    float v42=r.fields[42*p.n+i];
    float v43=r.fields[43*p.n+i];
    float v44=r.fields[44*p.n+i];
    float v45=r.fields[45*p.n+i];
    float v46=r.fields[46*p.n+i];
    float v47=r.fields[47*p.n+i];
    float v48=r.fields[48*p.n+i];
    float v49=r.fields[49*p.n+i];
    float v50=r.fields[50*p.n+i];
    float v51=r.fields[51*p.n+i];
    float v52=r.fields[52*p.n+i];
    float v53=r.fields[53*p.n+i];
    m_sf_mynn_run(v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,r.constants[0],r.constants[1],r.constants[2],r.constants[3],r.constants[4],v10,v11,v12,v13,v14,v15,v16,v17,v18,v19,v20,v21,v22,v23,v24,v25,v26,v27,v28,v29,v30,v31,v32,v33,v34,v35,v36,v37,v38,v39,v40,v41,v42,p.isfflx,v43,r.constants[5],r.constants[6],r.constants[7],r.constants[8],r.constants[9],r.constants[10],r.constants[11],v44,v45,p.itimestep,v46,v47,v48,v49,v50,v51,v52,p.spp_pbl,v53,p.isftcflx,p.iz0tlnd,0,0, (p.present & (1u<<0))!=0, (p.present & (1u<<1))!=0, (p.present & (1u<<2))!=0, (p.present & (1u<<3))!=0, (p.present & (1u<<4))!=0, (p.present & (1u<<5))!=0, (p.present & (1u<<6))!=0,r.tables);
    r.fields[0*p.n+i]=v0;
    r.fields[1*p.n+i]=v1;
    r.fields[2*p.n+i]=v2;
    r.fields[3*p.n+i]=v3;
    r.fields[4*p.n+i]=v4;
    r.fields[5*p.n+i]=v5;
    r.fields[6*p.n+i]=v6;
    r.fields[7*p.n+i]=v7;
    r.fields[8*p.n+i]=v8;
    r.fields[9*p.n+i]=v9;
    r.fields[10*p.n+i]=v10;
    r.fields[11*p.n+i]=v11;
    r.fields[12*p.n+i]=v12;
    r.fields[13*p.n+i]=v13;
    r.fields[14*p.n+i]=v14;
    r.fields[15*p.n+i]=v15;
    r.fields[16*p.n+i]=v16;
    r.fields[17*p.n+i]=v17;
    r.fields[18*p.n+i]=v18;
    r.fields[19*p.n+i]=v19;
    r.fields[20*p.n+i]=v20;
    r.fields[21*p.n+i]=v21;
    r.fields[22*p.n+i]=v22;
    r.fields[23*p.n+i]=v23;
    r.fields[24*p.n+i]=v24;
    r.fields[25*p.n+i]=v25;
    r.fields[26*p.n+i]=v26;
    r.fields[27*p.n+i]=v27;
    r.fields[28*p.n+i]=v28;
    r.fields[29*p.n+i]=v29;
    r.fields[30*p.n+i]=v30;
    r.fields[31*p.n+i]=v31;
    r.fields[32*p.n+i]=v32;
    r.fields[33*p.n+i]=v33;
    r.fields[34*p.n+i]=v34;
    r.fields[35*p.n+i]=v35;
    r.fields[36*p.n+i]=v36;
    r.fields[37*p.n+i]=v37;
    r.fields[38*p.n+i]=v38;
    r.fields[39*p.n+i]=v39;
    r.fields[40*p.n+i]=v40;
    r.fields[41*p.n+i]=v41;
    r.fields[42*p.n+i]=v42;
    r.fields[43*p.n+i]=v43;
    r.fields[44*p.n+i]=v44;
    r.fields[45*p.n+i]=v45;
    r.fields[46*p.n+i]=v46;
    r.fields[47*p.n+i]=v47;
    r.fields[48*p.n+i]=v48;
    r.fields[49*p.n+i]=v49;
    r.fields[50*p.n+i]=v50;
    r.fields[51*p.n+i]=v51;
    r.fields[52*p.n+i]=v52;
    r.fields[53*p.n+i]=v53;
}

kernel void hexv842_mpas_sf_mynn_prepare_f32(constant Hexv842MpasSfMynnPrepareF32Resources &r [[buffer(0)]], constant Hexv842MpasSfMynnPrepareF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if(i>=uint(p.n)) return;
    r.fields[0*p.n+i]=r.profiles[0*p.n+i];
    r.fields[1*p.n+i]=r.profiles[2*p.n+i];
    r.fields[2*p.n+i]=r.profiles[4*p.n+i];
    r.fields[3*p.n+i]=r.profiles[6*p.n+i];
    r.fields[4*p.n+i]=r.profiles[10*p.n+i];
    r.fields[6*p.n+i]=r.profiles[12*p.n+i];
    r.fields[5*p.n+i]=r.profiles[14*p.n+i];
    r.fields[7*p.n+i]=r.profiles[1*p.n+i];
    r.fields[8*p.n+i]=r.profiles[3*p.n+i];
    r.fields[9*p.n+i]=r.profiles[15*p.n+i];
    r.fields[53*p.n+i]=0.0f;
    r.fields[43*p.n+i]=cu_div(p.len_disp,gfk_pow(r.density[i],0.25f));
    if(p.first) {
        float u=r.profiles[i],v=r.profiles[2*p.n+i],q=r.profiles[6*p.n+i];
        r.fields[18*p.n+i]=fmax(mpas_mul(0.04f,cu_sqrt(mpas_add(mpas_mul(u,u),mpas_mul(v,v)))),0.001f);
        r.fields[21*p.n+i]=0.0f;
        r.fields[38*p.n+i]=cu_div(q,mpas_add(1.0f,q));
        r.fields[47*p.n+i]=0.0f;
    }
}
kernel void hexv842_mpas_sf_mynn_sea_f32(constant Hexv842MpasSfMynnSeaF32Resources &r [[buffer(0)]], constant Hexv842MpasSfMynnSeaF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if(i>=uint(p.n)) return;
    for(int j=0;j<54;++j) r.sea[j*p.n+i]=r.fields[j*p.n+i];
    r.sea[38*p.n+i]=p.first ? r.fields[38*p.n+i] : 0.0f;
    if(r.ice[i]>=p.threshold && r.ice[i]<=1.0f) {
        r.sea[25*p.n+i]=2.0f; r.sea[19*p.n+i]=1.0f;
        r.sea[17*p.n+i]=0.0001f; r.sea[28*p.n+i]=fmax(r.sst[i],271.4f);
        if(p.rp3) {
            // RP3 referee seaice_component_tsk, lines 87-98.
            float blended=r.fields[28*p.n+i];
            float recovered=cu_div(mpas_sub(blended,
                mpas_mul(mpas_sub(1.0f,r.ice[i]),r.sea[28*p.n+i])),r.ice[i]);
            r.fields[28*p.n+i]=fmax(recovered,221.4f);
            if(r.ice[i]<0.2f && blended<253.15f) r.fields[28*p.n+i]=253.15f;
            if(r.ice[i]<0.1f && blended<263.15f) r.fields[28*p.n+i]=263.15f;
        }
    }
}
kernel void hexv842_mpas_sf_mynn_blend_f32(constant Hexv842MpasSfMynnBlendF32Resources &r [[buffer(0)]], constant Hexv842MpasSfMynnBlendF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if(i>=uint(p.n) || r.ice[i]<p.threshold || r.ice[i]>1.0f) return;
    float x=r.ice[i],y=mpas_sub(1.0f,x);
    r.fields[42*p.n+i]=mpas_add(mpas_mul(r.fields[42*p.n+i],x),mpas_mul(y,r.sea[42*p.n+i]));
    r.fields[34*p.n+i]=mpas_add(mpas_mul(r.fields[34*p.n+i],x),mpas_mul(y,r.sea[34*p.n+i]));
    r.fields[35*p.n+i]=mpas_add(mpas_mul(r.fields[35*p.n+i],x),mpas_mul(y,r.sea[35*p.n+i]));
    r.fields[40*p.n+i]=mpas_add(mpas_mul(r.fields[40*p.n+i],x),mpas_mul(y,r.sea[40*p.n+i]));
    r.fields[21*p.n+i]=mpas_add(mpas_mul(r.fields[21*p.n+i],x),mpas_mul(y,r.sea[21*p.n+i]));
    r.fields[24*p.n+i]=mpas_add(mpas_mul(r.fields[24*p.n+i],x),mpas_mul(y,r.sea[24*p.n+i]));
    r.fields[23*p.n+i]=mpas_add(mpas_mul(r.fields[23*p.n+i],x),mpas_mul(y,r.sea[23*p.n+i]));
    r.fields[16*p.n+i]=mpas_add(mpas_mul(r.fields[16*p.n+i],x),mpas_mul(y,r.sea[16*p.n+i]));
    r.fields[18*p.n+i]=mpas_add(mpas_mul(r.fields[18*p.n+i],x),mpas_mul(y,r.sea[18*p.n+i]));
    r.fields[41*p.n+i]=mpas_add(mpas_mul(r.fields[41*p.n+i],x),mpas_mul(y,r.sea[41*p.n+i]));
    r.fields[20*p.n+i]=mpas_add(mpas_mul(r.fields[20*p.n+i],x),mpas_mul(y,r.sea[20*p.n+i]));
    r.fields[33*p.n+i]=mpas_add(mpas_mul(r.fields[33*p.n+i],x),mpas_mul(y,r.sea[33*p.n+i]));
    r.fields[32*p.n+i]=mpas_add(mpas_mul(r.fields[32*p.n+i],x),mpas_mul(y,r.sea[32*p.n+i]));
    r.fields[31*p.n+i]=mpas_add(mpas_mul(r.fields[31*p.n+i],x),mpas_mul(y,r.sea[31*p.n+i]));
    r.fields[29*p.n+i]=mpas_add(mpas_mul(r.fields[29*p.n+i],x),mpas_mul(y,r.sea[29*p.n+i]));
    r.fields[30*p.n+i]=mpas_add(mpas_mul(r.fields[30*p.n+i],x),mpas_mul(y,r.sea[30*p.n+i]));
    r.fields[51*p.n+i]=mpas_add(mpas_mul(r.fields[51*p.n+i],x),mpas_mul(y,r.sea[51*p.n+i]));
    r.fields[52*p.n+i]=mpas_add(mpas_mul(r.fields[52*p.n+i],x),mpas_mul(y,r.sea[52*p.n+i]));
    r.fields[49*p.n+i]=mpas_add(mpas_mul(r.fields[49*p.n+i],x),mpas_mul(y,r.sea[49*p.n+i]));
    r.fields[50*p.n+i]=mpas_add(mpas_mul(r.fields[50*p.n+i],x),mpas_mul(y,r.sea[50*p.n+i]));
    r.fields[48*p.n+i]=mpas_add(mpas_mul(r.fields[48*p.n+i],x),mpas_mul(y,r.sea[48*p.n+i]));
    r.fields[44*p.n+i]=mpas_add(mpas_mul(r.fields[44*p.n+i],x),mpas_mul(y,r.sea[44*p.n+i]));
    if(x>=0.5f) r.fields[22*p.n+i]=r.regime_hold[i];
}
