#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_seaice_math.metalh"
#include "hexv842_seaice_from_f32_bindings.metalh"
#include "hexv842_seaice_column_f32_bindings.metalh"
#include "hexv842_seaice_to_f32_bindings.metalh"
#include "hexv842_seaice_state_i32_bindings.metalh"
#pragma clang fp contract(off)
#define NSOLD 9

// MPAS module_sf_noah_seaice.F:222-1292 and wrapper .F:189-511:
// GNU scalar SSE keeps NaN operand payload/sign; CUDA arithmetic canonicalizes.
// Keep rounded device arithmetic for finite results, and reproduce the
// measured GNU operand words for nonfinite results. Shared libm bodies stay intact.
static inline float si_nan_result(float result,float a,float b) {
  unsigned int r=as_type<uint>(result);
  if ((r & 0x7fffffffu)<=0x7f800000u) return result;
  unsigned int x=as_type<uint>(a),y=as_type<uint>(b);
  if ((x & 0x7fffffffu)>0x7f800000u) return as_type<float>(x|0x00400000u);
  if ((y & 0x7fffffffu)>0x7f800000u) return as_type<float>(y|0x00400000u);
  return as_type<float>(0xffc00000u);
}
// MPAS physics_wrf/module_sf_noah_seaice.F:222-1292; module_sf_noahlsm.F:2541-2600.
static inline float si_add(float a,float b) { return si_nan_result(mpas_add(a,b),a,b); }
// MPAS physics_wrf/module_sf_noah_seaice.F:222-1292; module_sf_noahlsm.F:2541-2600.
static inline float si_sub(float a,float b) { return si_nan_result(mpas_sub(a,b),a,b); }
// MPAS physics_wrf/module_sf_noah_seaice.F:222-1292; module_sf_noahlsm.F:2541-2600.
static inline float si_mul(float a,float b) { return si_nan_result(mpas_mul(a,b),a,b); }
// MPAS physics_wrf/module_sf_noah_seaice.F:222-1292; module_sf_noahlsm.F:2541-2600.
static inline float si_div(float a,float b) { return si_nan_result(mpas_div(a,b),a,b); }
// MPAS physics_wrf/module_sf_noah_seaice.F:222-1292; module_sf_noahlsm.F:2541-2600.
static inline float si_neg(float a) {
  unsigned int bits=as_type<uint>(a);
bits ^= 0x80000000u;

  return as_type<float>(bits);
}

constant float cp=1004.5f, rd=287.04f, sigma=5.67e-8f, cph2o=4.218e3f, cpice=2.106e3f, lsubf=3.335e5f, tfreez=273.15f;
// Generated transcription reviewed against pinned MPAS-A v8.4.2; one-based private arrays.
// MPAS physics_wrf/module_sf_noahlsm.F:1857-1905 (forward declaration).
static void hstep(thread float * stcout, thread float * stcin, thread float * rhsts, thread float & dt, thread int & nsoil, thread float * ai, thread float * bi, thread float * ci, thread int & status);
// MPAS physics_wrf/module_sf_noahlsm.F:2541-2600 (forward declaration).
static void rosr12(thread float * p, thread float * a, thread float * b, thread float * c, thread float * d, thread float * delta, thread int & nsoil, thread int & status);
// MPAS physics_wrf/module_sf_noah_seaice.F:36-545 (forward declaration).
static void sflx_seaice(thread int & iiloc, thread int & jjloc, thread int & seaice_albedo_opt, thread float & seaice_albedo_default, thread int & seaice_snowdepth_opt, thread float & seaice_snowdepth_max, thread float & seaice_snowdepth_min, thread float & ffrozp, thread float & dt, thread float & zlvl, thread int & nsoil, thread float & sithick, thread float & lwdn, thread float & solnet, thread float & sfcprs, thread float & prcp, thread float & sfctmp, thread float & q2, thread float & th2, thread float & q2sat, thread float & dqsdt2, thread float & snoalb, thread float & tbot, thread float & z0brd, thread float & z0, thread float & emissi, thread float & t1, thread float * stc, thread float & snowh, thread float & sneqv, thread float & albedo, thread float & ch, thread float & albedosi, thread float & snowonsi, thread float & eta, thread float & sheat, thread float & eta_kinematic, thread float & fdown, thread float & esnow, thread float & dew, thread float & etp, thread float & ssoil, thread float & flx1, thread float & flx2, thread float & flx3, thread float & snomlt, thread float & sncovr, thread float & runoff1, thread float & q1, thread float & ribb, thread int & status);
// MPAS physics_wrf/module_sf_noah_seaice.F:548-587 (forward declaration).
static void csnow(thread float & sncond, thread float & dsnow, thread int & status);
// MPAS physics_wrf/module_sf_noah_seaice.F:589-707 (forward declaration).
static void hrtice(thread float * rhsts, thread float * stc, thread float & tbot, thread int & nsoil, thread float * zsoil, thread float & yy, thread float & zz1, thread float & df1, thread float * ai, thread float * bi, thread float * ci, thread int & status);
// MPAS physics_wrf/module_sf_noah_seaice.F:710-782 (forward declaration).
static void penman(thread float & sfctmp, thread float & sfcprs, thread float & ch, thread float & th2, thread float & prcp, thread float & fdown, thread float & t24, thread float & ssoil, thread float & q2, thread float & q2sat, thread float & etp, thread float & rch, thread float & rr, thread bool & snowng, thread bool & frzgra, thread float & dqsdt2, thread float & flx2, thread float & emissi, thread float & t1, thread int & status);
// MPAS physics_wrf/module_sf_noah_seaice.F:785-812 (forward declaration).
static void shflx(thread float * stc, thread int & nsoil, thread float & dt, thread float & yy, thread float & zz1, thread float * zsoil, thread float & tbot, thread float & df1, thread int & status);
// MPAS physics_wrf/module_sf_noah_seaice.F:815-1079 (forward declaration).
static void snopac(thread float & etp, thread float & eta, thread float & prcp, thread bool & snowng, thread int & nsoil, thread float & dt, thread float & df1, thread float & q2, thread float & t1, thread float & sfctmp, thread float & t24, thread float & th2, thread float & fdown, thread float & ssoil, thread float * stc, thread float & sfcprs, thread float & rch, thread float & rr, thread float & sncovr, thread float & esd, thread float & sndens, thread float & snowh, thread float * zsoil, thread float & tbot, thread float & snomlt, thread float & dew, thread float & flx1, thread float & flx2, thread float & flx3, thread float & esnow, thread float & emissi, thread float & ribb, thread int & seaice_albedo_opt, thread int & status);
// MPAS physics_wrf/module_sf_noah_seaice.F:1082-1207 (forward declaration).
static void snowpack(thread float & esd, thread float & dtsec, thread float & snowh, thread float & sndens, thread float & tsnow, thread float & tsoil, thread int & status);
// MPAS physics_wrf/module_sf_noah_seaice.F:1210-1239 (forward declaration).
static void snowz0(thread float & sncovr, thread float & z0, thread float & z0brd, thread float & snowh, thread int & status);
// MPAS physics_wrf/module_sf_noah_seaice.F:1243-1292 (forward declaration).
static void snow_new(thread float & temp, thread float & newsn, thread float & snowh, thread float & sndens, thread int & status);

// MPAS physics_wrf/module_sf_noahlsm.F:1857-1905
static void hstep(thread float * stcout, thread float * stcin, thread float * rhsts, thread float & dt, thread int & nsoil, thread float * ai, thread float * bi, thread float * ci, thread int & status) {
  int k;
  float rhstsin[NSOLD+1];
  float ciin[NSOLD+1];
  for (k = 1; k <= nsoil; k += 1) {
  rhsts[k] = si_mul(rhsts[k], dt);
  ai[k] = si_mul(ai[k], dt);
  bi[k] = si_add(1.0f, si_mul(bi[k], dt));
  ci[k] = si_mul(ci[k], dt);
  }
  for (k = 1; k <= nsoil; k += 1) {
  rhstsin[k] = rhsts[k];
  }
  for (k = 1; k <= nsoil; k += 1) {
  ciin[k] = ci[k];
  }
  rosr12(ci, ai, bi, ciin, rhstsin, rhsts, nsoil, status);
  for (k = 1; k <= nsoil; k += 1) {
  stcout[k] = si_add(stcin[k], ci[k]);
  }
}

// MPAS physics_wrf/module_sf_noahlsm.F:2541-2600
static void rosr12(thread float * p, thread float * a, thread float * b, thread float * c, thread float * d, thread float * delta, thread int & nsoil, thread int & status) {
  int k;
  int kk;
  c[nsoil] = 0.0f;
  // MPAS module_sf_noahlsm.F:2573,2583: unary minus follows each full chain.
  p[1] = si_neg(si_div(c[1], b[1]));
  delta[1] = si_div(d[1], b[1]);
  for (k = 2; k <= nsoil; k += 1) {
  p[k] = si_neg(si_mul(si_div(1.0f, si_add(si_mul(p[(k - 1)], a[k]), b[k])), c[k]));
  delta[k] = si_mul(si_sub(d[k], si_mul(a[k], delta[(k - 1)])), si_div(1.0f, si_add(si_mul(p[(k - 1)], a[k]), b[k])));
  }
  p[nsoil] = delta[nsoil];
  for (k = 2; k <= nsoil; k += 1) {
  kk = ((nsoil - k) + 1);
  p[kk] = si_add(si_mul(p[(kk + 1)], p[kk]), delta[kk]);
  }
}

// MPAS physics_wrf/module_sf_noah_seaice.F:36-545
static void sflx_seaice(thread int & iiloc, thread int & jjloc, thread int & seaice_albedo_opt, thread float & seaice_albedo_default, thread int & seaice_snowdepth_opt, thread float & seaice_snowdepth_max, thread float & seaice_snowdepth_min, thread float & ffrozp, thread float & dt, thread float & zlvl, thread int & nsoil, thread float & sithick, thread float & lwdn, thread float & solnet, thread float & sfcprs, thread float & prcp, thread float & sfctmp, thread float & q2, thread float & th2, thread float & q2sat, thread float & dqsdt2, thread float & snoalb, thread float & tbot, thread float & z0brd, thread float & z0, thread float & emissi, thread float & t1, thread float * stc, thread float & snowh, thread float & sneqv, thread float & albedo, thread float & ch, thread float & albedosi, thread float & snowonsi, thread float & eta, thread float & sheat, thread float & eta_kinematic, thread float & fdown, thread float & esnow, thread float & dew, thread float & etp, thread float & ssoil, thread float & flx1, thread float & flx2, thread float & flx3, thread float & snomlt, thread float & sncovr, thread float & runoff1, thread float & q1, thread float & ribb, thread int & status) {
  bool frzgra;
  bool snowng;
  float zsoil[NSOLD+1];
  float df1;
  float df1a;
  float dsoil;
  float dtot;
  float frcsno;
  float frcsoi;
  float rch;
  float rr;
  float sndens;
  float sncond;
  float sn_new;
  float t24;
  float t2v;
  float th2v;
  float tsnow;
  float rho;
  int kz;
  int k;
  float alb_snow;
  float alb_ice;
  float z0n;
  float sncovrr;
  const float lvh2o = 2501000.0f;
  const float lsubs = 2830000.0f;
  const float r = 287.04f;
  runoff1 = 0.0f;
  snomlt = 0.0f;
  for (kz = 1; kz <= nsoil; kz += 1) {
  zsoil[kz] = si_neg(si_div(si_mul(sithick, (float)(kz)), (float)(nsoil)));
  }
  z0brd = 0.001f;
  snowng = false;
  frzgra = false;
  switch (seaice_albedo_opt) {
  default:
  if ((sneqv < 0.01f)) {
  sneqv = 0.01f;
  snowh = 0.05f;
  }
  break;
  case 1:
  if ((sneqv < 0.0001f)) {
  sneqv = 0.0001f;
  snowh = 0.0005f;
  }
  break;
  }
  if ((seaice_snowdepth_opt == 0)) {
  sndens = si_div(sneqv, snowh);
  snowh = fmax(seaice_snowdepth_min, fmin(snowh, seaice_snowdepth_max));
  sneqv = si_mul(snowh, sndens);
  } else if ((seaice_snowdepth_opt == 1)) {
  sndens = 0.3f;
  snowh = snowonsi;
  sneqv = si_mul(snowh, sndens);
  }
  sndens = si_div(sneqv, snowh);
  if ((sndens > 1.0f)) {
  status = 1; return; // MPAS fatal: snow depth below water equivalent
  }
  csnow(sncond, sndens, status);
  if ((prcp > 0.0f)) {
  if ((ffrozp > 0.5f)) {
  snowng = true;
  } else {
  if ((t1 <= tfreez)) frzgra = true;
  }
  }
  if ((snowng || frzgra)) {
  sn_new = si_mul(si_mul(prcp, dt), 0.001f);
  sneqv = si_add(sneqv, sn_new);
  snow_new(sfctmp, sn_new, snowh, sndens, status);
  if ((sncovr > 0.99f)) {
  if ((stc[1] < si_sub(tfreez, 5.0f))) sndens = 0.2f;
  if ((snowng && (t1 < 273.0f) && (sfctmp < 273.0f))) sndens = 0.2f;
  }
  csnow(sncond, sndens, status);
  }
  switch (seaice_albedo_opt) {
  default:
  sncovr = 1.0f;
  emissi = 0.98f;
  albedo = seaice_albedo_default;
  break;
  case 1:
  if ((t1 < 268.15f)) {
  alb_snow = 0.8f;
  } else if (((t1 >= 268.15f) && (t1 < 273.15f))) {
  alb_snow = si_sub(0.65f, si_mul(0.03f, si_sub(t1, 273.15f)));
  } else {
  alb_snow = 0.65f;
  }
  if ((sfctmp <= 273.15f)) {
  alb_ice = 0.65f;
  } else if (((sfctmp > 273.15f) && (sfctmp < 278.15f))) {
  alb_ice = si_sub(0.65f, si_mul(0.04f, si_sub(sfctmp, 273.15f)));
  } else {
  alb_ice = 0.45f;
  }
  z0n = 0.1f;
  sncovrr = si_div(snowh, si_add(snowh, z0n));
  albedo = si_add(si_mul(sncovrr, alb_snow), si_mul(si_sub(1.0f, sncovrr), alb_ice));
  break;
  case 2:
  sncovr = 1.0f;
  emissi = 0.98f;
  albedo = albedosi;
  break;
  }
  df1 = 2.2f;
  dsoil = (si_neg(si_mul(0.5f, zsoil[1])));
  dtot = si_add(snowh, dsoil);
  frcsno = si_div(snowh, dtot);
  frcsoi = si_div(dsoil, dtot);
  df1a = si_add(si_mul(frcsno, sncond), si_mul(frcsoi, df1));
  df1 = si_add(si_mul(df1a, sncovr), si_mul(df1, si_sub(1.0f, sncovr)));
  ssoil = si_div(si_mul(df1, si_sub(t1, stc[1])), dtot);
  snowz0(sncovr, z0, z0brd, snowh, status);
  fdown = si_add(solnet, lwdn);
  t2v = si_mul(sfctmp, si_add(1.0f, si_mul(0.61f, q2)));
  t24 = si_mul(si_mul(si_mul(sfctmp, sfctmp), sfctmp), sfctmp);
  rho = si_div(sfcprs, si_mul(rd, t2v));
  rch = si_mul(si_mul(rho, 1004.6f), ch);
  penman(sfctmp, sfcprs, ch, th2, prcp, fdown, t24, ssoil, q2, q2sat, etp, rch, rr, snowng, frzgra, dqsdt2, flx2, emissi, t1, status);
  esnow = 0.0f;
  snopac(etp, eta, prcp, snowng, nsoil, dt, df1, q2, t1, sfctmp, t24, th2, fdown, ssoil, stc, sfcprs, rch, rr, sncovr, sneqv, sndens, snowh, zsoil, tbot, snomlt, dew, flx1, flx2, flx3, esnow, emissi, ribb, seaice_albedo_opt, status);
  eta_kinematic = etp;
  if ((seaice_snowdepth_opt == 0)) {
  sndens = si_div(sneqv, snowh);
  snowh = fmax(seaice_snowdepth_min, fmin(snowh, seaice_snowdepth_max));
  sneqv = si_mul(snowh, sndens);
  } else if ((seaice_snowdepth_opt == 1)) {
  sndens = 0.3f;
  snowh = snowonsi;
  sneqv = si_mul(snowh, sndens);
  }
  q1 = si_add(q2, si_div(si_mul(eta_kinematic, cp), rch));
  // MPAS module_sf_noah_seaice.F:514: GNU negates after the complete multiply/divide chain.
  sheat = si_neg(si_mul(si_div(si_mul(si_mul(ch, cp), sfcprs), si_mul(r, t2v)), si_sub(th2, t1)));
  esnow = si_mul(esnow, lsubs);
  etp = si_mul(etp, si_add(si_mul(si_sub(1.0f, sncovr), lvh2o), si_mul(sncovr, lsubs)));
  if ((etp > 0.0f)) {
  eta = esnow;
  } else {
  eta = etp;
  }
  // MPAS module_sf_noah_seaice.F:534: native uses sign-bit negation.
  ssoil = si_neg(ssoil);
  runoff1 = si_div(snomlt, dt);
}

// MPAS physics_wrf/module_sf_noah_seaice.F:548-587
static void csnow(thread float & sncond, thread float & dsnow, thread int & status) {
  float c;
  const float unit = 0.11631f;
  c = si_mul(0.328f, si_nan_result(seaice_pow(10, si_mul(2.25f, dsnow)), 10,  si_mul(2.25f, dsnow)));
  sncond = si_mul(si_mul(2.0f, unit), c);
}

// MPAS physics_wrf/module_sf_noah_seaice.F:589-707
static void hrtice(thread float * rhsts, thread float * stc, thread float & tbot, thread int & nsoil, thread float * zsoil, thread float & yy, thread float & zz1, thread float & df1, thread float * ai, thread float * bi, thread float * ci, thread int & status) {
  int k;
  float ddz;
  float ddz2;
  float denom;
  float dtsdz;
  float dtsdz2;
  float ssoil;
  float zbot;
  float hcpct;
  float df1k;
  float df1n;
  float zmd;
  hcpct = 1723960.0f;
  zbot = zsoil[nsoil];
  ddz = si_div(1.0f, si_mul((-0.5f), zsoil[2]));
  ai[1] = 0.0f;
  ci[1] = si_div(si_mul(df1, ddz), si_mul(zsoil[1], hcpct));
  bi[1] = si_add(si_neg(ci[1]), si_div(df1, si_mul(si_mul(si_mul(si_mul(0.5f, zsoil[1]), zsoil[1]), hcpct), zz1)));
  dtsdz = si_div(si_sub(stc[1], stc[2]), si_mul((-0.5f), zsoil[2]));
  ssoil = si_div(si_mul(df1, si_sub(stc[1], yy)), si_mul(si_mul(0.5f, zsoil[1]), zz1));
  rhsts[1] = si_div(si_sub(si_mul(df1, dtsdz), ssoil), si_mul(zsoil[1], hcpct));
  ddz2 = 0.0f;
  df1k = df1;
  df1n = df1;
  for (k = 2; k <= nsoil; k += 1) {
  if ((k != nsoil)) {
  denom = si_mul(0.5f, si_sub(zsoil[(k - 1)], zsoil[(k + 1)]));
  dtsdz2 = si_div(si_sub(stc[k], stc[(k + 1)]), denom);
  ddz2 = si_div(2.0f, si_sub(zsoil[(k - 1)], zsoil[(k + 1)]));
  ci[k] = si_neg(si_div(si_mul(df1n, ddz2), si_mul(si_sub(zsoil[(k - 1)], zsoil[k]), hcpct)));
  } else {
  dtsdz2 = si_div(si_sub(stc[k], tbot), si_sub(si_mul(0.5f, si_add(zsoil[(k - 1)], zsoil[k])), zbot));
  ci[k] = 0.0f;
  }
  denom = si_mul(si_sub(zsoil[k], zsoil[(k - 1)]), hcpct);
  rhsts[k] = si_div(si_sub(si_mul(df1n, dtsdz2), si_mul(df1k, dtsdz)), denom);
  ai[k] = si_neg(si_div(si_mul(df1k, ddz), si_mul(si_sub(zsoil[(k - 1)], zsoil[k]), hcpct)));
  bi[k] = (si_neg(si_add(ai[k], ci[k])));
  df1k = df1n;
  dtsdz = dtsdz2;
  ddz = ddz2;
  }
}

// MPAS physics_wrf/module_sf_noah_seaice.F:710-782
static void penman(thread float & sfctmp, thread float & sfcprs, thread float & ch, thread float & th2, thread float & prcp, thread float & fdown, thread float & t24, thread float & ssoil, thread float & q2, thread float & q2sat, thread float & etp, thread float & rch, thread float & rr, thread bool & snowng, thread bool & frzgra, thread float & dqsdt2, thread float & flx2, thread float & emissi, thread float & t1, thread int & status) {
  float elcp1;
  float lvs;
  float epsca;
  float a;
  float delta;
  float fnet;
  float rad;
  const float elcp = 2488.8f;
  const float lsubc = 2501000.0f;
  const float cp = 1004.6f;
  const float lsubs = 2830000.0f;
  if ((t1 > 273.15f)) {
  elcp1 = elcp;
  lvs = lsubc;
  } else {
  elcp1 = si_div(si_mul(elcp, lsubs), lsubc);
  lvs = lsubs;
  }
  flx2 = 0.0f;
  delta = si_mul(elcp1, dqsdt2);
  rr = si_add(si_div(si_mul(si_mul(emissi, t24), 6.48e-08f), si_mul(sfcprs, ch)), 1.0f);
  if ((prcp > 0.0f)) {
  if ((!snowng)) {
  rr = si_add(rr, si_div(si_mul(cph2o, prcp), rch));
  } else {
  rr = si_add(rr, si_div(si_mul(cpice, prcp), rch));
  }
  }
  fnet = si_sub(si_sub(fdown, si_mul(si_mul(emissi, sigma), t24)), ssoil);
  if (frzgra) {
  flx2 = si_neg(si_mul(lsubf, prcp));
  fnet = si_sub(fnet, flx2);
  }
  rad = si_sub(si_add(si_div(fnet, rch), th2), sfctmp);
  a = si_mul(elcp1, si_sub(q2sat, q2));
  epsca = si_div(si_add(si_mul(a, rr), si_mul(rad, delta)), si_add(delta, rr));
  etp = si_div(si_mul(epsca, rch), lvs);
}

// MPAS physics_wrf/module_sf_noah_seaice.F:785-812
static void shflx(thread float * stc, thread int & nsoil, thread float & dt, thread float & yy, thread float & zz1, thread float * zsoil, thread float & tbot, thread float & df1, thread int & status) {
  float ai[NSOLD+1];
  float bi[NSOLD+1];
  float ci[NSOLD+1];
  float stcf[NSOLD+1];
  float rhsts[NSOLD+1];
  int i;
  const float t0 = 273.15f;
  hrtice(rhsts, stc, tbot, nsoil, zsoil, yy, zz1, df1, ai, bi, ci, status);
  hstep(stcf, stc, rhsts, dt, nsoil, ai, bi, ci, status);
  for (i = 1; i <= nsoil; i += 1) {
  stc[i] = stcf[i];
  }
}

// MPAS physics_wrf/module_sf_noah_seaice.F:815-1079
static void snopac(thread float & etp, thread float & eta, thread float & prcp, thread bool & snowng, thread int & nsoil, thread float & dt, thread float & df1, thread float & q2, thread float & t1, thread float & sfctmp, thread float & t24, thread float & th2, thread float & fdown, thread float & ssoil, thread float * stc, thread float & sfcprs, thread float & rch, thread float & rr, thread float & sncovr, thread float & esd, thread float & sndens, thread float & snowh, thread float * zsoil, thread float & tbot, thread float & snomlt, thread float & dew, thread float & flx1, thread float & flx2, thread float & flx3, thread float & esnow, thread float & emissi, thread float & ribb, thread int & seaice_albedo_opt, thread int & status) {
  int k;
  float denom;
  float dsoil;
  float dtot;
  float esnow1;
  float esnow2;
  float eta1;
  float etp1;
  float etp2;
  float etanrg;
  float ex;
  float seh;
  float sncond;
  float t12;
  float t12a;
  float t12b;
  float t14;
  float yy;
  float zz1;
  const float esdmin = 1e-06f;
  const float lsubc = 2501000.0f;
  const float lsubs = 2830000.0f;
  const float snoexp = 2.0f;
  dew = 0.0f;
  esnow = 0.0f;
  esnow1 = 0.0f;
  esnow2 = 0.0f;
  if ((etp <= 0.0f)) {
  if (((ribb >= 0.1f) && (fdown > 150.0f))) {
  etp = si_div(si_add(si_div(si_mul(fmin(si_mul(etp, si_sub(1.0f, ribb)), 0.0f), sncovr), 0.98f), si_mul(etp, si_sub(0.98f, sncovr))), 0.98f);
  }
  etp1 = si_mul(etp, 0.001f);
  dew = si_neg(etp1);
  esnow2 = si_mul(etp1, dt);
  etanrg = si_mul(etp, si_add(si_mul(si_sub(1.0f, sncovr), lsubc), si_mul(sncovr, lsubs)));
  } else {
  etp1 = si_mul(etp, 0.001f);
  esnow = etp;
  esnow1 = si_mul(esnow, 0.001f);
  esnow2 = si_mul(esnow1, dt);
  etanrg = si_mul(esnow, lsubs);
  esnow = si_mul(etp, sncovr);
  esnow1 = si_mul(esnow, 0.001f);
  esnow2 = si_mul(esnow1, dt);
  etanrg = si_mul(esnow, lsubs);
  }
  flx1 = 0.0f;
  if (snowng) {
  flx1 = si_mul(si_mul(cpice, prcp), si_sub(t1, sfctmp));
  } else {
  if ((prcp > 0.0f)) flx1 = si_mul(si_mul(cph2o, prcp), si_sub(t1, sfctmp));
  }
  dsoil = (si_neg(si_mul(0.5f, zsoil[1])));
  dtot = si_add(snowh, dsoil);
  denom = si_add(1.0f, si_div(df1, si_mul(si_mul(dtot, rr), rch)));
  t12a = si_div(si_sub(si_sub(si_add(si_div(si_sub(si_sub(si_sub(fdown, flx1), flx2), si_mul(si_mul(emissi, sigma), t24)), rch), th2), sfctmp), si_div(etanrg, rch)), rr);
  t12b = si_div(si_mul(df1, stc[1]), si_mul(si_mul(dtot, rr), rch));
  t12 = si_div(si_add(si_add(sfctmp, t12a), t12b), denom);
  if ((t12 <= tfreez)) {
  t1 = t12;
  ssoil = si_div(si_mul(df1, si_sub(t1, stc[1])), dtot);
  esd = fmax(0.0f, si_sub(esd, esnow2));
  flx3 = 0.0f;
  ex = 0.0f;
  snomlt = 0.0f;
  } else {
  t1 = tfreez;
  ssoil = si_div(si_mul(df1, si_sub(t1, stc[1])), dtot);
  if ((si_sub(esd, esnow2) <= esdmin)) {
  esd = 0.0f;
  ex = 0.0f;
  snomlt = 0.0f;
  flx3 = 0.0f;
  } else {
  esd = si_sub(esd, esnow2);
  seh = si_mul(rch, si_sub(t1, th2));
  t14 = si_mul(si_mul(t1, t1), si_mul(t1, t1));
  flx3 = si_sub(si_sub(si_sub(si_sub(si_sub(si_sub(fdown, flx1), flx2), si_mul(si_mul(emissi, sigma), t14)), ssoil), seh), etanrg);
  if ((flx3 <= 0.0f)) flx3 = 0.0f;
  ex = si_div(si_mul(flx3, 0.001f), lsubf);
  snomlt = si_mul(ex, dt);
  if ((si_sub(esd, snomlt) >= esdmin)) {
  esd = si_sub(esd, snomlt);
  } else {
  ex = si_div(esd, dt);
  flx3 = si_mul(si_mul(ex, 1000.0f), lsubf);
  snomlt = esd;
  esd = 0.0f;
  }
  }
  }
  zz1 = 1.0f;
  yy = si_sub(stc[1], si_div(si_mul(si_mul(si_mul(0.5f, ssoil), zsoil[1]), zz1), df1));
  shflx(stc, nsoil, dt, yy, zz1, zsoil, tbot, df1, status);
  switch (seaice_albedo_opt) {
  default:
  if ((esd >= 0.01f)) {
  snowpack(esd, dt, snowh, sndens, t1, yy, status);
  } else {
  esd = 0.01f;
  snowh = 0.05f;
  sncovr = 1.0f;
  }
  break;
  case 1:
  if ((esd >= 0.0001f)) {
  snowpack(esd, dt, snowh, sndens, t1, yy, status);
  } else {
  esd = 0.0001f;
  snowh = 0.0005f;
  sncovr = 0.005f;
  }
  break;
  }
}

// MPAS physics_wrf/module_sf_noah_seaice.F:1082-1207
static void snowpack(thread float & esd, thread float & dtsec, thread float & snowh, thread float & sndens, thread float & tsnow, thread float & tsoil, thread int & status) {
  int ipol;
  int j;
  float bfac;
  float dsx;
  float dthr;
  float dw;
  float snowhc;
  float pexp;
  float tavgc;
  float tsnowc;
  float tsoilc;
  float esdc;
  float esdcx;
  const float c1 = 0.01f;
  const float c2 = 21.0f;
  const float g = 9.81f;
  const float kn = 4000.0f;
  snowhc = si_mul(snowh, 100.0f);
  esdc = si_mul(esd, 100.0f);
  dthr = si_div(dtsec, 3600.0f);
  tsnowc = si_sub(tsnow, 273.15f);
  tsoilc = si_sub(tsoil, 273.15f);
  tavgc = si_mul(0.5f, si_add(tsnowc, tsoilc));
  if ((esdc > 0.01f)) {
  esdcx = esdc;
  } else {
  esdcx = 0.01f;
  }
  bfac = si_mul(si_mul(dthr, c1), si_nan_result(seaice_exp(si_sub(si_mul(0.08f, tavgc), si_mul(c2, sndens))), si_sub(si_mul(0.08f, tavgc), si_mul(c2, sndens)), si_sub(si_mul(0.08f, tavgc), si_mul(c2, sndens))));
  ipol = 4;
  pexp = 0.0f;
  for (j = ipol; j >= 1; j += (-1)) {
  pexp = si_div(si_mul(si_mul(si_add(1.0f, pexp), bfac), esdcx), (float)((j + 1)));
  }
  pexp = si_add(pexp, 1.0f);
  dsx = si_mul(sndens, pexp);
  if ((dsx > 0.4f)) dsx = 0.4f;
  if ((dsx < 0.05f)) dsx = 0.05f;
  sndens = dsx;
  if ((tsnowc >= 0.0f)) {
  dw = si_div(si_mul(0.13f, dthr), 24.0f);
  sndens = si_add(si_mul(sndens, si_sub(1.0f, dw)), dw);
  if ((sndens >= 0.4f)) sndens = 0.4f;
  }
  snowhc = si_div(esdc, sndens);
  snowh = si_mul(snowhc, 0.01f);
}

// MPAS physics_wrf/module_sf_noah_seaice.F:1210-1239
static void snowz0(thread float & sncovr, thread float & z0, thread float & z0brd, thread float & snowh, thread int & status) {
  const float z0s = 0.001f;
  float burial;
  float z0eff;
  burial = si_sub(si_mul(7.0f, z0brd), snowh);
  if ((burial <= 0.0007f)) {
  z0eff = z0s;
  } else {
  z0eff = si_div(burial, 7.0f);
  }
  z0 = si_add(si_mul(si_sub(1.0f, sncovr), z0brd), si_mul(sncovr, z0eff));
}

// MPAS physics_wrf/module_sf_noah_seaice.F:1243-1292
static void snow_new(thread float & temp, thread float & newsn, thread float & snowh, thread float & sndens, thread int & status) {
  float dsnew;
  float hnewc;
  float snowhc;
  float newsnc;
  float tempc;
  snowhc = si_mul(snowh, 100.0f);
  newsnc = si_mul(newsn, 100.0f);
  tempc = si_sub(temp, 273.15f);
  if ((tempc <= (-15.0f))) {
  dsnew = 0.05f;
  } else {
  dsnew = si_add(0.05f, si_mul(0.0017f, si_nan_result(seaice_pow(si_add(tempc, 15.0f), 1.5f), si_add(tempc, 15.0f),  1.5f)));
  }
  hnewc = si_div(newsnc, dsnew);
  if ((si_add(snowhc, hnewc) < 0.001f)) {
  sndens = fmax(dsnew, sndens);
  } else {
  sndens = si_div(si_add(si_mul(snowhc, sndens), si_mul(hnewc, dsnew)), si_add(snowhc, hnewc));
  }
  snowhc = si_add(snowhc, hnewc);
  snowh = si_mul(snowhc, 0.01f);
}

kernel void hexv842_seaice_from_f32(
    constant Hexv842SeaiceFromF32Resources &r [[buffer(0)]],
    constant Hexv842SeaiceFromF32Params &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    int i=int(tid),n=p.n,nsoil=p.nsoil;
    float dt=p.dt,threshold=p.threshold,albedo_default=p.albedo_default;
    float thickness_default=p.thickness_default,snowdepth_max=p.snowdepth_max,snowdepth_min=p.snowdepth_min;
    int albedo_opt=p.albedo_opt,thickness_opt=p.thickness_opt,snowdepth_opt=p.snowdepth_opt;
    int frpcpn=p.frpcpn,optional_noah=p.optional_noah,microp=p.microp,convection=p.convection;

  if(i>=n) return;
  r.fields[7*n+i]=si_div(r.fields[38*n+i],si_sub(1.0f,r.fields[15*n+i]));
  r.fields[16*n+i]=albedo_default; r.fields[36*n+i]=snowdepth_min; r.fields[37*n+i]=thickness_default;
  r.fields[42*n+i]=r.fields[19*n+i];
  if(r.fields[10*n+i]>=threshold && r.fields[10*n+i]<=1.0f) {
    r.fields[15*n+i]=si_div(si_sub(r.fields[15*n+i],si_mul(0.08f,si_sub(1.0f,r.fields[10*n+i]))),r.fields[10*n+i]);
    r.fields[14*n+i]=si_div(si_sub(r.fields[14*n+i],si_mul(0.98f,si_sub(1.0f,r.fields[10*n+i]))),r.fields[10*n+i]);
    r.fields[42*n+i]=fmax(r.fields[42*n+i],271.4f);
    if(r.fields[10*n+i]<0.2f && r.fields[19*n+i]<253.15f) r.fields[19*n+i]=253.15f;
    if(r.fields[10*n+i]<0.1f && r.fields[19*n+i]<263.15f) r.fields[19*n+i]=263.15f;
  }
  r.fields[4*n+i]=0.0f; r.fields[8*n+i]=0.0f;
  if(microp) { r.fields[4*n+i]=r.fields[39*n+i]; r.fields[8*n+i]=si_add(r.fields[8*n+i],r.fields[40*n+i]); }
  if(convection) r.fields[8*n+i]=si_add(r.fields[8*n+i],r.fields[41*n+i]);

}

kernel void hexv842_seaice_column_f32(
    constant Hexv842SeaiceColumnF32Resources &r [[buffer(0)]],
    constant Hexv842SeaiceColumnF32Params &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    int i=int(tid),n=p.n,nsoil=p.nsoil;
    float dt=p.dt,threshold=p.threshold,albedo_default=p.albedo_default;
    float thickness_default=p.thickness_default,snowdepth_max=p.snowdepth_max,snowdepth_min=p.snowdepth_min;
    int albedo_opt=p.albedo_opt,thickness_opt=p.thickness_opt,snowdepth_opt=p.snowdepth_opt;
    int frpcpn=p.frpcpn,optional_noah=p.optional_noah,microp=p.microp,convection=p.convection;

  if (i >= n) return;
  int status = 0;
  if (r.fields[10*n+i] < threshold) {
    if (thickness_opt == 1) r.fields[37*n+i] = 0.0f;
    if (snowdepth_opt == 1) r.fields[36*n+i] = 0.0f;
    return;
  }
  float sithick;
  if (thickness_opt == 0) sithick = thickness_default;
  else if (thickness_opt == 1) {
    if (r.fields[37*n+i] < -1.0e6f) { r.errors[i]=2; return; }
    sithick = fmin(fmax(0.10f,r.fields[37*n+i]),10.0f);
    r.fields[37*n+i]=sithick;
  } else { r.errors[i]=3; return; }
  float sfctmp=r.fields[0*n+i], t1=r.fields[19*n+i], snoalb, albedosi=0.0f;
  if (albedo_opt == 2) {
    if (r.fields[16*n+i] < -1.0e6f) { r.errors[i]=4; return; }
    snoalb=r.fields[16*n+i]; r.fields[15*n+i]=r.fields[16*n+i]; albedosi=r.fields[16*n+i];
  } else snoalb=r.fields[9*n+i];
  float zlvl=si_mul(0.5f,r.fields[3*n+i]), emissi=r.fields[14*n+i], lwdn=si_mul(r.fields[5*n+i],emissi);
  float sneqv=si_mul(r.fields[18*n+i],0.001f), snowh=r.fields[21*n+i], sncovr=r.fields[20*n+i];
  float solnet=si_mul(r.fields[7*n+i],si_sub(1.0f,r.fields[15*n+i]));
  float sfcprs=si_mul(si_add(r.p8w2[i],r.fields[2*n+i]),0.5f);
  float psfc=r.fields[2*n+i], q2=si_div(r.fields[1*n+i],si_add(1.0f,r.fields[1*n+i]));
  const float capa=287.0f/1004.5f;
  const float a23m4=17.67f*(273.15f-29.65f);
  float apes=si_nan_result(seaice_pow(si_div(1.0e5f,psfc),capa), si_div(1.0e5f,psfc), capa);
  float apelm=si_nan_result(seaice_pow(si_div(1.0e5f,sfcprs),capa), si_div(1.0e5f,sfcprs), capa);
  float th2=si_div(si_mul(sfctmp,apelm),apes);
  float q2sat=si_div(r.fields[6*n+i],si_add(1.0f,r.fields[6*n+i]));
  float tempdiff=si_sub(sfctmp,29.65f);
  float dqsdt2=si_div(si_mul(q2sat,a23m4),si_mul(tempdiff,tempdiff));
  float snowonsi=0.0f;
  if (snowdepth_opt == 1) {
    snowonsi=fmax(snowdepth_min,fmin(r.fields[36*n+i],snowdepth_max));
    sneqv=si_mul(snowonsi,0.3f);
    r.fields[21*n+i]=snowonsi;
    // MPAS deliberately retains the previously loaded local SNOWH here.
  } else if (snowdepth_opt != 0) { r.errors[i]=5; return; }
  if (r.fields[18*n+i] > 0.0f) {
    float sfctsno=sfctmp;
    float e2sat=si_mul(611.2f,si_nan_result(seaice_exp(si_mul(6174.0f,si_sub(1.0f/273.15f,si_div(1.0f,sfctsno)))), si_mul(6174.0f,si_sub(1.0f/273.15f,si_div(1.0f,sfctsno))), si_mul(6174.0f,si_sub(1.0f/273.15f,si_div(1.0f,sfctsno)))));
    float q2sati=si_div(si_mul(0.622f,e2sat),si_sub(sfcprs,e2sat));
    q2sati=si_div(q2sati,si_add(1.0f,q2sati));
    if (t1 > 273.14f) {
      q2sat=si_add(si_mul(q2sat,si_sub(1.0f,r.fields[20*n+i])),si_mul(q2sati,r.fields[20*n+i]));
      dqsdt2=si_add(si_mul(dqsdt2,si_sub(1.0f,r.fields[20*n+i])),si_mul(si_div(si_mul(q2sati,6174.0f),si_mul(sfctsno,sfctsno)),r.fields[20*n+i]));
    } else {
      q2sat=q2sati;
      dqsdt2=si_div(si_mul(q2sati,6174.0f),si_mul(sfctsno,sfctsno));
    }
    if (t1 > 273.0f && r.fields[20*n+i] > 0.0f) dqsdt2=si_mul(dqsdt2,si_sub(1.0f,r.fields[20*n+i]));
  }
  float prcp=si_div(r.fields[8*n+i],dt);
  float ffrozp=frpcpn ? r.fields[4*n+i] : (sfctmp <= 273.15f ? 1.0f : 0.0f);
  float tbot=271.36f, z0brd=r.fields[17*n+i], stc[NSOLD+1];
  for(int k=1;k<=nsoil;k++) stc[k]=r.soil[(k-1)*n+i];
  float ch=r.fields[12*n+i],ribb=r.fields[11*n+i];
  float z0=-1.e36f,eta=-1.e36f,sheat=-1.e36f,eta_kinematic=-1.e36f;
  float fdown=-1.e36f,esnow=-1.e36f,dew=-1.e36f,etp=-1.e36f,ssoil=-1.e36f;
  float flx1=-1.e36f,flx2=-1.e36f,flx3=-1.e36f,snomlt=-1.e36f,runoff1=-1.e36f,q1=-1.e36f;
  float albedok=albedo_opt==2 ? r.fields[16*n+i] : 0.0f;
  int iloc=i+1,jloc=1;
  // MPAS physics_wrf/module_sf_noah_seaice.F:208-209 records the call indices.
  // Reduce in wrapper loop order, independently of device scheduling.
  r.ran[i]=1;
  sflx_seaice(iloc,jloc,albedo_opt,albedo_default,snowdepth_opt,snowdepth_max,
      snowdepth_min,ffrozp,dt,zlvl,nsoil,sithick,lwdn,solnet,sfcprs,prcp,sfctmp,
      q2,th2,q2sat,dqsdt2,snoalb,tbot,z0brd,z0,emissi,t1,stc,snowh,sneqv,
      albedok,ch,albedosi,snowonsi,eta,sheat,eta_kinematic,fdown,esnow,dew,
      etp,ssoil,flx1,flx2,flx3,snomlt,sncovr,runoff1,q1,ribb,status);
  if(status) { r.errors[i]=status; return; }
  r.fields[15*n+i]=albedok; r.fields[14*n+i]=emissi; r.fields[19*n+i]=t1; r.fields[17*n+i]=z0brd;
  r.fields[21*n+i]=snowh; r.fields[20*n+i]=sncovr; r.fields[18*n+i]=si_mul(sneqv,1000.0f);
  for(int k=1;k<=nsoil;k++) r.soil[(k-1)*n+i]=stc[k];
  r.fields[31*n+i]=z0; r.fields[29*n+i]=eta; r.fields[28*n+i]=sheat; r.fields[30*n+i]=eta_kinematic; r.fields[32*n+i]=ssoil;
  r.fields[34*n+i]=r.fields[22*n+i];
  if(q1 > r.fields[33*n+i]) r.fields[22*n+i]=r.fields[12*n+i];
  r.fields[33*n+i]=si_div(q1,si_sub(1.0f,q1));
  if(snowdepth_opt==1) r.fields[36*n+i]=snowonsi;
  const float rowliw=1000.0f*350000.0f;
  // MPAS module_sf_noah_seaice_drv.F:207-208: retain each rounded divide.
  // NVRTC can replace division by a constant with reciprocal multiplication.
  float fdtliw=si_div(dt,rowliw);
  float fdtw=si_div(dt,(2500000.0f*1000.0f));
  if(optional_noah & 1) r.fields[26*n+i]=si_add(r.fields[26*n+i],si_mul(etp,fdtw));
  if(ffrozp > 0.5f) r.fields[23*n+i]=si_add(r.fields[23*n+i],si_mul(prcp,dt));
  r.fields[24*n+i]=si_add(r.fields[24*n+i],si_mul(snomlt,1000.0f));
  // MPAS module_sf_noah_seaice_drv.F:483-484 (optional melt accumulator).
  if(optional_noah & 2) r.fields[27*n+i]=si_sub(r.fields[27*n+i],si_div(snomlt,fdtliw));
  r.fields[25*n+i]=si_add(r.fields[25*n+i],si_mul(si_mul(runoff1,dt),1000.0f));
  if(optional_noah & 4) {
    float t1sq=si_mul(t1,t1);
    r.fields[35*n+i]=si_sub(si_sub(si_sub(si_sub(si_sub(si_add(si_sub(si_add(solnet,lwdn),sheat),ssoil),eta),si_mul(si_mul(emissi,5.67051e-8f),si_mul(t1sq,t1sq))),flx1),flx2),flx3);
  }

}

kernel void hexv842_seaice_to_f32(
    constant Hexv842SeaiceToF32Resources &r [[buffer(0)]],
    constant Hexv842SeaiceToF32Params &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    int i=int(tid),n=p.n,nsoil=p.nsoil;
    float dt=p.dt,threshold=p.threshold,albedo_default=p.albedo_default;
    float thickness_default=p.thickness_default,snowdepth_max=p.snowdepth_max,snowdepth_min=p.snowdepth_min;
    int albedo_opt=p.albedo_opt,thickness_opt=p.thickness_opt,snowdepth_opt=p.snowdepth_opt;
    int frpcpn=p.frpcpn,optional_noah=p.optional_noah,microp=p.microp,convection=p.convection;

  if(i>=n) return;
  if(r.fields[10*n+i]>=threshold && r.fields[10*n+i]<=1.0f) {
    float x=r.fields[10*n+i],s=si_sub(1.0f,x);
    r.fields[43*n+i]=si_add(si_mul(x,r.fields[43*n+i]),si_mul(s,r.fields[44*n+i]));
    r.fields[12*n+i]=si_add(si_mul(x,r.fields[12*n+i]),si_mul(s,r.fields[45*n+i]));
    r.fields[34*n+i]=si_add(si_mul(x,r.fields[34*n+i]),si_mul(s,r.fields[46*n+i]));
    r.fields[22*n+i]=si_add(si_mul(x,r.fields[22*n+i]),si_mul(s,r.fields[47*n+i]));
    r.fields[28*n+i]=si_add(si_mul(x,r.fields[28*n+i]),si_mul(s,r.fields[48*n+i]));
    r.fields[29*n+i]=si_add(si_mul(x,r.fields[29*n+i]),si_mul(s,r.fields[49*n+i]));
    r.fields[30*n+i]=si_add(si_mul(x,r.fields[30*n+i]),si_mul(s,r.fields[50*n+i]));
    r.fields[6*n+i]=si_add(si_mul(x,r.fields[6*n+i]),si_mul(s,r.fields[51*n+i]));
    r.fields[33*n+i]=si_add(si_mul(x,r.fields[33*n+i]),si_mul(s,r.fields[52*n+i]));
    r.fields[19*n+i]=si_add(si_mul(x,r.fields[19*n+i]),si_mul(s,r.fields[42*n+i]));
    r.fields[15*n+i]=si_add(si_mul(x,r.fields[15*n+i]),si_mul(s,0.08f));
    r.fields[14*n+i]=si_add(si_mul(x,r.fields[14*n+i]),si_mul(s,0.98f));
  }

}

kernel void hexv842_seaice_state_i32(constant Hexv842SeaiceStateI32Resources &r [[buffer(0)]],
    constant Hexv842SeaiceStateI32Params &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if(tid) return;
    int last=-1; for(int i=0;i<p.n;i++) if(r.ran[i]) last=i;
    if(last>=0) { r.state[0]=p.i_start+last; r.state[1]=p.j_index; }
}
