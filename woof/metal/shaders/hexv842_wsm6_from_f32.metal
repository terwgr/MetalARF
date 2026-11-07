#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_scalar.metalh"
#include "hexv842_wsm6_from_f32_bindings.metalh"
#pragma clang fp contract(off)
#define FIELD_theta_m 0
#define FIELD_rho_zz 1
#define FIELD_zz 2
#define FIELD_exner 3
#define FIELD_pressure_base 4
#define FIELD_pressure_p 5
#define FIELD_rtheta_base 6
#define FIELD_rtheta_p 7
#define FIELD_exner_base 8
#define FIELD_zgrid 9
#define FIELD_qv 10
#define FIELD_qc 11
#define FIELD_qi 12
#define FIELD_qr 13
#define FIELD_qs 14
#define FIELD_qg 15
#define FIELD_rainprod 16
#define FIELD_evapprod 17
#define FIELD_re_cloud 18
#define FIELD_re_ice 19
#define FIELD_re_snow 20
#define FIELD_rthmpten 21
#define FIELD_rqvmpten 22
#define FIELD_rqcmpten 23
#define FIELD_rqimpten 24
#define FIELD_rqrmpten 25
#define FIELD_rqsmpten 26
#define FIELD_rqgmpten 27
#define FIELD_rt_diabatic_tend 28
#define FIELD_dtheta_dt_mp 29
#define FIELD_surface_pressure 30
#define FIELD_tend_sfc_pressure 31
#define FIELD_precipw 32
#define FIELD_rainnc 33
#define FIELD_rainncv 34
#define FIELD_snownc 35
#define FIELD_snowncv 36
#define FIELD_graupelnc 37
#define FIELD_graupelncv 38
#define FIELD_sr 39
#define FIELD_refl10cm 40
#define FIELD_refl10cm_max 41
#define FIELD_refl10cm_1km 42
#define FIELD_refl10cm_1km_max 43
#define F(name,k) fields[FIELD_##name][(k)*nc+i]
#define SF(name) fields[FIELD_##name][i]
constant int SPECIES[] = {FIELD_qv,FIELD_qc,FIELD_qi,FIELD_qr,FIELD_qs,FIELD_qg};
constant int RADII[] = {FIELD_re_cloud,FIELD_re_ice,FIELD_re_snow};
constant int PRODUCTION[] = {FIELD_rainprod,FIELD_evapprod};
constant int TENDENCIES[] = {FIELD_rqvmpten,FIELD_rqcmpten,FIELD_rqimpten,FIELD_rqrmpten,FIELD_rqsmpten,FIELD_rqgmpten};
kernel void hexv842_wsm6_from_f32(constant Hexv842Wsm6FromF32Resources &res [[buffer(0)]],constant Hexv842Wsm6FromF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
int nz=p.nz;
int nc=p.nc;
int ns=p.ns;
float rv_rd=p.rv_rd;
device float *v=res.v;
device float *s=res.s;
device float *r=res.r;
device float *prod=res.prod;
device float *pi=res.pi;
device float *fields[] = {res.theta_m,res.rho_zz,res.zz,res.exner,res.pressure_base,res.pressure_p,res.rtheta_base,res.rtheta_p,res.exner_base,res.zgrid,res.qv,res.qc,res.qi,res.qr,res.qs,res.qg,res.rainprod,res.evapprod,res.re_cloud,res.re_ice,res.re_snow,res.rthmpten,res.rqvmpten,res.rqcmpten,res.rqimpten,res.rqrmpten,res.rqsmpten,res.rqgmpten,res.rt_diabatic_tend,res.dtheta_dt_mp,res.surface_pressure,res.tend_sfc_pressure,res.precipw,res.rainnc,res.rainncv,res.snownc,res.snowncv,res.graupelnc,res.graupelncv,res.sr,res.refl10cm,res.refl10cm_max,res.refl10cm_1km,res.refl10cm_1km_max};


 int i=int(gid);if(i>=ns)return;
 for(int k=0;k<nz;++k) {
  const int x=k*nc+i;
  v[x]=tc_div(F(theta_m,k),tc_add(1.f,tc_mul(rv_rd,tc_max(F(qv,k), 0.f))));
  for(int n=0;n<6;++n) v[(n+1)*nz*nc+x]=fields[SPECIES[n]][x];
  v[7*nz*nc+x]=tc_mul(F(zz,k),F(rho_zz,k));
  v[8*nz*nc+x]=tc_add(F(pressure_base,k),F(pressure_p,k));
  v[9*nz*nc+x]=tc_sub(F(zgrid,k+1),F(zgrid,k));pi[x]=F(exner,k);
  for(int n=0;n<3;++n) r[n*nz*nc+x]=fields[RADII[n]][x];
  for(int n=0;n<2;++n) prod[n*nz*nc+x]=fields[PRODUCTION[n]][x];
  F(rthmpten,k)=v[x];
  for(int n=0;n<6;++n) fields[TENDENCIES[n]][x]=fields[SPECIES[n]][x];
 }
 for(int n=0;n<7;++n)s[n*nc+i]=0.f;

}
