#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
using namespace ntc;
#pragma clang fp contract(off)
#define A(a,b) tc_add((a),(b))
#define S(a,b) tc_sub((a),(b))
#define M(a,b) tc_mul((a),(b))
#define D(a,b) tc_div((a),(b))
#define MAX(a,b) tc_max((a),(b))
#define MIN(a,b) tc_min((a),(b))
#include "hexv842_mpas_kessler_to_f32_bindings.metalh"
kernel void hexv842_mpas_kessler_to_f32(constant Hexv842MpasKesslerToF32Resources &r [[buffer(0)]],constant Hexv842MpasKesslerToF32Params &params [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
device F* theta_m=(device F*)r.theta_m;
device F* qv=(device F*)r.qv;
device F* qc=(device F*)r.qc;
device F* qr=(device F*)r.qr;
device F* rho_zz=(device F*)r.rho_zz;
device F* zz=(device F*)r.zz;
device F* exner=(device F*)r.exner;
device F* exner_b=(device F*)r.exner_b;
device F* rtheta_b=(device F*)r.rtheta_b;
device F* rtheta_p=(device F*)r.rtheta_p;
device F* pressure_p=(device F*)r.pressure_p;
device F* pressure_b=(device F*)r.pressure_b;
device F* zgrid=(device F*)r.zgrid;
device F* surface_pressure=(device F*)r.surface_pressure;
device F* tend_sfc_pressure=(device F*)r.tend_sfc_pressure;
device F* rt_diabatic_tend=(device F*)r.rt_diabatic_tend;
device F* dtheta_dt_mp=(device F*)r.dtheta_dt_mp;
device F* t=(device F*)r.t;
device F* qvp=(device F*)r.qvp;
device F* qcp=(device F*)r.qcp;
device F* qrp=(device F*)r.qrp;
F rvord=F(params.rvord);
F rd=F(params.rd);
F p0=F(params.p0);
F rcv=F(params.rcv);
F gravity=F(params.gravity);
F dt_dyn=F(params.dt_dyn);
int mp_top=params.mp_top;
int nz=params.nz;
int nc=params.nc;
int ns=params.ns;

 int c=int(gid);
 if(c>=ns) return;
 for(int k=0;k<mp_top;++k) {
  int p=k*nc+c;
  F old=theta_m[p], olddry=D(old,A(1.0f,M(rvord,qv[p])));
  qv[p]=qvp[p]; qc[p]=qcp[p]; qr[p]=qrp[p];
  theta_m[p]=M(t[p],A(1.0f,M(rvord,qvp[p])));
  rt_diabatic_tend[p]=D(S(theta_m[p],old),dt_dyn);
  dtheta_dt_mp[p]=D(S(D(theta_m[p],A(1.0f,M(rvord,qv[p]))),olddry),dt_dyn);
  rtheta_p[p]=S(M(rho_zz[p],theta_m[p]),rtheta_b[p]);
  exner[p]=tc_pow(M(M(zz[p],D(rd,p0)),A(rtheta_p[p],rtheta_b[p])),rcv);
  pressure_p[p]=M(M(zz[p],rd),A(M(exner[p],rtheta_p[p]),M(S(exner[p],exner_b[p]),rtheta_b[p])));
 }
 for(int k=mp_top;k<nz;++k) { int p=k*nc+c; qc[p]=0.0f; qr[p]=0.0f; rt_diabatic_tend[p]=0.0f; }
 F tem1=S(zgrid[c+nc],zgrid[c]), tem2=S(zgrid[c+2*nc],zgrid[c+nc]);
 F rho1=M(M(rho_zz[c],zz[c]),A(1.0f,qvp[c]));
 F rho2=M(M(rho_zz[c+nc],zz[c+nc]),A(1.0f,qvp[c+nc]));
 F old=surface_pressure[c];
 surface_pressure[c]=M(M(M(0.5f,gravity),tem1),S(rho1,D(M(M(0.5f,S(rho2,rho1)),tem1),A(tem1,tem2))));
 surface_pressure[c]=A(A(surface_pressure[c],pressure_p[c]),pressure_b[c]);
 tend_sfc_pressure[c]=D(S(surface_pressure[c],old),dt_dyn);

}
