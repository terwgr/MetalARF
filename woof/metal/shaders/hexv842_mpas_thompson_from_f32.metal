#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_scalar.metalh"
#include "hexv842_mpas_thompson_from_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_thompson_from_f32(constant Hexv842MpasThompsonFromF32Resources &r [[buffer(0)]],constant Hexv842MpasThompsonFromF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
int nz=p.nz,columns=p.columns,top=p.top;float dt_dyn=p.dt_dyn;
float R_d=p.R_d;
float R_v=p.R_v;
float P0=p.P0;
float rcv=p.rcv;
float rvord=p.rvord;
float gravity=p.gravity;
float re_qc_bg=p.re_qc_bg;
float re_qi_bg=p.re_qi_bg;
float re_qs_bg=p.re_qs_bg;

    device float* m_qv=r.m_qv;
device float* m_qc=r.m_qc;
device float* m_qr=r.m_qr;
device float* m_qi=r.m_qi;
device float* m_qs=r.m_qs;
device float* m_qg=r.m_qg;
device float* m_ni=r.m_ni;
device float* m_nr=r.m_nr;
device float* m_theta_m=r.m_theta_m;
device float* m_rho_zz=r.m_rho_zz;
device float* m_w=r.m_w;
device float* g_zz=r.g_zz;
device float* g_zgrid=r.g_zgrid;
device float* d_exner=r.d_exner;
device float* d_pressure_base=r.d_pressure_base;
device float* d_pressure_p=r.d_pressure_p;
device float* d_rtheta_base=r.d_rtheta_base;
device float* d_rtheta_p=r.d_rtheta_p;
device float* d_exner_base=r.d_exner_base;
device float* d_rt_diabatic_tend=r.d_rt_diabatic_tend;
device float* d_dtheta_dt_mp=r.d_dtheta_dt_mp;
device float* t_rthmpten=r.t_rthmpten;
device float* t_rqvmpten=r.t_rqvmpten;
device float* t_rqcmpten=r.t_rqcmpten;
device float* t_rqrmpten=r.t_rqrmpten;
device float* t_rqimpten=r.t_rqimpten;
device float* t_rqsmpten=r.t_rqsmpten;
device float* t_rqgmpten=r.t_rqgmpten;
device float* t_rnimpten=r.t_rnimpten;
device float* t_rnrmpten=r.t_rnrmpten;
device float* f_qv=r.f_qv;
device float* f_qc=r.f_qc;
device float* f_qi=r.f_qi;
device float* f_qr=r.f_qr;
device float* f_qs=r.f_qs;
device float* f_qg=r.f_qg;
device float* f_ni=r.f_ni;
device float* f_nr=r.f_nr;
device float* f_th=r.f_th;
device float* f_pii=r.f_pii;
device float* f_p=r.f_p;
device float* f_w=r.f_w;
device float* f_dz=r.f_dz;
device float* f_rainprod=r.f_rainprod;
device float* f_evapprod=r.f_evapprod;
device float* f_re_cloud=r.f_re_cloud;
device float* f_re_ice=r.f_re_ice;
device float* f_re_snow=r.f_re_snow;
device float* f_rho=r.f_rho;
device float* f_z=r.f_z;
device float* d_surface_pressure=r.d_surface_pressure;
device float* d_tend_sfc_pressure=r.d_tend_sfc_pressure;
    int idx=int(gid);if(idx>=nz*columns)return;
    f_qv[idx]=m_qv[idx];f_qc[idx]=m_qc[idx];f_qr[idx]=m_qr[idx];
    f_qi[idx]=m_qi[idx];f_qs[idx]=m_qs[idx];f_qg[idx]=m_qg[idx];
    f_ni[idx]=m_ni[idx];f_nr[idx]=m_nr[idx];
    f_rho[idx]=tc_mul(g_zz[idx],m_rho_zz[idx]);
    f_th[idx]=tc_div(m_theta_m[idx],(tc_add(1.0f,tc_mul(tc_div(R_v,R_d),tc_max(0.0f,f_qv[idx])))));
    f_pii[idx]=d_exner[idx];f_p[idx]=tc_add(d_pressure_base[idx],d_pressure_p[idx]);
    f_z[idx]=g_zgrid[idx];f_dz[idx]=tc_sub(g_zgrid[idx+columns],g_zgrid[idx]);f_w[idx]=m_w[idx];
    t_rthmpten[idx]=tc_div(m_theta_m[idx],(tc_add(1.0f,tc_mul(tc_div(R_v,R_d),tc_max(0.0f,m_qv[idx])))));
    t_rqvmpten[idx]=m_qv[idx];t_rqcmpten[idx]=m_qc[idx];t_rqrmpten[idx]=m_qr[idx];
    t_rqimpten[idx]=m_qi[idx];t_rqsmpten[idx]=m_qs[idx];t_rqgmpten[idx]=m_qg[idx];
    t_rnimpten[idx]=m_ni[idx];t_rnrmpten[idx]=m_nr[idx];

}
