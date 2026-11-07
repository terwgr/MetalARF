#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_scalar.metalh"
#include "hexv842_mpas_thompson_to_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_thompson_to_f32(constant Hexv842MpasThompsonToF32Resources &r [[buffer(0)]],constant Hexv842MpasThompsonToF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
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
    int col=int(gid);if(col>=columns)return;
    for(int k=0;k<top;k++) {
        int idx=k*columns+col;
        d_rt_diabatic_tend[idx]=m_theta_m[idx];
        d_dtheta_dt_mp[idx]=tc_div(m_theta_m[idx],(tc_add(1.0f,tc_mul(rvord,m_qv[idx]))));
        m_qv[idx]=f_qv[idx];m_qc[idx]=f_qc[idx];m_qr[idx]=f_qr[idx];
        m_theta_m[idx]=tc_mul((tc_add(1.0f,tc_mul(rvord,f_qv[idx]))),f_th[idx]);
        d_rt_diabatic_tend[idx]=tc_div((tc_sub(m_theta_m[idx],d_rt_diabatic_tend[idx])),dt_dyn);
        d_dtheta_dt_mp[idx]=tc_div((tc_sub(tc_div(m_theta_m[idx],(tc_add(1.0f,tc_mul(rvord,m_qv[idx])))),d_dtheta_dt_mp[idx])),dt_dyn);
        d_rtheta_p[idx]=tc_sub(tc_mul(m_rho_zz[idx],m_theta_m[idx]),d_rtheta_base[idx]);
        d_exner[idx]=tc_pow(tc_mul(tc_mul(g_zz[idx],(tc_div(R_d,P0))),(tc_add(d_rtheta_p[idx],d_rtheta_base[idx]))),rcv);
        d_pressure_p[idx]=tc_mul(tc_mul(g_zz[idx],R_d),(tc_add(tc_mul(d_exner[idx],d_rtheta_p[idx]),tc_mul((tc_sub(d_exner[idx],d_exner_base[idx])),d_rtheta_base[idx]))));
    }
    for(int k=top;k<nz;k++) {
        int idx=k*columns+col;m_qc[idx]=0.0f;m_qr[idx]=0.0f;d_rt_diabatic_tend[idx]=0.0f;
    }
    float tem1=tc_sub(g_zgrid[columns+col],g_zgrid[col]);
    float tem2=tc_sub(g_zgrid[2*columns+col],g_zgrid[columns+col]);
    float rho1=tc_mul(tc_mul(m_rho_zz[col],g_zz[col]),(tc_add(1.0f,f_qv[col])));
    float rho2=tc_mul(tc_mul(m_rho_zz[columns+col],g_zz[columns+col]),(tc_add(1.0f,f_qv[columns+col])));
    d_tend_sfc_pressure[col]=d_surface_pressure[col];
    d_surface_pressure[col]=tc_mul((tc_sub(rho1,tc_div(tc_mul(tc_mul(0.5f,(tc_sub(rho2,rho1))),tem1),(tc_add(tem1,tem2))))),tc_mul(tc_mul(0.5f,gravity),(tc_sub(g_zgrid[columns+col],g_zgrid[col]))));
    d_surface_pressure[col]=tc_add(tc_add(d_surface_pressure[col],d_pressure_p[col]),d_pressure_base[col]);
    d_tend_sfc_pressure[col]=tc_div((tc_sub(d_surface_pressure[col],d_tend_sfc_pressure[col])),dt_dyn);
    for(int k=0;k<nz;k++) {
        int idx=k*columns+col;
        if(k<top) {
            m_qi[idx]=f_qi[idx];m_qs[idx]=f_qs[idx];m_qg[idx]=f_qg[idx];
            t_rthmpten[idx]=tc_div((tc_sub(tc_div(m_theta_m[idx],(tc_add(1.0f,tc_mul(tc_div(R_v,R_d),tc_max(0.0f,m_qv[idx]))))),t_rthmpten[idx])),dt_dyn);
            t_rqvmpten[idx]=tc_div((tc_sub(m_qv[idx],t_rqvmpten[idx])),dt_dyn);
            t_rqcmpten[idx]=tc_div((tc_sub(m_qc[idx],t_rqcmpten[idx])),dt_dyn);
            t_rqrmpten[idx]=tc_div((tc_sub(m_qr[idx],t_rqrmpten[idx])),dt_dyn);
            t_rqimpten[idx]=tc_div((tc_sub(m_qi[idx],t_rqimpten[idx])),dt_dyn);
            t_rqsmpten[idx]=tc_div((tc_sub(m_qs[idx],t_rqsmpten[idx])),dt_dyn);
            t_rqgmpten[idx]=tc_div((tc_sub(m_qg[idx],t_rqgmpten[idx])),dt_dyn);
        } else {
            m_qi[idx]=0.0f;m_qs[idx]=0.0f;m_qg[idx]=0.0f;
            f_rainprod[idx]=0.0f;f_evapprod[idx]=0.0f;
            f_re_cloud[idx]=re_qc_bg;f_re_ice[idx]=re_qi_bg;f_re_snow[idx]=re_qs_bg;
            t_rthmpten[idx]=0.0f;t_rqvmpten[idx]=0.0f;t_rqcmpten[idx]=0.0f;t_rqrmpten[idx]=0.0f;
            t_rqimpten[idx]=0.0f;t_rqsmpten[idx]=0.0f;t_rqgmpten[idx]=0.0f;
        }
        // MPAS updates number concentrations over kte, independently of top.
        m_ni[idx]=f_ni[idx];m_nr[idx]=f_nr[idx];
        t_rnimpten[idx]=tc_div((tc_sub(m_ni[idx],t_rnimpten[idx])),dt_dyn);
        t_rnrmpten[idx]=tc_div((tc_sub(m_nr[idx],t_rnrmpten[idx])),dt_dyn);
    }

}
