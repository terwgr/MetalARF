#include <metal_stdlib>
#include "urban_bep_couple_family_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubc_pow(float x, float y) { return pow(x, y); }
static inline float ubc_sqrt(float x) { return sqrt(max(x, 0.0f)); }

// Noah/Noah-MP BEP coupling, one horizontal column per thread. The resource
// vectors are flattened C-order views over the public (k, y, x) fields.
kernel void urban_bep_couple(
    constant UrbanBepCoupleResources &r [[buffer(0)]],
    constant UrbanBepCoupleParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const int ncol = p.ncol;
    if (col >= uint(ncol)) return;
    const float frc = r.frc_urb2d[col];
    const float omf = 1.0f - frc;
    const float hfx_rural = r.hfx[col], qfx_rural = r.qfx[col];
    const float grdflx_rural = r.grdflx[col], emiss_rural = r.emiss[col];
    const float tsk_rural = r.tsk[col], alb_rural = r.albedo[col];
    float umom_urb = 0.0f, vmom_urb = 0.0f;
    float hfx_urb = 0.0f, qfx_urb = 0.0f;
    const ulong ncol_u = ulong(ncol);
    for (int k = 0; k < p.nz; ++k) {
        const ulong q = ulong(k) * ncol_u + ulong(col);
        r.a_u_bep[q] *= frc; r.a_v_bep[q] *= frc;
        r.a_t_bep[q] *= frc; r.a_q_bep[q] = 0.0f; r.a_e_bep[q] = 0.0f;
        r.b_u_bep[q] *= frc; r.b_v_bep[q] *= frc; r.b_t_bep[q] *= frc;
        r.b_q_bep[q] *= frc; r.b_e_bep[q] *= frc;
        hfx_urb += r.b_t_bep[q] * r.rho[q] * 1004.5f * r.dz8w[q] * r.vl_bep[q];
        qfx_urb += r.b_q_bep[q] * r.dz8w[q] * r.vl_bep[q];
        umom_urb += (r.a_u_bep[q] * r.u_phy[q] + r.b_u_bep[q]) * r.dz8w[q] * r.vl_bep[q];
        vmom_urb += (r.a_v_bep[q] * r.v_phy[q] + r.b_v_bep[q]) * r.dz8w[q] * r.vl_bep[q];
        r.vl_bep[q] = omf + r.vl_bep[q] * frc;
        r.sf_bep[q] = omf + r.sf_bep[q] * frc;
    }
    const float u = r.u_phy[col], v = r.v_phy[col];
    const float dz = r.dz8w[col], rho = r.rho[col], us = r.ust[col];
    const float speed = max(ubc_sqrt(u*u + v*v), 1.0e-20f);
    r.a_u_bep[col] += omf * (-us*us) / dz / speed;
    r.a_v_bep[col] += omf * (-us*us) / dz / speed;
    r.b_t_bep[col] += omf * hfx_rural / dz / rho / 1004.5f;
    r.b_q_bep[col] += omf * qfx_rural / dz / rho;
    const float umom = omf * us * us * u / speed + umom_urb;
    const float vmom = omf * us * us * v / speed + vmom_urb;
    r.sf_bep[col] = 1.0f;

    if (frc > 0.0f) {
        const float g = r.glw[col], sw = r.swdown[col];
        const float eu = r.emiss_urb[col], rlu = r.rl_up_urb[col];
        const float rl_up_rural = -emiss_rural * 5.67e-8f * ubc_pow(tsk_rural, 4.0f)
            - (1.0f-emiss_rural) * g;
        const float rl_up_tot = omf * rl_up_rural + frc * rlu;
        const float em = omf * emiss_rural + frc * eu;
        r.emiss[col] = em;
        float x = (-rlu - (1.0f-eu)*g) / max(eu,1.0e-20f) / 5.67e-8f;
        r.ts_urb2d[col] = ubc_pow(max(0.0f,x),0.25f);
        x = (-rl_up_tot - (1.0f-em)*g) / max(em,1.0e-20f) / 5.67e-8f;
        r.tsk[col] = ubc_pow(max(0.0f,x),0.25f);
        const float rs_abs_tot = omf * sw * (1.0f-r.albedo[col]) + frc*r.rs_abs_urb[col];
        r.albedo[col] = sw > 0.0f ? 1.0f-rs_abs_tot/sw : alb_rural;
        const float gu = r.grdflx_urb[col];
        r.grdflx[col] = omf*grdflx_rural + frc*gu*(p.lsm==4 ? -1.0f : 1.0f);
        const float qf = omf*qfx_rural + qfx_urb;
        r.qfx[col] = qf; r.lh[col] = qf*2.5e6f;
        r.hfx[col] = hfx_urb + omf*hfx_rural;
        r.sh_urb2d[col] = hfx_urb/frc;
        r.lh_urb2d[col] = p.lsm==4 ? qfx_urb*2.5e6f/frc : qfx_urb*2.5e6f;
        r.g_urb2d[col] = gu;
        r.rn_urb2d[col] = r.rs_abs_urb[col] + eu*g - rlu;
        r.ust[col] = ubc_pow(umom*umom + vmom*vmom,0.25f);
    } else {
        r.sh_urb2d[col]=0.0f; r.lh_urb2d[col]=0.0f;
        r.g_urb2d[col]=0.0f; r.rn_urb2d[col]=0.0f;
    }
}

kernel void urban_bep_sfcdiag(
    constant UrbanBepSfcdiagResources &r [[buffer(0)]],
    constant UrbanBepSfcdiagParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const int ncol=p.ncol;
    if (col>=uint(ncol) || r.utype_urb2d[col]<=0) return;
    const float th=r.th_phy[col];
    r.t2[col]=th/ubc_pow(1.0e5f/r.psfc[col],287.0f/1004.5f);
    r.th2[col]=th; r.q2[col]=r.qv[col];
    r.u10[col]=r.u_phy[col]; r.v10[col]=r.v_phy[col];
}
