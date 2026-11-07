// Shared SLUCM canopy with the Noah and Noah-MP rural handoffs.
// Float inputs, the urban table and global arrays share float_bundle to keep
// the typed entry below the runtime's 64-resource ceiling. The canopy math is
// translated from urban_ucm.cu; central integration owns generated bindings.
#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_urban_trig.metalh"
#include "soft_libm64.metalh"
#include "ucm_noahmp_after_lsm_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)
#include "urban_ucm_core.metalh"

static inline float ucm_metal_wind(float u, float v) {
    float speed = FSQRT(FADD(gfk_pow(u, 2.0f), gfk_pow(v, 2.0f)));
    return speed < 1.0f ? 1.0f : speed;
}

kernel void ucm_noahmp_after_lsm(
    constant UcmNoahmpAfterLsmResources &r [[buffer(0)]],
    constant UcmNoahmpAfterLsmParams &q [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const uint plane = uint(q.ny) * uint(q.nx);
    if (index >= plane || r.urban[index] == 0) return;
    const device float *forcing = r.float_bundle;
    const float frc = forcing[index];
    const float u1 = forcing[plane + index];
    const float v1 = forcing[2u * plane + index];
    const float t3d1 = forcing[3u * plane + index];
    const float qv1 = forcing[4u * plane + index];
    const float p8w1 = forcing[5u * plane + index];
    const float p8w2 = forcing[6u * plane + index];
    const float dz8w1 = forcing[7u * plane + index];
    const float hrang = forcing[8u * plane + index];
    const device float *tab = forcing + (q.lsm == 2 ? 17u : 9u) * plane;
    const device float *glob = tab + uint(q.n_utype) * PT_NCOL;

    UcmCol c;
    c.utype = r.utype[index];
    c.jmonth = q.jmonth;
    c.ta = t3d1;
    c.qa = q.lsm == 2 ? qv1 : FDIV(qv1, FADD(1.0f, qv1));
    c.ua = ucm_metal_wind(u1, v1);
    c.u1 = u1;
    c.v1 = v1;
    c.ssg = q.lsm == 2 ? forcing[15u * plane + index] : r.field_swdown[index];
    c.llg = r.field_glw[index];
    c.rain = FMUL(FDIV(q.lsm == 2 ? forcing[16u * plane + index] : r.field_rainbl[index], q.dt), 3600.0f);
    c.rhoo = FDIV(q.lsm == 2 ? p8w1 : FMUL(FADD(p8w2, p8w1), 0.5f),
        FMUL(FMUL(287.04f, c.ta), FADD(1.0f, FMUL(0.61f, c.qa))));
    c.za = q.lsm == 2 ? dz8w1 : FMUL(0.5f, dz8w1);
    c.delt = q.dt;
    c.omg = hrang;
    c.znt = r.field_znt[index];
    if (r.field_chs[index] < 1.0e-02f) r.field_chs[index] = 1.0e-02f;
    if (r.field_chs2[index] < 1.0e-02f) r.field_chs2[index] = 1.0e-02f;
    if (r.field_cqs2[index] < 1.0e-02f) r.field_cqs2[index] = 1.0e-02f;
    c.chs = r.field_chs[index];
    if (q.lsm == 4) r.field_chs2[index] = r.field_cqs2[index];
    c.chs2 = r.field_chs2[index];
    c.tr = r.state_tr_urb2d[index];
    c.tb = r.state_tb_urb2d[index];
    c.tg = r.state_tg_urb2d[index];
    c.tc = r.state_tc_urb2d[index];
    c.qc = r.state_qc_urb2d[index];
    c.uc = r.state_uc_urb2d[index];
    c.xxxr = r.state_xxxr_urb2d[index];
    c.xxxb = r.state_xxxb_urb2d[index];
    c.xxxg = r.state_xxxg_urb2d[index];
    c.xxxc = r.state_xxxc_urb2d[index];
    c.cmr = r.state_cmr_sfcdif[index];
    c.chr = r.state_chr_sfcdif[index];
    c.cmc = r.state_cmc_sfcdif[index];
    c.chc = r.state_chc_sfcdif[index];
    c.cmgr = r.state_cmgr_sfcdif[index];
    c.chgr = r.state_chgr_sfcdif[index];
    c.cmcr = r.state_cmcr_urb2d[index];
    c.tgr = r.state_tgr_urb2d[index];
    c.drelr = r.state_drelr_urb2d[index];
    c.drelb = r.state_drelb_urb2d[index];
    c.drelg = r.state_drelg_urb2d[index];
    c.flxhumr = r.state_flxhumr_urb2d[index];
    c.flxhumb = r.state_flxhumb_urb2d[index];
    c.flxhumg = r.state_flxhumg_urb2d[index];
    for (uint level = 0; level < UCM_NL; ++level) {
        c.trl[level] = r.state_trl_urb3d[level * plane + index];
        c.tbl[level] = r.state_tbl_urb3d[level * plane + index];
        c.tgl[level] = r.state_tgl_urb3d[level * plane + index];
        c.tgrl[level] = r.state_tgrl_urb3d[level * plane + index];
        c.smr[level] = r.state_smr_urb3d[level * plane + index];
    }
    const int code = ucm_urban(tab, glob, r.switches, c);
    if (code != UCM_OK) {
        atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.status),
            1u << uint(code - 1), memory_order_relaxed);
        return;
    }
    const float omf = FSUB(1.0f, frc);
    r.field_albedo[index] = FADD(FMUL(frc, c.alb), FMUL(omf, q.lsm == 2 ? forcing[9u * plane + index] : r.field_albedo[index]));
    r.field_hfx[index] = FADD(FMUL(frc, c.sh), FMUL(omf, q.lsm == 2 ? forcing[10u * plane + index] : r.field_hfx[index]));
    r.field_qfx[index] = FADD(FMUL(frc, c.lh_kin), FMUL(omf, q.lsm == 2 ? forcing[11u * plane + index] : r.field_qfx[index]));
    r.field_lh[index] = FADD(FMUL(frc, c.lh), FMUL(omf, q.lsm == 2 ? forcing[12u * plane + index] : r.field_lh[index]));
    r.field_grdflx[index] = FADD(FMUL(frc, q.lsm == 2 ? c.g : FMUL(c.g, -1.0f)), FMUL(omf, q.lsm == 2 ? forcing[13u * plane + index] : r.field_grdflx[index]));
    r.field_tsk[index] = FADD(FMUL(frc, c.ts), FMUL(omf, q.lsm == 2 ? forcing[14u * plane + index] : r.field_tsk[index]));
    if (q.lsm == 2) {
        const float q1 = FADD(FMUL(frc, c.qs), FMUL(omf, p8w2));
        r.field_qsfc[index] = FDIV(q1, FSUB(1.0f, q1));
    } else {
        r.field_qsfc[index] = FADD(FMUL(frc, c.qs), FMUL(omf, r.field_qsfc[index]));
    }
    r.field_ust[index] = FADD(FMUL(frc, c.ust), FMUL(omf, r.field_ust[index]));
    r.state_ts_urb2d[index] = c.ts;
    r.state_tr_urb2d[index] = c.tr;
    r.state_tb_urb2d[index] = c.tb;
    r.state_tg_urb2d[index] = c.tg;
    r.state_tc_urb2d[index] = c.tc;
    r.state_qc_urb2d[index] = c.qc;
    r.state_uc_urb2d[index] = c.uc;
    r.state_tgr_urb2d[index] = c.tgr;
    r.state_cmcr_urb2d[index] = c.cmcr;
    r.state_flxhumr_urb2d[index] = c.flxhumr;
    r.state_flxhumb_urb2d[index] = c.flxhumb;
    r.state_flxhumg_urb2d[index] = c.flxhumg;
    r.state_drelr_urb2d[index] = c.drelr;
    r.state_drelb_urb2d[index] = c.drelb;
    r.state_drelg_urb2d[index] = c.drelg;
    r.state_xxxr_urb2d[index] = c.xxxr;
    r.state_xxxb_urb2d[index] = c.xxxb;
    r.state_xxxg_urb2d[index] = c.xxxg;
    r.state_xxxc_urb2d[index] = c.xxxc;
    r.state_sh_urb2d[index] = c.sh;
    r.state_lh_urb2d[index] = c.lh;
    r.state_g_urb2d[index] = c.g;
    r.state_rn_urb2d[index] = c.rn;
    r.state_psim_urb2d[index] = c.psim;
    r.state_psih_urb2d[index] = c.psih;
    r.state_gz1oz0_urb2d[index] = c.gz1oz0;
    r.state_u10_urb2d[index] = c.u10;
    r.state_v10_urb2d[index] = c.v10;
    r.state_th2_urb2d[index] = c.th2;
    r.state_q2_urb2d[index] = c.q2;
    r.state_ust_urb2d[index] = c.ust;
    r.state_cmr_sfcdif[index] = c.cmr;
    r.state_chr_sfcdif[index] = c.chr;
    r.state_cmgr_sfcdif[index] = c.cmgr;
    r.state_chgr_sfcdif[index] = c.chgr;
    r.state_cmc_sfcdif[index] = c.cmc;
    r.state_chc_sfcdif[index] = c.chc;
    r.state_akms_urb2d[index] = FDIV(FMUL(U_KARMAN, c.ust), FSUB(c.gz1oz0, c.psim));
    for (uint level = 0; level < UCM_NL; ++level) r.state_trl_urb3d[level * plane + index] = c.trl[level];
    for (uint level = 0; level < UCM_NL; ++level) r.state_smr_urb3d[level * plane + index] = c.smr[level];
    for (uint level = 0; level < UCM_NL; ++level) r.state_tgrl_urb3d[level * plane + index] = c.tgrl[level];
    for (uint level = 0; level < UCM_NL; ++level) r.state_tbl_urb3d[level * plane + index] = c.tbl[level];
    for (uint level = 0; level < UCM_NL; ++level) r.state_tgl_urb3d[level * plane + index] = c.tgl[level];
}
