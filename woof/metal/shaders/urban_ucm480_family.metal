// Full WRF 4.8 SLUCM canopy with Noah and Noah-MP handoffs.
// Forcing and the additional canopy state are packed to retain the
// typed runtime's 64-resource limit.
#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_urban_trig.metalh"
#include "soft_libm64.metalh"
#include "ucm480_after_lsm_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

#define FADD(a,b) as_type<float>(soft32_add(as_type<uint>(float(a)),as_type<uint>(float(b))))
#define FSUB(a,b) as_type<float>(soft32_sub(as_type<uint>(float(a)),as_type<uint>(float(b))))
#define FMUL(a,b) as_type<float>(soft32_mul(as_type<uint>(float(a)),as_type<uint>(float(b))))
#define FDIV(a,b) as_type<float>(soft32_div(as_type<uint>(float(a)),as_type<uint>(float(b))))
#define FSQRT(a) as_type<float>(soft32_sqrt(as_type<uint>(float(a))))
static inline float gfk_exp(float x) { return as_type<float>(soft_gfk_exp(as_type<uint>(x))); }
static inline float gfk_log(float x) { return as_type<float>(soft_gfk_log(as_type<uint>(x))); }
static inline float gfk_pow(float x, float y) { return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x), as_type<uint>(y))); }

#define UCM_NL 4
#define PT_NCOL 49
#define UCM_OK 0
#define U_KARMAN 0.4f

struct Ucm480Col {
    int utype, jmonth;
    float ta, qa, ua, u1, v1, ssg, llg, rain, rhoo, za, omg, delt, chs, chs2;
    float znt;
    float tr, tb, tg, tc, qc, uc;
    float trl[UCM_NL], tbl[UCM_NL], tgl[UCM_NL];
    float xxxr, xxxb, xxxg, xxxc;
    float cmr, chr, cmc, chc, cmgr, chgr;
    float cmcr, tgr, tgrl[UCM_NL], smr[UCM_NL];
    float drelr, drelb, drelg, flxhumr, flxhumb, flxhumg;
    float ts, qs, sh, lh, lh_kin, sw, alb, lw, g, rn, psim, psih, gz1oz0;
    float u10, v10, th2, q2, ust;
    float tvg,tt,xxxvg,tvgl[4],smg[4],cmcg,flxhumvg,flxhumt;
    float declin,cosz,xlat,ssgd,ssgq; bool lsolar;
};

static inline float u_max(float a, float b) { return (a > b) ? a : b; }
static inline float u_min(float a, float b) { return (a < b) ? a : b; }

// libgcc __powisf2: gfortran -O0 lowers REAL**INTEGER to it.
static inline float u_powi(float x, int n)
{
    unsigned int m = (n < 0) ? (0u - (unsigned int)n) : (unsigned int)n;
    float y = (m % 2u) ? x : 1.0f;
    while (m >>= 1) {
        x = FMUL(x, x);
        if (m % 2u) y = FMUL(y, x);
    }
    return (n < 0) ? FDIV(1.0f, y) : y;
}

// glibc 2.43 atanf / log10f are correctly rounded; see the header.
static inline float u_atan(float x) { return as_type<float>(soft_urban_atanf(as_type<uint>(x))); }
static inline float u_log10(float x) {
    return as_type<float>(soft64_to_f32_bits(soft64_div(
        soft64_plm_log(soft64_from_f32_bits(as_type<uint>(x))),
        0x40026bb1bbb55516ul)));
}


#include "urban_ucm480_bodies.metalh"
static inline int ucm480_urban(device const float* tab,device const float* glob,device const int* sw,thread Ucm480Col& c) {
if(c.utype<1 || c.utype>sw[10]) return 3;
float tabrow[49],globrow[99]; int switches[14];
for(int k=0;k<49;++k) tabrow[k]=tab[(c.utype-1)*49+k];
for(int k=0;k<99;++k) globrow[k]=glob[k];
for(int k=0;k<14;++k) switches[k]=sw[k];
const U480Env env{tabrow,globrow,switches};int status=0;const int layers=4;
const int rowtype=1;
float zero=0.0f;float lf[4]={};
u480_urban(c.lsolar,layers,layers,layers,globrow+0,globrow+4,globrow+8,rowtype,c.ta,c.qa,c.ua,c.u1,c.v1,c.ssg,c.ssgd,c.ssgq,c.llg,c.rain,c.rhoo,c.za,c.declin,c.cosz,c.omg,c.xlat,c.delt,c.znt,c.chs,c.chs2,c.tr,c.tb,c.tg,c.tc,c.qc,c.uc,c.trl,c.tbl,c.tgl,c.xxxr,c.xxxb,c.xxxg,c.xxxc,c.ts,c.qs,c.sh,c.lh,c.lh_kin,c.sw,c.alb,c.lw,c.g,c.rn,c.psim,c.psih,c.gz1oz0,c.cmr,c.chr,c.cmc,c.chc,c.u10,c.v10,c.th2,c.q2,c.ust,zero,zero,lf,zero,zero,zero,zero,zero,c.cmcr,c.tgr,c.tgrl,c.smr,c.cmgr,c.chgr,c.jmonth,c.drelr,c.drelb,c.drelg,c.flxhumr,c.flxhumb,c.flxhumg,c.tvg,c.tt,c.xxxvg,c.tvgl,c.smg,c.cmcg,c.flxhumvg,c.flxhumt,zero,zero,zero,env,status);
return status;
}
static inline float ucm_metal_wind(float u, float v) {
    float speed = FSQRT(FADD(gfk_pow(u, 2.0f), gfk_pow(v, 2.0f)));
    return speed < 1.0f ? 1.0f : speed;
}

kernel void ucm480_after_lsm(
    constant Ucm480AfterLsmResources &r [[buffer(0)]],
    constant Ucm480AfterLsmParams &q [[buffer(1)]],
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
    const device float *tab = forcing + (q.lsm == 2 ? 19u : 11u) * plane;
    const device float *glob = tab + uint(q.n_utype) * PT_NCOL;

    Ucm480Col c;
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
    c.declin=q.declin;
    const uint solar_offset=(q.lsm==2?17u:9u)*plane;
    c.cosz=forcing[solar_offset+index];
    c.xlat=forcing[solar_offset+plane+index];
    c.lsolar=false;c.ssgd=FMUL(0.8f,c.ssg);c.ssgq=FSUB(c.ssg,c.ssgd);
    c.tvg=r.extra_state[index];c.tt=r.extra_state[plane+index];c.xxxvg=r.extra_state[2u*plane+index];
    for(uint z=0;z<4;++z) {c.tvgl[z]=r.extra_state[(3u+z)*plane+index];c.smg[z]=r.extra_state[(7u+z)*plane+index];}
    c.cmcg=r.extra_state[11u*plane+index];c.flxhumvg=r.extra_state[12u*plane+index];c.flxhumt=r.extra_state[13u*plane+index];
    const int code = ucm480_urban(tab, glob, r.switches, c);
    if (code != UCM_OK) {
        atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.status),
            1u << uint(code - 1), memory_order_relaxed);
        return;
    }
    r.extra_state[index]=c.tvg;r.extra_state[plane+index]=c.tt;r.extra_state[2u*plane+index]=c.xxxvg;
    for(uint z=0;z<4;++z) {r.extra_state[(3u+z)*plane+index]=c.tvgl[z];r.extra_state[(7u+z)*plane+index]=c.smg[z];}
    r.extra_state[11u*plane+index]=c.cmcg;r.extra_state[12u*plane+index]=c.flxhumvg;r.extra_state[13u*plane+index]=c.flxhumt;
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
