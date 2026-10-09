#include <metal_stdlib>
#include "mynn_surface_column_bindings.metalh"
#include "mynn_surface_seed_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_gfk_atan.metalh"

using namespace metal;

// Keep source expressions separate where CUDA explicitly requests RN and
// prevent implicit contraction elsewhere. Transcendental parity is not claimed.
#pragma clang fp contract(off)

static inline float mynn_add_rn(float a, float b) { return a + b; }
static inline float mynn_mul_rn(float a, float b) { return a * b; }
static inline float mynn_div_rn(float a, float b) { return a / b; }

// The CPU oracle is the pinned WRF 4.6.1 source built with GNU Fortran 13
// against glibc 2.43. These deterministic glibc 2.39 REAL(4) libm
// translations substantially reduce, but do not eliminate, differences from
// that oracle over the four retained fixtures; they do not establish CUDA
// parity or equivalence between glibc versions.
static inline float mynn_sf_log(float x)
{ return as_type<float>(soft_gfk_log(as_type<uint>(x))); }
static inline float mynn_sf_exp(float x)
{ return as_type<float>(soft_gfk_exp(as_type<uint>(x))); }
static inline float mynn_sf_pow(float x, float y)
{ return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x), as_type<uint>(y))); }
static inline float mynn_sf_atan(float x)
{ return as_type<float>(soft_gfk_atanf(as_type<uint>(x))); }
static inline float mynn_sf_sqrt(float x)
{ return as_type<float>(soft32_sqrt(as_type<uint>(x))); }

// module_sf_mynn.F:329-337.  The wrapper performs this cold-start seed before
// SFCLAY1D_mynn; it remains a separate entrypoint so later calls retain their
// persistent inouts without repeating the initialization.
kernel void mynn_surface_seed(
    constant MynnSurfaceSeedResources &resources [[buffer(0)]],
    constant MynnSurfaceSeedParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    const int idx = int(tid);
    const device float *u1 = resources.u1;
    const device float *v1 = resources.v1;
    const device float *qv1 = resources.qv1;
    device float *ust = resources.ust;
    device float *mol = resources.mol;
    device float *qsfc = resources.qsfc;
    device float *qstar = resources.qstar;
    const float speed2 = u1[idx] * u1[idx] + v1[idx] * v1[idx];
    ust[idx] = fmax(0.04f * mynn_sf_sqrt(speed2), 0.001f);
    mol[idx] = 0.0f;
    qsfc[idx] = qv1[idx] / (1.0f + qv1[idx]);
    qstar[idx] = 0.0f;
}

// WRF v4.6.1 MYNN surface layer (sf_sfclay_physics=5).
//
// This is the CUDA transcription of module_sf_mynn.F:SFCLAY1D_mynn for the
// pinned option identities: every defined isftcflx over water, with
// iz0tlnd=0, spp_pbl=0 and psi_opt=0.  One thread handles one independent
// horizontal column.  The MYNN surface-layer runtime selector remains
// fail-closed until the PBL kernel and full driver coupling pass their
// separate WRF oracles.

static inline float mynn_psim_stable_full(float z)
{
    return -6.1f * mynn_sf_log(z + mynn_sf_pow(1.0f + mynn_sf_pow(z, 2.5f), 1.0f / 2.5f));
}

static inline float mynn_psih_stable_full(float z)
{
    return -5.3f * mynn_sf_log(z + mynn_sf_pow(1.0f + mynn_sf_pow(z, 1.1f), 1.0f / 1.1f));
}

static inline float mynn_psim_unstable_full(float z)
{
    float x = mynn_sf_pow(1.0f - 16.0f * z, 0.25f);
    float psimk = 2.0f * mynn_sf_log(0.5f * (1.0f + x))
               + mynn_sf_log(0.5f * (1.0f + x * x))
               - 2.0f * mynn_sf_atan(x) + 2.0f * mynn_sf_atan(1.0f);
    float ym = mynn_sf_pow(1.0f - 10.0f * z, 0.33f);
    float rt3 = mynn_sf_sqrt(3.0f);
    float psimc = 1.5f * mynn_sf_log(mynn_div_rn((ym * ym + ym + 1.0f), 3.0f))
               - rt3 * mynn_sf_atan(mynn_div_rn((2.0f * ym + 1.0f), rt3))
               + 4.0f * mynn_sf_atan(1.0f) / rt3;
    return (psimk + z * z * psimc) / (1.0f + z * z);
}

static inline float mynn_psih_unstable_full(float z)
{
    float y = mynn_sf_sqrt(1.0f - 16.0f * z);
    float psihk = 2.0f * mynn_sf_log((1.0f + y) / 2.0f);
    float yh = mynn_sf_pow(1.0f - 34.0f * z, 0.33f);
    float rt3 = mynn_sf_sqrt(3.0f);
    float psihc = 1.5f * mynn_sf_log(mynn_div_rn((yh * yh + yh + 1.0f), 3.0f))
               - rt3 * mynn_sf_atan(mynn_div_rn((2.0f * yh + 1.0f), rt3))
               + 4.0f * mynn_sf_atan(1.0f) / rt3;
    return (psihk + z * z * psihc) / (1.0f + z * z);
}

static inline float mynn_table(float z, int which)
{
    // which: 0 psim stable, 1 psih stable, 2 psim unstable, 3 psih unstable.
    bool unstable = which >= 2;
    float scaled = (unstable ? -z : z) * 100.0f;
    int index = (int)scaled;
    if (index + 1 > 1000) {
        if (which == 0) return mynn_psim_stable_full(z);
        if (which == 1) return mynn_psih_stable_full(z);
        if (which == 2) return mynn_psim_unstable_full(z);
        return mynn_psih_unstable_full(z);
    }
    float fraction = scaled - (float)index;
    float sign = unstable ? -1.0f : 1.0f;
    float z0 = sign * 0.01f * (float)index;
    float z1 = sign * 0.01f * (float)(index + 1);
    float f0, f1;
    if (which == 0) {
        f0 = mynn_psim_stable_full(z0); f1 = mynn_psim_stable_full(z1);
    } else if (which == 1) {
        f0 = mynn_psih_stable_full(z0); f1 = mynn_psih_stable_full(z1);
    } else if (which == 2) {
        f0 = mynn_psim_unstable_full(z0); f1 = mynn_psim_unstable_full(z1);
    } else {
        f0 = mynn_psih_unstable_full(z0); f1 = mynn_psih_unstable_full(z1);
    }
    return f0 + fraction * (f1 - f0);
}

static inline float mynn_psim_stable(float z)
{ return mynn_table(z, 0); }
static inline float mynn_psih_stable(float z)
{ return mynn_table(z, 1); }
static inline float mynn_psim_unstable(float z)
{ return mynn_table(z, 2); }
static inline float mynn_psih_unstable(float z)
{ return mynn_table(z, 3); }

static inline float mynn_li_etal_2010(
    float rib, float zaz0, float z0zt)
{
    float zaz02 = fmin(fmax(zaz0, 100.0f), 100000.0f);
    float z0zt2 = fmin(fmax(z0zt, 0.5f), 100.0f);
    float alfa = mynn_sf_log(zaz02), beta = mynn_sf_log(z0zt2), zl;
    if (rib <= 0.0f) {
        zl = 0.045f * alfa * rib * rib
           + ((0.003f * beta + 0.0059f) * alfa * alfa
              + (-0.0828f * beta + 0.8845f) * alfa
              + (0.1739f * beta * beta - 0.9213f * beta - 0.1057f)) * rib;
        return fmin(fmax(zl, -15.0f), 0.0f);
    }
    if (rib <= 0.2f) {
        zl = ((0.5738f * beta - 0.4399f) * alfa
              + (-4.901f * beta + 52.50f)) * rib * rib
           + ((-0.0539f * beta + 1.540f) * alfa
              + (-0.669f * beta - 3.282f)) * rib;
        return fmin(fmax(zl, 0.0f), 4.0f);
    }
    zl = (0.7529f * alfa + 14.94f) * rib
       + 0.1569f * alfa - 0.3091f * beta - 1.303f;
    return fmin(fmax(zl, 1.0f), 20.0f);
}

static inline float mynn_zolrib(
    float ri, float za, float z0, float zt, float logz0, float logzt, float zol1)
{
    if (zol1 * ri < 0.0f) zol1 = 0.0f;
    float zolold = ri < 0.0f ? -99999.0f : 99999.0f;
    float result = ri < 0.0f ? -66666.0f : 66666.0f;
    int iteration = 1;
    while (fabs(zolold - result) > 0.01f && iteration < 20) {
        zolold = iteration == 1 ? zol1 : result;
        float zol20 = zolold * z0 / za;
        float zol3 = zolold + zol20;
        float zolt = zolold * zt / za;
        float psit2, psix2;
        if (ri < 0.0f) {
            psit2 = fmax(logzt
                - (mynn_psih_unstable(zol3) - mynn_psih_unstable(zolt)), 1.0f);
            psix2 = fmax(logz0
                - (mynn_psim_unstable(zol3) - mynn_psim_unstable(zol20)), 1.0f);
        } else {
            psit2 = fmax(logzt
                - (mynn_psih_stable(zol3) - mynn_psih_stable(zolt)), 1.0f);
            psix2 = fmax(logz0
                - (mynn_psim_stable(zol3) - mynn_psim_stable(zol20)), 1.0f);
        }
        result = ri * psix2 * psix2 / psit2;
        ++iteration;
    }
    if (iteration == 20 && fabs(zolold - result) > 0.01f)
        result = mynn_li_etal_2010(ri, za / z0, z0 / zt);
    return result;
}

static inline void mynn_zilitinkevich_land(
    float z0, float restar, thread float &zt, thread float &zq)
{
    zt = z0 * mynn_sf_exp(-0.4f * 0.085f * mynn_sf_sqrt(restar));
    zt = fmin(zt, 0.75f * z0);
    zq = zt;
}

static inline float mynn_charnock_1955(
    float ustar, float wsp, float visc, float zu)
{
    float wsp10 = wsp * mynn_sf_log(10.0f / 1.0e-4f) / mynn_sf_log(mynn_div_rn(zu, 1.0e-4f));
    float czc = 0.011f + 0.007f
        * fmin(fmax((wsp10 - 10.0f) / 8.0f, 0.0f), 1.0f);
    float z0 = mynn_div_rn(czc * ustar * ustar, 9.81f)
            + 0.11f * visc / fmax(ustar, 0.05f);
    return fmin(fmax(z0, 1.27e-7f), 2.85e-3f);
}

// module_sf_mynn.F:1281.  ISFTCFLX=1 and 2.  A function of u* alone: the
// Fortran takes no wind speed.
static inline float mynn_davis_etal_2008(float ustar)
{
    float zw = fmin(mynn_sf_pow(mynn_div_rn(ustar, 1.06f), 0.3f), 1.0f);
    float zn1 = mynn_div_rn(0.011f * ustar * ustar, 9.81f) + 1.59e-5f;
    float zn2 = 10.0f * mynn_sf_exp(-9.5f * mynn_sf_pow(ustar, -0.3333f))
             + 0.11f * 1.5e-5f / fmax(ustar, 0.01f);
    float z0 = (1.0f - zw) * zn1 + zw * zn2;
    return fmin(fmax(z0, 1.27e-7f), 2.85e-3f);
}

// module_sf_mynn.F:1311.  ISFTCFLX=3.  ustar is an argument of the Fortran
// subroutine and is never read in its body, so it is not taken here either.
static inline float mynn_taylor_yelland_2001(float wsp)
{
    float hs = 0.0248f * mynn_sf_pow(wsp, 2.0f);
    float tp = 0.729f * fmax(wsp, 0.1f);
    float lp = mynn_div_rn(9.81f * (tp * tp), (2.0f * 3.14159265f));
    float z0 = 1200.0f * hs * mynn_sf_pow(hs / lp, 4.5f);
    return fmin(fmax(z0, 1.27e-7f), 2.85e-3f);
}

// module_sf_mynn.F:1425.  COARE 3.0 zt/zq.  The Ren<=2 test at :1442 selects
// between two arms that compute the identical expression.  Zq is assigned
// from the already-clamped Zt (:1466-1467), so zq == zt.
static inline void mynn_fairall_etal_2003(
    float restar, thread float &zt, thread float &zq)
{
    zt = 5.5e-5f * mynn_sf_pow(restar, -0.60f);
    zt = fmin(fmax(zt, 2.0e-9f), 1.0e-4f);
    zq = zt;
}

// module_sf_mynn.F:1392.  ISFTCFLX=2 zt/zq.  This leaf makes its own
// landsea-1.5 .GT. 0 test, which is not the .GE. 0 at :625 that selected the
// water roughness, so a column at exactly XLAND=1.5 arrives here and takes
// the LAND arm.
static inline void mynn_garratt_1992(
    float z0, float restar, float xland, thread float &zt, thread float &zq)
{
    if (xland - 1.5f > 0.0f) {
        float quarter = mynn_sf_pow(restar, 0.25f);
        zt = z0 * mynn_sf_exp(2.0f - 2.48f * quarter);
        zq = z0 * mynn_sf_exp(2.0f - 2.28f * quarter);
        zq = fmax(fmin(zq, 5.5e-5f), 2.0e-9f);
        zt = fmax(fmin(zt, 5.5e-5f), 2.0e-9f);
    } else {
        zq = mynn_div_rn(z0, mynn_sf_pow(2.71828183f, 2.0f));
        zt = zq;
    }
}

// module_sf_mynn.F:631-710.  ISFTCFLX picks the over-water z0 and then, from
// the restar that new z0 implies (:675), the zt/zq pair.  COARE_OPT is a REAL
// PARAMETER fixed at 3.0 (:85), so edson_etal_2013 and fairall_etal_2014 are
// unreachable through SFCLAY1D_mynn and are deliberately absent from this
// kernel: they exist in the CPU reference only because the leaf oracle can
// call them at their own entry points.  ISFTCFLX=4 sets a z0 and then leaves
// z_t/z_q unassigned (:680-702 has no arm for it), so the host rejects it
// before launch and this switch has no arm for it either.
static inline void mynn_water_roughness(
    int isftcflx, float ustar, float wsp, float visc, float za, float xland,
    thread float &z0, thread float &restar, thread float &zt, thread float &zq)
{
    if (isftcflx == 1 || isftcflx == 2) z0 = mynn_davis_etal_2008(ustar);
    else if (isftcflx == 3) z0 = mynn_taylor_yelland_2001(wsp);
    else z0 = mynn_charnock_1955(ustar, wsp, visc, za);
    restar = fmax(ustar * z0 / visc, 0.1f);
    if (isftcflx == 2) mynn_garratt_1992(z0, restar, xland, zt, zq);
    else mynn_fairall_etal_2003(restar, zt, zq);
}

static inline void mynn_andreas_snow(
    float visc, float ustar, thread float &zt, thread float &zq)
{
    float zntsno = 0.135f * visc / ustar
        + mynn_div_rn(0.035f * ustar * ustar, 9.8f)
        * (5.0f * mynn_sf_exp(-mynn_sf_pow(mynn_div_rn((ustar - 0.18f), 0.1f), 2.0f)) + 1.0f);
    float ren = fmin(ustar * zntsno / visc, 1000.0f);
    float log_ren = mynn_sf_log(ren);
    float bt0, bt1, bt2, bq0, bq1, bq2;
    if (ren <= 0.135f) {
        bt0 = 1.25f; bt1 = 0.0f; bt2 = 0.0f;
        bq0 = 1.61f; bq1 = 0.0f; bq2 = 0.0f;
    } else if (ren < 2.5f) {
        bt0 = 0.149f; bt1 = -0.55f; bt2 = 0.0f;
        bq0 = 0.351f; bq1 = -0.628f; bq2 = 0.0f;
    } else {
        bt0 = 0.317f; bt1 = -0.565f; bt2 = -0.183f;
        bq0 = 0.396f; bq1 = -0.512f; bq2 = -0.180f;
    }
    zt = zntsno * mynn_sf_exp(bt0 + bt1 * log_ren + bt2 * log_ren * log_ren);
    zq = zntsno * mynn_sf_exp(bq0 + bq1 * log_ren + bq2 * log_ren * log_ren);
}


kernel void mynn_surface_column(
    constant MynnSurfaceColumnResources &resources [[buffer(0)]],
    constant MynnSurfaceColumnParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.n)) return;
    const int idx = int(tid);
    const device float *u1_a = resources.input_u1;
    const device float *v1_a = resources.input_v1;
    const device float *t1_a = resources.input_t1;
    const device float *qv1_a = resources.input_qv1;
    const device float *p1_a = resources.input_p1;
    const device float *rho1_a = resources.input_rho1;
    const device float *dz1_a = resources.input_dz1;
    const device float *u2_a = resources.input_u2;
    const device float *v2_a = resources.input_v2;
    const device float *dz2_a = resources.input_dz2;
    const device float *psfc_a = resources.input_psfc;
    const device float *tsk_a = resources.input_tsk;
    const device float *pblh_a = resources.input_pblh;
    const device float *mavail_a = resources.input_mavail;
    const device float *hfx_a = resources.input_hfx;
    const device float *qfx_a = resources.input_qfx;
    const device float *znt_a = resources.input_znt;
    const device float *qsfc_a = resources.input_qsfc;
    const device float *ust_a = resources.input_ust;
    const device float *xland_a = resources.input_xland;
    const device float *snowh_a = resources.input_snowh;
    const device float *mol_a = resources.input_mol;
    const device float *ustm_a = resources.input_ustm;
    device float *regime_o = resources.output_regime;
    device float *zol_o = resources.output_zol;
    device float *rmol_o = resources.output_rmol;
    device float *ust_o = resources.output_ust;
    device float *ustm_o = resources.output_ustm;
    device float *mol_o = resources.output_mol;
    device float *psim_o = resources.output_psim;
    device float *psih_o = resources.output_psih;
    device float *chs_o = resources.output_chs;
    device float *chs2_o = resources.output_chs2;
    device float *cqs2_o = resources.output_cqs2;
    device float *ch_o = resources.output_ch;
    device float *flhc_o = resources.output_flhc;
    device float *flqc_o = resources.output_flqc;
    device float *qgh_o = resources.output_qgh;
    device float *qsfc_o = resources.output_qsfc;
    device float *hfx_o = resources.output_hfx;
    device float *qfx_o = resources.output_qfx;
    device float *lh_o = resources.output_lh;
    device float *u10_o = resources.output_u10;
    device float *v10_o = resources.output_v10;
    device float *th2_o = resources.output_th2;
    device float *t2_o = resources.output_t2;
    device float *q2_o = resources.output_q2;
    device float *gz1oz0_o = resources.output_gz1oz0;
    device float *wspd_o = resources.output_wspd;
    device float *br_o = resources.output_br;
    device float *ck_o = resources.output_ck;
    device float *cka_o = resources.output_cka;
    device float *cd_o = resources.output_cd;
    device float *cda_o = resources.output_cda;
    device float *wstar_o = resources.output_wstar;
    device float *qstar_o = resources.output_qstar;
    device float *cpm_o = resources.output_cpm;
    device float *znt_o = resources.output_znt;
    const float dx = p.dx;
    const int itimestep = p.itimestep;
    const int isfflx = p.isfflx;
    const int isftcflx = p.isftcflx;
    const int n = p.n;
    const float cp = 1004.5f, grav = 9.81f, r = 287.0f, rv = 461.6f;
    const float rovcp = r / cp, xlv = 2.5e6f;
    const float svp1 = 0.6112f, svp2 = 17.67f, svp3 = 29.65f;
    const float svpt0 = 273.15f, ep1 = rv / r - 1.0f;
    const float ep2 = r / rv, ep3 = 1.0f - ep2;
    const float karman = 0.4f, prt = 1.0f, wmin = 0.1f, vconvc = 1.25f;

    float u1 = u1_a[idx], v1 = v1_a[idx], t1 = t1_a[idx];
    float qv1 = qv1_a[idx], p1 = p1_a[idx], rho1 = rho1_a[idx];
    float dz1 = dz1_a[idx], u2 = u2_a[idx], v2 = v2_a[idx];
    float dz2 = dz2_a[idx], psfcpa = psfc_a[idx], tsk = tsk_a[idx];
    float pblh = pblh_a[idx], mavail = mavail_a[idx];
    float hfx = hfx_a[idx], qfx = qfx_a[idx], z0 = znt_a[idx];
    float qsfc = qsfc_a[idx], ust = ust_a[idx], xland = xland_a[idx];
    float snowh = snowh_a[idx], mol = mol_a[idx], ustm = ustm_a[idx];

    float psfc = mynn_div_rn(psfcpa, 1000.0f);
    float thgb = tsk * mynn_sf_pow(100.0f / psfc, rovcp);
    float pl = mynn_div_rn(p1, 1000.0f);
    float th1 = t1 * mynn_sf_pow(100.0f / pl, rovcp);
    float tc1 = t1 - 273.15f;
    float qvsh = qv1 / (1.0f + qv1);
    // See the THVGB comment below: TVCON/THV1D are pinned to unfused
    // round-to-nearest so they contract identically to their surface twins.
    float tvcon = mynn_add_rn(1.0f, mynn_mul_rn(ep1, qvsh));
    float thv1 = mynn_mul_rn(th1, tvcon);
    float za = 0.5f * dz1;
    float za2 = dz1 + 0.5f * dz2;
    float govrth = grav / th1;

    float e1;
    if (tsk < 273.15f) {
        e1 = svp1 * mynn_sf_exp(4648.0f * (1.0f / 273.15f - 1.0f / tsk)
             - 11.64f * mynn_sf_log(273.15f / tsk) + 0.02265f * (273.15f - tsk));
    } else {
        e1 = svp1 * mynn_sf_exp(svp2 * (tsk - svpt0) / (tsk - svp3));
    }
    // module_sf_mynn.F:532 tests the INCOMING QSFC: a land column whose LSM
    // never set QSFC is recomputed from saturation here.
    float qsfcmr;
    if (xland > 1.5f || qsfc <= 0.0f) {
        qsfc = ep2 * e1 / (psfc - ep3 * e1);
        qsfcmr = ep2 * e1 / (psfc - e1);
    } else {
        qsfcmr = qsfc / (1.0f - qsfc);
    }

    if (tsk < 273.15f) {
        e1 = svp1 * mynn_sf_exp(4648.0f * (1.0f / 273.15f - 1.0f / t1)
             - 11.64f * mynn_sf_log(273.15f / t1) + 0.02265f * (273.15f - t1));
    } else {
        e1 = svp1 * mynn_sf_exp(svp2 * (t1 - svpt0) / (t1 - svp3));
    }
    float qgh = ep2 * e1 / (pl - e1);
    float cpm = cp * (1.0f + 0.84f * qv1);

    // module_sf_mynn.F:556 -- WSPD is SQRT(u*u+v*v), not hypot().
    float wsp = mynn_sf_sqrt(u1 * u1 + v1 * v1);
    // THV1D (:513) and THVGB (:559) are the same product of a temperature and
    // (1+EP1*q).  gfortran evaluates both without contraction, so an exactly
    // saturated neutral column cancels to zero.  NVRTC is free to contract one
    // and not the other; pin both to unfused round-to-nearest so the identity
    // holds for the same reason it holds in WRF instead of by special case.
    float tvcon_gb = mynn_add_rn(1.0f, mynn_mul_rn(ep1, qsfc));
    float thvgb = mynn_mul_rn(thgb, tvcon_gb);
    float dthvdz = thv1 - thvgb;
    float fluxc = fmax(mynn_div_rn(hfx / rho1, cp) + mynn_div_rn(ep1 * thvgb * qfx, rho1), 0.0f);
    // module_sf_mynn.F:573 spells the :532 predicate a second time, but it runs
    // after the :533 saturation update, so a land column that entered with
    // QSFC<=0 now takes the LAND height scale.
    bool wstar_water = xland > 1.5f || qsfc <= 0.0f;
    float height = wstar_water ? pblh : fmin(1.5f * pblh, 4000.0f);
    float wstar = vconvc * mynn_sf_pow(grav / tsk * height * fluxc, 0.33f);
    float vsgd = 0.32f * mynn_sf_pow(fmax(mynn_div_rn(dx, 5000.0f) - 1.0f, 0.0f), 0.33f);
    wsp = fmax(mynn_sf_sqrt(wsp * wsp + wstar * wstar + vsgd * vsgd), wmin);
    float br = govrth * za * dthvdz / (wsp * wsp);
    float limit = itimestep == 1 ? 2.0f : 4.0f;
    br = fmin(fmax(br, -limit), limit);

    float visc = 1.326e-5f * (1.0f + 6.542e-3f * tc1
              + 8.301e-6f * tc1 * tc1 - 4.84e-9f * tc1 * tc1 * tc1);
    float restar, zt, zq;
    if (xland >= 1.5f) {
        mynn_water_roughness(isftcflx, ust, wsp, visc, za, xland,
                             z0, restar, zt, zq);
    } else {
        restar = fmax(ust * z0 / visc, 0.1f);
        if (snowh >= 0.1f) mynn_andreas_snow(visc, ust, zt, zq);
        else mynn_zilitinkevich_land(z0, restar, zt, zq);
    }

    float zratio = z0 / zt;
    float gz1oz0 = mynn_sf_log((za + z0) / z0);
    float gz1ozt = mynn_sf_log((za + z0) / zt);
    float gz2ozt = mynn_sf_log((2.0f + z0) / zt);
    float gz10oz0 = mynn_sf_log((10.0f + z0) / z0);
    float gz10ozt = mynn_sf_log((10.0f + z0) / zt);

    float regime, zol, psim, psih, psim10, psih10, psih2;
    if (br > 0.0f) {
        regime = br > 0.2f ? 1.0f : 2.0f;
        if (itimestep <= 1) zol = mynn_li_etal_2010(br, za / z0, zratio);
        else {
            zol = za * karman * grav * mol / (th1 * fmax(ust * ust, 0.0001f));
            zol = fmin(fmax(zol, 0.0f), 20.0f);
        }
        zol = fmin(fmax(mynn_zolrib(br, za, z0, zt, gz1oz0, gz1ozt, zol),
                           0.0f), 20.0f);
        float zolzt = zol * zt / za, zolz0 = zol * z0 / za;
        float zolza = zol * (za + z0) / za;
        float zol10 = zol * (10.0f + z0) / za;
        float zol2 = zol * (2.0f + z0) / za;
        psim = mynn_psim_stable(zolza) - mynn_psim_stable(zolz0);
        psih = mynn_psih_stable(zolza) - mynn_psih_stable(zolzt);
        psim10 = mynn_psim_stable(zol10) - mynn_psim_stable(zolz0);
        psih10 = mynn_psih_stable(zol10) - mynn_psih_stable(zolz0);
        psih2 = mynn_psih_stable(zol2) - mynn_psih_stable(zolz0);
    } else if (br == 0.0f) {
        regime = 3.0f; zol = 0.0f;
        psim = 0.0f; psih = 0.0f; psim10 = 0.0f; psih10 = 0.0f; psih2 = 0.0f;
    } else {
        regime = 4.0f;
        if (itimestep <= 1) zol = mynn_li_etal_2010(br, za / z0, zratio);
        else {
            zol = za * karman * grav * mol / (th1 * fmax(ust * ust, 0.001f));
            zol = fmin(fmax(zol, -20.0f), 0.0f);
        }
        zol = fmin(fmax(mynn_zolrib(br, za, z0, zt, gz1oz0, gz1ozt, zol),
                           -20.0f), 0.0f);
        float zolzt = zol * zt / za, zolz0 = zol * z0 / za;
        float zolza = zol * (za + z0) / za;
        float zol10 = zol * (10.0f + z0) / za;
        float zol2 = zol * (2.0f + z0) / za;
        psim = mynn_psim_unstable(zolza) - mynn_psim_unstable(zolz0);
        psih = mynn_psih_unstable(zolza) - mynn_psih_unstable(zolzt);
        psim10 = mynn_psim_unstable(zol10) - mynn_psim_unstable(zolz0);
        psih10 = mynn_psih_unstable(zol10) - mynn_psih_unstable(zolz0);
        psih2 = mynn_psih_unstable(zol2) - mynn_psih_unstable(zolz0);
        psih = fmin(psih, 0.9f * gz1ozt);
        psim = fmin(psim, 0.9f * gz1oz0);
        psih2 = fmin(psih2, 0.9f * gz2ozt);
        psim10 = fmin(psim10, 0.9f * gz10oz0);
        psih10 = fmin(psih10, 0.9f * gz10ozt);
    }
    float rmol = zol / za;

    float psix = gz1oz0 - psim;
    float psix10 = gz10oz0 - psim10;
    ust = 0.5f * ust + 0.5f * karman * wsp / psix;
    // module_sf_mynn.F:953 -- WSPDI is SQRT(u*u+v*v), not hypot().
    float wspdi = fmax(mynn_sf_sqrt(u1 * u1 + v1 * v1), wmin);
    ustm = 0.5f * ustm + 0.5f * karman * wspdi / psix;
    if (xland < 1.5f) {
        ust = fmax(ust, 0.005f);
        ustm = ust;
    }

    gz1ozt = mynn_sf_log((za + z0) / zt);
    gz2ozt = mynn_sf_log((2.0f + z0) / zt);
    float psit = fmax(gz1ozt - psih, 1.0f);
    float psit2 = fmax(gz2ozt - psih2, 1.0f);
    float psiq = fmax(mynn_sf_log((za + z0) / zq) - psih, 1.0f);
    float psiq2 = fmax(mynn_sf_log((2.0f + z0) / zq) - psih2, 1.0f);
    float psiq10 = fmax(mynn_sf_log((10.0f + z0) / zq) - psih10, 1.0f);
    mol = karman * (thv1 - thvgb) / psit / prt;
    float qstar = karman * (qvsh - qsfc) * 1000.0f / psiq / prt;

    // WRF deliberately recomputes moisture resistance with zq in numerator.
    psiq = fmax(mynn_sf_log((za + zq) / zq) - psih, 1.0f);
    psiq2 = fmax(mynn_sf_log((2.0f + zq) / zq) - psih2, 1.0f);
    psiq10 = fmax(mynn_sf_log((10.0f + zq) / zq) - psih10, 1.0f);
    float flhc, flqc, lh, chs, ch, cqs2, chs2, ck, cka, cd, cda;
    if (isfflx < 1) {
        qfx = 0.0f; hfx = 0.0f; flhc = 0.0f; flqc = 0.0f; lh = 0.0f;
        chs = 0.0f; ch = 0.0f; chs2 = 0.0f; cqs2 = 0.0f;
        ck = 0.0f; cka = 0.0f; cd = 0.0f; cda = 0.0f;
    } else {
        flqc = rho1 * mavail * ust * karman / psiq;
        flhc = rho1 * cpm * ust * karman / psit;
        qfx = fmax(flqc * (qsfcmr - qv1), -0.02f);
        lh = xlv * qfx;
        // module_sf_mynn.F:1065-1076.  The arms test XLAND-1.5 for .GT. and
        // .LT. and nothing else, so a column at exactly XLAND=1.5 takes
        // neither and HFX keeps the value it entered with.
        if (xland > 1.5f) {
            hfx = flhc * (thgb - th1);
            // :1067-1072, the AHW dissipative-heating term -- the third and
            // last thing ISFTCFLX selects, water-only, and keyed on
            // ISFTCFLX.NE.0, so every non-default roughness identity adds it.
            // USTM is the u* computed without the vconv/vsgd gust (:953-956).
            if (isftcflx != 0) hfx = hfx + rho1 * ustm * ustm * wspdi;
        } else if (xland < 1.5f) {
            hfx = flhc * (thgb - th1);
            hfx = fmax(hfx, -250.0f);
        }
        chs = ust * karman / psit;
        ch = flhc / (cpm * rho1);
        cqs2 = ust * karman / psiq2;
        chs2 = ust * karman / psit2;
        ck = (karman / psix10) * (karman / psiq10);
        cd = (karman / psix10) * (karman / psix10);
        cka = (karman / psix) * (karman / psiq);
        cda = (karman / psix) * (karman / psix);
    }

    float u10, v10;
    if (za <= 7.0f) {
        if (za2 > 7.0f && za2 < 13.0f) { u10 = u2; v10 = v2; }
        else {
            float ratio = mynn_sf_log(10.0f / z0) / mynn_sf_log(za / z0);
            u10 = u1 * ratio; v10 = v1 * ratio;
        }
    } else if (za < 13.0f) {
        float ratio = mynn_sf_log(10.0f / z0) / mynn_sf_log(za / z0);
        u10 = u1 * ratio; v10 = v1 * ratio;
    } else {
        u10 = u1 * psix10 / psix; v10 = v1 * psix10 / psix;
    }

    float th2 = thgb + (th1 - thgb) * psit2 / psit;
    if ((th1 > thgb && !(th2 >= thgb && th2 <= th1))
        || (th1 < thgb && !(th2 >= th1 && th2 <= thgb)))
        th2 = thgb + 2.0f * (th1 - thgb) / za;
    float t2 = th2 * mynn_sf_pow(mynn_div_rn(psfc, 100.0f), rovcp);
    float q2 = qsfcmr + (qv1 - qsfcmr) * psiq2 / psiq;
    q2 = fmax(q2, fmin(qsfcmr, qv1));
    q2 = fmin(q2, 1.05f * qv1);

    regime_o[idx] = regime; zol_o[idx] = zol; rmol_o[idx] = rmol;
    ust_o[idx] = ust; ustm_o[idx] = ustm; mol_o[idx] = mol;
    psim_o[idx] = psim; psih_o[idx] = psih; chs_o[idx] = chs;
    chs2_o[idx] = chs2; cqs2_o[idx] = cqs2; ch_o[idx] = ch;
    flhc_o[idx] = flhc; flqc_o[idx] = flqc; qgh_o[idx] = qgh;
    qsfc_o[idx] = qsfc; hfx_o[idx] = hfx; qfx_o[idx] = qfx;
    lh_o[idx] = lh; u10_o[idx] = u10; v10_o[idx] = v10;
    th2_o[idx] = th2; t2_o[idx] = t2; q2_o[idx] = q2;
    gz1oz0_o[idx] = gz1oz0; wspd_o[idx] = wsp; br_o[idx] = br;
    ck_o[idx] = ck; cka_o[idx] = cka; cd_o[idx] = cd; cda_o[idx] = cda;
    wstar_o[idx] = wstar; qstar_o[idx] = qstar; cpm_o[idx] = cpm;
    // module_sf_mynn.F:436 declares ZNT INTENT(INOUT); :657 (charnock_1955)
    // mutates it in place over water and the updated value persists into the
    // next step's ZNTstoch/restar/z_t/z_q chain.
    znt_o[idx] = z0;
}
