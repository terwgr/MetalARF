#include <metal_stdlib>
#include "../generated/ysu_column_480_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_pow.metalh"
using namespace metal;

// WRF 4.8 YSU generation (ysu_version="wrf_480"): ysu_480.cu and
// ysu_480_topo.cuh, transcribed statement for statement with the CUDA
// module's own arithmetic -- no multiply-add contraction, correctly rounded
// divide and square root, and glibc's powf/expf through the shared software
// gfk routines. The kernel keeps the CUDA entry points' shape: every column
// takes the ctopo arm (ctopo = 1 unless topo_wind supplies it), and the BEP
// arm is a runtime switch.
#pragma clang fp contract(off)

typedef float real;
#define G 9.81f
#define RD 287.0f
#define RV 461.6f
#define CP 1004.5f
#define RCP 0.2857142857142857f
#define XLV 2500000.0f
#define EP2 0.6217504332755632f
#define YSU_KMAX 128
#define YSUWS_SLOTS 18
#define YSUWS_LANES 32

struct YsuCol {
    device float *p;
    device float &operator[](int k) const { return p[k * YSUWS_LANES]; }
};
struct YsuBep {
    device const float *a_u, *a_v, *a_t, *a_q, *b_u, *b_v, *b_t, *b_q, *sf, *vl;
};
struct YsuTopo {
    device const float *ctopo, *ctopo2;
    device float *u10o, *v10o;
};

static inline float __fdiv_rn(float a, float b) { return a / b; }
static inline float __fmul_rn(float a, float b) { return a * b; }
static inline float __fadd_rn(float a, float b) { return a + b; }
static inline float __fsub_rn(float a, float b) { return a - b; }
// The GPU flushes binary32 subnormals; the tridiagonal solves and the
// tendencies they feed carry them (a cloud-water profile decaying away from a
// layer ends in subnormal words), so those use the integer-bit operations.
static inline float ysm(float a, float b)
{ return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float yss(float a, float b)
{ return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float __uint_as_float(uint x) { return as_type<float>(x); }
static inline uint __float_as_uint(float x) { return as_type<uint>(x); }
static inline float __int_as_float(int x) { return as_type<float>(x); }
static inline float fabsf(float x) { return fabs(x); }
static inline float fminf(float a, float b) { return fmin(a, b); }
static inline float fmaxf(float a, float b) { return fmax(a, b); }
static inline float sqrtf(float x)
{ return as_type<float>(soft32_sqrt(as_type<uint>(x))); }
static inline float gfk_exp(float x)
{ return as_type<float>(soft_gfk_exp(as_type<uint>(x))); }
static inline float gfk_pow(float x, float y)
{ return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x), as_type<uint>(y))); }

static inline float ysu_min(float a, float b)
{ return (a != a) ? a : (a < b ? a : b); }
static inline float ysu_max(float a, float b)
{ return (a != a) ? a : (a > b ? a : b); }
static inline float ysu_clip(float x, float lo, float hi) {
    return ysu_min(ysu_max(x, lo), hi);
}
static inline float ysu_pos(float x, float fallback) {
    return (x > 0.0f || x != x) ? x : fallback;
}
static inline float ysu_thx(device const float *theta, device const float *exner,
                            int q) {
    return __fdiv_rn(__fmul_rn(theta[q], exner[q]), exner[q]);
}

// bl_ysu.F90:674/:948 prfac2 where the float32 quotient is Inf/NaN: the same
// expression in binary64 on the shared integer-bit implementation.
static inline float yprfac2_rescue(float us, float wstar3, float wstar3_2, float karman) {
    const ulong c159 = 0x402fcccccccccccdul, c4 = 0x4010000000000000ul;
    const ulong one = 0x3ff0000000000000ul;
    ulong u = soft64_from_f32_bits(as_type<uint>(us));
    ulong u3 = soft64_mul(soft64_mul(u, u), u);
    ulong w = soft64_add(soft64_from_f32_bits(as_type<uint>(wstar3)),
                         soft64_from_f32_bits(as_type<uint>(wstar3_2)));
    if (soft64_greater(u3, 0ul)) {
        ulong k = soft64_from_f32_bits(as_type<uint>(karman));
        ulong numer = soft64_div(soft64_mul(c159, w), u3);
        ulong denom = soft64_add(one, soft64_div(soft64_mul(soft64_mul(c4, k), w), u3));
        return as_type<float>(soft64_to_f32_bits(soft64_div(numer, denom)));
    }
    if (soft64_greater(w, 0ul)) {
        ulong k = soft64_from_f32_bits(as_type<uint>(karman));
        return as_type<float>(soft64_to_f32_bits(soft64_div(c159, soft64_mul(c4, k))));
    }
    return 0.0f;
}

real ysu_scale_exponent(real y, int k)
{
    return __uint_as_float(__float_as_uint(y) + ((unsigned)k << 23));
}

real ysu_expm1f(real x)
{
    const real ln2_hi = __uint_as_float(0x3F317180u);
    const real ln2_lo = __uint_as_float(0x3717F7D1u);
    const real invln2 = __uint_as_float(0x3FB8AA3Bu);
    const real q1 = __uint_as_float(0xBD088889u);
    const real q2 = __uint_as_float(0x3AD00D01u);
    const real q3 = __uint_as_float(0xB8A670CDu);
    const real q4 = __uint_as_float(0x36867E54u);
    const real q5 = __uint_as_float(0xB457EDBBu);
    const real tiny = 1.0e-30f;

    unsigned word = __float_as_uint(x);
    unsigned sign = word & 0x80000000u;
    unsigned magnitude = word & 0x7FFFFFFFu;
    if (magnitude >= 0x4195B844u) {              // |x| >= 27*ln2
        if (magnitude >= 0x42B17218u) {          // |x| >= 88.72
            if (magnitude > 0x7F800000u) return __fadd_rn(x, x);
            if (magnitude == 0x7F800000u) return sign == 0u ? x : -1.0f;
            if (x > 8.8721679688e01f) return __int_as_float(0x7F800000);
        }
        if (sign != 0u) return __fsub_rn(tiny, 1.0f);
    }
    int k;
    real correction;
    if (magnitude > 0x3EB17218u) {               // |x| > 0.5*ln2
        real hi, lo;
        if (magnitude < 0x3F851592u) {           // |x| < 1.5*ln2
            if (sign == 0u) {
                hi = __fsub_rn(x, ln2_hi); lo = ln2_lo; k = 1;
            } else {
                hi = __fadd_rn(x, ln2_hi); lo = -ln2_lo; k = -1;
            }
        } else {
            k = (int)__fadd_rn(__fmul_rn(invln2, x), sign == 0u ? 0.5f : -0.5f);
            real scale = (real)k;
            hi = __fsub_rn(x, __fmul_rn(scale, ln2_hi));
            lo = __fmul_rn(scale, ln2_lo);
        }
        x = __fsub_rn(hi, lo);
        correction = __fsub_rn(__fsub_rn(hi, x), lo);
    } else if (magnitude < 0x33000000u) {        // |x| < 2**-25
        return x;
    } else {
        k = 0;
        correction = 0.0f;
    }

    real hfx = __fmul_rn(0.5f, x);
    real hxs = __fmul_rn(x, hfx);
    real r1 = __fadd_rn(1.0f, __fmul_rn(hxs, __fadd_rn(q1, __fmul_rn(hxs,
        __fadd_rn(q2, __fmul_rn(hxs, __fadd_rn(q3, __fmul_rn(hxs,
            __fadd_rn(q4, __fmul_rn(hxs, q5))))))))));
    real t = __fsub_rn(3.0f, __fmul_rn(r1, hfx));
    real e = __fmul_rn(hxs, __fdiv_rn(__fsub_rn(r1, t),
                                      __fsub_rn(6.0f, __fmul_rn(x, t))));
    if (k == 0) return __fsub_rn(x, __fsub_rn(__fmul_rn(x, e), hxs));
    e = __fsub_rn(__fmul_rn(x, __fsub_rn(e, correction)), correction);
    e = __fsub_rn(e, hxs);
    if (k == -1) return __fsub_rn(__fmul_rn(0.5f, __fsub_rn(x, e)), 0.5f);
    if (k == 1) {
        if (x < -0.25f)
            return __fmul_rn(-2.0f, __fsub_rn(e, __fadd_rn(x, 0.5f)));
        return __fadd_rn(1.0f, __fmul_rn(2.0f, __fsub_rn(x, e)));
    }
    real y;
    if (k <= -2 || k > 56) {
        y = __fsub_rn(1.0f, __fsub_rn(e, x));
        y = ysu_scale_exponent(y, k);
        return __fsub_rn(y, 1.0f);
    }
    if (k < 23) {
        t = __uint_as_float(0x3F800000u - (0x1000000u >> k));
        y = __fsub_rn(t, __fsub_rn(e, x));
    } else {
        t = __uint_as_float((unsigned)(0x7F - k) << 23);
        y = __fsub_rn(x, __fadd_rn(e, t));
        y = __fadd_rn(y, 1.0f);
    }
    return ysu_scale_exponent(y, k);
}

// 1 - 2/(expm1f(2|x|)+2) at |x| >= 1, -t/(t+2) below, saturated at 22.
real ysu_tanhf(real x)
{
    const real tiny = 1.0e-30f;
    unsigned word = __float_as_uint(x);
    unsigned magnitude = word & 0x7FFFFFFFu;
    real z;
    if (magnitude >= 0x7F800000u)                // inf or NaN: one/x +- one
        return (word & 0x80000000u) == 0u
            ? __fadd_rn(__fdiv_rn(1.0f, x), 1.0f)
            : __fsub_rn(__fdiv_rn(1.0f, x), 1.0f);
    if (magnitude < 0x41B00000u) {               // |x| < 22
        if (magnitude < 0x24000000u)             // |x| < 2**-55
            return __fmul_rn(x, __fadd_rn(1.0f, x));
        real ax = __uint_as_float(magnitude);
        if (magnitude >= 0x3F800000u) {          // |x| >= 1
            real t = ysu_expm1f(__fmul_rn(2.0f, ax));
            z = __fsub_rn(1.0f, __fdiv_rn(2.0f, __fadd_rn(t, 2.0f)));
        } else {
            real t = ysu_expm1f(__fmul_rn(-2.0f, ax));
            z = __fdiv_rn(-t, __fadd_rn(t, 2.0f));
        }
    } else {
        z = __fsub_rn(1.0f, tiny);
    }
    return (word & 0x80000000u) == 0u ? z : -z;
}

// bl_ysu.F90 get_pblh (:1586-1693, "Copied from MYNN PBL"), one column, on
// the kernel's 0-based workspace: thv[k] is thetav1d(k+1), tke[k] is
// qke1d(k+1) (written for k <= nz-2, which is all WRF writes and all the
// two searches below can reach), zq[k] is zw1d(k+1), and dz1d(k) is WRF's
// dzq(i,k) = zq(i,k+1)-zq(i,k) (:522), formed here from zq the same way.
real ysu_get_pblh(YsuCol thv, YsuCol tke, YsuCol zq, int nz, real landsea) {
    const real sbl_lim = 200.0f, sbl_damp = 400.0f;
    // Fortran index helpers (1-based, kts = 1, kte = nz).
    #define YT_THV(k) thv[(k) - 1]
    #define YT_QKE(k) tke[(k) - 1]
    #define YT_ZW(k) zq[(k) - 1]
    #define YT_DZ(k) __fsub_rn(zq[(k)], zq[(k) - 1])
    const int kte = nz, kts = 1;
    int k = kts + 1, kthv = 1, ktke = 1;
    real maxqke = 0.0f, minthv = 9.e9f;
    while (k <= kte && YT_ZW(k) <= 500.0f) {
        real qtke = fmaxf(YT_QKE(k), 0.0f);
        if (maxqke < qtke) { maxqke = qtke; ktke = k; }
        if (minthv > YT_THV(k)) { minthv = YT_THV(k); kthv = k; }
        k = k + 1;
    }
    real tkeeps = __fdiv_rn(maxqke, 40.0f);
    tkeeps = fmaxf(tkeeps, 0.025f);
    tkeeps = fminf(tkeeps, 0.25f);
    real delt_thv = (__fsub_rn(landsea, 1.5f) >= 0.0f) ? 0.75f : 1.5f;
    real zi = 0.0f;
    k = kthv + 1;
    real thr = __fadd_rn(minthv, delt_thv);
    while (zi == 0.0f && k <= kte) {
        if (YT_THV(k) >= thr) {
            real frac = fminf(__fdiv_rn(__fsub_rn(YT_THV(k), thr),
                                        fmaxf(__fsub_rn(YT_THV(k), YT_THV(k - 1)),
                                              1e-6f)), 1.0f);
            zi = __fsub_rn(YT_ZW(k), __fmul_rn(YT_DZ(k - 1), frac));
        }
        k = k + 1;
        if (k == kte - 1) zi = YT_ZW(kts + 1);
    }
    real pblh_tke = 0.0f;
    k = ktke + 1;
    while (pblh_tke == 0.0f && k <= kte) {
        real qtke = fmaxf(__fdiv_rn(YT_QKE(k), 2.0f), 0.0f);
        real qtkem1 = fmaxf(__fdiv_rn(YT_QKE(k - 1), 2.0f), 0.0f);
        if (qtke <= tkeeps) {
            real frac = fminf(__fdiv_rn(__fsub_rn(tkeeps, qtke),
                                        fmaxf(__fsub_rn(qtkem1, qtke), 1e-6f)),
                              1.0f);
            pblh_tke = __fsub_rn(YT_ZW(k), __fmul_rn(YT_DZ(k - 1), frac));
            pblh_tke = fmaxf(pblh_tke, YT_ZW(kts + 1));
        }
        k = k + 1;
        if (k == kte - 1) pblh_tke = YT_ZW(kts + 1);
    }
    real th = ysu_tanhf(__fdiv_rn(__fsub_rn(zi, sbl_lim), sbl_damp));
    real wt = __fadd_rn(__fmul_rn(0.5f, th), 0.5f);
    #undef YT_THV
    #undef YT_QKE
    #undef YT_ZW
    #undef YT_DZ
    return __fadd_rn(__fmul_rn(pblh_tke, __fsub_rn(1.0f, wt)),
                     __fmul_rn(zi, wt));
}



// The Thomas recurrence in register-carried form.  Every operation and both
// of its operands are the ones the plain form uses:
//     gamma[k] = upper[k] / (diag[k] - lower[k]*gamma[k-1])   (as * inv)
//     rhs[k]   = (rhs[k] - lower[k]*rhs[k-1]) * inv
//     rhs[k]  -= gamma[k]*rhs[k+1]                             (back pass)
// Only where the operands come from changes: gamma[k-1], rhs[k-1] and
// rhs[k+1] are the words this thread has just stored, held in registers
// instead of reloaded from the workspace, and the next level's
// coefficients are loaded before this level's stores, so those loads
// overlap the divide instead of waiting behind it.  The workspace ends
// with the same words.  Needs nz >= 2 (YSU's vertical bounds ask for 4).
void ysu_thomas(YsuCol lower, YsuCol diag,
                           YsuCol upper, YsuCol rhs, YsuCol gamma, int nz) {
    real inv = 1.0f / diag[0];
    real g = ysm(upper[0], inv);
    gamma[0] = g;
    real r = ysm(rhs[0], inv);
    rhs[0] = r;
    for (int k = 1; k < nz - 1; ++k) {
        inv = 1.0f / yss(diag[k], ysm(lower[k], g));
        g = ysm(upper[k], inv);
        gamma[k] = g;
        r = ysm(yss(rhs[k], ysm(lower[k], r)), inv);
        rhs[k] = r;
    }
    inv = 1.0f / yss(diag[nz - 1], ysm(lower[nz - 1], g));
    r = ysm(yss(rhs[nz - 1], ysm(lower[nz - 1], r)), inv);
    rhs[nz - 1] = r;
    for (int k = nz - 2; k >= 0; --k) {
        r = yss(rhs[k], ysm(gamma[k], r));
        rhs[k] = r;
    }
}
void ysu_diagnose(device const float *u, device const float *v, YsuCol thv,
                             YsuCol za, real thermal, real br0,
                             real brcrit, int nz, int st, int col,
                             thread float *hpbl, thread int *kpbl,
                             thread float *brdn_out, thread float *brup_out) {
    real brup = br0, brdn = brup;
    int kp = 1;
    bool crossed = false;
    for (int k = 1; k < nz; ++k) {
        if (!crossed) {
            brdn = brup;
            int q = k * st + col;
            real spdk2 = ysu_max(u[q] * u[q] + v[q] * v[q], 1.0f);
            brup = (thv[k] - thermal) * (G * za[k] / thv[0]) / spdk2;
            kp = k + 1;
            crossed = brup > brcrit;
        }
    }
    real frac;
    if (brdn >= brcrit) frac = 0.0f;
    else if (brup <= brcrit) frac = 1.0f;
    else frac = (brcrit - brdn) / (brup - brdn);
    int kh = kp - 1;
    *hpbl = za[kh - 1] + frac * (za[kh] - za[kh - 1]);
    *kpbl = kp;
    *brdn_out = brdn;
    *brup_out = brup;
}
// bl_ysu.F90 tridi2n's f2 arm, exactly as WRF solves it: the v column
// pivots on its OWN diagonal cm1 (ad1 = ad - a_v*dt2) but eliminates and
// back-substitutes with au, the factors the u column's diagonal cm produced.
// That is only the true v solve when cm1 == cm, i.e. always outside BEP --
// which is why the plain kernel can re-run ysu_thomas for v -- and it is
// what WRF does whenever a_v_bep differs from a_u_bep.  `gamma_u` must be
// the u solve's gamma, untouched since.
void ysu_tridi2n_v(YsuCol lower, YsuCol diag_v,
                              YsuCol gamma_u, YsuCol rhs, int nz) {
    real inv = 1.0f / diag_v[0];
    rhs[0] = ysm(rhs[0], inv);
    for (int k = 1; k < nz - 1; ++k) {
        inv = 1.0f / yss(diag_v[k], ysm(lower[k], gamma_u[k - 1]));
        rhs[k] = ysm(yss(rhs[k], ysm(lower[k], rhs[k - 1])), inv);
    }
    inv = 1.0f / yss(diag_v[nz - 1], ysm(lower[nz - 1], gamma_u[nz - 2]));
    rhs[nz - 1] = ysm(yss(rhs[nz - 1], ysm(lower[nz - 1], rhs[nz - 2])), inv);
    for (int k = nz - 2; k >= 0; --k)
        rhs[k] = yss(rhs[k], ysm(gamma_u[k], rhs[k + 1]));
}
static inline void ysu_topo_blend_u10(
        bool present, YsuTopo topo, device const float *u10,
        device const float *v10, float u1, float v1, int col) {
    if (!present) return;
    float c2 = topo.ctopo2[col];
    topo.u10o[col] = __fadd_rn(__fmul_rn(c2, u10[col]),
                               __fmul_rn(__fsub_rn(1.0f, c2), u1));
    topo.v10o[col] = __fadd_rn(__fmul_rn(c2, v10[col]),
                               __fmul_rn(__fsub_rn(1.0f, c2), v1));
}

kernel void ysu_column_480(
    constant YsuColumn480Resources &r [[buffer(0)]],
    constant YsuColumn480Params &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nz = prm.nz, ny = prm.ny, nx = prm.nx, st = ny * nx;
    const uint required_ws = ((uint(prm.tilecol) + YSUWS_LANES - 1u) / YSUWS_LANES)
                           * YSUWS_SLOTS * uint(prm.wskp) * YSUWS_LANES;
    if (tid >= uint(prm.tilecol) || prm.ncol != st || nz < 4 || nz > YSU_KMAX ||
        prm.wskp < nz + 1 || prm.ws_count <= 0 || uint(prm.ws_count) < required_ws)
        return;
    const int col = prm.col0 + int(tid);
    if (col >= st) return;
    const uint block = tid / YSUWS_LANES, lane = tid % YSUWS_LANES;
    device float *wsb = r.ws + ulong(block) * YSUWS_SLOTS * uint(prm.wskp) * YSUWS_LANES;
#define YSUWS_SLOT(i) YsuCol{wsb + (uint(i) * uint(prm.wskp)) * YSUWS_LANES + lane}
    YsuCol thv = YSUWS_SLOT(0), thli = YSUWS_SLOT(1), zq = YSUWS_SLOT(2);
    YsuCol za = YSUWS_SLOT(3), dza = YSUWS_SLOT(4), delp = YSUWS_SLOT(5);
    YsuCol xkzm = YSUWS_SLOT(6), xkzh = YSUWS_SLOT(7), xkzq = YSUWS_SLOT(8);
    YsuCol xkzml = YSUWS_SLOT(9), xkzhl = YSUWS_SLOT(10), zfacent = YSUWS_SLOT(11);
    YsuCol entfac = YSUWS_SLOT(12), lower = YSUWS_SLOT(13), diag = YSUWS_SLOT(14);
    YsuCol upper = YSUWS_SLOT(15), rhs = YSUWS_SLOT(16), gamma = YSUWS_SLOT(17);
#undef YSUWS_SLOT
    device const float *u = r.u, *v = r.v, *theta = r.theta, *qv = r.qv;
    device const float *qc = r.qc, *qi = r.qi, *p = r.p;
    device const float *p_interface = r.p_interface, *exner = r.exner, *dz = r.dz;
    device const float *rthraten = r.rthraten, *psfc = r.psfc, *znt = r.znt;
    device const float *ust = r.ust, *hfx = r.hfx, *qfx = r.qfx, *wspd = r.wspd;
    device const float *br = r.br, *psim = r.psim, *psih = r.psih;
    device const float *xland = r.xland, *u10 = r.u10, *v10 = r.v10;
    device float *du = r.du, *dv = r.dv, *dtheta = r.dtheta, *dqv = r.dqv;
    device float *dqc = r.dqc, *dqi = r.dqi, *hpbl_out = r.hpbl;
    device int *kpbl_out = r.kpbl;
    device float *exch_h = r.exch_h, *exch_m = r.exch_m, *wstar_out = r.wstar;
    device float *delta_out = r.delta, *topdown_radsum_out = r.topdown_radsum;
    device float *wstar3_2_out = r.wstar3_2;
    device int *cloudflg_out = r.cloudflg;
    const float dt = prm.dt;
    const int ysu_topdown_pblmix = prm.ysu_topdown_pblmix;
    const bool BEP = prm.bep != 0, TOPO = true;
    const YsuBep bep{r.a_u_bep, r.a_v_bep, r.a_t_bep, r.a_q_bep, r.b_u_bep,
                     r.b_v_bep, r.b_t_bep, r.b_q_bep, r.sf_bep, r.vl_bep};
    const YsuTopo topo{r.ctopo, r.ctopo2, r.u10o, r.v10o};

    real us = ust[col], hf = hfx[col], qf = qfx[col];
    // No zero-coupling short circuit.  Until lane/verify-ysu this kernel
    // returned zero tendencies, PBLH = dz(1) and KPBL = 1 whenever UST, HFX
    // and QFX were all zero, and -ftz fired the same branch on a subnormal
    // UST.  bl_ysu_run has no such branch: on a stable column it computes a
    // PBL (900 m and ten levels on the wide oracle's probe), and the kernel
    // now runs that column as WRF does.  Where WRF itself divides 0 by 0
    // (sfcflg true, ust**3 == 0: prfac2 below) the kernel keeps its defined
    // value; every other quantity on such a column is finite here because
    // wspd is guarded only where WRF would divide by zero (ysu_pos).
    // Graded by tests/test_ysu_wrf461_wide_parity.py (zero-coupling
    // probes) against tools/ysu_wrf461_oracle/run_bl_ysu_wide.F90.
    //
    //

    const real xkzminm = 0.1f, xkzminh = 0.01f, xkzmax = 1000.0f;
    const real rimin = -100.0f, rlam = 30.0f, prmin = 0.25f, prmax = 4.0f;
    const real brcr_ub = 0.0f, brcr_sb = 0.25f, cori = 1.0e-4f;
    const real afac = 6.8f, bfac = 6.8f, phifac = 8.0f, sfcfrac = 0.1f;
    const real d1 = 0.02f, d2 = 0.05f, d3 = 0.001f;
    const real h1 = 0.333333333333f, h2 = 0.666666666667f;
    const real zfmin = 1.0e-8f, aphi5 = 5.0f, aphi16 = 16.0f;
    const real tmin = 0.01f, gamcrt = 3.0f, gamcrq = 0.002f;
    // ep1 is WRF's EP_1, and WRF forms it in float32: module_model_constants
    // declares `REAL, PARAMETER :: EP_1 = R_v/R_d - 1.` so the quotient is a
    // float32 divide of two float32 values.  CUDA_DEFINES["RVOVRD"] is RV/RD
    // computed in Python doubles and rounded once to float32, which lands 1
    // ULP below WRF's quotient at 1.608 and therefore 2 ULP below it here at
    // 0.608.  Spelling the divide keeps the whole thv/Richardson/PBL-diagnosis
    // chain on WRF's constant; measured worth on the oracle fixture: hpbl
    // 112 -> 1 ULP, exch_h 283 -> 7, exch_m 48 -> 7, dv 46604 -> 23302.
    const real karman = 0.4f, ep1 = RV / RD - 1.0f;
    real dt2 = 2.0f * dt, rdt = 1.0f / dt2;

    // One pass instead of three: zq[k], za[k-1] and the interface pressure
    // ride in registers to the next level.  Each word is the same operation
    // on the same operands as the three separate loops formed:
    //   zq[k+1] = zq[k] + dz,  za[k] = 0.5*(zq[k] + zq[k+1]),
    //   dza[0] = za[0],  dza[k] = za[k] - za[k-1].
    zq[0] = 0.0f;
    real zq_k = 0.0f, za_prev = 0.0f, pi_k = p_interface[col];
    for (int k = 0; k < nz; ++k) {
        int q = k * st + col;
        thv[k] = ysu_thx(theta, exner, q) * (1.0f + ep1 * qv[q]);
        thli[k] = (theta[q] * exner[q] - __fdiv_rn(XLV * qc[q], CP)
                   - __fdiv_rn(2.834e6f * qi[q], CP)) / exner[q];
        real zq_k1 = zq_k + dz[q];
        zq[k + 1] = zq_k1;
        real pi_k1 = p_interface[(k + 1) * st + col];
        delp[k] = pi_k - pi_k1;
        real za_k = 0.5f * (zq_k + zq_k1);
        za[k] = za_k;
        dza[k] = k == 0 ? za_k : za_k - za_prev;
        zq_k = zq_k1;
        za_prev = za_k;
        pi_k = pi_k1;
    }

    real theta0 = ysu_thx(theta, exner, col), qv0 = qv[col];
    real temp0 = theta[col] * exner[col];
    real rho = psfc[col] / (RD * temp0 * (1.0f + ep1 * qv0));
    real govrth = G / theta0;
    real u0 = u[col], v0 = v[col];
    real wspd1 = sqrtf(u0 * u0 + v0 * v0) + 1.0e-9f;
    real sflux = __fdiv_rn(hf / rho, CP) + qf / rho * ep1 * theta0;
    real thermal = thv[0], thermalli = thli[0];
    real hpbl;
    int kpbl;
    real brdn, brup;
    ysu_diagnose(u, v, thv, za, thermal, br[col], brcr_ub, nz, st, col,
                 &hpbl, &kpbl, &brdn, &brup);
    if (hpbl < zq[1]) kpbl = 1;
    bool pblflg = kpbl > 1;
    bool sfcflg = !(br[col] > 0.0f);  // :613 in double; ysu_topo.cuh
    real zol1 = ysu_max(br[col] * psim[col] * psim[col] / psih[col], rimin);
    zol1 = sfcflg ? ysu_min(zol1, -zfmin) : ysu_max(zol1, zfmin);
    real hol1 = zol1 * hpbl / za[0] * sfcfrac;
    real phim, phih, wstar3, wstar;
    if (sfcflg) {
        phim = gfk_pow(1.0f - aphi16 * hol1, -0.25f);
        phih = gfk_pow(1.0f - aphi16 * hol1, -0.5f);
        wstar3 = govrth * ysu_max(sflux, 0.0f) * hpbl;
        wstar = gfk_pow(wstar3, h1);
    } else {
        phim = phih = 1.0f + aphi5 * hol1;
        wstar3 = wstar = 0.0f;
    }
    real ust3 = gfk_pow(us, 3.0f);
    real wscale = gfk_pow(ust3 + phifac * karman * wstar3 * 0.5f, h1);
    wscale = ysu_min(wscale, us * aphi16);
    wscale = ysu_max(wscale, __fdiv_rn(us, aphi5));
    real hgamt = 0.0f, hgamq = 0.0f, hgamu = 0.0f, hgamv = 0.0f;
    if (sfcflg && sflux > 0.0f) {
        real gamfac = bfac / rho / wscale;
        hgamt = ysu_min(__fdiv_rn(gamfac * hf, CP), gamcrt);
        hgamq = ysu_min(gamfac * qf, gamcrq);
        real vpert = __fdiv_rn((hgamt + ep1 * theta0 * hgamq), bfac) * afac;
        thermal += ysu_max(vpert, 0.0f)
                 * ysu_min(za[0] / (sfcfrac * hpbl), 1.0f);
        thermalli += ysu_max(vpert, 0.0f)
                   * ysu_min(za[0] / (sfcfrac * hpbl), 1.0f);
        hgamt = ysu_max(hgamt, 0.0f);
        hgamq = ysu_max(hgamq, 0.0f);
        real cg = -15.9f * us * us / ysu_pos(wspd[col], 1.0e-9f) * wstar3
                / ysu_pos(gfk_pow(wscale, 4.0f), 1.0e-20f);
        hgamu = cg * u0;
        hgamv = cg * v0;
        // bl_ysu.F90:703-728 guards all three thermal-enhanced statements
        // with if(pblflg(i)), and :684-698 can only LOWER pblflg -- WRF has
        // no path that raises it here.  A column whose FIRST guess sat
        // below zq(i,2) (:646-647) therefore keeps kpbl=1 and stays in the
        // local-K regime for the whole step, however far the thermal excess
        // could have pushed the enhanced sweep.
        //
        // The sweep is the WHOLE of :703-728: it leaves hpbl at the
        // zq(i,1) of :706 and never touches pblflg.  WRF interpolates
        // hpbl and applies the zq(i,2) clamp exactly once, at :754-768,
        // AFTER the theta-li scan -- so the scan below must see this
        // sweep's kpbl >= 2 and an unchanged pblflg.  ysu_diagnose folds
        // the interpolation in on its way out; that value is dead for
        // this call, because the post-scan if(pblflg) block recomputes it
        // from the same brdn/brup before any reader, exactly as :764 does.
        if (pblflg) {
            ysu_diagnose(u, v, thv, za, thermal, br[col], brcr_ub, nz, st,
                         col, &hpbl, &kpbl, &brdn, &brup);
        }
    } else pblflg = false;

    // WRF v4.6.1 bl_ysu.F90:732-768 runs the theta-li extension for every
    // column.  In-cloud liquid loading can therefore revive pblflg even
    // when the surface is stable and wstar3 remains zero.
    if (ysu_topdown_pblmix) {
        bool definebrup = false;
        int kpblold = kpbl;
        for (int fk = kpblold; fk <= nz - 1; ++fk) {
            int k = fk - 1;
            int q = k * st + col;
            real spdk2 = ysu_max(u[q] * u[q] + v[q] * v[q], 1.0f);
            real bruptmp = (thli[k] - thermalli)
                           * (G * za[k] / thli[0]) / spdk2;
            bool stable_li = bruptmp >= brcr_ub;
            if (definebrup) {
                kpbl = fk;
                brup = bruptmp;
                definebrup = false;
            }
            if (!stable_li) {
                brdn = bruptmp;
                definebrup = true;
                pblflg = true;
            }
        }
    }
    if (pblflg) {
        int kh = kpbl - 1;
        real frac;
        if (brdn >= brcr_ub) frac = 0.0f;
        else if (brup <= brcr_ub) frac = 1.0f;
        else frac = (brcr_ub - brdn) / (brup - brdn);
        hpbl = za[kh - 1] + frac * (za[kh] - za[kh - 1]);
        // bl_ysu.F90:765 and :766 are two INDEPENDENT statements.  The
        // second kills pblflg whenever kpbl <= 1, whatever hpbl came back
        // as -- including the case where hpbl was interpolated from WRF's
        // own za(i,0) overrun above and came back large.  Nesting :766
        // inside :765 let the theta-li revival (:745-749 raises pblflg on a
        // final unstable iteration without ever reassigning kpbl) survive
        // with kpbl == 1, and :833's k = kpbl(i)-1 is then one level below
        // the column.
        if (hpbl < zq[1]) kpbl = 1;
        if (kpbl <= 1) pblflg = false;
    }

    if (!sfcflg && hpbl < zq[1]) {
        real brcrit = brcr_sb;
        if (xland[col] >= 1.5f) {
            real ross = ysu_max(sqrtf(u10[col] * u10[col] + v10[col] * v10[col]),
                                1.0e-9f) / (cori * znt[col]);
            brcrit = ysu_min(0.16f * gfk_pow(1.0e-7f * ross, -0.18f), 0.3f);
        }
        ysu_diagnose(u, v, thv, za, thermal, br[col], brcrit, nz, st, col,
                     &hpbl, &kpbl, &brdn, &brup);
        if (hpbl < zq[1]) kpbl = 1;
        // Keep pblflg false: WRF uses kpbl for the stable K profile but
        // reserves countergradient and entrainment terms for convection.
    }

    real wm2 = 0.0f, we = 0.0f, hfxpbl = 0.0f, qfxpbl = 0.0f;
    real ufxpbl = 0.0f, vfxpbl = 0.0f, delta = 0.0f;
    real wstar3_2 = 0.0f, topdown_radsum = 0.0f;
    bool cloudflg = false;
    if (pblflg) {
        int kt = kpbl - 2;
        real wm3 = wstar3 + 5.0f * ust3;
        wm2 = gfk_pow(wm3, h2);
        real bfxpbl = __fdiv_rn(-0.15f * thv[0], G) * wm3 / hpbl;
        real dthv = ysu_max(thv[kt + 1] - thv[kt], tmin);
        we = ysu_max(bfxpbl / dthv, -sqrtf(wm2));
        // F90 839-897.  The kpbl<nz guard makes the source's k+2 access
        // explicit: a top-level PBL has no above-cloud comparison layer.
        if (ysu_topdown_pblmix && kpbl < nz
                && qc[kt * st + col] + qi[kt * st + col] > 1.0e-5f) {
            cloudflg = true;
            real ptop = p_interface[(kt + 1) * st + col];
            real templ = thli[kt] * gfk_pow(__fdiv_rn(ptop, 100000.0f), RCP);
            real rvls = 100.0f * 6.112f
                      * gfk_exp(17.67f * (templ - 273.16f)
                             / (templ - 29.65f)) * (EP2 / ptop);
            real qsum = qv[kt * st + col] + qc[kt * st + col];
            real temps = templ + (qsum - rvls)
                       / (CP / XLV + EP2 * XLV * rvls
                          / (RD * (templ * templ)));
            rvls = 100.0f * 6.112f
                 * gfk_exp(17.67f * (temps - 273.15f)
                        / (temps - 29.65f)) * (EP2 / ptop);
            real rcldb = ysu_max(qsum - rvls, 0.0f);
            int kabove = kt + 2;
            real dthv_cloud = (thli[kabove]
                    + ysu_thx(theta, exner, kabove * st + col) * ep1
                    * (qv[kabove * st + col] + qc[kabove * st + col]))
                    - (thli[kt] + ysu_thx(theta, exner, kt * st + col) * ep1 * qsum);
            dthv_cloud = ysu_max(dthv_cloud, 0.1f);
            real tmp1 = XLV / CP * rcldb
                      / (exner[kt * st + col] * dthv_cloud);
            // WRF 4.8 MMM 550b5b4:855 includes both upstream fixes.
            // Preserve (0.2 * 8.) * tmp1 before the cap; no offset.
#ifndef YSU_LEGACY_TOPDOWN
            real ent_eff = ysu_min(0.2f * 8.0f * tmp1, 0.4f);
#else
            // Explicit strict-arithmetic legacy arm: the WRF 4.6.1 term
            // with the source-ordered driver and shared no-FTZ math.
            real ent_eff = 0.2f * 8.0f * tmp1 + 0.2f;
#endif
            for (int kk = 0; kk <= kt; ++kk) {
                real radflux = rthraten[kk * st + col]
                             * exner[kk * st + col];
                // :859 radflux*cp/g*(dp) is ((radflux*cp)/g)*dp, not radflux*(cp/g*dp)
                radflux = __fmul_rn(__fdiv_rn(__fmul_rn(radflux, CP), G),
                    p_interface[kk * st + col] - p_interface[(kk + 1) * st + col]);
                if (radflux < 0.0f) topdown_radsum += fabsf(radflux);
            }
            topdown_radsum = ysu_max(topdown_radsum, 0.0f);
            real tk = theta[kt * st + col] * exner[kt * st + col];
            real rho2 = p[kt * st + col]
                      / (RD * tk * (1.0f + ep1 * qv[kt * st + col]));

            // Preserve the two overwritten bfx0 assignments in v4.6.1:
            // the executable values are max(sflux,0) and radsum/rho/cp.
            real bfx0 = ysu_max(sflux, 0.0f);
            wm3 = govrth * bfx0 * hpbl + 5.0f * ust3;
            wm2 = gfk_pow(wm3, h2);
            bfxpbl = __fdiv_rn(-0.15f * thv[0], G) * wm3 / hpbl;
            dthv = ysu_max(thv[kt + 1] - thv[kt], tmin);
            we = ysu_max(bfxpbl / dthv, -sqrtf(wm2));

            bfx0 = ysu_max(__fdiv_rn(topdown_radsum / rho2, CP), 0.0f);
            real wm3_top = G / thv[kt] * bfx0 * hpbl;
            real wm2_top = gfk_pow(wm3_top, h2);
            wm2 += wm2_top;
            bfxpbl = -ent_eff * bfx0;
            dthv = ysu_max(thv[kt + 1] - thv[kt], 0.1f);
            we += ysu_max(bfxpbl / dthv, -sqrtf(wm2_top));
            wstar3_2 = G / thv[kt] * bfx0 * hpbl;
            wscale = gfk_pow(ust3 + phifac * karman
                             * (wstar3 + wstar3_2) * 0.5f, h1);
            wscale = ysu_min(wscale, us * aphi16);
            wscale = ysu_max(wscale, __fdiv_rn(us, aphi5));
            real gamfac = bfac / rho / wscale;
            hgamt = ysu_min(__fdiv_rn(gamfac * hf, CP), gamcrt);
            hgamq = ysu_min(gamfac * qf, gamcrq);
            gamfac = bfac / rho2 / wscale;
            real hgamt2 = ysu_min(__fdiv_rn(gamfac * topdown_radsum, CP), gamcrt);
            hgamt = ysu_max(hgamt, 0.0f) + ysu_max(hgamt2, 0.0f);
            real cg = -15.9f * us * us / ysu_pos(wspd[col], 1.0e-9f)
                    * (wstar3 + wstar3_2)
                    / ysu_pos(gfk_pow(wscale, 4.0f), 1.0e-20f);
            hgamu = cg * u0;
            hgamv = cg * v0;
        }
        real dth = ysu_max(ysu_thx(theta, exner, (kt + 1) * st + col) - ysu_thx(theta, exner, kt * st + col), tmin);
        real dqq = ysu_min(qv[(kt + 1) * st + col] - qv[kt * st + col], 0.0f);
        hfxpbl = we * dth;
        qfxpbl = we * dqq;
        real dux = u[(kt + 1) * st + col] - u[kt * st + col];
        real dvx = v[(kt + 1) * st + col] - v[kt * st + col];
        if (dux > tmin) ufxpbl = ysu_max(we * dux, -us * us);
        else if (dux < -tmin) ufxpbl = ysu_min(we * dux, us * us);
        if (dvx > tmin) vfxpbl = ysu_max(we * dvx, -us * us);
        else if (dvx < -tmin) vfxpbl = ysu_min(we * dvx, us * us);
        real delb = govrth * d3 * hpbl;
        delta = ysu_min(d1 * hpbl + d2 * wm2 / delb, 100.0f);
    }
    // Each word is stored once with the value the overwritten stores left.
    for (int k = 0; k < nz; ++k) {
        real ef = 1.0e30f;
        if (pblflg && delta > 0.0f && k + 1 >= kpbl) {
            real e = (zq[k + 1] - hpbl) / delta;
            ef = e * e;               // :930 **2. is x*x in WRF's -O2 build
        }
        entfac[k] = ef;
        bool below_top = k < nz - 1;
        real xh0 = below_top ? xkzminh : 0.0f;
        xkzm[k] = below_top ? xkzminm : 0.0f;
        xkzh[k] = xh0;
        xkzq[k] = xh0;
        xkzml[k] = 0.0f;
        xkzhl[k] = 0.0f;
        zfacent[k] = 0.0f;
    }

    for (int k = 0; k < nz; ++k) {
        if (k + 1 < kpbl) {
            real zfac = ysu_clip(1.0f - (zq[k + 1] - za[0]) / (hpbl - za[0]),
                                  zfmin, 1.0f);
            zfacent[k] = gfk_pow(1.0f - zfac, 3.0f);
            real wsk = gfk_pow(ust3 + phifac * karman * wstar3
                               * (1.0f - zfac), h1);
            real wsk2 = gfk_pow(phifac * karman * wstar3_2 * zfac,
                                h1);
            real prfac, prfac2, prnumfac;
            if (sfcflg) {
                prfac = bfac * karman * sfcfrac;
                prfac2 = 15.9f * (wstar3 + wstar3_2) / ust3
                       / (1.0f + 4.0f * karman
                          * (wstar3 + wstar3_2) / ust3);
                if (!isfinite(prfac2)) {
                    // Even with the default no-FTZ route, cubing an
                    // extremely small ust can underflow to exactly zero.
                    // On the former FTZ route us*us*us reached zero for
                    // ust < ~2.3e-13 and the quotient above is Inf/Inf or
                    // 0/0 = NaN.  Pre-guard, ysu_min then laundered that
                    // NaN into xkzh = xkzmax: exch_h pinned at 1000 m2/s,
                    // no error raised.  WRF v4.6.1 has no floor here
                    // (bl_ysu.F90:674 cubes ust unguarded and :948 divides
                    // by it; the sfclay land floors, module_sf_sfclay.F:818
                    // and sf_sfclayrev.F90:772, are upstream and land-only),
                    // and on IEEE-subnormal hardware the same expression is
                    // finite: prfac2 -> 15.9/(4*karman) as ust3 -> 0.  So
                    // this is the rrtmg_sw.cu FP64-emulation countermeasure,
                    // not a WRF transcription: re-evaluate the same
                    // expression in float64, where us^3 down to the
                    // smallest normal float is itself normal.  It runs only
                    // when the FP32 result is already Inf/NaN, so every
                    // healthy lane keeps its exact FP32 word.  Where WRF
                    // itself is non-finite (w/ust3 overflowing FP32, or
                    // ust3 == 0 exactly), this produces the defined
                    // algebraic value instead -- never bit-exact to a bug.
                    prfac2 = yprfac2_rescue(us, wstar3, wstar3_2, karman);
                }
                real zz = ysu_max(zq[k + 1] - sfcfrac * hpbl, 0.0f);
                prnumfac = -(3.0f * (zz * zz)) / (hpbl * hpbl);   // :949, x*x
            } else {
                prfac = prfac2 = prnumfac = 0.0f;
                wsk = ysu_max(us / (1.0f + aphi5 * zol1 * zq[k + 1] / za[0]),
                              0.001f);
            }
            real prnum0 = ysu_clip(phih / phim + prfac, prmin, prmax);
            real km = wsk * karman * zq[k + 1] * (zfac * zfac)   // :960-961
                    + wsk2 * karman * (hpbl - zq[k + 1])          // **pfac (2.)
                    * ((1.0f - zfac) * (1.0f - zfac));            // is x*x at -O2
            if (k == kpbl - 2 && cloudflg && we < 0.0f) km = 0.0f;
            real prnum = 1.0f + (prnum0 - 1.0f) * gfk_exp(prnumfac);
            real kqq = km / prnum;
            prnum0 /= 1.0f + prfac2 * karman * sfcfrac;
            prnum = 1.0f + (prnum0 - 1.0f) * gfk_exp(prnumfac);
            real kh = km / prnum;
            xkzm[k] = ysu_min(km + xkzminm, xkzmax);
            xkzh[k] = ysu_min(kh + xkzminh, xkzmax);
            xkzq[k] = ysu_min(kqq + xkzminh, xkzmax);
        }
    }

    for (int k = 0; k < nz - 1; ++k) {
        if (k + 1 >= kpbl) {
            int q0i = k * st + col, q1i = (k + 1) * st + col;
            real ud = u[q1i] - u[q0i], vd = v[q1i] - v[q0i];
            real ss = (ud * ud + vd * vd) / (dza[k + 1] * dza[k + 1]) + 1.0e-9f;
            real ri = (G / (0.5f * (thv[k + 1] + thv[k])))
                    * (thv[k + 1] - thv[k]) / (ss * dza[k + 1]);
            if (qc[q0i] + qi[q0i] > 1.0e-5f && qc[q1i] + qi[q1i] > 1.0e-5f) {
                real qmean = 0.5f * (qv[q0i] + qv[q1i]);
                real tmean = 0.5f * (theta[q0i] * exner[q0i]
                                     + theta[q1i] * exner[q1i]);
                real alph = __fdiv_rn(XLV * qmean, RD) / tmean;
                real chi = __fdiv_rn(__fdiv_rn(XLV * XLV * qmean, CP), RV) / tmean / tmean;
                ri = (1.0f + alph) * (ri - __fdiv_rn(G * G / ss / tmean, CP)
                                      * ((chi - alph) / (1.0f + chi)));
            }
            real zk = karman * zq[k + 1];
            real rlamdz = ysu_min(ysu_max(0.1f * dza[k + 1], rlam), 300.0f);
            rlamdz = ysu_min(dza[k + 1], rlamdz);
            real rl = zk * rlamdz / (rlamdz + zk);
            real dk = rl * rl * sqrtf(ss), km, kh;
            if (ri < 0.0f) {
                ri = ysu_max(ri, rimin);
                real sri = sqrtf(-ri);
                km = dk * (1.0f + 8.0f * (-ri) / (1.0f + 1.746f * sri));
                kh = dk * (1.0f + 8.0f * (-ri) / (1.0f + 1.286f * sri));
            } else {
                kh = dk / ((1.0f + 5.0f * ri) * (1.0f + 5.0f * ri));
                km = kh * ysu_min(1.0f + 2.1f * ri, prmax);
            }
            real xm = ysu_min(km + xkzminm, xkzmax);
            real xh = ysu_min(kh + xkzminh, xkzmax);
            xkzm[k] = xm;
            xkzh[k] = xh;
            xkzml[k] = xm;               // the words just stored, not reloaded
            xkzhl[k] = xh;
        }
    }
    for (int k = kpbl - 1; k < nz - 1; ++k) xkzq[k] = xkzh[k];

#ifndef GPUWM_WRF_EXACT
    // The nonlocal entrainment flux is explicit although local diffusion
    // is implicit.  Its donor is the jump across kpbl-2 -> kpbl-1, not
    // the jump at each interface to which zfacent distributes the flux.
    // Bound its forward-Euler relaxation Courant number by one.  Otherwise
    // a weak virtual inversion permits we*dt to cross several layers and
    // vapor overshoots the donor (then condensation makes a hot layer).
    // Use the actual pressure-mass divergence, including the flux entering
    // the lower member of the donor pair; a geometric dz/dt cap is not the
    // discrete operator's stability limit.  Scale paired scalar fluxes,
    // retaining their conservative interface divergence and the diagnosed
    // turbulent diffusivity, countergradient heat and momentum closure.
    if (pblflg && we < 0.0f && kpbl >= 2) {
        const int kt = kpbl - 2;
        real atop = (p[kt * st + col] - p[(kt + 1) * st + col])
                        / dza[kt + 1] * zfacent[kt];
        real abelow = kt > 0
            ? (p[(kt - 1) * st + col] - p[kt * st + col])
                / dza[kt] * zfacent[kt - 1] : 0.0f;
        if (BEP) {
            atop *= bep.sf[kt * st + col];
            if (kt > 0) abelow *= bep.sf[(kt - 1) * st + col];
        }
        const real relax = atop / delp[kt + 1]
                         + (atop - abelow) / delp[kt];
        const real courant = dt2 * (-we) * relax;
        if (courant > 1.0f) {
            hfxpbl /= courant;
            qfxpbl /= courant;
        }
    }

#endif

    // Heat matrix, including the countergradient and entrainment fluxes.
    // The assembly below writes diag[0..nz-1], rhs[0..nz-1], upper[0..nz-2]
    // and lower[1..nz-1] before any of them is read, and ysu_thomas never
    // reads lower[0] or upper[nz-1]; zeroing the four whole columns first
    // only cost workspace traffic.  The two unread ends are still zeroed so
    // the workspace holds the same words as before.
    lower[0] = 0.0f;
    upper[nz - 1] = 0.0f;
    diag[0] = 1.0f;
    if (BEP) {
        // bl_ysu.F90:1045 with bepswitch = 1: (1.0-bepswitch)*hfx/cont/del*dt2
        // is a signed zero, so the surface heat flux enters only through
        // b_t_bep(1), where the couple folded the rural part in.
        rhs[0] = theta0 - 300.0f + __fdiv_rn(0.0f * hf, (CP / G)) / delp[0] * dt2;
    } else {
        rhs[0] = theta0 - 300.0f + __fdiv_rn(hf, (CP / G)) / delp[0] * dt2;
    }
    // diag[k] and rhs[k] ride in registers from the level that set them
    // to the level that finishes them (the same words, not reloaded).
    real diag_k = diag[0], rhs_k = rhs[0];
    for (int k = 0; k < nz - 1; ++k) {
        real dtodsd, dtodsu;
        real rhs_k1, up, lo;
        if (BEP) {
            // bl_ysu.F90:1050-1051 (and :1138, :1326): sfk2d(i,k)*dt2/del.
            real sfk = bep.sf[k * st + col];
            dtodsd = sfk * dt2 / delp[k];
            dtodsu = sfk * dt2 / delp[k + 1];
        } else {
            dtodsd = dt2 / delp[k], dtodsu = dt2 / delp[k + 1];
        }
        real dsig = p[k * st + col] - p[(k + 1) * st + col];
        real rdz = 1.0f / dza[k + 1];
        real xk_k = xkzh[k];
        real tem1 = dsig * xk_k * rdz;
        if (pblflg && k + 1 < kpbl) {
            real flux = tem1 * (-hgamt / hpbl - hfxpbl * zfacent[k] / xk_k);
            rhs_k += dtodsd * flux;
            rhs_k1 = ysu_thx(theta, exner, (k + 1) * st + col) - 300.0f - dtodsu * flux;
        } else if (pblflg && k + 1 >= kpbl && entfac[k] < 4.6f) {
            real knew = -we * dza[kpbl - 1] * gfk_exp(-entfac[k]);
            xk_k = ysu_clip(sqrtf(ysu_max(knew * xkzhl[k], 0.0f)),
                            xkzminh, xkzmax);
            xkzh[k] = xk_k;
            rhs_k1 = ysu_thx(theta, exner, (k + 1) * st + col) - 300.0f;
        } else rhs_k1 = ysu_thx(theta, exner, (k + 1) * st + col) - 300.0f;
        real dsdz2 = dsig * xk_k * rdz * rdz;
        if (BEP) {
            // bl_ysu.F90:1070-1071: au = -dtodsd*dsdz2/vlk2d(i,k),
            // al = -dtodsu*dsdz2/vlk2d(i,k) -- both divide by level k.
            real vlk = bep.vl[k * st + col];
            up = -dtodsd * dsdz2 / vlk;
            lo = -dtodsu * dsdz2 / vlk;
        } else {
            up = -dtodsd * dsdz2;
            lo = -dtodsu * dsdz2;
        }
        upper[k] = up;
        lower[k + 1] = lo;
        diag[k] = diag_k - up;          // diag[k] -= upper[k]
        rhs[k] = rhs_k;
        diag_k = 1.0f - lo;             // diag[k+1] = 1 - lower[k+1]
        rhs_k = rhs_k1;
    }
    diag[nz - 1] = diag_k;
    rhs[nz - 1] = rhs_k;
    if (BEP) {
        // bl_ysu.F90:1078-1083.  __fmul_rn: WRF rounds the product before
        // the add/subtract, NVRTC would otherwise fuse it.
        for (int k = 0; k < nz; ++k) {
            int q = k * st + col;
            diag[k] = diag[k] - __fmul_rn(bep.a_t[q], dt2);
            rhs[k] = rhs[k] + __fmul_rn(bep.b_t[q], dt2);
        }
    }
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) {
        int q = k * st + col;
        // WRF's own association, bl_ysu.F90:1103: (f1(i,k) - thx(i,k) + 300.).
        // Not cosmetic.  The solve leaves rhs[k] ~ theta - 300, so
        // (rhs - theta) is exactly -300 and adding 300 gives exactly zero,
        // whereas (rhs + 300) rounds at magnitude ~theta -- one ULP coarser
        // than rhs itself -- before the subtract, and the residue survives.
        // Measured: 3.4e-07 K/s where WRF writes exactly 0.0, which is
        // 884345697 ULP of that zero.
        dtheta[q] = __fdiv_rn(ysm(ysm((rhs[k] - ysu_thx(theta, exner, q) + 300.0f), rdt), exner[q]), exner[q]);
    }

    // Vapor matrix (Kq); cloud and ice reuse its matrix.
    lower[0] = 0.0f;
    upper[nz - 1] = 0.0f;
    diag[0] = 1.0f;
    if (BEP) {
        // bl_ysu.F90:1135 with bepswitch = 1: a signed zero is added, which
        // is what turns a -0.0 qv into +0.0 exactly as WRF does.
        rhs[0] = qv0 + 0.0f * qf * G / delp[0] * dt2;
    } else {
        rhs[0] = qv0 + qf * G / delp[0] * dt2;
    }
    // diag[k] and rhs[k] ride in registers from the level that set them
    // to the level that finishes them (the same words, not reloaded).
    diag_k = diag[0];
    rhs_k = rhs[0];
    for (int k = 0; k < nz - 1; ++k) {
        real dtodsd, dtodsu;
        real rhs_k1, up, lo;
        if (BEP) {
            // bl_ysu.F90:1050-1051 (and :1138, :1326): sfk2d(i,k)*dt2/del.
            real sfk = bep.sf[k * st + col];
            dtodsd = sfk * dt2 / delp[k];
            dtodsu = sfk * dt2 / delp[k + 1];
        } else {
            dtodsd = dt2 / delp[k], dtodsu = dt2 / delp[k + 1];
        }
        real dsig = p[k * st + col] - p[(k + 1) * st + col];
        real rdz = 1.0f / dza[k + 1];
        real xk_k = xkzq[k];
        real tem1 = dsig * xk_k * rdz;
        if (pblflg && k + 1 < kpbl) {
            real flux = tem1 * (-qfxpbl * zfacent[k] / xk_k);
            rhs_k += dtodsd * flux;
            rhs_k1 = qv[(k + 1) * st + col] - dtodsu * flux;
        } else if (pblflg && k + 1 >= kpbl && entfac[k] < 4.6f) {
            real knew = -we * dza[kpbl - 1] * gfk_exp(-entfac[k]);
            xk_k = ysu_clip(sqrtf(ysu_max(knew * xkzhl[k], 0.0f)),
                            xkzminh, xkzmax);
            xkzq[k] = xk_k;
            rhs_k1 = qv[(k + 1) * st + col];
        } else rhs_k1 = qv[(k + 1) * st + col];
        real dsdz2 = dsig * xk_k * rdz * rdz;
        if (BEP) {
            // bl_ysu.F90:1070-1071: au = -dtodsd*dsdz2/vlk2d(i,k),
            // al = -dtodsu*dsdz2/vlk2d(i,k) -- both divide by level k.
            real vlk = bep.vl[k * st + col];
            up = -dtodsd * dsdz2 / vlk;
            lo = -dtodsu * dsdz2 / vlk;
        } else {
            up = -dtodsd * dsdz2;
            lo = -dtodsu * dsdz2;
        }
        upper[k] = up;
        lower[k + 1] = lo;
        diag[k] = diag_k - up;          // diag[k] -= upper[k]
        rhs[k] = rhs_k;
        diag_k = 1.0f - lo;             // diag[k+1] = 1 - lower[k+1]
        rhs_k = rhs_k1;
    }
    diag[nz - 1] = diag_k;
    rhs[nz - 1] = rhs_k;
    if (BEP) {
        // bl_ysu.F90:1164-1168: vapor alone solves with adv = ad - a_q*dt2;
        // cloud water and ice reuse the unforced ad (:1177, :1190).  xkzhl
        // is dead after the vapor assembly above, so it holds adv.
        for (int k = 0; k < nz; ++k) {
            int q = k * st + col;
            xkzhl[k] = diag[k] - __fmul_rn(bep.a_q[q], dt2);
            rhs[k] = rhs[k] + __fmul_rn(bep.b_q[q], dt2);
        }
        ysu_thomas(lower, xkzhl, upper, rhs, gamma, nz);
    } else {
        ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    }
    for (int k = 0; k < nz; ++k) dqv[k * st + col] = ysm(yss(rhs[k], qv[k * st + col]), rdt);
    for (int k = 0; k < nz; ++k) rhs[k] = qc[k * st + col];
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) dqc[k * st + col] = ysm(yss(rhs[k], qc[k * st + col]), rdt);
    for (int k = 0; k < nz; ++k) rhs[k] = qi[k * st + col];
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) dqi[k * st + col] = ysm(yss(rhs[k], qi[k * st + col]), rdt);

    // Momentum matrix and implicit surface stress.
    lower[0] = 0.0f;
    upper[nz - 1] = 0.0f;
    real fric = us * us / wspd1 * rho * G / delp[0] * dt2      // :1306-1307
              * ((wspd1 / ysu_pos(wspd[col], 1.0e-9f))      // (wspd1/wspd)**2
                 * (wspd1 / ysu_pos(wspd[col], 1.0e-9f)));  // is formed first
    if (TOPO) {
        // bl_ysu.F90:1254-1314, the ctopo-present arm.  The paj TKE profile
        // (:1257-1279) reads the diffusivities as the heat and moisture
        // solves left them and before the momentum assembly below widens
        // xkzm (:1338-1342); thli is dead here and holds it.
        YsuCol tke = thli;
        // WRF writes only 1:kte-1 of this profile. Its get_pblh search can
        // reach the last level in a shallow domain. Define that missing word
        // as zero and retain the bounded searches in ysu_480_topo.cuh.
        // Checked public Fortran exposes qke1d(5) at nz=4, top=100 m.
        tke[nz - 1] = 0.0f;
        for (int k = 0; k < nz - 1; ++k) {
            int q0 = k * st + col, q1 = (k + 1) * st + col;
            real rdza = dza[k + 1];
            real dudz = __fdiv_rn(__fsub_rn(u[q1], u[q0]), rdza);
            real dvdz = __fdiv_rn(__fsub_rn(v[q1], v[q0]), rdza);
            real su = __fdiv_rn(__fmul_rn(__fadd_rn(-__fdiv_rn(hgamu, hpbl), dudz),
                                          __fsub_rn(u[q1], u[q0])), rdza);
            real sv = __fdiv_rn(__fmul_rn(__fadd_rn(-__fdiv_rn(hgamv, hpbl), dvdz),
                                          __fsub_rn(v[q1], v[q0])), rdza);
            real shear = __fmul_rn(xkzm[k], __fadd_rn(su, sv));
            real dthdz = __fdiv_rn(__fsub_rn(ysu_thx(theta, exner, q1), ysu_thx(theta, exner, q0)), rdza);
            real buoy = __fmul_rn(__fmul_rn(__fmul_rn(xkzh[k], G),
                                            __fdiv_rn(1.0f, ysu_thx(theta, exner, q0))),
                                  __fadd_rn(-__fdiv_rn(hgamt, hpbl), dthdz));
            real zk = __fmul_rn(karman, zq[k + 1]);
            real rlamdz;
            if (k + 1 >= kpbl) {
                rlamdz = ysu_min(ysu_max(__fmul_rn(0.1f, rdza), rlam), 300.0f);
                rlamdz = ysu_min(rdza, rlamdz);
            } else {
                rlamdz = 150.0f;
            }
            real el = __fdiv_rn(__fmul_rn(zk, rlamdz), __fadd_rn(rlamdz, zk));
            real t = __fmul_rn(__fmul_rn(16.6f, el), __fsub_rn(shear, buoy));
            tke[k] = (t <= 0.0f) ? 0.0f : gfk_pow(t, 0.66f);
        }
        real pblh_ysu = ysu_get_pblh(thv, tke, zq, nz, xland[col]);
        real vconv;
        if (xland[col] < 1.5f) {
            real fluxc = ysu_max(sflux, 0.0f);
            vconv = __fmul_rn(1.0f, gfk_pow(__fmul_rn(__fmul_rn(
                        __fdiv_rn(G, thv[0]), pblh_ysu), fluxc), 0.33f));
        } else {
            vconv = 0.0f;
        }
        real vconvnew = __fadd_rn(__fmul_rn(0.9f, vconv),
                                  __fmul_rn(1.5f, ysu_max(__fdiv_rn(
                                      __fsub_rn(pblh_ysu, 500.0f), 1000.0f),
                                      0.0f)));
        real vconvlim = ysu_min(vconvnew, 1.0f);
        // ctopo = 1 on the default and BEP entry points (YsuTopo{} carries
        // null pointers): module_bl_ysu.F:404 passes it on every WRF run.
        real ctopo = prm.topo ? topo.ctopo[col] : 1.0f;
        real drag_v = __fmul_rn(fric, vconvlim);
        real drag_c = __fmul_rn(__fmul_rn(ctopo, fric),
                                __fsub_rn(1.0f, vconvlim));
        diag[0] = __fadd_rn(__fadd_rn(1.0f, drag_v), drag_c);
        if (BEP) {
            // :1313-1314 with the declared fix below: the WHOLE bracket,
            // ad(1) - bepswitch*(fric*vconvlim+ctopo*fric*(1-vconvlim)),
            // bepswitch = 1.  Outside BEP WRF's line subtracts 0*0*bracket,
            // an exact no-op, so it is not spelled.
            diag[0] = __fsub_rn(diag[0], __fadd_rn(drag_v, drag_c));
        }
    } else {
        // bl_ysu.F90:1315, the ctopo-absent arm.  No WRF run reaches it
        // (module_bl_ysu.F:404 always passes ctopo), and no entry point of
        // this kernel instantiates it any longer; kept only so the template
        // stays total.
        diag[0] = 1.0f + fric;
        if (BEP) diag[0] = diag[0] - fric;
    }
    if (BEP) {
        // DECLARED DIVERGENCE FROM WRF v4.7.1 (docs/public/PHYSICS.md, "Urban
        // canopy models"; pinned by tests/test_ysu_bep_rural_drag.py).
        // bl_ysu.F90:1313-1314 removes only the URBAN fraction of YSU's own
        // surface drag, ad(1) - bepswitch*frc*(fric*vconvlim+ctopo*fric*
        // (1-vconvlim)), leaving (1-frc)*fric on the diagonal.  But the BEP
        // couple has already folded that same rural drag,
        // (1-frc)*(-ust*ust)/dz8w/|U|, into a_u_bep/a_v_bep at level 1
        // (module_sf_noahdrv.F:1708-1711, module_sf_noahmpdrv.F:3718-3721),
        // and :1359-1368 below adds a_u*dt2 to the same diagonal: WRF counts
        // the rural surface drag twice on every column, ocean included, and
        // the 750 m runs lost 1.25 m/s of 10 m wind over the sea in two
        // hours.  Heat and moisture are counted once (:1045 drops YSU's own
        // flux with (1-bepswitch) and takes the rural flux from b_t/b_q), and
        // WRF's myjurb takes the surface drag only through a_u (its VDIFV
        // never reads AKMS under BEP).  gpuwm removes the WHOLE of YSU's own
        // drag here -- WRF's line without the frc_urb1d factor -- so the
        // surface drag enters once, through a_u_bep, as heat already does.
        // WRF's own driver passes ctopo = 1 (module_bl_ysu.F:404), and so
        // does this kernel now: the removal is spelled on WRF's bracket in
        // the ctopo arm above.  WRF's stock line keeps the frc_urb1d factor:
        // `__fsub_rn(diag[0], __fmul_rn(bep.frc[col], __fadd_rn(drag_v, drag_c)))`.
    }
    rhs[0] = u0;
    // diag[k] and rhs[k] ride in registers from the level that set them
    // to the level that finishes them (the same words, not reloaded).
    diag_k = diag[0];
    rhs_k = rhs[0];
    for (int k = 0; k < nz - 1; ++k) {
        real dtodsd, dtodsu;
        real rhs_k1, up, lo;
        if (BEP) {
            // bl_ysu.F90:1050-1051 (and :1138, :1326): sfk2d(i,k)*dt2/del.
            real sfk = bep.sf[k * st + col];
            dtodsd = sfk * dt2 / delp[k];
            dtodsu = sfk * dt2 / delp[k + 1];
        } else {
            dtodsd = dt2 / delp[k], dtodsu = dt2 / delp[k + 1];
        }
        real dsig = p[k * st + col] - p[(k + 1) * st + col];
        real rdz = 1.0f / dza[k + 1];
        real xk_k = xkzm[k];
        real tem1 = dsig * xk_k * rdz;
        if (pblflg && k + 1 < kpbl) {
            real flux = tem1 * (-hgamu / hpbl - ufxpbl * zfacent[k] / xk_k);
            rhs_k += dtodsd * flux;
            rhs_k1 = u[(k + 1) * st + col] - dtodsu * flux;
        } else if (pblflg && k + 1 >= kpbl && entfac[k] < 4.6f) {
            xk_k = ysu_clip(sqrtf(ysu_max(xkzh[k] * xkzml[k], 0.0f)),
                            xkzminm, xkzmax);
            xkzm[k] = xk_k;
            rhs_k1 = u[(k + 1) * st + col];
        } else rhs_k1 = u[(k + 1) * st + col];
        real dsdz2 = dsig * xk_k * rdz * rdz;
        if (BEP) {
            // bl_ysu.F90:1070-1071: au = -dtodsd*dsdz2/vlk2d(i,k),
            // al = -dtodsu*dsdz2/vlk2d(i,k) -- both divide by level k.
            real vlk = bep.vl[k * st + col];
            up = -dtodsd * dsdz2 / vlk;
            lo = -dtodsu * dsdz2 / vlk;
        } else {
            up = -dtodsd * dsdz2;
            lo = -dtodsu * dsdz2;
        }
        upper[k] = up;
        lower[k + 1] = lo;
        diag[k] = diag_k - up;          // diag[k] -= upper[k]
        rhs[k] = rhs_k;
        diag_k = 1.0f - lo;             // diag[k+1] = 1 - lower[k+1]
        rhs_k = rhs_k1;
    }
    diag[nz - 1] = diag_k;
    rhs[nz - 1] = rhs_k;
    if (BEP) {
        // bl_ysu.F90:1359-1368: ad1 = ad; ad -= a_u*dt2; ad1 -= a_v*dt2;
        // f1 += b_u*dt2 (f2 += b_v*dt2 is added after the v RHS rebuild
        // below).  entfac is dead after the assembly above, so it holds
        // ad1, the v diagonal tridi2n solves with.
        for (int k = 0; k < nz; ++k) {
            int q = k * st + col;
            entfac[k] = diag[k] - __fmul_rn(bep.a_v[q], dt2);
            diag[k] = diag[k] - __fmul_rn(bep.a_u[q], dt2);
            rhs[k] = rhs[k] + __fmul_rn(bep.b_u[q], dt2);
        }
    }
    ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    for (int k = 0; k < nz; ++k) du[k * st + col] = ysm(yss(rhs[k], u[k * st + col]), rdt);
    for (int k = 0; k < nz; ++k) rhs[k] = v[k * st + col];
    // tridi2n assembles distinct u/v countergradient right-hand sides while
    // sharing one matrix.  Recreate the v RHS after the u solve.
    for (int k = 0; k < nz - 1; ++k) {
        if (pblflg && k + 1 < kpbl) {
            real dtodsd, dtodsu;
            if (BEP) {
                real sfk = bep.sf[k * st + col];
                dtodsd = sfk * dt2 / delp[k];
                dtodsu = sfk * dt2 / delp[k + 1];
            } else {
                dtodsd = dt2 / delp[k], dtodsu = dt2 / delp[k + 1];
            }
            real dsig = p[k * st + col] - p[(k + 1) * st + col];
            real rdz = 1.0f / dza[k + 1];
            real tem1 = dsig * xkzm[k] * rdz;
            real flux = tem1 * (-hgamv / hpbl - vfxpbl * zfacent[k] / xkzm[k]);
            rhs[k] += dtodsd * flux;
            rhs[k + 1] = v[(k + 1) * st + col] - dtodsu * flux;
        }
    }
    if (BEP) {
        for (int k = 0; k < nz; ++k)
            rhs[k] = rhs[k] + __fmul_rn(bep.b_v[k * st + col], dt2);
        ysu_tridi2n_v(lower, entfac, gamma, rhs, nz);
    } else {
        ysu_thomas(lower, diag, upper, rhs, gamma, nz);
    }
    for (int k = 0; k < nz; ++k) dv[k * st + col] = ysm(yss(rhs[k], v[k * st + col]), rdt);
    if (TOPO) ysu_topo_blend_u10(prm.topo != 0, topo, u10, v10, u0, v0, col);

    exch_h[col] = exch_m[col] = 0.0f;
    for (int k = 1; k < nz; ++k) {
        exch_h[k * st + col] = xkzh[k - 1];
        exch_m[k * st + col] = xkzm[k - 1];
    }
    hpbl_out[col] = hpbl;
    kpbl_out[col] = kpbl;
    wstar_out[col] = wstar;
    delta_out[col] = delta;
    topdown_radsum_out[col] = topdown_radsum;
    wstar3_2_out[col] = wstar3_2;
    cloudflg_out[col] = cloudflg ? 1 : 0;
}
