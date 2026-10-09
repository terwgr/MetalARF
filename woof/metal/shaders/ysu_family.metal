// Dependency-marked source translation for WRF's YSU column algorithms.
//
// Source: woof/core/kernels/ysu.cu (WRF v4.6.1, bl_ysu.F90). This file is
// deliberately not present in a production signature fragment. Public Python
// submission remains fail-closed until the int32 resource path and a consumer
// review admit these source proposals. It has not been compiled or run.

#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"

// Source-required operations for which the shared integer-bit math layer has
// no implementation yet. These declarations make the dependency boundary
// visible; they are intentionally unresolved and prevent this file from being
// treated as a build-ready shader.
uint ysu_required_sqrt_rn_bits(uint x);
uint ysu_required_cbrt_bits(uint x);
uint ysu_required_hypot_bits(uint x, uint y);

static inline float ysu_add(float a, float b)
{ return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ysu_sub(float a, float b)
{ return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ysu_mul(float a, float b)
{ return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ysu_div(float a, float b)
{ return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ysu_exp(float x)
{ return as_type<float>(soft_gfk_exp(as_type<uint>(x))); }
static inline float ysu_pow(float x, float y)
{ return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x), as_type<uint>(y))); }
static inline float ysu_sqrt(float x)
{ return as_type<float>(ysu_required_sqrt_rn_bits(as_type<uint>(x))); }
static inline float ysu_cbrt(float x)
{ return as_type<float>(ysu_required_cbrt_bits(as_type<uint>(x))); }
static inline float ysu_hypot(float x, float y)
{ return as_type<float>(ysu_required_hypot_bits(as_type<uint>(x), as_type<uint>(y))); }

static inline float ysu_min(float a, float b)
{ return (a != a) ? a : (a < b ? a : b); }
static inline float ysu_max(float a, float b)
{ return (a != a) ? a : (a > b ? a : b); }
static inline float ysu_clip(float x, float lo, float hi)
{ return ysu_min(ysu_max(x, lo), hi); }

// Direct source mapping: CUDA ysu_diagnose. One lane owns one column. The
// source uses k=1..nz-1 as the first crossing search and interpolates between
// za[kp-2] and za[kp-1] with the source brdn/brup values.
static inline void ysu_diagnose_bits(
    device const float *u, device const float *v,
    thread const float *thv, thread const float *za,
    float thermal, float br0, float brcrit, int nz, int st, int col,
    thread float &hpbl, thread int &kpbl,
    thread float &brdn_out, thread float &brup_out)
{
    float brup = br0;
    float brdn = brup;
    int kp = 1;
    bool crossed = false;
    for (int k = 1; k < nz; ++k) {
        if (!crossed) {
            brdn = brup;
            const int q = k * st + col;
            const float u2 = ysu_mul(u[q], u[q]);
            const float v2 = ysu_mul(v[q], v[q]);
            const float spdk2 = ysu_max(ysu_add(u2, v2), 1.0f);
            const float buoy = ysu_div(
                ysu_mul(ysu_sub(thv[k], thermal),
                        ysu_div(ysu_mul(9.81f, za[k]), thv[0])), spdk2);
            brup = buoy;
            kp = k + 1;
            crossed = brup > brcrit;
        }
    }
    float frac;
    if (brdn >= brcrit) frac = 0.0f;
    else if (brup <= brcrit) frac = 1.0f;
    else frac = ysu_div(ysu_sub(brcrit, brdn), ysu_sub(brup, brdn));
    const int kh = kp - 1;
    hpbl = ysu_add(za[kh - 1],
                   ysu_mul(frac, ysu_sub(za[kh], za[kh - 1])));
    kpbl = kp;
    brdn_out = brdn;
    brup_out = brup;
}

// Thomas factorization/back substitution, matching ysu_thomas's scalar
// operation order. The caller owns all nz-element work arrays.
static inline void ysu_thomas_bits(
    thread const float *lower, thread const float *diag,
    thread const float *upper, thread float *rhs,
    thread float *gamma, int nz)
{
    float inv = ysu_div(1.0f, diag[0]);
    gamma[0] = ysu_mul(upper[0], inv);
    rhs[0] = ysu_mul(rhs[0], inv);
    for (int k = 1; k < nz - 1; ++k) {
        const float pivot = ysu_sub(diag[k], ysu_mul(lower[k], gamma[k - 1]));
        inv = ysu_div(1.0f, pivot);
        gamma[k] = ysu_mul(upper[k], inv);
        const float numer = ysu_sub(rhs[k], ysu_mul(lower[k], rhs[k - 1]));
        rhs[k] = ysu_mul(numer, inv);
    }
    const int last = nz - 1;
    const float pivot = ysu_sub(diag[last], ysu_mul(lower[last], gamma[last - 1]));
    inv = ysu_div(1.0f, pivot);
    rhs[last] = ysu_mul(
        ysu_sub(rhs[last], ysu_mul(lower[last], rhs[last - 1])), inv);
    for (int k = nz - 2; k >= 0; --k)
        rhs[k] = ysu_sub(rhs[k], ysu_mul(gamma[k], rhs[k + 1]));
}

// BEP's tridi2n v arm pivots on diag_v but reuses the u solve's gamma. This
// source quirk is kept separate from ysu_thomas_bits, as in ysu_tridi2n_v.
static inline void ysu_tridi2n_v_bits(
    thread const float *lower, thread const float *diag_v,
    thread const float *gamma_u, thread float *rhs, int nz)
{
    float inv = ysu_div(1.0f, diag_v[0]);
    rhs[0] = ysu_mul(rhs[0], inv);
    for (int k = 1; k < nz - 1; ++k) {
        const float pivot = ysu_sub(diag_v[k], ysu_mul(lower[k], gamma_u[k - 1]));
        inv = ysu_div(1.0f, pivot);
        rhs[k] = ysu_mul(ysu_sub(rhs[k], ysu_mul(lower[k], rhs[k - 1])), inv);
    }
    const int last = nz - 1;
    const float pivot = ysu_sub(diag_v[last], ysu_mul(lower[last], gamma_u[last - 1]));
    inv = ysu_div(1.0f, pivot);
    rhs[last] = ysu_mul(
        ysu_sub(rhs[last], ysu_mul(lower[last], rhs[last - 1])), inv);
    for (int k = nz - 2; k >= 0; --k)
        rhs[k] = ysu_sub(rhs[k], ysu_mul(gamma_u[k], rhs[k + 1]));
}

// The theta-li top-down scan from bl_ysu.F90:732-768. `kpbl`, `brdn`,
// `brup`, and `pblflg` are in/out because the source can revive convection
// without changing kpbl on the final unstable iteration.
static inline void ysu_topdown_extend_bits(
    device const float *u, device const float *v,
    thread const float *thli, thread const float *za,
    float thermalli, int nz, int st, int col,
    thread int &kpbl, thread float &brdn, thread float &brup,
    thread bool &pblflg)
{
    bool definebrup = false;
    const int kpblold = kpbl;
    for (int fk = kpblold; fk <= nz - 1; ++fk) {
        const int k = fk - 1;
        const int q = k * st + col;
        const float spdk2 = ysu_max(
            ysu_add(ysu_mul(u[q], u[q]), ysu_mul(v[q], v[q])), 1.0f);
        const float bruptmp = ysu_div(
            ysu_mul(ysu_sub(thli[k], thermalli),
                    ysu_div(ysu_mul(9.81f, za[k]), thli[0])), spdk2);
        const bool stable_li = bruptmp >= 0.0f;
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

// Local-K Richardson/diffusivity segment from ysu_column_body. This is the
// source branch for levels k+1 >= kpbl; PBL mixing and matrix assembly follow
// in the owning CUDA body but remain to be ported before any kernel admission.
static inline void ysu_local_k_pair_bits(
    float u0, float v0, float u1, float v1, float thv0, float thv1,
    float condensate0, float condensate1, float qv0, float qv1,
    float theta0, float exner0, float theta1, float exner1,
    float dza, float zq_up, float g, float xlv, float rd, float rv,
    float cp, float karman, float rlam, float prmax,
    thread float &km_out, thread float &kh_out)
{
    const float ud = ysu_sub(u1, u0), vd = ysu_sub(v1, v0);
    const float shear = ysu_add(
        ysu_div(ysu_add(ysu_mul(ud, ud), ysu_mul(vd, vd)),
                ysu_mul(dza, dza)), 1.0e-9f);
    float ri = ysu_div(
        ysu_mul(ysu_div(9.81f, ysu_mul(0.5f, ysu_add(thv1, thv0))),
                ysu_sub(thv1, thv0)), ysu_mul(shear, dza));
    if (condensate0 > 1.0e-5f && condensate1 > 1.0e-5f) {
        const float qmean = ysu_mul(0.5f, ysu_add(qv0, qv1));
        const float tmean = ysu_mul(0.5f, ysu_add(
            ysu_mul(theta0, exner0), ysu_mul(theta1, exner1)));
        const float alph = ysu_div(ysu_div(ysu_mul(xlv, qmean), rd), tmean);
        const float chi = ysu_div(
            ysu_div(ysu_div(ysu_mul(ysu_mul(xlv, xlv), qmean), cp), rv),
            ysu_mul(tmean, tmean));
        const float moist = ysu_div(
            ysu_div(ysu_div(ysu_mul(g, g), shear), tmean), cp);
        ri = ysu_mul(ysu_add(1.0f, alph),
                     ysu_sub(ri, ysu_mul(moist,
                         ysu_div(ysu_sub(chi, alph), ysu_add(1.0f, chi)))));
    }
    const float zk = ysu_mul(karman, zq_up);
    float rlamdz = ysu_min(ysu_max(ysu_mul(0.1f, dza), rlam), 300.0f);
    rlamdz = ysu_min(dza, rlamdz);
    const float rl = ysu_div(ysu_mul(zk, rlamdz), ysu_add(rlamdz, zk));
    const float dk = ysu_mul(ysu_mul(rl, rl), ysu_sqrt(shear));
    if (ri < 0.0f) {
        ri = ysu_max(ri, -100.0f);
        const float sri = ysu_sqrt(ysu_sub(0.0f, ri));
        km_out = ysu_mul(dk, ysu_add(1.0f,
            ysu_div(ysu_mul(8.0f, ysu_sub(0.0f, ri)),
                    ysu_add(1.0f, ysu_mul(1.746f, sri)))));
        kh_out = ysu_mul(dk, ysu_add(1.0f,
            ysu_div(ysu_mul(8.0f, ysu_sub(0.0f, ri)),
                    ysu_add(1.0f, ysu_mul(1.286f, sri)))));
    } else {
        const float denom = ysu_add(1.0f, ysu_mul(5.0f, ri));
        kh_out = ysu_div(dk, ysu_mul(denom, denom));
        km_out = ysu_mul(kh_out, ysu_min(ysu_add(1.0f, ysu_mul(2.1f, ri)), prmax));
    }
}

// No kernel entrypoint is admitted yet. Remaining source body includes surface
// flux similarity, convective/top-down cloud mixing, BEP and topo arms, and
// heat/moisture/momentum matrix assembly/output stores. Those are tracked in
// woof/metal/handoffs/78_ysu_math.md rather than represented by a partial
// kernel that could be mistaken for an executable YSU port.
