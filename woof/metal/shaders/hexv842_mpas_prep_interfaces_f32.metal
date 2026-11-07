// MPAS v8.4.2 preparation interface; resident_v842.preparation_source.
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_prep_interfaces_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)

static inline uint prep_positive_q_bits_v841(const float raw)
{
    const uint bits = as_type<uint>(raw);
    return (bits & 0x80000000u) == 0u ? bits : 0u;
}

static inline float prep_positive_q_v841(const float raw)
{
    return as_type<float>(prep_positive_q_bits_v841(raw));
}

static inline bool prep_finite_bits_v841(const float raw)
{
    return (as_type<uint>(raw) & 0x7fffffffu) < 0x7f800000u;
}

static inline bool prep_negative_nonzero_bits_v841(const float raw)
{
    const uint bits = as_type<uint>(raw);
    return (bits & 0x80000000u) != 0u
        && (bits & 0x7fffffffu) != 0u;
}


// Contracted expf from flt32_expf_fma.cuh; binary64 operations use Soft64.
static inline float mpas_v842_exp_fma(float value) {
    const uint bits=as_type<uint>(value);
    const uint abstop=(bits>>20u)&0x7ffu;
    if (abstop>=0x42bu) return as_type<float>(soft_gfk_exp(bits));
    const ulong xd=soft64_from_f32_bits(bits);
    const ulong inv=0x40471547652b82feul, shift=0x4338000000000000ul;
    ulong kd=soft64_fma(inv,xd,shift);
    kd=soft64_sub(kd,shift);
    const ulong r=soft64_fma(inv,xd,kd^0x8000000000000000ul);
    const ulong scale=soft_gfk_exp2_core(kd,shift,0ul,0ul,0ul,0ul);
    const ulong z=soft64_fma(0x3ebc6af84b912394ul,r,0x3f2ebfce50fac4f3ul);
    const ulong r2=soft64_mul(r,r);
    ulong y=soft64_fma(0x3f962e42ff0c52d6ul,r,0x3ff0000000000000ul);
    y=soft64_mul(soft64_fma(z,r2,y),scale);
    return as_type<float>(soft64_to_f32_bits(y));
}

kernel void hexv842_mpas_prep_interfaces_f32(
    constant Hexv842MpasPrepInterfacesF32Resources &r [[buffer(0)]],
    constant Hexv842MpasPrepInterfacesF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float gravity = p.gravity;
    device const float *rho_zz = r.rho_zz;
    device const float *zz = r.zz;
    device const float *qv_p = r.qv_p;
    device const float *pres_p = r.pres_p;
    device const float *t_p = r.t_p;
    device const float *rho_p = r.rho_p;
    device const float *dz_p = r.dz_p;
    device const float *zgrid = r.zgrid;
    device const float *w = r.w;
    device float *w_p = r.w_p;
    device float *z_p = r.z_p;
    device float *t2_p = r.t2_p;
    device float *pres2_p = r.pres2_p;
    device float *pres_hyd_p = r.pres_hyd_p;
    device float *pres2_hyd_p = r.pres2_hyd_p;
    device float *pres_hydd_p = r.pres_hydd_p;
    device float *pres2_hydd_p = r.pres2_hydd_p;
    device float *psfc_p = r.psfc_p;
    device float *psfc_hyd_p = r.psfc_hyd_p;
    device float *psfc_hydd_p = r.psfc_hydd_p;
    device float *znu_p = r.znu_p;
    device float *znu_hyd_p = r.znu_hyd_p;
    device float *plrad = r.plrad;
    device int *invalid = r.invalid;

    const int cell = int(tid);
    if (cell >= ncells) return;
    for (int k = 0; k <= nlev; ++k) {
        const int i = k * ncells + cell;
        w_p[i] = w[i];
        z_p[i] = zgrid[i];
    }
    const float tem1 = mpas_sub(zgrid[ncells + cell], zgrid[cell]);
    const float tem2 = mpas_sub(zgrid[2 * ncells + cell], zgrid[ncells + cell]);
    const float rho1 = mpas_mul(
        mpas_mul(rho_zz[cell], zz[cell]), mpas_add(1.0f, qv_p[cell]));
    const float rho2 = mpas_mul(
        mpas_mul(rho_zz[ncells + cell], zz[ncells + cell]),
        mpas_add(1.0f, qv_p[ncells + cell]));
    const float correction = mpas_div(
        mpas_mul(mpas_mul(0.5f, mpas_sub(rho2, rho1)), tem1),
        mpas_add(tem1, tem2));
    const float hydro = mpas_mul(
        mpas_mul(mpas_mul(0.5f, gravity), tem1),
        mpas_sub(rho1, correction));
    psfc_p[cell] = mpas_add(mpas_add(hydro, r.pressure_p[cell]), r.pressure_base[cell]);
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        znu_p[i] = mpas_div(pres_p[i], psfc_p[cell]);
    }

    for (int k = 1; k < nlev; ++k) {
        const float inverse = mpas_div(
            1.0f, mpas_sub(zgrid[(k + 1) * ncells + cell],
                           zgrid[(k - 1) * ncells + cell]));
        const float fzm = mpas_mul(
            mpas_sub(zgrid[k * ncells + cell],
                     zgrid[(k - 1) * ncells + cell]), inverse);
        const float fzp = mpas_mul(
            mpas_sub(zgrid[(k + 1) * ncells + cell],
                     zgrid[k * ncells + cell]), inverse);
        const int i = k * ncells + cell;
        r.weights_minus[i] = fzm;
        r.weights_plus[i] = fzp;
        t2_p[i] = mpas_add(
            mpas_mul(fzm, t_p[i]), mpas_mul(fzp, t_p[i - ncells]));
        pres2_p[i] = mpas_add(
            mpas_mul(fzm, pres_p[i]), mpas_mul(fzp, pres_p[i - ncells]));
    }

    const int top = nlev * ncells + cell;
    float z0 = zgrid[top];
    float z1 = mpas_mul(0.5f, mpas_add(zgrid[top], zgrid[top - ncells]));
    float z2 = mpas_mul(
        0.5f, mpas_add(zgrid[top - ncells], zgrid[top - 2 * ncells]));
    float w1 = mpas_div(mpas_sub(z0, z2), mpas_sub(z1, z2));
    float w2 = mpas_sub(1.0f, w1);
    t2_p[top] = mpas_add(
        mpas_mul(w1, t_p[top - ncells]),
        mpas_mul(w2, t_p[top - 2 * ncells]));
    pres2_p[top] = mpas_v842_exp_fma(mpas_add(
        mpas_mul(w1, gfk_log(pres_p[top - ncells])),
        mpas_mul(w2, gfk_log(pres_p[top - 2 * ncells]))));

    z0 = zgrid[cell];
    z1 = mpas_mul(0.5f, mpas_add(zgrid[cell], zgrid[ncells + cell]));
    z2 = mpas_mul(
        0.5f, mpas_add(zgrid[ncells + cell], zgrid[2 * ncells + cell]));
    w1 = mpas_div(mpas_sub(z0, z2), mpas_sub(z1, z2));
    w2 = mpas_sub(1.0f, w1);
    t2_p[cell] = mpas_add(mpas_mul(w1, t_p[cell]),
                          mpas_mul(w2, t_p[ncells + cell]));
    pres2_p[cell] = mpas_add(mpas_mul(w1, pres_p[cell]),
                             mpas_mul(w2, pres_p[ncells + cell]));

    pres2_hyd_p[top] = pres2_p[top];
    pres2_hydd_p[top] = pres2_p[top];
    for (int k = nlev - 1; k >= 0; --k) {
        const int i = k * ncells + cell;
        const float rho_a = mpas_div(rho_p[i], mpas_add(1.0f, qv_p[i]));
        pres2_hyd_p[i] = mpas_add(
            pres2_hyd_p[i + ncells],
            mpas_mul(mpas_mul(gravity, rho_p[i]), dz_p[i]));
        pres2_hydd_p[i] = mpas_add(
            pres2_hydd_p[i + ncells],
            mpas_mul(mpas_mul(gravity, rho_a), dz_p[i]));
    }
    for (int k = nlev - 1; k >= 0; --k) {
        const int i = k * ncells + cell;
        pres_hyd_p[i] = mpas_mul(
            0.5f, mpas_add(pres2_hyd_p[i + ncells], pres2_hyd_p[i]));
        pres_hydd_p[i] = mpas_mul(
            0.5f, mpas_add(pres2_hydd_p[i + ncells], pres2_hydd_p[i]));
    }
    psfc_hyd_p[cell] = pres2_hyd_p[cell];
    psfc_hydd_p[cell] = pres2_hydd_p[cell];
    for (int k = nlev - 1; k >= 0; --k) {
        const int i = k * ncells + cell;
        znu_hyd_p[i] = mpas_div(pres_hyd_p[i], psfc_hyd_p[cell]);
    }
    plrad[cell] = pres2_p[top];
}
