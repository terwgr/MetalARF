// Hex v8.4.1 Metal transcription of CUDA kernel prep_noahmp_sounding_v841_f32 (woof/hex/cuda_physics_prep_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_prep_noahmp_sounding_v841_f32_bindings.metalh"
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


kernel void hexv841_prep_noahmp_sounding_v841_f32(
    constant Hexv841PrepNoahmpSoundingV841F32Resources &r [[buffer(0)]],
    constant Hexv841PrepNoahmpSoundingV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float rvord = p.rvord;
    device const float *raw_qv = r.raw_qv;
    device const float *theta_m = r.theta_m;
    device const float *exner = r.exner;
    device const float *pressure_base = r.pressure_base;
    device const float *pressure_p = r.pressure_p;
    device const float *zgrid = r.zgrid;
    device const float *u_p = r.u_p;
    device const float *v_p = r.v_p;
    device float *dz8w = r.dz8w;
    device float *qv_curr = r.qv_curr;
    device float *t_phy = r.t_phy;
    device float *u_phy = r.u_phy;
    device float *v_phy = r.v_phy;
    device float *p8w = r.p8w;
    device int *invalid = r.invalid;

    const int cell = int(tid);
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        const float qraw = raw_qv[i];
        const float thickness = mpas_sub(
            zgrid[i + ncells], zgrid[i]);
        const float denominator = mpas_add(
            1.0f, mpas_mul(rvord, qraw));
        const float temperature = mpas_mul(
            mpas_div(theta_m[i], denominator), exner[i]);
        dz8w[i] = thickness;
        qv_curr[i] = qraw;
        t_phy[i] = temperature;
        u_phy[i] = u_p[i];
        v_phy[i] = v_p[i];
        if (!isfinite(qraw) || !isfinite(temperature)
                || temperature <= 0.0f || !isfinite(thickness)
                || thickness <= 0.0f || !isfinite(u_p[i])
                || !isfinite(v_p[i]))
            atomicExch(invalid, 1);
    }

    float z0 = zgrid[cell];
    float z1 = mpas_mul(
        0.5f, mpas_add(zgrid[cell], zgrid[ncells + cell]));
    float z2 = mpas_mul(
        0.5f, mpas_add(zgrid[ncells + cell], zgrid[2 * ncells + cell]));
    float w1 = mpas_div(mpas_sub(z0, z2), mpas_sub(z1, z2));
    float w2 = mpas_sub(1.0f, w1);
    float totm = mpas_add(pressure_p[cell], pressure_base[cell]);
    float totp = mpas_add(
        pressure_p[ncells + cell], pressure_base[ncells + cell]);
    p8w[cell] = mpas_add(mpas_mul(w1, totm), mpas_mul(w2, totp));
    for (int k = 1; k < nlev; ++k) {
        const int i = k * ncells + cell;
        totm = mpas_add(pressure_p[i - ncells], pressure_base[i - ncells]);
        totp = mpas_add(pressure_p[i], pressure_base[i]);
        const float mult = mpas_div(
            1.0f,
            mpas_sub(zgrid[i + ncells], zgrid[i - ncells]));
        const float fzm = mpas_mul(
            mult, mpas_sub(zgrid[i], zgrid[i - ncells]));
        const float fzp = mpas_mul(
            mult, mpas_sub(zgrid[i + ncells], zgrid[i]));
        p8w[i] = mpas_add(mpas_mul(fzm, totp), mpas_mul(fzp, totm));
    }
    for (int k = 0; k < nlev; ++k) {
        const float value = p8w[k * ncells + cell];
        if (!isfinite(value) || value <= 0.0f)
            atomicExch(invalid, 1);
    }
}
