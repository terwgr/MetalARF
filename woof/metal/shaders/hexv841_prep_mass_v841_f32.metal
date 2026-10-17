// Hex v8.4.1 Metal transcription of CUDA kernel prep_mass_v841_f32 (woof/hex/cuda_physics_prep_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_prep_mass_v841_f32_bindings.metalh"
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


kernel void hexv841_prep_mass_v841_f32(
    constant Hexv841PrepMassV841F32Resources &r [[buffer(0)]],
    constant Hexv841PrepMassV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nspecies = p.nspecies;
    const float rvord = p.rvord;
    device const float *rho_zz = r.rho_zz;
    device const float *theta_m = r.theta_m;
    device const float *scalars = r.scalars;
    device const float *exner = r.exner;
    device const float *pressure_base = r.pressure_base;
    device const float *pressure_p = r.pressure_p;
    device const float *zgrid = r.zgrid;
    device const float *zz = r.zz;
    device float *qout = r.qout;
    device float *zz_p = r.zz_p;
    device float *rho_dry = r.rho_dry;
    device float *rho_p = r.rho_p;
    device float *th_p = r.th_p;
    device float *t_p = r.t_p;
    device float *pi_p = r.pi_p;
    device float *pres_p = r.pres_p;
    device float *zmid_p = r.zmid_p;
    device float *dz_p = r.dz_p;
    device int *invalid = r.invalid;

    const int cell = int(tid);
    if (cell >= ncells) return;
    const int plane = nlev * ncells;
    // qout is a DEVICE table of per-species output pointers, sized by the
    // scheme's species row, in the row's order.  It replaces six welded
    // parameters; the arithmetic below is unchanged, which is what the
    // WSM6 byte-identity arm measures.  Slot 0 is water vapour by the row's
    // own invariant -- the moist density, the dry-theta conversion, the
    // NoahMP sounding and the interface extrapolation all read it.
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        for (int s = 0; s < nspecies; ++s) {
            const float raw = scalars[s * plane + i];
            if (!prep_finite_bits_v841(raw)) atomicExch(invalid, 1);
            qout[s * plane + i] = prep_positive_q_v841(raw);
        }
        const float qv = qout[0 * plane + i];
        zz_p[i] = zz[i];
        const float dry_rho = mpas_mul(zz[i], rho_zz[i]);
        rho_dry[i] = dry_rho;
        rho_p[i] = mpas_mul(dry_rho, mpas_add(1.0f, qv));
        const float denominator = mpas_add(1.0f, mpas_mul(rvord, qv));
        th_p[i] = mpas_div(theta_m[i], denominator);
        t_p[i] = mpas_mul(th_p[i], exner[i]);
        pi_p[i] = exner[i];
        pres_p[i] = mpas_add(pressure_p[i], pressure_base[i]);
        const float bottom = zgrid[k * ncells + cell];
        const float top = zgrid[(k + 1) * ncells + cell];
        zmid_p[i] = mpas_mul(0.5f, mpas_add(top, bottom));
        dz_p[i] = mpas_sub(top, bottom);
        if (!isfinite(rho_zz[i]) || rho_zz[i] <= 0.0f
                || !isfinite(theta_m[i]) || theta_m[i] <= 0.0f
                || !isfinite(zz[i]) || !isfinite(exner[i])
                || !isfinite(pressure_p[i]) || !isfinite(pressure_base[i]))
            atomicExch(invalid, 1);
    }
}
