// Hex v8.4.1 Metal transcription of CUDA kernel recover_wsm6_v841_f32 (woof/hex/cuda_physics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_recover_wsm6_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


kernel void hexv841_recover_wsm6_v841_f32(
    constant Hexv841RecoverWsm6V841F32Resources &r [[buffer(0)]],
    constant Hexv841RecoverWsm6V841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int count = p.count;
    const int nspecies = p.nspecies;
    const int nradius = p.nradius;
    const float rvord = p.rvord;
    device const float *rho = r.rho;
    device const float *theta = r.theta;
    device const float *water = r.water;
    device const float *radius = r.radius;
    device float *rho_theta = r.rho_theta;
    device float *scalars = r.scalars;
    device int *invalid = r.invalid;

    const int i = int(tid);
    if (i >= count) return;
    // water and radius are DEVICE tables sized by the scheme's row: its
    // species in model order, and the effective radii it publishes (WSM6
    // has three, P3 two -- its one ice category has no separate snow).
    if (!isfinite(rho[i]) || rho[i] <= 0.0f
            || !isfinite(theta[i]) || theta[i] <= 0.0f) atomicExch(invalid, 1);
    for (int s = 0; s < nspecies; ++s) {
        const float value = water[s * count + i];
        // MPAS mpas_atmphys_interface.F:914 writes the post-microphysics
        // carriers back to scalars RAW -- no clamp, no refusal.  The +0 clamp
        // native applies is the POST-TRANSPORT one, reproduced by
        // clamp_wsm6_v841_f32 at the next step boundary.  Refusing a negative
        // here is stricter than the authority; only non-finite is refused.
        if (!isfinite(value)) atomicExch(invalid, 1);
        scalars[s * count + i] = value;
    }
    for (int r = 0; r < nradius; ++r) {
        const float value = radius[r * count + i];
        if (!isfinite(value) || value < 0.0f) atomicExch(invalid, 1);
    }
    // MPAS couples theta_m with max(0.,qv) AT THE POINT OF USE, in this exact
    // expression: mpas_atmphys_interface.F:665, :778 and :1078 (the last is the
    // POST-microphysics coupling this kernel implements).  The sign-bit form is
    // used instead of fmaxf because the certified terminal-FTZ route is
    // DAZ-sensitive: it maps -0 and every negative subnormal to the native +0
    // bit pattern while leaving positive subnormals exactly alone.
    const uint qv_bits = as_type<uint>(water[0 * count + i]);
    const float qv_coupled =
        (qv_bits & 0x80000000u) != 0u ? 0.0f : water[0 * count + i];
    const float coeff = mpas_add(1.0f, mpas_mul(rvord, qv_coupled));
    const float result = mpas_mul(rho[i], mpas_mul(theta[i], coeff));
    if (!isfinite(coeff) || coeff <= 0.0f || !isfinite(result))
        atomicExch(invalid, 1);
    rho_theta[i] = result;
}
