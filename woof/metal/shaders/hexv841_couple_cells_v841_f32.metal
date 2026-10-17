// Hex v8.4.1 Metal transcription of CUDA kernel couple_cells_v841_f32 (woof/hex/cuda_physics_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_couple_cells_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)


kernel void hexv841_couple_cells_v841_f32(
    constant Hexv841CoupleCellsV841F32Resources &r [[buffer(0)]],
    constant Hexv841CoupleCellsV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nspecies = p.nspecies;
    const float rvord = p.rvord;
    device const float *rho = r.rho;
    device const float *rho_theta = r.rho_theta;
    device const float *q = r.q;
    device const float *dtheta = r.dtheta;
    device const float *rates = r.rates;
    device float *out_rho = r.out_rho;
    device float *out_rtheta = r.out_rtheta;
    device float *out_q = r.out_q;
    device int *invalid = r.invalid;

    const int cell = int(tid);
    if (cell >= ncells) return;
    const int plane = nlev * ncells;
    // rates is a DEVICE table, one entry per species in the row's order.
    // A species the phase-one seam publishes no tendency for (P3 carries
    // ni, nr, qir and qib, which change only in phase two) points at a
    // shared zero buffer, so the coupling stays one straight loop and the
    // species still gets its coupled slot written.
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        const float mass = rho[i];
        const float rt = rho_theta[i];
        const float qt = q[i];
        const float th_rate = dtheta[i];
        if (!isfinite(mass) || mass <= 0.0f || !isfinite(rt)
                || !isfinite(qt) || !isfinite(th_rate)) atomicExch(invalid, 1);
        out_rho[i] = 0.0f;
        for (int s = 0; s < nspecies; ++s) {
            const float rate = rates[s * plane + i];
            if (!isfinite(rate)) atomicExch(invalid, 1);
            out_q[s * plane + i] = mpas_mul(mass, rate);
        }
        const float coeff = mpas_add(1.0f, mpas_mul(rvord, qt));
        const float theta_m = mpas_div(rt, mass);
        const float tend_th = mpas_mul(mass, th_rate);
        const float first = mpas_mul(coeff, tend_th);
        const float vapor = mpas_div(
            mpas_mul(mpas_mul(rvord, theta_m), out_q[i]), coeff);
        const float result = mpas_add(first, vapor);
        if (!isfinite(coeff) || coeff <= 0.0f || !isfinite(result))
            atomicExch(invalid, 1);
        out_rtheta[i] = result;
    }
}
