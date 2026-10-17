// Hex v8.4.1 Metal transcription of CUDA kernel recover_cells_f32 (woof/hex/cuda_driver.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "pressure_power.metalh"
#include "hexv841_recover_cells_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_recover_cells_f32(
    constant Hexv841RecoverCellsF32Resources &r [[buffer(0)]],
    constant Hexv841RecoverCellsF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int stage = p.stage;
    const float rgas = p.rgas;
    const float cp = p.cp;
    const float p0 = p.p0;
    device const float *rho_p_saved = r.rho_p_saved;
    device const float *rtheta_p_saved = r.rtheta_p_saved;
    device const float *rho_pp = r.rho_pp;
    device const float *rtheta_pp = r.rtheta_pp;
    device const float *rho_base = r.rho_base;
    device const float *rtheta_base = r.rtheta_base;
    device const float *exner_base = r.exner_base;
    device const float *zz = r.zz;
    device const float *exner_saved = r.exner_saved;
    device const float *pressure_p_saved = r.pressure_p_saved;
    device float *rho = r.rho;
    device float *rtheta = r.rtheta;
    device float *rho_p = r.rho_p;
    device float *rtheta_p = r.rtheta_p;
    device float *theta = r.theta;
    device float *exner = r.exner;
    device float *pressure_p = r.pressure_p;
    const int element = int(tid);
    if (element >= (nlev * ncells)) return;
    {

        const int index = element;
        const float density_p = mpas_add(
            rho_p_saved[index], rho_pp[index]);
        const float theta_p = mpas_add(
            rtheta_p_saved[index], rtheta_pp[index]);
        const float full_rho = mpas_add(density_p, rho_base[index]);
        const float full_rtheta = mpas_add(theta_p, rtheta_base[index]);
        rho_p[index] = density_p;
        rtheta_p[index] = theta_p;
        rho[index] = full_rho;
        rtheta[index] = full_rtheta;
        theta[index] = mpas_div(full_rtheta, full_rho);
        if (stage == 3) {
            const float argument = mpas_mul(mpas_mul(
                zz[index], mpas_div(rgas, p0)), full_rtheta);
            // Native pow is biased low on the Exner range; evaluate powf once in
            // binary64 and round, as the other Metal pressure paths do.
            const float value = fp32_pressure_power(argument, 1.0f, rgas / (cp - rgas));
            exner[index] = value;
            pressure_p[index] = mpas_mul(mpas_mul(zz[index], rgas),
                mpas_add(mpas_mul(value, theta_p),
                    mpas_mul(rtheta_base[index],
                        mpas_sub(value, exner_base[index]))));
        } else {
            exner[index] = exner_saved[index];
            pressure_p[index] = pressure_p_saved[index];
        }
    }
}
