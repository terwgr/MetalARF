// Hex v8.4.1 Metal transcription of CUDA kernel recover_pressure_f32 (woof/hex/cuda_backend/recovery.py).
#include <metal_stdlib>
using namespace metal;
#include "pressure_power.metalh"
#include "hexv841_recover_pressure_f32_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv841_recover_pressure_f32(
    constant Hexv841RecoverPressureF32Resources &r [[buffer(0)]],
    constant Hexv841RecoverPressureF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int index = int(tid);
    if (index >= p.nlev * p.ncells) return;
    const float rho = r.rho[index];
    const float rtheta = r.rtheta[index];
    const float rtheta_base = r.rtheta_base[index];
    const float zz = r.zz[index];
    const float rgas = p.rgas;
    r.theta[index] = rtheta / rho;
    const float argument = zz * (rgas / p.p0) * rtheta;
    // powf rounded once from binary64; native pow is biased low here.
    const float exner = fp32_pressure_power(argument, 1.0f, rgas / (p.cp - rgas));
    r.exner[index] = exner;
    r.pressure[index] = zz * rgas * rtheta * exner;
    r.rho_p[index] = rho - r.rho_base[index];
    const float rtheta_p = rtheta - rtheta_base;
    r.rtheta_p[index] = rtheta_p;
    r.pressure_p[index] = zz * rgas * (
        exner * rtheta_p + rtheta_base * (exner - r.exner_base[index]));
}
