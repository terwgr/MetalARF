// Hex v8.4.1 Metal transcription of CUDA kernel recover_flat_w_f32 (woof/hex/cuda_backend/recovery.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_recover_flat_w_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_recover_flat_w_f32(
    constant Hexv841RecoverFlatWF32Resources &r [[buffer(0)]],
    constant Hexv841RecoverFlatWF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    device const float *rho = r.rho;
    device const float *rw = r.rw;
    device const float *zz = r.zz;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device float *w = r.w;
    const int element = int(tid);
    if (element >= ((nlev + 1) * ncells)) return;
    {

    const int cell = element % ncells;
    const int level = element / ncells;
    const int index = level * ncells + cell;
    if (level == 0 || level == nlev) {
      w[index] = 0.0f;
    } else {
      const float metric = mpas_add(
          mpas_mul(fzm[level], zz[level * ncells + cell]),
          mpas_mul(fzp[level], zz[(level - 1) * ncells + cell]));
      const float density = mpas_add(
          mpas_mul(fzm[level], rho[level * ncells + cell]),
          mpas_mul(fzp[level], rho[(level - 1) * ncells + cell]));
      w[index] = mpas_div(mpas_div(rw[index], metric), density);
    }
    }
}
