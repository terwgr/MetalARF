// Hex v8.4.1 Metal transcription of CUDA kernel recover_terrain_w_f32 (woof/hex/cuda_backend/recovery.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_recover_terrain_w_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
constant float mpas_third_order_denominator = 12.0f;


kernel void hexv841_recover_terrain_w_f32(
    constant Hexv841RecoverTerrainWF32Resources &r [[buffer(0)]],
    constant Hexv841RecoverTerrainWF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;
    const float cf1 = p.cf1;
    const float cf2 = p.cf2;
    const float cf3 = p.cf3;
    device const float *rho = r.rho;
    device const float *ru = r.ru;
    device const float *rw = r.rw;
    device const float *zz = r.zz;
    device const float *fzm = r.fzm;
    device const float *fzp = r.fzp;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const float *edge_sign_on_cell = r.edge_sign_on_cell;
    device const float *zb_cell = r.zb_cell;
    device const float *zb3_cell = r.zb3_cell;
    device float *w = r.w;
    const int element = int(tid);
    if (element >= ((nlev + 1) * ncells)) return;
    {

    const int cell = element % ncells;
    const int level = element / ncells;
    const int index = level * ncells + cell;
    if (level == nlev) {
      w[index] = 0.0f;
      return;
    }
    const int slots = n_edges_on_cell[cell];
    if (level == 0) {
      float value = 0.0f;
      for (int slot = 0; slot < slots; ++slot) {
        const int edge = edges_on_cell[cell * max_edges + slot];
        const float cell_sign = edge_sign_on_cell[cell * max_edges + slot];
        const float flux = mpas_add(mpas_add(
            mpas_mul(cf1, ru[edge]), mpas_mul(cf2, ru[nedges + edge])),
            mpas_mul(cf3, ru[2 * nedges + edge]));
        const int metric_index = cell * max_edges + slot;
        value = mpas_add(value, mpas_mul(mpas_mul(cell_sign,
            mpas_add(zb_cell[metric_index],
                mpas_mul(mpas_copysign(1.0f, flux), zb3_cell[metric_index]))), flux));
      }
      const float bottom_density = mpas_add(mpas_add(
          mpas_mul(cf1, rho[cell]), mpas_mul(cf2, rho[ncells + cell])),
          mpas_mul(cf3, rho[2 * ncells + cell]));
      w[index] = mpas_div(value, bottom_density);
      return;
    }
    const float metric = mpas_add(
        mpas_mul(fzm[level], zz[level * ncells + cell]),
        mpas_mul(fzp[level], zz[(level - 1) * ncells + cell]));
    float value = mpas_div(rw[index], metric);
    for (int slot = 0; slot < slots; ++slot) {
      const int edge = edges_on_cell[cell * max_edges + slot];
      const float cell_sign = edge_sign_on_cell[cell * max_edges + slot];
      const float flux = mpas_add(
          mpas_mul(fzm[level], ru[level * nedges + edge]),
          mpas_mul(fzp[level], ru[(level - 1) * nedges + edge]));
      const int metric_index = (level * ncells + cell) * max_edges + slot;
      value = mpas_add(value,
          mpas_mul(mpas_mul(cell_sign,
              mpas_add(zb_cell[metric_index],
                  mpas_mul(mpas_copysign(1.0f, flux), zb3_cell[metric_index]))),
              flux));
    }
    const float density = mpas_add(
        mpas_mul(fzm[level], rho[level * ncells + cell]),
        mpas_mul(fzp[level], rho[(level - 1) * ncells + cell]));
    w[index] = mpas_div(value, density);
    }
}
