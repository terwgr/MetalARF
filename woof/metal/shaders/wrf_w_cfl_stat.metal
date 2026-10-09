// WRF vertical/horizontal CFL fold for the adaptive clock: Metal twin of
// w_cfl_stat in woof/core/kernels/openbc.cu (whole-domain form).
// out[0] = max vert_cfl bits, out[1] = cells above w_damp onset,
// out[2] = cells folded, out[3] = max horiz_cfl bits.
// The vertical CFL uses soft32 RN ops like the CUDA __f*_rn chain. Threads
// past the tail carry no cell instead of returning, because every thread of
// the threadgroup must reach the barriers.
#include <metal_stdlib>
#include "wrf_w_cfl_stat_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;


static inline float wcfl_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float wcfl_mul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}
static inline float wcfl_div(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}

kernel void wrf_w_cfl_stat(
    constant WrfWCflStatResources &r [[buffer(0)]],
    constant WrfWCflStatParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]])
{
    threadgroup atomic_uint block_max;
    threadgroup atomic_uint block_hmax;
    threadgroup atomic_uint block_hits;
    threadgroup atomic_uint block_count;
    if (lid == 0u) {
        atomic_store_explicit(&block_max, 0u, memory_order_relaxed);
        atomic_store_explicit(&block_hmax, 0u, memory_order_relaxed);
        atomic_store_explicit(&block_hits, 0u, memory_order_relaxed);
        atomic_store_explicit(&block_count, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nz = p.nz, ny = p.ny, nx = p.nx;
    const uint plane = uint(ny) * uint(nx);
    if (gid < uint(nz - 1) * plane) {
        const uint k = gid / plane + 1u;
        const uint c = gid - (k - 1u) * plane;
        const uint j = c / uint(nx), i = c - j * uint(nx);
        const float m = wcfl_add(wcfl_mul(r.c1f[k], wcfl_add(r.mub2d[c], r.mup[c])), r.c2f[k]);
        const float vert = fabs(wcfl_mul(wcfl_mul(wcfl_div(r.ww[k * plane + c], m), r.rdnw[k]), p.dt));
        const uint iu = k * uint(ny) * uint(nx + 1) + j * uint(nx + 1) + i;
        const uint iv = k * uint(ny + 1) * uint(nx) + j * uint(nx) + i;
        const float cu = fabs(wcfl_mul(wcfl_mul(wcfl_mul(r.u[iu], p.rdx), r.msfu[j * uint(nx + 1) + i]), p.dt));
        const float cv = fabs(wcfl_mul(wcfl_mul(wcfl_mul(r.v[iv], p.rdy), r.msfv[j * uint(nx) + i]), p.dt));
        atomic_fetch_max_explicit(&block_max, as_type<uint>(vert), memory_order_relaxed);
        atomic_fetch_max_explicit(&block_hmax, as_type<uint>(fmax(cu, cv)), memory_order_relaxed);
        if (vert > p.w_damp_on)
            atomic_fetch_add_explicit(&block_hits, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&block_count, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device atomic_uint *out = reinterpret_cast<device atomic_uint *>(r.out);
    if (lid == 0u) {
        atomic_fetch_max_explicit(out + 0, atomic_load_explicit(&block_max, memory_order_relaxed), memory_order_relaxed);
        atomic_fetch_add_explicit(out + 1, atomic_load_explicit(&block_hits, memory_order_relaxed), memory_order_relaxed);
        atomic_fetch_add_explicit(out + 2, atomic_load_explicit(&block_count, memory_order_relaxed), memory_order_relaxed);
        atomic_fetch_max_explicit(out + 3, atomic_load_explicit(&block_hmax, memory_order_relaxed), memory_order_relaxed);
    }

}
