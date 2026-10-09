// Resident vector Fourier weighting and vorticity/divergence assembly.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_vector_analysis_fragment_bindings.metalh"
using namespace metal;

kernel void global_semilag_vector_weights_fragment(
    constant GlobalSemilagVectorWeightsFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagVectorWeightsFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const uint mode = tid % uint(p.nfreq);
    const uint latitude = (tid / uint(p.nfreq)) % uint(p.nlat);
    const uint weight = latitude * uint(p.nfreq) + mode;
    const ulong a = r.values[2 * tid];
    const ulong b = r.values[2 * tid + 1];
    const ulong c = r.rotation_weight[2 * weight];
    const ulong d = r.rotation_weight[2 * weight + 1];
    const ulong s = r.shear_weight[weight];
    r.rotated[2 * tid] = soft64_sub(soft64_mul(a, c), soft64_mul(b, d));
    r.rotated[2 * tid + 1] = soft64_add(soft64_mul(a, d), soft64_mul(b, c));
    r.sheared[2 * tid] = soft64_mul(a, s);
    r.sheared[2 * tid + 1] = soft64_mul(b, s);
}

kernel void global_semilag_vordiv_combine_fragment(
    constant GlobalSemilagVordivCombineFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagVordivCombineFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlevel) * uint(p.ndegree) * uint(p.mcount);
    if (tid >= count) return;
    const uint mode = tid % uint(p.mcount);
    const uint degree = (tid / uint(p.mcount)) % uint(p.ndegree);
    if (mode > degree) {
        r.divergence[2 * tid] = 0ul;
        r.divergence[2 * tid + 1] = 0ul;
        r.vorticity[2 * tid] = 0ul;
        r.vorticity[2 * tid + 1] = 0ul;
        return;
    }
    const uint offset = count;
    const ulong d_real = soft64_sub(r.rotated[2 * tid], r.sheared[2 * (offset + tid)]);
    const ulong d_imag = soft64_sub(r.rotated[2 * tid + 1], r.sheared[2 * (offset + tid) + 1]);
    const ulong z_real = soft64_add(r.rotated[2 * (offset + tid)], r.sheared[2 * tid]);
    const ulong z_imag = soft64_add(r.rotated[2 * (offset + tid) + 1], r.sheared[2 * tid + 1]);
    r.divergence[2 * tid] = d_real;
    r.divergence[2 * tid + 1] = mode == 0 ? 0ul : d_imag;
    r.vorticity[2 * tid] = z_real;
    r.vorticity[2 * tid + 1] = mode == 0 ? 0ul : z_imag;
}
