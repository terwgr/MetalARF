// Project a resident complex128 spectral field onto its triangular basis.
#include <metal_stdlib>
#include "global_spectral_project_fragment_bindings.metalh"

using namespace metal;

kernel void global_spectral_project_fragment(
    constant GlobalSpectralProjectFragmentResources &r [[buffer(0)]],
    constant GlobalSpectralProjectFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;
    const uint mode = tid % uint(p.ndegree * p.nfreq);
    const uint degree = mode / uint(p.nfreq);
    const uint order = mode % uint(p.nfreq);
    const ulong index = ulong(tid) * 2ul;
    if (order > degree) {
        r.output[index] = 0ul;
        r.output[index + 1ul] = 0ul;
    } else {
        r.output[index] = r.input[index];
        r.output[index + 1ul] = order == 0u ? 0ul : r.input[index + 1ul];
    }
}
