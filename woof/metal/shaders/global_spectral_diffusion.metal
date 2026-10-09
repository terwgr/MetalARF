#include <metal_stdlib>
#include "global_spectral_diffusion_bindings.metalh"
#include "soft_binary64.metalh"

using namespace metal;

kernel void global_spectral_diffusion(
    constant GlobalSpectralDiffusionResources &r [[buffer(0)]],
    constant GlobalSpectralDiffusionParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.coefficient_count)) return;
    const uint width = uint(p.truncation) + 1u;
    const uint plane_count = width * width;
    // A coefficient buffer may contain one field or a contiguous stack of
    // fields (for example, every primitive-equation level). Degree and order
    // restart at each triangular spectral plane.
    const uint plane_tid = tid % plane_count;
    const uint degree = plane_tid / width;
    const uint order = plane_tid - degree * width;
    const ulong factor = r.factors[degree];
    const ulong zero = 0ul;
    ulong real_part = soft64_mul(r.coefficients[2u * tid], factor);
    ulong imag_part = soft64_mul(r.coefficients[2u * tid + 1u], factor);
    if (order > degree) {
        // Match the preview's multiply-then-triangular-mask ordering,
        // including NaN propagation from non-finite out-of-triangle values.
        real_part = soft64_mul(real_part, zero);
        imag_part = soft64_mul(imag_part, zero);
    } else if (order == 0u) {
        // transform.project() realifies the zonal (m=0) coefficient column.
        imag_part = zero;
    }
    r.output[2u * tid] = real_part;
    r.output[2u * tid + 1u] = imag_part;
}
