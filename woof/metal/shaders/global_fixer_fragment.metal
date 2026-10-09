// needs integration: FP64 Global column positivity and spectral mass fixers.
//
// The ABI owner must register this source before a Global caller can
// dispatch it. Reductions that produce the mass offset remain host-side.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_fixer_fragment_bindings.metalh"

using namespace metal;

static inline ulong global_fixer_min(ulong a, ulong b) {
    return soft64_less(a, b) ? a : b;
}

static inline ulong global_fixer_max(ulong a, ulong b) {
    return soft64_greater(a, b) ? a : b;
}

// One thread owns one horizontal column. qv/dp use [level, column]
// contiguous layout, and the level sum follows the NumPy axis-zero order.
kernel void global_column_positivity_fragment(
    constant GlobalColumnPositivityFragmentResources &r [[buffer(0)]],
    constant GlobalColumnPositivityFragmentParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(q.ncolumn)) return;
    ulong negative = 0ul;
    ulong positive = 0ul;
    for (uint level = 0u; level < q.nlevel; ++level) {
        const uint index = level * q.ncolumn + tid;
        const ulong weighted = soft64_mul(r.qv[index], r.dp[index]);
        negative = soft64_add(negative, global_fixer_min(weighted, 0ul));
        positive = soft64_add(positive, global_fixer_max(weighted, 0ul));
    }
    const ulong net = soft64_add(positive, negative);
    const bool fillable = soft64_greater(positive, 0ul)
        && soft64_greater(net, 0ul);
    const ulong scale = fillable
        ? soft64_div(net, positive)
        : 0x3ff0000000000000ul;
    r.negative_mass[tid] = negative;
    r.unfillable_mass[tid] = fillable ? 0ul : negative;
    r.rescale[tid] = soft64_sub(0x3ff0000000000000ul, scale);
    for (uint level = 0u; level < q.nlevel; ++level) {
        const uint index = level * q.ncolumn + tid;
        r.repaired_qv[index] = soft64_mul(global_fixer_max(r.qv[index], 0ul), scale);
    }
}

// Add the already-reduced uniform log-pressure correction at spectral (0,0).
// Coefficients are [degree, order, real/imag] binary64 words; the caller
// supplies correction*sqrt(4*pi), matching add_grid_constant().
kernel void global_spectral_mass_offset_fragment(
    constant GlobalSpectralMassOffsetFragmentResources &r [[buffer(0)]],
    constant GlobalSpectralMassOffsetFragmentParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) return;
    r.coefficients[0] = soft64_add(r.coefficients[0], r.offset[0]);
}
