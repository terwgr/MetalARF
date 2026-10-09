// Resident FP64 hybrid-pressure setup for the Global semi-Lagrangian path.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "global_semilag_pressure_fragment_bindings.metalh"

using namespace metal;

kernel void global_semilag_pressure_fragment(
    constant GlobalSemilagPressureFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagPressureFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncolumn = uint(p.ncolumn);
    const uint pressure_count = ncolumn * uint(p.nlevel + 1);
    if (tid >= pressure_count) return;
    const uint level = tid / ncolumn;
    const uint column = tid - level * ncolumn;
    // FP64 exp: an f32 exp put ~4e-7 relative error into ps, which the
    // mass fixer then "corrected" into real dry-mass drift.
    const ulong surface_pressure = soft64_plm_exp(r.log_surface_pressure[column]);
    if (level == 0u) r.surface_pressure[column] = surface_pressure;
    const ulong upper = soft64_add(
        r.a_half[level], soft64_mul(r.b_half[level], surface_pressure));
    r.pressure_half[tid] = upper;
    if (level < uint(p.nlevel)) {
        const ulong lower = soft64_add(
            r.a_half[level + 1],
            soft64_mul(r.b_half[level + 1], surface_pressure));
        const uint layer = level * ncolumn + column;
        r.layer_thickness[layer] = soft64_sub(lower, upper);
        r.pressure_full[layer] = soft64_sqrt(soft64_mul(upper, lower));
        r.ln_ratio[layer] = soft64_plm_log(soft64_div(lower, upper));
    }
}
