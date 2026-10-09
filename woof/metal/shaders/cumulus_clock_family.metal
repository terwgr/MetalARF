#include <metal_stdlib>
#include "cumulus_clock_advance_bindings.metalh"
#include "cumulus_clock_clear_bindings.metalh"

using namespace metal;

// Mirrors the separate CuPy float32 operators in advance_ppt. The production
// Metal build disables contraction, preserving the source's multiply/add
// rounding boundaries.
#pragma clang fp contract(off)

kernel void cumulus_clock_advance(
    constant CumulusClockAdvanceResources &r [[buffer(0)]],
    constant CumulusClockAdvanceParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    const int c = int(gid);
    if (c >= p.ncol) return;

    const float pr = r.pratec[c];
    const float rain_increment = pr * p.dt;
    r.rainc[c] = r.rainc[c] + rain_increment;

    // CuPy maximum propagates NaN and emits its canonical quiet NaN.
    const float wet = isnan(pr) ? as_type<float>(uint(0x7fffffff))
                                : max(pr, 0.0f);
    const float increment = wet * p.dt;
    r.pending_rainbl[c] = r.pending_rainbl[c] + increment;
    if (p.has_surface_raincv != 0)
        r.surface_raincv[c] = r.surface_raincv[c] + increment;

    const float held = r.nca[c];
    const bool active = held > 0.0f;
    const float quotient = held / p.dt;
    const float rounded = quotient + 0.5f;
    const bool expires = active && (floor(rounded) <= 1.0f);
    r.expiring[c] = expires ? 1.0f : 0.0f;
    const float decremented = held - p.dt;
    r.nca[c] = active ? decremented : held;
}

kernel void cumulus_clock_clear(
    constant CumulusClockClearResources &r [[buffer(0)]],
    constant CumulusClockClearParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    const int c = int(gid);
    if (c >= p.ncol) return;

    // Keep masked elements untouched at the word level, including NaNs and
    // signed zero. The selected columns receive an ordinary +0.0f store.
    if (r.expiring[c] != 0.0f) {
        for (int k = 0; k < p.nz; ++k) {
            const ulong offset = ulong(k) * ulong(p.ncol) + ulong(c);
            if (p.nvolume > 0) r.v0[offset] = 0.0f;
            if (p.nvolume > 1) r.v1[offset] = 0.0f;
            if (p.nvolume > 2) r.v2[offset] = 0.0f;
            if (p.nvolume > 3) r.v3[offset] = 0.0f;
            if (p.nvolume > 4) r.v4[offset] = 0.0f;
            if (p.nvolume > 5) r.v5[offset] = 0.0f;
            if (p.nvolume > 6) r.v6[offset] = 0.0f;
            if (p.nvolume > 7) r.v7[offset] = 0.0f;
            if (p.nvolume > 8) r.v8[offset] = 0.0f;
            if (p.nvolume > 9) r.v9[offset] = 0.0f;
            if (p.nvolume > 10) r.v10[offset] = 0.0f;
            if (p.nvolume > 11) r.v11[offset] = 0.0f;
        }
    }
    r.expiring[c] = 0.0f;
}
