#include <metal_stdlib>
#include "noah_forcing_bindings.metalh"
#include "noah_rain_clear_bindings.metalh"
#include "noah_rain_add_bindings.metalh"
#include "noah_sr_from_temperature_bindings.metalh"

using namespace metal;

// Preserve the CuPy statement boundaries in the forcing prologue. In
// particular, the pressure sum is rounded before multiplication by 0.5f.
#pragma clang fp contract(off)

kernel void noah_forcing(
    constant NoahForcingResources &r [[buffer(0)]],
    constant NoahForcingParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const int c = int(flat);
    if (c >= p.ncol) return;

    r.sfcprs_a[c] = 0.5f * (r.p_if0_a[c] + r.p_if1_a[c]);
    const float t = r.temperature0_a[c];
    r.sfctmp_a[c] = t;
    r.qv1_a[c] = r.qv0_a[c];
    r.dz8w1_a[c] = r.dz0_a[c];
    r.rib_a[c] = r.br_a[c];
    r.sr_a[c] = p.use_scheme_sr != 0
        ? r.sr_scheme_a[c] : ((t <= p.freezing) ? 1.0f : 0.0f);
    r.rainbl_a[c] = r.rainbl_a[c] + r.pending_a[c];
}

kernel void noah_rain_clear(
    constant NoahRainClearResources &r [[buffer(0)]],
    constant NoahRainClearParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const int c = int(flat);
    if (c >= p.ncol) return;
    r.rainbl_a[c] = 0.0f;
    r.pending_a[c] = 0.0f;
}

kernel void noah_rain_add(
    constant NoahRainAddResources &r [[buffer(0)]],
    constant NoahRainAddParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const int c = int(flat);
    if (c >= p.ncol) return;
    r.rainbl_a[c] = r.rainbl_a[c] + r.pending_a[c];
}

kernel void noah_sr_from_temperature(
    constant NoahSrFromTemperatureResources &r [[buffer(0)]],
    constant NoahSrFromTemperatureParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const int c = int(flat);
    if (c >= p.ncol) return;
    r.sr_a[c] = r.temperature_a[c] <= p.freezing ? 1.0f : 0.0f;
}
