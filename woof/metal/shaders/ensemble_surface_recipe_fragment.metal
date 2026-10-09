// Seeded surface-state member controls on Metal.
#include <metal_stdlib>
#include "ensemble_surface_recipe_fragment_bindings.metalh"
using namespace metal;

inline void surface_mark_bad(device uint *bad, uint bits) {
    atomic_fetch_or_explicit((device atomic_uint *)bad, bits,
                             memory_order_relaxed);
}

kernel void ensemble_surface_validate_soil(
    constant EnsembleSurfaceValidateSoilResources &r [[buffer(0)]],
    constant EnsembleSurfaceValidateSoilParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.cells)) return;
    uint column = tid % uint(p.columns);
    if (!(r.land[column] > 0.5f) || r.ice[column] != 0.0f) return;
    float total = r.total[tid];
    float liquid = r.liquid[tid];
    float frozen = r.frozen[tid];
    if (!isfinite(total) || total < 0.0f || total > p.maximum_total ||
        !isfinite(liquid) || liquid < 0.0f || liquid > total ||
        (p.has_frozen && (!isfinite(frozen) || frozen < 0.0f ||
                          frozen > p.frozen_limit))) {
        surface_mark_bad(r.bad, 1u);
    }
}

kernel void ensemble_surface_scale_soil(
    constant EnsembleSurfaceScaleSoilResources &r [[buffer(0)]],
    constant EnsembleSurfaceScaleSoilParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.cells)) return;
    uint column = tid % uint(p.columns);
    if (!(r.land[column] > 0.5f) || r.ice[column] != 0.0f) return;
    float scale = r.parameters[0];
    if (scale == 1.0f) return;
    float old_total = r.total[tid];
    float scaled = old_total * scale;
    float floor_value = 0.0f;
    if (p.floor_categories > 0) {
        int category = r.soil_type[column] - 1;
        floor_value = (category >= 0 && category < p.floor_categories &&
                       r.dry[category] > 0.0f) ? r.dry[category] : p.fallback_floor;
    }
    float updated = max(floor_value, min(p.maximum_total, scaled));
    float factor = updated == scaled ? scale :
        (old_total > 0.0f ? updated / old_total : 1.0f);
    float old_liquid = r.liquid[tid];
    float old_frozen = p.has_frozen ? r.frozen[tid] : 0.0f;
    r.total[tid] = updated;
    r.liquid[tid] = min(updated, old_liquid * factor);
    if (p.has_frozen) r.frozen[tid] = old_frozen * factor;
}

kernel void ensemble_surface_ruc_availability(
    constant EnsembleSurfaceRucAvailabilityResources &r [[buffer(0)]],
    constant EnsembleSurfaceRucAvailabilityParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.columns)) return;
    if (!(r.land[tid] > 0.5f) || r.ice[tid] != 0.0f ||
        r.parameters[0] == 1.0f) return;
    int category = r.soil_type[tid] - 1;
    if (category < 0 || category >= p.categories ||
        r.reference[category] <= r.dry[category]) {
        surface_mark_bad(r.bad, 4u);
        return;
    }
    if (p.apply) {
        float numerator = r.total[tid] - r.dry[category];
        float denominator = r.reference[category] - r.dry[category];
        r.availability[tid] = max(p.availability_floor,
            min(p.availability_ceiling, numerator / denominator));
    }
}

kernel void ensemble_surface_offset_sst(
    constant EnsembleSurfaceOffsetSstResources &r [[buffer(0)]],
    constant EnsembleSurfaceOffsetSstParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.columns)) return;
    if (!(r.land[tid] < 0.5f) || !(r.lake[tid] < 0.5f) ||
        r.ice[tid] != 0.0f) return;
    float offset = r.parameters[1];
    if (offset == 0.0f) return;
    float value = r.temperature[tid] + offset;
    if (!isfinite(value) || value < p.minimum_temperature ||
        value > p.maximum_temperature) {
        surface_mark_bad(r.bad, 2u);
        return;
    }
    if (p.apply) r.temperature[tid] = value;
}
