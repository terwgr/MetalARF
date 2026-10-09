// needs integration: FP64 periodic zonal sweep for Global grid tracers.
//
// This is the sub-step operation in GridTracerTransport._periodic_step.
// The host computes each latitude band's sub-step count and invokes this
// kernel for each sub-step. The ABI owner must register the fragment before
// dispatch. Arrays are contiguous FP64 bit patterns in [tracer, level, lat,
// lon] order for mass and [level, lat, lon] order for density/psi.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_tracer_zonal_fragment_bindings.metalh"

using namespace metal;

static inline ulong global_harmonic_fragment(ulong left, ulong right) {
    const ulong product = soft64_mul(left, right);
    if (!soft64_greater(product, 0ul)) return 0ul;
    return soft64_div(soft64_mul(soft64_from_u32(2u), product),
                      soft64_add(left, right));
}

static inline ulong global_min_fragment(ulong left, ulong right) {
    return soft64_less(left, right) ? left : right;
}

static inline ulong global_max_fragment(ulong left, ulong right) {
    return soft64_greater(left, right) ? left : right;
}

static inline ulong global_zonal_face_flux_fragment(
    device const ulong *mass, device const ulong *density,
    device const ulong *psi, uint tracer, uint cells, uint row_level,
    uint longitude, uint nlongitude, ulong dt) {
    const uint previous = row_level + (longitude + nlongitude - 1u) % nlongitude;
    const uint current = row_level + longitude;
    const uint next = row_level + (longitude + 1u) % nlongitude;
    const uint next_next = row_level + (longitude + 2u) % nlongitude;
    const uint idx = tracer * cells;
    const ulong q = soft64_div(mass[idx + current], density[current]);
    const ulong q_next = soft64_div(mass[idx + next], density[next]);
    const ulong q_previous = soft64_div(mass[idx + previous], density[previous]);
    const ulong q_next_next = soft64_div(mass[idx + next_next], density[next_next]);
    const ulong dq_previous = soft64_sub(q, q_previous);
    const ulong dq = soft64_sub(q_next, q);
    const ulong dq_next = soft64_sub(q_next_next, q_next);
    const ulong slope = global_harmonic_fragment(dq_previous, dq);
    const ulong slope_next = global_harmonic_fragment(dq, dq_next);
    const ulong flux_dt = soft64_mul(psi[current], dt);
    const ulong c_lo = soft64_div(flux_dt, density[current]);
    const ulong c_hi = soft64_div(soft64_negate_bits(flux_dt), density[next]);
    const ulong half_bits = 0x3fe0000000000000ul;
    const ulong one = 0x3ff0000000000000ul;
    const ulong w_lo = soft64_mul(half_bits, soft64_sub(one, c_lo));
    const ulong w_hi = soft64_mul(soft64_negate_bits(half_bits), soft64_sub(one, c_hi));
    const bool eastward = soft64_greater_equal(psi[current], 0ul);
    ulong face = eastward
        ? soft64_add(q, soft64_mul(slope, w_lo))
        : soft64_add(q_next, soft64_mul(slope_next, w_hi));
    const ulong q_low = global_min_fragment(q, q_next);
    const ulong q_high = global_max_fragment(q, q_next);
    const ulong q_up = eastward ? q : q_next;
    face = global_min_fragment(global_max_fragment(face, q_low),
        global_min_fragment(q_high, soft64_mul(soft64_from_u32(2u), q_up)));
    return soft64_mul(flux_dt, face);
}

kernel void global_tracer_zonal_fragment(
    constant GlobalTracerZonalFragmentResources &r [[buffer(0)]],
    constant GlobalTracerZonalFragmentParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint cells = q.nlevel * q.nlatitude * q.nlongitude;
    if (tid >= cells) return;
    const uint longitude = tid % q.nlongitude;
    const uint row_level = tid - longitude;
    const uint previous = row_level + (longitude + q.nlongitude - 1u) % q.nlongitude;
    const ulong flux_dt = soft64_mul(r.psi[tid], r.dt[0]);
    const ulong previous_flux_dt = soft64_mul(r.psi[previous], r.dt[0]);
    r.new_density[tid] = soft64_sub(r.density[tid],
        soft64_sub(flux_dt, previous_flux_dt));
    for (uint tracer = 0u; tracer < q.ntracer; ++tracer) {
        const ulong flux = global_zonal_face_flux_fragment(
            r.mass, r.density, r.psi, tracer, cells, row_level, longitude,
            q.nlongitude, r.dt[0]);
        const ulong previous_flux = global_zonal_face_flux_fragment(
            r.mass, r.density, r.psi, tracer, cells, row_level,
            (longitude + q.nlongitude - 1u) % q.nlongitude, q.nlongitude, r.dt[0]);
        const uint index = tracer * cells + tid;
        r.new_mass[index] = soft64_sub(r.mass[index],
            soft64_sub(flux, previous_flux));
    }
}
