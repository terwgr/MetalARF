// needs integration: FP64 walled meridional/vertical tracer transport step.
//
// One launch advances one sub-step of the shared monotone walled sweep in
// GridTracerTransport._walled_step. The host chooses the axis, coordinates and
// sub-step count. Layout is [outer, axis, inner] for density/psi and
// [tracer, outer, axis, inner] for mass. The ABI owner must register it.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_tracer_walled_fragment_bindings.metalh"

using namespace metal;

static inline ulong global_walled_harmonic(ulong left, ulong right) {
    const ulong product = soft64_mul(left, right);
    if (!soft64_greater(product, 0ul)) return 0ul;
    return soft64_div(soft64_mul(soft64_from_u32(2u), product),
                      soft64_add(left, right));
}

static inline ulong global_walled_min(ulong a, ulong b) {
    return soft64_less(a, b) ? a : b;
}

static inline ulong global_walled_max(ulong a, ulong b) {
    return soft64_greater(a, b) ? a : b;
}

static inline ulong global_walled_face_flux(
    device const ulong *mass, device const ulong *density,
    device const ulong *psi, device const ulong *centre,
    device const ulong *face, device const ulong *inv_width,
    uint tracer, uint ntracer, uint nouter, uint axis_length, uint stride,
    uint outer, uint interface, uint inner, ulong sub) {
    const uint cells = nouter * axis_length * stride;
    const uint low = outer * axis_length * stride + interface * stride + inner;
    const uint high = low + stride;
    const uint psi_index = outer * (axis_length - 1u) * stride
                         + interface * stride + inner;
    const uint qbase = tracer * cells;
    const ulong qlo = soft64_div(mass[qbase + low], density[low]);
    const ulong qhi = soft64_div(mass[qbase + high], density[high]);
    ulong slope_lo = 0ul;
    ulong slope_hi = 0ul;
    if (interface > 0u) {
        const uint prev = low - stride;
        const ulong qprev = soft64_div(mass[qbase + prev], density[prev]);
        const ulong left = soft64_div(soft64_sub(qlo, qprev),
            soft64_sub(centre[interface], centre[interface - 1u]));
        const ulong right = soft64_div(soft64_sub(qhi, qlo),
            soft64_sub(centre[interface + 1u], centre[interface]));
        slope_lo = global_walled_harmonic(left, right);
    }
    if (interface + 1u < axis_length - 1u) {
        const uint next = high + stride;
        const ulong qnext = soft64_div(mass[qbase + next], density[next]);
        const ulong left = soft64_div(soft64_sub(qhi, qlo),
            soft64_sub(centre[interface + 1u], centre[interface]));
        const ulong right = soft64_div(soft64_sub(qnext, qhi),
            soft64_sub(centre[interface + 2u], centre[interface + 1u]));
        slope_hi = global_walled_harmonic(left, right);
    }
    const ulong flux_dt = soft64_mul(psi[psi_index], sub);
    const ulong c_lo = soft64_div(
        soft64_mul(flux_dt, inv_width[interface]), density[low]);
    const ulong c_hi = soft64_div(
        soft64_negate_bits(soft64_mul(flux_dt, inv_width[interface + 1u])),
        density[high]);
    const ulong w_lo = soft64_mul(
        soft64_sub(face[interface], centre[interface]),
        soft64_sub(0x3ff0000000000000ul, c_lo));
    const ulong w_hi = soft64_mul(
        soft64_sub(face[interface], centre[interface + 1u]),
        soft64_sub(0x3ff0000000000000ul, c_hi));
    const bool positive = soft64_greater_equal(psi[psi_index], 0ul);
    ulong value = positive
        ? soft64_add(qlo, soft64_mul(slope_lo, w_lo))
        : soft64_add(qhi, soft64_mul(slope_hi, w_hi));
    const ulong lower = global_walled_min(qlo, qhi);
    const ulong upper = global_walled_max(qlo, qhi);
    const ulong upstream = positive ? qlo : qhi;
    value = global_walled_min(global_walled_max(value, lower),
        global_walled_min(upper,
            soft64_mul(soft64_from_u32(2u), upstream)));
    return soft64_mul(flux_dt, value);
}

kernel void global_tracer_walled_fragment(
    constant GlobalTracerWalledFragmentResources &r [[buffer(0)]],
    constant GlobalTracerWalledFragmentParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint cells = q.nouter * q.axis_length * q.stride;
    if (tid >= cells) return;
    const uint inner = tid % q.stride;
    const uint axis_index = (tid / q.stride) % q.axis_length;
    const uint outer = tid / (q.axis_length * q.stride);
    ulong lower_flux = 0ul;
    ulong upper_flux = 0ul;
    ulong updated_density = r.density[tid];
    if (axis_index > 0u) {
        lower_flux = global_walled_face_flux(
            r.mass, r.density, r.psi, r.centre, r.face, r.inv_width, 0u, q.ntracer,
            q.nouter, q.axis_length, q.stride, outer, axis_index - 1u, inner, r.sub[0]);
    }
    if (axis_index + 1u < q.axis_length) {
        upper_flux = global_walled_face_flux(
            r.mass, r.density, r.psi, r.centre, r.face, r.inv_width, 0u, q.ntracer,
            q.nouter, q.axis_length, q.stride, outer, axis_index, inner, r.sub[0]);
        updated_density = soft64_sub(updated_density,
            soft64_mul(upper_flux, r.inv_width[axis_index]));
    }
    if (axis_index > 0u) {
        updated_density = soft64_add(updated_density,
            soft64_mul(lower_flux, r.inv_width[axis_index]));
    }
    r.new_density[tid] = updated_density;
    const uint tracer_cells = cells;
    for (uint tracer = 0u; tracer < q.ntracer; ++tracer) {
        const uint index = tracer * tracer_cells + tid;
        ulong updated = r.mass[index];
        if (axis_index > 0u) {
            const ulong flux = global_walled_face_flux(
                r.mass, r.density, r.psi, r.centre, r.face, r.inv_width, tracer, q.ntracer,
                q.nouter, q.axis_length, q.stride, outer, axis_index - 1u, inner, r.sub[0]);
            // The NumPy reference applies outgoing subtraction before
            // incoming addition on each in-place cell.
        }
        if (axis_index + 1u < q.axis_length) {
            const ulong flux = global_walled_face_flux(
                r.mass, r.density, r.psi, r.centre, r.face, r.inv_width, tracer, q.ntracer,
                q.nouter, q.axis_length, q.stride, outer, axis_index, inner, r.sub[0]);
            updated = soft64_sub(updated,
                soft64_mul(flux, r.inv_width[axis_index]));
        }
        if (axis_index > 0u) {
            const ulong flux = global_walled_face_flux(
                r.mass, r.density, r.psi, r.centre, r.face, r.inv_width, tracer, q.ntracer,
                q.nouter, q.axis_length, q.stride, outer, axis_index - 1u, inner, r.sub[0]);
            updated = soft64_add(updated,
                soft64_mul(flux, r.inv_width[axis_index]));
        }
        r.new_mass[index] = updated;
    }
}
