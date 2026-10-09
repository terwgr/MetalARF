#include <metal_stdlib>
#include "lbc_dependent_batch_bindings.metalh"
#include "lbc_zero_gradient_w_bindings.metalh"

using namespace metal;

struct FlowFramePoint {
    int j;
    int i;
    int d;
    bool valid;
};

static inline FlowFramePoint flow_frame_point(int p, int ny, int nx,
                                               int width) {
    for (int d = 0; d < width; ++d) {
        const int ni = nx - 2 * d;
        const int nj = max(ny - 2 * d - 2, 0);
        const int ring = 2 * ni + 2 * nj;
        if (p >= ring) {
            p -= ring;
            continue;
        }
        FlowFramePoint point = {0, 0, d, true};
        if (p < ni) {
            point.j = d;
            point.i = d + p;
        } else if (p < 2 * ni) {
            point.j = ny - 1 - d;
            point.i = d + p - ni;
        } else if (p < 2 * ni + nj) {
            point.j = d + 1 + p - 2 * ni;
            point.i = d;
        } else {
            point.j = d + 1 + p - 2 * ni - nj;
            point.i = nx - 1 - d;
        }
        return point;
    }
    FlowFramePoint invalid = {0, 0, 0, false};
    return invalid;
}

static inline void flow_store(constant LbcDependentBatchResources &r,
                              int field_index, ulong destination,
                              ulong source, bool outflow, float inflow_value) {
    switch (field_index) {
        case 0: r.f0[destination] = outflow ? r.f0[source] : inflow_value; break;
        case 1: r.f1[destination] = outflow ? r.f1[source] : inflow_value; break;
        case 2: r.f2[destination] = outflow ? r.f2[source] : inflow_value; break;
        case 3: r.f3[destination] = outflow ? r.f3[source] : inflow_value; break;
        case 4: r.f4[destination] = outflow ? r.f4[source] : inflow_value; break;
        case 5: r.f5[destination] = outflow ? r.f5[source] : inflow_value; break;
        case 6: r.f6[destination] = outflow ? r.f6[source] : inflow_value; break;
        case 7: r.f7[destination] = outflow ? r.f7[source] : inflow_value; break;
        default: r.f8[destination] = outflow ? r.f8[source] : inflow_value; break;
    }
}

kernel void lbc_dependent_batch(
    constant LbcDependentBatchResources &r [[buffer(0)]],
    constant LbcDependentBatchParams &q [[buffer(1)]],
    uint t [[thread_position_in_grid]]) {
    if (t >= uint(q.nz * q.frame_count)) return;
    const int k = int(t) / q.frame_count;
    const int p = int(t) - k * q.frame_count;
    const FlowFramePoint point = flow_frame_point(p, q.ny, q.nx, q.spec_zone);
    if (!point.valid) return;
    const int j = point.j;
    const int i = point.i;
    const int d = point.d;
    const ulong destination = (ulong(k) * ulong(q.ny) + ulong(j)) * ulong(q.nx) + ulong(i);
    ulong source;
    bool outflow;
    if (j == d) {
        const int inner_i = min(max(i, q.spec_zone), q.nx - 1 - q.spec_zone);
        source = (ulong(k) * ulong(q.ny) + ulong(q.spec_zone)) * ulong(q.nx) + ulong(inner_i);
        outflow = r.v_flux[(ulong(k) * ulong(q.ny + 1) + ulong(j)) * ulong(q.nx) + ulong(i)] < 0.0f;
    } else if (j == q.ny - 1 - d) {
        const int inner_i = min(max(i, q.spec_zone), q.nx - 1 - q.spec_zone);
        source = (ulong(k) * ulong(q.ny) + ulong(q.ny - 1 - q.spec_zone)) * ulong(q.nx) + ulong(inner_i);
        outflow = r.v_flux[(ulong(k) * ulong(q.ny + 1) + ulong(j + 1)) * ulong(q.nx) + ulong(i)] > 0.0f;
    } else if (i == d) {
        const int inner_j = min(max(j, q.spec_zone), q.ny - 1 - q.spec_zone);
        source = (ulong(k) * ulong(q.ny) + ulong(inner_j)) * ulong(q.nx) + ulong(q.spec_zone);
        outflow = r.u_flux[(ulong(k) * ulong(q.ny) + ulong(j)) * ulong(q.nx + 1) + ulong(i)] < 0.0f;
    } else {
        const int inner_j = min(max(j, q.spec_zone), q.ny - 1 - q.spec_zone);
        source = (ulong(k) * ulong(q.ny) + ulong(inner_j)) * ulong(q.nx) + ulong(q.nx - 1 - q.spec_zone);
        outflow = r.u_flux[(ulong(k) * ulong(q.ny) + ulong(j)) * ulong(q.nx + 1) + ulong(i + 1)] > 0.0f;
    }
    for (int n = 0; n < q.nfield; ++n)
        flow_store(r, n, destination, source, outflow, q.inflow_value);
}

kernel void lbc_zero_gradient_w(
    constant LbcZeroGradientWResources &r [[buffer(0)]],
    constant LbcZeroGradientWParams &q [[buffer(1)]],
    uint t [[thread_position_in_grid]]) {
    if (t >= uint(q.nz * q.frame_count)) return;
    const int k = int(t) / q.frame_count;
    const int p = int(t) - k * q.frame_count;
    const FlowFramePoint point = flow_frame_point(p, q.ny, q.nx, q.spec_zone);
    if (!point.valid) return;
    int source_j;
    int source_i;
    if (point.j == point.d) {
        source_j = q.spec_zone;
        source_i = min(max(point.i, q.spec_zone), q.nx - 1 - q.spec_zone);
    } else if (point.j == q.ny - 1 - point.d) {
        source_j = q.ny - 1 - q.spec_zone;
        source_i = min(max(point.i, q.spec_zone), q.nx - 1 - q.spec_zone);
    } else if (point.i == point.d) {
        source_j = min(max(point.j, q.spec_zone), q.ny - 1 - q.spec_zone);
        source_i = q.spec_zone;
    } else {
        source_j = min(max(point.j, q.spec_zone), q.ny - 1 - q.spec_zone);
        source_i = q.nx - 1 - q.spec_zone;
    }
    const ulong destination = (ulong(k) * ulong(q.ny) + ulong(point.j)) * ulong(q.nx) + ulong(point.i);
    const ulong source = (ulong(k) * ulong(q.ny) + ulong(source_j)) * ulong(q.nx) + ulong(source_i);
    r.field[destination] = r.field[source];
}
