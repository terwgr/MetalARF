// WRF module_microphysics_zero_out.F: outer-ring floor followed by the
// configured interior threshold/floor rule. Each thread owns one cell.
#include <metal_stdlib>
#include "mp_zero_out_fragment_bindings.metalh"

using namespace metal;

kernel void mp_zero_out(
    constant MpZeroOutResources &r [[buffer(0)]],
    constant MpZeroOutParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.count)) return;

    const uint plane = uint(p.ny) * uint(p.nx);
    const uint k = tid / plane;
    const uint rem = tid - k * plane;
    const uint j = rem / uint(p.nx);
    const uint i = rem - j * uint(p.nx);
    float value = r.field[tid];

    const bool ring = j == 0 || j == uint(p.ny - 1)
                   || i == 0 || i == uint(p.nx - 1);
    if (ring && value < 0.0f) value = 0.0f;

    const uint sz = uint(p.spec_zone);
    const bool interior = j >= sz && int(j) < p.ny - p.spec_zone
                       && i >= sz && int(i) < p.nx - p.spec_zone;
    if (interior) {
        if (p.first_species != 0) {
            if (p.mode == 2 && value < 0.0f) value = 0.0f;
        } else if (value < p.threshold) {
            value = 0.0f;
        }
    }
    r.field[tid] = value;
}
