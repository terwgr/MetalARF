#include <metal_stdlib>
#include "spec_zone_ring_bindings.metalh"

using namespace metal;

kernel void spec_zone_ring_copy(
    constant SpecZoneRingCopyResources &r [[buffer(0)]],
    constant SpecZoneRingCopyParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.count)) return;
    const uint section = uint(p.rows * p.columns);
    const uint level = idx / section;
    const uint rem = idx - level * section;
    const uint row = rem / uint(p.columns);
    const uint column = rem - row * uint(p.columns);
    const uint address = level * uint(p.plane)
        + uint(p.row_start + int(row)) * uint(p.row_length)
        + uint(p.column_start + int(column));
    if (p.direction == 0) {
        r.target[idx] = r.source[address];
    } else if (p.direction == 1) {
        r.source[address] = r.target[idx];
    } else {
        r.source[address] = 0.0f;
    }
}
