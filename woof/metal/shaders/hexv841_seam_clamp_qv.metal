// Metal mirror of the MPAS column-batch seam arithmetic: run_phase1:cp.maximum(qv, 0)
// (woof/core/mpas_column_batch.py:1049).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_seam_clamp_qv_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv841_seam_clamp_qv(
    constant Hexv841SeamClampQvResources &r [[buffer(0)]],
    constant Hexv841SeamClampQvParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int gid = int(tid);
    if (gid >= p.n) return;
    // cp.maximum propagates NaN; max(qv, 0) otherwise.  Decided on the
    // bits so a subnormal positive qv is kept, not flushed by a compare.
    const uint bits = as_type<uint>(r.qv[gid]);
    const bool is_nan = (bits & 0x7fffffffu) > 0x7f800000u;
    const bool negative = (bits >> 31) != 0u;
    r.out[gid] = (is_nan || !negative) ? as_type<float>(bits) : 0.0f;
}
