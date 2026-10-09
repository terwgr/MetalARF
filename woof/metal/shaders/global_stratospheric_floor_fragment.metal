// needs integration: FP64 Global one-sided stratospheric temperature pull.
//
// This is ReferencePhysics._stratospheric_floor_pull. The caller supplies
// the host-computed relax ratio dt/tau. A thread handles one atmospheric
// point; the ABI owner must register the kernel before dispatch.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_stratospheric_floor_fragment_bindings.metalh"

using namespace metal;

kernel void global_stratospheric_floor_fragment(
    constant GlobalStratosphericFloorFragmentResources &r [[buffer(0)]],
    constant GlobalStratosphericFloorFragmentParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= q.count) return;
    if (!soft64_less(r.pressure[tid], r.floor_pressure[0])) {
        r.updated_theta[tid] = r.theta[tid];
        return;
    }
    const ulong temperature = soft64_mul(r.theta[tid], r.exner[tid]);
    const ulong cold = soft64_greater(
        soft64_sub(r.floor_temperature[0], temperature), 0ul)
        ? soft64_sub(r.floor_temperature[0], temperature) : 0ul;
    const ulong pull = soft64_div(soft64_mul(cold, r.relax[0]), r.exner[tid]);
    r.updated_theta[tid] = soft64_add(r.theta[tid], pull);
}
