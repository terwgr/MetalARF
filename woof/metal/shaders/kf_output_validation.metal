#include <metal_stdlib>
#include "kf_validate_outputs_bindings.metalh"
using namespace metal;

kernel void kf_validate_outputs(
    constant KfValidateOutputsResources &r [[buffer(0)]],
    constant KfValidateOutputsParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    const uint i = index;
    device atomic_uint *status =
        reinterpret_cast<device atomic_uint *>(r.status);
    uint flags = 0u;
    if (i < uint(p.count3)) {
        if ((p.active & 0x01) && !isfinite(r.rthcuten[i])) flags |= 0x01u;
        if ((p.active & 0x02) && !isfinite(r.rqvcuten[i])) flags |= 0x02u;
        if ((p.active & 0x04) && !isfinite(r.rqccuten[i])) flags |= 0x04u;
        if ((p.active & 0x08) && !isfinite(r.rqicuten[i])) flags |= 0x08u;
        if ((p.active & 0x10) && !isfinite(r.rqrcuten[i])) flags |= 0x10u;
        if ((p.active & 0x20) && !isfinite(r.rqscuten[i])) flags |= 0x20u;
    }
    if (i < uint(p.count2)) {
        if ((p.active & 0x40) && !isfinite(r.nca_seconds[i])) flags |= 0x40u;
        if ((p.active & 0x80) && !isfinite(r.pratec[i])) flags |= 0x80u;
    }
    if (flags != 0u)
        atomic_fetch_or_explicit(status, flags, memory_order_relaxed);
}
