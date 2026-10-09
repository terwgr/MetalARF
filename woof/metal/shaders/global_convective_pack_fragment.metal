// Pack separate resident Global fields into the species-major layout used by
// the convective column kernel.
#include <metal_stdlib>
#include "global_convective_pack_fragment_bindings.metalh"
using namespace metal;

kernel void global_convective_pack_fragment(
    constant GlobalConvectivePackFragmentResources &r [[buffer(0)]],
    constant GlobalConvectivePackFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlevel) * uint(p.ncolumn);
    if (tid >= count) return;
    r.theta_packed[tid] = r.theta[tid];
    r.q_packed[tid] = r.qv[tid];
    r.q_packed[count + tid] = r.qc[tid];
    r.q_packed[2u * count + tid] = r.qr[tid];
    r.q_packed[3u * count + tid] = r.qi[tid];
    r.q_packed[4u * count + tid] = r.qs[tid];
    r.q_packed[5u * count + tid] = r.qg[tid];
}
