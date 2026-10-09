// Pack separate resident Global fields into the field-major buffer expected
// by the implicit vertical diffusion kernel.
#include <metal_stdlib>
#include "global_implicit_diffusion_pack_fragment_bindings.metalh"
using namespace metal;

kernel void global_implicit_diffusion_pack_fragment(
    constant GlobalImplicitDiffusionPackFragmentResources &r [[buffer(0)]],
    constant GlobalImplicitDiffusionPackFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = uint(p.nlevel) * uint(p.ncolumn);
    if (tid >= count) return;
    r.fields[tid] = r.u[tid];
    r.fields[count + tid] = r.v[tid];
    r.fields[2u * count + tid] = r.theta[tid];
    r.fields[3u * count + tid] = r.qv[tid];
    r.fields[4u * count + tid] = r.qc[tid];
    r.fields[5u * count + tid] = r.qi[tid];
}
