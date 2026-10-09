#include <metal_stdlib>
#include "hex_scalar_fct_fragment_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

// Phase 0 forms raw horizontal residual and upwind fluxes. Phase 1 applies the
// completed cell scale factors to that residual.
kernel void hex_scalar_fct_fragment(
    constant HexScalarFctFragmentResources &r [[buffer(0)]],
    constant HexScalarFctFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint count = p.nlev * p.nedges;
    if (tid >= count) return;
    const uint level = tid / p.nedges;
    const uint edge = tid % p.nedges;
    const uint c0 = uint(r.cells_on_edge[2u * edge]);
    const uint c1 = uint(r.cells_on_edge[2u * edge + 1u]);
    if (p.phase == 0) {
        const float velocity = r.horizontal_velocity[tid];
        const float upwind = r.dv_edge[edge] * p.dt *
            (max(velocity, 0.0f) * r.old_scalar[level * p.ncells + c0]
             + min(velocity, 0.0f) * r.old_scalar[level * p.ncells + c1]);
        r.upwind_flux[tid] = upwind;
        float residual = p.dt * r.high_flux[tid] - upwind;
        if (p.mask_enabled != 0) {
            const int edge_mask = r.edge_mask[edge];
            if ((p.apply_lbcs != 0 && edge_mask == 5) || edge_mask == 4) {
                residual = 0.0f;
            }
        }
        r.residual[tid] = residual;
        return;
    }
    const float flux = r.residual[tid];
    const float positive = max(flux, 0.0f)
        * min(r.scale_out[level * p.ncells + c0], r.scale_in[level * p.ncells + c1]);
    const float negative = min(flux, 0.0f)
        * min(r.scale_in[level * p.ncells + c0], r.scale_out[level * p.ncells + c1]);
    r.limited_residual[tid] = positive + negative;
}
