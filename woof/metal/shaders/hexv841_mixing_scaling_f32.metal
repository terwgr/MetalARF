#include <metal_stdlib>
#include "pressure_power.metalh"
#include "../generated/hexv841_mixing_scaling_f32_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)


kernel void hexv841_mixing_scaling_f32(
    constant Hexv841MixingScalingF32Resources &r [[buffer(0)]],
    constant Hexv841MixingScalingF32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *mesh_density = r.mesh_density;
    device float *del2 = r.del2;
    device float *del4 = r.del4;
    const int nedges = p.nedges;
    const int scale_with_mesh = p.scale_with_mesh;

    const int edge = int(gid);
    if (edge >= nedges) return;
    if (!scale_with_mesh) {
        del2[edge] = 1.0f;
        del4[edge] = 1.0f;
        return;
    }
    const int cell0 = cells_on_edge[2 * edge];
    const int cell1 = cells_on_edge[2 * edge + 1];
    const float mean_density = 0.5f * (mesh_density[cell0] + mesh_density[cell1]);
    // powf rounded once from binary64; native pow is biased low.
    del2[edge] = 1.0f / fp32_pressure_power(mean_density, 1.0f, 0.25f);
    del4[edge] = 1.0f / fp32_pressure_power(mean_density, 1.0f, 0.75f);
}
