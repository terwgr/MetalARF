#include <metal_stdlib>
#include "hex_normal_velocity_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// Recover MPAS edge-normal velocity from mass-weighted momentum and the two
// adjacent cell densities, preserving the denominator and multiply/divide
// operation order from the preview driver.
kernel void hex_normal_velocity(
        constant HexNormalVelocityResources &r [[buffer(0)]],
        constant HexNormalVelocityParams &q [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(q.nlev) * ulong(q.nedges);
    if (q.nlev <= 0 || q.ncells <= 0 || q.nedges <= 0 || ulong(gid) >= count)
        return;
    const int level = int(gid / uint(q.nedges));
    const int edge = int(gid % uint(q.nedges));
    const int cell0 = r.cells_on_edge[2 * edge];
    const int cell1 = r.cells_on_edge[2 * edge + 1];
    if (cell0 < 0 || cell1 < 0 || cell0 >= q.ncells || cell1 >= q.ncells)
        return;
    const ulong at0 = ulong(level) * ulong(q.ncells) + ulong(cell0);
    const ulong at1 = ulong(level) * ulong(q.ncells) + ulong(cell1);
    const float denominator = r.rho[at0] + r.rho[at1];
    const float twice_momentum = 2.0f * r.rho_u[gid];
    r.normal_velocity[gid] = twice_momentum / denominator;
}
