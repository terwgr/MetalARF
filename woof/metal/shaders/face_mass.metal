#include <metal_stdlib>
#include "average_mass_faces_bindings.metalh"

using namespace metal;

// CUDA uses separate RN add and multiply operations for every periodic face.
#pragma clang fp contract(off)

kernel void average_mass_faces(
    constant AverageMassFacesResources &r [[buffer(0)]],
    constant AverageMassFacesParams &p [[buffer(1)]],
    uint at [[thread_position_in_grid]]) {
    const int width = p.nx + (p.yface ? 0 : 1);
    const int height = p.ny + (p.yface ? 1 : 0);
    if (at >= uint(width * height)) return;

    const int j = int(at) / width;
    const int i = int(at) - j * width;
    const int jj = j % p.ny;
    const int ii = i % p.nx;
    const int other_j = p.yface ? (jj + p.ny - 1) % p.ny : jj;
    const int other_i = p.yface ? ii : (ii + p.nx - 1) % p.nx;
    const ulong current = ulong(jj) * ulong(p.nx) + ulong(ii);
    const ulong previous = ulong(other_j) * ulong(p.nx) + ulong(other_i);

    const bool boundary = p.yface ? p.boundary_y != 0
                                  : p.boundary_x != 0;
    const int face = p.yface ? j : i;
    const int extent = p.yface ? p.ny : p.nx;
    if (boundary && (face == 0 || face == extent)) {
        const int edge_j = p.yface ? (face == 0 ? 0 : p.ny - 1) : j;
        const int edge_i = p.yface ? i : (face == 0 ? 0 : p.nx - 1);
        r.out[at] = r.mu[ulong(edge_j) * ulong(p.nx) + ulong(edge_i)];
        return;
    }

    // The final periodic face is computed independently from the wrapped
    // source indices, preserving the duplicate while supporting ny != nx.
    r.out[at] = 0.5f * (r.mu[current] + r.mu[previous]);
}
