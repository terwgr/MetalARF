// Hex v8.4.1 kernel tendency_w_to_omega, transcribed statement for statement from
// woof/hex/cuda_acoustic.py.
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_tendency_w_to_omega_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ZBS(k,c,s,nc,me) ((((k)*(nc) + (c))*(me)) + (s))

kernel void hexv841_tendency_w_to_omega(
    constant Hexv841TendencyWToOmegaResources &rsc [[buffer(0)]],
    constant Hexv841TendencyWToOmegaParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint grid_size [[threads_per_grid]]) {
    const int nlev = prm.nlev;
    const int ncells = prm.ncells;
    const int nedges = prm.nedges;
    const int max_edges = prm.max_edges;

    /* One element per (interior level, cell): the edge-slot accumulation
       into omega_tendency runs in the same ascending slot order the
       per-cell loop ran it for that level, then the metric multiply. */
    for (int element = int(tid); element < ((nlev - 1) * ncells); element += int(grid_size)) {
        const int cell = element % ncells;
        const int k = 1 + element / ncells;
        const int count = rsc.n_edges_on_cell[cell];
        const int index = C2(k, cell, ncells);
        float omega = rsc.omega_tendency[index];
        for (int slot = 0; slot < count; ++slot) {
            const int edge = rsc.edges_on_cell[CES(cell, slot, max_edges)];
            if (edge < 0 || edge >= nedges) continue;
            const float sign = rsc.edge_sign_on_cell[CES(cell, slot, max_edges)];
            const float ut = rsc.u_tendency[E2(k, edge, nedges)];
            const float flux = mpas_mul(sign, mpas_add(
                mpas_mul(rsc.fzm[k], ut),
                mpas_mul(rsc.fzp[k],
                    rsc.u_tendency[E2(k - 1, edge, nedges)])));
            const int metric = ZBS(k, cell, slot, ncells, max_edges);
            omega = mpas_sub(omega,
                mpas_mul(mpas_add(rsc.zb[metric], mpas_mul(
                    mpas_copysign(1.0f, ut), rsc.zb3[metric])), flux));
        }
        rsc.omega_tendency[index] = mpas_mul(omega, mpas_add(
            mpas_mul(rsc.fzm[k], rsc.zz[index]),
            mpas_mul(rsc.fzp[k], rsc.zz[C2(k - 1, cell, ncells)])));
    }
}
