// Hex v8.4.1 kernel lts_recover_edges, transcribed statement for statement from
// woof/hex/cuda_acoustic_lts.py.
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_lts_recover_edges_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))

kernel void hexv841_lts_recover_edges(
    constant Hexv841LtsRecoverEdgesResources &rsc [[buffer(0)]],
    constant Hexv841LtsRecoverEdgesParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm.nlev;
    const int nedges = prm.nedges;
    const int edge = int(tid);
    if (edge >= nedges) return;
    const float steps = (float)rsc.edge_steps[edge];
    for (int k = 0; k < nlev; ++k) {
        const int index = E2(k, edge, nedges);
        rsc.ru[index] = mpas_add(rsc.ru_saved[index], rsc.ru_p[index]);
        rsc.flux_u[index] = mpas_add(rsc.ru_saved[index],
            mpas_div(rsc.ru_avg_p[index], steps));
    }
}
