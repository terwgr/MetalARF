// Hex v8.4.1 kernel lts_reflux_accumulate, transcribed statement for statement from
// woof/hex/cuda_acoustic_lts.py.
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_lts_reflux_accumulate_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))

kernel void hexv841_lts_reflux_accumulate(
    constant Hexv841LtsRefluxAccumulateResources &rsc [[buffer(0)]],
    constant Hexv841LtsRefluxAccumulateParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm.nlev;
    const int nedges = prm.nedges;
    const int n_iface = prm.n_iface;
    const float dts = prm.dts;
    const float sign_scale = prm.sign_scale;
    const int slot = int(tid);
    if (slot >= n_iface) return;
    const int edge = rsc.iface_edges[slot];
    const float scaled = mpas_mul(sign_scale, mpas_mul(dts, rsc.dv_edge[edge]));
    for (int k = 0; k < nlev; ++k) {
        const int index = E2(k, edge, nedges);
        rsc.resid[index] = mpas_add(rsc.resid[index],
            mpas_mul(scaled, rsc.ru_p[index]));
    }
}
