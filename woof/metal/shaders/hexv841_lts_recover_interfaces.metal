// Hex v8.4.1 kernel lts_recover_interfaces, transcribed statement for statement from
// woof/hex/cuda_acoustic_lts.py.
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_lts_recover_interfaces_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))

kernel void hexv841_lts_recover_interfaces(
    constant Hexv841LtsRecoverInterfacesResources &rsc [[buffer(0)]],
    constant Hexv841LtsRecoverInterfacesParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm.nlev;
    const int ncells = prm.ncells;
    const int cell = int(tid);
    if (cell >= ncells) return;
    const float steps = (float)rsc.cell_steps[cell];
    for (int k = 0; k <= nlev; ++k) {
        const int index = C2(k, cell, ncells);
        if (k == 0 || k == nlev) {
            rsc.rw[index] = 0.0f;
        } else {
            rsc.rw[index] = mpas_add(rsc.rw_saved[index], rsc.rw_p[index]);
        }
        rsc.flux_w[index] = mpas_add(rsc.rw_saved[index],
            mpas_div(rsc.ww_avg_p[index], steps));
    }
}
