// Hex v8.4.1 kernel acoustic_prepare_v841, transcribed statement for statement from
// woof/hex/cuda_acoustic_v841.py.
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_acoustic_prepare_v841_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ZBS(k,c,s,nc,me) ((((k)*(nc) + (c))*(me)) + (s))

kernel void hexv841_acoustic_prepare_v841(
    constant Hexv841AcousticPrepareV841Resources &rsc [[buffer(0)]],
    constant Hexv841AcousticPrepareV841Params &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm.nlev;
    const int ncells = prm.ncells;
    const int small_step = prm.small_step;

    const int cell = int(tid);
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int index = C2(k, cell, ncells);
        if (small_step == 1) {
            rsc.rtheta_pp_old[index] = 0.0f;
            rsc.rtheta_pp[index] = 0.0f;
            rsc.rho_pp[index] = 0.0f;
        } else {
            rsc.rtheta_pp_old[index] = rsc.rtheta_pp[index];
        }
    }
    if (small_step == 1) {
        for (int k = 0; k <= nlev; ++k) {
            const int index = C2(k, cell, ncells);
            rsc.rw_p[index] = 0.0f;
            rsc.ww_avg[index] = 0.0f;
        }
    }
}
