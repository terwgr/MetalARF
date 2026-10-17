// Hex v8.4.1 kernel acoustic_prepare_lts, transcribed statement for statement from
// woof/hex/cuda_acoustic_lts.py (derived from acoustic_prepare_v841 with the index-list gather).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_acoustic_prepare_lts_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ZBS(k,c,s,nc,me) ((((k)*(nc) + (c))*(me)) + (s))

kernel void hexv841_acoustic_prepare_lts(
    constant Hexv841AcousticPrepareLtsResources &rsc [[buffer(0)]],
    constant Hexv841AcousticPrepareLtsParams &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm.nlev;
    const int ncells = prm.ncells;
    const int small_step = prm.small_step;

    const int lts_slot = int(tid);
    if (lts_slot >= prm.lts_n_active) return;
    const int cell = rsc.lts_active[lts_slot];
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
