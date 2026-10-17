// Hex v8.4.1 kernel acoustic_cofrz_v841, transcribed statement for statement from
// woof/hex/cuda_acoustic_v841.py.
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_acoustic_cofrz_v841_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define CES(c,s,me) ((c)*(me) + (s))
#define ZBS(k,c,s,nc,me) ((((k)*(nc) + (c))*(me)) + (s))

kernel void hexv841_acoustic_cofrz_v841(
    constant Hexv841AcousticCofrzV841Resources &rsc [[buffer(0)]],
    constant Hexv841AcousticCofrzV841Params &prm [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = prm.nlev;

    const int k = int(tid);
    if (k < nlev) rsc.cofrz[k] = rsc.rdzw[k];
}
