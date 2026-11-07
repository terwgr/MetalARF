#include <metal_stdlib>
using namespace metal;
#include "hex_fct_ieee32.metalh"
#pragma clang fp contract(off)
#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define Q3(q,k,c,nl,nc) ((((q)*(nl) + (k))*(nc)) + (c))
#define QE3(q,k,e,nl,ne) ((((q)*(nl) + (k))*(ne)) + (e))
#define QI3(q,k,c,nl,nc) ((((q)*((nl)+1) + (k))*(nc)) + (c))
#define CES(c,s,me) ((c)*(me) + (s))
#define ADV(e,s,width) ((e)*(width) + (s))
#include "hexv841_fct_copy_back_v841_bindings.metalh"

kernel void hexv841_fct_copy_back_v841(
    constant Hexv841FctCopyBackV841Resources &r [[buffer(0)]],
    constant Hexv841FctCopyBackV841Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_count)) return;
    const int ntracers = p.ntracers;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int regional = p.regional;
    device const int *bdy_mask_cell = r.bdy_mask_cell;
    device const float *stage = r.stage;
    device float *output = r.output;

    const int index = int(gid);
    if (index >= ntracers * nlev * ncells) return;
    if (regional && bdy_mask_cell[index % ncells] > 2)
        output[index] = stage[index];
    else output[index] = fct_max(0.0f, output[index]);

}
