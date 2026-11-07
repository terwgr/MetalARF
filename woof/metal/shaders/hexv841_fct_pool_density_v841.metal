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
#include "hexv841_fct_pool_density_v841_bindings.metalh"

kernel void hexv841_fct_pool_density_v841(
    constant Hexv841FctPoolDensityV841Resources &r [[buffer(0)]],
    constant Hexv841FctPoolDensityV841Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_count)) return;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int cell = p.cell;
    device float *density = r.density;

    const int k = int(gid);
    if (k < nlev) density[k * ncells + cell] = 1.0f;

}
