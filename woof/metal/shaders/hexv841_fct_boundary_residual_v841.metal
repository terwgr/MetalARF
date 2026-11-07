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
#include "hexv841_fct_boundary_residual_v841_bindings.metalh"

kernel void hexv841_fct_boundary_residual_v841(
    constant Hexv841FctBoundaryResidualV841Resources &r [[buffer(0)]],
    constant Hexv841FctBoundaryResidualV841Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_count)) return;
    const int ntracers = p.ntracers;
    const int nlev = p.nlev;
    const int nedges = p.nedges;
    device const int *bdy_mask_edge = r.bdy_mask_edge;
    device float *residual = r.residual;

    const int index = int(gid);
    if (index >= ntracers * nlev * nedges) return;
    const int mask = bdy_mask_edge[index % nedges];
    if (mask == 4 || mask == 5) residual[index] = 0.0f;

}
