#include <metal_stdlib>
using namespace metal;
#include "hex_fct_ieee32.metalh"
#pragma clang fp contract(off)
#include "hexv841_rrfs_regional_w_boundary_rrfs_bindings.metalh"
kernel void hexv841_rrfs_regional_w_boundary_rrfs(constant Hexv841RrfsRegionalWBoundaryRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsRegionalWBoundaryRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int nl = p.nl;
    int nc = p.nc;
    int first = p.first;
    int end = p.end;
    int relaxation = p.relaxation;
    int zero = p.zero;
    device const int *mask = r.mask;
    device const int *nearest = r.nearest;
    device const float *input = r.input;
    device float *output = r.output;
 int c=int(gid % p.dispatch_x);
 if(c<first || c>=end || mask[c]<=relaxation)return;
 for(int k=1;k<nl;++k)output[k*nc+c]=zero?0.0f:input[k*nc+nearest[c]];
}
