#include <metal_stdlib>
using namespace metal;
#include "hexv841_rrfs_transpose_bindings.metalh"
kernel void hexv841_rrfs_transpose(constant Hexv841RrfsTransposeResources &r [[buffer(0)]],
    constant Hexv841RrfsTransposeParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.rows * p.columns)) return;
    r.dst[(gid % p.columns) * p.rows + gid / p.columns] = r.src[gid];
}
