// One column of a [..., column] field copied to a column of another (or the
// same) field: the limited-area garbage-column lend of
// prepare_mpas_to_phys_cuda_v841 (woof/hex/cuda_physics_prep_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hexv841_copy_column_v841_bindings.metalh"

kernel void hexv841_copy_column_v841(
    constant Hexv841CopyColumnV841Resources &r [[buffer(0)]],
    constant Hexv841CopyColumnV841Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const long row = (long)tid;
    if (row >= (long)p.rows) return;
    r.dst[row * (long)p.dst_stride + (long)p.dst_col] =
        r.src[row * (long)p.src_stride + (long)p.src_col];
}
