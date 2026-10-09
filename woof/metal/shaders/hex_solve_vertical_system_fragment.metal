#include <metal_stdlib>
#include "hex_solve_vertical_system_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// One lane owns a complete column, preserving MPAS's forward and backward
// Thomas sweep order. The typed entry point remains a source fragment until
// its descriptor and generated binding are integrated.
kernel void hex_solve_vertical_system(
        constant HexSolveVerticalSystemResources &r [[buffer(0)]],
        constant HexSolveVerticalSystemParams &q [[buffer(1)]],
        uint cell [[thread_position_in_grid]]) {
    if (q.nlev <= 0 || q.ncells <= 0 || cell >= uint(q.ncells)) return;

    for (int level = 1; level < q.nlev; ++level) {
        const ulong here = ulong(level) * ulong(q.ncells) + ulong(cell);
        const ulong below = ulong(level - 1) * ulong(q.ncells) + ulong(cell);
        r.rhs[here] = (r.rhs[here] - q.a_scale * r.a_tri[here] * r.rhs[below])
                      * r.alpha_tri[here];
    }
    for (int level = q.nlev - 1; level >= 0; --level) {
        const ulong here = ulong(level) * ulong(q.ncells) + ulong(cell);
        const ulong above = ulong(level + 1) * ulong(q.ncells) + ulong(cell);
        r.rhs[here] -= r.gamma_tri[here] * r.rhs[above];
    }
}
