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
#include "hexv841_fct_limit_vertical_bindings.metalh"

kernel void hexv841_fct_limit_vertical(
    constant Hexv841FctLimitVerticalResources &r [[buffer(0)]],
    constant Hexv841FctLimitVerticalParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_count)) return;
    const int ntracers = p.ntracers;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    device const float *scale_in = r.scale_in;
    device const float *scale_out = r.scale_out;
    device float *residual = r.residual;

    const int cell = int(gid);
    if (cell >= ncells) return;
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int interface = 1; interface < nlev; ++interface) {
            const int qi = QI3(tracer, interface, cell, nlev, ncells);
            const float flux = residual[qi];
            residual[qi] = mpas_add(mpas_mul(fct_plus(flux), fct_min(
                    scale_out[Q3(
                        tracer, interface - 1, cell, nlev, ncells)],
                    scale_in[Q3(tracer, interface, cell, nlev, ncells)])),
                mpas_mul(fct_minus(flux), fct_min(
                    scale_out[Q3(tracer, interface, cell, nlev, ncells)],
                    scale_in[Q3(
                        tracer, interface - 1, cell, nlev, ncells)])));
        }
    }

}
