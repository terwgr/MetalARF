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
#include "hexv841_fct_edge_residual_v841_bindings.metalh"

kernel void hexv841_fct_edge_residual_v841(
    constant Hexv841FctEdgeResidualV841Resources &r [[buffer(0)]],
    constant Hexv841FctEdgeResidualV841Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_count)) return;
    const int ntracers = p.ntracers;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const float dt = p.dt;
    device const float *source_old = r.source_old;
    device const float *velocity = r.velocity;
    device const float *dv_edge = r.dv_edge;
    device const int *cells_on_edge = r.cells_on_edge;
    device const float *high_flux = r.high_flux;
    device float *upwind_flux = r.upwind_flux;
    device float *horizontal_residual = r.horizontal_residual;

    const int edge = int(gid);
    if (edge >= nedges) return;
    const int c0 = cells_on_edge[2 * edge];
    const int c1 = cells_on_edge[2 * edge + 1];
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = QE3(tracer, k, edge, nlev, nedges);
            const float vel = velocity[E2(k, edge, nedges)];
            const float upwind = mpas_mul(mpas_mul(dv_edge[edge], dt),
                mpas_add(mpas_mul(fct_plus(vel), source_old[Q3(
                    tracer, k, c0, nlev, ncells)]),
                    mpas_mul(fct_minus(vel), source_old[Q3(
                    tracer, k, c1, nlev, ncells)])));
            upwind_flux[index] = upwind;
            horizontal_residual[index] = mpas_sub(
                mpas_mul(dt, high_flux[index]), upwind);
        }
    }

}
