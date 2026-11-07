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
#include "hexv841_fct_minmax_source_bindings.metalh"

kernel void hexv841_fct_minmax_source(
    constant Hexv841FctMinmaxSourceResources &r [[buffer(0)]],
    constant Hexv841FctMinmaxSourceParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_count)) return;
    const int ntracers = p.ntracers;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int max_edges = p.max_edges;
    const float dt = p.dt;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *cells_on_cell = r.cells_on_cell;
    device const float *old = r.old;
    device const float *source = r.source;
    device const float *rho_old = r.rho_old;
    device float *source_old = r.source_old;
    device float *minimum = r.minimum;
    device float *maximum = r.maximum;

    const int cell = int(gid);
    if (cell >= ncells) return;
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            const float current = mpas_add(old[index], mpas_div(mpas_mul(dt, source[index]), rho_old[C2(k, cell, ncells)]));
            source_old[index] = current;
            float low = current;
            float high = current;
            if (k > 0) {
                const float value = mpas_add(old[Q3(
                    tracer, k - 1, cell, nlev, ncells)], mpas_div(mpas_mul(dt, source[Q3(
                        tracer, k - 1, cell, nlev, ncells)]), rho_old[C2(k - 1, cell, ncells)]));
                low = fct_min(low, value);
                high = fct_max(high, value);
            }
            if (k + 1 < nlev) {
                const float value = mpas_add(old[Q3(
                    tracer, k + 1, cell, nlev, ncells)], mpas_div(mpas_mul(dt, source[Q3(
                        tracer, k + 1, cell, nlev, ncells)]), rho_old[C2(k + 1, cell, ncells)]));
                low = fct_min(low, value);
                high = fct_max(high, value);
            }
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int neighbor = cells_on_cell[CES(cell, slot, max_edges)];
                const float value = mpas_add(old[Q3(
                    tracer, k, neighbor, nlev, ncells)], mpas_div(mpas_mul(dt, source[Q3(tracer, k, neighbor, nlev, ncells)]), rho_old[C2(k, neighbor, ncells)]));
                low = fct_min(low, value);
                high = fct_max(high, value);
            }
            minimum[index] = low;
            maximum[index] = high;
        }
    }

}
