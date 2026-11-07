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
#include "hexv841_fct_finish_v841_bindings.metalh"

kernel void hexv841_fct_finish_v841(
    constant Hexv841FctFinishV841Resources &r [[buffer(0)]],
    constant Hexv841FctFinishV841Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_count)) return;
    const int ntracers = p.ntracers;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *edges_on_cell = r.edges_on_cell;
    device const float *acoustic_sign = r.acoustic_sign;
    device const float *inv_area_cell = r.inv_area_cell;
    device const float *rdzw = r.rdzw;
    device const float *horizontal_residual = r.horizontal_residual;
    device const float *vertical_residual = r.vertical_residual;
    device const float *target_density = r.target_density;
    device const float *mass = r.mass;
    device float *output = r.output;

    const int cell = int(gid);
    if (cell >= ncells) return;
    const float inverse_area = inv_area_cell[cell];
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            float value = mass[index];
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                value = mpas_add(value, mpas_mul(mpas_mul(sign,
                    horizontal_residual[QE3(
                        tracer, k, edge, nlev, nedges)]), inverse_area));
            }
            value = mpas_sub(value, mpas_mul(rdzw[k], mpas_sub(
                vertical_residual[QI3(
                    tracer, k + 1, cell, nlev, ncells)],
                vertical_residual[QI3(
                    tracer, k, cell, nlev, ncells)])));
            output[index] = fct_max(0.0f,
                mpas_div(value, target_density[C2(k, cell, ncells)]));
        }
    }

}
