// Hex v8.4.1 scalar transport: standard RK stage finish (cuda_transport_v841.py transport_standard_finish_v841).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_transport_standard_finish_v841_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define Q3(q,k,c,nl,nc) ((((q)*(nl) + (k))*(nc)) + (c))
#define QE3(q,k,e,nl,ne) ((((q)*(nl) + (k))*(ne)) + (e))
#define QI3(q,k,c,nl,nc) ((((q)*((nl)+1) + (k))*(nc)) + (c))
#define CES(c,s,me) ((c)*(me) + (s))

kernel void hexv841_transport_standard_finish_v841(
    constant Hexv841TransportStandardFinishV841Resources &r [[buffer(0)]],
    constant Hexv841TransportStandardFinishV841Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int ntracers = p.ntracers, nlev = p.nlev, ncells = p.ncells;
    const int nedges = p.nedges, max_edges = p.max_edges;
    const float dt = p.dt;
    const int cell = int(tid);
    if (cell >= ncells) return;
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            float horizontal = 0.0f;
            for (int slot = 0; slot < r.n_edges_on_cell[cell]; ++slot) {
                const int edge = r.edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -r.acoustic_sign[CES(cell, slot, max_edges)];
                horizontal = mpas_add(horizontal, mpas_mul(mpas_mul(
                    sign, r.velocity[E2(k, edge, nedges)]),
                    r.edge_values[QE3(tracer, k, edge, nlev, nedges)]));
            }
            const float tendency = mpas_add(
                mpas_mul(horizontal, r.inv_area_cell[cell]), r.source[index]);
            const float vertical = mpas_mul(r.rdzw[k], mpas_sub(
                r.vertical_flux[QI3(tracer, k + 1, cell, nlev, ncells)],
                r.vertical_flux[QI3(tracer, k, cell, nlev, ncells)]));
            const float numerator = mpas_add(
                mpas_mul(r.old[index], r.rho_old[C2(k, cell, ncells)]),
                mpas_mul(dt, mpas_sub(tendency, vertical)));
            const float rho_new_inv = mpas_div(
                1.0f, r.target_density[C2(k, cell, ncells)]);
            r.output[index] = mpas_mul(numerator, rho_new_inv);
        }
    }
}
