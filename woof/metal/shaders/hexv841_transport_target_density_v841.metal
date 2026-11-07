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
#include "hexv841_transport_target_density_v841_bindings.metalh"

kernel void hexv841_transport_target_density_v841(
    constant Hexv841TransportTargetDensityV841Resources &r [[buffer(0)]],
    constant Hexv841TransportTargetDensityV841Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_count)) return;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int max_edges = p.max_edges;
    const int advance_density = p.advance_density;
    const float dt = p.dt;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const int *edges_on_cell = r.edges_on_cell;
    device const float *acoustic_sign = r.acoustic_sign;
    device const float *dv_edge = r.dv_edge;
    device const float *inv_area_cell = r.inv_area_cell;
    device const float *velocity = r.velocity;
    device const float *vertical_velocity = r.vertical_velocity;
    device const float *rdzw = r.rdzw;
    device const float *rho_old = r.rho_old;
    device const float *rho_new = r.rho_new;
    device float *target_density = r.target_density;

    const int cell = int(gid);
    if (cell >= ncells) return;
    for (int k = 0; k < nlev; ++k) {
        const int index = C2(k, cell, ncells);
        if (!advance_density) {
            target_density[index] = rho_new[index];
        } else {
            float horizontal = 0.0f;
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                float contribution = mpas_mul(
                    sign, velocity[E2(k, edge, nedges)]);
                contribution = mpas_mul(contribution, dv_edge[edge]);
                contribution = mpas_mul(contribution, inv_area_cell[cell]);
                horizontal = mpas_add(horizontal, contribution);
            }
            target_density[index] = mpas_add(rho_old[index], mpas_mul(dt,
                mpas_sub(horizontal, mpas_mul(rdzw[k], mpas_sub(
                    vertical_velocity[C2(k + 1, cell, ncells)],
                    vertical_velocity[index])))));
        }
    }

}
