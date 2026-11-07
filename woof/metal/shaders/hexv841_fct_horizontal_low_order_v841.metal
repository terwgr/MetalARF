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
#include "hexv841_fct_horizontal_low_order_v841_bindings.metalh"

kernel void hexv841_fct_horizontal_low_order_v841(
    constant Hexv841FctHorizontalLowOrderV841Resources &r [[buffer(0)]],
    constant Hexv841FctHorizontalLowOrderV841Params &p [[buffer(1)]],
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
    device const float *upwind_flux = r.upwind_flux;
    device const float *horizontal_residual = r.horizontal_residual;
    device float *mass = r.mass;
    device float *scale_in = r.scale_in;
    device float *scale_out = r.scale_out;

    const int cell = int(gid);
    if (cell >= ncells) return;
    const float inverse_area = inv_area_cell[cell];
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            float cell_mass = mass[index];
            float incoming = scale_in[index];
            float outgoing = scale_out[index];
            for (int slot = 0; slot < n_edges_on_cell[cell]; ++slot) {
                const int edge = edges_on_cell[CES(cell, slot, max_edges)];
                const float sign = -acoustic_sign[CES(cell, slot, max_edges)];
                const int qe = QE3(tracer, k, edge, nlev, nedges);
                cell_mass = mpas_add(cell_mass, mpas_mul(
                    mpas_mul(sign, upwind_flux[qe]), inverse_area));
                const float signed_residual = mpas_mul(
                    -sign, horizontal_residual[qe]);
                outgoing = mpas_sub(outgoing,
                    mpas_mul(fct_plus(signed_residual), inverse_area));
                incoming = mpas_sub(incoming,
                    mpas_mul(fct_minus(signed_residual), inverse_area));
            }
            mass[index] = cell_mass;
            scale_in[index] = incoming;
            scale_out[index] = outgoing;
        }
    }

}
