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
#include "hexv841_fct_vertical_low_order_v841_bindings.metalh"

kernel void hexv841_fct_vertical_low_order_v841(
    constant Hexv841FctVerticalLowOrderV841Resources &r [[buffer(0)]],
    constant Hexv841FctVerticalLowOrderV841Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_count)) return;
    const int ntracers = p.ntracers;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const float dt = p.dt;
    device const float *source_old = r.source_old;
    device const float *rho_old = r.rho_old;
    device const float *vertical_velocity = r.vertical_velocity;
    device const float *rdzw = r.rdzw;
    device const float *high_vertical = r.high_vertical;
    device float *mass = r.mass;
    device float *vertical_residual = r.vertical_residual;
    device float *scale_in = r.scale_in;
    device float *scale_out = r.scale_out;

    const int index = int(gid);
    const int total = ntracers * ncells;
    if (index >= total) return;
    const int cell = index % ncells;
    const int tracer = index / ncells;
    for (int k = 0; k < nlev; ++k) {
        const int q = Q3(tracer, k, cell, nlev, ncells);
        mass[q] = mpas_mul(source_old[q], rho_old[C2(k, cell, ncells)]);
    }
    for (int interface = 0; interface <= nlev; ++interface) {
        const int qi = QI3(tracer, interface, cell, nlev, ncells);
        vertical_residual[qi] = mpas_mul(dt, high_vertical[qi]);
    }
    /* Pass 1: level k loses interface k+1's upwind flux. */
    for (int k = 0; k + 1 < nlev; ++k) {
        const int interface = k + 1;
        const float velocity = vertical_velocity[C2(interface, cell, ncells)];
        const float flux = mpas_mul(dt, mpas_add(
            mpas_mul(fct_plus(velocity), source_old[Q3(
                tracer, interface - 1, cell, nlev, ncells)]),
            mpas_mul(fct_minus(velocity), source_old[Q3(
                tracer, interface, cell, nlev, ncells)])));
        const int q = Q3(tracer, k, cell, nlev, ncells);
        mass[q] = mpas_sub(mass[q], mpas_mul(flux, rdzw[k]));
    }
    /* Pass 2: level k gains interface k's flux; the residual is formed. */
    for (int k = 1; k < nlev; ++k) {
        const float velocity = vertical_velocity[C2(k, cell, ncells)];
        const float flux = mpas_mul(dt, mpas_add(
            mpas_mul(fct_plus(velocity), source_old[Q3(
                tracer, k - 1, cell, nlev, ncells)]),
            mpas_mul(fct_minus(velocity), source_old[Q3(
                tracer, k, cell, nlev, ncells)])));
        const int q = Q3(tracer, k, cell, nlev, ncells);
        mass[q] = mpas_add(mass[q], mpas_mul(flux, rdzw[k]));
        const int residual_q = QI3(tracer, k, cell, nlev, ncells);
        vertical_residual[residual_q] = mpas_sub(
            vertical_residual[residual_q], flux);
    }
    for (int k = 0; k < nlev; ++k) {
        const float upper = vertical_residual[QI3(tracer, k + 1, cell, nlev, ncells)];
        const float lower = vertical_residual[QI3(tracer, k, cell, nlev, ncells)];
        const int q = Q3(tracer, k, cell, nlev, ncells);
        scale_in[q] = mpas_mul(-rdzw[k],
            mpas_sub(fct_minus(upper), fct_plus(lower)));
        scale_out[q] = mpas_mul(-rdzw[k],
            mpas_sub(fct_plus(upper), fct_minus(lower)));
    }

}
