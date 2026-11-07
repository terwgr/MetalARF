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
#include "hexv841_fct_scale_bindings.metalh"

kernel void hexv841_fct_scale(
    constant Hexv841FctScaleResources &r [[buffer(0)]],
    constant Hexv841FctScaleParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_count)) return;
    const int ntracers = p.ntracers;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    device const float *minimum = r.minimum;
    device const float *maximum = r.maximum;
    device const float *target_density = r.target_density;
    device const float *mass = r.mass;
    device float *scale_in = r.scale_in;
    device float *scale_out = r.scale_out;

    const int cell = int(gid);
    if (cell >= ncells) return;
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            const int index = Q3(tracer, k, cell, nlev, ncells);
            const float density = target_density[C2(k, cell, ncells)];
            const float eps = 1.0e-20f;
            const float incoming = mpas_div(mpas_sub(
                mpas_mul(maximum[index], density), mass[index]),
                mpas_add(scale_in[index], eps));
            const float outgoing = mpas_div(mpas_sub(
                mpas_mul(minimum[index], density), mass[index]),
                mpas_sub(scale_out[index], eps));
            scale_in[index] = fct_min(1.0f, fct_max(0.0f, incoming));
            scale_out[index] = fct_min(1.0f, fct_max(0.0f, outgoing));
        }
    }

}
