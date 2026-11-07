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
#include "hexv841_transport_edge_values_mono_v841_bindings.metalh"

kernel void hexv841_transport_edge_values_mono_v841(
    constant Hexv841TransportEdgeValuesMonoV841Resources &r [[buffer(0)]],
    constant Hexv841TransportEdgeValuesMonoV841Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_count)) return;
    const int ntracers = p.ntracers;
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int width = p.width;
    const float coefficient = p.coefficient;
    device const float *stage = r.stage;
    device const float *velocity = r.velocity;
    device const float *adv = r.adv;
    device const float *adv3 = r.adv3;
    device const int *n_adv = r.n_adv;
    device const int *adv_cells = r.adv_cells;
    device float *edge_values = r.edge_values;

    const int edge = int(gid);
    if (edge >= nedges) return;
    const int count = n_adv[edge];
    if (count == 10) {
        for (int k = 0; k < nlev; ++k) {
            const float vel = velocity[E2(k, edge, nedges)];
            const float sign = mpas_copysign(1.0f, vel);
            for (int tracer = 0; tracer < ntracers; ++tracer) {
                const int first_cell = adv_cells[ADV(edge, 0, width)];
                float weight = mpas_add(adv[ADV(edge, 0, width)],
                    mpas_mul(mpas_mul(coefficient, sign),
                        adv3[ADV(edge, 0, width)]));
                float value = mpas_mul(weight,
                    stage[Q3(tracer, k, first_cell, nlev, ncells)]);
                for (int slot = 1; slot < 10; ++slot) {
                    const int cell = adv_cells[ADV(edge, slot, width)];
                    weight = mpas_add(adv[ADV(edge, slot, width)],
                        mpas_mul(mpas_mul(coefficient, sign),
                            adv3[ADV(edge, slot, width)]));
                    value = mpas_add(value, mpas_mul(weight,
                        stage[Q3(tracer, k, cell, nlev, ncells)]));
                }
                edge_values[QE3(tracer, k, edge, nlev, nedges)] =
                    mpas_mul(vel, value);
            }
        }
    } else {
        for (int tracer = 0; tracer < ntracers; ++tracer) {
            for (int k = 0; k < nlev; ++k) {
                edge_values[QE3(tracer, k, edge, nlev, nedges)] = 0.0f;
            }
        }
        for (int slot = 0; slot < count; ++slot) {
            const int cell = adv_cells[ADV(edge, slot, width)];
            for (int k = 0; k < nlev; ++k) {
                const float vel = velocity[E2(k, edge, nedges)];
                const float sign = mpas_copysign(1.0f, vel);
                float scalar_weight = mpas_add(adv[ADV(edge, slot, width)],
                    mpas_mul(mpas_mul(coefficient, sign),
                        adv3[ADV(edge, slot, width)]));
                scalar_weight = mpas_mul(vel, scalar_weight);
                for (int tracer = 0; tracer < ntracers; ++tracer) {
                    const int index = QE3(tracer, k, edge, nlev, nedges);
                    edge_values[index] = mpas_add(edge_values[index],
                        mpas_mul(scalar_weight,
                            stage[Q3(tracer, k, cell, nlev, ncells)]));
                }
            }
        }
    }

}
