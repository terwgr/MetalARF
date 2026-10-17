// Hex v8.4.1 scalar transport: horizontal edge values (cuda_transport.py transport_edge_values).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_transport_edge_values_bindings.metalh"
#pragma clang fp contract(off)

#define E2(k,e,ne) ((k)*(ne) + (e))
#define Q3(q,k,c,nl,nc) ((((q)*(nl) + (k))*(nc)) + (c))
#define QE3(q,k,e,nl,ne) ((((q)*(nl) + (k))*(ne)) + (e))
#define ADV(e,s,width) ((e)*(width) + (s))

kernel void hexv841_transport_edge_values(
    constant Hexv841TransportEdgeValuesResources &r [[buffer(0)]],
    constant Hexv841TransportEdgeValuesParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int ntracers = p.ntracers, nlev = p.nlev, ncells = p.ncells;
    const int nedges = p.nedges, width = p.width;
    const float coefficient = p.coefficient;
    const int edge = int(tid);
    if (edge >= nedges) return;
    for (int tracer = 0; tracer < ntracers; ++tracer) {
        for (int k = 0; k < nlev; ++k) {
            float value = 0.0f;
            const float velocity_sign = mpas_copysign(
                1.0f, r.velocity[E2(k, edge, nedges)]);
            bool ftz_sensitive = false;
            for (int slot = 0; slot < r.n_adv[edge]; ++slot) {
                const int cell = r.adv_cells[ADV(edge, slot, width)];
                const float base_weight = r.adv[ADV(edge, slot, width)];
                const float third_weight = r.adv3[ADV(edge, slot, width)];
                const float stage_value = r.stage[Q3(
                    tracer, k, cell, nlev, ncells)];
                const float correction = coefficient * velocity_sign
                    * third_weight;
                const float weight = base_weight + correction;
                const float contribution = weight * stage_value;
                const float next = value + contribution;
                const unsigned int coefficient_mag =
                    mpas_f32_magnitude_bits(coefficient);
                const unsigned int third_mag =
                    mpas_f32_magnitude_bits(third_weight);
                const unsigned int correction_mag =
                    mpas_f32_magnitude_bits(correction);
                const unsigned int base_mag =
                    mpas_f32_magnitude_bits(base_weight);
                const unsigned int weight_mag =
                    mpas_f32_magnitude_bits(weight);
                const unsigned int stage_mag =
                    mpas_f32_magnitude_bits(stage_value);
                const unsigned int contribution_mag =
                    mpas_f32_magnitude_bits(contribution);
                const unsigned int value_mag =
                    mpas_f32_magnitude_bits(value);
                const unsigned int next_mag = mpas_f32_magnitude_bits(next);
                ftz_sensitive = ftz_sensitive
                    || (base_mag != 0u && (base_mag >> 23) == 0u)
                    || (third_mag != 0u && (third_mag >> 23) == 0u)
                    || (stage_mag != 0u && (stage_mag >> 23) == 0u)
                    || (correction_mag == 0u && coefficient_mag != 0u
                        && third_mag != 0u)
                    || (weight_mag == 0u && base_mag != correction_mag
                        && base_mag != 0u && correction_mag != 0u)
                    || (contribution_mag == 0u && weight_mag != 0u
                        && stage_mag != 0u)
                    || (next_mag == 0u && value_mag != contribution_mag
                        && value_mag != 0u && contribution_mag != 0u);
                value = next;
            }
            if (ftz_sensitive) {
                value = 0.0f;
                for (int slot = 0; slot < r.n_adv[edge]; ++slot) {
                    const int cell = r.adv_cells[ADV(edge, slot, width)];
                    const float weight = mpas_add(
                        r.adv[ADV(edge, slot, width)],
                        mpas_mul(mpas_mul(coefficient, velocity_sign),
                            r.adv3[ADV(edge, slot, width)]));
                    value = mpas_add(value, mpas_mul(weight,
                        r.stage[Q3(tracer, k, cell, nlev, ncells)]));
                }
            }
            r.edge_values[QE3(tracer, k, edge, nlev, nedges)] = value;
        }
    }
}
