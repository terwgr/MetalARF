// Hex v8.4.1 Metal transcription of CUDA kernel transport_edge_values_regional_v841 (woof/hex/cuda_regional_v841.py:806).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_transport_edge_values_regional_v841_bindings.metalh"
#pragma clang fp contract(off)

#define RC2(k,c,nc) ((k)*(nc) + (c))
#define RE2(k,e,ne) ((k)*(ne) + (e))
#define RQ3(t,k,c,nlev,nc) (((t)*(nlev) + (k))*(nc) + (c))
#define RQE3(t,k,e,nlev,ne) (((t)*(nlev) + (k))*(ne) + (e))
#define RQI3(t,k,c,nlev,nc) (((t)*((nlev)+1) + (k))*(nc) + (c))
#define RCES(c,s,me) ((c)*(me) + (s))
#define RADV(e,s,w) ((e)*(w) + (s))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)
#define REGIONAL_N_RELAX_ZONE 5
#define REGIONAL_N_SPEC_ZONE 2
#define MPAS_FTZ_FALLBACK_ENABLED 1


kernel void hexv841_transport_edge_values_regional_v841(
    constant Hexv841TransportEdgeValuesRegionalV841Resources &rsc__ [[buffer(0)]],
    constant Hexv841TransportEdgeValuesRegionalV841Params &prm__ [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int ntracers = prm__.ntracers;
    const int nlev = prm__.nlev;
    const int ncells = prm__.ncells;
    const int nedges = prm__.nedges;
    const int width = prm__.width;
    const float coefficient = prm__.coefficient;
    device const int *bdy_mask_edge = rsc__.bdy_mask_edge;
    device const float *dv_edge = rsc__.dv_edge;
    device const int *cells_on_edge = rsc__.cells_on_edge;
    device const float *stage = rsc__.stage;
    device const float *velocity = rsc__.velocity;
    device const float *adv = rsc__.adv;
    device const float *adv3 = rsc__.adv3;
    device const int *n_adv = rsc__.n_adv;
    device const int *adv_cells = rsc__.adv_cells;
    device float *edge_values = rsc__.edge_values;

    const int element = int(tid);
    if (element >= (ntracers * nlev * nedges)) return;
    {
        const int edge = element % nedges;
        const int rest = element / nedges;
        const int k = rest % nlev;
        const int tracer = rest / nlev;
        const int mask = bdy_mask_edge[edge];
        if (mask > REGIONAL_N_RELAX_ZONE) {
            /* Native leaves the scratch unwritten here (F:4764-4842) and no
               updated cell reads it: an edge above nRelaxZone touches only
               specified-zone cells, which this routine does not update.  A
               device scratch buffer is uninitialized memory, so the lane is
               written zero instead of left alone -- unread either way, but
               byte-reproducible between two runs of the same configuration,
               which "left alone" is not. */
            edge_values[RQE3(tracer, k, edge, nlev, nedges)] = 0.0f;
            return;
        }
        if (mask >= REGIONAL_N_RELAX_ZONE - 1) {
            const int cell1 = cells_on_edge[2 * edge];
            const int cell2 = cells_on_edge[2 * edge + 1];
            const float vel = velocity[RE2(k, edge, nedges)];
            const float direction = mpas_copysign(0.5f, vel);
            const float u_positive = mpas_mul(
                dv_edge[edge], mpas_abs(mpas_add(direction, 0.5f)));
            const float u_negative = mpas_mul(
                dv_edge[edge], mpas_abs(mpas_sub(direction, 0.5f)));
            edge_values[RQE3(tracer, k, edge, nlev, nedges)] = mpas_add(
                mpas_mul(u_positive,
                    stage[RQ3(tracer, k, cell1, nlev, ncells)]),
                mpas_mul(u_negative,
                    stage[RQ3(tracer, k, cell2, nlev, ncells)]));
            return;
        }
        /* The full branch is the released transport_edge_values body verbatim,
           guarded fallback included, so a non-boundary edge produces the same
           bits as the global translation unit.  The contract deck proves that
           claim rather than asserting it. */
        float value = 0.0f;
        const float velocity_sign = mpas_copysign(
            1.0f, velocity[RE2(k, edge, nedges)]);
        bool ftz_sensitive = false;
        for (int slot = 0; slot < n_adv[edge]; ++slot) {
            const int cell = adv_cells[RADV(edge, slot, width)];
            const float base_weight = adv[RADV(edge, slot, width)];
            const float third_weight = adv3[RADV(edge, slot, width)];
            const float stage_value = stage[RQ3(
                tracer, k, cell, nlev, ncells)];
            const float correction = coefficient * velocity_sign
                * third_weight;
            const float weight = base_weight + correction;
            const float contribution = weight * stage_value;
            const float next = value + contribution;
#if MPAS_FTZ_FALLBACK_ENABLED
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
#endif
            value = next;
        }
        if (ftz_sensitive) {
            value = 0.0f;
            for (int slot = 0; slot < n_adv[edge]; ++slot) {
                const int cell = adv_cells[RADV(edge, slot, width)];
                const float weight = mpas_add(
                    adv[RADV(edge, slot, width)],
                    mpas_mul(mpas_mul(coefficient, velocity_sign),
                        adv3[RADV(edge, slot, width)]));
                value = mpas_add(value, mpas_mul(weight,
                    stage[RQ3(tracer, k, cell, nlev, ncells)]));
            }
        }
        edge_values[RQE3(tracer, k, edge, nlev, nedges)] = value;
    }
}
