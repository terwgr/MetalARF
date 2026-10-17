// Hex v8.4.1 scalar transport: vertical flux (cuda_transport.py transport_vertical_flux).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_transport_vertical_flux_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define Q3(q,k,c,nl,nc) ((((q)*(nl) + (k))*(nc)) + (c))
#define QI3(q,k,c,nl,nc) ((((q)*((nl)+1) + (k))*(nc)) + (c))

// CUDA: __constant__ float mpas_third_order_denominator = 12.0f;
constant float mpas_third_order_denominator = 12.0f;

kernel void hexv841_transport_vertical_flux(
    constant Hexv841TransportVerticalFluxResources &r [[buffer(0)]],
    constant Hexv841TransportVerticalFluxParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint grid_threads [[threads_per_grid]]) {
    const int ntracers = p.ntracers, nlev = p.nlev, ncells = p.ncells;
    const float coefficient = p.coefficient;
    // TRANSPORT_ELEMENT_LOOP(total)
    const int total = ntracers * (nlev + 1) * ncells;
    for (int element = int(tid); element < total; element += int(grid_threads)) {
        const int cell = element % ncells;
        const int rest = element / ncells;
        const int interface = rest % (nlev + 1);
        const int tracer = rest / (nlev + 1);
        float result = 0.0f;
        if (interface > 0 && interface < nlev) {
            const float vel = r.velocity[C2(interface, cell, ncells)];
            if (interface == 1 || interface == nlev - 1) {
                result = mpas_mul(vel, mpas_add(
                    mpas_mul(r.fzm[interface], r.stage[Q3(
                        tracer, interface, cell, nlev, ncells)]),
                    mpas_mul(r.fzp[interface], r.stage[Q3(
                        tracer, interface - 1, cell, nlev, ncells)])));
            } else {
                const float qim2 = r.stage[Q3(
                    tracer, interface - 2, cell, nlev, ncells)];
                const float qim1 = r.stage[Q3(
                    tracer, interface - 1, cell, nlev, ncells)];
                const float qi = r.stage[Q3(
                    tracer, interface, cell, nlev, ncells)];
                const float qip1 = r.stage[Q3(
                    tracer, interface + 1, cell, nlev, ncells)];
                const float base = mpas_div(mpas_mul(vel, mpas_sub(
                    mpas_mul(7.0f, mpas_add(qi, qim1)),
                    mpas_add(qip1, qim2))),
                    mpas_third_order_denominator);
                const float correction = mpas_div(mpas_mul(
                    mpas_mul(coefficient, mpas_abs(vel)),
                    mpas_sub(mpas_sub(qip1, qim2),
                        mpas_mul(3.0f, mpas_sub(qi, qim1)))),
                    mpas_third_order_denominator);
                result = mpas_add(base, correction);
            }
        }
        r.vertical_flux[QI3(tracer, interface, cell, nlev, ncells)] = result;
    }
}
