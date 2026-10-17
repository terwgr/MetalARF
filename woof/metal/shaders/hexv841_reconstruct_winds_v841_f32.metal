// Hex v8.4.1 Metal transcription of CUDA kernel reconstruct_winds_v841_f32 (woof/hex/cuda_physics_prep_v841.py).
#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv841_reconstruct_winds_v841_f32_bindings.metalh"
#pragma clang fp contract(off)

#define C2(k,c,nc) ((k)*(nc) + (c))
#define E2(k,e,ne) ((k)*(ne) + (e))
#define S3(s,k,c,nl,nc) ((((s)*(nl) + (k))*(nc)) + (c))
#define atomicExch(address, value) \
    atomic_exchange_explicit((device atomic_int *)(address), (value), memory_order_relaxed)

static inline uint prep_positive_q_bits_v841(const float raw)
{
    const uint bits = as_type<uint>(raw);
    return (bits & 0x80000000u) == 0u ? bits : 0u;
}

static inline float prep_positive_q_v841(const float raw)
{
    return as_type<float>(prep_positive_q_bits_v841(raw));
}

static inline bool prep_finite_bits_v841(const float raw)
{
    return (as_type<uint>(raw) & 0x7fffffffu) < 0x7f800000u;
}

static inline bool prep_negative_nonzero_bits_v841(const float raw)
{
    const uint bits = as_type<uint>(raw);
    return (bits & 0x80000000u) != 0u
        && (bits & 0x7fffffffu) != 0u;
}


kernel void hexv841_reconstruct_winds_v841_f32(
    constant Hexv841ReconstructWindsV841F32Resources &r [[buffer(0)]],
    constant Hexv841ReconstructWindsV841F32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nlev = p.nlev;
    const int ncells = p.ncells;
    const int nedges = p.nedges;
    const int maxedges = p.maxedges;
    device const float *normal_velocity = r.normal_velocity;
    device const int *edges_on_cell = r.edges_on_cell;
    device const int *n_edges_on_cell = r.n_edges_on_cell;
    device const float *coeffs_reconstruct = r.coeffs_reconstruct;
    device const float *lat_cell = r.lat_cell;
    device const float *lon_cell = r.lon_cell;
    device const int *sealed_edges_on_cell = r.sealed_edges_on_cell;
    device const int *sealed_n_edges_on_cell = r.sealed_n_edges_on_cell;
    device const float *sealed_coeffs_reconstruct = r.sealed_coeffs_reconstruct;
    device const float *sealed_lat_cell = r.sealed_lat_cell;
    device const float *sealed_lon_cell = r.sealed_lon_cell;
    device float *x = r.x;
    device float *y = r.y;
    device float *z = r.z;
    device float *u_p = r.u_p;
    device float *v_p = r.v_p;
    device int *invalid = r.invalid;

    const int cell = int(tid);
    if (cell >= ncells) return;
    bool geometry_changed =
        n_edges_on_cell[cell] != sealed_n_edges_on_cell[cell]
        || as_type<uint>(lat_cell[cell])
            != as_type<uint>(sealed_lat_cell[cell])
        || as_type<uint>(lon_cell[cell])
            != as_type<uint>(sealed_lon_cell[cell]);
    for (int slot = 0; slot < maxedges; ++slot) {
        const int edge_index = cell * maxedges + slot;
        const int coeff_index = edge_index * 3;
        geometry_changed = geometry_changed
            || edges_on_cell[edge_index] != sealed_edges_on_cell[edge_index]
            || as_type<uint>(coeffs_reconstruct[coeff_index])
                != as_type<uint>(sealed_coeffs_reconstruct[coeff_index])
            || as_type<uint>(coeffs_reconstruct[coeff_index + 1])
                != as_type<uint>(sealed_coeffs_reconstruct[coeff_index + 1])
            || as_type<uint>(coeffs_reconstruct[coeff_index + 2])
                != as_type<uint>(sealed_coeffs_reconstruct[coeff_index + 2]);
    }
    if (geometry_changed) {
        atomicExch(invalid, 1);
        return;
    }
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        x[i] = 0.0f;
        y[i] = 0.0f;
        z[i] = 0.0f;
    }
    const int count = n_edges_on_cell[cell];
    if (count < 0 || count > maxedges) {
        atomicExch(invalid, 1);
        return;
    }
    for (int slot = 0; slot < count; ++slot) {
        const int edge = edges_on_cell[cell * maxedges + slot];
        if (edge < 0 || edge >= nedges) {
            atomicExch(invalid, 1);
            continue;
        }
        const int ci = (cell * maxedges + slot) * 3;
        const float cx = coeffs_reconstruct[ci];
        const float cy = coeffs_reconstruct[ci + 1];
        const float cz = coeffs_reconstruct[ci + 2];
        if (!isfinite(cx) || !isfinite(cy) || !isfinite(cz))
            atomicExch(invalid, 1);
        for (int k = 0; k < nlev; ++k) {
            const int i = k * ncells + cell;
            const float value = normal_velocity[k * nedges + edge];
            if (!isfinite(value)) atomicExch(invalid, 1);
            x[i] = mpas_add(x[i], mpas_mul(cx, value));
            y[i] = mpas_add(y[i], mpas_mul(cy, value));
            z[i] = mpas_add(z[i], mpas_mul(cz, value));
        }
    }
    const float clat = precise::cos(lat_cell[cell]);
    const float slat = precise::sin(lat_cell[cell]);
    const float clon = precise::cos(lon_cell[cell]);
    const float slon = precise::sin(lon_cell[cell]);
    if (!isfinite(clat) || !isfinite(slat) || !isfinite(clon) || !isfinite(slon))
        atomicExch(invalid, 1);
    for (int k = 0; k < nlev; ++k) {
        const int i = k * ncells + cell;
        u_p[i] = mpas_add(
            mpas_mul(mpas_sub(0.0f, x[i]), slon), mpas_mul(y[i], clon));
        const float horizontal = mpas_add(
            mpas_mul(x[i], clon), mpas_mul(y[i], slon));
        v_p[i] = mpas_add(
            mpas_mul(mpas_sub(0.0f, horizontal), slat), mpas_mul(z[i], clat));
    }
}
