#include <metal_stdlib>
#include "urban_bem_atmosphere_gather_fragment_bindings.metalh"
using namespace metal;

// WRF module_sf_bep_bem.F:759-778. The shared BEP gather already computes
// z1d and copies u/v/theta/rho/p with the same source level ordering. This
// stage materializes BEM's local one-dimensional inputs, gathers qv, sets
// pt0 to 300 K, and zero-initializes its atmospheric tendency vectors.
kernel void urban_bem_atmosphere_gather_fragment(
    constant UrbanBemAtmosphereGatherFragmentResources &r [[buffer(0)]],
    constant UrbanBemAtmosphereGatherFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    device const int *active = r.active_cols;
    device const float *column_x = r.column_x;
    device const float *qv = r.qv;
    device float *ua1d = r.ua1d;
    device float *va1d = r.va1d;
    device float *pt1d = r.pt1d;
    device float *da1d = r.da1d;
    device float *pr1d = r.pr1d;
    device float *pt01d = r.pt01d;
    device float *z1d = r.z1d;
    device float *qv1d = r.qv1d;
    device float *a_u1d = r.a_u1d;
    device float *a_v1d = r.a_v1d;
    device float *a_t1d = r.a_t1d;
    device float *a_e1d = r.a_e1d;
    device float *b_u1d = r.b_u1d;
    device float *b_v1d = r.b_v1d;
    device float *b_t1d = r.b_t1d;
    device float *b_ac1d = r.b_ac1d;
    device float *b_e1d = r.b_e1d;
    device int *error = r.error;

    if (col >= uint(p.ncol)) return;
    error[col] = 0;
    if (p.ncol <= 0 || p.nz <= 0) { error[col] = 6520; return; }
    const ulong nlev = ulong(p.nz);
    const ulong slots = nlev + 1ul;
    const ulong source = ulong(col) * 15ul * slots;
    const ulong out = ulong(col) * nlev;
    const ulong qv_col = ulong(col);
    for (ulong k = 0; k <= nlev; ++k)
        z1d[ulong(col) * slots + k] = column_x[source + k];
    if (active[col] == 0) return;

    for (ulong k = 0; k < nlev; ++k) {
        const ulong level = k + 1ul;
        const ulong atmospheric = source + level;
        const ulong index = out + k;
        ua1d[index] = column_x[atmospheric + slots];
        va1d[index] = column_x[atmospheric + 2ul * slots];
        pt1d[index] = column_x[atmospheric + 3ul * slots];
        da1d[index] = column_x[atmospheric + 4ul * slots];
        pr1d[index] = column_x[atmospheric + 5ul * slots];
        pt01d[index] = column_x[atmospheric + 6ul * slots];
        qv1d[index] = qv[k * ulong(p.ncol) + qv_col];
        a_u1d[index] = 0.0f;
        a_v1d[index] = 0.0f;
        a_t1d[index] = 0.0f;
        a_e1d[index] = 0.0f;
        b_u1d[index] = 0.0f;
        b_v1d[index] = 0.0f;
        b_t1d[index] = 0.0f;
        b_ac1d[index] = 0.0f;
        b_e1d[index] = 0.0f;
    }
}
