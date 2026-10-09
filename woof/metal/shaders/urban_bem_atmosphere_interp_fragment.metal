#include <metal_stdlib>
#include "urban_bem_atmosphere_interp_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmi_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmi_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmi_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmi_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmi_min(float a, float b) { return a < b ? a : b; }
static inline float ubmi_max(float a, float b) { return a > b ? a : b; }

// WRF bb_interpol (urban_bep_bem.cu:1094-1145), applied to all seven
// atmospheric fields in source call order. One item is one active column;
// z and each field are laid out [column, mass-level], z_u is the shared
// 19-interface BEM grid, and output is [column, field, urban-level].
// Fields: ua, va, pt, pt0, pr, da, qv. The overlap sum preserves the source
// k-ascending FP32 order and zero-fills levels above the class-specific nzu.
kernel void urban_bem_atmosphere_interp_fragment(
    constant UrbanBemAtmosphereInterpFragmentResources &r [[buffer(0)]],
    constant UrbanBemAtmosphereInterpFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    device const int *active_cols = r.active_cols;
    device const int *utype = r.utype;
    device const int *nzu_class = r.nzu_class;
    device const float *z = r.z;
    device const float *z_u = r.z_u;
    device const float *ua = r.ua;
    device const float *va = r.va;
    device const float *pt = r.pt;
    device const float *pt0 = r.pt0;
    device const float *pr = r.pr;
    device const float *da = r.da;
    device const float *qv = r.qv;
    device float *interp = r.interp;
    device int *error = r.error;
    const int ncol = p.ncol;
    const int nz = p.nz;
    const int nurbm = p.nurbm;
    if (col >= uint(ncol)) return;
    const ulong out = ulong(col) * 7ul * 18ul;
    for (int i = 0; i < 7 * 18; ++i) interp[out + uint(i)] = 0.0f;
    error[col] = 0;
    if (active_cols[col] == 0) return;
    if (nz < 1 || nurbm < 1 || nurbm > 11) {
        error[col] = 5501;
        return;
    }
    const int one_based_type = utype[col];
    if (one_based_type < 1 || one_based_type > nurbm) {
        error[col] = 5502;
        return;
    }
    const int nzu = nzu_class[uint(one_based_type - 1)];
    if (nzu < 1 || nzu > 18) {
        error[col] = 5503;
        return;
    }
    const ulong column = ulong(col) * uint(nz);
    const ulong zbase = ulong(col) * (uint(nz) + 1ul);
    for (int field = 0; field < 7; ++field) {
        for (int izu = 0; izu < nzu; ++izu) {
            float total = 0.0f;
            const float lower_u = z_u[uint(izu)];
            const float upper_u = z_u[uint(izu + 1)];
            const float layer = ubmi_sub(upper_u, lower_u);
            if (!(layer > 0.0f)) {
                error[col] = 5504;
                return;
            }
            for (int iz = 0; iz < nz; ++iz) {
                const float overlap_hi = ubmi_min(z[zbase + uint(iz + 1)], upper_u);
                const float overlap_lo = ubmi_max(z[zbase + uint(iz)], lower_u);
                const float dz = ubmi_max(ubmi_sub(overlap_hi, overlap_lo), 0.0f);
                float value = 0.0f;
                switch (field) {
                    case 0: value = ua[column + uint(iz)]; break;
                    case 1: value = va[column + uint(iz)]; break;
                    case 2: value = pt[column + uint(iz)]; break;
                    case 3: value = pt0[column + uint(iz)]; break;
                    case 4: value = pr[column + uint(iz)]; break;
                    case 5: value = da[column + uint(iz)]; break;
                    default: value = qv[column + uint(iz)]; break;
                }
                total = ubmi_add(total, ubmi_mul(value, dz));
            }
            interp[out + uint(field * 18 + izu)] = ubmi_div(total, layer);
        }
    }
}
