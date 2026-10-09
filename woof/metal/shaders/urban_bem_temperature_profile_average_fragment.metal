#include <metal_stdlib>
#include "urban_bem_temperature_profile_average_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmt_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmt_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmt_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubmt_pow4(float x) { const float x2 = ubmt_mul(x, x); return ubmt_mul(x2, x2); }

// WRF bb_bep1d, module_sf_bep_bem.F:1524-1532. One thread owns a column.
// It copies the top urban-layer temperature for each direction and computes
// the roof/vegetated-roof fourth-power blend in the original FP32 order.
// State arrays keep their source packing: tg[col,10,2], tr[col,5,18,2],
// trv[col,10,18,2]. The roof fraction is class-float offset 1206 indexed by
// one-based UTYPE. Outputs are tg_av[col,2] and tr_av[col,18,2].
kernel void urban_bem_temperature_profile_average_fragment(
    constant UrbanBemTemperatureProfileAverageFragmentResources &r [[buffer(0)]],
    constant UrbanBemTemperatureProfileAverageFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    device const int *active_cols = r.active_cols;
    device const int *utype = r.utype;
    device const float *class_float = r.class_float;
    device const float *tg = r.tg;
    device const float *tr = r.tr;
    device const float *trv = r.trv;
    device float *tg_av = r.tg_av;
    device float *tr_av = r.tr_av;
    device int *error = r.error;
    const int ncol = p.ncol;
    if (col >= uint(ncol)) return;
    const ulong tg_out = ulong(col) * 2ul;
    const ulong tr_out = ulong(col) * 36ul;
    for (int dir = 0; dir < 2; ++dir) tg_av[tg_out + uint(dir)] = 0.0f;
    for (int iz = 0; iz < 18; ++iz)
        for (int dir = 0; dir < 2; ++dir)
            tr_av[tr_out + uint(iz * 2 + dir)] = 0.0f;
    error[col] = 0;
    if (active_cols[col] == 0) return;
    const int type = utype[col];
    if (type < 1 || type > 11) { error[col] = 5521; return; }
    const float gr_frac_roof = class_float[1206ul + uint(type - 1)];
    const float nongreen_fraction = ubmt_sub(1.0f, gr_frac_roof);
    const ulong tg_base = ulong(col) * 20ul + 18ul;
    const ulong tr_base = ulong(col) * 180ul + 144ul;
    const ulong trv_base = ulong(col) * 360ul + 324ul;
    for (int dir = 0; dir < 2; ++dir)
        tg_av[tg_out + uint(dir)] = tg[tg_base + uint(dir)];
    for (int iz = 0; iz < 18; ++iz) {
        for (int dir = 0; dir < 2; ++dir) {
            const float roof4 = ubmt_pow4(tr[tr_base + uint(iz * 2 + dir)]);
            const float green4 = ubmt_pow4(trv[trv_base + uint(iz * 2 + dir)]);
            const float roof_term = ubmt_mul(nongreen_fraction, roof4);
            const float green_term = ubmt_mul(gr_frac_roof, green4);
            const float mixed = ubmt_add(roof_term, green_term);
            tr_av[tr_out + uint(iz * 2 + dir)] = powr(mixed, 0.25f);
        }
    }
}
