#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "soft_glibc_solar_trig.metalh"
#include "hexrrfs_tempo_native_init_aerosols_bindings.metalh"
using namespace ntc;
#pragma clang fp contract(off)

static inline F load_word(device const float *p, ulong i) {
    return f32_word(as_type<uint>(p[i]));
}

// The framework folds the log of each concentration ratio before launch.
// Interfaces and species are stored level-major with the full cell stride.
kernel void hexrrfs_tempo_native_init_aerosols(
    constant HexrrfsTempoNativeInitAerosolsResources &r [[buffer(0)]],
    constant HexrrfsTempoNativeInitAerosolsParams &p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.solve)) return;
    const ulong stride = uint(p.nc);
    const F z0 = load_word(r.zgrid, c);
    const F z1 = load_word(r.zgrid, stride + c);
    if (p.initw || p.initi) {
        const F h1 = F(0.5f) * (z0 + z1);
        F h01;
        if (h1 <= F(1000.0f)) {
            h01 = F(0.8f);
        } else if (h1 >= F(2500.0f)) {
            h01 = F(0.01f);
        } else {
            const F angle = h1 * F(0.001f) - F(1.0f);
            h01 = F(0.8f) * f32_word(soft_glibc_solar_cosf(angle.bits));
        }
        const F decay_w = tc_div(F(-1.0f) * F(p.log_w), h01);
        const F decay_i = tc_div(F(-1.0f) * F(p.log_i), h01);
        for (int k = 0; k < p.nz; ++k) {
            const ulong x = ulong(k) * stride + c;
            const ulong z = k == 0 ? x + stride : x;
            const F h = F(0.5f) * (load_word(r.zgrid, z) + load_word(r.zgrid, z + stride));
            const F dh = -tc_div(h - h1, F(1000.0f));
            if (p.initw) {
                const F nwfa = F(50.0e6f) + F(300.0e6f) * tc_exp(dh * decay_w);
                r.nwfa[x] = as_type<float>(nwfa.bits);
            }
            if (p.initi) {
                const F nifa = F(0.5e6f) + F(1.5e6f) * tc_exp(dh * decay_i);
                r.nifa[x] = as_type<float>(nifa.bits);
            }
        }
    }
    F dz = z1 - z0;
    if (dz < F(9.0f)) dz = F(9.0f);
    const F nwfa = tc_max(F(11.1e6f), tc_min(load_word(r.nwfa, c), F(9999.0e6f)));
    const F source = nwfa * F(0.000196f) * tc_div(F(5.0f), dz)
                     * tc_div(load_word(r.area, c), F(9.0e6f));
    r.nwfa2d[c] = as_type<float>(source.bits);
    r.nifa2d[c] = 0.0f;
}
