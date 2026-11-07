#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_seaice_rp3_f32_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv842_mpas_seaice_rp3_f32(
    constant Hexv842MpasSeaiceRp3F32Resources &r [[buffer(0)]],
    constant Hexv842MpasSeaiceRp3F32Params &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    float x = r.ice[i], blended = r.skin[i];
    if (x >= p.threshold && x <= 1.0f) {
        float sea = fmax(r.sst[i], 271.4f);
        float ice = cu_div(mpas_sub(blended, mpas_mul(mpas_sub(1.0f, x), sea)), x);
        ice = fmax(ice, 221.4f);
        if (x < 0.2f && blended < 253.15f) ice = 253.15f;
        if (x < 0.1f && blended < 263.15f) ice = 263.15f;
        r.fields[p.sea_index * p.n + i] = sea;
        r.fields[p.ice_index * p.n + i] = ice;
    }
}
