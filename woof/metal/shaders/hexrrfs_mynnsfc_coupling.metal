#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_seaice_math.metalh"
#include "hexrrfs_mynnsfc_coupling_bindings.metalh"
#pragma clang fp contract(off)

// Packed row order from sf_mynnsfc_v600.FIELDS.
enum PackedRow : uint {
    ZNT = 21, UST, USTM, MAVAIL = 25, ZOL, MOL, RMOL, PSIM, PSIH, XLAND,
    TSK = 36, QSFC = 39, U10, V10, TH2, T2, Q2 = 45, GZ1OZ0 = 47,
    WSPD, BR, DX, CH, CK, CKA, CD, CDA, ROW_COUNT = 68,
};

kernel void hexrrfs_mynnsfc_from_mpas(
    constant HexrrfsMynnsfcFromMpasResources &r [[buffer(0)]],
    constant HexrrfsMynnsfcFromMpasParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.ncol)) return;
    const uint n = uint(p.ncol);
    const float power = seaice_pow(r.density[i], 0.25f);
    r.fields[DX * n + i] = as_type<float>(
        soft32_div(as_type<uint>(p.len_disp), as_type<uint>(power)));
    if (p.fraction) {
        for (uint row = 0; row < ROW_COUNT; ++row)
            r.sea[row * n + i] = r.fields[row * n + i];
        r.sea[QSFC * n + i] = 0.0f;
        const uint ice = as_type<uint>(r.ice[i]);
        if (soft32_greater_equal(ice, as_type<uint>(p.threshold))
                && soft32_less(ice, as_type<uint>(1.0f))) {
            r.sea[XLAND * n + i] = 2.0f;
            r.fields[XLAND * n + i] = 1.0f;
            r.sea[MAVAIL * n + i] = 1.0f;
            r.sea[ZNT * n + i] = 0.0001f;
            r.sea[TSK * n + i] = mpas_max(r.sst[i], 271.4f);
        }
    }
}

kernel void hexrrfs_mynnsfc_to_mpas(
    constant HexrrfsMynnsfcToMpasResources &r [[buffer(0)]],
    constant HexrrfsMynnsfcToMpasParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.ncol)) return;
    const uint ice = as_type<uint>(r.ice[i]);
    const uint one = as_type<uint>(1.0f);
    if (soft32_less(ice, as_type<uint>(p.threshold)) || soft32_greater_equal(ice, one)) return;
    const uint water = soft32_sub(one, ice);
    const uint n = uint(p.ncol);
    const uint rows[] = {BR, GZ1OZ0, MOL, PSIH, PSIM, RMOL, UST, WSPD, ZOL, Q2,
                         T2, TH2, U10, V10, CD, CDA, CK, CKA, USTM, CH};
    for (uint row : rows) {
        const uint index = row * n + i;
        const uint a = soft32_mul(as_type<uint>(r.fields[index]), ice);
        const uint b = soft32_mul(water, as_type<uint>(r.sea[index]));
        r.fields[index] = as_type<float>(soft32_add(a, b));
    }
}
