#include <metal_stdlib>
#include "p3_homofreeze_bindings.metalh"

using namespace metal;

// P3's homogeneous freezing stage (p3.cu:1784-1824, 1928-1938). One thread
// owns a complete column. The qv snapshot and then each level's source
// statements retain their original order.
#pragma clang fp contract(off)

constant float P3_QSMALL = 0x1.6849b80000000p-47f;
constant float P3_NSMALL = 0x1.cd2b2a0000000p-54f;
constant float P3_INV_RHO_RIMEMAX = 0x1.2345680000000p-10f;
constant float P3_XLF = 0x1.45e1000000000p+18f;
constant float P3_INV_CP = 0x1.04d6fe0000000p-10f;

kernel void p3_homofreeze(
    constant P3HomofreezeResources &r [[buffer(0)]],
    constant P3HomofreezeParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny) * uint(p.nx);
    if (column >= ncol || r.flags[ncol + column] == 0.0f) return;

    for (uint k = 0; k < uint(p.nz); ++k) {
        const ulong idx = ulong(k) * ulong(ncol) + ulong(column);
        r.qv_cld[idx] = r.qv[idx];
    }
    for (uint k = 0; k < uint(p.nz); ++k) {
        const ulong idx = ulong(k) * ulong(ncol) + ulong(column);
        const float t = r.temperature[idx];
        const float invexn = 1.0f / r.tmparr1[idx];

        if (r.qc[idx] >= P3_QSMALL && t < 233.15f) {
            const float q_nuc = r.qc[idx];
            r.nc[idx] = fmax(r.nc[idx], P3_NSMALL);
            const float n_nuc = r.nc[idx];
            r.qir[idx] = r.qir[idx] + q_nuc;
            r.qi[idx] = r.qi[idx] + q_nuc;
            r.qib[idx] = r.qib[idx] + q_nuc * P3_INV_RHO_RIMEMAX;
            r.ni[idx] = r.ni[idx] + n_nuc;
            r.theta[idx] = r.theta[idx]
                + invexn * q_nuc * P3_XLF * P3_INV_CP;
            r.qc[idx] = 0.0f;
            r.nc[idx] = 0.0f;
        }
        if (r.qr[idx] >= P3_QSMALL && t < 233.15f) {
            const float q_nuc = r.qr[idx];
            r.nr[idx] = fmax(r.nr[idx], P3_NSMALL);
            const float n_nuc = r.nr[idx];
            r.qir[idx] = r.qir[idx] + q_nuc;
            r.qi[idx] = r.qi[idx] + q_nuc;
            r.qib[idx] = r.qib[idx] + q_nuc * P3_INV_RHO_RIMEMAX;
            r.ni[idx] = r.ni[idx] + n_nuc;
            r.theta[idx] = r.theta[idx]
                + invexn * q_nuc * P3_XLF * P3_INV_CP;
            r.qr[idx] = 0.0f;
            r.nr[idx] = 0.0f;
        }
    }
}
