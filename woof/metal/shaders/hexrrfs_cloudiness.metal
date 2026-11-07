#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexrrfs_cloudiness_bindings.metalh"
#pragma clang fp contract(off)

// RRFS cloud_from and cloud_condensate, native [level,cell] layout.
kernel void hexrrfs_cloudiness(
    constant HexrrfsCloudinessResources &r [[buffer(0)]],
    constant HexrrfsCloudinessParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.nlev * p.ncells)) return;
    int i = int(gid), col = i % p.ncells;
    float qc = r.qc[i], qi = r.qi[i], frac = 0.f;
    if (p.mynn) {
        if (p.step > 0) {
            if (r.cldfrac_bl[i] > .001f) frac = r.cldfrac_bl[i];
        } else if (qc > 1.e-6f || qi > 1.e-9f || r.qs[i] > 1.e-8f) frac = 1.f;
    }
    if (p.gfl && r.cldfrac_cu[i] > .001f)
        frac = min(1.f, mpas_add(frac, r.cldfrac_cu[i]));
    if (p.mynn && frac > .001f) {
        if (qc < 1.e-6f) qc = mpas_add(qc, r.qc_bl[i]);
        if (qi < 1.e-9f) qi = mpas_add(qi, r.qi_bl[i]);
    }
    if (p.gfl && frac > .001f) {
        if (qc < 1.e-6f) qc = mpas_add(qc, r.qc_cu[i]);
        if (qi < 1.e-9f) qi = mpas_add(qi, r.qi_cu[i]);
    }
    r.qvrad[i] = r.qv[i]; r.qcrad[i] = qc; r.qirad[i] = qi; r.qsrad[i] = r.qs[i];
    r.cldfrac[i] = frac; r.diag_cldfrac[i] = frac;
    if (i < p.ncells) {
        r.dx[col] = mpas_mul(mpas_div(p.len_disp, gfk_pow(r.density[col], .25f)), .001f);
        r.land[col] = r.xland[col];
    }
}
