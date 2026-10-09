#include <metal_stdlib>
#include "vertical_interpolate_logp_bindings.metalh"

using namespace metal;

// Preserve the CUDA source's visible FP32 operation grouping. The default
// Metal log/pow implementations are not claimed bitwise identical to CUDA.
#pragma clang fp contract(off)

kernel void vertical_interpolate_logp(
    constant VerticalInterpolateLogpResources &r [[buffer(0)]],
    constant VerticalInterpolateLogpParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total = uint(p.ntarget * p.ncolumn);
    if (tid >= total) return;
    const int kt = int(tid) / p.ncolumn;
    const int c = int(tid) - kt * p.ncolumn;
    const float pt = r.target_p[ulong(kt) * ulong(p.ncolumn) + ulong(c)];
    const float pbottom = r.source_p[c];
    const float ptop = r.source_p[
        ulong(p.nsource - 1) * ulong(p.ncolumn) + ulong(c)];

    if (pt > pbottom) {
        float q = r.field[c];
        if (p.below_temperature != 0) {
            const float t1 = q * pow(pbottom / p.p0, p.rcp);
            const float pavg = 0.5f * (pt + pbottom);
            const float dhdp = 11880.516f * 0.1902632f
                * pow(pavg / 100.0f, 0.1902632f - 1.0f);
            const float dt = dhdp * ((pt - pbottom) / 100.0f) * 0.0065f;
            q = (t1 + dt) * pow(p.p0 / pt, p.rcp);
        }
        r.output[tid] = q;
        return;
    }
    if (pt < ptop) {
        // The caller rejects a target above the source top before submission.
        r.output[tid] = as_type<float>(0x7fc00000u);
        return;
    }

    for (int k = 0; k < p.nsource - 1; ++k) {
        const ulong aidx = ulong(k) * ulong(p.ncolumn) + ulong(c);
        const ulong bidx = ulong(k + 1) * ulong(p.ncolumn) + ulong(c);
        const float pa = r.source_p[aidx];
        const float pb = r.source_p[bidx];
        if (pa >= pt && pt >= pb) {
            const float qa = r.field[aidx];
            const float qb = r.field[bidx];
            const float weight = (log(pt) - log(pa)) / (log(pb) - log(pa));
            r.output[tid] = qa + weight * (qb - qa);
            return;
        }
    }
    r.output[tid] = as_type<float>(0x7fc00000u);
}
