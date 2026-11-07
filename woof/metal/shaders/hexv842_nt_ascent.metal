#include "hexv842_nt_ascent_bindings.metalh"
#include <metal_stdlib>
using namespace metal;

// One thread owns the whole solve slab for the source ascent latch.
kernel void hexv842_nt_ascent(constant Hexv842NtAscentResources &r [[buffer(0)]], constant Hexv842NtAscentParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const int *ldcum=r.ldcum;
    device const int *ktype=r.ktype;
    device const int *klab=r.klab;
    device const float *pqen=r.pqen;
    device const float *pqsen=r.pqsen;
    device const float *pgeo=r.pgeo;
    device int *result=r.result;

    if (gid >= 1) return;
    int start = 0;
    float zrg = 1.0f / p.grav;
    for (int i = 0; i < p.ncol; ++i) {
        int first = 0;
        if (ldcum[i] && ktype[i] != 3) {
            if (klab[p.klev * p.ncol + i] > 0) first = p.klev - 1;
        } else {
            for (int jk = p.klev - 1; jk >= 3; --jk) {
                int j = jk * p.ncol + i;
                float h = pgeo[j] * zrg;
                if (pqen[j] > 0.80f * pqsen[j] && h > 500.0f && h < 10000.0f) {
                    first = jk;
                    break;
                }
            }
        }
        start = max(start, first);
    }
    result[0] = start;
    for (int jk = p.klev - 1; jk >= 3; --jk)
        result[1 + jk] = int(jk <= start);
}
