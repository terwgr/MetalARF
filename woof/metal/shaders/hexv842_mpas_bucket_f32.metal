#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_mpas_bucket_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_bucket_f32(
    constant Hexv842MpasBucketF32Resources &r [[buffer(0)]],
    constant Hexv842MpasBucketF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int i=int(tid);
    if (i>=p.n || i<p.begin || i>=p.end) return;
    r.total[i]=mpas_add(r.total[i],mpas_mul(r.rate[i],p.dt));
    if (p.ringing && p.bucket>0.f && r.total[i]>p.bucket) {
        r.count[i]+=1;
        r.total[i]=mpas_sub(r.total[i],p.bucket);
    }
}
