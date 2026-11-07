// Generated from classic RRTM GASABS Pade index; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_rrtm_lw_make_itr_bindings.metalh"
using namespace metal;
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_greater_equal(b.bits,a.bits); }
kernel void wrf_exact_rrtm_lw_make_itr(
    constant WrfExactRrtmLwMakeItrResources &r [[buffer(0)]],
    constant WrfExactRrtmLwMakeItrParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.npoint))return;
    const Exact32 od=1.66f*reinterpret_cast<device const Exact32*>(r.taug)[tid];
    Exact32 tf=od/(Exact32(p.bpade)+od);
    if(od<=0.0f)tf=0.0f;
    r.itr[tid]=clamp(int(trunc(as_type<float>((5000.0f*tf+0.5f).bits))),0,5000);
}
