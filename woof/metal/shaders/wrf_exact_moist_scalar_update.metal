#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_moist_scalar_update_bindings.metalh"
using namespace metal;
static inline device const Exact32 *ms_words(device const float *p) { return reinterpret_cast<device const Exact32 *>(p); }
static inline device Exact32 *ms_words(device float *p) { return reinterpret_cast<device Exact32 *>(p); }
kernel void wrf_exact_moist_scalar_update(
    constant WrfExactMoistScalarUpdateResources &r [[buffer(0)]],
    constant WrfExactMoistScalarUpdateParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong plane=ulong(p.ny)*ulong(p.nx), flat=ulong(gid);
    if(flat>=ulong(p.nz)*plane) return;
    const ulong k=flat/plane, cell=flat-k*plane;
    Exact32 t=ms_words(r.tend)[flat];
    if(p.has_msf) t=t*ms_words(r.msft)[cell];
    const Exact32 chm0=ms_words(r.c1h)[k]*ms_words(r.mu0)[cell]+ms_words(r.c2h)[k];
    const Exact32 chm=ms_words(r.c1h)[k]*ms_words(r.mu)[cell]+ms_words(r.c2h)[k];
    Exact32 value=chm0*ms_words(r.q0)[flat];
    t=Exact32(p.dt_eff)*t;
    value=value+t;
    ms_words(r.q)[flat]=value/chm;
}
