// Strict nesting smoother; order follows core/kernels/nest.cu.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_nest_smooth_bindings.metalh"
using namespace metal;
static inline ulong nest_i3(int k,int j,int i,int ny,int nx) { return (ulong(k)*ulong(ny)+ulong(j))*ulong(nx)+ulong(i); }
static inline Exact32 nest_smooth_filter(int mode, Exact32 xnu,
                                       Exact32 a, Exact32 b, Exact32 c) {
    if (mode == 0) return Exact32(0.25f) * ((c + Exact32(2.0f) * b) + a);
    return b + xnu * ((c + a) * Exact32(0.5f) - b);
}

kernel void wrf_exact_nest_smooth_j(
    constant WrfExactNestSmoothJResources &r [[buffer(0)]],
    constant WrfExactNestSmoothJParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nit = p.niw + 2;
    const int njt = p.njw + 2;
    const ulong total = ulong(p.nz) * ulong(njt) * ulong(nit);
    if (ulong(tid) >= total) return;
    const int i = p.i0 - 1 + int(tid % uint(nit));
    const int j = p.j0 - 1 + int((tid / uint(nit)) % uint(njt));
    const int k = int(tid / (ulong(nit) * ulong(njt)));
    const bool inside = i >= p.i0 && i < p.i0 + p.niw &&
                        j >= p.j0 && j < p.j0 + p.njw;
    const ulong at = nest_i3(k, j, i, p.nyp, p.nxp);
    const Exact32 b = reinterpret_cast<device const Exact32 *>(r.parent)[at];
    if (!inside) { reinterpret_cast<device Exact32 *>(r.scratch)[at] = b; return; }
    const Exact32 a = reinterpret_cast<device const Exact32 *>(r.parent)[nest_i3(k, j - 1, i, p.nyp, p.nxp)];
    const Exact32 c = reinterpret_cast<device const Exact32 *>(r.parent)[nest_i3(k, j + 1, i, p.nyp, p.nxp)];
    reinterpret_cast<device Exact32 *>(r.scratch)[at] = nest_smooth_filter(p.mode, Exact32(p.xnu), a, b, c);
}

kernel void wrf_exact_nest_smooth_i(
    constant WrfExactNestSmoothIResources &r [[buffer(0)]],
    constant WrfExactNestSmoothIParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong total = ulong(p.nz) * ulong(p.njw) * ulong(p.niw);
    if (ulong(tid) >= total) return;
    const int i = p.i0 + int(tid % uint(p.niw));
    const int j = p.j0 + int((tid / uint(p.niw)) % uint(p.njw));
    const int k = int(tid / (ulong(p.niw) * ulong(p.njw)));
    const Exact32 a = reinterpret_cast<device const Exact32 *>(r.scratch)[nest_i3(k, j, i - 1, p.nyp, p.nxp)];
    const Exact32 b = reinterpret_cast<device const Exact32 *>(r.scratch)[nest_i3(k, j, i, p.nyp, p.nxp)];
    const Exact32 c = reinterpret_cast<device const Exact32 *>(r.scratch)[nest_i3(k, j, i + 1, p.nyp, p.nxp)];
    reinterpret_cast<device Exact32 *>(r.parent)[nest_i3(k, j, i, p.nyp, p.nxp)] =
        nest_smooth_filter(p.mode, Exact32(p.xnu), a, b, c);
}

kernel void wrf_exact_nest_clamp(constant WrfExactNestClampResources &r [[buffer(0)]],constant WrfExactNestClampParams &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 const uint count=uint(p.nz*p.njw*p.niw); if (tid>=count) return;
 int i=p.i0+int(tid%uint(p.niw)); int j=p.j0+int((tid/uint(p.niw))%uint(p.njw)); int k=int(tid/uint(p.niw*p.njw));
 device Exact32 *a=reinterpret_cast<device Exact32 *>(r.parent); ulong at=nest_i3(k,j,i,p.nyp,p.nxp); Exact32 x=a[at];
 if (!soft32_is_nan(x.bits)) a[at]=exact_max(x,Exact32(0.0f));
}
