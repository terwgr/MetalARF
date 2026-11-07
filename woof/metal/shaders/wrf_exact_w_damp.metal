// Generated from woof/core/kernels/openbc.cu by tools/w_damp_wrf_exact/generate.py; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_w_damp_bindings.metalh"
using namespace metal;
namespace wrf_exact_w_damp_source {
static inline Exact32 exact_copysign(Exact32 a, Exact32 b)
{ return Exact32::from_bits((a.bits & 0x7fffffffu) | (b.bits & 0x80000000u)); }
void w_damp(device Exact32*  rw_t,
            device const Exact32*  ww,
            device const Exact32*  w,
            device const Exact32*  mup,
            device const Exact32*  mub2d,
            device const Exact32*  c1f,
            device const Exact32*  c2f,
            device const Exact32*  rdnw,
            Exact32 dt, Exact32 w_damp_on, Exact32 w_crit_cfl,
            int nz, int ny, int nx, int t)
{

    int plane = ny * nx;
    if (t >= (nz - 1) * plane) return;
    int k = t / plane + 1;
    int c = t - (k - 1) * plane;
    Exact32 m = exact_add(exact_mul(c1f[k], exact_add(mub2d[c], mup[c])), c2f[k]);
    ulong ix = (ulong)k * plane + c;
    Exact32 vert_cfl = exact_abs(exact_mul(exact_mul(exact_div(ww[ix], m), rdnw[k]), dt));
    if (vert_cfl > w_damp_on) {
        Exact32 damp = exact_mul(exact_copysign(1.0f, w[ix]), 0.3f);
        damp = exact_mul(damp, exact_sub(vert_cfl, w_crit_cfl));
        rw_t[ix] = exact_sub(rw_t[ix], exact_mul(damp, m));
    }
}
}
kernel void wrf_exact_w_damp(
    constant WrfExactWDampResources &r [[buffer(0)]],
    constant WrfExactWDampParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= ulong(p.nz - 1) * ulong(p.ny) * ulong(p.nx)) return;
    wrf_exact_w_damp_source::w_damp(reinterpret_cast<device Exact32*>(r.rw_t), reinterpret_cast<device const Exact32*>(r.ww), reinterpret_cast<device const Exact32*>(r.w), reinterpret_cast<device const Exact32*>(r.mup), reinterpret_cast<device const Exact32*>(r.mub2d), reinterpret_cast<device const Exact32*>(r.c1f), reinterpret_cast<device const Exact32*>(r.c2f), reinterpret_cast<device const Exact32*>(r.rdnw), Exact32(p.dt), Exact32(p.w_damp_on), Exact32(p.w_crit_cfl), p.nz, p.ny, p.nx, int(gid));
}
