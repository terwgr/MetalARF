// Generated from WOOF CUDA upper-wind source; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "exact_gfk_math.metalh"
#include "wrf_exact_upper_wind_limiter_bindings.metalh"
using namespace metal;
namespace wrf_exact_upper_wind_source {
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_greater_equal(b.bits,a.bits); }
#define G Exact32(9.81f)
void upper_wind_limiter(device Exact32* u, device Exact32* v, device const Exact32* php,
                        device const Exact32* phb, Exact32 dts, Exact32 hdepth,
                        int base3d, int zone, int nz, int ny, int nx, int col)
{

    if (col >= ny * nx) return;
    const int j = col / nx, i = col % nx;
    if (i < zone || i >= nx-zone || j < zone || j >= ny-zone) return;
    Exact32 qq = 0.0f;
    for (int k = nz-1; k >= nz-3; --k) {
        const Exact32 uk = u[(k*ny+j)*(nx+1)+i];
        const Exact32 vk = v[(k*(ny+1)+j)*nx+i];
        qq = exact_max(qq, exact_add(exact_mul(uk, uk), exact_mul(vk, vk)));
    }
    qq = exact_div(qq, 12100.0f);
    if (qq <= 1.0f) return;
    const int top = nz*ny*nx+col;
    const Exact32 htop = exact_div(exact_add(php[top], phb[base3d ? top : nz]), 9.81f);
    const Exact32 hbot = exact_sub(htop, hdepth);
    for (int k = nz-1; k >= 1; --k) {
        const int pos = k*ny*nx+col;
        const Exact32 hk = exact_div(exact_add(php[pos], phb[base3d ? pos : k]), 9.81f);
        if (hk >= hbot && qq > 1.0f) {
            qq = exact_sqrt(exact_mul(exact_sub(qq, 1.0f), qq));
            Exact32 drag = exact_mul(exact_div(dts, 86400.0f), 2.0f);
            drag = exact_mul(drag, qq);
            drag = exact_mul(drag, exact_gfk_pow(2.0f, exact_mul(0.4f, Exact32(float(k-(nz-1))))));
            const Exact32 factor = exact_sub(1.0f, drag);
            const int up = (k*ny+j)*(nx+1)+i;
            const int vp = (k*(ny+1)+j)*nx+i;
            u[up] = exact_mul(u[up], factor);
            v[vp] = exact_mul(v[vp], factor);
        }
    }
}
}
kernel void wrf_exact_upper_wind_limiter(constant WrfExactUpperWindLimiterResources &r [[buffer(0)]], constant WrfExactUpperWindLimiterParams &p [[buffer(1)]], uint col [[thread_position_in_grid]]) {
  if (col >= uint(p.ny)*uint(p.nx)) return;
  wrf_exact_upper_wind_source::upper_wind_limiter(reinterpret_cast<device Exact32*>(r.u), reinterpret_cast<device Exact32*>(r.v), reinterpret_cast<device const Exact32*>(r.php), reinterpret_cast<device const Exact32*>(r.phb), Exact32(p.dts), Exact32(p.hdepth), p.base3d, p.zone, p.nz, p.ny, p.nx, int(col));
}
