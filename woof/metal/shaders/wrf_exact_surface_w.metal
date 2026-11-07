// Generated from woof/core/kernels/surface_w.cu by tools/surface_w_wrf_exact/generate.py; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_surface_w_bindings.metalh"
using namespace metal;
namespace wrf_exact_surface_w {
void set_surface_w(
    device const Exact32* u, device const Exact32* v, device const Exact32* ht, device const Exact32* msft,
    device Exact32* w, Exact32 cf1, Exact32 cf2, Exact32 cf3, Exact32 half_rdx,
    Exact32 half_rdy, int mapped, int bx, int by, int ny, int nx, int col)
{
    
    if (col >= ny * nx) return;
    int j = col / nx;
    int i = col - j * nx;
    int uplane = ny * (nx + 1);
    int vplane = (ny + 1) * nx;
    int ul = j * (nx + 1) + i;
    int vb = j * nx + i;
    Exact32 uc0 = exact_add(exact_add(exact_mul(cf1, u[ul]),
                                  exact_mul(cf2, u[uplane + ul])),
                         exact_mul(cf3, u[2 * uplane + ul]));
    Exact32 uc1 = exact_add(exact_add(exact_mul(cf1, u[ul + 1]),
                                  exact_mul(cf2, u[uplane + ul + 1])),
                         exact_mul(cf3, u[2 * uplane + ul + 1]));
    Exact32 vc0 = exact_add(exact_add(exact_mul(cf1, v[vb]),
                                  exact_mul(cf2, v[vplane + vb])),
                         exact_mul(cf3, v[2 * vplane + vb]));
    Exact32 vc1 = exact_add(exact_add(exact_mul(cf1, v[vb + nx]),
                                  exact_mul(cf2, v[vplane + vb + nx])),
                         exact_mul(cf3, v[2 * vplane + vb + nx]));
    Exact32 dyn = exact_sub(ht[((j + 1) % ny) * nx + i], ht[col]);
    Exact32 dys = exact_sub(ht[col], ht[((j + ny - 1) % ny) * nx + i]);
    Exact32 dxe = exact_sub(ht[j * nx + (i + 1) % nx], ht[col]);
    Exact32 dxw = exact_sub(ht[col], ht[j * nx + (i + nx - 1) % nx]);
    if (by && j == 0) dys = 0.0f;
    if (by && j == ny - 1) dyn = 0.0f;
    if (bx && i == 0) dxw = 0.0f;
    if (bx && i == nx - 1) dxe = 0.0f;
    Exact32 msf = mapped ? msft[col] : 1.0f;
    Exact32 cy = exact_mul(exact_mul(msf, 0.5f), exact_mul(2.0f, half_rdy));
    Exact32 cx = exact_mul(exact_mul(msf, 0.5f), exact_mul(2.0f, half_rdx));
    Exact32 y = exact_mul(cy, exact_add(exact_mul(dyn, vc1),
                                      exact_mul(dys, vc0)));
    Exact32 x = exact_mul(cx, exact_add(exact_mul(dxe, uc1),
                                      exact_mul(dxw, uc0)));
    w[col] = exact_add(y, x);
}
}
kernel void wrf_exact_set_surface_w(
    constant WrfExactSetSurfaceWResources &r [[buffer(0)]],
    constant WrfExactSetSurfaceWParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= ulong(p.ny) * ulong(p.nx)) return;
    wrf_exact_surface_w::set_surface_w(reinterpret_cast<device const Exact32*>(r.u), reinterpret_cast<device const Exact32*>(r.v), reinterpret_cast<device const Exact32*>(r.ht), reinterpret_cast<device const Exact32*>(r.msft), reinterpret_cast<device Exact32*>(r.w), Exact32(p.cf1), Exact32(p.cf2), Exact32(p.cf3), Exact32(p.half_rdx), Exact32(p.half_rdy), p.has_msf, p.boundary_x, p.boundary_y, p.ny, p.nx, int(gid));
}
