// Generated from woof/core/kernels/coriolis_map.cu by tools/coriolis_wrf_exact/generate.py; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_coriolis_off_bindings.metalh"
using namespace metal;
namespace wrf_exact_coriolis_off_source {
static
Exact32 rw_at(device const Exact32*  w, device const Exact32*  mut,
           device const Exact32*  msft,
           device const Exact32*  c1f, device const Exact32*  c2f,
           int kf, ulong c, ulong st)
{
    return (c1f[kf] * mut[c] + c2f[kf]) * w[(ulong)kf * st + c] / msft[c];
}
static
Exact32 vxgm_at(device const Exact32*  u, device const Exact32*  v,
             device const Exact32*  msfu, device const Exact32*  msfv,
             Exact32 rdx, Exact32 rdy, int k, int j, int i, int ny, int nx)
{
    int nxf = nx + 1;
    Exact32 ub = 0.5f * (u[((ulong)(k) * (ny) * (nxf) + (ulong)(j) * (nxf) + (i))] + u[((ulong)(k) * (ny) * (nxf) + (ulong)(j) * (nxf) + (i + 1))]);
    Exact32 vb = 0.5f * (v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))]
                      + v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j + 1) * (nx) + (i))]);
    return ub * (msfv[(ulong)(j + 1) * nx + i] - msfv[(ulong)j * nx + i])
               * rdy
         - vb * (msfu[(ulong)j * nxf + i + 1] - msfu[(ulong)j * nxf + i])
               * rdx;
}
void coriolis_curvature(device const Exact32*  ru,
                        device const Exact32*  rv,
                        device const Exact32*  u,
                        device const Exact32*  v,
                        device const Exact32*  w,
                        device const Exact32*  mut,
                        device const Exact32*  msft,
                        device const Exact32*  msfu,
                        device const Exact32*  msfv,
                        device const Exact32*  f,
                        device const Exact32*  e,
                        device const Exact32*  sina,
                        device const Exact32*  cosa,
                        device const Exact32*  c1f,
                        device const Exact32*  c2f,
                        device const Exact32*  fzm,
                        device const Exact32*  fzp,
                        Exact32 rdx, Exact32 rdy,
                        device Exact32*  ru_t,
                        device Exact32*  rv_t,
                        device Exact32*  rw_t,
                        int boundary_x, int boundary_y,
                        int nz, int ny, int nx, int tid)
{

    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    ulong st = (ulong)ny * nx;
    if (j < ny && (!boundary_x || (i > 0 && i < nx))) {
        int ic = i % nx, imc = (i - 1 + nx) % nx;
        ulong cA = (ulong)j * nx + ic, cB = (ulong)j * nx + imc;
        Exact32 rv4 = 0.25f * (rv[((ulong)(k) * (nyf) * (nx) + (ulong)(j) * (nx) + (imc))]
                          + rv[((ulong)(k) * (nyf) * (nx) + (ulong)(j) * (nx) + (ic))]
                          + rv[((ulong)(k) * (nyf) * (nx) + (ulong)(j + 1) * (nx) + (imc))]
                          + rv[((ulong)(k) * (nyf) * (nx) + (ulong)(j + 1) * (nx) + (ic))]);
        Exact32 rw4 = 0.25f * (rw_at(w, mut, msft, c1f, c2f, k, cB, st)
                          + rw_at(w, mut, msft, c1f, c2f, k + 1, cB, st)
                          + rw_at(w, mut, msft, c1f, c2f, k, cA, st)
                          + rw_at(w, mut, msft, c1f, c2f, k + 1, cA, st));
        Exact32 t = 0.5f * (f[cA] + f[cB]) * rv4
               - 0.5f * (e[cA] + e[cB])
                 * (0.5f * (cosa[cA] + cosa[cB])) * rw4;
        Exact32 vx = 0.5f * (vxgm_at(u, v, msfu, msfv, rdx, rdy,
                                  k, j, ic, ny, nx)
                        + vxgm_at(u, v, msfu, msfv, rdx, rdy,
                                  k, j, imc, ny, nx));
        t += vx * rv4
           - u[((ulong)(k) * (ny) * (nxf) + (ulong)(j) * (nxf) + (i))] * 1.5698587127158556e-07f * rw4;
        ru_t[((ulong)(k) * (ny) * (nxf) + (ulong)(j) * (nxf) + (i))] += t;
    }
    if (i < nx && (!boundary_y || (j > 0 && j < ny))) {
        int jc = j % ny, jmc = (j - 1 + ny) % ny;
        ulong cA = (ulong)jc * nx + i, cB = (ulong)jmc * nx + i;
        Exact32 ru4 = 0.25f * (ru[((ulong)(k) * (ny) * (nxf) + (ulong)(jc) * (nxf) + (i))]
                          + ru[((ulong)(k) * (ny) * (nxf) + (ulong)(jc) * (nxf) + (i + 1))]
                          + ru[((ulong)(k) * (ny) * (nxf) + (ulong)(jmc) * (nxf) + (i))]
                          + ru[((ulong)(k) * (ny) * (nxf) + (ulong)(jmc) * (nxf) + (i + 1))]);
        Exact32 rw4 = 0.25f * (rw_at(w, mut, msft, c1f, c2f, k, cB, st)
                          + rw_at(w, mut, msft, c1f, c2f, k + 1, cB, st)
                          + rw_at(w, mut, msft, c1f, c2f, k, cA, st)
                          + rw_at(w, mut, msft, c1f, c2f, k + 1, cA, st));
        Exact32 t = -0.5f * (f[cA] + f[cB]) * ru4;
        Exact32 vx = 0.5f * (vxgm_at(u, v, msfu, msfv, rdx, rdy,
                                  k, jc, i, ny, nx)
                        + vxgm_at(u, v, msfu, msfv, rdx, rdy,
                                  k, jmc, i, ny, nx));
        t += -vx * ru4
             - v[((ulong)(k) * (nyf) * (nx) + (ulong)(j) * (nx) + (i))] * 1.5698587127158556e-07f * rw4;
        t += 0.5f * (e[cA] + e[cB])
             * (0.5f * (sina[cA] + sina[cB])) * rw4;
        rv_t[((ulong)(k) * (nyf) * (nx) + (ulong)(j) * (nx) + (i))] += t;
    }
    if (i < nx && j < ny && k >= 1) {
        ulong c = (ulong)j * nx + i;
        Exact32 ruf = 0.5f * (fzm[k] * (ru[((ulong)(k) * (ny) * (nxf) + (ulong)(j) * (nxf) + (i))]
                                   + ru[((ulong)(k) * (ny) * (nxf) + (ulong)(j) * (nxf) + (i + 1))])
                         + fzp[k] * (ru[((ulong)(k - 1) * (ny) * (nxf) + (ulong)(j) * (nxf) + (i))]
                                   + ru[((ulong)(k - 1) * (ny) * (nxf) + (ulong)(j) * (nxf) + (i + 1))]));
        Exact32 uf = 0.5f * (fzm[k] * (u[((ulong)(k) * (ny) * (nxf) + (ulong)(j) * (nxf) + (i))]
                                  + u[((ulong)(k) * (ny) * (nxf) + (ulong)(j) * (nxf) + (i + 1))])
                        + fzp[k] * (u[((ulong)(k - 1) * (ny) * (nxf) + (ulong)(j) * (nxf) + (i))]
                                  + u[((ulong)(k - 1) * (ny) * (nxf) + (ulong)(j) * (nxf) + (i + 1))]));
        Exact32 rvf = 0.5f * (fzm[k] * (rv[((ulong)(k) * (nyf) * (nx) + (ulong)(j) * (nx) + (i))]
                                   + rv[((ulong)(k) * (nyf) * (nx) + (ulong)(j + 1) * (nx) + (i))])
                         + fzp[k] * (rv[((ulong)(k - 1) * (nyf) * (nx) + (ulong)(j) * (nx) + (i))]
                                   + rv[((ulong)(k - 1) * (nyf) * (nx) + (ulong)(j + 1) * (nx) + (i))]));
        Exact32 vf = 0.5f * (fzm[k] * (v[((ulong)(k) * (nyf) * (nx) + (ulong)(j) * (nx) + (i))]
                                  + v[((ulong)(k) * (nyf) * (nx) + (ulong)(j + 1) * (nx) + (i))])
                        + fzp[k] * (v[((ulong)(k - 1) * (nyf) * (nx) + (ulong)(j) * (nx) + (i))]
                                  + v[((ulong)(k - 1) * (nyf) * (nx) + (ulong)(j + 1) * (nx) + (i))]));
        rw_t[(ulong)k * st + c] += e[c] * (cosa[c] * ruf - sina[c] * rvf)
                                  + 1.5698587127158556e-07f * (ruf * uf + rvf * vf);
    }
}
}
kernel void wrf_exact_coriolis_off(constant WrfExactCoriolisOffResources &r [[buffer(0)]],
    constant WrfExactCoriolisOffParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= ulong(p.nz) * ulong(p.ny + 1) * ulong(p.nx + 1)) return;
    wrf_exact_coriolis_off_source::coriolis_curvature(reinterpret_cast<device const Exact32*>(r.ru), reinterpret_cast<device const Exact32*>(r.rv), reinterpret_cast<device const Exact32*>(r.u), reinterpret_cast<device const Exact32*>(r.v), reinterpret_cast<device const Exact32*>(r.w), reinterpret_cast<device const Exact32*>(r.mut), reinterpret_cast<device const Exact32*>(r.msft), reinterpret_cast<device const Exact32*>(r.msfu), reinterpret_cast<device const Exact32*>(r.msfv), reinterpret_cast<device const Exact32*>(r.f), reinterpret_cast<device const Exact32*>(r.e), reinterpret_cast<device const Exact32*>(r.sina), reinterpret_cast<device const Exact32*>(r.cosa), reinterpret_cast<device const Exact32*>(r.c1f), reinterpret_cast<device const Exact32*>(r.c2f), reinterpret_cast<device const Exact32*>(r.fnm), reinterpret_cast<device const Exact32*>(r.fnp), Exact32(p.rdx), Exact32(p.rdy), reinterpret_cast<device Exact32*>(r.ru_t), reinterpret_cast<device Exact32*>(r.rv_t), reinterpret_cast<device Exact32*>(r.rw_t), p.boundary_x, p.boundary_y, p.nz, p.ny, p.nx, int(gid));
}
