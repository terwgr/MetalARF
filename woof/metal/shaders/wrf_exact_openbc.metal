// Generated from WOOF open-boundary operations; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_open_u_radiative_bindings.metalh"
#include "wrf_exact_open_v_radiative_bindings.metalh"
#include "wrf_exact_open_zero_gradient_x_bindings.metalh"
#include "wrf_exact_open_zero_gradient_y_bindings.metalh"
#include "wrf_exact_open_zero_gradient_scalar_x_bindings.metalh"
#include "wrf_exact_open_zero_gradient_scalar_y_bindings.metalh"
using namespace metal;
static inline bool operator<(Exact32 a, Exact32 b) { return soft32_less(a.bits,b.bits); }
namespace wrf_exact_openbc_source {
void open_u_radiative(device Exact32*  ru_t,
                      device const Exact32*  u,
                      device const Exact32*  mup,
                      device const Exact32*  mub2d,
                      device const Exact32*  c1h,
                      device const Exact32*  c2h,
                      device const Exact32*  msfu,
                      Exact32 rdx, Exact32 cb,
                      int nz, int ny, int nx, int has_msf, int t)
{

    if (t >= nz * ny) return;
    int k = t / ny, j = t - k * ny;
    int nxf = nx + 1;
    ulong cw = (ulong)j * nx;
    Exact32 mw = c1h[k] * (mub2d[cw] + mup[cw]) + c2h[k];
    ulong f0 = ((ulong)(k) * (ny) * (nxf) + (ulong)(j) * (nxf) + (0));
    ulong f1 = ((ulong)(k) * (ny) * (nxf) + (ulong)(j) * (nxf) + (1));
    Exact32 ub = exact_min(mw * u[f0] - cb * mw, 0.0f);
    if (has_msf)
        ub = exact_min(exact_div(mw * u[f0], msfu[(ulong)j * nxf]) - cb * mw, 0.0f);
    ru_t[f0] += -rdx * ub * (u[f1] - u[f0]);
    ulong ce = (ulong)j * nx + (nx - 1);
    Exact32 me = c1h[k] * (mub2d[ce] + mup[ce]) + c2h[k];
    ulong fe = ((ulong)(k) * (ny) * (nxf) + (ulong)(j) * (nxf) + (nx));
    ulong fi = ((ulong)(k) * (ny) * (nxf) + (ulong)(j) * (nxf) + (nx - 1));
    ub = exact_max(me * u[fe] + cb * me, 0.0f);
    if (has_msf)
        ub = exact_max(exact_div(me * u[fe], msfu[(ulong)j * nxf + nx]) + cb * me, 0.0f);
    ru_t[fe] += -rdx * ub * (u[fe] - u[fi]);
}
void open_v_radiative(device Exact32*  rv_t,
                      device const Exact32*  v,
                      device const Exact32*  mup,
                      device const Exact32*  mub2d,
                      device const Exact32*  c1h,
                      device const Exact32*  c2h,
                      device const Exact32*  msfv,
                      Exact32 rdy, Exact32 cb,
                      int nz, int ny, int nx, int has_msf, int t)
{

    if (t >= nz * nx) return;
    int k = t / nx, i = t - k * nx;
    int nyf = ny + 1;
    ulong cs = (ulong)i;
    Exact32 ms = c1h[k] * (mub2d[cs] + mup[cs]) + c2h[k];
    ulong f0 = ((ulong)(k) * (nyf) * (nx) + (ulong)(0) * (nx) + (i));
    ulong f1 = ((ulong)(k) * (nyf) * (nx) + (ulong)(1) * (nx) + (i));
    Exact32 vb = exact_min(ms * v[f0] - cb * ms, 0.0f);
    if (has_msf)
        vb = exact_min(exact_div(ms * v[f0], msfv[i]) - cb * ms, 0.0f);
    rv_t[f0] += -rdy * vb * (v[f1] - v[f0]);
    ulong cn = (ulong)(ny - 1) * nx + i;
    Exact32 mn = c1h[k] * (mub2d[cn] + mup[cn]) + c2h[k];
    ulong fn = ((ulong)(k) * (nyf) * (nx) + (ulong)(ny) * (nx) + (i));
    ulong fi = ((ulong)(k) * (nyf) * (nx) + (ulong)(ny - 1) * (nx) + (i));
    vb = exact_max(mn * v[fn] + cb * mn, 0.0f);
    if (has_msf)
        vb = exact_max(exact_div(mn * v[fn], msfv[(ulong)ny * nx + i]) + cb * mn, 0.0f);
    rv_t[fn] += -rdy * vb * (v[fn] - v[fi]);
}
}
kernel void wrf_exact_open_u_radiative(constant WrfExactOpenURadiativeResources &r [[buffer(0)]], constant WrfExactOpenURadiativeParams &q [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  if (t>=uint(q.nz*q.ny)) return;
  wrf_exact_openbc_source::open_u_radiative(reinterpret_cast<device Exact32*>(r.ru_t), reinterpret_cast<device const Exact32*>(r.u), reinterpret_cast<device const Exact32*>(r.mup), reinterpret_cast<device const Exact32*>(r.mub2d), reinterpret_cast<device const Exact32*>(r.c1h), reinterpret_cast<device const Exact32*>(r.c2h), reinterpret_cast<device const Exact32*>(r.msfu), Exact32(q.rdx), Exact32(q.cb), q.nz, q.ny, q.nx, q.has_msf, int(t));
}
kernel void wrf_exact_open_v_radiative(constant WrfExactOpenVRadiativeResources &r [[buffer(0)]], constant WrfExactOpenVRadiativeParams &q [[buffer(1)]], uint t [[thread_position_in_grid]]) {
  if (t>=uint(q.nz*q.nx)) return;
  wrf_exact_openbc_source::open_v_radiative(reinterpret_cast<device Exact32*>(r.rv_t), reinterpret_cast<device const Exact32*>(r.v), reinterpret_cast<device const Exact32*>(r.mup), reinterpret_cast<device const Exact32*>(r.mub2d), reinterpret_cast<device const Exact32*>(r.c1h), reinterpret_cast<device const Exact32*>(r.c2h), reinterpret_cast<device const Exact32*>(r.msfv), Exact32(q.rdy), Exact32(q.cb), q.nz, q.ny, q.nx, q.has_msf, int(t));
}
kernel void wrf_exact_open_zero_gradient_x(constant WrfExactOpenZeroGradientXResources &r [[buffer(0)]],
                                 constant WrfExactOpenZeroGradientXParams &q [[buffer(1)]],
                                 uint t [[thread_position_in_grid]]) {
    const int nyf = q.ny + 1;
    if (t >= uint(q.nz * nyf)) return;
    const int k = int(t) / nyf;
    const int j = int(t) - k * nyf;
    const int nxf = q.nx + 1;
    const ulong plane = ulong(q.ny) * ulong(q.nx);
    const int mask_j = j < q.ny ? j : (q.boundary_y ? q.ny - 1 : 0);
    const ulong u0 = (ulong(k) * ulong(q.ny) + ulong(mask_j)) * ulong(nxf);
    const ulong ue = u0 + ulong(q.nx);
    const bool outw = reinterpret_cast<device const Exact32*>(r.u)[u0] < 0.0f;
    const bool oute = reinterpret_cast<device const Exact32*>(r.u)[ue] > 0.0f;
    if (j < q.ny) {
        const ulong th = (ulong(k) * ulong(q.ny) + ulong(j)) * ulong(q.nx);
        if (outw) reinterpret_cast<device Exact32*>(r.thp)[th] = reinterpret_cast<device Exact32*>(r.thp)[th + 1ul];
        if (oute) reinterpret_cast<device Exact32*>(r.thp)[th + ulong(q.nx - 1)] = reinterpret_cast<device Exact32*>(r.thp)[th + ulong(q.nx - 2)];
        if (k > 0 && k < q.nz) {
            const ulong w = ulong(k) * plane + ulong(j) * ulong(q.nx);
            if (reinterpret_cast<device const Exact32*>(r.u)[u0] + reinterpret_cast<device const Exact32*>(r.u)[u0 - ulong(q.ny) * ulong(nxf)] < 0.0f)
                reinterpret_cast<device Exact32*>(r.w)[w] = reinterpret_cast<device Exact32*>(r.w)[w + 1ul];
            if (reinterpret_cast<device const Exact32*>(r.u)[ue] + reinterpret_cast<device const Exact32*>(r.u)[ue - ulong(q.ny) * ulong(nxf)] > 0.0f)
                reinterpret_cast<device Exact32*>(r.w)[w + ulong(q.nx - 1)] = reinterpret_cast<device Exact32*>(r.w)[w + ulong(q.nx - 2)];
        }
    }
    const ulong v = (ulong(k) * ulong(nyf) + ulong(j)) * ulong(q.nx);
    if (outw) reinterpret_cast<device Exact32*>(r.v)[v] = reinterpret_cast<device Exact32*>(r.v)[v + 1ul];
    if (oute) reinterpret_cast<device Exact32*>(r.v)[v + ulong(q.nx - 1)] = reinterpret_cast<device Exact32*>(r.v)[v + ulong(q.nx - 2)];
}
kernel void wrf_exact_open_zero_gradient_y(constant WrfExactOpenZeroGradientYResources &r [[buffer(0)]],
                                 constant WrfExactOpenZeroGradientYParams &q [[buffer(1)]],
                                 uint t [[thread_position_in_grid]]) {
    const int nxf = q.nx + 1;
    if (t >= uint(q.nz * nxf)) return;
    const int k = int(t) / nxf;
    const int i = int(t) - k * nxf;
    const int nyf = q.ny + 1;
    const ulong plane = ulong(q.ny) * ulong(q.nx);
    const int mask_i = i < q.nx ? i : (q.boundary_x ? q.nx - 1 : 0);
    const ulong v0 = ulong(k) * ulong(nyf) * ulong(q.nx) + ulong(mask_i);
    const ulong vn = v0 + ulong(q.ny) * ulong(q.nx);
    const bool outs = reinterpret_cast<device const Exact32*>(r.v)[v0] < 0.0f;
    const bool outn = reinterpret_cast<device const Exact32*>(r.v)[vn] > 0.0f;
    if (i < q.nx) {
        const ulong ths = ulong(k) * plane + ulong(i);
        const ulong thn = ulong(k) * plane + ulong(q.ny - 1) * ulong(q.nx) + ulong(i);
        if (outs) reinterpret_cast<device Exact32*>(r.thp)[ths] = reinterpret_cast<device Exact32*>(r.thp)[ths + ulong(q.nx)];
        if (outn) reinterpret_cast<device Exact32*>(r.thp)[thn] = reinterpret_cast<device Exact32*>(r.thp)[thn - ulong(q.nx)];
        if (k > 0 && k < q.nz) {
            const ulong w = ulong(k) * plane + ulong(i);
            if (reinterpret_cast<device const Exact32*>(r.v)[v0] + reinterpret_cast<device const Exact32*>(r.v)[v0 - ulong(nyf) * ulong(q.nx)] < 0.0f)
                reinterpret_cast<device Exact32*>(r.w)[w] = reinterpret_cast<device Exact32*>(r.w)[w + ulong(q.nx)];
            if (reinterpret_cast<device const Exact32*>(r.v)[vn] + reinterpret_cast<device const Exact32*>(r.v)[vn - ulong(nyf) * ulong(q.nx)] > 0.0f)
                reinterpret_cast<device Exact32*>(r.w)[w + ulong(q.ny - 1) * ulong(q.nx)]
                    = reinterpret_cast<device Exact32*>(r.w)[w + ulong(q.ny - 2) * ulong(q.nx)];
        }
    }
    const ulong u = (ulong(k) * ulong(q.ny)) * ulong(nxf) + ulong(i);
    const ulong un = u + ulong(q.ny - 1) * ulong(nxf);
    if (outs) reinterpret_cast<device Exact32*>(r.u)[u] = reinterpret_cast<device Exact32*>(r.u)[u + ulong(nxf)];
    if (outn) reinterpret_cast<device Exact32*>(r.u)[un] = reinterpret_cast<device Exact32*>(r.u)[un - ulong(nxf)];
}
kernel void wrf_exact_open_zero_gradient_scalar_x(
    constant WrfExactOpenZeroGradientScalarXResources &r [[buffer(0)]],
    constant WrfExactOpenZeroGradientScalarXParams &q [[buffer(1)]],
    uint t [[thread_position_in_grid]]) {
    if (t >= uint(q.nz * q.ny)) return;
    const int k = int(t) / q.ny;
    const int j = int(t) - k * q.ny;
    const int nxf = q.nx + 1;
    const ulong u0 = (ulong(k) * ulong(q.ny) + ulong(j)) * ulong(nxf);
    const ulong ue = u0 + ulong(q.nx);
    const bool outw = reinterpret_cast<device const Exact32*>(r.u)[u0] < 0.0f;
    const bool oute = reinterpret_cast<device const Exact32*>(r.u)[ue] > 0.0f;
    const ulong cell = (ulong(k) * ulong(q.ny) + ulong(j)) * ulong(q.nx);
    if (outw) {
        reinterpret_cast<device Exact32*>(r.qv)[cell] = reinterpret_cast<device Exact32*>(r.qv)[cell + 1ul];
        reinterpret_cast<device Exact32*>(r.qc)[cell] = reinterpret_cast<device Exact32*>(r.qc)[cell + 1ul];
        reinterpret_cast<device Exact32*>(r.qr)[cell] = reinterpret_cast<device Exact32*>(r.qr)[cell + 1ul];
    }
    if (oute) {
        const ulong edge = cell + ulong(q.nx - 1);
        const ulong inner = edge - 1ul;
        reinterpret_cast<device Exact32*>(r.qv)[edge] = reinterpret_cast<device Exact32*>(r.qv)[inner];
        reinterpret_cast<device Exact32*>(r.qc)[edge] = reinterpret_cast<device Exact32*>(r.qc)[inner];
        reinterpret_cast<device Exact32*>(r.qr)[edge] = reinterpret_cast<device Exact32*>(r.qr)[inner];
    }
}
kernel void wrf_exact_open_zero_gradient_scalar_y(
    constant WrfExactOpenZeroGradientScalarYResources &r [[buffer(0)]],
    constant WrfExactOpenZeroGradientScalarYParams &q [[buffer(1)]],
    uint t [[thread_position_in_grid]]) {
    if (t >= uint(q.nz * q.nx)) return;
    const int k = int(t) / q.nx;
    const int i = int(t) - k * q.nx;
    const int nyf = q.ny + 1;
    const ulong vs = ulong(k) * ulong(nyf) * ulong(q.nx) + ulong(i);
    const ulong vn = vs + ulong(q.ny) * ulong(q.nx);
    const bool outs = reinterpret_cast<device const Exact32*>(r.v)[vs] < 0.0f;
    const bool outn = reinterpret_cast<device const Exact32*>(r.v)[vn] > 0.0f;
    const ulong south = ulong(k) * ulong(q.ny) * ulong(q.nx) + ulong(i);
    const ulong north = south + ulong(q.ny - 1) * ulong(q.nx);
    if (outs) {
        const ulong inner = south + ulong(q.nx);
        reinterpret_cast<device Exact32*>(r.qv)[south] = reinterpret_cast<device Exact32*>(r.qv)[inner];
        reinterpret_cast<device Exact32*>(r.qc)[south] = reinterpret_cast<device Exact32*>(r.qc)[inner];
        reinterpret_cast<device Exact32*>(r.qr)[south] = reinterpret_cast<device Exact32*>(r.qr)[inner];
    }
    if (outn) {
        const ulong inner = north - ulong(q.nx);
        reinterpret_cast<device Exact32*>(r.qv)[north] = reinterpret_cast<device Exact32*>(r.qv)[inner];
        reinterpret_cast<device Exact32*>(r.qc)[north] = reinterpret_cast<device Exact32*>(r.qc)[inner];
        reinterpret_cast<device Exact32*>(r.qr)[north] = reinterpret_cast<device Exact32*>(r.qr)[inner];
    }
}
