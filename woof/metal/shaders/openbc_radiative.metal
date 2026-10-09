#include <metal_stdlib>
#include "open_u_radiative_bindings.metalh"
#include "open_v_radiative_bindings.metalh"
#include "open_zero_gradient_x_bindings.metalh"
#include "open_zero_gradient_y_bindings.metalh"
#include "open_zero_gradient_scalar_x_bindings.metalh"
#include "open_zero_gradient_scalar_y_bindings.metalh"
#include "w_damp_bindings.metalh"

using namespace metal;

kernel void open_u_radiative(constant OpenURadiativeResources &r [[buffer(0)]],
                             constant OpenURadiativeParams &q [[buffer(1)]],
                             uint t [[thread_position_in_grid]]) {
    if (t >= uint(q.nz * q.ny)) return;
    const int k = int(t) / q.ny;
    const int j = int(t) - k * q.ny;
    const int nxf = q.nx + 1;
    const ulong cw = ulong(j) * ulong(q.nx);
    const float mw = r.c1h[k] * (r.mub2d[cw] + r.mup[cw]) + r.c2h[k];
    const ulong f0 = (ulong(k) * ulong(q.ny) + ulong(j)) * ulong(nxf);
    const ulong f1 = f0 + 1ul;
    float ub = min(mw * r.u[f0] - q.cb * mw, 0.0f);
    if (q.has_msf)
        ub = min((mw * r.u[f0]) / r.msfu[ulong(j) * ulong(nxf)]
                 - q.cb * mw, 0.0f);
    r.ru_t[f0] += -q.rdx * ub * (r.u[f1] - r.u[f0]);

    const ulong ce = cw + ulong(q.nx - 1);
    const ulong fe = f0 + ulong(q.nx);
    const ulong fi = fe - 1ul;
    const float me = r.c1h[k] * (r.mub2d[ce] + r.mup[ce]) + r.c2h[k];
    ub = max(me * r.u[fe] + q.cb * me, 0.0f);
    if (q.has_msf)
        ub = max((me * r.u[fe]) / r.msfu[ulong(j) * ulong(nxf) + ulong(q.nx)]
                 + q.cb * me, 0.0f);
    r.ru_t[fe] += -q.rdx * ub * (r.u[fe] - r.u[fi]);
}

kernel void open_v_radiative(constant OpenVRadiativeResources &r [[buffer(0)]],
                             constant OpenVRadiativeParams &q [[buffer(1)]],
                             uint t [[thread_position_in_grid]]) {
    if (t >= uint(q.nz * q.nx)) return;
    const int k = int(t) / q.nx;
    const int i = int(t) - k * q.nx;
    const int nyf = q.ny + 1;
    const ulong cs = ulong(i);
    const float ms = r.c1h[k] * (r.mub2d[cs] + r.mup[cs]) + r.c2h[k];
    const ulong f0 = ulong(k) * ulong(nyf) * ulong(q.nx) + ulong(i);
    const ulong f1 = f0 + ulong(q.nx);
    float vb = min(ms * r.v[f0] - q.cb * ms, 0.0f);
    if (q.has_msf)
        vb = min((ms * r.v[f0]) / r.msfv[ulong(i)] - q.cb * ms, 0.0f);
    r.rv_t[f0] += -q.rdy * vb * (r.v[f1] - r.v[f0]);

    const ulong cn = ulong(q.ny - 1) * ulong(q.nx) + ulong(i);
    const float mn = r.c1h[k] * (r.mub2d[cn] + r.mup[cn]) + r.c2h[k];
    const ulong fn = f0 + ulong(q.ny) * ulong(q.nx);
    const ulong fi = fn - ulong(q.nx);
    vb = max(mn * r.v[fn] + q.cb * mn, 0.0f);
    if (q.has_msf)
        vb = max((mn * r.v[fn]) / r.msfv[ulong(q.ny) * ulong(q.nx) + ulong(i)]
                 + q.cb * mn, 0.0f);
    r.rv_t[fn] += -q.rdy * vb * (r.v[fn] - r.v[fi]);
}

// Dry portion of WRF's outbound zero-gradient stage boundary update.  The
// caller submits x then y on the serial context, matching the CUDA ordering.
kernel void open_zero_gradient_x(constant OpenZeroGradientXResources &r [[buffer(0)]],
                                 constant OpenZeroGradientXParams &q [[buffer(1)]],
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
    const bool outw = r.u[u0] < 0.0f;
    const bool oute = r.u[ue] > 0.0f;
    if (j < q.ny) {
        const ulong th = (ulong(k) * ulong(q.ny) + ulong(j)) * ulong(q.nx);
        if (outw) r.thp[th] = r.thp[th + 1ul];
        if (oute) r.thp[th + ulong(q.nx - 1)] = r.thp[th + ulong(q.nx - 2)];
        if (k > 0 && k < q.nz) {
            const ulong w = ulong(k) * plane + ulong(j) * ulong(q.nx);
            if (r.u[u0] + r.u[u0 - ulong(nxf)] < 0.0f)
                r.w[w] = r.w[w + 1ul];
            if (r.u[ue] + r.u[ue - ulong(nxf)] > 0.0f)
                r.w[w + ulong(q.nx - 1)] = r.w[w + ulong(q.nx - 2)];
        }
    }
    const ulong v = (ulong(k) * ulong(nyf) + ulong(j)) * ulong(q.nx);
    if (outw) r.v[v] = r.v[v + 1ul];
    if (oute) r.v[v + ulong(q.nx - 1)] = r.v[v + ulong(q.nx - 2)];
}

kernel void open_zero_gradient_y(constant OpenZeroGradientYResources &r [[buffer(0)]],
                                 constant OpenZeroGradientYParams &q [[buffer(1)]],
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
    const bool outs = r.v[v0] < 0.0f;
    const bool outn = r.v[vn] > 0.0f;
    if (i < q.nx) {
        const ulong ths = ulong(k) * plane + ulong(i);
        const ulong thn = ulong(k) * plane + ulong(q.ny - 1) * ulong(q.nx) + ulong(i);
        if (outs) r.thp[ths] = r.thp[ths + ulong(q.nx)];
        if (outn) r.thp[thn] = r.thp[thn - ulong(q.nx)];
        if (k > 0 && k < q.nz) {
            const ulong w = ulong(k) * plane + ulong(i);
            if (r.v[v0] + r.v[v0 - ulong(nyf) * ulong(q.nx)] < 0.0f)
                r.w[w] = r.w[w + ulong(q.nx)];
            if (r.v[vn] + r.v[vn - ulong(nyf) * ulong(q.nx)] > 0.0f)
                r.w[w + ulong(q.ny - 1) * ulong(q.nx)]
                    = r.w[w + ulong(q.ny - 2) * ulong(q.nx)];
        }
    }
    const ulong u = (ulong(k) * ulong(q.ny)) * ulong(nxf) + ulong(i);
    const ulong un = u + ulong(q.ny - 1) * ulong(nxf);
    if (outs) r.u[u] = r.u[u + ulong(nxf)];
    if (outn) r.u[un] = r.u[un - ulong(nxf)];
}

// Outbound zero-gradient for the mandatory moist scalars. The caller submits
// the x pass before the y pass, matching the CPU field update order.
kernel void open_zero_gradient_scalar_x(
    constant OpenZeroGradientScalarXResources &r [[buffer(0)]],
    constant OpenZeroGradientScalarXParams &q [[buffer(1)]],
    uint t [[thread_position_in_grid]]) {
    if (t >= uint(q.nz * q.ny)) return;
    const int k = int(t) / q.ny;
    const int j = int(t) - k * q.ny;
    const int nxf = q.nx + 1;
    const ulong u0 = (ulong(k) * ulong(q.ny) + ulong(j)) * ulong(nxf);
    const ulong ue = u0 + ulong(q.nx);
    const bool outw = r.u[u0] < 0.0f;
    const bool oute = r.u[ue] > 0.0f;
    const ulong cell = (ulong(k) * ulong(q.ny) + ulong(j)) * ulong(q.nx);
    if (outw) {
        r.qv[cell] = r.qv[cell + 1ul];
        r.qc[cell] = r.qc[cell + 1ul];
        r.qr[cell] = r.qr[cell + 1ul];
    }
    if (oute) {
        const ulong edge = cell + ulong(q.nx - 1);
        const ulong inner = edge - 1ul;
        r.qv[edge] = r.qv[inner];
        r.qc[edge] = r.qc[inner];
        r.qr[edge] = r.qr[inner];
    }
}

kernel void open_zero_gradient_scalar_y(
    constant OpenZeroGradientScalarYResources &r [[buffer(0)]],
    constant OpenZeroGradientScalarYParams &q [[buffer(1)]],
    uint t [[thread_position_in_grid]]) {
    if (t >= uint(q.nz * q.nx)) return;
    const int k = int(t) / q.nx;
    const int i = int(t) - k * q.nx;
    const int nyf = q.ny + 1;
    const ulong vs = ulong(k) * ulong(nyf) * ulong(q.nx) + ulong(i);
    const ulong vn = vs + ulong(q.ny) * ulong(q.nx);
    const bool outs = r.v[vs] < 0.0f;
    const bool outn = r.v[vn] > 0.0f;
    const ulong south = ulong(k) * ulong(q.ny) * ulong(q.nx) + ulong(i);
    const ulong north = south + ulong(q.ny - 1) * ulong(q.nx);
    if (outs) {
        const ulong inner = south + ulong(q.nx);
        r.qv[south] = r.qv[inner];
        r.qc[south] = r.qc[inner];
        r.qr[south] = r.qr[inner];
    }
    if (outn) {
        const ulong inner = north - ulong(q.nx);
        r.qv[north] = r.qv[inner];
        r.qc[north] = r.qc[inner];
        r.qr[north] = r.qr[inner];
    }
}

kernel void w_damp(constant WDampResources &r [[buffer(0)]],
                   constant WDampParams &q [[buffer(1)]],
                   uint t [[thread_position_in_grid]]) {
    const int plane = q.ny * q.nx;
    if (t >= uint((q.nz - 1) * plane)) return;
    const int k = int(t) / plane + 1;
    const int c = int(t) - (k - 1) * plane;
    const float m = r.c1f[k] * (r.mub2d[c] + r.mup[c]) + r.c2f[k];
    const ulong ix = ulong(k) * ulong(plane) + ulong(c);
    const float vert_cfl = fabs(((r.ww[ix] / m) * r.rdnw[k]) * q.dt);
    if (vert_cfl > q.w_damp_on) {
        float damp = copysign(1.0f, r.w[ix]) * 0.3f;
        damp = damp * (vert_cfl - q.w_crit_cfl);
        r.rw_t[ix] = r.rw_t[ix] - damp * m;
    }
}
