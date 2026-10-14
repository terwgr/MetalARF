#include <metal_stdlib>
#include "coriolis_curvature_bindings.metalh"

using namespace metal;

static inline float rw_at(constant CoriolisCurvatureResources &r, int kf,
                          ulong c, ulong st, float msft) {
    return ((r.c1f[kf] * r.mut[c] + r.c2f[kf])
            * r.w[ulong(kf) * st + c]) / msft;
}

static inline float vxgm_at(constant CoriolisCurvatureResources &r,
                            constant CoriolisCurvatureParams &q,
                            int k, int j, int i) {
    const int nxf = q.nx + 1;
    const int nyf = q.ny + 1;
    const ulong u0 = (ulong(k) * ulong(q.ny) + ulong(j)) * ulong(nxf) + ulong(i);
    const ulong v0 = (ulong(k) * ulong(nyf) + ulong(j)) * ulong(q.nx) + ulong(i);
    const ulong c = ulong(j) * ulong(q.nx) + ulong(i);
    const float ub = 0.5f * (r.u[u0] + r.u[u0 + 1ul]);
    const float vb = 0.5f * (r.v[v0] + r.v[v0 + ulong(q.nx)]);
    return ub * (r.msfv[c + ulong(q.nx)] - r.msfv[c]) * q.rdy
         - vb * (r.msfu[ulong(j) * ulong(nxf) + ulong(i + 1)]
                 - r.msfu[ulong(j) * ulong(nxf) + ulong(i)]) * q.rdx;
}

kernel void coriolis_curvature(
    constant CoriolisCurvatureResources &r [[buffer(0)]],
    constant CoriolisCurvatureParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nyf = q.ny + 1;
    const int nxf = q.nx + 1;
    const uint plane2 = uint(nyf * nxf);
    if (tid >= uint(q.nz) * plane2) return;
    const int k = int(tid / plane2);
    const int rem = int(tid - uint(k) * plane2);
    const int j = rem / nxf;
    const int i = rem - j * nxf;
    const ulong st = ulong(q.ny) * ulong(q.nx);

    if (j < q.ny && (!q.boundary_x || (i > 0 && i < q.nx))) {
        const int ic = i == q.nx ? 0 : i;
        const int imc = i == 0 ? q.nx - 1 : i - 1;
        const ulong cA = ulong(j) * ulong(q.nx) + ulong(ic);
        const ulong cB = ulong(j) * ulong(q.nx) + ulong(imc);
        const ulong rvA = (ulong(k) * ulong(nyf) + ulong(j)) * ulong(q.nx);
        const ulong rvB = rvA + ulong(q.nx);
        // Coriolis and curvature add to coupled momentum tendencies.
        const float rv4 = 0.25f * (r.rv[rvA + ulong(imc)] + r.rv[rvA + ulong(ic)]
                                 + r.rv[rvB + ulong(imc)] + r.rv[rvB + ulong(ic)]);
        const float rw4 = 0.25f * (rw_at(r, k, cB, st, r.msft[cB])
                                 + rw_at(r, k + 1, cB, st, r.msft[cB])
                                 + rw_at(r, k, cA, st, r.msft[cA])
                                 + rw_at(r, k + 1, cA, st, r.msft[cA]));
        float t = 0.5f * (r.f[cA] + r.f[cB]) * rv4
                - 0.5f * (r.e[cA] + r.e[cB])
                  * (0.5f * (r.cosa[cA] + r.cosa[cB])) * rw4;
        const float vx = 0.5f * (vxgm_at(r, q, k, j, ic)
                               + vxgm_at(r, q, k, j, imc));
        const ulong out = (ulong(k) * ulong(q.ny) + ulong(j)) * ulong(nxf) + ulong(i);
        t += vx * rv4 - r.u[out] * q.reradius * rw4;
        r.ru_t[out] += t;
    }

    if (i < q.nx && (!q.boundary_y || (j > 0 && j < q.ny))) {
        const int jc = j == q.ny ? 0 : j;
        const int jmc = j == 0 ? q.ny - 1 : j - 1;
        const ulong cA = ulong(jc) * ulong(q.nx) + ulong(i);
        const ulong cB = ulong(jmc) * ulong(q.nx) + ulong(i);
        const ulong ruA = (ulong(k) * ulong(q.ny) + ulong(jc)) * ulong(nxf);
        const ulong ruB = (ulong(k) * ulong(q.ny) + ulong(jmc)) * ulong(nxf);
        const float ru4 = 0.25f * (r.ru[ruA + ulong(i)] + r.ru[ruA + ulong(i + 1)]
                                 + r.ru[ruB + ulong(i)] + r.ru[ruB + ulong(i + 1)]);
        const float rw4 = 0.25f * (rw_at(r, k, cB, st, r.msft[cB])
                                 + rw_at(r, k + 1, cB, st, r.msft[cB])
                                 + rw_at(r, k, cA, st, r.msft[cA])
                                 + rw_at(r, k + 1, cA, st, r.msft[cA]));
        float t = -0.5f * (r.f[cA] + r.f[cB]) * ru4;
        const float vx = 0.5f * (vxgm_at(r, q, k, jc, i)
                               + vxgm_at(r, q, k, jmc, i));
        const ulong out = (ulong(k) * ulong(nyf) + ulong(j)) * ulong(q.nx) + ulong(i);
        t += -vx * ru4 - r.v[out] * q.reradius * rw4;
        t += 0.5f * (r.e[cA] + r.e[cB])
             * (0.5f * (r.sina[cA] + r.sina[cB])) * rw4;
        r.rv_t[out] += t;
    }

    if (i < q.nx && j < q.ny && k >= 1) {
        const ulong c = ulong(j) * ulong(q.nx) + ulong(i);
        const ulong ru0 = (ulong(k) * ulong(q.ny) + ulong(j)) * ulong(nxf) + ulong(i);
        const ulong ru1 = ru0 + 1ul;
        const ulong ru_prev = ru0 - ulong(q.ny) * ulong(nxf);
        const ulong ru_prev1 = ru_prev + 1ul;
        const ulong rv0 = (ulong(k) * ulong(nyf) + ulong(j)) * ulong(q.nx) + ulong(i);
        const ulong rv1 = rv0 + ulong(q.nx);
        const ulong rv_prev = rv0 - ulong(nyf) * ulong(q.nx);
        const ulong rv_prev1 = rv_prev + ulong(q.nx);
        const float ruf = 0.5f * (r.fnm[k] * (r.ru[ru0] + r.ru[ru1])
                                 + r.fnp[k] * (r.ru[ru_prev] + r.ru[ru_prev1]));
        const float uf = 0.5f * (r.fnm[k] * (r.u[ru0] + r.u[ru1])
                                + r.fnp[k] * (r.u[ru_prev] + r.u[ru_prev1]));
        const float rvf = 0.5f * (r.fnm[k] * (r.rv[rv0] + r.rv[rv1])
                                 + r.fnp[k] * (r.rv[rv_prev] + r.rv[rv_prev1]));
        const float vf = 0.5f * (r.fnm[k] * (r.v[rv0] + r.v[rv1])
                                + r.fnp[k] * (r.v[rv_prev] + r.v[rv_prev1]));
        const ulong out = ulong(k) * st + c;
        r.rw_t[out] += r.e[c] * (r.cosa[c] * ruf - r.sina[c] * rvf)
                     + q.reradius * (ruf * uf + rvf * vf);
    }
}
