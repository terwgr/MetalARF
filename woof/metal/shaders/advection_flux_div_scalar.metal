#include <metal_stdlib>
#include "flux_div_scalar_bindings.metalh"

using namespace metal;

#ifndef GPUWM_WRF_EXACT
#define GPUWM_WRF_EXACT 0
#endif
#ifndef GPUWM_WRF_EXACT_C_ADVECTION
#define GPUWM_WRF_EXACT_C_ADVECTION GPUWM_WRF_EXACT
#endif

static inline ulong cell_index(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx) + ulong(i);
}
static inline ulong u_index(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx + 1) + ulong(i);
}
static inline ulong v_index(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny + 1) + ulong(j)) * ulong(nx) + ulong(i);
}
static inline ulong w_index(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx) + ulong(i);
}
static inline int periodic(int i, int n) {
    int r = i % n;
    return r < 0 ? r + n : r;
}

static inline float flux5(float qm3, float qm2, float qm1, float q0,
                          float qp1, float qp2, float vel) {
#if GPUWM_WRF_EXACT_C_ADVECTION
    float center = (37.0f * (q0 + qm1) - 8.0f * (qp1 + qm2)
                    + (qp2 + qm3)) / 60.0f;
    float dissipation = copysign(1.0f, vel)
        * ((qp2 - qm3) - 5.0f * (qp1 - qm2) + 10.0f * (q0 - qm1))
        / 60.0f;
    return vel * (center - dissipation);
#else
    return (vel * (37.0f * (q0 + qm1) - 8.0f * (qp1 + qm2)
                   + (qp2 + qm3))
            - abs(vel) * (10.0f * (q0 - qm1) - 5.0f * (qp1 - qm2)
                          + (qp2 - qm3))) / 60.0f;
#endif
}

static inline float flux3(float qm2, float qm1, float q0, float qp1,
                          float vel) {
#if GPUWM_WRF_EXACT_C_ADVECTION
    float center = (7.0f * (q0 + qm1) - (qp1 + qm2)) / 12.0f;
    float dissipation = copysign(1.0f, vel)
        * ((qp1 - qm2) - 3.0f * (q0 - qm1)) / 12.0f;
    return vel * (center - dissipation);
#else
    return (vel * (7.0f * (q0 + qm1) - (qp1 + qm2))
            + abs(vel) * (3.0f * (q0 - qm1) - (qp1 - qm2))) / 12.0f;
#endif
}

static inline float flux3h(float qm2, float qm1, float q0, float qp1,
                           float vel) {
#if GPUWM_WRF_EXACT_C_ADVECTION
    float center = (7.0f * (q0 + qm1) - (qp1 + qm2)) / 12.0f;
    float dissipation = copysign(1.0f, vel)
        * ((qp1 - qm2) - 3.0f * (q0 - qm1)) / 12.0f;
    return vel * (center + dissipation);
#else
    return (vel * (7.0f * (q0 + qm1) - (qp1 + qm2))
            - abs(vel) * (3.0f * (q0 - qm1) - (qp1 - qm2))) / 12.0f;
#endif
}

static inline float flux2(float qm1, float q0, float vel) {
    return 0.5f * vel * (q0 + qm1);
}

static inline float zface_half(device const float *q, float vel, int kf,
                               int j, int i, int nz, int ny, int nx,
                               device const float *fnm,
                               device const float *fnp) {
    if (kf == 0 || kf == nz) return 0.0f;
    if (kf == 1 || kf == nz - 1)
        return vel * (fnm[kf] * q[cell_index(kf, j, i, ny, nx)]
                    + fnp[kf] * q[cell_index(kf - 1, j, i, ny, nx)]);
    return flux3(q[cell_index(kf - 2, j, i, ny, nx)],
                 q[cell_index(kf - 1, j, i, ny, nx)],
                 q[cell_index(kf, j, i, ny, nx)],
                 q[cell_index(kf + 1, j, i, ny, nx)], vel);
}

static inline float xface_cell_open(constant FluxDivScalarResources &r,
                                    constant FluxDivScalarParams &p,
                                    float vel, int k, int j, int f) {
    if (f <= 0 || f >= p.nx) return 0.0f;
    const int d = min(f, p.nx - f);
    if (d == 1)
        return flux2(r.q[cell_index(k,j,f-1,p.ny,p.nx)],
                     r.q[cell_index(k,j,f,p.ny,p.nx)],vel);
    if (d == 2)
        return flux3h(r.q[cell_index(k,j,f-2,p.ny,p.nx)],
                      r.q[cell_index(k,j,f-1,p.ny,p.nx)],
                      r.q[cell_index(k,j,f,p.ny,p.nx)],
                      r.q[cell_index(k,j,f+1,p.ny,p.nx)],vel);
    return flux5(r.q[cell_index(k,j,f-3,p.ny,p.nx)],
                 r.q[cell_index(k,j,f-2,p.ny,p.nx)],
                 r.q[cell_index(k,j,f-1,p.ny,p.nx)],
                 r.q[cell_index(k,j,f,p.ny,p.nx)],
                 r.q[cell_index(k,j,f+1,p.ny,p.nx)],
                 r.q[cell_index(k,j,f+2,p.ny,p.nx)],vel);
}

static inline float xface_cell_periodic(constant FluxDivScalarResources &r,
                                        constant FluxDivScalarParams &p,
                                        float vel, int k, int j, int f) {
    return flux5(r.q[cell_index(k,j,periodic(f-3,p.nx),p.ny,p.nx)],
                 r.q[cell_index(k,j,periodic(f-2,p.nx),p.ny,p.nx)],
                 r.q[cell_index(k,j,periodic(f-1,p.nx),p.ny,p.nx)],
                 r.q[cell_index(k,j,periodic(f,p.nx),p.ny,p.nx)],
                 r.q[cell_index(k,j,periodic(f+1,p.nx),p.ny,p.nx)],
                 r.q[cell_index(k,j,periodic(f+2,p.nx),p.ny,p.nx)],vel);
}

static inline float yface_cell_open(constant FluxDivScalarResources &r,
                                    constant FluxDivScalarParams &p,
                                    float vel, int k, int g, int i) {
    if (g <= 0 || g >= p.ny) return 0.0f;
    const int d = min(g, p.ny - g);
    if (d == 1)
        return flux2(r.q[cell_index(k,g-1,i,p.ny,p.nx)],
                     r.q[cell_index(k,g,i,p.ny,p.nx)],vel);
    if (d == 2)
        return flux3h(r.q[cell_index(k,g-2,i,p.ny,p.nx)],
                      r.q[cell_index(k,g-1,i,p.ny,p.nx)],
                      r.q[cell_index(k,g,i,p.ny,p.nx)],
                      r.q[cell_index(k,g+1,i,p.ny,p.nx)],vel);
    return flux5(r.q[cell_index(k,g-3,i,p.ny,p.nx)],
                 r.q[cell_index(k,g-2,i,p.ny,p.nx)],
                 r.q[cell_index(k,g-1,i,p.ny,p.nx)],
                 r.q[cell_index(k,g,i,p.ny,p.nx)],
                 r.q[cell_index(k,g+1,i,p.ny,p.nx)],
                 r.q[cell_index(k,g+2,i,p.ny,p.nx)],vel);
}

static inline float yface_cell_periodic(constant FluxDivScalarResources &r,
                                        constant FluxDivScalarParams &p,
                                        float vel, int k, int g, int i) {
    return flux5(r.q[cell_index(k,periodic(g-3,p.ny),i,p.ny,p.nx)],
                 r.q[cell_index(k,periodic(g-2,p.ny),i,p.ny,p.nx)],
                 r.q[cell_index(k,periodic(g-1,p.ny),i,p.ny,p.nx)],
                 r.q[cell_index(k,periodic(g,p.ny),i,p.ny,p.nx)],
                 r.q[cell_index(k,periodic(g+1,p.ny),i,p.ny,p.nx)],
                 r.q[cell_index(k,periodic(g+2,p.ny),i,p.ny,p.nx)],vel);
}

kernel void flux_div_scalar(constant FluxDivScalarResources &r [[buffer(0)]],
                            constant FluxDivScalarParams &p [[buffer(1)]],
                            uint tid [[thread_position_in_grid]]) {
    const int n = p.nz * p.ny * p.nx;
    if (tid >= uint(n)) return;
    const int i = int(tid % uint(p.nx));
    const int j = int((tid / uint(p.nx)) % uint(p.ny));
    const int k = int(tid / uint(p.nx * p.ny));
    const ulong out = cell_index(k, j, i, p.ny, p.nx);

    const float mscale = p.has_msf
        ? r.msf[ulong(j)*ulong(p.nx)+ulong(i)] : 1.0f;
    if (p.open_x || p.open_y) {
        float tx = 0.0f, ty = 0.0f;
        float th = 0.0f, tb = 0.0f;
        bool applyx = false, applyy = false;
        if (p.open_x && (i == 0 || i == p.nx - 1)) {
            if (!p.spec) {
                float ru0 = r.ru[u_index(k,j,i == 0 ? 0 : p.nx-1,p.ny,p.nx)];
                float ru1 = r.ru[u_index(k,j,i == 0 ? 1 : p.nx,p.ny,p.nx)];
                float ub = i == 0 ? min(0.5f*(ru0+ru1),0.0f)
                                  : max(0.5f*(ru0+ru1),0.0f);
                int qi = i == 0 ? 0 : p.nx - 1;
                int qn = i == 0 ? 1 : p.nx - 2;
                tx = p.dx_inv * (ub * (r.q[cell_index(k,j,qi,p.ny,p.nx)]
                                       - r.q[cell_index(k,j,qn,p.ny,p.nx)])
                           + r.q[cell_index(k,j,qi,p.ny,p.nx)]*(ru1-ru0));
#if !GPUWM_WRF_EXACT_C_ADVECTION
                tb += -p.dx_inv * (ub * (r.q[cell_index(k,j,qi,p.ny,p.nx)]
                                         - r.q[cell_index(k,j,qn,p.ny,p.nx)])
                             + r.q[cell_index(k,j,qi,p.ny,p.nx)]*(ru1-ru0));
#endif
                applyx = true;
            }
        } else {
            float v0 = r.ru[u_index(k,j,i,p.ny,p.nx)];
            float v1 = r.ru[u_index(k,j,i+1,p.ny,p.nx)];
            float fx0 = p.open_x ? xface_cell_open(r,p,v0,k,j,i)
                                  : xface_cell_periodic(r,p,v0,k,j,i);
            float fx1 = p.open_x ? xface_cell_open(r,p,v1,k,j,i+1)
                                  : xface_cell_periodic(r,p,v1,k,j,i+1);
            tx = (p.has_msf ? mscale * p.dx_inv : p.dx_inv) * (fx1-fx0);
#if !GPUWM_WRF_EXACT_C_ADVECTION
            if (p.has_msf) th += -(fx1-fx0) * p.dx_inv;
            else tb += -(fx1-fx0) * p.dx_inv;
#endif
            applyx = true;
        }
        if (p.open_y && (j == 0 || j == p.ny - 1)) {
            if (!p.spec) {
                float rv0 = r.rv[v_index(k,j == 0 ? 0 : p.ny-1,i,p.ny,p.nx)];
                float rv1 = r.rv[v_index(k,j == 0 ? 1 : p.ny,i,p.ny,p.nx)];
                float vb = j == 0 ? min(0.5f*(rv0+rv1),0.0f)
                                  : max(0.5f*(rv0+rv1),0.0f);
                int qj = j == 0 ? 0 : p.ny - 1;
                int qn = j == 0 ? 1 : p.ny - 2;
                ty = p.dy_inv * (vb * (r.q[cell_index(k,qj,i,p.ny,p.nx)]
                                       - r.q[cell_index(k,qn,i,p.ny,p.nx)])
                           + r.q[cell_index(k,qj,i,p.ny,p.nx)]*(rv1-rv0));
#if !GPUWM_WRF_EXACT_C_ADVECTION
                tb += -p.dy_inv * (vb * (r.q[cell_index(k,qj,i,p.ny,p.nx)]
                                         - r.q[cell_index(k,qn,i,p.ny,p.nx)])
                             + r.q[cell_index(k,qj,i,p.ny,p.nx)]*(rv1-rv0));
#endif
                applyy = true;
            }
        } else {
            float v0 = r.rv[v_index(k,j,i,p.ny,p.nx)];
            float v1 = r.rv[v_index(k,j+1,i,p.ny,p.nx)];
            float fy0 = p.open_y ? yface_cell_open(r,p,v0,k,j,i)
                                  : yface_cell_periodic(r,p,v0,k,j,i);
            float fy1 = p.open_y ? yface_cell_open(r,p,v1,k,j+1,i)
                                  : yface_cell_periodic(r,p,v1,k,j+1,i);
            ty = (p.has_msf ? mscale * p.dy_inv : p.dy_inv) * (fy1-fy0);
#if !GPUWM_WRF_EXACT_C_ADVECTION
            if (p.has_msf) th += -(fy1-fy0) * p.dy_inv;
            else tb += -(fy1-fy0) * p.dy_inv;
#endif
            applyy = true;
        }
        float fz0 = zface_half(r.q,r.rw[w_index(k,j,i,p.ny,p.nx)],k,
                               j,i,p.nz,p.ny,p.nx,r.fnm,r.fnp);
        float fz1 = zface_half(r.q,r.rw[w_index(k+1,j,i,p.ny,p.nx)],k+1,
                               j,i,p.nz,p.ny,p.nx,r.fnm,r.fnp);
#if GPUWM_WRF_EXACT_C_ADVECTION
        float result = r.tend[out];
        if (applyy) result = result - ty;
        if (applyx) result = result - tx;
        r.tend[out] = result - r.rdnw[k] * (fz1-fz0);
#else
        float t = p.has_msf ? mscale * th + tb : tb + th;
        r.tend[out] += t - (fz1-fz0) * r.rdnw[k];
#endif
        return;
    }

    const int ix[6] = {periodic(i - 3, p.nx), periodic(i - 2, p.nx),
                       periodic(i - 1, p.nx), periodic(i, p.nx),
                       periodic(i + 1, p.nx), periodic(i + 2, p.nx)};
    const int iy[6] = {periodic(j - 3, p.ny), periodic(j - 2, p.ny),
                       periodic(j - 1, p.ny), periodic(j, p.ny),
                       periodic(j + 1, p.ny), periodic(j + 2, p.ny)};
    float fx0 = flux5(r.q[cell_index(k,j,ix[0],p.ny,p.nx)],
                      r.q[cell_index(k,j,ix[1],p.ny,p.nx)],
                      r.q[cell_index(k,j,ix[2],p.ny,p.nx)],
                      r.q[cell_index(k,j,ix[3],p.ny,p.nx)],
                      r.q[cell_index(k,j,ix[4],p.ny,p.nx)],
                      r.q[cell_index(k,j,ix[5],p.ny,p.nx)],
                      r.ru[u_index(k,j,i,p.ny,p.nx)]);
    const int ix1[6] = {periodic(i - 2, p.nx), periodic(i - 1, p.nx),
                        periodic(i, p.nx), periodic(i + 1, p.nx),
                        periodic(i + 2, p.nx), periodic(i + 3, p.nx)};
    float fx1 = flux5(r.q[cell_index(k,j,ix1[0],p.ny,p.nx)],
                      r.q[cell_index(k,j,ix1[1],p.ny,p.nx)],
                      r.q[cell_index(k,j,ix1[2],p.ny,p.nx)],
                      r.q[cell_index(k,j,ix1[3],p.ny,p.nx)],
                      r.q[cell_index(k,j,ix1[4],p.ny,p.nx)],
                      r.q[cell_index(k,j,ix1[5],p.ny,p.nx)],
                      r.ru[u_index(k,j,i+1,p.ny,p.nx)]);
    float fy0 = flux5(r.q[cell_index(k,iy[0],i,p.ny,p.nx)],
                      r.q[cell_index(k,iy[1],i,p.ny,p.nx)],
                      r.q[cell_index(k,iy[2],i,p.ny,p.nx)],
                      r.q[cell_index(k,iy[3],i,p.ny,p.nx)],
                      r.q[cell_index(k,iy[4],i,p.ny,p.nx)],
                      r.q[cell_index(k,iy[5],i,p.ny,p.nx)],
                      r.rv[v_index(k,j,i,p.ny,p.nx)]);
    const int iy1[6] = {periodic(j - 2, p.ny), periodic(j - 1, p.ny),
                        periodic(j, p.ny), periodic(j + 1, p.ny),
                        periodic(j + 2, p.ny), periodic(j + 3, p.ny)};
    float fy1 = flux5(r.q[cell_index(k,iy1[0],i,p.ny,p.nx)],
                      r.q[cell_index(k,iy1[1],i,p.ny,p.nx)],
                      r.q[cell_index(k,iy1[2],i,p.ny,p.nx)],
                      r.q[cell_index(k,iy1[3],i,p.ny,p.nx)],
                      r.q[cell_index(k,iy1[4],i,p.ny,p.nx)],
                      r.q[cell_index(k,iy1[5],i,p.ny,p.nx)],
                      r.rv[v_index(k,j+1,i,p.ny,p.nx)]);
    const int k0 = k;
    const int k1 = k + 1;
    float fz0 = zface_half(r.q, r.rw[w_index(k0,j,i,p.ny,p.nx)], k0,
                           j, i, p.nz, p.ny, p.nx, r.fnm, r.fnp);
    float fz1 = zface_half(r.q, r.rw[w_index(k1,j,i,p.ny,p.nx)], k1,
                           j, i, p.nz, p.ny, p.nx, r.fnm, r.fnp);

#if GPUWM_WRF_EXACT_C_ADVECTION
    float result = r.tend[out];
    result = result - ((p.has_msf ? r.msf[ulong(j)*ulong(p.nx)+ulong(i)] : 1.0f)
                        * p.dy_inv) * (fy1 - fy0);
    result = result - ((p.has_msf ? r.msf[ulong(j)*ulong(p.nx)+ulong(i)] : 1.0f)
                        * p.dx_inv) * (fx1 - fx0);
    result = result - r.rdnw[k] * (fz1 - fz0);
    r.tend[out] = result;
#else
    if (p.has_msf) {
        r.tend[out] += mscale * (-(fx1 - fx0) * p.dx_inv
                                 -(fy1 - fy0) * p.dy_inv)
                       -(fz1 - fz0) * r.rdnw[k];
    } else {
        r.tend[out] += -(fx1 - fx0) * p.dx_inv
                       -(fy1 - fy0) * p.dy_inv
                       -(fz1 - fz0) * r.rdnw[k];
    }
#endif
}
