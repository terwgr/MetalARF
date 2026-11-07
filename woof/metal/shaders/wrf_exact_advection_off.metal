// Generated from woof/core/kernels/advection.cu by tools/advection_wrf_exact/generate.py; do not edit.
// Flux formulations transcribed from WRF-ARW module_advect_em.F.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_advection_off_bindings.metalh"
using namespace metal;
namespace wrf_exact_advection_off {
static inline bool operator<(Exact32 a, Exact32 b)
{ return soft32_less(a.bits, b.bits); }
static inline Exact32 exact_copysign(Exact32 a, Exact32 b)
{ return Exact32::from_bits((a.bits & 0x7fffffffu) | (b.bits & 0x80000000u)); }
  int adv_periodic(int i, int n)
{
    int r = i % n;
    return r < 0 ? r + n : r;
}

Exact32 flux5(Exact32 qm3, Exact32 qm2, Exact32 qm1, Exact32 q0, Exact32 qp1, Exact32 qp2, Exact32 vel)
{
    return exact_div((vel * (37.0f * (q0 + qm1) - 8.0f * (qp1 + qm2) + (qp2 + qm3))
            - exact_abs(vel) * (10.0f * (q0 - qm1) - 5.0f * (qp1 - qm2)
                            + (qp2 - qm3))), 60.0f);
}

Exact32 flux3(Exact32 qm2, Exact32 qm1, Exact32 q0, Exact32 qp1, Exact32 vel)
{
    return exact_div((vel * (7.0f * (q0 + qm1) - (qp1 + qm2))
            + exact_abs(vel) * (3.0f * (q0 - qm1) - (qp1 - qm2))), 12.0f);
}

Exact32 flux5v(Exact32 qm3, Exact32 qm2, Exact32 qm1, Exact32 q0, Exact32 qp1, Exact32 qp2, Exact32 vel)
{
    return exact_div((vel * (37.0f * (q0 + qm1) - 8.0f * (qp1 + qm2) + (qp2 + qm3))
            + exact_abs(vel) * (10.0f * (q0 - qm1) - 5.0f * (qp1 - qm2)
                            + (qp2 - qm3))), 60.0f);
}

Exact32 flux3h(Exact32 qm2, Exact32 qm1, Exact32 q0, Exact32 qp1, Exact32 vel)
{
    return exact_div((vel * (7.0f * (q0 + qm1) - (qp1 + qm2))
            - exact_abs(vel) * (3.0f * (q0 - qm1) - (qp1 - qm2))), 12.0f);
}

Exact32 flux2(Exact32 qm1, Exact32 q0, Exact32 vel)
{
    return 0.5f * vel * (q0 + qm1);
}

Exact32 xface_cell_open(device const Exact32* q, Exact32 vel, int k, int j, int f,
                     int n, int ny, int nxs)
{
    if (f <= 0 || f >= n) return 0.0f;
    int d = min(f, n - f);
    if (d == 1)
        return flux2(q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (f - 1))],
                     q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (f))], vel);
    if (d == 2)
        return flux3h(q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (f - 2))],
                      q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (f - 1))],
                      q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (f))],
                      q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (f + 1))], vel);
    return flux5(q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (f - 3))],
                 q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (f - 2))],
                 q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (f - 1))],
                 q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (f))],
                 q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (f + 1))],
                 q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (f + 2))], vel);
}

Exact32 xface_cell_per(device const Exact32* q, Exact32 vel, int k, int j, int f,
                    int n, int ny, int nxs)
{
    return flux5(q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (((((f - 3) % (n)) + (n)) % (n))))],
                 q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (((((f - 2) % (n)) + (n)) % (n))))],
                 q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (((((f - 1) % (n)) + (n)) % (n))))],
                 q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (((((f) % (n)) + (n)) % (n))))],
                 q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (((((f + 1) % (n)) + (n)) % (n))))],
                 q[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (((((f + 2) % (n)) + (n)) % (n))))], vel);
}

Exact32 yface_cell_open(device const Exact32* q, Exact32 vel, int k, int g, int i,
                     int n, int nys, int nxs)
{
    if (g <= 0 || g >= n) return 0.0f;
    int d = min(g, n - g);
    if (d == 1)
        return flux2(q[((ulong)(k) * (nys) * (nxs) + (ulong)(g - 1) * (nxs) + (i))],
                     q[((ulong)(k) * (nys) * (nxs) + (ulong)(g) * (nxs) + (i))], vel);
    if (d == 2)
        return flux3h(q[((ulong)(k) * (nys) * (nxs) + (ulong)(g - 2) * (nxs) + (i))],
                      q[((ulong)(k) * (nys) * (nxs) + (ulong)(g - 1) * (nxs) + (i))],
                      q[((ulong)(k) * (nys) * (nxs) + (ulong)(g) * (nxs) + (i))],
                      q[((ulong)(k) * (nys) * (nxs) + (ulong)(g + 1) * (nxs) + (i))], vel);
    return flux5(q[((ulong)(k) * (nys) * (nxs) + (ulong)(g - 3) * (nxs) + (i))],
                 q[((ulong)(k) * (nys) * (nxs) + (ulong)(g - 2) * (nxs) + (i))],
                 q[((ulong)(k) * (nys) * (nxs) + (ulong)(g - 1) * (nxs) + (i))],
                 q[((ulong)(k) * (nys) * (nxs) + (ulong)(g) * (nxs) + (i))],
                 q[((ulong)(k) * (nys) * (nxs) + (ulong)(g + 1) * (nxs) + (i))],
                 q[((ulong)(k) * (nys) * (nxs) + (ulong)(g + 2) * (nxs) + (i))], vel);
}

Exact32 yface_cell_per(device const Exact32* q, Exact32 vel, int k, int g, int i,
                    int n, int nys, int nxs)
{
    return flux5(q[((ulong)(k) * (nys) * (nxs) + (ulong)(((((g - 3) % (n)) + (n)) % (n))) * (nxs) + (i))],
                 q[((ulong)(k) * (nys) * (nxs) + (ulong)(((((g - 2) % (n)) + (n)) % (n))) * (nxs) + (i))],
                 q[((ulong)(k) * (nys) * (nxs) + (ulong)(((((g - 1) % (n)) + (n)) % (n))) * (nxs) + (i))],
                 q[((ulong)(k) * (nys) * (nxs) + (ulong)(((((g) % (n)) + (n)) % (n))) * (nxs) + (i))],
                 q[((ulong)(k) * (nys) * (nxs) + (ulong)(((((g + 1) % (n)) + (n)) % (n))) * (nxs) + (i))],
                 q[((ulong)(k) * (nys) * (nxs) + (ulong)(((((g + 2) % (n)) + (n)) % (n))) * (nxs) + (i))], vel);
}

Exact32 zface_half(device const Exact32* q, Exact32 vel, int kf, int j, int i,
                int nz, int ny, int nxs,
                device const Exact32* fnm, device const Exact32* fnp, int vorder)
{
    if (kf == 0 || kf == nz) return 0.0f;
    if (kf == 1 || kf == nz - 1)
        return vel * (fnm[kf] * q[((ulong)(kf) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))]
                      + fnp[kf] * q[((ulong)(kf - 1) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))]);
    if (vorder == 5 && kf >= 3 && kf <= nz - 3)
        return flux5v(q[((ulong)(kf - 3) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                      q[((ulong)(kf - 2) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                      q[((ulong)(kf - 1) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                      q[((ulong)(kf) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                      q[((ulong)(kf + 1) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                      q[((ulong)(kf + 2) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))], vel);
    return flux3(q[((ulong)(kf - 2) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                 q[((ulong)(kf - 1) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                 q[((ulong)(kf) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                 q[((ulong)(kf + 1) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))], vel);
}
void flux_div_scalar(device const Exact32*  q,
                     device const Exact32*  ru,
                     device const Exact32*  rv,
                     device const Exact32*  rw,
                     device Exact32*  tend_out,
                     device const Exact32*  rdnw,
                     device const Exact32*  fnm,
                     device const Exact32*  fnp,
                     device const Exact32*  msf,
                     Exact32 dx_inv, Exact32 dy_inv,
                     int nz, int ny, int nx,
                     int open_x, int open_y, int has_msf, int spec,
                     int vorder, int i, int j, int k)
{



    if (i >= nx || j >= ny || k >= nz) return;
    if ((open_x || open_y) && !has_msf) {
        Exact32 t = 0.0f;
        if (open_x && (i == 0 || i == nx - 1)) {
            if (spec) {
            } else if (i == 0) {
                Exact32 ru0 = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (0))];
                Exact32 ru1 = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (1))];
                Exact32 ub = exact_min(0.5f * (ru0 + ru1), 0.0f);
                t += -dx_inv * (ub * (q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (1))] - q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (0))])
                                + q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (0))] * (ru1 - ru0));
            } else {
                Exact32 ru0 = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (nx - 1))];
                Exact32 ru1 = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (nx))];
                Exact32 ub = exact_max(0.5f * (ru0 + ru1), 0.0f);
                t += -dx_inv * (ub * (q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 1))]
                                      - q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 2))])
                                + q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 1))] * (ru1 - ru0));
            }
        } else {
            Exact32 fx0, fx1;
            Exact32 v0 = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))];
            Exact32 v1 = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i + 1))];
            if (open_x) {
                fx0 = xface_cell_open(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_open(q, v1, k, j, i + 1, nx, ny, nx);
            } else {
                fx0 = xface_cell_per(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_per(q, v1, k, j, i + 1, nx, ny, nx);
            }
            t += -(fx1 - fx0) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
            } else if (j == 0) {
                Exact32 rv0 = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(0) * (nx) + (i))];
                Exact32 rv1 = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(1) * (nx) + (i))];
                Exact32 vb = exact_min(0.5f * (rv0 + rv1), 0.0f);
                t += -dy_inv * (vb * (q[((ulong)(k) * (ny) * (nx) + (ulong)(1) * (nx) + (i))] - q[((ulong)(k) * (ny) * (nx) + (ulong)(0) * (nx) + (i))])
                                + q[((ulong)(k) * (ny) * (nx) + (ulong)(0) * (nx) + (i))] * (rv1 - rv0));
            } else {
                Exact32 rv0 = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny - 1) * (nx) + (i))];
                Exact32 rv1 = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny) * (nx) + (i))];
                Exact32 vb = exact_max(0.5f * (rv0 + rv1), 0.0f);
                t += -dy_inv * (vb * (q[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 1) * (nx) + (i))]
                                      - q[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 2) * (nx) + (i))])
                                + q[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 1) * (nx) + (i))] * (rv1 - rv0));
            }
        } else {
            Exact32 fy0, fy1;
            Exact32 v0 = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))];
            Exact32 v1 = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j + 1) * (nx) + (i))];
            if (open_y) {
                fy0 = yface_cell_open(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_open(q, v1, k, j + 1, i, ny, ny, nx);
            } else {
                fy0 = yface_cell_per(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_per(q, v1, k, j + 1, i, ny, ny, nx);
            }
            t += -(fy1 - fy0) * dy_inv;
        }
        Exact32 fzo[2];
        for (int s = 0; s < 2; ++s) {
            int kf = k + s;
            fzo[s] = zface_half(q, rw[((ulong)(kf) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], kf, j, i,
                                nz, ny, nx, fnm, fnp, vorder);
        }
        tend_out[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] += t - (fzo[1] - fzo[0]) * rdnw[k];
        return;
    }
    if (open_x || open_y) {
        Exact32 th = 0.0f;
        Exact32 tb = 0.0f;
        if (open_x && (i == 0 || i == nx - 1)) {
            if (spec) {
            } else if (i == 0) {
                Exact32 ru0 = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (0))];
                Exact32 ru1 = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (1))];
                Exact32 ub = exact_min(0.5f * (ru0 + ru1), 0.0f);
                tb += -dx_inv * (ub * (q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (1))] - q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (0))])
                                 + q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (0))] * (ru1 - ru0));
            } else {
                Exact32 ru0 = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (nx - 1))];
                Exact32 ru1 = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (nx))];
                Exact32 ub = exact_max(0.5f * (ru0 + ru1), 0.0f);
                tb += -dx_inv * (ub * (q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 1))]
                                       - q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 2))])
                                 + q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 1))] * (ru1 - ru0));
            }
        } else {
            Exact32 fx0, fx1;
            Exact32 v0 = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))];
            Exact32 v1 = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i + 1))];
            if (open_x) {
                fx0 = xface_cell_open(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_open(q, v1, k, j, i + 1, nx, ny, nx);
            } else {
                fx0 = xface_cell_per(q, v0, k, j, i, nx, ny, nx);
                fx1 = xface_cell_per(q, v1, k, j, i + 1, nx, ny, nx);
            }
            th += -(fx1 - fx0) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
            } else if (j == 0) {
                Exact32 rv0 = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(0) * (nx) + (i))];
                Exact32 rv1 = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(1) * (nx) + (i))];
                Exact32 vb = exact_min(0.5f * (rv0 + rv1), 0.0f);
                tb += -dy_inv * (vb * (q[((ulong)(k) * (ny) * (nx) + (ulong)(1) * (nx) + (i))] - q[((ulong)(k) * (ny) * (nx) + (ulong)(0) * (nx) + (i))])
                                 + q[((ulong)(k) * (ny) * (nx) + (ulong)(0) * (nx) + (i))] * (rv1 - rv0));
            } else {
                Exact32 rv0 = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny - 1) * (nx) + (i))];
                Exact32 rv1 = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny) * (nx) + (i))];
                Exact32 vb = exact_max(0.5f * (rv0 + rv1), 0.0f);
                tb += -dy_inv * (vb * (q[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 1) * (nx) + (i))]
                                       - q[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 2) * (nx) + (i))])
                                 + q[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 1) * (nx) + (i))] * (rv1 - rv0));
            }
        } else {
            Exact32 fy0, fy1;
            Exact32 v0 = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))];
            Exact32 v1 = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j + 1) * (nx) + (i))];
            if (open_y) {
                fy0 = yface_cell_open(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_open(q, v1, k, j + 1, i, ny, ny, nx);
            } else {
                fy0 = yface_cell_per(q, v0, k, j, i, ny, ny, nx);
                fy1 = yface_cell_per(q, v1, k, j + 1, i, ny, ny, nx);
            }
            th += -(fy1 - fy0) * dy_inv;
        }
        Exact32 fzo[2];
        for (int s = 0; s < 2; ++s) {
            int kf = k + s;
            fzo[s] = zface_half(q, rw[((ulong)(kf) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], kf, j, i,
                                nz, ny, nx, fnm, fnp, vorder);
        }
        Exact32 t = msf[(ulong)j * nx + i] * th + tb;
        tend_out[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] += t - (fzo[1] - fzo[0]) * rdnw[k];
        return;
    }
    Exact32 fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int f = i + s;
        fx[s] = flux5(q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (adv_periodic(f - 3, nx)))],
                      q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (adv_periodic(f - 2, nx)))],
                      q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (adv_periodic(f - 1, nx)))],
                      q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (adv_periodic(f, nx)))],
                      q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (adv_periodic(f + 1, nx)))],
                      q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (adv_periodic(f + 2, nx)))],
                      ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (f))]);
        int g = j + s;
        fy[s] = flux5(q[((ulong)(k) * (ny) * (nx) + (ulong)(adv_periodic(g - 3, ny)) * (nx) + (i))],
                      q[((ulong)(k) * (ny) * (nx) + (ulong)(adv_periodic(g - 2, ny)) * (nx) + (i))],
                      q[((ulong)(k) * (ny) * (nx) + (ulong)(adv_periodic(g - 1, ny)) * (nx) + (i))],
                      q[((ulong)(k) * (ny) * (nx) + (ulong)(adv_periodic(g, ny)) * (nx) + (i))],
                      q[((ulong)(k) * (ny) * (nx) + (ulong)(adv_periodic(g + 1, ny)) * (nx) + (i))],
                      q[((ulong)(k) * (ny) * (nx) + (ulong)(adv_periodic(g + 2, ny)) * (nx) + (i))],
                      rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(g) * (nx) + (i))]);
        int kf = k + s;
        Exact32 vel = rw[((ulong)(kf) * (ny) * (nx) + (ulong)(j) * (nx) + (i))];
        fz[s] = zface_half(q, vel, kf, j, i, nz, ny, nx, fnm, fnp, vorder);
    }
    if (has_msf) {
        tend_out[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] += msf[(ulong)j * nx + i]
                                   * (-(fx[1] - fx[0]) * dx_inv
                                      - (fy[1] - fy[0]) * dy_inv)
                                   - (fz[1] - fz[0]) * rdnw[k];
    } else {
        tend_out[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] += -(fx[1] - fx[0]) * dx_inv
                                   - (fy[1] - fy[0]) * dy_inv
                                   - (fz[1] - fz[0]) * rdnw[k];
    }
}
void flux_div_u(device const Exact32*  u,
                device const Exact32*  ru,
                device const Exact32*  rv,
                device const Exact32*  rw,
                device Exact32*  tend_out,
                device const Exact32*  rdnw,
                device const Exact32*  fnm,
                device const Exact32*  fnp,
                device const Exact32*  msf,
                Exact32 dx_inv, Exact32 dy_inv,
                int nz, int ny, int nx,
                int open_x, int open_y, int has_msf, int spec,
                     int vorder, int i, int j, int k)
{



    if (i >= nx + 1 || j >= ny || k >= nz) return;
    if (open_x || open_y) {
        int nxs = nx + 1;
        bool xedge = open_x && (i == 0 || i == nx);
        int cl = open_x ? max(i - 1, 0) : ((((i - 1) % (nx)) + (nx)) % (nx));
        int cr = open_x ? min(i, nx - 1) : ((((i) % (nx)) + (nx)) % (nx));
        int iu = open_x ? i : cr;
        Exact32 th = 0.0f;
        if (!xedge) {
            Exact32 fx2[2];
            for (int s = 0; s < 2; ++s) {
                int m = (s == 0) ? cl : cr;
                Exact32 velx = 0.5f * (ru[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m))]
                                    + ru[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m + 1))]);
                if (open_x) {
                    int d = min(m, nx - 1 - m);
                    if (d == 0) {
                        Exact32 qa = u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m))];
                        Exact32 qb = u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m + 1))];
                        if (spec) {
                            if (m == 0) { if (qb < 0.0f) qa = qb; }
                            else { if (qa > 0.0f) qb = qa; }
                        }
                        fx2[s] = flux2(qa, qb, velx);
                    }
                    else if (d == 1)
                        fx2[s] = flux3h(u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m - 1))],
                                        u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m))],
                                        u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m + 1))],
                                        u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m + 2))], velx);
                    else
                        fx2[s] = flux5(u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m - 2))],
                                       u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m - 1))],
                                       u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m))],
                                       u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m + 1))],
                                       u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m + 2))],
                                       u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (m + 3))], velx);
                } else {
                    fx2[s] = flux5(u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (((((m - 2) % (nx)) + (nx)) % (nx))))],
                                   u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (((((m - 1) % (nx)) + (nx)) % (nx))))],
                                   u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (((((m) % (nx)) + (nx)) % (nx))))],
                                   u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (((((m + 1) % (nx)) + (nx)) % (nx))))],
                                   u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (((((m + 2) % (nx)) + (nx)) % (nx))))],
                                   u[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (((((m + 3) % (nx)) + (nx)) % (nx))))],
                                   velx);
                }
            }
            th += -(fx2[1] - fx2[0]) * dx_inv;
        }
        if (!xedge) {
            if (open_y && (j == 0 || j == ny - 1)) {
                if (spec) {
                } else if (j == 0) {
                    Exact32 vw = 0.5f * (rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(0) * (nx) + (cl))]
                                      + rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(0) * (nx) + (cr))]);
                    Exact32 vb = exact_min(vw, 0.0f);
                    Exact32 dsum = (rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(1) * (nx) + (cl))]
                                 - rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(0) * (nx) + (cl))])
                              + (rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(1) * (nx) + (cr))]
                                 - rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(0) * (nx) + (cr))]);
                    th += -dy_inv * (vb * (u[((ulong)(k) * (ny) * (nxs) + (ulong)(1) * (nxs) + (iu))]
                                           - u[((ulong)(k) * (ny) * (nxs) + (ulong)(0) * (nxs) + (iu))])
                                     + 0.5f * u[((ulong)(k) * (ny) * (nxs) + (ulong)(0) * (nxs) + (iu))] * dsum);
                } else {
                    Exact32 vw = 0.5f * (rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny) * (nx) + (cl))]
                                      + rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny) * (nx) + (cr))]);
                    Exact32 vb = exact_max(vw, 0.0f);
                    Exact32 dsum = (rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny) * (nx) + (cl))]
                                 - rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny - 1) * (nx) + (cl))])
                              + (rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny) * (nx) + (cr))]
                                 - rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny - 1) * (nx) + (cr))]);
                    th += -dy_inv * (vb * (u[((ulong)(k) * (ny) * (nxs) + (ulong)(ny - 1) * (nxs) + (iu))]
                                           - u[((ulong)(k) * (ny) * (nxs) + (ulong)(ny - 2) * (nxs) + (iu))])
                                     + 0.5f * u[((ulong)(k) * (ny) * (nxs) + (ulong)(ny - 1) * (nxs) + (iu))]
                                       * dsum);
                }
            } else {
                Exact32 fy2[2];
                for (int s = 0; s < 2; ++s) {
                    int g = j + s;
                    Exact32 vely = 0.5f * (rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(g) * (nx) + (cl))]
                                        + rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(g) * (nx) + (cr))]);
                    fy2[s] = open_y
                        ? yface_cell_open(u, vely, k, g, iu, ny, ny, nxs)
                        : yface_cell_per(u, vely, k, g, iu, ny, ny, nxs);
                }
                th += -(fy2[1] - fy2[0]) * dy_inv;
            }
        }
        Exact32 t = has_msf ? msf[(ulong)j * (nx + 1) + i] * th : th;
        if (!(xedge && open_y)) {
            Exact32 fz2[2];
            for (int s = 0; s < 2; ++s) {
                int kf = k + s;
                Exact32 velz;
                if (open_x && i == 0)
                    velz = rw[((ulong)(kf) * (ny) * (nx) + (ulong)(j) * (nx) + (0))];
                else if (open_x && i == nx)
                    velz = rw[((ulong)(kf) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 1))];
                else
                    velz = 0.5f * (rw[((ulong)(kf) * (ny) * (nx) + (ulong)(j) * (nx) + (cl))]
                                   + rw[((ulong)(kf) * (ny) * (nx) + (ulong)(j) * (nx) + (cr))]);
                fz2[s] = zface_half(u, velz, kf, j, iu, nz, ny, nxs, fnm, fnp, vorder);
            }
            t += -(fz2[1] - fz2[0]) * rdnw[k];
        }
        tend_out[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))] += t;
        return;
    }
    int ic = ((((i) % (nx)) + (nx)) % (nx));
    int im1c = ((((i - 1) % (nx)) + (nx)) % (nx));
    Exact32 fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int m = (s == 0) ? im1c : ic;
        Exact32 velx = 0.5f * (ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (m))]
                          + ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (m + 1))]);
        fx[s] = flux5(u[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (((((m - 2) % (nx)) + (nx)) % (nx))))],
                      u[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (((((m - 1) % (nx)) + (nx)) % (nx))))],
                      u[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (((((m) % (nx)) + (nx)) % (nx))))],
                      u[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (((((m + 1) % (nx)) + (nx)) % (nx))))],
                      u[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (((((m + 2) % (nx)) + (nx)) % (nx))))],
                      u[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (((((m + 3) % (nx)) + (nx)) % (nx))))],
                      velx);
        int g = j + s;
        Exact32 vely = 0.5f * (rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(g) * (nx) + (im1c))]
                          + rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(g) * (nx) + (ic))]);
        fy[s] = flux5(u[((ulong)(k) * (ny) * (nx + 1) + (ulong)(((((g - 3) % (ny)) + (ny)) % (ny))) * (nx + 1) + (ic))],
                      u[((ulong)(k) * (ny) * (nx + 1) + (ulong)(((((g - 2) % (ny)) + (ny)) % (ny))) * (nx + 1) + (ic))],
                      u[((ulong)(k) * (ny) * (nx + 1) + (ulong)(((((g - 1) % (ny)) + (ny)) % (ny))) * (nx + 1) + (ic))],
                      u[((ulong)(k) * (ny) * (nx + 1) + (ulong)(((((g) % (ny)) + (ny)) % (ny))) * (nx + 1) + (ic))],
                      u[((ulong)(k) * (ny) * (nx + 1) + (ulong)(((((g + 1) % (ny)) + (ny)) % (ny))) * (nx + 1) + (ic))],
                      u[((ulong)(k) * (ny) * (nx + 1) + (ulong)(((((g + 2) % (ny)) + (ny)) % (ny))) * (nx + 1) + (ic))],
                      vely);
        int kf = k + s;
        Exact32 velz = 0.5f * (rw[((ulong)(kf) * (ny) * (nx) + (ulong)(j) * (nx) + (im1c))]
                          + rw[((ulong)(kf) * (ny) * (nx) + (ulong)(j) * (nx) + (ic))]);
        fz[s] = zface_half(u, velz, kf, j, ic, nz, ny, nx + 1, fnm, fnp, vorder);
    }
    if (has_msf) {
        tend_out[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))] += msf[(ulong)j * (nx + 1) + i]
                                             * (-(fx[1] - fx[0]) * dx_inv
                                                - (fy[1] - fy[0]) * dy_inv)
                                             - (fz[1] - fz[0]) * rdnw[k];
    } else {
        tend_out[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))] += -(fx[1] - fx[0]) * dx_inv
                                             - (fy[1] - fy[0]) * dy_inv
                                             - (fz[1] - fz[0]) * rdnw[k];
    }
}
void flux_div_v(device const Exact32*  v,
                device const Exact32*  ru,
                device const Exact32*  rv,
                device const Exact32*  rw,
                device Exact32*  tend_out,
                device const Exact32*  rdnw,
                device const Exact32*  fnm,
                device const Exact32*  fnp,
                device const Exact32*  msf,
                Exact32 dx_inv, Exact32 dy_inv,
                int nz, int ny, int nx,
                int open_x, int open_y, int has_msf, int spec,
                     int vorder, int i, int j, int k)
{



    if (i >= nx || j >= ny + 1 || k >= nz) return;
    if (open_x || open_y) {
        int nys = ny + 1;
        bool yedge = open_y && (j == 0 || j == ny);
        int rs = open_y ? max(j - 1, 0) : adv_periodic(j - 1, ny);
        int rn = open_y ? min(j, ny - 1) : adv_periodic(j, ny);
        int jv = open_y ? j : rn;
        Exact32 th = 0.0f;
        if (!yedge) {
            Exact32 fy2[2];
            for (int s = 0; s < 2; ++s) {
                int m = (s == 0) ? rs : rn;
                Exact32 vely = 0.5f * (rv[((ulong)(k) * (nys) * (nx) + (ulong)(m) * (nx) + (i))]
                                    + rv[((ulong)(k) * (nys) * (nx) + (ulong)(m + 1) * (nx) + (i))]);
                if (open_y) {
                    int d = min(m, ny - 1 - m);
                    if (d == 0) {
                        Exact32 qa = v[((ulong)(k) * (nys) * (nx) + (ulong)(m) * (nx) + (i))];
                        Exact32 qb = v[((ulong)(k) * (nys) * (nx) + (ulong)(m + 1) * (nx) + (i))];
                        if (spec) {
                            if (m == 0) { if (qb < 0.0f) qa = qb; }
                            else { if (qa > 0.0f) qb = qa; }
                        }
                        fy2[s] = flux2(qa, qb, vely);
                    }
                    else if (d == 1)
                        fy2[s] = flux3h(v[((ulong)(k) * (nys) * (nx) + (ulong)(m - 1) * (nx) + (i))],
                                        v[((ulong)(k) * (nys) * (nx) + (ulong)(m) * (nx) + (i))],
                                        v[((ulong)(k) * (nys) * (nx) + (ulong)(m + 1) * (nx) + (i))],
                                        v[((ulong)(k) * (nys) * (nx) + (ulong)(m + 2) * (nx) + (i))], vely);
                    else
                        fy2[s] = flux5(v[((ulong)(k) * (nys) * (nx) + (ulong)(m - 2) * (nx) + (i))],
                                       v[((ulong)(k) * (nys) * (nx) + (ulong)(m - 1) * (nx) + (i))],
                                       v[((ulong)(k) * (nys) * (nx) + (ulong)(m) * (nx) + (i))],
                                       v[((ulong)(k) * (nys) * (nx) + (ulong)(m + 1) * (nx) + (i))],
                                       v[((ulong)(k) * (nys) * (nx) + (ulong)(m + 2) * (nx) + (i))],
                                       v[((ulong)(k) * (nys) * (nx) + (ulong)(m + 3) * (nx) + (i))], vely);
                } else {
                    fy2[s] = flux5(v[((ulong)(k) * (nys) * (nx) + (ulong)(adv_periodic(m - 2, ny)) * (nx) + (i))],
                                   v[((ulong)(k) * (nys) * (nx) + (ulong)(adv_periodic(m - 1, ny)) * (nx) + (i))],
                                   v[((ulong)(k) * (nys) * (nx) + (ulong)(adv_periodic(m, ny)) * (nx) + (i))],
                                   v[((ulong)(k) * (nys) * (nx) + (ulong)(adv_periodic(m + 1, ny)) * (nx) + (i))],
                                   v[((ulong)(k) * (nys) * (nx) + (ulong)(adv_periodic(m + 2, ny)) * (nx) + (i))],
                                   v[((ulong)(k) * (nys) * (nx) + (ulong)(adv_periodic(m + 3, ny)) * (nx) + (i))],
                                   vely);
                }
            }
            th += -(fy2[1] - fy2[0]) * dy_inv;
        }
        if (!yedge) {
            if (open_x && (i == 0 || i == nx - 1)) {
                if (spec) {
                } else if (i == 0) {
                    Exact32 uw0 = 0.5f * (ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(rs) * (nx + 1) + (0))]
                                       + ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(rn) * (nx + 1) + (0))]);
                    Exact32 uw1 = 0.5f * (ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(rs) * (nx + 1) + (1))]
                                       + ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(rn) * (nx + 1) + (1))]);
                    Exact32 ub = exact_min(uw0, 0.0f);
                    th += -dx_inv * (ub * (v[((ulong)(k) * (nys) * (nx) + (ulong)(jv) * (nx) + (1))]
                                           - v[((ulong)(k) * (nys) * (nx) + (ulong)(jv) * (nx) + (0))])
                                     + v[((ulong)(k) * (nys) * (nx) + (ulong)(jv) * (nx) + (0))] * (uw1 - uw0));
                } else {
                    Exact32 uw0 = 0.5f * (ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(rs) * (nx + 1) + (nx - 1))]
                                       + ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(rn) * (nx + 1) + (nx - 1))]);
                    Exact32 uw1 = 0.5f * (ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(rs) * (nx + 1) + (nx))]
                                       + ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(rn) * (nx + 1) + (nx))]);
                    Exact32 ub = exact_max(uw1, 0.0f);
                    th += -dx_inv * (ub * (v[((ulong)(k) * (nys) * (nx) + (ulong)(jv) * (nx) + (nx - 1))]
                                           - v[((ulong)(k) * (nys) * (nx) + (ulong)(jv) * (nx) + (nx - 2))])
                                     + v[((ulong)(k) * (nys) * (nx) + (ulong)(jv) * (nx) + (nx - 1))]
                                       * (uw1 - uw0));
                }
            } else {
                Exact32 fx2[2];
                for (int s = 0; s < 2; ++s) {
                    int f = i + s;
                    Exact32 velx = 0.5f * (ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(rs) * (nx + 1) + (f))]
                                        + ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(rn) * (nx + 1) + (f))]);
                    fx2[s] = open_x
                        ? xface_cell_open(v, velx, k, jv, f, nx, nys, nx)
                        : xface_cell_per(v, velx, k, jv, f, nx, nys, nx);
                }
                th += -(fx2[1] - fx2[0]) * dx_inv;
            }
        }
        Exact32 t = has_msf ? msf[(ulong)j * nx + i] * th : th;
        if (!yedge) {
            Exact32 fz2[2];
            for (int s = 0; s < 2; ++s) {
                int kf = k + s;
                Exact32 velz = 0.5f * (rw[((ulong)(kf) * (ny) * (nx) + (ulong)(rs) * (nx) + (i))]
                                    + rw[((ulong)(kf) * (ny) * (nx) + (ulong)(rn) * (nx) + (i))]);
                fz2[s] = zface_half(v, velz, kf, jv, i, nz, nys, nx, fnm, fnp, vorder);
            }
            t += -(fz2[1] - fz2[0]) * rdnw[k];
        }
        tend_out[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + (i))] += t;
        return;
    }
    int jc = adv_periodic(j, ny);
    int jm1c = adv_periodic(j - 1, ny);
    Exact32 fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int f = i + s;
        Exact32 velx = 0.5f * (ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(jm1c) * (nx + 1) + (f))]
                          + ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(jc) * (nx + 1) + (f))]);
        fx[s] = flux5(v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(jc) * (nx) + (adv_periodic(f - 3, nx)))],
                      v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(jc) * (nx) + (adv_periodic(f - 2, nx)))],
                      v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(jc) * (nx) + (adv_periodic(f - 1, nx)))],
                      v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(jc) * (nx) + (adv_periodic(f, nx)))],
                      v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(jc) * (nx) + (adv_periodic(f + 1, nx)))],
                      v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(jc) * (nx) + (adv_periodic(f + 2, nx)))],
                      velx);
        int m = (s == 0) ? jm1c : jc;
        Exact32 vely = 0.5f * (rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(m) * (nx) + (i))]
                          + rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(m + 1) * (nx) + (i))]);
        fy[s] = flux5(v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(adv_periodic(m - 2, ny)) * (nx) + (i))],
                      v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(adv_periodic(m - 1, ny)) * (nx) + (i))],
                      v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(adv_periodic(m, ny)) * (nx) + (i))],
                      v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(adv_periodic(m + 1, ny)) * (nx) + (i))],
                      v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(adv_periodic(m + 2, ny)) * (nx) + (i))],
                      v[((ulong)(k) * (ny + 1) * (nx) + (ulong)(adv_periodic(m + 3, ny)) * (nx) + (i))],
                      vely);
        int kf = k + s;
        Exact32 velz = 0.5f * (rw[((ulong)(kf) * (ny) * (nx) + (ulong)(jm1c) * (nx) + (i))]
                          + rw[((ulong)(kf) * (ny) * (nx) + (ulong)(jc) * (nx) + (i))]);
        fz[s] = zface_half(v, velz, kf, jc, i, nz, ny + 1, nx, fnm, fnp, vorder);
    }
    if (has_msf) {
        tend_out[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))] += msf[(ulong)j * nx + i]
                                             * (-(fx[1] - fx[0]) * dx_inv
                                                - (fy[1] - fy[0]) * dy_inv)
                                             - (fz[1] - fz[0]) * rdnw[k];
    } else {
        tend_out[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))] += -(fx[1] - fx[0]) * dx_inv
                                             - (fy[1] - fy[0]) * dy_inv
                                             - (fz[1] - fz[0]) * rdnw[k];
    }
}

Exact32 w_lid_velocity(device const Exact32* flux, int j, int i, int nz, int nys, int nxs,
                    device const Exact32* fnm, device const Exact32* fnp)
{
    return (2.0f - fnm[nz - 1]) * flux[((ulong)(nz - 1) * (nys) * (nxs) + (ulong)(j) * (nxs) + (i))]
           - fnp[nz - 1] * flux[((ulong)(nz - 2) * (nys) * (nxs) + (ulong)(j) * (nxs) + (i))];
}
void flux_div_w(device const Exact32*  w,
                device const Exact32*  ru,
                device const Exact32*  rv,
                device const Exact32*  rw,
                device Exact32*  tend_out,
                device const Exact32*  rdn,
                device const Exact32*  fnm,
                device const Exact32*  fnp,
                device const Exact32*  msf,
                Exact32 dx_inv, Exact32 dy_inv,
                int nz, int ny, int nx,
                int open_x, int open_y, int has_msf, int spec,
                     int vorder, int i, int j, int k)
{



    if (i >= nx || j >= ny || k > nz) return;
    if (k == 0 || (k == nz && nz < 2)) return;
    if (k == nz) {
        Exact32 th = 0.0f;
        Exact32 tb = 0.0f;
        if (open_x && (i == 0 || i == nx - 1)) {
            if (!spec) {
                int f0 = i == 0 ? 0 : nx - 1;
                Exact32 r0 = w_lid_velocity(ru, j, f0, nz, ny, nx + 1, fnm, fnp);
                Exact32 r1 = w_lid_velocity(ru, j, f0 + 1, nz, ny, nx + 1, fnm, fnp);
                Exact32 ub = i == 0 ? exact_min(0.5f * (r0 + r1), 0.0f)
                                : exact_max(0.5f * (r0 + r1), 0.0f);
                Exact32 dq = i == 0 ? w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (1))] - w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (0))]
                                : w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 1))] - w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 2))];
                tb += -dx_inv * (ub * dq + w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] * (r1 - r0));
            }
        } else {
            Exact32 r0 = w_lid_velocity(ru, j, i, nz, ny, nx + 1, fnm, fnp);
            Exact32 r1 = w_lid_velocity(ru, j, i + 1, nz, ny, nx + 1, fnm, fnp);
            Exact32 f0 = open_x ? xface_cell_open(w, r0, k, j, i, nx, ny, nx)
                             : xface_cell_per(w, r0, k, j, i, nx, ny, nx);
            Exact32 f1 = open_x ? xface_cell_open(w, r1, k, j, i + 1, nx, ny, nx)
                             : xface_cell_per(w, r1, k, j, i + 1, nx, ny, nx);
            th += -(f1 - f0) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (!spec) {
                int g0 = j == 0 ? 0 : ny - 1;
                Exact32 r0 = w_lid_velocity(rv, g0, i, nz, ny + 1, nx, fnm, fnp);
                Exact32 r1 = w_lid_velocity(rv, g0 + 1, i, nz, ny + 1, nx, fnm, fnp);
                Exact32 vb = j == 0 ? exact_min(0.5f * (r0 + r1), 0.0f)
                                : exact_max(0.5f * (r0 + r1), 0.0f);
                Exact32 dq = j == 0 ? w[((ulong)(k) * (ny) * (nx) + (ulong)(1) * (nx) + (i))] - w[((ulong)(k) * (ny) * (nx) + (ulong)(0) * (nx) + (i))]
                                : w[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 1) * (nx) + (i))] - w[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 2) * (nx) + (i))];
                tb += -dy_inv * (vb * dq + w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] * (r1 - r0));
            }
        } else {
            Exact32 r0 = w_lid_velocity(rv, j, i, nz, ny + 1, nx, fnm, fnp);
            Exact32 r1 = w_lid_velocity(rv, j + 1, i, nz, ny + 1, nx, fnm, fnp);
            Exact32 f0 = open_y ? yface_cell_open(w, r0, k, j, i, ny, ny, nx)
                             : yface_cell_per(w, r0, k, j, i, ny, ny, nx);
            Exact32 f1 = open_y ? yface_cell_open(w, r1, k, j + 1, i, ny, ny, nx)
                             : yface_cell_per(w, r1, k, j + 1, i, ny, ny, nx);
            th += -(f1 - f0) * dy_inv;
        }
        Exact32 horizontal = has_msf ? msf[(ulong)j * nx + i] * th + tb : th + tb;
        Exact32 top_flux = 0.25f * (rw[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] + rw[((ulong)(k - 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))])
                             * (w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] + w[((ulong)(k - 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]);
        tend_out[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] += horizontal + 2.0f * rdn[k - 1] * top_flux;
        return;
    }
    if ((open_x || open_y) && !has_msf) {
        Exact32 t = 0.0f;
        if (open_x && (i == 0 || i == nx - 1)) {
            if (spec) {
            } else if (i == 0) {
                Exact32 ruw0 = fnm[k] * ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (0))]
                          + fnp[k] * ru[((ulong)(k - 1) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (0))];
                Exact32 ruw1 = fnm[k] * ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (1))]
                          + fnp[k] * ru[((ulong)(k - 1) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (1))];
                Exact32 ub = exact_min(0.5f * (ruw0 + ruw1), 0.0f);
                t += -dx_inv * (ub * (w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (1))] - w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (0))])
                                + w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (0))] * (ruw1 - ruw0));
            } else {
                Exact32 ruw0 = fnm[k] * ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (nx - 1))]
                          + fnp[k] * ru[((ulong)(k - 1) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (nx - 1))];
                Exact32 ruw1 = fnm[k] * ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (nx))]
                          + fnp[k] * ru[((ulong)(k - 1) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (nx))];
                Exact32 ub = exact_max(0.5f * (ruw0 + ruw1), 0.0f);
                t += -dx_inv * (ub * (w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 1))]
                                      - w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 2))])
                                + w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 1))] * (ruw1 - ruw0));
            }
        } else {
            Exact32 fx2[2];
            for (int s = 0; s < 2; ++s) {
                int f = i + s;
                Exact32 velx = fnm[k] * ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (f))]
                          + fnp[k] * ru[((ulong)(k - 1) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (f))];
                fx2[s] = open_x
                    ? xface_cell_open(w, velx, k, j, f, nx, ny, nx)
                    : xface_cell_per(w, velx, k, j, f, nx, ny, nx);
            }
            t += -(fx2[1] - fx2[0]) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
            } else if (j == 0) {
                Exact32 rvw0 = fnm[k] * rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(0) * (nx) + (i))]
                          + fnp[k] * rv[((ulong)(k - 1) * (ny + 1) * (nx) + (ulong)(0) * (nx) + (i))];
                Exact32 rvw1 = fnm[k] * rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(1) * (nx) + (i))]
                          + fnp[k] * rv[((ulong)(k - 1) * (ny + 1) * (nx) + (ulong)(1) * (nx) + (i))];
                Exact32 vb = exact_min(0.5f * (rvw0 + rvw1), 0.0f);
                t += -dy_inv * (vb * (w[((ulong)(k) * (ny) * (nx) + (ulong)(1) * (nx) + (i))] - w[((ulong)(k) * (ny) * (nx) + (ulong)(0) * (nx) + (i))])
                                + w[((ulong)(k) * (ny) * (nx) + (ulong)(0) * (nx) + (i))] * (rvw1 - rvw0));
            } else {
                Exact32 rvw0 = fnm[k] * rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny - 1) * (nx) + (i))]
                          + fnp[k] * rv[((ulong)(k - 1) * (ny + 1) * (nx) + (ulong)(ny - 1) * (nx) + (i))];
                Exact32 rvw1 = fnm[k] * rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny) * (nx) + (i))]
                          + fnp[k] * rv[((ulong)(k - 1) * (ny + 1) * (nx) + (ulong)(ny) * (nx) + (i))];
                Exact32 vb = exact_max(0.5f * (rvw0 + rvw1), 0.0f);
                t += -dy_inv * (vb * (w[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 1) * (nx) + (i))]
                                      - w[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 2) * (nx) + (i))])
                                + w[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 1) * (nx) + (i))] * (rvw1 - rvw0));
            }
        } else {
            Exact32 fy2[2];
            for (int s = 0; s < 2; ++s) {
                int g = j + s;
                Exact32 vely = fnm[k] * rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(g) * (nx) + (i))]
                          + fnp[k] * rv[((ulong)(k - 1) * (ny + 1) * (nx) + (ulong)(g) * (nx) + (i))];
                fy2[s] = open_y
                    ? yface_cell_open(w, vely, k, g, i, ny, ny, nx)
                    : yface_cell_per(w, vely, k, g, i, ny, ny, nx);
            }
            t += -(fy2[1] - fy2[0]) * dy_inv;
        }
        Exact32 fz2[2];
        for (int s = 0; s < 2; ++s) {
            int m = k - 1 + s;
            Exact32 velz = 0.5f * (rw[((ulong)(m) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] + rw[((ulong)(m + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]);
            if (m == 0 || m == nz - 1)
                fz2[s] = flux2(w[((ulong)(m) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], velz);
            else if (vorder == 5 && m >= 2 && m <= nz - 3)
                fz2[s] = flux5v(w[((ulong)(m - 2) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m - 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                                w[((ulong)(m) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                                w[((ulong)(m + 2) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m + 3) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                                velz);
            else
                fz2[s] = flux3(w[((ulong)(m - 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                               w[((ulong)(m + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m + 2) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                               velz);
        }
        tend_out[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] += t - (fz2[1] - fz2[0]) * rdn[k];
        return;
    }
    if (open_x || open_y) {
        Exact32 th = 0.0f;
        Exact32 tb = 0.0f;
        if (open_x && (i == 0 || i == nx - 1)) {
            if (spec) {
            } else if (i == 0) {
                Exact32 ruw0 = fnm[k] * ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (0))]
                          + fnp[k] * ru[((ulong)(k - 1) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (0))];
                Exact32 ruw1 = fnm[k] * ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (1))]
                          + fnp[k] * ru[((ulong)(k - 1) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (1))];
                Exact32 ub = exact_min(0.5f * (ruw0 + ruw1), 0.0f);
                tb += -dx_inv * (ub * (w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (1))] - w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (0))])
                                 + w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (0))] * (ruw1 - ruw0));
            } else {
                Exact32 ruw0 = fnm[k] * ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (nx - 1))]
                          + fnp[k] * ru[((ulong)(k - 1) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (nx - 1))];
                Exact32 ruw1 = fnm[k] * ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (nx))]
                          + fnp[k] * ru[((ulong)(k - 1) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (nx))];
                Exact32 ub = exact_max(0.5f * (ruw0 + ruw1), 0.0f);
                tb += -dx_inv * (ub * (w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 1))]
                                       - w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 2))])
                                 + w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (nx - 1))] * (ruw1 - ruw0));
            }
        } else {
            Exact32 fx2[2];
            for (int s = 0; s < 2; ++s) {
                int f = i + s;
                Exact32 velx = fnm[k] * ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (f))]
                          + fnp[k] * ru[((ulong)(k - 1) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (f))];
                fx2[s] = open_x
                    ? xface_cell_open(w, velx, k, j, f, nx, ny, nx)
                    : xface_cell_per(w, velx, k, j, f, nx, ny, nx);
            }
            th += -(fx2[1] - fx2[0]) * dx_inv;
        }
        if (open_y && (j == 0 || j == ny - 1)) {
            if (spec) {
            } else if (j == 0) {
                Exact32 rvw0 = fnm[k] * rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(0) * (nx) + (i))]
                          + fnp[k] * rv[((ulong)(k - 1) * (ny + 1) * (nx) + (ulong)(0) * (nx) + (i))];
                Exact32 rvw1 = fnm[k] * rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(1) * (nx) + (i))]
                          + fnp[k] * rv[((ulong)(k - 1) * (ny + 1) * (nx) + (ulong)(1) * (nx) + (i))];
                Exact32 vb = exact_min(0.5f * (rvw0 + rvw1), 0.0f);
                tb += -dy_inv * (vb * (w[((ulong)(k) * (ny) * (nx) + (ulong)(1) * (nx) + (i))] - w[((ulong)(k) * (ny) * (nx) + (ulong)(0) * (nx) + (i))])
                                 + w[((ulong)(k) * (ny) * (nx) + (ulong)(0) * (nx) + (i))] * (rvw1 - rvw0));
            } else {
                Exact32 rvw0 = fnm[k] * rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny - 1) * (nx) + (i))]
                          + fnp[k] * rv[((ulong)(k - 1) * (ny + 1) * (nx) + (ulong)(ny - 1) * (nx) + (i))];
                Exact32 rvw1 = fnm[k] * rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(ny) * (nx) + (i))]
                          + fnp[k] * rv[((ulong)(k - 1) * (ny + 1) * (nx) + (ulong)(ny) * (nx) + (i))];
                Exact32 vb = exact_max(0.5f * (rvw0 + rvw1), 0.0f);
                tb += -dy_inv * (vb * (w[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 1) * (nx) + (i))]
                                       - w[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 2) * (nx) + (i))])
                                 + w[((ulong)(k) * (ny) * (nx) + (ulong)(ny - 1) * (nx) + (i))] * (rvw1 - rvw0));
            }
        } else {
            Exact32 fy2[2];
            for (int s = 0; s < 2; ++s) {
                int g = j + s;
                Exact32 vely = fnm[k] * rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(g) * (nx) + (i))]
                          + fnp[k] * rv[((ulong)(k - 1) * (ny + 1) * (nx) + (ulong)(g) * (nx) + (i))];
                fy2[s] = open_y
                    ? yface_cell_open(w, vely, k, g, i, ny, ny, nx)
                    : yface_cell_per(w, vely, k, g, i, ny, ny, nx);
            }
            th += -(fy2[1] - fy2[0]) * dy_inv;
        }
        Exact32 fz2[2];
        for (int s = 0; s < 2; ++s) {
            int m = k - 1 + s;
            Exact32 velz = 0.5f * (rw[((ulong)(m) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] + rw[((ulong)(m + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]);
            if (m == 0 || m == nz - 1)
                fz2[s] = flux2(w[((ulong)(m) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], velz);
            else if (vorder == 5 && m >= 2 && m <= nz - 3)
                fz2[s] = flux5v(w[((ulong)(m - 2) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m - 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                                w[((ulong)(m) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                                w[((ulong)(m + 2) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m + 3) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                                velz);
            else
                fz2[s] = flux3(w[((ulong)(m - 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                               w[((ulong)(m + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m + 2) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                               velz);
        }
        Exact32 t = msf[(ulong)j * nx + i] * th + tb;
        tend_out[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] += t - (fz2[1] - fz2[0]) * rdn[k];
        return;
    }
    Exact32 fx[2], fy[2], fz[2];
    for (int s = 0; s < 2; ++s) {
        int f = i + s;
        Exact32 velx = fnm[k] * ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (f))]
                  + fnp[k] * ru[((ulong)(k - 1) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (f))];
        fx[s] = flux5(w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (((((f - 3) % (nx)) + (nx)) % (nx))))],
                      w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (((((f - 2) % (nx)) + (nx)) % (nx))))],
                      w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (((((f - 1) % (nx)) + (nx)) % (nx))))],
                      w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (((((f) % (nx)) + (nx)) % (nx))))],
                      w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (((((f + 1) % (nx)) + (nx)) % (nx))))],
                      w[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (((((f + 2) % (nx)) + (nx)) % (nx))))],
                      velx);
        int g = j + s;
        Exact32 vely = fnm[k] * rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(g) * (nx) + (i))]
                  + fnp[k] * rv[((ulong)(k - 1) * (ny + 1) * (nx) + (ulong)(g) * (nx) + (i))];
        fy[s] = flux5(w[((ulong)(k) * (ny) * (nx) + (ulong)(((((g - 3) % (ny)) + (ny)) % (ny))) * (nx) + (i))],
                      w[((ulong)(k) * (ny) * (nx) + (ulong)(((((g - 2) % (ny)) + (ny)) % (ny))) * (nx) + (i))],
                      w[((ulong)(k) * (ny) * (nx) + (ulong)(((((g - 1) % (ny)) + (ny)) % (ny))) * (nx) + (i))],
                      w[((ulong)(k) * (ny) * (nx) + (ulong)(((((g) % (ny)) + (ny)) % (ny))) * (nx) + (i))],
                      w[((ulong)(k) * (ny) * (nx) + (ulong)(((((g + 1) % (ny)) + (ny)) % (ny))) * (nx) + (i))],
                      w[((ulong)(k) * (ny) * (nx) + (ulong)(((((g + 2) % (ny)) + (ny)) % (ny))) * (nx) + (i))],
                      vely);
        int m = k - 1 + s;
        Exact32 velz = 0.5f * (rw[((ulong)(m) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] + rw[((ulong)(m + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]);
        if (m == 0 || m == nz - 1)
            fz[s] = flux2(w[((ulong)(m) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], velz);
        else if (vorder == 5 && m >= 2 && m <= nz - 3)
            fz[s] = flux5v(w[((ulong)(m - 2) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m - 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                            w[((ulong)(m) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                            w[((ulong)(m + 2) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m + 3) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], velz);
        else
            fz[s] = flux3(w[((ulong)(m - 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                          w[((ulong)(m + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], w[((ulong)(m + 2) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], velz);
    }
    if (has_msf) {
        tend_out[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] += msf[(ulong)j * nx + i]
                                   * (-(fx[1] - fx[0]) * dx_inv
                                      - (fy[1] - fy[0]) * dy_inv)
                                   - (fz[1] - fz[0]) * rdn[k];
    } else {
        tend_out[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] += -(fx[1] - fx[0]) * dx_inv
                                   - (fy[1] - fy[0]) * dy_inv
                                   - (fz[1] - fz[0]) * rdn[k];
    }
}
}
kernel void wrf_exact_advection_off_flux_div_scalar(
    constant WrfExactAdvectionOffFluxDivScalarResources &r [[buffer(0)]],
    constant WrfExactAdvectionOffFluxDivScalarParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= ulong(p.nz) * ulong(p.ny) * ulong(p.nx)) return;
    const int i = int(gid % uint(p.nx));
    const int j = int((gid / uint(p.nx)) % uint(p.ny));
    const int k = int(gid / (uint(p.nx) * uint(p.ny)));
    wrf_exact_advection_off::flux_div_scalar(reinterpret_cast<device const Exact32*>(r.q), reinterpret_cast<device const Exact32*>(r.ru), reinterpret_cast<device const Exact32*>(r.rv), reinterpret_cast<device const Exact32*>(r.rw), reinterpret_cast<device Exact32*>(r.tend), reinterpret_cast<device const Exact32*>(r.rdnw), reinterpret_cast<device const Exact32*>(r.fnm), reinterpret_cast<device const Exact32*>(r.fnp), reinterpret_cast<device const Exact32*>(r.msf), Exact32(p.dx_inv), Exact32(p.dy_inv), p.nz, p.ny, p.nx, p.open_x, p.open_y, p.has_msf, p.spec, p.vorder, i, j, k);
}
kernel void wrf_exact_advection_off_flux_div_u(
    constant WrfExactAdvectionOffFluxDivUResources &r [[buffer(0)]],
    constant WrfExactAdvectionOffFluxDivUParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= ulong(p.nz) * ulong(p.ny) * ulong(p.nx + 1)) return;
    const int i = int(gid % uint(p.nx + 1));
    const int j = int((gid / uint(p.nx + 1)) % uint(p.ny));
    const int k = int(gid / (uint(p.nx + 1) * uint(p.ny)));
    wrf_exact_advection_off::flux_div_u(reinterpret_cast<device const Exact32*>(r.u), reinterpret_cast<device const Exact32*>(r.ru), reinterpret_cast<device const Exact32*>(r.rv), reinterpret_cast<device const Exact32*>(r.rw), reinterpret_cast<device Exact32*>(r.tend), reinterpret_cast<device const Exact32*>(r.rdnw), reinterpret_cast<device const Exact32*>(r.fnm), reinterpret_cast<device const Exact32*>(r.fnp), reinterpret_cast<device const Exact32*>(r.msf), Exact32(p.dx_inv), Exact32(p.dy_inv), p.nz, p.ny, p.nx, p.open_x, p.open_y, p.has_msf, p.spec, p.vorder, i, j, k);
}
kernel void wrf_exact_advection_off_flux_div_v(
    constant WrfExactAdvectionOffFluxDivVResources &r [[buffer(0)]],
    constant WrfExactAdvectionOffFluxDivVParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= ulong(p.nz) * ulong(p.ny + 1) * ulong(p.nx)) return;
    const int i = int(gid % uint(p.nx));
    const int j = int((gid / uint(p.nx)) % uint(p.ny + 1));
    const int k = int(gid / (uint(p.nx) * uint(p.ny + 1)));
    wrf_exact_advection_off::flux_div_v(reinterpret_cast<device const Exact32*>(r.v), reinterpret_cast<device const Exact32*>(r.ru), reinterpret_cast<device const Exact32*>(r.rv), reinterpret_cast<device const Exact32*>(r.rw), reinterpret_cast<device Exact32*>(r.tend), reinterpret_cast<device const Exact32*>(r.rdnw), reinterpret_cast<device const Exact32*>(r.fnm), reinterpret_cast<device const Exact32*>(r.fnp), reinterpret_cast<device const Exact32*>(r.msf), Exact32(p.dx_inv), Exact32(p.dy_inv), p.nz, p.ny, p.nx, p.open_x, p.open_y, p.has_msf, p.spec, p.vorder, i, j, k);
}
kernel void wrf_exact_advection_off_flux_div_w(
    constant WrfExactAdvectionOffFluxDivWResources &r [[buffer(0)]],
    constant WrfExactAdvectionOffFluxDivWParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (ulong(gid) >= ulong(p.nz + 1) * ulong(p.ny) * ulong(p.nx)) return;
    const int i = int(gid % uint(p.nx));
    const int j = int((gid / uint(p.nx)) % uint(p.ny));
    const int k = int(gid / (uint(p.nx) * uint(p.ny)));
    wrf_exact_advection_off::flux_div_w(reinterpret_cast<device const Exact32*>(r.w), reinterpret_cast<device const Exact32*>(r.ru), reinterpret_cast<device const Exact32*>(r.rv), reinterpret_cast<device const Exact32*>(r.rw), reinterpret_cast<device Exact32*>(r.tend), reinterpret_cast<device const Exact32*>(r.rdn), reinterpret_cast<device const Exact32*>(r.fnm), reinterpret_cast<device const Exact32*>(r.fnp), reinterpret_cast<device const Exact32*>(r.msf), Exact32(p.dx_inv), Exact32(p.dy_inv), p.nz, p.ny, p.nx, p.open_x, p.open_y, p.has_msf, p.spec, p.vorder, i, j, k);
}
