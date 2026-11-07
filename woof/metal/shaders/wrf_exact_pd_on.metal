// Generated from the selected CUDA PD source; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "soft_binary64.metalh"
#include "wrf_exact_pd_on_bindings.metalh"
using namespace metal;
namespace wrf_exact_pd_on {
static inline bool operator<(Exact32 a,Exact32 b){return soft32_less(a.bits,b.bits);}
static inline Exact32 exact_copysign(Exact32 a,Exact32 b){return Exact32::from_bits((a.bits&0x7fffffffu)|(b.bits&0x80000000u));}
static inline int pd_shift(Exact32 cr){Soft64I32Result v=soft64_to_i32_rz(soft64_from_f32_bits(cr.bits));return v.valid?v.value:as_type<int>(0x80000000u);}
static inline Exact32 pd_double_ratio(Exact32 numerator,Exact32 cr){
 const ulong a=soft64_from_f32_bits(numerator.bits), b=soft64_from_f32_bits(soft32_abs_bits(cr.bits)), eps=0x3cd203af9ee75616ul;
 return Exact32::from_bits(soft64_to_f32_bits(soft64_div(a,soft64_greater(b,eps)?b:eps)));
}

int pd_periodic(int index, int extent)
{
    return ((((index) % (extent)) + (extent)) % (extent));
}

Exact32 pd_flux5(Exact32 qm3, Exact32 qm2, Exact32 qm1, Exact32 q0, Exact32 qp1, Exact32 qp2,
              Exact32 vel)
{
    Exact32 center = 0.6166666746139526f * (q0 + qm1)
                - 0.13333334028720856f * (qp1 + qm2)
                + 0.01666666753590107f * (qp2 + qm3);
    Exact32 dissipation = exact_copysign(1.0f, vel) * 0.01666666753590107f
        * ((qp2 - qm3) - 5.0f * (qp1 - qm2) + 10.0f * (q0 - qm1));
    return vel * (center - dissipation);
}

Exact32 pd_flux3(Exact32 qm2, Exact32 qm1, Exact32 q0, Exact32 qp1, Exact32 vel)
{
    Exact32 center = 0.5833333134651184f * (q0 + qm1)
                - 0.0833333358168602f * (qp1 + qm2);
    Exact32 dissipation = exact_copysign(1.0f, vel) * 0.0833333358168602f
        * ((qp1 - qm2) - 3.0f * (q0 - qm1));
    return vel * (center - dissipation);
}

Exact32 pd_flux5v(Exact32 qm3, Exact32 qm2, Exact32 qm1, Exact32 q0, Exact32 qp1, Exact32 qp2,
               Exact32 vel)
{
    Exact32 center = 0.6166666746139526f * (q0 + qm1)
                - 0.13333334028720856f * (qp1 + qm2)
                + 0.01666666753590107f * (qp2 + qm3);
    Exact32 dissipation = exact_copysign(1.0f, vel) * 0.01666666753590107f
        * ((qp2 - qm3) - 5.0f * (qp1 - qm2) + 10.0f * (q0 - qm1));
    return vel * (center + dissipation);
}

Exact32 pd_flux3h(Exact32 qm2, Exact32 qm1, Exact32 q0, Exact32 qp1, Exact32 vel)
{
    Exact32 center = 0.5833333134651184f * (q0 + qm1)
                - 0.0833333358168602f * (qp1 + qm2);
    Exact32 dissipation = exact_copysign(1.0f, vel) * 0.0833333358168602f
        * ((qp1 - qm2) - 3.0f * (q0 - qm1));
    return vel * (center + dissipation);
}

Exact32 pd_zface_half(device const Exact32* q, Exact32 vel, int kf, int j, int i,
                   int nz, int ny, int nxs,
                   device const Exact32* fnm, device const Exact32* fnp, int vorder)
{
    if (kf == 0 || kf == nz) return 0.0f;
    if (kf == 1 || kf == nz - 1)
        return vel * (fnm[kf] * q[((ulong)(kf) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))]
                      + fnp[kf] * q[((ulong)(kf - 1) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))]);
    if (vorder == 5 && kf >= 3 && kf <= nz - 3)
        return pd_flux5v(q[((ulong)(kf - 3) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                         q[((ulong)(kf - 2) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                         q[((ulong)(kf - 1) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                         q[((ulong)(kf) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                         q[((ulong)(kf + 1) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                         q[((ulong)(kf + 2) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))], vel);
    return pd_flux3(q[((ulong)(kf - 2) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                    q[((ulong)(kf - 1) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                    q[((ulong)(kf) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))],
                    q[((ulong)(kf + 1) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))], vel);
}

Exact32 flux_upwind(Exact32 qm1, Exact32 q0v, Exact32 cr)
{
    return 0.5f * exact_min(1.0f, cr + exact_abs(cr)) * qm1
         + 0.5f * exact_max(-1.0f, cr - exact_abs(cr)) * q0v;
}
void pd_fluxes(device const Exact32*  q,
               device const Exact32*  q0,
               device const Exact32*  ru,
               device const Exact32*  rv,
               device const Exact32*  rw,
               device const Exact32*  mut,
               device const Exact32*  c1h,
               device const Exact32*  c2h,
               device const Exact32*  rdnw,
               device const Exact32*  fnm,
               device const Exact32*  fnp,
               device const Exact32*  msft,
               Exact32 dx, Exact32 dy, Exact32 dt,
               device Exact32*  fxl,
               device Exact32*  fxc,
               device Exact32*  fyl,
               device Exact32*  fyc,
               device Exact32*  fzl,
               device Exact32*  fzc,
               int nz, int ny, int nx, int has_msf,
               int open_x, int open_y, int vorder, int i, int j, int k)
{

    if (i > nx || j > ny || k > nz) return;
    if (j < ny && k < nz) {
        if (open_x && (i == 0 || i == nx)) {
            fxl[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))] = 0.0f;
            fxc[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))] = 0.0f;
        } else if (open_x) {
            Exact32 vel = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))];
Exact32 muf=0.5f*((c1h[k]*mut[(ulong)j * nx + i]+c2h[k])+(c1h[k]*mut[(ulong)j * nx + i - 1]+c2h[k]));
            Exact32 dxp = dx;
            if (has_msf)
dxp=exact_div(exact_div(2.0f,msft[(ulong)j * nx + i]+msft[(ulong)j * nx + i - 1]),exact_div(1.0f,dx));
            Exact32 cr = vel * dt / dxp / muf;
            if (i == 1) cr = exact_div(exact_div(vel, muf) * dt, dxp);
            Exact32 fl = muf * (dxp / dt)
                    * flux_upwind(q0[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i - 1))],
                                  q0[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], cr);
            int d = min(i, nx - i);
            Exact32 fh;
            if (d == 1)
                fh = 0.5f * vel * (q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] + q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i - 1))]);
            else if (d == 2)
                fh = pd_flux3h(q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i - 2))], q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i - 1))],
                               q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i + 1))], vel);
            else
                fh = pd_flux5(q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i - 3))], q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i - 2))],
                              q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i - 1))], q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                              q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i + 1))], q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i + 2))],
                              vel);
            fxl[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))] = fl;
            fxc[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))] = fh - fl;
        } else {
            Exact32 vel = ru[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))];
Exact32 muf=0.5f*((c1h[k]*mut[(ulong)j * nx + pd_periodic(i, nx)]+c2h[k])+(c1h[k]*mut[(ulong)j * nx + pd_periodic(i - 1, nx)]+c2h[k]));
            Exact32 dxp = dx;
            if (has_msf)
dxp=exact_div(exact_div(2.0f,msft[(ulong)j * nx + pd_periodic(i, nx)]+msft[(ulong)j * nx + pd_periodic(i - 1, nx)]),exact_div(1.0f,dx));
            Exact32 cr = vel * dt / dxp / muf;
            Exact32 fl = muf * (dxp / dt)
                    * flux_upwind(q0[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (pd_periodic(i - 1, nx)))],
                                  q0[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (pd_periodic(i, nx)))], cr);
            Exact32 fh = pd_flux5(q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (pd_periodic(i - 3, nx)))],
                               q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (pd_periodic(i - 2, nx)))],
                               q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (pd_periodic(i - 1, nx)))],
                               q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (pd_periodic(i, nx)))],
                               q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (pd_periodic(i + 1, nx)))],
                               q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (pd_periodic(i + 2, nx)))], vel);
            fxl[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))] = fl;
            fxc[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))] = fh - fl;
        }
    }
    if (i < nx && k < nz) {
        if (open_y && (j == 0 || j == ny)) {
            fyl[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))] = 0.0f;
            fyc[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))] = 0.0f;
        } else if (open_y) {
            Exact32 vel = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))];
Exact32 muf=0.5f*((c1h[k]*mut[(ulong)j * nx + i]+c2h[k])+(c1h[k]*mut[(ulong)(j - 1) * nx + i]+c2h[k]));
            Exact32 dyp = dy;
            if (has_msf)
dyp=exact_div(exact_div(2.0f,msft[(ulong)j * nx + i]+msft[(ulong)(j - 1) * nx + i]),exact_div(1.0f,dy));
            Exact32 cr = vel * dt / dyp / muf;
            Exact32 fl = muf * (dyp / dt)
                    * flux_upwind(q0[((ulong)(k) * (ny) * (nx) + (ulong)(j - 1) * (nx) + (i))],
                                  q0[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], cr);
            int d = min(j, ny - j);
            Exact32 fh;
            if (d == 1)
                fh = 0.5f * vel * (q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] + q[((ulong)(k) * (ny) * (nx) + (ulong)(j - 1) * (nx) + (i))]);
            else if (d == 2)
                fh = pd_flux3h(q[((ulong)(k) * (ny) * (nx) + (ulong)(j - 2) * (nx) + (i))], q[((ulong)(k) * (ny) * (nx) + (ulong)(j - 1) * (nx) + (i))],
                               q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], q[((ulong)(k) * (ny) * (nx) + (ulong)(j + 1) * (nx) + (i))], vel);
            else
                fh = pd_flux5(q[((ulong)(k) * (ny) * (nx) + (ulong)(j - 3) * (nx) + (i))], q[((ulong)(k) * (ny) * (nx) + (ulong)(j - 2) * (nx) + (i))],
                              q[((ulong)(k) * (ny) * (nx) + (ulong)(j - 1) * (nx) + (i))], q[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))],
                              q[((ulong)(k) * (ny) * (nx) + (ulong)(j + 1) * (nx) + (i))], q[((ulong)(k) * (ny) * (nx) + (ulong)(j + 2) * (nx) + (i))],
                              vel);
            fyl[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))] = fl;
            fyc[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))] = fh - fl;
        } else {
            Exact32 vel = rv[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))];
Exact32 muf=0.5f*((c1h[k]*mut[(ulong)pd_periodic(j, ny) * nx + i]+c2h[k])+(c1h[k]*mut[(ulong)pd_periodic(j - 1, ny) * nx + i]+c2h[k]));
            Exact32 dyp = dy;
            if (has_msf)
dyp=exact_div(exact_div(2.0f,msft[(ulong)pd_periodic(j, ny) * nx + i]+msft[(ulong)pd_periodic(j - 1, ny) * nx + i]),exact_div(1.0f,dy));
            Exact32 cr = vel * dt / dyp / muf;
            Exact32 fl = muf * (dyp / dt)
                    * flux_upwind(q0[((ulong)(k) * (ny) * (nx) + (ulong)(pd_periodic(j - 1, ny)) * (nx) + (i))],
                                  q0[((ulong)(k) * (ny) * (nx) + (ulong)(pd_periodic(j, ny)) * (nx) + (i))], cr);
            Exact32 fh = pd_flux5(q[((ulong)(k) * (ny) * (nx) + (ulong)(pd_periodic(j - 3, ny)) * (nx) + (i))],
                               q[((ulong)(k) * (ny) * (nx) + (ulong)(pd_periodic(j - 2, ny)) * (nx) + (i))],
                               q[((ulong)(k) * (ny) * (nx) + (ulong)(pd_periodic(j - 1, ny)) * (nx) + (i))],
                               q[((ulong)(k) * (ny) * (nx) + (ulong)(pd_periodic(j, ny)) * (nx) + (i))],
                               q[((ulong)(k) * (ny) * (nx) + (ulong)(pd_periodic(j + 1, ny)) * (nx) + (i))],
                               q[((ulong)(k) * (ny) * (nx) + (ulong)(pd_periodic(j + 2, ny)) * (nx) + (i))], vel);
            fyl[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))] = fl;
            fyc[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))] = fh - fl;
        }
    }
    if (i < nx && j < ny) {
        Exact32 fl = 0.0f, fc = 0.0f;
        if (k > 0 && k < nz) {
            Exact32 vel = rw[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))];
            Exact32 dz = 2.0f / (rdnw[k] + rdnw[k - 1]);
            Exact32 muf = c1h[k] * mut[(ulong)j * nx + i] + c2h[k];
            Exact32 cr = exact_div(exact_div(vel * dt, dz), muf);
            fl = muf * (dz / dt)
               * flux_upwind(q0[((ulong)(k - 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], q0[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))], cr);
            fc = pd_zface_half(q, vel, k, j, i, nz, ny, nx, fnm, fnp, vorder) - fl;
        }
        fzl[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] = fl;
        fzc[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] = fc;
    }
}

Exact32 pd_scale(int k, int j, int i,
              device const Exact32* q0, device const Exact32* mu_old,
              device const Exact32* c1h, device const Exact32* c2h, device const Exact32* rdnw,
              device const Exact32* fxl, device const Exact32* fxc,
              device const Exact32* fyl, device const Exact32* fyc,
              device const Exact32* fzl, device const Exact32* fzc,
              device const Exact32* msft,
              Exact32 dx_inv, Exact32 dy_inv, Exact32 dt, int ny, int nx,
              int has_msf, int open_x, int open_y
              ,device const Exact32* mub, device const Exact32* mup_old
              )
{
    if ((open_x && (i == 0 || i == nx - 1))
        || (open_y && (j == 0 || j == ny - 1)))
        return 1.0f;
    Exact32 chm0=(c1h[k]*mub[(ulong)j*nx+i]+c2h[k])+(c1h[k]*mup_old[(ulong)j*nx+i]);
    Exact32 ph_low, fo;
    if (has_msf) {
        Exact32 m = msft[(ulong)j * nx + i];
        Exact32 m2 = m * m;
        ph_low = chm0 * q0[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]
               - dt * (m2 * (dx_inv * (fxl[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i + 1))]
                                       - fxl[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))])
                             + dy_inv * (fyl[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j + 1) * (nx) + (i))]
                                         - fyl[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))]))
                       + m * rdnw[k] * (fzl[((ulong)(k + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]
                                        - fzl[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]));
        fo = dt * (m2 * (dx_inv * (exact_max(0.0f, fxc[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i + 1))])
                                   - exact_min(0.0f, fxc[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))]))
                         + dy_inv * (exact_max(0.0f, fyc[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j + 1) * (nx) + (i))])
                                     - exact_min(0.0f, fyc[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))])))
                   + m * rdnw[k] * (exact_min(0.0f, fzc[((ulong)(k + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))])
                                    - exact_max(0.0f, fzc[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))])));
    } else {
        Exact32 dl = dx_inv * (fxl[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i + 1))]
                            - fxl[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))])
                + dy_inv * (fyl[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j + 1) * (nx) + (i))]
                            - fyl[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))])
                + rdnw[k] * (fzl[((ulong)(k + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] - fzl[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]);
        ph_low = chm0 * q0[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] - dt * dl;
        fo = dt * (dx_inv * (exact_max(0.0f, fxc[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i + 1))])
                             - exact_min(0.0f, fxc[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))]))
                   + dy_inv * (exact_max(0.0f, fyc[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j + 1) * (nx) + (i))])
                               - exact_min(0.0f, fyc[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))]))
                   + rdnw[k] * (exact_min(0.0f, fzc[((ulong)(k + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))])
                                - exact_max(0.0f, fzc[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))])));
    }
    if (fo > ph_low)
        return exact_max(0.0f, exact_div(ph_low,fo+1e-20f));
    return 1.0f;
}
void pd_renorm_apply(device const Exact32*  q0,
                     device const Exact32*  mu_old,
                     device const Exact32*  fxl,
                     device const Exact32*  fxc,
                     device const Exact32*  fyl,
                     device const Exact32*  fyc,
                     device const Exact32*  fzl,
                     device const Exact32*  fzc,
                     device const Exact32*  c1h,
                     device const Exact32*  c2h,
                     device const Exact32*  rdnw,
                     device const Exact32*  msft,
                     Exact32 dx_inv, Exact32 dy_inv, Exact32 dt,
                     device Exact32*  tend_out,
                     int nz, int ny, int nx, int has_msf,
                     int open_x, int open_y
                     ,device const Exact32*  mub, device const Exact32*  mup_old
                     , int i, int j, int k)
{

    if (i >= nx || j >= ny || k >= nz) return;
    Exact32 sc = pd_scale((k), (j), (i), q0, mu_old, c1h, c2h, rdnw, fxl, fxc, fyl, fyc, fzl, fzc, msft, dx_inv, dy_inv, dt, ny, nx, has_msf, open_x, open_y,mub,mup_old);
    Exact32 fxc_l = fxc[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i))];
    Exact32 sx_l = (fxc_l > 0.0f) ? pd_scale((k), (j), (pd_periodic(i - 1, nx)), q0, mu_old, c1h, c2h, rdnw, fxl, fxc, fyl, fyc, fzl, fzc, msft, dx_inv, dy_inv, dt, ny, nx, has_msf, open_x, open_y,mub,mup_old)
              : ((fxc_l < 0.0f) ? sc : 1.0f);
    Exact32 fxc_r = fxc[((ulong)(k) * (ny) * (nx + 1) + (ulong)(j) * (nx + 1) + (i + 1))];
    Exact32 sx_r = (fxc_r > 0.0f) ? sc
              : ((fxc_r < 0.0f) ? pd_scale((k), (j), (pd_periodic(i + 1, nx)), q0, mu_old, c1h, c2h, rdnw, fxl, fxc, fyl, fyc, fzl, fzc, msft, dx_inv, dy_inv, dt, ny, nx, has_msf, open_x, open_y,mub,mup_old) : 1.0f);
    Exact32 fyc_l = fyc[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j) * (nx) + (i))];
    Exact32 sy_l = (fyc_l > 0.0f) ? pd_scale((k), (pd_periodic(j - 1, ny)), (i), q0, mu_old, c1h, c2h, rdnw, fxl, fxc, fyl, fyc, fzl, fzc, msft, dx_inv, dy_inv, dt, ny, nx, has_msf, open_x, open_y,mub,mup_old)
              : ((fyc_l < 0.0f) ? sc : 1.0f);
    Exact32 fyc_r = fyc[((ulong)(k) * (ny + 1) * (nx) + (ulong)(j + 1) * (nx) + (i))];
    Exact32 sy_r = (fyc_r > 0.0f) ? sc
              : ((fyc_r < 0.0f) ? pd_scale((k), (pd_periodic(j + 1, ny)), (i), q0, mu_old, c1h, c2h, rdnw, fxl, fxc, fyl, fyc, fzl, fzc, msft, dx_inv, dy_inv, dt, ny, nx, has_msf, open_x, open_y,mub,mup_old) : 1.0f);
    Exact32 fzc_b = fzc[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))];
    Exact32 sz_b = (fzc_b > 0.0f) ? sc
              : ((fzc_b < 0.0f) ? pd_scale((k - 1), (j), (i), q0, mu_old, c1h, c2h, rdnw, fxl, fxc, fyl, fyc, fzl, fzc, msft, dx_inv, dy_inv, dt, ny, nx, has_msf, open_x, open_y,mub,mup_old) : 1.0f);
    Exact32 fzc_t = fzc[((ulong)(k + 1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))];
    Exact32 sz_t = (fzc_t < 0.0f) ? sc
              : ((fzc_t > 0.0f) ? pd_scale((k + 1), (j), (i), q0, mu_old, c1h, c2h, rdnw, fxl, fxc, fyl, fyc, fzl, fzc, msft, dx_inv, dy_inv, dt, ny, nx, has_msf, open_x, open_y,mub,mup_old) : 1.0f);
    Exact32 result=tend_out[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))];
    Exact32 zbracket=sz_t*fzc_t-sz_b*fzc_b+fzl[((ulong)(k+1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]-fzl[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))];
    result=result-rdnw[k]*zbracket;
    Exact32 m=has_msf?msft[(ulong)j*nx+i]:1.0f;
    if(!open_x || (i>=1 && i<=nx-2)) {
        Exact32 xbracket=sx_r*fxc_r-sx_l*fxc_l+fxl[((ulong)(k) * (ny) * (nx+1) + (ulong)(j) * (nx+1) + (i+1))]-fxl[((ulong)(k) * (ny) * (nx+1) + (ulong)(j) * (nx+1) + (i))];
        result=result-m*(dx_inv*xbracket);
    }
    if(!open_y || (j>=1 && j<=ny-2)) {
        Exact32 ybracket=sy_r*fyc_r-sy_l*fyc_l+fyl[((ulong)(k) * (ny+1) * (nx) + (ulong)(j+1) * (nx) + (i))]-fyl[((ulong)(k) * (ny+1) * (nx) + (ulong)(j) * (nx) + (i))];
        result=result-m*(dy_inv*ybracket);
    }
    tend_out[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]=result;
}
void pd_vertical_sl(device const Exact32*  q,
                    device const Exact32*  q0,
                    device const Exact32*  ru,
                    device const Exact32*  rv,
                    device const Exact32*  rw,
                    device const Exact32*  mut,
                    device const Exact32*  c1h,
                    device const Exact32*  c2h,
                    device const Exact32*  rdnw,
                    device const Exact32*  fnm,
                    device const Exact32*  fnp,
                    device const Exact32*  msft,
                    Exact32 dx, Exact32 dy, Exact32 dt,
                    device Exact32*  fxl, device Exact32*  fxc,
                    device Exact32*  fyl, device Exact32*  fyc,
                    device Exact32*  fzl, device Exact32*  fzc,
                    int nz, int ny, int nx, int has_msf,
                    int open_x, int open_y, int vorder, int i, int j, int k)
{

    if (i >= nx || j >= ny || k > nz) return;
    Exact32 fl = 0.0f, fc = 0.0f;
    if (k > 0 && k < nz) {
        Exact32 vel = rw[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))];
        bool multi = false;
        fl = exact_max(vel,0.0f)*q0[((ulong)(k-1) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]
           + exact_min(vel,0.0f)*q0[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))];
        if (k > 1 && k < nz-1) {
            Exact32 dz = exact_div(2.0f,rdnw[k]+rdnw[k-1]);
            Exact32 mu = c1h[k]*mut[(ulong)j*nx+i]+c2h[k];
            Exact32 cr = exact_div(exact_div(vel*dt,dz),mu);
            if (cr > 1.0f || cr < -1.0f) {
                int shift = pd_shift(cr);
                int low = k == nz-2 ? -2 : -3;
                int high = k == 2 ? 2 : 3;
                shift = min(max(shift,low),high);
                Exact32 sum = 0.0f;
                int first = cr > 1.0f ? k-shift : k;
                int last = cr > 1.0f ? k-1 : k-shift-1;
                for (int level = first; level <= last; ++level)
                    sum = sum + q0[((ulong)(level) * (ny) * (nx) + (ulong)(j) * (nx) + (i))];
                fl = pd_double_ratio(vel*sum,cr);
                multi = true;
            }
        }
        fc = pd_zface_half(q,vel,k,j,i,nz,ny,nx,fnm,fnp,5)-fl;
    }
    fzl[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] = fl;
    fzc[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] = fc;
}
}
kernel void wrf_exact_pd_on_pd_fluxes(constant WrfExactPdOnPdFluxesResources &r [[buffer(0)]],constant WrfExactPdOnPdFluxesParams &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if (ulong(gid)>=ulong((p.nx+1))*ulong((p.ny+1))*ulong((p.nz+1))) return;
 int i=int(gid%uint((p.nx+1))),j=int((gid/uint((p.nx+1)))%uint((p.ny+1))),k=int(gid/(uint((p.nx+1))*uint((p.ny+1))));
 wrf_exact_pd_on::pd_fluxes(reinterpret_cast<device const Exact32*>(r.q), reinterpret_cast<device const Exact32*>(r.q0), reinterpret_cast<device const Exact32*>(r.ru), reinterpret_cast<device const Exact32*>(r.rv), reinterpret_cast<device const Exact32*>(r.rw), reinterpret_cast<device const Exact32*>(r.mut), reinterpret_cast<device const Exact32*>(r.c1h), reinterpret_cast<device const Exact32*>(r.c2h), reinterpret_cast<device const Exact32*>(r.rdnw), reinterpret_cast<device const Exact32*>(r.fnm), reinterpret_cast<device const Exact32*>(r.fnp), reinterpret_cast<device const Exact32*>(r.msft), Exact32(p.dx), Exact32(p.dy), Exact32(p.dt), reinterpret_cast<device Exact32*>(r.fxl), reinterpret_cast<device Exact32*>(r.fxc), reinterpret_cast<device Exact32*>(r.fyl), reinterpret_cast<device Exact32*>(r.fyc), reinterpret_cast<device Exact32*>(r.fzl), reinterpret_cast<device Exact32*>(r.fzc), p.nz, p.ny, p.nx, p.has_msf, p.open_x, p.open_y, p.vorder,i,j,k);
}
kernel void wrf_exact_pd_on_pd_renorm_apply(constant WrfExactPdOnPdRenormApplyResources &r [[buffer(0)]],constant WrfExactPdOnPdRenormApplyParams &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if (ulong(gid)>=ulong(p.nx)*ulong(p.ny)*ulong(p.nz)) return;
 int i=int(gid%uint(p.nx)),j=int((gid/uint(p.nx))%uint(p.ny)),k=int(gid/(uint(p.nx)*uint(p.ny)));
 wrf_exact_pd_on::pd_renorm_apply(reinterpret_cast<device const Exact32*>(r.q0), reinterpret_cast<device const Exact32*>(r.mu_old), reinterpret_cast<device const Exact32*>(r.fxl), reinterpret_cast<device const Exact32*>(r.fxc), reinterpret_cast<device const Exact32*>(r.fyl), reinterpret_cast<device const Exact32*>(r.fyc), reinterpret_cast<device const Exact32*>(r.fzl), reinterpret_cast<device const Exact32*>(r.fzc), reinterpret_cast<device const Exact32*>(r.c1h), reinterpret_cast<device const Exact32*>(r.c2h), reinterpret_cast<device const Exact32*>(r.rdnw), reinterpret_cast<device const Exact32*>(r.msft), Exact32(p.dx_inv), Exact32(p.dy_inv), Exact32(p.dt), reinterpret_cast<device Exact32*>(r.tend), p.nz, p.ny, p.nx, p.has_msf, p.open_x, p.open_y, reinterpret_cast<device const Exact32*>(r.mub), reinterpret_cast<device const Exact32*>(r.mup_old),i,j,k);
}
kernel void wrf_exact_pd_on_pd_vertical_sl(constant WrfExactPdOnPdVerticalSlResources &r [[buffer(0)]],constant WrfExactPdOnPdVerticalSlParams &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if (ulong(gid)>=ulong((p.nx+1))*ulong((p.ny+1))*ulong((p.nz+1))) return;
 int i=int(gid%uint((p.nx+1))),j=int((gid/uint((p.nx+1)))%uint((p.ny+1))),k=int(gid/(uint((p.nx+1))*uint((p.ny+1))));
 wrf_exact_pd_on::pd_vertical_sl(reinterpret_cast<device const Exact32*>(r.q), reinterpret_cast<device const Exact32*>(r.q0), reinterpret_cast<device const Exact32*>(r.ru), reinterpret_cast<device const Exact32*>(r.rv), reinterpret_cast<device const Exact32*>(r.rw), reinterpret_cast<device const Exact32*>(r.mut), reinterpret_cast<device const Exact32*>(r.c1h), reinterpret_cast<device const Exact32*>(r.c2h), reinterpret_cast<device const Exact32*>(r.rdnw), reinterpret_cast<device const Exact32*>(r.fnm), reinterpret_cast<device const Exact32*>(r.fnp), reinterpret_cast<device const Exact32*>(r.msft), Exact32(p.dx), Exact32(p.dy), Exact32(p.dt), reinterpret_cast<device Exact32*>(r.fxl), reinterpret_cast<device Exact32*>(r.fxc), reinterpret_cast<device Exact32*>(r.fyl), reinterpret_cast<device Exact32*>(r.fyc), reinterpret_cast<device Exact32*>(r.fzl), reinterpret_cast<device Exact32*>(r.fzc), p.nz, p.ny, p.nx, p.has_msf, p.open_x, p.open_y, p.vorder,i,j,k);
}
