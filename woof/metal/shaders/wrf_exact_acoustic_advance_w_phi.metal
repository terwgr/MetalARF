#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "soft_glibc_solar_trig.metalh"
#include "wrf_exact_acoustic_advance_w_phi_bindings.metalh"
#include "wrf_exact_acoustic_boundary_frame_bindings.metalh"
using namespace metal;
#define WPHI_MAX_LEV 257
#define I3S(k,j,i,ny,nx) (((ulong)(k)*(ny)+(j))*(nx)+(i))
#define G Exact32(9.81f)
#define __fdiv_rn exact_div
#define arn_add exact_add
#define arn_sub exact_sub
#define arn_mul exact_mul
#define arn_div exact_div
static inline Exact32 glibc_sinf(Exact32 x) { return Exact32::from_bits(soft_glibc_solar_sinf(x.bits)); }
namespace wrf_wphi {
static inline
void diagnose_p_column(device Exact32*  p_pp,
                       device Exact32*  al_pp,
                       device const Exact32*  th_pp,
                       device const Exact32*  ph_pp,
                       device const Exact32*  mu_pp,
                       device const Exact32*  thp,
                       device const Exact32*  thb,
                       device const Exact32*  alt,
                       device const Exact32*  c2a,
                       device const Exact32*  mup,
                       device const Exact32*  rdnw,
                       device const Exact32*  c1h,
                       device const Exact32*  c2h,
                       device const Exact32*  mub2d,
                       ulong c, ulong st, int base3d, int nz)
{
    for (int k = 0; k < nz; ++k) {
        ulong tid = (ulong)k * st + c;
        Exact32 chm = c1h[k] * (mub2d[c] + mup[c] + mu_pp[c]) + c2h[k];
        Exact32 th_ref = thp[tid];
        Exact32 al = __fdiv_rn(Exact32(-1.0f), chm)
                * (alt[tid] * (c1h[k] * mu_pp[c])
                   + rdnw[k] * (ph_pp[tid + st] - ph_pp[tid]));
        al_pp[tid] = al;
        p_pp[tid] = c2a[tid]
                  * (__fdiv_rn(alt[tid]
                      * (th_pp[tid] - (c1h[k] * mu_pp[c]) * th_ref),
                      chm * (Exact32(300.0f) + th_ref)) - al);
    }
}
static inline
void diagnose_p_column_muts(device Exact32*  p_pp,
                            device Exact32*  al_pp,
                            device const Exact32*  th_pp,
                            device const Exact32*  ph_pp,
                            device const Exact32*  mu_pp,
                            device const Exact32*  thp,
                            device const Exact32*  alt,
                            device const Exact32*  c2a,
                            device const Exact32*  rdnw,
                            device const Exact32*  c1h,
                            device const Exact32*  c2h,
                            ulong c, ulong st, int nz, Exact32 muts)
{
    for (int k = 0; k < nz; ++k) {
        ulong tid = (ulong)k * st + c;
        Exact32 chm = c1h[k] * muts + c2h[k];
        Exact32 th_ref = thp[tid];
        Exact32 al = __fdiv_rn(Exact32(-1.0f), chm)
                * (alt[tid] * (c1h[k] * mu_pp[c])
                   + rdnw[k] * (ph_pp[tid + st] - ph_pp[tid]));
        al_pp[tid] = al;
        p_pp[tid] = c2a[tid]
                  * (__fdiv_rn(alt[tid]
                      * (th_pp[tid] - (c1h[k] * mu_pp[c]) * th_ref),
                      chm * (Exact32(300.0f) + th_ref)) - al);
    }
}
static inline
Exact32 wrf_theta_average(Exact32 current, Exact32 previous, Exact32 muave,
                       Exact32 c1, Exact32 c2, Exact32 muts, Exact32 theta, Exact32 epssm)
{
    Exact32 average = Exact32(0.5f)*((Exact32(1.0f)+epssm)*current+(Exact32(1.0f)-epssm)*previous);
    return __fdiv_rn(average+(c1*muave)*Exact32(300.0f),
                     (c1*muts+c2)*(Exact32(300.0f)+theta));
}
static inline
Exact32 wrf_full_theta_average(Exact32 current, Exact32 previous, Exact32 mu, Exact32 mu_old,
                            Exact32 muave, Exact32 c1, Exact32 c2, Exact32 muts,
                            Exact32 theta, Exact32 epssm)
{
    Exact32 current_wrf = current - (c1 * mu) * Exact32(300.0f);
    Exact32 previous_wrf = previous - (c1 * mu_old) * Exact32(300.0f);
    return wrf_theta_average(current_wrf, previous_wrf, muave,
                             c1, c2, muts, theta, epssm);
}

void advance_w_phi(device Exact32*  w_pp, device Exact32*  ph_pp,
                   device const Exact32*  rw_t,
                   device const Exact32*  rph_t,
                   device const Exact32*  ww_pp,
                   device const Exact32*  mu_pp,
                   device const Exact32*  mu_pp_old,
                   device const Exact32*  th_pp,
                   device const Exact32*  th_pp_old,
                   device const Exact32*  thp, device const Exact32*  thb,
                   device const Exact32*  php, device const Exact32*  phb,
                   device const Exact32*  alt, device const Exact32*  c2a,
                   device const Exact32*  a, device const Exact32*  alpha,
                   device const Exact32*  gam,
                   device const Exact32*  mup,
                   device const Exact32*  u_pp,
                   device const Exact32*  v_pp,
                   device const Exact32*  w_ref,
                   device const Exact32*  ht,
                   device const Exact32*  rdn, device const Exact32*  rdnw,
                   device const Exact32*  fnm, device const Exact32*  fnp,
                   device const Exact32*  c1h, device const Exact32*  c2h,
                   device const Exact32*  c1f, device const Exact32*  c2f,
                   device const Exact32*  mub2d,
                   device const Exact32*  cqw, int moist_cq,
                   device Exact32*  p_pp, device Exact32*  al_pp,
                   Exact32 cf1, Exact32 cf2, Exact32 cf3, Exact32 rdx, Exact32 rdy,
                   Exact32 dtau, Exact32 epssm,
                   Exact32 dampmag, Exact32 zdamp,
                   int boundary_x, int boundary_y,
                   int spec_zone, int base3d, int top_lid,
                   int nz, int ny, int nx, uint gid)
{
    int c = int(gid);
    if (c >= ny * nx || nz + 1 > 257) return;
    int j = c / nx, i = c - j * nx;
    if (i < spec_zone || i >= nx - spec_zone
        || j < spec_zone || j >= ny - spec_zone) return;
    ulong st = (ulong)ny * nx;
    ulong bstr = base3d ? st : 1;
    ulong boff = base3d ? (ulong)c : 0;
    Exact32 mut = mub2d[c] + mup[c];
    Exact32 muts = mut + mu_pp[c];
    Exact32 muave = Exact32(0.5f) * ((Exact32(1.0f) + epssm) * mu_pp[c]
                       + (Exact32(1.0f) - epssm) * mu_pp_old[c]);
    Exact32 rhs[257];
    rhs[0] = Exact32(0.0f);
    for (int k = 1; k <= nz; ++k) {
        ulong f = (ulong)k * st + c;
        rhs[k] = dtau * (rph_t[f]
                         + Exact32(0.5f) * G * (Exact32(1.0f) - epssm) * w_pp[f]);
    }
    Exact32 ph_lo = phb[boff] + php[c];
    Exact32 ph_hi = phb[bstr + boff] + php[(ulong)st + c];
    Exact32 wd_lo = Exact32(0.5f) * (ww_pp[(ulong)st + c] + ww_pp[c])
               * rdnw[0] * (php[st+c]-php[c]+phb[bstr+boff]-phb[boff]);
    for (int k = 1; k < nz; ++k) {
        ph_lo = ph_hi;
        ph_hi = phb[(ulong)(k + 1) * bstr + boff]
              + php[(ulong)(k + 1) * st + c];
        Exact32 wd_hi = Exact32(0.5f) * (ww_pp[(ulong)(k + 1) * st + c]
                           + ww_pp[(ulong)k * st + c])
                   * rdnw[k] * (php[(ulong)(k+1)*st+c]-php[(ulong)k*st+c]
                     +phb[(ulong)(k+1)*bstr+boff]-phb[(ulong)k*bstr+boff]);
        rhs[k] -= dtau * (fnm[k] * wd_hi + fnp[k] * wd_lo);
        wd_lo = wd_hi;
    }
    for (int k = 1; k <= nz; ++k)
        rhs[k] = ph_pp[(ulong)k * st + c]
               + rhs[k] / (c1f[k] * mut + c2f[k]);
    if (top_lid) rhs[nz] = Exact32(0.0f);
    {
        int ip = boundary_x ? min(i + 1, nx - 1) : (i + 1) % nx;
        int im = boundary_x ? max(i - 1, 0) : (i - 1 + nx) % nx;
        int jp = boundary_y ? min(j + 1, ny - 1) : (j + 1) % ny;
        int jm = boundary_y ? max(j - 1, 0) : (j - 1 + ny) % ny;
        int nxf = nx + 1;
        ulong cip = (ulong)j * nx + ip, cim = (ulong)j * nx + im;
        ulong cjp = (ulong)jp * nx + i, cjm = (ulong)jm * nx + i;
        Exact32 ue = cf1 * u_pp[I3S(0, j, i + 1, ny, nxf)]
                + cf2 * u_pp[I3S(1, j, i + 1, ny, nxf)]
                + cf3 * u_pp[I3S(2, j, i + 1, ny, nxf)];
        Exact32 uw = cf1 * u_pp[I3S(0, j, i, ny, nxf)]
                + cf2 * u_pp[I3S(1, j, i, ny, nxf)]
                + cf3 * u_pp[I3S(2, j, i, ny, nxf)];
        Exact32 vn = cf1 * v_pp[I3S(0, j + 1, i, ny + 1, nx)]
                + cf2 * v_pp[I3S(1, j + 1, i, ny + 1, nx)]
                + cf3 * v_pp[I3S(2, j + 1, i, ny + 1, nx)];
        Exact32 vs = cf1 * v_pp[I3S(0, j, i, ny + 1, nx)]
                + cf2 * v_pp[I3S(1, j, i, ny + 1, nx)]
                + cf3 * v_pp[I3S(2, j, i, ny + 1, nx)];
        w_pp[c] = Exact32(0.5f) * rdy * ((ht[cjp] - ht[c]) * vn
                                + (ht[c] - ht[cjm]) * vs)
                + Exact32(0.5f) * rdx * ((ht[cip] - ht[c]) * ue
                                + (ht[c] - ht[cim]) * uw);
    }
    Exact32 t2_dn = wrf_theta_average(th_pp[c],th_pp_old[c],muave,
                                        c1h[0],c2h[0],muts,thp[c],epssm);
    for (int k = 1; k < nz; ++k) {
        ulong h = (ulong)k * st + c;
        ulong hm = h - st;
        Exact32 t2_up = wrf_theta_average(th_pp[h],th_pp_old[h],muave,
                                            c1h[k],c2h[k],muts,thp[h],epssm);
        Exact32 dph_up = (Exact32(1.0f) + epssm) * (rhs[k + 1] - rhs[k])
                    + (Exact32(1.0f) - epssm) * (ph_pp[(ulong)(k + 1) * st + c]
                                        - ph_pp[(ulong)k * st + c]);
        Exact32 dph_dn = (Exact32(1.0f) + epssm) * (rhs[k] - rhs[k - 1])
                    + (Exact32(1.0f) - epssm) * (ph_pp[(ulong)k * st + c]
                                        - ph_pp[(ulong)(k - 1) * st + c]);
        ulong f = (ulong)k * st + c;
        if (moist_cq) {
            w_pp[f] = w_pp[f] + dtau * rw_t[f]
                     + cqw[f] * (Exact32(0.5f) * dtau * G * rdn[k]
                       * (c2a[h] * rdnw[k]
                          / (c1h[k] * mut + c2h[k]) * dph_up
                          - c2a[hm] * rdnw[k - 1]
                            / (c1h[k - 1] * mut + c2h[k - 1]) * dph_dn))
                     + dtau * G * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                             - c2a[hm] * alt[hm] * t2_dn)
                                   - c1f[k] * muave);
        } else {
            w_pp[f] = w_pp[f] + dtau * rw_t[f]
                     + Exact32(0.5f) * dtau * G * rdn[k]
                       * (c2a[h] * rdnw[k]
                          / (c1h[k] * mut + c2h[k]) * dph_up
                          - c2a[hm] * rdnw[k - 1]
                            / (c1h[k - 1] * mut + c2h[k - 1]) * dph_dn)
                     + dtau * G * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                             - c2a[hm] * alt[hm] * t2_dn)
                                   - c1f[k] * muave);
        }
        t2_dn = t2_up;
    }
    if (top_lid) {
        w_pp[(ulong)nz * st + c] = Exact32(0.0f);
    } else {
        ulong h = (ulong)(nz - 1) * st + c;
        ulong f = (ulong)nz * st + c;
        Exact32 dph_dn = (Exact32(1.0f) + epssm) * (rhs[nz] - rhs[nz - 1])
                    + (Exact32(1.0f) - epssm)
                      * (ph_pp[f] - ph_pp[(ulong)(nz - 1) * st + c]);
        w_pp[f] = w_pp[f] + dtau * rw_t[f]
                 + (__fdiv_rn(Exact32(-0.5f)*dtau*G,c1h[nz-1]*mut+c2h[nz-1])
                    * (rdnw[nz-1]*rdnw[nz-1])*Exact32(2.0f)*c2a[h]*dph_dn
                    -dtau*G*(Exact32(2.0f)*rdnw[nz-1]*c2a[h]*alt[h]*t2_dn+c1f[nz]*muave));
    }
    for (int k = 1; k <= nz; ++k)
        w_pp[(ulong)k * st + c] =
            (w_pp[(ulong)k * st + c]
             - a[(ulong)k * st + c] * w_pp[(ulong)(k - 1) * st + c])
            * alpha[(ulong)k * st + c];
    for (int k = nz - 1; k >= 1; --k)
        w_pp[(ulong)k * st + c] -= gam[(ulong)k * st + c]
                                  * w_pp[(ulong)(k + 1) * st + c];
    if (dampmag > Exact32(0.0f)) {
        Exact32 htop = __fdiv_rn((phb[(ulong)nz * bstr + boff]
                     + php[(ulong)nz * st + c]), G);
        Exact32 hbot = htop - zdamp;
        for (int k = 1; k <= nz; ++k) {
            Exact32 hk = __fdiv_rn((phb[(ulong)k * bstr + boff]
                       + php[(ulong)k * st + c]), G);
            if (hk >= hbot) {
                Exact32 angle = __fdiv_rn(Exact32(1.5707963267948966f)*(hk-hbot),zdamp);
                Exact32 sn = glibc_sinf(angle);
                Exact32 dw = dampmag * sn * sn;
                Exact32 cfm = c1f[k] * mut + c2f[k];
                ulong f = (ulong)k * st + c;
                w_pp[f] = (w_pp[f] - dw * cfm * w_ref[f]) / (Exact32(1.0f) + dw);
            }
        }
    }
    for (int k = 1; k <= nz; ++k)
        ph_pp[(ulong)k * st + c] =
            rhs[k] + Exact32(0.5f) * dtau * G * (Exact32(1.0f) + epssm)
                     * w_pp[(ulong)k * st + c]
                     / (c1f[k] * muts + c2f[k]);
    diagnose_p_column(p_pp, al_pp, th_pp, ph_pp, mu_pp, thp, thb, alt,
                      c2a, mup, rdnw, c1h, c2h, mub2d,
                      (ulong)c, st, base3d, nz);
}

void advance_w_phi_msf(device Exact32*  w_pp, device Exact32*  ph_pp,
                       device const Exact32*  rw_t,
                       device const Exact32*  rph_t,
                       device const Exact32*  ww_pp,
                       device const Exact32*  mu_pp,
                       device const Exact32*  mu_pp_old,
                       device const Exact32*  th_pp,
                       device const Exact32*  th_pp_old,
                       device const Exact32*  thp,
                       device const Exact32*  thb,
                       device const Exact32*  php,
                       device const Exact32*  phb,
                       device const Exact32*  alt,
                       device const Exact32*  c2a,
                       device const Exact32*  a,
                       device const Exact32*  alpha,
                       device const Exact32*  gam,
                       device const Exact32*  mup,
                       device const Exact32*  u_pp,
                       device const Exact32*  v_pp,
                       device const Exact32*  w_ref,
                       device const Exact32*  ht,
                       device const Exact32*  rdn,
                       device const Exact32*  rdnw,
                       device const Exact32*  fnm,
                       device const Exact32*  fnp,
                       device const Exact32*  c1h,
                       device const Exact32*  c2h,
                       device const Exact32*  c1f,
                       device const Exact32*  c2f,
                       device const Exact32*  mub2d,
                       device const Exact32*  cqw, int moist_cq,
                       device const Exact32*  msft,
                       device Exact32*  p_pp,
                       device Exact32*  al_pp,
                       Exact32 cf1, Exact32 cf2, Exact32 cf3, Exact32 rdx, Exact32 rdy,
                       Exact32 dtau, Exact32 epssm,
                       Exact32 dampmag, Exact32 zdamp,
                       int boundary_x, int boundary_y,
                       int spec_zone, int base3d, int top_lid,
                       int nz, int ny, int nx, uint gid)
{
    int c = int(gid);
    if (c >= ny * nx || nz + 1 > 257) return;
    int j = c / nx, i = c - j * nx;
    if (i < spec_zone || i >= nx - spec_zone
        || j < spec_zone || j >= ny - spec_zone) return;
    ulong st = (ulong)ny * nx;
    ulong bstr = base3d ? st : 1;
    ulong boff = base3d ? (ulong)c : 0;
    Exact32 mut = mub2d[c] + mup[c];
    Exact32 muts = mut + mu_pp[c];
    Exact32 muave = Exact32(0.5f) * ((Exact32(1.0f) + epssm) * mu_pp[c]
                       + (Exact32(1.0f) - epssm) * mu_pp_old[c]);
    Exact32 msf_c = msft[c];
    Exact32 msf_i = Exact32(1.0f) / msf_c;
    Exact32 rhs[257];
    rhs[0] = Exact32(0.0f);
    for (int k = 1; k <= nz; ++k) {
        ulong f = (ulong)k * st + c;
        rhs[k] = dtau * (rph_t[f]
                         + Exact32(0.5f) * G * (Exact32(1.0f) - epssm) * w_pp[f]);
    }
    Exact32 ph_lo = phb[boff] + php[c];
    Exact32 ph_hi = phb[bstr + boff] + php[(ulong)st + c];
    Exact32 wd_lo = Exact32(0.5f) * (ww_pp[(ulong)st + c] + ww_pp[c])
               * rdnw[0] * (php[st+c]-php[c]+phb[bstr+boff]-phb[boff]);
    for (int k = 1; k < nz; ++k) {
        ph_lo = ph_hi;
        ph_hi = phb[(ulong)(k + 1) * bstr + boff]
              + php[(ulong)(k + 1) * st + c];
        Exact32 wd_hi = Exact32(0.5f) * (ww_pp[(ulong)(k + 1) * st + c]
                           + ww_pp[(ulong)k * st + c])
                   * rdnw[k] * (php[(ulong)(k+1)*st+c]-php[(ulong)k*st+c]
                     +phb[(ulong)(k+1)*bstr+boff]-phb[(ulong)k*bstr+boff]);
        rhs[k] -= dtau * (fnm[k] * wd_hi + fnp[k] * wd_lo);
        wd_lo = wd_hi;
    }
    for (int k = 1; k <= nz; ++k)
        rhs[k] = ph_pp[(ulong)k * st + c]
               + msf_c * rhs[k] / (c1f[k] * mut + c2f[k]);
    if (top_lid) rhs[nz] = Exact32(0.0f);
    {
        int ip = boundary_x ? min(i + 1, nx - 1) : (i + 1) % nx;
        int im = boundary_x ? max(i - 1, 0) : (i - 1 + nx) % nx;
        int jp = boundary_y ? min(j + 1, ny - 1) : (j + 1) % ny;
        int jm = boundary_y ? max(j - 1, 0) : (j - 1 + ny) % ny;
        int nxf = nx + 1;
        ulong cip = (ulong)j * nx + ip, cim = (ulong)j * nx + im;
        ulong cjp = (ulong)jp * nx + i, cjm = (ulong)jm * nx + i;
        Exact32 ue = cf1 * u_pp[I3S(0, j, i + 1, ny, nxf)]
                + cf2 * u_pp[I3S(1, j, i + 1, ny, nxf)]
                + cf3 * u_pp[I3S(2, j, i + 1, ny, nxf)];
        Exact32 uw = cf1 * u_pp[I3S(0, j, i, ny, nxf)]
                + cf2 * u_pp[I3S(1, j, i, ny, nxf)]
                + cf3 * u_pp[I3S(2, j, i, ny, nxf)];
        Exact32 vn = cf1 * v_pp[I3S(0, j + 1, i, ny + 1, nx)]
                + cf2 * v_pp[I3S(1, j + 1, i, ny + 1, nx)]
                + cf3 * v_pp[I3S(2, j + 1, i, ny + 1, nx)];
        Exact32 vs = cf1 * v_pp[I3S(0, j, i, ny + 1, nx)]
                + cf2 * v_pp[I3S(1, j, i, ny + 1, nx)]
                + cf3 * v_pp[I3S(2, j, i, ny + 1, nx)];
        w_pp[c] = msf_c * Exact32(0.5f) * rdy * ((ht[cjp] - ht[c]) * vn
                                         + (ht[c] - ht[cjm]) * vs)
                           + msf_c * Exact32(0.5f) * rdx * ((ht[cip] - ht[c]) * ue
                                           + (ht[c] - ht[cim]) * uw);
    }
    Exact32 t2_dn = wrf_theta_average(th_pp[c],th_pp_old[c],muave,
                                        c1h[0],c2h[0],muts,thp[c],epssm);
    for (int k = 1; k < nz; ++k) {
        ulong h = (ulong)k * st + c;
        ulong hm = h - st;
        Exact32 t2_up = wrf_theta_average(th_pp[h],th_pp_old[h],muave,
                                            c1h[k],c2h[k],muts,thp[h],epssm);
        Exact32 dph_up = (Exact32(1.0f) + epssm) * (rhs[k + 1] - rhs[k])
                    + (Exact32(1.0f) - epssm) * (ph_pp[(ulong)(k + 1) * st + c]
                                        - ph_pp[(ulong)k * st + c]);
        Exact32 dph_dn = (Exact32(1.0f) + epssm) * (rhs[k] - rhs[k - 1])
                    + (Exact32(1.0f) - epssm) * (ph_pp[(ulong)k * st + c]
                                        - ph_pp[(ulong)(k - 1) * st + c]);
        ulong f = (ulong)k * st + c;
        if (moist_cq) {
            w_pp[f] = w_pp[f] + dtau * rw_t[f]
                     + msf_i * cqw[f] * (Exact32(0.5f) * dtau * G * rdn[k]
                                        * (c2a[h] * rdnw[k]
                                           / (c1h[k] * mut + c2h[k]) * dph_up
                                           - c2a[hm] * rdnw[k - 1]
                                             / (c1h[k - 1] * mut + c2h[k - 1])
                                             * dph_dn))
                     + dtau * G * msf_i
                       * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                    - c2a[hm] * alt[hm] * t2_dn)
                          - c1f[k] * muave);
        } else {
            w_pp[f] = w_pp[f] + dtau * rw_t[f]
                     + msf_i * (Exact32(0.5f) * dtau * G * rdn[k]
                                * (c2a[h] * rdnw[k]
                                   / (c1h[k] * mut + c2h[k]) * dph_up
                                   - c2a[hm] * rdnw[k - 1]
                                     / (c1h[k - 1] * mut + c2h[k - 1])
                                     * dph_dn))
                     + dtau * G * msf_i
                       * (rdn[k] * (c2a[h] * alt[h] * t2_up
                                    - c2a[hm] * alt[hm] * t2_dn)
                          - c1f[k] * muave);
        }
        t2_dn = t2_up;
    }
    if (top_lid) {
        w_pp[(ulong)nz * st + c] = Exact32(0.0f);
    } else {
        ulong h = (ulong)(nz - 1) * st + c;
        ulong f = (ulong)nz * st + c;
        Exact32 dph_dn = (Exact32(1.0f) + epssm) * (rhs[nz] - rhs[nz - 1])
                    + (Exact32(1.0f) - epssm)
                      * (ph_pp[f] - ph_pp[(ulong)(nz - 1) * st + c]);
        w_pp[f] = w_pp[f] + dtau * rw_t[f]
                 + msf_i
                   * (__fdiv_rn(Exact32(-0.5f)*dtau*G,c1h[nz-1]*mut+c2h[nz-1])
                      * (rdnw[nz-1]*rdnw[nz-1])*Exact32(2.0f)*c2a[h]*dph_dn
                      -dtau*G*(Exact32(2.0f)*rdnw[nz-1]*c2a[h]*alt[h]*t2_dn+c1f[nz]*muave));
    }
    for (int k = 1; k <= nz; ++k)
        w_pp[(ulong)k * st + c] =
            (w_pp[(ulong)k * st + c]
             - a[(ulong)k * st + c] * w_pp[(ulong)(k - 1) * st + c])
            * alpha[(ulong)k * st + c];
    for (int k = nz - 1; k >= 1; --k)
        w_pp[(ulong)k * st + c] -= gam[(ulong)k * st + c]
                                  * w_pp[(ulong)(k + 1) * st + c];
    if (dampmag > Exact32(0.0f)) {
        Exact32 htop = __fdiv_rn((phb[(ulong)nz * bstr + boff]
                     + php[(ulong)nz * st + c]), G);
        Exact32 hbot = htop - zdamp;
        for (int k = 1; k <= nz; ++k) {
            Exact32 hk = __fdiv_rn((phb[(ulong)k * bstr + boff]
                       + php[(ulong)k * st + c]), G);
            if (hk >= hbot) {
                Exact32 angle = __fdiv_rn(Exact32(1.5707963267948966f)*(hk-hbot),zdamp);
                Exact32 sn = glibc_sinf(angle);
                Exact32 dw = dampmag * sn * sn;
                Exact32 cfm = c1f[k] * mut + c2f[k];
                ulong f = (ulong)k * st + c;
                w_pp[f] = (w_pp[f] - dw * cfm * w_ref[f]) / (Exact32(1.0f) + dw);
            }
        }
    }
    for (int k = 1; k <= nz; ++k)
        ph_pp[(ulong)k * st + c] =
            rhs[k] + msf_c * Exact32(0.5f) * dtau * G * (Exact32(1.0f) + epssm)
                     * w_pp[(ulong)k * st + c]
                     / (c1f[k] * muts + c2f[k]);
    diagnose_p_column(p_pp, al_pp, th_pp, ph_pp, mu_pp, thp, thb, alt,
                      c2a, mup, rdnw, c1h, c2h, mub2d,
                      (ulong)c, st, base3d, nz);
}

void advance_specified_phi_w(device Exact32*  ph_pp,
                             device Exact32*  w_pp,
                             device Exact32*  p_pp,
                             device Exact32*  al_pp,
                             device const Exact32*  rph_t,
                             device const Exact32*  th_pp,
                             device const Exact32*  mup,
                             device const Exact32*  mu_pp,
                             device const Exact32*  mub2d,
                             device const Exact32*  rmu_t,
                             device const Exact32*  php,
                             device const Exact32*  thp,
                             device const Exact32*  thb,
                             device const Exact32*  alt,
                             device const Exact32*  c2a,
                             device const Exact32*  rdnw,
                             device const Exact32*  c1h,
                             device const Exact32*  c2h,
                             device const Exact32*  c1f,
                             device const Exact32*  c2f,
                             Exact32 dtau, int spec_zone, int base3d,
                             int nz, int ny, int nx,
                             device const Exact32*  muts_wrf, uint gid)
{
    ulong c = ulong(gid);
    ulong st = (ulong)ny * nx;
    if (c >= st) return;
    int j = (int)(c / nx), i = (int)(c - (ulong)j * nx);
    if (i >= spec_zone && i < nx - spec_zone
        && j >= spec_zone && j < ny - spec_zone) return;
    Exact32 muts = muts_wrf[c];
    Exact32 dt_rmu = arn_mul(dtau, rmu_t[c]);
    Exact32 mu_old = arn_sub(muts, dt_rmu);
    int source_j = min(max(j, spec_zone), ny - 1 - spec_zone);
    int source_i = min(max(i, spec_zone), nx - 1 - spec_zone);
    ulong source_c = (ulong)source_j * nx + source_i;
    for (int k = 0; k <= nz; ++k) {
        ulong tid = (ulong)k * st + c;
        Exact32 numerator = arn_add(arn_mul(c1f[k], mu_old), c2f[k]);
        Exact32 denominator = arn_add(arn_mul(c1f[k], muts), c2f[k]);
        Exact32 ratio = arn_div(numerator, denominator);
        Exact32 value0 = arn_div(arn_mul(ph_pp[tid], numerator), denominator);
        Exact32 value1 = arn_mul(dtau, rph_t[tid]);
        Exact32 denominator2 = arn_add(arn_mul(c1f[k], muts), c2f[k]);
        Exact32 value2 = arn_div(value1, denominator2);
        Exact32 value3 = arn_add(value0, value2);
        Exact32 value4 = arn_sub(ratio, Exact32(1.0f));
        Exact32 value5 = arn_mul(php[tid], value4);
        ph_pp[tid] = arn_add(value3, value5);
        w_pp[tid] = w_pp[(ulong)k * st + source_c];
    }
    diagnose_p_column_muts(p_pp, al_pp, th_pp, ph_pp, mu_pp, thp, alt, c2a,
                           rdnw, c1h, c2h, c, st, nz, muts);
}

void advance_nested_phi_w(device Exact32*  ph_pp,
                          device Exact32*  w_pp,
                          device Exact32*  p_pp,
                          device Exact32*  al_pp,
                          device const Exact32*  rph_t,
                          device const Exact32*  rw_t,
                          device const Exact32*  th_pp,
                          device const Exact32*  mup,
                          device const Exact32*  mu_pp,
                          device const Exact32*  mub2d,
                          device const Exact32*  rmu_t,
                          device const Exact32*  php,
                          device const Exact32*  thp,
                          device const Exact32*  thb,
                          device const Exact32*  alt,
                          device const Exact32*  c2a,
                          device const Exact32*  rdnw,
                          device const Exact32*  c1h,
                          device const Exact32*  c2h,
                          device const Exact32*  c1f,
                          device const Exact32*  c2f,
                          Exact32 dtau, int spec_zone, int base3d,
                          int nz, int ny, int nx,
                          device const Exact32*  muts_wrf, uint gid)
{
    ulong c = ulong(gid);
    ulong st = (ulong)ny * nx;
    if (c >= st) return;
    int j = (int)(c / nx), i = (int)(c - (ulong)j * nx);
    if (i >= spec_zone && i < nx - spec_zone
        && j >= spec_zone && j < ny - spec_zone) return;
    Exact32 muts = muts_wrf[c];
    Exact32 dt_rmu = arn_mul(dtau, rmu_t[c]);
    Exact32 mu_old = arn_sub(muts, dt_rmu);
    for (int k = 0; k <= nz; ++k) {
        ulong tid = (ulong)k * st + c;
        Exact32 numerator = arn_add(arn_mul(c1f[k], mu_old), c2f[k]);
        Exact32 denominator = arn_add(arn_mul(c1f[k], muts), c2f[k]);
        Exact32 ratio = arn_div(numerator, denominator);
        Exact32 value0 = arn_div(arn_mul(ph_pp[tid], numerator), denominator);
        Exact32 value1 = arn_mul(dtau, rph_t[tid]);
        Exact32 denominator2 = arn_add(arn_mul(c1f[k], muts), c2f[k]);
        Exact32 value2 = arn_div(value1, denominator2);
        Exact32 value3 = arn_add(value0, value2);
        Exact32 value4 = arn_sub(ratio, Exact32(1.0f));
        Exact32 value5 = arn_mul(php[tid], value4);
        ph_pp[tid] = arn_add(value3, value5);
        w_pp[tid] = arn_add(w_pp[tid], arn_mul(dtau, rw_t[tid]));
    }
    diagnose_p_column_muts(p_pp, al_pp, th_pp, ph_pp, mu_pp, thp, alt, c2a,
                           rdnw, c1h, c2h, c, st, nz, muts);
}

}
#define A(n) reinterpret_cast<device const Exact32 *>(r.n)
#define O(n) reinterpret_cast<device Exact32 *>(r.n)
kernel void wrf_exact_acoustic_advance_w_phi(constant WrfExactAcousticAdvanceWPhiResources &r [[buffer(0)]], constant WrfExactAcousticAdvanceWPhiParams &q [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 if (ulong(gid)>=ulong(q.ny)*q.nx) return;
 if (q.mapped) wrf_wphi::advance_w_phi_msf(O(w_pp),O(ph_pp),A(rw_t),A(rph_t),A(ww_pp),A(mu_pp),A(mu_pp_old),A(th_pp),A(th_pp_old),A(thp),A(thb),A(php),A(phb),A(alt),A(c2a),A(a),A(alpha),A(gam),A(mup),A(u_pp),A(v_pp),A(w_ref),A(ht),A(rdn),A(rdnw),A(fnm),A(fnp),A(c1h),A(c2h),A(c1f),A(c2f),A(mub2d),A(cqw),q.moist_cq,A(msft),O(p_pp),O(al_pp),Exact32(q.cf1),Exact32(q.cf2),Exact32(q.cf3),Exact32(q.rdx),Exact32(q.rdy),Exact32(q.dtau),Exact32(q.epssm),Exact32(q.dampmag),Exact32(q.zdamp),q.boundary_x,q.boundary_y,q.spec_zone,q.base3d,q.top_lid,q.nz,q.ny,q.nx,gid);
 else wrf_wphi::advance_w_phi(O(w_pp),O(ph_pp),A(rw_t),A(rph_t),A(ww_pp),A(mu_pp),A(mu_pp_old),A(th_pp),A(th_pp_old),A(thp),A(thb),A(php),A(phb),A(alt),A(c2a),A(a),A(alpha),A(gam),A(mup),A(u_pp),A(v_pp),A(w_ref),A(ht),A(rdn),A(rdnw),A(fnm),A(fnp),A(c1h),A(c2h),A(c1f),A(c2f),A(mub2d),A(cqw),q.moist_cq,O(p_pp),O(al_pp),Exact32(q.cf1),Exact32(q.cf2),Exact32(q.cf3),Exact32(q.rdx),Exact32(q.rdy),Exact32(q.dtau),Exact32(q.epssm),Exact32(q.dampmag),Exact32(q.zdamp),q.boundary_x,q.boundary_y,q.spec_zone,q.base3d,q.top_lid,q.nz,q.ny,q.nx,gid);
}
kernel void wrf_exact_acoustic_boundary_frame(constant WrfExactAcousticBoundaryFrameResources &r [[buffer(0)]], constant WrfExactAcousticBoundaryFrameParams &q [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 if (ulong(gid)>=ulong(q.ny)*q.nx) return;
 if (q.table_w) wrf_wphi::advance_nested_phi_w(O(ph_pp),O(w_pp),O(p_pp),O(al_pp),A(rph_t),A(rw_t),A(th_pp),A(mup),A(mu_pp),A(mub2d),A(rmu_t),A(php),A(thp),A(thb),A(alt),A(c2a),A(rdnw),A(c1h),A(c2h),A(c1f),A(c2f),Exact32(q.dtau),q.spec_zone,q.base3d,q.nz,q.ny,q.nx,A(muts_wrf),gid);
 else wrf_wphi::advance_specified_phi_w(O(ph_pp),O(w_pp),O(p_pp),O(al_pp),A(rph_t),A(th_pp),A(mup),A(mu_pp),A(mub2d),A(rmu_t),A(php),A(thp),A(thb),A(alt),A(c2a),A(rdnw),A(c1h),A(c2h),A(c1f),A(c2f),Exact32(q.dtau),q.spec_zone,q.base3d,q.nz,q.ny,q.nx,A(muts_wrf),gid);
}
#undef A
#undef O
