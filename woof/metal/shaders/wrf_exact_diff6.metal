// Generated from WOOF CUDA sixth-order stencils by tools/diff6_wrf_exact/generate.py; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_diff6_bindings.metalh"
#include "wrf_exact_diff6_seam_u_bindings.metalh"
#include "wrf_exact_diff6_seam_v_bindings.metalh"
#include "wrf_exact_diff6_edge_crop_bindings.metalh"
using namespace metal;
namespace wrf_exact_diff6_impl {
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_greater_equal(b.bits, a.bits); }
void diff6(device const Exact32*  f,
           device Exact32*  tend,
           device const Exact32*  mut,
           device const Exact32*  c1,
           device const Exact32*  c2,
           device const Exact32*  phb,
           device const Exact32*  msfu,
           device const Exact32*  msfv,
           device const Exact32*  msft,
           Exact32 coef, int mono, int slopeopt,
           Exact32 dzthr_x, Exact32 dzthr_y,
           int nlev, int ny, int nys, int nx, int nxs,
           int variant, int wstag,
           int i_lo, int i_hi, int j_lo, int j_hi, int i, int j, int k)
{



    if (i >= nxs || j >= nys || k >= nlev) return;
    if (wstag && (k == 0 || k == nlev - 1)) return;
    if (i < i_lo || i > i_hi || j < j_lo || j > j_hi) return;
    int ic = ((((i) % (nx)) + (nx)) % (nx));
    int jc = ((((j) % (ny)) + (ny)) % (ny));
    Exact32 dflux_x_p0 = exact_mul(10.0f, (f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (0)) % (nx)) + (nx)) % (nx))))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (-1)) % (nx)) + (nx)) % (nx))))])) - exact_mul(5.0f, (f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (1)) % (nx)) + (nx)) % (nx))))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (-2)) % (nx)) + (nx)) % (nx))))]))
                      + (f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (2)) % (nx)) + (nx)) % (nx))))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (-3)) % (nx)) + (nx)) % (nx))))]);
    Exact32 dflux_x_p1 = exact_mul(10.0f, (f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (1)) % (nx)) + (nx)) % (nx))))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (0)) % (nx)) + (nx)) % (nx))))])) - exact_mul(5.0f, (f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (2)) % (nx)) + (nx)) % (nx))))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (-1)) % (nx)) + (nx)) % (nx))))]))
                      + (f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (3)) % (nx)) + (nx)) % (nx))))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (-2)) % (nx)) + (nx)) % (nx))))]);
    if (mono == 2) {
        if (dflux_x_p0 * (f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (0)) % (nx)) + (nx)) % (nx))))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (-1)) % (nx)) + (nx)) % (nx))))]) <= 0.0f) dflux_x_p0 = 0.0f;
        if (dflux_x_p1 * (f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (1)) % (nx)) + (nx)) % (nx))))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(jc) * (nxs) + (((((ic + (0)) % (nx)) + (nx)) % (nx))))]) <= 0.0f) dflux_x_p1 = 0.0f;
    }
    Exact32 mu_x_p0, mu_x_p1;
    if (variant == 1) {
        mu_x_p0 = (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic - 1) % (nx)) + (nx)) % (nx))]) + c2[k]);
        mu_x_p1 = (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]);
    } else if (variant == 2) {
        mu_x_p0 = 0.25f * ((exact_mul(c1[k], mut[(ulong)((((jc - 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic - 1) % (nx)) + (nx)) % (nx))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((((jc - 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k])
                           + (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic - 1) % (nx)) + (nx)) % (nx))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]));
        mu_x_p1 = 0.25f * ((exact_mul(c1[k], mut[(ulong)((((jc - 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((((jc - 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic + 1) % (nx)) + (nx)) % (nx))]) + c2[k])
                           + (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic + 1) % (nx)) + (nx)) % (nx))]) + c2[k]));
    } else {
        mu_x_p0 = 0.5f * ((exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic - 1) % (nx)) + (nx)) % (nx))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]));
        mu_x_p1 = 0.5f * ((exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic + 1) % (nx)) + (nx)) % (nx))]) + c2[k]));
    }
    Exact32 sdx_p0 = 1.0f, sdx_p1 = 1.0f;
    if (slopeopt >= 1) {
        Exact32 a0 = exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic) % (nx)) + (nx)) % (nx))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic - 1) % (nx)) + (nx)) % (nx))))]) * msfu[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * (nx + 1) + ((((ic) % (nx)) + (nx)) % (nx))];
        Exact32 a1 = exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic + 1) % (nx)) + (nx)) % (nx))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic) % (nx)) + (nx)) % (nx))))]) * msfu[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * (nx + 1) + ((((ic + 1) % (nx)) + (nx)) % (nx))];
        Exact32 dz0 = a0, dz1 = a1;
        if (variant == 1) {
            dz0 = exact_max(a0, exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic - 1) % (nx)) + (nx)) % (nx))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic - 2) % (nx)) + (nx)) % (nx))))])
                            * msfu[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * (nx + 1) + ((((ic - 1) % (nx)) + (nx)) % (nx))]);
            dz1 = exact_max(a1, a0);
        } else if (variant == 2) {
            dz0 = exact_max(a0, exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc - 1) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic) % (nx)) + (nx)) % (nx))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc - 1) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic - 1) % (nx)) + (nx)) % (nx))))])
                            * msfu[(ulong)((((jc - 1) % (ny)) + (ny)) % (ny)) * (nx + 1) + ((((ic) % (nx)) + (nx)) % (nx))]);
            dz1 = exact_max(a1, exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc - 1) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic + 1) % (nx)) + (nx)) % (nx))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc - 1) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic) % (nx)) + (nx)) % (nx))))])
                            * msfu[(ulong)((((jc - 1) % (ny)) + (ny)) % (ny)) * (nx + 1) + ((((ic + 1) % (nx)) + (nx)) % (nx))]);
        }
        sdx_p0 = exact_max(1.0f - exact_div(dz0, dzthr_x), 0.0f);
        sdx_p1 = exact_max(1.0f - exact_div(dz1, dzthr_x), 0.0f);
    }
    Exact32 map = variant == 1 ? msfu[(ulong)j * (nx + 1) + i]
             : variant == 2 ? msfv[(ulong)j * nx + i]
             : msft[(ulong)jc * nx + ic];
    Exact32 tendency_x = exact_mul(exact_mul(coef, map),
        (exact_mul(exact_mul(sdx_p1, mu_x_p1), dflux_x_p1)
           - exact_mul(exact_mul(sdx_p0, mu_x_p0), dflux_x_p0)));
    Exact32 dflux_y_p0 = exact_mul(10.0f, (f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (0)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (-1)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))])) - exact_mul(5.0f, (f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (1)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (-2)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))]))
                      + (f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (2)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (-3)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))]);
    Exact32 dflux_y_p1 = exact_mul(10.0f, (f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (1)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (0)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))])) - exact_mul(5.0f, (f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (2)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (-1)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))]))
                      + (f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (3)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (-2)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))]);
    if (mono == 2) {
        if (dflux_y_p0 * (f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (0)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (-1)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))]) <= 0.0f) dflux_y_p0 = 0.0f;
        if (dflux_y_p1 * (f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (1)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))] - f[((ulong)(k) * (nys) * (nxs) + (ulong)(((((jc + (0)) % (ny)) + (ny)) % (ny))) * (nxs) + (ic))]) <= 0.0f) dflux_y_p1 = 0.0f;
    }
    Exact32 mu_y_p0, mu_y_p1;
    if (variant == 1) {
        mu_y_p0 = 0.25f * ((exact_mul(c1[k], mut[(ulong)((((jc - 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic - 1) % (nx)) + (nx)) % (nx))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((((jc - 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k])
                           + (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic - 1) % (nx)) + (nx)) % (nx))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]));
        mu_y_p1 = 0.25f * ((exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic - 1) % (nx)) + (nx)) % (nx))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k])
                           + (exact_mul(c1[k], mut[(ulong)((((jc + 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic - 1) % (nx)) + (nx)) % (nx))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((((jc + 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]));
    } else if (variant == 2) {
        mu_y_p0 = (exact_mul(c1[k], mut[(ulong)((((jc - 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]);
        mu_y_p1 = (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]);
    } else {
        mu_y_p0 = 0.5f * ((exact_mul(c1[k], mut[(ulong)((((jc - 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]));
        mu_y_p1 = 0.5f * ((exact_mul(c1[k], mut[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((((jc + 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]) + c2[k]));
    }
    Exact32 sdy_p0 = 1.0f, sdy_p1 = 1.0f;
    if (slopeopt >= 1) {
        Exact32 b0 = exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic) % (nx)) + (nx)) % (nx))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc - 1) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic) % (nx)) + (nx)) % (nx))))]) * msfv[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))];
        Exact32 b1 = exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc + 1) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic) % (nx)) + (nx)) % (nx))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic) % (nx)) + (nx)) % (nx))))]) * msfv[(ulong)((((jc + 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))];
        Exact32 dz0 = b0, dz1 = b1;
        if (variant == 1) {
            dz0 = exact_max(b0, exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic - 1) % (nx)) + (nx)) % (nx))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc - 1) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic - 1) % (nx)) + (nx)) % (nx))))])
                            * msfv[(ulong)((((jc) % (ny)) + (ny)) % (ny)) * nx + ((((ic - 1) % (nx)) + (nx)) % (nx))]);
            dz1 = exact_max(b1, exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc + 1) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic - 1) % (nx)) + (nx)) % (nx))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic - 1) % (nx)) + (nx)) % (nx))))])
                            * msfv[(ulong)((((jc + 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic - 1) % (nx)) + (nx)) % (nx))]);
        } else if (variant == 2) {
            dz0 = exact_max(b0, exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc - 1) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic) % (nx)) + (nx)) % (nx))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(((((jc - 2) % (ny)) + (ny)) % (ny))) * (nx) + (((((ic) % (nx)) + (nx)) % (nx))))])
                            * msfv[(ulong)((((jc - 1) % (ny)) + (ny)) % (ny)) * nx + ((((ic) % (nx)) + (nx)) % (nx))]);
            dz1 = exact_max(b1, b0);
        }
        sdy_p0 = exact_max(1.0f - exact_div(dz0, dzthr_y), 0.0f);
        sdy_p1 = exact_max(1.0f - exact_div(dz1, dzthr_y), 0.0f);
    }
    Exact32 tendency_y = exact_mul(exact_mul(coef, map),
        (exact_mul(exact_mul(sdy_p1, mu_y_p1), dflux_y_p1)
           - exact_mul(exact_mul(sdy_p0, mu_y_p0), dflux_y_p0)));
    tend[((ulong)(k) * (nys) * (nxs) + (ulong)(j) * (nxs) + (i))] = (tend[((ulong)(k) * (nys) * (nxs) + (ulong)(j) * (nxs) + (i))] + tendency_x) + tendency_y;
}
void diff6_seam_u(device const Exact32*  f,
                  device Exact32*  tend,
                  device const Exact32*  mut,
                  device const Exact32*  c1,
                  device const Exact32*  c2,
                  device const Exact32*  phb,
                  device const Exact32*  msfu,
                  device const Exact32*  msfv,
                  device const Exact32*  msft,
                  Exact32 coef, int mono, int slopeopt,
                  Exact32 dzthr_x, Exact32 dzthr_y,
                  int nlev, int ny, int nx,
                  int h0, int h1, int bndc, int t, int k)
{


    int j = h0 + t;
    if (j > h1 || k >= nlev) return;
    const int i = nx - 3;
    const int nxs = nx + 1;
    Exact32 dflux_x_p0 = exact_mul(10.0f, (f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (0)))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (-1)))])) - exact_mul(5.0f, (f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (1)))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (-2)))]))
                      + (f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (2)))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (-3)))]);
    Exact32 dflux_x_p1 = exact_mul(10.0f, (f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (1)))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (0)))])) - exact_mul(5.0f, (f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (2)))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (-1)))]))
                      + (f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (3)))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (-2)))]);
    if (mono == 2) {
        if (dflux_x_p0 * (f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (0)))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (-1)))]) <= 0.0f) dflux_x_p0 = 0.0f;
        if (dflux_x_p1 * (f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (1)))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i + (0)))]) <= 0.0f) dflux_x_p1 = 0.0f;
    }
    Exact32 sdx_p0 = 1.0f, sdx_p1 = 1.0f;
    if (slopeopt >= 1) {
        Exact32 a0 = exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i - 1))]) * msfu[(ulong)(j) * (nx + 1) + (i)];
        Exact32 a1 = exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i + 1))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]) * msfu[(ulong)(j) * (nx + 1) + (i + 1)];
        Exact32 dz0 = exact_max(a0, exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i - 1))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i - 2))])
                             * msfu[(ulong)(j) * (nx + 1) + (i - 1)]);
        Exact32 dz1 = exact_max(a1, a0);
        sdx_p0 = exact_max(1.0f - exact_div(dz0, dzthr_x), 0.0f);
        sdx_p1 = exact_max(1.0f - exact_div(dz1, dzthr_x), 0.0f);
    }
    Exact32 mapped_coef = exact_mul(coef, msfu[(ulong)(j) * (nx + 1) + (i)]);
    Exact32 tendency_x = exact_mul(mapped_coef,
        (exact_mul(exact_mul(sdx_p1, (exact_mul(c1[k], mut[(ulong)(j) * nx + (i)]) + c2[k])), dflux_x_p1)
           - exact_mul(exact_mul(sdx_p0, (exact_mul(c1[k], mut[(ulong)(j) * nx + (i - 1)]) + c2[k])), dflux_x_p0)));
    Exact32 dflux_y_p0 = exact_mul(10.0f, (f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (-1)) : ((((j + (-1)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))])) - exact_mul(5.0f, (f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (1)) : ((((j + (1)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (-2)) : ((((j + (-2)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))]))
                      + (f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (2)) : ((((j + (2)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (-3)) : ((((j + (-3)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))]);
    Exact32 dflux_y_p1 = exact_mul(10.0f, (f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (1)) : ((((j + (1)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))])) - exact_mul(5.0f, (f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (2)) : ((((j + (2)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (-1)) : ((((j + (-1)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))]))
                      + (f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (3)) : ((((j + (3)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (-2)) : ((((j + (-2)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))]);
    if (mono == 2) {
        if (dflux_y_p0 * (f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (-1)) : ((((j + (-1)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))]) <= 0.0f) dflux_y_p0 = 0.0f;
        if (dflux_y_p1 * (f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (1)) : ((((j + (1)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))] - f[((ulong)(k) * (ny) * (nxs) + (ulong)((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) * (nxs) + (i))]) <= 0.0f) dflux_y_p1 = 0.0f;
    }
    Exact32 sdy_p0 = 1.0f, sdy_p1 = 1.0f;
    if (slopeopt >= 1) {
        Exact32 b0 = exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) * (nx) + (i))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)((bndc ? (j + (-1)) : ((((j + (-1)) % (ny)) + (ny)) % (ny)))) * (nx) + (i))]) * msfv[(ulong)(bndc ? ((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) : (((((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) % (ny)) + (ny)) % (ny))) * nx + (i)];
        Exact32 b1 = exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)((bndc ? (j + (1)) : ((((j + (1)) % (ny)) + (ny)) % (ny)))) * (nx) + (i))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) * (nx) + (i))]) * msfv[(ulong)(bndc ? ((bndc ? (j + (1)) : ((((j + (1)) % (ny)) + (ny)) % (ny)))) : (((((bndc ? (j + (1)) : ((((j + (1)) % (ny)) + (ny)) % (ny)))) % (ny)) + (ny)) % (ny))) * nx + (i)];
        Exact32 dz0 = exact_max(b0, exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) * (nx) + (i - 1))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)((bndc ? (j + (-1)) : ((((j + (-1)) % (ny)) + (ny)) % (ny)))) * (nx) + (i - 1))])
                             * msfv[(ulong)(bndc ? ((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) : (((((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) % (ny)) + (ny)) % (ny))) * nx + (i - 1)]);
        Exact32 dz1 = exact_max(b1, exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)((bndc ? (j + (1)) : ((((j + (1)) % (ny)) + (ny)) % (ny)))) * (nx) + (i - 1))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) * (nx) + (i - 1))])
                             * msfv[(ulong)(bndc ? ((bndc ? (j + (1)) : ((((j + (1)) % (ny)) + (ny)) % (ny)))) : (((((bndc ? (j + (1)) : ((((j + (1)) % (ny)) + (ny)) % (ny)))) % (ny)) + (ny)) % (ny))) * nx + (i - 1)]);
        sdy_p0 = exact_max(1.0f - exact_div(dz0, dzthr_y), 0.0f);
        sdy_p1 = exact_max(1.0f - exact_div(dz1, dzthr_y), 0.0f);
    }
    Exact32 mu_y_p0 = 0.25f * ((exact_mul(c1[k], mut[(ulong)((bndc ? (j + (-1)) : ((((j + (-1)) % (ny)) + (ny)) % (ny)))) * nx + (i - 1)]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((bndc ? (j + (-1)) : ((((j + (-1)) % (ny)) + (ny)) % (ny)))) * nx + (i)]) + c2[k])
                            + (exact_mul(c1[k], mut[(ulong)((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) * nx + (i - 1)]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) * nx + (i)]) + c2[k]));
    Exact32 mu_y_p1 = 0.25f * ((exact_mul(c1[k], mut[(ulong)((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) * nx + (i - 1)]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((bndc ? (j + (0)) : ((((j + (0)) % (ny)) + (ny)) % (ny)))) * nx + (i)]) + c2[k])
                            + (exact_mul(c1[k], mut[(ulong)((bndc ? (j + (1)) : ((((j + (1)) % (ny)) + (ny)) % (ny)))) * nx + (i - 1)]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)((bndc ? (j + (1)) : ((((j + (1)) % (ny)) + (ny)) % (ny)))) * nx + (i)]) + c2[k]));
    Exact32 tendency_y = exact_mul(mapped_coef,
        (exact_mul(exact_mul(sdy_p1, mu_y_p1), dflux_y_p1) - exact_mul(exact_mul(sdy_p0, mu_y_p0), dflux_y_p0)));
    tend[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))] = (tend[((ulong)(k) * (ny) * (nxs) + (ulong)(j) * (nxs) + (i))] + tendency_x) + tendency_y;
}
void diff6_seam_v(device const Exact32*  f,
                  device Exact32*  tend,
                  device const Exact32*  mut,
                  device const Exact32*  c1,
                  device const Exact32*  c2,
                  device const Exact32*  phb,
                  device const Exact32*  msfu,
                  device const Exact32*  msfv,
                  device const Exact32*  msft,
                  Exact32 coef, int mono, int slopeopt,
                  Exact32 dzthr_x, Exact32 dzthr_y,
                  int nlev, int ny, int nx,
                  int h0, int h1, int bndc, int t, int k)
{


    int i = h0 + t;
    if (i > h1 || k >= nlev) return;
    const int j = ny - 3;
    const int nys = ny + 1;
    Exact32 dflux_x_p0 = exact_mul(10.0f, (f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx)))))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (-1)) : ((((i + (-1)) % (nx)) + (nx)) % (nx)))))])) - exact_mul(5.0f, (f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (1)) : ((((i + (1)) % (nx)) + (nx)) % (nx)))))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (-2)) : ((((i + (-2)) % (nx)) + (nx)) % (nx)))))]))
                      + (f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (2)) : ((((i + (2)) % (nx)) + (nx)) % (nx)))))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (-3)) : ((((i + (-3)) % (nx)) + (nx)) % (nx)))))]);
    Exact32 dflux_x_p1 = exact_mul(10.0f, (f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (1)) : ((((i + (1)) % (nx)) + (nx)) % (nx)))))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx)))))])) - exact_mul(5.0f, (f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (2)) : ((((i + (2)) % (nx)) + (nx)) % (nx)))))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (-1)) : ((((i + (-1)) % (nx)) + (nx)) % (nx)))))]))
                      + (f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (3)) : ((((i + (3)) % (nx)) + (nx)) % (nx)))))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (-2)) : ((((i + (-2)) % (nx)) + (nx)) % (nx)))))]);
    if (mono == 2) {
        if (dflux_x_p0 * (f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx)))))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (-1)) : ((((i + (-1)) % (nx)) + (nx)) % (nx)))))]) <= 0.0f) dflux_x_p0 = 0.0f;
        if (dflux_x_p1 * (f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (1)) : ((((i + (1)) % (nx)) + (nx)) % (nx)))))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx)))))]) <= 0.0f) dflux_x_p1 = 0.0f;
    }
    Exact32 sdx_p0 = 1.0f, sdx_p1 = 1.0f;
    if (slopeopt >= 1) {
        Exact32 a0 = exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx)))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (-1)) : ((((i + (-1)) % (nx)) + (nx)) % (nx)))))]) * msfu[(ulong)(j) * (nx + 1) + (bndc ? ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx)))) : (((((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx)))) % (nx)) + (nx)) % (nx)))];
        Exact32 a1 = exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (1)) : ((((i + (1)) % (nx)) + (nx)) % (nx)))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx)))))]) * msfu[(ulong)(j) * (nx + 1) + (bndc ? ((bndc ? (i + (1)) : ((((i + (1)) % (nx)) + (nx)) % (nx)))) : (((((bndc ? (i + (1)) : ((((i + (1)) % (nx)) + (nx)) % (nx)))) % (nx)) + (nx)) % (nx)))];
        Exact32 dz0 = exact_max(a0, exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(j - 1) * (nx) + ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx)))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(j - 1) * (nx) + ((bndc ? (i + (-1)) : ((((i + (-1)) % (nx)) + (nx)) % (nx)))))])
                             * msfu[(ulong)(j - 1) * (nx + 1) + (bndc ? ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx)))) : (((((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx)))) % (nx)) + (nx)) % (nx)))]);
        Exact32 dz1 = exact_max(a1, exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(j - 1) * (nx) + ((bndc ? (i + (1)) : ((((i + (1)) % (nx)) + (nx)) % (nx)))))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(j - 1) * (nx) + ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx)))))])
                             * msfu[(ulong)(j - 1) * (nx + 1) + (bndc ? ((bndc ? (i + (1)) : ((((i + (1)) % (nx)) + (nx)) % (nx)))) : (((((bndc ? (i + (1)) : ((((i + (1)) % (nx)) + (nx)) % (nx)))) % (nx)) + (nx)) % (nx)))]);
        sdx_p0 = exact_max(1.0f - exact_div(dz0, dzthr_x), 0.0f);
        sdx_p1 = exact_max(1.0f - exact_div(dz1, dzthr_x), 0.0f);
    }
    Exact32 mu_x_p0 = 0.25f * ((exact_mul(c1[k], mut[(ulong)(j - 1) * nx + ((bndc ? (i + (-1)) : ((((i + (-1)) % (nx)) + (nx)) % (nx))))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)(j - 1) * nx + ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx))))]) + c2[k])
                            + (exact_mul(c1[k], mut[(ulong)(j) * nx + ((bndc ? (i + (-1)) : ((((i + (-1)) % (nx)) + (nx)) % (nx))))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)(j) * nx + ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx))))]) + c2[k]));
    Exact32 mu_x_p1 = 0.25f * ((exact_mul(c1[k], mut[(ulong)(j - 1) * nx + ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx))))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)(j - 1) * nx + ((bndc ? (i + (1)) : ((((i + (1)) % (nx)) + (nx)) % (nx))))]) + c2[k])
                            + (exact_mul(c1[k], mut[(ulong)(j) * nx + ((bndc ? (i + (0)) : ((((i + (0)) % (nx)) + (nx)) % (nx))))]) + c2[k]) + (exact_mul(c1[k], mut[(ulong)(j) * nx + ((bndc ? (i + (1)) : ((((i + (1)) % (nx)) + (nx)) % (nx))))]) + c2[k]));
    Exact32 mapped_coef = exact_mul(coef, msfv[(ulong)(j) * nx + (i)]);
    Exact32 tendency_x = exact_mul(mapped_coef,
        (exact_mul(exact_mul(sdx_p1, mu_x_p1), dflux_x_p1) - exact_mul(exact_mul(sdx_p0, mu_x_p0), dflux_x_p0)));
    Exact32 dflux_y_p0 = exact_mul(10.0f, (f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (0)) * (nx) + (i))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (-1)) * (nx) + (i))])) - exact_mul(5.0f, (f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (1)) * (nx) + (i))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (-2)) * (nx) + (i))]))
                      + (f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (2)) * (nx) + (i))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (-3)) * (nx) + (i))]);
    Exact32 dflux_y_p1 = exact_mul(10.0f, (f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (1)) * (nx) + (i))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (0)) * (nx) + (i))])) - exact_mul(5.0f, (f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (2)) * (nx) + (i))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (-1)) * (nx) + (i))]))
                      + (f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (3)) * (nx) + (i))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (-2)) * (nx) + (i))]);
    if (mono == 2) {
        if (dflux_y_p0 * (f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (0)) * (nx) + (i))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (-1)) * (nx) + (i))]) <= 0.0f) dflux_y_p0 = 0.0f;
        if (dflux_y_p1 * (f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (1)) * (nx) + (i))] - f[((ulong)(k) * (nys) * (nx) + (ulong)(j + (0)) * (nx) + (i))]) <= 0.0f) dflux_y_p1 = 0.0f;
    }
    Exact32 sdy_p0 = 1.0f, sdy_p1 = 1.0f;
    if (slopeopt >= 1) {
        Exact32 b0 = exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(j - 1) * (nx) + (i))]) * msfv[(ulong)(j) * nx + (i)];
        Exact32 b1 = exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(j + 1) * (nx) + (i))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(j) * (nx) + (i))]) * msfv[(ulong)(j + 1) * nx + (i)];
        Exact32 dz0 = exact_max(b0, exact_abs(phb[((ulong)(k) * (ny) * (nx) + (ulong)(j - 1) * (nx) + (i))] - phb[((ulong)(k) * (ny) * (nx) + (ulong)(j - 2) * (nx) + (i))])
                             * msfv[(ulong)(j - 1) * nx + (i)]);
        Exact32 dz1 = exact_max(b1, b0);
        sdy_p0 = exact_max(1.0f - exact_div(dz0, dzthr_y), 0.0f);
        sdy_p1 = exact_max(1.0f - exact_div(dz1, dzthr_y), 0.0f);
    }
    Exact32 tendency_y = exact_mul(mapped_coef,
        (exact_mul(exact_mul(sdy_p1, (exact_mul(c1[k], mut[(ulong)(j) * nx + (i)]) + c2[k])), dflux_y_p1) - exact_mul(exact_mul(sdy_p0, (exact_mul(c1[k], mut[(ulong)(j - 1) * nx + (i)]) + c2[k])), dflux_y_p0)));
    tend[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + (i))] = (tend[((ulong)(k) * (nys) * (nx) + (ulong)(j) * (nx) + (i))] + tendency_x) + tendency_y;
}
}
kernel void wrf_exact_diff6(constant WrfExactDiff6Resources &r [[buffer(0)]], constant WrfExactDiff6Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  if (ulong(gid)>=ulong(p.nlev)*ulong(p.nys)*ulong(p.nxs)) return;
  int i=int(gid%uint(p.nxs)), j=int((gid/uint(p.nxs))%uint(p.nys)), k=int(gid/(uint(p.nxs)*uint(p.nys)));
  wrf_exact_diff6_impl::diff6(reinterpret_cast<device const Exact32*>(r.f), reinterpret_cast<device Exact32*>(r.tend), reinterpret_cast<device const Exact32*>(r.mut), reinterpret_cast<device const Exact32*>(r.c1), reinterpret_cast<device const Exact32*>(r.c2), reinterpret_cast<device const Exact32*>(r.phb), reinterpret_cast<device const Exact32*>(r.msfu), reinterpret_cast<device const Exact32*>(r.msfv), reinterpret_cast<device const Exact32*>(r.msft), Exact32(p.coef), p.mono, p.slopeopt, Exact32(p.dzthr_x), Exact32(p.dzthr_y), p.nlev, p.ny, p.nys, p.nx, p.nxs, p.variant, p.wstag, (p.boundary_x ? 3 : 0), (p.boundary_x ? p.nx-4 : p.nxs-1), (p.boundary_y ? 3 : 0), (p.boundary_y ? p.ny-4 : p.nys-1), i, j, k);
}
kernel void wrf_exact_diff6_seam_u(constant WrfExactDiff6SeamUResources &r [[buffer(0)]], constant WrfExactDiff6SeamUParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  if (ulong(gid)>=ulong(p.nlev)*ulong(p.h1-p.h0+1)) return;
  int t=int(gid%uint(p.h1-p.h0+1)), k=int(gid/uint(p.h1-p.h0+1));
  wrf_exact_diff6_impl::diff6_seam_u(reinterpret_cast<device const Exact32*>(r.f), reinterpret_cast<device Exact32*>(r.tend), reinterpret_cast<device const Exact32*>(r.mut), reinterpret_cast<device const Exact32*>(r.c1), reinterpret_cast<device const Exact32*>(r.c2), reinterpret_cast<device const Exact32*>(r.phb), reinterpret_cast<device const Exact32*>(r.msfu), reinterpret_cast<device const Exact32*>(r.msfv), reinterpret_cast<device const Exact32*>(r.msft), Exact32(p.coef), p.mono, p.slopeopt, Exact32(p.dzthr_x), Exact32(p.dzthr_y), p.nlev, p.ny, p.nx, p.h0, p.h1, p.boundary_cross, t, k);
}
kernel void wrf_exact_diff6_seam_v(constant WrfExactDiff6SeamVResources &r [[buffer(0)]], constant WrfExactDiff6SeamVParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
  if (ulong(gid)>=ulong(p.nlev)*ulong(p.h1-p.h0+1)) return;
  int t=int(gid%uint(p.h1-p.h0+1)), k=int(gid/uint(p.h1-p.h0+1));
  wrf_exact_diff6_impl::diff6_seam_v(reinterpret_cast<device const Exact32*>(r.f), reinterpret_cast<device Exact32*>(r.tend), reinterpret_cast<device const Exact32*>(r.mut), reinterpret_cast<device const Exact32*>(r.c1), reinterpret_cast<device const Exact32*>(r.c2), reinterpret_cast<device const Exact32*>(r.phb), reinterpret_cast<device const Exact32*>(r.msfu), reinterpret_cast<device const Exact32*>(r.msfv), reinterpret_cast<device const Exact32*>(r.msft), Exact32(p.coef), p.mono, p.slopeopt, Exact32(p.dzthr_x), Exact32(p.dzthr_y), p.nlev, p.ny, p.nx, p.h0, p.h1, p.boundary_cross, t, k);
}
kernel void wrf_exact_diff6_edge_crop(constant WrfExactDiff6EdgeCropResources &r [[buffer(0)]], constant WrfExactDiff6EdgeCropParams &p [[buffer(1)]], uint flat [[thread_position_in_grid]]) {
  if (flat >= uint(p.nlev)*uint(p.ny)*uint(p.nx)) return;
  uint i=flat%uint(p.nx), j=(flat/uint(p.nx))%uint(p.ny), k=flat/(uint(p.nx)*uint(p.ny));
  uint src=(k*uint(p.ny+7)+j+3)*uint(p.nx+7)+i+3;
  reinterpret_cast<device uint*>(r.tend)[flat]=reinterpret_cast<device const uint*>(r.padded)[src];
}
