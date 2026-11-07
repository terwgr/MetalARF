#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_acoustic_advance_uv_bindings.metalh"
using namespace metal;
#define I3S(k,j,i,ny,nx) (((ulong)(k)*(ny)+(j))*(nx)+(i))
#define arn_add exact_add
#define arn_mul exact_mul
namespace wrf_uv {
static inline
Exact32 pdmp(device const Exact32*  p, device const Exact32*  pold,
          ulong idx, Exact32 smdiv)
{
    return p[idx] + smdiv * (p[idx] - pold[idx]);
}
static
Exact32 dpn_face(device const Exact32*  p, device const Exact32*  pold,
              Exact32 smdiv, int kf, int nz, ulong cA, ulong cB, ulong st,
              device const Exact32*  fnm, device const Exact32*  fnp,
              Exact32 cf1, Exact32 cf2, Exact32 cf3, int top_lid)
{
    if (kf == nz) {
        if (!top_lid) return Exact32(0.0f);
        return Exact32(0.5f) * (
            cf1 * (pdmp(p, pold, cA + (ulong)(nz - 1) * st, smdiv)
                 + pdmp(p, pold, cB + (ulong)(nz - 1) * st, smdiv))
          + cf2 * (pdmp(p, pold, cA + (ulong)(nz - 2) * st, smdiv)
                 + pdmp(p, pold, cB + (ulong)(nz - 2) * st, smdiv))
          + cf3 * (pdmp(p, pold, cA + (ulong)(nz - 3) * st, smdiv)
                 + pdmp(p, pold, cB + (ulong)(nz - 3) * st, smdiv)));
    }
    if (kf == 0)
        return Exact32(0.5f) * (cf1 * (pdmp(p, pold, cA, smdiv)
                            + pdmp(p, pold, cB, smdiv))
                     + cf2 * (pdmp(p, pold, cA + st, smdiv)
                            + pdmp(p, pold, cB + st, smdiv))
                     + cf3 * (pdmp(p, pold, cA + 2*st, smdiv)
                            + pdmp(p, pold, cB + 2*st, smdiv)));
    return Exact32(0.5f) * (fnm[kf] * (pdmp(p, pold, cA + (ulong)kf * st, smdiv)
                            + pdmp(p, pold, cB + (ulong)kf * st, smdiv))
                 + fnp[kf] * (pdmp(p, pold, cA + (ulong)(kf-1) * st, smdiv)
                            + pdmp(p, pold, cB + (ulong)(kf-1) * st, smdiv)));
}
static
Exact32 pgrad_face(ulong cA, ulong cB, int k, int nz, ulong st, Exact32 rd,
                device const Exact32*  ph_pp, device const Exact32*  php,
                device const Exact32*  phb,
                device const Exact32*  alt,
                device const Exact32*  al_pp, device const Exact32*  pb,
                device const Exact32*  p_pp, device const Exact32*  p_old,
                Exact32 smdiv,
                device const Exact32*  mup, device const Exact32*  mu_pp,
                device const Exact32*  mub2d,
                device const Exact32*  c1h, device const Exact32*  c2h,
                device const Exact32*  fnm, device const Exact32*  fnp,
                device const Exact32*  rdnw,
                Exact32 cf1, Exact32 cf2, Exact32 cf3, int base3d, int top_lid)
{
    Exact32 muf = Exact32(0.5f) * (mup[cA] + mup[cB] + mub2d[cA] + mub2d[cB]);
    ulong a = cA + (ulong)k * st, b = cB + (ulong)k * st;
    Exact32 dph = (ph_pp[a + st] - ph_pp[b + st]) + (ph_pp[a] - ph_pp[b]);
    Exact32 dpe = pdmp(p_pp, p_old, a, smdiv) - pdmp(p_pp, p_old, b, smdiv);
    Exact32 dpb = base3d ? (pb[a] - pb[b]) : Exact32(0.0f);
    Exact32 dpxy = Exact32(0.5f) * rd * (c1h[k] * muf + c2h[k])
              * (dph + (alt[a] + alt[b]) * dpe + (al_pp[a] + al_pp[b]) * dpb);
    Exact32 dpn_up = dpn_face(p_pp, p_old, smdiv, k + 1, nz, cA, cB, st,
                           fnm, fnp, cf1, cf2, cf3, top_lid);
    Exact32 dpn_dn = dpn_face(p_pp, p_old, smdiv, k, nz, cA, cB, st,
                           fnm, fnp, cf1, cf2, cf3, top_lid);
    ulong ba = base3d ? a : (ulong)k;
    ulong bb = base3d ? b : (ulong)k;
    ulong bst = base3d ? st : 1;
    Exact32 half_a = Exact32(0.5f) * (phb[ba] + phb[ba + bst] + php[a] + php[a + st]);
    Exact32 half_b = Exact32(0.5f) * (phb[bb] + phb[bb + bst] + php[b] + php[b + st]);
    Exact32 dphp = half_a - half_b;
    dpxy += rd * dphp * (rdnw[k] * (dpn_up - dpn_dn)
                         - Exact32(0.5f) * ((c1h[k] * mu_pp[cB]) + (c1h[k] * mu_pp[cA])));
    return dpxy;
}

void advance_uv(device Exact32*  u_pp, device Exact32*  v_pp,
                device const Exact32*  ru_t, device const Exact32*  rv_t,
                device const Exact32*  p_pp,
                device const Exact32*  p_pp_old,
                device const Exact32*  ph_pp, device const Exact32*  php,
                device const Exact32*  phb,
                device const Exact32*  alt,
                device const Exact32*  al_pp, device const Exact32*  pb,
                device const Exact32*  mup, device const Exact32*  mu_pp,
                device const Exact32*  mub2d,
                 device const Exact32*  c1h, device const Exact32*  c2h,
                 device const Exact32*  fnm, device const Exact32*  fnp,
                 device const Exact32*  rdnw,
                 device const Exact32*  cqu,
                 device const Exact32*  cqv, int moist_cq,
                 Exact32 cf1, Exact32 cf2, Exact32 cf3,
                int top_lid,
                Exact32 rdx, Exact32 rdy, Exact32 dtau, Exact32 smdiv,
                int spec_zone, int base3d, int nz, int ny, int nx,
                device const Exact32*  mudf,
                device const Exact32*  msfu, device const Exact32*  msfv,
                Exact32 xscale, Exact32 yscale, int use_emdiv, int has_msf, int radiative_x, int radiative_y, uint tid)
{
    
    int nyf = ny + 1, nxf = nx + 1;
    if (tid >= nz * nyf * nxf) return;
    int k = tid / (nyf * nxf);
    int r = tid - k * nyf * nxf;
    int j = r / nxf;
    int i = r - j * nxf;
    ulong st = (ulong)ny * nx;
    if (j < ny
        && j >= spec_zone && j < ny - spec_zone
        && i >= spec_zone && i <= nx - spec_zone
        && !(radiative_x && (i == 0 || i == nx))) {
        int iA = i % nx, iB = (i - 1 + nx) % nx;
        ulong cA = (ulong)j * nx + iA, cB = (ulong)j * nx + iB;
        Exact32 dpxy = pgrad_face(cA, cB, k, nz, st, rdx, ph_pp, php, phb, alt,
                               al_pp, pb, p_pp, p_pp_old, smdiv,
                               mup, mu_pp, mub2d, c1h, c2h,
                               fnm, fnp, rdnw, cf1, cf2, cf3, base3d,
                               top_lid);
        ulong uix = I3S(k, j, i, ny, nxf);
        u_pp[uix] = u_pp[uix] + dtau * ru_t[uix];
        Exact32 cq = moist_cq ? cqu[uix] : Exact32(1.0f);
        u_pp[uix] = u_pp[uix] - dtau * cq * dpxy;
        if (use_emdiv) {
            Exact32 gx = xscale * (mudf[cA] - mudf[cB]);
            if (has_msf) gx = gx / msfu[(ulong)j * nxf + i];
            u_pp[uix] = u_pp[uix] + c1h[k] * gx;
        }
    } else if (j < ny) {
        ulong uix = I3S(k, j, i, ny, nxf);
        u_pp[uix] = arn_add(u_pp[uix], arn_mul(dtau, ru_t[uix]));
    }
    if (i < nx
        && i >= spec_zone && i < nx - spec_zone
        && j >= spec_zone && j <= ny - spec_zone
        && !(radiative_y && (j == 0 || j == ny))) {
        int jA = j % ny, jB = (j - 1 + ny) % ny;
        ulong cA = (ulong)jA * nx + i, cB = (ulong)jB * nx + i;
        Exact32 dpxy = pgrad_face(cA, cB, k, nz, st, rdy, ph_pp, php, phb, alt,
                               al_pp, pb, p_pp, p_pp_old, smdiv,
                               mup, mu_pp, mub2d, c1h, c2h,
                               fnm, fnp, rdnw, cf1, cf2, cf3, base3d,
                               top_lid);
        ulong vix = I3S(k, j, i, nyf, nx);
        v_pp[vix] = v_pp[vix] + dtau * rv_t[vix];
        Exact32 cq = moist_cq ? cqv[vix] : Exact32(1.0f);
        v_pp[vix] = v_pp[vix] - dtau * cq * dpxy;
        if (use_emdiv) {
            Exact32 gy = yscale * (mudf[cA] - mudf[cB]);
            if (has_msf) gy = gy * msfv[(ulong)j * nx + i];
            v_pp[vix] = v_pp[vix] + c1h[k] * gy;
        }
    } else if (i < nx) {
        ulong vix = I3S(k, j, i, nyf, nx);
        v_pp[vix] = arn_add(v_pp[vix], arn_mul(dtau, rv_t[vix]));
    }
}

}
#define A(n) reinterpret_cast<device const Exact32 *>(r.n)
#define O(n) reinterpret_cast<device Exact32 *>(r.n)
kernel void wrf_exact_acoustic_advance_uv(constant WrfExactAcousticAdvanceUvResources &r [[buffer(0)]], constant WrfExactAcousticAdvanceUvParams &q [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
 if (ulong(tid)>=ulong(q.nz)*(q.ny+1)*(q.nx+1)) return;
 wrf_uv::advance_uv(O(u_pp),O(v_pp),A(ru_t),A(rv_t),A(p_pp),A(p_pp_old),A(ph_pp),A(php),A(phb),A(alt),A(al_pp),A(pb),A(mup),A(mu_pp),A(mub2d),A(c1h),A(c2h),A(fnm),A(fnp),A(rdnw),A(cqu),A(cqv),q.moist_cq,Exact32(q.cf1),Exact32(q.cf2),Exact32(q.cf3),q.top_lid,Exact32(q.rdx),Exact32(q.rdy),Exact32(q.dtau),Exact32(q.smdiv),q.spec_zone,q.base3d,q.nz,q.ny,q.nx,A(mudf),A(msfu),A(msfv),Exact32(q.xscale),Exact32(q.yscale),q.use_emdiv,q.has_msf,q.radiative_x,q.radiative_y,tid);
}
#undef A
#undef O
