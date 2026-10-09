#include <metal_stdlib>
#include "smag2d_km_bindings.metalh"
#include "smag_hd_s_bindings.metalh"
#include "smag_hd_u_bindings.metalh"
#include "smag_hd_v_bindings.metalh"
#include "smag_hd_w_bindings.metalh"
#include "smag_zero_boundary_strips_bindings.metalh"
#include "wrf_smag2d_km_bindings.metalh"
#include "wrf_smag_deform_bindings.metalh"
#include "wrf_smag_flux_s_bindings.metalh"
#include "wrf_smag_hd_s_bindings.metalh"
#include "wrf_smag_hd_u_bindings.metalh"
#include "wrf_smag_hd_v_bindings.metalh"
#include "wrf_smag_hd_w_bindings.metalh"
#include "wrf_smag_hd_w_cached_bindings.metalh"
#include "wrf_smag_hd_w_stress_bindings.metalh"
#include "wrf_smag_km_bc_bindings.metalh"
#include "smag2d_couple_map_factor_bindings.metalh"
#include "wrf_smag_surface_heat_const_bindings.metalh"
#include "wrf_smag_surface_scalars_bindings.metalh"
#include "wrf_smag_surface_u_bindings.metalh"
#include "wrf_smag_surface_u_cd0_bindings.metalh"
#include "wrf_smag_surface_v_bindings.metalh"
#include "wrf_smag_surface_v_cd0_bindings.metalh"
#include "wrf_smag_vd_s_bindings.metalh"
#include "wrf_smag_vd_u_bindings.metalh"
#include "wrf_smag_vd_v_bindings.metalh"
#include "wrf_smag_vd_w_bindings.metalh"
#include "wrf_smag_w_primitives_bindings.metalh"
#include "wrf_smag_w_stress_bindings.metalh"
#include "wrf_tke_rhs_bindings.metalh"
#include "bound_tke_bindings.metalh"

using namespace metal;

// Flatten all leading dimensions into planes and mask the last two axes.
// This matches Python slice semantics for 2-D and higher-rank buffers.
kernel void smag_zero_boundary_strips(
    constant SmagZeroBoundaryStripsResources &r [[buffer(0)]],
    constant SmagZeroBoundaryStripsParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    if (int(flat) >= p.size) return;
    const int i = int(flat) % p.nx;
    const int j = (int(flat) / p.nx) % p.ny;
    const bool zero_x = p.boundary_x &&
        (i < p.width || p.x_hi == 0 || i >= p.nx - p.x_hi);
    const bool zero_y = p.boundary_y &&
        (j < p.width || p.y_hi == 0 || j >= p.ny - p.y_hi);
    if (zero_x || zero_y) r.buf[flat] = 0.0f;
}

// The CUDA family is explicitly FP32 (common.cuh:7). Preserve expression
// order and keep contraction disabled for the retained oracle kernels.
#pragma clang fp contract(off)

static inline int sm2_wrap(int i, int n) {
    const int r = i % n;
    return r < 0 ? r + n : r;
}
static inline ulong sm2_i3(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx) + ulong(i);
}
static inline float sm2_chm(device const float *mut, device const float *c1,
                            device const float *c2, int k, int j, int i,
                            int nx) {
    return c1[k] * mut[ulong(j) * ulong(nx) + ulong(i)] + c2[k];
}
static inline float sm2_d12(device const float *u, device const float *v,
                            float rdx, float rdy, int k, int j, int i,
                            int ny, int nx) {
    const int jm = sm2_wrap(j - 1, ny), im = sm2_wrap(i - 1, nx);
    const float du = u[sm2_i3(k, j, i, ny, nx + 1)]
                   - u[sm2_i3(k, jm, i, ny, nx + 1)];
    const float dv = v[sm2_i3(k, j, i, ny + 1, nx)]
                   - v[sm2_i3(k, j, im, ny + 1, nx)];
    return (rdy * du) + (rdx * dv);
}

struct WrfSmagGrid {
    device const float *u, *v, *w, *php, *phb, *alt, *qv;
    device const float *msft, *msfu, *msfv, *fnm, *fnp, *dn, *dnw;
    float rdx, rdy, dx, dy, cf1, cf2, cf3;
    int nz, ny, nx, phb3d, boundary_x, boundary_y, moist;
};
struct WrfSmagScalarField {
    device const float *f;
    device const float *thb;
    int full_theta, thb3d;
};

template <typename R, typename P>
static inline WrfSmagGrid wrf_sm2_grid(constant R &r, constant P &p) {
    WrfSmagGrid q;
    q.u=r.u; q.v=r.v; q.w=r.w; q.php=r.php; q.phb=r.phb;
    q.alt=r.alt; q.qv=r.qv; q.msft=r.msft; q.msfu=r.msfu; q.msfv=r.msfv;
    q.fnm=r.fnm; q.fnp=r.fnp; q.dn=r.dn; q.dnw=r.dnw;
    q.rdx=p.rdx; q.rdy=p.rdy; q.dx=p.dx; q.dy=p.dy;
    q.cf1=p.cf1; q.cf2=p.cf2; q.cf3=p.cf3;
    q.nz=p.nz; q.ny=p.ny; q.nx=p.nx; q.phb3d=p.phb3d;
    q.boundary_x=p.boundary_x; q.boundary_y=p.boundary_y; q.moist=p.moist;
    return q;
}
static inline int wrf_sm2_ix(thread const WrfSmagGrid &q, int i) {
    return q.boundary_x ? clamp(i,0,q.nx-1) : sm2_wrap(i,q.nx);
}
static inline int wrf_sm2_iy(thread const WrfSmagGrid &q, int j) {
    return q.boundary_y ? clamp(j,0,q.ny-1) : sm2_wrap(j,q.ny);
}
static inline int wrf_sm2_iu(thread const WrfSmagGrid &q, int i) {
    return q.boundary_x ? clamp(i,0,q.nx) : sm2_wrap(i,q.nx);
}
static inline int wrf_sm2_jv(thread const WrfSmagGrid &q, int j) {
    return q.boundary_y ? clamp(j,0,q.ny) : sm2_wrap(j,q.ny);
}
static inline float wrf_sm2_phi(thread const WrfSmagGrid &q, int kw, int j, int i) {
    const int ii=wrf_sm2_ix(q,i), jj=wrf_sm2_iy(q,j);
    kw=clamp(kw,0,q.nz);
    const ulong s=ulong(q.ny)*ulong(q.nx), c=ulong(jj)*ulong(q.nx)+ulong(ii);
    const float base=q.phb3d ? q.phb[ulong(kw)*s+c] : q.phb[kw];
    return base+q.php[ulong(kw)*s+c];
}
static inline float wrf_sm2_rdzw(thread const WrfSmagGrid &q, int k, int j, int i) {
    k=clamp(k,0,q.nz-1);
    return 9.81f/(wrf_sm2_phi(q,k+1,j,i)-wrf_sm2_phi(q,k,j,i));
}
static inline float wrf_sm2_rdz(thread const WrfSmagGrid &q, int kw, int j, int i) {
    if(kw<=0) return 2.0f*9.81f/(wrf_sm2_phi(q,1,j,i)-wrf_sm2_phi(q,0,j,i));
    if(kw>=q.nz) return 2.0f*9.81f/(wrf_sm2_phi(q,q.nz,j,i)-wrf_sm2_phi(q,q.nz-1,j,i));
    return 2.0f*9.81f/(wrf_sm2_phi(q,kw+1,j,i)-wrf_sm2_phi(q,kw-1,j,i));
}
static inline float wrf_sm2_zx(thread const WrfSmagGrid &q, int kw, int j, int iface) {
    if(q.boundary_x && iface<=0) return 0.0f;
    if(q.boundary_x && iface>=q.nx) iface=q.nx-1;
    return q.rdx*(wrf_sm2_phi(q,kw,j,iface)-wrf_sm2_phi(q,kw,j,iface-1))/9.81f;
}
static inline float wrf_sm2_zy(thread const WrfSmagGrid &q, int kw, int jface, int i) {
    if(q.boundary_y && jface<=0) return 0.0f;
    if(q.boundary_y && jface>=q.ny) jface=q.ny-1;
    return q.rdy*(wrf_sm2_phi(q,kw,jface,i)-wrf_sm2_phi(q,kw,jface-1,i))/9.81f;
}
static inline float wrf_sm2_uhat(thread const WrfSmagGrid &q,int k,int j,int i) {
    const int jj=wrf_sm2_iy(q,j),ii=wrf_sm2_iu(q,i);
    return q.u[sm2_i3(k,jj,ii,q.ny,q.nx+1)]/q.msfu[ulong(jj)*ulong(q.nx+1)+ulong(ii)];
}
static inline float wrf_sm2_vhat(thread const WrfSmagGrid &q,int k,int j,int i) {
    const int jj=wrf_sm2_jv(q,j),ii=wrf_sm2_ix(q,i);
    return q.v[sm2_i3(k,jj,ii,q.ny+1,q.nx)]/q.msfv[ulong(jj)*ulong(q.nx)+ulong(ii)];
}
static inline float wrf_sm2_what(thread const WrfSmagGrid &q,int k,int j,int i) {
    const int jj=wrf_sm2_iy(q,j),ii=wrf_sm2_ix(q,i);
    return q.w[sm2_i3(k,jj,ii,q.ny,q.nx)]/q.msft[ulong(jj)*ulong(q.nx)+ulong(ii)];
}
static inline float wrf_sm2_weights(thread const WrfSmagGrid &q,int kw,float p0,float p1,float p2,
                                    float plast,float pprev,float pcur,float pbelow) {
    if(kw<=0) return q.cf1*p0+q.cf2*p1+q.cf3*p2;
    if(kw>=q.nz) {
        const float cft2=-0.5f*q.dnw[q.nz-1]/q.dn[q.nz-1];
        return (1.0f-cft2)*plast+cft2*pprev;
    }
    return q.fnm[kw]*pcur+q.fnp[kw]*pbelow;
}
static inline float wrf_sm2_uwx(thread const WrfSmagGrid &q,int kw,int j,int i) {
    const float p0=wrf_sm2_uhat(q,0,j,i)+wrf_sm2_uhat(q,0,j,i+1);
    const float p1=wrf_sm2_uhat(q,1,j,i)+wrf_sm2_uhat(q,1,j,i+1);
    const float p2=wrf_sm2_uhat(q,2,j,i)+wrf_sm2_uhat(q,2,j,i+1);
    const float pl=wrf_sm2_uhat(q,q.nz-1,j,i)+wrf_sm2_uhat(q,q.nz-1,j,i+1);
    const float pp=wrf_sm2_uhat(q,q.nz-2,j,i)+wrf_sm2_uhat(q,q.nz-2,j,i+1);
    const int kc=kw>=q.nz?q.nz-1:kw,kb=max(kc-1,0);
    const float pc=wrf_sm2_uhat(q,kc,j,i)+wrf_sm2_uhat(q,kc,j,i+1);
    const float pb=wrf_sm2_uhat(q,kb,j,i)+wrf_sm2_uhat(q,kb,j,i+1);
    return 0.5f*wrf_sm2_weights(q,kw,p0,p1,p2,pl,pp,pc,pb);
}
static inline float wrf_sm2_vwy(thread const WrfSmagGrid &q,int kw,int j,int i) {
    const float p0=wrf_sm2_vhat(q,0,j,i)+wrf_sm2_vhat(q,0,j+1,i);
    const float p1=wrf_sm2_vhat(q,1,j,i)+wrf_sm2_vhat(q,1,j+1,i);
    const float p2=wrf_sm2_vhat(q,2,j,i)+wrf_sm2_vhat(q,2,j+1,i);
    const float pl=wrf_sm2_vhat(q,q.nz-1,j,i)+wrf_sm2_vhat(q,q.nz-1,j+1,i);
    const float pp=wrf_sm2_vhat(q,q.nz-2,j,i)+wrf_sm2_vhat(q,q.nz-2,j+1,i);
    const int kc=kw>=q.nz?q.nz-1:kw,kb=max(kc-1,0);
    const float pc=wrf_sm2_vhat(q,kc,j,i)+wrf_sm2_vhat(q,kc,j+1,i);
    const float pb=wrf_sm2_vhat(q,kb,j,i)+wrf_sm2_vhat(q,kb,j+1,i);
    return 0.5f*wrf_sm2_weights(q,kw,p0,p1,p2,pl,pp,pc,pb);
}
static inline float wrf_sm2_uwc(thread const WrfSmagGrid &q,int kw,int j,int i) {
    const float p0=wrf_sm2_uhat(q,0,j-1,i)+wrf_sm2_uhat(q,0,j,i);
    const float p1=wrf_sm2_uhat(q,1,j-1,i)+wrf_sm2_uhat(q,1,j,i);
    const float p2=wrf_sm2_uhat(q,2,j-1,i)+wrf_sm2_uhat(q,2,j,i);
    const float pl=wrf_sm2_uhat(q,q.nz-1,j-1,i)+wrf_sm2_uhat(q,q.nz-1,j,i);
    const float pp=wrf_sm2_uhat(q,q.nz-2,j-1,i)+wrf_sm2_uhat(q,q.nz-2,j,i);
    const int kc=kw>=q.nz?q.nz-1:kw,kb=max(kc-1,0);
    const float pc=wrf_sm2_uhat(q,kc,j-1,i)+wrf_sm2_uhat(q,kc,j,i);
    const float pb=wrf_sm2_uhat(q,kb,j-1,i)+wrf_sm2_uhat(q,kb,j,i);
    return 0.5f*wrf_sm2_weights(q,kw,p0,p1,p2,pl,pp,pc,pb);
}
static inline float wrf_sm2_vwc(thread const WrfSmagGrid &q,int kw,int j,int i) {
    const float p0=wrf_sm2_vhat(q,0,j,i-1)+wrf_sm2_vhat(q,0,j,i);
    const float p1=wrf_sm2_vhat(q,1,j,i-1)+wrf_sm2_vhat(q,1,j,i);
    const float p2=wrf_sm2_vhat(q,2,j,i-1)+wrf_sm2_vhat(q,2,j,i);
    const float pl=wrf_sm2_vhat(q,q.nz-1,j,i-1)+wrf_sm2_vhat(q,q.nz-1,j,i);
    const float pp=wrf_sm2_vhat(q,q.nz-2,j,i-1)+wrf_sm2_vhat(q,q.nz-2,j,i);
    const int kc=kw>=q.nz?q.nz-1:kw,kb=max(kc-1,0);
    const float pc=wrf_sm2_vhat(q,kc,j,i-1)+wrf_sm2_vhat(q,kc,j,i);
    const float pb=wrf_sm2_vhat(q,kb,j,i-1)+wrf_sm2_vhat(q,kb,j,i);
    return 0.5f*wrf_sm2_weights(q,kw,p0,p1,p2,pl,pp,pc,pb);
}
static inline float wrf_sm2_d11(thread const WrfSmagGrid &q,int k,int j,int i) {
    const float zx=0.25f*(wrf_sm2_zx(q,k,j,i)+wrf_sm2_zx(q,k,j,i+1)+wrf_sm2_zx(q,k+1,j,i)+wrf_sm2_zx(q,k+1,j,i+1));
    const float slope=(wrf_sm2_uwx(q,k+1,j,i)-wrf_sm2_uwx(q,k,j,i))*zx*wrf_sm2_rdzw(q,k,j,i);
    const float mm=q.msft[ulong(wrf_sm2_iy(q,j))*ulong(q.nx)+ulong(wrf_sm2_ix(q,i))];
    return 2.0f*mm*mm*(q.rdx*(wrf_sm2_uhat(q,k,j,i+1)-wrf_sm2_uhat(q,k,j,i))-slope);
}
static inline float wrf_sm2_d22(thread const WrfSmagGrid &q,int k,int j,int i) {
    const float zy=0.25f*(wrf_sm2_zy(q,k,j,i)+wrf_sm2_zy(q,k,j+1,i)+wrf_sm2_zy(q,k+1,j,i)+wrf_sm2_zy(q,k+1,j+1,i));
    const float slope=(wrf_sm2_vwy(q,k+1,j,i)-wrf_sm2_vwy(q,k,j,i))*zy*wrf_sm2_rdzw(q,k,j,i);
    const float mm=q.msft[ulong(wrf_sm2_iy(q,j))*ulong(q.nx)+ulong(wrf_sm2_ix(q,i))];
    return 2.0f*mm*mm*(q.rdy*(wrf_sm2_vhat(q,k,j+1,i)-wrf_sm2_vhat(q,k,j,i))-slope);
}
static inline float wrf_sm2_d12(thread const WrfSmagGrid &q,int k,int j,int i) {
    if(q.boundary_x)i=(i<=0?(q.nx>1?1:0):(i>=q.nx?q.nx-1:i));
    if(q.boundary_y)j=(j<=0?(q.ny>1?1:0):(j>=q.ny?q.ny-1:j));
    const int iu=wrf_sm2_iu(q,i),jm=wrf_sm2_iy(q,j-1),jc=wrf_sm2_iy(q,j);
    const int iv=wrf_sm2_ix(q,i),im=wrf_sm2_ix(q,i-1),jv=wrf_sm2_jv(q,j);
    const float mm=0.25f*(q.msfu[ulong(jm)*ulong(q.nx+1)+ulong(iu)]+q.msfu[ulong(jc)*ulong(q.nx+1)+ulong(iu)])
        *(q.msfv[ulong(jv)*ulong(q.nx)+ulong(im)]+q.msfv[ulong(jv)*ulong(q.nx)+ulong(iv)]);
    const float rr=wrf_sm2_rdzw(q,k,j,i)+wrf_sm2_rdzw(q,k,j,i-1)+wrf_sm2_rdzw(q,k,j-1,i-1)+wrf_sm2_rdzw(q,k,j-1,i);
    const float zy=0.25f*(wrf_sm2_zy(q,k,j,i-1)+wrf_sm2_zy(q,k,j,i)+wrf_sm2_zy(q,k+1,j,i-1)+wrf_sm2_zy(q,k+1,j,i));
    const float uslope=(wrf_sm2_uwc(q,k+1,j,i)-wrf_sm2_uwc(q,k,j,i))*0.25f*zy*rr;
    const float zx=0.25f*(wrf_sm2_zx(q,k,j-1,i)+wrf_sm2_zx(q,k,j,i)+wrf_sm2_zx(q,k+1,j-1,i)+wrf_sm2_zx(q,k+1,j,i));
    const float vslope=(wrf_sm2_vwc(q,k+1,j,i)-wrf_sm2_vwc(q,k,j,i))*0.25f*zx*rr;
    return mm*(q.rdy*(wrf_sm2_uhat(q,k,j,i)-wrf_sm2_uhat(q,k,j-1,i))-uslope
              +q.rdx*(wrf_sm2_vhat(q,k,j,i)-wrf_sm2_vhat(q,k,j,i-1))-vslope);
}

kernel void smag2d_km(constant Smag2DKmResources &r [[buffer(0)]],
                      constant Smag2DKmParams &p [[buffer(1)]],
                      uint flat [[thread_position_in_grid]]) {
    const ulong count = ulong(p.nz) * ulong(p.ny) * ulong(p.nx);
    if (ulong(flat) >= count) return;
    const int i = int(flat % uint(p.nx));
    const int j = int((flat / uint(p.nx)) % uint(p.ny));
    const int k = int(flat / uint(p.nx * p.ny));
    const int ip = sm2_wrap(i + 1, p.nx), jp = sm2_wrap(j + 1, p.ny);
    const float d11 = 2.0f * p.rdx *
        (r.u[sm2_i3(k,j,ip,p.ny,p.nx+1)]-r.u[sm2_i3(k,j,i,p.ny,p.nx+1)]);
    const float d22 = 2.0f * p.rdy *
        (r.v[sm2_i3(k,jp,i,p.ny+1,p.nx)]-r.v[sm2_i3(k,j,i,p.ny+1,p.nx)]);
    const float s12 = 0.25f * (sm2_d12(r.u,r.v,p.rdx,p.rdy,k,j,i,p.ny,p.nx)
        + sm2_d12(r.u,r.v,p.rdx,p.rdy,k,j,ip,p.ny,p.nx)
        + sm2_d12(r.u,r.v,p.rdx,p.rdy,k,jp,i,p.ny,p.nx)
        + sm2_d12(r.u,r.v,p.rdx,p.rdy,k,jp,ip,p.ny,p.nx));
    const float delta = d11-d22;
    const float def2 = 0.25f * delta * delta + s12 * s12;
    float km = p.c_s * p.c_s * p.mlen * p.mlen * sqrt(def2);
    km = min(km, 10.0f * p.mlen);
    const ulong q = sm2_i3(k,j,i,p.ny,p.nx);
    r.xkmh[q] = km;
    r.xkhh[q] = km / p.prandtl;
}

kernel void smag_hd_s(constant SmagHdSResources &r [[buffer(0)]],
                      constant SmagHdSParams &p [[buffer(1)]],
                      uint flat [[thread_position_in_grid]]) {
    const ulong count = ulong(p.nz) * ulong(p.ny) * ulong(p.nx);
    if (ulong(flat) >= count) return;
    const int i=int(flat%uint(p.nx)), j=int((flat/uint(p.nx))%uint(p.ny));
    const int k=int(flat/uint(p.nx*p.ny));
    const int ip=sm2_wrap(i+1,p.nx), im=sm2_wrap(i-1,p.nx);
    const int jp=sm2_wrap(j+1,p.ny), jm=sm2_wrap(j-1,p.ny);
    const float fc=r.f[sm2_i3(k,j,i,p.ny,p.nx)];
    const float kmxm=0.5f*(r.xk[sm2_i3(k,j,i,p.ny,p.nx)]+r.xk[sm2_i3(k,j,im,p.ny,p.nx)])
        *0.5f*(sm2_chm(r.mut,r.c1,r.c2,k,j,i,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,j,im,p.nx))*p.rdx;
    const float kmxp=0.5f*(r.xk[sm2_i3(k,j,ip,p.ny,p.nx)]+r.xk[sm2_i3(k,j,i,p.ny,p.nx)])
        *0.5f*(sm2_chm(r.mut,r.c1,r.c2,k,j,ip,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,j,i,p.nx))*p.rdx;
    const float kmym=0.5f*(r.xk[sm2_i3(k,j,i,p.ny,p.nx)]+r.xk[sm2_i3(k,jm,i,p.ny,p.nx)])
        *0.5f*(sm2_chm(r.mut,r.c1,r.c2,k,j,i,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,jm,i,p.nx))*p.rdy;
    const float kmyp=0.5f*(r.xk[sm2_i3(k,jp,i,p.ny,p.nx)]+r.xk[sm2_i3(k,j,i,p.ny,p.nx)])
        *0.5f*(sm2_chm(r.mut,r.c1,r.c2,k,jp,i,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,j,i,p.nx))*p.rdy;
    r.tend[sm2_i3(k,j,i,p.ny,p.nx)] +=
        p.rdx*(kmxp*(r.f[sm2_i3(k,j,ip,p.ny,p.nx)]-fc)-kmxm*(fc-r.f[sm2_i3(k,j,im,p.ny,p.nx)]))
       +p.rdy*(kmyp*(r.f[sm2_i3(k,jp,i,p.ny,p.nx)]-fc)-kmym*(fc-r.f[sm2_i3(k,jm,i,p.ny,p.nx)]));
}

kernel void smag_hd_u(constant SmagHdUResources &r [[buffer(0)]],
                      constant SmagHdUParams &p [[buffer(1)]],
                      uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny)*ulong(p.nx+1);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx+1)),j=int((flat/uint(p.nx+1))%uint(p.ny));
    const int k=int(flat/uint((p.nx+1)*p.ny));
    const int ic=sm2_wrap(i,p.nx), im=sm2_wrap(ic-1,p.nx);
    int ip=sm2_wrap(ic+1,p.nx); if(p.open_x&&ic==p.nx-1)ip=p.nx;
    const int jp=sm2_wrap(j+1,p.ny),jm=sm2_wrap(j-1,p.ny);
    const float uc=r.u[sm2_i3(k,j,ic,p.ny,p.nx+1)];
    const float kmxm=sm2_chm(r.mut,r.c1,r.c2,k,j,im,p.nx)*r.xk[sm2_i3(k,j,im,p.ny,p.nx)]*p.rdx;
    const float kmxp=sm2_chm(r.mut,r.c1,r.c2,k,j,ic,p.nx)*r.xk[sm2_i3(k,j,ic,p.ny,p.nx)]*p.rdx;
    const float kmym=0.25f*(sm2_chm(r.mut,r.c1,r.c2,k,j,ic,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,jm,ic,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,jm,im,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,j,im,p.nx))
        *0.25f*(r.xk[sm2_i3(k,j,ic,p.ny,p.nx)]+r.xk[sm2_i3(k,jm,ic,p.ny,p.nx)]+r.xk[sm2_i3(k,jm,im,p.ny,p.nx)]+r.xk[sm2_i3(k,j,im,p.ny,p.nx)])*p.rdy;
    const float kmyp=0.25f*(sm2_chm(r.mut,r.c1,r.c2,k,jp,ic,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,j,ic,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,j,im,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,jp,im,p.nx))
        *0.25f*(r.xk[sm2_i3(k,jp,ic,p.ny,p.nx)]+r.xk[sm2_i3(k,j,ic,p.ny,p.nx)]+r.xk[sm2_i3(k,j,im,p.ny,p.nx)]+r.xk[sm2_i3(k,jp,im,p.ny,p.nx)])*p.rdy;
    r.tend[sm2_i3(k,j,i,p.ny,p.nx+1)] += p.rdx*(kmxp*(r.u[sm2_i3(k,j,ip,p.ny,p.nx+1)]-uc)-kmxm*(uc-r.u[sm2_i3(k,j,im,p.ny,p.nx+1)]))
        +p.rdy*(kmyp*(r.u[sm2_i3(k,jp,ic,p.ny,p.nx+1)]-uc)-kmym*(uc-r.u[sm2_i3(k,jm,ic,p.ny,p.nx+1)]));
}

kernel void smag_hd_v(constant SmagHdVResources &r [[buffer(0)]],
                      constant SmagHdVParams &p [[buffer(1)]],
                      uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny+1)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny+1));
    const int k=int(flat/uint(p.nx*(p.ny+1)));
    const int jc=sm2_wrap(j,p.ny),jm=sm2_wrap(jc-1,p.ny);
    int jp=sm2_wrap(jc+1,p.ny); if(p.open_y&&jc==p.ny-1)jp=p.ny;
    const int ip=sm2_wrap(i+1,p.nx),im=sm2_wrap(i-1,p.nx);
    const float vc=r.v[sm2_i3(k,jc,i,p.ny+1,p.nx)];
    const float kmym=r.xk[sm2_i3(k,jm,i,p.ny,p.nx)]*p.rdy;
    const float kmyp=r.xk[sm2_i3(k,jc,i,p.ny,p.nx)]*p.rdy;
    const float kmxm=0.25f*(sm2_chm(r.mut,r.c1,r.c2,k,jc,i,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,jc,im,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,jm,im,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,jm,i,p.nx))
        *0.25f*(r.xk[sm2_i3(k,jc,i,p.ny,p.nx)]+r.xk[sm2_i3(k,jc,im,p.ny,p.nx)]+r.xk[sm2_i3(k,jm,im,p.ny,p.nx)]+r.xk[sm2_i3(k,jm,i,p.ny,p.nx)])*p.rdx;
    const float kmxp=0.25f*(sm2_chm(r.mut,r.c1,r.c2,k,jc,ip,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,jc,i,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,jm,i,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,jm,ip,p.nx))
        *0.25f*(r.xk[sm2_i3(k,jc,ip,p.ny,p.nx)]+r.xk[sm2_i3(k,jc,i,p.ny,p.nx)]+r.xk[sm2_i3(k,jm,i,p.ny,p.nx)]+r.xk[sm2_i3(k,jm,ip,p.ny,p.nx)])*p.rdx;
    r.tend[sm2_i3(k,j,i,p.ny+1,p.nx)] += p.rdx*(kmxp*(r.v[sm2_i3(k,jc,ip,p.ny+1,p.nx)]-vc)-kmxm*(vc-r.v[sm2_i3(k,jc,im,p.ny+1,p.nx)]))
        +p.rdy*(kmyp*(r.v[sm2_i3(k,jp,i,p.ny+1,p.nx)]-vc)-kmym*(vc-r.v[sm2_i3(k,jm,i,p.ny+1,p.nx)]));
}

kernel void smag_hd_w(constant SmagHdWResources &r [[buffer(0)]],
                      constant SmagHdWParams &p [[buffer(1)]],
                      uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz+1)*ulong(p.ny)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny));
    const int k=int(flat/uint(p.nx*p.ny));
    if(k==0||k==p.nz)return;
    const int ip=sm2_wrap(i+1,p.nx),im=sm2_wrap(i-1,p.nx),jp=sm2_wrap(j+1,p.ny),jm=sm2_wrap(j-1,p.ny);
    const float wc=r.w[sm2_i3(k,j,i,p.ny,p.nx)];
    const float kmxm=0.5f*(sm2_chm(r.mut,r.c1,r.c2,k,j,i,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,j,im,p.nx))*0.25f*(r.xk[sm2_i3(k,j,i,p.ny,p.nx)]+r.xk[sm2_i3(k,j,im,p.ny,p.nx)]+r.xk[sm2_i3(k-1,j,i,p.ny,p.nx)]+r.xk[sm2_i3(k-1,j,im,p.ny,p.nx)])*p.rdx;
    const float kmxp=0.5f*(sm2_chm(r.mut,r.c1,r.c2,k,j,ip,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,j,i,p.nx))*0.25f*(r.xk[sm2_i3(k,j,ip,p.ny,p.nx)]+r.xk[sm2_i3(k,j,i,p.ny,p.nx)]+r.xk[sm2_i3(k-1,j,ip,p.ny,p.nx)]+r.xk[sm2_i3(k-1,j,i,p.ny,p.nx)])*p.rdx;
    const float kmym=0.5f*(sm2_chm(r.mut,r.c1,r.c2,k,j,i,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,jm,i,p.nx))*0.25f*(r.xk[sm2_i3(k,j,i,p.ny,p.nx)]+r.xk[sm2_i3(k,jm,i,p.ny,p.nx)]+r.xk[sm2_i3(k-1,j,i,p.ny,p.nx)]+r.xk[sm2_i3(k-1,jm,i,p.ny,p.nx)])*p.rdy;
    const float kmyp=0.5f*(sm2_chm(r.mut,r.c1,r.c2,k,jp,i,p.nx)+sm2_chm(r.mut,r.c1,r.c2,k,j,i,p.nx))*0.25f*(r.xk[sm2_i3(k,jp,i,p.ny,p.nx)]+r.xk[sm2_i3(k,j,i,p.ny,p.nx)]+r.xk[sm2_i3(k-1,jp,i,p.ny,p.nx)]+r.xk[sm2_i3(k-1,j,i,p.ny,p.nx)])*p.rdy;
    r.tend[sm2_i3(k,j,i,p.ny,p.nx)] += p.rdx*(kmxp*(r.w[sm2_i3(k,j,ip,p.ny,p.nx)]-wc)-kmxm*(wc-r.w[sm2_i3(k,j,im,p.ny,p.nx)]))
        +p.rdy*(kmyp*(r.w[sm2_i3(k,jp,i,p.ny,p.nx)]-wc)-kmym*(wc-r.w[sm2_i3(k,jm,i,p.ny,p.nx)]));
}

static inline float wrf_sm2_d(device const float *d, thread const WrfSmagGrid &q,
                              int k,int j,int i) {
    return d[sm2_i3(k,wrf_sm2_iy(q,j),wrf_sm2_ix(q,i),q.ny,q.nx)];
}

kernel void wrf_smag_deform(constant WrfSmagDeformResources &r [[buffer(0)]],
                            constant WrfSmagDeformParams &p [[buffer(1)]],
                            uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny));
    const int k=int(flat/uint(p.nx*p.ny));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const ulong ix=sm2_i3(k,j,i,p.ny,p.nx);
    r.d11[ix]=wrf_sm2_d11(q,k,j,i);
    r.d22[ix]=wrf_sm2_d22(q,k,j,i);
    r.d12[ix]=wrf_sm2_d12(q,k,j,i);
}

kernel void wrf_smag2d_km(constant WrfSmag2DKmResources &r [[buffer(0)]],
                           constant WrfSmag2DKmParams &p [[buffer(1)]],
                           uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny));
    const int k=int(flat/uint(p.nx*p.ny));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const float d11=wrf_sm2_d(r.d11,q,k,j,i),d22=wrf_sm2_d(r.d22,q,k,j,i);
    const float d12=0.25f*(wrf_sm2_d(r.d12,q,k,j,i)+wrf_sm2_d(r.d12,q,k,j+1,i)
        +wrf_sm2_d(r.d12,q,k,j,i+1)+wrf_sm2_d(r.d12,q,k,j+1,i+1));
    const float delta=d11-d22;
    const float strain=sqrt(0.25f*delta*delta+d12*d12);
    const float map=r.msft[ulong(j)*ulong(p.nx)+ulong(i)];
    const float dxm=p.dx/map,dym=p.dy/map;
    const float mlen=sqrt(dxm*dym);
    float km=min(p.c_s*p.c_s*mlen*mlen*strain,10.0f*mlen);
    const float zx=0.25f*(fabs(wrf_sm2_zx(q,k,j,i))+fabs(wrf_sm2_zx(q,k,j,i+1))
        +fabs(wrf_sm2_zx(q,k+1,j,i))+fabs(wrf_sm2_zx(q,k+1,j,i+1)))
        *wrf_sm2_rdzw(q,k,j,i)*dxm;
    const float zy=0.25f*(fabs(wrf_sm2_zy(q,k,j,i))+fabs(wrf_sm2_zy(q,k,j+1,i))
        +fabs(wrf_sm2_zy(q,k+1,j,i))+fabs(wrf_sm2_zy(q,k+1,j+1,i)))
        *wrf_sm2_rdzw(q,k,j,i)*dym;
    const float alpha=max(sqrt(zx*zx+zy*zy),1.0f);
    const float def_limit=max(10.0f/mlen,1.0e-3f);
    km /= strain>def_limit ? alpha*alpha : alpha;
    const ulong ix=sm2_i3(k,j,i,p.ny,p.nx);
    r.xkmh[ix]=km;
    r.xkhh[ix]=km/p.prandtl;
}

kernel void wrf_smag_km_bc(constant WrfSmagKmBcResources &r [[buffer(0)]],
                           constant WrfSmagKmBcParams &p [[buffer(1)]],
                           uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny));
    const int k=int(flat/uint(p.nx*p.ny));
    if((p.boundary_x&&(i==0||i==p.nx-1))||(p.boundary_y&&(j==0||j==p.ny-1))){
        const ulong ix=sm2_i3(k,j,i,p.ny,p.nx);r.xkmh[ix]=0.0f;r.xkhh[ix]=0.0f;
    }
}


static inline float wrf_sm2_rho(thread const WrfSmagGrid &q,int k,int j,int i) {
    const int jj=wrf_sm2_iy(q,j),ii=wrf_sm2_ix(q,i);
    const ulong ix=sm2_i3(k,jj,ii,q.ny,q.nx);
    return q.moist ? (1.0f+q.qv[ix])/q.alt[ix] : 1.0f/q.alt[ix];
}
static inline float wrf_sm2_k(thread const WrfSmagGrid &q,device const float *km,int k,int j,int i) {
    return km[sm2_i3(k,wrf_sm2_iy(q,j),wrf_sm2_ix(q,i),q.ny,q.nx)];
}
static inline float wrf_sm2_tau11(thread const WrfSmagGrid &q,device const float *km,
                                  device const float *d11,int k,int j,int i) {
    return -wrf_sm2_rho(q,k,j,i)*wrf_sm2_k(q,km,k,j,i)*wrf_sm2_d(d11,q,k,j,i);
}
static inline float wrf_sm2_tau22(thread const WrfSmagGrid &q,device const float *km,
                                  device const float *d22,int k,int j,int i) {
    return -wrf_sm2_rho(q,k,j,i)*wrf_sm2_k(q,km,k,j,i)*wrf_sm2_d(d22,q,k,j,i);
}
static inline float wrf_sm2_tau12(thread const WrfSmagGrid &q,device const float *km,
                                  device const float *d12,int k,int j,int i) {
    const float rho=0.25f*(wrf_sm2_rho(q,k,j,i)+wrf_sm2_rho(q,k,j,i-1)
        +wrf_sm2_rho(q,k,j-1,i-1)+wrf_sm2_rho(q,k,j-1,i));
    const float kval=0.25f*(wrf_sm2_k(q,km,k,j,i)+wrf_sm2_k(q,km,k,j,i-1)
        +wrf_sm2_k(q,km,k,j-1,i-1)+wrf_sm2_k(q,km,k,j-1,i));
    return -rho*kval*wrf_sm2_d(d12,q,k,j,i);
}
static inline float wrf_sm2_tau11_uavg(thread const WrfSmagGrid &q,device const float *km,
                                       device const float *d11,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    return 0.5f*(q.fnm[kw]*(wrf_sm2_tau11(q,km,d11,kw,j,i-1)+wrf_sm2_tau11(q,km,d11,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_tau11(q,km,d11,kw-1,j,i-1)+wrf_sm2_tau11(q,km,d11,kw-1,j,i)));
}
static inline float wrf_sm2_tau12_uavg(thread const WrfSmagGrid &q,device const float *km,
                                       device const float *d12,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    return 0.5f*(q.fnm[kw]*(wrf_sm2_tau12(q,km,d12,kw,j+1,i)+wrf_sm2_tau12(q,km,d12,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_tau12(q,km,d12,kw-1,j+1,i)+wrf_sm2_tau12(q,km,d12,kw-1,j,i)));
}
static inline float wrf_sm2_tau12_vavg(thread const WrfSmagGrid &q,device const float *km,
                                       device const float *d12,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    return 0.5f*(q.fnm[kw]*(wrf_sm2_tau12(q,km,d12,kw,j,i+1)+wrf_sm2_tau12(q,km,d12,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_tau12(q,km,d12,kw-1,j,i+1)+wrf_sm2_tau12(q,km,d12,kw-1,j,i)));
}
static inline float wrf_sm2_tau22_vavg(thread const WrfSmagGrid &q,device const float *km,
                                       device const float *d22,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    return 0.5f*(q.fnm[kw]*(wrf_sm2_tau22(q,km,d22,kw,j-1,i)+wrf_sm2_tau22(q,km,d22,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_tau22(q,km,d22,kw-1,j-1,i)+wrf_sm2_tau22(q,km,d22,kw-1,j,i)));
}

static inline float wrf_sm2_wxavg(thread const WrfSmagGrid &q,int k,int j,int i) {
    return 0.25f*(wrf_sm2_what(q,k,j,i)+wrf_sm2_what(q,k+1,j,i)
        +wrf_sm2_what(q,k,j,i-1)+wrf_sm2_what(q,k+1,j,i-1));
}
static inline float wrf_sm2_wyavg(thread const WrfSmagGrid &q,int k,int j,int i) {
    return 0.25f*(wrf_sm2_what(q,k,j,i)+wrf_sm2_what(q,k+1,j,i)
        +wrf_sm2_what(q,k,j-1,i)+wrf_sm2_what(q,k+1,j-1,i));
}
static inline float wrf_sm2_d13(thread const WrfSmagGrid &q,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    if(q.boundary_x)i=(i<=0?(q.nx>1?1:0):(i>=q.nx?q.nx-1:i));
    if(q.boundary_y)j=(j<=0?(q.ny>1?1:0):(j>=q.ny?q.ny-1:j));
    const int jj=wrf_sm2_iy(q,j),iu=wrf_sm2_iu(q,i);
    const float msf=q.msfu[ulong(jj)*ulong(q.nx+1)+ulong(iu)];
    const float rz=0.5f*(wrf_sm2_rdz(q,kw,j,i)+wrf_sm2_rdz(q,kw,j,i-1));
    const float slope=(wrf_sm2_wxavg(q,kw,j,i)-wrf_sm2_wxavg(q,kw-1,j,i))*wrf_sm2_zx(q,kw,j,i)*rz;
    const float dwdx=msf*msf*(q.rdx*(wrf_sm2_what(q,kw,j,i)-wrf_sm2_what(q,kw,j,i-1))-slope);
    const float dudz=(q.u[sm2_i3(kw,jj,iu,q.ny,q.nx+1)]-q.u[sm2_i3(kw-1,jj,iu,q.ny,q.nx+1)])*rz;
    return dwdx+dudz;
}
static inline float wrf_sm2_d23(thread const WrfSmagGrid &q,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    if(q.boundary_x)i=(i<=0?(q.nx>1?1:0):(i>=q.nx?q.nx-1:i));
    if(q.boundary_y)j=(j<=0?(q.ny>1?1:0):(j>=q.ny?q.ny-1:j));
    const int jv=wrf_sm2_jv(q,j),ii=wrf_sm2_ix(q,i);
    const float msf=q.msfv[ulong(jv)*ulong(q.nx)+ulong(ii)];
    const float rz=0.5f*(wrf_sm2_rdz(q,kw,j,i)+wrf_sm2_rdz(q,kw,j-1,i));
    const float slope=(wrf_sm2_wyavg(q,kw,j,i)-wrf_sm2_wyavg(q,kw-1,j,i))*wrf_sm2_zy(q,kw,j,i)*rz;
    const float dwdy=msf*msf*(q.rdy*(wrf_sm2_what(q,kw,j,i)-wrf_sm2_what(q,kw,j-1,i))-slope);
    const float dvdz=(q.v[sm2_i3(kw,jv,ii,q.ny+1,q.nx)]-q.v[sm2_i3(kw-1,jv,ii,q.ny+1,q.nx)])*rz;
    return dwdy+dvdz;
}
static inline float wrf_sm2_tau13(thread const WrfSmagGrid &q,device const float *km,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    const float rho=0.5f*(q.fnm[kw]*(wrf_sm2_rho(q,kw,j,i-1)+wrf_sm2_rho(q,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_rho(q,kw-1,j,i-1)+wrf_sm2_rho(q,kw-1,j,i)));
    const float kval=0.5f*(q.fnm[kw]*(wrf_sm2_k(q,km,kw,j,i-1)+wrf_sm2_k(q,km,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_k(q,km,kw-1,j,i-1)+wrf_sm2_k(q,km,kw-1,j,i)));
    return -rho*kval*wrf_sm2_d13(q,kw,j,i);
}
static inline float wrf_sm2_tau23(thread const WrfSmagGrid &q,device const float *km,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    const float rho=0.5f*(q.fnm[kw]*(wrf_sm2_rho(q,kw,j-1,i)+wrf_sm2_rho(q,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_rho(q,kw-1,j-1,i)+wrf_sm2_rho(q,kw-1,j,i)));
    const float kval=0.5f*(q.fnm[kw]*(wrf_sm2_k(q,km,kw,j-1,i)+wrf_sm2_k(q,km,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_k(q,km,kw-1,j-1,i)+wrf_sm2_k(q,km,kw-1,j,i)));
    return -rho*kval*wrf_sm2_d23(q,kw,j,i);
}
static inline float wrf_sm2_tau33(thread const WrfSmagGrid &q,device const float *km,int k,int j,int i) {
    if(k<0||k>=q.nz)return 0.0f;
    const int jj=wrf_sm2_iy(q,j),ii=wrf_sm2_ix(q,i);
    const float d=2.0f*(q.w[sm2_i3(k+1,jj,ii,q.ny,q.nx)]-q.w[sm2_i3(k,jj,ii,q.ny,q.nx)])*wrf_sm2_rdzw(q,k,j,i);
    return -wrf_sm2_rho(q,k,j,i)*wrf_sm2_k(q,km,k,j,i)*d;
}
static inline float wrf_sm2_tau13m(thread const WrfSmagGrid &q,device const float *km,int k,int j,int i) {
    return 0.25f*(wrf_sm2_tau13(q,km,k,j,i)+wrf_sm2_tau13(q,km,k,j,i+1)
        +wrf_sm2_tau13(q,km,k+1,j,i)+wrf_sm2_tau13(q,km,k+1,j,i+1));
}
static inline float wrf_sm2_tau23m(thread const WrfSmagGrid &q,device const float *km,int k,int j,int i) {
    return 0.25f*(wrf_sm2_tau23(q,km,k,j,i)+wrf_sm2_tau23(q,km,k,j+1,i)
        +wrf_sm2_tau23(q,km,k+1,j,i)+wrf_sm2_tau23(q,km,k+1,j+1,i));
}
static inline float wrf_sm2_scalar(thread const WrfSmagGrid &q,
                                   thread const WrfSmagScalarField &s,int k,int j,int i) {
    const int jj=wrf_sm2_iy(q,j),ii=wrf_sm2_ix(q,i);
    float value=s.f[sm2_i3(k,jj,ii,q.ny,q.nx)];
    if(s.full_theta){
        const ulong h=s.thb3d?sm2_i3(k,jj,ii,q.ny,q.nx):ulong(k);
        value=(s.thb[h]+value)-300.0f;
    }
    return value;
}
static inline float wrf_sm2_scalar_wx(thread const WrfSmagGrid &q,
                                     thread const WrfSmagScalarField &s,int kw,int j,int i) {
    const float p0=wrf_sm2_scalar(q,s,0,j,i-1)+wrf_sm2_scalar(q,s,0,j,i);
    const float p1=wrf_sm2_scalar(q,s,1,j,i-1)+wrf_sm2_scalar(q,s,1,j,i);
    const float p2=wrf_sm2_scalar(q,s,2,j,i-1)+wrf_sm2_scalar(q,s,2,j,i);
    const float pl=wrf_sm2_scalar(q,s,q.nz-1,j,i-1)+wrf_sm2_scalar(q,s,q.nz-1,j,i);
    const float pp=wrf_sm2_scalar(q,s,q.nz-2,j,i-1)+wrf_sm2_scalar(q,s,q.nz-2,j,i);
    const int kc=kw>=q.nz?q.nz-1:kw,kb=max(kc-1,0);
    const float pc=wrf_sm2_scalar(q,s,kc,j,i-1)+wrf_sm2_scalar(q,s,kc,j,i);
    const float pb=wrf_sm2_scalar(q,s,kb,j,i-1)+wrf_sm2_scalar(q,s,kb,j,i);
    return 0.5f*wrf_sm2_weights(q,kw,p0,p1,p2,pl,pp,pc,pb);
}
static inline float wrf_sm2_scalar_wy(thread const WrfSmagGrid &q,
                                     thread const WrfSmagScalarField &s,int kw,int j,int i) {
    const float p0=wrf_sm2_scalar(q,s,0,j-1,i)+wrf_sm2_scalar(q,s,0,j,i);
    const float p1=wrf_sm2_scalar(q,s,1,j-1,i)+wrf_sm2_scalar(q,s,1,j,i);
    const float p2=wrf_sm2_scalar(q,s,2,j-1,i)+wrf_sm2_scalar(q,s,2,j,i);
    const float pl=wrf_sm2_scalar(q,s,q.nz-1,j-1,i)+wrf_sm2_scalar(q,s,q.nz-1,j,i);
    const float pp=wrf_sm2_scalar(q,s,q.nz-2,j-1,i)+wrf_sm2_scalar(q,s,q.nz-2,j,i);
    const int kc=kw>=q.nz?q.nz-1:kw,kb=max(kc-1,0);
    const float pc=wrf_sm2_scalar(q,s,kc,j-1,i)+wrf_sm2_scalar(q,s,kc,j,i);
    const float pb=wrf_sm2_scalar(q,s,kb,j-1,i)+wrf_sm2_scalar(q,s,kb,j,i);
    return 0.5f*wrf_sm2_weights(q,kw,p0,p1,p2,pl,pp,pc,pb);
}
static inline float wrf_sm2_h1(thread const WrfSmagGrid &q,thread const WrfSmagScalarField &s,
                               device const float *kh,int k,int j,int i) {
    const float rho=0.5f*(wrf_sm2_rho(q,k,j,i-1)+wrf_sm2_rho(q,k,j,i));
    const float kval=0.5f*(wrf_sm2_k(q,kh,k,j,i-1)+wrf_sm2_k(q,kh,k,j,i));
    const float zx=0.5f*(wrf_sm2_zx(q,k,j,i)+wrf_sm2_zx(q,k+1,j,i));
    const float rdzu=2.0f/(1.0f/wrf_sm2_rdzw(q,k,j,i)+1.0f/wrf_sm2_rdzw(q,k,j,i-1));
    const int jj=wrf_sm2_iy(q,j),iu=wrf_sm2_iu(q,i);
    const float map=q.msfu[ulong(jj)*ulong(q.nx+1)+ulong(iu)];
    const float grad=q.rdx*(wrf_sm2_scalar(q,s,k,j,i)-wrf_sm2_scalar(q,s,k,j,i-1))
        -zx*(wrf_sm2_scalar_wx(q,s,k+1,j,i)-wrf_sm2_scalar_wx(q,s,k,j,i))*rdzu;
    return -map*rho*kval*grad;
}
static inline float wrf_sm2_h2(thread const WrfSmagGrid &q,thread const WrfSmagScalarField &s,
                               device const float *kh,int k,int j,int i) {
    const float rho=0.5f*(wrf_sm2_rho(q,k,j-1,i)+wrf_sm2_rho(q,k,j,i));
    const float kval=0.5f*(wrf_sm2_k(q,kh,k,j-1,i)+wrf_sm2_k(q,kh,k,j,i));
    const float zy=0.5f*(wrf_sm2_zy(q,k,j,i)+wrf_sm2_zy(q,k+1,j,i));
    const float rdzv=2.0f/(1.0f/wrf_sm2_rdzw(q,k,j,i)+1.0f/wrf_sm2_rdzw(q,k,j-1,i));
    const int jv=wrf_sm2_jv(q,j),ii=wrf_sm2_ix(q,i);
    const float map=q.msfv[ulong(jv)*ulong(q.nx)+ulong(ii)];
    const float grad=q.rdy*(wrf_sm2_scalar(q,s,k,j,i)-wrf_sm2_scalar(q,s,k,j-1,i))
        -zy*(wrf_sm2_scalar_wy(q,s,k+1,j,i)-wrf_sm2_scalar_wy(q,s,k,j,i))*rdzv;
    return -map*rho*kval*grad;
}
static inline float wrf_sm2_fx(thread const WrfSmagGrid &q,device const float *fx,int k,int j,int i) {
    return fx[sm2_i3(k,wrf_sm2_iy(q,j),wrf_sm2_iu(q,i),q.ny,q.nx+1)];
}
static inline float wrf_sm2_fy(thread const WrfSmagGrid &q,device const float *fy,int k,int j,int i) {
    return fy[sm2_i3(k,wrf_sm2_jv(q,j),wrf_sm2_ix(q,i),q.ny+1,q.nx)];
}
static inline float wrf_sm2_h1m(thread const WrfSmagGrid &q,device const float *fx,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    return 0.5f*(q.fnm[kw]*(wrf_sm2_fx(q,fx,kw,j,i+1)+wrf_sm2_fx(q,fx,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_fx(q,fx,kw-1,j,i+1)+wrf_sm2_fx(q,fx,kw-1,j,i)));
}
static inline float wrf_sm2_h2m(thread const WrfSmagGrid &q,device const float *fy,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    return 0.5f*(q.fnm[kw]*(wrf_sm2_fy(q,fy,kw,j+1,i)+wrf_sm2_fy(q,fy,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_fy(q,fy,kw-1,j+1,i)+wrf_sm2_fy(q,fy,kw-1,j,i)));
}
static inline float wrf_sm2_vd_s_h3(thread const WrfSmagGrid &q,device const float *var,
                                    device const float *thb,device const float *khv,
                                    int full_theta,int thb3d,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    const int jj=wrf_sm2_iy(q,j),ii=wrf_sm2_ix(q,i);
    const ulong hc=sm2_i3(kw,jj,ii,q.ny,q.nx),hm=sm2_i3(kw-1,jj,ii,q.ny,q.nx);
    float xkx=q.fnm[kw]*khv[hc]+q.fnp[kw]*khv[hm];
    xkx *= q.fnm[kw]*wrf_sm2_rho(q,kw,j,i)+q.fnp[kw]*wrf_sm2_rho(q,kw-1,j,i);
    float dvar=var[hc]-var[hm];
    if(full_theta){
        const ulong bc=thb3d?hc:ulong(kw),bm=thb3d?hm:ulong(kw-1);
        dvar += thb[bc]-thb[bm];
    }
    return -xkx*dvar*wrf_sm2_rdz(q,kw,j,i);
}
static inline float wrf_sm2_chat(thread const WrfSmagGrid &q,device const float *what,int k,int j,int i) {
    return what[sm2_i3(k,wrf_sm2_iy(q,j),wrf_sm2_ix(q,i),q.ny,q.nx)];
}
static inline float wrf_sm2_crdz(thread const WrfSmagGrid &q,device const float *rdz,int k,int j,int i) {
    k=clamp(k,0,q.nz);return rdz[sm2_i3(k,wrf_sm2_iy(q,j),wrf_sm2_ix(q,i),q.ny,q.nx)];
}
static inline float wrf_sm2_czx(thread const WrfSmagGrid &q,device const float *zx,int k,int j,int i) {
    return zx[sm2_i3(k,wrf_sm2_iy(q,j),wrf_sm2_iu(q,i),q.ny,q.nx+1)];
}
static inline float wrf_sm2_czy(thread const WrfSmagGrid &q,device const float *zy,int k,int j,int i) {
    return zy[sm2_i3(k,wrf_sm2_jv(q,j),wrf_sm2_ix(q,i),q.ny+1,q.nx)];
}
static inline float wrf_sm2_cwx(thread const WrfSmagGrid &q,device const float *what,int k,int j,int i) {
    return 0.25f*(wrf_sm2_chat(q,what,k,j,i)+wrf_sm2_chat(q,what,k+1,j,i)
        +wrf_sm2_chat(q,what,k,j,i-1)+wrf_sm2_chat(q,what,k+1,j,i-1));
}
static inline float wrf_sm2_cwy(thread const WrfSmagGrid &q,device const float *what,int k,int j,int i) {
    return 0.25f*(wrf_sm2_chat(q,what,k,j,i)+wrf_sm2_chat(q,what,k+1,j,i)
        +wrf_sm2_chat(q,what,k,j-1,i)+wrf_sm2_chat(q,what,k+1,j-1,i));
}
static inline float wrf_sm2_cd13(thread const WrfSmagGrid &q,int kw,int j,int i,
                                 device const float *what,device const float *rdz,device const float *zx) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    if(q.boundary_x)i=(i<=0?(q.nx>1?1:0):(i>=q.nx?q.nx-1:i));
    if(q.boundary_y)j=(j<=0?(q.ny>1?1:0):(j>=q.ny?q.ny-1:j));
    const int jj=wrf_sm2_iy(q,j),iu=wrf_sm2_iu(q,i);
    const float msf=q.msfu[ulong(jj)*ulong(q.nx+1)+ulong(iu)];
    const float rz=0.5f*(wrf_sm2_crdz(q,rdz,kw,j,i)+wrf_sm2_crdz(q,rdz,kw,j,i-1));
    const float slope=(wrf_sm2_cwx(q,what,kw,j,i)-wrf_sm2_cwx(q,what,kw-1,j,i))*wrf_sm2_czx(q,zx,kw,j,i)*rz;
    const float dwdx=msf*msf*(q.rdx*(wrf_sm2_chat(q,what,kw,j,i)-wrf_sm2_chat(q,what,kw,j,i-1))-slope);
    const float du=q.u[sm2_i3(kw,jj,iu,q.ny,q.nx+1)]-q.u[sm2_i3(kw-1,jj,iu,q.ny,q.nx+1)];
    return fma(rz,du,dwdx);
}
static inline float wrf_sm2_cd23(thread const WrfSmagGrid &q,int kw,int j,int i,
                                 device const float *what,device const float *rdz,device const float *zy) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    if(q.boundary_x)i=(i<=0?(q.nx>1?1:0):(i>=q.nx?q.nx-1:i));
    if(q.boundary_y)j=(j<=0?(q.ny>1?1:0):(j>=q.ny?q.ny-1:j));
    const int jv=wrf_sm2_jv(q,j),ii=wrf_sm2_ix(q,i);
    const float msf=q.msfv[ulong(jv)*ulong(q.nx)+ulong(ii)];
    const float rz=0.5f*(wrf_sm2_crdz(q,rdz,kw,j,i)+wrf_sm2_crdz(q,rdz,kw,j-1,i));
    const float slope=(wrf_sm2_cwy(q,what,kw,j,i)-wrf_sm2_cwy(q,what,kw-1,j,i))*wrf_sm2_czy(q,zy,kw,j,i)*rz;
    const float dwdy=msf*msf*(q.rdy*(wrf_sm2_chat(q,what,kw,j,i)-wrf_sm2_chat(q,what,kw,j-1,i))-slope);
    const float dv=q.v[sm2_i3(kw,jv,ii,q.ny+1,q.nx)]-q.v[sm2_i3(kw-1,jv,ii,q.ny+1,q.nx)];
    return fma(rz,dv,dwdy);
}
static inline float wrf_sm2_ctau13(thread const WrfSmagGrid &q,device const float *km,int kw,int j,int i,
                                   device const float *what,device const float *rdz,device const float *zx) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    const float rho=0.5f*(q.fnm[kw]*(wrf_sm2_rho(q,kw,j,i-1)+wrf_sm2_rho(q,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_rho(q,kw-1,j,i-1)+wrf_sm2_rho(q,kw-1,j,i)));
    const float kval=0.5f*(q.fnm[kw]*(wrf_sm2_k(q,km,kw,j,i-1)+wrf_sm2_k(q,km,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_k(q,km,kw-1,j,i-1)+wrf_sm2_k(q,km,kw-1,j,i)));
    return -rho*kval*wrf_sm2_cd13(q,kw,j,i,what,rdz,zx);
}
static inline float wrf_sm2_ctau23(thread const WrfSmagGrid &q,device const float *km,int kw,int j,int i,
                                   device const float *what,device const float *rdz,device const float *zy) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    const float rho=0.5f*(q.fnm[kw]*(wrf_sm2_rho(q,kw,j-1,i)+wrf_sm2_rho(q,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_rho(q,kw-1,j-1,i)+wrf_sm2_rho(q,kw-1,j,i)));
    const float kval=0.5f*(q.fnm[kw]*(wrf_sm2_k(q,km,kw,j-1,i)+wrf_sm2_k(q,km,kw,j,i))
        +q.fnp[kw]*(wrf_sm2_k(q,km,kw-1,j-1,i)+wrf_sm2_k(q,km,kw-1,j,i)));
    return -rho*kval*wrf_sm2_cd23(q,kw,j,i,what,rdz,zy);
}
static inline float wrf_sm2_ctau13m(thread const WrfSmagGrid &q,device const float *km,int k,int j,int i,
                                    device const float *what,device const float *rdz,device const float *zx) {
    return 0.25f*(wrf_sm2_ctau13(q,km,k,j,i,what,rdz,zx)+wrf_sm2_ctau13(q,km,k,j,i+1,what,rdz,zx)
        +wrf_sm2_ctau13(q,km,k+1,j,i,what,rdz,zx)+wrf_sm2_ctau13(q,km,k+1,j,i+1,what,rdz,zx));
}
static inline float wrf_sm2_ctau23m(thread const WrfSmagGrid &q,device const float *km,int k,int j,int i,
                                    device const float *what,device const float *rdz,device const float *zy) {
    return 0.25f*(wrf_sm2_ctau23(q,km,k,j,i,what,rdz,zy)+wrf_sm2_ctau23(q,km,k,j+1,i,what,rdz,zy)
        +wrf_sm2_ctau23(q,km,k+1,j,i,what,rdz,zy)+wrf_sm2_ctau23(q,km,k+1,j+1,i,what,rdz,zy));
}
static inline float wrf_sm2_tau_add4(float a,float b,float c,float d) { return ((a+b)+c)+d; }
static inline float wrf_sm2_tau_def13(thread const WrfSmagGrid &q,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    if(q.boundary_x)i=(i<=0?(q.nx>1?1:0):(i>=q.nx?q.nx-1:i));
    if(q.boundary_y)j=(j<=0?(q.ny>1?1:0):(j>=q.ny?q.ny-1:j));
    const int jj=wrf_sm2_iy(q,j),iu=wrf_sm2_iu(q,i);
    const float msf=q.msfu[ulong(jj)*ulong(q.nx+1)+ulong(iu)];
    const float rz=0.5f*(wrf_sm2_rdz(q,kw,j,i)+wrf_sm2_rdz(q,kw,j,i-1));
    const float upper=wrf_sm2_tau_add4(wrf_sm2_what(q,kw,j,i),wrf_sm2_what(q,kw+1,j,i),wrf_sm2_what(q,kw,j,i-1),wrf_sm2_what(q,kw+1,j,i-1));
    const float lower=wrf_sm2_tau_add4(wrf_sm2_what(q,kw-1,j,i),wrf_sm2_what(q,kw,j,i),wrf_sm2_what(q,kw-1,j,i-1),wrf_sm2_what(q,kw,j,i-1));
    const float avg=fma(upper,0.25f,-(lower*0.25f));
    const float slope=(avg*wrf_sm2_zx(q,kw,j,i))*rz;
    const float du=wrf_sm2_what(q,kw,j,i)-wrf_sm2_what(q,kw,j,i-1);
    const float grad=fma(du,q.rdx,-slope), horizontal=(msf*msf)*grad;
    const float delta=q.u[sm2_i3(kw,jj,iu,q.ny,q.nx+1)]-q.u[sm2_i3(kw-1,jj,iu,q.ny,q.nx+1)];
    return fma(rz,delta,horizontal);
}
static inline float wrf_sm2_tau_def23(thread const WrfSmagGrid &q,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    if(q.boundary_x)i=(i<=0?(q.nx>1?1:0):(i>=q.nx?q.nx-1:i));
    if(q.boundary_y)j=(j<=0?(q.ny>1?1:0):(j>=q.ny?q.ny-1:j));
    const int jv=wrf_sm2_jv(q,j),ii=wrf_sm2_ix(q,i);
    const float msf=q.msfv[ulong(jv)*ulong(q.nx)+ulong(ii)];
    const float rz=0.5f*(wrf_sm2_rdz(q,kw,j,i)+wrf_sm2_rdz(q,kw,j-1,i));
    const float upper=wrf_sm2_tau_add4(wrf_sm2_what(q,kw,j,i),wrf_sm2_what(q,kw+1,j,i),wrf_sm2_what(q,kw,j-1,i),wrf_sm2_what(q,kw+1,j-1,i));
    const float lower=wrf_sm2_tau_add4(wrf_sm2_what(q,kw-1,j,i),wrf_sm2_what(q,kw,j,i),wrf_sm2_what(q,kw-1,j-1,i),wrf_sm2_what(q,kw,j-1,i));
    const float avg=fma(upper,0.25f,-(lower*0.25f));
    const float slope=(avg*wrf_sm2_zy(q,kw,j,i))*rz;
    const float dw=wrf_sm2_what(q,kw,j,i)-wrf_sm2_what(q,kw,j-1,i);
    const float grad=fma(dw,q.rdy,-slope), horizontal=(msf*msf)*grad;
    const float delta=q.v[sm2_i3(kw,jv,ii,q.ny+1,q.nx)]-q.v[sm2_i3(kw-1,jv,ii,q.ny+1,q.nx)];
    return fma(rz,delta,horizontal);
}
template <bool Axis13>
static inline float wrf_sm2_tau_face(thread const WrfSmagGrid &q,device const float *km,int kw,int j,int i) {
    if(kw<=0||kw>=q.nz)return 0.0f;
    const int ja=Axis13?j:j-1,ia=Axis13?i-1:i;
    const float ru=wrf_sm2_rho(q,kw,ja,ia)+wrf_sm2_rho(q,kw,j,i);
    const float rl=wrf_sm2_rho(q,kw-1,ja,ia)+wrf_sm2_rho(q,kw-1,j,i);
    const float ku=wrf_sm2_k(q,km,kw,ja,ia)+wrf_sm2_k(q,km,kw,j,i);
    const float kl=wrf_sm2_k(q,km,kw-1,ja,ia)+wrf_sm2_k(q,km,kw-1,j,i);
    const float rhoavg=0.5f*fma(q.fnm[kw],ru,q.fnp[kw]*rl);
    const float kavg=0.5f*fma(q.fnp[kw],kl,q.fnm[kw]*ku);
    const float deformation=Axis13?wrf_sm2_tau_def13(q,kw,j,i):wrf_sm2_tau_def23(q,kw,j,i);
    return (-rhoavg*kavg)*deformation;
}
static inline float wrf_sm2_stress_tx(device const float *tx,thread const WrfSmagGrid &q,int k,int j,int i) {
    if(k<=0||k>=q.nz)return 0.0f;
    return tx[sm2_i3(k,wrf_sm2_iy(q,j),wrf_sm2_iu(q,i),q.ny,q.nx+1)];
}
static inline float wrf_sm2_stress_ty(device const float *ty,thread const WrfSmagGrid &q,int k,int j,int i) {
    if(k<=0||k>=q.nz)return 0.0f;
    return ty[sm2_i3(k,wrf_sm2_jv(q,j),wrf_sm2_ix(q,i),q.ny+1,q.nx)];
}
static inline float wrf_sm2_stress_sum13(device const float *tx,thread const WrfSmagGrid &q,int k,int j,int i) {
    return wrf_sm2_tau_add4(wrf_sm2_stress_tx(tx,q,k,j,i),wrf_sm2_stress_tx(tx,q,k,j,i+1),wrf_sm2_stress_tx(tx,q,k+1,j,i),wrf_sm2_stress_tx(tx,q,k+1,j,i+1));
}
static inline float wrf_sm2_stress_sum23(device const float *ty,thread const WrfSmagGrid &q,int k,int j,int i) {
    return wrf_sm2_tau_add4(wrf_sm2_stress_ty(ty,q,k,j,i),wrf_sm2_stress_ty(ty,q,k,j+1,i),wrf_sm2_stress_ty(ty,q,k+1,j,i),wrf_sm2_stress_ty(ty,q,k+1,j+1,i));
}

kernel void wrf_smag_hd_u(constant WrfSmagHdUResources &r [[buffer(0)]],
                          constant WrfSmagHdUParams &p [[buffer(1)]],
                          uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny)*ulong(p.nx+1);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx+1)),j=int((flat/uint(p.nx+1))%uint(p.ny));
    const int k=int(flat/(ulong(p.nx+1)*ulong(p.ny)));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const int ic=wrf_sm2_ix(q,i),im=wrf_sm2_ix(q,i-1),iu=wrf_sm2_iu(q,i),jj=wrf_sm2_iy(q,j);
    const float msf=r.msfu[ulong(jj)*ulong(p.nx+1)+ulong(iu)];
    const float tmpdz=0.5f*(1.0f/wrf_sm2_rdzw(q,k,j,ic)+1.0f/wrf_sm2_rdzw(q,k,j,im));
    const float zx=0.5f*(wrf_sm2_zx(q,k,j,i)+wrf_sm2_zx(q,k+1,j,i));
    const float zy=0.125f*(wrf_sm2_zy(q,k,j,im)+wrf_sm2_zy(q,k,j,ic)+wrf_sm2_zy(q,k,j+1,im)+wrf_sm2_zy(q,k,j+1,ic)
        +wrf_sm2_zy(q,k+1,j,im)+wrf_sm2_zy(q,k+1,j,ic)+wrf_sm2_zy(q,k+1,j+1,im)+wrf_sm2_zy(q,k+1,j+1,ic));
    const float divh=msf*p.rdx*(wrf_sm2_tau11(q,r.km,r.d11,k,j,ic)-wrf_sm2_tau11(q,r.km,r.d11,k,j,im))
        +msf*p.rdy*(wrf_sm2_tau12(q,r.km,r.d12,k,j+1,i)-wrf_sm2_tau12(q,r.km,r.d12,k,j,i));
    const float divz=msf*zx*(wrf_sm2_tau11_uavg(q,r.km,r.d11,k+1,j,i)-wrf_sm2_tau11_uavg(q,r.km,r.d11,k,j,i))/tmpdz
        +msf*zy*(wrf_sm2_tau12_uavg(q,r.km,r.d12,k+1,j,i)-wrf_sm2_tau12_uavg(q,r.km,r.d12,k,j,i))/tmpdz;
    r.tend[sm2_i3(k,j,i,p.ny,p.nx+1)] += 9.81f*tmpdz/q.dnw[k]*(divh-divz);
}

kernel void wrf_smag_hd_v(constant WrfSmagHdVResources &r [[buffer(0)]],
                          constant WrfSmagHdVParams &p [[buffer(1)]],
                          uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny+1)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny+1));
    const int k=int(flat/(ulong(p.nx)*ulong(p.ny+1)));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const int jc=wrf_sm2_iy(q,j),jm=wrf_sm2_iy(q,j-1),jv=wrf_sm2_jv(q,j),ii=wrf_sm2_ix(q,i);
    const float msf=r.msfv[ulong(jv)*ulong(p.nx)+ulong(ii)];
    const float tmpdz=0.5f*(1.0f/wrf_sm2_rdzw(q,k,jc,i)+1.0f/wrf_sm2_rdzw(q,k,jm,i));
    const float zx=0.125f*(wrf_sm2_zx(q,k,jm,i)+wrf_sm2_zx(q,k,jm,i+1)+wrf_sm2_zx(q,k,jc,i)+wrf_sm2_zx(q,k,jc,i+1)
        +wrf_sm2_zx(q,k+1,jm,i)+wrf_sm2_zx(q,k+1,jm,i+1)+wrf_sm2_zx(q,k+1,jc,i)+wrf_sm2_zx(q,k+1,jc,i+1));
    const float zy=0.5f*(wrf_sm2_zy(q,k,j,i)+wrf_sm2_zy(q,k+1,j,i));
    const float divh=msf*p.rdy*(wrf_sm2_tau22(q,r.km,r.d22,k,jc,i)-wrf_sm2_tau22(q,r.km,r.d22,k,jm,i))
        +msf*p.rdx*(wrf_sm2_tau12(q,r.km,r.d12,k,j,i+1)-wrf_sm2_tau12(q,r.km,r.d12,k,j,i));
    const float divz=msf*zx*(wrf_sm2_tau12_vavg(q,r.km,r.d12,k+1,j,i)-wrf_sm2_tau12_vavg(q,r.km,r.d12,k,j,i))/tmpdz
        +msf*zy*(wrf_sm2_tau22_vavg(q,r.km,r.d22,k+1,j,i)-wrf_sm2_tau22_vavg(q,r.km,r.d22,k,j,i))/tmpdz;
    r.tend[sm2_i3(k,j,i,p.ny+1,p.nx)] += 9.81f*tmpdz/q.dnw[k]*(divh-divz);
}

kernel void wrf_smag_vd_u(constant WrfSmagVdUResources &r [[buffer(0)]],
                          constant WrfSmagVdUParams &p [[buffer(1)]],
                          uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny)*ulong(p.nx+1);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx+1)),j=int((flat/uint(p.nx+1))%uint(p.ny));
    const int k=int(flat/(ulong(p.nx+1)*ulong(p.ny)));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const float lo=wrf_sm2_tau13(q,r.km,k,j,i),hi=wrf_sm2_tau13(q,r.km,k+1,j,i);
    r.tend[sm2_i3(k,j,i,p.ny,p.nx+1)] += 9.81f*(hi-lo)/q.dnw[k];
}

kernel void wrf_smag_vd_v(constant WrfSmagVdVResources &r [[buffer(0)]],
                          constant WrfSmagVdVParams &p [[buffer(1)]],
                          uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny+1)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny+1));
    const int k=int(flat/(ulong(p.nx)*ulong(p.ny+1)));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const float lo=wrf_sm2_tau23(q,r.km,k,j,i),hi=wrf_sm2_tau23(q,r.km,k+1,j,i);
    r.tend[sm2_i3(k,j,i,p.ny+1,p.nx)] += 9.81f*(hi-lo)/q.dnw[k];
}

kernel void wrf_smag_vd_w(constant WrfSmagVdWResources &r [[buffer(0)]],
                          constant WrfSmagVdWParams &p [[buffer(1)]],
                          uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz+1)*ulong(p.ny)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny));
    const int kw=int(flat/(ulong(p.nx)*ulong(p.ny)));
    if(kw<=0||kw>=p.nz)return;
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const float hi=wrf_sm2_tau33(q,r.km,kw,j,i),lo=wrf_sm2_tau33(q,r.km,kw-1,j,i);
    r.tend[sm2_i3(kw,j,i,p.ny,p.nx)] += 9.81f*(hi-lo)/q.dn[kw];
}

template <typename P>
static inline float wrf_sm2_surface_u(device const float *ustm,
                                      constant P &p,
                                      thread const WrfSmagGrid &q,int j,int i,
                                      bool use_ustm,float cd0) {
    const int ic=wrf_sm2_ix(q,i),im=wrf_sm2_ix(q,i-1),jj=wrf_sm2_iy(q,j);
    const float vv=0.25f*(q.v[sm2_i3(0,wrf_sm2_jv(q,j),ic,q.ny+1,q.nx)]
        +q.v[sm2_i3(0,wrf_sm2_jv(q,j+1),ic,q.ny+1,q.nx)]
        +q.v[sm2_i3(0,wrf_sm2_jv(q,j),im,q.ny+1,q.nx)]
        +q.v[sm2_i3(0,wrf_sm2_jv(q,j+1),im,q.ny+1,q.nx)]);
    const float uu=q.u[sm2_i3(0,jj,wrf_sm2_iu(q,i),q.ny,q.nx+1)];
    const float speed=sqrt(uu*uu+vv*vv)+1.0e-15f;
    const float ustar=use_ustm?0.5f*(ustm[ulong(jj)*ulong(p.nx)+ulong(ic)]
                                      +ustm[ulong(jj)*ulong(p.nx)+ulong(im)]):0.0f;
    const float stress=use_ustm?ustar*ustar*uu/speed:cd0*speed*uu;
    const float rho=0.5f*(wrf_sm2_rho(q,0,j,ic)+wrf_sm2_rho(q,0,j,im));
    return 9.81f*stress*rho/q.dnw[0];
}
template <typename P>
static inline float wrf_sm2_surface_v(device const float *ustm,
                                      constant P &p,
                                      thread const WrfSmagGrid &q,int j,int i,
                                      bool use_ustm,float cd0) {
    const int jc=wrf_sm2_iy(q,j),jm=wrf_sm2_iy(q,j-1),ii=wrf_sm2_ix(q,i);
    const float uu=0.25f*(q.u[sm2_i3(0,jc,wrf_sm2_iu(q,i),q.ny,q.nx+1)]
        +q.u[sm2_i3(0,jc,wrf_sm2_iu(q,i+1),q.ny,q.nx+1)]
        +q.u[sm2_i3(0,jm,wrf_sm2_iu(q,i),q.ny,q.nx+1)]
        +q.u[sm2_i3(0,jm,wrf_sm2_iu(q,i+1),q.ny,q.nx+1)]);
    const float vv=q.v[sm2_i3(0,wrf_sm2_jv(q,j),ii,q.ny+1,q.nx)];
    const float speed=sqrt(vv*vv+uu*uu)+1.0e-15f;
    const float ustar=use_ustm?0.5f*(ustm[ulong(jc)*ulong(p.nx)+ulong(ii)]
                                      +ustm[ulong(jm)*ulong(p.nx)+ulong(ii)]):0.0f;
    const float stress=use_ustm?ustar*ustar*vv/speed:cd0*speed*vv;
    const float rho=0.5f*(wrf_sm2_rho(q,0,jc,i)+wrf_sm2_rho(q,0,jm,i));
    return 9.81f*stress*rho/q.dnw[0];
}

kernel void wrf_smag_surface_u(constant WrfSmagSurfaceUResources &r [[buffer(0)]],
                               constant WrfSmagSurfaceUParams &p [[buffer(1)]],
                               uint flat [[thread_position_in_grid]]) {
    if(!p.active||flat>=uint(p.ny*(p.nx+1)))return;
    const int i=int(flat%uint(p.nx+1)),j=int(flat/uint(p.nx+1));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    r.tend[sm2_i3(0,j,i,p.ny,p.nx+1)] += wrf_sm2_surface_u(r.ustm,p,q,j,i,true,0.0f);
}
kernel void wrf_smag_surface_v(constant WrfSmagSurfaceVResources &r [[buffer(0)]],
                               constant WrfSmagSurfaceVParams &p [[buffer(1)]],
                               uint flat [[thread_position_in_grid]]) {
    if(!p.active||flat>=uint((p.ny+1)*p.nx))return;
    const int i=int(flat%uint(p.nx)),j=int(flat/uint(p.nx));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    r.tend[sm2_i3(0,j,i,p.ny+1,p.nx)] += wrf_sm2_surface_v(r.ustm,p,q,j,i,true,0.0f);
}
kernel void wrf_smag_surface_u_cd0(constant WrfSmagSurfaceUCd0Resources &r [[buffer(0)]],
                                   constant WrfSmagSurfaceUCd0Params &p [[buffer(1)]],
                                   uint flat [[thread_position_in_grid]]) {
    if(flat>=uint(p.ny*(p.nx+1)))return;
    const int i=int(flat%uint(p.nx+1)),j=int(flat/uint(p.nx+1));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    r.tend[sm2_i3(0,j,i,p.ny,p.nx+1)] += wrf_sm2_surface_u(nullptr,p,q,j,i,false,p.cd0);
}
kernel void wrf_smag_surface_v_cd0(constant WrfSmagSurfaceVCd0Resources &r [[buffer(0)]],
                                   constant WrfSmagSurfaceVCd0Params &p [[buffer(1)]],
                                   uint flat [[thread_position_in_grid]]) {
    if(flat>=uint((p.ny+1)*p.nx))return;
    const int i=int(flat%uint(p.nx)),j=int(flat/uint(p.nx));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    r.tend[sm2_i3(0,j,i,p.ny+1,p.nx)] += wrf_sm2_surface_v(nullptr,p,q,j,i,false,p.cd0);
}
kernel void wrf_smag_surface_scalars(constant WrfSmagSurfaceScalarsResources &r [[buffer(0)]],
                                     constant WrfSmagSurfaceScalarsParams &p [[buffer(1)]],
                                     uint flat [[thread_position_in_grid]]) {
    if((!p.apply_heat&&!p.apply_moist)||flat>=uint(p.ny*p.nx))return;
    const int i=int(flat%uint(p.nx)),j=int(flat/uint(p.nx));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const float vapor=q.moist?q.qv[sm2_i3(0,j,i,p.ny,p.nx)]:0.0f;
    const float cpm=1004.5f*(1.0f+0.8f*vapor);
    const ulong ix=ulong(j)*ulong(p.nx)+ulong(i);
    if(p.apply_heat)r.rth[sm2_i3(0,j,i,p.ny,p.nx)] -= 9.81f*r.hfx[ix]/cpm/q.dnw[0];
    if(p.apply_moist&&q.moist)r.rqv[sm2_i3(0,j,i,p.ny,p.nx)] -= 9.81f*r.qfx[ix]/q.dnw[0];
}
kernel void wrf_smag_surface_heat_const(constant WrfSmagSurfaceHeatConstResources &r [[buffer(0)]],
                                        constant WrfSmagSurfaceHeatConstParams &p [[buffer(1)]],
                                        uint flat [[thread_position_in_grid]]) {
    if(flat>=uint(p.ny*p.nx))return;
    const int i=int(flat%uint(p.nx)),j=int(flat/uint(p.nx));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const ulong ix=ulong(j)*ulong(p.nx)+ulong(i);
    const float rho=wrf_sm2_rho(q,0,j,i);
    if(p.write_hfx){
        const float vapor=q.moist?q.qv[sm2_i3(0,j,i,p.ny,p.nx)]:0.0f;
        const float cpm=1004.5f*(1.0f+0.8f*vapor);
        r.hfx[ix]=p.heat_flux*cpm*rho;
    }
    r.rth[sm2_i3(0,j,i,p.ny,p.nx)] -= 9.81f*p.heat_flux*rho/q.dnw[0];
}

kernel void wrf_smag_flux_s(constant WrfSmagFluxSResources &r [[buffer(0)]],
                            constant WrfSmagFluxSParams &p [[buffer(1)]],
                            uint flat [[thread_position_in_grid]]) {
    const ulong padded=ulong(p.ny+1)*ulong(p.nx+1);
    const int i=int(ulong(flat)%ulong(p.nx+1));
    const int j=int((ulong(flat)/ulong(p.nx+1))%ulong(p.ny+1));
    const int k=int(ulong(flat)/padded);
    if(k>=p.nz)return;
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const WrfSmagScalarField s={r.f,r.thb,p.full_theta,p.thb3d};
    if(i<=p.nx&&j<p.ny)r.fx[sm2_i3(k,j,i,p.ny,p.nx+1)]=wrf_sm2_h1(q,s,r.kh,k,j,i);
    if(i<p.nx&&j<=p.ny)r.fy[sm2_i3(k,j,i,p.ny+1,p.nx)]=wrf_sm2_h2(q,s,r.kh,k,j,i);
}

kernel void wrf_smag_hd_s(constant WrfSmagHdSResources &r [[buffer(0)]],
                          constant WrfSmagHdSParams &p [[buffer(1)]],
                          uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny));
    const int k=int(flat/(ulong(p.nx)*ulong(p.ny)));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const int ii=wrf_sm2_ix(q,i),jj=wrf_sm2_iy(q,j);
    const float map=r.msft[ulong(jj)*ulong(p.nx)+ulong(ii)];
    const float zx=0.25f*(wrf_sm2_zx(q,k,j,i)+wrf_sm2_zx(q,k,j,i+1)+wrf_sm2_zx(q,k+1,j,i)+wrf_sm2_zx(q,k+1,j,i+1));
    const float zy=0.25f*(wrf_sm2_zy(q,k,j,i)+wrf_sm2_zy(q,k,j+1,i)+wrf_sm2_zy(q,k+1,j,i)+wrf_sm2_zy(q,k+1,j+1,i));
    const float rz=wrf_sm2_rdzw(q,k,j,i);
    const float divh=map*p.rdx*(wrf_sm2_fx(q,r.fx,k,j,i+1)-wrf_sm2_fx(q,r.fx,k,j,i))
        +map*p.rdy*(wrf_sm2_fy(q,r.fy,k,j+1,i)-wrf_sm2_fy(q,r.fy,k,j,i));
    const float divz=map*zx*(wrf_sm2_h1m(q,r.fx,k+1,j,i)-wrf_sm2_h1m(q,r.fx,k,j,i))*rz
        +map*zy*(wrf_sm2_h2m(q,r.fy,k+1,j,i)-wrf_sm2_h2m(q,r.fy,k,j,i))*rz;
    r.tend[sm2_i3(k,j,i,p.ny,p.nx)] += 9.81f/(q.dnw[k]*rz)*(divh-divz);
}

kernel void wrf_smag_w_primitives(constant WrfSmagWPrimitivesResources &r [[buffer(0)]],
                                  constant WrfSmagWPrimitivesParams &p [[buffer(1)]],
                                  uint flat [[thread_position_in_grid]]) {
    const ulong i_extent=ulong(p.nx+1),j_extent=ulong(p.ny+1),k_extent=ulong(p.nz+1);
    const ulong count=i_extent*j_extent*k_extent;
    if(ulong(flat)>=count)return;
    const int i=int(ulong(flat)%i_extent),j=int((ulong(flat)/i_extent)%j_extent);
    const int kw=int(ulong(flat)/(i_extent*j_extent));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    if(i<p.nx&&j<p.ny)
        r.cached_what[sm2_i3(kw,j,i,p.ny,p.nx)]=wrf_sm2_what(q,kw,j,i);
    if(kw<p.nz&&i<p.nx&&j<p.ny)
        r.cached_rdz[sm2_i3(kw,j,i,p.ny,p.nx)]=wrf_sm2_rdz(q,kw,j,i);
    if(kw<p.nz&&i<=p.nx&&j<p.ny)
        r.cached_zx[sm2_i3(kw,j,i,p.ny,p.nx+1)]=wrf_sm2_zx(q,kw,j,i);
    if(kw<p.nz&&i<p.nx&&j<=p.ny)
        r.cached_zy[sm2_i3(kw,j,i,p.ny+1,p.nx)]=wrf_sm2_zy(q,kw,j,i);
}

kernel void wrf_smag_hd_w_cached(constant WrfSmagHdWCachedResources &r [[buffer(0)]],
                                 constant WrfSmagHdWCachedParams &p [[buffer(1)]],
                                 uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz+1)*ulong(p.ny)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny));
    const int kw=int(flat/(ulong(p.nx)*ulong(p.ny)));
    if(kw==0||kw>=p.nz)return;
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const int ii=wrf_sm2_ix(q,i),jj=wrf_sm2_iy(q,j);
    const float map=r.msft[ulong(jj)*ulong(p.nx)+ulong(ii)];
    const float zx=0.5f*(wrf_sm2_czx(q,r.cached_zx,kw,j,i)+wrf_sm2_czx(q,r.cached_zx,kw,j,i+1));
    const float zy=0.5f*(wrf_sm2_czy(q,r.cached_zy,kw,j,i)+wrf_sm2_czy(q,r.cached_zy,kw,j+1,i));
    const float rz=wrf_sm2_crdz(q,r.cached_rdz,kw,j,i);
    const float divh=map*p.rdx*(wrf_sm2_ctau13(q,r.km,kw,j,i+1,r.cached_what,r.cached_rdz,r.cached_zx)-wrf_sm2_ctau13(q,r.km,kw,j,i,r.cached_what,r.cached_rdz,r.cached_zx))
        +map*p.rdy*(wrf_sm2_ctau23(q,r.km,kw,j+1,i,r.cached_what,r.cached_rdz,r.cached_zy)-wrf_sm2_ctau23(q,r.km,kw,j,i,r.cached_what,r.cached_rdz,r.cached_zy));
    const float divz=map*rz*(zx*(wrf_sm2_ctau13m(q,r.km,kw,j,i,r.cached_what,r.cached_rdz,r.cached_zx)-wrf_sm2_ctau13m(q,r.km,kw-1,j,i,r.cached_what,r.cached_rdz,r.cached_zx))
        +zy*(wrf_sm2_ctau23m(q,r.km,kw,j,i,r.cached_what,r.cached_rdz,r.cached_zy)-wrf_sm2_ctau23m(q,r.km,kw-1,j,i,r.cached_what,r.cached_rdz,r.cached_zy)));
    r.tend[sm2_i3(kw,j,i,p.ny,p.nx)] += 9.81f/(q.dn[kw]*rz)*(divh-divz);
}

// Direct reference form retained by CUDA for exact and diagnostic builds.
// The Metal public launcher continues to select the cached ordinary path.
kernel void wrf_smag_hd_w(constant WrfSmagHdWResources &r [[buffer(0)]],
                          constant WrfSmagHdWParams &p [[buffer(1)]],
                          uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz+1)*ulong(p.ny)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny));
    const int kw=int(ulong(flat)/(ulong(p.nx)*ulong(p.ny)));
    if(kw==0||kw>=p.nz)return;
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const int ii=wrf_sm2_ix(q,i),jj=wrf_sm2_iy(q,j);
    const float map=r.msft[ulong(jj)*ulong(p.nx)+ulong(ii)];
    const float zx=0.5f*(wrf_sm2_zx(q,kw,j,i)+wrf_sm2_zx(q,kw,j,i+1));
    const float zy=0.5f*(wrf_sm2_zy(q,kw,j,i)+wrf_sm2_zy(q,kw,j+1,i));
    const float rz=wrf_sm2_rdz(q,kw,j,i);
    const float divh=map*p.rdx*(wrf_sm2_tau13(q,r.km,kw,j,i+1)-wrf_sm2_tau13(q,r.km,kw,j,i))
        +map*p.rdy*(wrf_sm2_tau23(q,r.km,kw,j+1,i)-wrf_sm2_tau23(q,r.km,kw,j,i));
    const float divz=map*rz*(zx*(wrf_sm2_tau13m(q,r.km,kw,j,i)-wrf_sm2_tau13m(q,r.km,kw-1,j,i))
        +zy*(wrf_sm2_tau23m(q,r.km,kw,j,i)-wrf_sm2_tau23m(q,r.km,kw-1,j,i)));
    r.tend[sm2_i3(kw,j,i,p.ny,p.nx)] += 9.81f/(q.dn[kw]*rz)*(divh-divz);
}

kernel void wrf_smag_w_stress(constant WrfSmagWStressResources &r [[buffer(0)]],
                              constant WrfSmagWStressParams &p [[buffer(1)]],
                              uint flat [[thread_position_in_grid]]) {
    const ulong ix_extent=ulong(p.nx+1),jy_extent=ulong(p.ny+1),plane=ix_extent*jy_extent;
    if(ulong(flat)>=plane*ulong(p.nz))return;
    const int i=int(ulong(flat)%ix_extent),j=int((ulong(flat)/ix_extent)%jy_extent);
    const int kw=int(ulong(flat)/plane);
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    if(i<=p.nx&&j<p.ny)
        r.tx[sm2_i3(kw,j,i,p.ny,p.nx+1)]=wrf_sm2_tau_face<true>(q,r.km,kw,j,i);
    if(i<p.nx&&j<=p.ny)
        r.ty[sm2_i3(kw,j,i,p.ny+1,p.nx)]=wrf_sm2_tau_face<false>(q,r.km,kw,j,i);
}

kernel void wrf_smag_hd_w_stress(constant WrfSmagHdWStressResources &r [[buffer(0)]],
                                 constant WrfSmagHdWStressParams &p [[buffer(1)]],
                                 uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz+1)*ulong(p.ny)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny));
    const int kw=int(ulong(flat)/(ulong(p.nx)*ulong(p.ny)));
    if(kw<=0||kw>=p.nz)return;
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const int ii=wrf_sm2_ix(q,i),jj=wrf_sm2_iy(q,j);
    const float map=r.msft[ulong(jj)*ulong(p.nx)+ulong(ii)];
    const float zx=0.5f*(wrf_sm2_zx(q,kw,j,i)+wrf_sm2_zx(q,kw,j,i+1));
    const float zy=0.5f*(wrf_sm2_zy(q,kw,j,i)+wrf_sm2_zy(q,kw,j+1,i));
    const float rz=wrf_sm2_rdz(q,kw,j,i);
    const float xd=wrf_sm2_stress_tx(r.tx,q,kw,j,i+1)-wrf_sm2_stress_tx(r.tx,q,kw,j,i);
    const float yd=wrf_sm2_stress_ty(r.ty,q,kw,j+1,i)-wrf_sm2_stress_ty(r.ty,q,kw,j,i);
    const float divh=fma(xd,map*p.rdx,(map*p.rdy)*yd);
    const float xavg=fma(wrf_sm2_stress_sum13(r.tx,q,kw,j,i),0.25f,
                         -(wrf_sm2_stress_sum13(r.tx,q,kw-1,j,i)*0.25f));
    const float yavg=fma(wrf_sm2_stress_sum23(r.ty,q,kw,j,i),0.25f,
                         -(wrf_sm2_stress_sum23(r.ty,q,kw-1,j,i)*0.25f));
    const float inner=fma(yavg,zy,xavg*zx);
    const float divz=(map*rz)*inner;
    const float quotient=9.81f/(q.dn[kw]*rz);
    const ulong at=sm2_i3(kw,j,i,p.ny,p.nx);
    r.tend[at]=fma(divh-divz,quotient,r.tend[at]);
}

kernel void wrf_smag_vd_s(constant WrfSmagVdSResources &r [[buffer(0)]],
                          constant WrfSmagVdSParams &p [[buffer(1)]],
                          uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny));
    const int k=int(ulong(flat)/(ulong(p.nx)*ulong(p.ny)));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const float lo=wrf_sm2_vd_s_h3(q,r.var,r.thb,r.khv,p.full_theta,p.thb3d,k,j,i);
    const float hi=wrf_sm2_vd_s_h3(q,r.var,r.thb,r.khv,p.full_theta,p.thb3d,k+1,j,i);
    r.tend[sm2_i3(k,j,i,p.ny,p.nx)] += 9.81f*(hi-lo)/q.dnw[k];
}

// Apply rk_addtend_dry's map-factor division to one already-accumulated
// tendency. The dedicated y big-step mode retains source operation order:
// rounded FP32 reciprocal, then a separate FP32 multiply.
kernel void smag2d_couple_map_factor(constant Smag2DCoupleMapFactorResources &r [[buffer(0)]],
                                     constant Smag2DCoupleMapFactorParams &p [[buffer(1)]],
                                     uint flat [[thread_position_in_grid]]) {
    const ulong nx_t = ulong(p.nx + (p.stagger == 1));
    const ulong ny_t = ulong(p.ny + (p.stagger == 2));
    const ulong nz_t = ulong(p.nz + (p.stagger == 3));
    const ulong count = nx_t * ny_t * nz_t;
    if (ulong(flat) >= count) return;
    const ulong i = ulong(flat) % nx_t;
    const ulong j = (ulong(flat) / nx_t) % ny_t;
    const ulong factor_nx = ulong(p.nx + (p.stagger == 1));
    const float factor = r.map_factor[j * factor_nx + i];
    if (p.multiply_reciprocal != 0) {
        const float reciprocal = 1.0f / factor;
        r.tendency[flat] = r.tendency[flat] * reciprocal;
    } else {
        r.tendency[flat] = r.tendency[flat] / factor;
    }
}

// Alternate Smagorinsky and TKE entrypoints share the grid/deformation
// helpers above and are compiled into the same family library.
#include "smag2d_alternates_fragment.metal"
#include "wrf_tke_km_fragment.metal"
#include "wrf_tke_rhs_fragment.metal"
#include "bound_tke_fragment.metal"
