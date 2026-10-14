#include <metal_stdlib>
#include "diff6_bindings.metalh"
#include "diff6_seam_u_bindings.metalh"
#include "diff6_seam_v_bindings.metalh"
#include "diff6_edge_pad_bindings.metalh"
#include "diff6_edge_crop_add_bindings.metalh"

using namespace metal;

// Copy WRF's zero-gradient physical halo: low=3, high=4. This leaves the
// existing periodic diff6 arithmetic untouched; real stencil reads cannot
// reach the padded grid's wrap. Padding supports surface planes and fields.
kernel void diff6_edge_pad(constant Diff6EdgePadResources &r [[buffer(0)]],
                          constant Diff6EdgePadParams &p [[buffer(1)]],
                          uint flat [[thread_position_in_grid]]) {
    const uint py=uint(p.ny+7), px=uint(p.nx+7);
    const uint count=uint(p.nlev)*py*px;
    if (flat>=count) return;
    const int i=clamp(int(flat%px)-3,0,p.nx-1);
    const int j=clamp(int((flat/px)%py)-3,0,p.ny-1);
    const uint k=flat/(py*px);
    r.dst[flat]=r.src[(k*uint(p.ny)+uint(j))*uint(p.nx)+uint(i)];
}

kernel void diff6_edge_crop_add(constant Diff6EdgeCropAddResources &r [[buffer(0)]],
                               constant Diff6EdgeCropAddParams &p [[buffer(1)]],
                               uint flat [[thread_position_in_grid]]) {
    const uint count=uint(p.nlev)*uint(p.ny)*uint(p.nx);
    if (flat>=count) return;
    const uint i=flat%uint(p.nx), j=(flat/uint(p.nx))%uint(p.ny);
    const uint k=flat/(uint(p.ny)*uint(p.nx));
    const uint src=(k*uint(p.ny+7)+j+3)*uint(p.nx+7)+i+3;
    r.tend[flat]=r.tend[flat]+r.padded[src];
}

// Keep CUDA's explicit FP32 operation boundaries and the Fortran face-flux
// sequence visible in both libraries. No contraction is permitted here.
#pragma clang fp contract(off)

static inline int d6_wrap(int x, int n) {
    const int r = x % n;
    return r < 0 ? r + n : r;
}
static inline ulong d6_field_index(int k, int j, int i, int nys, int nxs) {
    return (ulong(k) * ulong(nys) + ulong(j)) * ulong(nxs) + ulong(i);
}
static inline ulong d6_mass_index(int j, int i, int nx) {
    return ulong(j) * ulong(nx) + ulong(i);
}

// mode: 0 main periodic stencil, 1 u east seam (direct x), 2 v north
// seam (direct y). On a seam the cross axis wraps only when periodic.
static inline float d6_f(device const float *f, int k, int j, int i,
                         int nlev, int ny, int nys, int nx, int nxs,
                         int mode, bool boundary_cross) {
    const int ii = (mode == 0 || (mode == 2 && !boundary_cross))
        ? d6_wrap(i, nx) : i;
    const int jj = (mode == 0 || (mode == 1 && !boundary_cross))
        ? d6_wrap(j, ny) : j;
    return f[d6_field_index(k, jj, ii, nys, nxs)];
}
static inline float d6_mu(device const float *mut,
                          device const float *c1, device const float *c2,
                          int k, int j, int i, int ny, int nx,
                          int mode, bool boundary_cross) {
    const int jj = mode == 0 || (mode == 1 && !boundary_cross)
        ? d6_wrap(j, ny) : j;
    const int ii = mode == 0 || (mode == 2 && !boundary_cross)
        ? d6_wrap(i, nx) : i;
    const float mass = mut[d6_mass_index(jj, ii, nx)];
    return c1[k] * mass + c2[k];
}
static inline float d6_phb(device const float *phb, int k, int j, int i,
                           int ny, int nx, int mode, bool boundary_cross) {
    const int jj = mode == 0 || (mode == 1 && !boundary_cross)
        ? d6_wrap(j, ny) : j;
    const int ii = mode == 0 || (mode == 2 && !boundary_cross)
        ? d6_wrap(i, nx) : i;
    return phb[d6_mass_index(jj, ii, nx) + ulong(k) * ulong(ny) * ulong(nx)];
}
static inline float d6_msfu(device const float *msfu, int j, int i,
                            int ny, int nx, int mode, bool boundary_cross) {
    const int jj = mode == 0 || (mode == 1 && !boundary_cross)
        ? d6_wrap(j, ny) : j;
    const int ii = mode == 0 || (mode == 2 && !boundary_cross)
        ? d6_wrap(i, nx) : i;
    return msfu[ulong(jj) * ulong(nx + 1) + ulong(ii)];
}
static inline float d6_msfv(device const float *msfv, int j, int i,
                            int ny, int nx, int mode, bool boundary_cross) {
    const int jj = mode == 0 || (mode == 1 && !boundary_cross)
        ? d6_wrap(j, ny) : j;
    const int ii = mode == 0 || (mode == 2 && !boundary_cross)
        ? d6_wrap(i, nx) : i;
    return msfv[ulong(jj) * ulong(nx) + ulong(ii)];
}
static inline float d6_flux(device const float *f, int k, int j, int i,
                            bool x_axis, int p, int nlev, int ny, int nys,
                            int nx, int nxs, int mode, bool boundary_cross) {
    const float q0 = x_axis
        ? d6_f(f,k,j,i+p,nlev,ny,nys,nx,nxs,mode,boundary_cross)
        : d6_f(f,k,j+p,i,nlev,ny,nys,nx,nxs,mode,boundary_cross);
    const float qm1 = x_axis
        ? d6_f(f,k,j,i+p-1,nlev,ny,nys,nx,nxs,mode,boundary_cross)
        : d6_f(f,k,j+p-1,i,nlev,ny,nys,nx,nxs,mode,boundary_cross);
    const float qp1 = x_axis
        ? d6_f(f,k,j,i+p+1,nlev,ny,nys,nx,nxs,mode,boundary_cross)
        : d6_f(f,k,j+p+1,i,nlev,ny,nys,nx,nxs,mode,boundary_cross);
    const float qm2 = x_axis
        ? d6_f(f,k,j,i+p-2,nlev,ny,nys,nx,nxs,mode,boundary_cross)
        : d6_f(f,k,j+p-2,i,nlev,ny,nys,nx,nxs,mode,boundary_cross);
    const float qp2 = x_axis
        ? d6_f(f,k,j,i+p+2,nlev,ny,nys,nx,nxs,mode,boundary_cross)
        : d6_f(f,k,j+p+2,i,nlev,ny,nys,nx,nxs,mode,boundary_cross);
    const float qm3 = x_axis
        ? d6_f(f,k,j,i+p-3,nlev,ny,nys,nx,nxs,mode,boundary_cross)
        : d6_f(f,k,j+p-3,i,nlev,ny,nys,nx,nxs,mode,boundary_cross);
    return (10.0f * (q0 - qm1) - 5.0f * (qp1 - qm2)) + (qp2 - qm3);
}
static inline float d6_limited_flux(device const float *f, int k, int j,
                                   int i, bool x_axis, int p, int mono,
                                   int nlev, int ny, int nys, int nx, int nxs,
                                   int mode, bool boundary_cross) {
    float flux = d6_flux(f,k,j,i,x_axis,p,nlev,ny,nys,nx,nxs,mode,boundary_cross);
    if (mono == 2) {
        const float a = x_axis
            ? d6_f(f,k,j,i+p,nlev,ny,nys,nx,nxs,mode,boundary_cross)
            : d6_f(f,k,j+p,i,nlev,ny,nys,nx,nxs,mode,boundary_cross);
        const float b = x_axis
            ? d6_f(f,k,j,i+p-1,nlev,ny,nys,nx,nxs,mode,boundary_cross)
            : d6_f(f,k,j+p-1,i,nlev,ny,nys,nx,nxs,mode,boundary_cross);
        if (flux * (a - b) <= 0.0f) flux = 0.0f;
    }
    return flux;
}
static inline float d6_face_mu(device const float *mut,
                               device const float *c1, device const float *c2,
                               int k, int j, int i, int variant, bool x_axis,
                               int ny, int nx, int mode, bool boundary_cross) {
    if (x_axis) {
        if (variant == 1) return d6_mu(mut,c1,c2,k,j,i,ny,nx,mode,boundary_cross);
        if (variant == 2)
            return 0.25f * (d6_mu(mut,c1,c2,k,j-1,i-1,ny,nx,mode,boundary_cross)
                          + d6_mu(mut,c1,c2,k,j-1,i,ny,nx,mode,boundary_cross)
                          + d6_mu(mut,c1,c2,k,j,i-1,ny,nx,mode,boundary_cross)
                          + d6_mu(mut,c1,c2,k,j,i,ny,nx,mode,boundary_cross));
        return 0.5f * (d6_mu(mut,c1,c2,k,j,i-1,ny,nx,mode,boundary_cross)
                     + d6_mu(mut,c1,c2,k,j,i,ny,nx,mode,boundary_cross));
    }
    if (variant == 2) return d6_mu(mut,c1,c2,k,j,i,ny,nx,mode,boundary_cross);
    if (variant == 1)
        return 0.25f * (d6_mu(mut,c1,c2,k,j-1,i-1,ny,nx,mode,boundary_cross)
                      + d6_mu(mut,c1,c2,k,j-1,i,ny,nx,mode,boundary_cross)
                      + d6_mu(mut,c1,c2,k,j,i-1,ny,nx,mode,boundary_cross)
                      + d6_mu(mut,c1,c2,k,j,i,ny,nx,mode,boundary_cross));
    return 0.5f * (d6_mu(mut,c1,c2,k,j-1,i,ny,nx,mode,boundary_cross)
                 + d6_mu(mut,c1,c2,k,j,i,ny,nx,mode,boundary_cross));
}
static inline float d6_taper(device const float *phb,
                             device const float *msfu, device const float *msfv,
                             int k, int j, int i, int variant, bool x_axis,
                             int p, float threshold, int ny, int nx,
                             int mode, bool boundary_cross) {
    float face0, face1, dz;
    if (x_axis) {
        const float a0=fabs(d6_phb(phb,k,j,i,ny,nx,mode,boundary_cross)
                           -d6_phb(phb,k,j,i-1,ny,nx,mode,boundary_cross))
                         *d6_msfu(msfu,j,i,ny,nx,mode,boundary_cross);
        const float a1=fabs(d6_phb(phb,k,j,i+1,ny,nx,mode,boundary_cross)
                           -d6_phb(phb,k,j,i,ny,nx,mode,boundary_cross))
                         *d6_msfu(msfu,j,i+1,ny,nx,mode,boundary_cross);
        face0=a0; face1=a1;
        if (variant==1) {
            const float far=fabs(d6_phb(phb,k,j,i-1,ny,nx,mode,boundary_cross)
                                -d6_phb(phb,k,j,i-2,ny,nx,mode,boundary_cross))
                                *d6_msfu(msfu,j,i-1,ny,nx,mode,boundary_cross);
            dz=p==0?max(a0,far):max(a1,a0);
        } else if (variant==2) {
            const float b0=fabs(d6_phb(phb,k,j-1,i,ny,nx,mode,boundary_cross)
                               -d6_phb(phb,k,j-1,i-1,ny,nx,mode,boundary_cross))
                               *d6_msfu(msfu,j-1,i,ny,nx,mode,boundary_cross);
            const float b1=fabs(d6_phb(phb,k,j-1,i+1,ny,nx,mode,boundary_cross)
                               -d6_phb(phb,k,j-1,i,ny,nx,mode,boundary_cross))
                               *d6_msfu(msfu,j-1,i+1,ny,nx,mode,boundary_cross);
            dz=p==0?max(a0,b0):max(a1,b1);
        } else dz=p==0?a0:a1;
    } else {
        const float b0=fabs(d6_phb(phb,k,j,i,ny,nx,mode,boundary_cross)
                           -d6_phb(phb,k,j-1,i,ny,nx,mode,boundary_cross))
                         *d6_msfv(msfv,j,i,ny,nx,mode,boundary_cross);
        const float b1=fabs(d6_phb(phb,k,j+1,i,ny,nx,mode,boundary_cross)
                           -d6_phb(phb,k,j,i,ny,nx,mode,boundary_cross))
                         *d6_msfv(msfv,j+1,i,ny,nx,mode,boundary_cross);
        face0=b0; face1=b1;
        if (variant==1) {
            const float a0=fabs(d6_phb(phb,k,j,i-1,ny,nx,mode,boundary_cross)
                               -d6_phb(phb,k,j-1,i-1,ny,nx,mode,boundary_cross))
                             *d6_msfv(msfv,j,i-1,ny,nx,mode,boundary_cross);
            const float a1=fabs(d6_phb(phb,k,j+1,i-1,ny,nx,mode,boundary_cross)
                               -d6_phb(phb,k,j,i-1,ny,nx,mode,boundary_cross))
                             *d6_msfv(msfv,j+1,i-1,ny,nx,mode,boundary_cross);
            dz=p==0?max(b0,a0):max(b1,a1);
        } else if (variant==2) {
            const float far=fabs(d6_phb(phb,k,j-1,i,ny,nx,mode,boundary_cross)
                                -d6_phb(phb,k,j-2,i,ny,nx,mode,boundary_cross))
                              *d6_msfv(msfv,j-1,i,ny,nx,mode,boundary_cross);
            dz=p==0?max(b0,far):max(b1,b0);
        } else dz=p==0?b0:b1;
    }
    return max(1.0f - dz / threshold, 0.0f);
}

static inline void d6_cell(device const float *f, device float *tend,
                           device const float *mut, device const float *c1,
                           device const float *c2, device const float *phb,
                           device const float *msfu, device const float *msfv,
                           device const float *msft, int k, int j, int i,
                           int nlev, int ny, int nys, int nx, int nxs,
                           int variant, int wstag, int mono, int slopeopt,
                           float coef, float dzthr_x, float dzthr_y,
                           int mode, bool boundary_cross, bool replace) {
    if (wstag && (k == 0 || k == nlev - 1)) return;
    const float dx0 = d6_limited_flux(f,k,j,i,true,0,mono,nlev,ny,nys,nx,nxs,mode,boundary_cross);
    const float dx1 = d6_limited_flux(f,k,j,i,true,1,mono,nlev,ny,nys,nx,nxs,mode,boundary_cross);
    const float dy0 = d6_limited_flux(f,k,j,i,false,0,mono,nlev,ny,nys,nx,nxs,mode,boundary_cross);
    const float dy1 = d6_limited_flux(f,k,j,i,false,1,mono,nlev,ny,nys,nx,nxs,mode,boundary_cross);
    float sx0=1.0f, sx1=1.0f, sy0=1.0f, sy1=1.0f;
    if (slopeopt >= 1) {
        sx0=d6_taper(phb,msfu,msfv,k,j,i,variant,true,0,dzthr_x,ny,nx,mode,boundary_cross);
        sx1=d6_taper(phb,msfu,msfv,k,j,i,variant,true,1,dzthr_x,ny,nx,mode,boundary_cross);
        sy0=d6_taper(phb,msfu,msfv,k,j,i,variant,false,0,dzthr_y,ny,nx,mode,boundary_cross);
        sy1=d6_taper(phb,msfu,msfv,k,j,i,variant,false,1,dzthr_y,ny,nx,mode,boundary_cross);
    }
    float map;
    if (variant == 1) map=msfu[ulong(j)*ulong(nx+1)+ulong(i)];
    else if (variant == 2) map=msfv[ulong(j)*ulong(nx)+ulong(i)];
    else map=msft[d6_mass_index(d6_wrap(j,ny),d6_wrap(i,nx),nx)];
    // u's x flux and v's y flux use donor mass centers. Other fluxes
    // average the two/four cells bracketing the face: left uses (i-1,i),
    // right uses (i,i+1), and the y analogue. Do not shift that average.
    const int x0=i-(variant==1 ? 1 : 0), x1=i+(variant==1 ? 0 : 1);
    const int y0=j-(variant==2 ? 1 : 0), y1=j+(variant==2 ? 0 : 1);
    const float mx0=d6_face_mu(mut,c1,c2,k,j,x0,variant,true,ny,nx,mode,boundary_cross);
    const float mx1=d6_face_mu(mut,c1,c2,k,j,x1,variant,true,ny,nx,mode,boundary_cross);
    const float my0=d6_face_mu(mut,c1,c2,k,y0,i,variant,false,ny,nx,mode,boundary_cross);
    const float my1=d6_face_mu(mut,c1,c2,k,y1,i,variant,false,ny,nx,mode,boundary_cross);
    const float tx=coef*map*((sx1*mx1)*dx1-(sx0*mx0)*dx0);
    const float ty=coef*map*((sy1*my1)*dy1-(sy0*my0)*dy0);
    const ulong out=d6_field_index(k,j,i,nys,nxs);
    // CUDA's seam wrappers clear the target face before their += launch.
    // The Metal main kernel omits that face, so the seam can write the
    // equivalent replacement directly without unsupported host slicing.
    tend[out]=replace ? (tx+ty) : ((tend[out]+tx)+ty);
}

kernel void diff6(constant Diff6Resources &r [[buffer(0)]],
                  constant Diff6Params &p [[buffer(1)]],
                  uint flat [[thread_position_in_grid]]) {
    const uint count=uint(p.nlev)*uint(p.nys)*uint(p.nxs);
    if (flat>=count) return;
    const int i=int(flat%uint(p.nxs));
    const int j=int((flat/uint(p.nxs))%uint(p.nys));
    const int k=int(flat/(uint(p.nxs)*uint(p.nys)));
    if ((p.boundary_x && p.variant==1 && i==p.nx-3)
        || (p.boundary_y && p.variant==2 && j==p.ny-3)) return;
    d6_cell(r.f,r.tend,r.mut,r.c1,r.c2,r.phb,r.msfu,r.msfv,r.msft,
            k,j,i,p.nlev,p.ny,p.nys,p.nx,p.nxs,p.variant,p.wstag,p.mono,
            p.slopeopt,p.coef,p.dzthr_x,p.dzthr_y,0,false,false);
}

kernel void diff6_seam_u(constant Diff6SeamUResources &r [[buffer(0)]],
                         constant Diff6SeamUParams &p [[buffer(1)]],
                         uint flat [[thread_position_in_grid]]) {
    const int span=p.h1-p.h0+1;
    const uint count=uint(p.nlev)*uint(span);
    if (flat>=count) return;
    const int k=int(flat/uint(span));
    const int j=p.h0+int(flat%uint(span));
    d6_cell(r.f,r.tend,r.mut,r.c1,r.c2,r.phb,r.msfu,r.msfv,r.msft,
            k,j,p.nx-3,p.nlev,p.ny,p.ny,p.nx,p.nx+1,1,0,p.mono,p.slopeopt,
            p.coef,p.dzthr_x,p.dzthr_y,1,p.boundary_cross!=0,true);
}

kernel void diff6_seam_v(constant Diff6SeamVResources &r [[buffer(0)]],
                         constant Diff6SeamVParams &p [[buffer(1)]],
                         uint flat [[thread_position_in_grid]]) {
    const int span=p.h1-p.h0+1;
    const uint count=uint(p.nlev)*uint(span);
    if (flat>=count) return;
    const int k=int(flat/uint(span));
    const int i=p.h0+int(flat%uint(span));
    d6_cell(r.f,r.tend,r.mut,r.c1,r.c2,r.phb,r.msfu,r.msfv,r.msft,
            k,p.ny-3,i,p.nlev,p.ny,p.ny+1,p.nx,p.nx,2,0,p.mono,p.slopeopt,
            p.coef,p.dzthr_x,p.dzthr_y,2,p.boundary_cross!=0,true);
}
