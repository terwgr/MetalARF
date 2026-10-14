#include <metal_stdlib>
#include "pd_fluxes_bindings.metalh"
#include "pd_renorm_apply_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;

// Ordinary FP32 transcription of pd_advection.cu. Keep contraction disabled
// so each source-level update remains a distinct round point.
#pragma clang fp contract(off)

static inline ulong cidx(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx) + ulong(i);
}
static inline ulong xidx(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx + 1) + ulong(i);
}
static inline ulong yidx(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny + 1) + ulong(j)) * ulong(nx) + ulong(i);
}
static inline int wrap(int i, int n) { int r=i%n; return r<0?r+n:r; }
static inline float f5(float a,float b,float c,float d,float e,float f,float v) {
    return (v*(37.0f*(d+c)-8.0f*(e+b)+(f+a))
      -abs(v)*(10.0f*(d-c)-5.0f*(e-b)+(f-a)))/60.0f;
}
static inline float f3(float a,float b,float c,float d,float v) {
    return (v*(7.0f*(c+b)-(d+a))
      +abs(v)*(3.0f*(c-b)-(d-a)))/12.0f;
}
static inline float f3h(float a,float b,float c,float d,float v) {
    return (v*(7.0f*(c+b)-(d+a))
      -abs(v)*(3.0f*(c-b)-(d-a)))/12.0f;
}
static inline float f5v(float a,float b,float c,float d,float e,float f,float v) {
    return (v*(37.0f*(d+c)-8.0f*(e+b)+(f+a))
      +abs(v)*(10.0f*(d-c)-5.0f*(e-b)+(f-a)))/60.0f;
}
static inline float upwind(float a,float b,float cr) {
    return 0.5f*min(1.0f,cr+abs(cr))*a
         + 0.5f*max(-1.0f,cr-abs(cr))*b;
}
static inline float zhigh(constant PdFluxesResources &r, int k, int j,
                           int i, float vel, constant PdFluxesParams &p) {
    if (k==0 || k==p.nz) return 0.0f;
    if (k==1 || k==p.nz-1)
        return vel*(r.fnm[k]*r.q[cidx(k,j,i,p.ny,p.nx)]
                   +r.fnp[k]*r.q[cidx(k-1,j,i,p.ny,p.nx)]);
    if (p.vorder==5 && k>=3 && k<=p.nz-3)
        return f5v(r.q[cidx(k-3,j,i,p.ny,p.nx)],
                   r.q[cidx(k-2,j,i,p.ny,p.nx)],
                   r.q[cidx(k-1,j,i,p.ny,p.nx)],
                   r.q[cidx(k,j,i,p.ny,p.nx)],
                   r.q[cidx(k+1,j,i,p.ny,p.nx)],
                   r.q[cidx(k+2,j,i,p.ny,p.nx)],vel);
    return f3(r.q[cidx(k-2,j,i,p.ny,p.nx)],
              r.q[cidx(k-1,j,i,p.ny,p.nx)],
              r.q[cidx(k,j,i,p.ny,p.nx)],
              r.q[cidx(k+1,j,i,p.ny,p.nx)],vel);
}

kernel void pd_fluxes(constant PdFluxesResources &r [[buffer(0)]],
                      constant PdFluxesParams &p [[buffer(1)]],
                      uint flat [[thread_position_in_grid]]) {
    const ulong total=ulong(p.nz+1)*ulong(p.ny+1)*ulong(p.nx+1);
    if (ulong(flat)>=total) return;
    const int i=int(flat%uint(p.nx+1));
    const int j=int((flat/uint(p.nx+1))%uint(p.ny+1));
    const int k=int(flat/(uint(p.nx+1)*uint(p.ny+1)));
    if (j<p.ny && k<p.nz) {
        float fl=0.0f, fc=0.0f;
        if (!(p.open_x && (i==0 || i==p.nx))) {
            const int il=p.open_x?i-1:wrap(i-1,p.nx);
            const int ir=p.open_x?i:wrap(i,p.nx);
            const float vel=r.ru[xidx(k,j,i,p.ny,p.nx)];
    const float muf=r.c1h[k]*0.5f*(r.mut[ulong(j)*p.nx+il]
                                             +r.mut[ulong(j)*p.nx+ir])+r.c2h[k];
            const float dxp=p.has_msf ? p.dx*2.0f/
                (r.msft[ulong(j)*p.nx+il]+r.msft[ulong(j)*p.nx+ir]) : p.dx;
            const float cr=vel*p.dt/dxp/muf;
            fl=muf*(dxp/p.dt)*upwind(r.q0[cidx(k,j,il,p.ny,p.nx)],
                                     r.q0[cidx(k,j,ir,p.ny,p.nx)],cr);
            float fh;
            if (p.open_x) {
                const int d=min(i,p.nx-i);
                if (d==1) fh=0.5f*vel*(r.q[cidx(k,j,i,p.ny,p.nx)]+r.q[cidx(k,j,i-1,p.ny,p.nx)]);
                else if (d==2) fh=f3h(r.q[cidx(k,j,i-2,p.ny,p.nx)],r.q[cidx(k,j,i-1,p.ny,p.nx)],r.q[cidx(k,j,i,p.ny,p.nx)],r.q[cidx(k,j,i+1,p.ny,p.nx)],vel);
                else fh=f5(r.q[cidx(k,j,i-3,p.ny,p.nx)],r.q[cidx(k,j,i-2,p.ny,p.nx)],r.q[cidx(k,j,i-1,p.ny,p.nx)],r.q[cidx(k,j,i,p.ny,p.nx)],r.q[cidx(k,j,i+1,p.ny,p.nx)],r.q[cidx(k,j,i+2,p.ny,p.nx)],vel);
            } else {
                fh=f5(r.q[cidx(k,j,wrap(i-3,p.nx),p.ny,p.nx)],r.q[cidx(k,j,wrap(i-2,p.nx),p.ny,p.nx)],r.q[cidx(k,j,wrap(i-1,p.nx),p.ny,p.nx)],r.q[cidx(k,j,wrap(i,p.nx),p.ny,p.nx)],r.q[cidx(k,j,wrap(i+1,p.nx),p.ny,p.nx)],r.q[cidx(k,j,wrap(i+2,p.nx),p.ny,p.nx)],vel);
            }
            fc=fh-fl;
        }
        r.fxl[xidx(k,j,i,p.ny,p.nx)]=fl; r.fxc[xidx(k,j,i,p.ny,p.nx)]=fc;
    }
    if (i<p.nx && k<p.nz) {
        float fl=0.0f, fc=0.0f;
        if (!(p.open_y && (j==0 || j==p.ny))) {
            const int jl=p.open_y?j-1:wrap(j-1,p.ny);
            const int jr=p.open_y?j:wrap(j,p.ny);
            const float vel=r.rv[yidx(k,j,i,p.ny,p.nx)];
            const float muf=r.c1h[k]*0.5f*(r.mut[ulong(jl)*p.nx+i]
                                             +r.mut[ulong(jr)*p.nx+i])+r.c2h[k];
            const float dyp=p.has_msf ? p.dy*2.0f/
                (r.msft[ulong(jl)*p.nx+i]+r.msft[ulong(jr)*p.nx+i]) : p.dy;
            const float cr=vel*p.dt/dyp/muf;
            fl=muf*(dyp/p.dt)*upwind(r.q0[cidx(k,jl,i,p.ny,p.nx)],r.q0[cidx(k,jr,i,p.ny,p.nx)],cr);
            float fh;
            if (p.open_y) {
                const int d=min(j,p.ny-j);
                if (d==1) fh=0.5f*vel*(r.q[cidx(k,j,i,p.ny,p.nx)]+r.q[cidx(k,j-1,i,p.ny,p.nx)]);
                else if (d==2) fh=f3h(r.q[cidx(k,j-2,i,p.ny,p.nx)],r.q[cidx(k,j-1,i,p.ny,p.nx)],r.q[cidx(k,j,i,p.ny,p.nx)],r.q[cidx(k,j+1,i,p.ny,p.nx)],vel);
                else fh=f5(r.q[cidx(k,j-3,i,p.ny,p.nx)],r.q[cidx(k,j-2,i,p.ny,p.nx)],r.q[cidx(k,j-1,i,p.ny,p.nx)],r.q[cidx(k,j,i,p.ny,p.nx)],r.q[cidx(k,j+1,i,p.ny,p.nx)],r.q[cidx(k,j+2,i,p.ny,p.nx)],vel);
            } else {
                fh=f5(r.q[cidx(k,wrap(j-3,p.ny),i,p.ny,p.nx)],r.q[cidx(k,wrap(j-2,p.ny),i,p.ny,p.nx)],r.q[cidx(k,wrap(j-1,p.ny),i,p.ny,p.nx)],r.q[cidx(k,wrap(j,p.ny),i,p.ny,p.nx)],r.q[cidx(k,wrap(j+1,p.ny),i,p.ny,p.nx)],r.q[cidx(k,wrap(j+2,p.ny),i,p.ny,p.nx)],vel);
            }
            fc=fh-fl;
        }
        r.fyl[yidx(k,j,i,p.ny,p.nx)]=fl; r.fyc[yidx(k,j,i,p.ny,p.nx)]=fc;
    }
    if (i<p.nx && j<p.ny) {
        float fl=0.0f,fc=0.0f;
        if (k>0 && k<p.nz) {
            const float vel=r.rw[cidx(k,j,i,p.ny,p.nx)];
            const float dz=2.0f/(r.rdnw[k]+r.rdnw[k-1]);
            const float muf=r.c1h[k]*r.mut[ulong(j)*p.nx+i]+r.c2h[k];
            const float cr=vel*p.dt/(dz*muf);
            fl=muf*(dz/p.dt)*upwind(r.q0[cidx(k-1,j,i,p.ny,p.nx)],r.q0[cidx(k,j,i,p.ny,p.nx)],cr);
            const float sl_dz=2.0f/(r.rdnw[k]+r.rdnw[k-1]);
            const float sl_cr=((vel*p.dt)/sl_dz)/muf;
            if (p.vorder==5 && k>1 && k<p.nz-1
                && (sl_cr>1.0f || sl_cr< -1.0f)) {
                int shift=int(floor(abs(sl_cr)));
                if (sl_cr<0.0f) shift=-shift;
                const int low=(k==p.nz-2)?-2:-3;
                const int high=(k==2)?2:3;
                shift=clamp(shift,low,high);
                const int first=(sl_cr>1.0f)?k-shift:k;
                const int last=(sl_cr>1.0f)?k-1:k-shift-1;
                float sum=0.0f;
                for (int level=first; level<=last; ++level)
                    sum=sum+r.q0[cidx(level,j,i,p.ny,p.nx)];
                // The source rounds vel*sum in binary32, then divides in
                // binary64 and rounds the result back to binary32.
                const uint product_bits=as_type<uint>(vel*sum);
                const ulong numerator=soft64_from_f32_bits(product_bits);
                const ulong denominator=soft64_from_f32_bits(as_type<uint>(abs(sl_cr)));
                fl=as_type<float>(soft64_to_f32_bits(soft64_div(numerator,denominator)));
            }
            fc=zhigh(r,k,j,i,vel,p)-fl;
        }
        r.fzl[cidx(k,j,i,p.ny,p.nx)]=fl; r.fzc[cidx(k,j,i,p.ny,p.nx)]=fc;
    }
}

static inline float pdscale(constant PdRenormApplyResources &r,
                            constant PdRenormApplyParams &p,
                            int k,int j,int i) {
    if ((p.open_x&&(i==0||i==p.nx-1)) || (p.open_y&&(j==0||j==p.ny-1))) return 1.0f;
    const ulong cell=ulong(j)*p.nx+i;
    const float chm=r.c1h[k]*r.mu_old[cell]+r.c2h[k];
    const float m=p.has_msf?r.msft[cell]:1.0f, m2=m*m;
    const float xl=r.fxl[xidx(k,j,i,p.ny,p.nx)], xr=r.fxl[xidx(k,j,i+1,p.ny,p.nx)];
    const float yl=r.fyl[yidx(k,j,i,p.ny,p.nx)], yr=r.fyl[yidx(k,j+1,i,p.ny,p.nx)];
    const float zl=r.fzl[cidx(k,j,i,p.ny,p.nx)], zr=r.fzl[cidx(k+1,j,i,p.ny,p.nx)];
    const float qcell=r.q0[cidx(k,j,i,p.ny,p.nx)];
    float ph, out;
    if (p.has_msf) {
        // WRF's mapped branch weights both horizontal divergences by m^2
        // and the vertical divergence by m. Keep its nested expression tree.
        ph=chm*qcell-p.dt*(m2*(p.dx_inv*(xr-xl)+p.dy_inv*(yr-yl))
                           +m*r.rdnw[k]*(zr-zl));
    } else {
        // The original no-map branch first forms dl in x, y, z order.
        // Substituting m=1 into the mapped expression changes FP32 grouping.
        const float dl=p.dx_inv*(xr-xl)+p.dy_inv*(yr-yl)
                     +r.rdnw[k]*(zr-zl);
        ph=chm*qcell-p.dt*dl;
    }
    const float xc0=r.fxc[xidx(k,j,i,p.ny,p.nx)], xc1=r.fxc[xidx(k,j,i+1,p.ny,p.nx)];
    const float yc0=r.fyc[yidx(k,j,i,p.ny,p.nx)], yc1=r.fyc[yidx(k,j+1,i,p.ny,p.nx)];
    const float zc0=r.fzc[cidx(k,j,i,p.ny,p.nx)], zc1=r.fzc[cidx(k+1,j,i,p.ny,p.nx)];
    if (p.has_msf) {
        out=p.dt*(m2*(p.dx_inv*(max(0.0f,xc1)-min(0.0f,xc0))
                       +p.dy_inv*(max(0.0f,yc1)-min(0.0f,yc0)))
                  +m*r.rdnw[k]*(min(0.0f,zc1)-max(0.0f,zc0)));
    } else {
        // Match CUDA's no-map fo grouping and directional order exactly.
        out=p.dt*(p.dx_inv*(max(0.0f,xc1)-min(0.0f,xc0))
                 +p.dy_inv*(max(0.0f,yc1)-min(0.0f,yc0))
                 +r.rdnw[k]*(min(0.0f,zc1)-max(0.0f,zc0)));
    }
    return out>ph?clamp(ph/(out+1.0e-20f),0.0f,1.0f):1.0f;
}

kernel void pd_renorm_apply(constant PdRenormApplyResources &r [[buffer(0)]],
                            constant PdRenormApplyParams &p [[buffer(1)]],
                            uint flat [[thread_position_in_grid]]) {
    if (flat>=uint(p.nz*p.ny*p.nx)) return;
    const int i=int(flat%uint(p.nx)), j=int((flat/uint(p.nx))%uint(p.ny));
    const int k=int(flat/uint(p.nx*p.ny));
    const float sc=pdscale(r,p,k,j,i);
    const float xl=r.fxc[xidx(k,j,i,p.ny,p.nx)], xr=r.fxc[xidx(k,j,i+1,p.ny,p.nx)];
    const float yl=r.fyc[yidx(k,j,i,p.ny,p.nx)], yr=r.fyc[yidx(k,j+1,i,p.ny,p.nx)];
    const float zl=r.fzc[cidx(k,j,i,p.ny,p.nx)], zr=r.fzc[cidx(k+1,j,i,p.ny,p.nx)];
    const float sxl=xl>0.0f?pdscale(r,p,k,wrap(j,p.ny),wrap(i-1,p.nx)):(xl<0.0f?sc:1.0f);
    const float sxr=xr>0.0f?sc:(xr<0.0f?pdscale(r,p,k,wrap(j,p.ny),wrap(i+1,p.nx)):1.0f);
    const float syl=yl>0.0f?pdscale(r,p,k,wrap(j-1,p.ny),i):(yl<0.0f?sc:1.0f);
    const float syr=yr>0.0f?sc:(yr<0.0f?pdscale(r,p,k,wrap(j+1,p.ny),i):1.0f);
    const float szl=zl>0.0f?sc:(zl<0.0f?(k>0?pdscale(r,p,k-1,j,i):1.0f):1.0f);
    const float szr=zr<0.0f?sc:(zr>0.0f?(k+1<p.nz?pdscale(r,p,k+1,j,i):1.0f):1.0f);
    const float zb=(szr*zr+r.fzl[cidx(k+1,j,i,p.ny,p.nx)])
                  -(szl*zl+r.fzl[cidx(k,j,i,p.ny,p.nx)]);
    const float xb=(sxr*xr+r.fxl[xidx(k,j,i+1,p.ny,p.nx)])
                  -(sxl*xl+r.fxl[xidx(k,j,i,p.ny,p.nx)]);
    const float yb=(syr*yr+r.fyl[yidx(k,j+1,i,p.ny,p.nx)])
                  -(syl*yl+r.fyl[yidx(k,j,i,p.ny,p.nx)]);
    const ulong cell_index=cidx(k,j,i,p.ny,p.nx);
    if (!p.open_x && !p.open_y) {
        const float m=p.has_msf?r.msft[ulong(j)*p.nx+i]:1.0f;
        float t;
        if (p.has_msf)
            t=-m*p.dx_inv*xb-m*p.dy_inv*yb-r.rdnw[k]*zb;
        else
            t=-p.dx_inv*xb-p.dy_inv*yb-r.rdnw[k]*zb;
        r.tend[cell_index]+=t;
        return;
    }
    float t=-r.rdnw[k]*zb;
    if (!p.open_x || (i>=1&&i<=p.nx-2))
        t+=-(p.has_msf?r.msft[ulong(j)*p.nx+i]:1.0f)*p.dx_inv*xb;
    if (!p.open_y || (j>=1&&j<=p.ny-2))
        t+=-(p.has_msf?r.msft[ulong(j)*p.nx+i]:1.0f)*p.dy_inv*yb;
    r.tend[cell_index]+=t;
}
