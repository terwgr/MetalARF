#include <metal_stdlib>
#include "slow_geopotential_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float sg_base(device const float *x, uint k, ulong c, ulong s, bool b3) {
    return x[b3 ? ulong(k)*s+c : ulong(k)];
}
static inline float sg_total(device const float *php, device const float *phb,
                             uint k, ulong c, ulong s, bool b3) {
    return sg_base(phb,k,c,s,b3)+php[ulong(k)*s+c];
}
static inline float sg_mut(device const float *mup, device const float *mub, ulong c) {
    return mub[c]+mup[c];
}
static inline int sg_face(int x, int n) { return x == n ? 0 : x; }
static inline int sg_prev(int x, int n) { return x == 0 ? n-1 : x-1; }

static inline float sg_flux(int axis, int face, int i, int j, uint k, bool top,
    constant SlowGeopotentialResources &r, constant SlowGeopotentialParams &q,
    ulong s, bool b3, bool supplied, bool include_map) {
    const int n = axis ? q.nx : q.ny;
    const int fi = sg_face(face,n);
    float mass;
    if (supplied) {
        mass = r.c1f[k] * (axis ? r.mux[ulong(j)*(q.nx+1)+fi]
                              : r.muy[ulong(fi)*q.nx+i]) + r.c2f[k];
    } else {
        const ulong ca = axis ? ulong(j)*q.nx+fi : ulong(fi)*q.nx+i;
        const ulong cb = axis ? ulong(j)*q.nx+sg_prev(fi,q.nx)
                              : ulong(sg_prev(fi,q.ny))*q.nx+i;
        mass = r.c1f[k] * (0.5f*(sg_mut(r.mup,r.mub2d,ca)+sg_mut(r.mup,r.mub2d,cb))) + r.c2f[k];
    }
    float wind;
    float map_factor=1.0f;
    if (axis) {
        const ulong f=ulong(k-1)*q.ny*(q.nx+1)+ulong(j)*(q.nx+1)+fi;
        wind=top ? q.cfn*r.u[f]+q.cfn1*r.u[f-ulong(q.ny)*(q.nx+1)]
                 : r.u[f+ulong(q.ny)*(q.nx+1)]+r.u[f];
        if (include_map && q.has_msf) map_factor=r.msfu[ulong(j)*(q.nx+1)+fi];
    } else {
        const ulong f=ulong(k-1)*(q.ny+1)*q.nx+ulong(fi)*q.nx+i;
        wind=top ? q.cfn*r.v[f]+q.cfn1*r.v[f-ulong((q.ny+1)*q.nx)]
                 : r.v[f+ulong((q.ny+1)*q.nx)]+r.v[f];
        if (include_map && q.has_msf) map_factor=r.msfv[ulong(fi)*q.nx+i];
    }
    float value=mass*wind;
    if (include_map && q.has_msf) value=value*map_factor;
    return value;
}

static inline float sg_phi_diff(device const float *php, device const float *phb,
    int axis,int i,int j,uint k,int off,int ny,int nx,ulong s,bool b3) {
    const int pos=axis?i:j, n=axis?nx:ny;
    const int plus=(pos+off+n)%n, minus=(pos-off+n)%n;
    const ulong cp=axis?ulong(j)*nx+plus:ulong(plus)*nx+i;
    const ulong cm=axis?ulong(j)*nx+minus:ulong(minus)*nx+i;
    return sg_total(php,phb,k,cp,s,b3)-sg_total(php,phb,k,cm,s,b3);
}

kernel void slow_geopotential(constant SlowGeopotentialResources &r [[buffer(0)]],
                              constant SlowGeopotentialParams &q [[buffer(1)]],
                              uint flat [[thread_position_in_grid]]) {
    const uint nz=uint(q.nz), ny=uint(q.ny), nx=uint(q.nx);
    const ulong s=ulong(ny)*nx, count=ulong(nz)*s;
    if (ulong(flat)>=count) return;
    const uint k=flat/uint(s)+1;
    const ulong c=ulong(flat)-ulong(k-1)*s;
    const int j=int(c/nx), i=int(c-ulong(j)*nx);
    const ulong ix=ulong(k)*s+c;
    const bool top=k==nz, b3=bool(q.base3d), supplied=q.mode==1;
    float t=(top && q.add_vertical)?0.0f:r.rph_t[ix];
    if (q.mode==2 || q.add_vertical) {
        if (!top) {
            const float phm=sg_total(r.php,r.phb,k-1,c,s,b3);
            const float ph=sg_total(r.php,r.phb,k,c,s,b3);
            const float php=sg_total(r.php,r.phb,k+1,c,s,b3);
            const float lo=((0.5f*(r.ww[ix]+r.ww[ix-s]))*r.rdnw[k-1])*(ph-phm);
            const float hi=((0.5f*(r.ww[ix+s]+r.ww[ix]))*r.rdnw[k])*(php-ph);
            t=t-(r.fnm[k]*hi+r.fnp[k]*lo);
        }
        const float mass=r.c1f[k]*sg_mut(r.mup,r.mub2d,c)+r.c2f[k];
        float gw=mass*(9.81f*r.w[ix]);
        if(q.has_msf) gw=gw/r.msft[c];
        t=t+gw;
    }
    if(q.mode==2) { r.rph_t[ix]=t; return; }
    float qx=q.quarter_rdx, qy=q.quarter_rdy;
    if(top) { qx=2.0f*qx; qy=2.0f*qy; }
    if(q.order==2) {
        // The ordinary rhs_ph source applies the x contribution before y.
        for(int pass=0;pass<2;++pass) {
            const int axis = pass==0;
            const int pos=axis?i:j, n=axis?int(nx):int(ny);
            const bool boundary=axis?bool(q.boundary_x):bool(q.boundary_y);
            const float qr=axis?qx:qy;
            float h=0.0f;
            if(!boundary || (pos!=0 && pos!=n-1)) {
                const int prev=sg_prev(pos,n), next=sg_face(pos+1,n);
                const ulong cm=axis?ulong(j)*nx+prev:ulong(prev)*nx+i;
                const ulong cp=axis?ulong(j)*nx+next:ulong(next)*nx+i;
                const float here=sg_total(r.php,r.phb,k,c,s,b3);
                const float d0=here-sg_total(r.php,r.phb,k,cm,s,b3);
                const float d1=sg_total(r.php,r.phb,k,cp,s,b3)-here;
                float f0=sg_flux(axis,pos,i,j,k,top,r,q,s,b3,supplied,false);
                float f1=sg_flux(axis,pos+1,i,j,k,top,r,q,s,b3,supplied,false);
                if(q.has_msf) {
                    const float mf0=axis?r.msfu[ulong(j)*(nx+1)+pos]:r.msfv[ulong(pos)*nx+i];
                    const float mf1=axis?r.msfu[ulong(j)*(nx+1)+sg_face(pos+1,n)]:r.msfv[ulong(sg_face(pos+1,n))*nx+i];
                    f0=f0*mf0; f1=f1*mf1;
                }
                h=qr*(f1*d1+f0*d0);
                if(q.has_msf) h=h/r.msft[c];
            }
            t=t-h;
        }
    } else {
        float hx=0.0f, hy=0.0f;
        for(int axis=0;axis<2;++axis) {
            const int pos=axis?i:j, n=axis?int(nx):int(ny);
            const bool boundary=axis?bool(q.boundary_x):bool(q.boundary_y);
            const bool radiative=axis?bool(q.open_x):bool(q.open_y);
            const float qr=axis?qx:qy;
            int stencil=6;
            if(boundary && (pos==0 || pos==n-1)) {
                if(radiative) {
                    const bool lower=pos==0;
                    const ulong c0=axis?ulong(j)*nx+ulong(pos)
                                       :ulong(pos)*nx+i;
                    const ulong c1=axis?ulong(j)*nx+ulong(pos+(lower?1:-1))
                                       :ulong(pos+(lower?1:-1))*nx+i;
                    const ulong f0=axis?ulong(k-1)*ny*(nx+1)
                                           +ulong(j)*(nx+1)+ulong(pos)
                                       :ulong(k-1)*(ny+1)*nx
                                           +ulong(pos)*nx+i;
                    const ulong f1=axis?f0+1:f0+ulong(nx);
                    float normal;
                    if(top) {
                        const ulong plane=axis?ulong(ny)*(nx+1)
                                             :ulong(ny+1)*nx;
                        const float a=axis
                            ? r.fnm[k-1]*(r.u[f0]+r.u[f1])
                              +r.fnp[k-1]*(r.u[f0-plane]+r.u[f1-plane])
                            : r.fnm[k-1]*(r.v[f0]+r.v[f1])
                              +r.fnp[k-1]*(r.v[f0-plane]+r.v[f1-plane]);
                        normal=0.5f*a;
                    } else {
                        const ulong plane=axis?ulong(ny)*(nx+1)
                                             :ulong(ny+1)*nx;
                        const float a=axis
                            ? r.fnm[k]*(r.u[f0+plane]+r.u[f1+plane])
                              +r.fnp[k]*(r.u[f0]+r.u[f1])
                            : r.fnm[k]*(r.v[f0+plane]+r.v[f1+plane])
                              +r.fnp[k]*(r.v[f0]+r.v[f1]);
                        normal=0.5f*a;
                    }
                    normal=lower?min(normal,0.0f):max(normal,0.0f);
                    // WRF rhs_ph passes its ph argument as ph_old here;
                    // this boundary-only upwind term uses perturbation phi,
                    // while the centered interior stencil uses ph + phb.
                    const float phi0=r.php[ulong(k)*s+c0];
                    const float phi1=r.php[ulong(k)*s+c1];
                    const float dphi=lower?phi1-phi0:phi0-phi1;
                    const float rd=axis?4.0f*q.quarter_rdx
                                       :4.0f*q.quarter_rdy;
                    const float advective=normal*dphi;
                    t=t-(rd*sg_mut(r.mup,r.mub2d,c0))*advective;
                }
                continue;
            }
            if(radiative && !q.specified
                    && (pos==1 || pos==n-2)) stencil=3;
            else if(radiative && !q.specified
                    && (pos==2 || pos==n-3)) stencil=4;
            if(q.specified && (pos<3 || pos>=n-3)) {
                if(pos==1 || pos==n-2) stencil=2;
                else if(!axis && (pos==2 || pos==n-3)) stencil=4;
                else continue;
            }
            const float f0=sg_flux(axis,pos,i,j,k,top,r,q,s,b3,supplied,true);
            const float f1=sg_flux(axis,pos+1,i,j,k,top,r,q,s,b3,supplied,true);
            float contribution;
            if(stencil==2) {
                const int pm=sg_prev(pos,n), pp=sg_face(pos+1,n);
                const ulong cm=axis?ulong(j)*nx+pm:ulong(pm)*nx+i;
                const ulong cp=axis?ulong(j)*nx+pp:ulong(pp)*nx+i;
                const float here=sg_total(r.php,r.phb,k,c,s,b3);
                const float d0=here-sg_total(r.php,r.phb,k,cm,s,b3);
                const float d1=sg_total(r.php,r.phb,k,cp,s,b3)-here;
                contribution=qr*(f1*d1+f0*d0);
            } else {
                float raw;
                if(stencil==3) {
                    raw=0.5f*sg_phi_diff(r.php,r.phb,axis,i,j,k,1,ny,nx,s,b3);
                } else if(stencil==4) {
                    const float t1=8.0f*sg_phi_diff(r.php,r.phb,axis,i,j,k,1,ny,nx,s,b3);
                    const float t2=-1.0f*sg_phi_diff(r.php,r.phb,axis,i,j,k,2,ny,nx,s,b3);
                    raw=(t1+t2)/12.0f;
                } else {
                    const float t1=45.0f*sg_phi_diff(r.php,r.phb,axis,i,j,k,1,ny,nx,s,b3);
                    const float t2=-9.0f*sg_phi_diff(r.php,r.phb,axis,i,j,k,2,ny,nx,s,b3);
                    const float t3=sg_phi_diff(r.php,r.phb,axis,i,j,k,3,ny,nx,s,b3);
                    raw=((t1+t2)+t3)/60.0f;
                }
                contribution=qr*((f1+f0)*raw);
            }
            if(q.has_msf) contribution=contribution/r.msft[c];
            if(axis) hx=contribution; else hy=contribution;
        }
        // Source order is y then x; both were computed independently above.
        t=t-(hy+hx);
    }
    r.rph_t[ix]=t;
}
