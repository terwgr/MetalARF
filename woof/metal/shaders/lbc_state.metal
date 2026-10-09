#include <metal_stdlib>
#include "couple_nest_field_bindings.metalh"
#include "couple_nest_field_window_bindings.metalh"
#include "couple_nest_field_frame_bindings.metalh"
#include "state_specified_relaxation_bindings.metalh"
#include "install_mu_boundary_bindings.metalh"
#include "uncouple_feedback_field_bindings.metalh"
#include "finalize_state_field_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline int frame_rows(int d, int ny) { return ny - 1 - d > d ? 2 : 1; }
static inline int frame_cols(int d, int nx) { return nx - 1 - d > d ? 2 : 1; }

static inline bool frame_point(int p, int ny, int nx, int width,
                               thread int &j, thread int &i) {
    for (int d = 0; d < width; ++d) {
        int ni = nx - 2*d, nj = max(ny - 2*d - 2, 0);
        int rows = frame_rows(d, ny), ring = rows*ni + frame_cols(d, nx)*nj;
        if (p >= ring) { p -= ring; continue; }
        if (p < ni) { j = d; i = d + p; }
        else if (p < rows*ni) { j = ny - 1 - d; i = d + p - ni; }
        else if (p < rows*ni + nj) { j = d + 1 + p - rows*ni; i = d; }
        else { j = d + 1 + p - rows*ni - nj; i = nx - 1 - d; }
        return true;
    }
    return false;
}

static inline float hybrid_point(float c1, float c2,
                                 device const float *mub,
                                 device const float *mup,
                                 int j, int i, int nx) {
    ulong q = ulong(j)*ulong(nx) + ulong(i);
    float base = c1*mub[q] + c2;
    float perturbation = c1*mup[q];
    return base + perturbation;
}

static inline float u_hybrid(float c1, float c2,
                             device const float *mub, device const float *mup,
                             int j, int i, int ny, int nx) {
    if (i == 0) {
        ulong q = ulong(j)*ulong(nx);
        float base = c1*mub[q] + c2, perturbation = c1*mup[q];
        float sum = base + base; sum = sum + perturbation;
        sum = sum + perturbation; return 0.5f*sum;
    }
    if (i == nx) return hybrid_point(c1, c2, mub, mup, j, nx-1, nx);
    ulong right = ulong(j)*ulong(nx) + ulong(i), left = right-1;
    float br = c1*mub[right] + c2, bl = c1*mub[left] + c2;
    float sum = br + bl; sum = sum + c1*mup[right];
    sum = sum + c1*mup[left]; return 0.5f*sum;
}

static inline float v_hybrid(float c1, float c2,
                             device const float *mub, device const float *mup,
                             int j, int i, int ny, int nx) {
    if (j == 0) {
        ulong q = ulong(i);
        float base = c1*mub[q] + c2, perturbation = c1*mup[q];
        float sum = base + base; sum = sum + perturbation;
        sum = sum + perturbation; return 0.5f*sum;
    }
    if (j == ny) return hybrid_point(c1, c2, mub, mup, ny-1, i, nx);
    ulong north = ulong(j)*ulong(nx) + ulong(i), south = north-ulong(nx);
    float bn = c1*mub[north] + c2, bs = c1*mub[south] + c2;
    float sum = bn + bs; sum = sum + c1*mup[north];
    sum = sum + c1*mup[south]; return 0.5f*sum;
}

static inline void couple_value(device const float *target,
                                device const float *mub2d,
                                device const float *mup,
                                device const float *thb,
                                device const float *c1h,
                                device const float *c2h,
                                device const float *c1f,
                                device const float *c2f,
                                device const float *msft,
                                device const float *msfu,
                                device const float *msfv,
                                device float *out,
                                int has_msf,int thb_3d,int kind,int wrf_exact,
                                int ny,int nx,int mny,int mnx,
                                int k,int j,int i,ulong index) {
    if (kind == 4) { out[index] = target[index]; return; }
    float ch, value = target[index];
    if (kind == 0) {
        ch = u_hybrid(c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf) ch = ch/msfu[ulong(j)*ulong(nx)+ulong(i)];
        out[index] = ch*value; return;
    }
    if (kind == 1) {
        ch = v_hybrid(c1h[k], c2h[k], mub2d, mup, j, i, mny, mnx);
        if (has_msf) ch = ch/msfv[ulong(j)*ulong(nx)+ulong(i)];
        out[index] = ch*value; return;
    }
    if (kind == 6) {
        ch = hybrid_point(c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        if (has_msf) ch = ch/msft[ulong(j)*ulong(nx)+ulong(i)];
        out[index] = ch*value; return;
    }
    if (kind == 3) {
        ch = hybrid_point(c1f[k], c2f[k], mub2d, mup, j, i, mnx);
        out[index] = ch*value; return;
    }
    ch = hybrid_point(c1h[k], c2h[k], mub2d, mup, j, i, mnx);
    if (kind == 2 && !wrf_exact) {
        ulong h = thb_3d ? (ulong(k)*ulong(mny)+ulong(j))*ulong(mnx)+ulong(i) : ulong(k);
        value = (thb[h] + value) - 300.0f;
    }
    out[index] = ch*value;
}

kernel void couple_nest_field(constant CoupleNestFieldResources &r [[buffer(0)]],
                              constant CoupleNestFieldParams &q [[buffer(1)]],
                              uint tid [[thread_position_in_grid]]) {
    if (ulong(tid) >= ulong(q.nz)*ulong(q.ny)*ulong(q.nx)) return;
    int k = int(ulong(tid)/(ulong(q.ny)*ulong(q.nx)));
    ulong rem = ulong(tid)-ulong(k)*ulong(q.ny)*ulong(q.nx);
    int j = int(rem/ulong(q.nx)), i = int(rem-ulong(j)*ulong(q.nx));
    couple_value(r.target,r.mub2d,r.mup,r.thb,r.c1h,r.c2h,r.c1f,r.c2f,
                 r.msft,r.msfu,r.msfv,r.out,q.has_msf,q.thb_3d,q.kind,
                 q.wrf_exact,q.ny,q.nx,q.mny,q.mnx,k,j,i,ulong(tid));
}

kernel void couple_nest_field_window(constant CoupleNestFieldWindowResources &r [[buffer(0)]],
                                     constant CoupleNestFieldWindowParams &q [[buffer(1)]],
                                     uint tid [[thread_position_in_grid]]) {
    if (ulong(tid) >= ulong(q.nz)*ulong(q.nj)*ulong(q.ni)) return;
    int k = int(ulong(tid)/(ulong(q.nj)*ulong(q.ni)));
    ulong rem = ulong(tid)-ulong(k)*ulong(q.nj)*ulong(q.ni);
    int j = q.j0 + int(rem/ulong(q.ni)), i = q.i0 + int(rem%ulong(q.ni));
    ulong index = (ulong(k)*ulong(q.ny)+ulong(j))*ulong(q.nx)+ulong(i);
    couple_value(r.target,r.mub2d,r.mup,r.thb,r.c1h,r.c2h,r.c1f,r.c2f,
                 r.msft,r.msfu,r.msfv,r.out,q.has_msf,q.thb_3d,q.kind,
                 q.wrf_exact,q.ny,q.nx,q.mny,q.mnx,k,j,i,index);
}

kernel void couple_nest_field_frame(constant CoupleNestFieldFrameResources &r [[buffer(0)]],
                                    constant CoupleNestFieldFrameParams &q [[buffer(1)]],
                                    uint tid [[thread_position_in_grid]]) {
    if (ulong(tid) >= ulong(q.nz)*ulong(q.points)) return;
    int k = int(ulong(tid)/ulong(q.points)), j, i;
    if (!frame_point(int(ulong(tid)-ulong(k)*ulong(q.points)), q.ny, q.nx,
                     q.width, j, i)) return;
    ulong index = (ulong(k)*ulong(q.ny)+ulong(j))*ulong(q.nx)+ulong(i);
    couple_value(r.target,r.mub2d,r.mup,r.thb,r.c1h,r.c2h,r.c1f,r.c2f,
                 r.msft,r.msfu,r.msfv,r.out,q.has_msf,q.thb_3d,q.kind,
                 q.wrf_exact,q.ny,q.nx,q.mny,q.mnx,k,j,i,index);
}

// Shared source-order helpers for state relaxation, MU installation and the
// two inverse transforms. Explicit temporaries plus contract(off) retain the
// CUDA expression tree; parity remains unverified until compiled and compared.
static inline float current_mu(device const float *mub, device const float *mup,
                               int j, int i, int nx) {
    ulong q = ulong(j)*ulong(nx)+ulong(i); return mub[q]+mup[q];
}
static inline float u_mu(device const float *mub, device const float *mup,
                         int j, int i, int ny, int nx) {
    if (i==0) return current_mu(mub,mup,j,0,nx);
    if (i==nx) return current_mu(mub,mup,j,nx-1,nx);
    return 0.5f*(current_mu(mub,mup,j,i,nx)+current_mu(mub,mup,j,i-1,nx));
}
static inline float v_mu(device const float *mub, device const float *mup,
                         int j, int i, int ny, int nx) {
    if (j==0) return current_mu(mub,mup,0,i,nx);
    if (j==ny) return current_mu(mub,mup,ny-1,i,nx);
    return 0.5f*(current_mu(mub,mup,j,i,nx)+current_mu(mub,mup,j-1,i,nx));
}
static inline int frame_offset(int j,int i,int ny,int nx,int width) {
    int d=min(min(j,ny-1-j),min(i,nx-1-i));
    if (d<0 || d>=width) return -1;
    int off=0;
    for(int q=0;q<d;++q)
        off+=frame_rows(q,ny)*(nx-2*q)+frame_cols(q,nx)*max(ny-2*q-2,0);
    int ni=nx-2*d,nj=max(ny-2*d-2,0),rows=frame_rows(d,ny);
    if(j==d) return off+i-d;
    if(j==ny-1-d) return off+ni+i-d;
    if(i==d) return off+rows*ni+j-d-1;
    return off+rows*ni+nj+j-d-1;
}
static inline float old_mu_value(device const float *oldf,device const float *mub,
                                 device const float *mup,int j,int i,
                                 int ny,int nx,int zone) {
    int f=frame_offset(j,i,ny,nx,zone);
    return mub[ulong(j)*ulong(nx)+ulong(i)]+(f>=0?oldf[f]:mup[ulong(j)*ulong(nx)+ulong(i)]);
}
static inline float old_u_mu(device const float *oldf,device const float *mub,
                             device const float *mup,int j,int i,int ny,int nx,int zone) {
    if(i==0)return old_mu_value(oldf,mub,mup,j,0,ny,nx,zone);
    if(i==nx)return old_mu_value(oldf,mub,mup,j,nx-1,ny,nx,zone);
    return 0.5f*(old_mu_value(oldf,mub,mup,j,i,ny,nx,zone)+old_mu_value(oldf,mub,mup,j,i-1,ny,nx,zone));
}
static inline float old_v_mu(device const float *oldf,device const float *mub,
                             device const float *mup,int j,int i,int ny,int nx,int zone) {
    if(j==0)return old_mu_value(oldf,mub,mup,0,i,ny,nx,zone);
    if(j==ny)return old_mu_value(oldf,mub,mup,ny-1,i,ny,nx,zone);
    return 0.5f*(old_mu_value(oldf,mub,mup,j,i,ny,nx,zone)+old_mu_value(oldf,mub,mup,j-1,i,ny,nx,zone));
}

static inline float coupled_relax_value(constant StateSpecifiedRelaxationResources &r,
                                        constant StateSpecifiedRelaxationParams &q,
                                        int kind,int k,int j,int i) {
    int mny=q.mny,mnx=q.mnx;
    if(kind==4)return r.mup[ulong(j)*ulong(mnx)+ulong(i)];
    ulong ix=(ulong(k)*ulong(q.ny)+ulong(j))*ulong(q.nx)+ulong(i);
    float mass,ch;
    if(kind==0){mass=u_mu(r.mub2d,r.mup,j,i,mny,mnx);ch=r.c1h[k]*mass+r.c2h[k];float z=ch*r.u[ix];return q.has_msf?z/r.msfu[ulong(j)*ulong(q.nx)+ulong(i)]:z;}
    if(kind==1){mass=v_mu(r.mub2d,r.mup,j,i,mny,mnx);ch=r.c1h[k]*mass+r.c2h[k];float z=ch*r.v[ix];return q.has_msf?z/r.msfv[ulong(j)*ulong(q.nx)+ulong(i)]:z;}
    mass=current_mu(r.mub2d,r.mup,j,i,mnx);
    if(kind==6){ch=r.c1f[k]*mass+r.c2f[k];return ch*r.w[ix];}
    if(kind==3){ch=r.c1f[k]*mass+r.c2f[k];return ch*r.php[ix];}
    ch=r.c1h[k]*mass+r.c2h[k];
    if(kind==2){ulong h=q.thb_3d?(ulong(k)*ulong(mny)+ulong(j))*ulong(mnx)+ulong(i):ulong(k);return ch*((r.thb[h]+r.thp[ix])-300.0f);}
    return ch*r.scalar[ix];
}
static inline float lbc_interp(device const float *v,device const float *t,ulong b,float dt){return v[b]+dt*t[b];}
static inline float installed_interp(device const float *v,device const float *t,ulong b,float dt){return v[b]+dt*t[b];}

static inline bool state_boundary_index(constant StateSpecifiedRelaxationResources &r,
                                       constant StateSpecifiedRelaxationParams &q,
                                       int k,int j,int i,thread float &value,bool tendency_only) {
    int ds=j,dn=q.ny-1-j,dw=i,de=q.nx-1-i; ulong b; device const float *v,*t;
    if(ds<q.spec_zone && i>=ds && i<q.nx-ds){b=(ulong(k)*ulong(q.width)+ulong(ds))*ulong(q.nx)+ulong(i);v=r.south;t=r.south_t;}
    else if(dn<q.spec_zone && i>=dn && i<q.nx-dn){b=(ulong(k)*ulong(q.width)+ulong(dn))*ulong(q.nx)+ulong(i);v=r.north;t=r.north_t;}
    else if(dw<q.spec_zone && j>=dw+1 && j<q.ny-dw-1){b=(ulong(k)*ulong(q.ny)+ulong(j))*ulong(q.width)+ulong(dw);v=r.west;t=r.west_t;}
    else if(de<q.spec_zone && j>=de+1 && j<q.ny-de-1){b=(ulong(k)*ulong(q.ny)+ulong(j))*ulong(q.width)+ulong(de);v=r.east;t=r.east_t;}
    else return false;
    value=tendency_only?t[b]:installed_interp(v,t,b,q.dtbc); return true;
}
static inline float residual_at(constant StateSpecifiedRelaxationResources &r,
                                constant StateSpecifiedRelaxationParams &q,
                                int side,int k,int j,int i,int d) {
    int ny=q.ny,nx=q.nx,w=q.width;
    device const float *v=side==0?r.south:(side==1?r.north:(side==2?r.west:r.east));
    device const float *t=side==0?r.south_t:(side==1?r.north_t:(side==2?r.west_t:r.east_t));
    ulong b0,bm,bp,bo,bi; int jm=j,jp=j;
    if(side<2){b0=(ulong(k)*ulong(w)+ulong(d))*ulong(nx)+ulong(i);bm=b0-(i>0?1:0);bp=b0+(i+1<nx?1:0);bo=b0-ulong(nx);bi=b0+ulong(nx);}
    else {jm=max(j-1,0);jp=min(j+1,ny-1);b0=(ulong(k)*ulong(ny)+ulong(j))*ulong(w)+ulong(d);bm=(ulong(k)*ulong(ny)+ulong(jm))*ulong(w)+ulong(d);bp=(ulong(k)*ulong(ny)+ulong(jp))*ulong(w)+ulong(d);bo=b0-1;bi=b0+1;}
    // CUDA orders f1/f2 as lateral neighbors and f3/f4 as normal neighbors.
    int j1=j,i1=i,j2=j,i2=i,j3=j,i3=i,j4=j,i4=i;
    if(side==0){i1=max(i-1,0);i2=min(i+1,nx-1);j3=j-1;j4=j+1;}
    else if(side==1){i1=max(i-1,0);i2=min(i+1,nx-1);j3=j+1;j4=j-1;}
    else if(side==2){j1=max(j-1,0);j2=min(j+1,ny-1);i3=i-1;i4=i+1;}
    else {j1=max(j-1,0);j2=min(j+1,ny-1);i3=i+1;i4=i-1;}
    float f0=lbc_interp(v,t,b0,q.dtbc)-coupled_relax_value(r,q,q.kind,k,j,i);
    float f1=lbc_interp(v,t,bm,q.dtbc)-coupled_relax_value(r,q,q.kind,k,j1,i1);
    float f2=lbc_interp(v,t,bp,q.dtbc)-coupled_relax_value(r,q,q.kind,k,j2,i2);
    float f3=lbc_interp(v,t,bo,q.dtbc)-coupled_relax_value(r,q,q.kind,k,j3,i3);
    float f4=lbc_interp(v,t,bi,q.dtbc)-coupled_relax_value(r,q,q.kind,k,j4,i4);
    return r.fcx[d]*f0-r.gcx[d]*(f1+f2+f3+f4-4.0f*f0);
}

kernel void state_specified_relaxation(constant StateSpecifiedRelaxationResources &r [[buffer(0)]],
                                       constant StateSpecifiedRelaxationParams &q [[buffer(1)]],
                                       uint tid [[thread_position_in_grid]]) {
    ulong total=ulong(q.nz)*ulong(q.ny)*ulong(q.nx); if(ulong(tid)>=total)return;
    int k=int(ulong(tid)/(ulong(q.ny)*ulong(q.nx)));
    ulong rem=ulong(tid)-ulong(k)*ulong(q.ny)*ulong(q.nx);
    int j=int(rem/ulong(q.nx)),i=int(rem-ulong(j)*ulong(q.nx));
    int ds=j,dn=q.ny-1-j,dw=i,de=q.nx-1-i;
    int active=max(q.spec_zone,q.relax_zone);
    if(min(min(ds,dn),min(dw,de))>=active && !q.divide_msf && !q.add_held)return;
    ulong idx=ulong(tid); float result=r.field_tend[idx],specified;
    if(state_boundary_index(r,q,k,j,i,specified,true)) result=q.clear_specified?0.0f:specified;
    else if(q.apply_relax){
        bool os=(q.relax_sides&1)!=0,on=(q.relax_sides&2)!=0,ow=(q.relax_sides&4)!=0,oe=(q.relax_sides&8)!=0;
        if(os&&ds>=q.spec_zone&&ds<q.relax_zone&&(!ow||i>=ds)&&(!oe||i<q.nx-ds))result+=residual_at(r,q,0,k,j,i,ds);
        else if(on&&dn>=q.spec_zone&&dn<q.relax_zone&&(!ow||i>=dn)&&(!oe||i<q.nx-dn))result+=residual_at(r,q,1,k,j,i,dn);
        else if(ow&&dw>=q.spec_zone&&dw<q.relax_zone&&(!os||j>=dw+1)&&(!on||j<q.ny-dw-1))result+=residual_at(r,q,2,k,j,i,dw);
        else if(oe&&de>=q.spec_zone&&de<q.relax_zone&&(!os||j>=de+1)&&(!on||j<q.ny-de-1))result+=residual_at(r,q,3,k,j,i,de);
    }
    if(q.divide_msf)result=result/r.msft[ulong(j)*ulong(q.nx)+ulong(i)];
    if(q.add_held)result=result+r.held[idx];
    r.field_tend[idx]=result;
}

kernel void install_mu_boundary(constant InstallMuBoundaryResources &r [[buffer(0)]],
                                constant InstallMuBoundaryParams &q [[buffer(1)]],
                                uint p [[thread_position_in_grid]]) {
    if(p>=uint(q.frame_count))return; int j,i;
    if(!frame_point(int(p),q.ny,q.nx,q.spec_zone,j,i))return;
    ulong ix=ulong(j)*ulong(q.nx)+ulong(i); r.old_mup_frame[p]=r.mup[ix];
    float value;
    // This is the same Y-corners-own boundary selector as state relaxation.
    int ds=j,dn=q.ny-1-j,dw=i,de=q.nx-1-i; ulong b;
    if(ds<q.spec_zone&&i>=ds&&i<q.nx-ds){b=ulong(ds)*ulong(q.nx)+ulong(i);value=r.south[b]+q.dtbc*r.south_t[b];r.mup[ix]=value;}
    else if(dn<q.spec_zone&&i>=dn&&i<q.nx-dn){b=ulong(dn)*ulong(q.nx)+ulong(i);value=r.north[b]+q.dtbc*r.north_t[b];r.mup[ix]=value;}
    else if(dw<q.spec_zone&&j>=dw+1&&j<q.ny-dw-1){b=ulong(j)*ulong(q.width)+ulong(dw);r.mup[ix]=r.west[b]+q.dtbc*r.west_t[b];}
    else if(de<q.spec_zone&&j>=de+1&&j<q.ny-de-1){b=ulong(j)*ulong(q.width)+ulong(de);r.mup[ix]=r.east[b]+q.dtbc*r.east_t[b];}
}

kernel void uncouple_feedback_field(constant UncoupleFeedbackFieldResources &r [[buffer(0)]],
                                    constant UncoupleFeedbackFieldParams &q [[buffer(1)]],
                                    uint tid [[thread_position_in_grid]]) {
    int ni=q.i_hi-q.i_lo+1,nj=q.j_hi-q.j_lo+1;
    if(ulong(tid)>=ulong(q.nz)*ulong(nj)*ulong(ni))return;
    int k=int(ulong(tid)/(ulong(nj)*ulong(ni)));
    ulong rem=ulong(tid)-ulong(k)*ulong(nj)*ulong(ni);
    int j=q.j_lo+int(rem/ulong(ni)),i=q.i_lo+int(rem%ulong(ni));
    ulong ix=(ulong(k)*ulong(q.ny)+ulong(j))*ulong(q.nx)+ulong(i);
    float value=r.coupled[ix],ch;
    if(q.kind==0){ch=u_hybrid(r.c1h[k],r.c2h[k],r.mub2d,r.mup,j,i,q.mny,q.mnx);if(q.has_msf)value=value*r.msfu[ulong(j)*ulong(q.nx)+ulong(i)];r.target[ix]=value/ch;return;}
    if(q.kind==1){ch=v_hybrid(r.c1h[k],r.c2h[k],r.mub2d,r.mup,j,i,q.mny,q.mnx);if(q.has_msf)value=value*r.msfv[ulong(j)*ulong(q.nx)+ulong(i)];r.target[ix]=value/ch;return;}
    if(q.kind==6){ch=hybrid_point(r.c1f[k],r.c2f[k],r.mub2d,r.mup,j,i,q.mnx);if(q.has_msf)value=value*r.msft[ulong(j)*ulong(q.nx)+ulong(i)];r.target[ix]=value/ch;return;}
    if(q.kind==3){ch=hybrid_point(r.c1f[k],r.c2f[k],r.mub2d,r.mup,j,i,q.mnx);r.target[ix]=value/ch;return;}
    ch=hybrid_point(r.c1h[k],r.c2h[k],r.mub2d,r.mup,j,i,q.mnx);float result=value/ch;
    if(q.kind==2){ulong h=q.thb_3d?(ulong(k)*ulong(q.mny)+ulong(j))*ulong(q.mnx)+ulong(i):ulong(k);result=(result+300.0f)-r.thb[h];}
    r.target[ix]=result;
}

static inline float coupled_old(constant FinalizeStateFieldResources &r,
                                constant FinalizeStateFieldParams &q,
                                int k,int j,int i) {
    ulong ix=(ulong(k)*ulong(q.ny)+ulong(j))*ulong(q.nx)+ulong(i);
    float mass,ch,z;
    if(q.kind==0){mass=old_u_mu(r.old_mup_frame,r.mub2d,r.mup,j,i,q.mny,q.mnx,q.spec_zone);ch=r.c1h[k]*mass+r.c2h[k];z=ch*r.target[ix];return q.has_msf?z/r.msfu[ulong(j)*ulong(q.nx)+ulong(i)]:z;}
    if(q.kind==1){mass=old_v_mu(r.old_mup_frame,r.mub2d,r.mup,j,i,q.mny,q.mnx,q.spec_zone);ch=r.c1h[k]*mass+r.c2h[k];z=ch*r.target[ix];return q.has_msf?z/r.msfv[ulong(j)*ulong(q.nx)+ulong(i)]:z;}
    mass=old_mu_value(r.old_mup_frame,r.mub2d,r.mup,j,i,q.mny,q.mnx,q.spec_zone);
    if(q.kind==6){ch=r.c1f[k]*mass+r.c2f[k];z=ch*r.target[ix];return q.has_msf?z/r.msft[ulong(j)*ulong(q.nx)+ulong(i)]:z;}
    if(q.kind==3){ch=r.c1f[k]*mass+r.c2f[k];return ch*r.target[ix];}
    ch=r.c1h[k]*mass+r.c2h[k];
    if(q.kind==2){ulong h=q.thb_3d?ix:ulong(k);return ch*((r.thb[h]+r.target[ix])-300.0f);}
    return ch*r.target[ix];
}
static inline bool final_boundary(constant FinalizeStateFieldResources &r,
                                 constant FinalizeStateFieldParams &q,
                                 int k,int j,int i,thread float &value) {
    int ds=j,dn=q.ny-1-j,dw=i,de=q.nx-1-i;ulong b;
    if(ds<q.spec_zone&&i>=ds&&i<q.nx-ds){b=(ulong(k)*ulong(q.width)+ulong(ds))*ulong(q.nx)+ulong(i);value=r.south[b]+q.dtbc*r.south_t[b];return true;}
    if(dn<q.spec_zone&&i>=dn&&i<q.nx-dn){b=(ulong(k)*ulong(q.width)+ulong(dn))*ulong(q.nx)+ulong(i);value=r.north[b]+q.dtbc*r.north_t[b];return true;}
    if(dw<q.spec_zone&&j>=dw+1&&j<q.ny-dw-1){b=(ulong(k)*ulong(q.ny)+ulong(j))*ulong(q.width)+ulong(dw);value=r.west[b]+q.dtbc*r.west_t[b];return true;}
    if(de<q.spec_zone&&j>=de+1&&j<q.ny-de-1){b=(ulong(k)*ulong(q.ny)+ulong(j))*ulong(q.width)+ulong(de);value=r.east[b]+q.dtbc*r.east_t[b];return true;}
    return false;
}
kernel void finalize_state_field(constant FinalizeStateFieldResources &r [[buffer(0)]],
                                 constant FinalizeStateFieldParams &q [[buffer(1)]],
                                 uint tid [[thread_position_in_grid]]) {
    if(ulong(tid)>=ulong(q.nz)*ulong(q.ny)*ulong(q.nx))return;
    int k=int(ulong(tid)/(ulong(q.ny)*ulong(q.nx)));ulong rem=ulong(tid)-ulong(k)*ulong(q.ny)*ulong(q.nx);
    int j=int(rem/ulong(q.nx)),i=int(rem-ulong(j)*ulong(q.nx));
    float coupled;
    if(!final_boundary(r,q,k,j,i,coupled))coupled=coupled_old(r,q,k,j,i);
    ulong ix=ulong(tid);float mass,ch;
    if(q.kind==0){mass=u_mu(r.mub2d,r.mup,j,i,q.mny,q.mnx);ch=r.c1h[k]*mass+r.c2h[k];r.target[ix]=(coupled*r.msfu[ulong(j)*ulong(q.nx)+ulong(i)])/ch;}
    else if(q.kind==1){mass=v_mu(r.mub2d,r.mup,j,i,q.mny,q.mnx);ch=r.c1h[k]*mass+r.c2h[k];r.target[ix]=(coupled*r.msfv[ulong(j)*ulong(q.nx)+ulong(i)])/ch;}
    else if(q.kind==3){mass=current_mu(r.mub2d,r.mup,j,i,q.mnx);ch=r.c1f[k]*mass+r.c2f[k];r.target[ix]=coupled/ch;}
    else if(q.kind==6){mass=current_mu(r.mub2d,r.mup,j,i,q.mnx);ch=r.c1f[k]*mass+r.c2f[k];r.target[ix]=(q.has_msf?coupled*r.msft[ulong(j)*ulong(q.nx)+ulong(i)]:coupled)/ch;}
    else {mass=current_mu(r.mub2d,r.mup,j,i,q.mnx);ch=r.c1h[k]*mass+r.c2h[k];float z=coupled/ch;if(q.kind==2){ulong h=q.thb_3d?(ulong(k)*ulong(q.mny)+ulong(j))*ulong(q.mnx)+ulong(i):ulong(k);z=(z+300.0f)-r.thb[h];}else if(!isnan(z))z=max(z,0.0f);r.target[ix]=z;}
}
