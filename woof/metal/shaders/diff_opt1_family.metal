#include <metal_stdlib>
#include "wrf_diff_opt1_horizontal_bindings.metalh"
#include "wrf_diff_opt1_km4_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

// CUDA uses explicit __f*_rn intrinsics. These helpers and the disabled
// contraction preserve each source operation boundary for ordinary FP32.
inline float d1add(float a, float b) { return a + b; }
inline float d1sub(float a, float b) { return a - b; }
inline float d1mul(float a, float b) { return a * b; }
inline float d1div(float a, float b) { return a / b; }
inline float d1four(float a, float b, float c, float d) {
    return d1mul(0.25f, d1add(d1add(d1add(a, b), c), d));
}
inline int d1index(int n, int size, int boundary) {
    if (boundary) return max(0, min(n, size - 1));
    const int r = n % size;
    return r < 0 ? r + size : r;
}
inline ulong d1mass_index(int j, int i, int ny, int nx, int bx, int by) {
    return ulong(d1index(j, ny, by)) * ulong(nx)
         + ulong(d1index(i, nx, bx));
}
inline float d1mass(device const float *mu, device const float *c1,
                    device const float *c2, int k, int j, int i,
                    int ny, int nx, int bx, int by) {
    const ulong col = d1mass_index(j, i, ny, nx, bx, by);
    return d1add(d1mul(c1[k], mu[col]), c2[k]);
}
inline float d1kval(device const float *km, int k, int j, int i,
                    int nz, int ny, int nx, int bx, int by) {
    const int kk = max(0, min(k, nz - 1));
    const ulong col = d1mass_index(j, i, ny, nx, bx, by);
    return km[(ulong(kk) * ulong(ny) + ulong(d1index(j, ny, by)))
              * ulong(nx) + ulong(d1index(i, nx, bx))];
}
inline float d1field(device const float *field, int k, int j, int i,
                     int nz, int ny, int nx, int stag, int bx, int by) {
    const int nxs = nx + int(stag == 1);
    const int nys = ny + int(stag == 2);
    const int ii = stag == 1 && bx ? max(0, min(i, nx))
                                    : d1index(i, nx, stag == 1 ? 0 : bx);
    const int jj = stag == 2 && by ? max(0, min(j, ny))
                                    : d1index(j, ny, stag == 2 ? 0 : by);
    return field[(ulong(k) * ulong(nys) + ulong(jj)) * ulong(nxs) + ulong(ii)];
}
inline float d1base(device const float *base, int k, int j, int i,
                    int ny, int nx, int bx, int by) {
    return base[(ulong(k) * ulong(ny) + ulong(d1index(j, ny, by)))
                * ulong(nx) + ulong(d1index(i, nx, bx))];
}
inline float d1map(device const float *msft, device const float *msfu,
                   device const float *msfv, int stagger, int j, int i,
                   int ny, int nx, int bx, int by) {
    const int width = nx + int(stagger == 1);
    const int ii = stagger == 1 && bx ? max(0, min(i, nx))
                                      : d1index(i, nx, bx);
    const int jj = stagger == 2 && by ? max(0, min(j, ny))
                                      : d1index(j, ny, by);
    const ulong index = ulong(jj) * ulong(width) + ulong(ii);
    return stagger == 1 ? msfu[index] : stagger == 2 ? msfv[index] : msft[index];
}
inline float d1delta(device const float *field, device const float *base,
                     int k, int j1, int i1, int j0, int i0,
                     int nz, int ny, int nx, int stag, int bx, int by,
                     int perturb) {
    float value = d1sub(d1field(field, k, j1, i1, nz, ny, nx, stag, bx, by),
                        d1field(field, k, j0, i0, nz, ny, nx, stag, bx, by));
    if (perturb) {
        value = d1add(d1sub(value, d1base(base, k, j1, i1, ny, nx, bx, by)),
                      d1base(base, k, j0, i0, ny, nx, bx, by));
    }
    return value;
}

kernel void wrf_diff_opt1_horizontal(
    constant WrfDiffOpt1HorizontalResources &r [[buffer(0)]],
    constant WrfDiffOpt1HorizontalParams &q [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const int nz=q.nz, ny=q.ny, nx=q.nx, stag=q.stagger;
    const int nxs=nx+int(stag==1), nys=ny+int(stag==2), nzs=nz+int(stag==3);
    const ulong count=ulong(nxs)*ulong(nys)*ulong(nzs);
    if (ulong(flat)>=count) return;
    const int i=int(flat%uint(nxs));
    const int j=int((flat/uint(nxs))%uint(nys));
    const int k=int(flat/(uint(nxs)*uint(nys)));
    if (stag==3 && (k==0 || k==nz)) return;
    if (q.boundary_x && (i==0 || i>=nx-int(stag!=1))) return;
    if (q.boundary_y && (j==0 || j>=ny-int(stag!=2))) return;
    const int bx=q.boundary_x, by=q.boundary_y;
    float xm, xp, ym, yp;
    if (stag==1) {
        xm=d1mul(d1mul(d1mass(r.mu,r.c1,r.c2,k,j,i-1,ny,nx,bx,by),
                        d1kval(r.km,k,j,i-1,nz,ny,nx,bx,by)),q.rdx);
        xp=d1mul(d1mul(d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),
                        d1kval(r.km,k,j,i,nz,ny,nx,bx,by)),q.rdx);
        ym=d1mul(d1mul(d1four(d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j-1,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j-1,i-1,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i-1,ny,nx,bx,by)),
                        d1four(d1kval(r.km,k,j,i,nz,ny,nx,bx,by),d1kval(r.km,k,j-1,i,nz,ny,nx,bx,by),d1kval(r.km,k,j-1,i-1,nz,ny,nx,bx,by),d1kval(r.km,k,j,i-1,nz,ny,nx,bx,by))),q.rdy);
        yp=d1mul(d1mul(d1four(d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j+1,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j+1,i-1,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i-1,ny,nx,bx,by)),
                        d1four(d1kval(r.km,k,j,i,nz,ny,nx,bx,by),d1kval(r.km,k,j+1,i,nz,ny,nx,bx,by),d1kval(r.km,k,j+1,i-1,nz,ny,nx,bx,by),d1kval(r.km,k,j,i-1,nz,ny,nx,bx,by))),q.rdy);
    } else if (stag==2) {
        xm=d1mul(d1mul(d1four(d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j-1,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j-1,i-1,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i-1,ny,nx,bx,by)),
                        d1four(d1kval(r.km,k,j,i,nz,ny,nx,bx,by),d1kval(r.km,k,j-1,i,nz,ny,nx,bx,by),d1kval(r.km,k,j-1,i-1,nz,ny,nx,bx,by),d1kval(r.km,k,j,i-1,nz,ny,nx,bx,by))),q.rdx);
        xp=d1mul(d1mul(d1four(d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j-1,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j-1,i+1,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i+1,ny,nx,bx,by)),
                        d1four(d1kval(r.km,k,j,i,nz,ny,nx,bx,by),d1kval(r.km,k,j-1,i,nz,ny,nx,bx,by),d1kval(r.km,k,j-1,i+1,nz,ny,nx,bx,by),d1kval(r.km,k,j,i+1,nz,ny,nx,bx,by))),q.rdx);
        // Pinned WRF v4.7.1 coordinate-v meridional flux omits dry mass.
        ym=d1mul(d1kval(r.km,k,j-1,i,nz,ny,nx,bx,by),q.rdy);
        yp=d1mul(d1kval(r.km,k,j,i,nz,ny,nx,bx,by),q.rdy);
    } else if (stag==3) {
        xm=d1mul(d1mul(d1four(d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i-1,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i-1,ny,nx,bx,by)),
                        d1four(d1kval(r.km,k,j,i,nz,ny,nx,bx,by),d1kval(r.km,k,j,i-1,nz,ny,nx,bx,by),d1kval(r.km,k-1,j,i,nz,ny,nx,bx,by),d1kval(r.km,k-1,j,i-1,nz,ny,nx,bx,by))),q.rdx);
        xp=d1mul(d1mul(d1four(d1mass(r.mu,r.c1,r.c2,k,j,i+1,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i+1,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by)),
                        d1four(d1kval(r.km,k,j,i+1,nz,ny,nx,bx,by),d1kval(r.km,k,j,i,nz,ny,nx,bx,by),d1kval(r.km,k-1,j,i+1,nz,ny,nx,bx,by),d1kval(r.km,k-1,j,i,nz,ny,nx,bx,by))),q.rdx);
        ym=d1mul(d1mul(d1mul(d1map(r.msft,r.msfu,r.msfv,2,j,i,ny,nx,bx,by),d1div(1.0f,d1map(r.msft,r.msfu,r.msfv,2,j,i,ny,nx,bx,by))),
                        d1four(d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j-1,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j-1,i,ny,nx,bx,by))),
                        d1four(d1kval(r.km,k,j,i,nz,ny,nx,bx,by),d1kval(r.km,k,j-1,i,nz,ny,nx,bx,by),d1kval(r.km,k-1,j,i,nz,ny,nx,bx,by),d1kval(r.km,k-1,j-1,i,nz,ny,nx,bx,by)));
        ym=d1mul(ym,q.rdy);
        yp=d1mul(d1mul(d1mul(d1map(r.msft,r.msfu,r.msfv,2,j+1,i,ny,nx,bx,by),d1div(1.0f,d1map(r.msft,r.msfu,r.msfv,2,j+1,i,ny,nx,bx,by))),
                        d1four(d1mass(r.mu,r.c1,r.c2,k,j+1,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j+1,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by))),
                        d1four(d1kval(r.km,k,j+1,i,nz,ny,nx,bx,by),d1kval(r.km,k,j,i,nz,ny,nx,bx,by),d1kval(r.km,k-1,j+1,i,nz,ny,nx,bx,by),d1kval(r.km,k-1,j,i,nz,ny,nx,bx,by)));
        yp=d1mul(yp,q.rdy);
    } else {
        xm=d1mul(d1mul(d1mul(d1mul(0.5f,d1add(d1kval(r.km,k,j,i,nz,ny,nx,bx,by),d1kval(r.km,k,j,i-1,nz,ny,nx,bx,by))),0.5f),
                        d1add(d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i-1,ny,nx,bx,by))),q.rdx);
        xp=d1mul(d1mul(d1mul(d1mul(0.5f,d1add(d1kval(r.km,k,j,i+1,nz,ny,nx,bx,by),d1kval(r.km,k,j,i,nz,ny,nx,bx,by))),0.5f),
                        d1add(d1mass(r.mu,r.c1,r.c2,k,j,i+1,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by))),q.rdx);
        ym=d1mul(d1mul(d1mul(d1mul(d1mul(d1map(r.msft,r.msfu,r.msfv,2,j,i,ny,nx,bx,by),d1div(1.0f,d1map(r.msft,r.msfu,r.msfv,2,j,i,ny,nx,bx,by))),0.5f),
                        d1add(d1kval(r.km,k,j,i,nz,ny,nx,bx,by),d1kval(r.km,k,j-1,i,nz,ny,nx,bx,by))),0.5f),
                        d1add(d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j-1,i,ny,nx,bx,by))),q.rdy);
        yp=d1mul(d1mul(d1mul(d1mul(d1mul(d1map(r.msft,r.msfu,r.msfv,2,j+1,i,ny,nx,bx,by),d1div(1.0f,d1map(r.msft,r.msfu,r.msfv,2,j+1,i,ny,nx,bx,by))),0.5f),
                        d1add(d1kval(r.km,k,j+1,i,nz,ny,nx,bx,by),d1kval(r.km,k,j,i,nz,ny,nx,bx,by))),0.5f),
                        d1add(d1mass(r.mu,r.c1,r.c2,k,j+1,i,ny,nx,bx,by),d1mass(r.mu,r.c1,r.c2,k,j,i,ny,nx,bx,by))),q.rdy);
    }
    const int map_stag=stag==3 ? 0 : stag;
    const float map=d1map(r.msft,r.msfu,r.msfv,map_stag,j,i,ny,nx,bx,by);
    const float map2=d1mul(map,map);
    const float tx=d1mul(d1mul(map2,q.rdx),
        d1sub(d1mul(xp,d1delta(r.field,r.base,k,j,i+1,j,i,nz,ny,nx,stag,bx,by,q.perturb)),
              d1mul(xm,d1delta(r.field,r.base,k,j,i,j,i-1,nz,ny,nx,stag,bx,by,q.perturb))));
    const float ty=d1mul(d1mul(map2,q.rdy),
        d1sub(d1mul(yp,d1delta(r.field,r.base,k,j+1,i,j,i,nz,ny,nx,stag,bx,by,q.perturb)),
              d1mul(ym,d1delta(r.field,r.base,k,j,i,j-1,i,nz,ny,nx,stag,bx,by,q.perturb))));
    const ulong index=(ulong(k)*ulong(nys)+ulong(j))*ulong(nxs)+ulong(i);
    r.tendency[index]=d1add(r.tendency[index],d1add(tx,ty));
}

kernel void wrf_diff_opt1_km4(
    constant WrfDiffOpt1Km4Resources &r [[buffer(0)]],
    constant WrfDiffOpt1Km4Params &q [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const int nz=q.nz, ny=q.ny, nx=q.nx;
    const ulong count=ulong(nz)*ulong(ny)*ulong(nx);
    if (ulong(flat)>=count) return;
    const int i=int(flat%uint(nx));
    const int j=int((flat/uint(nx))%uint(ny));
    const int k=int(flat/(uint(nx)*uint(ny)));
    const ulong index=(ulong(k)*ulong(ny)+ulong(j))*ulong(nx)+ulong(i);
    if ((q.boundary_x && (i==0 || i==nx-1))
        || (q.boundary_y && (j==0 || j==ny-1))) {
        r.km[index]=0.0f;
        r.kh[index]=0.0f;
        return;
    }
    const int ip=d1index(i+1,nx,q.boundary_x);
    const int jp=d1index(j+1,ny,q.boundary_y);
    const float d=d1sub(r.d11[index],r.d22[index]);
    const ulong cross_index=(ulong(k)*ulong(ny)+ulong(j))*ulong(nx);
    const float cross=d1four(r.d12[cross_index+ulong(i)],
        r.d12[ulong(k*ny+jp)*ulong(nx)+ulong(i)],
        r.d12[ulong(k*ny+j)*ulong(nx)+ulong(ip)],
        r.d12[ulong(k*ny+jp)*ulong(nx)+ulong(ip)]);
    const float strain=sqrt(d1add(d1mul(0.25f,d1mul(d,d)),d1mul(cross,cross)));
    const float dxm=d1div(q.dx,r.msft[ulong(j)*ulong(nx)+ulong(i)]);
    const float maplen=d1div(d1mul(dxm,q.dy),r.msft[ulong(j)*ulong(nx)+ulong(i)]);
    const float length=sqrt(maplen);
    const float value=fmin(
        d1mul(d1mul(d1mul(d1mul(q.cs,q.cs),length),length),strain),
        d1mul(10.0f,length));
    r.km[index]=value;
    r.kh[index]=d1div(value,q.pr);
}
