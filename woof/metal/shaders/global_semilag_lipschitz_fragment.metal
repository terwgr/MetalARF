// Chunked FP64 spatial derivatives for the default SL Lipschitz diagnostic.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_lipschitz_fragment_bindings.metalh"

using namespace metal;

static inline uint gl_wrap(int x, uint n) {
    int m = x % int(n);
    return uint(m < 0 ? m + int(n) : m);
}

static inline ulong gl_sample(device const ulong *a, int k, int j, int i,
                              uint nk, uint nj, uint ni) {
    if (j < 0) { j = 0; i += int(ni / 2); }
    if (j >= int(nj)) { j = int(nj) - 1; i += int(ni / 2); }
    k = clamp(k, 0, int(nk) - 1);
    return a[uint(k) * nj * ni + uint(j) * ni + gl_wrap(i, ni)];
}

static inline void gl_derivatives(device const ulong *a, uint k, uint j, uint i,
                                  device const ulong *lat_ext,
                                  uint nk, uint nj, uint ni,
                                  ulong dlam, ulong inv_x, ulong inv_y,
                                  thread ulong &dx, thread ulong &dy,
                                  thread ulong &dk) {
    const ulong two = 0x4000000000000000ul;
    const ulong f = gl_sample(a, int(k), int(j), int(i), nk, nj, ni);
    const ulong xp = gl_sample(a, int(k), int(j), int(i)+1, nk, nj, ni);
    const ulong xm = gl_sample(a, int(k), int(j), int(i)-1, nk, nj, ni);
    const ulong dlam_diff = soft64_sub(xp, xm);
    dx = soft64_mul(inv_x, soft64_div(dlam_diff, soft64_mul(two, dlam)));
    const ulong f0 = gl_sample(a, int(k), int(j)-1, int(i), nk, nj, ni);
    const ulong f1 = f;
    const ulong f2 = gl_sample(a, int(k), int(j)+1, int(i), nk, nj, ni);
    const ulong x0 = lat_ext[j+1], x1 = lat_ext[j+2], x2 = lat_ext[j+3];
    const ulong c0 = soft64_div(soft64_sub(x1,x2),
        soft64_mul(soft64_sub(x0,x1),soft64_sub(x0,x2)));
    const ulong c1 = soft64_div(soft64_sub(soft64_sub(soft64_mul(x1,soft64_from_i32(2)),x0),x2),
        soft64_mul(soft64_sub(x1,x0),soft64_sub(x1,x2)));
    const ulong c2 = soft64_div(soft64_sub(x1,x0),
        soft64_mul(soft64_sub(x2,x0),soft64_sub(x2,x1)));
    dy = soft64_mul(inv_y, soft64_add(soft64_add(
        soft64_mul(f0,c0),soft64_mul(f1,c1)),soft64_mul(f2,c2)));
    ulong dlevel;
    if (k == 0u) dlevel = soft64_sub(gl_sample(a,1,int(j),int(i),nk,nj,ni), f);
    else if (k+1u == nk) dlevel = soft64_sub(f,gl_sample(a,int(k)-1,int(j),int(i),nk,nj,ni));
    else dlevel = soft64_mul(0x3fe0000000000000ul,
        soft64_sub(gl_sample(a,int(k)+1,int(j),int(i),nk,nj,ni),
                   gl_sample(a,int(k)-1,int(j),int(i),nk,nj,ni)));
    dk = dlevel;
}

kernel void global_semilag_lipschitz_fragment(
    constant GlobalSemilagLipschitzFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagLipschitzFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nlev=uint(p.nlevel), nlat=uint(p.nlat), nlon=uint(p.nlon);
    const uint plane=nlat*nlon, count=uint(p.chunk_levels)*plane;
    if(tid>=count)return;
    const uint k=uint(p.start_level)+tid/plane, cell=tid%plane;
    const uint j=cell/nlon, i=cell-j*nlon;
    const ulong dlam=r.dlam[0];
    const ulong inv_x=r.inv_x[j], inv_y=r.inv_y[0];
    ulong dx,dy,dk;
    gl_derivatives(r.vx,k,j,i,r.lat_ext,nlev,nlat,nlon,dlam,inv_x,inv_y,dx,dy,dk); r.vx_dx[tid]=dx; r.vx_dy[tid]=dy; r.vx_dk[tid]=dk;
    gl_derivatives(r.vy,k,j,i,r.lat_ext,nlev,nlat,nlon,dlam,inv_x,inv_y,dx,dy,dk); r.vy_dx[tid]=dx; r.vy_dy[tid]=dy; r.vy_dk[tid]=dk;
    gl_derivatives(r.vz,k,j,i,r.lat_ext,nlev,nlat,nlon,dlam,inv_x,inv_y,dx,dy,dk); r.vz_dx[tid]=dx; r.vz_dy[tid]=dy; r.vz_dk[tid]=dk;
    gl_derivatives(r.rate,k,j,i,r.lat_ext,nlev,nlat,nlon,dlam,inv_x,inv_y,dx,dy,dk); r.rate_dx[tid]=dx; r.rate_dy[tid]=dy; r.rate_dk[tid]=dk;
}
