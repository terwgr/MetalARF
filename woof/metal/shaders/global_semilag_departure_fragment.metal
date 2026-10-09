// FP64 spherical fixed-point departure search for the default SL integrator.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_semilag_departure_fragment_bindings.metalh"

using namespace metal;

static inline int sld_floor(ulong x) {
    Soft64I32Result r = soft64_to_i32_rz(x);
    if (soft64_less(x, soft64_from_i32(r.value))) --r.value;
    return r.value;
}
static inline ulong sld_atan_unit(ulong x) {
    const ulong one=0x3ff0000000000000ul, pi4=0x3fe921fb54442d18ul;
    const ulong tanpi8=0x3fda827999fcef32ul;
    bool shifted=soft64_greater(x,tanpi8);
    if (shifted) x=soft64_div(soft64_sub(x,one),soft64_add(x,one));
    ulong x2=soft64_mul(x,x), term=x, sum=x;
    for (int n=1;n<24;++n) {
        term=soft64_negate_bits(soft64_mul(term,x2));
        sum=soft64_add(sum,soft64_div(term,soft64_from_i32(2*n+1)));
    }
    return shifted ? soft64_add(pi4,sum) : sum;
}
static inline ulong sld_atan2(ulong y, ulong x) {
    const ulong pi=0x400921fb54442d18ul, pi2=0x3ff921fb54442d18ul;
    ulong ax=soft64_abs_bits(x), ay=soft64_abs_bits(y);
    ulong a=soft64_greater_equal(ax,ay)
        ? sld_atan_unit(soft64_div(ay,ax))
        : soft64_sub(pi2,sld_atan_unit(soft64_div(ax,ay)));
    if (soft64_sign(x)) a=soft64_sub(pi,a);
    if (soft64_sign(y)) a=soft64_negate_bits(a);
    return a;
}
static inline int sld_bracket(ulong phi,
    device const ulong *lat_ext, device const int *mlut,
    int bins, int nlat, ulong lut_scale) {
    const ulong half_pi=0x3ff921fb54442d18ul;
    const int b=clamp(sld_floor(soft64_mul(
        soft64_add(phi,half_pi),lut_scale)),0,bins-1);
    int m=mlut[b];
    while (m>1 && soft64_less(phi,lat_ext[m])) --m;
    while (m<nlat+1 && !soft64_less(phi,lat_ext[m+1])) ++m;
    return m;
}
static inline void sld_trilinear(
    device const ulong *wind_vx, device const ulong *wind_vy,
    device const ulong *wind_vz, device const ulong *wind_level,
    ulong lam, ulong phi, ulong level,
    device const ulong *lat_ext, device const int *mlut,
    device const int *rowoff, device const int *rowshift,
    int bins, int nk, int nj, int ni, ulong inv_dlam, ulong lut_scale,
    thread ulong *out) {
    const ulong zero=0ul, one=0x3ff0000000000000ul;
    const ulong half_pi=0x3ff921fb54442d18ul;
    if (soft64_less(phi,soft64_negate_bits(half_pi))) phi=soft64_negate_bits(half_pi);
    if (soft64_greater(phi,half_pi)) phi=half_pi;
    const ulong top=soft64_from_i32(nk-1);
    if (soft64_less(level,zero)) level=zero;
    if (soft64_greater(level,top)) level=top;
    ulong xi=soft64_mul(lam,inv_dlam);
    int i0=sld_floor(xi); ulong ax=soft64_sub(xi,soft64_from_i32(i0));
    int c0=i0%ni; if(c0<0)c0+=ni; int c1=(c0+1)%ni;
    int d0=(c0+ni/2)%ni, d1=(c1+ni/2)%ni;
    const int m=sld_bracket(phi,lat_ext,mlut,bins,nj,lut_scale);
    const ulong ay=soft64_div(soft64_sub(phi,lat_ext[m]),
        soft64_sub(lat_ext[m+1],lat_ext[m]));
    const int r0=rowoff[m],r1=rowoff[m+1];
    int i00=r0+(rowshift[m]?d0:c0), i01=r0+(rowshift[m]?d1:c1);
    int i10=r1+(rowshift[m+1]?d0:c0), i11=r1+(rowshift[m+1]?d1:c1);
    int k0=clamp(sld_floor(level),0,nk-2);
    const ulong az=soft64_sub(level,soft64_from_i32(k0));
    const ulong bx=soft64_sub(one,ax), by=soft64_sub(one,ay), bz=soft64_sub(one,az);
    const uint plane=uint(nj)*uint(ni);
    for(uint f=0;f<4;++f) {
        device const ulong *wind = f == 0 ? wind_vx :
            (f == 1 ? wind_vy : (f == 2 ? wind_vz : wind_level));
        const uint o=uint(k0)*plane;
        ulong a0=wind[o+uint(i00)], a1=wind[o+uint(i01)];
        ulong a2=wind[o+uint(i10)], a3=wind[o+uint(i11)];
        ulong t0=soft64_add(soft64_mul(by,soft64_add(soft64_mul(bx,a0),soft64_mul(ax,a1))),
                           soft64_mul(ay,soft64_add(soft64_mul(bx,a2),soft64_mul(ax,a3))));
        a0=wind[o+plane+uint(i00)]; a1=wind[o+plane+uint(i01)];
        a2=wind[o+plane+uint(i10)]; a3=wind[o+plane+uint(i11)];
        ulong t1=soft64_add(soft64_mul(by,soft64_add(soft64_mul(bx,a0),soft64_mul(ax,a1))),
                           soft64_mul(ay,soft64_add(soft64_mul(bx,a2),soft64_mul(ax,a3))));
        out[f]=soft64_add(soft64_mul(bz,t0),soft64_mul(az,t1));
    }
}

kernel void global_semilag_departure_fragment(
    constant GlobalSemilagDepartureFragmentResources &r [[buffer(0)]],
    constant GlobalSemilagDepartureFragmentParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const uint n=uint(p.nlevel)*uint(p.nlat)*uint(p.nlon);
    if(gid>=n)return;
    const uint plane=uint(p.nlat)*uint(p.nlon);
    const int k=int(gid/plane);
    const ulong one=0x3ff0000000000000ul, two=0x4000000000000000ul;
    ulong ax=r.axis_x[gid], ay=r.axis_y[gid], az=r.axis_z[gid];
    const ulong inv_r=soft64_div(one,r.radius[0]);
    ulong qx=soft64_sub(ax,soft64_mul(soft64_mul(r.dt[0],r.wind_vx[gid]),inv_r));
    ulong qy=soft64_sub(ay,soft64_mul(soft64_mul(r.dt[0],r.wind_vy[gid]),inv_r));
    ulong qz=soft64_sub(az,soft64_mul(soft64_mul(r.dt[0],r.wind_vz[gid]),inv_r));
    ulong norm=soft64_sqrt(soft64_add(soft64_add(soft64_mul(qx,qx),soft64_mul(qy,qy)),soft64_mul(qz,qz)));
    qx=soft64_div(qx,norm); qy=soft64_div(qy,norm); qz=soft64_div(qz,norm);
    ulong qk=soft64_sub(soft64_from_i32(k),soft64_mul(r.dt[0],r.wind_level[gid]));
    const ulong top=soft64_from_i32(p.nlevel-1);
    if(soft64_less(qk,0ul))qk=0ul; if(soft64_greater(qk,top))qk=top;
    const ulong hdt=soft64_mul(0x3fe0000000000000ul,r.dt[0]);
    ulong move_m=0ul, move_k=0ul;
    for(int it=0;it<p.iterations;++it) {
        ulong lam=sld_atan2(qy,qx);
        ulong phi=sld_atan2(qz,soft64_sqrt(soft64_add(soft64_mul(qx,qx),soft64_mul(qy,qy))));
        ulong w[4];
        sld_trilinear(r.wind_vx,r.wind_vy,r.wind_vz,r.wind_level,
            lam,phi,qk,r.lat_ext,r.mlut,r.rowoff,r.rowshift,
            p.lookup_bins,p.nlevel,p.nlat,p.nlon,r.inv_dlam[0],r.lut_scale[0],w);
        ulong px=soft64_sub(ax,soft64_mul(soft64_mul(hdt,soft64_add(r.extrapolated_vx[gid],w[0])),inv_r));
        ulong py=soft64_sub(ay,soft64_mul(soft64_mul(hdt,soft64_add(r.extrapolated_vy[gid],w[1])),inv_r));
        ulong pz=soft64_sub(az,soft64_mul(soft64_mul(hdt,soft64_add(r.extrapolated_vz[gid],w[2])),inv_r));
        norm=soft64_sqrt(soft64_add(soft64_add(soft64_mul(px,px),soft64_mul(py,py)),soft64_mul(pz,pz)));
        px=soft64_div(px,norm); py=soft64_div(py,norm); pz=soft64_div(pz,norm);
        ulong pk=soft64_sub(soft64_from_i32(k),soft64_mul(hdt,soft64_add(r.extrapolated_level[gid],w[3])));
        if(soft64_less(pk,0ul))pk=0ul; if(soft64_greater(pk,top))pk=top;
        ulong dx=soft64_sub(px,qx),dy=soft64_sub(py,qy),dz=soft64_sub(pz,qz);
        ulong chord=soft64_sqrt(soft64_add(soft64_add(soft64_mul(dx,dx),soft64_mul(dy,dy)),soft64_mul(dz,dz)));
        ulong half_chord=soft64_mul(0x3fe0000000000000ul,chord);
        if(soft64_greater(half_chord,one))half_chord=one;
        ulong angle=sld_atan2(half_chord,soft64_sqrt(soft64_sub(one,soft64_mul(half_chord,half_chord))));
        move_m=soft64_mul(soft64_mul(two,r.radius[0]),angle);
        move_k=soft64_abs_bits(soft64_sub(pk,qk));
        qx=px;qy=py;qz=pz;qk=pk;
    }
    r.xi[gid]=soft64_mul(sld_atan2(qy,qx),r.inv_dlam[0]);
    r.phi[gid]=sld_atan2(qz,soft64_sqrt(soft64_add(soft64_mul(qx,qx),soft64_mul(qy,qy))));
    r.departure_level[gid]=qk;

    // Keep the trajectory diagnostics beside their source fields.  The two
    // output resources are packed into contiguous planes so the stencil and
    // all six scalar readings can remain resident through the following
    // gather.  Their signatures remain flat float64 buffers; the caller
    // exposes each plane as a MetalArray view.
    const uint row=(gid/uint(p.nlon))%uint(p.nlat);
    const ulong meridional_cell=soft64_mul(r.radius[0],
        soft64_mul(0x3fe0000000000000ul,
            soft64_sub(r.lat_ext[row+3u],r.lat_ext[row+1u])));
    const ulong arrival_dx=soft64_sub(ax,qx);
    const ulong arrival_dy=soft64_sub(ay,qy);
    const ulong arrival_dz=soft64_sub(az,qz);
    ulong arrival_chord=soft64_sqrt(soft64_add(
        soft64_add(soft64_mul(arrival_dx,arrival_dx),
                   soft64_mul(arrival_dy,arrival_dy)),
        soft64_mul(arrival_dz,arrival_dz)));
    ulong arrival_half=soft64_mul(0x3fe0000000000000ul,arrival_chord);
    if(soft64_greater(arrival_half,one))arrival_half=one;
    const ulong displacement=soft64_mul(soft64_mul(two,r.radius[0]),
        sld_atan2(arrival_half,
            soft64_sqrt(soft64_sub(one,soft64_mul(arrival_half,arrival_half)))));
    const ulong count=uint(p.nlevel)*uint(p.nlat)*uint(p.nlon);
    r.move_m[gid]=move_m;
    r.move_m[gid+count]=soft64_div(move_m,meridional_cell);
    r.move_m[gid+2u*count]=displacement;
    r.move_m[gid+3u*count]=soft64_div(displacement,meridional_cell);
    r.move_k[gid]=move_k;
    r.move_k[gid+count]=soft64_abs_bits(
        soft64_sub(qk,soft64_from_i32(k)));
}
