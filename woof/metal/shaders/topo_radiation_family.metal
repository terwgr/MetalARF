#include <metal_stdlib>
#include "topo_slope_geometry_bindings.metalh"
#include "topo_shadow_init_bindings.metalh"
#include "topo_shadow_scan_bindings.metalh"
#include "topo_diffuse_frac_bindings.metalh"
#include "topo_rad_adjust_bindings.metalh"
#include "topo_rad_restore_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

static inline ulong tr_m(int i, int j, int nx) {
    return ulong(j) * ulong(nx) + ulong(i);
}
static inline ulong tr_w(int i, int j, int nx) {
    return ulong(j + 1) * ulong(nx + 4) + ulong(i + 1);
}
static inline float tr_pi() { return 4.0f * atan(1.0f); }
static inline float tr_degrad() { return as_type<float>(0x3C8EFA35u); }

kernel void topo_slope_geometry(
    constant TopoSlopeGeometryResources &r [[buffer(0)]],
    constant TopoSlopeGeometryParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint count = uint(p.nx * p.ny);
    if (flat >= count) return;
    const int i = int(flat) % p.nx, j = int(flat) / p.nx;
    const int im1 = max(i - 1, 0), ip1 = min(i + 1, p.nx - 1);
    const int jm1 = max(j - 1, 0), jp1 = min(j + 1, p.ny - 1);
    float hx = (r.ht[tr_m(ip1,j,p.nx)] - r.ht[tr_m(im1,j,p.nx)]) * r.msftx[flat];
    hx = hx * p.rdx / float(ip1 - im1);
    float hy = (r.ht[tr_m(i,jp1,p.nx)] - r.ht[tr_m(i,jm1,p.nx)]) * r.msfty[flat];
    hy = hy * p.rdy / float(jp1 - jm1);
    const float pi = tr_pi();
    const float s = atan(pow(hx * hx + hy * hy, 0.5f));
    float azi, slp = s;
    if (s < 1.0e-4f) { slp = 0.0f; azi = 0.0f; }
    else {
        azi = atan2(hx, hy) + pi;
        const float asa = asin(r.sina[flat]);
        azi = r.cosa[flat] >= 0.0f ? azi - asa : azi - (pi - asa);
    }
    r.slope[flat] = slp;
    r.slp_azi[flat] = azi;
}

kernel void topo_shadow_init(
    constant TopoShadowInitResources &r [[buffer(0)]],
    constant TopoShadowInitParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint count = uint(p.nx * p.ny);
    if (flat >= count) return;
    const int i = int(flat) % p.nx + 1, j = int(flat) / p.nx + 1;
    int mask = 0;
    float shad = r.ht_shad[flat], loc = r.ht_loc[tr_w(i,j,p.nx)];
    if (!p.nested || (i >= 3 && i <= p.nx - 2 && j >= 3 && j <= p.ny - 2))
        shad = loc - 0.001f;
    if (p.nested) {
        const bool edge = i <= 2 || i >= p.nx - 1 || j <= 2 || j >= p.ny - 1;
        if (edge && shad > loc) { mask = 1; loc = shad; }
        r.ht_loc[tr_w(i,j,p.nx)] = loc;
    }
    r.shadowmask[flat] = mask;
    r.ht_shad[flat] = shad;
}

static inline void tr_scan(device const float *ht, int nx, int ny,
                           int i, int j, int step, bool along_j, float t,
                           float dx, float dy, float csza, int gpshad,
                           thread int &mask, thread float &shad) {
    const float h0 = ht[tr_w(i,j,nx)];
    for (int n = 1; n <= gpshad; ++n) {
        const int k = along_j ? j + step*n : i + step*n;
        float q; int c1; float dxabs;
        if (along_j) {
            q = float(i) + float(k-j) * t; c1 = int(q);
            const float a = dy * float(k-j), b = dx * (q-float(i));
            dxabs = sqrt(a*a + b*b);
        } else {
            q = float(j) - float(k-i) * t; c1 = int(q);
            const float a = dx * float(k-i), b = dy * (q-float(j));
            dxabs = sqrt(a*a + b*b);
        }
        const int c2 = c1 + 1; const float wgt = q - float(c1);
        const bool stop = along_j
            ? ((step > 0 ? k >= ny+3 : k <= -2) || c1 <= -2 || c2 >= nx+3)
            : ((step > 0 ? k >= nx+3 : k <= -2) || c1 <= -2 || c2 >= ny+3);
        if (stop) return;
        const float h1 = along_j ? ht[tr_w(c1,k,nx)] : ht[tr_w(k,c1,nx)];
        const float h2 = along_j ? ht[tr_w(c2,k,nx)] : ht[tr_w(k,c2,nx)];
        const float topoelev = atan(((wgt*h2 + (1.0f-wgt)*h1-h0)/dxabs));
        if (sin(topoelev) >= csza) {
            mask = 1;
            shad = max(shad, h0 + dxabs * (tan(topoelev)-tan(asin(csza))));
        }
    }
}

kernel void topo_shadow_scan(
    constant TopoShadowScanResources &r [[buffer(0)]],
    constant TopoShadowScanParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint count = uint(p.nx*p.ny);
    if (flat >= count) return;
    const int i = int(flat)%p.nx+1, j = int(flat)/p.nx+1;
    const float degrad = tr_degrad(), pi = tr_pi();
    const float xt24 = fmod(p.xtime + p.radfrq*0.5f, 1440.0f);
    const int gpshad = int(p.shadlen/p.dx + 1.0f);
    int mask = r.shadowmask[flat]; float shad = r.ht_shad[flat];
    const float tloctm = p.gmt + xt24/60.0f + r.xlong[flat]/15.0f;
    const float hrang = 15.0f*(tloctm-12.0f)*degrad;
    const float xxlat = r.xlat[flat]*degrad;
    const float csza = sin(xxlat)*sin(p.declin) + cos(xxlat)*cos(p.declin)*cos(hrang);
    if (csza < 1.0e-2f) {
        r.shadowmask[flat] = 0; r.ht_shad[flat] = r.ht_loc[tr_w(i,j,p.nx)]-0.001f; return;
    }
    float argu = (csza*sin(xxlat)-sin(p.declin))/(sin(acos(csza))*cos(xxlat));
    argu = clamp(argu, -1.0f, 1.0f);
    float sol_azi = copysign(fabs(acos(argu)), sin(hrang)) + pi;
    if (r.cosa[flat] >= 0.0f) sol_azi += asin(r.sina[flat]);
    else sol_azi = sol_azi + pi - asin(r.sina[flat]);
    const float pq = 0.25f*pi;
    if (sol_azi > 1.75f*pi || sol_azi < pq)
        tr_scan(r.ht_loc,p.nx,p.ny,i,j,+1,true,tan(sol_azi),p.dx,p.dy,csza,gpshad,mask,shad);
    else if (sol_azi < 0.75f*pi)
        tr_scan(r.ht_loc,p.nx,p.ny,i,j,+1,false,tan(pi/2.0f+sol_azi),p.dx,p.dy,csza,gpshad,mask,shad);
    else if (sol_azi < 1.25f*pi)
        tr_scan(r.ht_loc,p.nx,p.ny,i,j,-1,true,tan(sol_azi),p.dx,p.dy,csza,gpshad,mask,shad);
    else
        tr_scan(r.ht_loc,p.nx,p.ny,i,j,-1,false,tan(pi/2.0f+sol_azi),p.dx,p.dy,csza,gpshad,mask,shad);
    r.shadowmask[flat] = mask; r.ht_shad[flat] = shad;
}

kernel void topo_diffuse_frac(
    constant TopoDiffuseFracResources &r [[buffer(0)]],
    constant TopoDiffuseFracParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const float mu=r.coszen[idx], down=r.swdown[idx]; float dif=r.swddif[idx];
    if (p.ruiz && mu > 1.0e-3f) {
        const float ioh=p.solcon*mu; float kt=down/max(ioh,1.0e-3f);
        const float airmass=exp(-r.ht[idx]/8434.5f)/(mu+0.50572f*pow(asin(mu)*57.295779513082323f+6.07995f,-1.6364f));
        kt=kt/(0.1f+1.031f*exp(-1.4f/(0.9f+9.4f/max(airmass,1.0e-3f))));
        const float kd=0.952f-1.041f*exp(-exp(2.300f-4.702f*kt));
        dif=kd*down; r.swddif[idx]=dif;
    }
    float frac=0.0f;
    if (down > 0.001f) frac=min(dif/down,1.0f);
    r.diffuse_frac[idx]=frac;
}

kernel void topo_rad_adjust(
    constant TopoRadAdjustResources &r [[buffer(0)]],
    constant TopoRadAdjustParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const float degrad=tr_pi()/180.0f, sw=r.swdown[idx];
    r.swnorm[idx]=sw; if (!(sw > 1.0e-3f)) return;
    float teradj=sw; const int shadow=r.shadowmask[idx];
    const float csza=r.coszen[idx], xxlat=r.xlat[idx]*degrad;
    if (!(csza <= 1.0e-4f)) {
        const float frac=r.diffuse_frac[idx], s=r.slope[idx]; float corr;
        if (s == 0.0f || frac == 1.0f || csza <= 1.0e-4f) {
            corr=1.0f; if (shadow == 1) corr=frac;
        } else {
            const float h=r.hrang[idx], azi=r.slp_azi[idx];
            const float sins=sin(s), coss=cos(s);
            float slp=((sin(xxlat)*cos(h)*(-cos(azi)*sins)-sin(h)*(sin(azi)*sins)+cos(xxlat)*cos(h)*coss)*cos(p.declin))
                    + ((cos(xxlat)*cos(azi)*sins+sin(xxlat)*coss)*sin(p.declin));
            if (slp <= 1.0e-4f) slp=0.0f;
            if (shadow == 1) slp=0.0f;
            corr=frac+((1.0f-frac)*slp)/csza;
        }
        teradj=1.0f*sw*corr;
    }
    r.gswsave[idx]=r.gsw[idx]; r.gsw[idx]=(r.gsw[idx]*teradj)/sw; r.swdown[idx]=teradj;
}

kernel void topo_rad_restore(
    constant TopoRadRestoreResources &r [[buffer(0)]],
    constant TopoRadRestoreParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    if (r.swnorm[idx] > 1.0e-3f) {
        const float save=r.swdown[idx]; r.swdown[idx]=r.swnorm[idx];
        r.swnorm[idx]=save; r.gsw[idx]=r.gswsave[idx];
    }
}
