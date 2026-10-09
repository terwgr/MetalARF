#include <metal_stdlib>
#include "urban_bep_bem_flat_roof_flux_fragment_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubr_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubr_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubr_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubr_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubr_pow(float x, float y) { return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x), as_type<uint>(y))); }
static inline float ubr_log(float x) { return as_type<float>(soft_gfk_log(as_type<uint>(x))); }
static inline float ubr_sqrt(float x) { return as_type<float>(soft32_sqrt(as_type<uint>(x))); }
static inline float ubr_min(float a, float b) { return a < b ? a : b; }
static inline float ubr_max(float a, float b) { return a > b ? a : b; }

// WRF module_sf_bep_bem.F bb_flux_flat_roof (4499-4588). This is a direct
// per-roof translation; every arithmetic operation uses the project's
// explicit binary32 helpers and invalid vegetation types are returned through
// the per-item error buffer.
kernel void urban_bep_bem_flat_roof_flux_fragment(
    constant UrbanBepBemFlatRoofFluxFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemFlatRoofFluxFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.nitem)) return;
    device const float *qr = r.qr;
    device float *uhb = r.uhb, *vhb = r.vhb, *sf = r.sf, *lf = r.lf, *ehb = r.ehb;
    device float *resg = r.resg, *rsveg = r.rsveg, *f1 = r.f1, *f2 = r.f2, *f3 = r.f3, *f4 = r.f4;
    device int *error = r.error;
    const float dz = r.dz[item], z0 = r.z0[item], ua = r.ua[item], va = r.va[item];
    const float pt = r.pt[item], pt0 = r.pt0[item], ptg = r.ptg[item];
    const float da = r.da[item], qv = r.qv[item], pr = r.pr[item], rsg = r.rsg[item];
    const float pvf = r.pv_frac_roof[item];
    const int gr_type = r.gr_type[item];
    float rsmin, rsmax, lai;
    if (gr_type == 1) { rsmin = 40.0f; rsmax = 5000.0f; lai = 2.0f; }
    else if (gr_type == 2) { rsmin = 150.0f; rsmax = 5000.0f; lai = 3.0f; }
    else { error[item] = 22; return; }

    const float b = 9.4f, cm = 7.4f, ch = 5.3f, rr = 0.74f;
    const float sdlim = 100.0f, qw = 0.06f, qref = 0.37f, hs = 36.35f;
    const float z0t = ubr_div(z0, 10.0f);
    const float rs_eff = ubr_mul(ubr_sub(1.0f, pvf), rsg);
    float utot = ubr_sqrt(ubr_add(ubr_mul(ua, ua), ubr_mul(va, va)));
    const float zz = ubr_div(dz, 2.0f);
    utot = ubr_max(utot, 0.01f);
    const float ric = ubr_div(ubr_mul(ubr_mul(ubr_mul(2.0f, p.g_u), zz), ubr_sub(pt, ptg)),
        ubr_mul(ubr_add(pt, ptg), ubr_mul(utot, utot)));
    const float aa = ubr_div(p.vk, ubr_log(ubr_div(zz, z0)));
    const float ah = ubr_div(p.vk, ubr_log(ubr_div(zz, z0t)));
    float fm, fh, c;
    if (ric > 0.0f) {
        fm = ubr_div(1.0f, ubr_pow(ubr_add(1.0f, ubr_mul(ubr_mul(0.5f, b), ric)), 2.0f));
        fh = fm;
    } else {
        c = ubr_mul(ubr_mul(ubr_mul(ubr_mul(b, cm), aa), aa), ubr_pow(ubr_div(zz, z0), 0.5f));
        fm = ubr_sub(1.0f, ubr_div(ubr_mul(b, ric), ubr_add(1.0f, ubr_mul(c, ubr_pow(-ric, 0.5f)))));
        c = ubr_mul(ubr_mul(ubr_mul(ubr_mul(b, cm), aa), ah), ubr_pow(ubr_div(zz, z0t), 0.5f));
        c = ubr_div(ubr_mul(c, ch), cm);
        fh = ubr_sub(1.0f, ubr_div(ubr_mul(b, ric), ubr_add(1.0f, ubr_mul(c, ubr_pow(-ric, 0.5f)))));
    }
    const float fbuw = ubr_mul(ubr_mul(ubr_mul(ubr_mul(-aa, aa), utot), utot), fm);
    const float fbpt = ubr_div(ubr_mul(ubr_mul(ubr_mul(ubr_mul(-aa, ah), utot), ubr_sub(pt, ptg)), fh), rr);
    const float tmpg = ubr_sub(ubr_mul(ptg, ubr_pow(ubr_div(pr, p.p0), p.rcp_u)), 273.15f);
    const float es = ubr_mul(6.11f, ubr_pow(10.0f, ubr_div(ubr_mul(tmpg, 7.5f), ubr_add(237.7f, tmpg))));
    const float qvsg = ubr_div(ubr_mul(0.62197f, es), ubr_sub(ubr_mul(0.01f, pr), ubr_mul(0.378f, es)));
    const float f = ubr_div(ubr_mul(ubr_div(ubr_mul(0.55f, rs_eff), sdlim), 2.0f), lai);
    const float f1v = ubr_div(ubr_add(f, ubr_div(rsmin, rsmax)), ubr_add(1.0f, f));
    const float ta = ubr_sub(ubr_mul(pt, ubr_pow(ubr_div(pr, p.p0), p.rcp_u)), 273.15f);
    const float esa = ubr_mul(6.11f, ubr_pow(10.0f, ubr_div(ubr_mul(ta, 7.5f), ubr_add(237.7f, ta))));
    const float qvs = ubr_div(ubr_mul(0.62197f, esa), ubr_sub(ubr_mul(0.01f, pr), ubr_mul(0.378f, esa)));
    const float f2v = ubr_div(1.0f, ubr_add(1.0f, ubr_mul(hs, ubr_sub(qvs, qv))));
    const float f3v = ubr_sub(1.0f, ubr_mul(0.0016f, ubr_pow(ubr_sub(25.0f, ta), 2.0f)));
    constexpr float dzgr[10] = {0.1f,0.003f,0.06f,0.003f,0.05f,0.04f,0.02f,0.0125f,0.005f,0.0025f};
    float f4v = 0.0f, dzgr_tot = 0.0f;
    for (int iz = 4; iz < 10; ++iz) {
        float gx = ubr_div(ubr_sub(qr[int(item) * 10 + iz], qw), ubr_sub(qref, qw));
        gx = ubr_min(1.0f, ubr_max(0.0f, gx));
        f4v = ubr_add(f4v, ubr_mul(gx, dzgr[iz]));
        dzgr_tot = ubr_add(dzgr_tot, dzgr[iz]);
    }
    f4v = ubr_div(f4v, dzgr_tot);
    const float rsveg_v = ubr_min(ubr_div(rsmin, ubr_max(ubr_mul(ubr_mul(ubr_mul(ubr_mul(lai, f1v), f2v), f3v), f4v), 1.0e-9f)), rsmax);
    const float resg_v = ubr_div(rr, ubr_mul(ubr_mul(ubr_mul(aa, aa), utot), fh));
    const float fbqq = ubr_div(-ubr_sub(qv, qvsg), ubr_add(resg_v, rsveg_v));
    const float ustar = ubr_sqrt(-fbuw);
    const float tstar = ubr_div(-fbpt, ustar);
    const float qstar = ubr_div(-fbqq, ustar);
    const float buu = ubr_mul(ubr_mul(ubr_div(-p.g_u, pt0), ustar), tstar);
    uhb[item] = ubr_div(ubr_mul(ubr_mul(-ustar, ustar), ua), utot);
    vhb[item] = ubr_div(ubr_mul(ubr_mul(-ustar, ustar), va), utot);
    sf[item] = ubr_mul(ubr_mul(ubr_mul(ustar, tstar), da), p.cp_u);
    lf[item] = ubr_mul(ubr_mul(ubr_mul(ustar, qstar), da), p.latent);
    ehb[item] = buu;
    resg[item] = resg_v; rsveg[item] = rsveg_v;
    f1[item] = f1v; f2[item] = f2v; f3[item] = f3v; f4[item] = f4v;
    error[item] = 0;
}
