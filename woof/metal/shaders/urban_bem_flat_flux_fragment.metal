#include <metal_stdlib>
#include "urban_bem_flat_flux_fragment_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubf_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubf_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubf_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubf_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubf_pow(float x, float y) { return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x), as_type<uint>(y))); }
static inline float ubf_log(float x) { return as_type<float>(soft_gfk_log(as_type<uint>(x))); }

// WRF bb_flux_flat (urban_bep_bem.cu:3516-3639). One item is one
// column/direction/urban level. Level zero is the ground solve; positive
// levels are roof solves and only active where the selected ss profile is
// positive, matching bb_buildings' Fortran index range 2:nz.
kernel void urban_bem_flat_flux_fragment(
    constant UrbanBemFlatFluxFragmentResources &r [[buffer(0)]],
    constant UrbanBemFlatFluxFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.nitem)) return;
    device const int *active = r.active_cols;
    device const float *interp = r.interp;
    device const float *z0 = r.z0;
    device const float *ss = r.ss;
    device const float *ptg = r.ptg;
    device const float *tr = r.tr;
    device const float *dg = r.dg;
    device const float *dgr = r.dgr;
    device float *uhb = r.uhb, *vhb = r.vhb, *sf = r.sf, *lf = r.lf, *ehb = r.ehb;
    device float *thb = r.thb, *qhb = r.qhb;
    device int *error = r.error;
    const uint level = item % 18u;
    const uint direction = (item / 18u) % 2u;
    const uint col = item / 36u;
    error[item] = 0;
    uhb[item] = 0.0f; vhb[item] = 0.0f; sf[item] = 0.0f; lf[item] = 0.0f; ehb[item] = 0.0f;
    thb[item] = 0.0f; qhb[item] = 0.0f;
    if (col >= uint(p.ncol) || active[col] == 0) return;
    if (level > 0u && !(ss[ulong(col) * 19ul + level] > 0.0f)) return;
    const ulong atmos = ulong(col) * 7ul * 18ul;
    const float ua = interp[atmos + level];
    const float va = interp[atmos + 18ul + level];
    const float pt = interp[atmos + 36ul + level];
    const float pt0 = interp[atmos + 54ul + level];
    const float pr = interp[atmos + 72ul + level];
    const float da = interp[atmos + 90ul + level];
    const float qv = interp[atmos + 108ul + level];
    const ulong surface = (ulong(col) * 2ul + direction) * 18ul + level;
    const float z0v = z0[surface];
    const float ptgv = level == 0u
        ? ptg[ulong(col) * 2ul + direction]
        : ubf_mul(tr[((ulong(col) * 10ul + 9ul) * 18ul + level) * 2ul + direction],
                  ubf_pow(ubf_div(pr, p.p0), -p.rcp_u));
    const float log_argument = ubf_div(ubf_div(p.dz_u, 2.0f), z0v);
    if (!(z0v > 0.0f) || !(da > 0.0f) || !(pr > 0.0f)
            || !(log_argument > 0.0f) || log_argument == 1.0f) {
        error[item] = 7420; return;
    }
    const float speed0 = ubf_pow(ubf_add(ubf_mul(ua, ua), ubf_mul(va, va)), 0.5f);
    const float speed = max(speed0, 0.01f);
    const float zz = ubf_div(p.dz_u, 2.0f);
    const float ric = ubf_div(ubf_mul(ubf_mul(ubf_mul(2.0f, p.g_u), zz), ubf_sub(pt, ptgv)),
                              ubf_mul(ubf_add(pt, ptgv), ubf_mul(speed, speed)));
    const float aa = ubf_div(p.vk, ubf_log(ubf_div(zz, z0v)));
    const float tmpg = ubf_sub(ubf_mul(ptgv, ubf_pow(ubf_div(pr, p.p0), p.rcp_u)), 273.15f);
    const float es = ubf_mul(6.11f, ubf_pow(10.0f,
        ubf_div(ubf_mul(tmpg, 7.5f), ubf_add(237.7f, tmpg))));
    const float qvsg = ubf_div(ubf_mul(0.62197f, es),
        ubf_sub(ubf_mul(0.01f, pr), ubf_mul(0.378f, es)));
    float fm, fh;
    if (ric > 0.0f) {
        fm = ubf_div(1.0f, ubf_pow(ubf_add(1.0f, ubf_mul(ubf_mul(0.5f, 9.4f), ric)), 2.0f));
        fh = fm;
    } else {
        float c = ubf_mul(ubf_mul(ubf_mul(ubf_mul(9.4f, 7.4f), aa), aa),
                          ubf_pow(ubf_div(zz, z0v), 0.5f));
        fm = ubf_sub(1.0f, ubf_div(ubf_mul(9.4f, ric),
            ubf_add(1.0f, ubf_mul(c, ubf_pow(-ric, 0.5f)))));
        c = ubf_div(ubf_mul(c, 5.3f), 7.4f);
        fh = ubf_sub(1.0f, ubf_div(ubf_mul(9.4f, ric),
            ubf_add(1.0f, ubf_mul(c, ubf_pow(-ric, 0.5f)))));
    }
    const float resg = ubf_div(0.74f, ubf_mul(ubf_mul(ubf_mul(aa, aa), speed), fh));
    const float fbuw = ubf_mul(ubf_mul(ubf_mul(ubf_mul(-aa, aa), speed), speed), fm);
    const float fbpt = ubf_div(ubf_mul(ubf_mul(ubf_mul(ubf_mul(-aa, aa), speed), ubf_sub(pt, ptgv)), fh), 0.74f);
    const float fbqq = ubf_div(-ubf_sub(qv, qvsg), resg);
    const float ustar = ubf_pow(-fbuw, 0.5f);
    const float tstar = ubf_div(-fbpt, ustar);
    const float qstar = ubf_div(-fbqq, ustar);
    const float buu = ubf_mul(ubf_mul(ubf_div(-p.g_u, pt0), ustar), tstar);
    uhb[item] = ubf_div(ubf_mul(ubf_mul(-ustar, ustar), ua), speed);
    vhb[item] = ubf_div(ubf_mul(ubf_mul(-ustar, ustar), va), speed);
    sf[item] = ubf_mul(ubf_mul(ubf_mul(ustar, tstar), da), p.cp_u);
    lf[item] = ubf_mul(ubf_mul(ubf_mul(ustar, qstar), da), p.latent);
    ehb[item] = buu;
    const float drainage = level == 0u
        ? dg[ulong(col) * 2ul + direction]
        : dgr[(ulong(col) * 18ul + level) * 2ul + direction];
    if (drainage > 0.0f) {
        const float w = ubf_div(drainage, level == 0u ? p.dgmax : p.drmax);
        const float negative_density_latent = ubf_mul(-da, p.latent);
        const float negative_weight_flux = ubf_mul(-w, lf[item]);
        lf[item] = ubf_mul(negative_density_latent,
            ubf_div(negative_weight_flux, ubf_mul(da, p.latent)));
    } else {
        lf[item] = 0.0f;
    }
    thb[item] = ubf_div(-sf[item], ubf_mul(da, p.cp_u));
    qhb[item] = ubf_div(-lf[item], ubf_mul(da, p.latent));
}
