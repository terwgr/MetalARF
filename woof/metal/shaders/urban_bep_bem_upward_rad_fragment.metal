#include <metal_stdlib>
#include "urban_bep_bem_upward_rad_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubr_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubr_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubr_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubr_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubr_pow4(float x) { float x2 = ubr_mul(x, x); return ubr_mul(x2, x2); }

// WRF module_sf_bep_bem.F bb_upward_rad (5288-5352). This BEM caller packs
// canyon columns as [ncol,2], wall faces as [ncol,4,18], and roof fields as
// [ncol,18,2]; UTYPE selects class parameters and each profile's nzu/nd.
kernel void urban_bep_bem_upward_rad_fragment(
    constant UrbanBepBemUpwardRadFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemUpwardRadFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item >= uint(p.nitem)) return;
    device const float *ws = r.ws;
    device const float *bs = r.bs;
    device const float *pb = r.pb;
    device const float *ss = r.ss;
    device const float *tg_av = r.tg_av;
    device const float *rlg = r.rlg;
    device const float *rsg = r.rsg;
    device const float *sfg = r.sfg;
    device const float *lfg = r.lfg;
    device const float *tw = r.tw;
    device const float *rlw = r.rlw;
    device const float *rsw = r.rsw;
    device const float *sfw = r.sfw;
    device const float *tr_av = r.tr_av;
    device const float *twlev = r.twlev;
    device const float *sfwind = r.sfwind;
    device const float *tpvlev = r.tpvlev;
    device const float *sfr = r.sfr;
    device const float *sfrv = r.sfrv;
    device const float *lfr = r.lfr;
    device const float *lfrv = r.lfrv;
    device const int *utype = r.utype;
    device const int *class_int = r.class_int;
    device const float *class_float = r.class_float;
    device const int *nzu_class = r.nzu_class;
    device const float *rld = r.rld;
    device const float *rs = r.rs;
    device const int *active_cols = r.active_cols;
    device float *rs_abs_out = r.rs_abs;
    device float *rl_up_out = r.rl_up;
    device float *emiss_out = r.emiss;
    device float *grdflx_out = r.grdflx_urb;
    device int *error = r.error;

    if (active_cols[item] == 0) {
        rs_abs_out[item] = 0.0f; rl_up_out[item] = 0.0f;
        emiss_out[item] = 0.0f; grdflx_out[item] = 0.0f;
        error[item] = 0;
        return;
    }

    const int type = utype[item];
    if (p.nurbm < 1 || p.nurbm > 11 || type < 1 || type > p.nurbm) {
        error[item] = 7101; return;
    }
    const int ndu = class_int[uint(type)];
    const int nzu = nzu_class[uint(type - 1)];
    if (ndu < 1 || ndu > 2 || nzu < 1 || nzu >= 18) {
        error[item] = 7102; return;
    }
    const ulong c2 = ulong(item) * 2ul;
    const ulong c4 = ulong(item) * 72ul;
    const ulong c2z = ulong(item) * 36ul;
    const ulong pbase = ulong(item) * 19ul;
    const uint cls = uint(type - 1);
    const float emg_u = class_float[98u + cls], albg_u = class_float[65u + cls];
    const float emw_u = class_float[109u + cls], albw_u = class_float[76u + cls];
    const float emr_u = class_float[120u + cls], albr_u = class_float[87u + cls];
    const float emwind_u = class_float[1062u + cls], albwind_u = class_float[1073u + cls];
    const float pwin_u = class_float[1084u + cls];
    const float grf = class_float[1205u + cls], pvf = class_float[1216u + cls];

    for (int iz = 0; iz <= nzu; ++iz) {
        for (int id = 0; id < ndu; ++id) {
            if (tr_av[c2z + ulong(iz) * 2ul + uint(id)] < 100.0f) {
                error[item] = 10;
                return;
            }
        }
    }

    float rl_emit = 0.0f, rl_inc = 0.0f, rs_abs = 0.0f, grdflx_urb = 0.0f;
    for (int id = 0; id < ndu; ++id) {
        const ulong c = c2 + uint(id);
        const float wb = ubr_add(ws[c], bs[c]);
        const float tg4 = ubr_pow4(tg_av[c]);
        rl_emit = ubr_sub(rl_emit, ubr_div(ubr_div(ubr_mul(ubr_add(
            ubr_mul(ubr_mul(emg_u, p.sigma), tg4), ubr_mul(ubr_sub(1.0f, emg_u), rlg[c])), ws[c]), wb), float(ndu)));
        rl_inc = ubr_add(rl_inc, ubr_div(ubr_div(ubr_mul(rlg[c], ws[c]), wb), float(ndu)));
        rs_abs = ubr_add(rs_abs, ubr_div(ubr_div(ubr_mul(ubr_mul(ubr_sub(1.0f, albg_u), rsg[c]), ws[c]), wb), float(ndu)));
        const float gfl = ubr_add(ubr_add(ubr_sub(ubr_add(
            ubr_mul(ubr_sub(1.0f, albg_u), rsg[c]), ubr_mul(emg_u, rlg[c])),
            ubr_mul(ubr_mul(emg_u, p.sigma), tg4)), sfg[c]), lfg[c]);
        grdflx_urb = ubr_sub(grdflx_urb, ubr_div(ubr_div(ubr_mul(gfl, ws[c]), wb), float(ndu)));

        for (int iz = 1; iz < nzu; ++iz) {
            const ulong zc = c2z + ulong(iz) * 2ul + uint(id);
            const float tr4 = ubr_pow4(tr_av[zc]);
            const float panel4 = ubr_pow4(tpvlev[zc]);
            const float rl_surface = ubr_add(ubr_add(
                ubr_mul(ubr_mul(ubr_mul(emr_u, p.sigma), ubr_sub(1.0f, pvf)), tr4),
                ubr_mul(ubr_mul(ubr_mul(0.79f, p.sigma), pvf), panel4)),
                ubr_add(ubr_mul(ubr_mul(ubr_sub(1.0f, emr_u), rld[item]), ubr_sub(1.0f, pvf)),
                    ubr_mul(ubr_mul(ubr_sub(1.0f, 0.79f), pvf), rld[item])));
            rl_emit = ubr_sub(rl_emit, ubr_div(ubr_div(ubr_mul(rl_surface,
                ss[pbase + uint(iz)]), bs[c]), ubr_mul(wb, float(ndu))));
            rl_inc = ubr_add(rl_inc, ubr_div(ubr_div(ubr_mul(ubr_mul(rld[item], ss[pbase + uint(iz)]), bs[c]), wb), float(ndu)));
            const float roof_sw = ubr_add(ubr_mul(ubr_mul(ubr_sub(1.0f, albr_u), rs[item]), ubr_sub(1.0f, pvf)),
                ubr_mul(ubr_mul(ubr_sub(1.0f, 0.11f), rs[item]), pvf));
            rs_abs = ubr_add(rs_abs, ubr_div(ubr_div(ubr_mul(ubr_mul(roof_sw,
                ss[pbase + uint(iz)]), bs[c]), wb), float(ndu)));
            const float gflr0 = ubr_sub(ubr_add(ubr_add(
                ubr_mul(ubr_mul(ubr_sub(1.0f, albr_u), rs[item]), ubr_sub(1.0f, pvf)),
                ubr_mul(ubr_mul(emr_u, rld[item]), ubr_sub(1.0f, pvf))),
                ubr_mul(ubr_mul(ubr_mul(pvf, emr_u), p.sigma), panel4)),
                ubr_mul(ubr_mul(emr_u, p.sigma), tr4));
            const float gflr1 = ubr_add(ubr_mul(ubr_sub(1.0f, grf), sfr[zc]),
                ubr_mul(ubr_add(sfrv[zc], lfrv[zc]), grf));
            const float gflr = ubr_add(gflr0, ubr_add(gflr1, ubr_mul(ubr_sub(1.0f, grf), lfr[zc])));
            grdflx_urb = ubr_sub(grdflx_urb, ubr_div(ubr_div(ubr_mul(ubr_mul(gflr,
                ss[pbase + uint(iz)]), bs[c]), wb), float(ndu)));
        }

        for (int iz = 0; iz < nzu; ++iz) {
            const ulong w0 = c4 + uint(2 * id) * 18ul + uint(iz);
            const ulong w1 = c4 + uint(2 * id + 1) * 18ul + uint(iz);
            const ulong z = pbase + uint(iz + 1);
            const float walls_rl = ubr_add(rlw[w0], rlw[w1]);
            const float walls_rs = ubr_add(rsw[w0], rsw[w1]);
            const float walls_t4 = ubr_add(ubr_pow4(tw[w0]), ubr_pow4(tw[w1]));
            const float wind_t4 = ubr_add(ubr_pow4(twlev[w0]), ubr_pow4(twlev[w1]));
            const float win_sf = ubr_add(sfw[w0], sfw[w1]);
            const float wind_sf = ubr_add(sfwind[w0], sfwind[w1]);
            const float pressure = pb[z];
            const float wall_lw_em = ubr_add(ubr_mul(ubr_mul(ubr_mul(emw_u, ubr_sub(1.0f, pwin_u)), p.sigma), walls_t4),
                ubr_mul(ubr_mul(ubr_mul(emwind_u, pwin_u), p.sigma), wind_t4));
            const float wall_lw_inc = ubr_mul(ubr_add(
                ubr_mul(ubr_sub(1.0f, emw_u), ubr_sub(1.0f, pwin_u)),
                ubr_mul(pwin_u, ubr_sub(1.0f, emwind_u))), walls_rl);
            rl_emit = ubr_sub(rl_emit, ubr_div(ubr_div(ubr_mul(ubr_add(wall_lw_em, wall_lw_inc),
                p.dz_u), pressure), ubr_mul(wb, float(ndu))));
            rl_inc = ubr_add(rl_inc, ubr_div(ubr_div(ubr_mul(ubr_mul(walls_rl, p.dz_u), pressure), wb), float(ndu)));
            const float wall_abs = ubr_mul(ubr_add(ubr_mul(ubr_sub(1.0f, albw_u), ubr_sub(1.0f, pwin_u)),
                ubr_mul(ubr_sub(1.0f, albwind_u), pwin_u)), walls_rs);
            rs_abs = ubr_add(rs_abs, ubr_div(ubr_div(ubr_mul(ubr_mul(wall_abs, p.dz_u), pressure), wb), float(ndu)));
            const float gflw = ubr_add(ubr_sub(ubr_add(ubr_mul(ubr_sub(1.0f, albw_u), walls_rs),
                ubr_mul(emw_u, walls_rl)), ubr_mul(ubr_mul(emw_u, p.sigma), walls_t4)), win_sf);
            const float gflwin = ubr_add(ubr_sub(ubr_add(ubr_mul(ubr_sub(1.0f, albwind_u), walls_rs),
                ubr_mul(emwind_u, walls_rl)), ubr_mul(ubr_mul(emwind_u, p.sigma), wind_t4)), wind_sf);
            const float gflmix = ubr_add(ubr_mul(gflw, ubr_sub(1.0f, pwin_u)), ubr_mul(pwin_u, gflwin));
            grdflx_urb = ubr_sub(grdflx_urb, ubr_div(ubr_div(ubr_mul(ubr_mul(gflmix, p.dz_u), pressure), wb), float(ndu)));
        }
    }
    emiss_out[item] = ubr_div(ubr_add(ubr_add(emg_u, emw_u), emr_u), 3.0f);
    rl_up_out[item] = ubr_sub(ubr_add(rl_inc, rl_emit), rld[item]);
    rs_abs_out[item] = rs_abs;
    grdflx_out[item] = grdflx_urb;
    error[item] = 0;
}
