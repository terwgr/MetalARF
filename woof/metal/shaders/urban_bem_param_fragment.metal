#include <metal_stdlib>
#include "urban_bem_param_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmp_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubmp_sub(float a, float b) {
    return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubmp_mul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}
static inline float ubmp_div(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}

constant uint BEMP_NZ = 18u;
constant uint BEMP_NG = 10u;
constant uint BEMP_NWR = 10u;
constant uint BEMP_NF = 10u;
constant uint BEMP_NGB = 10u;
constant uint BEMP_NDM = 2u;

// WRF module_sf_bep_bem.F bb_param (1964-2093), one thread per domain
// column. Arrays use [column, Fortran-local-index...] in row-major storage;
// class tables preserve their packed Fortran word offsets. This stage writes
// scratch geometry/profile data only and does not mutate prognostic state.
kernel void urban_bem_param_fragment(
    constant UrbanBemParamFragmentResources &r [[buffer(0)]],
    constant UrbanBemParamFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    device const int *active = r.active_cols;
    device const int *utype = r.utype;
    device const float *cf = r.class_float;
    device const int *ci = r.class_int;
    device const float *ss_class = r.ss_class;
    device const float *pb_class = r.pb_class;
    device const int *nzu_class = r.nzu_class;
    device const float *ss_urban = r.ss_urban;
    device const float *pb_urban = r.pb_urban;
    device const int *nzurb_column = r.nzurb_column;
    device const float *lp = r.lp_urb;
    device const float *lb = r.lb_urb;
    device const float *hgt = r.hgt_urb;
    device const float *frc = r.frc_urb;

    device int *error = r.error;
    device int *nzurban_out = r.nzurban;
    device float *ss = r.ss;
    device float *pb = r.pb;
    device float *ss_urb_out = r.ss_urb;
    device float *pb_urb_out = r.pb_urb;
    device float *csg = r.csg;
    device float *alag = r.alag;
    device float *csgb = r.csgb;
    device float *alagb = r.alagb;
    device float *csf = r.csf;
    device float *alaf = r.alaf;
    device float *csr = r.csr;
    device float *alar = r.alar;
    device float *csw = r.csw;
    device float *alaw = r.alaw;
    device float *z0 = r.z0;
    device float *strd = r.strd;
    device float *drst = r.drst;
    device float *ws = r.ws;
    device float *bs = r.bs;
    device float *ws_urb = r.ws_urb;
    device float *bs_urb = r.bs_urb;
    device float *dzw = r.dzw;
    device float *dzr = r.dzr;
    device float *dzf = r.dzf;
    device float *dzgb = r.dzgb;

    if (col >= uint(p.ncol)) return;
    error[col] = 0;
    if (active[col] == 0) return;
    if (p.ncol <= 0 || p.nurbm < 1 || p.nurbm > 11) {
        error[col] = 5014;
        return;
    }
    const int row = utype[col] - 1;
    if (row < 0 || row >= p.nurbm) { error[col] = 5015; return; }
    const int nzu = nzu_class[uint(row)];
    const int nzurb = nzurb_column[col];
    const int ndu = ci[1u + uint(row)];
    if (nzu < 0 || nzu > int(BEMP_NZ) ||
        nzurb < 0 || nzurb > int(BEMP_NZ) ||
        ndu < 1 || ndu > int(BEMP_NDM)) {
        error[col] = 5016;
        return;
    }

    const ulong cbase = ulong(row) * 19ul;
    const ulong ssbase = ulong(col) * 19ul;
    const ulong zbase = ulong(col) * BEMP_NDM * BEMP_NZ;
    nzurban_out[col] = 0;
    for (uint iz = 0; iz <= BEMP_NZ; ++iz) {
        ss[ssbase + iz] = 0.0f;
        pb[ssbase + iz] = 0.0f;
        ss_urb_out[ssbase + iz] = 0.0f;
        pb_urb_out[ssbase + iz] = 0.0f;
    }
    for (uint iz = 0; iz < BEMP_NZ; ++iz)
        for (uint id = 0; id < BEMP_NDM; ++id)
            z0[zbase + ulong(id) * BEMP_NZ + iz] = 0.0f;

    // bb_param tests all NZ_UM entries for any nonzero local profile.
    bool has_urban_profile = false;
    for (uint iz = 0; iz < BEMP_NZ; ++iz) {
        const float value = ss_urban[ssbase + iz];
        ss_urb_out[ssbase + iz] = value;
        pb_urb_out[ssbase + iz] = pb_urban[ssbase + iz];
        if (value != 0.0f) has_urban_profile = true;
    }
    const int active_nz = has_urban_profile ? nzurb : nzu;
    nzurban_out[col] = active_nz;
    for (int iz = 0; iz <= active_nz; ++iz) {
        if (has_urban_profile) {
            ss[ssbase + uint(iz)] = ss_urban[ssbase + uint(iz)];
            pb[ssbase + uint(iz)] = pb_urban[ssbase + uint(iz)];
        } else {
            const float class_ss = ss_class[cbase + uint(iz)];
            const float class_pb = pb_class[cbase + uint(iz)];
            ss[ssbase + uint(iz)] = class_ss;
            pb[ssbase + uint(iz)] = class_pb;
            ss_urb_out[ssbase + uint(iz)] = class_ss;
            pb_urb_out[ssbase + uint(iz)] = class_pb;
        }
    }

    const float csg_u = cf[33u + uint(row)];
    const float alag_u = cf[uint(row)];
    const float csr_u = cf[55u + uint(row)];
    const float alar_u = cf[22u + uint(row)];
    const float csw_u = cf[44u + uint(row)];
    const float alaw_u = cf[11u + uint(row)];
    for (uint i = 0; i < BEMP_NG; ++i) {
        csg[ulong(col) * BEMP_NG + i] = csg_u;
        alag[ulong(col) * BEMP_NG + i] = alag_u;
        csgb[ulong(col) * BEMP_NGB + i] = csg_u;
        alagb[ulong(col) * (BEMP_NGB + 1u) + i] = ubmp_mul(csg_u, alag_u);
    }
    alagb[ulong(col) * (BEMP_NGB + 1u) + BEMP_NGB] = ubmp_mul(csg_u, alag_u);
    for (uint i = 0; i < BEMP_NF; ++i) {
        csf[ulong(col) * BEMP_NF + i] = csw_u;
        alaf[ulong(col) * (BEMP_NF + 1u) + i] = ubmp_mul(csw_u, alaw_u);
    }
    alaf[ulong(col) * (BEMP_NF + 1u) + BEMP_NF] = ubmp_mul(csw_u, alaw_u);
    for (uint i = 0; i < BEMP_NWR; ++i) {
        csr[ulong(col) * BEMP_NWR + i] = csr_u;
        alar[ulong(col) * (BEMP_NWR + 1u) + i] = ubmp_mul(csr_u, alar_u);
        csw[ulong(col) * BEMP_NWR + i] = csw_u;
        alaw[ulong(col) * (BEMP_NWR + 1u) + i] = ubmp_mul(csw_u, alaw_u);
    }
    alar[ulong(col) * (BEMP_NWR + 1u) + BEMP_NWR] = ubmp_mul(csr_u, alar_u);
    alaw[ulong(col) * (BEMP_NWR + 1u) + BEMP_NWR] = ubmp_mul(csw_u, alaw_u);

    const float z0g_u = cf[132u + uint(row)];
    const float z0r_u = cf[143u + uint(row)];
    for (uint id = 0; id < uint(ndu); ++id) {
        z0[zbase + ulong(id) * BEMP_NZ] = z0g_u;
        for (int iz = 1; iz <= active_nz; ++iz)
            z0[zbase + ulong(id) * BEMP_NZ + uint(iz)] = z0r_u;
        strd[ulong(col) * BEMP_NDM + id] = cf[154u + ulong(row) * BEMP_NDM + id];
        drst[ulong(col) * BEMP_NDM + id] = cf[176u + ulong(row) * BEMP_NDM + id];
    }

    const float lpv = lp[col];
    const float lbv = lb[col];
    const float hgtv = hgt[col];
    const float frcv = frc[col];
    for (uint id = 0; id < BEMP_NDM; ++id) {
        const ulong out = ulong(col) * BEMP_NDM + id;
        if (id >= uint(ndu)) {
            bs[out] = 0.0f;
            ws[out] = 0.0f;
            bs_urb[out] = 0.0f;
            ws_urb[out] = 0.0f;
            strd[ulong(col) * BEMP_NDM + id] = 0.0f;
            drst[ulong(col) * BEMP_NDM + id] = 0.0f;
            continue;
        }
        // The packed Fortran table has direction as the fastest index.
        const ulong class_geom = ulong(row) * BEMP_NDM + id;
        const float base_ws = cf[198u + class_geom];
        const float base_bs = cf[220u + class_geom];
        float ws_value = base_ws;
        float bs_value = base_bs;
        if (!(hgtv <= 0.0f || lpv <= 0.0f || lbv <= 0.0f)) {
            if (ubmp_div(lpv, frcv) < 1.0f && lpv < lbv) {
                const float denominator = ubmp_sub(lbv, lpv);
                bs_value = ubmp_div(
                    ubmp_mul(ubmp_mul(2.0f, hgtv), lpv), denominator);
                ws_value = ubmp_div(
                    ubmp_mul(ubmp_mul(ubmp_mul(2.0f, hgtv), lpv),
                             ubmp_sub(ubmp_div(frcv, lpv), 1.0f)),
                    denominator);
            }
        }
        if (bs_value <= 1.0f || bs_value >= 150.0f) {
            bs_value = base_bs;
            ws_value = base_ws;
        }
        if (ws_value <= 1.0f || ws_value >= 150.0f) {
            ws_value = base_ws;
            bs_value = base_bs;
        }
        bs[out] = bs_value;
        ws[out] = ws_value;
        bs_urb[out] = bs_value;
        ws_urb[out] = ws_value;
    }
    // bb_param's four fixed layer arrays, in metres.
    const float dzgb_values[10] = {0.2f, 0.12f, 0.08f, 0.05f, 0.03f,
                                   0.02f, 0.02f, 0.01f, 0.005f, 0.0025f};
    const float dzr_values[10] = {0.02f, 0.02f, 0.02f, 0.02f, 0.02f,
                                  0.02f, 0.02f, 0.01f, 0.005f, 0.0025f};
    const float dzw_values[10] = {0.02f, 0.02f, 0.02f, 0.02f, 0.02f,
                                  0.02f, 0.02f, 0.01f, 0.005f, 0.0025f};
    const float dzf_values[10] = {0.02f, 0.02f, 0.02f, 0.02f, 0.02f,
                                  0.02f, 0.02f, 0.02f, 0.02f, 0.02f};
    for (uint i = 0; i < 10u; ++i) {
        dzgb[ulong(col) * 10ul + i] = dzgb_values[i];
        dzr[ulong(col) * 10ul + i] = dzr_values[i];
        dzw[ulong(col) * 10ul + i] = dzw_values[i];
        dzf[ulong(col) * 10ul + i] = dzf_values[i];
    }
    error[col] = 0;
}
