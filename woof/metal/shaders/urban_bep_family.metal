#include <metal_stdlib>
#include "urban_bep_family_bindings.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"
using namespace metal;
#pragma clang fp contract(off)

// WRF v4.7.1 Urban BEP class/view initialization and column-preparation
// source proposals. The full column solve remains out of this packet pending
// its solver call graph and source-matched glibc trig interfaces.
constant int BEP_NZ_UM = 18;
constant int BEP_NDM = 2;
constant int BEP_NURBM = 11;
constant float BEP_PI = 3.141592653f;

static inline float bep_add(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float bep_sub(float a, float b) {
    return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)));
}
static inline float bep_mul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}
static inline float bep_div(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}
static inline float bep_pow(float x, float y) {
    return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x), as_type<uint>(y)));
}
static inline float bep_log(float x) {
    return as_type<float>(soft_gfk_log(as_type<uint>(x)));
}
static inline float bep_atan(float x) {
    return precise::atan(x);
}
static inline float bep_sin(float x) {
    return precise::sin(x);
}
static inline float bep_cos(float x) {
    return precise::cos(x);
}
static inline float bep_atan2(float y, float x) {
    return precise::atan2(y, x);
}
static inline float bep_asin(float x) {
    return precise::asin(x);
}
static inline float bep_acos(float x) {
    return precise::acos(x);
}
static inline float bep_tan(float x) {
    return precise::tan(x);
}
// Compose the shared experimental RN-even binary32 square-root proposal.
static inline float bep_sqrtf(float x) {
    return as_type<float>(soft32_sqrt(as_type<uint>(x)));
}
static inline float bep_fmax(float a, float b) {
    if (soft32_is_nan(as_type<uint>(a))) return b;
    if (soft32_is_nan(as_type<uint>(b))) return a;
    if (a == b && a == 0.0f) return 0.0f;
    return a > b ? a : b;
}
static inline float bep_fmin(float a, float b) {
    if (soft32_is_nan(as_type<uint>(a))) return b;
    if (soft32_is_nan(as_type<uint>(b))) return a;
    if (a == b && a == 0.0f)
        return as_type<float>(as_type<uint>(a) | as_type<uint>(b));
    return a < b ? a : b;
}

static inline float bep_abs(float x) {
    return as_type<float>(as_type<uint>(x) & 0x7fffffffu);
}

static inline float bep_fprls(float a, float b, float c) {
    const float x = bep_div(a, c);
    const float y = bep_div(b, c);
    if (a == 0.0f || b == 0.0f) return 0.0f;
    const float xx = bep_mul(x, x);
    const float yy = bep_mul(y, y);
    const float one_xx = bep_add(1.0f, xx);
    const float one_yy = bep_add(1.0f, yy);
    const float common = bep_add(bep_add(1.0f, xx), yy);
    const float term0 = bep_log(bep_pow(bep_div(bep_mul(one_xx, one_yy), common), 0.5f));
    const float root_x = bep_pow(one_xx, 0.5f);
    const float term1 = bep_mul(bep_mul(y, root_x), bep_atan(bep_div(y, root_x)));
    const float root_y = bep_pow(one_yy, 0.5f);
    const float term2 = bep_mul(bep_mul(x, root_y), bep_atan(bep_div(x, root_y)));
    const float term3 = bep_mul(y, bep_atan(y));
    const float term4 = bep_mul(x, bep_atan(x));
    float result = bep_sub(bep_sub(bep_add(bep_add(term0, term1), term2), term3), term4);
    return bep_div(bep_mul(result, 2.0f), bep_mul(bep_mul(BEP_PI, x), y));
}

static inline float bep_fnrms(float a, float b, float c) {
    const float x = bep_div(a, b);
    const float y = bep_div(c, b);
    const float xx = bep_mul(x, x);
    const float yy = bep_mul(y, y);
    const float z = bep_add(xx, yy);
    if (y == 0.0f || x == 0.0f) return 0.0f;
    const float one_xx = bep_add(1.0f, xx);
    const float one_yy = bep_add(1.0f, yy);
    const float one_z = bep_add(1.0f, z);
    const float a1 = bep_log(bep_div(bep_mul(one_xx, one_yy), one_z));
    const float a2inner = bep_div(
        bep_div(bep_mul(yy, one_z), z), one_yy);
    const float a2 = bep_mul(yy, bep_log(a2inner));
    const float a3inner = bep_div(
        bep_div(bep_mul(xx, one_z), z), one_xx);
    const float a3 = bep_mul(xx, bep_log(a3inner));
    const float a4 = bep_mul(y, bep_atan(bep_div(1.0f, y)));
    const float a5 = bep_mul(x, bep_atan(bep_div(1.0f, x)));
    const float root0 = bep_sqrtf(z);
    const float root1 = bep_sqrtf(z);
    const float a6 = bep_mul(root0, bep_atan(bep_div(1.0f, root1)));
    const float sum = bep_sub(bep_add(bep_add(
        bep_mul(0.25f, bep_add(bep_add(a1, a2), a3)), a4), a5), a6);
    return bep_div(sum, bep_mul(BEP_PI, y));
}

static inline void bep_view_factor(int nz_u, int id, const thread float *z,
                                   float dxy, float ws, device float *v) {
    const int hut_i = nz_u;
    const float hut = z[hut_i];
    for (int jz = 1; jz <= nz_u; ++jz) {
        for (int iz = 1; iz <= nz_u; ++iz) {
            const float f123 = bep_fprls(dxy, bep_abs(bep_sub(z[jz], z[iz])), ws);
            const float f23 = bep_fprls(dxy, bep_abs(bep_sub(z[jz], z[iz + 1])), ws);
            const float f12 = bep_fprls(dxy, bep_abs(bep_sub(z[jz - 1], z[iz])), ws);
            const float f2 = bep_fprls(dxy, bep_abs(bep_sub(z[jz - 1], z[iz + 1])), ws);
            const float a123 = bep_mul(dxy, bep_abs(bep_sub(z[jz], z[iz])));
            const float a12 = bep_mul(dxy, bep_abs(bep_sub(z[jz - 1], z[iz])));
            const float a23 = bep_mul(dxy, bep_abs(bep_sub(z[jz], z[iz + 1])));
            const float a1 = bep_mul(dxy, bep_abs(bep_sub(z[iz], z[iz + 1])));
            const float a2 = bep_mul(dxy, bep_abs(bep_sub(z[jz - 1], z[iz + 1])));
            const float a3 = bep_mul(dxy, bep_abs(bep_sub(z[jz], z[jz - 1])));
            const float q = bep_add(bep_sub(bep_sub(bep_mul(a123, f123),
                bep_mul(a23, f23)), bep_mul(a12, f12)), bep_mul(a2, f2));
            const float ftot = bep_div(bep_mul(0.5f, q), a1);
            const ulong at = ulong(iz - 1) + 18ul * ulong(jz - 1) + 324ul * ulong(id);
            v[at] = bep_div(bep_mul(ftot, a1), a3);
        }
        const float f12 = bep_fnrms(z[jz], dxy, ws);
        const float f1 = bep_fnrms(z[jz - 1], dxy, ws);
        const float a1 = bep_mul(ws, dxy);
        const float a12 = bep_mul(ws, dxy);
        const float a4 = bep_mul(bep_sub(z[jz], z[jz - 1]), dxy);
        const float ftot = bep_div(bep_sub(bep_mul(a12, f12),
                                           bep_mul(a12, f1)), a1);
        v[684ul + ulong(jz - 1) + 18ul * ulong(id)] =
            bep_div(bep_mul(ftot, a1), a4);

        const float f12s = bep_fnrms(bep_sub(hut, z[jz - 1]), dxy, ws);
        const float f1s = bep_fnrms(bep_sub(hut, z[jz]), dxy, ws);
        const float a1s = bep_mul(ws, dxy);
        const float a12s = bep_mul(ws, dxy);
        const float a4s = bep_mul(bep_sub(z[jz], z[jz - 1]), dxy);
        const float ftots = bep_div(bep_sub(bep_mul(a12s, f12s),
                                            bep_mul(a12s, f1s)), a1s);
        v[720ul + ulong(jz - 1) + 18ul * ulong(id)] =
            bep_div(bep_mul(ftots, a1s), a4s);
    }
    for (int iz = 1; iz <= nz_u; ++iz) {
        const float f12 = bep_fnrms(ws, dxy, bep_sub(hut, z[iz - 1]));
        const float f1 = bep_fnrms(ws, dxy, bep_sub(hut, z[iz]));
        const float a1 = bep_mul(bep_sub(z[iz], z[iz - 1]), dxy);
        const float a2 = bep_mul(bep_sub(hut, z[iz]), dxy);
        const float a12 = bep_mul(bep_sub(hut, z[iz - 1]), dxy);
        const float a4 = bep_mul(ws, dxy);
        const float ftot = bep_div(bep_sub(bep_mul(a12, f12),
                                           bep_mul(a2, f1)), a1);
        v[756ul + ulong(iz - 1) + 18ul * ulong(id)] =
            bep_div(bep_mul(ftot, a1), a4);
    }
    for (int iz = 1; iz <= nz_u; ++iz) {
        const float f12 = bep_fnrms(ws, dxy, z[iz]);
        const float f1 = bep_fnrms(ws, dxy, z[iz - 1]);
        const float a1 = bep_mul(bep_sub(z[iz], z[iz - 1]), dxy);
        const float a2 = bep_mul(z[iz - 1], dxy);
        const float a12 = bep_mul(z[iz], dxy);
        const float a4 = bep_mul(ws, dxy);
        const float ftot = bep_div(bep_sub(bep_mul(a12, f12),
                                           bep_mul(a2, f1)), a1);
        v[648ul + ulong(iz - 1) + 18ul * ulong(id)] =
            bep_div(bep_mul(ftot, a1), a4);
    }
    v[792ul + ulong(id)] = bep_fprls(dxy, ws, hut);
}

// Per-case packed inputs are a canonical split of UrbanBepTable: table_x
// contains its float fields in declaration order, with each two-dimensional
// Fortran table stored column-major; table_ix contains icate, numdir, numhgt.
// class_x similarly packs the float fields of UrbanBepClass with integer
// fields omitted; class_ix is nd_u[11], nz_u[11], error.
kernel void urban_bep_class_init(
    constant UrbanBepClassInitResources &r [[buffer(0)]],
    constant UrbanBepClassInitParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncase)) return;
    const ulong tx = ulong(tid) * 1386ul;
    const ulong ti = ulong(tid) * 23ul;
    const ulong cx = ulong(tid) * 1085ul;
    const ulong ci = ulong(tid) * 23ul;

    for (int j = 0; j < 1085; ++j) r.class_x[cx + ulong(j)] = 0.0f;
    for (int j = 0; j < 23; ++j) r.class_ix[ci + ulong(j)] = 0;

    const int icate = r.table_ix[ti];
    // Valid table records are range checked by the host packer. Keep a bad
    // record from indexing past this fixed WRF module extent.
    if (icate < 1 || icate > BEP_NURBM) {
        r.class_ix[ci + 22ul] = 3000;
        return;
    }

    // UrbanBepClass float fields: seventeen 11-class arrays first.
    const int ALAG = 0, ALAW = 11, ALAR = 22, CSG = 33, CSW = 44, CSR = 55;
    const int TWINI = 66, TRINI = 77, TGINI = 88, ALBG = 99, ALBW = 110;
    const int ALBR = 121, EMG = 132, EMW = 143, EMR = 154, Z0G = 165, Z0R = 176;
    const int STRD = 187, DRST = 209, WS = 231, BS = 253;
    const int HB = 275, DB = 473, SS = 671, PB = 869, ZU = 1067;

    // Copy the all-class values that bep_init_para assigns unconditionally.
    for (int iu = 0; iu < BEP_NURBM; ++iu) {
        const ulong o = ulong(iu);
        r.class_x[cx + ulong(TWINI) + o] = r.table_x[tx + 66ul + o];
        r.class_x[cx + ulong(TRINI) + o] = r.table_x[tx + 77ul + o];
        r.class_x[cx + ulong(TGINI) + o] = r.table_x[tx + 88ul + o];
        r.class_x[cx + ulong(ALBW) + o] = r.table_x[tx + 99ul + o];
        r.class_x[cx + ulong(ALBR) + o] = r.table_x[tx + 110ul + o];
        r.class_x[cx + ulong(ALBG) + o] = r.table_x[tx + 121ul + o];
        r.class_x[cx + ulong(EMW) + o] = r.table_x[tx + 132ul + o];
        r.class_x[cx + ulong(EMR) + o] = r.table_x[tx + 143ul + o];
        r.class_x[cx + ulong(EMG) + o] = r.table_x[tx + 154ul + o];
        r.class_x[cx + ulong(Z0R) + o] = r.table_x[tx + 165ul + o];
        r.class_x[cx + ulong(Z0G) + o] = r.table_x[tx + 176ul + o];
        r.class_ix[ci + o] = r.table_ix[ti + 1ul + o];
    }

    const float cv = bep_mul(bep_div(1.0f, 4.1868f), 1.0e-6f);
    const float av = bep_mul(bep_div(1.0f, 4.1868f), 1.0e-2f);
    for (int iu = 0; iu < BEP_NURBM; ++iu) {
        const ulong o = ulong(iu);
        r.class_x[cx + ulong(CSW) + o] = bep_div(r.table_x[tx + o], cv);
        r.class_x[cx + ulong(CSR) + o] = bep_div(r.table_x[tx + 11ul + o], cv);
        r.class_x[cx + ulong(CSG) + o] = bep_div(r.table_x[tx + 22ul + o], cv);
    }
    for (int iu = 0; iu < icate; ++iu) {
        const ulong o = ulong(iu);
        r.class_x[cx + ulong(ALAW) + o] = bep_div(bep_div(r.table_x[tx + 33ul + o], r.class_x[cx + ulong(CSW) + o]), av);
        r.class_x[cx + ulong(ALAR) + o] = bep_div(bep_div(r.table_x[tx + 44ul + o], r.class_x[cx + ulong(CSR) + o]), av);
        r.class_x[cx + ulong(ALAG) + o] = bep_div(bep_div(r.table_x[tx + 55ul + o], r.class_x[cx + ulong(CSG) + o]), av);
    }

    for (int iu = 0; iu < icate; ++iu) {
        const int nd = r.table_ix[ti + 1ul + ulong(iu)];
        if (nd > BEP_NDM) { r.class_ix[ci + 22ul] = 3081; return; }
        for (int id = 0; id < nd; ++id) {
            const ulong src = ulong(id + 3 * iu);
            const ulong dst = ulong(id + 2 * iu);
            r.class_x[cx + ulong(DRST) + dst] =
                bep_div(bep_mul(r.table_x[tx + 187ul + src], BEP_PI), 180.0f);
            r.class_x[cx + ulong(WS) + dst] = r.table_x[tx + 220ul + src];
            r.class_x[cx + ulong(BS) + dst] = r.table_x[tx + 253ul + src];
        }
    }
    for (int iu = 0; iu < icate; ++iu) {
        const int nh = r.table_ix[ti + 12ul + ulong(iu)];
        if (BEP_NZ_UM < nh + 3) { r.class_ix[ci + 22ul] = 3093; return; }
        for (int k = 0; k < nh; ++k) {
            const ulong src = ulong(k + 50 * iu);
            const ulong dst = ulong(k + BEP_NZ_UM * iu);
            r.class_x[cx + ulong(HB) + dst] = r.table_x[tx + 286ul + src];
            r.class_x[cx + ulong(DB) + dst] = r.table_x[tx + 836ul + src];
        }
    }

    // bep_init_para assigns the default direction only after validation.
    for (int iu = 0; iu < BEP_NURBM; ++iu) {
        r.class_x[cx + ulong(STRD + 2 * iu)] = 100000.0f;
        r.class_x[cx + ulong(STRD + 2 * iu + 1)] = 100000.0f;
    }

    // bep_icbep: fixed urban-level heights, normalize every class' building
    // fractions, determine each class' active height, then form ss_u/pb_u.
    r.class_x[cx + ulong(ZU)] = 0.0f;
    for (int iz = 0; iz < BEP_NZ_UM - 1; ++iz)
        r.class_x[cx + ulong(ZU + iz + 1)] = bep_add(r.class_x[cx + ulong(ZU + iz)], 5.0f);
    for (int iu = 0; iu < BEP_NURBM; ++iu) {
        float dtot = 0.0f;
        for (int ilu = 0; ilu < BEP_NZ_UM; ++ilu)
            dtot = bep_add(dtot, r.class_x[cx + ulong(DB + ilu + BEP_NZ_UM * iu)]);
        for (int ilu = 0; ilu < BEP_NZ_UM; ++ilu) {
            const ulong j = cx + ulong(DB + ilu + BEP_NZ_UM * iu);
            r.class_x[j] = bep_div(r.class_x[j], dtot);
        }
    }
    for (int iu = 0; iu < BEP_NURBM; ++iu) {
        float hbmax = 0.0f;
        for (int ilu = 0; ilu < BEP_NZ_UM; ++ilu) {
            const float h = r.class_x[cx + ulong(HB + ilu + BEP_NZ_UM * iu)];
            if (h > hbmax) hbmax = h;
        }
        int iz_u = 1;
        for (; iz_u <= BEP_NZ_UM - 1; ++iz_u) {
            if (r.class_x[cx + ulong(ZU + iz_u)] > hbmax) break;
        }
        const int nzu = iz_u + 1;
        r.class_ix[ci + 11ul + ulong(iu)] = nzu;
        if (nzu + 1 > BEP_NZ_UM) {
            r.class_ix[ci + 22ul] = 2740;
            return;
        }
        const int nd = r.class_ix[ci + ulong(iu)];
        for (int id = 0; id < nd; ++id) {
            for (int iz = 0; iz < nzu; ++iz) {
                float ss = 0.0f;
                for (int ilu = 0; ilu < BEP_NZ_UM; ++ilu) {
                    const float h = r.class_x[cx + ulong(HB + ilu + BEP_NZ_UM * iu)];
                    if (r.class_x[cx + ulong(ZU + iz)] <= h &&
                        r.class_x[cx + ulong(ZU + iz + 1)] > h)
                        ss = bep_add(ss, r.class_x[cx + ulong(DB + ilu + BEP_NZ_UM * iu)]);
                }
                const ulong at = cx + ulong(SS + iz + BEP_NZ_UM * iu);
                r.class_x[at] = ss;
            }
        }
        r.class_x[cx + ulong(PB + BEP_NZ_UM * iu)] = 1.0f;
        for (int iz = 0; iz < nzu; ++iz) {
            const ulong curr = cx + ulong(PB + iz + BEP_NZ_UM * iu);
            const ulong next = cx + ulong(PB + iz + 1 + BEP_NZ_UM * iu);
            r.class_x[next] = bep_fmax(0.0f, bep_sub(r.class_x[curr], r.class_x[cx + ulong(SS + iz + BEP_NZ_UM * iu)]));
        }
    }
}

// Flat ncase x 11 expansion of CUDA's one-block-per-class view initializer.
// All routines in this call chain allocate no arena slots; the CUDA scratch
// argument is therefore not read by this faithful subset.
kernel void urban_bep_view_init(
    constant UrbanBepViewInitResources &r [[buffer(0)]],
    constant UrbanBepViewInitParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint total = uint(p.ncase) * 11u;
    if (tid >= total) return;
    const ulong case_x = ulong(tid / 11u) * 1085ul;
    const ulong case_i = ulong(tid / 11u) * 23ul;
    const ulong view_x = ulong(tid) * 794ul;
    const int iurb = int(tid % 11u);
    for (int i = 0; i < 794; ++i) r.view_x[view_x + ulong(i)] = 0.0f;
    int error = r.class_ix[case_i + 22ul];
    const int nz_u = r.class_ix[case_i + 11ul + ulong(iurb)];
    if (error != 0 || nz_u < 1 || nz_u + 1 > BEP_NZ_UM) {
        r.view_ix[tid] = error != 0 ? error : 2736;
        return;
    }
    const int nd = r.class_ix[case_i + ulong(iurb)];
    thread float z[BEP_NZ_UM];
    for (int i = 0; i < BEP_NZ_UM; ++i)
        z[i] = r.class_x[case_x + 1067ul + ulong(i)];
    for (int id = 0; id < nd; ++id) {
        const float dxy = r.class_x[case_x + 187ul + ulong(2 * iurb + id)];
        const float ws = r.class_x[case_x + 231ul + ulong(2 * iurb + id)];
        bep_view_factor(nz_u, id, z, dxy, ws, r.view_x + view_x);
    }
    r.view_ix[tid] = 0;
}

// Column preparation stage matching urban_bep_column's source lines 352-369:
// z is accumulated in source order, the prognostic fields are gathered from
// [level, column] storage, pt0 is initialized to 300 K, and a/b coefficients
// start at zero. `column_x` is [column, source-field, one-based-slot], with
// source fields z,ua,va,pt,da,pr,pt0,a_u,a_v,a_t,a_e,b_u,b_v,b_t,b_e.
// This stage does not claim to execute bep_bep1d or scatter a column update.
kernel void urban_bep_column_gather(
    constant UrbanBepColumnGatherResources &r [[buffer(0)]],
    constant UrbanBepColumnGatherParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong slots = ulong(p.nz) + 1ul;
    const ulong row = ulong(col) * 15ul * slots;
    const bool active = r.frc_urb[col] > 0.0f;
    r.active_cols[col] = active ? 1 : 0;
    for (int field = 0; field < 15; ++field)
        for (int k = 0; k <= p.nz; ++k)
            r.column_x[row + ulong(field) * slots + ulong(k)] = 0.0f;
    if (!active) return;

    r.column_x[row] = 0.0f;
    for (int k = 1; k <= p.nz; ++k) {
        const ulong src = ulong(k - 1) * ulong(p.ncol) + ulong(col);
        const ulong z_at = row + ulong(k);
        const ulong slot = ulong(k);
        r.column_x[z_at] = as_type<float>(soft32_add(
            as_type<uint>(r.column_x[z_at - 1ul]), as_type<uint>(r.dz8w[src])));
        r.column_x[row + 1ul * slots + slot] = r.u_phy[src];
        r.column_x[row + 2ul * slots + slot] = r.v_phy[src];
        r.column_x[row + 3ul * slots + slot] = r.th_phy[src];
        r.column_x[row + 4ul * slots + slot] = r.rho[src];
        r.column_x[row + 5ul * slots + slot] = r.p_phy[src];
        r.column_x[row + 6ul * slots + slot] = 300.0f;
    }
}

// Column-specific translation of bep_icbephi_xy (urban_bep.cu:2286-2358).
// Profile channels are hb_u, hi_urb1d, ss_urb, pb_urb; ix channels are nzurb
// and the source error. Each lane owns a separate column's work arrays.
kernel void urban_bep_column_profile(
    constant UrbanBepColumnProfileResources &r [[buffer(0)]],
    constant UrbanBepColumnProfileParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong base = ulong(col) * 72ul;
    const ulong ix = ulong(col) * 2ul;
    for (int field = 0; field < 4; ++field)
        for (int k = 0; k < BEP_NZ_UM; ++k)
            r.urban_profile[base + ulong(field * BEP_NZ_UM + k)] = 0.0f;
    r.urban_profile_ix[ix] = 0;
    r.urban_profile_ix[ix + 1ul] = 0;
    if (r.active_cols[col] == 0) return;

    int error = r.class_ix[22ul];
    if (error == 0 && (p.num_urban_hi < 0 || p.num_urban_hi >= BEP_NZ_UM))
        error = 348;
    if (error != 0) {
        r.urban_profile_ix[ix + 1ul] = error;
        return;
    }

    float total = 0.0f;
    bool negative = false;
    for (int k = 0; k < BEP_NZ_UM; ++k) {
        const float h = k < p.num_urban_hi
            ? r.hi_urb[ulong(k) * ulong(p.ncol) + ulong(col)] : 0.0f;
        r.urban_profile[base + 18ul + ulong(k)] = h;
        total = bep_add(total, h);
        negative = negative || h < 0.0f;
    }
    if (negative || !(total > 0.0f)) return;

    r.urban_profile[base] = 5.0f;
    for (int k = 1; k < BEP_NZ_UM; ++k)
        r.urban_profile[base + ulong(k)] =
            bep_add(5.0f, r.urban_profile[base + ulong(k - 1)]);

    float hbmax = 0.0f;
    for (int k = 0; k < BEP_NZ_UM; ++k) {
        const ulong hi_at = base + 18ul + ulong(k);
        const float normalized = bep_div(r.urban_profile[hi_at], total);
        r.urban_profile[hi_at] = normalized;
        if (normalized > 0.0f && normalized <= 1.0f)
            hbmax = r.urban_profile[ulong(k)];
    }

    int iz_u = 1;
    for (; iz_u <= BEP_NZ_UM - 1; ++iz_u) {
        if (r.class_x[1067ul + ulong(iz_u)] > hbmax) break;
    }
    const int nzu = iz_u + 1;
    r.urban_profile_ix[ix] = nzu;
    if (nzu + 1 > BEP_NZ_UM) {
        r.urban_profile_ix[ix + 1ul] = 3477;
        return;
    }

    for (int iz = 1; iz <= nzu; ++iz) {
        float ss = 0.0f;
        const float zlo = r.class_x[1067ul + ulong(iz - 1)];
        const float zhi = r.class_x[1067ul + ulong(iz)];
        for (int k = 0; k < BEP_NZ_UM; ++k) {
            const float hb = r.urban_profile[ulong(k)];
            if (zlo <= hb && zhi > hb)
                ss = bep_add(ss, r.urban_profile[base + 18ul + ulong(k)]);
        }
        r.urban_profile[base + 36ul + ulong(iz - 1)] = ss;
    }
    r.urban_profile[base + 54ul] = 1.0f;
    for (int iz = 1; iz <= nzu; ++iz) {
        const float prev = r.urban_profile[base + 54ul + ulong(iz - 1)];
        const float ss = r.urban_profile[base + 36ul + ulong(iz - 1)];
        r.urban_profile[base + 54ul + ulong(iz)] = bep_fmax(0.0f, bep_sub(prev, ss));
    }
}

// Column-specific translation of bep_param (urban_bep.cu:306-477). Packed
// parameter floats: csg[10], alag[10], csr[10], alar[10], csw[10], alaw[10],
// z0[2*18], bs[2], ws[2], strd[2], drst[2], ss[18], pb[18].
kernel void urban_bep_column_params(
    constant UrbanBepColumnParamsResources &r [[buffer(0)]],
    constant UrbanBepColumnParamsParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong ox = ulong(col) * 140ul;
    const ulong oi = ulong(col) * 3ul;
    for (int k = 0; k < 140; ++k) r.column_param[ox + ulong(k)] = 0.0f;
    r.column_param_ix[oi] = 0;
    r.column_param_ix[oi + 1ul] = 0;
    r.column_param_ix[oi + 2ul] = 0;
    if (r.active_cols[col] == 0) return;

    int error = r.class_ix[22ul];
    if (error == 0) error = r.urban_profile_ix[ulong(col) * 2ul + 1ul];
    const int iurb = r.utype_urb[col];
    if (error == 0 && (iurb < 1 || iurb > BEP_NURBM)) error = 392;
    if (error != 0) {
        r.column_param_ix[oi + 2ul] = error;
        return;
    }
    const ulong u = ulong(iurb - 1);
    const ulong ci = ulong(iurb - 1);
    const int ndu = r.class_ix[ci];
    const int nzu = r.class_ix[11ul + ci];
    const ulong prof = ulong(col) * 72ul;
    bool has_urban_profile = false;
    for (int iz = 0; iz < BEP_NZ_UM; ++iz)
        has_urban_profile = has_urban_profile ||
            (r.urban_profile[prof + 36ul + ulong(iz)] != 0.0f);

    int nzurban;
    if (has_urban_profile) {
        nzurban = r.urban_profile_ix[ulong(col) * 2ul];
        for (int iz = 0; iz <= nzurban; ++iz) {
            r.column_param[ox + 104ul + ulong(iz)] =
                r.urban_profile[prof + 36ul + ulong(iz)];
            r.column_param[ox + 122ul + ulong(iz)] =
                r.urban_profile[prof + 54ul + ulong(iz)];
        }
    } else {
        nzurban = nzu;
        for (int iz = 0; iz <= nzu; ++iz) {
            r.column_param[ox + 104ul + ulong(iz)] =
                r.class_x[671ul + ulong(iz) + 18ul * u];
            r.column_param[ox + 122ul + ulong(iz)] =
                r.class_x[869ul + ulong(iz) + 18ul * u];
        }
    }
    r.column_param_ix[oi] = nzurban;
    r.column_param_ix[oi + 1ul] = ndu;

    for (int id = 0; id < ndu; ++id) {
        const ulong d = ulong(id);
        r.column_param[ox + 60ul + d] = r.class_x[165ul + u];
        for (int iz = 1; iz <= nzurban; ++iz)
            r.column_param[ox + 60ul + d + 2ul * ulong(iz)] =
                r.class_x[176ul + u];
        r.column_param[ox + 100ul + d] = r.class_x[187ul + d + 2ul * u];
        r.column_param[ox + 102ul + d] = r.class_x[209ul + d + 2ul * u];

        float ws = 0.0f;
        float bs = 0.0f;
        const float lp = r.lp_urb[col];
        const float lb = r.lb_urb[col];
        const float hgt = r.hgt_urb[col];
        const float frc = r.frc_urb[col];
        if (hgt <= 0.0f || lp <= 0.0f || lb <= 0.0f) {
            ws = r.class_x[231ul + d + 2ul * u];
            bs = r.class_x[253ul + d + 2ul * u];
        } else if (bep_div(lp, frc) < 1.0f && lp < lb) {
            bs = bep_div(bep_mul(bep_mul(2.0f, hgt), lp), bep_sub(lb, lp));
            ws = bep_div(bep_mul(bep_mul(bep_mul(2.0f, hgt), lp),
                                 bep_sub(bep_div(frc, lp), 1.0f)),
                         bep_sub(lb, lp));
        } else {
            ws = r.class_x[231ul + d + 2ul * u];
            bs = r.class_x[253ul + d + 2ul * u];
        }
        if (bs <= 1.0f || bs >= 150.0f) {
            bs = r.class_x[253ul + d + 2ul * u];
            ws = r.class_x[231ul + d + 2ul * u];
        }
        if (ws <= 1.0f || ws >= 150.0f) {
            bs = r.class_x[253ul + d + 2ul * u];
            ws = r.class_x[231ul + d + 2ul * u];
        }
        r.column_param[ox + 96ul + d] = bs;
        r.column_param[ox + 98ul + d] = ws;
    }
    for (int ig = 0; ig < 10; ++ig) {
        const ulong k = ulong(ig);
        r.column_param[ox + k] = r.class_x[33ul + u];
        r.column_param[ox + 10ul + k] = r.class_x[u];
        r.column_param[ox + 20ul + k] = r.class_x[55ul + u];
        r.column_param[ox + 30ul + k] = r.class_x[22ul + u];
        r.column_param[ox + 40ul + k] = r.class_x[44ul + u];
        r.column_param[ox + 50ul + k] = r.class_x[11ul + u];
    }
}

// Column-local view recompute matching the uncached bep_icbep_xy path. It
// uses the effective nzurban/strd/ws from bep_param, unlike class-init views.
kernel void urban_bep_column_views(
    constant UrbanBepColumnViewsResources &r [[buffer(0)]],
    constant UrbanBepColumnViewsParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong ox = ulong(col) * 794ul;
    const ulong oi = ulong(col) * 3ul;
    for (int k = 0; k < 794; ++k) r.column_view_x[ox + ulong(k)] = 0.0f;
    if (r.active_cols[col] == 0) {
        r.column_view_ix[col] = 0;
        return;
    }
    int error = r.column_param_ix[oi + 2ul];
    const int nzurban = r.column_param_ix[oi];
    const int ndu = r.column_param_ix[oi + 1ul];
    if (error == 0 && (nzurban < 1 || nzurban + 1 > BEP_NZ_UM)) error = 2736;
    if (error != 0) {
        r.column_view_ix[col] = error;
        return;
    }
    thread float z[BEP_NZ_UM];
    for (int k = 0; k < BEP_NZ_UM; ++k)
        z[k] = r.class_x[1067ul + ulong(k)];
    for (int id = 0; id < ndu; ++id) {
        const float strd = r.column_param[ox + 100ul + ulong(id)];
        const float ws = r.column_param[ox + 98ul + ulong(id)];
        bep_view_factor(nzurban, id, z, strd, ws, r.column_view_x + ox);
    }
    r.column_view_ix[col] = 0;
}

// Persistent-state gather matching urban_bep_column's collapsed maps at
// module_sf_bep.F:318-346. The flat destination preserves each local
// Fortran array's first-index-fastest order.
kernel void urban_bep_column_state_gather(
    constant UrbanBepColumnStateGatherResources &r [[buffer(0)]],
    constant UrbanBepColumnStateGatherParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong out = ulong(col) * 1210ul;
    for (int k = 0; k < 1210; ++k) r.column_state[out + ulong(k)] = 0.0f;
    r.column_state_ix[col] = 0;
    if (r.active_cols[col] == 0) return;

    const ulong param_ix = ulong(col) * 3ul;
    int error = r.column_param_ix[param_ix + 2ul];
    const int nzurban = r.column_param_ix[param_ix];
    if (error == 0 && (nzurban < 1 || nzurban + 1 > BEP_NZ_UM)) error = 2736;
    if (error != 0) {
        r.column_state_ix[col] = error;
        return;
    }

    for (int id = 0; id < 2; ++id) {
        for (int iz = 0; iz < BEP_NZ_UM; ++iz) {
            const ulong zd = ulong(iz * 2 + id) * ulong(p.ncol) + ulong(col);
            const ulong sfw_base = out + 1100ul + ulong(iz * 4 + id * 2);
            r.column_state[sfw_base] = r.sfw1_urb[zd];
            r.column_state[sfw_base + 1ul] = r.sfw2_urb[zd];
            r.column_state[out + 1174ul + ulong(id + 2 * iz)] = r.sfr_urb[zd];
            for (int iw = 0; iw < 10; ++iw) {
                const ulong zwd = ulong((iz * 10 + iw) * 2 + id) *
                                  ulong(p.ncol) + ulong(col);
                const ulong tw_base = out + ulong(4 * (iz + 18 * iw) + id * 2);
                const ulong tr_base = out + 740ul + ulong(2 * (iz + 18 * iw) + id);
                r.column_state[tw_base] = r.tw1_urb[zwd];
                r.column_state[tw_base + 1ul] = r.tw2_urb[zwd];
                r.column_state[tr_base] = r.trb_urb[zwd];
            }
        }
        r.column_state[out + 720ul + ulong(id)] =
            r.tgb_urb[ulong(id) * ulong(p.ncol) + ulong(col)];
        r.column_state[out + 1172ul + ulong(id)] =
            r.sfg_urb[ulong(id) * ulong(p.ncol) + ulong(col)];
    }
    r.column_state_ix[col] = 0;
}

// Inverse of the persistent-state gather, corresponding to the successful
// post-bep_bep1d scatter in urban_bep_column. It leaves failed/inactive
// columns untouched, matching the wrapper's error return before writeback.
kernel void urban_bep_column_state_scatter(
    constant UrbanBepColumnStateScatterResources &r [[buffer(0)]],
    constant UrbanBepColumnStateScatterParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    if (r.active_cols[col] == 0 || r.column_state_ix[col] != 0) return;
    const ulong in = ulong(col) * 1210ul;
    for (int id = 0; id < 2; ++id) {
        for (int iz = 0; iz < BEP_NZ_UM; ++iz) {
            const ulong zd = ulong(iz * 2 + id) * ulong(p.ncol) + ulong(col);
            const ulong sfw_base = in + 1100ul + ulong(iz * 4 + id * 2);
            r.sfw1_urb[zd] = r.column_state[sfw_base];
            r.sfw2_urb[zd] = r.column_state[sfw_base + 1ul];
            r.sfr_urb[zd] = r.column_state[in + 1174ul + ulong(id + 2 * iz)];
            for (int iw = 0; iw < 10; ++iw) {
                const ulong zwd = ulong((iz * 10 + iw) * 2 + id) *
                                  ulong(p.ncol) + ulong(col);
                const ulong tw_base = in + ulong(4 * (iz + 18 * iw) + id * 2);
                const ulong tr_base = in + 740ul + ulong(2 * (iz + 18 * iw) + id);
                r.tw1_urb[zwd] = r.column_state[tw_base];
                r.tw2_urb[zwd] = r.column_state[tw_base + 1ul];
                r.trb_urb[zwd] = r.column_state[tr_base];
            }
        }
        r.tgb_urb[ulong(id) * ulong(p.ncol) + ulong(col)] =
            r.column_state[in + 720ul + ulong(id)];
        r.sfg_urb[ulong(id) * ulong(p.ncol) + ulong(col)] =
            r.column_state[in + 1172ul + ulong(id)];
    }
}

// Initial bep_bep1d preparation: mass-layer thickness and the six
// source-ordered vertical overlap interpolations (urban_bep.cu:811-821).
// Destination channels are ua_u, va_u, pt_u, pt0_u, pr_u, da_u.
kernel void urban_bep_column_interp(
    constant UrbanBepColumnInterpResources &r [[buffer(0)]],
    constant UrbanBepColumnInterpParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong out = ulong(col) * 108ul;
    for (int k = 0; k < 108; ++k) r.column_interp_x[out + ulong(k)] = 0.0f;
    r.column_interp_ix[col] = 0;
    if (r.active_cols[col] == 0) return;
    if (r.column_param_ix[ulong(col) * 3ul + 2ul] != 0) {
        r.column_interp_ix[col] = r.column_param_ix[ulong(col) * 3ul + 2ul];
        return;
    }
    const int nzu = r.column_param_ix[ulong(col) * 3ul];
    if (nzu < 1 || nzu + 1 > BEP_NZ_UM) {
        r.column_interp_ix[col] = 2736;
        return;
    }
    const ulong in = ulong(col) * 15ul * (ulong(p.nz) + 1ul);
    const ulong zbase = in;
    const int src_fields[6] = {1, 2, 3, 6, 5, 4};
    for (int field = 0; field < 6; ++field) {
        const ulong src = in + ulong(src_fields[field]) * (ulong(p.nz) + 1ul);
        for (int iz_u = 1; iz_u <= nzu; ++iz_u) {
            float ctot = 0.0f;
            const float zu_lo = r.class_x[1067ul + ulong(iz_u - 1)];
            const float zu_hi = r.class_x[1067ul + ulong(iz_u)];
            for (int iz = 1; iz <= p.nz; ++iz) {
                const float zlo = r.column_x[zbase + ulong(iz - 1)];
                const float zhi = r.column_x[zbase + ulong(iz)];
                const float overlap_hi = bep_fmin(zhi, zu_hi);
                const float overlap_lo = bep_fmax(zlo, zu_lo);
                const float dz = bep_fmax(bep_sub(overlap_hi, overlap_lo), 0.0f);
                const float c = r.column_x[src + ulong(iz - 1)];
                ctot = bep_add(ctot, bep_mul(c, dz));
            }
            const float layer = bep_sub(zu_hi, zu_lo);
            r.column_interp_x[out + ulong(field * BEP_NZ_UM + iz_u - 1)] =
                bep_div(ctot, layer);
        }
    }
}

// Translate bep_interp_length (urban_bep.cu:916-971), which computes the
// urban mixing length dl_u and grid mixing length dlg after bep_bep1d.
kernel void urban_bep_column_lengths(
    constant UrbanBepColumnLengthsResources &r [[buffer(0)]],
    constant UrbanBepColumnLengthsParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong out = ulong(col) * 2ul * ulong(p.nz);
    for (int k = 0; k < 2 * p.nz; ++k) r.column_lengths[out + ulong(k)] = 0.0f;
    r.column_lengths_ix[col] = 0;
    if (r.active_cols[col] == 0) return;
    const ulong pix = ulong(col) * 3ul;
    int error = r.column_state_ix[col];
    const int nzu = r.column_param_ix[pix];
    const int nd = r.column_param_ix[pix + 1ul];
    if (error == 0 && (nzu < 1 || nzu + 1 > BEP_NZ_UM)) error = 2736;
    if (error != 0) {
        r.column_lengths_ix[col] = error;
        return;
    }
    const ulong xin = ulong(col) * 15ul * (ulong(p.nz) + 1ul);
    const ulong pin = ulong(col) * 140ul;
    const float nd_f = float(nd);
    for (int iz = 1; iz <= p.nz; ++iz) {
        const float z = r.column_x[xin + ulong(iz - 1)];
        float ulu = 0.0f;
        float ssl = 0.0f;
        for (int id = 1; id <= nd; ++id) {
            for (int iz_u = 2; iz_u <= nzu; ++iz_u) {
                const float zu = r.class_x[1067ul + ulong(iz_u - 1)];
                if (zu > z) {
                    const float ss = r.column_param[pin + 104ul + ulong(iz_u - 1)];
                    ulu = bep_add(ulu, bep_div(bep_div(ss, zu), nd_f));
                    ssl = bep_add(ssl, bep_div(ss, nd_f));
                }
            }
        }
        r.column_lengths[out + ulong(iz - 1)] =
            ulu != 0.0f ? bep_div(ssl, ulu) : 0.0f;

        const float znext = r.column_x[xin + ulong(iz)];
        const float zmid = bep_div(bep_add(z, znext), 2.0f);
        float dlg = 0.0f;
        for (int id = 1; id <= nd; ++id) {
            const ulong d = ulong(id - 1);
            const float ws = r.column_param[pin + 98ul + d];
            const float bs = r.column_param[pin + 96ul + d];
            float sftot = ws;
            float dlgtmp = bep_div(ws, zmid);
            for (int iz_u = 1; iz_u <= nzu; ++iz_u) {
                const float zu = r.class_x[1067ul + ulong(iz_u - 1)];
                if (zmid > zu) {
                    const float ss = r.column_param[pin + 104ul + ulong(iz_u - 1)];
                    dlgtmp = bep_add(dlgtmp,
                        bep_div(bep_mul(ss, bs), bep_sub(zmid, zu)));
                    sftot = bep_add(sftot, bep_mul(ss, bs));
                }
            }
            dlg = bep_add(dlg, bep_div(bep_div(dlgtmp, sftot), nd_f));
        }
        r.column_lengths[out + ulong(p.nz + iz - 1)] = bep_div(1.0f, dlg);
    }
}

// Translate bep_urban_meso (urban_bep.cu:695-914), consuming the building
// fluxes and emitting the eight BEP coefficients plus sf/vl. building_x holds
// uva,vva,uvb,vvb,tva,tvb,evb (4x18 each), then uhb,vhb,thb,ehb (2x18 each).
kernel void urban_bep_column_meso(
    constant UrbanBepColumnMesoResources &r [[buffer(0)]],
    constant UrbanBepColumnMesoParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong slots = ulong(p.nz) + 1ul;
    const ulong out = ulong(col) * 10ul * slots;
    for (int k = 0; k < int(10ul * slots); ++k)
        r.column_meso_x[out + ulong(k)] = 0.0f;
    r.column_meso_ix[col] = 0;
    if (r.active_cols[col] == 0) return;
    int error = r.column_state_ix[col];
    const ulong pix = ulong(col) * 3ul;
    const int nzu = r.column_param_ix[pix];
    const int nd = r.column_param_ix[pix + 1ul];
    if (error == 0 && (nzu < 1 || nzu + 1 > BEP_NZ_UM)) error = 2736;
    if (error != 0) {
        r.column_meso_ix[col] = error;
        return;
    }

    const ulong xin = ulong(col) * 15ul * slots;
    const ulong pin = ulong(col) * 140ul;
    const ulong bin = ulong(col) * 648ul;
    const float nd_f = float(nd);

    for (int id = 1; id <= nd; ++id) {
        const ulong d = ulong(id - 1);
        const float ws = r.column_param[pin + 98ul + d];
        const float bs = r.column_param[pin + 96ul + d];
        for (int iz = 2; iz <= p.nz + 1; ++iz) {
            float sr = 0.0f;
            const float z = r.column_x[xin + ulong(iz - 1)];
            for (int iz_u = 2; iz_u <= nzu; ++iz_u) {
                const float zu_hi = r.class_x[1067ul + ulong(iz_u - 1)];
                const float zu_lo = r.class_x[1067ul + ulong(iz_u - 2)];
                if (z < zu_hi && z >= zu_lo)
                    sr = r.column_param[pin + 122ul + ulong(iz_u - 1)];
            }
            const float frac = bep_div(
                bep_div(bep_add(ws, bep_mul(bep_sub(1.0f, sr), bs)),
                        bep_add(ws, bs)), nd_f);
            r.column_meso_x[out + 8ul * slots + ulong(iz - 1)] =
                bep_add(r.column_meso_x[out + 8ul * slots + ulong(iz - 1)], frac);
        }
    }

    for (int iz = 1; iz <= p.nz; ++iz) {
        const ulong k = ulong(iz - 1);
        const float zlo = r.column_x[xin + k];
        const float zhi = r.column_x[xin + k + 1ul];
        const float dz = bep_sub(zhi, zlo);
        for (int id = 1; id <= nd; ++id) {
            const ulong d = ulong(id - 1);
            const float ws = r.column_param[pin + 98ul + d];
            const float bs = r.column_param[pin + 96ul + d];
            float vtot = 0.0f;
            for (int iz_u = 1; iz_u <= nzu; ++iz_u) {
                const float ulo = r.class_x[1067ul + ulong(iz_u - 1)];
                const float uhi = r.class_x[1067ul + ulong(iz_u)];
                const float overlap = bep_fmax(
                    bep_sub(bep_fmin(uhi, zhi), bep_fmax(ulo, zlo)), 0.0f);
                const float pb = r.column_param[pin + 122ul + ulong(iz_u)];
                vtot = bep_add(vtot, bep_mul(pb, overlap));
            }
            vtot = bep_div(vtot, dz);
            const float vf = bep_div(
                bep_sub(1.0f, bep_div(bep_mul(vtot, bs), bep_add(ws, bs))), nd_f);
            const ulong vl_at = out + 9ul * slots + k;
            r.column_meso_x[vl_at] = bep_add(r.column_meso_x[vl_at], vf);
        }
    }

    for (int id = 1; id <= nd; ++id) {
        const ulong d = ulong(id - 1);
        const float ws = r.column_param[pin + 98ul + d];
        const float bs = r.column_param[pin + 96ul + d];
        const float z0 = r.column_x[xin];
        const float z1 = r.column_x[xin + 1ul];
        const float dz0 = bep_sub(z1, z0);
        const float vl0 = r.column_meso_x[out + 9ul * slots];
        const float fact0 = bep_div(
            bep_div(bep_mul(bep_div(bep_div(1.0f, vl0), dz0), ws),
                    bep_add(ws, bs)), nd_f);
        r.column_meso_x[out + 6ul * slots] = bep_add(
            r.column_meso_x[out + 6ul * slots],
            bep_mul(r.column_building_x[bin + 576ul + d], fact0));
        r.column_meso_x[out + 4ul * slots] = bep_add(
            r.column_meso_x[out + 4ul * slots],
            bep_mul(r.column_building_x[bin + 504ul + d], fact0));
        r.column_meso_x[out + 5ul * slots] = bep_add(
            r.column_meso_x[out + 5ul * slots],
            bep_mul(r.column_building_x[bin + 540ul + d], fact0));
        const float dz_u0 = bep_sub(r.class_x[1068ul], r.class_x[1067ul]);
        r.column_meso_x[out + 7ul * slots] = bep_add(
            r.column_meso_x[out + 7ul * slots],
            bep_mul(bep_mul(r.column_building_x[bin + 612ul + d], fact0), dz_u0));

        for (int iz = 1; iz <= p.nz; ++iz) {
            const ulong k = ulong(iz - 1);
            const float zlo = r.column_x[xin + k];
            const float zhi = r.column_x[xin + k + 1ul];
            const float dz = bep_sub(zhi, zlo);
            float st = 0.0f, su = 0.0f, sv = 0.0f, se = 0.0f;
            for (int iz_u = 2; iz_u <= nzu; ++iz_u) {
                const float zu = r.class_x[1067ul + ulong(iz_u - 1)];
                if (zlo <= zu && zhi > zu) {
                    const ulong u = ulong(iz_u - 1);
                    const float ss = r.column_param[pin + 104ul + u];
                    st = bep_add(st, bep_mul(ss, r.column_building_x[bin + 576ul + d + 2ul * u]));
                    su = bep_add(su, bep_mul(ss, r.column_building_x[bin + 504ul + d + 2ul * u]));
                    sv = bep_add(sv, bep_mul(ss, r.column_building_x[bin + 540ul + d + 2ul * u]));
                    const float ehb = r.column_building_x[bin + 612ul + d + 2ul * u];
                    const float dz_u = bep_sub(r.class_x[1067ul + ulong(iz_u)], zu);
                    se = bep_add(se, bep_mul(bep_mul(ss, ehb), dz_u));
                }
            }
            const float fact = bep_div(
                bep_div(bep_div(bep_div(bs, bep_add(ws, bs)),
                                r.column_meso_x[out + 9ul * slots + k]), dz), nd_f);
            r.column_meso_x[out + 6ul * slots + k] = bep_add(
                r.column_meso_x[out + 6ul * slots + k], bep_mul(st, fact));
            r.column_meso_x[out + 4ul * slots + k] = bep_add(
                r.column_meso_x[out + 4ul * slots + k], bep_mul(su, fact));
            r.column_meso_x[out + 5ul * slots + k] = bep_add(
                r.column_meso_x[out + 5ul * slots + k], bep_mul(sv, fact));
            r.column_meso_x[out + 7ul * slots + k] = bep_add(
                r.column_meso_x[out + 7ul * slots + k], bep_mul(se, fact));
        }
    }

    for (int iz = 1; iz <= p.nz; ++iz) {
        const ulong k = ulong(iz - 1);
        const float zlo = r.column_x[xin + k];
        const float zhi = r.column_x[xin + k + 1ul];
        const float dz = bep_sub(zhi, zlo);
        const float vl = r.column_meso_x[out + 9ul * slots + k];
        for (int id = 1; id <= nd; ++id) {
            const ulong d = ulong(id - 1);
            const float ws = r.column_param[pin + 98ul + d];
            const float bs = r.column_param[pin + 96ul + d];
            float vtb = 0.0f, vta = 0.0f, vua = 0.0f, vva = 0.0f;
            float vub = 0.0f, vvb = 0.0f, veb = 0.0f;
            for (int iz_u = 1; iz_u <= nzu; ++iz_u) {
                const float ulo = r.class_x[1067ul + ulong(iz_u - 1)];
                const float uhi = r.class_x[1067ul + ulong(iz_u)];
                const float dzz = bep_fmax(
                    bep_sub(bep_fmin(uhi, zhi), bep_fmax(ulo, zlo)), 0.0f);
                const float fact = bep_div(dzz, bep_add(ws, bs));
                const ulong u = ulong(iz_u - 1);
                const float pb = r.column_param[pin + 122ul + ulong(iz_u)];
                const ulong wall = 2ul * d + 4ul * u;
                vtb = bep_add(vtb, bep_mul(bep_mul(pb,
                    bep_add(r.column_building_x[bin + 360ul + wall],
                            r.column_building_x[bin + 360ul + wall + 1ul])), fact));
                vta = bep_add(vta, bep_mul(bep_mul(pb,
                    bep_add(r.column_building_x[bin + 288ul + wall],
                            r.column_building_x[bin + 288ul + wall + 1ul])), fact));
                vua = bep_add(vua, bep_mul(bep_mul(pb,
                    bep_add(r.column_building_x[bin + wall],
                            r.column_building_x[bin + wall + 1ul])), fact));
                vva = bep_add(vva, bep_mul(bep_mul(pb,
                    bep_add(r.column_building_x[bin + 72ul + wall],
                            r.column_building_x[bin + 72ul + wall + 1ul])), fact));
                vub = bep_add(vub, bep_mul(bep_mul(pb,
                    bep_add(r.column_building_x[bin + 144ul + wall],
                            r.column_building_x[bin + 144ul + wall + 1ul])), fact));
                vvb = bep_add(vvb, bep_mul(bep_mul(pb,
                    bep_add(r.column_building_x[bin + 216ul + wall],
                            r.column_building_x[bin + 216ul + wall + 1ul])), fact));
                veb = bep_add(veb, bep_mul(bep_mul(pb,
                    bep_add(r.column_building_x[bin + 432ul + wall],
                            r.column_building_x[bin + 432ul + wall + 1ul])), fact));
            }
            const float fact = bep_div(
                bep_div(bep_div(1.0f, vl), dz), nd_f);
            r.column_meso_x[out + 6ul * slots + k] = bep_add(
                r.column_meso_x[out + 6ul * slots + k], bep_mul(vtb, fact));
            r.column_meso_x[out + 2ul * slots + k] = bep_add(
                r.column_meso_x[out + 2ul * slots + k], bep_mul(vta, fact));
            r.column_meso_x[out + 0ul * slots + k] = bep_add(
                r.column_meso_x[out + 0ul * slots + k], bep_mul(vua, fact));
            r.column_meso_x[out + 1ul * slots + k] = bep_add(
                r.column_meso_x[out + 1ul * slots + k], bep_mul(vva, fact));
            r.column_meso_x[out + 4ul * slots + k] = bep_add(
                r.column_meso_x[out + 4ul * slots + k], bep_mul(vub, fact));
            r.column_meso_x[out + 5ul * slots + k] = bep_add(
                r.column_meso_x[out + 5ul * slots + k], bep_mul(vvb, fact));
            r.column_meso_x[out + 7ul * slots + k] = bep_add(
                r.column_meso_x[out + 7ul * slots + k], bep_mul(veb, fact));
        }
    }
    r.column_meso_ix[col] = 0;
}

// Integrated source-ordered flat/wall flux helpers.
static inline float urban_bep_sin(float x) {
    return precise::sin(x);
}
static inline float urban_bep_cos(float x) {
    return precise::cos(x);
}

static inline void urban_bep_flux_wall(
        float ua, float va, float pt, float da, float ptw, float drst,
        float dt,
        thread float &uva, thread float &vva,
        thread float &uvb, thread float &vvb,
        thread float &tva, thread float &tvb, thread float &evb) {
    constexpr float cdrag=0.4f, cp_u=1004.0f;
    const float sinr=urban_bep_sin(drst), cosr=urban_bep_cos(drst);
    const float speed=as_type<float>(soft_gfk_pow_bits(
        as_type<uint>(bep_add(bep_mul(ua,ua),bep_mul(va,va))),
        as_type<uint>(0.5f)));
    const float ort=fabs(bep_sub(bep_mul(cosr,ua),bep_mul(sinr,va)));
    uva=bep_mul(bep_mul(bep_div(bep_mul(-cdrag,ort),2.0f),cosr),cosr);
    vva=bep_mul(bep_mul(bep_div(bep_mul(-cdrag,ort),2.0f),sinr),sinr);
    uvb=bep_mul(bep_mul(bep_mul(bep_div(bep_mul(cdrag,ort),2.0f),sinr),cosr),va);
    vvb=bep_mul(bep_mul(bep_mul(bep_div(bep_mul(cdrag,ort),2.0f),sinr),cosr),ua);
    float hc=bep_mul(5.678f,bep_add(1.09f,bep_mul(0.23f,bep_div(speed,0.3048f))));
    hc=min(hc,bep_div(bep_mul(da,cp_u),dt));
    tvb=bep_sub(bep_div(bep_div(bep_mul(hc,ptw),da),cp_u),
                bep_mul(bep_div(bep_div(hc,da),cp_u),pt));
    tva=0.0f;
    evb=bep_div(bep_mul(cdrag,as_type<float>(soft_gfk_pow_bits(
        as_type<uint>(ort),as_type<uint>(3.0f)))),2.0f);
}

static inline void urban_bep_flux_flat(
        float dz, float z0, float ua, float va, float pt, float pt0,
        float ptg, thread float &uhb, thread float &vhb,
        thread float &thb, thread float &ehb) {
    constexpr float g_u=9.81f, vk=0.40f;
    constexpr float b=9.4f, cm=7.4f, ch=5.3f, rr=0.74f;
    float utot=as_type<float>(soft_gfk_pow_bits(
        as_type<uint>(bep_add(bep_mul(ua,ua),bep_mul(va,va))),
        as_type<uint>(0.5f)));
    const float zz=bep_mul(dz,0.5f);
    utot=max(utot,0.01f);
    const float ric=bep_div(bep_mul(bep_mul(bep_mul(2.0f,g_u),zz),
        bep_sub(pt,ptg)),bep_mul(bep_add(pt,ptg),bep_mul(utot,utot)));
    const float aa=bep_div(vk,bep_log(bep_div(zz,z0)));
    float fm,fh,c;
    if(ric>0.0f) {
        const float br=bep_mul(0.5f,b);
        fm=bep_div(1.0f,bep_mul(bep_add(1.0f,bep_mul(br,ric)),
                                 bep_add(1.0f,bep_mul(br,ric))));
        fh=fm;
    } else {
        c=bep_mul(bep_mul(bep_mul(b,cm),bep_mul(aa,aa)),
            as_type<float>(soft_gfk_pow_bits(as_type<uint>(bep_div(zz,z0)),
                                             as_type<uint>(0.5f))));
        const float root=as_type<float>(soft_gfk_pow_bits(
            as_type<uint>(-ric),as_type<uint>(0.5f)));
        fm=bep_sub(1.0f,bep_div(bep_mul(b,ric),bep_add(1.0f,bep_mul(c,root))));
        c=bep_div(bep_mul(c,ch),cm);
        fh=bep_sub(1.0f,bep_div(bep_mul(b,ric),bep_add(1.0f,bep_mul(c,root))));
    }
    const float fbuw=bep_mul(bep_mul(bep_mul(bep_mul(-aa,aa),utot),utot),fm);
    const float fbpt=bep_div(bep_mul(bep_mul(bep_mul(bep_mul(-aa,aa),utot),
        bep_sub(pt,ptg)),fh),rr);
    const float ustar=as_type<float>(soft_gfk_pow_bits(
        as_type<uint>(-fbuw),as_type<uint>(0.5f)));
    const float tstar=bep_div(-fbpt,ustar);
    const float buu=bep_mul(bep_mul(bep_div(-g_u,pt0),ustar),tstar);
    uhb=bep_div(bep_mul(bep_mul(-ustar,ustar),ua),utot);
    vhb=bep_div(bep_mul(bep_mul(-ustar,ustar),va),utot);
    thb=bep_mul(-ustar,tstar);
    ehb=buu;
}

// One serial source-order building exchange column per Metal thread.
kernel void urban_bep_column_buildings_fragment(
    constant UrbanBepColumnBuildingsFragmentResources &r [[buffer(0)]],
    constant UrbanBepColumnBuildingsFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong pin=ulong(col)*140ul;
    const ulong pix=ulong(col)*3ul;
    const int nz=r.column_param_ix[pix];
    const int nd=r.column_param_ix[pix+1ul];
    if (r.column_param_ix[pix+2ul] != 0 || nz < 1 || nz > 18 || nd < 1 || nd > 2) return;
    const ulong interp=ulong(col)*108ul;
    const ulong surface=ulong(col)*330ul;
    for (int direction=0; direction<nd; ++direction) {
        const ulong out_flat=ulong(col)*648ul;
        const ulong flat_out=out_flat+504ul+ulong(direction)*18ul;
        float uh=0.0f,vh=0.0f,th=0.0f,eh=0.0f;
        urban_bep_flux_flat(5.0f,r.column_param[pin+60ul+ulong(direction)],
            r.column_interp_x[interp],r.column_interp_x[interp+18ul],
            r.column_interp_x[interp+36ul],r.column_interp_x[interp+54ul],
            r.surface_x[surface+ulong(direction)],uh,vh,th,eh);
        r.column_building_x[flat_out]=uh;
        r.column_building_x[flat_out+36ul]=vh;
        r.column_building_x[flat_out+72ul]=th;
        r.column_building_x[flat_out+108ul]=eh;
        for (int k=1;k<nz;++k) {
            uh=vh=th=eh=0.0f;
            if (r.column_param[pin+104ul+ulong(k+1)]>0.0f) {
                urban_bep_flux_flat(5.0f,
                    r.column_param[pin+60ul+ulong(direction)+2ul*ulong(k+1)],
                    r.column_interp_x[interp+ulong(k)],
                    r.column_interp_x[interp+18ul+ulong(k)],
                    r.column_interp_x[interp+36ul+ulong(k)],
                    r.column_interp_x[interp+54ul+ulong(k)],
                    r.surface_x[surface+2ul+ulong(direction)*18ul+ulong(k+1)],
                    uh,vh,th,eh);
            }
            r.column_building_x[flat_out+ulong(k)]=uh;
            r.column_building_x[flat_out+36ul+ulong(k)]=vh;
            r.column_building_x[flat_out+72ul+ulong(k)]=th;
            r.column_building_x[flat_out+108ul+ulong(k)]=eh;
        }
        for (int side=0;side<2;++side) {
            const int wall_index=2*direction+side;
            for (int k=0;k<nz;++k) {
            const ulong wall0=out_flat+ulong(wall_index)*18ul+ulong(k);
            float uva=0.0f,vva=0.0f,uvb=0.0f,vvb=0.0f;
            float tva=0.0f,tvb=0.0f,evb=0.0f;
                urban_bep_flux_wall(r.column_interp_x[interp+ulong(k)],
                    r.column_interp_x[interp+18ul+ulong(k)],
                    r.column_interp_x[interp+36ul+ulong(k)],
                    r.column_interp_x[interp+90ul+ulong(k)],
                    r.surface_x[surface+38ul+ulong(wall_index)*18ul+ulong(k)],
                    r.column_param[pin+100ul+2ul*ulong(direction)],p.dt,
                uva,vva,uvb,vvb,tva,tvb,evb);
            r.column_building_x[wall0]=uva;
            r.column_building_x[wall0+72ul]=vva;
            r.column_building_x[wall0+144ul]=uvb;
            r.column_building_x[wall0+216ul]=vvb;
            r.column_building_x[wall0+288ul]=tva;
            r.column_building_x[wall0+360ul]=tvb;
            r.column_building_x[wall0+432ul]=evb;
            }
        }
    }
}
