#include <metal_stdlib>
#include "rrtmg_legacy_lw_inatm_bindings.metalh"
#include "rrtmg_legacy_lw_planck_bindings.metalh"
#include "rrtmg_legacy_lw_taugb2_bindings.metalh"
#include "rrtmg_legacy_lw_taugb10_bindings.metalh"
#include "rrtmg_legacy_lw_taugb11_bindings.metalh"
#include "rrtmg_legacy_lw_taugb12_bindings.metalh"
#include "rrtmg_legacy_lw_taugb1_bindings.metalh"
#include "rrtmg_legacy_lw_taugb4_bindings.metalh"
#include "rrtmg_legacy_lw_taugb3_bindings.metalh"
#include "rrtmg_legacy_lw_taugb5_bindings.metalh"
#include "rrtmg_legacy_lw_taugb6_bindings.metalh"
#include "rrtmg_legacy_lw_taugb7_bindings.metalh"
#include "rrtmg_legacy_lw_taugb8_bindings.metalh"
#include "rrtmg_legacy_lw_taugb9_bindings.metalh"
#include "rrtmg_legacy_lw_taugb14_bindings.metalh"
#include "rrtmg_legacy_lw_taugb15_bindings.metalh"
#include "rrtmg_legacy_lw_taugb13_bindings.metalh"
#include "rrtmg_legacy_lw_taugb16_bindings.metalh"
#include "rrtmg_legacy_lw_rtrn_secdiff_bindings.metalh"
#include "rrtmg_legacy_lw_rtrn_prol_bindings.metalh"
#include "rrtmg_legacy_lw_rtrn_accum_bindings.metalh"
#include "rrtmg_legacy_lw_rtrn_final_bindings.metalh"
#include "rrtmg_legacy_lw_rtrn_march_bindings.metalh"
#include "rrtmg_legacy_lw_cldprmc_bindings.metalh"
#include "rrtmg_legacy_lw_setcoef_indices_bindings.metalh"
#include "rrtmg_legacy_lw_setcoef_gas_bindings.metalh"
#include "soft_gfk_log.metalh"

using namespace metal;


struct RrtmgLegacySpec {
    float comb;
    float parm;
    int j;
    float f;
};
struct RrtmgLegacyFac {
    int branch;
    float f0p, f1p, f2p, f0t, f1t, f2t;
};
inline RrtmgLegacySpec rrtmg_legacy_spec(float col1, float rat, float col2,
                                          float oneminus, float mult) {
    RrtmgLegacySpec s;
    s.comb = col1 + rat * col2;
    s.parm = col1 / s.comb;
    if (s.parm >= oneminus) s.parm = oneminus;
    const float specmult = mult * s.parm;
    s.j = 1 + int(specmult);
    s.f = fmod(specmult, 1.0f);
    return s;
}
inline RrtmgLegacyFac rrtmg_legacy_fac(float parm, float sf,
                                       float facp, float fact) {
    RrtmgLegacyFac r;
    if (parm < 0.125f) {
        const float x = sf - 1.0f;
        const float x2 = x * x;
        const float x4 = x2 * x2;
        const float fk0 = x4;
        const float fk1 = (1.0f - x) - 2.0f * x4;
        const float fk2 = x + x4;
        r.branch = 0;
        r.f0p = fk0 * facp; r.f1p = fk1 * facp; r.f2p = fk2 * facp;
        r.f0t = fk0 * fact; r.f1t = fk1 * fact; r.f2t = fk2 * fact;
    } else if (parm > 0.875f) {
        const float x = -sf;
        const float x2 = x * x;
        const float x4 = x2 * x2;
        const float fk0 = x4;
        const float fk1 = (1.0f - x) - 2.0f * x4;
        const float fk2 = x + x4;
        r.branch = 1;
        r.f0p = fk0 * facp; r.f1p = fk1 * facp; r.f2p = fk2 * facp;
        r.f0t = fk0 * fact; r.f1t = fk1 * fact; r.f2t = fk2 * fact;
    } else {
        r.branch = 2;
        r.f0p = (1.0f - sf) * facp; r.f0t = (1.0f - sf) * fact;
        r.f1p = sf * facp; r.f1t = sf * fact;
        r.f2p = 0.0f; r.f2t = 0.0f;
    }
    return r;
}
inline float rrtmg_legacy_major(RrtmgLegacyFac f, device const float *absa,
                                 int ind, int ig, float comb) {
    const ulong base = ulong(ind - 1) + 585ul * ulong(ig - 1);
    float t;
    if (f.branch == 0) {
        t = f.f0p * absa[base];
        t = t + f.f1p * absa[base + 1ul];
        t = t + f.f2p * absa[base + 2ul];
        t = t + f.f0t * absa[base + 9ul];
        t = t + f.f1t * absa[base + 10ul];
        t = t + f.f2t * absa[base + 11ul];
    } else if (f.branch == 1) {
        t = f.f2p * absa[base - 1ul];
        t = t + f.f1p * absa[base];
        t = t + f.f0p * absa[base + 1ul];
        t = t + f.f2t * absa[base + 8ul];
        t = t + f.f1t * absa[base + 9ul];
        t = t + f.f0t * absa[base + 10ul];
    } else {
        t = f.f0p * absa[base];
        t = t + f.f1p * absa[base + 1ul];
        t = t + f.f0t * absa[base + 9ul];
        t = t + f.f1t * absa[base + 10ul];
    }
    return comb * t;
}
inline float rrtmg_legacy_bmajor(float f0p, float f1p, float f0t, float f1t,
                                  device const float *absb, int ind, int ig,
                                  float comb) {
    const ulong base = ulong(ind - 1) + 1175ul * ulong(ig - 1);
    float t = f0p * absb[base];
    t = t + f1p * absb[base + 1ul];
    t = t + f0t * absb[base + 5ul];
    t = t + f1t * absb[base + 6ul];
    return comb * t;
}

#pragma clang fp contract(off)

// Port of rlw_inatm in woof/core/kernels/rrtmg_lw_chain.cu. Layout and
// operation order mirror the CUDA launcher and the NumPy reference.
kernel void rrtmg_legacy_lw_inatm(
    constant RrtmgLegacyLwInatmResources &r [[buffer(0)]],
    constant RrtmgLegacyLwInatmParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const int nl = p.nl;
    const ulong col_base = ulong(col) * ulong(nl);
    const ulong wkl_base = ulong(col) * 35ul * ulong(nl);
    const ulong wx_base = ulong(col) * 4ul * ulong(nl);
    const ulong lev_base = ulong(col) * ulong(nl + 1);
    const float amd = 28.9660f;
    const float amw = 18.0160f;

    for (int l = 1; l <= nl; ++l) {
        const ulong i = col_base + ulong(l - 1);
        const float w1 = r.wkl[wkl_base + ulong(l - 1)];
        const float amm = (1.0f - w1) * amd + w1 * amw;
        const float num = (r.plev[lev_base + ulong(l - 1)]
                         - r.plev[lev_base + ulong(l)]) * 1.0e3f * p.avogad;
        const float den = 1.0e2f * p.grav * amm * (1.0f + w1);
        r.coldry[i] = num / den;
    }

    float amttl = 0.0f;
    float wvttl = 0.0f;
    for (int l = 1; l <= nl; ++l) {
        const ulong i = col_base + ulong(l - 1);
        const float cd = r.coldry[i];
        float summol = 0.0f;
        for (int imol = 2; imol <= 7; ++imol)
            summol = summol + r.wkl[wkl_base + ulong(imol - 1) * ulong(nl)
                                    + ulong(l - 1)];
        r.wbrodl[i] = cd * (1.0f - summol);
        for (int imol = 1; imol <= 7; ++imol) {
            const ulong j = wkl_base + ulong(imol - 1) * ulong(nl)
                          + ulong(l - 1);
            r.wkl[j] = cd * r.wkl[j];
        }
        const float h2o = r.wkl[wkl_base + ulong(l - 1)];
        amttl = (amttl + cd) + h2o;
        wvttl = wvttl + h2o;
        for (int ix = 1; ix <= 4; ++ix) {
            const ulong j = wx_base + ulong(ix - 1) * ulong(nl)
                          + ulong(l - 1);
            r.wx[j] = (cd * r.wx[j]) * 1.0e-20f;
        }
    }
    const float wvsh = (amw * wvttl) / (amd * amttl);
    r.pwvcm[col] = (wvsh * (1.0e3f * r.plev[lev_base]))
                 / (1.0e2f * p.grav);
}

// Planck interpolation from rlw_setcoef in woof/core/kernels/rrtmg_lw.cu.
kernel void rrtmg_legacy_lw_planck(
    constant RrtmgLegacyLwPlanckResources &r [[buffer(0)]],
    constant RrtmgLegacyLwPlanckParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const int nl = p.nl;
    const ulong colnl = ulong(col) * ulong(nl);
    const ulong collev = ulong(col) * ulong(nl + 1);
    const ulong colband = ulong(col) * 16ul;
    const float tbound = r.tbound[col];
    const int indbound = clamp(int(tbound - 159.0f), 1, 180);
    const float tbndfrac = (tbound - 159.0f) - float(indbound);
    const int indlev0 = clamp(int(r.tz[collev] - 159.0f), 1, 180);
    const float t0frac = (r.tz[collev] - 159.0f) - float(indlev0);

    for (int lay = 1; lay <= nl; ++lay) {
        const float tav = r.tavel[colnl + ulong(lay - 1)];
        const int indlay = clamp(int(tav - 159.0f), 1, 180);
        const float tlayfrac = (tav - 159.0f) - float(indlay);
        const float tzl = r.tz[collev + ulong(lay)];
        const int indlev = clamp(int(tzl - 159.0f), 1, 180);
        const float tlevfrac = (tzl - 159.0f) - float(indlev);

        for (int ib = 1; ib <= 15; ++ib) {
            const ulong table = 181ul * ulong(ib - 1);
            if (lay == 1) {
                const float dbdtlev = r.totplnk[table + ulong(indbound)]
                                    - r.totplnk[table + ulong(indbound - 1)];
                r.plankbnd[colband + ulong(ib - 1)] = r.sem[ colband + ulong(ib - 1)]
                    * (r.totplnk[table + ulong(indbound - 1)]
                       + tbndfrac * dbdtlev);
                const float dbdtlev0 = r.totplnk[table + ulong(indlev0)]
                                     - r.totplnk[table + ulong(indlev0 - 1)];
                r.planklev[collev * 16ul + ulong(ib - 1)] =
                    r.totplnk[table + ulong(indlev0 - 1)] + t0frac * dbdtlev0;
            }
            const float dbdtlev = r.totplnk[table + ulong(indlev)]
                                - r.totplnk[table + ulong(indlev - 1)];
            const float dbdtlay = r.totplnk[table + ulong(indlay)]
                                - r.totplnk[table + ulong(indlay - 1)];
            r.planklay[(colnl + ulong(lay - 1)) * 16ul + ulong(ib - 1)] =
                r.totplnk[table + ulong(indlay - 1)] + tlayfrac * dbdtlay;
            r.planklev[(collev + ulong(lay)) * 16ul + ulong(ib - 1)] =
                r.totplnk[table + ulong(indlev - 1)] + tlevfrac * dbdtlev;
        }

        const int ib = 16;
        if (p.istart == 16) {
            if (lay == 1) {
                const float dbdtlev = r.totplk16[indbound]
                                    - r.totplk16[indbound - 1];
                r.plankbnd[colband + 15ul] = r.sem[colband + 15ul]
                    * (r.totplk16[indbound - 1] + tbndfrac * dbdtlev);
                const ulong table = 181ul * 15ul;
                const float dbdtlev0 = r.totplnk[table + ulong(indlev0)]
                                     - r.totplnk[table + ulong(indlev0 - 1)];
                r.planklev[collev * 16ul + 15ul] =
                    r.totplnk[table + ulong(indlev0 - 1)] + t0frac * dbdtlev0;
            }
            const float dbdtlev = r.totplk16[indlev] - r.totplk16[indlev - 1];
            const float dbdtlay = r.totplk16[indlay] - r.totplk16[indlay - 1];
            r.planklay[(colnl + ulong(lay - 1)) * 16ul + 15ul] =
                r.totplk16[indlay - 1] + tlayfrac * dbdtlay;
            r.planklev[(collev + ulong(lay)) * 16ul + 15ul] =
                r.totplk16[indlev - 1] + tlevfrac * dbdtlev;
        } else {
            const ulong table = 181ul * 15ul;
            if (lay == 1) {
                const float dbdtlev = r.totplnk[table + ulong(indbound)]
                                    - r.totplnk[table + ulong(indbound - 1)];
                r.plankbnd[colband + 15ul] = r.sem[colband + 15ul]
                    * (r.totplnk[table + ulong(indbound - 1)]
                       + tbndfrac * dbdtlev);
                const float dbdtlev0 = r.totplnk[table + ulong(indlev0)]
                                     - r.totplnk[table + ulong(indlev0 - 1)];
                r.planklev[collev * 16ul + 15ul] =
                    r.totplnk[table + ulong(indlev0 - 1)] + t0frac * dbdtlev0;
            }
            const float dbdtlev = r.totplnk[table + ulong(indlev)]
                                - r.totplnk[table + ulong(indlev - 1)];
            const float dbdtlay = r.totplnk[table + ulong(indlay)]
                                - r.totplnk[table + ulong(indlay - 1)];
            r.planklay[(colnl + ulong(lay - 1)) * 16ul + 15ul] =
                r.totplnk[table + ulong(indlay - 1)] + tlayfrac * dbdtlay;
            r.planklev[(collev + ulong(lay)) * 16ul + 15ul] =
                r.totplnk[table + ulong(indlev - 1)] + tlevfrac * dbdtlev;
        }
    }
}

#include "rrtmg_legacy_lw_taut_bindings.metalh"

// Port of rlw_taut in woof/core/kernels/rrtmg_lw_chain.cu.
kernel void rrtmg_legacy_lw_taut(
    constant RrtmgLegacyLwTautResources &r [[buffer(0)]],
    constant RrtmgLegacyLwTautParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong total = ulong(p.ncol) * ulong(p.nl) * ulong(p.ngpt);
    if (ulong(tid) >= total) return;
    const int ig = int(tid % uint(p.ngpt));
    const ulong collay = ulong(tid) / ulong(p.ngpt);
    r.taut[tid] = r.taug[tid]
        + r.taua[collay * ulong(p.nbnd) + ulong(r.ngb[ig] - 1)];
}

static inline float rrtmg_legacy_lw_logf(float x) {
    uint ix = as_type<uint>(x);
    if (ix == 0x3f800000u) return 0.0f;
    if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {
        if (ix * 2u == 0u) return as_type<float>(0xff800000u);
        if (ix == 0x7f800000u) return x;
        if ((ix & 0x80000000u) != 0u || ix * 2u >= 0xff000000u)
            return as_type<float>(0x7fc00000u);
        ix = as_type<uint>(x * 8388608.0f);
        ix -= 23u << 23;
    }
    const uint tmp = ix - 0x3f330000u;
    const uint i = (tmp >> 19) & 15u;
    const int k = int(tmp) >> 23;
    const uint iz = ix - (tmp & 0xff800000u);
    const ulong z = soft64_from_f32_bits(iz);
    const ulong r = soft64_sub(
        soft64_mul(z, SOFT_GFK_LOGF_INVC[i]), 0x3ff0000000000000ul);
    const ulong y0 = soft64_add(
        SOFT_GFK_LOGF_LOGC[i],
        soft64_mul(soft_gfk_log_i32_to_f64(k), SOFT_GFK_LOGF_LN2));
    const ulong r2 = soft64_mul(r, r);
    ulong y = soft64_add(soft64_mul(SOFT_GFK_LOGF_A1, r),
                         SOFT_GFK_LOGF_A2);
    y = soft64_add(soft64_mul(SOFT_GFK_LOGF_A0, r2), y);
    y = soft64_add(soft64_mul(y, r2), soft64_add(y0, r));
    return as_type<float>(soft64_to_f32_bits(y));
}

// RRTMG setcoef's pressure/temperature interpolation coordinates. This is
// split from the remaining coefficient preparation so the typed Metal ABI
// has a compact, auditable input/output surface.
kernel void rrtmg_legacy_lw_setcoef_indices(
    constant RrtmgLegacyLwSetcoefIndicesResources &r [[buffer(0)]],
    constant RrtmgLegacyLwSetcoefIndicesParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong base = ulong(col) * ulong(p.nl);
    int laytrop = 0;
    for (int lay = 0; lay < p.nl; ++lay) {
        float pav = r.pavel[base + ulong(lay)];
        if (!(pav > 0.0f)) pav = 1.0e-6f;
        const float plog = rrtmg_legacy_lw_logf(pav);
        int jp = int(trunc(36.0f - 5.0f * (plog + 0.04f)));
        jp = clamp(jp, 1, 58);
        const float fp = 5.0f * (r.preflog[jp - 1] - plog);
        int jt = int(trunc(3.0f + (r.tavel[base + ulong(lay)]
                                   - r.tref[jp - 1]) / 15.0f));
        jt = clamp(jt, 1, 4);
        const float ft = (r.tavel[base + ulong(lay)] - r.tref[jp - 1])
                       / 15.0f - float(jt - 3);
        int jt1 = int(trunc(3.0f + (r.tavel[base + ulong(lay)]
                                    - r.tref[jp]) / 15.0f));
        jt1 = clamp(jt1, 1, 4);
        const float ft1 = (r.tavel[base + ulong(lay)] - r.tref[jp])
                        / 15.0f - float(jt1 - 3);
        r.jp[base + ulong(lay)] = jp;
        r.jt[base + ulong(lay)] = jt;
        r.jt1[base + ulong(lay)] = jt1;
        r.fp[base + ulong(lay)] = fp;
        r.ft[base + ulong(lay)] = ft;
        r.ft1[base + ulong(lay)] = ft1;
        if (!(plog <= 4.56f)) ++laytrop;
    }
    r.laytrop[col] = laytrop;
}

// RRTMG SETCOEF's per-layer gas columns and interpolation factors. The
// CUDA host packs the named arrays into frozen FS/IS slabs before TAUMOL.
#define RRTMG_LW_FS(slot, col, lay) \
    r.fs[(ulong(slot) * ulong(p.ncol) + ulong(col)) * ulong(p.nl) + ulong(lay)]
#define RRTMG_LW_IS(slot, col, lay) \
    r.isv[(ulong(slot) * ulong(p.ncol) + ulong(col)) * ulong(p.nl) + ulong(lay)]
#define RRTMG_LW_WKL(mol, col, lay) \
    r.wkl[(ulong(col) * 35ul + ulong(mol)) * ulong(p.nl) + ulong(lay)]
#define RRTMG_LW_CHI(species, row) \
    r.chi_mls[ulong(species) + 7ul * ulong(row)]

kernel void rrtmg_legacy_lw_setcoef_gas(
    constant RrtmgLegacyLwSetcoefGasResources &r [[buffer(0)]],
    constant RrtmgLegacyLwSetcoefGasParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong n = ulong(p.ncol) * ulong(p.nl);
    if (ulong(gid) >= n) return;
    const int col = int(gid) / p.nl;
    const int lay = int(gid) % p.nl;
    const ulong i = ulong(gid);
    const float pav = r.pavel[i];
    const float tav = r.tavel[i];
    const float cdry = r.coldry[i];
    const float water = RRTMG_LW_WKL(0, col, lay) / cdry;
    const float stpfac = 296.0f / 1013.0f;
    const float scalefac = (pav * stpfac) / tav;
    const int jp = r.jp[i];
    const int jt = r.jt[i];
    const int jt1 = r.jt1[i];
    const float fp = r.fp[i];
    const float ft = r.ft[i];
    const float ft1 = r.ft1[i];
    const float plog = rrtmg_legacy_lw_logf(pav > 0.0f ? pav : 1.0e-6f);
    const int upper = !(plog <= 4.56f);

    RRTMG_LW_FS(0, col, lay) = pav;
    RRTMG_LW_FS(1, col, lay) = cdry;
    RRTMG_LW_FS(2, col, lay) = 1.0e-20f * RRTMG_LW_WKL(0, col, lay);
    RRTMG_LW_FS(3, col, lay) = 1.0e-20f * RRTMG_LW_WKL(1, col, lay);
    RRTMG_LW_FS(4, col, lay) = 1.0e-20f * RRTMG_LW_WKL(2, col, lay);
    RRTMG_LW_FS(5, col, lay) = 1.0e-20f * RRTMG_LW_WKL(3, col, lay);
    RRTMG_LW_FS(6, col, lay) = 1.0e-20f * RRTMG_LW_WKL(4, col, lay);
    RRTMG_LW_FS(7, col, lay) = 1.0e-20f * RRTMG_LW_WKL(5, col, lay);
    RRTMG_LW_FS(8, col, lay) = 1.0e-20f * RRTMG_LW_WKL(6, col, lay);
    const float floor = 1.0e-32f * cdry;
    if (RRTMG_LW_FS(3, col, lay) == 0.0f) RRTMG_LW_FS(3, col, lay) = floor;
    if (RRTMG_LW_FS(4, col, lay) == 0.0f) RRTMG_LW_FS(4, col, lay) = floor;
    if (RRTMG_LW_FS(5, col, lay) == 0.0f) RRTMG_LW_FS(5, col, lay) = floor;
    if (RRTMG_LW_FS(6, col, lay) == 0.0f) RRTMG_LW_FS(6, col, lay) = floor;
    if (RRTMG_LW_FS(7, col, lay) == 0.0f) RRTMG_LW_FS(7, col, lay) = floor;
    RRTMG_LW_FS(9, col, lay) = 1.0e-20f * r.wbrodl[i];

    const float compfp = 1.0f - fp;
    RRTMG_LW_FS(10, col, lay) = compfp * (1.0f - ft);
    RRTMG_LW_FS(11, col, lay) = fp * (1.0f - ft1);
    RRTMG_LW_FS(12, col, lay) = compfp * ft;
    RRTMG_LW_FS(13, col, lay) = fp * ft1;

    RRTMG_LW_FS(26, col, lay) = RRTMG_LW_FS(2, col, lay)
        * (water * (scalefac / (1.0f + water)));
    RRTMG_LW_FS(28, col, lay) = RRTMG_LW_FS(2, col, lay)
        * (scalefac / (1.0f + water));
    float factor = 0.0f;
    int indfor = 0;
    if (upper) {
        factor = (332.0f - tav) / 36.0f;
        indfor = clamp(int(factor), 1, 2);
        RRTMG_LW_FS(29, col, lay) = factor - float(indfor);
        factor = (tav - 188.0f) / 7.2f;
        const int indself = clamp(int(factor) - 7, 1, 9);
        RRTMG_LW_IS(3, col, lay) = indself;
        RRTMG_LW_FS(27, col, lay) = factor - float(indself + 7);
        RRTMG_LW_FS(31, col, lay) = pav / tav;
        RRTMG_LW_FS(32, col, lay) = (pav / tav)
            * (r.wbrodl[i] / (cdry + RRTMG_LW_WKL(0, col, lay)));
        factor = (tav - 180.8f) / 7.2f;
        const int indminor = clamp(int(factor), 1, 18);
        RRTMG_LW_IS(5, col, lay) = indminor;
        RRTMG_LW_FS(30, col, lay) = factor - float(indminor);
        RRTMG_LW_FS(14, col, lay) = RRTMG_LW_CHI(0, jp - 1) / RRTMG_LW_CHI(1, jp - 1);
        RRTMG_LW_FS(15, col, lay) = RRTMG_LW_CHI(0, jp) / RRTMG_LW_CHI(1, jp);
        RRTMG_LW_FS(16, col, lay) = RRTMG_LW_CHI(0, jp - 1) / RRTMG_LW_CHI(2, jp - 1);
        RRTMG_LW_FS(17, col, lay) = RRTMG_LW_CHI(0, jp) / RRTMG_LW_CHI(2, jp);
        RRTMG_LW_FS(18, col, lay) = RRTMG_LW_CHI(0, jp - 1) / RRTMG_LW_CHI(3, jp - 1);
        RRTMG_LW_FS(19, col, lay) = RRTMG_LW_CHI(0, jp) / RRTMG_LW_CHI(3, jp);
        RRTMG_LW_FS(20, col, lay) = RRTMG_LW_CHI(0, jp - 1) / RRTMG_LW_CHI(5, jp - 1);
        RRTMG_LW_FS(21, col, lay) = RRTMG_LW_CHI(0, jp) / RRTMG_LW_CHI(5, jp);
        RRTMG_LW_FS(22, col, lay) = RRTMG_LW_CHI(3, jp - 1) / RRTMG_LW_CHI(1, jp - 1);
        RRTMG_LW_FS(23, col, lay) = RRTMG_LW_CHI(3, jp) / RRTMG_LW_CHI(1, jp);
        RRTMG_LW_FS(24, col, lay) = 0.0f;
        RRTMG_LW_FS(25, col, lay) = 0.0f;
    } else {
        factor = (tav - 188.0f) / 36.0f;
        indfor = 3;
        RRTMG_LW_FS(29, col, lay) = factor - 1.0f;
        RRTMG_LW_FS(31, col, lay) = pav / tav;
        RRTMG_LW_FS(32, col, lay) = (pav / tav)
            * (r.wbrodl[i] / (cdry + RRTMG_LW_WKL(0, col, lay)));
        factor = (tav - 180.8f) / 7.2f;
        const int indminor = clamp(int(factor), 1, 18);
        RRTMG_LW_IS(5, col, lay) = indminor;
        RRTMG_LW_FS(30, col, lay) = factor - float(indminor);
        RRTMG_LW_FS(14, col, lay) = RRTMG_LW_CHI(0, jp - 1) / RRTMG_LW_CHI(1, jp - 1);
        RRTMG_LW_FS(15, col, lay) = RRTMG_LW_CHI(0, jp) / RRTMG_LW_CHI(1, jp);
        RRTMG_LW_FS(16, col, lay) = 0.0f;
        RRTMG_LW_FS(17, col, lay) = 0.0f;
        RRTMG_LW_FS(18, col, lay) = 0.0f;
        RRTMG_LW_FS(19, col, lay) = 0.0f;
        RRTMG_LW_FS(20, col, lay) = 0.0f;
        RRTMG_LW_FS(21, col, lay) = 0.0f;
        RRTMG_LW_FS(22, col, lay) = 0.0f;
        RRTMG_LW_FS(23, col, lay) = 0.0f;
        RRTMG_LW_FS(24, col, lay) = RRTMG_LW_CHI(2, jp - 1) / RRTMG_LW_CHI(1, jp - 1);
        RRTMG_LW_FS(25, col, lay) = RRTMG_LW_CHI(2, jp) / RRTMG_LW_CHI(1, jp);
        RRTMG_LW_IS(3, col, lay) = 0;
        RRTMG_LW_FS(27, col, lay) = 0.0f;
    }
    RRTMG_LW_IS(0, col, lay) = jp;
    RRTMG_LW_IS(1, col, lay) = jt;
    RRTMG_LW_IS(2, col, lay) = jt1;
    RRTMG_LW_IS(4, col, lay) = indfor;
}
#undef RRTMG_LW_FS
#undef RRTMG_LW_IS
#undef RRTMG_LW_WKL
#undef RRTMG_LW_CHI


// Band 2 optical depth from rlw_taugb2 in
// woof/core/kernels/rrtmg_lw_taugb02_10_11_12.cu.
kernel void rrtmg_legacy_lw_taugb2(
    constant RrtmgLegacyLwTaugb2Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb2Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nitems = uint(p.ncol * p.nl);
    if (tid >= nitems) return;
    const int ncol = p.ncol;
    const int nl = p.nl;
    const int col = int(tid) / nl;
    const int lay = int(tid) % nl + 1;
    const int laytrop = r.laytrop[col];
    const ulong cell = ulong(col) * ulong(nl) + ulong(lay - 1);
    const ulong fsbase = cell;
    const ulong isbase = cell;
    const int jp = r.isv[0ul * ulong(ncol) * ulong(nl) + isbase];
    const int jt = r.isv[1ul * ulong(ncol) * ulong(nl) + isbase];
    const int jt1 = r.isv[2ul * ulong(ncol) * ulong(nl) + isbase];
    const int indself = r.isv[3ul * ulong(ncol) * ulong(nl) + isbase];
    const int indfor = r.isv[4ul * ulong(ncol) * ulong(nl) + isbase];
    const ulong tabarea_a = 65ul;
    const ulong tabarea_b = 235ul;
    const ulong tabarea_self = 10ul;
    const ulong tabarea_for = 4ul;
    const int ng = 12;
    const int gs = 10;
    const float colh2o = r.fs[2ul * ulong(ncol) * ulong(nl) + fsbase];
    const float fac00 = r.fs[10ul * ulong(ncol) * ulong(nl) + fsbase];
    const float fac01 = r.fs[11ul * ulong(ncol) * ulong(nl) + fsbase];
    const float fac10 = r.fs[12ul * ulong(ncol) * ulong(nl) + fsbase];
    const float fac11 = r.fs[13ul * ulong(ncol) * ulong(nl) + fsbase];
    const float selffac = r.fs[26ul * ulong(ncol) * ulong(nl) + fsbase];
    const float selffrac = r.fs[27ul * ulong(ncol) * ulong(nl) + fsbase];
    const float forfac = r.fs[28ul * ulong(ncol) * ulong(nl) + fsbase];
    const float forfrac = r.fs[29ul * ulong(ncol) * ulong(nl) + fsbase];
    const float pp = r.fs[0ul * ulong(ncol) * ulong(nl) + fsbase];
    const ulong outbase = cell * 140ul + ulong(gs);

    if (lay <= laytrop) {
        const int ind0 = ((jp - 1) * 5 + (jt - 1)) + 1;
        const int ind1 = (jp * 5 + (jt1 - 1)) + 1;
        const float corradj = 1.0f - (0.05f * (pp - 100.0f)) / 900.0f;
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            const float self0 = r.selfref[ulong(indself - 1) + g * tabarea_self];
            const float self1 = r.selfref[ulong(indself) + g * tabarea_self];
            const float for0 = r.forref[ulong(indfor - 1) + g * tabarea_for];
            const float for1 = r.forref[ulong(indfor) + g * tabarea_for];
            const ulong ia0 = ulong(ind0 - 1) + g * tabarea_a;
            const ulong ia1 = ulong(ind1 - 1) + g * tabarea_a;
            float tauself = selffac * (self0 + selffrac * (self1 - self0));
            const float taufor = forfac * (for0 + forfrac * (for1 - for0));
            float tmaj = fac00 * r.absa[ia0];
            tmaj = tmaj + fac10 * r.absa[ia0 + 1ul];
            tmaj = tmaj + fac01 * r.absa[ia1];
            tmaj = tmaj + fac11 * r.absa[ia1 + 1ul];
            float tau = colh2o * tmaj;
            tau = tau + tauself;
            tau = tau + taufor;
            r.taug[outbase + g] = corradj * tau;
            r.fracs[outbase + g] = r.fracrefa[g];
        }
    } else {
        const int ind0 = ((jp - 13) * 5 + (jt - 1)) + 1;
        const int ind1 = ((jp - 12) * 5 + (jt1 - 1)) + 1;
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            const float for0 = r.forref[ulong(indfor - 1) + g * tabarea_for];
            const float for1 = r.forref[ulong(indfor) + g * tabarea_for];
            const ulong ib0 = ulong(ind0 - 1) + g * tabarea_b;
            const ulong ib1 = ulong(ind1 - 1) + g * tabarea_b;
            const float taufor = forfac * (for0 + forfrac * (for1 - for0));
            float tmaj = fac00 * r.absb[ib0];
            tmaj = tmaj + fac10 * r.absb[ib0 + 1ul];
            tmaj = tmaj + fac01 * r.absb[ib1];
            tmaj = tmaj + fac11 * r.absb[ib1 + 1ul];
            float tau = colh2o * tmaj;
            tau = tau + taufor;
            r.taug[outbase + g] = tau;
            r.fracs[outbase + g] = r.fracrefb[g];
        }
    }
}


// Bands 10 and 11 from rlw_taugb10/11 in
// woof/core/kernels/rrtmg_lw_taugb02_10_11_12.cu.
kernel void rrtmg_legacy_lw_taugb10(
    constant RrtmgLegacyLwTaugb10Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb10Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncol * p.nl)) return;
    const int ncol = p.ncol, nl = p.nl;
    const int col = int(tid) / nl, lay = int(tid) % nl + 1;
    const int laytrop = r.laytrop[col];
    const ulong cell = ulong(col) * ulong(nl) + ulong(lay - 1);
    const ulong fss = ulong(ncol) * ulong(nl);
    const ulong iss = fss;
    const int jp = r.isv[cell];
    const int jt = r.isv[iss + cell];
    const int jt1 = r.isv[2ul * iss + cell];
    const int indself = r.isv[3ul * iss + cell];
    const int indfor = r.isv[4ul * iss + cell];
    const float colh2o = r.fs[2ul * fss + cell];
    const float fac00 = r.fs[10ul * fss + cell];
    const float fac01 = r.fs[11ul * fss + cell];
    const float fac10 = r.fs[12ul * fss + cell];
    const float fac11 = r.fs[13ul * fss + cell];
    const float selffac = r.fs[26ul * fss + cell];
    const float selffrac = r.fs[27ul * fss + cell];
    const float forfac = r.fs[28ul * fss + cell];
    const float forfrac = r.fs[29ul * fss + cell];
    const ulong outbase = cell * 140ul + 108ul;
    const int ng = 6;
    if (lay <= laytrop) {
        const int ind0 = ((jp - 1) * 5 + (jt - 1)) + 1;
        const int ind1 = (jp * 5 + (jt1 - 1)) + 1;
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            const float sr0 = r.selfref[ulong(indself - 1) + 10ul * g];
            const float sr1 = r.selfref[ulong(indself) + 10ul * g];
            const float fr0 = r.forref[ulong(indfor - 1) + 4ul * g];
            const float fr1 = r.forref[ulong(indfor) + 4ul * g];
            const float tauself = selffac * (sr0 + selffrac * (sr1 - sr0));
            const float taufor = forfac * (fr0 + forfrac * (fr1 - fr0));
            const ulong a0 = ulong(ind0 - 1) + 65ul * g;
            const ulong a1 = ulong(ind1 - 1) + 65ul * g;
            float tmaj = fac00 * r.absa[a0];
            tmaj = tmaj + fac10 * r.absa[a0 + 1ul];
            tmaj = tmaj + fac01 * r.absa[a1];
            tmaj = tmaj + fac11 * r.absa[a1 + 1ul];
            float tau = colh2o * tmaj;
            tau = tau + tauself;
            tau = tau + taufor;
            r.taug[outbase + g] = tau;
            r.fracs[outbase + g] = r.fracrefa[g];
        }
    } else {
        const int ind0 = ((jp - 13) * 5 + (jt - 1)) + 1;
        const int ind1 = ((jp - 12) * 5 + (jt1 - 1)) + 1;
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            const float fr0 = r.forref[ulong(indfor - 1) + 4ul * g];
            const float fr1 = r.forref[ulong(indfor) + 4ul * g];
            const float taufor = forfac * (fr0 + forfrac * (fr1 - fr0));
            const ulong b0 = ulong(ind0 - 1) + 235ul * g;
            const ulong b1 = ulong(ind1 - 1) + 235ul * g;
            float tmaj = fac00 * r.absb[b0];
            tmaj = tmaj + fac10 * r.absb[b0 + 1ul];
            tmaj = tmaj + fac01 * r.absb[b1];
            tmaj = tmaj + fac11 * r.absb[b1 + 1ul];
            r.taug[outbase + g] = colh2o * tmaj + taufor;
            r.fracs[outbase + g] = r.fracrefb[g];
        }
    }
}

kernel void rrtmg_legacy_lw_taugb11(
    constant RrtmgLegacyLwTaugb11Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb11Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncol * p.nl)) return;
    const int ncol = p.ncol, nl = p.nl;
    const int col = int(tid) / nl, lay = int(tid) % nl + 1;
    const int laytrop = r.laytrop[col];
    const ulong cell = ulong(col) * ulong(nl) + ulong(lay - 1);
    const ulong fss = ulong(ncol) * ulong(nl);
    const ulong iss = fss;
    const int jp = r.isv[cell];
    const int jt = r.isv[iss + cell];
    const int jt1 = r.isv[2ul * iss + cell];
    const int indself = r.isv[3ul * iss + cell];
    const int indfor = r.isv[4ul * iss + cell];
    const int indminor = r.isv[5ul * iss + cell];
    const float colh2o = r.fs[2ul * fss + cell];
    const float colo2 = r.fs[8ul * fss + cell];
    const float fac00 = r.fs[10ul * fss + cell];
    const float fac01 = r.fs[11ul * fss + cell];
    const float fac10 = r.fs[12ul * fss + cell];
    const float fac11 = r.fs[13ul * fss + cell];
    const float selffac = r.fs[26ul * fss + cell];
    const float selffrac = r.fs[27ul * fss + cell];
    const float forfac = r.fs[28ul * fss + cell];
    const float forfrac = r.fs[29ul * fss + cell];
    const float minorfrac = r.fs[30ul * fss + cell];
    const float scaleminor = r.fs[31ul * fss + cell];
    const float scaleo2 = colo2 * scaleminor;
    const ulong outbase = cell * 140ul + 114ul;
    const int ng = 8;
    if (lay <= laytrop) {
        const int ind0 = ((jp - 1) * 5 + (jt - 1)) + 1;
        const int ind1 = (jp * 5 + (jt1 - 1)) + 1;
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            const float sr0 = r.selfref[ulong(indself - 1) + 10ul * g];
            const float sr1 = r.selfref[ulong(indself) + 10ul * g];
            const float fr0 = r.forref[ulong(indfor - 1) + 4ul * g];
            const float fr1 = r.forref[ulong(indfor) + 4ul * g];
            const float mo20 = r.ka_mo2[ulong(indminor - 1) + 19ul * g];
            const float mo21 = r.ka_mo2[ulong(indminor) + 19ul * g];
            const float tauself = selffac * (sr0 + selffrac * (sr1 - sr0));
            const float taufor = forfac * (fr0 + forfrac * (fr1 - fr0));
            const float tauo2 = scaleo2 * (mo20 + minorfrac * (mo21 - mo20));
            const ulong a0 = ulong(ind0 - 1) + 65ul * g;
            const ulong a1 = ulong(ind1 - 1) + 65ul * g;
            float tmaj = fac00 * r.absa[a0];
            tmaj = tmaj + fac10 * r.absa[a0 + 1ul];
            tmaj = tmaj + fac01 * r.absa[a1];
            tmaj = tmaj + fac11 * r.absa[a1 + 1ul];
            float tau = colh2o * tmaj;
            tau = tau + tauself;
            tau = tau + taufor;
            tau = tau + tauo2;
            r.taug[outbase + g] = tau;
            r.fracs[outbase + g] = r.fracrefa[g];
        }
    } else {
        const int ind0 = ((jp - 13) * 5 + (jt - 1)) + 1;
        const int ind1 = ((jp - 12) * 5 + (jt1 - 1)) + 1;
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            const float fr0 = r.forref[ulong(indfor - 1) + 4ul * g];
            const float fr1 = r.forref[ulong(indfor) + 4ul * g];
            const float mo20 = r.kb_mo2[ulong(indminor - 1) + 19ul * g];
            const float mo21 = r.kb_mo2[ulong(indminor) + 19ul * g];
            const float taufor = forfac * (fr0 + forfrac * (fr1 - fr0));
            const float tauo2 = scaleo2 * (mo20 + minorfrac * (mo21 - mo20));
            const ulong b0 = ulong(ind0 - 1) + 235ul * g;
            const ulong b1 = ulong(ind1 - 1) + 235ul * g;
            float tmaj = fac00 * r.absb[b0];
            tmaj = tmaj + fac10 * r.absb[b0 + 1ul];
            tmaj = tmaj + fac01 * r.absb[b1];
            tmaj = tmaj + fac11 * r.absb[b1 + 1ul];
            float tau = colh2o * tmaj;
            tau = tau + taufor;
            tau = tau + tauo2;
            r.taug[outbase + g] = tau;
            r.fracs[outbase + g] = r.fracrefb[g];
        }
    }
}


// Band 12 from rlw_taugb12 in
// woof/core/kernels/rrtmg_lw_taugb02_10_11_12.cu.
kernel void rrtmg_legacy_lw_taugb12(
    constant RrtmgLegacyLwTaugb12Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb12Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncol * p.nl)) return;
    const int ncol = p.ncol, nl = p.nl;
    const int col = int(tid) / nl, lay = int(tid) % nl + 1;
    const int laytrop = r.laytrop[col];
    const ulong cell = ulong(col) * ulong(nl) + ulong(lay - 1);
    const ulong fss = ulong(ncol) * ulong(nl);
    const ulong iss = fss;
    const int jp = r.isv[cell];
    const int jt = r.isv[iss + cell];
    const int jt1 = r.isv[2ul * iss + cell];
    const int indself = r.isv[3ul * iss + cell];
    const int indfor = r.isv[4ul * iss + cell];
    const float oneminus = p.oneminus;
    const float colh2o = r.fs[2ul * fss + cell];
    const float colco2 = r.fs[3ul * fss + cell];
    const float fac00 = r.fs[10ul * fss + cell];
    const float fac01 = r.fs[11ul * fss + cell];
    const float fac10 = r.fs[12ul * fss + cell];
    const float fac11 = r.fs[13ul * fss + cell];
    const float rat0 = r.fs[14ul * fss + cell];
    const float rat1 = r.fs[15ul * fss + cell];
    const float selffac = r.fs[26ul * fss + cell];
    const float selffrac = r.fs[27ul * fss + cell];
    const float forfac = r.fs[28ul * fss + cell];
    const float forfrac = r.fs[29ul * fss + cell];
    const ulong outbase = cell * 140ul + 122ul;
    const int ng = 8;

    if (lay <= laytrop) {
        const float refrat_planck_a = r.chi_mls[0ul + 7ul * 9ul]
                                    / r.chi_mls[1ul + 7ul * 9ul];
        const float speccomb = colh2o + rat0 * colco2;
        float specparm = colh2o / speccomb;
        if (specparm >= oneminus) specparm = oneminus;
        const float specmult = 8.0f * specparm;
        const int js = 1 + int(specmult);
        const float fs0 = fmod(specmult, 1.0f);

        const float speccomb1 = colh2o + rat1 * colco2;
        float specparm1 = colh2o / speccomb1;
        if (specparm1 >= oneminus) specparm1 = oneminus;
        const float specmult1 = 8.0f * specparm1;
        const int js1 = 1 + int(specmult1);
        const float fs1 = fmod(specmult1, 1.0f);

        const float speccomb_planck = colh2o + refrat_planck_a * colco2;
        float specparm_planck = colh2o / speccomb_planck;
        if (specparm_planck >= oneminus) specparm_planck = oneminus;
        const float specmult_planck = 8.0f * specparm_planck;
        const int jpl = 1 + int(specmult_planck);
        const float fpl = fmod(specmult_planck, 1.0f);

        const int ind0 = ((jp - 1) * 5 + (jt - 1)) * 9 + js;
        const int ind1 = (jp * 5 + (jt1 - 1)) * 9 + js1;
        const ulong selfbase = ulong(indself - 1);
        const ulong forbase = ulong(indfor - 1);
        float fac000 = 0.0f, fac100 = 0.0f, fac200 = 0.0f;
        float fac010 = 0.0f, fac110 = 0.0f, fac210 = 0.0f;
        if (specparm < 0.125f) {
            const float q = fs0 - 1.0f;
            const float q2 = q * q;
            const float q4 = q2 * q2;
            const float fk0 = q4;
            const float fk1 = (1.0f - q) - 2.0f * q4;
            const float fk2 = q + q4;
            fac000 = fk0 * fac00; fac100 = fk1 * fac00; fac200 = fk2 * fac00;
            fac010 = fk0 * fac10; fac110 = fk1 * fac10; fac210 = fk2 * fac10;
        } else if (specparm > 0.875f) {
            const float q = -fs0;
            const float q2 = q * q;
            const float q4 = q2 * q2;
            const float fk0 = q4;
            const float fk1 = (1.0f - q) - 2.0f * q4;
            const float fk2 = q + q4;
            fac000 = fk0 * fac00; fac100 = fk1 * fac00; fac200 = fk2 * fac00;
            fac010 = fk0 * fac10; fac110 = fk1 * fac10; fac210 = fk2 * fac10;
        } else {
            fac000 = (1.0f - fs0) * fac00;
            fac010 = (1.0f - fs0) * fac10;
            fac100 = fs0 * fac00;
            fac110 = fs0 * fac10;
        }
        float fac001 = 0.0f, fac101 = 0.0f, fac201 = 0.0f;
        float fac011 = 0.0f, fac111 = 0.0f, fac211 = 0.0f;
        if (specparm1 < 0.125f) {
            const float q = fs1 - 1.0f;
            const float q2 = q * q;
            const float q4 = q2 * q2;
            const float fk0 = q4;
            const float fk1 = (1.0f - q) - 2.0f * q4;
            const float fk2 = q + q4;
            fac001 = fk0 * fac01; fac101 = fk1 * fac01; fac201 = fk2 * fac01;
            fac011 = fk0 * fac11; fac111 = fk1 * fac11; fac211 = fk2 * fac11;
        } else if (specparm1 > 0.875f) {
            const float q = -fs1;
            const float q2 = q * q;
            const float q4 = q2 * q2;
            const float fk0 = q4;
            const float fk1 = (1.0f - q) - 2.0f * q4;
            const float fk2 = q + q4;
            fac001 = fk0 * fac01; fac101 = fk1 * fac01; fac201 = fk2 * fac01;
            fac011 = fk0 * fac11; fac111 = fk1 * fac11; fac211 = fk2 * fac11;
        } else {
            fac001 = (1.0f - fs1) * fac01;
            fac011 = (1.0f - fs1) * fac11;
            fac101 = fs1 * fac01;
            fac111 = fs1 * fac11;
        }

        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            const float sr0 = r.selfref[selfbase + 10ul * g];
            const float sr1 = r.selfref[selfbase + 1ul + 10ul * g];
            const float fr0 = r.forref[forbase + 4ul * g];
            const float fr1 = r.forref[forbase + 1ul + 4ul * g];
            const float tauself = selffac * (sr0 + selffrac * (sr1 - sr0));
            const float taufor = forfac * (fr0 + forfrac * (fr1 - fr0));
            float tau_major;
            if (specparm < 0.125f) {
                const ulong base = ulong(ind0 - 1) + 585ul * g;
                float t = fac000 * r.absa[base];
                t = t + fac100 * r.absa[base + 1ul];
                t = t + fac200 * r.absa[base + 2ul];
                t = t + fac010 * r.absa[base + 9ul];
                t = t + fac110 * r.absa[base + 10ul];
                t = t + fac210 * r.absa[base + 11ul];
                tau_major = speccomb * t;
            } else if (specparm > 0.875f) {
                const ulong base = ulong(ind0 - 1) + 585ul * g;
                float t = fac200 * r.absa[base - 1ul];
                t = t + fac100 * r.absa[base];
                t = t + fac000 * r.absa[base + 1ul];
                t = t + fac210 * r.absa[base + 8ul];
                t = t + fac110 * r.absa[base + 9ul];
                t = t + fac010 * r.absa[base + 10ul];
                tau_major = speccomb * t;
            } else {
                const ulong base = ulong(ind0 - 1) + 585ul * g;
                float t = fac000 * r.absa[base];
                t = t + fac100 * r.absa[base + 1ul];
                t = t + fac010 * r.absa[base + 9ul];
                t = t + fac110 * r.absa[base + 10ul];
                tau_major = speccomb * t;
            }
            float tau_major1;
            if (specparm1 < 0.125f) {
                const ulong base = ulong(ind1 - 1) + 585ul * g;
                float t = fac001 * r.absa[base];
                t = t + fac101 * r.absa[base + 1ul];
                t = t + fac201 * r.absa[base + 2ul];
                t = t + fac011 * r.absa[base + 9ul];
                t = t + fac111 * r.absa[base + 10ul];
                t = t + fac211 * r.absa[base + 11ul];
                tau_major1 = speccomb1 * t;
            } else if (specparm1 > 0.875f) {
                const ulong base = ulong(ind1 - 1) + 585ul * g;
                float t = fac201 * r.absa[base - 1ul];
                t = t + fac101 * r.absa[base];
                t = t + fac001 * r.absa[base + 1ul];
                t = t + fac211 * r.absa[base + 8ul];
                t = t + fac111 * r.absa[base + 9ul];
                t = t + fac011 * r.absa[base + 10ul];
                tau_major1 = speccomb1 * t;
            } else {
                const ulong base = ulong(ind1 - 1) + 585ul * g;
                float t = fac001 * r.absa[base];
                t = t + fac101 * r.absa[base + 1ul];
                t = t + fac011 * r.absa[base + 9ul];
                t = t + fac111 * r.absa[base + 10ul];
                tau_major1 = speccomb1 * t;
            }
            float tau = tau_major + tau_major1;
            tau = tau + tauself;
            tau = tau + taufor;
            r.taug[outbase + g] = tau;
            const ulong frac0 = ulong(ig - 1) + 8ul * ulong(jpl - 1);
            const ulong frac1 = ulong(ig - 1) + 8ul * ulong(jpl);
            const float f0 = r.fracrefa[frac0];
            const float f1 = r.fracrefa[frac1];
            r.fracs[outbase + g] = f0 + fpl * (f1 - f0);
        }
    } else {
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            r.taug[outbase + g] = 0.0f;
            r.fracs[outbase + g] = 0.0f;
        }
    }
}


// Band 1 from rlw_taugb1 in woof/core/kernels/rrtmg_lw.cu.
kernel void rrtmg_legacy_lw_taugb1(
    constant RrtmgLegacyLwTaugb1Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb1Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncol * p.nl)) return;
    const int ncol = p.ncol, nl = p.nl;
    const int col = int(tid) / nl, lay = int(tid) % nl + 1;
    const int laytrop = r.laytrop[col];
    const ulong cell = ulong(col) * ulong(nl) + ulong(lay - 1);
    const ulong fss = ulong(ncol) * ulong(nl);
    const ulong iss = fss;
    const int jp = r.isv[cell];
    const int jt = r.isv[iss + cell];
    const int jt1 = r.isv[2ul * iss + cell];
    const int indself = r.isv[3ul * iss + cell];
    const int indfor = r.isv[4ul * iss + cell];
    const int indminor = r.isv[5ul * iss + cell];
    const float colh2o = r.fs[2ul * fss + cell];
    const float colbrd = r.fs[9ul * fss + cell];
    const float fac00 = r.fs[10ul * fss + cell];
    const float fac01 = r.fs[11ul * fss + cell];
    const float fac10 = r.fs[12ul * fss + cell];
    const float fac11 = r.fs[13ul * fss + cell];
    const float selffac = r.fs[26ul * fss + cell];
    const float selffrac = r.fs[27ul * fss + cell];
    const float forfac = r.fs[28ul * fss + cell];
    const float forfrac = r.fs[29ul * fss + cell];
    const float minorfrac = r.fs[30ul * fss + cell];
    const float scaleminorn2 = r.fs[32ul * fss + cell];
    const float scale_n2 = colbrd * scaleminorn2;
    const ulong outbase = cell * 140ul;
    const int ng = 10;

    if (lay <= laytrop) {
        const int ind0 = ((jp - 1) * 5 + (jt - 1)) + 1;
        const int ind1 = (jp * 5 + (jt1 - 1)) + 1;
        const float pp = r.fs[cell];
        float corradj = 1.0f;
        if (pp < 250.0f) corradj = 1.0f - (0.15f * (250.0f - pp)) / 154.4f;
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            const float sr0 = r.selfref[ulong(indself - 1) + 10ul * g];
            const float sr1 = r.selfref[ulong(indself) + 10ul * g];
            const float fr0 = r.forref[ulong(indfor - 1) + 4ul * g];
            const float fr1 = r.forref[ulong(indfor) + 4ul * g];
            const float n20 = r.ka_mn2[ulong(indminor - 1) + 19ul * g];
            const float n21 = r.ka_mn2[ulong(indminor) + 19ul * g];
            const float tauself = selffac * (sr0 + selffrac * (sr1 - sr0));
            const float taufor = forfac * (fr0 + forfrac * (fr1 - fr0));
            const float taun2 = scale_n2 * (n20 + minorfrac * (n21 - n20));
            const ulong a0 = ulong(ind0 - 1) + 65ul * g;
            const ulong a1 = ulong(ind1 - 1) + 65ul * g;
            float tmaj = fac00 * r.absa[a0];
            tmaj = tmaj + fac10 * r.absa[a0 + 1ul];
            tmaj = tmaj + fac01 * r.absa[a1];
            tmaj = tmaj + fac11 * r.absa[a1 + 1ul];
            float tau = colh2o * tmaj;
            tau = tau + tauself;
            tau = tau + taufor;
            tau = tau + taun2;
            r.taug[outbase + g] = corradj * tau;
            r.fracs[outbase + g] = r.fracrefa[g];
        }
    } else {
        const int ind0 = ((jp - 13) * 5 + (jt - 1)) + 1;
        const int ind1 = ((jp - 12) * 5 + (jt1 - 1)) + 1;
        const float pp = r.fs[cell];
        const float corradj = 1.0f - 0.15f * (pp / 95.6f);
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            const float fr0 = r.forref[ulong(indfor - 1) + 4ul * g];
            const float fr1 = r.forref[ulong(indfor) + 4ul * g];
            const float n20 = r.kb_mn2[ulong(indminor - 1) + 19ul * g];
            const float n21 = r.kb_mn2[ulong(indminor) + 19ul * g];
            const float taufor = forfac * (fr0 + forfrac * (fr1 - fr0));
            const float taun2 = scale_n2 * (n20 + minorfrac * (n21 - n20));
            const ulong b0 = ulong(ind0 - 1) + 235ul * g;
            const ulong b1 = ulong(ind1 - 1) + 235ul * g;
            float tmaj = fac00 * r.absb[b0];
            tmaj = tmaj + fac10 * r.absb[b0 + 1ul];
            tmaj = tmaj + fac01 * r.absb[b1];
            tmaj = tmaj + fac11 * r.absb[b1 + 1ul];
            float tau = colh2o * tmaj;
            tau = tau + taufor;
            tau = tau + taun2;
            r.taug[outbase + g] = corradj * tau;
            r.fracs[outbase + g] = r.fracrefb[g];
        }
    }
}


// Band 4 from rlw_taugb4 in woof/core/kernels/rrtmg_lw_taugb03_05.cu.
kernel void rrtmg_legacy_lw_taugb4(
    constant RrtmgLegacyLwTaugb4Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb4Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncol * p.nl)) return;
    const int ncol = p.ncol, nl = p.nl;
    const int col = int(tid) / nl, lay = int(tid) % nl + 1;
    const int laytrop = r.laytrop[col];
    const ulong cell = ulong(col) * ulong(nl) + ulong(lay - 1);
    const ulong fss = ulong(ncol) * ulong(nl), iss = fss;
    const int jp = r.isv[cell];
    const int jt = r.isv[iss + cell], jt1 = r.isv[2ul * iss + cell];
    const int indself = r.isv[3ul * iss + cell], indfor = r.isv[4ul * iss + cell];
    const float colco2 = r.fs[3ul * fss + cell];
    const float oneminus = p.oneminus;
    const float ref_a = r.chi_mls[0ul + 7ul * 10ul] / r.chi_mls[1ul + 7ul * 10ul];
    const float ref_b = r.chi_mls[2ul + 7ul * 12ul] / r.chi_mls[1ul + 7ul * 12ul];
    const ulong outbase = cell * 140ul + 38ul;
    const int ng = 14;
    if (lay <= laytrop) {
        const float colh2o = r.fs[2ul * fss + cell];
        const float rat0 = r.fs[14ul * fss + cell], rat1 = r.fs[15ul * fss + cell];
        const RrtmgLegacySpec s0 = rrtmg_legacy_spec(colh2o, rat0, colco2, oneminus, 8.0f);
        const RrtmgLegacySpec s1 = rrtmg_legacy_spec(colh2o, rat1, colco2, oneminus, 8.0f);
        const RrtmgLegacySpec spl = rrtmg_legacy_spec(colh2o, ref_a, colco2, oneminus, 8.0f);
        const int ind0 = ((jp - 1) * 5 + (jt - 1)) * 9 + s0.j;
        const int ind1 = (jp * 5 + (jt1 - 1)) * 9 + s1.j;
        const RrtmgLegacyFac f0 = rrtmg_legacy_fac(s0.parm, s0.f,
            r.fs[10ul * fss + cell], r.fs[12ul * fss + cell]);
        const RrtmgLegacyFac f1 = rrtmg_legacy_fac(s1.parm, s1.f,
            r.fs[11ul * fss + cell], r.fs[13ul * fss + cell]);
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            const float sr0 = r.selfref[ulong(indself - 1) + 10ul * g];
            const float sr1 = r.selfref[ulong(indself) + 10ul * g];
            const float fr0 = r.forref[ulong(indfor - 1) + 4ul * g];
            const float fr1 = r.forref[ulong(indfor) + 4ul * g];
            const float tauself = r.fs[26ul * fss + cell]
                * (sr0 + r.fs[27ul * fss + cell] * (sr1 - sr0));
            const float taufor = r.fs[28ul * fss + cell]
                * (fr0 + r.fs[29ul * fss + cell] * (fr1 - fr0));
            float tau = rrtmg_legacy_major(f0, r.absa, ind0, ig, s0.comb);
            tau = tau + rrtmg_legacy_major(f1, r.absa, ind1, ig, s1.comb);
            tau = tau + tauself;
            tau = tau + taufor;
            r.taug[outbase + g] = tau;
            const ulong f0i = g + 14ul * ulong(spl.j - 1);
            const ulong f1i = g + 14ul * ulong(spl.j);
            const float a = r.fracrefa[f0i];
            r.fracs[outbase + g] = a + spl.f * (r.fracrefa[f1i] - a);
        }
    } else {
        const float colo3 = r.fs[4ul * fss + cell];
        const float rat0 = r.fs[24ul * fss + cell], rat1 = r.fs[25ul * fss + cell];
        const RrtmgLegacySpec s0 = rrtmg_legacy_spec(colo3, rat0, colco2, oneminus, 4.0f);
        const RrtmgLegacySpec s1 = rrtmg_legacy_spec(colo3, rat1, colco2, oneminus, 4.0f);
        const RrtmgLegacySpec spl = rrtmg_legacy_spec(colo3, ref_b, colco2, oneminus, 4.0f);
        const int ind0 = ((jp - 13) * 5 + (jt - 1)) * 5 + s0.j;
        const int ind1 = ((jp - 12) * 5 + (jt1 - 1)) * 5 + s1.j;
        const float fac000 = (1.0f - s0.f) * r.fs[10ul * fss + cell];
        const float fac010 = (1.0f - s0.f) * r.fs[12ul * fss + cell];
        const float fac100 = s0.f * r.fs[10ul * fss + cell];
        const float fac110 = s0.f * r.fs[12ul * fss + cell];
        const float fac001 = (1.0f - s1.f) * r.fs[11ul * fss + cell];
        const float fac011 = (1.0f - s1.f) * r.fs[13ul * fss + cell];
        const float fac101 = s1.f * r.fs[11ul * fss + cell];
        const float fac111 = s1.f * r.fs[13ul * fss + cell];
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            float tau = rrtmg_legacy_bmajor(fac000, fac100, fac010, fac110,
                                            r.absb, ind0, ig, s0.comb);
            tau = tau + rrtmg_legacy_bmajor(fac001, fac101, fac011, fac111,
                                            r.absb, ind1, ig, s1.comb);
            if (ig >= 8) {
                const float scale[7] = {0.92f, 0.88f, 1.07f, 1.1f, 0.99f, 0.88f, 0.943f};
                tau = tau * scale[ig - 8];
            }
            r.taug[outbase + g] = tau;
            const ulong f0i = g + 14ul * ulong(spl.j - 1);
            const ulong f1i = g + 14ul * ulong(spl.j);
            const float b = r.fracrefb[f0i];
            r.fracs[outbase + g] = b + spl.f * (r.fracrefb[f1i] - b);
        }
    }
}


// Band 3 from rlw_taugb3 in woof/core/kernels/rrtmg_lw_taugb03_05.cu.
kernel void rrtmg_legacy_lw_taugb3(
    constant RrtmgLegacyLwTaugb3Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb3Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncol * p.nl)) return;
    const int ncol = p.ncol, nl = p.nl;
    const int col = int(tid) / nl, lay = int(tid) % nl + 1;
    const int laytrop = r.laytrop[col];
    const ulong cell = ulong(col) * ulong(nl) + ulong(lay - 1);
    const ulong fss = ulong(ncol) * ulong(nl), iss = fss;
    const int jp = r.isv[cell], jt = r.isv[iss + cell], jt1 = r.isv[2ul * iss + cell];
    const int indself = r.isv[3ul * iss + cell], indfor = r.isv[4ul * iss + cell];
    const int indminor = r.isv[5ul * iss + cell];
    const float oneminus = p.oneminus;
    const float colh2o = r.fs[2ul * fss + cell], colco2 = r.fs[3ul * fss + cell];
    const float colo3 = r.fs[4ul * fss + cell], coln2o = r.fs[5ul * fss + cell];
    const float coldry = r.fs[fss + cell];
    const float fac00 = r.fs[10ul * fss + cell], fac01 = r.fs[11ul * fss + cell];
    const float fac10 = r.fs[12ul * fss + cell], fac11 = r.fs[13ul * fss + cell];
    const float rat0 = r.fs[14ul * fss + cell], rat1 = r.fs[15ul * fss + cell];
    const float selffac = r.fs[26ul * fss + cell], selffrac = r.fs[27ul * fss + cell];
    const float forfac = r.fs[28ul * fss + cell], forfrac = r.fs[29ul * fss + cell];
    const float minorfrac = r.fs[30ul * fss + cell];
    const ulong outbase = cell * 140ul + 22ul;
    const int ng = 16;
    const float refrat_planck_a = r.chi_mls[0ul + 7ul * 8ul] / r.chi_mls[1ul + 7ul * 8ul];
    const float refrat_planck_b = r.chi_mls[0ul + 7ul * 12ul] / r.chi_mls[1ul + 7ul * 12ul];
    const float refrat_m_a = r.chi_mls[0ul + 7ul * 2ul] / r.chi_mls[1ul + 7ul * 2ul];
    const float refrat_m_b = r.chi_mls[0ul + 7ul * 12ul] / r.chi_mls[1ul + 7ul * 12ul];
    const float chi_n2o = coln2o / coldry;
    const float ratn2o = (1.0e20f * chi_n2o) / r.chi_mls[3ul + 7ul * ulong(jp)];
    float adjcoln2o = coln2o;
    if (ratn2o > 1.5f) {
        const float adjfac = 0.5f + pow(ratn2o - 0.5f, 0.65f);
        adjcoln2o = ((adjfac * r.chi_mls[3ul + 7ul * ulong(jp)]) * coldry) * 1.0e-20f;
    }

    if (lay <= laytrop) {
        const RrtmgLegacySpec s0 = rrtmg_legacy_spec(colh2o, rat0, colco2, oneminus, 8.0f);
        const RrtmgLegacySpec s1 = rrtmg_legacy_spec(colh2o, rat1, colco2, oneminus, 8.0f);
        const RrtmgLegacySpec smn = rrtmg_legacy_spec(colh2o, refrat_m_a, colco2, oneminus, 8.0f);
        const RrtmgLegacySpec spl = rrtmg_legacy_spec(colh2o, refrat_planck_a, colco2, oneminus, 8.0f);
        const int ind0 = ((jp - 1) * 5 + (jt - 1)) * 9 + s0.j;
        const int ind1 = (jp * 5 + (jt1 - 1)) * 9 + s1.j;
        const RrtmgLegacyFac f0 = rrtmg_legacy_fac(s0.parm, s0.f, fac00, fac10);
        const RrtmgLegacyFac f1 = rrtmg_legacy_fac(s1.parm, s1.f, fac01, fac11);
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            const float sr0 = r.selfref[ulong(indself - 1) + 10ul * g];
            const float sr1 = r.selfref[ulong(indself) + 10ul * g];
            const float fr0 = r.forref[ulong(indfor - 1) + 4ul * g];
            const float fr1 = r.forref[ulong(indfor) + 4ul * g];
            const float tauself = selffac * (sr0 + selffrac * (sr1 - sr0));
            const float taufor = forfac * (fr0 + forfrac * (fr1 - fr0));
            const ulong ka0 = ulong(smn.j - 1) + 9ul * ulong(indminor - 1) + 171ul * g;
            const ulong ka1 = ulong(smn.j) + 9ul * ulong(indminor - 1) + 171ul * g;
            const ulong ka2 = ulong(smn.j - 1) + 9ul * ulong(indminor) + 171ul * g;
            const ulong ka3 = ulong(smn.j) + 9ul * ulong(indminor) + 171ul * g;
            const float n2om1 = r.ka_mn2o[ka0] + smn.f * (r.ka_mn2o[ka1] - r.ka_mn2o[ka0]);
            const float n2om2 = r.ka_mn2o[ka2] + smn.f * (r.ka_mn2o[ka3] - r.ka_mn2o[ka2]);
            const float absn2o = n2om1 + minorfrac * (n2om2 - n2om1);
            float tau = rrtmg_legacy_major(f0, r.absa, ind0, ig, s0.comb);
            tau = tau + rrtmg_legacy_major(f1, r.absa, ind1, ig, s1.comb);
            tau = tau + tauself;
            tau = tau + taufor;
            tau = tau + adjcoln2o * absn2o;
            r.taug[outbase + g] = tau;
            const ulong f0i = g + 16ul * ulong(spl.j - 1);
            const ulong f1i = g + 16ul * ulong(spl.j);
            const float fr0p = r.fracrefa[f0i];
            r.fracs[outbase + g] = fr0p + spl.f * (r.fracrefa[f1i] - fr0p);
        }
    } else {
        const RrtmgLegacySpec s0 = rrtmg_legacy_spec(colh2o, rat0, colco2, oneminus, 4.0f);
        const RrtmgLegacySpec s1 = rrtmg_legacy_spec(colh2o, rat1, colco2, oneminus, 4.0f);
        const float fac000 = (1.0f - s0.f) * fac00, fac010 = (1.0f - s0.f) * fac10;
        const float fac100 = s0.f * fac00, fac110 = s0.f * fac10;
        const float fac001 = (1.0f - s1.f) * fac01, fac011 = (1.0f - s1.f) * fac11;
        const float fac101 = s1.f * fac01, fac111 = s1.f * fac11;
        const RrtmgLegacySpec smn = rrtmg_legacy_spec(colh2o, refrat_m_b, colco2, oneminus, 4.0f);
        const RrtmgLegacySpec spl = rrtmg_legacy_spec(colh2o, refrat_planck_b, colco2, oneminus, 4.0f);
        const int ind0 = ((jp - 13) * 5 + (jt - 1)) * 5 + s0.j;
        const int ind1 = ((jp - 12) * 5 + (jt1 - 1)) * 5 + s1.j;
        for (int ig = 1; ig <= ng; ++ig) {
            const ulong g = ulong(ig - 1);
            const float fr0 = r.forref[ulong(indfor - 1) + 4ul * g];
            const float fr1 = r.forref[ulong(indfor) + 4ul * g];
            const float taufor = forfac * (fr0 + forfrac * (fr1 - fr0));
            const ulong kb0 = ulong(smn.j - 1) + 5ul * ulong(indminor - 1) + 95ul * g;
            const ulong kb1 = ulong(smn.j) + 5ul * ulong(indminor - 1) + 95ul * g;
            const ulong kb2 = ulong(smn.j - 1) + 5ul * ulong(indminor) + 95ul * g;
            const ulong kb3 = ulong(smn.j) + 5ul * ulong(indminor) + 95ul * g;
            const float n2om1 = r.kb_mn2o[kb0] + smn.f * (r.kb_mn2o[kb1] - r.kb_mn2o[kb0]);
            const float n2om2 = r.kb_mn2o[kb2] + smn.f * (r.kb_mn2o[kb3] - r.kb_mn2o[kb2]);
            const float absn2o = n2om1 + minorfrac * (n2om2 - n2om1);
            float tau = rrtmg_legacy_bmajor(fac000, fac100, fac010, fac110,
                                            r.absb, ind0, ig, s0.comb);
            tau = tau + rrtmg_legacy_bmajor(fac001, fac101, fac011, fac111,
                                            r.absb, ind1, ig, s1.comb);
            tau = tau + taufor;
            tau = tau + adjcoln2o * absn2o;
            r.taug[outbase + g] = tau;
            const ulong f0i = g + 16ul * ulong(spl.j - 1);
            const ulong f1i = g + 16ul * ulong(spl.j);
            const float fr0p = r.fracrefb[f0i];
            r.fracs[outbase + g] = fr0p + spl.f * (r.fracrefb[f1i] - fr0p);
        }
    }
}


// Band 5 from rlw_taugb5 in woof/core/kernels/rrtmg_lw_taugb03_05.cu.
kernel void rrtmg_legacy_lw_taugb5(
    constant RrtmgLegacyLwTaugb5Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb5Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncol * p.nl)) return;
    const int ncol=p.ncol, nl=p.nl;
    const int col=int(tid)/nl, lay=int(tid)%nl+1;
    const int laytrop=r.laytrop[col];
    const ulong cell=ulong(col)*ulong(nl)+ulong(lay-1);
    const ulong fss=ulong(ncol)*ulong(nl), iss=fss;
    const int jp=r.isv[cell], jt=r.isv[iss+cell], jt1=r.isv[2ul*iss+cell];
    const int indself=r.isv[3ul*iss+cell], indfor=r.isv[4ul*iss+cell];
    const int indminor=r.isv[5ul*iss+cell];
    const float oneminus=p.oneminus;
    const float colco2=r.fs[3ul*fss+cell], colo3=r.fs[4ul*fss+cell];
    const float rat0=r.fs[14ul*fss+cell], rat1=r.fs[15ul*fss+cell];
    const float rat_o30=r.fs[24ul*fss+cell], rat_o31=r.fs[25ul*fss+cell];
    const float fac00=r.fs[10ul*fss+cell], fac01=r.fs[11ul*fss+cell];
    const float fac10=r.fs[12ul*fss+cell], fac11=r.fs[13ul*fss+cell];
    const float selffac=r.fs[26ul*fss+cell], selffrac=r.fs[27ul*fss+cell];
    const float forfac=r.fs[28ul*fss+cell], forfrac=r.fs[29ul*fss+cell];
    const float minorfrac=r.fs[30ul*fss+cell];
    const float refrat_planck_a=r.chi_mls[0ul+7ul*4ul]/r.chi_mls[1ul+7ul*4ul];
    const float refrat_planck_b=r.chi_mls[2ul+7ul*42ul]/r.chi_mls[1ul+7ul*42ul];
    const float refrat_m_a=r.chi_mls[0ul+7ul*6ul]/r.chi_mls[1ul+7ul*6ul];
    const ulong wxbase=ulong(col)*4ul*ulong(nl)+ulong(lay-1);
    const float wx=r.wx[wxbase];
    const ulong outbase=cell*140ul+52ul;
    const int ng=16;
    if (lay<=laytrop) {
        const float colh2o=r.fs[2ul*fss+cell];
        const RrtmgLegacySpec s0=rrtmg_legacy_spec(colh2o,rat0,colco2,oneminus,8.0f);
        const RrtmgLegacySpec s1=rrtmg_legacy_spec(colh2o,rat1,colco2,oneminus,8.0f);
        const RrtmgLegacySpec smo=rrtmg_legacy_spec(colh2o,refrat_m_a,colco2,oneminus,8.0f);
        const RrtmgLegacySpec spl=rrtmg_legacy_spec(colh2o,refrat_planck_a,colco2,oneminus,8.0f);
        const int ind0=((jp-1)*5+(jt-1))*9+s0.j;
        const int ind1=(jp*5+(jt1-1))*9+s1.j;
        const RrtmgLegacyFac f0=rrtmg_legacy_fac(s0.parm,s0.f,fac00,fac10);
        const RrtmgLegacyFac f1=rrtmg_legacy_fac(s1.parm,s1.f,fac01,fac11);
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            const float sr0=r.selfref[ulong(indself-1)+10ul*g], sr1=r.selfref[ulong(indself)+10ul*g];
            const float fr0=r.forref[ulong(indfor-1)+4ul*g], fr1=r.forref[ulong(indfor)+4ul*g];
            const float tauself=selffac*(sr0+selffrac*(sr1-sr0));
            const float taufor=forfac*(fr0+forfrac*(fr1-fr0));
            const ulong ka0=ulong(smo.j-1)+9ul*ulong(indminor-1)+171ul*g;
            const ulong ka1=ulong(smo.j)+9ul*ulong(indminor-1)+171ul*g;
            const ulong ka2=ulong(smo.j-1)+9ul*ulong(indminor)+171ul*g;
            const ulong ka3=ulong(smo.j)+9ul*ulong(indminor)+171ul*g;
            const float o30=r.ka_mo3[ka0]+smo.f*(r.ka_mo3[ka1]-r.ka_mo3[ka0]);
            const float o31=r.ka_mo3[ka2]+smo.f*(r.ka_mo3[ka3]-r.ka_mo3[ka2]);
            const float abso3=o30+minorfrac*(o31-o30);
            float tau=rrtmg_legacy_major(f0,r.absa,ind0,ig,s0.comb);
            tau=tau+rrtmg_legacy_major(f1,r.absa,ind1,ig,s1.comb);
            tau=tau+tauself; tau=tau+taufor;
            tau=tau+abso3*colo3;
            tau=tau+wx*r.ccl4[g];
            r.taug[outbase+g]=tau;
            const ulong f0i=g+16ul*ulong(spl.j-1), f1i=g+16ul*ulong(spl.j);
            const float a=r.fracrefa[f0i]; r.fracs[outbase+g]=a+spl.f*(r.fracrefa[f1i]-a);
        }
    } else {
        const RrtmgLegacySpec s0=rrtmg_legacy_spec(colo3,rat_o30,colco2,oneminus,4.0f);
        const RrtmgLegacySpec s1=rrtmg_legacy_spec(colo3,rat_o31,colco2,oneminus,4.0f);
        const RrtmgLegacySpec spl=rrtmg_legacy_spec(colo3,refrat_planck_b,colco2,oneminus,4.0f);
        const int ind0=((jp-13)*5+(jt-1))*5+s0.j;
        const int ind1=((jp-12)*5+(jt1-1))*5+s1.j;
        const float fac000=(1.0f-s0.f)*fac00, fac010=(1.0f-s0.f)*fac10;
        const float fac100=s0.f*fac00, fac110=s0.f*fac10;
        const float fac001=(1.0f-s1.f)*fac01, fac011=(1.0f-s1.f)*fac11;
        const float fac101=s1.f*fac01, fac111=s1.f*fac11;
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            float tau=rrtmg_legacy_bmajor(fac000,fac100,fac010,fac110,r.absb,ind0,ig,s0.comb);
            tau=tau+rrtmg_legacy_bmajor(fac001,fac101,fac011,fac111,r.absb,ind1,ig,s1.comb);
            tau=tau+wx*r.ccl4[g];
            r.taug[outbase+g]=tau;
            const ulong f0i=g+16ul*ulong(spl.j-1), f1i=g+16ul*ulong(spl.j);
            const float b=r.fracrefb[f0i]; r.fracs[outbase+g]=b+spl.f*(r.fracrefb[f1i]-b);
        }
    }
}


// Band 6 from rlw_taugb6 in woof/core/kernels/rrtmg_lw_taugb06_09.cu.
kernel void rrtmg_legacy_lw_taugb6(
    constant RrtmgLegacyLwTaugb6Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb6Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncol*p.nl)) return;
    const int ncol=p.ncol,nl=p.nl;
    const int col=int(tid)/nl,lay=int(tid)%nl+1;
    const int laytrop=r.laytrop[col];
    const ulong cell=ulong(col)*ulong(nl)+ulong(lay-1);
    const ulong fss=ulong(ncol)*ulong(nl),iss=fss;
    const int jp=r.isv[cell],jt=r.isv[iss+cell],jt1=r.isv[2ul*iss+cell];
    const int indself=r.isv[3ul*iss+cell],indfor=r.isv[4ul*iss+cell],indminor=r.isv[5ul*iss+cell];
    const float colh2o=r.fs[2ul*fss+cell],colco2=r.fs[3ul*fss+cell];
    const float coldry=r.fs[fss+cell],fac00=r.fs[10ul*fss+cell];
    const float fac01=r.fs[11ul*fss+cell],fac10=r.fs[12ul*fss+cell],fac11=r.fs[13ul*fss+cell];
    const float selffac=r.fs[26ul*fss+cell],selffrac=r.fs[27ul*fss+cell];
    const float forfac=r.fs[28ul*fss+cell],forfrac=r.fs[29ul*fss+cell];
    const float minorfrac=r.fs[30ul*fss+cell];
    const ulong wx2=ulong(col)*4ul*ulong(nl)+ulong(nl)+ulong(lay-1);
    const ulong wx3=ulong(col)*4ul*ulong(nl)+2ul*ulong(nl)+ulong(lay-1);
    const float cfc11=r.wx[wx2],cfc12x=r.wx[wx3];
    const ulong outbase=cell*140ul+68ul;
    const int ng=8;
    if(lay<=laytrop) {
        const float chi=colco2/coldry;
        const float ratio=(1.0e20f*chi)/r.chi_mls[1ul+7ul*ulong(jp)];
        float adjcol=colco2;
        if(ratio>3.0f) {
            const float adjfac=2.0f+pow(ratio-2.0f,0.77f);
            adjcol=((adjfac*r.chi_mls[1ul+7ul*ulong(jp)])*coldry)*1.0e-20f;
        }
        const int ind0=((jp-1)*5+(jt-1))+1,ind1=(jp*5+(jt1-1))+1;
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            const float sr0=r.selfref[ulong(indself-1)+10ul*g],sr1=r.selfref[ulong(indself)+10ul*g];
            const float fr0=r.forref[ulong(indfor-1)+4ul*g],fr1=r.forref[ulong(indfor)+4ul*g];
            const float tauself=selffac*(sr0+selffrac*(sr1-sr0));
            const float taufor=forfac*(fr0+forfrac*(fr1-fr0));
            const ulong m0=ulong(indminor-1)+19ul*g,m1=ulong(indminor)+19ul*g;
            const float absco2=r.ka_mco2[m0]+minorfrac*(r.ka_mco2[m1]-r.ka_mco2[m0]);
            const ulong a0=ulong(ind0-1)+65ul*g,a1=ulong(ind1-1)+65ul*g;
            float tmaj=fac00*r.absa[a0];tmaj=tmaj+fac10*r.absa[a0+1ul];
            tmaj=tmaj+fac01*r.absa[a1];tmaj=tmaj+fac11*r.absa[a1+1ul];
            float tau=colh2o*tmaj;tau=tau+tauself;tau=tau+taufor;
            tau=tau+adjcol*absco2;tau=tau+cfc11*r.cfc11adj[g];tau=tau+cfc12x*r.cfc12[g];
            r.taug[outbase+g]=tau;r.fracs[outbase+g]=r.fracrefa[g];
        }
    } else {
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            float tau=0.0f+cfc11*r.cfc11adj[g];tau=tau+cfc12x*r.cfc12[g];
            r.taug[outbase+g]=tau;r.fracs[outbase+g]=r.fracrefa[g];
        }
    }
}


// Band 7 from rlw_taugb7 in woof/core/kernels/rrtmg_lw_taugb06_09.cu.
kernel void rrtmg_legacy_lw_taugb7(
    constant RrtmgLegacyLwTaugb7Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb7Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncol*p.nl)) return;
    const int ncol=p.ncol,nl=p.nl;
    const int col=int(tid)/nl,lay=int(tid)%nl+1;
    const int laytrop=r.laytrop[col];
    const ulong cell=ulong(col)*ulong(nl)+ulong(lay-1);
    const ulong fss=ulong(ncol)*ulong(nl),iss=fss;
    const int jp=r.isv[cell],jt=r.isv[iss+cell],jt1=r.isv[2ul*iss+cell];
    const int indself=r.isv[3ul*iss+cell],indfor=r.isv[4ul*iss+cell],indminor=r.isv[5ul*iss+cell];
    const float oneminus=p.oneminus;
    const float colh2o=r.fs[2ul*fss+cell],colco2=r.fs[3ul*fss+cell];
    const float colo3=r.fs[4ul*fss+cell],coldry=r.fs[fss+cell];
    const float rat0=r.fs[16ul*fss+cell],rat1=r.fs[17ul*fss+cell];
    const float fac00=r.fs[10ul*fss+cell],fac01=r.fs[11ul*fss+cell];
    const float fac10=r.fs[12ul*fss+cell],fac11=r.fs[13ul*fss+cell];
    const float selffac=r.fs[26ul*fss+cell],selffrac=r.fs[27ul*fss+cell];
    const float forfac=r.fs[28ul*fss+cell],forfrac=r.fs[29ul*fss+cell];
    const float minorfrac=r.fs[30ul*fss+cell];
    const ulong outbase=cell*140ul+76ul;
    const int ng=12;
    if(lay<=laytrop) {
        const float refrat=r.chi_mls[0ul+7ul*2ul]/r.chi_mls[2ul+7ul*2ul];
        const RrtmgLegacySpec s0=rrtmg_legacy_spec(colh2o,rat0,colo3,oneminus,8.0f);
        const RrtmgLegacySpec s1=rrtmg_legacy_spec(colh2o,rat1,colo3,oneminus,8.0f);
        const RrtmgLegacySpec sm=rrtmg_legacy_spec(colh2o,refrat,colo3,oneminus,8.0f);
        const float chi=colco2/coldry;
        const float ratio=(1.0e20f*chi)/r.chi_mls[1ul+7ul*ulong(jp)];
        float adjcol=colco2;
        if(ratio>3.0f) {
            const float adjfac=3.0f+pow(ratio-3.0f,0.79f);
            adjcol=((adjfac*r.chi_mls[1ul+7ul*ulong(jp)])*coldry)*1.0e-20f;
        }
        const RrtmgLegacySpec spl=rrtmg_legacy_spec(colh2o,refrat,colo3,oneminus,8.0f);
        const int ind0=((jp-1)*5+(jt-1))*9+s0.j;
        const int ind1=(jp*5+(jt1-1))*9+s1.j;
        const RrtmgLegacyFac f0=rrtmg_legacy_fac(s0.parm,s0.f,fac00,fac10);
        const RrtmgLegacyFac f1=rrtmg_legacy_fac(s1.parm,s1.f,fac01,fac11);
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            const float sr0=r.selfref[ulong(indself-1)+10ul*g],sr1=r.selfref[ulong(indself)+10ul*g];
            const float fr0=r.forref[ulong(indfor-1)+4ul*g],fr1=r.forref[ulong(indfor)+4ul*g];
            const float tauself=selffac*(sr0+selffrac*(sr1-sr0));
            const float taufor=forfac*(fr0+forfrac*(fr1-fr0));
            const ulong ka0=ulong(sm.j-1)+9ul*ulong(indminor-1)+171ul*g;
            const ulong ka1=ulong(sm.j)+9ul*ulong(indminor-1)+171ul*g;
            const ulong ka2=ulong(sm.j-1)+9ul*ulong(indminor)+171ul*g;
            const ulong ka3=ulong(sm.j)+9ul*ulong(indminor)+171ul*g;
            const float co20=r.ka_mco2[ka0]+sm.f*(r.ka_mco2[ka1]-r.ka_mco2[ka0]);
            const float co21=r.ka_mco2[ka2]+sm.f*(r.ka_mco2[ka3]-r.ka_mco2[ka2]);
            const float absco2=co20+minorfrac*(co21-co20);
            float tau=rrtmg_legacy_major(f0,r.absa,ind0,ig,s0.comb);
            tau=tau+rrtmg_legacy_major(f1,r.absa,ind1,ig,s1.comb);
            tau=tau+tauself;tau=tau+taufor;tau=tau+adjcol*absco2;
            r.taug[outbase+g]=tau;
            const ulong f0i=g+12ul*ulong(spl.j-1),f1i=g+12ul*ulong(spl.j);
            const float f=r.fracrefa[f0i];r.fracs[outbase+g]=f+spl.f*(r.fracrefa[f1i]-f);
        }
    } else {
        const float chi=colco2/coldry;
        const float ratio=(1.0e20f*chi)/r.chi_mls[1ul+7ul*ulong(jp)];
        float adjcol=colco2;
        if(ratio>3.0f) {
            const float adjfac=2.0f+pow(ratio-2.0f,0.79f);
            adjcol=((adjfac*r.chi_mls[1ul+7ul*ulong(jp)])*coldry)*1.0e-20f;
        }
        const int ind0=((jp-13)*5+(jt-1))+1,ind1=((jp-12)*5+(jt1-1))+1;
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            const ulong m0=ulong(indminor-1)+19ul*g,m1=ulong(indminor)+19ul*g;
            const float absco2=r.kb_mco2[m0]+minorfrac*(r.kb_mco2[m1]-r.kb_mco2[m0]);
            const ulong b0=ulong(ind0-1)+235ul*g,b1=ulong(ind1-1)+235ul*g;
            float tmaj=fac00*r.absb[b0];tmaj=tmaj+fac10*r.absb[b0+1ul];
            tmaj=tmaj+fac01*r.absb[b1];tmaj=tmaj+fac11*r.absb[b1+1ul];
            float tau=colo3*tmaj;tau=tau+adjcol*absco2;
            if(ig>=6&&ig<=11) {
                const float scale[6]={0.92f,0.88f,1.07f,1.1f,0.99f,0.855f};
                tau=tau*scale[ig-6];
            }
            r.taug[outbase+g]=tau;r.fracs[outbase+g]=r.fracrefb[g];
        }
    }
}


// Band 8 from rlw_taugb8 in woof/core/kernels/rrtmg_lw_taugb06_09.cu.
kernel void rrtmg_legacy_lw_taugb8(
    constant RrtmgLegacyLwTaugb8Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb8Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncol*p.nl)) return;
    const int ncol=p.ncol,nl=p.nl;
    const int col=int(tid)/nl,lay=int(tid)%nl+1;
    const int laytrop=r.laytrop[col];
    const ulong cell=ulong(col)*ulong(nl)+ulong(lay-1), fss=ulong(ncol)*ulong(nl),iss=fss;
    const int jp=r.isv[cell],jt=r.isv[iss+cell],jt1=r.isv[2ul*iss+cell];
    const int indself=r.isv[3ul*iss+cell],indfor=r.isv[4ul*iss+cell],indminor=r.isv[5ul*iss+cell];
    const float colh2o=r.fs[2ul*fss+cell],colco2=r.fs[3ul*fss+cell];
    const float colo3=r.fs[4ul*fss+cell],coln2o=r.fs[5ul*fss+cell],coldry=r.fs[fss+cell];
    const float fac00=r.fs[10ul*fss+cell],fac01=r.fs[11ul*fss+cell];
    const float fac10=r.fs[12ul*fss+cell],fac11=r.fs[13ul*fss+cell];
    const float selffac=r.fs[26ul*fss+cell],selffrac=r.fs[27ul*fss+cell];
    const float forfac=r.fs[28ul*fss+cell],forfrac=r.fs[29ul*fss+cell];
    const float minorfrac=r.fs[30ul*fss+cell];
    const ulong wx3=ulong(col)*4ul*ulong(nl)+2ul*ulong(nl)+ulong(lay-1);
    const ulong wx4=ulong(col)*4ul*ulong(nl)+3ul*ulong(nl)+ulong(lay-1);
    const float cfc12x=r.wx[wx3],cfc22x=r.wx[wx4];
    const ulong outbase=cell*140ul+88ul;
    const int ng=8;
    const float chi=colco2/coldry;
    const float ratio=(1.0e20f*chi)/r.chi_mls[1ul+7ul*ulong(jp)];
    float adjcol=colco2;
    if(ratio>3.0f) {
        const float adjfac=2.0f+pow(ratio-2.0f,0.65f);
        adjcol=((adjfac*r.chi_mls[1ul+7ul*ulong(jp)])*coldry)*1.0e-20f;
    }
    if(lay<=laytrop) {
        const int ind0=((jp-1)*5+(jt-1))+1,ind1=(jp*5+(jt1-1))+1;
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            const float sr0=r.selfref[ulong(indself-1)+10ul*g],sr1=r.selfref[ulong(indself)+10ul*g];
            const float fr0=r.forref[ulong(indfor-1)+4ul*g],fr1=r.forref[ulong(indfor)+4ul*g];
            const float tauself=selffac*(sr0+selffrac*(sr1-sr0));
            const float taufor=forfac*(fr0+forfrac*(fr1-fr0));
            const ulong m0=ulong(indminor-1)+19ul*g,m1=ulong(indminor)+19ul*g;
            const float absco2=r.ka_mco2[m0]+minorfrac*(r.ka_mco2[m1]-r.ka_mco2[m0]);
            const float abso3=r.ka_mo3[m0]+minorfrac*(r.ka_mo3[m1]-r.ka_mo3[m0]);
            const float absn2o=r.ka_mn2o[m0]+minorfrac*(r.ka_mn2o[m1]-r.ka_mn2o[m0]);
            const ulong a0=ulong(ind0-1)+65ul*g,a1=ulong(ind1-1)+65ul*g;
            float tmaj=fac00*r.absa[a0];tmaj=tmaj+fac10*r.absa[a0+1ul];
            tmaj=tmaj+fac01*r.absa[a1];tmaj=tmaj+fac11*r.absa[a1+1ul];
            float tau=colh2o*tmaj;tau=tau+tauself;tau=tau+taufor;
            tau=tau+adjcol*absco2;tau=tau+colo3*abso3;tau=tau+coln2o*absn2o;
            tau=tau+cfc12x*r.cfc12[g];tau=tau+cfc22x*r.cfc22adj[g];
            r.taug[outbase+g]=tau;r.fracs[outbase+g]=r.fracrefa[g];
        }
    } else {
        const int ind0=((jp-13)*5+(jt-1))+1,ind1=((jp-12)*5+(jt1-1))+1;
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            const ulong m0=ulong(indminor-1)+19ul*g,m1=ulong(indminor)+19ul*g;
            const float absco2=r.kb_mco2[m0]+minorfrac*(r.kb_mco2[m1]-r.kb_mco2[m0]);
            const float absn2o=r.kb_mn2o[m0]+minorfrac*(r.kb_mn2o[m1]-r.kb_mn2o[m0]);
            const ulong b0=ulong(ind0-1)+235ul*g,b1=ulong(ind1-1)+235ul*g;
            float tmaj=fac00*r.absb[b0];tmaj=tmaj+fac10*r.absb[b0+1ul];
            tmaj=tmaj+fac01*r.absb[b1];tmaj=tmaj+fac11*r.absb[b1+1ul];
            float tau=colo3*tmaj;tau=tau+adjcol*absco2;tau=tau+coln2o*absn2o;
            tau=tau+cfc12x*r.cfc12[g];tau=tau+cfc22x*r.cfc22adj[g];
            r.taug[outbase+g]=tau;r.fracs[outbase+g]=r.fracrefb[g];
        }
    }
}


// Band 9 from rlw_taugb9 in woof/core/kernels/rrtmg_lw_taugb06_09.cu.
kernel void rrtmg_legacy_lw_taugb9(
    constant RrtmgLegacyLwTaugb9Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb9Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncol*p.nl)) return;
    const int ncol=p.ncol,nl=p.nl;
    const int col=int(tid)/nl,lay=int(tid)%nl+1;
    const int laytrop=r.laytrop[col];
    const ulong cell=ulong(col)*ulong(nl)+ulong(lay-1),fss=ulong(ncol)*ulong(nl),iss=fss;
    const int jp=r.isv[cell],jt=r.isv[iss+cell],jt1=r.isv[2ul*iss+cell];
    const int indself=r.isv[3ul*iss+cell],indfor=r.isv[4ul*iss+cell],indminor=r.isv[5ul*iss+cell];
    const float oneminus=p.oneminus;
    const float colh2o=r.fs[2ul*fss+cell],coln2o=r.fs[5ul*fss+cell];
    const float colch4=r.fs[7ul*fss+cell],coldry=r.fs[fss+cell];
    const float rat0=r.fs[20ul*fss+cell],rat1=r.fs[21ul*fss+cell];
    const float fac00=r.fs[10ul*fss+cell],fac01=r.fs[11ul*fss+cell];
    const float fac10=r.fs[12ul*fss+cell],fac11=r.fs[13ul*fss+cell];
    const float selffac=r.fs[26ul*fss+cell],selffrac=r.fs[27ul*fss+cell];
    const float forfac=r.fs[28ul*fss+cell],forfrac=r.fs[29ul*fss+cell],minorfrac=r.fs[30ul*fss+cell];
    const ulong outbase=cell*140ul+96ul;
    const int ng=12;
    const float chi_n2o=coln2o/coldry;
    const float ratio=(1.0e20f*chi_n2o)/r.chi_mls[3ul+7ul*ulong(jp)];
    float adjcoln2o=coln2o;
    if(ratio>1.5f) {
        const float adjfac=0.5f+pow(ratio-0.5f,0.65f);
        adjcoln2o=((adjfac*r.chi_mls[3ul+7ul*ulong(jp)])*coldry)*1.0e-20f;
    }
    if(lay<=laytrop) {
        const float refp=r.chi_mls[0ul+7ul*8ul]/r.chi_mls[5ul+7ul*8ul];
        const float refm=r.chi_mls[0ul+7ul*2ul]/r.chi_mls[5ul+7ul*2ul];
        const RrtmgLegacySpec s0=rrtmg_legacy_spec(colh2o,rat0,colch4,oneminus,8.0f);
        const RrtmgLegacySpec s1=rrtmg_legacy_spec(colh2o,rat1,colch4,oneminus,8.0f);
        const RrtmgLegacySpec sm=rrtmg_legacy_spec(colh2o,refm,colch4,oneminus,8.0f);
        const RrtmgLegacySpec spl=rrtmg_legacy_spec(colh2o,refp,colch4,oneminus,8.0f);
        const int ind0=((jp-1)*5+(jt-1))*9+s0.j,ind1=(jp*5+(jt1-1))*9+s1.j;
        const RrtmgLegacyFac f0=rrtmg_legacy_fac(s0.parm,s0.f,fac00,fac10);
        const RrtmgLegacyFac f1=rrtmg_legacy_fac(s1.parm,s1.f,fac01,fac11);
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            const float sr0=r.selfref[ulong(indself-1)+10ul*g],sr1=r.selfref[ulong(indself)+10ul*g];
            const float fr0=r.forref[ulong(indfor-1)+4ul*g],fr1=r.forref[ulong(indfor)+4ul*g];
            const float tauself=selffac*(sr0+selffrac*(sr1-sr0));
            const float taufor=forfac*(fr0+forfrac*(fr1-fr0));
            const ulong ka0=ulong(sm.j-1)+9ul*ulong(indminor-1)+171ul*g;
            const ulong ka1=ulong(sm.j)+9ul*ulong(indminor-1)+171ul*g;
            const ulong ka2=ulong(sm.j-1)+9ul*ulong(indminor)+171ul*g;
            const ulong ka3=ulong(sm.j)+9ul*ulong(indminor)+171ul*g;
            const float n20=r.ka_mn2o[ka0]+sm.f*(r.ka_mn2o[ka1]-r.ka_mn2o[ka0]);
            const float n21=r.ka_mn2o[ka2]+sm.f*(r.ka_mn2o[ka3]-r.ka_mn2o[ka2]);
            const float absn2o=n20+minorfrac*(n21-n20);
            float tau=rrtmg_legacy_major(f0,r.absa,ind0,ig,s0.comb);
            tau=tau+rrtmg_legacy_major(f1,r.absa,ind1,ig,s1.comb);
            tau=tau+tauself;tau=tau+taufor;tau=tau+adjcoln2o*absn2o;
            r.taug[outbase+g]=tau;
            const ulong f0i=g+12ul*ulong(spl.j-1),f1i=g+12ul*ulong(spl.j);
            const float f=r.fracrefa[f0i];r.fracs[outbase+g]=f+spl.f*(r.fracrefa[f1i]-f);
        }
    } else {
        const int ind0=((jp-13)*5+(jt-1))+1,ind1=((jp-12)*5+(jt1-1))+1;
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1),m0=ulong(indminor-1)+19ul*g,m1=ulong(indminor)+19ul*g;
            const float absn2o=r.kb_mn2o[m0]+minorfrac*(r.kb_mn2o[m1]-r.kb_mn2o[m0]);
            const ulong b0=ulong(ind0-1)+235ul*g,b1=ulong(ind1-1)+235ul*g;
            float tmaj=fac00*r.absb[b0];tmaj=tmaj+fac10*r.absb[b0+1ul];
            tmaj=tmaj+fac01*r.absb[b1];tmaj=tmaj+fac11*r.absb[b1+1ul];
            float tau=colch4*tmaj;tau=tau+adjcoln2o*absn2o;
            r.taug[outbase+g]=tau;r.fracs[outbase+g]=r.fracrefb[g];
        }
    }
}


// Band 14 from rlw_taugb14 in woof/core/kernels/rrtmg_lw_taugb13_16.cu.
kernel void rrtmg_legacy_lw_taugb14(
    constant RrtmgLegacyLwTaugb14Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb14Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncol*p.nl))return;
    const int ncol=p.ncol,nl=p.nl;
    const int col=int(tid)/nl,lay=int(tid)%nl+1;
    const int laytrop=r.laytrop[col];
    const ulong cell=ulong(col)*ulong(nl)+ulong(lay-1),fss=ulong(ncol)*ulong(nl),iss=fss;
    const int jp=r.isv[cell],jt=r.isv[iss+cell],jt1=r.isv[2ul*iss+cell];
    const int indself=r.isv[3ul*iss+cell],indfor=r.isv[4ul*iss+cell];
    const float colco2=r.fs[3ul*fss+cell],fac00=r.fs[10ul*fss+cell];
    const float fac01=r.fs[11ul*fss+cell],fac10=r.fs[12ul*fss+cell],fac11=r.fs[13ul*fss+cell];
    const ulong outbase=cell*140ul+134ul;
    const int ng=2;
    if(lay<=laytrop) {
        const int ind0=((jp-1)*5+(jt-1))+1,ind1=(jp*5+(jt1-1))+1;
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            const float sr0=r.selfref[ulong(indself-1)+10ul*g],sr1=r.selfref[ulong(indself)+10ul*g];
            const float fr0=r.forref[ulong(indfor-1)+4ul*g],fr1=r.forref[ulong(indfor)+4ul*g];
            const float tauself=r.fs[26ul*fss+cell]*(sr0+r.fs[27ul*fss+cell]*(sr1-sr0));
            const float taufor=r.fs[28ul*fss+cell]*(fr0+r.fs[29ul*fss+cell]*(fr1-fr0));
            const ulong a0=ulong(ind0-1)+65ul*g,a1=ulong(ind1-1)+65ul*g;
            float tmaj=fac00*r.absa[a0];tmaj=tmaj+fac10*r.absa[a0+1ul];
            tmaj=tmaj+fac01*r.absa[a1];tmaj=tmaj+fac11*r.absa[a1+1ul];
            float tau=colco2*tmaj;tau=tau+tauself;tau=tau+taufor;
            r.taug[outbase+g]=tau;r.fracs[outbase+g]=r.fracrefa[g];
        }
    } else {
        const int ind0=((jp-13)*5+(jt-1))+1,ind1=((jp-12)*5+(jt1-1))+1;
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1),b0=ulong(ind0-1)+235ul*g,b1=ulong(ind1-1)+235ul*g;
            float tmaj=fac00*r.absb[b0];tmaj=tmaj+fac10*r.absb[b0+1ul];
            tmaj=tmaj+fac01*r.absb[b1];tmaj=tmaj+fac11*r.absb[b1+1ul];
            r.taug[outbase+g]=colco2*tmaj;r.fracs[outbase+g]=r.fracrefb[g];
        }
    }
}


// Band 15 from rlw_taugb15 in woof/core/kernels/rrtmg_lw_taugb13_16.cu.
kernel void rrtmg_legacy_lw_taugb15(
    constant RrtmgLegacyLwTaugb15Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb15Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncol*p.nl))return;
    const int ncol=p.ncol,nl=p.nl;
    const int col=int(tid)/nl,lay=int(tid)%nl+1;
    const int laytrop=r.laytrop[col];
    const ulong cell=ulong(col)*ulong(nl)+ulong(lay-1),fss=ulong(ncol)*ulong(nl),iss=fss;
    const int jp=r.isv[cell],jt=r.isv[iss+cell],jt1=r.isv[2ul*iss+cell];
    const int indself=r.isv[3ul*iss+cell],indfor=r.isv[4ul*iss+cell],indminor=r.isv[5ul*iss+cell];
    const float oneminus=p.oneminus;
    const float coln2o=r.fs[5ul*fss+cell],colco2=r.fs[3ul*fss+cell];
    const float fac00=r.fs[10ul*fss+cell],fac01=r.fs[11ul*fss+cell];
    const float fac10=r.fs[12ul*fss+cell],fac11=r.fs[13ul*fss+cell];
    const float selffac=r.fs[26ul*fss+cell],selffrac=r.fs[27ul*fss+cell];
    const float forfac=r.fs[28ul*fss+cell],forfrac=r.fs[29ul*fss+cell];
    const float minorfrac=r.fs[30ul*fss+cell];
    const float scalen2=r.fs[9ul*fss+cell]*r.fs[31ul*fss+cell];
    const ulong outbase=cell*140ul+136ul;
    const int ng=2;
    if(lay<=laytrop) {
        const float refrat=r.chi_mls[3ul]/r.chi_mls[1ul];
        const RrtmgLegacySpec s0=rrtmg_legacy_spec(coln2o,r.fs[22ul*fss+cell],colco2,oneminus,8.0f);
        const RrtmgLegacySpec s1=rrtmg_legacy_spec(coln2o,r.fs[23ul*fss+cell],colco2,oneminus,8.0f);
        const RrtmgLegacySpec sm=rrtmg_legacy_spec(coln2o,refrat,colco2,oneminus,8.0f);
        const RrtmgLegacySpec spl=rrtmg_legacy_spec(coln2o,refrat,colco2,oneminus,8.0f);
        const int ind0=((jp-1)*5+(jt-1))*9+s0.j,ind1=(jp*5+(jt1-1))*9+s1.j;
        const RrtmgLegacyFac f0=rrtmg_legacy_fac(s0.parm,s0.f,fac00,fac10);
        const RrtmgLegacyFac f1=rrtmg_legacy_fac(s1.parm,s1.f,fac01,fac11);
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            const float sr0=r.selfref[ulong(indself-1)+10ul*g],sr1=r.selfref[ulong(indself)+10ul*g];
            const float fr0=r.forref[ulong(indfor-1)+4ul*g],fr1=r.forref[ulong(indfor)+4ul*g];
            const float tauself=selffac*(sr0+selffrac*(sr1-sr0));
            const float taufor=forfac*(fr0+forfrac*(fr1-fr0));
            const ulong ka0=ulong(sm.j-1)+9ul*ulong(indminor-1)+171ul*g;
            const ulong ka1=ulong(sm.j)+9ul*ulong(indminor-1)+171ul*g;
            const ulong ka2=ulong(sm.j-1)+9ul*ulong(indminor)+171ul*g;
            const ulong ka3=ulong(sm.j)+9ul*ulong(indminor)+171ul*g;
            const float n20=r.ka_mn2[ka0]+sm.f*(r.ka_mn2[ka1]-r.ka_mn2[ka0]);
            const float n21=r.ka_mn2[ka2]+sm.f*(r.ka_mn2[ka3]-r.ka_mn2[ka2]);
            const float taun2=scalen2*(n20+minorfrac*(n21-n20));
            float tau=rrtmg_legacy_major(f0,r.absa,ind0,ig,s0.comb);
            tau=tau+rrtmg_legacy_major(f1,r.absa,ind1,ig,s1.comb);
            tau=tau+tauself;tau=tau+taufor;tau=tau+taun2;
            r.taug[outbase+g]=tau;
            const ulong f0i=g+2ul*ulong(spl.j-1),f1i=g+2ul*ulong(spl.j);
            const float f=r.fracrefa[f0i];r.fracs[outbase+g]=f+spl.f*(r.fracrefa[f1i]-f);
        }
    } else {
        for(int ig=1;ig<=ng;++ig) { const ulong g=ulong(ig-1);r.taug[outbase+g]=0.0f;r.fracs[outbase+g]=0.0f; }
    }
}


// Band 13 from rlw_taugb13 in woof/core/kernels/rrtmg_lw_taugb13_16.cu.
kernel void rrtmg_legacy_lw_taugb13(
    constant RrtmgLegacyLwTaugb13Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb13Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncol*p.nl))return;
    const int ncol=p.ncol,nl=p.nl;
    const int col=int(tid)/nl,lay=int(tid)%nl+1;
    const int laytrop=r.laytrop[col];
    const ulong cell=ulong(col)*ulong(nl)+ulong(lay-1),fss=ulong(ncol)*ulong(nl),iss=fss;
    const int jp=r.isv[cell],jt=r.isv[iss+cell],jt1=r.isv[2ul*iss+cell];
    const int indself=r.isv[3ul*iss+cell],indfor=r.isv[4ul*iss+cell],indminor=r.isv[5ul*iss+cell];
    const float oneminus=p.oneminus;
    const float colh2o=r.fs[2ul*fss+cell],coln2o=r.fs[5ul*fss+cell];
    const float coldry=r.fs[fss+cell],colco2=r.fs[3ul*fss+cell];
    const float fac00=r.fs[10ul*fss+cell],fac01=r.fs[11ul*fss+cell];
    const float fac10=r.fs[12ul*fss+cell],fac11=r.fs[13ul*fss+cell];
    const float selffac=r.fs[26ul*fss+cell],selffrac=r.fs[27ul*fss+cell];
    const float forfac=r.fs[28ul*fss+cell],forfrac=r.fs[29ul*fss+cell];
    const float minorfrac=r.fs[30ul*fss+cell];
    const ulong outbase=cell*140ul+130ul;
    const int ng=4;
    if(lay<=laytrop) {
        const float refratp=r.chi_mls[28ul]/r.chi_mls[31ul];
        const float refratm=r.chi_mls[0ul]/r.chi_mls[3ul];
        const float refratm3=r.chi_mls[14ul]/r.chi_mls[17ul];
        const RrtmgLegacySpec s0=rrtmg_legacy_spec(colh2o,r.fs[18ul*fss+cell],coln2o,oneminus,8.0f);
        const RrtmgLegacySpec s1=rrtmg_legacy_spec(colh2o,r.fs[19ul*fss+cell],coln2o,oneminus,8.0f);
        const RrtmgLegacySpec smco2=rrtmg_legacy_spec(colh2o,refratm,coln2o,oneminus,8.0f);
        const RrtmgLegacySpec smco=rrtmg_legacy_spec(colh2o,refratm3,coln2o,oneminus,8.0f);
        const RrtmgLegacySpec spl=rrtmg_legacy_spec(colh2o,refratp,coln2o,oneminus,8.0f);
        const float chi_co2=colco2/coldry;
        const float ratio=(1.0e20f*chi_co2)/3.55e-4f;
        float adjcolco2=colco2;
        if(ratio>3.0f) {
            const float adjfac=2.0f+pow(ratio-2.0f,0.68f);
            adjcolco2=((adjfac*3.55e-4f)*coldry)*1.0e-20f;
        }
        const int ind0=((jp-1)*5+(jt-1))*9+s0.j;
        const int ind1=(jp*5+(jt1-1))*9+s1.j;
        const RrtmgLegacyFac f0=rrtmg_legacy_fac(s0.parm,s0.f,fac00,fac10);
        const RrtmgLegacyFac f1=rrtmg_legacy_fac(s1.parm,s1.f,fac01,fac11);
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            const float sr0=r.selfref[ulong(indself-1)+10ul*g],sr1=r.selfref[ulong(indself)+10ul*g];
            const float fr0=r.forref[ulong(indfor-1)+4ul*g],fr1=r.forref[ulong(indfor)+4ul*g];
            const float tauself=selffac*(sr0+selffrac*(sr1-sr0));
            const float taufor=forfac*(fr0+forfrac*(fr1-fr0));
            const ulong c20=ulong(smco2.j-1)+9ul*ulong(indminor-1)+171ul*g;
            const ulong c21=ulong(smco2.j)+9ul*ulong(indminor-1)+171ul*g;
            const ulong c22=ulong(smco2.j-1)+9ul*ulong(indminor)+171ul*g;
            const ulong c23=ulong(smco2.j)+9ul*ulong(indminor)+171ul*g;
            const float co20=r.ka_mco2[c20]+smco2.f*(r.ka_mco2[c21]-r.ka_mco2[c20]);
            const float co21=r.ka_mco2[c22]+smco2.f*(r.ka_mco2[c23]-r.ka_mco2[c22]);
            const float absco2=co20+minorfrac*(co21-co20);
            const ulong co0=ulong(smco.j-1)+9ul*ulong(indminor-1)+171ul*g;
            const ulong co1=ulong(smco.j)+9ul*ulong(indminor-1)+171ul*g;
            const ulong co2=ulong(smco.j-1)+9ul*ulong(indminor)+171ul*g;
            const ulong co3=ulong(smco.j)+9ul*ulong(indminor)+171ul*g;
            const float co00=r.ka_mco[co0]+smco.f*(r.ka_mco[co1]-r.ka_mco[co0]);
            const float co01=r.ka_mco[co2]+smco.f*(r.ka_mco[co3]-r.ka_mco[co2]);
            const float absco=co00+minorfrac*(co01-co00);
            float tau=rrtmg_legacy_major(f0,r.absa,ind0,ig,s0.comb);
            tau=tau+rrtmg_legacy_major(f1,r.absa,ind1,ig,s1.comb);
            tau=tau+tauself;tau=tau+taufor;
            tau=tau+adjcolco2*absco2;tau=tau+r.fs[6ul*fss+cell]*absco;
            r.taug[outbase+g]=tau;
            const ulong fi0=g+4ul*ulong(spl.j-1),fi1=g+4ul*ulong(spl.j);
            r.fracs[outbase+g]=r.fracrefa[fi0]+spl.f*(r.fracrefa[fi1]-r.fracrefa[fi0]);
        }
    } else {
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1),m0=ulong(indminor-1)+19ul*g,m1=ulong(indminor)+19ul*g;
            const float o30=r.kb_mo3[m0],o31=r.kb_mo3[m1];
            const float abso3=o30+minorfrac*(o31-o30);
            r.taug[outbase+g]=r.fs[4ul*fss+cell]*abso3;
            r.fracs[outbase+g]=r.fracrefb[g];
        }
    }
}


// Band 16 from rlw_taugb16 in woof/core/kernels/rrtmg_lw_taugb13_16.cu.
kernel void rrtmg_legacy_lw_taugb16(
    constant RrtmgLegacyLwTaugb16Resources &r [[buffer(0)]],
    constant RrtmgLegacyLwTaugb16Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncol*p.nl))return;
    const int ncol=p.ncol,nl=p.nl;
    const int col=int(tid)/nl,lay=int(tid)%nl+1;
    const int laytrop=r.laytrop[col];
    const ulong cell=ulong(col)*ulong(nl)+ulong(lay-1),fss=ulong(ncol)*ulong(nl),iss=fss;
    const int jp=r.isv[cell],jt=r.isv[iss+cell],jt1=r.isv[2ul*iss+cell];
    const int indself=r.isv[3ul*iss+cell],indfor=r.isv[4ul*iss+cell];
    const float oneminus=p.oneminus;
    const float colh2o=r.fs[2ul*fss+cell],colch4=r.fs[7ul*fss+cell];
    const float fac00=r.fs[10ul*fss+cell],fac01=r.fs[11ul*fss+cell];
    const float fac10=r.fs[12ul*fss+cell],fac11=r.fs[13ul*fss+cell];
    const ulong outbase=cell*140ul+138ul;
    const int ng=2;
    if(lay<=laytrop) {
        const float refratp=r.chi_mls[35ul]/r.chi_mls[40ul];
        const RrtmgLegacySpec s0=rrtmg_legacy_spec(colh2o,r.fs[20ul*fss+cell],colch4,oneminus,8.0f);
        const RrtmgLegacySpec s1=rrtmg_legacy_spec(colh2o,r.fs[21ul*fss+cell],colch4,oneminus,8.0f);
        const RrtmgLegacySpec spl=rrtmg_legacy_spec(colh2o,refratp,colch4,oneminus,8.0f);
        const int ind0=((jp-1)*5+(jt-1))*9+s0.j;
        const int ind1=(jp*5+(jt1-1))*9+s1.j;
        const RrtmgLegacyFac f0=rrtmg_legacy_fac(s0.parm,s0.f,fac00,fac10);
        const RrtmgLegacyFac f1=rrtmg_legacy_fac(s1.parm,s1.f,fac01,fac11);
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1);
            const float sr0=r.selfref[ulong(indself-1)+10ul*g],sr1=r.selfref[ulong(indself)+10ul*g];
            const float fr0=r.forref[ulong(indfor-1)+4ul*g],fr1=r.forref[ulong(indfor)+4ul*g];
            const float tauself=r.fs[26ul*fss+cell]*(sr0+r.fs[27ul*fss+cell]*(sr1-sr0));
            const float taufor=r.fs[28ul*fss+cell]*(fr0+r.fs[29ul*fss+cell]*(fr1-fr0));
            float tau=rrtmg_legacy_major(f0,r.absa,ind0,ig,s0.comb);
            tau=tau+rrtmg_legacy_major(f1,r.absa,ind1,ig,s1.comb);
            tau=tau+tauself;tau=tau+taufor;
            r.taug[outbase+g]=tau;
            const ulong fi0=g+2ul*ulong(spl.j-1),fi1=g+2ul*ulong(spl.j);
            r.fracs[outbase+g]=r.fracrefa[fi0]+spl.f*(r.fracrefa[fi1]-r.fracrefa[fi0]);
        }
    } else {
        // This WRF copy has nspb16=0: both pressure/temperature key
        // expressions collapse to table rows 1 and 2.
        for(int ig=1;ig<=ng;++ig) {
            const ulong g=ulong(ig-1),base=235ul*g;
            float tmaj=fac00*r.absb[base];
            tmaj=tmaj+fac10*r.absb[base+1ul];
            tmaj=tmaj+fac01*r.absb[base];
            tmaj=tmaj+fac11*r.absb[base+1ul];
            r.taug[outbase+g]=colch4*tmaj;
            r.fracs[outbase+g]=r.fracrefb[g];
        }
    }
}


kernel void rrtmg_legacy_lw_rtrn_secdiff(
    constant RrtmgLegacyLwRtrnSecdiffResources &r [[buffer(0)]],
    constant RrtmgLegacyLwRtrnSecdiffParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if(col>=uint(p.ncol))return;
    const float pwvcm=r.pwvcm[col];
    for(uint ib=0;ib<16u;++ib) {
        float sd;
        if(ib==0u||ib==3u||ib>=9u) sd=1.66f;
        else {
            sd=r.a0[ib]+r.a1[ib]*precise::exp(r.a2[ib]*pwvcm);
            sd=clamp(sd,1.50f,1.80f);
        }
        r.secdiff[ulong(col)*16ul+ulong(ib)]=sd;
    }
}

kernel void rrtmg_legacy_lw_rtrn_prol(
    constant RrtmgLegacyLwRtrnProlResources &r [[buffer(0)]],
    constant RrtmgLegacyLwRtrnProlParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncol*p.nl))return;
    const int col=int(tid)/p.nl,lay=int(tid)%p.nl;
    const ulong cell=ulong(col)*ulong(p.nl)+ulong(lay);
    int flag=0;
    for(uint ig=0;ig<140u;++ig) {
        const ulong q=(ulong(col)*140ul+ulong(ig))*ulong(p.nl)+ulong(lay);
        const float cf=r.cldfmc[q];
        if(cf==1.0f) {
            const int ib=r.ngb[ig]-1;
            const float od=r.secdiff[ulong(col)*16ul+ulong(ib)]*r.taucmc[q];
            r.odcld[q]=od;
            const float transcld=precise::exp(-od);
            r.efclfrac[q]=(1.0f-transcld)*cf;
            flag=1;
        } else {
            r.odcld[q]=0.0f;
            r.efclfrac[q]=0.0f;
        }
    }
    r.icldlyr[cell]=flag;
}


kernel void rrtmg_legacy_lw_rtrn_accum(
    constant RrtmgLegacyLwRtrnAccumResources &r [[buffer(0)]],
    constant RrtmgLegacyLwRtrnAccumParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong total=ulong(p.ncol)*ulong(p.nl+1);
    if(ulong(tid)>=total)return;
    const int col=int(tid)/ (p.nl+1),lev=int(tid)% (p.nl+1);
    float totu=0.0f,totd=0.0f,totuc=0.0f,totdc=0.0f;
    for(int iband=0;iband<16;++iband) {
        const int glo=iband==0?0:r.ngs[iband-1];
        const int ghi=r.ngs[iband];
        float urad=0.0f,drad=0.0f,clru=0.0f,clrd=0.0f;
        for(int lane=glo;lane<ghi;++lane) {
            const ulong q=(ulong(col)*140ul+ulong(lane))*ulong(p.nl);
            if(lev<=p.nl-1) {
                drad=drad+r.radld_p[q+ulong(lev)];
                if(r.iclddn_p[q+ulong(lev)]!=0u)
                    clrd=clrd+r.radclrd_p[q+ulong(lev)];
                else clrd=drad;
            }
            if(lev==0) {
                urad=urad+r.radlu_sfc[ulong(col)*140ul+ulong(lane)];
                if(r.iclddn_p[q]!=0u)
                    clru=clru+r.radclru_sfc[ulong(col)*140ul+ulong(lane)];
                else clru=urad;
            } else {
                urad=urad+r.radlu_p[q+ulong(lev-1)];
                if(r.iclddn_p[q]!=0u) clru=clru+r.radclru_p[q+ulong(lev-1)];
                else clru=urad;
            }
        }
        const float dw=r.delwave[iband];
        totu=totu+(urad*p.wtdiff)*dw;
        totd=totd+(drad*p.wtdiff)*dw;
        totuc=totuc+(clru*p.wtdiff)*dw;
        totdc=totdc+(clrd*p.wtdiff)*dw;
    }
    const ulong out=ulong(col)*ulong(p.nl+1)+ulong(lev);
    r.totuflux[out]=totu;r.totdflux[out]=totd;
    r.totuclfl[out]=totuc;r.totdclfl[out]=totdc;
}

kernel void rrtmg_legacy_lw_rtrn_final(
    constant RrtmgLegacyLwRtrnFinalResources &r [[buffer(0)]],
    constant RrtmgLegacyLwRtrnFinalParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if(col>=uint(p.ncol))return;
    const ulong base=ulong(col)*ulong(p.nl+1);
    r.totuflux[base]=r.totuflux[base]*p.fluxfac;
    r.totdflux[base]=r.totdflux[base]*p.fluxfac;
    r.fnet[base]=r.totuflux[base]-r.totdflux[base];
    r.totuclfl[base]=r.totuclfl[base]*p.fluxfac;
    r.totdclfl[base]=r.totdclfl[base]*p.fluxfac;
    r.fnetc[base]=r.totuclfl[base]-r.totdclfl[base];
    for(int lev=1;lev<=p.nl;++lev) {
        const ulong q=base+ulong(lev),prev=q-1ul;
        r.totuflux[q]=r.totuflux[q]*p.fluxfac;
        r.totdflux[q]=r.totdflux[q]*p.fluxfac;
        r.fnet[q]=r.totuflux[q]-r.totdflux[q];
        r.totuclfl[q]=r.totuclfl[q]*p.fluxfac;
        r.totdclfl[q]=r.totdclfl[q]*p.fluxfac;
        r.fnetc[q]=r.totuclfl[q]-r.totdclfl[q];
        r.htr[prev]=(p.heatfac*(r.fnet[prev]-r.fnet[q]))/(r.pz[prev]-r.pz[q]);
        r.htrc[prev]=(p.heatfac*(r.fnetc[prev]-r.fnetc[q]))/(r.pz[prev]-r.pz[q]);
    }
    r.htr[base+ulong(p.nl)]=0.0f;
    r.htrc[base+ulong(p.nl)]=0.0f;
}


kernel void rrtmg_legacy_lw_rtrn_march(
    constant RrtmgLegacyLwRtrnMarchResources &r [[buffer(0)]],
    constant RrtmgLegacyLwRtrnMarchParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncol*140))return;
    if(p.nl>128)return;
    const int col=int(tid)/140,ig=int(tid)%140;
    const int iband=r.ngb[ig]-1;
    const ulong colg=ulong(col)*140ul+ulong(ig);
    const ulong laybase=colg*ulong(p.nl);
    const ulong layrow=ulong(col)*ulong(p.nl);
    const ulong laylev=ulong(col)*ulong(p.nl+1);
    const float sd=r.secdiff[ulong(col)*16ul+ulong(iband)];
    float atrans[128],atot[128],bbugas[128],bbutot[128];
    float radld=0.0f,radclrd=0.0f;
    int iclddn=0;
    for(int lev=p.nl;lev>=1;--lev) {
        const ulong li=ulong(lev-1), cell=layrow+li;
        const ulong plume=laybase+li;
        const float plfrac=r.fracs[cell*140ul+ulong(ig)];
        const float blay=r.planklay[cell*16ul+ulong(iband)];
        const float dplankup=r.planklev[(laylev+ulong(lev))*16ul+ulong(iband)]-blay;
        const float dplankdn=r.planklev[(laylev+li)*16ul+ulong(iband)]-blay;
        float odepth=sd*r.taut[cell*140ul+ulong(ig)];
        if(odepth<0.0f)odepth=0.0f;
        float bbd;
        if(r.icldlyr[cell]==1) {
            iclddn=1;
            const float oc=r.odcld[plume];
            const float odtot=odepth+oc;
            float gassrc,bbdtot,at,ao,bug,but;
            if(odtot<0.06f) {
                at=odepth-(0.5f*(odepth*odepth));
                const float od_rec=0.166667f*odepth;
                gassrc=(plfrac*(blay+dplankdn*od_rec))*at;
                ao=odtot-(0.5f*(odtot*odtot));
                const float odtot_rec=0.166667f*odtot;
                bbdtot=plfrac*(blay+dplankdn*odtot_rec);
                bbd=plfrac*(blay+dplankdn*od_rec);
                bug=plfrac*(blay+dplankup*od_rec);
                but=plfrac*(blay+dplankup*odtot_rec);
            } else if(odepth<=0.06f) {
                at=odepth-(0.5f*(odepth*odepth));
                const float od_rec=0.166667f*odepth;
                gassrc=(plfrac*(blay+dplankdn*od_rec))*at;
                const float tblind=odtot/(p.bpade+odtot);
                const int ittot=int(10000.0f*tblind+0.5f);
                const float tfactot=r.tfn_tbl[ittot];
                bbdtot=plfrac*(blay+tfactot*dplankdn);
                ao=1.0f-r.exp_tbl[ittot];
                bbd=plfrac*(blay+dplankdn*od_rec);
                bug=plfrac*(blay+dplankup*od_rec);
                but=plfrac*(blay+tfactot*dplankup);
            } else {
                const float tblind=odepth/(p.bpade+odepth);
                const int itgas=int(10000.0f*tblind+0.5f);
                odepth=r.tau_tbl[itgas];
                at=1.0f-r.exp_tbl[itgas];
                const float tfacgas=r.tfn_tbl[itgas];
                gassrc=(at*plfrac)*(blay+tfacgas*dplankdn);
                const float odtot2=odepth+oc;
                const float tblind2=odtot2/(p.bpade+odtot2);
                const int ittot=int(10000.0f*tblind2+0.5f);
                const float tfactot=r.tfn_tbl[ittot];
                bbdtot=plfrac*(blay+tfactot*dplankdn);
                ao=1.0f-r.exp_tbl[ittot];
                bbd=plfrac*(blay+tfacgas*dplankdn);
                bug=plfrac*(blay+tfacgas*dplankup);
                but=plfrac*(blay+tfactot*dplankup);
            }
            const float cf=r.cldfmc[plume];
            const float ef=r.efclfrac[plume];
            radld=radld-radld*(at+ef*(1.0f-at));
            radld=radld+gassrc;
            radld=radld+cf*(bbdtot*ao-gassrc);
            atrans[li]=at;atot[li]=ao;bbugas[li]=bug;bbutot[li]=but;
        } else {
            float at,bug;
            if(odepth<=0.06f) {
                at=odepth-(0.5f*(odepth*odepth));
                const float od2=0.166667f*odepth;
                bbd=plfrac*(blay+dplankdn*od2);
                bug=plfrac*(blay+dplankup*od2);
            } else {
                const float tblind=odepth/(p.bpade+odepth);
                const int itr=int(10000.0f*tblind+0.5f);
                at=1.0f-r.exp_tbl[itr];
                const float tausfac=r.tfn_tbl[itr];
                bbd=plfrac*(blay+tausfac*dplankdn);
                bug=plfrac*(blay+tausfac*dplankup);
            }
            radld=radld+(bbd-radld)*at;
            atrans[li]=at;bbugas[li]=bug;
        }
        r.radld_p[plume]=radld;
        if(iclddn==1)radclrd=radclrd+(bbd-radclrd)*atrans[li];
        else radclrd=radld;
        r.radclrd_p[plume]=radclrd;
        r.iclddn_p[plume]=uint(iclddn);
    }
    const float rad0=r.fracs[layrow*140ul+ulong(ig)]
                    *r.plankbnd[ulong(col)*16ul+ulong(iband)];
    const float reflect=1.0f-r.semiss[ulong(col)*16ul+ulong(iband)];
    float radlu=rad0+reflect*radld;
    float radclru=rad0+reflect*radclrd;
    r.radlu_sfc[colg]=radlu;r.radclru_sfc[colg]=radclru;
    for(int lev=1;lev<=p.nl;++lev) {
        const ulong li=ulong(lev-1),cell=layrow+li,plume=laybase+li;
        if(r.icldlyr[cell]==1) {
            const float gassrc=bbugas[li]*atrans[li];
            const float cf=r.cldfmc[plume],ef=r.efclfrac[plume];
            radlu=radlu-radlu*(atrans[li]+ef*(1.0f-atrans[li]));
            radlu=radlu+gassrc;
            radlu=radlu+cf*(bbutot[li]*atot[li]-gassrc);
        } else radlu=radlu+(bbugas[li]-radlu)*atrans[li];
        r.radlu_p[plume]=radlu;
        if(iclddn==1)radclru=radclru+(bbugas[li]-radclru)*atrans[li];
        else radclru=radlu;
        r.radclru_p[plume]=radclru;
    }
}


kernel void rrtmg_legacy_lw_cldprmc(
    constant RrtmgLegacyLwCldprmcResources &r [[buffer(0)]],
    constant RrtmgLegacyLwCldprmcParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if(col>=uint(p.ncol))return;
    r.ncb_flag[col]=0;
    r.err_flag[col]=0;
    const float cldmin=1.0e-20f;
    for(int lay=0;lay<p.nl;++lay) {
        const ulong layer=ulong(col)*ulong(p.nl)+ulong(lay);
        const float radice=r.reicmc[layer];
        for(int ig=0;ig<140;++ig) {
            const ulong cell=(ulong(col)*140ul+ulong(ig))*ulong(p.nl)+ulong(lay);
            float cwp=r.ciwpmc[cell]+r.clwpmc[cell];
            cwp=cwp+r.cswpmc[cell];
            if(!(r.cldfmc[cell]>=cldmin
                 &&(cwp>=cldmin||r.taucmc[cell]>=cldmin)))continue;
            if(p.inflag==0)return;
            if(p.inflag==1){r.err_flag[col]=1;return;}
            float abscoice,abscosno;
            const float ice_plus_snow=r.ciwpmc[cell]+r.cswpmc[cell];
            if(ice_plus_snow==0.0f){abscoice=0.0f;abscosno=0.0f;}
            else if(p.iceflag==0) {
                if(radice<10.0f){r.err_flag[col]=2;return;}
                abscoice=r.absice0[0]+r.absice0[1]/radice;abscosno=0.0f;
            } else if(p.iceflag==1) {
                if(radice<13.0f||radice>130.0f){r.err_flag[col]=3;return;}
                r.ncb_flag[col]=1;
                const int band=r.ngb[ig];
                const int ib=band<=2?band:(band<=5?3:(band<=8?4:5));
                const ulong base=2ul*ulong(ib-1);
                abscoice=r.absice1[base]+r.absice1[base+1ul]/radice;
                abscosno=0.0f;
            } else if(p.iceflag==2) {
                if(radice<5.0f||radice>131.0f){r.err_flag[col]=4;return;}
                r.ncb_flag[col]=1;
                const float factor=(radice-2.0f)/3.0f;
                int index=int(factor);if(index==43)index=42;
                const float fint=factor-float(index);
                const int ib=r.ngb[ig];
                const ulong a=ulong(index-1)+43ul*ulong(ib-1);
                const ulong b=ulong(index)+43ul*ulong(ib-1);
                abscoice=r.absice2[a]+fint*(r.absice2[b]-r.absice2[a]);
                abscosno=0.0f;
            } else {
                if(radice<5.0f||radice>140.0f){r.err_flag[col]=5;return;}
                r.ncb_flag[col]=1;
                const float factor=(radice-2.0f)/3.0f;
                int index=int(factor);if(index==46)index=45;
                const float fint=factor-float(index);
                const int ib=r.ngb[ig];
                const ulong a=ulong(index-1)+46ul*ulong(ib-1);
                const ulong b=ulong(index)+46ul*ulong(ib-1);
                abscoice=r.absice3[a]+fint*(r.absice3[b]-r.absice3[a]);
                abscosno=0.0f;
            }
            if(r.cswpmc[cell]>0.0f&&p.iceflag==5) {
                const float radsno=r.resnmc[layer];
                if(radsno<5.0f||radsno>140.0f){r.err_flag[col]=6;return;}
                r.ncb_flag[col]=1;
                const float factor=(radsno-2.0f)/3.0f;
                int index=int(factor);if(index==46)index=45;
                const float fint=factor-float(index);
                const int ib=r.ngb[ig];
                const ulong a=ulong(index-1)+46ul*ulong(ib-1);
                const ulong b=ulong(index)+46ul*ulong(ib-1);
                abscosno=r.absice3[a]+fint*(r.absice3[b]-r.absice3[a]);
            }
            float abscoliq;
            if(r.clwpmc[cell]==0.0f)abscoliq=0.0f;
            else if(p.liqflag==0)abscoliq=p.absliq0;
            else {
                const float radliq=r.relqmc[layer];
                if(radliq<2.5f||radliq>60.0f){r.err_flag[col]=7;return;}
                int index=int(radliq-1.5f);
                if(index==0)index=1;if(index==58)index=57;
                const float fint=(radliq-1.5f)-float(index);
                const int ib=r.ngb[ig];
                const ulong a=ulong(index-1)+58ul*ulong(ib-1);
                const ulong b=ulong(index)+58ul*ulong(ib-1);
                abscoliq=r.absliq1[a]+fint*(r.absliq1[b]-r.absliq1[a]);
            }
            float tau=r.ciwpmc[cell]*abscoice;
            tau=tau+r.clwpmc[cell]*abscoliq;
            tau=tau+r.cswpmc[cell]*abscosno;
            r.taucmc[cell]=tau;
        }
    }
}
