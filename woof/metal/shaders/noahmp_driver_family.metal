#include <metal_stdlib>
#include "noahmp_driver_snow_init_bindings.metalh"
#include "noahmp_driver_noahmp_init_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"

using namespace metal;

// The CUDA source uses explicit RN FP32 intrinsics and a bit-pattern constant
// table. Preserve operation boundaries and prevent FMA contraction here.
#pragma clang fp contract(off)

constant uint NMP_SNOW_CONSTANTS[35] = {
    0x00000000u, 0x3F800000u, 0xBF800000u, 0x40000000u,
    0x3F000000u, 0x3CCCCCCDu, 0x3D4CCCCDu, 0x3DCCCCCDu,
    0x3E800000u, 0x3EE66666u, 0x3E4CCCCDu, 0x48A2D780u,
    0x411CF5C3u, 0x43889333u, 0x43889312u, 0x43839333u,
    0x44FA0000u, 0x41200000u, 0x3BA3D70Au, 0x3C23D70Au,
    0x3CA3D70Au, 0x3D4CCCCDu, 0x3DCCCCCDu, 0x447A0000u,
    0x40400000u, 0x43A6AAABu, 0x45992000u, 0x41C80000u,
    0x3E4CCCCDu, 0x44FA0000u, 0x3F266666u, 0x3DCCCCCDu,
    0x2EDBE6FFu, 0x43FA0000u, 0x447A0000u
};

static inline float nmp_k(int i) {
    return as_type<float>(NMP_SNOW_CONSTANTS[i]);
}
static inline float nmp_add(float a, float b) { return a + b; }
static inline float nmp_sub(float a, float b) { return a - b; }
static inline float nmp_mul(float a, float b) { return a * b; }
static inline float nmp_div(float a, float b) { return a / b; }
static inline float nmp_min(float a, float b) { return a < b ? a : b; }
static inline float nmp_max(float a, float b) { return a > b ? a : b; }

// NOAHMP_INIT packed offsets.  These are the host layout exported by
// woof.core.noahmp_driver_gpu.NI_IN/NI_IX/NI_OUT.
constant int NI_NSOIL_MAX = 9;
constant int NI_NSNOW = 3;
constant int NI_NLAY_MAX = 12;
constant int NI_IX_STRIDE = 21;
constant int NI_IN_STRIDE = 58;
constant int NI_OUT_STRIDE = 93;
constant int NI_IN_DZS = 10;
constant int NI_IN_TSLB = 19;
constant int NI_IN_SMOIS = 28;
constant int NI_IN_ZSNSO = 37;
constant int NI_IN_TSNO = 49;
constant int NI_IN_SNICE = 52;
constant int NI_IN_SNLIQ = 55;
constant int NI_OUT_TSLB = 36;
constant int NI_OUT_SMOIS = 45;
constant int NI_OUT_SH2O = 54;
constant int NI_OUT_ZSOIL = 63;
constant int NI_OUT_ZSNSO = 72;
constant int NI_OUT_TSNO = 84;
constant int NI_OUT_SNICE = 87;
constant int NI_OUT_SNLIQ = 90;

// The oracle's x**y is glibc's powf (2095-2096); the water and glacier
// leaves use the same transcription.
static inline float nmp_powf(float x, float y) {
    return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x), as_type<uint>(y)));
}

static inline float nmp_supercooled_guess(float tslb, float bexp,
                                           float smcmax, float psisat,
                                           float smois) {
    const float base = nmp_mul(
        nmp_div(nmp_k(11), nmp_mul(nmp_k(12), nmp_sub(nmp_k(0), psisat))),
        nmp_div(nmp_sub(tslb, nmp_k(13)), tslb));
    float fk = nmp_mul(nmp_powf(base, nmp_div(nmp_k(2), bexp)), smcmax);
    fk = nmp_max(fk, nmp_k(20));
    return nmp_min(fk, smois);
}

static inline void nmp_snow_init_core(
    int nsnow, int nsoil, thread const float *zsoil,
    float swe, float tgxy, float snodep,
    thread float *zsnso, thread float *tsno,
    thread float *snice, thread float *snliq,
    thread int &isnow_out) {
    thread float dzsno[3];
    thread float dzsnso[12];
    for (int k = 0; k < 3; ++k) dzsno[k] = as_type<float>(0x7FC00000u);
    for (int k = 0; k < 12; ++k) dzsnso[k] = as_type<float>(0x7FC00000u);

    int isnow;
    if (snodep < nmp_k(5)) {
        isnow = 0;
        for (int k = 0; k < 3; ++k) dzsno[k] = nmp_k(0);
    } else if (snodep >= nmp_k(5) && snodep <= nmp_k(6)) {
        isnow = -1;
        dzsno[nsnow - 1] = snodep;
    } else if (snodep > nmp_k(6) && snodep <= nmp_k(7)) {
        isnow = -2;
        dzsno[nsnow - 2] = nmp_div(snodep, nmp_k(3));
        dzsno[nsnow - 1] = nmp_div(snodep, nmp_k(3));
    } else if (snodep > nmp_k(7) && snodep <= nmp_k(8)) {
        isnow = -2;
        dzsno[nsnow - 2] = nmp_k(6);
        dzsno[nsnow - 1] = nmp_sub(snodep, dzsno[nsnow - 2]);
    } else if (snodep > nmp_k(8) && snodep <= nmp_k(9)) {
        isnow = -3;
        dzsno[nsnow - 3] = nmp_k(6);
        dzsno[nsnow - 2] = nmp_mul(
            nmp_k(4), nmp_sub(snodep, dzsno[nsnow - 3]));
        dzsno[nsnow - 1] = nmp_mul(
            nmp_k(4), nmp_sub(snodep, dzsno[nsnow - 3]));
    } else if (snodep > nmp_k(9)) {
        isnow = -3;
        dzsno[nsnow - 3] = nmp_k(6);
        dzsno[nsnow - 2] = nmp_k(10);
        dzsno[nsnow - 1] = nmp_sub(
            nmp_sub(snodep, dzsno[nsnow - 2]), dzsno[nsnow - 3]);
    } else {
        // The CUDA fallback is reachable only for NaN SNODEP; retain its
        // NaN-filled local behavior instead of inventing a repaired value.
        isnow = 0;
        for (int k = 0; k < 3; ++k) dzsno[k] = as_type<float>(0x7FC00000u);
    }

    for (int k = 0; k < nsnow; ++k) {
        tsno[k] = nmp_k(0);
        snice[k] = nmp_k(0);
        snliq[k] = nmp_k(0);
    }
    for (int iz = isnow + 1; iz <= 0; ++iz) {
        const int slot = iz + nsnow - 1;
        tsno[slot] = tgxy;
        snliq[slot] = nmp_k(0);
        snice[slot] = nmp_mul(dzsno[slot], nmp_div(swe, snodep));
    }

    for (int iz = isnow + 1; iz <= 0; ++iz)
        dzsnso[iz + nsnow - 1] = nmp_sub(nmp_k(0), dzsno[iz + nsnow - 1]);
    dzsnso[nsnow] = zsoil[0];
    for (int iz = 2; iz <= nsoil; ++iz)
        dzsnso[iz + nsnow - 1] = nmp_sub(zsoil[iz - 1], zsoil[iz - 2]);

    // The assigned lower snow layer comes from DZSNSO. Slots above it keep
    // the caller's original ZSNSO value, as in the Fortran INTENT(OUT) path.
    zsnso[isnow + 1 + nsnow - 1] = dzsnso[isnow + 1 + nsnow - 1];
    for (int iz = isnow + 2; iz <= nsoil; ++iz) {
        const int slot = iz + nsnow - 1;
        zsnso[slot] = nmp_add(zsnso[slot - 1], dzsnso[slot]);
    }
    isnow_out = isnow;
}

kernel void noahmp_driver_snow_init(
    constant NoahmpDriverSnowInitResources &r [[buffer(0)]],
    constant NoahmpDriverSnowInitParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncase)) return;
    const ulong xbase = ulong(tid) * 33ul;
    const ulong ixbase = ulong(tid) * 2ul;
    const ulong ybase = ulong(tid) * 22ul;
    const int nsnow = r.ix[ixbase];
    const int nsoil = r.ix[ixbase + 1ul];
    thread float zsoil[9];
    thread float zsnso[12];
    thread float tsno[3], snice[3], snliq[3];
    for (int k = 0; k < nsoil; ++k) zsoil[k] = r.x[xbase + 3ul + ulong(k)];
    for (int k = 0; k < nsnow + nsoil; ++k)
        zsnso[k] = r.x[xbase + 12ul + ulong(k)];
    for (int k = 0; k < nsnow; ++k) {
        tsno[k] = r.x[xbase + 24ul + ulong(k)];
        snice[k] = r.x[xbase + 27ul + ulong(k)];
        snliq[k] = r.x[xbase + 30ul + ulong(k)];
    }
    int isnow = 0;
    nmp_snow_init_core(nsnow, nsoil, zsoil, r.x[xbase], r.x[xbase + 1ul],
                       r.x[xbase + 2ul], zsnso, tsno, snice, snliq, isnow);
    r.y[ybase] = float(isnow);
    for (int k = 0; k < nsnow + nsoil; ++k)
        r.y[ybase + 1ul + ulong(k)] = zsnso[k];
    for (int k = 0; k < nsnow; ++k) {
        r.y[ybase + 13ul + ulong(k)] = tsno[k];
        r.y[ybase + 16ul + ulong(k)] = snice[k];
        r.y[ybase + 19ul + ulong(k)] = snliq[k];
    }
}

kernel void noahmp_driver_noahmp_init(
    constant NoahmpDriverNoahmpInitResources &r [[buffer(0)]],
    constant NoahmpDriverNoahmpInitParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncase)) return;
    const ulong xbase = ulong(tid) * ulong(NI_IN_STRIDE);
    const ulong ibase = ulong(tid) * ulong(NI_IX_STRIDE);
    const ulong ybase = ulong(tid) * ulong(NI_OUT_STRIDE);
    device const float *x = r.x + xbase;
    device const int *ix = r.ix + ibase;
    device float *y = r.y + ybase;

    const int nsoil = ix[0];
    const int nsnow = ix[1];
    const int fndsnowh = ix[2];
    const int vegtyp = ix[3];
    const int cropcat_in = ix[4];
    const int sf_urban_physics = ix[5];
    const int isice = ix[6];
    const int isurban = ix[7];
    const int iswater = ix[8];
    const int isbarren = ix[9];
    device const int *lcz = ix + 10;

    thread float tslb[NI_NSOIL_MAX], smois[NI_NSOIL_MAX];
    thread float sh2o[NI_NSOIL_MAX], zsoil[NI_NSOIL_MAX];
    thread float dzs[NI_NSOIL_MAX], zsnso[NI_NLAY_MAX];
    thread float tsno[NI_NSNOW], snice[NI_NSNOW], snliq[NI_NSNOW];
    for (int k = 0; k < nsoil; ++k) {
        tslb[k] = x[NI_IN_TSLB + k];
        smois[k] = x[NI_IN_SMOIS + k];
        dzs[k] = x[NI_IN_DZS + k];
    }
    for (int k = 0; k < nsnow + nsoil; ++k)
        zsnso[k] = x[NI_IN_ZSNSO + k];
    for (int k = 0; k < nsnow; ++k) {
        tsno[k] = x[NI_IN_TSNO + k];
        snice[k] = x[NI_IN_SNICE + k];
        snliq[k] = x[NI_IN_SNLIQ + k];
    }

    float snow = x[8];
    float snowh = x[9];
    if (fndsnowh == 0) snowh = nmp_mul(snow, nmp_k(18));
    if (snow > nmp_k(16)) {
        snowh = nmp_div(nmp_mul(snowh, nmp_k(16)), snow);
        snow = nmp_k(16);
    }

    if (vegtyp == isice && x[0] <= nmp_k(0)) {
        for (int k = 0; k < nsoil; ++k) {
            smois[k] = nmp_k(1);
            sh2o[k] = nmp_k(0);
            tslb[k] = nmp_min(tslb[k], nmp_k(15));
        }
        snow = nmp_max(snow, nmp_k(17));
        snowh = nmp_mul(snow, nmp_k(19));
    } else {
        for (int k = 0; k < nsoil; ++k)
            if (smois[k] > x[4]) smois[k] = x[4];
        if (x[3] > nmp_k(0) && x[4] > nmp_k(0) && x[5] > nmp_k(0)) {
            for (int k = 0; k < nsoil; ++k) {
                if (tslb[k] < nmp_k(14))
                    sh2o[k] = nmp_supercooled_guess(
                        tslb[k], x[3], x[4], x[5], smois[k]);
                else
                    sh2o[k] = smois[k];
            }
        } else {
            for (int k = 0; k < nsoil; ++k) sh2o[k] = smois[k];
        }
    }

    float cold_skin = x[1];
    if (snow > nmp_k(0) && x[1] > nmp_k(13)) cold_skin = nmp_k(13);
    const float wa = nmp_k(26);
    const float zwt = nmp_sub(
        nmp_add(nmp_k(27), nmp_k(3)),
        nmp_div(nmp_div(wa, nmp_k(23)), nmp_k(28)));

    bool urbanpt = (vegtyp == isurban);
    for (int k = 0; k < 11; ++k) urbanpt = urbanpt || (vegtyp == lcz[k]);
    const bool bare = (vegtyp == isbarren) || (vegtyp == isice)
        || (sf_urban_physics == 0 && urbanpt) || (vegtyp == iswater);
    float lai, xsai, lfmass, stmass, rtmass, wood, stblcp, fastcp;
    int cropcat;
    if (bare) {
        lai = nmp_k(0); xsai = nmp_k(0); lfmass = nmp_k(0);
        stmass = nmp_k(0); rtmass = nmp_k(0); wood = nmp_k(0);
        stblcp = nmp_k(0); fastcp = nmp_k(0); cropcat = 0;
    } else {
        lai = nmp_max(x[2], nmp_k(21));
        xsai = nmp_max(nmp_mul(nmp_k(22), lai), nmp_k(21));
        const float sla_used = urbanpt ? x[7] : x[6];
        const float masslai = nmp_div(nmp_k(23), nmp_max(sla_used, nmp_k(1)));
        lfmass = nmp_mul(lai, masslai);
        stmass = nmp_mul(xsai, nmp_k(25));
        rtmass = nmp_k(33); wood = nmp_k(33);
        stblcp = nmp_k(34); fastcp = nmp_k(34);
        cropcat = cropcat_in;
    }

    zsoil[0] = nmp_sub(nmp_k(0), dzs[0]);
    for (int k = 1; k < nsoil; ++k)
        zsoil[k] = nmp_sub(zsoil[k - 1], dzs[k]);
    int isnow = 0;
    nmp_snow_init_core(nsnow, nsoil, zsoil, snow, cold_skin, snowh,
                       zsnso, tsno, snice, snliq, isnow);

    y[0] = snow; y[1] = snowh; y[2] = nmp_k(0);
    y[3] = cold_skin; y[4] = cold_skin; y[5] = nmp_k(0);
    y[6] = nmp_k(0); y[7] = nmp_k(29); y[8] = cold_skin;
    y[9] = nmp_k(0); y[10] = nmp_k(0); y[11] = nmp_k(0);
    y[12] = nmp_k(0); y[13] = nmp_k(30); y[14] = nmp_k(0);
    y[15] = nmp_k(0); y[16] = nmp_k(0); y[17] = zwt;
    y[18] = wa; y[19] = wa; y[20] = lai; y[21] = xsai;
    y[22] = lfmass; y[23] = rtmass; y[24] = stmass; y[25] = wood;
    y[26] = stblcp; y[27] = fastcp; y[28] = nmp_k(32); y[29] = nmp_k(0);
    y[30] = cold_skin; y[31] = cold_skin; y[32] = nmp_k(31);
    y[33] = nmp_k(0); y[34] = float(isnow); y[35] = float(cropcat);
    for (int k = 0; k < nsoil; ++k) {
        y[NI_OUT_TSLB + k] = tslb[k];
        y[NI_OUT_SMOIS + k] = smois[k];
        y[NI_OUT_SH2O + k] = sh2o[k];
        y[NI_OUT_ZSOIL + k] = zsoil[k];
    }
    for (int k = 0; k < nsnow + nsoil; ++k)
        y[NI_OUT_ZSNSO + k] = zsnso[k];
    for (int k = 0; k < nsnow; ++k) {
        y[NI_OUT_TSNO + k] = tsno[k];
        y[NI_OUT_SNICE + k] = snice[k];
        y[NI_OUT_SNLIQ + k] = snliq[k];
    }
}
