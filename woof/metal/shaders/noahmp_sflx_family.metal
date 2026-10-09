#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "k_sflx_error_bindings.metalh"
#include "k_sflx_marshal_bindings.metalh"

using namespace metal;

// Dependency-blocked consumer proposal. The shared integer-bit primitives
// perform RN-even FP32 arithmetic without native float operations. This
// source remains uncompiled/unverified and is not admitted for production.
static inline float sflx_fadd(float a, float b) {
    return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)));
}
static inline float sflx_fsub(float a, float b) {
    return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)));
}
static inline float sflx_fmul(float a, float b) {
    return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)));
}
static inline float sflx_fdiv(float a, float b) {
    return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)));
}
static inline float sflx_abs(float x) {
    return as_type<float>(soft32_abs_bits(as_type<uint>(x)));
}

enum {
    SC_SWDOWN = 0, SC_FSA = 1, SC_FSR = 2, SC_FIRA = 3, SC_FSH = 4,
    SC_FCEV = 5, SC_FGEV = 6, SC_FCTR = 7, SC_SSOIL = 8, SC_SAV = 9,
    SC_SAG = 10, SC_BEG_WB = 11, SC_CANLIQ = 12, SC_CANICE = 13,
    SC_SNEQV = 14, SC_WA = 15, SC_PRCP = 16, SC_ECAN = 17,
    SC_ETRAN = 18, SC_EDIR = 19, SC_RUNSRF = 20, SC_RUNSUB = 21,
    SC_DT = 22, SC_QTLDRN = 23, SC_PAH = 24, SC_FIRR = 25,
    SC_CANHS = 26, SC_IRMIRATE = 27, SC_IRFIRATE = 28,
    SC_ACC_DWATER = 29, SC_ACC_PRCP = 30, SC_ACC_ECAN = 31,
    SC_ACC_ETRAN = 32, SC_ACC_EDIR = 33, NSC = 34
};
enum {
    OU_ERRWAT = 0, OU_ACC_DWATER = 1, OU_ACC_PRCP = 2, OU_ACC_ECAN = 3,
    OU_ACC_ETRAN = 4, OU_ACC_EDIR = 5, OU_ERRSW = 6, OU_ERRENG = 7,
    OU_END_WB = 8, NOUT = 9
};
constant int NSOIL = 4;
constant int NSNOW = 3;
constant int NFULL = NSNOW + NSOIL;
constant float ERRSW_TOL = 0.01f;
constant float ERRENG_TOL = 0.01f;
constant float ERRWAT_TOL = 0.1f;

kernel void k_sflx_error(
    constant KSflxErrorResources &r [[buffer(0)]],
    constant KSflxErrorParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    if (flat >= uint(p.ncase)) return;
    const int c = int(flat);
    const ulong sbase = ulong(c) * ulong(NSC);
    const ulong obase = ulong(c) * ulong(NOUT);
    const ulong soilbase = ulong(c) * ulong(NSOIL);

    for (int k = 0; k < NOUT; ++k) r.out[obase + ulong(k)] = 0.0f;
    r.status[c] = 0;

    const float errsw = sflx_fsub(
        r.sc[sbase + SC_SWDOWN],
        sflx_fadd(r.sc[sbase + SC_FSA], r.sc[sbase + SC_FSR]));
    r.out[obase + OU_ERRSW] = errsw;
    if (sflx_abs(errsw) > ERRSW_TOL) {
        r.status[c] = 1;
        return;
    }

    float sink = sflx_fadd(r.sc[sbase + SC_FIRA], r.sc[sbase + SC_FSH]);
    sink = sflx_fadd(sink, r.sc[sbase + SC_FCEV]);
    sink = sflx_fadd(sink, r.sc[sbase + SC_FGEV]);
    sink = sflx_fadd(sink, r.sc[sbase + SC_FCTR]);
    sink = sflx_fadd(sink, r.sc[sbase + SC_SSOIL]);
    sink = sflx_fadd(sink, r.sc[sbase + SC_FIRR]);
    sink = sflx_fadd(sink, r.sc[sbase + SC_CANHS]);
    const float sav_sag = sflx_fadd(r.sc[sbase + SC_SAV], r.sc[sbase + SC_SAG]);
    const float energy_minus_sink = sflx_fsub(sav_sag, sink);
    const float erreng = sflx_fadd(energy_minus_sink, r.sc[sbase + SC_PAH]);
    r.out[obase + OU_ERRENG] = erreng;
    if (sflx_abs(erreng) > ERRENG_TOL) {
        r.status[c] = 2;
        return;
    }

    float acc_dwater = r.sc[sbase + SC_ACC_DWATER];
    float acc_prcp = r.sc[sbase + SC_ACC_PRCP];
    float acc_ecan = r.sc[sbase + SC_ACC_ECAN];
    float acc_etran = r.sc[sbase + SC_ACC_ETRAN];
    float acc_edir = r.sc[sbase + SC_ACC_EDIR];
    float errwat = 0.0f;
    float end_wb = 0.0f;

    if (r.ist[c] == 1) {
        float stores = sflx_fadd(r.sc[sbase + SC_CANLIQ],
                                 r.sc[sbase + SC_CANICE]);
        stores = sflx_fadd(stores, r.sc[sbase + SC_SNEQV]);
        end_wb = sflx_fadd(stores, r.sc[sbase + SC_WA]);
        for (int k = 0; k < NSOIL; ++k) {
            const float soil_mass = sflx_fmul(
                r.smc[soilbase + ulong(k)], r.dzsnso[soilbase + ulong(k)]);
            end_wb = sflx_fadd(end_wb, sflx_fmul(soil_mass, 1000.0f));
        }

        acc_dwater = sflx_fadd(
            acc_dwater, sflx_fsub(end_wb, r.sc[sbase + SC_BEG_WB]));
        acc_prcp = sflx_fadd(acc_prcp,
            sflx_fmul(r.sc[sbase + SC_PRCP], r.sc[sbase + SC_DT]));
        acc_ecan = sflx_fadd(acc_ecan,
            sflx_fmul(r.sc[sbase + SC_ECAN], r.sc[sbase + SC_DT]));
        acc_etran = sflx_fadd(acc_etran,
            sflx_fmul(r.sc[sbase + SC_ETRAN], r.sc[sbase + SC_DT]));
        acc_edir = sflx_fadd(acc_edir,
            sflx_fmul(r.sc[sbase + SC_EDIR], r.sc[sbase + SC_DT]));

        if (r.calc[c] != 0) {
            float inner = sflx_fadd(acc_prcp,
                sflx_fmul(r.sc[sbase + SC_IRFIRATE], 1000.0f));
            inner = sflx_fadd(inner,
                sflx_fmul(r.sc[sbase + SC_IRMIRATE], 1000.0f));
            inner = sflx_fsub(inner, acc_ecan);
            inner = sflx_fsub(inner, acc_etran);
            inner = sflx_fsub(inner, acc_edir);
            inner = sflx_fsub(inner, r.sc[sbase + SC_RUNSRF]);
            inner = sflx_fsub(inner, r.sc[sbase + SC_RUNSUB]);
            inner = sflx_fsub(inner, r.sc[sbase + SC_QTLDRN]);
            errwat = sflx_fsub(acc_dwater, inner);
            if (sflx_abs(errwat) > ERRWAT_TOL) r.status[c] = 3;
        }
    }

    r.out[obase + OU_ERRWAT] = errwat;
    r.out[obase + OU_ACC_DWATER] = acc_dwater;
    r.out[obase + OU_ACC_PRCP] = acc_prcp;
    r.out[obase + OU_ACC_ECAN] = acc_ecan;
    r.out[obase + OU_ACC_ETRAN] = acc_etran;
    r.out[obase + OU_ACC_EDIR] = acc_edir;
    r.out[obase + OU_END_WB] = end_wb;
}

kernel void k_sflx_marshal(
    constant KSflxMarshalResources &r [[buffer(0)]],
    constant KSflxMarshalParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    if (flat >= uint(p.ncol)) return;
    const int c = int(flat);
    const ulong full_base = ulong(c) * ulong(NFULL);
    const ulong soil_base = ulong(c) * ulong(NSOIL);
    for (int k = 0; k < NFULL; ++k)
        r.dzsnso[full_base + ulong(k)] = 0.0f;

    const int is = r.isnow[c];
    for (int iz = is + 1; iz <= NSOIL; ++iz) {
        const int k = iz + NSNOW - 1;
        if (iz == is + 1) {
            r.dzsnso[full_base + ulong(k)] =
                sflx_fsub(0.0f, r.zsnso[full_base + ulong(k)]);
        } else {
            r.dzsnso[full_base + ulong(k)] = sflx_fsub(
                r.zsnso[full_base + ulong(k - 1)],
                r.zsnso[full_base + ulong(k)]);
        }
    }

    const int nr = r.nroot[c];
    const float denom = sflx_fsub(0.0f, r.zsoil[soil_base + ulong(nr - 1)]);
    float tr = 0.0f;
    for (int iz = 1; iz <= nr; ++iz) {
        const int k = iz + NSNOW - 1;
        const float weighted_temp = sflx_fmul(
            r.stc[full_base + ulong(k)], r.dzsnso[full_base + ulong(k)]);
        const float layer_temp = sflx_fdiv(weighted_temp, denom);
        tr = sflx_fadd(tr, layer_temp);
    }
    r.troot[c] = tr;

    float wb = 0.0f;
    if (r.ist[c] == 1) {
        wb = sflx_fadd(r.canliq[c], r.canice[c]);
        wb = sflx_fadd(wb, r.sneqv[c]);
        wb = sflx_fadd(wb, r.wa[c]);
        for (int iz = 1; iz <= NSOIL; ++iz) {
            const int k = iz + NSNOW - 1;
            const float soil_mass = sflx_fmul(
                r.smc[soil_base + ulong(iz - 1)],
                r.dzsnso[full_base + ulong(k)]);
            wb = sflx_fadd(wb, sflx_fmul(soil_mass, 1000.0f));
        }
    }
    r.beg_wb[c] = wb;
}
