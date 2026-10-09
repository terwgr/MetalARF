#include <metal_stdlib>
#include "microphysics_edge_field_bindings.metalh"

using namespace metal;

// CUDA's __f*rn intrinsics are represented by separate ordinary float
// expressions here. Strict hardware equivalence remains unverified.
static inline float edge_rn_add(float a, float b) { return a + b; }
static inline float edge_rn_sub(float a, float b) { return a - b; }
static inline float edge_rn_mul(float a, float b) { return a * b; }
static inline float edge_rn_div(float a, float b) { return a / b; }

static inline float edge_source_mass(int code, int idx,
                                     constant MicrophysicsEdgeFieldResources &r) {
    if (code == 0) return r.source_qv[idx];
    if (code == 1) return r.source_qc[idx];
    if (code == 2) return r.source_qr[idx];
    if (code == 3) return r.source_qi[idx];
    if (code == 4) return r.source_qs[idx];
    if (code == 5) return r.source_qg[idx];
    if (code == 6) return r.source_qh[idx];
    return 0.0f;
}

static inline float thompson_edge_rain_number(float qr) {
    if (qr <= 1.0e-12f) return 0.0f;
    const float am_r = 3.1415926536f * 1000.0f / 6.0f;
    const float lambda = 3.672f / 1.0e-3f;
    return edge_rn_div((1.0f / 6.0f) * qr, am_r) * lambda * lambda * lambda;
}

static inline float thompson_edge_ice_number(float qi, float rho) {
    if (qi <= 1.0e-12f) return 0.0f;
    const float am_i = 3.1415926536f * 890.0f / 6.0f;
    const float ice_mass = qi * rho;
    float lambda = 4.0f / 5.0e-6f;
    float number = min(999.0e3f,
        edge_rn_div((1.0f / 6.0f) * ice_mass, am_i) * lambda * lambda * lambda);
    lambda = cbrt(am_i * 6.0f * number / ice_mass);
    const float diameter = 4.0f / lambda;
    if (diameter < 5.0e-6f) {
        lambda = 4.0f / 5.0e-6f;
        number = min(999.0e3f,
            edge_rn_div((1.0f / 6.0f) * ice_mass, am_i) * lambda * lambda * lambda);
    } else if (diameter > 300.0e-6f) {
        lambda = 4.0f / 300.0e-6f;
        number = edge_rn_div((1.0f / 6.0f) * ice_mass, am_i) * lambda * lambda * lambda;
    }
    return number / rho;
}

static inline float morrison_edge_number(float q, int kind, float rimed_density) {
    if (q < 1.0e-14f) return 0.0f;
    const float pi = 3.14159265358979323846f;
    float six_c;
    float lambda;
    if (kind == 0) {
        six_c = pi * 997.0f;
        lambda = 1.0f / 2800.0e-6f;
    } else if (kind == 1) {
        six_c = 500.0f * pi;
        lambda = 1.0f / 350.0e-6f;
    } else if (kind == 2) {
        six_c = 100.0f * pi;
        lambda = 1.0f / 2000.0e-6f;
    } else {
        six_c = rimed_density * pi;
        lambda = 1.0f / 2000.0e-6f;
    }
    return edge_rn_div(q * lambda * lambda * lambda, six_c);
}

static inline void p3_edge_bulk_rho_rime(float qi_tot,
                                         thread float &qi_rim,
                                         thread float &bi_rim) {
    float rho_rime = 0.0f;
    if (bi_rim >= 1.0e-15f) {
        rho_rime = edge_rn_div(qi_rim, bi_rim);
        if (rho_rime < 50.0f) {
            rho_rime = 50.0f;
            bi_rim = edge_rn_div(qi_rim, rho_rime);
        } else if (rho_rime > 900.0f) {
            rho_rime = 900.0f;
            bi_rim = edge_rn_div(qi_rim, rho_rime);
        }
    } else {
        qi_rim = 0.0f;
        bi_rim = 0.0f;
        rho_rime = 0.0f;
    }
    if (qi_rim > qi_tot && rho_rime > 0.0f) {
        qi_rim = qi_tot;
        bi_rim = edge_rn_div(qi_rim, rho_rime);
    }
    if (qi_rim < 1.0e-14f) {
        qi_rim = 0.0f;
        bi_rim = 0.0f;
    }
}

static inline float p3_edge_field(int field, float inv_rho, float qv,
                                  float qc, float qr, float qi, float qs,
                                  float qg, float qh) {
    const float qsmall = 1.0e-14f;
    float vapor = qv;
    const float frozen_dense = edge_rn_add(qg, qh);
    float qitot = edge_rn_add(edge_rn_add(qi, qs), frozen_dense);
    float qirim = edge_rn_add(qs, frozen_dense);
    float birim = edge_rn_add(edge_rn_div(qs, 100.0f),
                              edge_rn_div(frozen_dense, 400.0f));
    if (qitot < qsmall) {
        vapor = edge_rn_add(vapor, qitot);
        qitot = 0.0f;
        qirim = 0.0f;
        birim = 0.0f;
    }
    p3_edge_bulk_rho_rime(qitot, qirim, birim);
    if (field == 0) return vapor;
    if (field == 1) return qc;
    if (field == 2) return qr;
    if (field == 3) return qitot;
    if (field == 6) return qr < qsmall ? 0.0f : edge_rn_mul(qr, 39788.727f);
    if (field == 7) {
        if (qitot < qsmall) return 0.0f;
        return min(edge_rn_div(qitot, 3.7699116e-15f),
                   edge_rn_mul(2000.0e3f, inv_rho));
    }
    if (field == 20) return qirim;
    if (field == 21) return birim;
    return 0.0f;
}

static inline float edge_plain_or_moment_field(
    int field, int target_mp, float rho, float morr_rhog, float wdm6_ccn,
    float qv, float qc, float qr, float qi, float qs, float qg, float qh) {
    if (field == 0) return qv;
    if (field == 1) return qc;
    if (field == 2) return qr;
    if (field == 3) return qi;
    if (field == 4) return qs;
    if (field == 5) return qg;
    if (field == 10) return qh;
    if (target_mp == 8) {
        if (field == 6) return thompson_edge_rain_number(qr);
        return thompson_edge_ice_number(qi, rho);
    }
    if (target_mp == 10) {
        if (field == 6) return morrison_edge_number(qr, 0, morr_rhog);
        if (field == 7) return morrison_edge_number(qi, 1, morr_rhog);
        if (field == 8) return morrison_edge_number(qs, 2, morr_rhog);
        return morrison_edge_number(qg, 3, morr_rhog);
    }
    if (target_mp == 16) {
        if (field == 24) return wdm6_ccn;
        return 0.0f;
    }
    if (target_mp == 28) {
        if (field == 6) return thompson_edge_rain_number(qr);
        if (field == 7) return thompson_edge_ice_number(qi, rho);
        if (field == 22) return edge_rn_div(100.0e6f, rho);
        if (field == 25) return edge_rn_div(11.1e6f, rho);
        if (field == 26) return edge_rn_div(5.0e3f, rho);
        return 0.0f;
    }
    return 0.0f;
}

kernel void microphysics_edge_field(
    constant MicrophysicsEdgeFieldResources &r [[buffer(0)]],
    constant MicrophysicsEdgeFieldParams &p [[buffer(1)]],
    uint t [[thread_position_in_grid]]) {
    const uint count = uint(p.nz * p.ny * p.nx);
    if (t >= count) return;
    // These two entry closures are fail-closed in the host helper: MP18
    // requires binary64 and MP9 currently receives a CuPy-only CK table.
    if (p.target_mp == 18 || p.target_mp == 9) return;
    const int idx = int(t);
    const int plane = p.ny * p.nx;
    const int k = idx / plane;
    const int rem = idx - k * plane;
    const int j = rem / p.nx;
    const int i = rem - j * p.nx;
    const float rho = 1.0f / r.alt[idx];
    const float qv = edge_source_mass(p.qv_source, idx, r);
    const float qc = edge_source_mass(p.qc_source, idx, r);
    const float qr = edge_source_mass(p.qr_source, idx, r);
    float qi = edge_source_mass(p.qi_source, idx, r);
    float qs = edge_source_mass(p.qs_source, idx, r);
    float qg = edge_source_mass(p.qg_source, idx, r);
    float qh = edge_source_mass(p.qh_source, idx, r);
    if (p.source_mp == 50) {
        const float qitot = max(qi, 0.0f);
        float qirim = max(r.source_qir[idx], 0.0f);
        float birim = max(r.source_qib[idx], 0.0f);
        p3_edge_bulk_rho_rime(qitot, qirim, birim);
        float graupel = 0.0f;
        if (qirim > 0.0f) {
            const float scale = edge_rn_div(400.0f, 300.0f);
            graupel = edge_rn_mul(edge_rn_sub(qirim,
                edge_rn_mul(100.0f, birim)), scale);
            graupel = min(max(graupel, 0.0f), qirim);
        }
        qg = graupel;
        qs = edge_rn_sub(qirim, graupel);
        qi = edge_rn_sub(qitot, qirim);
        qh = 0.0f;
    }
    float value;
    if (p.target_mp == 50) {
        value = p3_edge_field(p.field, r.alt[idx], qv, qc, qr,
                              qi, qs, qg, qh);
    } else {
        value = edge_plain_or_moment_field(
            p.field, p.target_mp, rho, p.morr_rhog, p.wdm6_ccn,
            qv, qc, qr, qi, qs, qg, qh);
    }
    if (p.coupled != 0) {
        const ulong q = ulong(j) * ulong(p.nx) + ulong(i);
        const float base = edge_rn_add(edge_rn_mul(r.c1h[k], r.mub2d[q]), r.c2h[k]);
        const float perturbation = edge_rn_mul(r.c1h[k], r.mup[q]);
        const float hybrid_mass = edge_rn_add(base, perturbation);
        value = edge_rn_mul(hybrid_mass, value);
    }
    r.out[idx] = value;
}
