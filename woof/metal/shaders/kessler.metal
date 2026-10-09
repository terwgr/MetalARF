#include <metal_stdlib>
#include "kessler_column_bindings.metalh"

using namespace metal;

#pragma clang fp contract(off)

#define KESS_KMAX 256

kernel void kessler_column(constant KesslerColumnResources &r [[buffer(0)]],
                           constant KesslerColumnParams &p [[buffer(1)]],
                           uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ny) * uint(p.nx)) return;
    const int j = int(col / uint(p.nx));
    const int i = int(col - uint(j) * uint(p.nx));

    const float c1k = 0.001f;
    const float c2k = 0.001f;
    const float c3k = 2.2f;
    const float c4k = 0.875f;
    const float max_cr_sedimentation = 0.75f;
    const float f5 = p.svp2 * (p.svpt0 - p.svp3) * p.xlv / p.cp;

    float prodk[KESS_KMAX];
    float vt[KESS_KMAX];
    float vtden[KESS_KMAX];
    float rhok[KESS_KMAX];
    float rdzk[KESS_KMAX];

    const uint column = uint(j) * uint(p.nx) + uint(i);
    const uint plane = uint(p.ny) * uint(p.nx);
    const float rho0 = r.rho[column];
    float crmax = 0.0f;
    for (int k = 0; k < p.nz; ++k) {
        const uint idx = uint(k) * plane + column;
        prodk[k] = r.qr[idx];
        rhok[k] = r.rho[idx];
        const float qrr = fmax(0.0f, r.qr[idx] * 0.001f * rhok[k]);
        vtden[k] = sqrt(rho0 / rhok[k]);
        vt[k] = 36.34f * pow(qrr, 0.1364f) * vtden[k];
        const float rdzw = 1.0f / r.dz8w[idx];
        crmax = fmax(vt[k] * p.dt * rdzw, crmax);
    }
    for (int k = 0; k < p.nz - 1; ++k) {
        const uint idx = uint(k) * plane + column;
        rdzk[k] = 1.0f / (r.z[idx + plane] - r.z[idx]);
    }
    const uint top = uint(p.nz - 1) * plane + column;
    const uint below_top = uint(p.nz - 2) * plane + column;
    rdzk[p.nz - 1] = 1.0f / (r.z[top] - r.z[below_top]);

    int nfall = max(1, int(floor(0.5f + crmax / max_cr_sedimentation
                                + 0.5f)));
    float dtfall = p.dt / float(nfall);
    float time_sediment = p.dt;

    while (nfall > 0) {
        time_sediment -= dtfall;
        const float ppt = (rhok[0] * prodk[0] * vt[0] * dtfall)
                          / p.rhowater;
        r.rainncv[column] = ppt * 1000.0f;
        r.rainnc[column] += ppt * 1000.0f;

        for (int k = 0; k < p.nz - 1; ++k) {
            const float factor = dtfall * rdzk[k] / rhok[k];
            prodk[k] = prodk[k] - factor
                * (rhok[k] * prodk[k] * vt[k]
                   - rhok[k + 1] * prodk[k + 1] * vt[k + 1]);
        }
        prodk[p.nz - 1] = prodk[p.nz - 1]
            - (dtfall * rdzk[p.nz - 1]) * prodk[p.nz - 1] * vt[p.nz - 1];

        if (nfall > 1) {
            nfall = nfall - 1;
            crmax = 0.0f;
            for (int k = 0; k < p.nz; ++k) {
                const float qrr = fmax(0.0f, prodk[k] * 0.001f * rhok[k]);
                vt[k] = 36.34f * pow(qrr, 0.1364f) * vtden[k];
                const uint idx = uint(k) * plane + column;
                const float rdzw = 1.0f / r.dz8w[idx];
                crmax = fmax(vt[k] * time_sediment * rdzw, crmax);
            }
            const int nfall_new = max(1, int(floor(
                0.5f + crmax / max_cr_sedimentation + 0.5f)));
            if (nfall_new != nfall) {
                nfall = nfall_new;
                dtfall = time_sediment / float(nfall);
            }
        } else {
            nfall = 0;
        }
    }

    for (int k = 0; k < p.nz; ++k) {
        const uint idx = uint(k) * plane + column;
        const float qrk_source = r.qr[idx];
        float qck = r.qc[idx];
        const float qvk = r.qv[idx];
        const float tk = r.t[idx];
        const float piik = r.pii[idx];

        const float factorn = 1.0f / (1.0f + c3k * p.dt
                                     * pow(fmax(0.0f, qrk_source), c4k));
        const float qrprod = qck * (1.0f - factorn)
            + factorn * c1k * p.dt * fmax(qck - c2k, 0.0f);
        const float rcgs = 0.001f * r.rho[idx];

        qck = fmax(qck - qrprod, 0.0f);
        float qrk = prodk[k];
        qrk = fmax(qrk + qrprod, 0.0f);

        const float temp = piik * tk;
        const float pressure = 1.0e5f * pow(piik, 1004.0f / 287.0f);
        const float gam = 2.5e6f / (1004.0f * piik);
        const float es = 1000.0f * p.svp1
            * exp(p.svp2 * (temp - p.svpt0) / (temp - p.svp3));
        const float qvs = p.ep2 * es / (pressure - es);
        const float prod = (qvk - qvs)
            / (1.0f + pressure / (pressure - es) * qvs * f5
               / ((temp - p.svp3) * (temp - p.svp3)));
        const float ern = fmin(fmin(
            p.dt * (((1.6f + 124.9f * pow(rcgs * qrk, 0.2046f))
                     * pow(rcgs * qrk, 0.525f))
                    / (2.55e8f / (pressure * qvs) + 5.4e5f))
                * (fmax(qvs - qvk, 0.0f) / (rcgs * qvs)),
            fmax(-prod - qck, 0.0f)), qrk);

        const float product = fmax(prod, -qck);
        r.t[idx] = tk + gam * (product - ern);
        r.qv[idx] = fmax(qvk - product + ern, 0.0f);
        r.qc[idx] = qck + product;
        r.qr[idx] = qrk - ern;
    }
}
