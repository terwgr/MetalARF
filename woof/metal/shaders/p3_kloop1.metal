#include <metal_stdlib>
#include "p3_kloop1_bindings.metalh"
#include "p3_constants.metalh"
#include "p3_math.metalh"

using namespace metal;

// P3 k_loop_1 (module_mp_p3.F:2320-2411; p3.cu:651-780).  One invocation
// owns one complete column, retaining its level order and source update
// order. Native MSL pow replaces source-pinned r_pow.
#pragma clang fp contract(off)

static inline float p3_polysvp1(float t, int ice) {
    if (ice == 1 && t < 273.15f) {
        if (t >= 195.8f) {
            const float dt = t - 273.15f;
            const float a0 = 6.11147274f, a1 = 0.503160820f;
            const float a2 = 0.188439774e-1f, a3 = 0.420895665e-3f;
            const float a4 = 0.615021634e-5f, a5 = 0.602588177e-7f;
            const float a6 = 0.385852041e-9f, a7 = 0.146898966e-11f;
            const float a8 = 0.252751365e-14f;
            const float out = a0 + dt * (a1 + dt * (a2 + dt * (a3 + dt
                * (a4 + dt * (a5 + dt * (a6 + dt * (a7 + a8 * dt)))))));
            return out * 100.0f;
        }
        return pow(10.0f, -9.09718f * (273.16f / t - 1.0f)
            - 3.56654f * p3_log10(273.16f / t)
            + 0.876793f * (1.0f - t / 273.16f)
            + p3_log10(6.1071f)) * 100.0f;
    }
    if (t >= 202.0f) {
        const float dt = t - 273.15f;
        const float a0 = 6.11239921f, a1 = 0.443987641f;
        const float a2 = 0.142986287e-1f, a3 = 0.264847430e-3f;
        const float a4 = 0.302950461e-5f, a5 = 0.206739458e-7f;
        const float a6 = 0.640689451e-10f, a7 = -0.952447341e-13f;
        const float a8 = -0.976195544e-15f;
        const float out = a0 + dt * (a1 + dt * (a2 + dt * (a3 + dt
            * (a4 + dt * (a5 + dt * (a6 + dt * (a7 + a8 * dt)))))));
        return out * 100.0f;
    }
    return pow(10.0f, -7.90298f * (373.16f / t - 1.0f)
        + 5.02808f * p3_log10(373.16f / t)
        - 1.3816e-7f * (pow(10.0f, 11.344f * (1.0f - t / 373.16f)) - 1.0f)
        + 8.1328e-3f * (pow(10.0f, -3.49149f * (373.16f / t - 1.0f)) - 1.0f)
        + p3_log10(1013.246f)) * 100.0f;
}

static inline float p3_qv_sat(float t, float pressure, int ice,
                              device const float *pc) {
    const float e_pres = p3_polysvp1(t, ice);
    return P3_EP_2 * e_pres / fmax(1.0e-3f, pressure - e_pres);
}

kernel void p3_kloop1(
    constant P3Kloop1Resources &r [[buffer(0)]],
    constant P3Kloop1Params &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny) * uint(p.nx);
    if (column >= ncol) return;
    device const float *pc = r.constants;
    const float xxlv = p.xxlv, xxls = p.xxls, xlf = p.xlf;
    for (uint k = 0; k < uint(p.nz); ++k) {
        const ulong idx = ulong(k) * ulong(ncol) + ulong(column);
        const float t = r.temperature[idx];
        const float tm = r.tmparr1[idx];
        const float invexn = 1.0f / tm;
        const float rho = r.pressure[idx] / (P3_RD * t);
        const float inv_rho = 1.0f / rho;
        r.rho[idx] = rho;
        r.inv_rho[idx] = inv_rho;

        const float t_old = r.th_old[idx] * tm;
        float qvs = p3_qv_sat(fmax(t_old, 1.0f), r.pressure[idx], 0, pc);
        float qvi = p3_qv_sat(fmax(t_old, 1.0f), r.pressure[idx], 1, pc);
        qvs = fmax(qvs, 1.0e-20f);
        qvi = fmax(qvi, 1.0e-20f);
        r.qvs[idx] = qvs;
        r.qvi[idx] = qvi;

        r.ssat[idx] = r.qv_old[idx] - qvs;
        const float sup = r.qv_old[idx] / qvs - 1.0f;
        const float supi = r.qv_old[idx] / qvi - 1.0f;
        r.sup[idx] = sup;
        r.supi[idx] = supi;
        r.rhofacr[idx] = pow(P3_RHOSUR * inv_rho, 0.54f);
        r.rhofaci[idx] = pow(P3_RHOSUI * inv_rho, 0.54f);
        const float dum = 1.496e-6f * pow(t, 1.5f) / (t + 120.0f);
        r.acn[idx] = P3_G * P3_RHOW / (18.0f * dum);
        if (p.log_predict_nc == 0) r.nc[idx] = P3_NCCNST * inv_rho;

        if ((t < 273.15f && supi >= -0.05f)
            || (t >= 273.15f && sup >= -0.05f)) {
            r.flags[column] = 1.0f;
        }

        if (r.qc[idx] < P3_QSMALL
            || (r.qc[idx] < 1.0e-8f && sup < -0.1f)) {
            r.qv[idx] = r.qv[idx] + r.qc[idx];
            r.theta[idx] = r.theta[idx]
                - invexn * r.qc[idx] * xxlv * P3_INV_CP;
            r.qc[idx] = 0.0f;
            r.nc[idx] = 0.0f;
        } else {
            r.flags[ncol + column] = 1.0f;
        }

        if (r.qr[idx] < P3_QSMALL
            || (r.qr[idx] < 1.0e-8f && sup < -0.1f)) {
            r.qv[idx] = r.qv[idx] + r.qr[idx];
            r.theta[idx] = r.theta[idx]
                - invexn * r.qr[idx] * xxlv * P3_INV_CP;
            r.qr[idx] = 0.0f;
            r.nr[idx] = 0.0f;
        } else {
            r.flags[ncol + column] = 1.0f;
        }

        if (r.qi[idx] < P3_QSMALL
            || (r.qi[idx] < 1.0e-8f && supi < -0.1f)) {
            r.qv[idx] = r.qv[idx] + r.qi[idx];
            r.theta[idx] = r.theta[idx]
                - invexn * r.qi[idx] * xxls * P3_INV_CP;
            r.qi[idx] = 0.0f;
            r.ni[idx] = 0.0f;
            r.qir[idx] = 0.0f;
            r.qib[idx] = 0.0f;
        } else {
            r.flags[ncol + column] = 1.0f;
        }

        if (r.qi[idx] >= P3_QSMALL && r.qi[idx] < 1.0e-8f
            && t >= 273.15f) {
            r.qr[idx] = r.qr[idx] + r.qi[idx];
            r.nr[idx] = r.nr[idx] + r.ni[idx];
            r.theta[idx] = r.theta[idx]
                - invexn * r.qi[idx] * xlf * P3_INV_CP;
            r.qi[idx] = 0.0f;
            r.ni[idx] = 0.0f;
            r.qir[idx] = 0.0f;
            r.qib[idx] = 0.0f;
        }
        r.qv_cld[idx] = r.qv[idx];
    }
}
