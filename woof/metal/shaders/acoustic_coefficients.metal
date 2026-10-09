#include <metal_stdlib>
using namespace metal;

struct CalcCoefsParams {
    uint abi_size;
    uint nz;
    uint ny;
    uint nx;
    int moist_cq;
    int top_lid;
    int wrf_exact;
    float dtau;
    float epssm;
    float gravity;
    float gamma;
};

// Port of woof/core/kernels/acoustic.cu::calc_coefs. The thread mapping and
// assignment set preserve the one-column serial vertical recurrence; in
// particular alpha[0] and a[0] are intentionally not written.
kernel void calc_coefs(
    device const float *p [[buffer(0)]],
    device const float *alt [[buffer(1)]],
    device const float *mup [[buffer(2)]],
    device float *c2a [[buffer(3)]],
    device float *a [[buffer(4)]],
    device float *alpha [[buffer(5)]],
    device float *gam [[buffer(6)]],
    device const float *rdn [[buffer(7)]],
    device const float *rdnw [[buffer(8)]],
    device const float *c1h [[buffer(9)]],
    device const float *c2h [[buffer(10)]],
    device const float *c1f [[buffer(11)]],
    device const float *c2f [[buffer(12)]],
    device const float *mub2d [[buffer(13)]],
    device const float *cqw [[buffer(14)]],
    constant CalcCoefsParams &q [[buffer(15)]],
    uint column [[thread_position_in_grid]]) {
    const ulong st = ulong(q.ny) * ulong(q.nx);
    if (ulong(column) >= st) return;

    const float mut = mub2d[column] + mup[column];
    float cof = 0.5f * q.dtau * q.gravity * (1.0f + q.epssm);
    cof *= cof;

    for (uint k = 0; k < q.nz; ++k) {
        const ulong idx = ulong(k) * st + column;
        c2a[idx] = q.gamma * p[idx] / alt[idx];
    }

    gam[column] = 0.0f;
    a[st + column] = 0.0f;
    for (uint k = 2; k < q.nz; ++k) {
        const ulong idx = ulong(k) * st + column;
        const ulong prev = ulong(k - 1) * st + column;
        const float denom = (c1h[k - 1] * mut + c2h[k - 1])
                          * (c1f[k - 1] * mut + c2f[k - 1]);
        if (q.moist_cq) {
            a[idx] = -cqw[idx] * cof * rdn[k] * rdnw[k - 1]
                   * c2a[prev] / denom;
        } else {
            a[idx] = -cof * rdn[k] * rdnw[k - 1] * c2a[prev] / denom;
        }
    }

    if (q.top_lid) {
        a[ulong(q.nz) * st + column] = q.wrf_exact ? -0.0f : 0.0f;
    } else {
        const uint k = q.nz - 1;
        const float denom = (c1h[k] * mut + c2h[k])
                          * (c1f[k] * mut + c2f[k]);
        a[ulong(q.nz) * st + column] = -2.0f * cof * rdnw[k] * rdnw[k]
                                     * c2a[ulong(k) * st + column] / denom;
    }

    for (uint k = 1; k < q.nz; ++k) {
        const ulong idx = ulong(k) * st + column;
        const float chm_k = c1h[k] * mut + c2h[k];
        const float chm_km = c1h[k - 1] * mut + c2h[k - 1];
        const float cfm_k = c1f[k] * mut + c2f[k];
        const float cfm_kp = c1f[k + 1] * mut + c2f[k + 1];
        float b;
        float cc;
        if (q.moist_cq) {
            const float cq = cqw[idx];
            b = 1.0f + cq * cof * rdn[k]
                * (rdnw[k] * c2a[idx] / (chm_k * cfm_k)
                   + rdnw[k - 1] * c2a[ulong(k - 1) * st + column]
                     / (chm_km * cfm_k));
            cc = -cq * cof * rdn[k] * rdnw[k] * c2a[idx]
               / (chm_k * cfm_kp);
        } else {
            b = 1.0f + cof * rdn[k]
                * (rdnw[k] * c2a[idx] / (chm_k * cfm_k)
                   + rdnw[k - 1] * c2a[ulong(k - 1) * st + column]
                     / (chm_km * cfm_k));
            cc = -cof * rdn[k] * rdnw[k] * c2a[idx]
               / (chm_k * cfm_kp);
        }
        const float al = 1.0f / (b - a[idx] * gam[ulong(k - 1) * st + column]);
        alpha[idx] = al;
        gam[idx] = cc * al;
    }

    const uint top = q.nz - 1;
    const float b_top = 1.0f + 2.0f * cof * rdnw[top] * rdnw[top]
        * c2a[ulong(top) * st + column]
        / ((c1h[top] * mut + c2h[top])
           * (c1f[q.nz] * mut + c2f[q.nz]));
    const ulong top_idx = ulong(q.nz) * st + column;
    alpha[top_idx] = 1.0f / (b_top - a[top_idx]
                                 * gam[ulong(top) * st + column]);
    gam[top_idx] = 0.0f;
}
