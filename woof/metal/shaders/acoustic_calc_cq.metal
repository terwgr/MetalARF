#include <metal_stdlib>
#include "calc_cq_bindings.metalh"
using namespace metal;

#pragma clang fp contract(off)

static inline float cq_pair(device const float *qv,
                            device const float *qc,
                            device const float *qr,
                            device const float *qi,
                            device const float *qs,
                            device const float *qg,
                            device const float *qh,
                            ulong a, ulong b, int n_mass) {
    float qtot = 0.0f;
    qtot = qtot + qv[a];
    qtot = qtot + qv[b];
    if (n_mass >= 3) {
        qtot = qtot + qc[a];
        qtot = qtot + qc[b];
        qtot = qtot + qr[a];
        qtot = qtot + qr[b];
    }
    if (n_mass >= 6) {
        qtot = qtot + qi[a];
        qtot = qtot + qi[b];
        qtot = qtot + qs[a];
        qtot = qtot + qs[b];
        qtot = qtot + qg[a];
        qtot = qtot + qg[b];
    }
    if (n_mass == 7) {
        qtot = qtot + qh[a];
        qtot = qtot + qh[b];
    }
    return 1.0f / (1.0f + 0.5f * qtot);
}

// WRF calc_cq: build periodic u/v face coefficients and interior w-face
// coefficients from the active moist-package mass fields. Number moments are
// not included; the unused lower/upper w faces are unity.
kernel void calc_cq(constant CalcCqResources &r [[buffer(0)]],
                    constant CalcCqParams &q [[buffer(1)]],
                    uint tid [[thread_position_in_grid]]) {
    const ulong nz = ulong(q.nz), ny = ulong(q.ny), nx = ulong(q.nx);
    const ulong nxf = nx + 1, nyf = ny + 1;
    const ulong nu = nz * ny * nxf;
    const ulong nv = nz * nyf * nx;
    const ulong nw = (nz + 1) * ny * nx;

    if (ulong(tid) < nu) {
        const ulong k = ulong(tid) / (ny * nxf);
        const ulong rem = ulong(tid) - k * ny * nxf;
        const ulong j = rem / nxf, i = rem - j * nxf;
        const ulong ia = i % nx;
        const ulong ib = (i + nx - 1) % nx;
        const ulong a = (k * ny + j) * nx + ia;
        const ulong b = (k * ny + j) * nx + ib;
        r.cqu[tid] = cq_pair(r.qv, r.qc, r.qr, r.qi, r.qs, r.qg, r.qh,
                             a, b, q.n_mass);
    }
    if (ulong(tid) < nv) {
        const ulong k = ulong(tid) / (nyf * nx);
        const ulong rem = ulong(tid) - k * nyf * nx;
        const ulong j = rem / nx, i = rem - j * nx;
        const ulong ja = j % ny;
        const ulong jb = (j + ny - 1) % ny;
        const ulong a = (k * ny + ja) * nx + i;
        const ulong b = (k * ny + jb) * nx + i;
        r.cqv[tid] = cq_pair(r.qv, r.qc, r.qr, r.qi, r.qs, r.qg, r.qh,
                             a, b, q.n_mass);
    }
    if (ulong(tid) < nw) {
        const ulong k = ulong(tid) / (ny * nx);
        const ulong rem = ulong(tid) - k * ny * nx;
        if (k == 0 || k == nz) {
            r.cqw[tid] = 1.0f;
        } else {
            const ulong a = k * ny * nx + rem;
            const ulong b = (k - 1) * ny * nx + rem;
            r.cqw[tid] = cq_pair(r.qv, r.qc, r.qr, r.qi, r.qs, r.qg, r.qh,
                                 a, b, q.n_mass);
        }
    }
}
