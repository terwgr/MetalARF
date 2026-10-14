#include <metal_stdlib>
#include "physics_atmosphere_prepare_bindings.metalh"
#include "pressure_power.metalh"
using namespace metal;

// Keep the field-wise FP32 order used by woof.core.physics._prepare_atmosphere.
#pragma clang fp contract(off)

static inline ulong pa_index(uint k, ulong cell, ulong stride) {
    return ulong(k) * stride + cell;
}

static inline float pa_base(device const float *field, uint k, ulong cell,
                            ulong stride, bool is3d) {
    return field[is3d ? pa_index(k, cell, stride) : ulong(k)];
}

static inline float pa_qtotal(device const float *qv, device const float *qc,
        device const float *qr, device const float *qi,
        device const float *qs, device const float *qg,
        device const float *qh, ulong idx, bool hv, bool hc, bool hr,
        bool hi, bool hs, bool hg, bool hh) {
    float q = 0.0f;
    if (hv) q = q + qv[idx];
    if (hc) q = q + qc[idx];
    if (hr) q = q + qr[idx];
    if (hi) q = q + qi[idx];
    if (hs) q = q + qs[idx];
    if (hg) q = q + qg[idx];
    if (hh) q = q + qh[idx];
    return q;
}

kernel void physics_atmosphere_prepare(
    constant PhysicsAtmospherePrepareResources &r [[buffer(0)]],
    constant PhysicsAtmospherePrepareParams &p [[buffer(1)]],
    uint cell_id [[thread_position_in_grid]]) {
    const uint nz = uint(p.nz), ny = uint(p.ny), nx = uint(p.nx);
    const ulong stride = ulong(ny) * ulong(nx);
    if (ulong(cell_id) >= stride) return;
    const ulong cell = ulong(cell_id);
    const bool thb3d = bool(p.thb_3d), phb3d = bool(p.phb_3d);
    const bool hv = bool(p.has_qv), hc = bool(p.has_qc);
    const bool hr = bool(p.has_qr), hi = bool(p.has_qi);
    const bool hs = bool(p.has_qs), hg = bool(p.has_qg);
    const bool hh = bool(p.has_qh);

    // Surface interface initialization is followed by the same descending
    // Fortran loop as the Python/CuPy transcription.
    const ulong bottom_interface = ulong(nz) * stride + cell;
    r.p_interface[bottom_interface] = p.p_top;
    for (uint k = 0; k < nz + 1; ++k) {
        const ulong idxi = pa_index(k, cell, stride);
        const float phb = pa_base(r.phb, k, cell, stride, phb3d);
        r.z_interface[idxi] = (phb + r.php[idxi]) / 9.81f;
    }
    for (uint k = 0; k < nz; ++k) {
        const ulong idx = pa_index(k, cell, stride);
        const float theta = pa_base(r.thb, k, cell, stride, thb3d) + r.thp[idx];
        const float exner = fp32_pressure_power(
            r.p[idx], 100000.0f, 0.2857142984867096f);
        r.theta[idx] = theta;
        r.exner[idx] = exner;
        r.temperature[idx] = theta * exner;
        const ulong uleft = ulong(k) * ulong(ny) * ulong(nx + 1)
                          + ulong(cell_id / nx) * ulong(nx + 1)
                          + ulong(cell_id % nx);
        const ulong vbelow = ulong(k) * ulong(ny + 1) * ulong(nx)
                           + ulong(cell_id / nx) * ulong(nx)
                           + ulong(cell_id % nx);
        r.u_mass[idx] = 0.5f * (r.u[uleft] + r.u[uleft + 1]);
        r.v_mass[idx] = 0.5f * (r.v[vbelow] + r.v[vbelow + ulong(nx)]);
        const ulong zidx = pa_index(k, cell, stride);
        r.dz[idx] = r.z_interface[zidx + stride] - r.z_interface[zidx];
    }

    // Each column is one thread, preserving the sequential downward
    // hydrostatic integration order without atomics or a CPU staging path.
    for (int k = int(nz) - 1; k >= 0; --k) {
        const ulong idx = pa_index(uint(k), cell, stride);
        float qtot = pa_qtotal(r.qv, r.qc, r.qr, r.qi, r.qs, r.qg, r.qh,
                               idx, hv, hc, hr, hi, hs, hg, hh);
        const float moist = 1.0f + qtot;
        const float base_layer = (r.c1h[k] * r.total_mu[cell]) + r.c2h[k];
        const float layer = (moist * base_layer) * r.dnw[k];
        const ulong lower = pa_index(uint(k) + 1, cell, stride);
        r.p_interface[idx] = r.p_interface[lower] - layer;
        r.pressure[idx] = 0.5f * (r.p_interface[idx] + r.p_interface[lower]);
        r.rho[idx] = (1.0f + (hv ? r.qv[idx] : 0.0f)) / r.alt[idx];
    }
}

