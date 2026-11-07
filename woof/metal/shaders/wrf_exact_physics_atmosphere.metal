#include <metal_stdlib>
#include "exact_gfk_math.metalh"
#include "wrf_exact_physics_atmosphere_prepare_bindings.metalh"
using namespace metal;

// Source-ordered phy_prep products; scalar power follows physics_exner.cu.
kernel void wrf_exact_physics_atmosphere_prepare(
    constant WrfExactPhysicsAtmospherePrepareResources &r [[buffer(0)]],
    constant WrfExactPhysicsAtmospherePrepareParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong stride = ulong(p.ny) * ulong(p.nx);
    if (ulong(gid) >= stride) return;
    const ulong cell = ulong(gid);
#define A(name) reinterpret_cast<device const Exact32 *>(r.name)
#define O(name) reinterpret_cast<device Exact32 *>(r.name)
    for (int k = 0; k <= p.nz; ++k) {
        const ulong idx = ulong(k) * stride + cell;
        const ulong base = p.phb_3d ? idx : ulong(k);
        O(z_interface)[idx] = (A(phb)[base] + A(php)[idx]) / Exact32(9.81f);
    }
    for (int k = 0; k < p.nz; ++k) {
        const ulong idx = ulong(k) * stride + cell;
        const ulong base = p.thb_3d ? idx : ulong(k);
        const Exact32 theta = A(thb)[base] + A(thp)[idx];
        const Exact32 exner = exact_gfk_pow(A(p)[idx] / Exact32(100000.0f),
                                           Exact32(0.2857142984867096f));
        O(theta)[idx] = theta;
        O(exner)[idx] = exner;
        O(temperature)[idx] = theta * exner;
        const ulong uidx = ulong(k) * ulong(p.ny) * ulong(p.nx + 1)
                          + ulong(gid / uint(p.nx)) * ulong(p.nx + 1)
                          + ulong(gid % uint(p.nx));
        const ulong vidx = ulong(k) * ulong(p.ny + 1) * ulong(p.nx)
                          + ulong(gid / uint(p.nx)) * ulong(p.nx)
                          + ulong(gid % uint(p.nx));
        O(u_mass)[idx] = Exact32(0.5f) * (A(u)[uidx] + A(u)[uidx + 1]);
        O(v_mass)[idx] = Exact32(0.5f) * (A(v)[vidx] + A(v)[vidx + ulong(p.nx)]);
        O(dz)[idx] = A(z_interface)[idx + stride] - A(z_interface)[idx];
    }
    O(p_interface)[ulong(p.nz) * stride + cell] = Exact32(p.p_top);
    for (int k = p.nz - 1; k >= 0; --k) {
        const ulong idx = ulong(k) * stride + cell;
        Exact32 qtot(0.0f);
        if (p.has_qv) qtot = qtot + A(qv)[idx];
        if (p.has_qc) qtot = qtot + A(qc)[idx];
        if (p.has_qr) qtot = qtot + A(qr)[idx];
        if (p.has_qi) qtot = qtot + A(qi)[idx];
        if (p.has_qs) qtot = qtot + A(qs)[idx];
        if (p.has_qg) qtot = qtot + A(qg)[idx];
        if (p.has_qh) qtot = qtot + A(qh)[idx];
        const Exact32 layer = ((Exact32(1.0f) + qtot)
            * (A(c1h)[k] * A(total_mu)[cell] + A(c2h)[k])) * A(dnw)[k];
        O(p_interface)[idx] = A(p_interface)[idx + stride] - layer;
        O(pressure)[idx] = Exact32(0.5f) * (A(p_interface)[idx] + A(p_interface)[idx + stride]);
        const Exact32 qv = p.has_qv ? A(qv)[idx] : Exact32(0.0f);
        O(rho)[idx] = (Exact32(1.0f) / A(alt)[idx]) * (Exact32(1.0f) + qv);
    }
#undef A
#undef O
}
