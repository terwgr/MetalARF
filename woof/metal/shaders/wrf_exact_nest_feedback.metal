// Strict feedback restriction, matching the CUDA copy_fcn operation order.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_nest_feedback_bindings.metalh"
using namespace metal;
static inline ulong nest_i3(int k, int j, int i, int ny, int nx) { return (ulong(k)*ulong(ny)+ulong(j))*ulong(nx)+ulong(i); }
kernel void wrf_exact_nest_copy_fcn(
    constant WrfExactNestCopyFcnResources &r [[buffer(0)]],
    constant WrfExactNestCopyFcnParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong total = ulong(p.nz) * ulong(p.loop_ni) * ulong(p.loop_nj);
    if (ulong(tid) >= total || p.loop_ni <= 0 || p.loop_nj <= 0) return;
    const int ci = p.loop_i0 + int(tid % uint(p.loop_ni)) + 1;
    const int cj = p.loop_j0 + int((tid / uint(p.loop_ni)) % uint(p.loop_nj)) + 1;
    const int k = int(tid / (ulong(p.loop_ni) * ulong(p.loop_nj)));
    const bool odd = (p.nrj % 2) != 0;
    int ni, nj, ij_lo, ij_hi, ij_stride, sub;
    Exact32 weight;
    if (odd) {
        if (!p.xstag && !p.ystag) {
            ni = (ci - p.ipos) * p.nri + p.nri / 2 + 1;
            nj = (cj - p.jpos) * p.nrj + p.nrj / 2 + 1;
            ij_lo = 1; ij_hi = p.nri * p.nrj; ij_stride = 1;
            weight = Exact32(1.0f) / Exact32(float(p.nri * p.nrj));
        } else if (p.xstag) {
            ni = (ci - p.ipos) * p.nri + 1;
            nj = (cj - p.jpos) * p.nrj + p.nrj / 2 + 1;
            ij_lo = (p.nri + 1) / 2;
            ij_hi = (p.nri + 1) / 2 + p.nri * (p.nri - 1);
            ij_stride = p.nri;
            weight = Exact32(1.0f) / Exact32(float(p.nri));
        } else {
            ni = (ci - p.ipos) * p.nri + p.nri / 2 + 1;
            nj = (cj - p.jpos) * p.nrj + 1;
            ij_lo = (p.nrj * p.nrj + 1) / 2 - p.nrj / 2;
            ij_hi = ij_lo + p.nrj - 1;
            ij_stride = 1;
            weight = Exact32(1.0f) / Exact32(float(p.nrj));
        }
        sub = 1;
    } else {
        ni = (ci - p.ipos) * p.nri + 1;
        nj = (cj - p.jpos) * p.nrj + 1;
        if (!p.xstag && !p.ystag) {
            ij_lo = 1; ij_hi = p.nri * p.nrj; ij_stride = 1;
            weight = Exact32(1.0f) / Exact32(float(p.nri * p.nrj));
        } else if (p.xstag) {
            ij_lo = 1; ij_hi = p.nri * p.nrj; ij_stride = p.nri;
            weight = Exact32(1.0f) / Exact32(float(p.nri));
        } else {
            ij_lo = 1; ij_hi = p.nri; ij_stride = 1;
            weight = Exact32(1.0f) / Exact32(float(p.nri));
        }
        sub = 0;
    }
    Exact32 acc = Exact32(0.0f);
    for (int ij = ij_lo; ij <= ij_hi; ij += ij_stride) {
        const int ipoints = (ij - 1) % p.nri + sub * (1 - p.nri / 2 - 1);
        const int jpoints = (ij - 1) / p.nri + sub * (1 - p.nrj / 2 - 1);
        acc = acc + weight * reinterpret_cast<device const Exact32 *>(r.child)[nest_i3(
            k, nj + jpoints - 1 - p.child_j0,
            ni + ipoints - 1 - p.child_i0, p.nyc, p.nxc)];
    }
    reinterpret_cast<device Exact32 *>(r.parent)[nest_i3(k, cj - 1 - p.parent_j0,
                     ci - 1 - p.parent_i0, p.nyp, p.nxp)] = acc;
}

kernel void wrf_exact_nest_copy_fcn_theta(
    constant WrfExactNestCopyFcnThetaResources &r [[buffer(0)]],
    constant WrfExactNestCopyFcnThetaParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong total = ulong(p.nz) * ulong(p.loop_ni) * ulong(p.loop_nj);
    if (ulong(tid) >= total || p.loop_ni <= 0 || p.loop_nj <= 0) return;
    const int ci = p.loop_i0 + int(tid % uint(p.loop_ni)) + 1;
    const int cj = p.loop_j0 + int((tid / uint(p.loop_ni)) % uint(p.loop_nj)) + 1;
    const int k = int(tid / (ulong(p.loop_ni) * ulong(p.loop_nj)));
    const bool odd = (p.nrj % 2) != 0;
    int ni, nj, ij_lo, ij_hi, ij_stride, sub;
    Exact32 weight;
    if (odd) {
        if (!p.xstag && !p.ystag) {
            ni = (ci - p.ipos) * p.nri + p.nri / 2 + 1;
            nj = (cj - p.jpos) * p.nrj + p.nrj / 2 + 1;
            ij_lo = 1; ij_hi = p.nri * p.nrj; ij_stride = 1;
            weight = Exact32(1.0f) / Exact32(float(p.nri * p.nrj));
        } else if (p.xstag) {
            ni = (ci - p.ipos) * p.nri + 1;
            nj = (cj - p.jpos) * p.nrj + p.nrj / 2 + 1;
            ij_lo = (p.nri + 1) / 2;
            ij_hi = (p.nri + 1) / 2 + p.nri * (p.nri - 1);
            ij_stride = p.nri;
            weight = Exact32(1.0f) / Exact32(float(p.nri));
        } else {
            ni = (ci - p.ipos) * p.nri + p.nri / 2 + 1;
            nj = (cj - p.jpos) * p.nrj + 1;
            ij_lo = (p.nrj * p.nrj + 1) / 2 - p.nrj / 2;
            ij_hi = ij_lo + p.nrj - 1;
            ij_stride = 1;
            weight = Exact32(1.0f) / Exact32(float(p.nrj));
        }
        sub = 1;
    } else {
        ni = (ci - p.ipos) * p.nri + 1;
        nj = (cj - p.jpos) * p.nrj + 1;
        if (!p.xstag && !p.ystag) {
            ij_lo = 1; ij_hi = p.nri * p.nrj; ij_stride = 1;
            weight = Exact32(1.0f) / Exact32(float(p.nri * p.nrj));
        } else if (p.xstag) {
            ij_lo = 1; ij_hi = p.nri * p.nrj; ij_stride = p.nri;
            weight = Exact32(1.0f) / Exact32(float(p.nri));
        } else {
            ij_lo = 1; ij_hi = p.nri; ij_stride = 1;
            weight = Exact32(1.0f) / Exact32(float(p.nri));
        }
        sub = 0;
    }
    Exact32 acc = Exact32(0.0f);
    for (int ij = ij_lo; ij <= ij_hi; ij += ij_stride) {
        const int ipoints = (ij - 1) % p.nri + sub * (1 - p.nri / 2 - 1);
        const int jpoints = (ij - 1) / p.nri + sub * (1 - p.nrj / 2 - 1);
        const int child_i = ni + ipoints - 1 - p.child_i0;
        const int child_j = nj + jpoints - 1 - p.child_j0;
        const ulong child_index = nest_i3(
            k, child_j, child_i, p.nyc, p.nxc);
        Exact32 value = reinterpret_cast<device const Exact32 *>(r.child)[child_index];
        if (p.transform_child) {
            const ulong base_index = p.child_thb_3d
                ? child_index : ulong(k);
            value = value + reinterpret_cast<device const Exact32 *>(r.child_thb)[base_index];
            value = value - Exact32(300.0f);
        }
        acc = acc + weight * value;
    }
    const ulong parent_index = nest_i3(
        k, cj - 1 - p.parent_j0, ci - 1 - p.parent_i0, p.nyp, p.nxp);
    const ulong parent_base_index = p.parent_thb_3d
        ? parent_index : ulong(k);
    Exact32 result = acc + Exact32(300.0f);
    result = result - reinterpret_cast<device const Exact32 *>(r.parent_thb)[parent_base_index];
    reinterpret_cast<device Exact32 *>(r.parent)[parent_index] = result;
}
