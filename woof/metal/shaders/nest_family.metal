#include <metal_stdlib>
#include "nest_blend_terrain_bindings.metalh"
#include "nest_copy_fcn_bindings.metalh"
#include "nest_copy_fcn_theta_bindings.metalh"
#include "nest_copy_fcnm_bindings.metalh"
#include "nest_smooth_j_bindings.metalh"
#include "nest_smooth_i_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

static inline ulong nest_i3(int k, int j, int i, int ny, int nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx) + ulong(i);
}

kernel void nest_blend_terrain(
    constant NestBlendTerrainResources &r [[buffer(0)]],
    constant NestBlendTerrainParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong total = ulong(p.nk) * ulong(p.ny) * ulong(p.nx);
    if (ulong(tid) >= total) return;
    const int i = int(tid % uint(p.nx)) + 1 + p.i0;
    const int j = int((tid / uint(p.nx)) % uint(p.ny)) + 1 + p.j0;
    const int ide = p.domain_nx + 1;
    const int jde = p.domain_ny + 1;
    const float fine = r.fine[tid];
    const float coarse = r.interpolated[tid];
    const float reciprocal = 1.0f / float(p.blend_width + 1);
    float value = fine;
    for (int blend_cell = p.blend_width; blend_cell >= 1; --blend_cell) {
        if (i == p.spec_bdy_width + blend_cell ||
            j == p.spec_bdy_width + blend_cell ||
            i == ide - p.spec_bdy_width - blend_cell ||
            j == jde - p.spec_bdy_width - blend_cell) {
            value = ((float(blend_cell) * fine +
                      float(p.blend_width + 1 - blend_cell) * coarse) * reciprocal);
        }
    }
    if (i <= p.spec_bdy_width || j <= p.spec_bdy_width ||
        i >= ide - p.spec_bdy_width || j >= jde - p.spec_bdy_width)
        value = coarse;
    r.fine[tid] = value;
}

kernel void nest_copy_fcn(
    constant NestCopyFcnResources &r [[buffer(0)]],
    constant NestCopyFcnParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong total = ulong(p.nz) * ulong(p.loop_ni) * ulong(p.loop_nj);
    if (ulong(tid) >= total || p.loop_ni <= 0 || p.loop_nj <= 0) return;
    const int ci = p.loop_i0 + int(tid % uint(p.loop_ni)) + 1;
    const int cj = p.loop_j0 + int((tid / uint(p.loop_ni)) % uint(p.loop_nj)) + 1;
    const int k = int(tid / (ulong(p.loop_ni) * ulong(p.loop_nj)));
    const bool odd = (p.nrj % 2) != 0;
    int ni, nj, ij_lo, ij_hi, ij_stride, sub;
    float weight;
    if (odd) {
        if (!p.xstag && !p.ystag) {
            ni = (ci - p.ipos) * p.nri + p.nri / 2 + 1;
            nj = (cj - p.jpos) * p.nrj + p.nrj / 2 + 1;
            ij_lo = 1; ij_hi = p.nri * p.nrj; ij_stride = 1;
            weight = 1.0f / float(p.nri * p.nrj);
        } else if (p.xstag) {
            ni = (ci - p.ipos) * p.nri + 1;
            nj = (cj - p.jpos) * p.nrj + p.nrj / 2 + 1;
            ij_lo = (p.nri + 1) / 2;
            ij_hi = (p.nri + 1) / 2 + p.nri * (p.nri - 1);
            ij_stride = p.nri;
            weight = 1.0f / float(p.nri);
        } else {
            ni = (ci - p.ipos) * p.nri + p.nri / 2 + 1;
            nj = (cj - p.jpos) * p.nrj + 1;
            ij_lo = (p.nrj * p.nrj + 1) / 2 - p.nrj / 2;
            ij_hi = ij_lo + p.nrj - 1;
            ij_stride = 1;
            weight = 1.0f / float(p.nrj);
        }
        sub = 1;
    } else {
        ni = (ci - p.ipos) * p.nri + 1;
        nj = (cj - p.jpos) * p.nrj + 1;
        if (!p.xstag && !p.ystag) {
            ij_lo = 1; ij_hi = p.nri * p.nrj; ij_stride = 1;
            weight = 1.0f / float(p.nri * p.nrj);
        } else if (p.xstag) {
            ij_lo = 1; ij_hi = p.nri * p.nrj; ij_stride = p.nri;
            weight = 1.0f / float(p.nri);
        } else {
            ij_lo = 1; ij_hi = p.nri; ij_stride = 1;
            weight = 1.0f / float(p.nri);
        }
        sub = 0;
    }
    float acc = 0.0f;
    for (int ij = ij_lo; ij <= ij_hi; ij += ij_stride) {
        const int ipoints = (ij - 1) % p.nri + sub * (1 - p.nri / 2 - 1);
        const int jpoints = (ij - 1) / p.nri + sub * (1 - p.nrj / 2 - 1);
        acc = acc + weight * r.child[nest_i3(
            k, nj + jpoints - 1 - p.child_j0,
            ni + ipoints - 1 - p.child_i0, p.nyc, p.nxc)];
    }
    r.parent[nest_i3(k, cj - 1 - p.parent_j0,
                     ci - 1 - p.parent_i0, p.nyp, p.nxp)] = acc;
}

kernel void nest_copy_fcn_theta(
    constant NestCopyFcnThetaResources &r [[buffer(0)]],
    constant NestCopyFcnThetaParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong total = ulong(p.nz) * ulong(p.loop_ni) * ulong(p.loop_nj);
    if (ulong(tid) >= total || p.loop_ni <= 0 || p.loop_nj <= 0) return;
    const int ci = p.loop_i0 + int(tid % uint(p.loop_ni)) + 1;
    const int cj = p.loop_j0 + int((tid / uint(p.loop_ni)) % uint(p.loop_nj)) + 1;
    const int k = int(tid / (ulong(p.loop_ni) * ulong(p.loop_nj)));
    const bool odd = (p.nrj % 2) != 0;
    int ni, nj, ij_lo, ij_hi, ij_stride, sub;
    float weight;
    if (odd) {
        if (!p.xstag && !p.ystag) {
            ni = (ci - p.ipos) * p.nri + p.nri / 2 + 1;
            nj = (cj - p.jpos) * p.nrj + p.nrj / 2 + 1;
            ij_lo = 1; ij_hi = p.nri * p.nrj; ij_stride = 1;
            weight = 1.0f / float(p.nri * p.nrj);
        } else if (p.xstag) {
            ni = (ci - p.ipos) * p.nri + 1;
            nj = (cj - p.jpos) * p.nrj + p.nrj / 2 + 1;
            ij_lo = (p.nri + 1) / 2;
            ij_hi = (p.nri + 1) / 2 + p.nri * (p.nri - 1);
            ij_stride = p.nri;
            weight = 1.0f / float(p.nri);
        } else {
            ni = (ci - p.ipos) * p.nri + p.nri / 2 + 1;
            nj = (cj - p.jpos) * p.nrj + 1;
            ij_lo = (p.nrj * p.nrj + 1) / 2 - p.nrj / 2;
            ij_hi = ij_lo + p.nrj - 1;
            ij_stride = 1;
            weight = 1.0f / float(p.nrj);
        }
        sub = 1;
    } else {
        ni = (ci - p.ipos) * p.nri + 1;
        nj = (cj - p.jpos) * p.nrj + 1;
        if (!p.xstag && !p.ystag) {
            ij_lo = 1; ij_hi = p.nri * p.nrj; ij_stride = 1;
            weight = 1.0f / float(p.nri * p.nrj);
        } else if (p.xstag) {
            ij_lo = 1; ij_hi = p.nri * p.nrj; ij_stride = p.nri;
            weight = 1.0f / float(p.nri);
        } else {
            ij_lo = 1; ij_hi = p.nri; ij_stride = 1;
            weight = 1.0f / float(p.nri);
        }
        sub = 0;
    }
    float acc = 0.0f;
    for (int ij = ij_lo; ij <= ij_hi; ij += ij_stride) {
        const int ipoints = (ij - 1) % p.nri + sub * (1 - p.nri / 2 - 1);
        const int jpoints = (ij - 1) / p.nri + sub * (1 - p.nrj / 2 - 1);
        const int child_i = ni + ipoints - 1 - p.child_i0;
        const int child_j = nj + jpoints - 1 - p.child_j0;
        const ulong child_index = nest_i3(
            k, child_j, child_i, p.nyc, p.nxc);
        float value = r.child[child_index];
        if (p.transform_child) {
            const ulong base_index = p.child_thb_3d
                ? child_index : ulong(k);
            value = value + r.child_thb[base_index];
            value = value - 300.0f;
        }
        acc = acc + weight * value;
    }
    const ulong parent_index = nest_i3(
        k, cj - 1 - p.parent_j0, ci - 1 - p.parent_i0, p.nyp, p.nxp);
    const ulong parent_base_index = p.parent_thb_3d
        ? parent_index : ulong(k);
    float result = acc + 300.0f;
    result = result - r.parent_thb[parent_base_index];
    r.parent[parent_index] = result;
}

static inline bool nest_feedback_cell(
    ulong tid, constant NestCopyFcnmParams &p,
    thread int &ci, thread int &cj, thread int &k) {
    const int ni_span = p.xstag ? p.nxc - 1 : p.nxc;
    const int nj_span = p.ystag ? p.nyc - 1 : p.nyc;
    const int istag = p.xstag ? 0 : 1;
    const int jstag = p.ystag ? 0 : 1;
    const int ci_lo = p.ipos + p.spec_zone;
    const int ci_hi = p.ipos + ni_span / p.nri - istag - p.spec_zone;
    const int cj_lo = p.jpos + p.spec_zone;
    const int cj_hi = p.jpos + nj_span / p.nrj - jstag - p.spec_zone;
    const int nci = ci_hi - ci_lo + 1;
    const int ncj = cj_hi - cj_lo + 1;
    if (nci <= 0 || ncj <= 0 || tid >= ulong(p.nz) * ulong(nci) * ulong(ncj))
        return false;
    ci = ci_lo + int(tid % ulong(nci));
    cj = cj_lo + int((tid / ulong(nci)) % ulong(ncj));
    k = int(tid / (ulong(nci) * ulong(ncj)));
    return true;
}

kernel void nest_copy_fcnm(
    constant NestCopyFcnmResources &r [[buffer(0)]],
    constant NestCopyFcnmParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    int ci, cj, k;
    if (!nest_feedback_cell(ulong(tid), p, ci, cj, k)) return;
    const int istag = p.xstag ? 0 : 1;
    const int jstag = p.ystag ? 0 : 1;
    int ni, nj;
    if ((p.nrj % 2) != 0) {
        ni = (ci - p.ipos) * p.nri + istag + 1;
        nj = (cj - p.jpos) * p.nrj + jstag + 1;
    } else {
        ni = (ci - p.ipos) * p.nri + 1 + (p.nri / 2 - 1);
        nj = (cj - p.jpos) * p.nrj + 1 + (p.nrj / 2 - 1);
    }
    r.parent[nest_i3(k, cj - 1, ci - 1, p.nyp, p.nxp)] =
        r.child[nest_i3(k, nj - 1, ni - 1, p.nyc, p.nxc)];
}

static inline float nest_smooth_filter(int mode, float xnu,
                                       float a, float b, float c) {
    if (mode == 0) return 0.25f * ((c + 2.0f * b) + a);
    return b + xnu * ((c + a) * 0.5f - b);
}

kernel void nest_smooth_j(
    constant NestSmoothJResources &r [[buffer(0)]],
    constant NestSmoothJParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int nit = p.niw + 2;
    const int njt = p.njw + 2;
    const ulong total = ulong(p.nz) * ulong(njt) * ulong(nit);
    if (ulong(tid) >= total) return;
    const int i = p.i0 - 1 + int(tid % uint(nit));
    const int j = p.j0 - 1 + int((tid / uint(nit)) % uint(njt));
    const int k = int(tid / (ulong(nit) * ulong(njt)));
    const bool inside = i >= p.i0 && i < p.i0 + p.niw &&
                        j >= p.j0 && j < p.j0 + p.njw;
    const ulong at = nest_i3(k, j, i, p.nyp, p.nxp);
    const float b = r.parent[at];
    if (!inside) { r.scratch[at] = b; return; }
    const float a = r.parent[nest_i3(k, j - 1, i, p.nyp, p.nxp)];
    const float c = r.parent[nest_i3(k, j + 1, i, p.nyp, p.nxp)];
    r.scratch[at] = nest_smooth_filter(p.mode, p.xnu, a, b, c);
}

kernel void nest_smooth_i(
    constant NestSmoothIResources &r [[buffer(0)]],
    constant NestSmoothIParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong total = ulong(p.nz) * ulong(p.njw) * ulong(p.niw);
    if (ulong(tid) >= total) return;
    const int i = p.i0 + int(tid % uint(p.niw));
    const int j = p.j0 + int((tid / uint(p.niw)) % uint(p.njw));
    const int k = int(tid / (ulong(p.niw) * ulong(p.njw)));
    const float a = r.scratch[nest_i3(k, j, i - 1, p.nyp, p.nxp)];
    const float b = r.scratch[nest_i3(k, j, i, p.nyp, p.nxp)];
    const float c = r.scratch[nest_i3(k, j, i + 1, p.nyp, p.nxp)];
    r.parent[nest_i3(k, j, i, p.nyp, p.nxp)] =
        nest_smooth_filter(p.mode, p.xnu, a, b, c);
}
