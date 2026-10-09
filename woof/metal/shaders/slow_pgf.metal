#include <metal_stdlib>
#include "slow_pgf_bindings.metalh"
using namespace metal;

// This kernel follows the ordinary slow_pgf CUDA source. CuPy's former
// operator temporaries round each FP32 operation, so contraction stays off.
#pragma clang fp contract(off)

static inline ulong cell_index(uint k, uint j, uint i, uint ny, uint nx) {
    return (ulong(k) * ulong(ny) + ulong(j)) * ulong(nx) + ulong(i);
}

static inline float base_value(device const float *field, uint k, ulong cell,
                               ulong stride, bool base3d) {
    return field[base3d ? ulong(k) * stride + cell : ulong(k)];
}

static inline float pressure_perturbation(
    device const float *p, device const float *pb, uint k, ulong cell,
    ulong stride, bool base3d) {
    return p[ulong(k) * stride + cell]
         - base_value(pb, k, cell, stride, base3d);
}

static inline float dpn_value(
    device const float *p, device const float *pb, uint kf, ulong cell,
    ulong stride, uint nz, bool base3d, device const float *fnm,
    device const float *fnp, float cf1, float cf2, float cf3,
    bool top_lid, float cfn, float cfn1) {
    if (kf == nz) {
        if (!top_lid) return 0.0f;
        const float top = cfn * pressure_perturbation(
            p, pb, nz - 1, cell, stride, base3d);
        const float below = cfn1 * pressure_perturbation(
            p, pb, nz - 2, cell, stride, base3d);
        return top + below;
    }
    if (kf == 0) {
        const float t0 = cf1 * pressure_perturbation(
            p, pb, 0, cell, stride, base3d);
        const float t1 = cf2 * pressure_perturbation(
            p, pb, 1, cell, stride, base3d);
        const float t2 = cf3 * pressure_perturbation(
            p, pb, 2, cell, stride, base3d);
        return (t0 + t1) + t2;
    }
    const float high = fnm[kf] * pressure_perturbation(
        p, pb, kf, cell, stride, base3d);
    const float low = fnp[kf] * pressure_perturbation(
        p, pb, kf - 1, cell, stride, base3d);
    return high + low;
}

static inline void dpn_pair_value(
    device const float *p, device const float *pb, uint k, ulong cell,
    ulong stride, uint nz, bool base3d, device const float *fnm,
    device const float *fnp, float cf1, float cf2, float cf3,
    bool top_lid, float cfn, float cfn1, float pp_k,
    thread float &high, thread float &low) {
    if (nz < 3) {
        high = dpn_value(p, pb, k + 1, cell, stride, nz, base3d,
                         fnm, fnp, cf1, cf2, cf3, top_lid, cfn, cfn1);
        low = dpn_value(p, pb, k, cell, stride, nz, base3d,
                        fnm, fnp, cf1, cf2, cf3, top_lid, cfn, cfn1);
        return;
    }
    const float pp_below = k > 0
        ? pressure_perturbation(p, pb, k - 1, cell, stride, base3d) : 0.0f;
    const float pp_above = k + 1 < nz
        ? pressure_perturbation(p, pb, k + 1, cell, stride, base3d) : 0.0f;
    if (k + 1 == nz) {
        if (top_lid) {
            const float top = cfn * pp_k;
            const float below = cfn1 * pp_below;
            high = top + below;
        } else {
            high = 0.0f;
        }
    } else {
        const float upper = fnm[k + 1] * pp_above;
        const float lower = fnp[k + 1] * pp_k;
        high = upper + lower;
    }
    if (k == 0) {
        const float t0 = cf1 * pp_k;
        const float t1 = cf2 * pp_above;
        const float t2 = cf3 * pressure_perturbation(
            p, pb, 2, cell, stride, base3d);
        low = (t0 + t1) + t2;
    } else {
        const float upper = fnm[k] * pp_k;
        const float lower = fnp[k] * pp_below;
        low = upper + lower;
    }
}

static inline float php_half_value(
    device const float *php, device const float *phb, uint k, ulong cell,
    ulong stride, bool base3d) {
    const float perturbation = php[ulong(k) * stride + cell]
                             + php[ulong(k + 1) * stride + cell];
    if (!base3d) return 0.5f * perturbation;
    const float base = phb[ulong(k) * stride + cell]
                     + phb[ulong(k + 1) * stride + cell];
    return 0.5f * (base + perturbation);
}

static inline float pgf_face(
    ulong ca, ulong cb, uint k, ulong stride, uint nz, float rd,
    float half_rd, device const float *p, device const float *pb,
    device const float *al, device const float *alt,
    device const float *php, device const float *phb,
    device const float *mup, device const float *mub2d,
    device const float *c1h, device const float *c2h,
    device const float *rdnw, device const float *fnm,
    device const float *fnp, float cf1, float cf2, float cf3,
    bool base3d, bool top_lid, float cfn, float cfn1) {
    const ulong a = ulong(k) * stride + ca;
    const ulong b = ulong(k) * stride + cb;
    const float mut_a = mub2d[ca] + mup[ca];
    const float mut_b = mub2d[cb] + mup[cb];
    const float muf = 0.5f * (mut_a + mut_b);
    const float dmu = 0.5f * (mup[ca] + mup[cb]);
    const float layer_mass = c1h[k] * muf + c2h[k];

    const float dph_hi = php[a + stride] - php[b + stride];
    const float dph_lo = php[a] - php[b];
    float bracket = dph_hi + dph_lo;
    const float alt_sum = alt[a] + alt[b];
    const float pp_a = pressure_perturbation(p, pb, k, ca, stride, base3d);
    const float pp_b = pressure_perturbation(p, pb, k, cb, stride, base3d);
    const float dpp = pp_a - pp_b;
    bracket = bracket + alt_sum * dpp;
    const float al_sum = al[a] + al[b];
    const float dpb = base_value(pb, k, ca, stride, base3d)
                    - base_value(pb, k, cb, stride, base3d);
    bracket = bracket + al_sum * dpb;
    const float left = (half_rd * layer_mass) * bracket;

    const float dphp = php_half_value(php, phb, k, ca, stride, base3d)
                     - php_half_value(php, phb, k, cb, stride, base3d);
    float dp_a_hi, dp_a_lo, dp_b_hi, dp_b_lo;
    dpn_pair_value(p, pb, k, ca, stride, nz, base3d, fnm, fnp,
                   cf1, cf2, cf3, top_lid, cfn, cfn1, pp_a,
                   dp_a_hi, dp_a_lo);
    dpn_pair_value(p, pb, k, cb, stride, nz, base3d, fnm, fnp,
                   cf1, cf2, cf3, top_lid, cfn, cfn1, pp_b,
                   dp_b_hi, dp_b_lo);
    const float dpf_hi = 0.5f * (dp_a_hi + dp_b_hi);
    const float dpf_lo = 0.5f * (dp_a_lo + dp_b_lo);
    float vertical = rdnw[k] * (dpf_hi - dpf_lo);
    vertical = vertical - c1h[k] * dmu;
    const float right = (rd * dphp) * vertical;
    return left + right;
}

kernel void slow_pgf(
    constant SlowPgfResources &r [[buffer(0)]],
    constant SlowPgfParams &q [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint nz = uint(q.nz), ny = uint(q.ny), nx = uint(q.nx);
    const uint width = nx + 1;
    const uint height = ny + 1;
    const uint count = nz * height * width;
    if (flat >= count) return;
    if (nz < 3) return;
    const uint k = flat / (height * width);
    const uint rem = flat - k * height * width;
    const uint j = rem / width;
    const uint i = rem - j * width;
    const ulong stride = ulong(ny) * ulong(nx);
    const float rdx = q.rdx;
    const float rdy = q.rdy;
    if (j < ny && ((!q.boundary_x && i <= nx)
                   || (q.boundary_x && i > 0 && i < nx))) {
        const uint ia = q.boundary_x ? i : (i == nx ? 0 : i);
        const uint ib = q.boundary_x ? i - 1 : (i == 0 ? nx - 1 : i - 1);
        const ulong ca = ulong(j) * nx + ia;
        const ulong cb = ulong(j) * nx + ib;
        float term = pgf_face(ca, cb, k, stride, nz, rdx, q.half_rdx,
            r.p, r.pb, r.al, r.alt, r.php, r.phb, r.mup, r.mub2d,
            r.c1h, r.c2h, r.rdnw, r.fnm, r.fnp, q.cf1, q.cf2, q.cf3,
            bool(q.base3d), bool(q.top_lid), q.cfn, q.cfn1);
        const ulong out = (ulong(k) * ny + j) * width + i;
        if (q.use_cq) term = r.cqu[out] * term;
        r.ru_t[out] = r.ru_t[out] - term;
    }
    if (i < nx && ((!q.boundary_y && j <= ny)
                   || (q.boundary_y && j > 0 && j < ny))) {
        const uint ja = q.boundary_y ? j : (j == ny ? 0 : j);
        const uint jb = q.boundary_y ? j - 1 : (j == 0 ? ny - 1 : j - 1);
        const ulong ca = ulong(ja) * nx + i;
        const ulong cb = ulong(jb) * nx + i;
        float term = pgf_face(ca, cb, k, stride, nz, rdy, q.half_rdy,
            r.p, r.pb, r.al, r.alt, r.php, r.phb, r.mup, r.mub2d,
            r.c1h, r.c2h, r.rdnw, r.fnm, r.fnp, q.cf1, q.cf2, q.cf3,
            bool(q.base3d), bool(q.top_lid), q.cfn, q.cfn1);
        const ulong out = (ulong(k) * height + j) * nx + i;
        if (q.use_cq) term = r.cqv[out] * term;
        r.rv_t[out] = r.rv_t[out] - term;
    }
}
