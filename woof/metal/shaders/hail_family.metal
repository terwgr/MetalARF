// WOOF hail diagnostics, counterpart of woof/core/kernels/hail.cu.
#include <metal_stdlib>
#include "hail_family_bindings.metalh"
#include "hailcast_support.metalh"
#include "hailcast_wrf_fragment.metalh"
#include "hailcast_wrf39_fragment.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void hail_prepare(
    constant HailPrepareResources &r [[buffer(0)]],
    constant HailPrepareParams &q [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int n = q.nx * q.ny, nz = q.nz, base_profile = q.base_profile;
    const int c = int(gid);
    if (c >= n) return;
    float strong = 0;
    for (int k = 0; k < nz; k++) {
        int i = k * n + c;
        r.t[i] = r.theta[i] * lake_pow(r.p[i] / 100000.f, 287.f / 1004.f);
        r.rho[i] = (1.f + r.qv[i]) / r.alt[i];
        float za = (r.phb[base_profile ? k : i] + r.php[i]) / 9.81f;
        float zb = (r.phb[base_profile ? k + 1 : i + n] + r.php[i + n]) / 9.81f;
        r.z[i] = .5f * (za + zb);
        if (r.w[i] >= 10.f) strong = 1;
    }
    if (r.w[nz * n + c] >= 10.f) strong = 1;
    r.mask[c] = strong;
}

kernel void hail_duration(
    constant HailDurationResources &r [[buffer(0)]],
    constant HailDurationParams &q [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int nx = q.nx, ny = q.ny, c = int(gid), n = nx * ny;
    if (c >= n) return;
    int x = c % nx, y = c / nx;
    r.dur[c] = 0;
    if ((!q.periodic_x && (x == 0 || x == nx - 1)) || (!q.periodic_y && (y == 0 || y == ny - 1))) return;
    float seen = r.mask[c], longest = 0;
    for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) {
        int xx = (x + dx + nx) % nx, yy = (y + dy + ny) % ny, j = yy * nx + xx;
        seen = fmax(seen, r.oldmask[j]); longest = fmax(longest, r.olddur[j]);
    }
    if (seen == 1.f) r.dur[c] = longest + q.dt;
}

kernel void hailcast_columns(
    constant HailcastColumnsResources &r [[buffer(0)]],
    constant HailcastColumnsParams &q [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int nx = q.nx, ny = q.ny, nz = q.nz, c = int(gid), n = nx * ny;
    if (c >= n || r.dur[c] <= 900.f) return;
    HailStorage<float, 256> ta, za, pa, ra, va, ia, ca, raina, sa, ga, wa;
    for (int k = 0; k < nz; k++) {
        const int i = k * n + c;
        ta.storage[k] = r.t[i]; za.storage[k] = r.z[i]; pa.storage[k] = r.p[i];
        ra.storage[k] = r.rho[i]; va.storage[k] = r.qv[i]; ia.storage[k] = r.qi[i];
        ca.storage[k] = r.qc[i]; raina.storage[k] = r.qr[i]; sa.storage[k] = r.qs[i];
        ga.storage[k] = r.qg[i]; wa.storage[k] = r.w[i];
    }
    float h = r.ht[c], duration = r.dur[c], d[5] = {0, 0, 0, 0, 0};
    HailColumn<256> column;
    column.hailstone_driver(ta, za, h, pa, ra, va, ia, ca, raina, sa, ga, wa,
        duration, nz, d[0], d[1], d[2], d[3], d[4]);
    device float* embryos[5] = {r.e1, r.e2, r.e3, r.e4, r.e5};
    // Invalid integration must remain visible to history/verification. Never
    // turn a failed stone trajectory into a plausible no-hail observation.
    if (!isfinite(r.runmax[c])) return;
    const float nan = as_type<float>(0x7fc00000u);
    for (int j = 0; j < 5; j++) if (!isfinite(d[j])) {
        r.maximum[c] = r.mean[c] = r.sd[c] = r.runmax[c] = nan;
        embryos[j][c] = nan; return;
    }
    float m = 0, mx = 0, s = 0;
    for (int j = 0; j < 5; j++) {
        float value = fmax(d[j], 0.f);
        embryos[j][c] = fmax(embryos[j][c], value);
        m += embryos[j][c] * .2f; mx = fmax(mx, embryos[j][c]);
    }
    for (int j = 0; j < 5; j++) { float e = embryos[j][c] - m; s += e * e; }
    r.maximum[c] = mx; r.mean[c] = m; r.sd[c] = hail_sqrtf(s * .25f);
    r.runmax[c] = fmax(r.runmax[c], mx);
}

kernel void hailcast_columns_wrf39(
    constant HailcastColumnsWrf39Resources &r [[buffer(0)]],
    constant HailcastColumnsWrf39Params &q [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int nx = q.nx, ny = q.ny, nz = q.nz, c = int(gid), n = nx * ny;
    if (c >= n || r.dur[c] <= 900.f) return;
    HailStorage<float, 256> ta, za, pa, ra, va, ia, ca, raina, sa, ga, wa;
    for (int k = 0; k < nz; k++) {
        const int i = k * n + c;
        ta.storage[k] = r.t[i]; za.storage[k] = r.z[i]; pa.storage[k] = r.p[i];
        ra.storage[k] = r.rho[i]; va.storage[k] = r.qv[i]; ia.storage[k] = r.qi[i];
        ca.storage[k] = r.qc[i]; raina.storage[k] = r.qr[i]; sa.storage[k] = r.qs[i];
        ga.storage[k] = r.qg[i]; wa.storage[k] = r.w[i];
    }
    float h = r.ht[c], duration = r.dur[c], d[5] = {0, 0, 0, 0, 0};
    HailColumnWrf39<256> column;
    column.hailstone_driver(ta, za, h, pa, ra, va, ia, ca, raina, sa, ga, wa,
        duration, nz, d[0], d[1], d[2], d[3], d[4]);
    device float* embryos[5] = {r.e1, r.e2, r.e3, r.e4, r.e5};
    // Invalid integration must remain visible to history/verification. Never
    // turn a failed stone trajectory into a plausible no-hail observation.
    if (!isfinite(r.runmax[c])) return;
    const float nan = as_type<float>(0x7fc00000u);
    for (int j = 0; j < 5; j++) if (!isfinite(d[j])) {
        r.maximum[c] = r.mean[c] = r.sd[c] = r.runmax[c] = nan;
        embryos[j][c] = nan; return;
    }
    float m = 0, mx = 0, s = 0;
    for (int j = 0; j < 5; j++) {
        float value = fmax(d[j], 0.f);
        embryos[j][c] = fmax(embryos[j][c], value);
        m += embryos[j][c] * .2f; mx = fmax(mx, embryos[j][c]);
    }
    for (int j = 0; j < 5; j++) { float e = embryos[j][c] - m; s += e * e; }
    r.maximum[c] = mx; r.mean[c] = m; r.sd[c] = hail_sqrtf(s * .25f);
    r.runmax[c] = fmax(r.runmax[c], mx);
}

// N*Q(alpha+1,lambda*D) = 0.0005 m^-3, WRF HAILMAXD criterion. Only the
// closed forms (exponential graupel, alpha=1 hail) are ported.
static HailR8 hail_tail(HailR8 a, HailR8 x) {
    if (x <= 0) return HailR8(1);
    if (a == 1) return hail_exp(-x);
    return hail_exp(-x) * (1 + x);
}
static float hail_psd_size(HailR8 mass, HailR8 number, HailR8 density, HailR8 alpha) {
    if (!(mass > HAIL_D(0x3eb0c6f7a0b5ed8dul) && number > HAIL_D(0x3f40624dd2f1a9fcul)
          && density > 0 && alpha > -1)) return 0;
    HailR8 lambda = hail_cbrt(HAIL_D(0x3fe0c152382d7365ul) * density * (alpha + 1)
        * (alpha + 2) * (alpha + 3) * number / mass);
    HailR8 target = HAIL_D(0x3f40624dd2f1a9fcul) / number, lo = 0,
           hi = hail_fmax(HAIL_D(0x4050000000000000ul), alpha * 4 + HAIL_D(0x4050000000000000ul));
    if (alpha == 0) return float(-hail_log(target) / lambda);
    for (int j = 0; j < 40; j++) {
        HailR8 mid = HAIL_D(0x3fe0000000000000ul) * (lo + hi);
        if (hail_tail(alpha + 1, mid) > target) lo = mid; else hi = mid;
    }
    return float(HAIL_D(0x3fe0000000000000ul) * (lo + hi) / lambda);
}

kernel void hail_psd(
    constant HailPsdResources &r [[buffer(0)]],
    constant HailPsdParams &q [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int nz = q.nz, c = int(gid), n = q.nx * q.ny;
    if (c >= n) return;
    const HailR8 pi = HAIL_D(0x400921fb54442d18ul), quarter = HAIL_D(0x3fd0000000000000ul);
    float mx = 0, low = 0;
    for (int k = 0; k < nz; k++) {
        int i = k * n + c;
        HailR8 m = fmax(r.qg[i], 0.f) * r.rho[i];
        HailR8 density = q.volume_g && r.vg[i] > 0 ? r.qg[i] / r.vg[i] : q.density_g;
        HailR8 number = q.number_g ? fmax(r.ng[i], 0.f) * r.rho[i] : 0.f;
        if (!q.number_g && m > HAIL_D(0x3eb0c6f7a0b5ed8dul)) {
            HailR8 n0 = q.intercept;
            if (q.thompson) {
                float lg = lake_log10(fmax(1.e-9f, float(m)));
                float exponent = fmax(2.f, fmin(6.f, 3.f + 2.f / 7.f * (lg + 8.f)));
                n0 = lake_pow(10.f, exponent);
            }
            HailR8 lambda = hail_pow(n0 * pi * density / m, quarter);
            number = n0 / lambda;
        }
        float size = hail_psd_size(m, number, density, q.alpha_g);
        if (q.number_h || q.intercept_h > 0) {
            HailR8 mh = fmax(r.qh[i], 0.f) * r.rho[i];
            HailR8 dh = q.volume_h && r.vh[i] > 0 ? r.qh[i] / r.vh[i] : q.density_h;
            HailR8 numberh = q.number_h ? fmax(r.nh[i], 0.f) * r.rho[i] : 0.f;
            if (!q.number_h && mh > HAIL_D(0x3eb0c6f7a0b5ed8dul)) {
                HailR8 lambda = hail_pow(q.intercept_h * pi * dh / mh, quarter);
                numberh = q.intercept_h / lambda;
            }
            size = fmax(size, hail_psd_size(mh, numberh, dh, q.alpha_h));
        }
        mx = fmax(mx, size);
        if (k == 0) low = size;
    }
    r.lowest[c] = fmax(r.lowest[c], low);
    r.column[c] = fmax(r.column[c], mx);
    r.runlowest[c] = fmax(r.runlowest[c], low); r.runcolumn[c] = fmax(r.runcolumn[c], mx);
}
