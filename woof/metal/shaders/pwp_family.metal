// WRF 3DPWP ocean (woof/ocean/mixed_layer/pwp.cu), transcribed statement by
// statement. The CUDA unit keeps subnormals and IEEE division and square
// root, so binary32 runs on soft32 bit ops; expf, sinf and cosf are the
// binary64 glibc/fdlibm results rounded once. Columns use (level, y, x).
#include <metal_stdlib>
#include "pwp_geometry_bindings.metalh"
#include "pwp_transport_bindings.metalh"
#include "pwp_column_bindings.metalh"
#include "pwp_fallback_bindings.metalh"
#include "pwp_diagnose_bindings.metalh"
#include "pwp_mld_profiles_bindings.metalh"
#include "soft_fdlibm64_trig.metalh"
using namespace metal;

struct P {
    float v;
    P() = default;
    constexpr P(float x) : v(x) {}
};
static inline uint pb(P a) { return as_type<uint>(a.v); }
static inline P pm(uint b) { return P(as_type<float>(b)); }
#define PWP_ARITH(OP, FN) \
static inline P operator OP(P a, P b) { return pm(FN(pb(a), pb(b))); } \
static inline P operator OP(P a, float b) { return a OP P(b); } \
static inline P operator OP(float a, P b) { return P(a) OP b; }
PWP_ARITH(+, soft32_add)
PWP_ARITH(-, soft32_sub)
PWP_ARITH(*, soft32_mul)
PWP_ARITH(/, soft32_div)
#define PWP_CMP(OP, FN) \
static inline bool operator OP(P a, P b) { return FN(pb(a), pb(b)); } \
static inline bool operator OP(P a, float b) { return a OP P(b); } \
static inline bool operator OP(float a, P b) { return P(a) OP b; }
PWP_CMP(<, soft32_less)
PWP_CMP(>, soft32_greater)
PWP_CMP(<=, soft32_less_equal)
PWP_CMP(>=, soft32_greater_equal)
static inline bool operator==(P a, P b) { return soft32_equal(pb(a), pb(b)); }
static inline P operator-(P a) { return pm(pb(a) ^ 0x80000000u); }
static inline thread P &operator+=(thread P &a, P b) { a = a + b; return a; }
static inline thread P &operator-=(thread P &a, P b) { a = a - b; return a; }
static inline P fmaxf(P a, P b) { return P(aa_fmaxf(a.v, b.v)); }
static inline P fminf(P a, P b) { return P(aa_fminf(a.v, b.v)); }
static inline P fabsf(P a) { return pm(pb(a) & 0x7fffffffu); }
static inline P sqrtf(P a) { return pm(soft32_sqrt(pb(a))); }
static inline bool p_isfinite(P a) { return soft32_is_finite(pb(a)); }
static inline P expf(P a) {
    const Soft8 x = Soft8(a.v);
    if ((x.bits & 0x7fffffffffffffffUL) > 0x7ff0000000000000UL) return a;
    return P(float(Soft8::fromBits(sgp_exp_inline(x.bits, 0ul, 0u))));
}
static inline P sinf(P a) { return P(float(fd64_sin(Soft8(a.v)))); }
static inline P cosf(P a) { return P(float(fd64_cos(Soft8(a.v)))); }

// One column of a (level, y, x) field.
struct Col {
    device float *p; uint n, i;
    P operator[](int k) const { return P(p[ulong(k) * n + i]); }
    void set(int k, P x) const { p[ulong(k) * n + i] = x.v; }
};
struct CCol {
    device const float *p; uint n, i;
    P operator[](int k) const { return P(p[ulong(k) * n + i]); }
};
static inline P lv(device const float *a, int q) { return P(a[q]); }

static inline P sigma(P tt, P s) {
    P t = tt - 273.f, sg = ((6.76786136e-6f * s - 4.8249614e-4f) * s + 0.814876577f) * s - 0.0934458632f;
    return ((((-1.43803061e-7f * t - 1.98248399e-3f) * t - 0.545939111f) * t + 4.53168426f) * t) / (t + 67.26f)
        + ((((1.667e-8f * t - 8.164e-7f) * t + 1.803e-5f) * t) * sg + ((-1.0843e-6f * t + 9.8185e-5f) * t - 4.7867e-3f) * t + 1.f) * sg;
}
static inline void mix(Col a, CCol z, int m) {
    P sum = 0.f, depth = 0.f;
    for (int k = 0; k < m; k++) { P dz = z[k] - (k ? z[k - 1] : P(0.f)); sum += a[k] * dz; depth += dz; }
    for (int k = 0; k < m; k++) a.set(k, sum / depth);
}
static inline int mld(Col r, int L) {
    for (int k = 0; k < L - 1; k++) if (fabsf(r[k + 1] - r[k]) > 1.e-5f) return k + 1;
    return L;
}
static inline P depth_mld(CCol z, int m) { return z[m - 1] + (m > 1 ? (z[m - 1] - z[m - 2]) * 0.5f : P(0.f)); }
static inline void rotate(Col u, Col v, P sa, P ca, int L) {
    for (int k = 0; k < L; k++) { P a = u[k], b = v[k]; u.set(k, a * ca + b * sa); v.set(k, b * ca - a * sa); }
}
static inline void stir(Col a, CCol z, int k, P ff) {
    P dz = z[k] - (k ? z[k - 1] : P(0.f)), dn = z[k + 1] - z[k], da = (a[k + 1] - a[k]) * ff * 0.5f;
    P b = 2.f / (1.f + dz / dn); a.set(k + 1, a[k + 1] - da * b * dz / dn); a.set(k, a[k] + da * b);
}
static inline P richardson(Col r, CCol z, Col u, Col v, int k) {
    P dz = (z[k + 1] - (k ? z[k - 1] : P(0.f))) * 0.5f, du = u[k + 1] - u[k], dv = v[k + 1] - v[k];
    return 9.8f * dz * fmaxf(r[k + 1] - r[k], P(1.e-3f)) / (1000.f * fmaxf(du * du + dv * dv, P(1.e-6f)));
}

kernel void pwp_geometry(constant PwpGeometryResources &r [[buffer(0)]],
                         constant PwpGeometryParams &p [[buffer(1)]],
                         uint gi [[thread_position_in_grid]]) {
    const int nx = p.nx, ny = p.ny, n = nx * ny, L = p.levels, i = int(gi);
    if (i >= n) return;
    const P rdx = P(p.rdx), rdy = P(p.rdy);
    int x = i % nx, y = i / nx, cx = min(max(x, 1), nx - 2), cy = min(max(y, 1), ny - 2), c = cy * nx + cx;
    device const float *land = r.land;
    bool interior = lv(land, i) > 1.5f && lv(land, c) > 1.5f && lv(land, c + 1) > 1.5f && lv(land, c - 1) > 1.5f
                 && lv(land, c + nx) > 1.5f && lv(land, c - nx) > 1.5f;
    P div = 0.f, press = 0.f;
    for (int k = 0; k < L; k++) {
        int q = k * n + c; P dz = lv(r.z, q) - (k ? lv(r.z, q - n) : P(0.f));
        if (interior) div += dz * ((lv(r.u, q + 1) - lv(r.u, q - 1)) * rdx * lv(r.mx, c)
                                   + (lv(r.v, q + nx) - lv(r.v, q - nx)) * rdy * lv(r.my, c)) * 0.5f;
        r.w[k * n + i] = div.v;
    }
    for (int k = L - 1; k >= 0; k--) {
        int q = k * n + i; P dz = lv(r.z, q) - (k ? lv(r.z, q - n) : P(0.f));
        P tr = lv(r.tr, q), sr = lv(r.sr, q);
        P alpha = sigma(tr + 0.5f, sr) - sigma(tr - 0.5f, sr);
        P beta = sigma(tr, sr + 0.5f) - sigma(tr, sr - 0.5f);
        if (lv(land, i) > 1.5f) press -= 9.8f * dz * (alpha * (lv(r.t, q) - tr) + beta * (lv(r.s, q) - sr));
        r.p[q] = press.v;
    }
}

static inline P transport(device const float *a, device const float *z, device const float *w,
                          device const float *land, device const float *mx, device const float *my,
                          P u, P v, int i, int k, int nx, int ny, P rdx, P rdy, int L) {
    int n = nx * ny, q = k * n + i, x = i % nx, y = i / nx; P tendency = 0.f;
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
        if (x == 0 && lv(land, i + 1) > 1.5f) tendency -= 1.5f * rdx * lv(mx, i) * (lv(a, q) - lv(a, q + 1));
        if (x == nx - 1 && lv(land, i - 1) > 1.5f) tendency -= 1.5f * rdx * lv(mx, i) * (lv(a, q) - lv(a, q - 1));
        if (y == 0 && lv(land, i + nx) > 1.5f) tendency -= 1.5f * rdy * lv(my, i) * (lv(a, q) - lv(a, q + nx));
        if (y == ny - 1 && lv(land, i - nx) > 1.5f) tendency -= 1.5f * rdy * lv(my, i) * (lv(a, q) - lv(a, q - nx));
        return tendency;
    }
    int ix = u >= 0.f ? -1 : 1, iy = v >= 0.f ? -nx : nx;
    if (lv(land, i + ix) > 1.5f) tendency -= fabsf(u) * (lv(a, q) - lv(a, q + ix)) * rdx * lv(mx, i);
    if (lv(land, i + iy) > 1.5f) tendency -= fabsf(v) * (lv(a, q) - lv(a, q + iy)) * rdy * lv(my, i);
    if (k > 0) {
        int km = q - n, kp = k == L - 1 ? q : q + n; P wv = 0.5f * (lv(w, q - n) + lv(w, q));
        if (k < L - 1 || wv < 0.f) tendency += wv * (lv(a, km) - lv(a, kp)) / (lv(z, km) - lv(z, kp));
    }
    return tendency;
}

kernel void pwp_transport(constant PwpTransportResources &r [[buffer(0)]],
                          constant PwpTransportParams &p [[buffer(1)]],
                          uint gq [[thread_position_in_grid]]) {
    const int nx = p.nx, ny = p.ny, n = nx * ny, L = p.levels, q = int(gq);
    if (q >= n * L) return;
    const P rdx = P(p.rdx), rdy = P(p.rdy);
    int i = q % n, k = q / n, x = i % nx, y = i / nx;
    if (p.first) for (int a = 0; a < 4; a++) r.tend[a * n * L + q] = 0.f;
    if (lv(r.land, i) <= 1.5f) return;
    device const float *a[4] = {r.t, r.s, r.u, r.v};
    for (int j = 0; j < 4; j++)
        r.tend[j * n * L + q] = (P(r.tend[j * n * L + q])
            + transport(a[j], r.z, r.w, r.land, r.mx, r.my, lv(r.u, q), lv(r.v, q), i, k, nx, ny, rdx, rdy, L)).v;
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
        if (lv(r.land, i + 1) > 1.5f && lv(r.land, i - 1) > 1.5f)
            r.tend[2 * n * L + q] = (P(r.tend[2 * n * L + q]) - (lv(r.p, q + 1) - lv(r.p, q - 1)) * rdx * lv(r.mx, i) / 2046.f).v;
        if (lv(r.land, i + nx) > 1.5f && lv(r.land, i - nx) > 1.5f)
            r.tend[3 * n * L + q] = (P(r.tend[3 * n * L + q]) - (lv(r.p, q + nx) - lv(r.p, q - nx)) * rdy * lv(r.my, i) / 2046.f).v;
    }
}

// The CUDA column keeps t, s, u, v, r and ri in thread arrays; here they are
// the output profiles, the density field and the vertical-velocity scratch,
// which this kernel fully rewrites before any later reader.
kernel void pwp_column(constant PwpColumnResources &r [[buffer(0)]],
                       constant PwpColumnParams &p [[buffer(1)]],
                       uint gi [[thread_position_in_grid]]) {
    const int n = p.n, L = p.levels, i = int(gi);
    if (i >= n || lv(r.land, i) <= 1.5f) return;
    const uint un = uint(n), ui = uint(i);
    const P dt = P(p.dt), stage_dt = P(p.stage_dt), stbolt = P(p.stbolt);
    Col t{r.tout, un, ui}, s{r.sout, un, ui}, u{r.uout, un, ui}, v{r.vout, un, ui}, rr{r.density, un, ui}, ri{r.ri, un, ui};
    CCol z{r.zall, un, ui};
    for (int k = 0; k < L; k++) {
        int q = k * n + i;
        t.set(k, lv(r.base, q)); s.set(k, lv(r.base, n * L + q));
        u.set(k, lv(r.base, 2 * n * L + q)); v.set(k, lv(r.base, 3 * n * L + q));
    }
    P tr = lv(r.trall, i), sr = lv(r.srall, i), dr = sigma(tr, sr) + 1000.f;
    P alpha = sigma(tr + 0.5f, sr) - sigma(tr - 0.5f, sr), beta = sigma(tr, sr + 0.5f) - sigma(tr, sr - 0.5f);
    const P sw = lv(r.sw, i), lw = lv(r.lw, i);
    const bool ice_owned = p.coupled_ice && lv(r.ice_mask, i) > 0.f;
    for (int k = 0; k < L && !ice_owned; k++) {
        P z1 = k ? z[k - 1] : P(0.f), dz = z[k] - z1;
        P heat = (z[k] / 0.6f < 70.f ? sw * (expf(-z1 / 0.6f) - expf(-z[k] / 0.6f)) : P(0.f))
               + (z[k] / 20.f < 70.f ? lw * (expf(-z1 / 20.f) - expf(-z[k] / 20.f)) : P(0.f));
        if (k == 0) heat -= lv(r.lh, i) + lv(r.hfx, i) + stbolt * t[0] * t[0] * t[0] * t[0];
        t.set(k, t[k] + dt * heat / (dz * 1024.f * 4183.3f));
    }
    s.set(0, s[0] + s[0] * lv(r.qfx, i) * dt / (1024.f * z[0]));
    for (int k = 0; k < L; k++) rr.set(k, dr + alpha * (t[k] - tr) + beta * (s[k] - sr));
    int ml = 1;
    for (int k = 1; k < L; k++) { if (rr[k] > rr[k - 1]) break; if (rr[k] < rr[k - 1]) { ml = k + 1; mix(rr, z, ml); } }
    if (ml > 1) { mix(t, z, ml); mix(s, z, ml); mix(u, z, ml); mix(v, z, ml); mix(rr, z, ml); }
    P sa = sinf(lv(r.f, i) * dt * 0.5f), ca = cosf(lv(r.f, i) * dt * 0.5f); rotate(u, v, sa, ca, L);
    const P ua = lv(r.ua, i), va = lv(r.va, i), ust = lv(r.ust, i);
    ml = mld(rr, L); P dml = depth_mld(z, ml), speed = fmaxf(sqrtf(ua * ua + va * va), P(0.1f));
    P du = ust * ust * ua / speed * dt / (1024.f * dml), dv = ust * ust * va / speed * dt / (1024.f * dml);
    for (int k = 0; k < ml; k++) { u.set(k, u[k] + du); v.set(k, v[k] + dv); }
    for (int k = ml - 1; k < L - 1; k++) {
        P dz = depth_mld(z, k + 1), delr = (rr[k + 1] - rr[k]) / 1024.f, a = u[k + 1] - u[k], b = v[k + 1] - v[k];
        if (9.8f * delr * dz / (a * a + b * b + 1.e-8f) > 0.65f) break;
        mix(t, z, k + 2); mix(s, z, k + 2); mix(u, z, k + 2); mix(v, z, k + 2); mix(rr, z, k + 2);
    }
    for (int k = 0; k < L - 1; k++) ri.set(k, richardson(rr, z, u, v, k));
    int iterations = 0;
    device atomic_int *status = (device atomic_int *)r.status;
    while (true) {
        int ks = 0; for (int k = 1; k < L - 1; k++) if (ri[k] < ri[ks]) ks = k;
        if (ri[ks] >= 0.25f) break;
        if (++iterations > 100000) { atomic_fetch_or_explicit(status, 1, memory_order_relaxed); break; }
        P ff = 1.f - ri[ks] / 0.3f; stir(t, z, ks, ff); stir(s, z, ks, ff); stir(u, z, ks, ff); stir(v, z, ks, ff); stir(rr, z, ks, ff);
        for (int k = max(0, ks - 2); k <= min(L - 2, ks + 2); k++) ri.set(k, richardson(rr, z, u, v, k));
    }
    rotate(u, v, sa, ca, L); r.mlout[i] = depth_mld(z, mld(rr, L)).v;
    Col out[4] = {t, s, u, v};
    for (int j = 0; j < 4; j++) for (int k = 0; k < L; k++) {
        int q = k * n + i, off = j * n * L + q;
        P ak = out[j][k];
        P tend = P(r.tend[off]) + (ak - lv(r.base, off)) / dt;
        r.tend[off] = tend.v;
        P value = lv(r.base, off) + stage_dt * tend; out[j].set(k, value);
        if (!p_isfinite(value)) atomic_fetch_or_explicit(status, 2, memory_order_relaxed);
    }
    for (int k = 0; k < L; k++) rr.set(k, dr + alpha * (t[k] - tr) + beta * (s[k] - sr));
    r.tsk[i] = r.tout[i];
}

kernel void pwp_fallback(constant PwpFallbackResources &r [[buffer(0)]],
                         constant PwpFallbackParams &p [[buffer(1)]],
                         uint gq [[thread_position_in_grid]]) {
    const int n = p.n, q = int(gq);
    if (q >= n * p.levels) return;
    int i = q % n, k = q / n; P depth = 5.f * P(float(k + 1));
    r.z[q] = depth.v; r.t[q] = (lv(r.tsk, i) - P(p.gamma) * fmaxf(depth - P(p.hml), P(0.f))).v;
    r.s[q] = 35.f; r.u[q] = 0.f; r.v[q] = 0.f;
}

kernel void pwp_diagnose(constant PwpDiagnoseResources &r [[buffer(0)]],
                         constant PwpDiagnoseParams &p [[buffer(1)]],
                         uint gi [[thread_position_in_grid]]) {
    const int n = p.n, L = p.levels, i = int(gi);
    if (i >= n || lv(r.land, i) <= 1.5f) return;
    const uint un = uint(n), ui = uint(i);
    Col rout{r.rout, un, ui}; CCol depth{r.z, un, ui};
    P tr = lv(r.tr, i), sr = lv(r.sr, i), dr = sigma(tr, sr) + 1000.f;
    P alpha = sigma(tr + 0.5f, sr) - sigma(tr - 0.5f, sr);
    P beta = sigma(tr, sr + 0.5f) - sigma(tr, sr - 0.5f);
    for (int k = 0; k < L; k++) { int q = k * n + i; rout.set(k, dr + alpha * (lv(r.t, q) - tr) + beta * (lv(r.s, q) - sr)); }
    r.ml[i] = depth_mld(depth, mld(rout, L)).v;
}

kernel void pwp_mld_profiles(constant PwpMldProfilesResources &r [[buffer(0)]],
                             constant PwpMldProfilesParams &p [[buffer(1)]],
                             uint gq [[thread_position_in_grid]]) {
    const int n = p.n, q = int(gq);
    if (q >= n * p.levels) return;
    int i = q % n, k = q / n, j = r.index[i]; P depth = 5.f * P(float(k + 1)); r.z[q] = depth.v;
    const P sst = lv(r.sst, i);
    r.t[q] = (j < 0 ? P(0.f) : fmaxf(sst - P(p.gamma) * fmaxf(depth - lv(r.mld, j), P(0.f)), fminf(sst, P(271.25f)))).v;
    r.s[q] = (j < 0 ? 0.f : 35.f); r.u[q] = 0.f; r.v[q] = 0.f;
}
