#include <metal_stdlib>
#include "ieva_solve_family_bindings.metalh"
#include "soft_binary64.metalh"

using namespace metal;
#pragma clang fp contract(off)

// CUDA IEVA uses FP32 coefficients and FP64 Thomas work arrays. Metal has no
// native FP64, so keep the Thomas values as IEEE binary64 words and perform
// every solve operation with the shared integer-only soft64 implementation.
static inline ulong ieva_widen(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}
static inline float ieva_narrow(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}
static inline float ieva_fadd(float a, float b) { return a + b; }
static inline float ieva_fsub(float a, float b) { return a - b; }
static inline float ieva_fmul(float a, float b) { return a * b; }
static inline float ieva_fdiv(float a, float b) { return a / b; }
static inline float ieva_fmax(float a, float b) { return a > b ? a : b; }
static inline float ieva_fmin(float a, float b) { return a < b ? a : b; }

static inline void ieva_coefficients(float wiL, float wiR, float dt,
                                     float zero, thread ulong& at,
                                     thread ulong& ct, thread ulong& btmp) {
    at = ieva_widen(-ieva_fmul(dt, ieva_fmax(wiL, zero)));
    ct = ieva_widen(ieva_fmul(dt, ieva_fmin(wiR, zero)));
    btmp = ieva_widen(ieva_fmul(
        dt, ieva_fsub(ieva_fmax(wiR, zero), ieva_fmin(wiL, zero))));
}

static inline ulong ieva_rhs(float lead, float mass, ulong at, ulong btmp,
                             ulong ct, float xm, float x0, float xp) {
    ulong sum = soft64_add(soft64_mul(at, ieva_widen(xm)),
                           soft64_mul(btmp, ieva_widen(x0)));
    sum = soft64_add(sum, soft64_mul(ct, ieva_widen(xp)));
    return soft64_sub(ieva_widen(lead),
                      soft64_mul(ieva_widen(mass), sum));
}

static inline void ieva_forward(int k, int kstart, ulong a, ulong b,
                                ulong cprev, ulong rhs,
                                thread ulong& bet,
                                thread ulong* gamma,
                                thread ulong* values) {
    if (k == kstart) {
        bet = b;
        values[k] = soft64_div(rhs, bet);
    } else {
        gamma[k] = soft64_div(cprev, bet);
        bet = soft64_sub(b, soft64_mul(a, gamma[k]));
        values[k] = soft64_div(
            soft64_sub(rhs, soft64_mul(a, values[k - 1])), bet);
    }
}

static inline void ieva_backward(int kstart, int kend,
                                 thread ulong* gamma,
                                 thread ulong* values) {
    for (int k = kend - 1; k >= kstart; --k)
        values[k] = soft64_sub(
            values[k], soft64_mul(gamma[k + 1], values[k + 1]));
}

// Current supported Metal tier is IEVA_KMAX=65 (nz <= 64), matching the
// production HRRR configuration. Larger vertical tiers remain refused.
kernel void ieva_solve_s(constant IevaSolveSResources& r [[buffer(0)]],
                          constant IevaSolveSParams& p [[buffer(1)]],
                          uint c [[thread_position_in_grid]]) {
    const ulong st = ulong(p.ny) * ulong(p.nx);
    if (ulong(c) >= st) return;
    ulong gamma[65];
    ulong values[65];
    ulong bet = 0ul;
    ulong cprev = 0ul;
    const float mo = r.mut_old[c];
    const float mn = r.mut_new[c];
    for (int k = 0; k < p.nz; ++k) {
        const int km1 = k == 0 ? 0 : k - 1;
        const int kp1 = k == p.nz - 1 ? p.nz - 1 : k + 1;
        float sm, s0, sp;
        if (p.base_mode == 0) {
            sm = r.pert[ulong(km1) * st + ulong(c)];
            s0 = r.pert[ulong(k) * st + ulong(c)];
            sp = r.pert[ulong(kp1) * st + ulong(c)];
        } else {
            const ulong bm = p.base_mode == 1 ? ulong(km1) : ulong(km1) * st + ulong(c);
            const ulong b0 = p.base_mode == 1 ? ulong(k) : ulong(k) * st + ulong(c);
            const ulong bp = p.base_mode == 1 ? ulong(kp1) : ulong(kp1) * st + ulong(c);
            sm = ieva_fadd(ieva_fsub(r.base[bm], p.shift),
                           r.pert[ulong(km1) * st + ulong(c)]);
            s0 = ieva_fadd(ieva_fsub(r.base[b0], p.shift),
                           r.pert[ulong(k) * st + ulong(c)]);
            sp = ieva_fadd(ieva_fsub(r.base[bp], p.shift),
                           r.pert[ulong(kp1) * st + ulong(c)]);
        }
        const float mass_new = ieva_fadd(ieva_fmul(r.c1h[k], mn), r.c2h[k]);
        const float wiL = ieva_fdiv(ieva_fmul(r.rom[ulong(k) * st + c],
                                               r.rdnw[k]), mass_new);
        const float wiR = ieva_fdiv(ieva_fmul(r.rom[ulong(k + 1) * st + c],
                                               r.rdnw[k]), mass_new);
        ulong at, ct, btmp;
        ieva_coefficients(wiL, wiR, p.dt, p.zero, at, ct, btmp);
        const ulong bt = soft64_add(ieva_widen(p.one), btmp);
        const float mass_old = ieva_fadd(ieva_fmul(r.c1h[k], mo), r.c2h[k]);
        const float lead = ieva_fmul(p.dt, r.tend[ulong(k) * st + c]);
        const ulong rhs = ieva_rhs(lead, mass_old, at, btmp, ct, sm, s0, sp);
        ieva_forward(k, 0, at, bt, cprev, rhs, bet, gamma, values);
        cprev = ct;
    }
    ieva_backward(0, p.nz - 1, gamma, values);
    for (int k = 0; k < p.nz; ++k)
        r.tend[ulong(k) * st + c] = ieva_fdiv(ieva_narrow(values[k]), p.dt);
}

static inline float ieva_face_mass(device const float* mu, ulong row,
                                   int face, int n, int stride,
                                   int periodic, float half_val) {
    int a = face;
    int b = face - 1;
    if (periodic) {
        a = (a % n + n) % n;
        b = (b % n + n) % n;
    } else {
        if (a > n - 1) a = n - 1;
        if (b < 0) b = 0;
        if (face == 0) b = a;
        if (face == n) a = b;
    }
    return half_val * (mu[row + ulong(a * stride)]
                   + mu[row + ulong(b * stride)]);
}

kernel void ieva_solve_u(constant IevaSolveUResources& r [[buffer(0)]],
                         constant IevaSolveUParams& p [[buffer(1)]],
                         uint t [[thread_position_in_grid]]) {
    const int nfaces = p.f_hi - p.f_lo + 1;
    if (nfaces <= 0 || ulong(t) >= ulong(p.ny) * ulong(nfaces)) return;
    const int j = int(t) / nfaces;
    const int f = p.f_lo + int(t - uint(j * nfaces));
    const ulong nxu = ulong(p.nx + 1);
    const ulong st = ulong(p.ny) * ulong(p.nx);
    const ulong stu = ulong(p.ny) * nxu;
    const ulong row = ulong(j) * ulong(p.nx);
    const int ia = p.periodic ? f % p.nx : f;
    const int ib = p.periodic ? (f - 1 + p.nx) % p.nx : f - 1;
    const ulong fc = ulong(j) * nxu + ulong(f);
    float mo = ieva_face_mass(r.mut_old, row, f, p.nx, 1,
                              p.periodic, p.half_val);
    float mn = ieva_face_mass(r.mut_new, row, f, p.nx, 1,
                              p.periodic, p.half_val);
    if (p.legacy) mo = mn = r.legacy_face[fc];
    const float msf = p.has_msf ? r.msfu[fc] : p.one;
    ulong gamma[65];
    ulong values[65];
    ulong bet = 0ul;
    ulong cprev = 0ul;
    for (int k = 0; k < p.nz; ++k) {
        const int km1 = k == 0 ? 0 : k - 1;
        const int kp1 = k == p.nz - 1 ? p.nz - 1 : k + 1;
        const float mass_new = ieva_fadd(ieva_fmul(r.c1h[k], mn), r.c2h[k]);
        const float rl = p.half_val * (r.rom[ulong(k) * st + row + ulong(ib)]
                                  + r.rom[ulong(k) * st + row + ulong(ia)]);
        const float rr = p.half_val * (r.rom[ulong(k + 1) * st + row + ulong(ib)]
                                  + r.rom[ulong(k + 1) * st + row + ulong(ia)]);
        float wiL = rl * r.rdnw[k];
        float wiR = rr * r.rdnw[k];
        if (p.has_msf) { wiL *= msf; wiR *= msf; }
        wiL = ieva_fdiv(wiL, mass_new);
        wiR = ieva_fdiv(wiR, mass_new);
        ulong at, ct, btmp;
        ieva_coefficients(wiL, wiR, p.dt, p.zero, at, ct, btmp);
        const ulong bt = soft64_add(ieva_widen(p.one), btmp);
        const float mass_old = ieva_fadd(ieva_fmul(r.c1h[k], mo), r.c2h[k]);
        float lead = ieva_fmul(p.dt, r.tend[ulong(k) * stu + fc]);
        if (p.has_msf) lead = ieva_fmul(lead, msf);
        const ulong rhs = ieva_rhs(
            lead, mass_old, at, btmp, ct,
            r.u_old[ulong(km1) * stu + fc],
            r.u_old[ulong(k) * stu + fc],
            r.u_old[ulong(kp1) * stu + fc]);
        ieva_forward(k, 0, at, bt, cprev, rhs, bet, gamma, values);
        cprev = ct;
    }
    ieva_backward(0, p.nz - 1, gamma, values);
    for (int k = 0; k < p.nz; ++k) {
        float out = ieva_fdiv(ieva_narrow(values[k]), p.dt);
        if (p.has_msf) out = ieva_fdiv(out, msf);
        r.tend[ulong(k) * stu + fc] = out;
    }
}

kernel void ieva_solve_v(constant IevaSolveVResources& r [[buffer(0)]],
                         constant IevaSolveVParams& p [[buffer(1)]],
                         uint t [[thread_position_in_grid]]) {
    const int nfaces = p.f_hi - p.f_lo + 1;
    if (nfaces <= 0 || ulong(t) >= ulong(p.nx) * ulong(nfaces)) return;
    const int f = p.f_lo + int(t / uint(p.nx));
    const int i = int(t - uint((f - p.f_lo) * p.nx));
    const ulong st = ulong(p.ny) * ulong(p.nx);
    const ulong stv = ulong(p.ny + 1) * ulong(p.nx);
    const int ja = p.periodic ? f % p.ny : f;
    const int jb = p.periodic ? (f - 1 + p.ny) % p.ny : f - 1;
    const ulong fc = ulong(f) * ulong(p.nx) + ulong(i);
    float mo = ieva_face_mass(r.mut_old, ulong(i), f, p.ny, p.nx,
                              p.periodic, p.half_val);
    float mn = ieva_face_mass(r.mut_new, ulong(i), f, p.ny, p.nx,
                              p.periodic, p.half_val);
    if (p.legacy) mo = mn = r.legacy_face[fc];
    const float msf = p.has_msf ? r.msfv[fc] : p.one;
    ulong gamma[65];
    ulong values[65];
    ulong bet = 0ul;
    ulong cprev = 0ul;
    for (int k = 0; k < p.nz; ++k) {
        const int km1 = k == 0 ? 0 : k - 1;
        const int kp1 = k == p.nz - 1 ? p.nz - 1 : k + 1;
        const float mass_new = ieva_fadd(ieva_fmul(r.c1h[k], mn), r.c2h[k]);
        const float rl = p.half_val * (r.rom[ulong(k) * st + ulong(jb) * ulong(p.nx) + ulong(i)]
                                  + r.rom[ulong(k) * st + ulong(ja) * ulong(p.nx) + ulong(i)]);
        const float rr = p.half_val * (r.rom[ulong(k + 1) * st + ulong(jb) * ulong(p.nx) + ulong(i)]
                                  + r.rom[ulong(k + 1) * st + ulong(ja) * ulong(p.nx) + ulong(i)]);
        float wiL = rl * r.rdnw[k];
        float wiR = rr * r.rdnw[k];
        if (p.has_msf) { wiL *= msf; wiR *= msf; }
        wiL = ieva_fdiv(wiL, mass_new);
        wiR = ieva_fdiv(wiR, mass_new);
        ulong at, ct, btmp;
        ieva_coefficients(wiL, wiR, p.dt, p.zero, at, ct, btmp);
        const ulong bt = soft64_add(ieva_widen(p.one), btmp);
        const float mass_old = ieva_fadd(ieva_fmul(r.c1h[k], mo), r.c2h[k]);
        float lead = ieva_fmul(p.dt, r.tend[ulong(k) * stv + fc]);
        if (p.has_msf) lead = ieva_fmul(lead, msf);
        const ulong rhs = ieva_rhs(
            lead, mass_old, at, btmp, ct,
            r.v_old[ulong(km1) * stv + fc],
            r.v_old[ulong(k) * stv + fc],
            r.v_old[ulong(kp1) * stv + fc]);
        ieva_forward(k, 0, at, bt, cprev, rhs, bet, gamma, values);
        cprev = ct;
    }
    ieva_backward(0, p.nz - 1, gamma, values);
    for (int k = 0; k < p.nz; ++k) {
        float out = ieva_fdiv(ieva_narrow(values[k]), p.dt);
        if (p.has_msf) out = ieva_fdiv(out, msf);
        r.tend[ulong(k) * stv + fc] = out;
    }
}

kernel void ieva_solve_ph(constant IevaSolvePhResources& r [[buffer(0)]],
                          constant IevaSolvePhParams& p [[buffer(1)]],
                          uint c [[thread_position_in_grid]]) {
    const ulong st = ulong(p.ny) * ulong(p.nx);
    if (ulong(c) >= st || p.nz < 2) return;
    const float mt = r.mut[c];
    const float msf = p.has_msf ? r.msft[c] : p.one;
    ulong gamma[65];
    ulong values[65];
    ulong bet = 0ul;
    ulong cprev = 0ul;
    for (int k = 1; k < p.nz; ++k) {
        const float mass = ieva_fadd(ieva_fmul(r.c1f[k], mt), r.c2f[k]);
        float wi = ieva_fmul(
            ieva_fmul(p.half_val, r.rom[ulong(k) * st + ulong(c)]),
            ieva_fadd(r.rdnw[k - 1], r.rdnw[k]));
        if (p.has_msf) wi = ieva_fmul(wi, msf);
        wi = ieva_fdiv(wi, mass);
        const ulong at = ieva_widen(-ieva_fmul(p.dt, ieva_fmax(wi, p.zero)));
        const ulong ct = ieva_widen(ieva_fmul(p.dt, ieva_fmin(wi, p.zero)));
        const ulong btmp = soft64_sub(soft64_sub(0ul, at), ct);
        const ulong bt = soft64_add(ieva_widen(p.one), btmp);
        float lead = ieva_fmul(r.tend[ulong(k) * st + ulong(c)], p.dt);
        if (p.has_msf) lead = ieva_fmul(lead, msf);
        lead = ieva_fdiv(lead, mass);
        const ulong km = ulong(k - 1) * st + ulong(c);
        const ulong k0 = ulong(k) * st + ulong(c);
        const ulong kp = ulong(k + 1) * st + ulong(c);
        const ulong bm = p.base3d ? km : ulong(k - 1);
        const ulong b0 = p.base3d ? k0 : ulong(k);
        const ulong bp = p.base3d ? kp : ulong(k + 1);
        ulong rhs = ieva_widen(lead);
        rhs = soft64_sub(rhs, soft64_mul(at, ieva_widen(r.ph_old[km])));
        rhs = soft64_sub(rhs, soft64_mul(btmp, ieva_widen(r.ph_old[k0])));
        rhs = soft64_sub(rhs, soft64_mul(ct, ieva_widen(r.ph_old[kp])));
        rhs = soft64_sub(rhs, soft64_mul(at, ieva_widen(r.phb[bm])));
        rhs = soft64_sub(rhs, soft64_mul(btmp, ieva_widen(r.phb[b0])));
        rhs = soft64_sub(rhs, soft64_mul(ct, ieva_widen(r.phb[bp])));
        ieva_forward(k, 1, at, bt, cprev, rhs, bet, gamma, values);
        cprev = ct;
    }
    ieva_backward(1, p.nz - 1, gamma, values);
    for (int k = 1; k < p.nz; ++k) {
        const float mass = ieva_fadd(ieva_fmul(r.c1f[k], mt), r.c2f[k]);
        float out = ieva_fdiv(ieva_fmul(ieva_narrow(values[k]), mass), p.dt);
        if (p.has_msf) out = ieva_fdiv(out, msf);
        r.tend[ulong(k) * st + ulong(c)] = out;
    }
}

static inline float ieva_ht_at(device const float* ht, int j, int i,
                               int ny, int nx, int bx, int by) {
    if (bx) i = clamp(i, 0, nx - 1);
    else i = (i % nx + nx) % nx;
    if (by) j = clamp(j, 0, ny - 1);
    else j = (j % ny + ny) % ny;
    return ht[ulong(j) * ulong(nx) + ulong(i)];
}

static inline float ieva_uncoupled(float tend, float msf, int has_msf,
                                   float c1, float c2, float mu) {
    const float scaled = has_msf ? ieva_fmul(tend, msf) : tend;
    return ieva_fdiv(scaled, ieva_fadd(ieva_fmul(c1, mu), c2));
}

kernel void ieva_solve_w(constant IevaSolveWResources& r [[buffer(0)]],
                         constant IevaSolveWParams& p [[buffer(1)]],
                         uint c [[thread_position_in_grid]]) {
    const ulong st = ulong(p.ny) * ulong(p.nx);
    if (ulong(c) >= st || p.nz < 2) return;
    const int j = int(c) / p.nx;
    const int i = int(c) - j * p.nx;
    const ulong stu = ulong(p.ny) * ulong(p.nx + 1);
    const ulong stv = ulong(p.ny + 1) * ulong(p.nx);
    const float mt = r.mut[c];
    const float mo = r.mut_old[c];
    const float mn = r.mut_new[c];
    const float msf = p.has_msf ? r.msft[c] : p.one;
    const int kend = p.nz - 1;
    ulong gamma[65];
    ulong values[65];
    ulong bet = 0ul;
    ulong cprev = 0ul;
    for (int k = 1; k <= kend; ++k) {
        const float mass_new = ieva_fadd(ieva_fmul(r.c1f[k], mn), r.c2f[k]);
        const float rl = ieva_fmul(p.half_val,
            ieva_fadd(r.rom[ulong(k - 1) * st + ulong(c)],
                     r.rom[ulong(k) * st + ulong(c)]));
        const float rr = ieva_fmul(p.half_val,
            ieva_fadd(r.rom[ulong(k + 1) * st + ulong(c)],
                     r.rom[ulong(k) * st + ulong(c)]));
        float wiL = ieva_fmul(rl, r.rdn[k]);
        float wiR = ieva_fmul(rr, r.rdn[k]);
        if (p.has_msf) {
            wiL = ieva_fmul(wiL, msf);
            wiR = ieva_fmul(wiR, msf);
        }
        wiL = ieva_fdiv(wiL, mass_new);
        wiR = ieva_fdiv(wiR, mass_new);
        ulong at, ct, btmp;
        ieva_coefficients(wiL, wiR, p.dt, p.zero, at, ct, btmp);
        const ulong bt = soft64_add(ieva_widen(p.one), btmp);
        const float mass_old = ieva_fadd(ieva_fmul(r.c1f[k], mo), r.c2f[k]);
        float lead = ieva_fmul(p.dt, r.tend[ulong(k) * st + ulong(c)]);
        if (p.has_msf) lead = ieva_fmul(lead, msf);
        ulong rhs = ieva_rhs(lead, mass_old, at, btmp, ct,
            r.w_old[ulong(k - 1) * st + ulong(c)],
            r.w_old[ulong(k) * st + ulong(c)],
            r.w_old[ulong(k + 1) * st + ulong(c)]);
        const float mass = ieva_fadd(ieva_fmul(r.c1f[k], mt), r.c2f[k]);
        if (k == 1) {
            const ulong vn = ulong(j + 1) * ulong(p.nx) + ulong(i);
            const ulong vs = ulong(j) * ulong(p.nx) + ulong(i);
            const float mvn = p.has_msf ? r.msfv[vn] : p.one;
            const float mvs = p.has_msf ? r.msfv[vs] : p.one;
            float vcn = ieva_fadd(
                ieva_fadd(
                    ieva_fmul(p.cf1, ieva_uncoupled(r.vtend[vn], mvn,
                        p.has_msf, r.c1h[0], r.c2h[0], r.muy[vn])),
                    ieva_fmul(p.cf2, ieva_uncoupled(r.vtend[stv + vn], mvn,
                        p.has_msf, r.c1h[1], r.c2h[1], r.muy[vn]))),
                ieva_fmul(p.cf3, ieva_uncoupled(r.vtend[2ul * stv + vn], mvn,
                    p.has_msf, r.c1h[2], r.c2h[2], r.muy[vn])));
            float vcs = ieva_fadd(
                ieva_fadd(
                    ieva_fmul(p.cf1, ieva_uncoupled(r.vtend[vs], mvs,
                        p.has_msf, r.c1h[0], r.c2h[0], r.muy[vs])),
                    ieva_fmul(p.cf2, ieva_uncoupled(r.vtend[stv + vs], mvs,
                        p.has_msf, r.c1h[1], r.c2h[1], r.muy[vs]))),
                ieva_fmul(p.cf3, ieva_uncoupled(r.vtend[2ul * stv + vs], mvs,
                    p.has_msf, r.c1h[2], r.c2h[2], r.muy[vs])));
            const ulong ue = ulong(j) * ulong(p.nx + 1) + ulong(i + 1);
            const ulong uw = ulong(j) * ulong(p.nx + 1) + ulong(i);
            const float mue = p.has_msf ? r.msfu[ue] : p.one;
            const float muw = p.has_msf ? r.msfu[uw] : p.one;
            const float uce = ieva_fadd(
                ieva_fadd(
                    ieva_fmul(p.cf1, ieva_uncoupled(r.utend[ue], mue,
                        p.has_msf, r.c1h[0], r.c2h[0], r.mux[ue])),
                    ieva_fmul(p.cf2, ieva_uncoupled(r.utend[stu + ue], mue,
                        p.has_msf, r.c1h[1], r.c2h[1], r.mux[ue]))),
                ieva_fmul(p.cf3, ieva_uncoupled(r.utend[2ul * stu + ue], mue,
                    p.has_msf, r.c1h[2], r.c2h[2], r.mux[ue])));
            const float ucw = ieva_fadd(
                ieva_fadd(
                    ieva_fmul(p.cf1, ieva_uncoupled(r.utend[uw], muw,
                        p.has_msf, r.c1h[0], r.c2h[0], r.mux[uw])),
                    ieva_fmul(p.cf2, ieva_uncoupled(r.utend[stu + uw], muw,
                        p.has_msf, r.c1h[1], r.c2h[1], r.mux[uw]))),
                ieva_fmul(p.cf3, ieva_uncoupled(r.utend[2ul * stu + uw], muw,
                    p.has_msf, r.c1h[2], r.c2h[2], r.mux[uw])));
            const float h0 = r.ht[c];
            const float hn = ieva_ht_at(r.ht, j + 1, i, p.ny, p.nx, p.bx, p.by);
            const float hs = ieva_ht_at(r.ht, j - 1, i, p.ny, p.nx, p.bx, p.by);
            const float he = ieva_ht_at(r.ht, j, i + 1, p.ny, p.nx, p.bx, p.by);
            const float hw = ieva_ht_at(r.ht, j, i - 1, p.ny, p.nx, p.bx, p.by);
            const float yterm = ieva_fadd(ieva_fmul(ieva_fsub(hn, h0), vcn),
                                          ieva_fmul(ieva_fsub(h0, hs), vcs));
            const float xterm = ieva_fadd(ieva_fmul(ieva_fsub(he, h0), uce),
                                          ieva_fmul(ieva_fsub(h0, hw), ucw));
            const float dw = ieva_fadd(
                ieva_fmul(ieva_fmul(ieva_fmul(msf, p.half_val), p.rdy), yterm),
                ieva_fmul(ieva_fmul(ieva_fmul(msf, p.half_val), p.rdx), xterm));
            ulong correction = soft64_mul(ieva_widen(mass), at);
            correction = soft64_mul(correction, ieva_widen(p.dt));
            correction = soft64_mul(correction, ieva_widen(dw));
            rhs = soft64_sub(rhs, correction);
        }
        if (k == kend) {
            const ulong top = ulong(k + 1) * st + ulong(c);
            float dph = ieva_fdiv(ieva_fdiv(
                ieva_fsub(r.ph_new[top], r.ph_old[top]), p.dt), p.g);
            float pt = ieva_fdiv(ieva_fdiv(r.ph_tend[top], mass), p.g);
            float dw = ieva_fsub(dph, pt);
            if (p.has_msf) dw = ieva_fmul(msf, dw);
            if (p.legacy) {
                dph = ieva_fdiv(ieva_fsub(r.ph_new[top], r.ph_old[top]), p.dt);
                pt = ieva_fdiv(r.ph_tend[top], mass);
                dw = ieva_fdiv(ieva_fmul(msf, ieva_fsub(dph, pt)), p.g);
            }
            float boundary = ieva_fsub(dw, r.w_old[top]);
            rhs = soft64_sub(rhs, soft64_mul(
                soft64_mul(ieva_widen(mass), ct), ieva_widen(boundary)));
        }
        ieva_forward(k, 1, at, bt, cprev, rhs, bet, gamma, values);
        cprev = ct;
    }
    ieva_backward(1, kend, gamma, values);
    for (int k = 1; k <= kend; ++k) {
        float out = ieva_fdiv(ieva_narrow(values[k]), p.dt);
        if (p.has_msf) out = ieva_fdiv(out, msf);
        r.tend[ulong(k) * st + ulong(c)] = out;
    }
}
