#include <metal_stdlib>
#include "sase_ddz_coefficients_bindings.metalh"
#include "sase_n2_bindings.metalh"
#include "sase_box_filter_bindings.metalh"
#include "sase_strain_bindings.metalh"
#include "sase_e_hflux_bindings.metalh"
#include "sase_scalar_hflux_km_bindings.metalh"
#include "sase_hflux_div_bindings.metalh"
#include "sase_model_stress_gov_bindings.metalh"
#include "sase_split_tendencies_bindings.metalh"
#include "sase_velocity_products_bindings.metalh"
#include "sase_lift_combine_bindings.metalh"
#include "sase_vertical_production_bindings.metalh"
#include "sase_blackadar_length_bindings.metalh"
#include "sase_basis_premultiply_bindings.metalh"
#include "sase_solve_partial_bindings.metalh"
#include "sase_model_stress_bindings.metalh"
#include "sase_scalar_hflux_bindings.metalh"
#include "sase_diff_flux_diag_bindings.metalh"
#include "sase_vent_flux_diag_bindings.metalh"
#include "sase_destagger_w_bindings.metalh"
#include "sase_surface_source_bindings.metalh"
#include "sase_surface_rho1_bindings.metalh"
#include "sase_plume_vent_flux_bindings.metalh"
#include "sase_vent_deposit_scale_bindings.metalh"
#include "sase_vent_deposit_bindings.metalh"
#include "sase_rate_bindings.metalh"
#include "sase_structure_partial_bindings.metalh"
#include "sase_w_sensor_partial_bindings.metalh"
#include "sase_zi_column_bindings.metalh"
#include "sase_thomas_scalar_bindings.metalh"
#include "sase_thomas_momentum_bindings.metalh"
#include "sase_vertical_channel_bindings.metalh"
#include "sase_split_e_update_bindings.metalh"
#include "sase_taper_weights_bindings.metalh"
#include "sase_moist_n2_bindings.metalh"
#include "sase_couple_w_tendency_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "soft_binary32.metalh"

using namespace metal;
#pragma clang fp contract(off)

static inline uint sase_idx3(uint k, uint j, uint i, uint ny, uint nx) {
    return (k * ny + j) * nx + i;
}

static inline int sase_periodic(int i, int n) {
    return (i % n + n) % n;
}

static inline uint sase_km(uint k) { return k == 0u ? 0u : k - 1u; }
static inline uint sase_kp(uint k, uint nz) {
    return k + 1u == nz ? k : k + 1u;
}

static inline ulong sase_f64(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}

static inline float sase_f32(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}

static inline bool sase_ge64(ulong a, ulong b) {
    const bool sa = (a & 0x8000000000000000ul) != 0ul;
    const bool sb = (b & 0x8000000000000000ul) != 0ul;
    if (sa != sb) return !sa;
    return sa ? a <= b : a >= b;
}

static inline float sase_face_spacing(device const float *t, uint q,
                                      uint plane, bool upper) {
    const ulong half_w = 0x3fe0000000000000ul;
    const ulong a = sase_f64(t[q]);
    const ulong b = sase_f64(t[upper ? q + plane : q - plane]);
    return sase_f32(soft64_mul(half_w, soft64_add(a, b)));
}

static inline ulong sase_thick64(device const float *t, uint k, uint col,
                                 uint ncol, float dz, int mode) {
    if (mode == 0) return sase_f64(dz);
    return sase_f64(t[k * ncol + col]);
}

static inline ulong sase_min64(ulong a, ulong b) {
    return sase_ge64(a, b) ? b : a;
}

static inline ulong sase_max64(ulong a, ulong b) {
    return sase_ge64(a, b) ? a : b;
}

static inline ulong sase_cbrt64(ulong x) {
    const ulong two = 0x4000000000000000ul;
    const ulong three = 0x4008000000000000ul;
    ulong y = sase_f64(pow(sase_f32(x), 1.0f / 3.0f));
    for (uint i = 0u; i < 4u; ++i) {
        const ulong y2 = soft64_mul(y, y);
        const ulong quotient = soft64_div(x, y2);
        y = soft64_div(soft64_add(soft64_mul(two, y), quotient), three);
    }
    return y;
}

static inline ulong sase_negative_two_thirds_mean(ulong a, ulong b) {
    const ulong two = 0x4000000000000000ul;
    const ulong ca = sase_cbrt64(a);
    const ulong cb = sase_cbrt64(b);
    const ulong aa = soft64_mul(ca, ca);
    const ulong bb = soft64_mul(cb, cb);
    const ulong harmonic = soft64_div(soft64_mul(two,
                                      soft64_mul(aa, bb)),
                                      soft64_add(aa, bb));
    return soft64_mul(harmonic, soft64_sqrt(harmonic));
}

static inline ulong sase_bl89_crossing64(ulong c1, ulong c2,
                                          ulong rem, ulong h) {
    const ulong zero = 0ul;
    const ulong half_w = 0x3fe0000000000000ul;
    const ulong four = 0x4010000000000000ul;
    const ulong tol = soft64_mul(0x3d719799812dea11ul, h); // 1e-12*h
    const ulong neg_tol = soft64_negate_bits(tol);
    ulong best = soft64_negate_bits(0x3ff0000000000000ul);
    if ((c2 & 0x7ffffffffffffffful) == 0ul) {
        if (sase_ge64(c1, zero) && (c1 & 0x7ffffffffffffffful) != 0ul) {
            ulong s = soft64_div(rem, c1);
            if (sase_ge64(s, neg_tol)
                    && sase_ge64(soft64_add(h, tol), s))
                best = sase_min64(h, sase_max64(s, zero));
        }
        return best;
    }
    const ulong disc = soft64_add(soft64_mul(c1, c1),
                                  soft64_mul(four, soft64_mul(c2, rem)));
    if (!sase_ge64(disc, zero)) return best;
    const ulong sq = soft64_sqrt(disc);
    const ulong signed_sq = sase_ge64(c1, zero)
        ? sq : soft64_negate_bits(sq);
    const ulong q = soft64_negate_bits(soft64_mul(half_w,
                                               soft64_add(c1, signed_sq)));
    ulong roots[2];
    roots[0] = soft64_div(q, c2);
    roots[1] = (q & 0x7ffffffffffffffful) != 0ul
        ? soft64_negate_bits(soft64_div(rem, q)) : zero;
    for (uint i = 0u; i < 2u; ++i) {
        const ulong root = roots[i];
        if (sase_ge64(root, neg_tol)
                && sase_ge64(soft64_add(h, tol), root)) {
            const ulong clipped = sase_min64(h, sase_max64(root, zero));
            if (!sase_ge64(best, zero) || sase_ge64(best, clipped))
                best = clipped;
        }
    }
    return best;
}

static inline float sase_ddz(float fm, float fc, float fp, uint k, uint q,
                             uint nz, device const float *cm,
                             device const float *c0, device const float *cp,
                             constant SaseStrainParams &p) {
    if (p.z_mode == 3) return cm[q] * fm + c0[q] * fc + cp[q] * fp;
    if (k == 0u) return (fp - fc) / p.dz;
    if (k + 1u == nz) return (fc - fm) / (p.z_mode ? p.h_hi : p.dz);
    if (p.z_mode != 0) return cm[k] * fm + c0[k] * fc + cp[k] * fp;
    return (fp - fm) / p.two_dz;
}

kernel void sase_box_filter(
    constant SaseBoxFilterResources &r [[buffer(0)]],
    constant SaseBoxFilterParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const int i = int(tid % p.nx);
    const int j = int((tid / p.nx) % p.ny);
    const uint k = tid / (p.nx * p.ny);
    const float w2[3] = {0.25f, 0.5f, 0.25f};
    const float w4[5] = {0.125f, 0.25f, 0.25f, 0.25f, 0.125f};
    const bool wide = p.width == 4;
    const int half_w = wide ? 2 : 1;
    float acc = 0.0f;
    for (int jj = -half_w; jj <= half_w; ++jj) {
        const int jc = sase_periodic(j + jj, int(p.ny));
        float row = 0.0f;
        for (int ii = -half_w; ii <= half_w; ++ii) {
            const int ic = sase_periodic(i + ii, int(p.nx));
            const float weight = wide ? w4[ii + half_w] : w2[ii + half_w];
            row += weight * r.f[sase_idx3(k, uint(jc), uint(ic), p.ny, p.nx)];
        }
        const float weight = wide ? w4[jj + half_w] : w2[jj + half_w];
        acc += weight * row;
    }
    r.out[tid] = acc;
}

kernel void sase_model_stress(
    constant SaseModelStressResources &r [[buffer(0)]],
    constant SaseModelStressParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const float ef = fmax(r.e[tid], p.e_min);
    const float root_e = sqrt(ef);
    const float nu_eddy = p.c_nu * p.delta_eddy * root_e;
    const float nu_mom = p.ck_over_prt * p.delta_mom * root_e;
    const float nu = p.f_blend * nu_eddy + (1.0f - p.f_blend) * nu_mom;
    const float a = r.s0[tid], b = r.s1[tid];
    const float div3 = (a + b + r.s2[tid]) / 3.0f;
    const float iso_visc = 2.0f * nu * div3;
    const float iso_e = (2.0f / 3.0f) * ef;
    const float txx = (-2.0f * nu * a + iso_visc) + iso_e;
    const float tyy = (-2.0f * nu * b + iso_visc) + iso_e;
    r.t0[tid] = txx;
    r.t1[tid] = tyy;
    r.t2[tid] = (2.0f * ef - txx) - tyy;
    r.t3[tid] = -2.0f * nu * r.s3[tid];
    r.t4[tid] = -2.0f * nu * r.s4[tid];
    r.t5[tid] = -2.0f * nu * r.s5[tid];
}

kernel void sase_scalar_hflux(
    constant SaseScalarHfluxResources &r [[buffer(0)]],
    constant SaseScalarHfluxParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const int i = int(tid % p.nx);
    const int j = int((tid / p.nx) % p.ny);
    const uint k = tid / (p.nx * p.ny);
    const uint ip = uint(sase_periodic(i + 1, int(p.nx)));
    const uint im = uint(sase_periodic(i - 1, int(p.nx)));
    const uint jp = uint(sase_periodic(j + 1, int(p.ny)));
    const uint jm = uint(sase_periodic(j - 1, int(p.ny)));
    const float kh = p.kh_coef * sqrt(fmax(r.e[tid], p.e_min));
    const float dsdx = (r.s[sase_idx3(k, uint(j), ip, p.ny, p.nx)]
                      - r.s[sase_idx3(k, uint(j), im, p.ny, p.nx)]) / p.two_dx;
    const float dsdy = (r.s[sase_idx3(k, jp, uint(i), p.ny, p.nx)]
                      - r.s[sase_idx3(k, jm, uint(i), p.ny, p.nx)]) / p.two_dy;
    r.fx[tid] = kh * dsdx;
    r.fy[tid] = kh * dsdy;
}

kernel void sase_vent_flux_diag(
    constant SaseVentFluxDiagResources &r [[buffer(0)]],
    constant SaseVentFluxDiagParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncol = uint(p.ny) * uint(p.nx);
    if (tid >= uint(p.nzp1) * ncol) return;
    const ulong scaled = soft64_mul(r.scale[tid % ncol], sase_f64(r.f_row[tid]));
    r.out[tid] = sase_f32(soft64_mul(r.constants[0], scaled));
}

kernel void sase_diff_flux_diag(
    constant SaseDiffFluxDiagResources &r [[buffer(0)]],
    constant SaseDiffFluxDiagParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    const uint nface = (p.nz + 1u) * ncol;
    if (tid >= nface) return;
    const uint k = tid / ncol;
    const uint col = tid % ncol;
    if (k == 0u || k == p.nz) {
        r.out[tid] = 0.0f;
        return;
    }
    const uint q = k * ncol + col;
    const ulong half_w = 0x3fe0000000000000ul;
    const ulong tk0 = sase_thick64(r.t, k - 1u, col, ncol, p.dz, p.t_mode);
    const ulong tk1 = sase_thick64(r.t, k, col, ncol, p.dz, p.t_mode);
    const ulong h = soft64_mul(half_w, soft64_add(tk0, tk1));
    const ulong kfac = r.constants[0];
    const ulong fac = r.constants[1];
    const ulong kvsum = soft64_add(sase_f64(r.kv[q - ncol]),
                                   sase_f64(r.kv[q]));
    const ulong kf = soft64_mul(kfac, soft64_mul(half_w, kvsum));
    const ulong grad = soft64_div(soft64_sub(sase_f64(r.phi[q]),
                                             sase_f64(r.phi[q - ncol])), h);
    const ulong flux = soft64_mul(fac, soft64_mul(
        soft64_negate_bits(sase_f64(r.rho1[col])), soft64_mul(kf, grad)));
    r.out[tid] = sase_f32(flux);
}

kernel void sase_destagger_w(
    constant SaseDestaggerWResources &r [[buffer(0)]],
    constant SaseDestaggerWParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const uint plane = p.ny * p.nx;
    r.out[tid] = 0.5f * (r.w[tid] + r.w[tid + plane]);
}

kernel void sase_surface_source(
    constant SaseSurfaceSourceResources &r [[buffer(0)]],
    constant SaseSurfaceSourceParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    if (col >= ncol) return;
    const float rho = r.rho1[col];
    const float wtv = r.hfx[col] / (rho * p.cp_air)
                    + p.ep1 * r.theta1[col] * r.qfx[col] / rho;
    const float buoy = fmax((p.g_accel / r.theta1[col]) * wtv, 0.0f);
    const float ustar = r.ust[col];
    const float shear = ustar * ustar * ustar
                      / (p.karman * 0.5f * r.dz1[col]);
    r.e_sgs[col] += p.dt * (shear + buoy);
}

kernel void sase_surface_rho1(
    constant SaseSurfaceRho1Resources &r [[buffer(0)]],
    constant SaseSurfaceRho1Params &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    if (col >= ncol) return;
    r.rho1[col] = r.pressure[col]
        / (p.rd * r.temperature[col] * (1.0f + p.ep1 * r.qv[col]));
}

kernel void sase_rate(
    constant SaseRateResources &r [[buffer(0)]],
    constant SaseRateParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    if (p.mode == 0) {
        r.out[tid] = (r.a[tid] - r.b[tid]) * p.inv_dt;
    } else if (p.mode == 1) {
        r.out[tid] += (r.a[tid] * p.heat_scale) / r.b[tid];
    } else {
        r.out[tid] = r.a[tid] + p.inv_dt * r.b[tid];
    }
}

kernel void sase_structure_partial(
    constant SaseStructurePartialResources &r [[buffer(0)]],
    constant SaseStructurePartialParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup ulong sums[3][128];
    const uint ncell = p.nz * p.ny * p.nx;
    ulong local[3] = {0ul, 0ul, 0ul};
    if (tid < ncell) {
        const int i = int(tid % p.nx);
        const int j = int((tid / p.nx) % p.ny);
        const uint k = tid / (p.nx * p.ny);
        const float fc = r.f[tid];
        const int rs[3] = {1, 2, 4};
        for (uint t = 0u; t < 3u; ++t) {
            const float dxr = r.f[sase_idx3(k, uint(j),
                uint(sase_periodic(i + rs[t], int(p.nx))), p.ny, p.nx)] - fc;
            const float dyr = r.f[sase_idx3(k,
                uint(sase_periodic(j + rs[t], int(p.ny))), uint(i), p.ny, p.nx)] - fc;
            local[t] = soft64_add(soft64_mul(sase_f64(dxr), sase_f64(dxr)),
                                  soft64_mul(sase_f64(dyr), sase_f64(dyr)));
        }
    }
    for (uint t = 0u; t < 3u; ++t) sums[t][lane] = local[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (lane < stride)
            for (uint t = 0u; t < 3u; ++t)
                sums[t][lane] = soft64_add(sums[t][lane], sums[t][lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u)
        for (uint t = 0u; t < 3u; ++t)
            r.partials[t * p.nblocks + group] = sums[t][0];
}

kernel void sase_w_sensor_partial(
    constant SaseWSensorPartialResources &r [[buffer(0)]],
    constant SaseWSensorPartialParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup ulong sums[5][128];
    const uint ncell = p.nz * p.ny * p.nx;
    ulong local[5] = {0ul, 0ul, 0ul, 0ul, 0ul};
    if (tid < ncell) {
        const int i = int(tid % p.nx);
        const int j = int((tid / p.nx) % p.ny);
        const uint k = tid / (p.nx * p.ny);
        const bool interior = !(p.bw > 0 && (i < p.bw || i >= int(p.nx) - p.bw
                                 || j < p.bw || j >= int(p.ny) - p.bw));
        if (interior) {
            local[4] = sase_f64(fmax(r.e[tid], p.e_min));
            const bool pass = (p.has_n2 == 0) || (r.n2[tid] <= p.n2_screen);
            if (pass) {
                const float wc = r.w[tid];
                const int rs[3] = {1, 2, 4};
                for (uint t = 0u; t < 3u; ++t) {
                    const float dxr = r.w[sase_idx3(k, uint(j),
                        uint(sase_periodic(i + rs[t], int(p.nx))), p.ny, p.nx)] - wc;
                    const float dyr = r.w[sase_idx3(k,
                        uint(sase_periodic(j + rs[t], int(p.ny))), uint(i), p.ny, p.nx)] - wc;
                    local[t] = soft64_add(soft64_mul(sase_f64(dxr), sase_f64(dxr)),
                                          soft64_mul(sase_f64(dyr), sase_f64(dyr)));
                }
                local[3] = 0x3ff0000000000000ul;
            }
        }
    }
    for (uint t = 0u; t < 5u; ++t) sums[t][lane] = local[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (lane < stride)
            for (uint t = 0u; t < 5u; ++t)
                sums[t][lane] = soft64_add(sums[t][lane], sums[t][lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u)
        for (uint t = 0u; t < 5u; ++t)
            r.partials[t * p.nblocks + group] = sums[t][0];
}

kernel void sase_zi_column(
    constant SaseZiColumnResources &r [[buffer(0)]],
    constant SaseZiColumnParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    if (col >= ncol || p.nz == 0u) return;
    const ulong half_w = 0x3fe0000000000000ul;
    const ulong rib_crit = sase_f64(p.rib_crit);
    const ulong wspd_floor = sase_f64(p.wspd2_floor);
    const ulong gravity = sase_f64(p.g_accel);
    ulong tk = sase_thick64(r.t, 0u, col, ncol, p.dz, p.t_mode);
    ulong z_prev = soft64_mul(half_w, tk);
    if (p.nz == 1u) {
        r.zi[col] = sase_f32(z_prev);
        return;
    }
    ulong zc = tk;
    const ulong thermal = sase_f64(r.theta[col]);
    ulong rib_prev = 0ul;
    ulong z1_int = 0ul;
    ulong out = 0xfffffffffffffffful;
    ulong zk = z_prev;
    for (uint k = 1u; k < p.nz; ++k) {
        tk = sase_thick64(r.t, k, col, ncol, p.dz, p.t_mode);
        zk = soft64_add(zc, soft64_mul(half_w, tk));
        zc = soft64_add(zc, tk);
        if (k == 1u) z1_int = zk;
        const uint q = k * ncol + col;
        const ulong uu = sase_f64(r.u[q]);
        const ulong vv = sase_f64(r.v[q]);
        ulong spd2 = soft64_add(soft64_mul(uu, uu), soft64_mul(vv, vv));
        if (spd2 < wspd_floor) spd2 = wspd_floor;
        const ulong rib = soft64_div(soft64_mul(
            soft64_sub(sase_f64(r.theta[q]), thermal),
            soft64_div(soft64_mul(gravity, zk), thermal)), spd2);
        if (out == 0xfffffffffffffffful && sase_ge64(rib, rib_crit)) {
            const ulong frac = soft64_div(soft64_sub(rib_crit, rib_prev),
                                          soft64_sub(rib, rib_prev));
            out = soft64_add(z_prev, soft64_mul(frac, soft64_sub(zk, z_prev)));
        }
        z_prev = zk;
        rib_prev = rib;
    }
    if (out == 0xfffffffffffffffful) out = zk;
    if (!sase_ge64(out, z1_int)) out = z1_int;
    r.zi[col] = sase_f32(out);
}

kernel void sase_thomas_scalar(
    constant SaseThomasScalarResources &r [[buffer(0)]],
    constant SaseThomasScalarParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    if (col >= ncol || p.nz < 2u || p.nz > 128u) return;
    thread ulong conductance[127];
    thread ulong cprime[128];
    thread ulong dprime[128];
    const ulong half_w = 0x3fe0000000000000ul;
    const ulong one = 0x3ff0000000000000ul;
    const ulong dt64 = sase_f64(p.dt);
    const ulong kfac64 = sase_f64(p.kfac);
    ulong tk = sase_thick64(r.t, 0u, col, ncol, p.dz, p.t_mode);
    ulong deposit = 0ul;
    if (p.has_sfc != 0) {
        const ulong flux = sase_f64(r.sfc_flux[col]);
        if ((flux & 0x7ffffffffffffffful) != 0ul) {
            const ulong rho = sase_f64(r.sfc_rho1[col]);
            const ulong scale = soft64_mul(
                soft64_mul(rho, r.constants[0]), tk);
            deposit = soft64_div(soft64_mul(dt64, flux), scale);
        }
    }
    for (uint k = 0u; k + 1u < p.nz; ++k) {
        const ulong tk1 = sase_thick64(r.t, k + 1u, col, ncol,
                                       p.dz, p.t_mode);
        const ulong h = soft64_mul(half_w, soft64_add(tk, tk1));
        const uint q = k * ncol + col;
        const ulong kf = soft64_mul(kfac64, soft64_mul(half_w,
            soft64_add(sase_f64(r.kv[q]), sase_f64(r.kv[q + ncol]))));
        conductance[k] = soft64_div(soft64_mul(dt64, kf), h);
        tk = tk1;
    }
    tk = sase_thick64(r.t, 0u, col, ncol, p.dz, p.t_mode);
    ulong sup = soft64_negate_bits(soft64_div(conductance[0], tk));
    ulong diag = soft64_sub(one, sup);
    uint q = col;
    ulong rhs = sase_f64(r.phi[q]);
    if ((deposit & 0x7ffffffffffffffful) != 0ul)
        rhs = soft64_add(rhs, deposit);
    cprime[0] = soft64_div(sup, diag);
    dprime[0] = soft64_div(rhs, diag);
    for (uint k = 1u; k < p.nz; ++k) {
        q = k * ncol + col;
        tk = sase_thick64(r.t, k, col, ncol, p.dz, p.t_mode);
        const ulong sub = soft64_negate_bits(soft64_div(conductance[k - 1u], tk));
        sup = (k + 1u < p.nz)
            ? soft64_negate_bits(soft64_div(conductance[k], tk)) : 0ul;
        diag = soft64_sub(soft64_sub(one, sub), sup);
        const ulong denom = soft64_sub(diag, soft64_mul(sub, cprime[k - 1u]));
        rhs = sase_f64(r.phi[q]);
        cprime[k] = soft64_div(sup, denom);
        dprime[k] = soft64_div(soft64_sub(rhs,
                    soft64_mul(sub, dprime[k - 1u])), denom);
    }
    ulong prev = 0ul;
    for (int k = int(p.nz) - 1; k >= 0; --k) {
        q = uint(k) * ncol + col;
        const ulong solved = (uint(k) + 1u == p.nz)
            ? dprime[k] : soft64_sub(dprime[k], soft64_mul(cprime[k], prev));
        prev = solved;
        float stored = sase_f32(solved);
        if (p.has_floor != 0) stored = fmax(stored, p.floor_val);
        r.phi[q] = stored;
    }
}

static inline void sase_thomas_momentum_column(
    device float *phi, device const float *dphi,
    thread const ulong *conductance, device const float *thickness,
    float dz, int t_mode, ulong dt64, ulong bottom,
    uint nz, uint ncol, uint col, thread ulong *cprime,
    thread ulong *dprime, thread ulong &m_expl,
    thread ulong &m_impl) {
    const ulong one = 0x3ff0000000000000ul;
    ulong tk = sase_thick64(thickness, 0u, col, ncol, dz, t_mode);
    ulong sup = soft64_negate_bits(soft64_div(conductance[0], tk));
    ulong diag = soft64_add(soft64_sub(one, sup), bottom);
    uint q = col;
    ulong old = sase_f64(phi[q]);
    ulong rhs = soft64_add(old, soft64_mul(dt64, sase_f64(dphi[q])));
    m_expl = soft64_add(m_expl, soft64_mul(old, soft64_sub(rhs, old)));
    cprime[0] = soft64_div(sup, diag);
    dprime[0] = soft64_div(rhs, diag);
    for (uint k = 1u; k < nz; ++k) {
        q = k * ncol + col;
        tk = sase_thick64(thickness, k, col, ncol, dz, t_mode);
        const ulong sub = soft64_negate_bits(soft64_div(conductance[k - 1u], tk));
        sup = (k + 1u < nz)
            ? soft64_negate_bits(soft64_div(conductance[k], tk)) : 0ul;
        diag = soft64_sub(soft64_sub(one, sub), sup);
        const ulong denom = soft64_sub(diag, soft64_mul(sub, cprime[k - 1u]));
        old = sase_f64(phi[q]);
        rhs = soft64_add(old, soft64_mul(dt64, sase_f64(dphi[q])));
        m_expl = soft64_add(m_expl, soft64_mul(old, soft64_sub(rhs, old)));
        cprime[k] = soft64_div(sup, denom);
        dprime[k] = soft64_div(soft64_sub(rhs,
                         soft64_mul(sub, dprime[k - 1u])), denom);
    }
    ulong prev = 0ul;
    for (int k = int(nz) - 1; k >= 0; --k) {
        q = uint(k) * ncol + col;
        const ulong solved = (uint(k) + 1u == nz)
            ? dprime[k] : soft64_sub(dprime[k], soft64_mul(cprime[k], prev));
        prev = solved;
        const float stored = sase_f32(solved);
        const ulong stored64 = sase_f64(stored);
        old = sase_f64(phi[q]);
        rhs = soft64_add(old, soft64_mul(dt64, sase_f64(dphi[q])));
        m_impl = soft64_add(m_impl, soft64_mul(stored64,
                                               soft64_sub(stored64, rhs)));
        phi[q] = stored;
    }
}

kernel void sase_thomas_momentum(
    constant SaseThomasMomentumResources &r [[buffer(0)]],
    constant SaseThomasMomentumParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup ulong sums[3][128];
    const uint ncol = p.ny * p.nx;
    ulong local[3] = {0ul, 0ul, 0ul};
    if (tid < ncol) {
        thread ulong conductance[127];
        thread ulong cprime[128];
        thread ulong dprime[128];
        const ulong half_w = 0x3fe0000000000000ul;
        const ulong dt64 = sase_f64(p.dt);
        ulong t0 = sase_thick64(r.t, 0u, tid, ncol, p.dz, p.t_mode);
        if (p.nz > 1u) {
            ulong tk = t0;
            for (uint k = 0u; k + 1u < p.nz; ++k) {
                const ulong tk1 = sase_thick64(r.t, k + 1u, tid, ncol,
                                               p.dz, p.t_mode);
                const ulong h = soft64_mul(half_w, soft64_add(tk, tk1));
                const uint q = k * ncol + tid;
                const ulong kf = soft64_mul(half_w, soft64_add(
                    sase_f64(r.kv[q]), sase_f64(r.kv[q + ncol])));
                conductance[k] = soft64_div(soft64_mul(dt64, kf), h);
                tk = tk1;
            }
            ulong bottom = 0ul, cdrag = 0ul;
            if (p.has_drag != 0) {
                const ulong u1 = sase_f64(r.u[tid]);
                const ulong v1 = sase_f64(r.v[tid]);
                const ulong spd2 = soft64_add(soft64_mul(u1, u1),
                                               soft64_mul(v1, v1));
                ulong spd = soft64_sqrt(spd2);
                const ulong floor64 = r.constants[0];
                if (!sase_ge64(spd, floor64)) spd = floor64;
                const ulong us = sase_f64(r.ust[tid]);
                cdrag = soft64_div(soft64_mul(us, us), spd);
                if (p.has_wspd != 0) {
                    ulong wspd = sase_f64(r.wspd[tid]);
                    const ulong tiny = 0x3e112e0be826d695ul; // 1e-9
                    if (!sase_ge64(wspd, tiny)) wspd = tiny;
                    const ulong ratio = soft64_div(spd, wspd);
                    cdrag = soft64_mul(cdrag, soft64_mul(ratio, ratio));
                }
                bottom = soft64_div(soft64_mul(dt64, cdrag), t0);
            }
            sase_thomas_momentum_column(r.u, r.du, conductance, r.t,
                p.dz, p.t_mode, dt64, bottom, p.nz, ncol, tid,
                cprime, dprime, local[0], local[1]);
            sase_thomas_momentum_column(r.v, r.dv, conductance, r.t,
                p.dz, p.t_mode, dt64, bottom, p.nz, ncol, tid,
                cprime, dprime, local[0], local[1]);
            sase_thomas_momentum_column(r.w, r.dw, conductance, r.t,
                p.dz, p.t_mode, dt64, 0ul, p.nz, ncol, tid,
                cprime, dprime, local[0], local[1]);
            if (p.has_drag != 0) {
                const ulong u1n = sase_f64(r.u[tid]);
                const ulong v1n = sase_f64(r.v[tid]);
                const ulong speed2 = soft64_add(soft64_mul(u1n, u1n),
                                                soft64_mul(v1n, v1n));
                local[2] = soft64_negate_bits(soft64_div(
                    soft64_mul(dt64, soft64_mul(cdrag, speed2)), t0));
            }
        } else {
            device float *fields[3] = {r.u, r.v, r.w};
            device const float *tends[3] = {r.du, r.dv, r.dw};
            for (uint c = 0u; c < 3u; ++c) {
                const ulong old = sase_f64(fields[c][tid]);
                const ulong rhs = soft64_add(old,
                    soft64_mul(dt64, sase_f64(tends[c][tid])));
                local[0] = soft64_add(local[0], soft64_mul(old,
                                                           soft64_sub(rhs, old)));
                const float stored = sase_f32(rhs);
                const ulong stored64 = sase_f64(stored);
                local[1] = soft64_add(local[1], soft64_mul(stored64,
                                                         soft64_sub(stored64, rhs)));
                fields[c][tid] = stored;
            }
        }
    }
    for (uint c = 0u; c < 3u; ++c) sums[c][lane] = local[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (lane < stride)
            for (uint c = 0u; c < 3u; ++c)
                sums[c][lane] = soft64_add(sums[c][lane], sums[c][lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u)
        for (uint c = 0u; c < 3u; ++c)
            r.partials[c * p.nblocks + group] = sums[c][0];
}

kernel void sase_vertical_channel(
    constant SaseVerticalChannelResources &r [[buffer(0)]],
    constant SaseVerticalChannelParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    if (col >= ncol || p.nz == 0u || p.nz > 128u) return;
    thread ulong zc[128];
    thread ulong th[128];
    thread ulong energy[128];
    thread ulong n2eff[128];
    const ulong half_w = 0x3fe0000000000000ul;
    const ulong one = 0x3ff0000000000000ul;
    const ulong two = 0x4000000000000000ul;
    const ulong zero = 0ul;
    ulong zsum = zero;
    for (uint k = 0u; k < p.nz; ++k) {
        const uint q = k * ncol + col;
        const ulong tk = sase_thick64(r.t, k, col, ncol, p.dz, p.t_mode);
        zc[k] = soft64_add(zsum, soft64_mul(half_w, tk));
        zsum = soft64_add(zsum, tk);
        th[k] = sase_f64(r.theta[q]);
        energy[k] = sase_f64(fmax(r.e[q], p.e_min));
        if (p.has_moist != 0) n2eff[k] = sase_f64(r.n2[q]);
    }
    const ulong htop = zsum;
    const ulong z0 = sase_f64(p.z0);
    const ulong karman = sase_f64(p.karman);
    const ulong lambda = sase_f64(p.blk_lambda);
    const ulong ckv = sase_f64(p.c_kv);
    const ulong lscoef = sase_f64(p.ls_coef);
    const ulong fblend = sase_f64(p.f_blend);
    const ulong gravity = sase_f64(p.g_accel);
    const ulong cks = sase_f64(p.c_ks);
    const ulong cks_exp = sase_f64(p.cks_exp);
    for (uint k = 0u; k < p.nz; ++k) {
        const uint q = k * ncol + col;
        const ulong z = zc[k];
        const ulong kz = soft64_mul(karman, soft64_add(z, z0));
        const ulong lb = soft64_div(kz, soft64_add(one,
                                                  soft64_div(kz, lambda)));
        const ulong e64 = energy[k];
        const ulong root_e = soft64_sqrt(e64);
        ulong l_les = lb;
        ulong ls_v = zero;
        if (p.has_n2 != 0) {
            const ulong n2v = sase_f64(r.n2[q]);
            if (sase_ge64(n2v, zero) && (n2v & 0x7ffffffffffffffful) != 0ul) {
                const ulong ls = soft64_div(soft64_mul(lscoef, root_e),
                                            soft64_sqrt(n2v));
                ls_v = ls;
                l_les = sase_min64(l_les, ls);
            }
        }
        const ulong beta = soft64_div(gravity, th[k]);
        ulong bl[2];
        for (uint down = 0u; down < 2u; ++down) {
            ulong rem = e64;
            ulong acc = zero;
            ulong got = soft64_negate_bits(one);
            ulong prev_z = zc[k], prev_t = th[k];
            const uint nsteps = down != 0u ? k + 1u : p.nz - k;
            for (uint m = 0u; m < nsteps && !sase_ge64(got, zero); ++m) {
                const int j = down != 0u ? int(k) - 1 - int(m)
                                         : int(k) + 1 + int(m);
                ulong node_z, node_t, h, c1, c2d;
                if (down != 0u) {
                    node_z = j >= 0 ? zc[uint(j)] : zero;
                    node_t = j >= 0 ? th[uint(j)] : th[0];
                    h = soft64_sub(prev_z, node_z);
                    c1 = soft64_mul(beta, soft64_sub(th[k], prev_t));
                    c2d = soft64_mul(beta, soft64_sub(prev_t, node_t));
                } else {
                    node_z = j < int(p.nz) ? zc[uint(j)] : htop;
                    node_t = j < int(p.nz) ? th[uint(j)] : th[p.nz - 1u];
                    h = soft64_sub(node_z, prev_z);
                    c1 = soft64_mul(beta, soft64_sub(prev_t, th[k]));
                    c2d = soft64_mul(beta, soft64_sub(node_t, prev_t));
                }
                if (sase_ge64(h, zero) && (h & 0x7ffffffffffffffful) != 0ul) {
                    const ulong c2 = soft64_div(c2d, soft64_mul(two, h));
                    const ulong s = sase_bl89_crossing64(c1, c2, rem, h);
                    if (sase_ge64(s, zero)) {
                        got = soft64_add(acc, s);
                    } else {
                        const ulong seg = soft64_mul(
                            soft64_add(c1, soft64_mul(c2, h)), h);
                        rem = sase_max64(soft64_sub(rem, seg), zero);
                    }
                    acc = soft64_add(acc, h);
                }
                prev_z = node_z;
                prev_t = node_t;
            }
            if (!sase_ge64(got, zero))
                got = down != 0u ? z : soft64_sub(htop, z);
            bl[down] = got;
        }
        const ulong l_mix_bl = sase_negative_two_thirds_mean(bl[0], bl[1]);
        const ulong l_eps_bl = sase_min64(bl[0], bl[1]);
        ulong l_mix_r = sase_min64(l_les, l_mix_bl);
        ulong l_eps_r = sase_min64(lb, l_eps_bl);
        if (p.has_moist != 0 && r.n2[q] != r.n2d[q]) {
            ulong lm[2];
            for (uint down = 0u; down < 2u; ++down) {
                ulong rem = e64, acc = zero, run = zero;
                ulong got = soft64_negate_bits(one);
                ulong prev_z = zc[k];
                const uint nsteps = down != 0u ? k + 1u : p.nz - k;
                for (uint m = 0u; m < nsteps && !sase_ge64(got, zero); ++m) {
                    const int j = down != 0u ? int(k) - 1 - int(m)
                                             : int(k) + 1 + int(m);
                    ulong node_z, h, slope;
                    if (down != 0u) {
                        node_z = j >= 0 ? zc[uint(j)] : zero;
                        h = soft64_sub(prev_z, node_z);
                        slope = j >= 0 ? soft64_mul(half_w,
                            soft64_add(n2eff[uint(j)], n2eff[uint(j) + 1u]))
                            : zero;
                    } else {
                        node_z = j < int(p.nz) ? zc[uint(j)] : htop;
                        h = soft64_sub(node_z, prev_z);
                        slope = j < int(p.nz) ? soft64_mul(half_w,
                            soft64_add(n2eff[uint(j) - 1u], n2eff[uint(j)]))
                            : zero;
                    }
                    const ulong c1 = run;
                    const ulong c2 = sase_ge64(h, zero)
                        ? soft64_mul(half_w, slope) : zero;
                    const ulong s = sase_bl89_crossing64(c1, c2, rem, h);
                    if (sase_ge64(s, zero)) {
                        got = soft64_add(acc, s);
                    } else if (sase_ge64(h, zero)
                               && (h & 0x7ffffffffffffffful) != 0ul) {
                        const ulong seg = soft64_mul(
                            soft64_add(c1, soft64_mul(c2, h)), h);
                        rem = sase_max64(soft64_sub(rem, seg), zero);
                    }
                    acc = soft64_add(acc, h);
                    if (sase_ge64(h, zero)) run = soft64_add(run,
                                                         soft64_mul(slope, h));
                    prev_z = node_z;
                }
                if (!sase_ge64(got, zero))
                    got = down != 0u ? z : soft64_sub(htop, z);
                lm[down] = got;
            }
            const ulong l_m = sase_min64(lm[0], lm[1]);
            l_mix_r = sase_min64(l_mix_r, l_m);
            l_eps_r = sase_min64(l_eps_r, l_m);
        }
        ulong c_r = ckv;
        if (sase_ge64(ls_v, zero) && (ls_v & 0x7ffffffffffffffful) != 0ul) {
            const ulong rho = sase_min64(soft64_div(l_mix_r, ls_v), one);
            const ulong rho_pow = soft64_mul(rho, rho); // CKS_BLEND_EXP = 2
            c_r = soft64_add(ckv, soft64_mul(
                soft64_sub(soft64_div(cks, lscoef), ckv), rho_pow));
        }
        const ulong kv_les = soft64_mul(soft64_mul(ckv, l_les), root_e);
        const ulong kv_rans = soft64_mul(soft64_mul(c_r, l_mix_r), root_e);
        const ulong kv_out = soft64_add(soft64_mul(fblend, kv_les),
            soft64_mul(soft64_sub(one, fblend), kv_rans));
        r.kv[q] = sase_f32(kv_out);
        r.leps[q] = sase_f32(l_eps_r);
    }
}

kernel void sase_split_e_update(
    constant SaseSplitEUpdateResources &r [[buffer(0)]],
    constant SaseSplitEUpdateParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup ulong sums[4][128];
    const uint ncell = p.nz * p.ny * p.nx;
    ulong local[4] = {0ul, 0ul, 0ul, 0ul};
    if (tid < ncell) {
        const int i = int(tid % p.nx);
        const int j = int((tid / p.nx) % p.ny);
        const uint k = tid / (p.nx * p.ny);
        const uint plane = p.ny * p.nx;
        const int ip = sase_periodic(i + 1, int(p.nx));
        const int im = sase_periodic(i - 1, int(p.nx));
        const int jp = sase_periodic(j + 1, int(p.ny));
        const int jm = sase_periodic(j - 1, int(p.ny));
        const uint km = k == 0u ? 0u : k - 1u;
        const uint kp = k + 1u == p.nz ? k : k + 1u;
        const uint qm = km * plane + tid % plane;
        const uint qp = kp * plane + tid % plane;
        const float fc = r.theta[tid];
        const float fm = r.theta[qm];
        const float fp = r.theta[qp];
        float dthdz;
        if (p.z_mode == 3) {
            dthdz = r.cm[tid] * fm + r.c0[tid] * fc + r.cp[tid] * fp;
        } else if (k == 0u) {
            dthdz = (fp - fc) / p.dz;
        } else if (k + 1u == p.nz) {
            dthdz = (fc - fm) / (p.z_mode != 0 ? p.h_hi : p.dz);
        } else if (p.z_mode != 0) {
            dthdz = r.cm[k] * fm + r.c0[k] * fc + r.cp[k] * fp;
        } else {
            dthdz = (fp - fm) / p.two_dz;
        }
        const float ec = fmax(r.e[tid], p.e_min);
        const float root_e = sqrt(ec);
        float buoy = -(p.g_accel / fc) * (r.kv[tid] / p.pr_t) * dthdz;
        if (p.has_moist != 0 && r.n2[tid] != r.n2d[tid])
            buoy = -(r.kv[tid] / p.pr_t) * r.n2[tid];
        float ell;
        if (p.f_blend == 0.0f) ell = r.leps[tid];
        else if (p.f_blend == 1.0f) ell = p.delta;
        else ell = pow(p.delta, p.f_blend)
                   * pow(r.leps[tid], 1.0f - p.f_blend);
        float ls_v = 0.0f;
        if (p.has_n2 != 0 && r.n2[tid] > 0.0f) {
            ls_v = p.ls_coef * root_e / sqrt(r.n2[tid]);
            ell = fmin(ell, ls_v);
        }
        const float th = (r.fx[k * plane + uint(j) * p.nx + uint(ip)]
                         - r.fx[k * plane + uint(j) * p.nx + uint(im)])
                         / p.two_dx
                       + (r.fy[k * plane + uint(jp) * p.nx + uint(i)]
                         - r.fy[k * plane + uint(jm) * p.nx + uint(i)])
                         / p.two_dy;
        const float g = p.has_taper != 0 ? r.gtap[tid] : 1.0f;
        const float x_prod = r.ph_e[tid] + r.pv[tid];
        const float src = g * x_prod;
        const float gb = g * buoy;
        const float old = r.e[tid];
        const float e_star = old + p.dt * ((src + gb) + th);
        const ulong es64 = sase_f64(e_star);
        const ulong c_e = sase_f64(p.c_e);
        const ulong f64 = sase_f64(p.f_blend);
        const ulong one64 = 0x3ff0000000000000ul;
        ulong c_eps = c_e;
        ulong rho = 0ul, weight = 0ul;
        if (p.has_ces != 0 && ls_v > 0.0f) {
            rho = sase_min64(soft64_div(sase_f64(ell), sase_f64(ls_v)), one64);
            weight = soft64_mul(rho, rho); // CKS_BLEND_EXP = 2
            const ulong c_es = sase_f64(p.c_es);
            const ulong crn = soft64_add(
                soft64_mul(soft64_sub(one64, weight), c_e),
                soft64_mul(weight, c_es));
            c_eps = soft64_add(soft64_mul(f64, c_e),
                soft64_mul(soft64_sub(one64, f64), crn));
        }
        if (p.has_ced != 0 && ls_v > 0.0f) {
            rho = sase_min64(soft64_div(sase_f64(ell), sase_f64(ls_v)), one64);
            weight = soft64_mul(rho, rho);
            float lref;
            if (p.f_blend == 0.0f) lref = r.lb[tid];
            else if (p.f_blend == 1.0f) lref = p.delta;
            else lref = pow(p.delta, p.f_blend)
                        * pow(r.lb[tid], 1.0f - p.f_blend);
            const ulong added = soft64_mul(
                soft64_mul(soft64_mul(soft64_sub(one64, f64), weight),
                           sase_f64(p.c_ed)),
                soft64_div(sase_f64(ell), sase_f64(lref)));
            c_eps = soft64_add(c_eps, added);
        }
        ulong e_floor64 = sase_f64(p.e_min);
        ulong e_for_root = sase_max64(es64, e_floor64);
        const ulong two64 = 0x4000000000000000ul;
        const ulong b = soft64_div(soft64_mul(c_eps, soft64_sqrt(e_for_root)),
                                   soft64_mul(two64, sase_f64(ell)));
        const ulong decay_fac = soft64_add(one64,
                                  soft64_mul(b, sase_f64(p.dt)));
        const ulong dec_den = soft64_mul(decay_fac, decay_fac);
        const float e_dec = sase_f32(soft64_div(es64, dec_den));
        const float decay = e_star - e_dec;
        const float e_clip = fmax(e_dec, p.e_min);
        const float clip_gain = e_clip - e_dec;
        const float ht = (decay - clip_gain)
                       + p.dt * (r.ph_heat[tid] + (x_prod - src));
        r.e[tid] = e_clip;
        r.heat[tid] = ht;
        local[0] = soft64_sub(sase_f64(e_clip), sase_f64(old));
        local[1] = soft64_mul(sase_f64(p.dt), sase_f64(gb));
        local[2] = soft64_mul(sase_f64(p.dt), sase_f64(th));
        local[3] = sase_f64(ht);
    }
    for (uint c = 0u; c < 4u; ++c) sums[c][lane] = local[c];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (lane < stride)
            for (uint c = 0u; c < 4u; ++c)
                sums[c][lane] = soft64_add(sums[c][lane], sums[c][lane + stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u)
        for (uint c = 0u; c < 4u; ++c)
            r.partials[c * p.nblocks + group] = sums[c][0];
}

kernel void sase_taper_weights(
    constant SaseTaperWeightsResources &r [[buffer(0)]],
    constant SaseTaperWeightsParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    if (col >= ncol || p.nz == 0) return;
    ulong htop = 0ul;
    for (uint k = 0u; k < p.nz; ++k)
        htop = soft64_add(htop, sase_f64(r.t[k * ncol + col]));
    const ulong zd = sase_f64(p.zdamp);
    const ulong one = 0x3ff0000000000000ul;
    const ulong half_w = 0x3fe0000000000000ul;
    const ulong half_pi = sase_f64(1.5707963267948966f);
    ulong zsum = 0ul;
    for (uint k = 0u; k < p.nz; ++k) {
        const uint q = k * ncol + col;
        const ulong tk = sase_f64(r.t[q]);
        const ulong zc = soft64_add(zsum, soft64_mul(half_w, tk));
        zsum = soft64_add(zsum, tk);
        ulong arg = soft64_div(soft64_sub(zc, soft64_sub(htop, zd)), zd);
        if (!sase_ge64(arg, 0ul)) arg = 0ul;
        if (sase_ge64(arg, one)) arg = one;
        const float angle = sase_f32(soft64_mul(half_pi, arg));
        const float sine = sin(angle);
        r.out[q] = 1.0f - sine * sine;
    }
}

// SASE's per-column variable-spacing stencil.  These coefficient rows
// mirror sase_ddz_coefficients in core/kernels/sase.cu: layer values are
// promoted to binary64, the final coefficient alone is rounded to FP32.
kernel void sase_ddz_coefficients(
    constant SaseDdzCoefficientsResources &r [[buffer(0)]],
    constant SaseDdzCoefficientsParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    if (tid >= ncol || p.nz == 0u) return;
    const ulong half_w = 0x3fe0000000000000ul;
    const ulong one = 0x3ff0000000000000ul;
    for (uint k = 0u; k < p.nz; ++k) {
        const uint q = k * ncol + tid;
        const ulong tk = soft64_from_f32_bits(as_type<uint>(r.t[q]));
        ulong cm = 0ul, c0 = 0ul, cp = 0ul;
        if (k == 0u) {
            const ulong tn = soft64_from_f32_bits(as_type<uint>(r.t[q + ncol]));
            const ulong h = soft64_mul(half_w, soft64_add(tk, tn));
            const ulong inv = soft64_div(one, h);
            c0 = soft64_negate_bits(inv);
            cp = inv;
        } else if (k + 1u == p.nz) {
            const ulong tm = soft64_from_f32_bits(as_type<uint>(r.t[q - ncol]));
            const ulong h = soft64_mul(half_w, soft64_add(tm, tk));
            const ulong inv = soft64_div(one, h);
            cm = soft64_negate_bits(inv);
            c0 = inv;
        } else {
            const ulong tm = soft64_from_f32_bits(as_type<uint>(r.t[q - ncol]));
            const ulong tn = soft64_from_f32_bits(as_type<uint>(r.t[q + ncol]));
            const ulong hm = soft64_mul(half_w, soft64_add(tm, tk));
            const ulong hp = soft64_mul(half_w, soft64_add(tk, tn));
            const ulong hsum = soft64_add(hm, hp);
            const ulong cden = soft64_mul(hm, hsum);
            const ulong pden = soft64_mul(hp, hsum);
            cm = soft64_negate_bits(soft64_div(hp, cden));
            c0 = soft64_div(soft64_sub(hp, hm), soft64_mul(hp, hm));
            cp = soft64_div(hm, pden);
        }
        r.cm[q] = as_type<float>(soft64_to_f32_bits(cm));
        r.c0[q] = as_type<float>(soft64_to_f32_bits(c0));
        r.cp[q] = as_type<float>(soft64_to_f32_bits(cp));
    }
}

static inline uint sase_clamp_k(int k, uint nz) {
    return uint(clamp(k, 0, int(nz) - 1));
}

kernel void sase_n2(
    constant SaseN2Resources &r [[buffer(0)]],
    constant SaseN2Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const uint plane = p.ny * p.nx;
    const uint k = tid / plane;
    const uint km = sase_clamp_k(int(k) - 1, p.nz);
    const uint kp = sase_clamp_k(int(k) + 1, p.nz);
    const uint qm = km * plane + tid % plane;
    const uint qp = kp * plane + tid % plane;
    const float fc = r.theta[tid];
    const float fm = r.theta[qm];
    const float fp = r.theta[qp];
    float dthdz;
    if (p.z_mode == 3) {
        dthdz = r.cm[tid] * fm + r.c0[tid] * fc + r.cp[tid] * fp;
    } else if (k == 0u) {
        dthdz = (fp - fc) / p.dz;
    } else if (k + 1u == p.nz) {
        dthdz = (fc - fm) / p.dz;
    } else if (p.z_mode != 0) {
        dthdz = r.cm[k] * fm + r.c0[k] * fc + r.cp[k] * fp;
    } else {
        dthdz = (fp - fm) / p.two_dz;
    }
    r.n2[tid] = (p.g_accel / fc) * dthdz;
}

kernel void sase_strain(
    constant SaseStrainResources &r [[buffer(0)]],
    constant SaseStrainParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const int i = int(tid % p.nx);
    const int j = int((tid / p.nx) % p.ny);
    const uint k = tid / (p.nx * p.ny);
    const uint ip = uint(sase_periodic(i + 1, int(p.nx)));
    const uint im = uint(sase_periodic(i - 1, int(p.nx)));
    const uint jp = uint(sase_periodic(j + 1, int(p.ny)));
    const uint jm = uint(sase_periodic(j - 1, int(p.ny)));
    const uint km = sase_km(k), kp = sase_kp(k, p.nz);
    const uint q = tid;
    const float dudx = (r.u[sase_idx3(k, uint(j), ip, p.ny, p.nx)]
                       - r.u[sase_idx3(k, uint(j), im, p.ny, p.nx)]) / p.two_dx;
    const float dvdx = (r.v[sase_idx3(k, uint(j), ip, p.ny, p.nx)]
                       - r.v[sase_idx3(k, uint(j), im, p.ny, p.nx)]) / p.two_dx;
    const float dwdx = (r.w[sase_idx3(k, uint(j), ip, p.ny, p.nx)]
                       - r.w[sase_idx3(k, uint(j), im, p.ny, p.nx)]) / p.two_dx;
    const float dudy = (r.u[sase_idx3(k, jp, uint(i), p.ny, p.nx)]
                       - r.u[sase_idx3(k, jm, uint(i), p.ny, p.nx)]) / p.two_dy;
    const float dvdy = (r.v[sase_idx3(k, jp, uint(i), p.ny, p.nx)]
                       - r.v[sase_idx3(k, jm, uint(i), p.ny, p.nx)]) / p.two_dy;
    const float dwdy = (r.w[sase_idx3(k, jp, uint(i), p.ny, p.nx)]
                       - r.w[sase_idx3(k, jm, uint(i), p.ny, p.nx)]) / p.two_dy;
    const float dudz = sase_ddz(r.u[sase_idx3(km, uint(j), uint(i), p.ny, p.nx)],
        r.u[q], r.u[sase_idx3(kp, uint(j), uint(i), p.ny, p.nx)], k, q,
        p.nz, r.cm, r.c0, r.cp, p);
    const float dvdz = sase_ddz(r.v[sase_idx3(km, uint(j), uint(i), p.ny, p.nx)],
        r.v[q], r.v[sase_idx3(kp, uint(j), uint(i), p.ny, p.nx)], k, q,
        p.nz, r.cm, r.c0, r.cp, p);
    const float dwdz = sase_ddz(r.w[sase_idx3(km, uint(j), uint(i), p.ny, p.nx)],
        r.w[q], r.w[sase_idx3(kp, uint(j), uint(i), p.ny, p.nx)], k, q,
        p.nz, r.cm, r.c0, r.cp, p);
    r.sxx[q] = dudx;
    r.syy[q] = dvdy;
    r.szz[q] = dwdz;
    r.sxy[q] = 0.5f * (dudy + dvdx);
    r.sxz[q] = 0.5f * (dudz + dwdx);
    r.syz[q] = 0.5f * (dvdz + dwdy);
}

kernel void sase_e_hflux(
    constant SaseEHfluxResources &r [[buffer(0)]],
    constant SaseEHfluxParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const int i = int(tid % p.nx);
    const int j = int((tid / p.nx) % p.ny);
    const uint k = tid / (p.nx * p.ny);
    const uint ip = uint(sase_periodic(i + 1, int(p.nx)));
    const uint im = uint(sase_periodic(i - 1, int(p.nx)));
    const uint jp = uint(sase_periodic(j + 1, int(p.ny)));
    const uint jm = uint(sase_periodic(j - 1, int(p.ny)));
    const float kh = 2.0f * r.km[tid];
    const float dfdx = (fmax(r.e[sase_idx3(k, uint(j), ip, p.ny, p.nx)], p.e_min)
                      - fmax(r.e[sase_idx3(k, uint(j), im, p.ny, p.nx)], p.e_min))
                      / p.two_dx;
    const float dfdy = (fmax(r.e[sase_idx3(k, jp, uint(i), p.ny, p.nx)], p.e_min)
                      - fmax(r.e[sase_idx3(k, jm, uint(i), p.ny, p.nx)], p.e_min))
                      / p.two_dy;
    r.fx[tid] = kh * dfdx;
    r.fy[tid] = kh * dfdy;
}

kernel void sase_scalar_hflux_km(
    constant SaseScalarHfluxKmResources &r [[buffer(0)]],
    constant SaseScalarHfluxKmParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const int i = int(tid % p.nx);
    const int j = int((tid / p.nx) % p.ny);
    const uint k = tid / (p.nx * p.ny);
    const uint ip = uint(sase_periodic(i + 1, int(p.nx)));
    const uint im = uint(sase_periodic(i - 1, int(p.nx)));
    const uint jp = uint(sase_periodic(j + 1, int(p.ny)));
    const uint jm = uint(sase_periodic(j - 1, int(p.ny)));
    const float kh = p.kh_fac * r.km[tid];
    const float dsdx = (r.s[sase_idx3(k, uint(j), ip, p.ny, p.nx)]
                       - r.s[sase_idx3(k, uint(j), im, p.ny, p.nx)]) / p.two_dx;
    const float dsdy = (r.s[sase_idx3(k, jp, uint(i), p.ny, p.nx)]
                       - r.s[sase_idx3(k, jm, uint(i), p.ny, p.nx)]) / p.two_dy;
    r.fx[tid] = kh * dsdx;
    r.fy[tid] = kh * dsdy;
}

kernel void sase_hflux_div(
    constant SaseHfluxDivResources &r [[buffer(0)]],
    constant SaseHfluxDivParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const int i = int(tid % p.nx);
    const int j = int((tid / p.nx) % p.ny);
    const uint k = tid / (p.nx * p.ny);
    const uint ip = uint(sase_periodic(i + 1, int(p.nx)));
    const uint im = uint(sase_periodic(i - 1, int(p.nx)));
    const uint jp = uint(sase_periodic(j + 1, int(p.ny)));
    const uint jm = uint(sase_periodic(j - 1, int(p.ny)));
    r.out[tid] = (r.fx[sase_idx3(k, uint(j), ip, p.ny, p.nx)]
                - r.fx[sase_idx3(k, uint(j), im, p.ny, p.nx)]) / p.two_dx
               + (r.fy[sase_idx3(k, jp, uint(i), p.ny, p.nx)]
               - r.fy[sase_idx3(k, jm, uint(i), p.ny, p.nx)]) / p.two_dy;
}

kernel void sase_model_stress_gov(
    constant SaseModelStressGovResources &r [[buffer(0)]],
    constant SaseModelStressGovParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const float ef = fmax(r.e[tid], p.e_min);
    const float root_e = sqrt(ef);
    const float a = r.s0[tid], b = r.s1[tid], d12 = r.s3[tid];
    const float def_h = sqrt((a - b) * (a - b) + 4.0f * (d12 * d12));
    const float cd = p.c_s * p.delta;
    const float k_smag = fmin(cd * cd * def_h, p.km_cap * p.delta);
    const float nu_eddy = p.c_nu * p.delta * root_e;
    const float nu_smag = (1.0f - p.f_blend) * k_smag;
    const float nu = p.f_blend * nu_eddy + nu_smag;
    const float div3 = as_type<float>(soft32_div(
        as_type<uint>(a + b + r.s2[tid]), as_type<uint>(3.0f)));
    const float iso_visc = 2.0f * nu * div3;
    const float iso_e = (2.0f / 3.0f) * ef;
    const float txx = (-2.0f * nu * a + iso_visc) + iso_e;
    const float tyy = (-2.0f * nu * b + iso_visc) + iso_e;
    const float tzz = (2.0f * ef - txx) - tyy;
    r.t0[tid] = txx;
    r.t1[tid] = tyy;
    r.t2[tid] = tzz;
    r.t3[tid] = -2.0f * nu * d12;
    r.t4[tid] = -2.0f * nu * r.s4[tid];
    r.t5[tid] = -2.0f * nu * r.s5[tid];
    r.km[tid] = nu;
    r.r_out[tid] = nu_smag / fmax(nu, p.nu_eps);
}

kernel void sase_split_tendencies(
    constant SaseSplitTendenciesResources &r [[buffer(0)]],
    constant SaseSplitTendenciesParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const int i = int(tid % p.nx);
    const int j = int((tid / p.nx) % p.ny);
    const uint k = tid / (p.nx * p.ny);
    const uint ip = uint(sase_periodic(i + 1, int(p.nx)));
    const uint im = uint(sase_periodic(i - 1, int(p.nx)));
    const uint jp = uint(sase_periodic(j + 1, int(p.ny)));
    const uint jm = uint(sase_periodic(j - 1, int(p.ny)));
    const uint qxp = sase_idx3(k, uint(j), ip, p.ny, p.nx);
    const uint qxm = sase_idx3(k, uint(j), im, p.ny, p.nx);
    const uint qyp = sase_idx3(k, jp, uint(i), p.ny, p.nx);
    const uint qym = sase_idx3(k, jm, uint(i), p.ny, p.nx);
    const float du = -((r.t0[qxp] - r.t0[qxm]) / p.two_dx
                      + (r.t3[qyp] - r.t3[qym]) / p.two_dy);
    const float dv = -((r.t3[qxp] - r.t3[qxm]) / p.two_dx
                      + (r.t1[qyp] - r.t1[qym]) / p.two_dy);
    const float dw = -((r.t4[qxp] - r.t4[qxm]) / p.two_dx
                      + (r.t5[qyp] - r.t5[qym]) / p.two_dy);
    const float dudx = (r.u[qxp] - r.u[qxm]) / p.two_dx;
    const float dvdy = (r.v[qyp] - r.v[qym]) / p.two_dy;
    const float dudy = (r.u[qyp] - r.u[qym]) / p.two_dy;
    const float dvdx = (r.v[qxp] - r.v[qxm]) / p.two_dx;
    const float dwdx = (r.w[qxp] - r.w[qxm]) / p.two_dx;
    const float dwdy = (r.w[qyp] - r.w[qym]) / p.two_dy;
    const float ph_tot = -(r.t0[tid] * dudx + r.t1[tid] * dvdy
        + r.t3[tid] * (dudy + dvdx) + r.t4[tid] * dwdx
        + r.t5[tid] * dwdy);
    const float phh = r.r[tid] * (ph_tot
        + (2.0f / 3.0f) * fmax(r.e[tid], p.e_min) * (dudx + dvdy));
    r.du[tid] = du;
    r.dv[tid] = dv;
    r.dw[tid] = dw;
    r.ph_heat[tid] = phh;
    r.ph_e[tid] = ph_tot - phh;
}

kernel void sase_velocity_products(
    constant SaseVelocityProductsResources &r [[buffer(0)]],
    constant SaseVelocityProductsParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const float a = r.u[tid], b = r.v[tid], c = r.w[tid];
    r.uu[tid] = a * a;
    r.vv[tid] = b * b;
    r.ww[tid] = c * c;
    r.uv[tid] = a * b;
    r.uw[tid] = a * c;
    r.vw[tid] = b * c;
}

kernel void sase_lift_combine(
    constant SaseLiftCombineResources &r [[buffer(0)]],
    constant SaseLiftCombineParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const float a = r.fu[tid], b = r.fv[tid], c = r.fw[tid];
    r.l0[tid] = r.fuu[tid] - a * a;
    r.l1[tid] = r.fvv[tid] - b * b;
    r.l2[tid] = r.fww[tid] - c * c;
    r.l3[tid] = r.fuv[tid] - a * b;
    r.l4[tid] = r.fuw[tid] - a * c;
    r.l5[tid] = r.fvw[tid] - b * c;
}

kernel void sase_vertical_production(
    constant SaseVerticalProductionResources &r [[buffer(0)]],
    constant SaseVerticalProductionParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const uint plane = p.ny * p.nx;
    const uint k = tid / plane;
    if (p.nz < 2u) { r.pv[tid] = 0.0f; return; }
    const float tk = r.t[tid];
    float eps_lo = 0.0f, eps_hi = 0.0f;
    if (k > 0u) {
        const uint qm = tid - plane;
        const float h = sase_face_spacing(r.t, tid, plane, false);
        const float kf = 0.5f * (r.kv[qm] + r.kv[tid]);
        const float duz = r.u[tid] - r.u[qm];
        const float dvz = r.v[tid] - r.v[qm];
        const float dwz = r.w[tid] - r.w[qm];
        eps_lo = kf * duz * duz / h;
        eps_lo += kf * dvz * dvz / h;
        eps_lo += kf * dwz * dwz / h;
    }
    if (k + 1u < p.nz) {
        const uint qp = tid + plane;
        const float h = sase_face_spacing(r.t, tid, plane, true);
        const float kf = 0.5f * (r.kv[tid] + r.kv[qp]);
        const float duz = r.u[qp] - r.u[tid];
        const float dvz = r.v[qp] - r.v[tid];
        const float dwz = r.w[qp] - r.w[tid];
        eps_hi = kf * duz * duz / h;
        eps_hi += kf * dvz * dvz / h;
        eps_hi += kf * dwz * dwz / h;
    }
    r.pv[tid] = 0.5f * eps_hi / tk + 0.5f * eps_lo / tk;
}

kernel void sase_blackadar_length(
    constant SaseBlackadarLengthResources &r [[buffer(0)]],
    constant SaseBlackadarLengthParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    if (col >= ncol) return;
    const ulong half_w = 0x3fe0000000000000ul;
    const ulong one = 0x3ff0000000000000ul;
    const ulong z0 = r.constants[0], karman = r.constants[1];
    const ulong lambda = r.constants[2];
    ulong csum = 0ul;
    for (uint k = 0u; k < p.nz; ++k) {
        const uint q = k * ncol + col;
        const ulong tk = sase_f64(r.t[q]);
        csum = soft64_add(csum, tk);
        const ulong z = soft64_sub(csum, soft64_mul(half_w, tk));
        const ulong kz = soft64_mul(karman, soft64_add(z, z0));
        const ulong denom = soft64_add(one, soft64_div(kz, lambda));
        r.lb[q] = sase_f32(soft64_div(kz, denom));
    }
}

kernel void sase_basis_premultiply(
    constant SaseBasisPremultiplyResources &r [[buffer(0)]],
    constant SaseBasisPremultiplyParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const float root_e = sqrt(fmax(r.e[tid], p.e_min));
    const float c = -2.0f * p.delta * root_e;
    const float div3 = as_type<float>(soft32_div(
        as_type<uint>(r.s0[tid] + r.s1[tid] + r.s2[tid]),
        as_type<uint>(3.0f)));
    r.p0[tid] = c * (r.s0[tid] - div3);
    r.p1[tid] = c * (r.s1[tid] - div3);
    r.p2[tid] = c * (r.s2[tid] - div3);
    r.p3[tid] = c * r.s3[tid];
    r.p4[tid] = c * r.s4[tid];
    r.p5[tid] = c * r.s5[tid];
}

kernel void sase_solve_partial(
    constant SaseSolvePartialResources &r [[buffer(0)]],
    constant SaseSolvePartialParams &p [[buffer(1)]],
    uint q [[thread_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup ulong sums[5][128];
    const uint ncell = p.nz * p.ny * p.nx;
    ulong m[5] = {0ul, 0ul, 0ul, 0ul, 0ul};
    bool excluded = false;
    if (p.bw > 0 && q < ncell) {
        const uint i = q % p.nx;
        const uint j = (q / p.nx) % p.ny;
        excluded = (i < p.bw || i >= p.nx - p.bw
                    || j < p.bw || j >= p.ny - p.bw);
    }
    if (q < ncell && !excluded) {
        const float root_e = sqrt(fmax(r.e[q], p.e_min));
        const float ca = -2.0f * p.delta_a * root_e;
        const float cb = -2.0f * p.delta_b * root_e;
        const float sc[6] = {r.sc0[q], r.sc1[q], r.sc2[q],
                             r.sc3[q], r.sc4[q], r.sc5[q]};
        const float rf[6] = {r.rf0[q], r.rf1[q], r.rf2[q],
                             r.rf3[q], r.rf4[q], r.rf5[q]};
        const float lv[6] = {r.l0[q], r.l1[q], r.l2[q],
                             r.l3[q], r.l4[q], r.l5[q]};
        const float div3 = as_type<float>(soft32_div(
            as_type<uint>(sc[0] + sc[1] + sc[2]), as_type<uint>(3.0f)));
        const float trl = as_type<float>(soft32_div(
            as_type<uint>(lv[0] + lv[1] + lv[2]), as_type<uint>(3.0f)));
        for (uint k = 0u; k < 6u; ++k) {
            const float scd = k < 3u ? sc[k] - div3 : sc[k];
            const float av = ca * scd - rf[k];
            const float bv = cb * scd - rf[k];
            const float rv = k < 3u ? lv[k] - trl : lv[k];
            m[0] = soft64_add(m[0], soft64_mul(sase_f64(av), sase_f64(av)));
            m[1] = soft64_add(m[1], soft64_mul(sase_f64(av), sase_f64(bv)));
            m[2] = soft64_add(m[2], soft64_mul(sase_f64(bv), sase_f64(bv)));
            m[3] = soft64_add(m[3], soft64_mul(sase_f64(av), sase_f64(rv)));
            m[4] = soft64_add(m[4], soft64_mul(sase_f64(bv), sase_f64(rv)));
        }
    }
    for (uint t = 0u; t < 5u; ++t) sums[t][lane] = m[t];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 64u; stride > 0u; stride >>= 1u) {
        if (lane < stride) {
            for (uint t = 0u; t < 5u; ++t)
                sums[t][lane] = soft64_add(sums[t][lane], sums[t][lane + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) {
        for (uint t = 0u; t < 5u; ++t)
            r.partials[t * p.nblocks + group] = sums[t][0];
    }
}

// Moist effective-stability companion to sase_n2.  The saturation
// thermodynamics and variable-height derivatives use the same integer
// binary64 arithmetic as the authority path; dry cells copy n2 verbatim.
kernel void sase_moist_n2(
    constant SaseMoistN2Resources &r [[buffer(0)]],
    constant SaseMoistN2Params &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    if (col >= ncol || p.nz > 128) return;
    constexpr ulong one = 0x3ff0000000000000ul;
    constexpr ulong thousand = 0x408f400000000000ul;
    ulong z[128], th[128], temp[128];
    ulong qs[128], qt[128];
    ulong zrun = 0ul;
    const ulong kappa = r.constants[0];
    const ulong p0 = r.constants[1];
    const ulong svp1 = r.constants[2];
    const ulong svp2 = r.constants[3];
    const ulong svp3 = r.constants[4];
    const ulong svpt0 = r.constants[5];
    const ulong ep2 = r.constants[6];
    const ulong xlv = r.constants[7];
    const ulong rd = r.constants[8];
    const ulong cp_air = r.constants[9];
    const ulong gravity = r.constants[10];
    const ulong half_bits = 0x3fe0000000000000ul;
    for (uint k = 0u; k < p.nz; ++k) {
        const uint q = k * ncol + col;
        const ulong tk = sase_f64(r.t[q]);
        zrun = soft64_add(zrun, tk);
        z[k] = soft64_sub(zrun, soft64_mul(half_bits, tk));
        th[k] = sase_f64(r.theta[q]);
        const ulong pressure = sase_f64(r.pressure[q]);
        const ulong exner = soft64_plm_pow(soft64_div(pressure, p0), kappa);
        temp[k] = soft64_mul(th[k], exner);
        const ulong exponent = soft64_div(
            soft64_mul(svp2, soft64_sub(temp[k], svpt0)),
            soft64_sub(temp[k], svp3));
        const ulong es = soft64_mul(soft64_mul(thousand, svp1),
                                    soft64_plm_exp(exponent));
        qs[k] = soft64_div(soft64_mul(ep2, es),
                           soft64_sub(pressure, es));
        qt[k] = soft64_add(sase_f64(r.qv[q]), sase_f64(r.qc[q]));
    }
    for (uint k = 0u; k < p.nz; ++k) {
        const uint q = k * ncol + col;
        const ulong qc = sase_f64(r.qc[q]);
        const ulong qv = sase_f64(r.qv[q]);
        const bool qc_positive = (qc != 0ul)
            && ((qc & 0x8000000000000000ul) == 0ul);
        if (!qc_positive && !sase_ge64(qv, qs[k])) {
            r.out[q] = r.n2dry[q];
            continue;
        }
        ulong dth = 0ul, dqs = 0ul, dqw = 0ul;
        if (p.nz > 1u) {
            if (k == 0u) {
                const ulong h = soft64_sub(z[1], z[0]);
                dth = soft64_div(soft64_sub(th[1], th[0]), h);
                dqs = soft64_div(soft64_sub(qs[1], qs[0]), h);
                dqw = soft64_div(soft64_sub(qt[1], qt[0]), h);
            } else if (k + 1u == p.nz) {
                const ulong h = soft64_sub(z[k], z[k - 1u]);
                dth = soft64_div(soft64_sub(th[k], th[k - 1u]), h);
                dqs = soft64_div(soft64_sub(qs[k], qs[k - 1u]), h);
                dqw = soft64_div(soft64_sub(qt[k], qt[k - 1u]), h);
            } else {
                const ulong hm = soft64_sub(z[k], z[k - 1u]);
                const ulong hp = soft64_sub(z[k + 1u], z[k]);
                const ulong cm = soft64_negate_bits(soft64_div(
                    hp, soft64_mul(hm, soft64_add(hm, hp))));
                const ulong c0 = soft64_div(soft64_sub(hp, hm),
                                            soft64_mul(hm, hp));
                const ulong cp = soft64_div(hm,
                    soft64_mul(hp, soft64_add(hm, hp)));
                dth = soft64_add(soft64_add(soft64_mul(cm, th[k - 1u]),
                    soft64_mul(c0, th[k])), soft64_mul(cp, th[k + 1u]));
                dqs = soft64_add(soft64_add(soft64_mul(cm, qs[k - 1u]),
                    soft64_mul(c0, qs[k])), soft64_mul(cp, qs[k + 1u]));
                dqw = soft64_add(soft64_add(soft64_mul(cm, qt[k - 1u]),
                    soft64_mul(c0, qt[k])), soft64_mul(cp, qt[k + 1u]));
            }
        }
        const ulong a = soft64_add(one, soft64_div(
            soft64_mul(xlv, qs[k]), soft64_mul(rd, temp[k])));
        const ulong b = soft64_add(one, soft64_div(
            soft64_mul(soft64_mul(ep2, soft64_mul(xlv, xlv)), qs[k]),
            soft64_mul(soft64_mul(cp_air, rd),
                       soft64_mul(temp[k], temp[k]))));
        const ulong bracket = soft64_sub(
            soft64_add(soft64_div(dth, th[k]),
                       soft64_mul(soft64_div(xlv,
                           soft64_mul(cp_air, temp[k])), dqs)), dqw);
        const ulong value = soft64_mul(gravity,
            soft64_mul(soft64_div(a, b), bracket));
        r.out[q] = sase_f32(value);
    }
}

static inline ulong sase_vent_es(ulong temp, ulong svp1, ulong svp2,
                                  ulong svp3, ulong svpt0) {
    const ulong exponent = soft64_div(
        soft64_mul(svp2, soft64_sub(temp, svpt0)),
        soft64_sub(temp, svp3));
    return soft64_mul(soft64_mul(0x408f400000000000ul, svp1),
                      soft64_plm_exp(exponent));
}

static inline ulong sase_vent_qs(ulong temp, ulong pressure, ulong kappa,
                                  ulong p0, ulong svp1, ulong svp2,
                                  ulong svp3, ulong svpt0, ulong ep2) {
    const ulong exner = soft64_plm_pow(soft64_div(pressure, p0), kappa);
    const ulong t = soft64_mul(temp, exner);
    const ulong es = sase_vent_es(t, svp1, svp2, svp3, svpt0);
    return soft64_div(soft64_mul(ep2, es), soft64_sub(pressure, es));
}

static inline void sase_vent_sat_adjust(
    ulong thl, ulong qt, ulong pressure, int iters, ulong kappa, ulong p0,
    ulong svp1, ulong svp2, ulong svp3, ulong svpt0, ulong ep2,
    ulong xlv_cp, thread ulong *theta_out, thread ulong *qv_out,
    thread ulong *qc_out) {
    const ulong one = 0x3ff0000000000000ul;
    const ulong exner = soft64_plm_pow(soft64_div(pressure, p0), kappa);
    ulong temp = soft64_mul(exner, thl);
    for (int it = 0; it < iters; ++it) {
        const ulong es = sase_vent_es(temp, svp1, svp2, svp3, svpt0);
        const ulong qs = soft64_div(soft64_mul(ep2, es),
                                    soft64_sub(pressure, es));
        const ulong qc = sase_max64(soft64_sub(qt, qs), 0ul);
        const ulong residual = soft64_sub(
            soft64_sub(temp, soft64_mul(exner, thl)),
            soft64_mul(xlv_cp, qc));
        const ulong den = soft64_sub(temp, svp3);
        const ulong des = soft64_div(
            soft64_mul(soft64_mul(es, svp2), soft64_sub(svpt0, svp3)),
            soft64_mul(den, den));
        const ulong dqs = soft64_div(
            soft64_mul(soft64_mul(ep2, pressure), des),
            soft64_mul(soft64_sub(pressure, es),
                       soft64_sub(pressure, es)));
        const ulong slope = sase_ge64(qc, 0x0000000000000001ul)
            ? soft64_add(one, soft64_mul(xlv_cp, dqs)) : one;
        temp = soft64_sub(temp, soft64_div(residual, slope));
    }
    const ulong es = sase_vent_es(temp, svp1, svp2, svp3, svpt0);
    const ulong qs = soft64_div(soft64_mul(ep2, es),
                                soft64_sub(pressure, es));
    *qc_out = sase_max64(soft64_sub(qt, qs), 0ul);
    *theta_out = soft64_div(temp, exner);
    *qv_out = soft64_sub(qt, *qc_out);
}

kernel void sase_plume_vent_flux(
    constant SasePlumeVentFluxResources &r [[buffer(0)]],
    constant SasePlumeVentFluxParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    if (col >= ncol || p.nz > 128) return;
    const ulong zero = 0ul, one = 0x3ff0000000000000ul;
    const ulong f = r.constants[0], kappa = r.constants[1];
    const ulong p0 = r.constants[2], svp1 = r.constants[3];
    const ulong svp2 = r.constants[4], svp3 = r.constants[5];
    const ulong svpt0 = r.constants[6], ep2 = r.constants[7];
    const ulong xlv_cp = r.constants[8], xlv = r.constants[9];
    const ulong cp_air = r.constants[10], rvm1 = r.constants[11];
    const ulong gravity = r.constants[12], e_min = r.constants[13];
    const ulong mb_coef = r.constants[14], ent_coef = r.constants[15];
    const ulong sigw_share = r.constants[16], depth_cap = r.constants[17];
    ulong thes[128], buoy[128], dth[128], dqv[128], dqc[128], shape[128];
    bool member[128];
    for (uint face = 0u; face <= p.nz; ++face) {
        const uint qf = face * ncol + col;
        r.fth[qf] = 0.0f; r.fqv[qf] = 0.0f; r.fqc[qf] = 0.0f;
    }
    if (p.has_idx) for (uint row = 0u; row < 7u; ++row)
        r.idx[row * ncol + col] = -1;
    ulong zrun = 0ul;
    for (uint k = 0u; k < p.nz; ++k) {
        const uint q = k * ncol + col;
        const ulong th = sase_f64(r.theta[q]);
        const ulong pressure = sase_f64(r.pressure[q]);
        const ulong temp = soft64_mul(th,
            soft64_plm_pow(soft64_div(pressure, p0), kappa));
        const ulong qs = soft64_div(soft64_mul(ep2,
            sase_vent_es(temp, svp1, svp2, svp3, svpt0)),
            soft64_sub(pressure, sase_vent_es(temp, svp1, svp2, svp3, svpt0)));
        thes[k] = soft64_mul(th, soft64_plm_exp(
            soft64_div(soft64_mul(xlv, qs), soft64_mul(cp_air, temp))));
        const ulong qt = soft64_add(sase_f64(r.qv[q]), sase_f64(r.qc[q]));
        const bool m1 = r.n2m[q] != r.n2d[q];
        member[k] = m1 && sase_ge64(qt, qs);
        const ulong tk = sase_f64(r.t[q]);
        zrun = soft64_add(zrun, tk);
    }
    bool in_run = false, mono = false, chosen = false;
    int run_s = 0, k_base = -1, k_top = -1;
    ulong base_thes = zero, prev_thes = zero;
    for (uint k = 0u; k < p.nz; ++k) {
        const bool mk = member[k];
        const bool start = mk && !in_run;
        const bool cont = mk && in_run;
        if (start) { run_s = int(k); base_thes = thes[k]; mono = true; }
        else if (cont) mono = mono && !sase_ge64(thes[k], prev_thes);
        if (mk) prev_thes = thes[k];
        const bool end_here = k + 1u < p.nz ? (mk && !member[k + 1u]) : mk;
        const bool long_enough = int(k) - run_s >= p.min_run_cells - 1;
        const bool reading = p.per_level ? mono : !sase_ge64(thes[k], base_thes);
        const bool take = end_here && long_enough && reading && !chosen;
        if (take) { k_base = run_s; k_top = int(k); }
        chosen = chosen || take; in_run = mk;
    }
    if (!chosen || k_base == 0) return;
    int k_r = k_base;
    const int k_floor = k_base - (k_top - k_base) - 1;
    for (int k = 1; k < int(p.nz) - 1; ++k)
        if (sase_ge64(thes[k], thes[k - 1]) && sase_ge64(thes[k], thes[k + 1])
            && thes[k] != thes[k - 1] && thes[k] != thes[k + 1]
            && k <= k_base && k >= k_floor) k_r = k;
    const int k_hi = min(k_top + 1, int(p.nz) - 1);
    ulong e_num = zero, e_den = zero;
    for (int k = k_base; k <= k_hi; ++k) {
        const uint q = uint(k) * ncol + col;
        const ulong tk = sase_f64(r.t[q]);
        e_num = soft64_add(e_num, soft64_mul(tk,
            sase_max64(sase_f64(r.e[q]), e_min)));
        e_den = soft64_add(e_den, tk);
    }
    const ulong ebar = sase_max64(soft64_div(e_num, e_den), e_min);
    const ulong sigma = soft64_sqrt(soft64_mul(sigw_share, ebar));
    const ulong rho = sase_f64(r.rho1[col]);
    const ulong mbase = soft64_mul(mb_coef, soft64_mul(rho, sigma));
    const ulong mused = soft64_mul(soft64_sub(one, f), mbase);
    ulong thlp = zero, qtp = zero, thprev = zero, qtprev = zero, zprev = zero;
    ulong zc[128];
    bool started = false;
    zrun = zero;
    for (uint k = 0u; k < p.nz; ++k) {
        const uint q = k * ncol + col;
        const ulong tk = sase_f64(r.t[q]);
        zrun = soft64_add(zrun, tk);
        zc[k] = soft64_sub(zrun, soft64_mul(0x3fe0000000000000ul, tk));
        const ulong th = sase_f64(r.theta[q]), qv = sase_f64(r.qv[q]);
        const ulong qc = sase_f64(r.qc[q]), pressure = sase_f64(r.pressure[q]);
        const ulong temp = soft64_mul(th,
            soft64_plm_pow(soft64_div(pressure, p0), kappa));
        const ulong qt_env = soft64_add(qv, qc);
        const ulong thl_env = soft64_sub(th, soft64_div(
            soft64_mul(soft64_mul(xlv_cp, qc), th), temp));
        ulong b = zero; dth[k] = zero; dqv[k] = zero; dqc[k] = zero;
        if (k > 0u && started) {
            const ulong thlf = soft64_mul(0x3fe0000000000000ul,
                soft64_add(thprev, thl_env));
            const ulong qtf = soft64_mul(0x3fe0000000000000ul,
                soft64_add(qtprev, qt_env));
            const ulong fac = soft64_plm_pow(soft64_div(zprev, zc[k]), ent_coef);
            const ulong thln = soft64_add(thlf,
                soft64_mul(soft64_sub(thlp, thlf), fac));
            const ulong qtn = soft64_add(qtf,
                soft64_mul(soft64_sub(qtp, qtf), fac));
            ulong thp, qvp, qcp;
            sase_vent_sat_adjust(thln, qtn, pressure, p.sat_iters, kappa, p0,
                svp1, svp2, svp3, svpt0, ep2, xlv_cp,
                &thp, &qvp, &qcp);
            const ulong tvp = soft64_mul(thp,
                soft64_add(one, soft64_sub(soft64_mul(rvm1, qvp), qcp)));
            const ulong tve = soft64_mul(th,
                soft64_add(one, soft64_sub(soft64_mul(rvm1, qv), qc)));
            b = soft64_div(soft64_mul(gravity, soft64_sub(tvp, tve)), tve);
            dth[k] = soft64_sub(thp, th);
            dqv[k] = soft64_sub(qvp, qv);
            dqc[k] = soft64_sub(qcp, qc);
            thlp = thln; qtp = qtn;
        }
        if (k_r == int(k)) { thlp = thl_env; qtp = qt_env; started = true; }
        buoy[k] = b; thprev = thl_env; qtprev = qt_env; zprev = zc[k];
    }
    const int k_lid = k_top + 2;
    ulong zr = zero; zrun = zero;
    for (int k = 0; k <= k_r; ++k) {
        const uint q = uint(k) * ncol + col;
        const ulong tk = sase_f64(r.t[q]); zrun = soft64_add(zrun, tk);
        zr = soft64_sub(zrun, soft64_mul(0x3fe0000000000000ul, tk));
    }
    bool lfc_found = false, nb_found = false;
    int k_nb = -1, kb = -1, k_lfc = -1;
    ulong bmax = zero; zrun = zero;
    for (uint k = 0u; k < p.nz; ++k) {
        const uint q = k * ncol + col;
        const ulong tk = sase_f64(r.t[q]); zrun = soft64_add(zrun, tk);
        const ulong zk = soft64_sub(zrun, soft64_mul(0x3fe0000000000000ul, tk));
        const bool above = int(k) > k_r;
        const bool incap = above && int(k) < k_lid
            && !sase_ge64(soft64_sub(zk, zr), depth_cap);
        const bool pos = sase_ge64(buoy[k], 0x0000000000000001ul);
        const bool upd_lfc = incap && !lfc_found && pos;
        if (upd_lfc) k_lfc = int(k);
        lfc_found = lfc_found || upd_lfc;
        const bool cand = incap && lfc_found && !nb_found && pos;
        if (cand && sase_ge64(buoy[k], bmax)) { bmax = buoy[k]; kb = int(k); }
        const bool upd_nb = incap && lfc_found && !nb_found && !pos;
        if (upd_nb) k_nb = int(k);
        nb_found = nb_found || upd_nb;
        const bool at_lid = above && int(k) == k_lid && lfc_found && !nb_found
            && !sase_ge64(soft64_sub(zk, zr), depth_cap);
        if (at_lid) k_nb = int(k);
        nb_found = nb_found || at_lid;
    }
    if (!nb_found || kb < 0) return;
    if (p.has_idx) {
        const int ix[7] = {k_base, k_top, k_r, k_lid, k_lfc, k_nb, kb};
        for (uint row = 0u; row < 7u; ++row)
            r.idx[row * ncol + col] = ix[row];
    }
    ulong accum = zero;
    for (int k = int(p.nz) - 1; k >= 0; --k) {
        ulong v = zero;
        if (k > kb && k < k_nb) {
            const uint q = uint(k) * ncol + col;
            v = soft64_mul(buoy[k], sase_f64(r.t[q]));
        }
        accum = soft64_add(accum, v); shape[k] = accum;
    }
    const int ja = max(k_base, 1);
    ulong zfa = zero, zfpk = zero; zrun = zero;
    for (uint face = 0u; face <= p.nz; ++face) {
        if (int(face) == ja) zfa = zrun;
        if (int(face) == kb + 1) zfpk = zrun;
        if (face < p.nz) zrun = soft64_add(zrun,
            sase_f64(r.t[face * ncol + col]));
    }
    const ulong den = shape[kb + 1];
    const ulong mhpk = soft64_plm_pow(soft64_div(zfpk, zfa), ent_coef);
    zrun = sase_f64(r.t[col]);
    for (uint face = 1u; face < p.nz; ++face) {
        const bool grow = int(face) >= k_r + 1 && int(face) <= kb + 1
                          && int(face) < k_nb;
        const bool tap = int(face) > kb + 1 && int(face) < k_nb
                         && den != zero;
        const ulong g = soft64_plm_pow(soft64_div(zrun, zfa), ent_coef);
        const ulong rr = soft64_div(shape[face], den == zero ? one : den);
        const ulong mh = grow ? g : (tap ? soft64_mul(mhpk, rr) : zero);
        if (sase_ge64(mh, 0x0000000000000001ul)
            && sase_ge64(mused, 0x0000000000000001ul)) {
            const uint q = face * ncol + col;
            const ulong mass = soft64_mul(mused, mh);
            r.fth[q] = sase_f32(soft64_mul(mass,
                soft64_mul(0x3fe0000000000000ul,
                           soft64_add(dth[face - 1u], dth[face]))));
            r.fqv[q] = sase_f32(soft64_mul(mass,
                soft64_mul(0x3fe0000000000000ul,
                           soft64_add(dqv[face - 1u], dqv[face]))));
            r.fqc[q] = sase_f32(soft64_mul(mass,
                soft64_mul(0x3fe0000000000000ul,
                           soft64_add(dqc[face - 1u], dqc[face]))));
        }
        zrun = soft64_add(zrun, sase_f64(r.t[face * ncol + col]));
    }
}

kernel void sase_vent_deposit_scale(
    constant SaseVentDepositScaleResources &r [[buffer(0)]],
    constant SaseVentDepositScaleParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    if (col >= ncol) return;
    const ulong dt = r.constants[0], theta_cap = r.constants[1];
    const ulong qt_cap = r.constants[2];
    const ulong rho = sase_f64(r.rho1[col]);
    ulong dmax[3] = {0ul, 0ul, 0ul};
    for (uint k = 0u; k < p.nz; ++k) {
        const ulong tk = sase_f64(r.t[k * ncol + col]);
        const ulong rho_t = soft64_mul(rho, tk);
        const ulong factor = soft64_div(dt, rho_t);
        const uint q = k * ncol + col;
        const ulong df = soft64_mul(soft64_sub(sase_f64(r.fth[q]),
            sase_f64(r.fth[q + ncol])), factor);
        const ulong dqv = soft64_mul(soft64_sub(sase_f64(r.fqv[q]),
            sase_f64(r.fqv[q + ncol])), factor);
        const ulong dqc = soft64_mul(soft64_sub(sase_f64(r.fqc[q]),
            sase_f64(r.fqc[q + ncol])), factor);
        dmax[0] = sase_max64(dmax[0], soft64_abs_bits(df));
        dmax[1] = sase_max64(dmax[1], soft64_abs_bits(dqv));
        dmax[2] = sase_max64(dmax[2], soft64_abs_bits(dqc));
    }
    ulong scale = 0x3ff0000000000000ul;
    if (dmax[0] != 0ul) scale = sase_min64(scale, soft64_div(theta_cap, dmax[0]));
    if (dmax[1] != 0ul) scale = sase_min64(scale, soft64_div(qt_cap, dmax[1]));
    if (dmax[2] != 0ul) scale = sase_min64(scale, soft64_div(qt_cap, dmax[2]));
    r.scale[col] = scale;
}

kernel void sase_vent_deposit(
    constant SaseVentDepositResources &r [[buffer(0)]],
    constant SaseVentDepositParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncell = p.nz * p.ny * p.nx;
    if (tid >= ncell) return;
    const uint col = tid % (p.ny * p.nx);
    const ulong dt = r.constants[0];
    const ulong fac = soft64_div(dt,
        soft64_mul(sase_f64(r.rho1[col]), sase_f64(r.t[tid])));
    const ulong scale = r.scale[col];
    const ulong f0 = soft64_mul(scale, sase_f64(r.f[tid]));
    const ulong f1 = soft64_mul(scale, sase_f64(r.f[tid + p.ny * p.nx]));
    const ulong deposit = soft64_mul(soft64_sub(f0, f1), fac);
    r.phi[tid] = sase_f32(soft64_add(sase_f64(r.phi[tid]), deposit));
}

kernel void sase_couple_w_tendency(
    constant SaseCoupleWTendencyResources &r [[buffer(0)]],
    constant SaseCoupleWTendencyParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint ncol = p.ny * p.nx;
    const uint nface = p.nz + 1;
    const uint n = nface * ncol;
    if (tid >= n) return;
    const uint face = tid / ncol;
    const uint col = tid % ncol;
    float full = 0.0f;
    if (face > 0u && face < p.nz) {
        const uint lo = (face - 1u) * ncol + col;
        const uint hi = face * ncol + col;
        full = 0.5f * (r.dw[lo] + r.dw[hi]);
    }
    const uint j = col / p.nx, i = col % p.nx;
    const bool ring = p.mask_ring != 0
        && (j == 0u || j + 1u == p.ny || i == 0u || i + 1u == p.nx);
    const float chf = r.c1f[face] * r.mut[col] + r.c2f[face];
    const float msft = p.has_msf != 0 ? r.msft[col] : 1.0f;
    r.rw[tid] = ring ? 0.0f : (chf * full) / msft;
}
