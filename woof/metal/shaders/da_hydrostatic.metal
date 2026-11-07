// Source-ordered prepared DA hydrostatic integration.
#include <metal_stdlib>
#include "da_hydrostatic_bindings.metalh"
#include "soft_libm64.metalh"
using namespace metal;

static inline ulong hydro_alt(ulong theta, ulong qv, ulong p, device const ulong *c) {
    ulong v = soft64_mul(c[0], theta);
    v = soft64_mul(v, soft64_add(0x3ff0000000000000ul, soft64_mul(c[1], qv)));
    v = soft64_mul(v, soft64_plm_pow(soft64_div(p, c[2]), c[3]));
    return soft64_div(v, p);
}

static inline ulong hydro_operator(uint k, ulong mu, uint option,
    device const ulong *c1h, device const ulong *c2h,
    device const ulong *c3f, device const ulong *c4f,
    device const ulong *c3h, device const ulong *c4h,
    device const ulong *dnw, device const ulong *c) {
    if (option == 1) {
        return soft64_mul(soft64_negate_bits(dnw[k]),
            soft64_add(soft64_mul(c1h[k], mu), c2h[k]));
    }
    ulong up = soft64_add(soft64_add(soft64_mul(c3f[k + 1], mu), c4f[k + 1]), c[4]);
    ulong down = soft64_add(soft64_add(soft64_mul(c3f[k], mu), c4f[k]), c[4]);
    ulong mid = soft64_add(soft64_add(soft64_mul(c3h[k], mu), c4h[k]), c[4]);
    return soft64_mul(mid, soft64_plm_log(soft64_div(down, up)));
}

kernel void da_hydrostatic_column(
    constant DaHydrostaticColumnResources &r [[buffer(0)]],
    constant DaHydrostaticColumnParams &params [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint levels = uint(params.levels), columns = uint(params.columns);
    const uint option = uint(params.option), mass = uint(params.mass);
    if (col >= columns) return;
    uint plane = levels * columns, stride = levels + 1;
    device const ulong *p = r.field3;
    device const ulong *thpb = r.field3 + plane;
    device const ulong *thpa = r.field3 + 2 * plane;
    device const ulong *qvb = r.field3 + 3 * plane;
    device const ulong *qva = r.field3 + 4 * plane;
    device const ulong *thb = r.field3 + 5 * plane;
    ulong mub = r.mass2[col], mupb = r.mass2[columns + col], mupa = r.mass2[2 * columns + col];
    device const ulong *c1f = r.profiles;
    device const ulong *c2f = r.profiles + stride;
    device const ulong *c1h = r.profiles + 2 * stride;
    device const ulong *c2h = r.profiles + 3 * stride;
    device const ulong *c3f = r.profiles + 4 * stride;
    device const ulong *c4f = r.profiles + 5 * stride;
    device const ulong *c3h = r.profiles + 6 * stride;
    device const ulong *c4h = r.profiles + 7 * stride;
    device const ulong *rdnw = r.profiles + 8 * stride;
    device const ulong *rdn = r.profiles + 9 * stride;
    device const ulong *dnw = r.profiles + 10 * stride;
    device const ulong *c = r.constants;
    const ulong one = 0x3ff0000000000000ul, positive_half = 0x3fe0000000000000ul;
    const ulong neg_half = 0xbfe0000000000000ul;
    uint top = levels - 1;
    r.status[col] = 0;
    // Rust binary starts at +0, rounds and checks after every species.
    for (uint a = 0; a < 2; ++a) {
        device ulong *q = a ? r.pressure : r.scratch;
        for (uint k = 0; k < levels; ++k) {
            uint i = k * columns + col;
            ulong total = soft64_add(0ul, (a ? qva : qvb)[i]);
            if (!soft64_is_finite(total)) { r.status[col] = 2; return; }
            for (uint species = 1; species < (uint(params.field_count) - 4) / 2; ++species) {
                total = soft64_add(total, r.field3[(6 + 2 * (species - 1) + a) * plane + i]);
                if (!soft64_is_finite(total)) { r.status[col] = 2; return; }
            }
            q[i] = total;
        }
    }
    for (uint a = 0; a < 2; ++a) {
        device ulong *out = a ? r.pressure : r.scratch;
        ulong mup = a ? mupa : mupb;
        ulong qt = out[top * columns + col];
        ulong cq = soft64_div(one, soft64_add(one, qt));
        ulong load = soft64_mul(qt, cq);
        ulong base = soft64_add(soft64_mul(c1f[levels], mub), c2f[levels]);
        ulong value = soft64_add(soft64_mul(c1f[levels], mup), soft64_mul(load, base));
        out[top * columns + col] = soft64_div(soft64_div(soft64_mul(neg_half, value), rdnw[top]), cq);
        for (uint k = top; k > 0; --k) {
            uint lower = k - 1;
            ulong qlower = out[lower * columns + col];
            ulong qbar = soft64_mul(positive_half, soft64_add(qlower, qt));
            qt = qlower;
            cq = soft64_div(one, soft64_add(one, qbar));
            load = soft64_mul(qbar, cq);
            base = soft64_add(soft64_mul(c1f[k], mub), c2f[k]);
            value = soft64_add(soft64_mul(c1f[k], mup), soft64_mul(load, base));
            out[lower * columns + col] = soft64_sub(out[k * columns + col], soft64_div(soft64_div(value, cq), rdn[k]));
        }
    }
    r.phi[col] = 0;
    ulong integrated = 0;
    for (uint k = 0; k < levels; ++k) {
        uint i = k * columns + col;
        ulong dp = soft64_sub(r.pressure[i], r.scratch[i]);
        r.pressure[i] = dp;
        ulong analysed = soft64_add(p[i], dp);
        if (!soft64_is_finite(analysed) || soft64_less_equal(analysed, 0ul)) {
            r.status[col] = 1; return;
        }
        ulong aa = hydro_alt(soft64_add(thb[i], thpa[i]), qva[i], analysed, c);
        ulong ab = hydro_alt(soft64_add(thb[i], thpb[i]), qvb[i], p[i], c);
        ulong before_mu = soft64_add(mub, mupb);
        ulong opb = hydro_operator(k, before_mu, option, c1h, c2h, c3f, c4f, c3h, c4h, dnw, c);
        ulong dd;
        if (mass == 1) {
            ulong after_mu = soft64_add(mub, mupa);
            ulong opa = hydro_operator(k, after_mu, option, c1h, c2h, c3f, c4f, c3h, c4h, dnw, c);
            dd = soft64_sub(soft64_mul(opa, aa), soft64_mul(opb, ab));
        } else {
            dd = soft64_mul(opb, soft64_sub(aa, ab));
        }
        integrated = k == 0 ? dd : soft64_add(integrated, dd);
        if (!soft64_is_finite(integrated)) { r.status[col] = 1; return; }
        r.phi[(k + 1) * columns + col] = integrated;
    }
}

kernel void da_hydrostatic_apply(
    constant DaHydrostaticApplyResources &r [[buffer(0)]],
    constant DaHydrostaticApplyParams &params [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const uint columns = uint(params.columns);
    if (col >= columns) return;
    for (uint k = 0; k <= uint(params.levels); ++k) {
        uint dst = k * uint(params.total_columns) + uint(params.start) + col;
        ulong delta = r.phi[k * columns + col];
        uint original = r.php[dst];
        r.result[dst] = (delta & 0x7ffffffffffffffful) == 0 ? original :
            soft64_to_f32_bits(soft64_add(soft64_from_f32_bits(original), delta));
    }
}
