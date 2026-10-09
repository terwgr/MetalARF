#include <metal_stdlib>
#include "urban_bep_radiosity_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubrad_add(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubrad_sub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubrad_mul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ubrad_div(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }

// Source-only port of module_sf_bep.F bep_long_rad/bep_short_rad and
// bep_gaussj (2006-2205). Needs integration: dispatch once per urban column
// and street direction after urban_bep_radiation_fragment. Fixed shapes use
// BEP's nz_um=18; wall-pair and ground view factors are packed per column.
static inline bool ubrad_gaussj(thread float *a, thread float *b, int n) {
    int ipiv[37];
    for (int j = 0; j < n; ++j) ipiv[j] = 0;
    for (int i = 0; i < n; ++i) {
        float big = 0.0f;
        int irow = -1, icol = -1;
        for (int j = 0; j < n; ++j) {
            if (ipiv[j] != 1) {
                for (int k = 0; k < n; ++k) {
                    if (ipiv[k] == 0) {
                        const float v = fabs(a[j * 37 + k]);
                        if (v >= big) { big = v; irow = j; icol = k; }
                    } else if (ipiv[k] > 1) return false;
                }
            }
        }
        if (irow < 0 || icol < 0) return false;
        ++ipiv[icol];
        if (irow != icol) {
            for (int l = 0; l < n; ++l) {
                const float t = a[irow * 37 + l];
                a[irow * 37 + l] = a[icol * 37 + l];
                a[icol * 37 + l] = t;
            }
            const float t = b[irow]; b[irow] = b[icol]; b[icol] = t;
        }
        const float pivot = a[icol * 37 + icol];
        if (pivot == 0.0f) return false;
        const float pivinv = ubrad_div(1.0f, pivot);
        a[icol * 37 + icol] = 1.0f;
        for (int l = 0; l < n; ++l)
            a[icol * 37 + l] = ubrad_mul(a[icol * 37 + l], pivinv);
        b[icol] = ubrad_mul(b[icol], pivinv);
        for (int ll = 0; ll < n; ++ll) if (ll != icol) {
            const float dum = a[ll * 37 + icol];
            a[ll * 37 + icol] = 0.0f;
            for (int l = 0; l < n; ++l)
                a[ll * 37 + l] = ubrad_sub(a[ll * 37 + l], ubrad_mul(a[icol * 37 + l], dum));
            b[ll] = ubrad_sub(b[ll], ubrad_mul(b[icol], dum));
        }
    }
    return true;
}

// Read BEP's canonical per-column parameter, view, and state records directly.
// Direction slots are padded to two so one thread can own each column/slot.
kernel void urban_bep_radiosity_fragment(
    constant UrbanBepRadiosityFragmentResources &r [[buffer(0)]],
    constant UrbanBepRadiosityFragmentParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    const int ncol = p.ncol;
    if (item >= uint(ncol) * 2u) return;
    const int col = int(item) / 2, dir = int(item) % 2;
    const ulong pix = ulong(col) * 3ul;
    const int nz_u = r.column_param_ix[pix];
    const int nd = r.column_param_ix[pix + 1ul];
    const int type = r.utype_urb[col] - 1;
    const ulong param = ulong(col) * 140ul;
    const ulong view = ulong(col) * 794ul;
    const ulong state = ulong(col) * 1210ul;
    const ulong fwwbase = view + ulong(dir) * 324ul;
    const ulong fwgbase = view + 648ul + ulong(dir) * 18ul;
    const ulong fgwbase = view + 684ul + ulong(dir) * 18ul;
    const ulong fswbase = view + 720ul + ulong(dir) * 18ul;
    const ulong fsg_at = view + 792ul + ulong(dir);
    const ulong wallbase = item * 36ul;
    const ulong state_walls = state + 648ul;
    const ulong state_ground = state + 720ul + ulong(dir) + 18ul;
    const ulong direct_wall = item * 36ul;
    if (dir >= nd) {
        for (int k = 0; k < 36; ++k) {
            r.rlw[wallbase + ulong(k)] = 0.0f;
            r.rsw[wallbase + ulong(k)] = 0.0f;
        }
        r.rlg[item] = r.rsg[item] = 0.0f;
        r.error[item] = 0;
        return;
    }
    if (type < 0 || type >= 11 || r.active_cols[col] == 0) {
        for (int k = 0; k < 36; ++k) {
            r.rlw[wallbase + ulong(k)] = 0.0f;
            r.rsw[wallbase + ulong(k)] = 0.0f;
        }
        r.rlg[item] = r.rsg[item] = 0.0f;
        r.error[item] = 0;
        return;
    }
    if (r.column_param_ix[pix + 2ul] != 0) {
        r.error[item] = r.column_param_ix[pix + 2ul];
        return;
    }
    if (nz_u < 1 || nz_u > 18 || nd < 1 || nd > 2) { r.error[item] = 2097; return; }
    const float emw = r.class_x[143ul + ulong(type)];
    const float emg = r.class_x[132ul + ulong(type)];
    const float albw = r.class_x[110ul + ulong(type)];
    const float albg = r.class_x[99ul + ulong(type)];
    const float sigma = 5.67e-8f;
    const float rl = r.glw[col];
    float a[37 * 37];
    float b[37];
    const int n = 2 * nz_u + 1;
    for (int q = 0; q < n * 37; ++q) a[q] = 0.0f;

    // Longwave radiosity matrix and source vector, WRF source order.
    for (int i = 0; i < nz_u; ++i) {
        a[i * 37 + i] = 1.0f;
        for (int j = 0; j < nz_u; ++j) {
            const float vf = r.column_view_x[fwwbase + ulong(i) * 18ul + uint(j)];
            a[i * 37 + nz_u + j] = ubrad_mul(ubrad_mul(-(1.0f - emw), vf), r.column_param[param + 122ul + uint(j)]);
        }
        const float groundvf = r.column_view_x[fgwbase + uint(i)];
        a[i * 37 + 2 * nz_u] = ubrad_mul(-(1.0f - emg), groundvf);
        float rhs = ubrad_add(ubrad_mul(r.column_view_x[fswbase + uint(i)], rl),
            ubrad_mul(ubrad_mul(ubrad_mul(emg, groundvf), sigma), powr(r.column_state[state_ground], 4.0f)));
        for (int j = 0; j < nz_u; ++j) {
            const float vf = r.column_view_x[fwwbase + ulong(i) * 18ul + uint(j)];
            const float t = r.column_state[state_walls + ulong(4 * j + 2 * dir + 1)];
            const float emitted = ubrad_mul(ubrad_mul(ubrad_mul(ubrad_mul(r.column_param[param + 122ul + uint(j)], emw), sigma), vf), powr(t, 4.0f));
            const float transmitted = ubrad_mul(ubrad_mul(vf, rl), 1.0f - r.column_param[param + 122ul + uint(j)]);
            rhs = ubrad_add(ubrad_add(rhs, emitted), transmitted);
        }
        b[i] = rhs;
    }
    for (int i = 0; i < nz_u; ++i) {
        const int row = nz_u + i;
        for (int j = 0; j < nz_u; ++j)
            a[row * 37 + j] = ubrad_mul(ubrad_mul(-(1.0f - emw), r.column_view_x[fwwbase + ulong(i) * 18ul + uint(j)]), r.column_param[param + 122ul + uint(j)]);
        a[row * 37 + row] = 1.0f;
        const float gv = r.column_view_x[fgwbase + uint(i)];
        a[row * 37 + 2 * nz_u] = ubrad_mul(-(1.0f - emg), gv);
        const float t = r.column_state[state_walls + ulong(4 * i + 2 * dir)];
        float rhs = ubrad_add(ubrad_mul(r.column_view_x[fswbase + uint(i)], rl),
            ubrad_mul(ubrad_mul(ubrad_mul(emg, gv), sigma), powr(r.column_state[state_ground], 4.0f)));
        for (int j = 0; j < nz_u; ++j) {
            const float vf = r.column_view_x[fwwbase + ulong(i) * 18ul + uint(j)];
            const float wt = r.column_state[state_walls + ulong(4 * j + 2 * dir)];
            rhs = ubrad_add(ubrad_add(rhs,
                ubrad_mul(ubrad_mul(ubrad_mul(ubrad_mul(r.column_param[param + 122ul + uint(j)], emw), sigma), vf), powr(wt, 4.0f))),
                ubrad_mul(ubrad_mul(vf, rl), 1.0f - r.column_param[param + 122ul + uint(j)]));
        }
        b[row] = rhs;
        (void)t;
    }
    const int groundrow = 2 * nz_u;
    for (int j = 0; j < nz_u; ++j) {
        const float vf = r.column_view_x[fwgbase + uint(j)];
        a[groundrow * 37 + j] = ubrad_mul(ubrad_mul(-(1.0f - emw), vf), r.column_param[param + 122ul + uint(j)]);
        a[groundrow * 37 + nz_u + j] = ubrad_mul(ubrad_mul(-(1.0f - emw), vf), r.column_param[param + 122ul + uint(j)]);
    }
    a[groundrow * 37 + groundrow] = 1.0f;
    float ground_rhs = ubrad_mul(r.column_view_x[fsg_at], rl);
    for (int j = 0; j < nz_u; ++j) {
        const float vf = r.column_view_x[fwgbase + uint(j)];
        const float t1 = r.column_state[state_walls + ulong(4 * j + 2 * dir)];
        const float t2 = r.column_state[state_walls + ulong(4 * j + 2 * dir + 1)];
        const float emit = ubrad_mul(ubrad_mul(ubrad_mul(emw, sigma), vf),
            ubrad_mul(r.column_param[param + 122ul + uint(j)], ubrad_add(powr(t1, 4.0f), powr(t2, 4.0f))));
        const float refl = ubrad_mul(ubrad_mul(2.0f, vf),
            ubrad_mul(rl, 1.0f - r.column_param[param + 122ul + uint(j)]));
        ground_rhs = ubrad_add(ubrad_add(ground_rhs, emit), refl);
    }
    b[groundrow] = ground_rhs;
    if (!ubrad_gaussj(a, b, n)) { r.error[item] = 2287; return; }
    for (int i = 0; i < nz_u; ++i) {
        r.rlw[wallbase + uint(i)] = b[i];
        r.rlw[wallbase + 18ul + uint(i)] = b[nz_u + i];
    }
    r.rlg[item] = b[groundrow];

    // Shortwave radiosity, using the direct-shadow inputs in rsw/rsg.
    for (int q = 0; q < n * 37; ++q) a[q] = 0.0f;
    for (int i = 0; i < nz_u; ++i) {
        a[i * 37 + i] = 1.0f;
        for (int j = 0; j < nz_u; ++j)
            a[i * 37 + nz_u + j] = ubrad_mul(ubrad_mul(-albw, r.column_view_x[fwwbase + ulong(i) * 18ul + uint(j)]), r.column_param[param + 122ul + uint(j)]);
        a[i * 37 + groundrow] = ubrad_mul(-albg, r.column_view_x[fgwbase + uint(i)]);
        b[i] = r.direct_rsw[direct_wall + uint(i)];
    }
    for (int i = 0; i < nz_u; ++i) {
        const int row = nz_u + i;
        for (int j = 0; j < nz_u; ++j)
            a[row * 37 + j] = ubrad_mul(ubrad_mul(-albw, r.column_view_x[fwwbase + ulong(i) * 18ul + uint(j)]), r.column_param[param + 122ul + uint(j)]);
        a[row * 37 + row] = 1.0f;
        a[row * 37 + groundrow] = ubrad_mul(-albg, r.column_view_x[fgwbase + uint(i)]);
        b[row] = r.direct_rsw[direct_wall + 18ul + uint(i)];
    }
    for (int j = 0; j < nz_u; ++j) {
        const float vf = r.column_view_x[fwgbase + uint(j)];
        a[groundrow * 37 + j] = ubrad_mul(ubrad_mul(-albw, vf), r.column_param[param + 122ul + uint(j)]);
        a[groundrow * 37 + nz_u + j] = ubrad_mul(ubrad_mul(-albw, vf), r.column_param[param + 122ul + uint(j)]);
    }
    a[groundrow * 37 + groundrow] = 1.0f;
    b[groundrow] = r.direct_rsg[item];
    if (!ubrad_gaussj(a, b, n)) { r.error[item] = 2192; return; }
    for (int i = 0; i < nz_u; ++i) {
        r.rsw[wallbase + uint(i)] = b[i];
        r.rsw[wallbase + 18ul + uint(i)] = b[nz_u + i];
    }
    r.rsg[item] = b[groundrow];
    r.error[item] = 0;
}
