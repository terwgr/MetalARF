#include <metal_stdlib>
#include "add_diff2_bindings.metalh"
#include "add_diff2_raw_bindings.metalh"

using namespace metal;

#pragma clang fp contract(off)

inline uint diff_periodic(int value, uint extent) {
    int result = value % int(extent);
    return uint(result < 0 ? result + int(extent) : result);
}

inline float diff_laplacian(device const float *f,
                            device const float *rdzf,
                            device const float *rdzc,
                            int k, int j, int i,
                            int nlev, int ny, int nys, int nx, int nxs,
                            float kh, float kv,
                            float dx_inv2, float dy_inv2,
                            int wstag) {
    const uint ic = diff_periodic(i, uint(nx));
    const uint jc = diff_periodic(j, uint(ny));
    const uint center_index = (uint(k) * uint(nys) + jc) * uint(nxs) + ic;
    const float fc = f[center_index];

    const uint east = (uint(k) * uint(nys) + jc) * uint(nxs)
                    + diff_periodic(i + 1, uint(nx));
    const uint west = (uint(k) * uint(nys) + jc) * uint(nxs)
                    + diff_periodic(i - 1, uint(nx));
    const uint north = (uint(k) * uint(nys)
                      + diff_periodic(j + 1, uint(ny))) * uint(nxs) + ic;
    const uint south = (uint(k) * uint(nys)
                      + diff_periodic(j - 1, uint(ny))) * uint(nxs) + ic;
    const float dxx = f[east] - 2.0f * fc + f[west];
    const float dyy = f[north] - 2.0f * fc + f[south];
    const float horizontal = kh * (dx_inv2 * dxx + dy_inv2 * dyy);

    const float fup = (k < nlev - 1)
        ? (f[((uint(k + 1) * uint(nys) + jc) * uint(nxs)) + ic] - fc)
            * rdzf[k]
        : 0.0f;
    const float fdn = (k > 0)
        ? (fc - f[((uint(k - 1) * uint(nys) + jc) * uint(nxs)) + ic])
            * rdzf[k - 1]
        : 0.0f;
    float lap = horizontal;
    lap += kv * (fup - fdn) * rdzc[k];
    return lap;
}

kernel void add_diff2_raw(constant AddDiff2RawResources &r [[buffer(0)]],
                          constant AddDiff2RawParams &p [[buffer(1)]],
                          uint flat [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.nys) * uint(p.nxs);
    if (flat >= count) return;
    const int i = int(flat % uint(p.nxs));
    const int j = int((flat / uint(p.nxs)) % uint(p.nys));
    const int k = int(flat / (uint(p.nxs) * uint(p.nys)));
    if (p.stagger == 3 && (k == 0 || k == p.nlev - 1)) return;
    const float lap = diff_laplacian(
        r.f, r.rdzf, r.rdzc, k, j, i, p.nlev, p.ny, p.nys, p.nx,
        p.nxs, p.kh, p.kv, p.dx_inv2, p.dy_inv2, p.stagger == 3);
    const uint idx = flat;
    r.tend[idx] += lap;
}

kernel void add_diff2(constant AddDiff2Resources &r [[buffer(0)]],
                      constant AddDiff2Params &p [[buffer(1)]],
                      uint flat [[thread_position_in_grid]]) {
    const uint count = uint(p.nlev) * uint(p.nys) * uint(p.nxs);
    if (flat >= count) return;
    const int i = int(flat % uint(p.nxs));
    const int j = int((flat / uint(p.nxs)) % uint(p.nys));
    const int k = int(flat / (uint(p.nxs) * uint(p.nys)));
    if (p.stagger == 3 && (k == 0 || k == p.nlev - 1)) return;
    const float lap = diff_laplacian(
        r.f, r.rdzf, r.rdzc, k, j, i, p.nlev, p.ny, p.nys, p.nx,
        p.nxs, p.kh, p.kv, p.dx_inv2, p.dy_inv2, p.stagger == 3);

    const uint ic = diff_periodic(i, uint(p.nx));
    const uint jc = diff_periodic(j, uint(p.ny));
    float mu = r.mub2d[jc * uint(p.nx) + ic]
             + r.mup[jc * uint(p.nx) + ic];
    if (p.stagger == 1) {
        const uint left = jc * uint(p.nx) + diff_periodic(i - 1, uint(p.nx));
        const uint right = jc * uint(p.nx) + diff_periodic(i, uint(p.nx));
        const float mul = r.mub2d[left] + r.mup[left];
        const float mur = r.mub2d[right] + r.mup[right];
        mu = 0.5f * (mul + mur);
    } else if (p.stagger == 2) {
        const uint south = diff_periodic(j - 1, uint(p.ny)) * uint(p.nx) + ic;
        const uint north = diff_periodic(j, uint(p.ny)) * uint(p.nx) + ic;
        const float mus = r.mub2d[south] + r.mup[south];
        const float mun = r.mub2d[north] + r.mup[north];
        mu = 0.5f * (mus + mun);
    }
    const float coupling = r.c1[k] * mu + r.c2[k];
    r.tend[flat] += coupling * lap;
}
