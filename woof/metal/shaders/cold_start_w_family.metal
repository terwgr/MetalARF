#include <metal_stdlib>
#include "wrf_cold_start_w_bindings.metalh"

using namespace metal;

// The CUDA export uses explicit round-to-nearest FP32 operations. Preserve
// those operation boundaries and source expression order without contraction.
#pragma clang fp contract(off)

kernel void wrf_cold_start_w(
    constant WrfColdStartWResources &r [[buffer(0)]],
    constant WrfColdStartWParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    const uint count = uint(p.ny * p.nx);
    if (flat >= count) return;

    const int j = int(flat) / p.nx;
    const int i = int(flat) - j * p.nx;
    const int im1 = (i == 0) ? (p.periodic_x ? p.nx - 1 : 0) : i - 1;
    const int ip1 = (i == p.nx - 1) ? (p.periodic_x ? 0 : p.nx - 1) : i + 1;
    const int jm1 = (j == 0) ? (p.periodic_y ? p.ny - 1 : 0) : j - 1;
    const int jp1 = (j == p.ny - 1) ? (p.periodic_y ? 0 : p.ny - 1) : j + 1;

    const ulong cell = ulong(flat);
    const ulong stride = ulong(p.ny) * ulong(p.nx);
    const ulong uplane = ulong(p.ny) * ulong(p.nx + 1);
    const ulong vplane = ulong(p.ny + 1) * ulong(p.nx);
    const ulong ul = ulong(j) * ulong(p.nx + 1) + ulong(i);
    const ulong vb = cell;

    float uc0 = p.cf1 * r.u[ul];
    uc0 = uc0 + p.cf2 * r.u[uplane + ul];
    uc0 = uc0 + p.cf3 * r.u[2 * uplane + ul];
    float uc1 = p.cf1 * r.u[ul + 1];
    uc1 = uc1 + p.cf2 * r.u[uplane + ul + 1];
    uc1 = uc1 + p.cf3 * r.u[2 * uplane + ul + 1];
    float vc0 = p.cf1 * r.v[vb];
    vc0 = vc0 + p.cf2 * r.v[vplane + vb];
    vc0 = vc0 + p.cf3 * r.v[2 * vplane + vb];
    float vc1 = p.cf1 * r.v[vb + ulong(p.nx)];
    vc1 = vc1 + p.cf2 * r.v[vplane + vb + ulong(p.nx)];
    vc1 = vc1 + p.cf3 * r.v[2 * vplane + vb + ulong(p.nx)];

    const ulong north = ulong(jp1) * ulong(p.nx) + ulong(i);
    const ulong south = ulong(jm1) * ulong(p.nx) + ulong(i);
    const ulong east = ulong(j) * ulong(p.nx) + ulong(ip1);
    const ulong west = ulong(j) * ulong(p.nx) + ulong(im1);
    const float dyn = r.ht[north] - r.ht[cell];
    const float dys = r.ht[cell] - r.ht[south];
    const float dxe = r.ht[east] - r.ht[cell];
    const float dxw = r.ht[cell] - r.ht[west];

    float yscale = r.msfty[cell] * 0.5f;
    yscale = yscale * p.rdy;
    float xscale = r.msftx[cell] * 0.5f;
    xscale = xscale * p.rdx;
    float yterm = dyn * vc1;
    yterm = yterm + dys * vc0;
    const float y = yscale * yterm;
    float xterm = dxe * uc1;
    xterm = xterm + dxw * uc0;
    const float x = xscale * xterm;
    const float surface = y + x;

    r.w[cell] = surface;
    for (int k = 1; k <= p.nz; ++k) {
        float value = surface * r.znw[ulong(k)];
        value = value * r.znw[ulong(k)];
        r.w[ulong(k) * stride + cell] = value;
    }
}
