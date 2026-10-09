#include <metal_stdlib>
#include "urban_bep_bem_class_init_fragment_bindings.metalh"
using namespace metal;

// WRF module_sf_bep_bem.F bb_init_para (5052-5211). One lane owns a class;
// table and class arrays retain the source's packed Fortran field offsets.
kernel void urban_bep_bem_class_init_fragment(
    constant UrbanBepBemClassInitFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemClassInitFragmentParams &p [[buffer(1)]],
    uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.nurbm)) return;
    device const float *t = r.table_float;
    device const int *ti = r.table_int;
    device float *f = r.class_float;
    device int *i = r.class_int;
    if (p.nurbm < 1 || p.nurbm > 11) { i[0] = 9002; return; }
    const float heat_scale = 1.0f / 4.1868f * 1.0e-6f;
    const float conductivity_scale = 1.0f / 4.1868f * 1.0e-2f;
    const float csw = t[121 + c] / heat_scale;
    const float csr = t[143 + c] / heat_scale;
    const float csg = t[132 + c] / heat_scale;
    f[44 + c] = csw;
    f[55 + c] = csr;
    f[33 + c] = csg;
    f[11 + c] = (t[11 + c] / csw) / conductivity_scale;
    f[22 + c] = (t[22 + c] / csr) / conductivity_scale;
    f[c] = (t[c] / csg) / conductivity_scale;
    f[1287 + c] = t[1522 + c];
    f[1298 + c] = t[1566 + c];
    f[1309 + c] = t[1533 + c];
    f[77 + c] = t[33 + c];
    f[88 + c] = t[55 + c];
    f[66 + c] = t[44 + c];
    f[110 + c] = t[176 + c];
    f[121 + c] = t[198 + c];
    f[99 + c] = t[187 + c];
    f[143 + c] = t[1588 + c];
    f[132 + c] = t[1577 + c];
    i[1 + c] = ti[3 + c];
    f[1107 + c] = t[77 + c];
    f[1118 + c] = t[154 + c];
    f[1074 + c] = t[165 + c];
    f[1085 + c] = t[1423 + c];
    f[1096 + c] = t[66 + c];
    i[23 + c] = ti[25 + c];
    f[1129 + c] = t[1555 + c];
    f[1140 + c] = t[1544 + c];
    f[1151 + c] = t[1511 + c];
    f[1162 + c] = t[220 + c];
    f[1173 + c] = t[1500 + c];
    f[1184 + c] = t[209 + c];
    f[1195 + c] = t[1401 + c];
    f[1206 + c] = t[231 + c];
    f[1217 + c] = t[1412 + c];
    f[1228 + c] = t[1366 + c];
    if (c == 0) {
        for (uint h = 0; h < 24; ++h) {
            f[1239 + h] = t[1342 + h];
            f[1263 + h] = t[1377 + h];
        }
    }
    const int nd = ti[3 + c];
    const int nh = ti[14 + c];
    if (nd > 2) { i[0] = 8; return; }
    if (nh + 3 > 18) { i[0] = 9; return; }
    for (uint d = 0; d < 2; ++d) {
        f[154 + 2 * c + d] = 100000.0f;
        f[176 + 2 * c + d] = t[1434 + 3 * c + d] * 3.14159265358979323846f / 180.0f;
        f[198 + 2 * c + d] = t[1467 + 3 * c + d];
        f[220 + 2 * c + d] = t[88 + 3 * c + d];
    }
    for (uint b = 0; b < 18; ++b) {
        const uint dst = 242 + 18 * c + b;
        f[dst] = b < uint(nh) ? t[242 + 50 * c + b] : 0.0f;
        f[440 + 18 * c + b] = b < uint(nh) ? t[792 + 50 * c + b] : 0.0f;
    }
    for (uint iz = 0; iz < 18; ++iz) {
        f[638 + 18 * c + iz] = 0.0f;
        f[836 + 18 * c + iz] = 0.0f;
    }
    i[12 + c] = 0;
    f[1063 + c] = 0.9f;
    if (c == 0) {
        i[34] = ti[0];
        i[35] = ti[1];
        i[36] = p.nurbm;
        i[0] = 0;
    }
}
