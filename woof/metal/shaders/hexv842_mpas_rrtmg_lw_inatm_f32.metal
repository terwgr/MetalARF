#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#pragma clang fp contract(off)
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#define MXMOL 38
#define MAXXSEC 4
#include "hexv842_mpas_rrtmg_lw_inatm_f32_bindings.metalh"
kernel void hexv842_mpas_rrtmg_lw_inatm_f32(constant Hexv842MpasRrtmgLwInatmF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwInatmF32Params &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {

    int col=int(c);
    if (col >= p.ncol) return;
    const float amd = 28.9660f;
    const float amw = 18.0160f;
#define WKLC(m, l) r.wkl[((ulong)col * MXMOL + ((m) - 1)) * p.nl + ((l) - 1)]
#define WXC(m, l) r.wx[((ulong)col * MAXXSEC + ((m) - 1)) * p.nl + ((l) - 1)]
    device const float* pz = r.plev + (ulong)col * (p.nl + 1);

    float amttl = 0.0f;
    float wvttl = 0.0f;
    for (int l = 1; l <= p.nl; ++l) {
        float w1 = WKLC(1, l);
        float amm = RLW_AD(RLW_MU(RLW_SU(1.0f, w1), amd), RLW_MU(w1, amw));
        float num = RLW_MU(RLW_MU(RLW_SU(pz[l - 1], pz[l]), 1.e3f), p.avogad);
        float den = RLW_MU(RLW_MU(RLW_MU(1.e2f, p.grav), amm),
                           RLW_AD(1.0f, w1));
        r.coldry[(ulong)col * p.nl + (l - 1)] = RLW_DV(num, den);
    }
    for (int l = 1; l <= p.nl; ++l) {
        float cd = r.coldry[(ulong)col * p.nl + (l - 1)];
        float summol = 0.0f;
        for (int imol = 2; imol <= 7; ++imol)
            summol = RLW_AD(summol, WKLC(imol, l));
        r.wbrodl[(ulong)col * p.nl + (l - 1)] =
            RLW_MU(cd, RLW_SU(1.0f, summol));
        for (int imol = 1; imol <= 7; ++imol)
            WKLC(imol, l) = RLW_MU(cd, WKLC(imol, l));
        amttl = RLW_AD(RLW_AD(amttl, cd), WKLC(1, l));
        wvttl = RLW_AD(wvttl, WKLC(1, l));
        for (int ix = 1; ix <= MAXXSEC; ++ix) {
            // ixindx = identity (1,2,3,4) in lwdatinit
            WXC(ix, l) = RLW_MU(RLW_MU(cd, WXC(ix, l)), 1.e-20f);
        }
    }
    float wvsh = RLW_DV(RLW_MU(amw, wvttl), RLW_MU(amd, amttl));
    r.pwvcm_v[col] = RLW_DV(RLW_MU(wvsh, RLW_MU(1.e3f, pz[0])),
                          RLW_MU(1.e2f, p.grav));
#undef WKLC
#undef WXC
}
