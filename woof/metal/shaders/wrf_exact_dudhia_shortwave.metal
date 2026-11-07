// Source-ordered WRF Dudhia column recurrence with software binary32.
#include <metal_stdlib>
#include "exact_gfk_math.metalh"
#include "wrf_exact_dudhia_shortwave_bindings.metalh"
using namespace metal;
static inline bool operator<(Exact32 a, Exact32 b) { return soft32_less(a.bits,b.bits); }
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_less_equal(a.bits,b.bits); }
static inline Exact32 FADD(Exact32 a, Exact32 b) { return a+b; }
static inline Exact32 FSUB(Exact32 a, Exact32 b) { return a-b; }
static inline Exact32 FMUL(Exact32 a, Exact32 b) { return a*b; }
static inline Exact32 FDIV(Exact32 a, Exact32 b) { return a/b; }
static inline Exact32 dsw_log10f(Exact32 x)
{
    unsigned int hx = x.bits;
    if ((hx & 0x7fffffffu) == 0u) return Exact32::from_bits(0xff800000);
    if (hx & 0x80000000u) return Exact32::from_bits(0x7fc00000);
    if (hx >= 0x7f800000u) return FADD(x, x);
    int k = 0;
    if (hx < 0x00800000u) {
        k = -25;
        hx = FMUL(x, 33554432.0f).bits;
    }
    k += (int)(hx >> 23) - 127;
    const int i = (int)(((unsigned int)k & 0x80000000u) >> 31);
    const Exact32 y = (Exact32)(k + i);
    const Exact32 m = Exact32::from_bits((hx & 0x007fffffu)
                                    | ((unsigned int)(0x7f - i) << 23));
    const Exact32 z = FADD(FMUL(y, Exact32::from_bits(0x355427dbu)),
                         FMUL(Exact32::from_bits(0x3ede5bd9u),
                              exact_gfk_log(m)));
    return FADD(z, FMUL(y, Exact32::from_bits(0x3e9a2080u)));
}
constant float DSW_ALBTAB[20] = {
    0.f, 0.f, 0.f, 0.f,   69.f, 58.f, 40.f, 15.f,   90.f, 80.f, 70.f, 60.f,
    94.f, 90.f, 82.f, 78.f,   96.f, 92.f, 85.f, 80.f};
constant float DSW_ABSTAB[20] = {
    0.f, 0.f, 0.f, 0.f,   0.f, 2.5f, 4.f, 5.f,   0.f, 2.6f, 7.f, 10.f,
    0.f, 3.3f, 10.f, 14.f,   0.f, 3.7f, 10.f, 15.f};
constant float DSW_XMUVAL[4] = {0.f, 0.2f, 0.5f, 1.0f};
static inline Exact32 dsw_interp(constant float* tab, int iil, int iu,
                                            int jjl, int ju, Exact32 xi,
                                            Exact32 yj)
{
    const Exact32 a = FSUB(xi, (Exact32)iil);
    const Exact32 b = FSUB((Exact32)iu, xi);
    const Exact32 c = FSUB(yj, (Exact32)jjl);
    const Exact32 d = FSUB((Exact32)ju, yj);
    Exact32 s = FMUL(FMUL(tab[(ju - 1) * 4 + (iu - 1)], a), c);
    s = FADD(s, FMUL(FMUL(tab[(ju - 1) * 4 + (iil - 1)], b), c));
    s = FADD(s, FMUL(FMUL(tab[(jjl - 1) * 4 + (iu - 1)], a), d));
    s = FADD(s, FMUL(FMUL(tab[(jjl - 1) * 4 + (iil - 1)], b), d));
    return FDIV(s, (Exact32)((iu - iil) * (ju - jjl)));
}
static inline void exact_dudhia_column(int col, device const Exact32* t,
                      device const Exact32* p,
                      device const Exact32* qv,
                      device const Exact32* qc,
                      device const Exact32* qr,
                      device const Exact32* qi,
                      device const Exact32* qs,
                      device const Exact32* qg,
                      device const Exact32* dz,
                      device const Exact32* exner,
                      device const Exact32* coszen,
                      device const Exact32* albedo,
                      device Exact32* heating,
                      device Exact32* gsw,
                      device Exact32* swdown,
                      Exact32 soltop, Exact32 cssca, int icloud, int use_exner,
                      int ncol, int nlay, long col_stride,
                      long lay_stride, long base)
{
    if (col >= ncol) return;
    const Exact32 r = 287.0f;
    const Exact32 cp = 1004.5f;
    const Exact32 xmu = coszen[col];
    const Exact32 alb = albedo[col];
    const long c0 = base + (long)col * col_stride;
    if (xmu <= 1.0e-9f || soltop <= 1.0e-9f) {
        for (int k = 0; k < nlay; ++k) {
            const long o = c0 + (long)k * lay_stride;
            const Exact32 tten = 0.0f;
            heating[o] = FADD(0.0f, use_exner ? FDIV(tten, exner[o]) : tten);
        }
        const Exact32 g = 0.0f;
        gsw[col] = g;
        swdown[col] = FDIV(g, FSUB(1.0f, alb));
        return;
    }
    const Exact32 sdown1 = FMUL(soltop, xmu);
    Exact32 sdown = sdown1;
    Exact32 ww = 0.0f, uv = 0.0f, oldalb = 0.0f, oldabc = 0.0f;
    Exact32 totabs = 0.0f, dsca = 0.0f, dabs = 0.0f, dscld = 0.0f;
    for (int k = 0; k < nlay; ++k) {
        const long o = c0 + (long)k * lay_stride;
        const Exact32 tk = t[o];
        const Exact32 dzk = dz[o];
        const Exact32 qvk = exact_max(0.0f, qv[o]);
        const Exact32 ro = FDIV(p[o], FMUL(r, tk));
        const Exact32 xwvp = FMUL(FMUL(FMUL(ro, qvk), dzk), 1000.0f);
        const Exact32 xatp = FMUL(ro, dzk);
        Exact32 xlwp = 0.0f;
        if (icloud != 0) {
            Exact32 q = FADD(exact_max(0.0f, qc[o]),
                           FMUL(0.1f, exact_max(0.0f, qi[o])));
            q = FADD(q, FMUL(0.05f, exact_max(0.0f, qr[o])));
            q = FADD(q, FMUL(0.02f, exact_max(0.0f, qs[o])));
            q = FADD(q, FMUL(0.05f, exact_max(0.0f, qg[o])));
            xlwp = FMUL(FMUL(FMUL(ro, 1000.0f), dzk), q);
        }
        ww = FADD(ww, xlwp);
        uv = FADD(uv, xwvp);
        const Exact32 wgm = FDIV(ww, xmu);
        const Exact32 ugcm = FDIV(FMUL(uv, 0.0001f), xmu);
        const Exact32 oldabs = totabs;
        totabs = FDIV(FMUL(2.9f, ugcm),
                      FADD(exact_gfk_pow(FADD(1.0f, FMUL(141.5f, ugcm)), 0.635f),
                           FMUL(5.925f, ugcm)));
        Exact32 xsca = FDIV(FMUL(cssca, xatp), xmu);
        Exact32 xabs = FDIV(FMUL(FSUB(totabs, oldabs),
                               FSUB(FSUB(sdown1, dscld), dsca)), sdown);
        if (xabs < 0.0f) xabs = 0.0f;
        Exact32 alw = dsw_log10f(FADD(wgm, 1.0f));
        if (alw > 3.999f) alw = 3.999f;
        int iil = 1, iu = 2;
        Exact32 xi = 0.0f;
        for (int ii = 1; ii <= 3; ++ii) {
            if (xmu > DSW_XMUVAL[ii - 1]) {
                iil = ii;
                iu = ii + 1;
                xi = FADD(FDIV(FSUB(xmu, DSW_XMUVAL[ii - 1]),
                               FSUB(DSW_XMUVAL[ii], DSW_XMUVAL[ii - 1])),
                          (Exact32)iil);
            }
        }
        const int jjl = int(as_type<float>(alw.bits)) + 1;
        const int ju = jjl + 1;
        const Exact32 yj = FADD(alw, 1.0f);
        const Exact32 alba = dsw_interp(DSW_ALBTAB, iil, iu, jjl, ju, xi, yj);
        const Exact32 absc = dsw_interp(DSW_ABSTAB, iil, iu, jjl, ju, xi, yj);
        const Exact32 avail = FSUB(FSUB(sdown1, dsca), dabs);
        Exact32 xalb = FDIV(FMUL(FSUB(alba, oldalb), avail), sdown);
        Exact32 xabsc = FDIV(FMUL(FSUB(absc, oldabc), avail), sdown);
        if (xalb < 0.0f) xalb = 0.0f;
        if (xabsc < 0.0f) xabsc = 0.0f;
        dscld = FADD(dscld, FMUL(FMUL(FADD(xalb, xabsc), sdown), 0.01f));
        dsca = FADD(dsca, FMUL(xsca, sdown));
        dabs = FADD(dabs, FMUL(xabs, sdown));
        oldalb = alba;
        oldabc = absc;
        Exact32 trans0 = FSUB(FSUB(FSUB(FSUB(100.0f, xalb), xabsc),
                                 FMUL(xabs, 100.0f)), FMUL(xsca, 100.0f));
        if (trans0 < 1.0f) {
            const Exact32 ff = FDIV(99.0f,
                                  FADD(FADD(FADD(xalb, xabsc),
                                            FMUL(xabs, 100.0f)),
                                       FMUL(xsca, 100.0f)));
            xalb = FMUL(xalb, ff);
            xabsc = FMUL(xabsc, ff);
            xabs = FMUL(xabs, ff);
            xsca = FMUL(xsca, ff);
            trans0 = 1.0f;
        }
        const Exact32 absorbed = FADD(FADD(xabsc, FMUL(xabs, 100.0f)),
                                    FMUL(0.0f, 100.0f));
        const Exact32 tten = FDIV(FMUL(FMUL(sdown, absorbed), 0.01f),
                                FMUL(FMUL(ro, cp), dzk));
        heating[o] = FADD(0.0f, use_exner ? FDIV(tten, exner[o]) : tten);
        sdown = exact_max(1.0e-9f, FMUL(FMUL(sdown, trans0), 0.01f));
    }
    const Exact32 g = FMUL(FSUB(1.0f, alb), sdown);
    gsw[col] = g;
    swdown[col] = FDIV(g, FSUB(1.0f, alb));
}
kernel void wrf_exact_dudhia_shortwave_columns(
    constant WrfExactDudhiaShortwaveColumnsResources &r [[buffer(0)]],
    constant WrfExactDudhiaShortwaveColumnsParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    const long ncol = long(p.ny)*long(p.nx);
    if (long(col) >= ncol) return;
#define A(name) reinterpret_cast<device const Exact32 *>(r.name)
#define O(name) reinterpret_cast<device Exact32 *>(r.name)
    exact_dudhia_column(int(col), A(temperature), A(pressure), A(qv),
        A(qc), A(qr), A(qi), A(qs), A(qg), A(dz), A(exner),
        A(cosine_zenith), A(albedo), O(heating), O(gsw), O(swdown),
        Exact32(p.solcon), Exact32(p.swrad_scat)*Exact32(1.0e-5f),
        p.icloud, 1, int(ncol), p.nz, 1, -ncol, long(p.nz-1)*ncol);
    O(coszen)[col] = A(cosine_zenith)[col];
#undef A
#undef O
}
