// New Tiedtke (WRF cu_physics=16), Metal stage family.
#include <metal_stdlib>
#include "ntiedtke_prep_bindings.metalh"
#include "ntiedtke_convert_bindings.metalh"
#include "ntiedtke_cuinin_bindings.metalh"
#include "ntiedtke_cutypen_bindings.metalh"
#include "ntiedtke_midlevel_bindings.metalh"
#include "ntiedtke_mfub_bindings.metalh"
#include "ntiedtke_closure_bindings.metalh"
#include "ntiedtke_cuascn_bindings.metalh"
#include "ntiedtke_cudtdqn_bindings.metalh"
#include "ntiedtke_cududvn_bindings.metalh"
#include "ntiedtke_cudlfsn_bindings.metalh"
#include "ntiedtke_cuddrafn_bindings.metalh"
#include "ntiedtke_cuflxn_bindings.metalh"
#include "ntiedtke_cloud_depth_bindings.metalh"
#include "ntiedtke_adjust_bindings.metalh"
#include "ntiedtke_momentum_rescale_bindings.metalh"
#include "ntiedtke_updraft_scale_bindings.metalh"
#include "ntiedtke_momentum_profile_bindings.metalh"
#include "ntiedtke_ke_dissipation_bindings.metalh"
#include "ntiedtke_post_run_bindings.metalh"
#include "ntiedtke_post_conversion_bindings.metalh"
#include "ntiedtke_force_fold_bindings.metalh"
#include "ntiedtke_gather_level_bindings.metalh"
#include "ntiedtke_gather_interface_bindings.metalh"
#include "ntiedtke_gather_surface_bindings.metalh"
#include "ntiedtke_scatter_level_bindings.metalh"
#include "ntiedtke_scatter_surface_bindings.metalh"
#include "ntiedtke_llo3_reduce_columns_bindings.metalh"
#include "ntiedtke_llo3_reduce_finish_bindings.metalh"
#include "ntiedtke_driver_rain_merge_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float nt_add(float a,float b){return a+b;}
static inline float nt_mul(float a,float b){return a*b;}
static inline float nt_div(float a,float b){return a/b;}

#define NT_DXREF 15000.0f
#define NT_TMELT 273.16f
#define NT_RTWAT NT_TMELT
#define NT_RTICE (NT_TMELT - 23.0f)
#define NT_C1ES 610.78f
#define NT_C3LES 17.2693882f
#define NT_C3IES 21.875f
#define NT_C4LES 35.86f
#define NT_C4IES 7.66f
#define NT_T13 (1.0f / 3.0f)
#define NT_ZDNOPRC 2.0e4f
#define NT_CMFCMIN 1.0e-10f
#define NT_CMFCMAX 1.0f
#define NT_CPRCON 1.4e-3f
#define NT_RTBER (NT_TMELT - 5.0f)
#define NT_CMFDEPS 0.30f
#define NT_ENTRDD 2.0e-4f
#define NT_ZTAUMEL 18000.0f
#define NT_ZCUCOV 0.05f
#define NT_PGCOEF 0.7f

struct NtConst {
    float cpd, rcpd, c2es, vtmpc1, r5alvcp, r5alscp, ralvdcp, ralsdcp, g;
    float rd, zrg, ralfdcp, c5les, c5ies, alv;
    /* als and alf are needed by cudtdqn (alf*plglac) and by foelhm. */
    float als, alf;
};

#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]
#define NTM(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]
#define NTK(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

static inline void nt_scale_factors(float dx, device float *sf, device float *sf2) {

    if (dx < NT_DXREF) {
        float ratio = nt_div(NT_DXREF, dx);
        float base = nt_add(1.06133f, log(ratio));
        /* **3 on an INTEGER exponent: gfortran emits a multiply chain, not
         * powf.  Left to right, so (base*base)*base. */
        *sf = nt_mul(nt_mul(base, base), base);
        *sf2 = sqrt(*sf);
    } else {
        *sf = nt_add(1.0f, nt_mul(1.33e-5f, dx));
        *sf2 = 1.0f;
    }

}

static inline void nt_body_ntiedtke_prep(int i, device const float *t3d, device const float *qv3d, device const float *qc3d, device const float *qi3d, device const float *u3d, device const float *v3d, device const float *pcps, device const float *dz8w, device const float *rho3d, device const float *p8w, device const float *w, device const float *qvften, device const float *thften, device const float *xland, device const float *hfx, device const float *qfx, device const float *dx, device float *prsl, device float *ghtl, device float *omg, device float *tf, device float *qvf, device float *qcf, device float *qif, device float *uf, device float *vf, device float *qvftenz, device float *thftenz, device float *prsi, device float *ghti, device int *slimsk, device float *scale_fac, device float *scale_fac2, device float *delt_out, int ncol, int nz, float dt, int stepcu, int itimestep, float grav) {

    if (i >= ncol) return;

#define NT_IN(a, k)  (a)[(ulong)(k) * (ulong)ncol + (ulong)i]
#define NT_OUT(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    /* :396.  delt = dt*stepcu. */
    const float delt = nt_mul(dt, (float)stepcu);
    delt_out[i] = delt;

    /* :400-402.  slimsk = (abs(xland-2.)) -- a REAL expression assigned to
     * an INTEGER dummy, so it TRUNCATES toward zero rather than rounding.
     * xland = 2 (water) -> 0, xland = 1 (land) -> 1. */
    slimsk[i] = (int)fabs(nt_add(xland[i], -2.0f));

    nt_scale_factors(dx[i], &scale_fac[i], &scale_fac2[i]);

    /* :404-411 and the flip at :419-431 fused.
     *
     * zi[0] = 0; zi[k+1] = zi[k] + dz[k], accumulated upward, so every
     * interface carries the rounding of every layer beneath it -- the
     * accumulation ORDER is part of the answer and cannot be reassociated.
     *
     * ghti is zi reversed: ghti[nz - m] = zi[m].  Writing straight into the
     * reversed slot avoids materialising zi as a column array, which is what
     * keeps this kernel's local frame at zero.
     *
     * zl[k] = 0.5*(zi[k] + zi[k+1]) needs both ends of the layer, so the
     * previous accumulator is carried in a register.
     *
     * dot[k] = -0.5*grav*rho[k]*(w[k] + w[k+1]).  Fortran associates this
     * left to right over the parenthesised add: ((0.5*grav)*rho)*(w+w),
     * negated.  Three separate rounded multiplies -- NOT an FMA, and not
     * regrouped as 0.5*(grav*rho).
     */
    float zi_lo = 0.0f;
    NT_OUT(ghti, nz) = 0.0f;                   /* zi[0] -> ghti[nz] */
    const float half_g = nt_mul(0.5f, grav);

    for (int k = 0; k < nz; ++k) {
        const float zi_hi = nt_add(zi_lo, NT_IN(dz8w, k));
        NT_OUT(ghti, nz - 1 - k) = zi_hi;      /* zi[k+1] -> ghti[nz-k-1] */

        const float zl = nt_mul(0.5f, nt_add(zi_lo, zi_hi));
        const float wsum = nt_add(NT_IN(w, k), NT_IN(w, k + 1));
        const float dot = -nt_mul(nt_mul(half_g, NT_IN(rho3d, k)),
                                     wsum);

        /* Half-level fields reverse over nz: out[nz-1-k] = in[k]. */
        const int o = nz - 1 - k;
        NT_OUT(ghtl, o) = zl;
        NT_OUT(omg, o)  = dot;
        NT_OUT(prsl, o) = NT_IN(pcps, k);
        NT_OUT(tf, o)   = NT_IN(t3d, k);
        NT_OUT(qvf, o)  = NT_IN(qv3d, k);
        NT_OUT(qcf, o)  = NT_IN(qc3d, k);
        NT_OUT(qif, o)  = NT_IN(qi3d, k);
        NT_OUT(uf, o)   = NT_IN(u3d, k);
        NT_OUT(vf, o)   = NT_IN(v3d, k);

        /* :449-462.  The forcing tendencies are ZEROED on the first
         * timestep rather than flipped.  Not a nicety: the nonequil closure
         * builds zcape2 entirely out of ptte/pqte, so itimestep == 1 loses
         * that term and the deep closure answers differently. */
        if (itimestep == 1) {
            NT_OUT(qvftenz, o) = 0.0f;
            NT_OUT(thftenz, o) = 0.0f;
        } else {
            NT_OUT(qvftenz, o) = NT_IN(qvften, k);
            NT_OUT(thftenz, o) = NT_IN(thften, k);
        }

        zi_lo = zi_hi;
    }

    /* Interface pressure reverses over nz+1. */
    for (int k = 0; k <= nz; ++k) {
        NT_OUT(prsi, nz - k) = NT_IN(p8w, k);
    }

#undef NT_IN
#undef NT_OUT

}
kernel void ntiedtke_prep(
    constant NtiedtkePrepResources &r [[buffer(0)]],
    constant NtiedtkePrepParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_prep(i, r.t3d, r.qv3d, r.qc3d, r.qi3d, r.u3d, r.v3d, r.pcps, r.dz8w, r.rho3d, r.p8w, r.w, r.qvften, r.thften, r.xland, r.hfx, r.qfx, r.dx, r.prsl, r.ghtl, r.omg, r.tf, r.qvf, r.qcf, r.qif, r.uf, r.vf, r.qvftenz, r.thftenz, r.prsi, r.ghti, r.slimsk, r.scale_fac, r.scale_fac2, r.delt_out, p.ncol, p.nz, p.dt, p.stepcu, p.itimestep, p.grav);
}

static inline NtConst nt_init(float cp, float rd, float rv, float xlv, float xlf, float g) {

    NtConst c;
    const float als = nt_add(xlv, xlf);
    c.cpd  = cp;
    c.g    = g;
    c.rcpd = nt_div(1.0f, cp);
    c.c2es = nt_div(nt_mul(NT_C1ES, rd), rv);
    const float c5les = nt_mul(NT_C3LES, nt_add(NT_TMELT, -NT_C4LES));
    const float c5ies = nt_mul(NT_C3IES, nt_add(NT_TMELT, -NT_C4IES));
    c.r5alvcp = nt_mul(nt_mul(c5les, xlv), c.rcpd);
    c.r5alscp = nt_mul(nt_mul(c5ies, als), c.rcpd);
    c.ralvdcp = nt_mul(xlv, c.rcpd);
    c.ralsdcp = nt_mul(als, c.rcpd);
    c.vtmpc1  = nt_add(nt_div(rv, rd), -1.0f);
    c.rd      = rd;
    c.zrg     = nt_div(1.0f, g);
    c.ralfdcp = nt_mul(xlf, c.rcpd);
    c.c5les   = c5les;
    c.c5ies   = c5ies;
    c.alv     = xlv;
    c.als     = als;
    c.alf     = xlf;
    return c;

}

static inline float nt_foealfa(float tt) {

    const float clamped = max(NT_RTICE, min(NT_RTWAT, tt));
    const float r = nt_div(nt_add(clamped, -NT_RTICE),
                              nt_add(NT_RTWAT, -NT_RTICE));
    return min(1.0f, nt_mul(r, r));

}

static inline float nt_foelhm(float tt, thread const NtConst &c) {

    const float a = nt_foealfa(tt);
    return nt_add(nt_mul(a, c.alv),
                     nt_mul(nt_add(1.0f, -a), c.als));

}

static inline float nt_foeewm(float tt, thread const NtConst &c) {

    const float a = nt_foealfa(tt);
    const float el = exp(nt_div(
        nt_mul(NT_C3LES, nt_add(tt, -NT_TMELT)),
        nt_add(tt, -NT_C4LES)));
    const float ei = exp(nt_div(
        nt_mul(NT_C3IES, nt_add(tt, -NT_TMELT)),
        nt_add(tt, -NT_C4IES)));
    return nt_mul(c.c2es, nt_add(nt_mul(a, el),
                                       nt_mul(nt_add(1.0f, -a), ei)));

}

static inline float nt_foedem(float tt, thread const NtConst &c) {

    const float a = nt_foealfa(tt);
    const float dl = nt_add(tt, -NT_C4LES);
    const float di = nt_add(tt, -NT_C4IES);
    return nt_add(
        nt_mul(nt_mul(a, c.r5alvcp),
                  nt_div(1.0f, nt_mul(dl, dl))),
        nt_mul(nt_mul(nt_add(1.0f, -a), c.r5alscp),
                  nt_div(1.0f, nt_mul(di, di))));

}

static inline float nt_foeldcpm(float tt, thread const NtConst &c) {

    const float a = nt_foealfa(tt);
    return nt_add(nt_mul(a, c.ralvdcp),
                     nt_mul(nt_add(1.0f, -a), c.ralsdcp));

}

static inline void nt_body_ntiedtke_convert(int i, device const float *tf, device const float *qvf, device const float *uf, device const float *vf, device const float *omg, device const float *ghtl, device const float *ghti, device const float *prsl, device const float *qvftenz, device const float *thftenz, device float *ztp1, device float *zqp1, device float *zqsat, device float *pgeo, device float *pgeoh, device float *pum1, device float *pvm1, device float *pverv, device float *ptte, device float *pqte, int ncol, int nz, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;
    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define A(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    A(pgeoh, nz) = nt_mul(c.g, A(ghti, nz));
    for (int k = 0; k < nz; ++k) {
        const float q = A(qvf, k);
        A(zqp1, k) = nt_div(q, nt_add(1.0f, q));
        A(pgeo, k)  = nt_mul(c.g, A(ghtl, k));
        A(pgeoh, k) = nt_mul(c.g, A(ghti, k));
        const float t = A(tf, k);
        A(ztp1, k) = t;
        float zqs = nt_div(nt_foeewm(t, c), A(prsl, k));
        zqs = min(0.5f, zqs);
        const float zcor = nt_div(
            1.0f, nt_add(1.0f, -nt_mul(c.vtmpc1, zqs)));
        A(zqsat, k) = nt_mul(zqs, zcor);
        A(pum1, k)  = A(uf, k);
        A(pvm1, k)  = A(vf, k);
        A(pverv, k) = A(omg, k);
        A(ptte, k)  = A(thftenz, k);
        A(pqte, k)  = A(qvftenz, k);
    }
#undef A

}
kernel void ntiedtke_convert(
    constant NtiedtkeConvertResources &r [[buffer(0)]],
    constant NtiedtkeConvertParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_convert(i, r.tf, r.qvf, r.uf, r.vf, r.omg, r.ghtl, r.ghti, r.prsl, r.qvftenz, r.thftenz, r.ztp1, r.zqp1, r.zqsat, r.pgeo, r.pgeoh, r.pum1, r.pvm1, r.pverv, r.ptte, r.pqte, p.ncol, p.nz, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_cuadjtqn0(thread float *pt, thread float *pq, float psp, thread const NtConst &c) {

    const float zqp = nt_div(1.0f, psp);
    for (int pass = 0; pass < 2; ++pass) {
        float zqsat = nt_mul(nt_foeewm(*pt, c), zqp);
        zqsat = min(0.5f, zqsat);
        const float zcor = nt_div(
            1.0f, nt_add(1.0f, -nt_mul(c.vtmpc1, zqsat)));
        zqsat = nt_mul(zqsat, zcor);
        const float den = nt_add(
            1.0f, nt_mul(nt_mul(zqsat, zcor), nt_foedem(*pt, c)));
        const float zcond1 = nt_div(nt_add(*pq, -zqsat), den);
        *pt = nt_add(*pt, nt_mul(nt_foeldcpm(*pt, c), zcond1));
        *pq = nt_add(*pq, -zcond1);
    }

}

static inline void nt_body_ntiedtke_cuinin(int i, device const float *pten, device const float *pqen, device const float *pqsen, device const float *puen, device const float *pven, device const float *pverv, device const float *pgeo, device const float *paph, device const float *pgeoh, device float *ptenh, device float *pqenh, device float *pqsenh, device float *ptu, device float *pqu, device float *ptd, device float *pqd, device float *puu, device float *pvu, device float *pud, device float *pvd, device float *plu, device int *klab, device int *klwmin, int ncol, int nz, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;
    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define A(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    A(ptenh, 0) = 0.0f;
    A(pqenh, 0) = 0.0f;
    A(pqsenh, 0) = 0.0f;

    /* do jk = 2, klev  ->  k = 1 .. nz-1 */
    for (int k = 1; k < nz; ++k) {
        const float a = nt_add(nt_mul(c.cpd, A(pten, k - 1)),
                                  A(pgeo, k - 1));
        const float b = nt_add(nt_mul(c.cpd, A(pten, k)), A(pgeo, k));
        A(ptenh, k) = nt_mul(nt_add(max(a, b), -A(pgeoh, k)), c.rcpd);
        A(pqenh, k)  = A(pqen, k - 1);
        A(pqsenh, k) = A(pqsen, k - 1);
        /* if ( jk >= klev-1 .or. jk < 2 ) cycle  ->  skip k >= nz-2 */
        if (k >= nz - 2) continue;
        float t = A(ptenh, k), q = A(pqsenh, k);
        nt_cuadjtqn0(&t, &q, A(paph, k), c);
        A(ptenh, k) = t;
        A(pqsenh, k) = q;
        const float v = nt_add(min(A(pqen, k - 1), A(pqsen, k - 1)),
                                  nt_add(q, -A(pqsen, k - 1)));
        A(pqenh, k) = max(v, 0.0f);
    }

    A(ptenh, nz - 1) = nt_mul(
        nt_add(nt_add(nt_mul(c.cpd, A(pten, nz - 1)),
                            A(pgeo, nz - 1)), -A(pgeoh, nz - 1)), c.rcpd);
    A(pqenh, nz - 1) = A(pqen, nz - 1);
    A(ptenh, 0) = A(pten, 0);
    A(pqenh, 0) = A(pqen, 0);

    int lwmin = nz;                 /* 1-based klev */
    float zwmax = 0.0f;

    /* do jk = klevm1, 2, -1  ->  k = nz-2 .. 1 */
    for (int k = nz - 2; k >= 1; --k) {
        const float z1 = nt_add(nt_mul(c.cpd, A(ptenh, k)),
                                   A(pgeoh, k));
        const float z2 = nt_add(nt_mul(c.cpd, A(ptenh, k + 1)),
                                   A(pgeoh, k + 1));
        A(ptenh, k) = nt_mul(nt_add(max(z1, z2), -A(pgeoh, k)),
                                c.rcpd);
    }

    /* do jk = klev, 3, -1  ->  k = nz-1 .. 2; klwmin stays 1-BASED */
    for (int k = nz - 1; k >= 2; --k) {
        if (A(pverv, k) < zwmax) {
            zwmax = A(pverv, k);
            lwmin = k + 1;
        }
    }
    klwmin[i] = lwmin;

    for (int k = 0; k < nz; ++k) {
        const int ik = (k == 0) ? 0 : k - 1;   /* ik = jk-1, 1 when jk = 1 */
        A(ptu, k) = A(ptenh, k);
        A(ptd, k) = A(ptenh, k);
        A(pqu, k) = A(pqenh, k);
        A(pqd, k) = A(pqenh, k);
        A(plu, k) = 0.0f;
        A(puu, k) = A(puen, ik);
        A(pud, k) = A(puen, ik);
        A(pvu, k) = A(pven, ik);
        A(pvd, k) = A(pven, ik);
        A(klab, k) = 0;
    }
#undef A

}
kernel void ntiedtke_cuinin(
    constant NtiedtkeCuininResources &r [[buffer(0)]],
    constant NtiedtkeCuininParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_cuinin(i, r.pten, r.pqen, r.pqsen, r.puen, r.pven, r.pverv, r.pgeo, r.paph, r.pgeoh, r.ptenh, r.pqenh, r.pqsenh, r.ptu, r.pqu, r.ptd, r.pqd, r.puu, r.pvu, r.pud, r.pvd, r.plu, r.klab, r.klwmin, p.ncol, p.nz, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_cuadjtqn1(thread float *pt, thread float *pq, float psp, thread const NtConst &c) {

    const float zqp = nt_div(1.0f, psp);
    float zl = nt_div(1.0f, nt_add(*pt, -NT_C4LES));
    float zi = nt_div(1.0f, nt_add(*pt, -NT_C4IES));
    float a = nt_foealfa(*pt);
    float el = exp(nt_mul(nt_mul(NT_C3LES,
                                           nt_add(*pt, -NT_TMELT)), zl));
    float ei = exp(nt_mul(nt_mul(NT_C3IES,
                                           nt_add(*pt, -NT_TMELT)), zi));
    float zqsat = nt_mul(c.c2es,
        nt_add(nt_mul(a, el), nt_mul(nt_add(1.0f, -a), ei)));
    zqsat = nt_mul(zqsat, zqp);
    zqsat = min(0.5f, zqsat);
    float zcor = nt_add(1.0f, -nt_mul(c.vtmpc1, zqsat));
    float zf = nt_add(
        nt_mul(nt_mul(a, c.r5alvcp), nt_mul(zl, zl)),
        nt_mul(nt_mul(nt_add(1.0f, -a), c.r5alscp),
                  nt_mul(zi, zi)));
    const float zcond = nt_div(
        nt_add(nt_mul(*pq, nt_mul(zcor, zcor)),
                  -nt_mul(zqsat, zcor)),
        nt_add(nt_mul(zcor, zcor), nt_mul(zqsat, zf)));
    if (zcond > 0.0f) {
        *pt = nt_add(*pt, nt_mul(nt_foeldcpm(*pt, c), zcond));
        *pq = nt_add(*pq, -zcond);
        zl = nt_div(1.0f, nt_add(*pt, -NT_C4LES));
        zi = nt_div(1.0f, nt_add(*pt, -NT_C4IES));
        a = nt_foealfa(*pt);
        el = exp(nt_mul(nt_mul(NT_C3LES,
                                         nt_add(*pt, -NT_TMELT)), zl));
        ei = exp(nt_mul(nt_mul(NT_C3IES,
                                         nt_add(*pt, -NT_TMELT)), zi));
        zqsat = nt_mul(c.c2es,
            nt_add(nt_mul(a, el), nt_mul(nt_add(1.0f, -a), ei)));
        zqsat = nt_mul(zqsat, zqp);
        zqsat = min(0.5f, zqsat);
        zcor = nt_add(1.0f, -nt_mul(c.vtmpc1, zqsat));
        zf = nt_add(
            nt_mul(nt_mul(a, c.r5alvcp), nt_mul(zl, zl)),
            nt_mul(nt_mul(nt_add(1.0f, -a), c.r5alscp),
                      nt_mul(zi, zi)));
        float zcond1 = nt_div(
            nt_add(nt_mul(*pq, nt_mul(zcor, zcor)),
                      -nt_mul(zqsat, zcor)),
            nt_add(nt_mul(zcor, zcor), nt_mul(zqsat, zf)));
        if (fabs(zcond) < 1.0e-20f) zcond1 = 0.0f;
        *pt = nt_add(*pt, nt_mul(nt_foeldcpm(*pt, c), zcond1));
        *pq = nt_add(*pq, -zcond1);
    }

}

static inline void nt_cuadjtqn1_device(device float *pt, device float *pq, float psp, thread const NtConst &c) {

    const float zqp = nt_div(1.0f, psp);
    float zl = nt_div(1.0f, nt_add(*pt, -NT_C4LES));
    float zi = nt_div(1.0f, nt_add(*pt, -NT_C4IES));
    float a = nt_foealfa(*pt);
    float el = exp(nt_mul(nt_mul(NT_C3LES,
                                           nt_add(*pt, -NT_TMELT)), zl));
    float ei = exp(nt_mul(nt_mul(NT_C3IES,
                                           nt_add(*pt, -NT_TMELT)), zi));
    float zqsat = nt_mul(c.c2es,
        nt_add(nt_mul(a, el), nt_mul(nt_add(1.0f, -a), ei)));
    zqsat = nt_mul(zqsat, zqp);
    zqsat = min(0.5f, zqsat);
    float zcor = nt_add(1.0f, -nt_mul(c.vtmpc1, zqsat));
    float zf = nt_add(
        nt_mul(nt_mul(a, c.r5alvcp), nt_mul(zl, zl)),
        nt_mul(nt_mul(nt_add(1.0f, -a), c.r5alscp),
                  nt_mul(zi, zi)));
    const float zcond = nt_div(
        nt_add(nt_mul(*pq, nt_mul(zcor, zcor)),
                  -nt_mul(zqsat, zcor)),
        nt_add(nt_mul(zcor, zcor), nt_mul(zqsat, zf)));
    if (zcond > 0.0f) {
        *pt = nt_add(*pt, nt_mul(nt_foeldcpm(*pt, c), zcond));
        *pq = nt_add(*pq, -zcond);
        zl = nt_div(1.0f, nt_add(*pt, -NT_C4LES));
        zi = nt_div(1.0f, nt_add(*pt, -NT_C4IES));
        a = nt_foealfa(*pt);
        el = exp(nt_mul(nt_mul(NT_C3LES,
                                         nt_add(*pt, -NT_TMELT)), zl));
        ei = exp(nt_mul(nt_mul(NT_C3IES,
                                         nt_add(*pt, -NT_TMELT)), zi));
        zqsat = nt_mul(c.c2es,
            nt_add(nt_mul(a, el), nt_mul(nt_add(1.0f, -a), ei)));
        zqsat = nt_mul(zqsat, zqp);
        zqsat = min(0.5f, zqsat);
        zcor = nt_add(1.0f, -nt_mul(c.vtmpc1, zqsat));
        zf = nt_add(
            nt_mul(nt_mul(a, c.r5alvcp), nt_mul(zl, zl)),
            nt_mul(nt_mul(nt_add(1.0f, -a), c.r5alscp),
                      nt_mul(zi, zi)));
        float zcond1 = nt_div(
            nt_add(nt_mul(*pq, nt_mul(zcor, zcor)),
                      -nt_mul(zqsat, zcor)),
            nt_add(nt_mul(zcor, zcor), nt_mul(zqsat, zf)));
        if (fabs(zcond) < 1.0e-20f) zcond1 = 0.0f;
        *pt = nt_add(*pt, nt_mul(nt_foeldcpm(*pt, c), zcond1));
        *pq = nt_add(*pq, -zcond1);
    }

}

static inline void nt_cloud_base(int jk, int klev, int ncol, int i, device float *ptu, device float *pqu, device float *plu, device float *kup, device int *klab, device const float *paph, thread int *kcbot, thread const NtConst &c) {

    const int ik = jk + 1;
    float zqsu = nt_div(nt_foeewm(NTC(ptu, ik), c), NTC(paph, ik));
    zqsu = min(0.5f, zqsu);
    float zcor = nt_div(1.0f, nt_add(1.0f, -nt_mul(c.vtmpc1, zqsu)));
    zqsu = nt_mul(zqsu, zcor);
    const float zdq = min(0.0f, nt_add(NTC(pqu, ik), -zqsu));
    const float zalfaw = nt_foealfa(NTC(ptu, ik));
    const float dl = nt_add(NTC(ptu, ik), -NT_C4LES);
    const float di = nt_add(NTC(ptu, ik), -NT_C4IES);
    const float zfacw = nt_div(c.c5les, nt_mul(dl, dl));
    const float zfaci = nt_div(c.c5ies, nt_mul(di, di));
    const float zfac = nt_add(nt_mul(zalfaw, zfacw),
                                 nt_mul(nt_add(1.0f, -zalfaw), zfaci));
    const float zesdp = nt_div(nt_foeewm(NTC(ptu, ik), c), NTC(paph, ik));
    zcor = nt_div(1.0f, nt_add(1.0f, -nt_mul(c.vtmpc1, zesdp)));
    const float zdqsdt = nt_mul(nt_mul(zfac, zcor), zqsu);
    const float zdtdp = nt_div(nt_mul(c.rd, NTC(ptu, ik)),
                                  nt_mul(c.cpd, NTC(paph, ik)));
    const float zdp = nt_div(zdq, nt_mul(zdqsdt, zdtdp));
    const float zcbase = nt_add(NTC(paph, ik), zdp);
    const float zpdifftop = nt_add(zcbase, -NTC(paph, jk));
    const float zpdiffbot = nt_add(NTC(paph, jk + 1), -zcbase);
    if (zpdifftop > zpdiffbot && NTC(kup, jk + 1) > 0.0f) {
        const int ikb = min(klev - 1, jk + 1);
        NTC(klab, ikb) = 2;
        NTC(klab, jk) = 2;
        *kcbot = ikb;
        NTC(plu, jk + 1) = 1.0e-8f;
    } else if (zpdifftop <= zpdiffbot && NTC(kup, jk) > 0.0f) {
        NTC(klab, jk) = 2;
        *kcbot = jk;
    }

}

static inline void nt_ascent_step(int jk, int levels, int klev, int ncol, int i, bool deep, device float *ptu, device float *pqu, device float *plu, device float *dh, device float *dhen, device float *kup, device float *vptu, device float *vten, device float *zbuo, device float *abuoy, device int *klab, device const float *ptenh, device const float *pqenh, device const float *pgeo, device const float *pgeoh, device const float *paph, device const float *pqsen, thread int *kcbot, thread int *kctop, thread bool *loflag, thread bool *lldcum, thread const NtConst &c) {

    float eta;
    if (deep) {
        const float r = nt_div(NTC(pqsen, jk), NTC(pqsen, levels));
        const float fscale = min(1.0f, nt_mul(nt_mul(r, r), r));
        eta = nt_mul(1.75e-3f, fscale);
    } else {
        eta = nt_add(nt_div(0.8f, nt_mul(NTC(pgeo, jk), c.zrg)),
                        2.0e-4f);
    }
    const float dz = nt_mul(nt_add(NTC(pgeoh, jk),
                                         -NTC(pgeoh, jk + 1)), c.zrg);
    const float coef = nt_mul(nt_mul(0.5f, eta), dz);
    NTC(dhen, jk) = nt_add(NTC(pgeoh, jk),
                              nt_mul(c.cpd, NTC(ptenh, jk)));
    NTC(dh, jk) = nt_div(
        nt_add(nt_mul(coef, nt_add(NTC(dhen, jk + 1), NTC(dhen, jk))),
                  nt_mul(nt_add(1.0f, -coef), NTC(dh, jk + 1))),
        nt_add(1.0f, coef));
    NTC(pqu, jk) = nt_div(
        nt_add(nt_mul(coef, nt_add(NTC(pqenh, jk + 1),
                                            NTC(pqenh, jk))),
                  nt_mul(nt_add(1.0f, -coef), NTC(pqu, jk + 1))),
        nt_add(1.0f, coef));
    NTC(ptu, jk) = nt_mul(nt_add(NTC(dh, jk), -NTC(pgeoh, jk)), c.rcpd);
    const float zqold = NTC(pqu, jk);

    float t = NTC(ptu, jk), q = NTC(pqu, jk);
    nt_cuadjtqn1(&t, &q, NTC(paph, jk), c);
    NTC(ptu, jk) = t;
    NTC(pqu, jk) = q;

    const float zdq = max(nt_add(zqold, -NTC(pqu, jk)), 0.0f);
    NTC(plu, jk) = nt_add(NTC(plu, jk + 1), zdq);
    const float zlglac = nt_mul(zdq,
        nt_add(nt_add(1.0f, -nt_foealfa(NTC(ptu, jk))),
                  -nt_add(1.0f, -nt_foealfa(NTC(ptu, jk + 1)))));
    NTC(plu, jk) = deep ? nt_mul(0.5f, NTC(plu, jk))
                        : min(NTC(plu, jk), 5.0e-3f);
    NTC(dh, jk) = nt_add(NTC(pgeoh, jk),
        nt_mul(c.cpd, nt_add(NTC(ptu, jk),
                                   nt_mul(c.ralfdcp, zlglac))));
    NTC(vptu, jk) = nt_add(
        nt_mul(NTC(ptu, jk),
                  nt_add(nt_add(1.0f, nt_mul(c.vtmpc1,
                                                      NTC(pqu, jk))),
                            -NTC(plu, jk))),
        nt_mul(c.ralfdcp, zlglac));
    NTC(vten, jk) = nt_mul(NTC(ptenh, jk),
        nt_add(1.0f, nt_mul(c.vtmpc1, NTC(pqenh, jk))));
    NTC(zbuo, jk) = nt_div(nt_add(NTC(vptu, jk), -NTC(vten, jk)),
                              NTC(vten, jk));
    NTC(abuoy, jk) = nt_mul(
        nt_mul(nt_add(NTC(zbuo, jk), NTC(zbuo, jk + 1)), 0.5f), c.g);
    const float atop1 = nt_add(1.0f, -nt_mul(2.0f, coef));
    const float atop2 = nt_mul(nt_mul(2.0f, dz), NTC(abuoy, jk));
    const float abot = nt_add(1.0f, nt_mul(2.0f, coef));
    NTC(kup, jk) = nt_div(
        nt_add(nt_mul(atop1, NTC(kup, jk + 1)), atop2), abot);

    if (NTC(plu, jk) > 0.0f && NTC(klab, jk + 1) == 1) {
        nt_cloud_base(jk, klev, ncol, i, ptu, pqu, plu, kup, klab,
                      paph, kcbot, c);
    }
    if (NTC(kup, jk) < 0.0f) {
        *loflag = false;
        if (NTC(plu, jk + 1) > 0.0f) {
            *kctop = jk;
            *lldcum = true;
        } else {
            *lldcum = false;
        }
    } else {
        NTC(klab, jk) = (NTC(plu, jk) > 0.0f) ? 2 : 1;
    }

}

static inline void nt_body_ntiedtke_cutypen(int i, device const float *pqen, device const float *ptenh, device const float *pqenh, device const float *pgeoh, device const float *paph, device const float *pgeo, device const float *pqsen, device const float *pap, device const float *pten, device const float *hfx, device const float *qfx, device float *cutu, device float *cuqu, device float *culu, device int *culab, device float *scr, device int *scr_i, device int *ldcum_o, device int *ktype_o, device int *cubot_o, device int *cutop_o, device int *kdpl_o, device float *wbase_o, int ncol, int nz, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;
    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
    const int klev = nz, klevm1 = nz - 1;
    const size_t stride = (ulong)(nz + 2) * (ulong)ncol;

    device float *ptu = scr + 0 * stride, *pqu = scr + 1 * stride;
    device float *plu = scr + 2 * stride, *dh = scr + 3 * stride;
    device float *dhen = scr + 4 * stride, *kup = scr + 5 * stride;
    device float *vptu = scr + 6 * stride, *vten = scr + 7 * stride;
    device float *zbuo = scr + 8 * stride, *abuoy = scr + 9 * stride;
    device int *klab = scr_i;

    int kcbot = klev, kctop = klev, kdpl = klev, ktype = 0;
    int cubot = -1, cutop = -1;
    float wbase = 0.0f;
    bool ldcum = false, lldcum = false, loflag = true;

    /* ---- shallow pass (:1332-1345) --------------------------------- */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(plu, jk) = NTC(culu, jk);
        NTC(ptu, jk) = NTC(cutu, jk);
        NTC(pqu, jk) = NTC(cuqu, jk);
        NTC(klab, jk) = NTC(culab, jk);
        NTC(dh, jk) = 0.0f;  NTC(dhen, jk) = 0.0f;  NTC(kup, jk) = 0.0f;
        NTC(vptu, jk) = 0.0f; NTC(vten, jk) = 0.0f;
        NTC(zbuo, jk) = 0.0f; NTC(abuoy, jk) = 0.0f;
    }

    for (int jk = klevm1; jk >= 2; --jk) {
        if (jk == klevm1) {
            const float rho = nt_div(NTC(pap, klev),
                nt_mul(c.rd, nt_mul(NTC(pten, klev),
                    nt_add(1.0f, nt_mul(c.vtmpc1, NTC(pqen, klev))))));
            const float part1 = nt_div(
                nt_mul(nt_mul(1.5f, 0.4f), NTC(pgeo, klev)),
                nt_mul(rho, NTC(pten, klev)));
            const float part2 = nt_add(
                -nt_mul(hfx[i], c.rcpd),
                -nt_mul(nt_mul(c.vtmpc1, NTC(pten, klev)), qfx[i]));
            const float root = nt_add(0.001f, -nt_mul(part1, part2));
            if (part2 < 0.0f) {
                /* conw = 1.2*root**t13 -- a genuine cube root through powf,
                 * so pow(glibc's) and not CUDA's. */
                const float conw = nt_mul(1.2f, pow(root, NT_T13));
                const float deltt = max(nt_div(
                    nt_mul(1.5f, hfx[i]),
                    nt_mul(nt_mul(rho, c.cpd), conw)), 0.0f);
                const float deltq = max(nt_div(
                    nt_mul(1.5f, qfx[i]), nt_mul(rho, conw)), 0.0f);
                NTC(kup, klev) = nt_mul(0.5f, nt_mul(conw, conw));
                NTC(pqu, klev) = nt_add(NTC(pqenh, klev), deltq);
                NTC(dhen, klev) = nt_add(NTC(pgeoh, klev),
                                            nt_mul(NTC(ptenh, klev),
                                                      c.cpd));
                NTC(dh, klev) = nt_add(NTC(dhen, klev),
                                          nt_mul(deltt, c.cpd));
                NTC(ptu, klev) = nt_mul(
                    nt_add(NTC(dh, klev), -NTC(pgeoh, klev)), c.rcpd);
                NTC(vptu, klev) = nt_mul(NTC(ptu, klev),
                    nt_add(1.0f, nt_mul(c.vtmpc1, NTC(pqu, klev))));
                NTC(vten, klev) = nt_mul(NTC(ptenh, klev),
                    nt_add(1.0f, nt_mul(c.vtmpc1, NTC(pqenh, klev))));
                NTC(zbuo, klev) = nt_div(
                    nt_add(NTC(vptu, klev), -NTC(vten, klev)),
                    NTC(vten, klev));
                NTC(klab, klev) = 1;
            } else {
                loflag = false;
            }
        }
        if (!loflag) break;                        /* is == 0 -> exit */
        nt_ascent_step(jk, 0, klev, ncol, i, false,
                       ptu, pqu, plu, dh, dhen, kup, vptu, vten, zbuo,
                       abuoy, klab, ptenh, pqenh, pgeo, pgeoh, paph, pqsen,
                       &kcbot, &kctop, &loflag, &lldcum, c);
    }

    if (nt_add(NTC(paph, kcbot), -NTC(paph, kctop)) > NT_ZDNOPRC)
        lldcum = false;
    if (lldcum) {
        ktype = 2;  ldcum = true;
        wbase = sqrt(max(nt_mul(2.0f, NTC(kup, kcbot)), 0.0f));
        cubot = kcbot;  cutop = kctop;  kdpl = klev;
    } else {
        cutop = -1;  cubot = -1;  kdpl = klev - 1;
        ldcum = false;  wbase = 0.0f;
    }
    for (int jk = klev; jk >= 1; --jk) {
        if (jk >= kctop) {
            NTC(culab, jk) = NTC(klab, jk);
            NTC(cutu, jk) = NTC(ptu, jk);
            NTC(cuqu, jk) = NTC(pqu, jk);
            NTC(culu, jk) = NTC(plu, jk);
        }
    }

    /* ---- deep pass (:1517-1746) ------------------------------------ */
    const float deltt_d = 0.2f, deltq_d = 1.0e-4f;
    bool deepflag = false;
    int itoppacel = klev;
    for (int jk = klev; jk >= 1; --jk) {
        if (nt_add(NTC(paph, klev + 1), -NTC(paph, jk)) < 350.0e2f)
            itoppacel = jk;
    }

    for (int levels = klevm1 - 1; levels >= klev / 2 + 1; --levels) {
        // Inactive trials cannot update outputs. Keep the final scratch reset.
        if ((deepflag || levels < itoppacel) && levels > klev / 2 + 1)
            continue;
        // Only the final trial needs to leave every unused scratch row clear.
        if (levels == klev / 2 + 1) {
            for (int jk = 1; jk <= klev; ++jk) {
                NTC(plu, jk) = 0.0f; NTC(ptu, jk) = 0.0f; NTC(pqu, jk) = 0.0f;
                NTC(dh, jk) = 0.0f;  NTC(dhen, jk) = 0.0f; NTC(kup, jk) = 0.0f;
                NTC(vptu, jk) = 0.0f; NTC(vten, jk) = 0.0f;
                NTC(abuoy, jk) = 0.0f; NTC(zbuo, jk) = 0.0f;
                NTC(klab, jk) = 0;
            }
        }
        // The seed liquid and each visited label are read before replacement.
        NTC(plu, levels + 1) = 0.0f;
        kcbot = levels;  kctop = levels;
        lldcum = false;
        bool resetflag = false;
        loflag = (!deepflag) && (levels >= itoppacel);

        for (int jk = levels; jk >= 2; --jk) {
            if (!loflag) break;
            if (jk == levels) {
                float tmix, qmix, zmix;
                if (nt_add(NTC(paph, klev + 1), -NTC(paph, jk)) < 60.0e2f) {
                    tmix = 0.0f; qmix = 0.0f; zmix = 0.0f;
                    float pmix = 0.0f;
                    for (int nk = jk + 2; nk >= jk; --nk) {
                        if (pmix < 50.0e2f) {
                            const float dp = nt_add(NTC(paph, nk),
                                                       -NTC(paph, nk - 1));
                            tmix = nt_add(tmix,
                                             nt_mul(dp, NTC(ptenh, nk)));
                            qmix = nt_add(qmix,
                                             nt_mul(dp, NTC(pqenh, nk)));
                            zmix = nt_add(zmix,
                                             nt_mul(dp, NTC(pgeoh, nk)));
                            pmix = nt_add(pmix, dp);
                        }
                    }
                    tmix = nt_div(tmix, pmix);
                    qmix = nt_div(qmix, pmix);
                    zmix = nt_div(zmix, pmix);
                } else {
                    tmix = NTC(ptenh, jk + 1);
                    qmix = NTC(pqenh, jk + 1);
                    zmix = NTC(pgeoh, jk + 1);
                }
                NTC(pqu, jk + 1) = nt_add(qmix, deltq_d);
                NTC(dhen, jk + 1) = nt_add(zmix, nt_mul(tmix, c.cpd));
                NTC(dh, jk + 1) = nt_add(NTC(dhen, jk + 1),
                                            nt_mul(deltt_d, c.cpd));
                NTC(ptu, jk + 1) = nt_mul(
                    nt_add(NTC(dh, jk + 1), -NTC(pgeoh, jk + 1)), c.rcpd);
                NTC(kup, jk + 1) = 0.5f;
                NTC(klab, jk + 1) = 1;
                NTC(vptu, jk + 1) = nt_mul(NTC(ptu, jk + 1),
                    nt_add(1.0f, nt_mul(c.vtmpc1, NTC(pqu, jk + 1))));
                NTC(vten, jk + 1) = nt_mul(NTC(ptenh, jk + 1),
                    nt_add(1.0f, nt_mul(c.vtmpc1, NTC(pqenh, jk + 1))));
                NTC(zbuo, jk + 1) = nt_div(
                    nt_add(NTC(vptu, jk + 1), -NTC(vten, jk + 1)),
                    NTC(vten, jk + 1));
            }
            NTC(klab, jk) = 0;
            nt_ascent_step(jk, levels, klev, ncol, i, true,
                           ptu, pqu, plu, dh, dhen, kup, vptu, vten, zbuo,
                           abuoy, klab, ptenh, pqenh, pgeo, pgeoh, paph,
                           pqsen, &kcbot, &kctop, &loflag, &lldcum, c);
        }

        if (nt_add(NTC(paph, kcbot), -NTC(paph, kctop)) < NT_ZDNOPRC)
            lldcum = false;
        if (lldcum) {
            ktype = 1;  ldcum = true;  deepflag = true;
            wbase = sqrt(max(nt_mul(2.0f, NTC(kup, kcbot)), 0.0f));
            cubot = kcbot;  cutop = kctop;
            kdpl = levels + 1;
            resetflag = true;
        }
        if (resetflag) {
            const int ikt = kctop, ikb = kdpl;
            for (int jk = klev; jk >= 1; --jk) {
                if (jk >= ikt && jk <= ikb) {
                    NTC(culab, jk) = NTC(klab, jk);
                    NTC(cutu, jk) = NTC(ptu, jk);
                    NTC(cuqu, jk) = NTC(pqu, jk);
                    NTC(culu, jk) = NTC(plu, jk);
                } else {
                    NTC(culab, jk) = 1;
                    NTC(cutu, jk) = NTC(ptenh, jk);
                    NTC(cuqu, jk) = NTC(pqenh, jk);
                    NTC(culu, jk) = 0.0f;
                }
                if (jk < ikt) NTC(culab, jk) = 0;
            }
        }
    }

    ldcum_o[i] = ldcum ? 1 : 0;
    ktype_o[i] = ktype;
    cubot_o[i] = cubot;
    cutop_o[i] = cutop;
    kdpl_o[i] = kdpl;
    wbase_o[i] = wbase;

}
kernel void ntiedtke_cutypen(
    constant NtiedtkeCutypenResources &r [[buffer(0)]],
    constant NtiedtkeCutypenParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_cutypen(i, r.pqen, r.ptenh, r.pqenh, r.pgeoh, r.paph, r.pgeo, r.pqsen, r.pap, r.pten, r.hfx, r.qfx, r.cutu, r.cuqu, r.culu, r.culab, r.scr, r.scr_i, r.ldcum_o, r.ktype_o, r.cubot_o, r.cutop_o, r.kdpl_o, r.wbase_o, p.ncol, p.nz, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline bool nt_cubasmcn(int kk, int ncol, int i, device const float *pten, device const float *pqen, device const float *pqsen, device const float *pverv, device const float *pgeo, device const float *pgeoh, device int *ldcum, device int *ktype, device int *klab, device float *pmfu, device float *pmfub, device int *kcbot, device float *ptu, device float *pqu, device float *plu, device float *pmfus, device float *pmfuq, device float *pmful, device float *pdmfup, thread const NtConst &c) {

    /* if(.not.ldcum .and. klab(kk+1) == 0) */
    if (ldcum[i] != 0 || NTC(klab, kk + 1) != 0) return false;
    /* lmfmid is a parameter, .true.  pgeo*zrg is height in metres, so the
     * window below is 500 m to 10 km. */
    const float hgt = nt_mul(NTC(pgeo, kk), c.zrg);
    if (!(NTC(pqen, kk) > nt_mul(0.80f, NTC(pqsen, kk))
          && hgt > 5.0e2f && hgt < 1.0e4f)) return false;

    NTC(ptu, kk + 1) = nt_mul(
        nt_add(nt_add(nt_mul(c.cpd, NTC(pten, kk)), NTC(pgeo, kk)),
                  -NTC(pgeoh, kk + 1)), c.rcpd);
    NTC(pqu, kk + 1) = NTC(pqen, kk);
    NTC(plu, kk + 1) = 0.0f;
    float zzzmb = max(NT_CMFCMIN, -nt_mul(NTC(pverv, kk), c.zrg));
    zzzmb = min(zzzmb, NT_CMFCMAX);
    pmfub[i] = zzzmb;
    NTC(pmfu, kk + 1) = pmfub[i];
    NTC(pmfus, kk + 1) = nt_mul(pmfub[i],
        nt_add(nt_mul(c.cpd, NTC(ptu, kk + 1)), NTC(pgeoh, kk + 1)));
    NTC(pmfuq, kk + 1) = nt_mul(pmfub[i], NTC(pqu, kk + 1));
    NTC(pmful, kk + 1) = 0.0f;
    NTC(pdmfup, kk + 1) = 0.0f;
    kcbot[i] = kk;
    NTC(klab, kk + 1) = 1;
    ktype[i] = 3;
    return true;

}

static inline void nt_cuentrn(int kk, int ncol, int i, device const int *kcbot, device const int *ldcum, bool ldwork, device const float *pgeoh, device const float *pmfu, thread float *pdmfen, thread float *pdmfde, thread const NtConst &c) {

    if (!ldwork) return;
    *pdmfen = 0.0f;
    *pdmfde = 0.0f;
    const float zentr = 0.0f;
    if (ldcum[i] == 0) return;
    const float zdz = nt_mul(
        nt_add(NTC(pgeoh, kk), -NTC(pgeoh, kk + 1)), c.zrg);
    const float zmf = nt_mul(NTC(pmfu, kk + 1), zdz);
    if (kk < kcbot[i]) {
        *pdmfen = nt_mul(zentr, zmf);
        *pdmfde = nt_mul(0.75e-4f, zmf);
    }

}

static inline void nt_body_ntiedtke_midlevel(int i, device const float *pten, device const float *pqen, device const float *pqsen, device const float *pverv, device const float *pgeo, device const float *pgeoh, device int *ldcum, device int *ktype, device int *kcbot, device int *klab, device float *plrain, device float *pmfu, device float *pmfub, device float *ptu, device float *pqu, device float *plu, device float *pmfus, device float *pmfuq, device float *pmful, device float *pdmfup, device float *dmfen, device float *dmfde, int ncol, int nz, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;
    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);

    for (int kk = nz - 1; kk >= 3; --kk) {
        if (nt_cubasmcn(kk, ncol, i, pten, pqen, pqsen, pverv, pgeo, pgeoh,
                    ldcum, ktype, klab,
                    pmfu, pmfub, kcbot,
                    ptu, pqu, plu, pmfus, pmfuq, pmful, pdmfup, c))
            NTC(plrain, kk + 1) = 0.0f;
        /* cuentrn's outputs are per-column scalars in the reference; the
         * fixture captures them per level, so they are taken by pointer
         * and stored.  ONE body, called once -- an inlined second copy
         * here would be exactly the two-clone shape ptxas can contract
         * differently. */
        float e0, d0;
        nt_cuentrn(kk, ncol, i, kcbot, ldcum, true, pgeoh, pmfu,
                   &e0, &d0, c);
        NTC(dmfen, kk) = e0;
        NTC(dmfde, kk) = d0;
    }

}
kernel void ntiedtke_midlevel(
    constant NtiedtkeMidlevelResources &r [[buffer(0)]],
    constant NtiedtkeMidlevelParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_midlevel(i, r.pten, r.pqen, r.pqsen, r.pverv, r.pgeo, r.pgeoh, r.ldcum, r.ktype, r.kcbot, r.klab, r.plrain, r.pmfu, r.pmfub, r.ptu, r.pqu, r.plu, r.pmfus, r.pmfuq, r.pmful, r.pdmfup, r.dmfen, r.dmfde, p.ncol, p.nz, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_body_ntiedtke_mfub(int i, device const float *ptte, device const float *pqte, device const float *paph, device const float *puen, device const float *pven, device const float *ptu, device const float *pqu, device const float *plu, device const float *ztenh, device const float *zqenh, device const int *ktype, device const int *kcbot, device const int *lndj, device int *ldcum, device float *zdhpbl, device float *upbl, device float *zmfub, int tiedtke_closure, int ncol, int nz, float dt, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;
    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
    const int klev = nz;

    /* :468-469.  zcons2 = 3/(g*dt); the 3 is not a typo -- zcons at :468 is
     * the same quantity with 1. */
    const float zcons2 = nt_div(3.0f, nt_mul(c.g, dt));

    float dhpbl = 0.0f;
    float zdqpbl = 0.0f;   /* cu6 :832-839 */
    float up = 0.0f;
    bool ld = ldcum[i] != 0;

    /* do jk = 2, klev */
    for (int jk = 2; jk <= klev; ++jk) {
        if (!(jk >= kcbot[i] && ld)) continue;
        const float dp = nt_add(NTM(paph, jk + 1), -NTM(paph, jk));
        dhpbl = nt_add(dhpbl, nt_mul(
            nt_add(nt_mul(c.alv, NTM(pqte, jk)),
                      nt_mul(c.cpd, NTM(ptte, jk))), dp));
        /* module_cu_tiedtke.F:839 -- MOISTURE convergence alone, the
         * quantity cu6's deep first guess divides.  Accumulated in the
         * same loop over the same dp so the two agree cell for cell. */
        zdqpbl = nt_add(zdqpbl, nt_mul(NTM(pqte, jk), dp));
        if (lndj[i] == 0) {
            const float u = NTM(puen, jk), v = NTM(pven, jk);
            const float wspeed = sqrt(nt_add(nt_mul(u, u),
                                                 nt_mul(v, v)));
            up = nt_add(up, nt_mul(wspeed, dp));
        }
    }

    float mfub = 0.0f;
    if (ld) {
        const int ikb = kcbot[i];
        const float zmfmax = nt_mul(
            nt_add(NTM(paph, ikb), -NTM(paph, ikb - 1)), zcons2);
        if (ktype[i] == 1) {
            if (tiedtke_closure) {
                /* module_cu_tiedtke.F:855-866.  cu6's deep first guess is
                 * MOISTURE CONVERGENCE, not cu16's geometric 0.1*zmfmax.
                 * Same zqumqe/zdqmin construction as the shallow arm
                 * below; the 0.01 floor is cu6 :862 and the cap :866. */
                const float zqumqe = nt_add(
                    nt_add(NTM(pqu, ikb), NTM(plu, ikb)), -NTM(zqenh, ikb));
                const float zdqmin = max(
                    nt_mul(0.01f, NTM(zqenh, ikb)), 1.0e-10f);
                if (zdqpbl > 0.0f && zqumqe > zdqmin) {
                    mfub = nt_div(zdqpbl,
                                     nt_mul(c.g, max(zqumqe, zdqmin)));
                    mfub = min(mfub, zmfmax);
                } else {
                    mfub = 0.01f;
                }
            } else {
                mfub = nt_mul(0.1f, zmfmax);
            }
        } else if (ktype[i] == 2) {
            const float zqumqe = nt_add(
                nt_add(NTM(pqu, ikb), NTM(plu, ikb)), -NTM(zqenh, ikb));
            const float zdqmin = max(
                nt_mul(0.01f, NTM(zqenh, ikb)), 1.0e-10f);
            float zdh = nt_add(
                nt_mul(c.cpd, nt_add(NTM(ptu, ikb), -NTM(ztenh, ikb))),
                nt_mul(c.alv, zqumqe));
            zdh = nt_mul(c.g, max(zdh, nt_mul(1.0e5f, zdqmin)));
            if (dhpbl > 0.0f) {
                mfub = nt_div(dhpbl, zdh);
                mfub = min(mfub, zmfmax);
            } else {
                mfub = nt_mul(0.1f, zmfmax);
                ld = false;                    /* :536 -- clears ldcum */
            }
        }
        /* ktype 0 or 3 with ldcum true is unreachable from cumastrn --
         * cutypen produces only 0/1/2 and ldcum is false whenever ktype is
         * 0 -- and the reference assigns nothing there, leaving zmfub at
         * its initialised zero.  So does this. */
    }

    zdhpbl[i] = dhpbl;
    upbl[i] = up;
    zmfub[i] = mfub;
    ldcum[i] = ld ? 1 : 0;

}
kernel void ntiedtke_mfub(
    constant NtiedtkeMfubResources &r [[buffer(0)]],
    constant NtiedtkeMfubParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_mfub(i, r.ptte, r.pqte, r.paph, r.puen, r.pven, r.ptu, r.pqu, r.plu, r.ztenh, r.zqenh, r.ktype, r.kcbot, r.lndj, r.ldcum, r.zdhpbl, r.upbl, r.zmfub, p.tiedtke_closure, p.ncol, p.nz, p.dt, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_body_ntiedtke_closure(int i, device const int *ldcum, device const int *ktype, device const int *kcbot, device const int *kctop, device const int *kdpl, device const int *loddraf, device const int *lndj, device const float *wup, device const float *zmfub, device const float *zdhpbl, device const float *scale_fac, device const float *scale_fac2, device const float *pgeoh, device const float *paph, device const float *pap, device const float *pgeo, device const float *pten, device const float *pqen, device const float *ptenh, device const float *pqenh, device const float *ptu, device const float *pqu, device const float *plu, device const float *pmfu, device const float *ptd, device const float *pqd, device const float *ptte, device const float *pqte, device float *pmfd, device float *pmfds, device float *pmfdq, device float *pdmfdp, device float *pmfdde_rate, device float *zheat, device float *zcape, device float *zcape1, device float *zcape2, device float *ztauc, device float *ztaubl, device float *ztau_o, device float *upbl, device float *zmfub1, int tiedtke_closure, int ncol, int nz, float dt, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;
    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
    const int klev = nz;
    const float zcons2 = nt_div(3.0f, nt_mul(c.g, dt));
    const bool ld = ldcum[i] != 0;
    const int kt = ktype[i];
    const bool deep = ld && kt == 1;

    /* :622-641  timescales.  Written only on the deep arm. */
    if (deep) {
        const int ikb = kcbot[i], ikt = kctop[i];
        zmfub1[i] = zmfub[i];
        ztauc[i] = nt_div(
            nt_add(NTK(pgeoh, ikt), -NTK(pgeoh, ikb)),
            nt_mul(nt_add(2.0f, min(15.0f, wup[i])), c.g));
        if (lndj[i] == 0) {
            upbl[i] = nt_add(2.0f, nt_div(
                upbl[i], nt_add(NTK(paph, klev + 1), -NTK(paph, ikb))));
            ztaubl[i] = nt_div(
                nt_add(NTK(pgeoh, ikb), -NTK(pgeoh, klev + 1)),
                nt_mul(c.g, upbl[i]));
            ztaubl[i] = min(300.0f, ztaubl[i]);
        } else {
            ztaubl[i] = ztauc[i];
        }
        zheat[i] = 0.0f; zcape[i] = 0.0f;
        zcape1[i] = 0.0f; zcape2[i] = 0.0f;
    }

    /* :644-668  the CAPE and heating integrals */
    for (int jk = 1; jk <= klev; ++jk) {
        if (deep && jk <= kcbot[i] && jk > kctop[i]) {
            const float zdz = nt_add(NTK(pgeo, jk - 1), -NTK(pgeo, jk));
            const float zdp = nt_add(NTK(pap, jk), -NTK(pap, jk - 1));
            zheat[i] = nt_add(zheat[i], nt_mul(
                nt_add(
                    nt_div(nt_add(nt_add(NTK(pten, jk - 1),
                                                  -NTK(pten, jk)),
                                        nt_mul(zdz, c.rcpd)),
                              NTK(ptenh, jk)),
                    nt_mul(c.vtmpc1, nt_add(NTK(pqen, jk - 1),
                                                  -NTK(pqen, jk)))),
                nt_mul(c.g, nt_add(NTK(pmfu, jk), NTK(pmfd, jk)))));
            zcape1[i] = nt_add(zcape1[i], nt_mul(
                nt_add(
                    nt_div(nt_add(NTK(ptu, jk), -NTK(ptenh, jk)),
                              NTK(ptenh, jk)),
                    nt_add(nt_mul(c.vtmpc1,
                                        nt_add(NTK(pqu, jk),
                                                  -NTK(pqenh, jk))),
                              -NTK(plu, jk))),
                zdp));
        }
        if (deep && jk >= kcbot[i]) {
            if (nt_add(NTK(paph, klev + 1), -NTK(paph, kdpl[i]))
                    < 50.0e2f) {
                const float zdp = nt_add(NTK(paph, jk + 1),
                                            -NTK(paph, jk));
                zcape2[i] = nt_add(zcape2[i], nt_mul(
                    nt_mul(ztaubl[i], nt_add(
                        nt_mul(nt_add(1.0f,
                                            nt_mul(c.vtmpc1,
                                                      NTK(pqen, jk))),
                                  NTK(ptte, jk)),
                        nt_mul(nt_mul(c.vtmpc1, NTK(pten, jk)),
                                  NTK(pqte, jk)))),
                    zdp));
            }
        }
    }

    /* :670-694  the deep closure */
    if (deep) {
        const int ikb = kcbot[i];
        float tc = ztauc[i];
        tc = max(dt, tc);
        tc = max(360.0f, tc);
        tc = min(10800.0f, tc);
        ztauc[i] = tc;
        /* modC.  cu16 :676 scales the adjustment time by scale_fac
         * (11.6246 at dx=4500); cu6 uses a FIXED 2400 s
         * (module_cu_tiedtke.F:105 parameter ztau = 2400.0), which is
         * precisely the scale-awareness deletion that separates the two
         * schemes' deep closures. */
        const float ztau = tiedtke_closure
            ? 2400.0f : nt_mul(tc, scale_fac[i]);
        ztau_o[i] = ztau;
        /* nonequil is .true. (cu_ntiedtke_common:49) */
        zcape2[i] = max(0.0f, zcape2[i]);
        zcape[i] = max(0.0f, min(nt_add(zcape1[i], -zcape2[i]),
                                     5000.0f));
        zheat[i] = max(1.0e-4f, zheat[i]);
        float b1 = nt_div(nt_mul(zcape[i], zmfub[i]),
                             nt_mul(zheat[i], ztau));
        b1 = max(b1, 0.001f);
        const float zmfmax = nt_mul(
            nt_add(NTK(paph, ikb), -NTK(paph, ikb - 1)), zcons2);
        zmfub1[i] = min(b1, zmfmax);
    }

    /* :696-720  the shallow closure -- the ONLY use of scale_fac2 */
    if (ld && kt == 2) {
        const int ikb = kcbot[i];
        float zeps;
        if (NTK(pmfd, ikb) < 0.0f && loddraf[i] != 0) {
            zeps = nt_div(-NTK(pmfd, ikb),
                             max(zmfub[i], NT_CMFCMIN));
        } else {
            zeps = 0.0f;
        }
        const float zqumqe = nt_add(
            nt_add(nt_add(NTK(pqu, ikb), NTK(plu, ikb)),
                      -nt_mul(zeps, NTK(pqd, ikb))),
            -nt_mul(nt_add(1.0f, -zeps), NTK(pqenh, ikb)));
        const float zdqmin = max(nt_mul(0.01f, NTK(pqenh, ikb)),
                                   NT_CMFCMIN);
        const float zmfmax = nt_mul(
            nt_add(NTK(paph, ikb), -NTK(paph, ikb - 1)), zcons2);
        float zdh = nt_add(
            nt_mul(c.cpd, nt_add(
                nt_add(NTK(ptu, ikb), -nt_mul(zeps, NTK(ptd, ikb))),
                -nt_mul(nt_add(1.0f, -zeps), NTK(ptenh, ikb)))),
            nt_mul(c.alv, zqumqe));
        zdh = nt_mul(c.g, max(zdh, nt_mul(1.0e5f, zdqmin)));
        /* zdhpbl is an INPUT: cumastrn built it at :506-517 from CUTYPEN's
         * kcbot and ldcum, and cuascn has since changed both.  Recomputing
         * it from closure-time state is wrong on every shallow column. */
        float b1 = (zdhpbl[i] > 0.0f) ? nt_div(zdhpbl[i], zdh)
                                      : zmfub[i];
        b1 = nt_div(b1, scale_fac2[i]);
        zmfub1[i] = min(b1, zmfmax);
    }

    /* :722-724  mid-level takes NEITHER factor */
    if (ld && kt == 3) zmfub1[i] = zmfub[i];

    /* :726-740  scale the downdraft by zmfub1/zmfub */
    if (ld) {
        const float zfac = nt_div(zmfub1[i],
                                     max(zmfub[i], NT_CMFCMIN));
        for (int jk = 1; jk <= klev; ++jk) {
            NTK(pmfd, jk) = nt_mul(NTK(pmfd, jk), zfac);
            NTK(pmfds, jk) = nt_mul(NTK(pmfds, jk), zfac);
            NTK(pmfdq, jk) = nt_mul(NTK(pmfdq, jk), zfac);
            NTK(pdmfdp, jk) = nt_mul(NTK(pdmfdp, jk), zfac);
            NTK(pmfdde_rate, jk) = nt_mul(NTK(pmfdde_rate, jk), zfac);
        }
    }

}
kernel void ntiedtke_closure(
    constant NtiedtkeClosureResources &r [[buffer(0)]],
    constant NtiedtkeClosureParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_closure(i, r.ldcum, r.ktype, r.kcbot, r.kctop, r.kdpl, r.loddraf, r.lndj, r.wup, r.zmfub, r.zdhpbl, r.scale_fac, r.scale_fac2, r.pgeoh, r.paph, r.pap, r.pgeo, r.pten, r.pqen, r.ptenh, r.pqenh, r.ptu, r.pqu, r.plu, r.pmfu, r.ptd, r.pqd, r.ptte, r.pqte, r.pmfd, r.pmfds, r.pmfdq, r.pdmfdp, r.pmfdde_rate, r.zheat, r.zcape, r.zcape1, r.zcape2, r.ztauc, r.ztaubl, r.ztau_o, r.upbl, r.zmfub1, p.tiedtke_closure, p.ncol, p.nz, p.dt, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_body_ntiedtke_cuascn(int i, device const float *pten, device const float *pqen, device const float *pqsen, device const float *pgeo, device const float *pgeoh, device const float *pap, device const float *paph, device const float *pverv, device const int *lndj, device const int *kdpl, device const float *wbase, device float *ptenh, device float *pqenh, device float *ptu, device float *pqu, device float *plu, device float *pmfu, device float *pmfus, device float *pmfuq, device float *pmful, device float *plude, device float *pdmfup, device float *plglac, device float *pmfude_rate, device int *klab, device int *ldcum, device int *ktype, device int *kcbot, device int *kctop, device int *kctop0, device float *pmfub, device float *wup, int ncol, int klev, float ztmst, int llo3, float cp, float rd, float rv, float xlv, float xlf, float grav, device const int *llo3_mask) {

    if (i >= ncol) return;
    // The deferred mask is uniform across this chunk.
    if (llo3 < 0) llo3 = llo3_mask[0] & 1;

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]
    const int klevm1 = klev - 1;

    /* :1893-1899.  Each in the reference's own association.
     *
     * CORRECTED.  This said cuascn's cap is "three times looser than the
     * closure's".  It is not: zcons2 is 3/(g*dt) in all three scopes that
     * declare it and they are identical.  The factor of three separates
     * zcons2 from zcons (1/(g*dt), cumastrn:468) -- one character apart,
     * both live in cumastrn, and zcons has exactly one consumer: the
     * momentum rescale at :1000. */
    const float zcons2 = nt_div(3.0f, nt_mul(c.g, ztmst));
    const float zfacbuo = nt_div(0.5f, nt_add(1.0f, 0.5f));
    const float zprcdgw = nt_mul(NT_CPRCON, c.zrg);
    const float z_cldmax = 5.0e-3f;
    const float z_cwifrac = 0.5f;
    const float z_cprc2 = 0.5f;
    const float z_cwdrag =
        nt_div(nt_mul(nt_div(3.0f, 8.0f), 0.506f), 0.2f);

#define LD   ldcum[i]
#define KT   ktype[i]
#define KB   kcbot[i]
#define MFUB pmfub[i]
    int kt0 = kctop0[i];
    int ktp;
    const int kdp = kdpl[i];
    const float wb = wbase[i];

    float zluold = 0.0f, wupa = 0.0f, zdpmean = 0.0f, zoentr = 0.0f;

    /* ---- 2. defaults (:1903-1937) ---------------------------------- */
    if (!LD) {
        KT = 0;
        KB = -1;
        MFUB = 0.0f;
        NTC(pqu, klev) = 0.0f;
    }
    for (int jk = 1; jk <= klev; ++jk) {
        if (jk != KB) NTC(plu, jk) = 0.0f;
        NTC(pmfu, jk) = 0.0f;   NTC(pmfus, jk) = 0.0f;
        NTC(pmfuq, jk) = 0.0f;  NTC(pmful, jk) = 0.0f;
        NTC(plude, jk) = 0.0f;  NTC(plglac, jk) = 0.0f;
        NTC(pdmfup, jk) = 0.0f; NTC(pmfude_rate, jk) = 0.0f;
        if (!LD || KT == 3) NTC(klab, jk) = 0;
        if (!LD && NTC(paph, jk) < 4.0e4f) kt0 = jk;
    }
    if (KT == 3) LD = 0;

    /* ---- 3. cloud base (:1943-1953) -------------------------------- */
    ktp = KB;
    if (LD) {
        NTC(pmfu, KB) = MFUB;
        NTC(pmfus, KB) = nt_mul(
            MFUB, nt_add(nt_mul(c.cpd, NTC(ptu, KB)), NTC(pgeoh, KB)));
        NTC(pmfuq, KB) = nt_mul(MFUB, NTC(pqu, KB));
        NTC(pmful, KB) = nt_mul(MFUB, NTC(plu, KB));
    }

    /* The three one-level-lookback registers, seeded at level klev by the
     * same rule section 2 and section 3 applied to the arrays.
     *
     * kb0 is the cloud base AS SECTION 3 SAW IT, not the live kcbot.  Only
     * section 3 wrote kup, and only at that level; cubasmcn can move kcbot
     * later in the loop and must not drag the seed with it. */
    const int kb0 = LD ? KB : -1;
    float zlrain_prev = 0.0f;
    float zbuo_prev = 0.0f;
    float kup_prev = (kb0 == klev)
                   ? nt_mul(0.5f, nt_mul(wb, wb)) : 0.0f;

    float zdmfen = 0.0f, zdmfde = 0.0f;

    /* ---- 4. the ascent (:1959-2245) -------------------------------- */
    for (int jk = klevm1; jk >= 3; --jk) {
        /* cubasmcn (:1968).  It can zero zlrain at jk+1, so the register
         * goes in by reference. */
        if (nt_cubasmcn(jk, ncol, i, pten, pqen, pqsen, pverv, pgeo, pgeoh,
                    ldcum, ktype, klab, pmfu, pmfub, kcbot,
                    ptu, pqu, plu, pmfus, pmfuq, pmful, pdmfup, c))
            zlrain_prev = 0.0f;

        float kup_cur = (jk == kb0)
                      ? nt_mul(0.5f, nt_mul(wb, wb)) : 0.0f;
        float zlrain_cur = 0.0f;
        float zbuo_cur = 0.0f;
        int llo1 = 0;
        float zprecip = 0.0f;

        /* :1980-2001 */
        if (NTC(klab, jk + 1) == 0) NTC(klab, jk) = 0;
        const int loflag = (LD && NTC(klab, jk + 1) == 2)
                        || (KT == 3 && NTC(klab, jk + 1) == 1);
        const float zph = NTC(paph, jk);
        if (KT == 3 && jk == KB) {
            const float zmfmax = nt_mul(
                nt_add(NTC(paph, jk), -NTC(paph, jk - 1)), zcons2);
            if (MFUB > zmfmax) {
                const float zfac = nt_div(zmfmax, MFUB);
                NTC(pmfu, jk + 1) = nt_mul(NTC(pmfu, jk + 1), zfac);
                NTC(pmfus, jk + 1) = nt_mul(NTC(pmfus, jk + 1), zfac);
                NTC(pmfuq, jk + 1) = nt_mul(NTC(pmfuq, jk + 1), zfac);
                MFUB = zmfmax;
            }
            MFUB = min(MFUB, zmfmax);
        }

        nt_cuentrn(jk, ncol, i, kcbot, ldcum, llo3 != 0, pgeoh, pmfu,
                   &zdmfen, &zdmfde, c);

        if (!llo3) continue;

        /* :2015-2065 */
        float zqold = 0.0f;
        if (loflag) {
            zdmfde = min(zdmfde, nt_mul(0.75f, NTC(pmfu, jk + 1)));
            if (jk == KB) {
                const float r = min(1.0f, nt_div(NTC(pqen, jk),
                                                      NTC(pqsen, jk)));
                zoentr = nt_mul(
                    nt_mul(nt_mul(-1.75e-3f, nt_add(r, -1.0f)),
                              nt_add(NTC(pgeoh, jk),
                                        -NTC(pgeoh, jk + 1))), c.zrg);
                zoentr = nt_mul(min(0.4f, zoentr), NTC(pmfu, jk + 1));
            }
            if (jk < KB) {
                const float zmfmax = nt_mul(
                    nt_add(NTC(paph, jk), -NTC(paph, jk - 1)), zcons2);
                const float zxs =
                    max(nt_add(NTC(pmfu, jk + 1), -zmfmax), 0.0f);
                const float dpap =
                    nt_add(NTC(pap, jk + 1), -NTC(pap, jk));
                wupa = nt_add(wupa, nt_mul(kup_prev, dpap));
                /* :2036 has NO parentheses, so it associates left to
                 * right.  The wup accumulator one line up IS parenthesised.
                 * Bracketing the difference here costs 1 ULP in wup. */
                zdpmean = nt_add(nt_add(zdpmean, NTC(pap, jk + 1)),
                                    -NTC(pap, jk));
                zdmfen = zoentr;
                if (KT >= 2) {
                    zdmfen = nt_mul(2.0f, zdmfen);
                    zdmfde = zdmfen;
                }
                zdmfde = nt_mul(zdmfde, nt_add(1.6f,
                    -min(1.0f, nt_div(NTC(pqen, jk),
                                           NTC(pqsen, jk)))));
                const float zmftest =
                    nt_add(nt_add(NTC(pmfu, jk + 1), zdmfen), -zdmfde);
                float zchange = max(nt_add(zmftest, -zmfmax), 0.0f);
                const float zxe = max(nt_add(zchange, -zxs), 0.0f);
                zdmfen = nt_add(zdmfen, -zxe);
                zchange = nt_add(zchange, -zxe);
                zdmfde = nt_add(zdmfde, zchange);
            }
            /* :2050 writes pdmfen, which is a dead local.  Not stored. */
            NTC(pmfu, jk) =
                nt_add(nt_add(NTC(pmfu, jk + 1), zdmfen), -zdmfde);
            const float zqeen = nt_mul(NTC(pqenh, jk + 1), zdmfen);
            const float zseen = nt_mul(
                nt_add(nt_mul(c.cpd, NTC(ptenh, jk + 1)),
                          NTC(pgeoh, jk + 1)), zdmfen);
            const float zscde = nt_mul(
                nt_add(nt_mul(c.cpd, NTC(ptu, jk + 1)),
                          NTC(pgeoh, jk + 1)), zdmfde);
            const float zqude = nt_mul(NTC(pqu, jk + 1), zdmfde);
            NTC(plude, jk) = nt_mul(NTC(plu, jk + 1), zdmfde);
            const float zmfusk =
                nt_add(nt_add(NTC(pmfus, jk + 1), zseen), -zscde);
            const float zmfuqk =
                nt_add(nt_add(NTC(pmfuq, jk + 1), zqeen), -zqude);
            const float zmfulk =
                nt_add(NTC(pmful, jk + 1), -NTC(plude, jk));
            const float inv =
                nt_div(1.0f, max(NT_CMFCMIN, NTC(pmfu, jk)));
            NTC(plu, jk) = nt_mul(zmfulk, inv);
            NTC(pqu, jk) = nt_mul(zmfuqk, inv);
            NTC(ptu, jk) = nt_mul(
                nt_add(nt_mul(zmfusk, inv), -NTC(pgeoh, jk)), c.rcpd);
            NTC(ptu, jk) = max(100.0f, NTC(ptu, jk));
            NTC(ptu, jk) = min(400.0f, NTC(ptu, jk));
            zqold = NTC(pqu, jk);
            zlrain_cur = nt_mul(
                nt_mul(zlrain_prev,
                          nt_add(NTC(pmfu, jk + 1), -zdmfde)), inv);
            zluold = NTC(plu, jk);
        }

        /* :2069-2075.  NOT guarded by loflag -- every column. */
        if (jk > kdp) {
            NTC(ptu, jk) = NTC(ptenh, jk);
            NTC(pqu, jk) = NTC(pqenh, jk);
            NTC(plu, jk) = 0.0f;
            zluold = NTC(plu, jk);
        }

        /* :2081-2083 */
        if (loflag) nt_cuadjtqn1_device(&NTC(ptu, jk), &NTC(pqu, jk), zph, c);

        const int adjusted = loflag && (NTC(pqu, jk) != zqold);

        /* :2086-2093 */
        if (adjusted) {
            NTC(plglac, jk) = nt_mul(NTC(plu, jk),
                nt_add(nt_add(1.0f, -nt_foealfa(NTC(ptu, jk))),
                          -nt_add(1.0f, -nt_foealfa(NTC(ptu, jk + 1)))));
            NTC(ptu, jk) = nt_add(NTC(ptu, jk),
                                     nt_mul(c.ralfdcp, NTC(plglac, jk)));
        }

        /* :2096-2179 */
        if (adjusted) {
            NTC(klab, jk) = 2;
            NTC(plu, jk) = nt_add(nt_add(NTC(plu, jk), zqold),
                                     -NTC(pqu, jk));
            const float zbc = nt_mul(NTC(ptu, jk), nt_add(nt_add(
                nt_add(1.0f, nt_mul(c.vtmpc1, NTC(pqu, jk))),
                -NTC(plu, jk + 1)), -zlrain_prev));
            const float zbe = nt_mul(NTC(ptenh, jk),
                nt_add(1.0f, nt_mul(c.vtmpc1, NTC(pqenh, jk))));
            zbuo_cur = nt_add(zbc, -zbe);
            if (KT == 3 && NTC(klab, jk + 1) == 1) {
                if (zbuo_cur > -0.5f) {
                    LD = 1;
                    ktp = jk;
                    kup_cur = 0.5f;
                } else {
                    NTC(klab, jk) = 0;
                    NTC(pmfu, jk) = 0.0f;
                    NTC(plude, jk) = 0.0f;
                    NTC(plu, jk) = 0.0f;
                }
            }
            if (NTC(klab, jk + 1) == 2) {
                if (zbuo_cur < 0.0f) {
                    NTC(ptenh, jk) = nt_mul(0.5f,
                        nt_add(NTC(pten, jk), NTC(pten, jk - 1)));
                    NTC(pqenh, jk) = nt_mul(0.5f,
                        nt_add(NTC(pqen, jk), NTC(pqen, jk - 1)));
                    zbuo_cur = nt_add(zbc, -nt_mul(NTC(ptenh, jk),
                        nt_add(1.0f,
                                  nt_mul(c.vtmpc1, NTC(pqenh, jk)))));
                }
                const float zbuoc = nt_mul(nt_add(
                    nt_div(zbuo_cur, nt_mul(NTC(ptenh, jk),
                        nt_add(1.0f,
                                  nt_mul(c.vtmpc1, NTC(pqenh, jk))))),
                    nt_div(zbuo_prev, nt_mul(NTC(ptenh, jk + 1),
                        nt_add(1.0f, nt_mul(c.vtmpc1,
                                                 NTC(pqenh, jk + 1)))))),
                    0.5f);
                const float zdkbuo = nt_mul(nt_mul(
                    nt_add(NTC(pgeoh, jk), -NTC(pgeoh, jk + 1)),
                    zfacbuo), zbuoc);
                /* Both arms are the same shape on purpose: a runtime branch
                 * over differently-shaped arithmetic is what let ptxas
                 * contract two clones differently once before. */
                const float zsel = (zdmfen > 0.0f) ? zdmfen : zdmfde;
                const float zdken = min(1.0f, nt_div(
                    nt_mul(nt_add(1.0f, z_cwdrag), zsel),
                    max(NT_CMFCMIN, NTC(pmfu, jk + 1))));
                kup_cur = nt_div(
                    nt_add(nt_mul(kup_prev, nt_add(1.0f, -zdken)),
                              zdkbuo),
                    nt_add(1.0f, zdken));
                if (zbuo_cur < 0.0f) {
                    float zkedke =
                        nt_div(kup_cur, max(1.0e-10f, kup_prev));
                    zkedke = max(0.0f, min(1.0f, zkedke));
                    const float zmfun =
                        nt_mul(sqrt(zkedke), NTC(pmfu, jk + 1));
                    zdmfde = max(zdmfde,
                                   nt_add(NTC(pmfu, jk + 1), -zmfun));
                    NTC(plude, jk) = nt_mul(NTC(plu, jk + 1), zdmfde);
                    NTC(pmfu, jk) = nt_add(
                        nt_add(NTC(pmfu, jk + 1), zdmfen), -zdmfde);
                }
                if (zbuo_cur > -0.2f) {
                    const float rr = min(1.0f,
                        nt_div(NTC(pqen, jk - 1), NTC(pqsen, jk - 1)));
                    const float q3 = min(1.0f,
                        nt_div(NTC(pqsen, jk), NTC(pqsen, KB)));
                    zoentr = nt_mul(nt_mul(nt_mul(
                        nt_mul(1.75e-3f,
                                  nt_add(0.3f, -nt_add(rr, -1.0f))),
                        nt_add(NTC(pgeoh, jk - 1), -NTC(pgeoh, jk))),
                        c.zrg), nt_mul(nt_mul(q3, q3), q3));
                    zoentr = nt_mul(min(0.4f, zoentr), NTC(pmfu, jk));
                } else {
                    zoentr = 0.0f;
                }
                if (jk > kdp) {
                    NTC(pmfu, jk) = NTC(pmfu, jk + 1);
                    kup_cur = 0.5f;
                }
                if (kup_cur > 0.0f && NTC(pmfu, jk) > 0.0f) {
                    ktp = jk;
                    llo1 = 1;
                } else {
                    NTC(klab, jk) = 0;
                    NTC(pmfu, jk) = 0.0f;
                    kup_cur = 0.0f;
                    zdmfde = NTC(pmfu, jk + 1);
                    NTC(plude, jk) = nt_mul(NTC(plu, jk + 1), zdmfde);
                }
                if (NTC(pmfu, jk + 1) > 0.0f) NTC(pmfude_rate, jk) = zdmfde;
            }
        } else if (loflag && KT == 2 && NTC(pqu, jk) == zqold) {
            NTC(klab, jk) = 0;
            NTC(pmfu, jk) = 0.0f;
            kup_cur = 0.0f;
            zdmfde = NTC(pmfu, jk + 1);
            NTC(plude, jk) = nt_mul(NTC(plu, jk + 1), zdmfde);
            NTC(pmfude_rate, jk) = zdmfde;
        }

        /* :2182-2216  precipitation conversion */
        if (llo1) {
            const float zdshrd = (lndj[i] == 1) ? 5.0e-4f : 3.0e-4f;
            if (NTC(plu, jk) > zdshrd) {
                const float zwu = min(15.0f, sqrt(
                    nt_mul(2.0f, max(0.1f, kup_prev))));
                const float zprcon =
                    nt_div(zprcdgw, nt_mul(0.75f, zwu));
                const float zdt = min(NT_RTBER - NT_RTICE,
                    max(nt_add(NT_RTBER, -NTC(ptu, jk)), 0.0f));
                const float zcbf =
                    nt_add(1.0f, nt_mul(z_cprc2, sqrt(zdt)));
                const float zzco = nt_mul(zprcon, zcbf);
                const float zlcrit = nt_div(zdshrd, zcbf);
                const float zdfi =
                    nt_add(NTC(pgeoh, jk), -NTC(pgeoh, jk + 1));
                const float zc = nt_add(NTC(plu, jk), -zluold);
                const float rq = nt_div(NTC(plu, jk), zlcrit);
                const float zarg = nt_mul(rq, rq);
                const float zd = (zarg < 25.0f)
                    ? nt_mul(nt_mul(zzco,
                          nt_add(1.0f, -exp(-zarg))), zdfi)
                    : nt_mul(zzco, zdfi);
                const float zint = exp(-zd);
                float zlnew = nt_add(nt_mul(zluold, zint),
                    nt_mul(nt_div(zc, zd), nt_add(1.0f, -zint)));
                zlnew = max(0.0f, min(NTC(plu, jk), zlnew));
                zlnew = min(z_cldmax, zlnew);
                zprecip = max(0.0f,
                    nt_add(nt_add(zluold, zc), -zlnew));
                NTC(pdmfup, jk) = nt_mul(zprecip, NTC(pmfu, jk));
                zlrain_cur = nt_add(zlrain_cur, zprecip);
                NTC(plu, jk) = zlnew;
            }
        }

        /* :2219-2236  rain fallout */
        if (llo1 && zlrain_cur > 0.0f) {
            const float zvw = nt_mul(21.18f, pow(zlrain_cur, 0.2f));
            const float zvi = nt_mul(z_cwifrac, zvw);
            const float zalfaw = nt_foealfa(NTC(ptu, jk));
            const float zvv = nt_add(nt_mul(zalfaw, zvw),
                nt_mul(nt_add(1.0f, -zalfaw), zvi));
            const float zrold = nt_add(zlrain_cur, -zprecip);
            const float zwu = min(15.0f, sqrt(
                nt_mul(2.0f, max(0.1f, kup_cur))));
            const float zd = nt_div(zvv, zwu);
            const float zint = exp(-zd);
            float zrnew = nt_add(nt_mul(zrold, zint),
                nt_mul(nt_div(zprecip, zd), nt_add(1.0f, -zint)));
            zrnew = max(0.0f, min(zlrain_cur, zrnew));
            zlrain_cur = zrnew;
        }

        /* :2239-2243 */
        if (loflag) {
            NTC(pmful, jk) = nt_mul(NTC(plu, jk), NTC(pmfu, jk));
            NTC(pmfus, jk) = nt_mul(
                nt_add(nt_mul(c.cpd, NTC(ptu, jk)), NTC(pgeoh, jk)),
                NTC(pmfu, jk));
            NTC(pmfuq, jk) = nt_mul(NTC(pqu, jk), NTC(pmfu, jk));
        }

        kup_prev = kup_cur;
        zlrain_prev = zlrain_cur;
        zbuo_prev = zbuo_cur;
    }

    /* ---- 5. final (:2248-2256) ------------------------------------- */
    if (ktp == -1) LD = 0;
    KB = max(KB, ktp);
    if (LD) {
        wupa = max(1.0e-2f, nt_div(wupa, max(1.0f, zdpmean)));
        wupa = sqrt(nt_mul(2.0f, wupa));
    }

    kctop[i] = ktp;
    kctop0[i] = kt0;
    wup[i] = wupa;

#undef LD
#undef KT
#undef KB
#undef MFUB
#undef NTC

}
kernel void ntiedtke_cuascn(
    constant NtiedtkeCuascnResources &r [[buffer(0)]],
    constant NtiedtkeCuascnParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_cuascn(i, r.pten, r.pqen, r.pqsen, r.pgeo, r.pgeoh, r.pap, r.paph, r.pverv, r.lndj, r.kdpl, r.wbase, r.ptenh, r.pqenh, r.ptu, r.pqu, r.plu, r.pmfu, r.pmfus, r.pmfuq, r.pmful, r.plude, r.pdmfup, r.plglac, r.pmfude_rate, r.klab, r.ldcum, r.ktype, r.kcbot, r.kctop, r.kctop0, r.pmfub, r.wup, p.ncol, p.klev, p.ztmst, p.llo3, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav, r.llo3_mask);
}

static inline void nt_body_ntiedtke_cudtdqn(int i, device const int *ldcum, device const float *paph, device const float *pten, device const float *plglac, device const float *plude, device const float *pmfus, device const float *pmfds, device const float *pmfuq, device const float *pmfdq, device const float *pmful, device const float *pdmfup, device const float *pdmfdp, device const float *pdpmel, device float *ptent, device float *ptenq, device float *pcte, int ncol, int klev, int ktopm2, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;
    if (!ldcum[i]) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    for (int jk = ktopm2; jk <= klev; ++jk) {
        const float zdp = nt_div(
            c.g, nt_add(NTC(paph, jk + 1), -NTC(paph, jk)));
        const float zalv = nt_foelhm(NTC(pten, jk), c);
        float zdtdt, zdqdt;
        if (jk < klev) {
            const float inner = nt_add(nt_add(nt_add(
                nt_add(NTC(pmful, jk + 1), -NTC(pmful, jk)),
                -NTC(plude, jk)), -NTC(pdmfup, jk)), -NTC(pdmfdp, jk));
            const float big = nt_add(nt_add(nt_add(nt_add(
                nt_add(nt_add(NTC(pmfus, jk + 1), -NTC(pmfus, jk)),
                          NTC(pmfds, jk + 1)), -NTC(pmfds, jk)),
                nt_mul(c.alf, NTC(plglac, jk))),
                -nt_mul(c.alf, NTC(pdpmel, jk))),
                -nt_mul(zalv, inner));
            zdtdt = nt_mul(nt_mul(zdp, c.rcpd), big);
            /* :3117-3119 chains all NINE terms left to right -- there
             * are no parentheses inside, so the last two do not group. */
            zdqdt = nt_mul(zdp, nt_add(nt_add(nt_add(nt_add(
                nt_add(nt_add(nt_add(nt_add(NTC(pmfuq, jk + 1),
                    -NTC(pmfuq, jk)), NTC(pmfdq, jk + 1)),
                    -NTC(pmfdq, jk)), NTC(pmful, jk + 1)),
                    -NTC(pmful, jk)), -NTC(plude, jk)),
                    -NTC(pdmfup, jk)), -NTC(pdmfdp, jk)));
        } else {
            const float big = nt_add(nt_add(
                nt_add(NTC(pmfus, jk), NTC(pmfds, jk)),
                nt_mul(c.alf, NTC(pdpmel, jk))),
                -nt_mul(zalv, nt_add(nt_add(
                    nt_add(NTC(pmful, jk), NTC(pdmfup, jk)),
                    NTC(pdmfdp, jk)), NTC(plude, jk))));
            zdtdt = -nt_mul(nt_mul(zdp, c.rcpd), big);
            zdqdt = -nt_mul(zdp, nt_add(
                nt_add(nt_add(NTC(pmfuq, jk), NTC(plude, jk)),
                          NTC(pmfdq, jk)),
                nt_add(nt_add(NTC(pmful, jk), NTC(pdmfup, jk)),
                          NTC(pdmfdp, jk))));
        }
        /* ADD.  See the header. */
        NTC(ptent, jk) = nt_add(NTC(ptent, jk), zdtdt);
        NTC(ptenq, jk) = nt_add(NTC(ptenq, jk), zdqdt);
        NTC(pcte, jk) = nt_mul(zdp, NTC(plude, jk));
    }
#undef NTC

}
kernel void ntiedtke_cudtdqn(
    constant NtiedtkeCudtdqnResources &r [[buffer(0)]],
    constant NtiedtkeCudtdqnParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_cudtdqn(i, r.ldcum, r.paph, r.pten, r.plglac, r.plude, r.pmfus, r.pmfds, r.pmfuq, r.pmfdq, r.pmful, r.pdmfup, r.pdmfdp, r.pdpmel, r.ptent, r.ptenq, r.pcte, p.ncol, p.klev, p.ktopm2, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_body_ntiedtke_cududvn(int i, device const int *ldcum, device const int *ktype, device const int *kcbot, device const float *paph, device const float *puen, device const float *pven, device const float *pmfu, device const float *pmfd, device const float *puu, device const float *pud, device const float *pvu, device const float *pvd, device float *ptenu, device float *ptenv, device float *zmfuu, device float *zmfuv, device float *zmfdu, device float *zmfdv, int ncol, int klev, int ktopm2, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;
    if (!ldcum[i]) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]
    const int kb = kcbot[i];
    const int kt3 = (ktype[i] == 3);

    /* 1.0 the fluxes (:3191-3201) */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        const int ik = jk - 1;
        NTC(zmfuu, jk) = nt_mul(NTC(pmfu, jk),
            nt_add(NTC(puu, jk), -NTC(puen, ik)));
        NTC(zmfuv, jk) = nt_mul(NTC(pmfu, jk),
            nt_add(NTC(pvu, jk), -NTC(pven, ik)));
        NTC(zmfdu, jk) = nt_mul(NTC(pmfd, jk),
            nt_add(NTC(pud, jk), -NTC(puen, ik)));
        NTC(zmfdv, jk) = nt_mul(NTC(pmfd, jk),
            nt_add(NTC(pvd, jk), -NTC(pven, ik)));
    }

    /* linear fluxes below cloud (:3203-3215) */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        if (jk <= kb) continue;
        float zzp = nt_div(nt_add(NTC(paph, klev + 1), -NTC(paph, jk)),
                              nt_add(NTC(paph, klev + 1), -NTC(paph, kb)));
        if (kt3) zzp = nt_mul(zzp, zzp);
        NTC(zmfuu, jk) = nt_mul(NTC(zmfuu, kb), zzp);
        NTC(zmfuv, jk) = nt_mul(NTC(zmfuv, kb), zzp);
        NTC(zmfdu, jk) = nt_mul(NTC(zmfdu, kb), zzp);
        NTC(zmfdv, jk) = nt_mul(NTC(zmfdv, kb), zzp);
    }

    /* 2.0 and 3.0 fused (:3219-3249).  ADD, do not assign. */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        const float zdp = nt_div(
            c.g, nt_add(NTC(paph, jk + 1), -NTC(paph, jk)));
        float zdudt, zdvdt;
        if (jk < klev) {
            const int ik = jk + 1;
            zdudt = nt_mul(zdp, nt_add(nt_add(
                nt_add(NTC(zmfuu, ik), -NTC(zmfuu, jk)), NTC(zmfdu, ik)),
                -NTC(zmfdu, jk)));
            zdvdt = nt_mul(zdp, nt_add(nt_add(
                nt_add(NTC(zmfuv, ik), -NTC(zmfuv, jk)), NTC(zmfdv, ik)),
                -NTC(zmfdv, jk)));
        } else {
            zdudt = -nt_mul(zdp,
                nt_add(NTC(zmfuu, jk), NTC(zmfdu, jk)));
            zdvdt = -nt_mul(zdp,
                nt_add(NTC(zmfuv, jk), NTC(zmfdv, jk)));
        }
        NTC(ptenu, jk) = nt_add(NTC(ptenu, jk), zdudt);
        NTC(ptenv, jk) = nt_add(NTC(ptenv, jk), zdvdt);
    }
#undef NTC

}
kernel void ntiedtke_cududvn(
    constant NtiedtkeCududvnResources &r [[buffer(0)]],
    constant NtiedtkeCududvnParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_cududvn(i, r.ldcum, r.ktype, r.kcbot, r.paph, r.puen, r.pven, r.pmfu, r.pmfd, r.puu, r.pud, r.pvu, r.pvd, r.ptenu, r.ptenv, r.zmfuu, r.zmfuv, r.zmfdu, r.zmfdv, p.ncol, p.klev, p.ktopm2, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_cuadjtqn2(thread float *pt, thread float *pq, float psp, thread const NtConst &c) {

    const float zqp = nt_div(1.0f, psp);
    float zqsat = nt_mul(nt_foeewm(*pt, c), zqp);
    zqsat = min(0.5f, zqsat);
    float zcor = nt_div(1.0f, nt_add(1.0f, -nt_mul(c.vtmpc1, zqsat)));
    zqsat = nt_mul(zqsat, zcor);
    float zcond = nt_div(
        nt_add(*pq, -zqsat),
        nt_add(1.0f, nt_mul(nt_mul(zqsat, zcor),
                                  nt_foedem(*pt, c))));
    zcond = min(zcond, 0.0f);
    *pt = nt_add(*pt, nt_mul(nt_foeldcpm(*pt, c), zcond));
    *pq = nt_add(*pq, -zcond);
    zqsat = nt_mul(nt_foeewm(*pt, c), zqp);
    zqsat = min(0.5f, zqsat);
    zcor = nt_div(1.0f, nt_add(1.0f, -nt_mul(c.vtmpc1, zqsat)));
    zqsat = nt_mul(zqsat, zcor);
    float zcond1 = nt_div(
        nt_add(*pq, -zqsat),
        nt_add(1.0f, nt_mul(nt_mul(zqsat, zcor),
                                  nt_foedem(*pt, c))));
    if (fabs(zcond) < 1.0e-20f) zcond1 = min(zcond1, 0.0f);
    *pt = nt_add(*pt, nt_mul(nt_foeldcpm(*pt, c), zcond1));
    *pq = nt_add(*pq, -zcond1);

}

static inline void nt_cuadjtqn2_device(device float *pt, device float *pq, float psp, thread const NtConst &c) {

    const float zqp = nt_div(1.0f, psp);
    float zqsat = nt_mul(nt_foeewm(*pt, c), zqp);
    zqsat = min(0.5f, zqsat);
    float zcor = nt_div(1.0f, nt_add(1.0f, -nt_mul(c.vtmpc1, zqsat)));
    zqsat = nt_mul(zqsat, zcor);
    float zcond = nt_div(
        nt_add(*pq, -zqsat),
        nt_add(1.0f, nt_mul(nt_mul(zqsat, zcor),
                                  nt_foedem(*pt, c))));
    zcond = min(zcond, 0.0f);
    *pt = nt_add(*pt, nt_mul(nt_foeldcpm(*pt, c), zcond));
    *pq = nt_add(*pq, -zcond);
    zqsat = nt_mul(nt_foeewm(*pt, c), zqp);
    zqsat = min(0.5f, zqsat);
    zcor = nt_div(1.0f, nt_add(1.0f, -nt_mul(c.vtmpc1, zqsat)));
    zqsat = nt_mul(zqsat, zcor);
    float zcond1 = nt_div(
        nt_add(*pq, -zqsat),
        nt_add(1.0f, nt_mul(nt_mul(zqsat, zcor),
                                  nt_foedem(*pt, c))));
    if (fabs(zcond) < 1.0e-20f) zcond1 = min(zcond1, 0.0f);
    *pt = nt_add(*pt, nt_mul(nt_foeldcpm(*pt, c), zcond1));
    *pq = nt_add(*pq, -zcond1);

}

static inline void nt_body_ntiedtke_cudlfsn(int i, device const int *ldcum, device const int *kcbot, device const int *kctop, device const float *ptenh, device const float *pqenh, device const float *pten, device const float *pqsen, device const float *pgeo, device const float *pgeoh, device const float *paph, device const float *ptu, device const float *pqu, device const float *pmfub, device float *ptd, device float *pqd, device float *pmfd, device float *pmfds, device float *pmfdq, device float *pdmfdp, device float *prfl, device int *kdtop, device int *lddraf, int ncol, int klev, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    /* 1. defaults (:2393-2398).  Only the two scalars; the level arrays
     * are class 2 and are deliberately left alone. */
    int ld_out = 0;
    int kdt = klev + 1;
    kdtop[i] = kdt;
    lddraf[i] = 0;
    if (!ldcum[i]) return;

    const int kb = kcbot[i];
    const int ktop = kctop[i];
    float rfl = prfl[i];

    /* 2. the level of minimum saturation moist static energy (:2422-2431) */
    int ikhsmin = klev + 1;
    float zhsmin = 1.0e8f;
    for (int jk = 3; jk <= klev - 2; ++jk) {
        const float zhsk = nt_add(
            nt_add(nt_mul(c.cpd, NTC(pten, jk)), NTC(pgeo, jk)),
            nt_mul(nt_foelhm(NTC(pten, jk), c), NTC(pqsen, jk)));
        if (zhsk < zhsmin) { zhsmin = zhsk; ikhsmin = jk; }
    }

    /* 2.1-2.2 the descent (:2435-2484) */
    const int ike = klev - 3;
    for (int jk = 3; jk <= ike; ++jk) {
        float ztenwb = NTC(ptenh, jk);
        float zqenwb = NTC(pqenh, jk);
        const float zph = NTC(paph, jk);
        const bool llo2 = (rfl > 0.0f) && !ld_out
                          && (jk < kb) && (jk > ktop) && (jk >= ikhsmin);
        if (!llo2) continue;

        nt_cuadjtqn2(&ztenwb, &zqenwb, zph, c);

        const float zttest = nt_mul(0.5f,
            nt_add(NTC(ptu, jk), ztenwb));
        const float zqtest = nt_mul(0.5f,
            nt_add(NTC(pqu, jk), zqenwb));
        const float zbuo = nt_add(
            nt_mul(zttest, nt_add(1.0f, nt_mul(c.vtmpc1, zqtest))),
            -nt_mul(NTC(ptenh, jk),
                nt_add(1.0f, nt_mul(c.vtmpc1, NTC(pqenh, jk)))));
        const float zcond = nt_add(NTC(pqenh, jk), -zqenwb);
        const float zmftop = -nt_mul(NT_CMFDEPS, pmfub[i]);
        if (zbuo < 0.0f
            && rfl > nt_mul(nt_mul(10.0f, zmftop), zcond)) {
            kdt = jk;
            ld_out = 1;
            NTC(ptd, jk) = zttest;
            NTC(pqd, jk) = zqtest;
            NTC(pmfd, jk) = zmftop;
            NTC(pmfds, jk) = nt_mul(NTC(pmfd, jk),
                nt_add(nt_mul(c.cpd, NTC(ptd, jk)), NTC(pgeoh, jk)));
            NTC(pmfdq, jk) = nt_mul(NTC(pmfd, jk), NTC(pqd, jk));
            NTC(pdmfdp, jk - 1) = nt_mul(-0.5f,
                nt_mul(NTC(pmfd, jk), zcond));
            rfl = nt_add(rfl, NTC(pdmfdp, jk - 1));
        }
    }

    prfl[i] = rfl;
    kdtop[i] = kdt;
    lddraf[i] = ld_out;
#undef NTC

}
kernel void ntiedtke_cudlfsn(
    constant NtiedtkeCudlfsnResources &r [[buffer(0)]],
    constant NtiedtkeCudlfsnParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_cudlfsn(i, r.ldcum, r.kcbot, r.kctop, r.ptenh, r.pqenh, r.pten, r.pqsen, r.pgeo, r.pgeoh, r.paph, r.ptu, r.pqu, r.pmfub, r.ptd, r.pqd, r.pmfd, r.pmfds, r.pmfdq, r.pdmfdp, r.prfl, r.kdtop, r.lddraf, p.ncol, p.klev, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_body_ntiedtke_cuddrafn(int i, device const int *lddraf, device const float *ptenh, device const float *pqenh, device const float *pgeo, device const float *pgeoh, device const float *paph, device const float *pmfu, device float *ptd, device float *pqd, device float *pmfd, device float *pmfds, device float *pmfdq, device float *pdmfdp, device float *pmfdde_rate, device float *prfl, int ncol, int klev, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    float zoentr = 0.0f, zbuoy = 0.0f;
    float zdmfen = 0.0f, zdmfde = 0.0f;
    float rfl = prfl[i];
    const int ld = lddraf[i];

    /* itopde (:2616-2621).  The loop DESCENDS and the last assignment wins,
     * so itopde ends at the TOPMOST level within 60 hPa of the surface. */
    int itopde = -1;
    for (int jk = klev; jk >= 1; --jk) {
        NTC(pmfdde_rate, jk) = 0.0f;
        if (nt_add(NTC(paph, klev + 1), -NTC(paph, jk)) < 60.0e2f)
            itopde = jk;
    }
    if (itopde < 0) return;          /* never set: nothing to descend into */

    for (int jk = 3; jk <= klev; ++jk) {
        const float zph = NTC(paph, jk);
        if (!(ld && NTC(pmfd, jk - 1) < 0.0f)) continue;

        const float zentr = nt_mul(nt_mul(
            nt_mul(NT_ENTRDD, NTC(pmfd, jk - 1)),
            nt_add(NTC(pgeoh, jk - 1), -NTC(pgeoh, jk))), c.zrg);
        zdmfen = zentr;
        zdmfde = zentr;

        if (jk > itopde) {
            zdmfen = 0.0f;
            zdmfde = nt_div(
                nt_mul(NTC(pmfd, itopde),
                          nt_add(NTC(paph, jk), -NTC(paph, jk - 1))),
                nt_add(NTC(paph, klev + 1), -NTC(paph, itopde)));
        }
        if (jk <= itopde) {
            const float zdz = -nt_mul(
                nt_add(NTC(pgeoh, jk - 1), -NTC(pgeoh, jk)), c.zrg);
            const float zzentr = nt_mul(nt_mul(zoentr, zdz),
                                           NTC(pmfd, jk - 1));
            zdmfen = nt_add(zdmfen, zzentr);
            zdmfen = max(zdmfen, nt_mul(0.3f, NTC(pmfd, jk - 1)));
            zdmfen = max(zdmfen,
                nt_add(-nt_mul(0.75f, NTC(pmfu, jk)),
                          -nt_add(NTC(pmfd, jk - 1), -zdmfde)));
            zdmfen = min(zdmfen, 0.0f);
        }

        NTC(pmfd, jk) = nt_add(
            nt_add(NTC(pmfd, jk - 1), zdmfen), -zdmfde);
        const float zseen = nt_mul(
            nt_add(nt_mul(c.cpd, NTC(ptenh, jk - 1)),
                      NTC(pgeoh, jk - 1)), zdmfen);
        const float zqeen = nt_mul(NTC(pqenh, jk - 1), zdmfen);
        const float zsdde = nt_mul(
            nt_add(nt_mul(c.cpd, NTC(ptd, jk - 1)),
                      NTC(pgeoh, jk - 1)), zdmfde);
        const float zqdde = nt_mul(NTC(pqd, jk - 1), zdmfde);
        const float zmfdsk = nt_add(
            nt_add(NTC(pmfds, jk - 1), zseen), -zsdde);
        const float zmfdqk = nt_add(
            nt_add(NTC(pmfdq, jk - 1), zqeen), -zqdde);
        const float inv = nt_div(1.0f,
                                    min(-NT_CMFCMIN, NTC(pmfd, jk)));
        NTC(pqd, jk) = nt_mul(zmfdqk, inv);
        NTC(ptd, jk) = nt_mul(
            nt_add(nt_mul(zmfdsk, inv), -NTC(pgeoh, jk)), c.rcpd);
        NTC(ptd, jk) = min(400.0f, NTC(ptd, jk));
        NTC(ptd, jk) = max(100.0f, NTC(ptd, jk));
        float zcond = NTC(pqd, jk);

        nt_cuadjtqn2_device(&NTC(ptd, jk), &NTC(pqd, jk), zph, c);

        zcond = nt_add(zcond, -NTC(pqd, jk));
        float zbuo = nt_add(
            nt_mul(NTC(ptd, jk),
                nt_add(1.0f, nt_mul(c.vtmpc1, NTC(pqd, jk)))),
            -nt_mul(NTC(ptenh, jk),
                nt_add(1.0f, nt_mul(c.vtmpc1, NTC(pqenh, jk)))));
        if (rfl > 0.0f && NTC(pmfu, jk) > 0.0f) {
            const float zrain = nt_div(rfl, NTC(pmfu, jk));
            zbuo = nt_add(zbuo, -nt_mul(NTC(ptd, jk), zrain));
        }
        if (zbuo >= 0.0f || rfl <= nt_mul(NTC(pmfd, jk), zcond)) {
            NTC(pmfd, jk) = 0.0f;
            zbuo = 0.0f;
        }
        NTC(pmfds, jk) = nt_mul(
            nt_add(nt_mul(c.cpd, NTC(ptd, jk)), NTC(pgeoh, jk)),
            NTC(pmfd, jk));
        NTC(pmfdq, jk) = nt_mul(NTC(pqd, jk), NTC(pmfd, jk));
        const float zdmfdp = -nt_mul(NTC(pmfd, jk), zcond);
        NTC(pdmfdp, jk - 1) = zdmfdp;
        rfl = nt_add(rfl, zdmfdp);

        /* organised entrainment for the next level down */
        float zbuoyz = nt_div(zbuo, NTC(ptenh, jk));
        zbuoyz = min(zbuoyz, 0.0f);
        const float zdz2 = -nt_add(NTC(pgeo, jk - 1), -NTC(pgeo, jk));
        zbuoy = nt_add(zbuoy, nt_mul(zbuoyz, zdz2));
        zoentr = nt_div(nt_mul(nt_mul(c.g, zbuoyz), 0.5f),
                           nt_add(1.0f, zbuoy));
        NTC(pmfdde_rate, jk) = -zdmfde;
    }

    prfl[i] = rfl;
#undef NTC

}
kernel void ntiedtke_cuddrafn(
    constant NtiedtkeCuddrafnResources &r [[buffer(0)]],
    constant NtiedtkeCuddrafnParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_cuddrafn(i, r.lddraf, r.ptenh, r.pqenh, r.pgeo, r.pgeoh, r.paph, r.pmfu, r.ptd, r.pqd, r.pmfd, r.pmfds, r.pmfdq, r.pdmfdp, r.pmfdde_rate, r.prfl, p.ncol, p.klev, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_body_ntiedtke_cuflxn(int i, device const int *kcbot, device const int *kctop, device const int *kdtop, device const int *lndj, device const float *pten, device const float *ptenh, device const float *pqenh, device const float *paph, device const float *pap, device const float *pqen, device const float *pgeoh, device int *ldcum, device int *lddraf, device int *ktype, device float *pmfu, device float *pmfd, device float *pmfus, device float *pmfds, device float *pmfuq, device float *pmfdq, device float *pmful, device float *plude, device float *plglac, device float *pdmfup, device float *pdmfdp, device float *pmfdde_rate, device float *pqsen, device float *pdpmel, device float *pmflxr, device float *pmflxs, device float *prain, int ncol, int klev, float ztmst, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    const float zcons1a = nt_div(
        c.cpd, nt_mul(nt_mul(c.alf, c.g), NT_ZTAUMEL));
    const float zcons2 = nt_div(3.0f, nt_mul(c.g, ztmst));
    const float zcpecons = nt_div(5.44e-4f, c.g);

    /* 1.0 (:2865-2875) */
    float pr = 0.0f;
    const int kb = kcbot[i];
    const int ktop = kctop[i];
    const int kdt = kdtop[i];
    const int ld = ldcum[i];
    int ldd = lddraf[i];
    if (!ld || kdt < ktop) ldd = 0;
    if (!ld) ktype[i] = 0;
    const int kt3 = (ktype[i] == 3);
    int idbas = klev;
    const float rhevap = (lndj[i] == 1) ? 0.7f : 0.9f;

    const int ktopm2 = 2;             /* :2877, unconditional.  See header. */

    for (int jk = ktopm2; jk <= klev; ++jk) {
        const int ikb = min(jk + 1, klev);
        NTC(pmflxr, jk) = 0.0f;
        NTC(pmflxs, jk) = 0.0f;
        NTC(pdpmel, jk) = 0.0f;
        if (ld && jk >= ktop) {
            NTC(pmfus, jk) = nt_add(NTC(pmfus, jk),
                -nt_mul(NTC(pmfu, jk),
                    nt_add(nt_mul(c.cpd, NTC(ptenh, jk)),
                              NTC(pgeoh, jk))));
            NTC(pmfuq, jk) = nt_add(NTC(pmfuq, jk),
                -nt_mul(NTC(pmfu, jk), NTC(pqenh, jk)));
            NTC(plglac, jk) = nt_mul(NTC(pmfu, jk), NTC(plglac, jk));
            const int llddraf = ldd && (jk >= kdt);
            if (llddraf && jk >= kdt) {
                NTC(pmfds, jk) = nt_add(NTC(pmfds, jk),
                    -nt_mul(NTC(pmfd, jk),
                        nt_add(nt_mul(c.cpd, NTC(ptenh, jk)),
                                  NTC(pgeoh, jk))));
                NTC(pmfdq, jk) = nt_add(NTC(pmfdq, jk),
                    -nt_mul(NTC(pmfd, jk), NTC(pqenh, jk)));
            } else {
                NTC(pmfd, jk) = 0.0f;
                NTC(pmfds, jk) = 0.0f;
                NTC(pmfdq, jk) = 0.0f;
                NTC(pdmfdp, jk - 1) = 0.0f;
            }
            if (llddraf && NTC(pmfd, jk) < 0.0f
                && fabs(NTC(pmfd, ikb)) < 1.0e-20f) {
                idbas = jk;
            }
        } else {
            NTC(pmfu, jk) = 0.0f;  NTC(pmfd, jk) = 0.0f;
            NTC(pmfus, jk) = 0.0f; NTC(pmfds, jk) = 0.0f;
            NTC(pmfuq, jk) = 0.0f; NTC(pmfdq, jk) = 0.0f;
            NTC(pmful, jk) = 0.0f; NTC(plglac, jk) = 0.0f;
            NTC(pdmfup, jk - 1) = 0.0f;
            NTC(pdmfdp, jk - 1) = 0.0f;
            NTC(plude, jk - 1) = 0.0f;
        }
    }

    NTC(pmflxr, klev + 1) = 0.0f;
    NTC(pmflxs, klev + 1) = 0.0f;

    /* the cloud-base taper (:2926-2938) */
    if (ld) {
        const int ikb = kb;
        const int ik = ikb + 1;
        float zzp = nt_div(
            nt_add(NTC(paph, klev + 1), -NTC(paph, ik)),
            nt_add(NTC(paph, klev + 1), -NTC(paph, ikb)));
        if (kt3) zzp = nt_mul(zzp, zzp);
        NTC(pmfu, ik) = nt_mul(NTC(pmfu, ikb), zzp);
        NTC(pmfus, ik) = nt_mul(
            nt_add(NTC(pmfus, ikb),
                -nt_mul(nt_foelhm(NTC(ptenh, ikb), c), NTC(pmful, ikb))),
            zzp);
        NTC(pmfuq, ik) = nt_mul(
            nt_add(NTC(pmfuq, ikb), NTC(pmful, ikb)), zzp);
        NTC(pmful, ik) = 0.0f;
    }

    for (int jk = ktopm2; jk <= klev; ++jk) {
        if (ld && jk > kb + 1) {
            const int ikb = kb + 1;
            float zzp = nt_div(
                nt_add(NTC(paph, klev + 1), -NTC(paph, jk)),
                nt_add(NTC(paph, klev + 1), -NTC(paph, ikb)));
            if (kt3) zzp = nt_mul(zzp, zzp);
            NTC(pmfu, jk) = nt_mul(NTC(pmfu, ikb), zzp);
            NTC(pmfus, jk) = nt_mul(NTC(pmfus, ikb), zzp);
            NTC(pmfuq, jk) = nt_mul(NTC(pmfuq, ikb), zzp);
            NTC(pmful, jk) = 0.0f;
        }
        const int ik = idbas;
        const int llddraf = ldd && (jk > ik) && (ik < klev);
        if (llddraf && ik == kb + 1) {
            float zzp = nt_div(
                nt_add(NTC(paph, klev + 1), -NTC(paph, jk)),
                nt_add(NTC(paph, klev + 1), -NTC(paph, ik)));
            if (kt3) zzp = nt_mul(zzp, zzp);
            NTC(pmfd, jk) = nt_mul(NTC(pmfd, ik), zzp);
            NTC(pmfds, jk) = nt_mul(NTC(pmfds, ik), zzp);
            NTC(pmfdq, jk) = nt_mul(NTC(pmfdq, ik), zzp);
            NTC(pmfdde_rate, jk) =
                -nt_add(NTC(pmfd, jk - 1), -NTC(pmfd, jk));
        } else if (llddraf && ik != kb + 1 && jk == ik + 1) {
            NTC(pmfdde_rate, jk) =
                -nt_add(NTC(pmfd, jk - 1), -NTC(pmfd, jk));
        }
    }

    /* 2. melting and the rain/snow split (:2975-3011) */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        if (!(ld && jk >= ktop - 1)) continue;
        pr = nt_add(pr, NTC(pdmfup, jk));
        if (NTC(pmflxs, jk) > 0.0f && NTC(pten, jk) > NT_TMELT) {
            const float zcons1 = nt_mul(zcons1a, nt_add(1.0f,
                nt_mul(0.5f, nt_add(NTC(pten, jk), -NT_TMELT))));
            const float zfac = nt_mul(zcons1,
                nt_add(NTC(paph, jk + 1), -NTC(paph, jk)));
            const float zsnmlt = min(NTC(pmflxs, jk),
                nt_mul(zfac, nt_add(NTC(pten, jk), -NT_TMELT)));
            NTC(pdpmel, jk) = zsnmlt;
            NTC(pqsen, jk) = nt_div(
                nt_foeewm(nt_add(NTC(pten, jk),
                                    -nt_div(zsnmlt, zfac)), c),
                NTC(pap, jk));
        }
        float zalfaw = nt_foealfa(NTC(pten, jk));
        /* No liquid precipitation above the melting level. */
        if (NTC(pten, jk) < NT_TMELT && zalfaw > 0.0f) {
            NTC(plglac, jk) = nt_add(NTC(plglac, jk),
                nt_mul(zalfaw,
                    nt_add(NTC(pdmfup, jk), NTC(pdmfdp, jk))));
            zalfaw = 0.0f;
        }
        NTC(pmflxr, jk + 1) = nt_add(
            nt_add(NTC(pmflxr, jk),
                nt_mul(zalfaw,
                    nt_add(NTC(pdmfup, jk), NTC(pdmfdp, jk)))),
            NTC(pdpmel, jk));
        NTC(pmflxs, jk + 1) = nt_add(
            nt_add(NTC(pmflxs, jk),
                nt_mul(nt_add(1.0f, -zalfaw),
                    nt_add(NTC(pdmfup, jk), NTC(pdmfdp, jk)))),
            -NTC(pdpmel, jk));
        if (nt_add(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1)) < 0.0f) {
            NTC(pdmfdp, jk) = -nt_add(
                nt_add(NTC(pmflxr, jk), NTC(pmflxs, jk)),
                NTC(pdmfup, jk));
            NTC(pmflxr, jk + 1) = 0.0f;
            NTC(pmflxs, jk + 1) = 0.0f;
            NTC(pdpmel, jk) = 0.0f;
        } else if (NTC(pmflxr, jk + 1) < 0.0f) {
            NTC(pmflxs, jk + 1) =
                nt_add(NTC(pmflxs, jk + 1), NTC(pmflxr, jk + 1));
            NTC(pmflxr, jk + 1) = 0.0f;
        } else if (NTC(pmflxs, jk + 1) < 0.0f) {
            NTC(pmflxr, jk + 1) =
                nt_add(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1));
            NTC(pmflxs, jk + 1) = 0.0f;
        }
    }


    /* the sub-cloud evaporation (:3012-3057).
     *
     * NAMED COVERAGE GAP: measured, 48 columns reach this guard and ZERO
     * move pdmfup, because zdrfl1 carries max(0, pqsen - pqen) and the
     * fixture's soundings are saturated below cloud base wherever the block
     * runs.  So the 0.5777 power law, zrmin and the rhevap land/sea split
     * are transcribed and graded only in the sense that their guard is
     * evaluated -- and rhevap is the ONLY path lndj takes into cuflxn, so
     * land/sea is untested here too.  See docs/ntiedtke/PORT-RECORD.md section 16.
     */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        if (!(ld && jk >= kb)) continue;
        const float zrfl = nt_add(NTC(pmflxr, jk), NTC(pmflxs, jk));
        if (zrfl > 1.0e-20f) {
            const float zdp = nt_add(NTC(paph, jk + 1), -NTC(paph, jk));
            const float zbase = nt_mul(
                nt_div(sqrt(nt_div(NTC(paph, jk),
                                               NTC(paph, klev + 1))),
                          5.09e-3f),
                nt_div(zrfl, NT_ZCUCOV));
            const float zdrfl1 = nt_mul(
                nt_mul(nt_mul(nt_mul(zcpecons,
                    max(0.0f, nt_add(NTC(pqsen, jk), -NTC(pqen, jk)))),
                    NT_ZCUCOV),
                    pow(zbase, 0.5777f)),
                zdp);
            float zrnew = nt_add(zrfl, -zdrfl1);
            const float zrmin = nt_add(zrfl,
                -nt_mul(nt_mul(NT_ZCUCOV,
                    max(0.0f, nt_add(
                        nt_mul(rhevap, NTC(pqsen, jk)), -NTC(pqen, jk)))),
                    nt_mul(zcons2, zdp)));
            zrnew = max(zrnew, zrmin);
            const float zrfln = max(zrnew, 0.0f);
            const float zdrfl = min(0.0f, nt_add(zrfln, -zrfl));
            const float zdenom = nt_div(1.0f, max(1.0e-20f,
                nt_add(NTC(pmflxr, jk), NTC(pmflxs, jk))));
            float zalfaw = nt_foealfa(NTC(pten, jk));
            if (NTC(pten, jk) < NT_TMELT) zalfaw = 0.0f;
            const float zpdr = nt_mul(zalfaw, NTC(pdmfdp, jk));
            const float zpds =
                nt_mul(nt_add(1.0f, -zalfaw), NTC(pdmfdp, jk));
            NTC(pmflxr, jk + 1) = nt_add(
                nt_add(nt_add(NTC(pmflxr, jk), zpdr), NTC(pdpmel, jk)),
                nt_mul(nt_mul(zdrfl, NTC(pmflxr, jk)), zdenom));
            NTC(pmflxs, jk + 1) = nt_add(
                nt_add(nt_add(NTC(pmflxs, jk), zpds),
                          -NTC(pdpmel, jk)),
                nt_mul(nt_mul(zdrfl, NTC(pmflxs, jk)), zdenom));
            NTC(pdmfup, jk) = nt_add(NTC(pdmfup, jk), zdrfl);
            if (nt_add(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1)) < 0.0f) {
                NTC(pdmfup, jk) = nt_add(NTC(pdmfup, jk),
                    -nt_add(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1)));
                NTC(pmflxr, jk + 1) = 0.0f;
                NTC(pmflxs, jk + 1) = 0.0f;
                NTC(pdpmel, jk) = 0.0f;
            } else if (NTC(pmflxr, jk + 1) < 0.0f) {
                NTC(pmflxs, jk + 1) =
                    nt_add(NTC(pmflxs, jk + 1), NTC(pmflxr, jk + 1));
                NTC(pmflxr, jk + 1) = 0.0f;
            } else if (NTC(pmflxs, jk + 1) < 0.0f) {
                NTC(pmflxr, jk + 1) =
                    nt_add(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1));
                NTC(pmflxs, jk + 1) = 0.0f;
            }
        } else {
            NTC(pmflxr, jk + 1) = 0.0f;
            NTC(pmflxs, jk + 1) = 0.0f;
            NTC(pdmfdp, jk) = 0.0f;
            NTC(pdpmel, jk) = 0.0f;
        }
    }

    prain[i] = pr;
    lddraf[i] = ldd;
#undef NTC

}
kernel void ntiedtke_cuflxn(
    constant NtiedtkeCuflxnResources &r [[buffer(0)]],
    constant NtiedtkeCuflxnParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_cuflxn(i, r.kcbot, r.kctop, r.kdtop, r.lndj, r.pten, r.ptenh, r.pqenh, r.paph, r.pap, r.pqen, r.pgeoh, r.ldcum, r.lddraf, r.ktype, r.pmfu, r.pmfd, r.pmfus, r.pmfds, r.pmfuq, r.pmfdq, r.pmful, r.plude, r.plglac, r.pdmfup, r.pdmfdp, r.pmfdde_rate, r.pqsen, r.pdpmel, r.pmflxr, r.pmflxs, r.prain, p.ncol, p.klev, p.ztmst, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_body_ntiedtke_cloud_depth(int i, device const int *ldcum, device const int *kcbot, device const int *kctop, device const float *paph, device const float *pdmfup, device int *ktype, device int *ictop0, device float *prfl, device float *pmfd, device float *pmfds, device float *pmfdq, device float *pdmfdp, device float *pdpmel, int ncol, int klev) {

    if (i >= ncol) return;

#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    if (ldcum[i]) {
        const int ikb = kcbot[i];
        const int itopm2 = kctop[i];
        const float zpbmpt = nt_add(NTC(paph, ikb), -NTC(paph, itopm2));
        /* Two sequential ifs, NOT if/else: the reference lets a column
         * flipped 1 -> 2 be examined by the second test.  It cannot flip
         * back (zpbmpt cannot be both < and >= the threshold), but the
         * shape is transcribed as written. */
        if (ktype[i] == 1 && zpbmpt < NT_ZDNOPRC) ktype[i] = 2;
        if (ktype[i] == 2 && zpbmpt >= NT_ZDNOPRC) ktype[i] = 1;
        ictop0[i] = itopm2;
    }

    /* :571-577.  zrfl starts at level 1 and sums upward through klev. */
    float zrfl = NTC(pdmfup, 1);
    for (int jk = 2; jk <= klev; ++jk)
        zrfl = nt_add(zrfl, NTC(pdmfup, jk));
    prfl[i] = zrfl;

    /* :580-588 */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(pmfd, jk) = 0.0f;
        NTC(pmfds, jk) = 0.0f;
        NTC(pmfdq, jk) = 0.0f;
        NTC(pdmfdp, jk) = 0.0f;
        NTC(pdpmel, jk) = 0.0f;
    }
#undef NTC

}
kernel void ntiedtke_cloud_depth(
    constant NtiedtkeCloudDepthResources &r [[buffer(0)]],
    constant NtiedtkeCloudDepthParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_cloud_depth(i, r.ldcum, r.kcbot, r.kctop, r.paph, r.pdmfup, r.ktype, r.ictop0, r.prfl, r.pmfd, r.pmfds, r.pmfdq, r.pdmfdp, r.pdpmel, p.ncol, p.klev);
}

static inline void nt_body_ntiedtke_adjust(int i, device const int *ldcum, device const int *loddraf, device const int *idtop, device const int *kctop, device const int *kcbot, device const float *paph, device const float *pqen, device const float *pmfu, device const float *pmfuq, device const float *pmful, device float *pmfd, device float *pmfds, device float *pmfdq, device float *plude, device float *pdmfup, device float *pdmfdp, device float *pmfdde_rate, device float *pmfude_rate, device float *pmflxr, device float *pmflxs, device float *prsfc, device float *pssfc, int ncol, int klev, float ztmst, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    const int ld = ldcum[i];
    const int ldd = loddraf[i];
    const int idt = idtop[i];
    const int ktop = kctop[i];
    const int kb = kcbot[i];

    /* :834-847 */
    float zmfs = 1.0f;
    for (int jk = 2; jk <= klev; ++jk) {
        if (ldd && jk >= idt - 1) {
            const float zmfmax = nt_mul(NTC(pmfu, jk), 0.98f);
            if (nt_add(nt_add(NTC(pmfd, jk), zmfmax), 1.0e-15f)
                    < 0.0f) {
                zmfs = min(zmfs, nt_div(-zmfmax, NTC(pmfd, jk)));
            }
        }
    }

    /* :849-861.  zmfuub runs DOWNWARD through the column. */
    float zmfuub = 0.0f;
    for (int jk = 2; jk <= klev; ++jk) {
        if (zmfs < 1.0f && jk >= idt - 1) {
            NTC(pmfd, jk) = nt_mul(NTC(pmfd, jk), zmfs);
            NTC(pmfds, jk) = nt_mul(NTC(pmfds, jk), zmfs);
            NTC(pmfdq, jk) = nt_mul(NTC(pmfdq, jk), zmfs);
            NTC(pmfdde_rate, jk) = nt_mul(NTC(pmfdde_rate, jk), zmfs);
            zmfuub = nt_add(zmfuub,
                -nt_mul(nt_add(1.0f, -zmfs), NTC(pdmfdp, jk)));
            NTC(pmflxr, jk + 1) = nt_add(NTC(pmflxr, jk + 1), zmfuub);
            NTC(pdmfdp, jk) = nt_mul(NTC(pdmfdp, jk), zmfs);
        }
    }

    /* :863-880 */
    for (int jk = 2; jk <= klev - 1; ++jk) {
        if (ldd && jk >= idt - 1) {
            const float zerate = nt_add(
                nt_add(-NTC(pmfd, jk), NTC(pmfd, jk - 1)),
                NTC(pmfdde_rate, jk));
            if (zerate < 0.0f)
                NTC(pmfdde_rate, jk) =
                    nt_add(NTC(pmfdde_rate, jk), -zerate);
        }
        if (ld && jk >= ktop - 1) {
            const float zerate = nt_add(
                nt_add(NTC(pmfu, jk), -NTC(pmfu, jk + 1)),
                NTC(pmfude_rate, jk));
            if (zerate < 0.0f)
                NTC(pmfude_rate, jk) =
                    nt_add(NTC(pmfude_rate, jk), -zerate);
            NTC(pdmfup, jk) = nt_add(nt_add(
                nt_add(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1)),
                -NTC(pmflxr, jk)), -NTC(pmflxs, jk));
            NTC(pdmfdp, jk) = 0.0f;
        }
    }

    /* :883-892  the downdraft-top humidity guard */
    if (ldd) {
        const int jk = idt;
        const int ik = min(jk + 1, klev);
        if (NTC(pmfdq, jk) < nt_mul(0.3f, NTC(pmfdq, ik)))
            NTC(pmfdq, jk) = nt_mul(0.3f, NTC(pmfdq, ik));
    }

    /* :896-913  the near-cloud-top humidity guard */
    for (int jk = 2; jk <= klev; ++jk) {
        if (ld && jk >= ktop - 1 && jk < kb) {
            const float zdz = nt_div(nt_mul(ztmst, c.g),
                nt_add(NTC(paph, jk + 1), -NTC(paph, jk)));
            float zmfa = nt_add(nt_add(nt_add(nt_add(nt_add(
                nt_add(NTC(pmfuq, jk + 1), NTC(pmfdq, jk + 1)),
                -NTC(pmfuq, jk)), -NTC(pmfdq, jk)),
                NTC(pmful, jk + 1)), -NTC(pmful, jk)), NTC(pdmfup, jk));
            zmfa = nt_mul(nt_add(zmfa, -NTC(plude, jk)), zdz);
            if (nt_add(NTC(pqen, jk), zmfa) < 0.0f) {
                NTC(plude, jk) = nt_add(NTC(plude, jk), nt_div(
                    nt_mul(2.0f, nt_add(NTC(pqen, jk), zmfa)), zdz));
            }
            if (NTC(plude, jk) < 0.0f) NTC(plude, jk) = 0.0f;
        }
        if (!ld) NTC(pmfude_rate, jk) = 0.0f;
        if (fabs(NTC(pmfd, jk - 1)) < 1.0e-20f)
            NTC(pmfdde_rate, jk) = 0.0f;
    }

    prsfc[i] = NTC(pmflxr, klev + 1);
    pssfc[i] = NTC(pmflxs, klev + 1);
#undef NTC

}
kernel void ntiedtke_adjust(
    constant NtiedtkeAdjustResources &r [[buffer(0)]],
    constant NtiedtkeAdjustParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_adjust(i, r.ldcum, r.loddraf, r.idtop, r.kctop, r.kcbot, r.paph, r.pqen, r.pmfu, r.pmfuq, r.pmful, r.pmfd, r.pmfds, r.pmfdq, r.plude, r.pdmfup, r.pdmfdp, r.pmfdde_rate, r.pmfude_rate, r.pmflxr, r.pmflxs, r.prsfc, r.pssfc, p.ncol, p.klev, p.ztmst, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_body_ntiedtke_momentum_rescale(int i, device const int *ldcum, device const int *kctop, device const float *paph, device const float *pmfu, device const float *pmfd, device float *zmfuus, device float *zmfdus, int ncol, int klev, float ztmst, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    /* zcons, NOT zcons2.  See the header. */
    const float zcons = nt_div(1.0f, nt_mul(c.g, ztmst));
    const int ld = ldcum[i];
    const int ktop = kctop[i];

    float zmfs = 1.0f;
    if (ld) {
        for (int jk = 2; jk <= klev; ++jk) {
            if (jk >= ktop - 1) {
                const float zmfmax = nt_mul(
                    nt_add(NTC(paph, jk), -NTC(paph, jk - 1)), zcons);
                if (NTC(pmfu, jk) > zmfmax && jk >= ktop)
                    zmfs = min(zmfs, nt_div(zmfmax, NTC(pmfu, jk)));
            }
        }
    }

    for (int jk = 1; jk <= klev; ++jk) {
        NTC(zmfuus, jk) = NTC(pmfu, jk);
        NTC(zmfdus, jk) = NTC(pmfd, jk);
        if (ld && jk >= ktop - 1) {
            NTC(zmfuus, jk) = nt_mul(NTC(pmfu, jk), zmfs);
            NTC(zmfdus, jk) = nt_mul(NTC(pmfd, jk), zmfs);
        }
    }
#undef NTC

}
kernel void ntiedtke_momentum_rescale(
    constant NtiedtkeMomentumRescaleResources &r [[buffer(0)]],
    constant NtiedtkeMomentumRescaleParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_momentum_rescale(i, r.ldcum, r.kctop, r.paph, r.pmfu, r.pmfd, r.zmfuus, r.zmfdus, p.ncol, p.klev, p.ztmst, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_body_ntiedtke_updraft_scale(int i, device const int *loddraf, device const int *kcbot, device const int *kctop, device const float *zmfub1, device const float *zmfub, device const float *paph, device int *ldcum, device int *ktype, device int *idtop, device float *pmfu, device float *pmfus, device float *pmfuq, device float *pmful, device float *pdmfup, device float *plude, device float *pmfude_rate, device float *pmfd, device float *pmfds, device float *pmfdq, device float *pdmfdp, device float *pmfdde_rate, int ncol, int klev, float ztmst, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    const float zcons2 = nt_div(3.0f, nt_mul(c.g, ztmst));
    const int ld0 = ldcum[i];
    const int kb = kcbot[i];
    const int ktop = kctop[i];
    const int ldd = loddraf[i];

    /* :743-746 */
    float zmfs = 1.0f;
    if (ld0) zmfs = nt_div(zmfub1[i], max(NT_CMFCMIN, zmfub[i]));

    /* :747-761  taper below cloud base, then cap */
    for (int jk = 2; jk <= klev; ++jk) {
        if (ld0 && jk >= ktop - 1) {
            if (jk > kb) {
                const float zdz = nt_div(
                    nt_add(NTC(paph, klev + 1), -NTC(paph, jk)),
                    nt_add(NTC(paph, klev + 1), -NTC(paph, kb)));
                NTC(pmfu, jk) = nt_mul(NTC(pmfu, kb), zdz);
            }
            const float zmfmax = nt_mul(
                nt_add(NTC(paph, jk), -NTC(paph, jk - 1)), zcons2);
            if (nt_mul(NTC(pmfu, jk), zmfs) > zmfmax)
                zmfs = min(zmfs, nt_div(zmfmax, NTC(pmfu, jk)));
        }
    }

    /* :762-774  apply */
    for (int jk = 2; jk <= klev; ++jk) {
        if (ld0 && jk <= kb && jk >= ktop - 1) {
            NTC(pmfu, jk) = nt_mul(NTC(pmfu, jk), zmfs);
            NTC(pmfus, jk) = nt_mul(NTC(pmfus, jk), zmfs);
            NTC(pmfuq, jk) = nt_mul(NTC(pmfuq, jk), zmfs);
            NTC(pmful, jk) = nt_mul(NTC(pmful, jk), zmfs);
            NTC(pdmfup, jk) = nt_mul(NTC(pdmfup, jk), zmfs);
            NTC(plude, jk) = nt_mul(NTC(plude, jk), zmfs);
            NTC(pmfude_rate, jk) =
                nt_mul(NTC(pmfude_rate, jk), zmfs);
        }
    }

    /* 6.6 (:777-783) */
    if (ktype[i] == 2 && kb == ktop && kb >= klev - 1) {
        ldcum[i] = 0;
        ktype[i] = 0;
    }

    /* 6.7 (:798-818) */
    int idt = idtop[i];
    if (ldd && idt <= ktop) idt = ktop + 1;
    for (int jk = 2; jk <= klev; ++jk) {
        if (ldd) {
            if (jk < idt) {
                NTC(pmfd, jk) = 0.0f;
                NTC(pmfds, jk) = 0.0f;
                NTC(pmfdq, jk) = 0.0f;
                NTC(pmfdde_rate, jk) = 0.0f;
                NTC(pdmfdp, jk) = 0.0f;
            } else if (jk == idt) {
                NTC(pmfdde_rate, jk) = 0.0f;
            }
        }
    }
    idtop[i] = idt;
#undef NTC

}
kernel void ntiedtke_updraft_scale(
    constant NtiedtkeUpdraftScaleResources &r [[buffer(0)]],
    constant NtiedtkeUpdraftScaleParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_updraft_scale(i, r.loddraf, r.kcbot, r.kctop, r.zmfub1, r.zmfub, r.paph, r.ldcum, r.ktype, r.idtop, r.pmfu, r.pmfus, r.pmfuq, r.pmful, r.pdmfup, r.plude, r.pmfude_rate, r.pmfd, r.pmfds, r.pmfdq, r.pdmfdp, r.pmfdde_rate, p.ncol, p.klev, p.ztmst, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_body_ntiedtke_momentum_profile(int i, device const int *ldcum, device const int *ktype, device const int *kcbot, device const int *kctop, device const int *kdpl, device const int *idtop, device const float *puen, device const float *pven, device const float *pmfu, device const float *pmfd, device const float *pmfude_rate, device const float *pmfdde_rate, device float *puu, device float *pvu, device float *pud, device float *pvd, int ncol, int klev) {

    if (i >= ncol) return;
    if (!ldcum[i]) return;

#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]
    const int kt = ktype[i];
    const int kb = kcbot[i];
    const int ktop = kctop[i];
    const int kdp = kdpl[i];
    const int idt = idtop[i];

    /* the updraft profile (:930-971), DOWNWARD in index */
    for (int jk = klev - 1; jk >= 2; --jk) {
        const int ik = jk + 1;
        if (jk == kb && kt < 3) {
            NTC(puu, jk) = NTC(puen, kdp - 1);
            NTC(pvu, jk) = NTC(pven, kdp - 1);
        } else if (jk == kb && kt == 3) {
            NTC(puu, jk) = NTC(puen, jk - 1);
            NTC(pvu, jk) = NTC(pven, jk - 1);
        }
        if (jk < kb && jk >= ktop) {
            const float pgf_u = -nt_mul(
                nt_mul(NT_PGCOEF, 0.5f),
                nt_add(
                    nt_mul(NTC(pmfu, ik),
                        nt_add(NTC(puen, ik), -NTC(puen, jk))),
                    nt_mul(NTC(pmfu, jk),
                        nt_add(NTC(puen, jk), -NTC(puen, jk - 1)))));
            const float pgf_v = -nt_mul(
                nt_mul(NT_PGCOEF, 0.5f),
                nt_add(
                    nt_mul(NTC(pmfu, ik),
                        nt_add(NTC(pven, ik), -NTC(pven, jk))),
                    nt_mul(NTC(pmfu, jk),
                        nt_add(NTC(pven, jk), -NTC(pven, jk - 1)))));
            const float zerate = nt_add(
                nt_add(NTC(pmfu, jk), -NTC(pmfu, ik)),
                NTC(pmfude_rate, jk));
            const float zderate = NTC(pmfude_rate, jk);
            const float zmfa = nt_div(
                1.0f, max(NT_CMFCMIN, NTC(pmfu, jk)));
            NTC(puu, jk) = nt_mul(nt_add(nt_add(nt_add(
                nt_mul(NTC(puu, ik), NTC(pmfu, ik)),
                nt_mul(zerate, NTC(puen, jk))),
                -nt_mul(zderate, NTC(puu, ik))), pgf_u), zmfa);
            NTC(pvu, jk) = nt_mul(nt_add(nt_add(nt_add(
                nt_mul(NTC(pvu, ik), NTC(pmfu, ik)),
                nt_mul(zerate, NTC(pven, jk))),
                -nt_mul(zderate, NTC(pvu, ik))), pgf_v), zmfa);
        }
    }

    /* the downdraft profile (:972-991), UPWARD in index */
    for (int jk = 3; jk <= klev; ++jk) {
        const int ik = jk - 1;
        if (jk == idt) {
            NTC(pud, jk) = nt_mul(0.5f,
                nt_add(NTC(puu, jk), NTC(puen, ik)));
            NTC(pvd, jk) = nt_mul(0.5f,
                nt_add(NTC(pvu, jk), NTC(pven, ik)));
        } else if (jk > idt) {
            const float zerate = nt_add(
                nt_add(-NTC(pmfd, jk), NTC(pmfd, ik)),
                NTC(pmfdde_rate, jk));
            const float zmfa = nt_div(
                1.0f, min(-NT_CMFCMIN, NTC(pmfd, jk)));
            NTC(pud, jk) = nt_mul(nt_add(nt_add(
                nt_mul(NTC(pud, ik), NTC(pmfd, ik)),
                -nt_mul(zerate, NTC(puen, ik))),
                nt_mul(NTC(pmfdde_rate, jk), NTC(pud, ik))), zmfa);
            NTC(pvd, jk) = nt_mul(nt_add(nt_add(
                nt_mul(NTC(pvd, ik), NTC(pmfd, ik)),
                -nt_mul(zerate, NTC(pven, ik))),
                nt_mul(NTC(pmfdde_rate, jk), NTC(pvd, ik))), zmfa);
        }
    }
#undef NTC

}
kernel void ntiedtke_momentum_profile(
    constant NtiedtkeMomentumProfileResources &r [[buffer(0)]],
    constant NtiedtkeMomentumProfileParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_momentum_profile(i, r.ldcum, r.ktype, r.kcbot, r.kctop, r.kdpl, r.idtop, r.puen, r.pven, r.pmfu, r.pmfd, r.pmfude_rate, r.pmfdde_rate, r.puu, r.pvu, r.pud, r.pvd, p.ncol, p.klev);
}

static inline void nt_body_ntiedtke_ke_dissipation(int i, device const int *ldcum, device const int *kctop, device const float *paph, device const float *puen, device const float *pven, device const float *ztenu, device const float *ztenv, device const float *pvom, device const float *pvol, device float *ptte, device float *zuv2, int ncol, int klev, float cp, float rd, float rv, float xlv, float xlf, float grav) {

    if (i >= ncol) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xlf, grav);
#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    const int ld = ldcum[i];
    const int ktop = kctop[i];
    float zsum12 = 0.0f;
    float zsum22 = 0.0f;

    /* :1034-1048  the integrals */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(zuv2, jk) = 0.0f;
        if (ld && jk >= ktop - 1) {
            const float zdz =
                nt_add(NTC(paph, jk + 1), -NTC(paph, jk));
            const float zduten =
                nt_add(NTC(pvom, jk), -NTC(ztenu, jk));
            const float zdvten =
                nt_add(NTC(pvol, jk), -NTC(ztenv, jk));
            NTC(zuv2, jk) = sqrt(nt_add(
                nt_mul(zduten, zduten), nt_mul(zdvten, zdvten)));
            zsum22 = nt_add(zsum22, nt_mul(NTC(zuv2, jk), zdz));
            zsum12 = nt_add(zsum12, -nt_mul(nt_add(
                nt_mul(NTC(puen, jk), zduten),
                nt_mul(NTC(pven, jk), zdvten)), zdz));
        }
    }

    /* :1049-1056  the heating.  ADD, do not assign. */
    for (int jk = 1; jk <= klev; ++jk) {
        if (ld && jk >= ktop - 1) {
            const float ztdis = nt_div(
                nt_mul(nt_mul(c.rcpd, zsum12), NTC(zuv2, jk)),
                max(1.0e-15f, zsum22));
            NTC(ptte, jk) = nt_add(NTC(ptte, jk), ztdis);
        }
    }
#undef NTC

}
kernel void ntiedtke_ke_dissipation(
    constant NtiedtkeKeDissipationResources &r [[buffer(0)]],
    constant NtiedtkeKeDissipationParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_ke_dissipation(i, r.ldcum, r.kctop, r.paph, r.puen, r.pven, r.ztenu, r.ztenv, r.pvom, r.pvol, r.ptte, r.zuv2, p.ncol, p.klev, p.cp, p.rd, p.rv, p.xlv, p.xlf, p.grav);
}

static inline void nt_body_ntiedtke_post_run(int i, device const float *exner, device const float *qv, device const float *qc, device const float *qi, device const float *t, device const float *u, device const float *v, device const float *tf, device const float *qvf, device const float *qcf, device const float *qif, device const float *uf, device const float *vf, device const float *rn, device float *rthcuten, device float *rqvcuten, device float *rqccuten, device float *rqicuten, device float *rucuten, device float *rvcuten, device float *raincv, device float *pratec, int ncol, int klev, int stepcu, float dt) {

    if (i >= ncol) return;

#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    /* :504-505.  stepcu is an INTEGER in the reference and Fortran
     * promotes it; the reciprocal is a separate rounding and both are
     * kept rather than folded into one constant. */
    const float fstepcu = (float)stepcu;
    const float delt = nt_mul(dt, fstepcu);
    const float rdelt = nt_div(1.0f, delt);

    /* :506-509.  pratec divides by ONE product, stepcu*dt -- not by
     * stepcu and then by dt, which rounds twice. */
    raincv[i] = nt_div(rn[i], fstepcu);
    pratec[i] = nt_div(rn[i], nt_mul(fstepcu, dt));

    /* :511-524 */
    for (int jk = 1; jk <= klev; ++jk) {
        const int zz = klev + 1 - jk;
        NTC(rthcuten, jk) = nt_mul(
            nt_div(nt_add(NTC(tf, zz), -NTC(t, jk)), NTC(exner, jk)),
            rdelt);
        NTC(rqvcuten, jk) =
            nt_mul(nt_add(NTC(qvf, zz), -NTC(qv, jk)), rdelt);
        NTC(rqccuten, jk) =
            nt_mul(nt_add(NTC(qcf, zz), -NTC(qc, jk)), rdelt);
        NTC(rqicuten, jk) =
            nt_mul(nt_add(NTC(qif, zz), -NTC(qi, jk)), rdelt);
        NTC(rucuten, jk) =
            nt_mul(nt_add(NTC(uf, zz), -NTC(u, jk)), rdelt);
        NTC(rvcuten, jk) =
            nt_mul(nt_add(NTC(vf, zz), -NTC(v, jk)), rdelt);
    }
#undef NTC

}
kernel void ntiedtke_post_run(
    constant NtiedtkePostRunResources &r [[buffer(0)]],
    constant NtiedtkePostRunParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_post_run(i, r.exner, r.qv, r.qc, r.qi, r.t, r.u, r.v, r.tf, r.qvf, r.qcf, r.qif, r.uf, r.vf, r.rn, r.rthcuten, r.rqvcuten, r.rqccuten, r.rqicuten, r.rucuten, r.rvcuten, r.raincv, r.pratec, p.ncol, p.klev, p.stepcu, p.dt);
}

static inline void nt_body_ntiedtke_post_conversion(int i, device const float *pcte, device const float *ztp1, device const float *ptte, device const float *ztt, device const float *pqte, device const float *zqq, device float *zqp1, device const float *qcf, device const float *qif, device const float *uf, device const float *vf, device const float *pvom, device const float *pvol, device const float *prsfc, device const float *pssfc, device float *pqc, device float *pqi, device float *pt, device float *pqv, device float *pu, device float *pv, device float *zprecc, int ncol, int klev, float delt) {

    if (i >= ncol) return;

#define NTC(a, k) (a)[(ulong)(k) * (ulong)ncol + (ulong)i]

    /* :296-305.  fliq/fice split the detrained condensate by temperature.
     * The false arm CARRIES the incoming value; it does not zero. */
    for (int jk = 1; jk <= klev; ++jk) {
        if (NTC(pcte, jk) > 0.0f) {
            const float fliq = nt_foealfa(NTC(ztp1, jk));
            const float fice = nt_add(1.0f, -fliq);
            NTC(pqc, jk) = nt_add(
                NTC(qcf, jk),
                nt_mul(nt_mul(fliq, NTC(pcte, jk)), delt));
            NTC(pqi, jk) = nt_add(
                NTC(qif, jk),
                nt_mul(nt_mul(fice, NTC(pcte, jk)), delt));
        } else {
            NTC(pqc, jk) = NTC(qcf, jk);
            NTC(pqi, jk) = NTC(qif, jk);
        }
    }

    /* :308-314 */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(pt, jk) = nt_add(
            NTC(ztp1, jk),
            nt_mul(nt_add(NTC(ptte, jk), -NTC(ztt, jk)), delt));
        const float q = nt_add(
            NTC(zqp1, jk),
            nt_mul(nt_add(NTC(pqte, jk), -NTC(zqq, jk)), delt));
        NTC(zqp1, jk) = q;
        NTC(pqv, jk) = nt_div(q, nt_add(1.0f, -q));
    }

    /* :316-318.  amax1 clamps a negative flux product to zero. */
    zprecc[i] = max(0.0f, nt_mul(nt_add(prsfc[i], pssfc[i]), delt));

    /* :319-325, lmfdudv is .true. at :55 */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(pu, jk) = nt_add(NTC(uf, jk),
                                nt_mul(NTC(pvom, jk), delt));
        NTC(pv, jk) = nt_add(NTC(vf, jk),
                                nt_mul(NTC(pvol, jk), delt));
    }
#undef NTC

}
kernel void ntiedtke_post_conversion(
    constant NtiedtkePostConversionResources &r [[buffer(0)]],
    constant NtiedtkePostConversionParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    nt_body_ntiedtke_post_conversion(i, r.pcte, r.ztp1, r.ptte, r.ztt, r.pqte, r.zqq, r.zqp1, r.qcf, r.qif, r.uf, r.vf, r.pvom, r.pvol, r.prsfc, r.pssfc, r.pqc, r.pqi, r.pt, r.pqv, r.pu, r.pv, r.zprecc, p.ncol, p.klev, p.delt);
}

kernel void ntiedtke_force_fold(
    constant NtiedtkeForceFoldResources &r [[buffer(0)]],
    constant NtiedtkeForceFoldParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(p.nz) * ulong(p.ncol);
    if (ulong(gid) >= count) return;
    const ulong i = ulong(gid);
    const float radiation = nt_add(r.rthratenlw[i], r.rthratensw[i]);
    const float theta_forcing = nt_add(
        nt_add(r.gf_rthdynten[i], radiation), r.gf_rthblten[i]);
    r.thften[i] = nt_mul(theta_forcing, r.exner[i]);
    r.qvften[i] = nt_add(r.gf_rqvdynten[i], r.gf_rqvblten[i]);
}

kernel void ntiedtke_gather_level(
    constant NtiedtkeGatherLevelResources &r [[buffer(0)]],
    constant NtiedtkeGatherLevelParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(p.nz) * ulong(p.width);
    if (ulong(gid) >= count) return;
    const int k = int(gid) / p.width;
    const int local = int(gid) - k * p.width;
    const int source_col = p.start + min(local, p.count - 1);
    const ulong source = ulong(k) * ulong(p.full_ncol) + ulong(source_col);
    const ulong target = ulong(k) * ulong(p.width) + ulong(local);
    r.target[target] = r.source[source];
}

kernel void ntiedtke_gather_interface(
    constant NtiedtkeGatherInterfaceResources &r [[buffer(0)]],
    constant NtiedtkeGatherInterfaceParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(p.nzi) * ulong(p.width);
    if (ulong(gid) >= count) return;
    const int k = int(gid) / p.width;
    const int local = int(gid) - k * p.width;
    const int source_col = p.start + min(local, p.count - 1);
    const ulong source = ulong(k) * ulong(p.full_ncol) + ulong(source_col);
    const ulong target = ulong(k) * ulong(p.width) + ulong(local);
    r.target[target] = r.source[source];
}

kernel void ntiedtke_gather_surface(
    constant NtiedtkeGatherSurfaceResources &r [[buffer(0)]],
    constant NtiedtkeGatherSurfaceParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int local = int(gid);
    if (local >= p.width) return;
    const int source_col = p.start + min(local, p.count - 1);
    r.target[local] = r.source[source_col];
}

kernel void ntiedtke_scatter_level(
    constant NtiedtkeScatterLevelResources &r [[buffer(0)]],
    constant NtiedtkeScatterLevelParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const ulong count = ulong(p.nz) * ulong(p.count);
    if (ulong(gid) >= count) return;
    const int k = int(gid) / p.count;
    const int local = int(gid) - k * p.count;
    const ulong source = ulong(k) * ulong(p.width) + ulong(local);
    const ulong target = ulong(k) * ulong(p.full_ncol)
                       + ulong(p.start + local);
    r.target[target] = r.source[source];
}

kernel void ntiedtke_scatter_surface(
    constant NtiedtkeScatterSurfaceResources &r [[buffer(0)]],
    constant NtiedtkeScatterSurfaceParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int local = int(gid);
    if (local >= p.count) return;
    r.target[p.start + local] = r.source[local];
}

kernel void ntiedtke_llo3_reduce_columns(
    constant NtiedtkeLlo3ReduceColumnsResources &r [[buffer(0)]],
    constant NtiedtkeLlo3ReduceColumnsParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int column = int(gid);
    if (column >= p.ncol) return;
    const ulong top = ulong(p.nz) * ulong(p.ncol) + ulong(column);
    int any_label = 0;
    for (int k = 1; k <= p.nz; ++k) {
        const ulong index = ulong(k) * ulong(p.ncol) + ulong(column);
        any_label |= r.klab[index] > 0 ? 2 : 0;
    }
    r.partials[2 * column] = r.klab[top] > 0 ? 1 : 0;
    r.partials[2 * column + 1] = any_label;
}

kernel void ntiedtke_llo3_reduce_finish(
    constant NtiedtkeLlo3ReduceFinishResources &r [[buffer(0)]],
    constant NtiedtkeLlo3ReduceFinishParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;
    int first = 0;
    int ever = 0;
    for (int column = 0; column < p.ncol; ++column) {
        first |= r.partials[2 * column];
        ever |= r.partials[2 * column + 1];
    }
    const int mask = first | ever;
    r.llo3_mask[0] = mask;
    r.flags[p.chunk_index] = mask;
}

kernel void ntiedtke_driver_rain_merge(
    constant NtiedtkeDriverRainMergeResources &r [[buffer(0)]],
    constant NtiedtkeDriverRainMergeParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    const int i = int(gid);
    if (i >= p.ncol) return;
    const float amount = r.increment[i];
    r.rainc[i] = nt_add(r.rainc[i], amount);
    const float wetting = max(amount, 0.0f);
    r.pending_rainbl[i] = nt_add(r.pending_rainbl[i], wetting);
    if (p.has_surface_raincv != 0) {
        r.surface_raincv[i] = nt_add(r.surface_raincv[i], wetting);
    }
}
