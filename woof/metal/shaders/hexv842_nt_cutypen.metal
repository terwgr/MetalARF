#include "hexv842_nt_cutypen_bindings.metalh"
#include <metal_stdlib>
#include "soft_gfk_exp.metalh"
#include "soft_gfk_pow.metalh"
using namespace metal;
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
#define NTC(a,k) (a)[(ulong)(k)*(ulong)ncol+(ulong)i]
static inline float mpas_nt_add(float a,float b) { return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b))); }
static inline float mpas_nt_sub(float a,float b) { return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b))); }
static inline float mpas_nt_mul(float a,float b) { return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b))); }
static inline float mpas_nt_div(float a,float b) { return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b))); }
static inline float mpas_nt_sqrt(float a) { return as_type<float>(soft32_sqrt(as_type<uint>(a))); }
static inline float sqrt32(float a) { return mpas_nt_sqrt(a); }
static inline float mpas_nt_neg(float a) { return as_type<float>(as_type<uint>(a)^0x80000000u); }
static inline int mpas_nt_int(float x) { if (!(x >= -2147483648.0f && x < 2147483648.0f)) return -2147483647-1; return int(x); }
static inline float gfk_exp(float a) { return as_type<float>(soft_gfk_exp(as_type<uint>(a))); }
static inline float gfk_pow(float a,float b) { return as_type<float>(soft_gfk_pow_bits(as_type<uint>(a),as_type<uint>(b))); }
struct NtConst { float cpd,rcpd,c2es,vtmpc1,r5alvcp,r5alscp,ralvdcp,ralsdcp,g; float rd,zrg,ralfdcp,c5les,c5ies,alv; float als,alf; };
static inline NtConst nt_init(float cp, float rd, float rv,
                                           float xlv, float xls, float xlf, float g) {
    NtConst c;
    const float als = xls;
    c.cpd  = cp;
    c.g    = g;
    c.rcpd = mpas_nt_div(1.0f, cp);
    c.c2es = mpas_nt_div(mpas_nt_mul(NT_C1ES, rd), rv);
    const float c5les = mpas_nt_mul(NT_C3LES, mpas_nt_sub(NT_TMELT, NT_C4LES));
    const float c5ies = mpas_nt_mul(NT_C3IES, mpas_nt_sub(NT_TMELT, NT_C4IES));
    c.r5alvcp = mpas_nt_mul(mpas_nt_mul(c5les, xlv), c.rcpd);
    c.r5alscp = mpas_nt_mul(mpas_nt_mul(c5ies, als), c.rcpd);
    c.ralvdcp = mpas_nt_mul(xlv, c.rcpd);
    c.ralsdcp = mpas_nt_mul(als, c.rcpd);
    c.vtmpc1  = mpas_nt_sub(mpas_nt_div(rv, rd), 1.0f);
    c.rd      = rd;
    c.zrg     = mpas_nt_div(1.0f, g);
    c.ralfdcp = mpas_nt_mul(xlf, c.rcpd);
    c.c5les   = c5les;
    c.c5ies   = c5ies;
    c.alv     = xlv;
    c.als     = als;
    c.alf     = xlf;
    return c;
}
static inline float nt_foealfa(float tt) {
    const float clamped = max(NT_RTICE, min(NT_RTWAT, tt));
    const float r = mpas_nt_div(mpas_nt_sub(clamped, NT_RTICE),
                              mpas_nt_sub(NT_RTWAT, NT_RTICE));
    return min(1.0f, mpas_nt_mul(r, r));
}
static inline float nt_foeewm(float tt, thread const NtConst &c) {
    const float a = nt_foealfa(tt);
    const float el = gfk_exp(mpas_nt_div(
        mpas_nt_mul(NT_C3LES, mpas_nt_sub(tt, NT_TMELT)),
        mpas_nt_sub(tt, NT_C4LES)));
    const float ei = gfk_exp(mpas_nt_div(
        mpas_nt_mul(NT_C3IES, mpas_nt_sub(tt, NT_TMELT)),
        mpas_nt_sub(tt, NT_C4IES)));
    return mpas_nt_mul(c.c2es, mpas_nt_add(mpas_nt_mul(a, el),
                                       mpas_nt_mul(mpas_nt_sub(1.0f, a), ei)));
}
static inline float nt_foeldcpm(float tt, thread const NtConst &c) {
    const float a = nt_foealfa(tt);
    return mpas_nt_add(mpas_nt_mul(a, c.ralvdcp),
                     mpas_nt_mul(mpas_nt_sub(1.0f, a), c.ralsdcp));
}
static inline void nt_cuadjtqn1(thread float *pt, thread float *pq,
                                             float psp, thread const NtConst &c) {
    const float zqp = mpas_nt_div(1.0f, psp);
    float zl = mpas_nt_div(1.0f, mpas_nt_sub(*pt, NT_C4LES));
    float zi = mpas_nt_div(1.0f, mpas_nt_sub(*pt, NT_C4IES));
    float a = nt_foealfa(*pt);
    float el = gfk_exp(mpas_nt_mul(mpas_nt_mul(NT_C3LES,
                                           mpas_nt_sub(*pt, NT_TMELT)), zl));
    float ei = gfk_exp(mpas_nt_mul(mpas_nt_mul(NT_C3IES,
                                           mpas_nt_sub(*pt, NT_TMELT)), zi));
    float zqsat = mpas_nt_mul(c.c2es,
        mpas_nt_add(mpas_nt_mul(a, el), mpas_nt_mul(mpas_nt_sub(1.0f, a), ei)));
    zqsat = mpas_nt_mul(zqsat, zqp);
    zqsat = min(0.5f, zqsat);
    float zcor = mpas_nt_sub(1.0f, mpas_nt_mul(c.vtmpc1, zqsat));
    float zf = mpas_nt_add(
        mpas_nt_mul(mpas_nt_mul(a, c.r5alvcp), mpas_nt_mul(zl, zl)),
        mpas_nt_mul(mpas_nt_mul(mpas_nt_sub(1.0f, a), c.r5alscp),
                  mpas_nt_mul(zi, zi)));
    const float zcond = mpas_nt_div(
        mpas_nt_sub(mpas_nt_mul(*pq, mpas_nt_mul(zcor, zcor)),
                  mpas_nt_mul(zqsat, zcor)),
        mpas_nt_add(mpas_nt_mul(zcor, zcor), mpas_nt_mul(zqsat, zf)));
    if (zcond > 0.0f) {
        *pt = mpas_nt_add(*pt, mpas_nt_mul(nt_foeldcpm(*pt, c), zcond));
        *pq = mpas_nt_sub(*pq, zcond);
        zl = mpas_nt_div(1.0f, mpas_nt_sub(*pt, NT_C4LES));
        zi = mpas_nt_div(1.0f, mpas_nt_sub(*pt, NT_C4IES));
        a = nt_foealfa(*pt);
        el = gfk_exp(mpas_nt_mul(mpas_nt_mul(NT_C3LES,
                                         mpas_nt_sub(*pt, NT_TMELT)), zl));
        ei = gfk_exp(mpas_nt_mul(mpas_nt_mul(NT_C3IES,
                                         mpas_nt_sub(*pt, NT_TMELT)), zi));
        zqsat = mpas_nt_mul(c.c2es,
            mpas_nt_add(mpas_nt_mul(a, el), mpas_nt_mul(mpas_nt_sub(1.0f, a), ei)));
        zqsat = mpas_nt_mul(zqsat, zqp);
        zqsat = min(0.5f, zqsat);
        zcor = mpas_nt_sub(1.0f, mpas_nt_mul(c.vtmpc1, zqsat));
        zf = mpas_nt_add(
            mpas_nt_mul(mpas_nt_mul(a, c.r5alvcp), mpas_nt_mul(zl, zl)),
            mpas_nt_mul(mpas_nt_mul(mpas_nt_sub(1.0f, a), c.r5alscp),
                      mpas_nt_mul(zi, zi)));
        float zcond1 = mpas_nt_div(
            mpas_nt_sub(mpas_nt_mul(*pq, mpas_nt_mul(zcor, zcor)),
                      mpas_nt_mul(zqsat, zcor)),
            mpas_nt_add(mpas_nt_mul(zcor, zcor), mpas_nt_mul(zqsat, zf)));
        if (fabs(zcond) < 1.0e-20f) zcond1 = 0.0f;
        *pt = mpas_nt_add(*pt, mpas_nt_mul(nt_foeldcpm(*pt, c), zcond1));
        *pq = mpas_nt_sub(*pq, zcond1);
    }
}
static inline void nt_cloud_base(
        int jk, int klev, int ncol, int i,
        device float *ptu, device float *pqu, device float *plu, device float *kup, device int *klab,
        device const float *paph, thread int *kcbot, thread const NtConst &c) {
    const int ik = jk + 1;
    float zqsu = mpas_nt_div(nt_foeewm(NTC(ptu, ik), c), NTC(paph, ik));
    zqsu = min(0.5f, zqsu);
    float zcor = mpas_nt_div(1.0f, mpas_nt_sub(1.0f, mpas_nt_mul(c.vtmpc1, zqsu)));
    zqsu = mpas_nt_mul(zqsu, zcor);
    const float zdq = min(0.0f, mpas_nt_sub(NTC(pqu, ik), zqsu));
    const float zalfaw = nt_foealfa(NTC(ptu, ik));
    const float dl = mpas_nt_sub(NTC(ptu, ik), NT_C4LES);
    const float di = mpas_nt_sub(NTC(ptu, ik), NT_C4IES);
    const float zfacw = mpas_nt_div(c.c5les, mpas_nt_mul(dl, dl));
    const float zfaci = mpas_nt_div(c.c5ies, mpas_nt_mul(di, di));
    const float zfac = mpas_nt_add(mpas_nt_mul(zalfaw, zfacw),
                                 mpas_nt_mul(mpas_nt_sub(1.0f, zalfaw), zfaci));
    const float zesdp = mpas_nt_div(nt_foeewm(NTC(ptu, ik), c), NTC(paph, ik));
    zcor = mpas_nt_div(1.0f, mpas_nt_sub(1.0f, mpas_nt_mul(c.vtmpc1, zesdp)));
    const float zdqsdt = mpas_nt_mul(mpas_nt_mul(zfac, zcor), zqsu);
    const float zdtdp = mpas_nt_div(mpas_nt_mul(c.rd, NTC(ptu, ik)),
                                  mpas_nt_mul(c.cpd, NTC(paph, ik)));
    const float zdp = mpas_nt_div(zdq, mpas_nt_mul(zdqsdt, zdtdp));
    const float zcbase = (float)mpas_nt_int(mpas_nt_add(NTC(paph, ik), zdp));
    const float zpdifftop = mpas_nt_sub(zcbase, NTC(paph, jk));
    const float zpdiffbot = mpas_nt_sub(NTC(paph, jk + 1), zcbase);
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
static inline void nt_ascent_step(
        int jk, int levels, int klev, int ncol, int i, bool deep,
        device float *ptu, device float *pqu, device float *plu, device float *dh, device float *dhen,
        device float *kup, device float *vptu, device float *vten, device float *zbuo, device float *abuoy,
        device int *klab,
        device const float *ptenh, device const float *pqenh, device const float *pgeo,
        device const float *pgeoh, device const float *paph, device const float *pqsen,
        thread int *kcbot, thread int *kctop, thread bool *loflag, thread bool *lldcum,
        thread const NtConst &c) {
    float eta;
    if (deep) {
        const float r = mpas_nt_div(NTC(pqsen, jk), NTC(pqsen, levels));
        const float fscale = min(1.0f, mpas_nt_mul(mpas_nt_mul(r, r), r));
        eta = mpas_nt_mul(1.75e-3f, fscale);
    } else {
        eta = mpas_nt_add(mpas_nt_div(0.8f, mpas_nt_mul(NTC(pgeo, jk), c.zrg)),
                        2.0e-4f);
    }
    const float dz = mpas_nt_mul(mpas_nt_sub(NTC(pgeoh, jk),
                                         NTC(pgeoh, jk + 1)), c.zrg);
    const float coef = mpas_nt_mul(mpas_nt_mul(0.5f, eta), dz);
    NTC(dhen, jk) = mpas_nt_add(NTC(pgeoh, jk),
                              mpas_nt_mul(c.cpd, NTC(ptenh, jk)));
    NTC(dh, jk) = mpas_nt_div(
        mpas_nt_add(mpas_nt_mul(coef, mpas_nt_add(NTC(dhen, jk + 1), NTC(dhen, jk))),
                  mpas_nt_mul(mpas_nt_sub(1.0f, coef), NTC(dh, jk + 1))),
        mpas_nt_add(1.0f, coef));
    NTC(pqu, jk) = mpas_nt_div(
        mpas_nt_add(mpas_nt_mul(coef, mpas_nt_add(NTC(pqenh, jk + 1),
                                            NTC(pqenh, jk))),
                  mpas_nt_mul(mpas_nt_sub(1.0f, coef), NTC(pqu, jk + 1))),
        mpas_nt_add(1.0f, coef));
    NTC(ptu, jk) = mpas_nt_mul(mpas_nt_sub(NTC(dh, jk), NTC(pgeoh, jk)), c.rcpd);
    const float zqold = NTC(pqu, jk);

    float t = NTC(ptu, jk), q = NTC(pqu, jk);
    nt_cuadjtqn1(&t, &q, NTC(paph, jk), c);
    NTC(ptu, jk) = t;
    NTC(pqu, jk) = q;

    const float zdq = max(mpas_nt_sub(zqold, NTC(pqu, jk)), 0.0f);
    NTC(plu, jk) = mpas_nt_add(NTC(plu, jk + 1), zdq);
    const float zlglac = mpas_nt_mul(zdq,
        mpas_nt_sub(mpas_nt_sub(1.0f, nt_foealfa(NTC(ptu, jk))),
                  mpas_nt_sub(1.0f, nt_foealfa(NTC(ptu, jk + 1)))));
    NTC(plu, jk) = deep ? mpas_nt_mul(0.5f, NTC(plu, jk))
                        : min(NTC(plu, jk), 5.0e-3f);
    NTC(dh, jk) = mpas_nt_add(NTC(pgeoh, jk),
        mpas_nt_mul(c.cpd, mpas_nt_add(NTC(ptu, jk),
                                   mpas_nt_mul(c.ralfdcp, zlglac))));
    NTC(vptu, jk) = mpas_nt_add(
        mpas_nt_mul(NTC(ptu, jk),
                  mpas_nt_sub(mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1,
                                                      NTC(pqu, jk))),
                            NTC(plu, jk))),
        mpas_nt_mul(c.ralfdcp, zlglac));
    NTC(vten, jk) = mpas_nt_mul(NTC(ptenh, jk),
        mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1, NTC(pqenh, jk))));
    NTC(zbuo, jk) = mpas_nt_div(mpas_nt_sub(NTC(vptu, jk), NTC(vten, jk)),
                              NTC(vten, jk));
    NTC(abuoy, jk) = mpas_nt_mul(
        mpas_nt_mul(mpas_nt_add(NTC(zbuo, jk), NTC(zbuo, jk + 1)), 0.5f), c.g);
    const float atop1 = mpas_nt_sub(1.0f, mpas_nt_mul(2.0f, coef));
    const float atop2 = mpas_nt_mul(mpas_nt_mul(2.0f, dz), NTC(abuoy, jk));
    const float abot = mpas_nt_add(1.0f, mpas_nt_mul(2.0f, coef));
    NTC(kup, jk) = mpas_nt_div(
        mpas_nt_add(mpas_nt_mul(atop1, NTC(kup, jk + 1)), atop2), abot);

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

kernel void hexv842_nt_cutypen(constant Hexv842NtCutypenResources &r [[buffer(0)]], constant Hexv842NtCutypenParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *inputs=r.inputs;
    device const float *surface=r.surface;
    device float *outs=r.outs;
    device int *iouts=r.iouts;
    device float *scr=r.scr;
    device int *scr_i=r.scr_i;

    int i=int(gid), ncol=p.ncol, nz=p.nz; if (i>=ncol) return;
    float cp=p.cp,rd=p.rd,rv=p.rv,xlv=p.xlv,xls=p.xls,xlf=p.xlf,grav=p.grav;
    ulong st=ulong(nz+2)*ulong(ncol);
    device const float *pqen=inputs+0*st,*ptenh=inputs+1*st,*pqenh=inputs+2*st,*pgeoh=inputs+3*st;
    device const float *paph=inputs+4*st,*pgeo=inputs+5*st,*pqsen=inputs+6*st,*pap=inputs+7*st,*pten=inputs+8*st;
    device const float *hfx=surface,*qfx=surface+ncol;
    device float *cutu=outs+0*st,*cuqu=outs+1*st,*culu=outs+2*st,*wbase_o=outs+3*st;
    device int *culab=iouts,*ldcum_o=iouts+st,*ktype_o=ldcum_o+ncol,*cubot_o=ktype_o+ncol;
    device int *cutop_o=cubot_o+ncol,*kdpl_o=cutop_o+ncol;
    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);
    const int klev = nz, klevm1 = nz - 1;
    const ulong stride = (ulong)(nz + 2) * (ulong)ncol;

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
            const float rho = mpas_nt_div(NTC(pap, klev),
                mpas_nt_mul(c.rd, mpas_nt_mul(NTC(pten, klev),
                    mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1, NTC(pqen, klev))))));
            const float part1 = mpas_nt_div(
                mpas_nt_mul(mpas_nt_mul(1.5f, 0.4f), NTC(pgeo, klev)),
                mpas_nt_mul(rho, NTC(pten, klev)));
            const float part2 = mpas_nt_sub(
                mpas_nt_neg(mpas_nt_mul(hfx[i], c.rcpd)),
                mpas_nt_mul(mpas_nt_mul(c.vtmpc1, NTC(pten, klev)), qfx[i]));
            const float root = mpas_nt_sub(0.001f, mpas_nt_mul(part1, part2));
            if (part2 < 0.0f) {
                /* conw = 1.2*root**t13 -- a genuine cube root through powf,
                 * so gfk_pow (glibc's) and not CUDA's. */
                const float conw = mpas_nt_mul(1.2f, gfk_pow(root, NT_T13));
                const float deltt = max(mpas_nt_div(
                    mpas_nt_mul(1.5f, hfx[i]),
                    mpas_nt_mul(mpas_nt_mul(rho, c.cpd), conw)), 0.0f);
                const float deltq = max(mpas_nt_div(
                    mpas_nt_mul(1.5f, qfx[i]), mpas_nt_mul(rho, conw)), 0.0f);
                NTC(kup, klev) = mpas_nt_mul(0.5f, mpas_nt_mul(conw, conw));
                NTC(pqu, klev) = mpas_nt_add(NTC(pqenh, klev), deltq);
                NTC(dhen, klev) = mpas_nt_add(NTC(pgeoh, klev),
                                            mpas_nt_mul(NTC(ptenh, klev),
                                                      c.cpd));
                NTC(dh, klev) = mpas_nt_add(NTC(dhen, klev),
                                          mpas_nt_mul(deltt, c.cpd));
                NTC(ptu, klev) = mpas_nt_mul(
                    mpas_nt_sub(NTC(dh, klev), NTC(pgeoh, klev)), c.rcpd);
                NTC(vptu, klev) = mpas_nt_mul(NTC(ptu, klev),
                    mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1, NTC(pqu, klev))));
                NTC(vten, klev) = mpas_nt_mul(NTC(ptenh, klev),
                    mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1, NTC(pqenh, klev))));
                NTC(zbuo, klev) = mpas_nt_div(
                    mpas_nt_sub(NTC(vptu, klev), NTC(vten, klev)),
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

    if (mpas_nt_sub(NTC(paph, kcbot), NTC(paph, kctop)) > NT_ZDNOPRC)
        lldcum = false;
    if (lldcum) {
        ktype = 2;  ldcum = true;
        wbase = sqrt32(max(mpas_nt_mul(2.0f, NTC(kup, kcbot)), 0.0f));
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
        if (mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, jk)) < 350.0e2f)
            itoppacel = jk;
    }

    for (int levels = klevm1 - 1; levels >= klev / 2 + 1; --levels) {
        // MPAS cutypen:1540-1555: clear scratch for every parcel trial.
        {
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
                if (mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, jk)) < 60.0e2f) {
                    tmix = 0.0f; qmix = 0.0f; zmix = 0.0f;
                    float pmix = 0.0f;
                    for (int nk = jk + 2; nk >= jk; --nk) {
                        if (pmix < 50.0e2f) {
                            const float dp = mpas_nt_sub(NTC(paph, nk),
                                                       NTC(paph, nk - 1));
                            tmix = mpas_nt_add(tmix,
                                             mpas_nt_mul(dp, NTC(ptenh, nk)));
                            qmix = mpas_nt_add(qmix,
                                             mpas_nt_mul(dp, NTC(pqenh, nk)));
                            zmix = mpas_nt_add(zmix,
                                             mpas_nt_mul(dp, NTC(pgeoh, nk)));
                            pmix = mpas_nt_add(pmix, dp);
                        }
                    }
                    tmix = mpas_nt_div(tmix, pmix);
                    qmix = mpas_nt_div(qmix, pmix);
                    zmix = mpas_nt_div(zmix, pmix);
                } else {
                    tmix = NTC(ptenh, jk + 1);
                    qmix = NTC(pqenh, jk + 1);
                    zmix = NTC(pgeoh, jk + 1);
                }
                NTC(pqu, jk + 1) = mpas_nt_add(qmix, deltq_d);
                NTC(dhen, jk + 1) = mpas_nt_add(zmix, mpas_nt_mul(tmix, c.cpd));
                NTC(dh, jk + 1) = mpas_nt_add(NTC(dhen, jk + 1),
                                            mpas_nt_mul(deltt_d, c.cpd));
                NTC(ptu, jk + 1) = mpas_nt_mul(
                    mpas_nt_sub(NTC(dh, jk + 1), NTC(pgeoh, jk + 1)), c.rcpd);
                NTC(kup, jk + 1) = 0.5f;
                NTC(klab, jk + 1) = 1;
                NTC(vptu, jk + 1) = mpas_nt_mul(NTC(ptu, jk + 1),
                    mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1, NTC(pqu, jk + 1))));
                NTC(vten, jk + 1) = mpas_nt_mul(NTC(ptenh, jk + 1),
                    mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1, NTC(pqenh, jk + 1))));
                NTC(zbuo, jk + 1) = mpas_nt_div(
                    mpas_nt_sub(NTC(vptu, jk + 1), NTC(vten, jk + 1)),
                    NTC(vten, jk + 1));
            }
            NTC(klab, jk) = 0;
            nt_ascent_step(jk, levels, klev, ncol, i, true,
                           ptu, pqu, plu, dh, dhen, kup, vptu, vten, zbuo,
                           abuoy, klab, ptenh, pqenh, pgeo, pgeoh, paph,
                           pqsen, &kcbot, &kctop, &loflag, &lldcum, c);
        }

        if (mpas_nt_sub(NTC(paph, kcbot), NTC(paph, kctop)) < NT_ZDNOPRC)
            lldcum = false;
        if (lldcum) {
            ktype = 1;  ldcum = true;  deepflag = true;
            wbase = sqrt32(max(mpas_nt_mul(2.0f, NTC(kup, kcbot)), 0.0f));
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
