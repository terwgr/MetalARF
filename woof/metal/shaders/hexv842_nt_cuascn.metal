#include "hexv842_nt_cuascn_bindings.metalh"
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

#define NT_CPRCON 1.4e-3f
#define NT_RTBER (NT_TMELT - 5.0f)
#define NT_CMFCMIN 1.0e-10f
#define NT_CMFCMAX 1.0f
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
static inline void nt_cubasmcn(
        int kk, int ncol, int i,
        device const float *pten, device const float *pqen, device const float *pqsen,
        device const float *pverv, device const float *pgeo, device const float *pgeoh,
        device int *ldcum, device int *ktype, device int *klab,
        /* The single plrain slot to clear -- kk+1 -- taken by ADDRESS
         * so cuascn, which keeps zlrain in a register, shares this
         * body instead of forking it. */
        thread float *plrain_slot,
        device float *pmfu, device float *pmfub, device int *kcbot,
        device float *ptu, device float *pqu, device float *plu,
        device float *pmfus, device float *pmfuq, device float *pmful, device float *pdmfup,
        thread const NtConst &c) {
    /* if(.not.ldcum .and. klab(kk+1) == 0) */
    if (ldcum[i] != 0 || NTC(klab, kk + 1) != 0) return;
    /* lmfmid is a parameter, .true.  pgeo*zrg is height in metres, so the
     * window below is 500 m to 10 km. */
    const float hgt = mpas_nt_mul(NTC(pgeo, kk), c.zrg);
    if (!(NTC(pqen, kk) > mpas_nt_mul(0.80f, NTC(pqsen, kk))
          && hgt > 5.0e2f && hgt < 1.0e4f)) return;

    NTC(ptu, kk + 1) = mpas_nt_mul(
        mpas_nt_sub(mpas_nt_add(mpas_nt_mul(c.cpd, NTC(pten, kk)), NTC(pgeo, kk)),
                  NTC(pgeoh, kk + 1)), c.rcpd);
    NTC(pqu, kk + 1) = NTC(pqen, kk);
    NTC(plu, kk + 1) = 0.0f;
    float zzzmb = max(NT_CMFCMIN, mpas_nt_neg(mpas_nt_mul(NTC(pverv, kk), c.zrg)));
    zzzmb = min(zzzmb, NT_CMFCMAX);
    pmfub[i] = zzzmb;
    NTC(pmfu, kk + 1) = pmfub[i];
    NTC(pmfus, kk + 1) = mpas_nt_mul(pmfub[i],
        mpas_nt_add(mpas_nt_mul(c.cpd, NTC(ptu, kk + 1)), NTC(pgeoh, kk + 1)));
    NTC(pmfuq, kk + 1) = mpas_nt_mul(pmfub[i], NTC(pqu, kk + 1));
    NTC(pmful, kk + 1) = 0.0f;
    NTC(pdmfup, kk + 1) = 0.0f;
    kcbot[i] = kk;
    NTC(klab, kk + 1) = 1;
    *plrain_slot = 0.0f;
    ktype[i] = 3;
}
static inline void nt_cuentrn(
        int kk, int ncol, int i,
        device const int *kcbot, device const int *ldcum, bool ldwork,
        device const float *pgeoh, device const float *pmfu,
        thread float *pdmfen, thread float *pdmfde, thread const NtConst &c) {
    if (!ldwork) return;
    *pdmfen = 0.0f;
    *pdmfde = 0.0f;
    const float zentr = 0.0f;
    if (ldcum[i] == 0) return;
    const float zdz = mpas_nt_mul(
        mpas_nt_sub(NTC(pgeoh, kk), NTC(pgeoh, kk + 1)), c.zrg);
    const float zmf = mpas_nt_mul(NTC(pmfu, kk + 1), zdz);
    if (kk < kcbot[i]) {
        *pdmfen = mpas_nt_mul(zentr, zmf);
        *pdmfde = mpas_nt_mul(0.75e-4f, zmf);
    }
}
static inline void nt_cuadjtqn1(device float *pt,device float *pq,float psp,thread const NtConst &c) {
    float t=*pt,q=*pq; nt_cuadjtqn1(&t,&q,psp,c); *pt=t; *pq=q;
}


kernel void hexv842_nt_cuascn(constant Hexv842NtCuascnResources &r [[buffer(0)]], constant Hexv842NtCuascnParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *inputs=r.inputs;
    device const int *surf_in_i=r.surf_in_i;
    device const float *surf_in_f=r.surf_in_f;
    device float *levels=r.levels;
    device int *flags=r.flags;
    device float *surf_out=r.surf_out;
    device const int *mpas_llo3_start=r.mpas_llo3_start;

    int i=int(gid),ncol=p.ncol,klev=p.klev; if (i>=ncol) return;
    float ztmst=p.ztmst,cp=p.cp,rd=p.rd,rv=p.rv,xlv=p.xlv,xls=p.xls,xlf=p.xlf,grav=p.grav;
    ulong st=ulong(klev+2)*ulong(ncol);
    device const float *pten=inputs,*pqen=inputs+st,*pqsen=inputs+2*st,*pgeo=inputs+3*st;
    device const float *pgeoh=inputs+4*st,*pap=inputs+5*st,*paph=inputs+6*st,*pverv=inputs+7*st;
    device const int *lndj=surf_in_i,*kdpl=surf_in_i+ncol; device const float *wbase=surf_in_f;
    device float *ptenh=levels,*pqenh=levels+st,*ptu=levels+2*st,*pqu=levels+3*st,*plu=levels+4*st;
    device float *pmfu=levels+5*st,*pmfus=levels+6*st,*pmfuq=levels+7*st,*pmful=levels+8*st;
    device float *plude=levels+9*st,*pdmfup=levels+10*st,*plglac=levels+11*st,*pmfude_rate=levels+12*st;
    device int *klab=flags,*ldcum=flags+st,*ktype=ldcum+ncol,*kcbot=ktype+ncol,*kctop=kcbot+ncol,*kctop0=kctop+ncol;
    device float *pmfub=surf_out,*wup=surf_out+ncol;
    int llo3 = 0;

    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);

    const int klevm1 = klev - 1;

    /* :1893-1899.  Each in the reference's own association.
     *
     * CORRECTED.  This said cuascn's cap is "three times looser than the
     * closure's".  It is not: zcons2 is 3/(g*dt) in all three scopes that
     * declare it and they are identical.  The factor of three separates
     * zcons2 from zcons (1/(g*dt), cumastrn:468) -- one character apart,
     * both live in cumastrn, and zcons has exactly one consumer: the
     * momentum rescale at :1000. */
    const float zcons2 = mpas_nt_div(3.0f, mpas_nt_mul(c.g, ztmst));
    const float zfacbuo = mpas_nt_div(0.5f, mpas_nt_add(1.0f, 0.5f));
    const float zprcdgw = mpas_nt_mul(NT_CPRCON, c.zrg);
    const float z_cldmax = 5.0e-3f;
    const float z_cwifrac = 0.5f;
    const float z_cprc2 = 0.5f;
    const float z_cwdrag =
        mpas_nt_div(mpas_nt_mul(mpas_nt_div(3.0f, 8.0f), 0.506f), 0.2f);

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
        NTC(pmfus, KB) = mpas_nt_mul(
            MFUB, mpas_nt_add(mpas_nt_mul(c.cpd, NTC(ptu, KB)), NTC(pgeoh, KB)));
        NTC(pmfuq, KB) = mpas_nt_mul(MFUB, NTC(pqu, KB));
        NTC(pmful, KB) = mpas_nt_mul(MFUB, NTC(plu, KB));
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
                   ? mpas_nt_mul(0.5f, mpas_nt_mul(wb, wb)) : 0.0f;

    float zdmfen = 0.0f, zdmfde = 0.0f;

    /* ---- 4. the ascent (:1959-2245) -------------------------------- */
    for (int jk = klevm1; jk >= 3; --jk) {
        if (jk <= mpas_llo3_start[0]) llo3 = 1;
        /* cubasmcn (:1968).  It can zero zlrain at jk+1, so the register
         * goes in by reference. */
        nt_cubasmcn(jk, ncol, i, pten, pqen, pqsen, pverv, pgeo, pgeoh,
                    ldcum, ktype, klab, &zlrain_prev, pmfu, pmfub, kcbot,
                    ptu, pqu, plu, pmfus, pmfuq, pmful, pdmfup, c);

        float kup_cur = (jk == kb0)
                      ? mpas_nt_mul(0.5f, mpas_nt_mul(wb, wb)) : 0.0f;
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
            const float zmfmax = mpas_nt_mul(
                mpas_nt_sub(NTC(paph, jk), NTC(paph, jk - 1)), zcons2);
            if (MFUB > zmfmax) {
                const float zfac = mpas_nt_div(zmfmax, MFUB);
                NTC(pmfu, jk + 1) = mpas_nt_mul(NTC(pmfu, jk + 1), zfac);
                NTC(pmfus, jk + 1) = mpas_nt_mul(NTC(pmfus, jk + 1), zfac);
                NTC(pmfuq, jk + 1) = mpas_nt_mul(NTC(pmfuq, jk + 1), zfac);
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
            const float mpas_detrain_bound = mpas_nt_mul(0.75f, NTC(pmfu, jk + 1));
            if (mpas_detrain_bound < zdmfde) zdmfde = mpas_detrain_bound;
            if (jk == KB) {
                const float r = min(1.0f, mpas_nt_div(NTC(pqen, jk),
                                                      NTC(pqsen, jk)));
                zoentr = mpas_nt_mul(
                    mpas_nt_mul(mpas_nt_mul(-1.75e-3f, mpas_nt_sub(r, 1.0f)),
                              mpas_nt_sub(NTC(pgeoh, jk),
                                        NTC(pgeoh, jk + 1))), c.zrg);
                zoentr = mpas_nt_mul(min(0.4f, zoentr), NTC(pmfu, jk + 1));
            }
            if (jk < KB) {
                const float zmfmax = mpas_nt_mul(
                    mpas_nt_sub(NTC(paph, jk), NTC(paph, jk - 1)), zcons2);
                const float zxs =
                    max(mpas_nt_sub(NTC(pmfu, jk + 1), zmfmax), 0.0f);
                const float dpap =
                    mpas_nt_sub(NTC(pap, jk + 1), NTC(pap, jk));
                wupa = mpas_nt_add(wupa, mpas_nt_mul(kup_prev, dpap));
                /* :2036 has NO parentheses, so it associates left to
                 * right.  The wup accumulator one line up IS parenthesised.
                 * Bracketing the difference here costs 1 ULP in wup. */
                zdpmean = mpas_nt_sub(mpas_nt_add(zdpmean, NTC(pap, jk + 1)),
                                    NTC(pap, jk));
                zdmfen = zoentr;
                if (KT >= 2) {
                    zdmfen = mpas_nt_mul(2.0f, zdmfen);
                    zdmfde = zdmfen;
                }
                zdmfde = mpas_nt_mul(zdmfde, mpas_nt_sub(1.6f,
                    min(1.0f, mpas_nt_div(NTC(pqen, jk),
                                           NTC(pqsen, jk)))));
                const float zmftest =
                    mpas_nt_sub(mpas_nt_add(NTC(pmfu, jk + 1), zdmfen), zdmfde);
                float zchange = max(mpas_nt_sub(zmftest, zmfmax), 0.0f);
                const float zxe = max(mpas_nt_sub(zchange, zxs), 0.0f);
                zdmfen = mpas_nt_sub(zdmfen, zxe);
                zchange = mpas_nt_sub(zchange, zxe);
                zdmfde = mpas_nt_add(zdmfde, zchange);
            }
            /* :2050 writes pdmfen, which is a dead local.  Not stored. */
            NTC(pmfu, jk) =
                mpas_nt_sub(mpas_nt_add(NTC(pmfu, jk + 1), zdmfen), zdmfde);
            const float zqeen = mpas_nt_mul(NTC(pqenh, jk + 1), zdmfen);
            const float zseen = mpas_nt_mul(
                mpas_nt_add(mpas_nt_mul(c.cpd, NTC(ptenh, jk + 1)),
                          NTC(pgeoh, jk + 1)), zdmfen);
            const float zscde = mpas_nt_mul(
                mpas_nt_add(mpas_nt_mul(c.cpd, NTC(ptu, jk + 1)),
                          NTC(pgeoh, jk + 1)), zdmfde);
            const float zqude = mpas_nt_mul(NTC(pqu, jk + 1), zdmfde);
            NTC(plude, jk) = mpas_nt_mul(NTC(plu, jk + 1), zdmfde);
            const float zmfusk =
                mpas_nt_sub(mpas_nt_add(NTC(pmfus, jk + 1), zseen), zscde);
            const float zmfuqk =
                mpas_nt_sub(mpas_nt_add(NTC(pmfuq, jk + 1), zqeen), zqude);
            const float zmfulk =
                mpas_nt_sub(NTC(pmful, jk + 1), NTC(plude, jk));
            const float inv =
                mpas_nt_div(1.0f, max(NT_CMFCMIN, NTC(pmfu, jk)));
            NTC(plu, jk) = mpas_nt_mul(zmfulk, inv);
            NTC(pqu, jk) = mpas_nt_mul(zmfuqk, inv);
            NTC(ptu, jk) = mpas_nt_mul(
                mpas_nt_sub(mpas_nt_mul(zmfusk, inv), NTC(pgeoh, jk)), c.rcpd);
            NTC(ptu, jk) = max(100.0f, NTC(ptu, jk));
            NTC(ptu, jk) = min(400.0f, NTC(ptu, jk));
            zqold = NTC(pqu, jk);
            zlrain_cur = mpas_nt_mul(
                mpas_nt_mul(zlrain_prev,
                          mpas_nt_sub(NTC(pmfu, jk + 1), zdmfde)), inv);
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
        if (loflag) nt_cuadjtqn1(&NTC(ptu, jk), &NTC(pqu, jk), zph, c);

        const int adjusted = loflag && (NTC(pqu, jk) != zqold);

        /* :2086-2093 */
        if (adjusted) {
            NTC(plglac, jk) = mpas_nt_mul(NTC(plu, jk),
                mpas_nt_sub(mpas_nt_sub(1.0f, nt_foealfa(NTC(ptu, jk))),
                          mpas_nt_sub(1.0f, nt_foealfa(NTC(ptu, jk + 1)))));
            NTC(ptu, jk) = mpas_nt_add(NTC(ptu, jk),
                                     mpas_nt_mul(c.ralfdcp, NTC(plglac, jk)));
        }

        /* :2096-2179 */
        if (adjusted) {
            NTC(klab, jk) = 2;
            NTC(plu, jk) = mpas_nt_sub(mpas_nt_add(NTC(plu, jk), zqold),
                                     NTC(pqu, jk));
            const float zbc = mpas_nt_mul(NTC(ptu, jk), mpas_nt_sub(mpas_nt_sub(
                mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1, NTC(pqu, jk))),
                NTC(plu, jk + 1)), zlrain_prev));
            const float zbe = mpas_nt_mul(NTC(ptenh, jk),
                mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1, NTC(pqenh, jk))));
            zbuo_cur = mpas_nt_sub(zbc, zbe);
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
                    NTC(ptenh, jk) = mpas_nt_mul(0.5f,
                        mpas_nt_add(NTC(pten, jk), NTC(pten, jk - 1)));
                    NTC(pqenh, jk) = mpas_nt_mul(0.5f,
                        mpas_nt_add(NTC(pqen, jk), NTC(pqen, jk - 1)));
                    zbuo_cur = mpas_nt_sub(zbc, mpas_nt_mul(NTC(ptenh, jk),
                        mpas_nt_add(1.0f,
                                  mpas_nt_mul(c.vtmpc1, NTC(pqenh, jk)))));
                }
                const float zbuoc = mpas_nt_mul(mpas_nt_add(
                    mpas_nt_div(zbuo_cur, mpas_nt_mul(NTC(ptenh, jk),
                        mpas_nt_add(1.0f,
                                  mpas_nt_mul(c.vtmpc1, NTC(pqenh, jk))))),
                    mpas_nt_div(zbuo_prev, mpas_nt_mul(NTC(ptenh, jk + 1),
                        mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1,
                                                 NTC(pqenh, jk + 1)))))),
                    0.5f);
                const float zdkbuo = mpas_nt_mul(mpas_nt_mul(
                    mpas_nt_sub(NTC(pgeoh, jk), NTC(pgeoh, jk + 1)),
                    zfacbuo), zbuoc);
                /* Both arms are the same shape on purpose: a runtime branch
                 * over differently-shaped arithmetic is what let ptxas
                 * contract two clones differently once before. */
                const float zsel = (zdmfen > 0.0f) ? zdmfen : zdmfde;
                const float zdken = min(1.0f, mpas_nt_div(
                    mpas_nt_mul(mpas_nt_add(1.0f, z_cwdrag), zsel),
                    max(NT_CMFCMIN, NTC(pmfu, jk + 1))));
                kup_cur = mpas_nt_div(
                    mpas_nt_add(mpas_nt_mul(kup_prev, mpas_nt_sub(1.0f, zdken)),
                              zdkbuo),
                    mpas_nt_add(1.0f, zdken));
                if (zbuo_cur < 0.0f) {
                    float zkedke =
                        mpas_nt_div(kup_cur, max(1.0e-10f, kup_prev));
                    zkedke = max(0.0f, min(1.0f, zkedke));
                    const float zmfun =
                        mpas_nt_mul(mpas_nt_sqrt(zkedke), NTC(pmfu, jk + 1));
                    zdmfde = max(zdmfde,
                                   mpas_nt_sub(NTC(pmfu, jk + 1), zmfun));
                    NTC(plude, jk) = mpas_nt_mul(NTC(plu, jk + 1), zdmfde);
                    NTC(pmfu, jk) = mpas_nt_sub(
                        mpas_nt_add(NTC(pmfu, jk + 1), zdmfen), zdmfde);
                }
                if (zbuo_cur > -0.2f) {
                    const float rr = min(1.0f,
                        mpas_nt_div(NTC(pqen, jk - 1), NTC(pqsen, jk - 1)));
                    const float q3 = min(1.0f,
                        mpas_nt_div(NTC(pqsen, jk), NTC(pqsen, KB)));
                    zoentr = mpas_nt_mul(mpas_nt_mul(mpas_nt_mul(
                        mpas_nt_mul(1.75e-3f,
                                  mpas_nt_sub(0.3f, mpas_nt_sub(rr, 1.0f))),
                        mpas_nt_sub(NTC(pgeoh, jk - 1), NTC(pgeoh, jk))),
                        c.zrg), mpas_nt_mul(mpas_nt_mul(q3, q3), q3));
                    zoentr = mpas_nt_mul(min(0.4f, zoentr), NTC(pmfu, jk));
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
                    NTC(plude, jk) = mpas_nt_mul(NTC(plu, jk + 1), zdmfde);
                }
                if (NTC(pmfu, jk + 1) > 0.0f) NTC(pmfude_rate, jk) = zdmfde;
            }
        } else if (loflag && KT == 2 && NTC(pqu, jk) == zqold) {
            NTC(klab, jk) = 0;
            NTC(pmfu, jk) = 0.0f;
            kup_cur = 0.0f;
            zdmfde = NTC(pmfu, jk + 1);
            NTC(plude, jk) = mpas_nt_mul(NTC(plu, jk + 1), zdmfde);
            NTC(pmfude_rate, jk) = zdmfde;
        }

        /* :2182-2216  precipitation conversion */
        if (llo1) {
            const float zdshrd = (lndj[i] == 1) ? 5.0e-4f : 3.0e-4f;
            if (NTC(plu, jk) > zdshrd) {
                const float zwu = min(15.0f, mpas_nt_sqrt(
                    mpas_nt_mul(2.0f, max(0.1f, kup_prev))));
                const float zprcon =
                    mpas_nt_div(zprcdgw, mpas_nt_mul(0.75f, zwu));
                const float zdt = min(NT_RTBER - NT_RTICE,
                    max(mpas_nt_sub(NT_RTBER, NTC(ptu, jk)), 0.0f));
                const float zcbf =
                    mpas_nt_add(1.0f, mpas_nt_mul(z_cprc2, mpas_nt_sqrt(zdt)));
                const float zzco = mpas_nt_mul(zprcon, zcbf);
                const float zlcrit = mpas_nt_div(zdshrd, zcbf);
                const float zdfi =
                    mpas_nt_sub(NTC(pgeoh, jk), NTC(pgeoh, jk + 1));
                const float zc = mpas_nt_sub(NTC(plu, jk), zluold);
                const float rq = mpas_nt_div(NTC(plu, jk), zlcrit);
                const float zarg = mpas_nt_mul(rq, rq);
                const float zd = (zarg < 25.0f)
                    ? mpas_nt_mul(mpas_nt_mul(zzco,
                          mpas_nt_sub(1.0f, gfk_exp(mpas_nt_neg(zarg)))), zdfi)
                    : mpas_nt_mul(zzco, zdfi);
                const float zint = gfk_exp(mpas_nt_neg(zd));
                float zlnew = mpas_nt_add(mpas_nt_mul(zluold, zint),
                    mpas_nt_mul(mpas_nt_div(zc, zd), mpas_nt_sub(1.0f, zint)));
                zlnew = max(0.0f, min(NTC(plu, jk), zlnew));
                zlnew = min(z_cldmax, zlnew);
                zprecip = max(0.0f,
                    mpas_nt_sub(mpas_nt_add(zluold, zc), zlnew));
                NTC(pdmfup, jk) = mpas_nt_mul(zprecip, NTC(pmfu, jk));
                zlrain_cur = mpas_nt_add(zlrain_cur, zprecip);
                NTC(plu, jk) = zlnew;
            }
        }

        /* :2219-2236  rain fallout */
        if (llo1 && zlrain_cur > 0.0f) {
            const float zvw = mpas_nt_mul(21.18f, gfk_pow(zlrain_cur, 0.2f));
            const float zvi = mpas_nt_mul(z_cwifrac, zvw);
            const float zalfaw = nt_foealfa(NTC(ptu, jk));
            const float zvv = mpas_nt_add(mpas_nt_mul(zalfaw, zvw),
                mpas_nt_mul(mpas_nt_sub(1.0f, zalfaw), zvi));
            const float zrold = mpas_nt_sub(zlrain_cur, zprecip);
            const float zwu = min(15.0f, mpas_nt_sqrt(
                mpas_nt_mul(2.0f, max(0.1f, kup_cur))));
            const float zd = mpas_nt_div(zvv, zwu);
            const float zint = gfk_exp(mpas_nt_neg(zd));
            float zrnew = mpas_nt_add(mpas_nt_mul(zrold, zint),
                mpas_nt_mul(mpas_nt_div(zprecip, zd), mpas_nt_sub(1.0f, zint)));
            zrnew = max(0.0f, min(zlrain_cur, zrnew));
            zlrain_cur = zrnew;
        }

        /* :2239-2243 */
        if (loflag) {
            NTC(pmful, jk) = mpas_nt_mul(NTC(plu, jk), NTC(pmfu, jk));
            NTC(pmfus, jk) = mpas_nt_mul(
                mpas_nt_add(mpas_nt_mul(c.cpd, NTC(ptu, jk)), NTC(pgeoh, jk)),
                NTC(pmfu, jk));
            NTC(pmfuq, jk) = mpas_nt_mul(NTC(pqu, jk), NTC(pmfu, jk));
        }

        kup_prev = kup_cur;
        zlrain_prev = zlrain_cur;
        zbuo_prev = zbuo_cur;
    }

    /* ---- 5. final (:2248-2256) ------------------------------------- */
    if (ktp == -1) LD = 0;
    KB = max(KB, ktp);
    if (LD) {
        wupa = max(1.0e-2f, mpas_nt_div(wupa, max(1.0f, zdpmean)));
        wupa = mpas_nt_sqrt(mpas_nt_mul(2.0f, wupa));
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
