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

#define NT_CMFDEPS 0.30f
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
static inline float nt_foelhm(float tt, thread const NtConst &c) {
    const float a = nt_foealfa(tt);
    return mpas_nt_add(mpas_nt_mul(a, c.alv),
                     mpas_nt_mul(mpas_nt_sub(1.0f, a), c.als));
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
static inline float nt_foedem(float tt, thread const NtConst &c) {
    const float a = nt_foealfa(tt);
    const float dl = mpas_nt_sub(tt, NT_C4LES);
    const float di = mpas_nt_sub(tt, NT_C4IES);
    return mpas_nt_add(
        mpas_nt_mul(mpas_nt_mul(a, c.r5alvcp),
                  mpas_nt_div(1.0f, mpas_nt_mul(dl, dl))),
        mpas_nt_mul(mpas_nt_mul(mpas_nt_sub(1.0f, a), c.r5alscp),
                  mpas_nt_div(1.0f, mpas_nt_mul(di, di))));
}
static inline float nt_foeldcpm(float tt, thread const NtConst &c) {
    const float a = nt_foealfa(tt);
    return mpas_nt_add(mpas_nt_mul(a, c.ralvdcp),
                     mpas_nt_mul(mpas_nt_sub(1.0f, a), c.ralsdcp));
}
static inline void nt_cuadjtqn2(thread float *pt, thread float *pq,
                                             float psp, thread const NtConst &c) {
    const float zqp = mpas_nt_div(1.0f, psp);
    float zqsat = mpas_nt_mul(nt_foeewm(*pt, c), zqp);
    zqsat = min(0.5f, zqsat);
    float zcor = mpas_nt_div(1.0f, mpas_nt_sub(1.0f, mpas_nt_mul(c.vtmpc1, zqsat)));
    zqsat = mpas_nt_mul(zqsat, zcor);
    float zcond = mpas_nt_div(
        mpas_nt_sub(*pq, zqsat),
        mpas_nt_add(1.0f, mpas_nt_mul(mpas_nt_mul(zqsat, zcor),
                                  nt_foedem(*pt, c))));
    zcond = min(zcond, 0.0f);
    *pt = mpas_nt_add(*pt, mpas_nt_mul(nt_foeldcpm(*pt, c), zcond));
    *pq = mpas_nt_sub(*pq, zcond);
    zqsat = mpas_nt_mul(nt_foeewm(*pt, c), zqp);
    zqsat = min(0.5f, zqsat);
    zcor = mpas_nt_div(1.0f, mpas_nt_sub(1.0f, mpas_nt_mul(c.vtmpc1, zqsat)));
    zqsat = mpas_nt_mul(zqsat, zcor);
    float zcond1 = mpas_nt_div(
        mpas_nt_sub(*pq, zqsat),
        mpas_nt_add(1.0f, mpas_nt_mul(mpas_nt_mul(zqsat, zcor),
                                  nt_foedem(*pt, c))));
    if (fabs(zcond) < 1.0e-20f) zcond1 = min(zcond1, 0.0f);
    *pt = mpas_nt_add(*pt, mpas_nt_mul(nt_foeldcpm(*pt, c), zcond1));
    *pq = mpas_nt_sub(*pq, zcond1);
}
#include "hexv842_nt_cudlfsn_bindings.metalh"
kernel void hexv842_nt_cudlfsn(constant Hexv842NtCudlfsnResources &r [[buffer(0)]], constant Hexv842NtCudlfsnParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *inputs = r.inputs;
    device const int *surf_i = r.surf_i;
    device const float *surf_f = r.surf_f;
    device float *out = r.out;
    device int *out_i = r.out_i;
    int i=int(gid), ncol=p.ncol, klev=p.klev; if(i>=ncol) return;
    float cp=p.cp,rd=p.rd,rv=p.rv,xlv=p.xlv,xls=p.xls,xlf=p.xlf,grav=p.grav;
    ulong st=ulong(klev+2)*ulong(ncol);
    device const float *ptenh=inputs,*pqenh=inputs+st,*pten=inputs+2*st,*pqsen=inputs+3*st;
    device const float *pgeo=inputs+4*st,*pgeoh=inputs+5*st,*paph=inputs+6*st,*ptu=inputs+7*st,*pqu=inputs+8*st;
    device const int *ldcum=surf_i,*kcbot=surf_i+ncol,*kctop=surf_i+2*ncol;
    device const float *pmfub=surf_f;
    device float *ptd=out,*pqd=out+st,*pmfd=out+2*st,*pmfds=out+3*st,*pmfdq=out+4*st,*pdmfdp=out+5*st,*prfl=out+6*st;
    device int *kdtop=out_i,*lddraf=out_i+ncol;
    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);


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
        const float zhsk = mpas_nt_add(
            mpas_nt_add(mpas_nt_mul(c.cpd, NTC(pten, jk)), NTC(pgeo, jk)),
            mpas_nt_mul(nt_foelhm(NTC(pten, jk), c), NTC(pqsen, jk)));
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

        const float zttest = mpas_nt_mul(0.5f,
            mpas_nt_add(NTC(ptu, jk), ztenwb));
        const float zqtest = mpas_nt_mul(0.5f,
            mpas_nt_add(NTC(pqu, jk), zqenwb));
        const float zbuo = mpas_nt_sub(
            mpas_nt_mul(zttest, mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1, zqtest))),
            mpas_nt_mul(NTC(ptenh, jk),
                mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1, NTC(pqenh, jk)))));
        const float zcond = mpas_nt_sub(NTC(pqenh, jk), zqenwb);
        const float zmftop = mpas_nt_neg(mpas_nt_mul(NT_CMFDEPS, pmfub[i]));
        if (zbuo < 0.0f
            && rfl > mpas_nt_mul(mpas_nt_mul(10.0f, zmftop), zcond)) {
            kdt = jk;
            ld_out = 1;
            NTC(ptd, jk) = zttest;
            NTC(pqd, jk) = zqtest;
            NTC(pmfd, jk) = zmftop;
            NTC(pmfds, jk) = mpas_nt_mul(NTC(pmfd, jk),
                mpas_nt_add(mpas_nt_mul(c.cpd, NTC(ptd, jk)), NTC(pgeoh, jk)));
            NTC(pmfdq, jk) = mpas_nt_mul(NTC(pmfd, jk), NTC(pqd, jk));
            NTC(pdmfdp, jk - 1) = mpas_nt_mul(-0.5f,
                mpas_nt_mul(NTC(pmfd, jk), zcond));
            rfl = mpas_nt_add(rfl, NTC(pdmfdp, jk - 1));
        }
    }

    prfl[i] = rfl;
    kdtop[i] = kdt;
    lddraf[i] = ld_out;
#undef NTC

}
