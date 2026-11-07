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

#define NT_ENTRDD 2.0e-4f
#define NT_CMFCMIN 1.0e-10f
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
static inline void nt_cuadjtqn2(device float *pt,device float *pq,float psp,thread const NtConst &c) {
    float t=*pt,q=*pq; nt_cuadjtqn2(&t,&q,psp,c); *pt=t; *pq=q;
}

#include "hexv842_nt_cuddrafn_bindings.metalh"
kernel void hexv842_nt_cuddrafn(constant Hexv842NtCuddrafnResources &r [[buffer(0)]], constant Hexv842NtCuddrafnParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *inputs = r.inputs;
    device const int *surf_i = r.surf_i;
    device float *out = r.out;
    int i=int(gid), ncol=p.ncol,klev=p.klev; if(i>=ncol) return;
    float cp=p.cp,rd=p.rd,rv=p.rv,xlv=p.xlv,xls=p.xls,xlf=p.xlf,grav=p.grav;
    ulong st=ulong(klev+2)*ulong(ncol);
    device const float *ptenh=inputs,*pqenh=inputs+st,*pgeo=inputs+2*st,*pgeoh=inputs+3*st,*paph=inputs+4*st,*pmfu=inputs+5*st;
    device const int *lddraf=surf_i;
    device float *ptd=out,*pqd=out+st,*pmfd=out+2*st,*pmfds=out+3*st,*pmfdq=out+4*st,*pdmfdp=out+5*st,*pmfdde_rate=out+6*st,*prfl=out+7*st;
    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);


    float zoentr = 0.0f, zbuoy = 0.0f;
    float zdmfen = 0.0f, zdmfde = 0.0f;
    float rfl = prfl[i];
    const int ld = lddraf[i];

    /* itopde (:2616-2621).  The loop DESCENDS and the last assignment wins,
     * so itopde ends at the TOPMOST level within 60 hPa of the surface. */
    int itopde = 0;
    for (int jk = klev; jk >= 1; --jk) {
        NTC(pmfdde_rate, jk) = 0.0f;
        if (mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, jk)) < 60.0e2f)
            itopde = jk;
    }


    for (int jk = 3; jk <= klev; ++jk) {
        const float zph = NTC(paph, jk);
        if (!(ld && NTC(pmfd, jk - 1) < 0.0f)) continue;
        // The native source requires a qualifying half-level when descent is active.
        if (!itopde) {
            atomic_store_explicit((device atomic_uint *)r.status, 1u, memory_order_relaxed);
            return;
        }

        const float zentr = mpas_nt_mul(mpas_nt_mul(
            mpas_nt_mul(NT_ENTRDD, NTC(pmfd, jk - 1)),
            mpas_nt_sub(NTC(pgeoh, jk - 1), NTC(pgeoh, jk))), c.zrg);
        zdmfen = zentr;
        zdmfde = zentr;

        if (jk > itopde) {
            zdmfen = 0.0f;
            zdmfde = mpas_nt_div(
                mpas_nt_mul(NTC(pmfd, itopde),
                          mpas_nt_sub(NTC(paph, jk), NTC(paph, jk - 1))),
                mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, itopde)));
        }
        if (jk <= itopde) {
            const float zdz = mpas_nt_neg(mpas_nt_mul(
                mpas_nt_sub(NTC(pgeoh, jk - 1), NTC(pgeoh, jk)), c.zrg));
            const float zzentr = mpas_nt_mul(mpas_nt_mul(zoentr, zdz),
                                           NTC(pmfd, jk - 1));
            zdmfen = mpas_nt_add(zdmfen, zzentr);
            zdmfen = max(zdmfen, mpas_nt_mul(0.3f, NTC(pmfd, jk - 1)));
            zdmfen = max(zdmfen,
                mpas_nt_sub(mpas_nt_neg(mpas_nt_mul(0.75f, NTC(pmfu, jk))),
                          mpas_nt_sub(NTC(pmfd, jk - 1), zdmfde)));
            zdmfen = min(zdmfen, 0.0f);
        }

        NTC(pmfd, jk) = mpas_nt_sub(
            mpas_nt_add(NTC(pmfd, jk - 1), zdmfen), zdmfde);
        const float zseen = mpas_nt_mul(
            mpas_nt_add(mpas_nt_mul(c.cpd, NTC(ptenh, jk - 1)),
                      NTC(pgeoh, jk - 1)), zdmfen);
        const float zqeen = mpas_nt_mul(NTC(pqenh, jk - 1), zdmfen);
        const float zsdde = mpas_nt_mul(
            mpas_nt_add(mpas_nt_mul(c.cpd, NTC(ptd, jk - 1)),
                      NTC(pgeoh, jk - 1)), zdmfde);
        const float zqdde = mpas_nt_mul(NTC(pqd, jk - 1), zdmfde);
        const float zmfdsk = mpas_nt_sub(
            mpas_nt_add(NTC(pmfds, jk - 1), zseen), zsdde);
        const float zmfdqk = mpas_nt_sub(
            mpas_nt_add(NTC(pmfdq, jk - 1), zqeen), zqdde);
        const float inv = mpas_nt_div(1.0f,
                                    min(-NT_CMFCMIN, NTC(pmfd, jk)));
        NTC(pqd, jk) = mpas_nt_mul(zmfdqk, inv);
        NTC(ptd, jk) = mpas_nt_mul(
            mpas_nt_sub(mpas_nt_mul(zmfdsk, inv), NTC(pgeoh, jk)), c.rcpd);
        NTC(ptd, jk) = min(400.0f, NTC(ptd, jk));
        NTC(ptd, jk) = max(100.0f, NTC(ptd, jk));
        float zcond = NTC(pqd, jk);

        nt_cuadjtqn2(&NTC(ptd, jk), &NTC(pqd, jk), zph, c);

        zcond = mpas_nt_sub(zcond, NTC(pqd, jk));
        float zbuo = mpas_nt_sub(
            mpas_nt_mul(NTC(ptd, jk),
                mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1, NTC(pqd, jk)))),
            mpas_nt_mul(NTC(ptenh, jk),
                mpas_nt_add(1.0f, mpas_nt_mul(c.vtmpc1, NTC(pqenh, jk)))));
        if (rfl > 0.0f && NTC(pmfu, jk) > 0.0f) {
            const float zrain = mpas_nt_div(rfl, NTC(pmfu, jk));
            zbuo = mpas_nt_sub(zbuo, mpas_nt_mul(NTC(ptd, jk), zrain));
        }
        if (zbuo >= 0.0f || rfl <= mpas_nt_mul(NTC(pmfd, jk), zcond)) {
            NTC(pmfd, jk) = 0.0f;
            zbuo = 0.0f;
        }
        NTC(pmfds, jk) = mpas_nt_mul(
            mpas_nt_add(mpas_nt_mul(c.cpd, NTC(ptd, jk)), NTC(pgeoh, jk)),
            NTC(pmfd, jk));
        NTC(pmfdq, jk) = mpas_nt_mul(NTC(pqd, jk), NTC(pmfd, jk));
        const float zdmfdp = mpas_nt_neg(mpas_nt_mul(NTC(pmfd, jk), zcond));
        NTC(pdmfdp, jk - 1) = zdmfdp;
        rfl = mpas_nt_add(rfl, zdmfdp);

        /* organised entrainment for the next level down */
        float zbuoyz = mpas_nt_div(zbuo, NTC(ptenh, jk));
        zbuoyz = min(zbuoyz, 0.0f);
        const float zdz2 = mpas_nt_neg(mpas_nt_sub(NTC(pgeo, jk - 1), NTC(pgeo, jk)));
        zbuoy = mpas_nt_add(zbuoy, mpas_nt_mul(zbuoyz, zdz2));
        zoentr = mpas_nt_div(mpas_nt_mul(mpas_nt_mul(c.g, zbuoyz), 0.5f),
                           mpas_nt_add(1.0f, zbuoy));
        NTC(pmfdde_rate, jk) = -zdmfde;
    }

    prfl[i] = rfl;
#undef NTC

}
