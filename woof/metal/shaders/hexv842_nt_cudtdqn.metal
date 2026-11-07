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
static inline float nt_foelhm(float tt, thread const NtConst &c) {
    const float a = nt_foealfa(tt);
    return mpas_nt_add(mpas_nt_mul(a, c.alv),
                     mpas_nt_mul(mpas_nt_sub(1.0f, a), c.als));
}
#include "hexv842_nt_cudtdqn_bindings.metalh"
kernel void hexv842_nt_cudtdqn(constant Hexv842NtCudtdqnResources &r [[buffer(0)]], constant Hexv842NtCudtdqnParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *in = r.in;
    device const int *ldcum = r.ldcum;
    device float *out = r.out;
    int i=int(gid),ncol=p.ncol,klev=p.klev,ktopm2=p.ktopm2; if(i>=ncol) return;
    float cp=p.cp,rd=p.rd,rv=p.rv,xlv=p.xlv,xls=p.xls,xlf=p.xlf,grav=p.grav;
    ulong st=ulong(klev+2)*ulong(ncol);
    device const float *paph=in,*pten=in+st,*plglac=in+2*st,*plude=in+3*st,*pmfus=in+4*st,*pmfds=in+5*st;
    device const float *pmfuq=in+6*st,*pmfdq=in+7*st,*pmful=in+8*st,*pdmfup=in+9*st,*pdmfdp=in+10*st,*pdpmel=in+11*st;
    device float *ptent=out,*ptenq=out+st,*pcte=out+2*st;
    if (!ldcum[i]) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);


    for (int jk = ktopm2; jk <= klev; ++jk) {
        const float zdp = mpas_nt_div(
            c.g, mpas_nt_sub(NTC(paph, jk + 1), NTC(paph, jk)));
        const float zalv = nt_foelhm(NTC(pten, jk), c);
        float zdtdt, zdqdt;
        if (jk < klev) {
            const float inner = mpas_nt_sub(mpas_nt_sub(mpas_nt_sub(
                mpas_nt_sub(NTC(pmful, jk + 1), NTC(pmful, jk)),
                NTC(plude, jk)), NTC(pdmfup, jk)), NTC(pdmfdp, jk));
            const float big = mpas_nt_sub(mpas_nt_sub(mpas_nt_add(mpas_nt_sub(
                mpas_nt_add(mpas_nt_sub(NTC(pmfus, jk + 1), NTC(pmfus, jk)),
                          NTC(pmfds, jk + 1)), NTC(pmfds, jk)),
                mpas_nt_mul(c.alf, NTC(plglac, jk))),
                mpas_nt_mul(c.alf, NTC(pdpmel, jk))),
                mpas_nt_mul(zalv, inner));
            zdtdt = mpas_nt_mul(mpas_nt_mul(zdp, c.rcpd), big);
            /* :3117-3119 chains all NINE terms left to right -- there
             * are no parentheses inside, so the last two do not group. */
            zdqdt = mpas_nt_mul(zdp, mpas_nt_sub(mpas_nt_sub(mpas_nt_sub(mpas_nt_sub(
                mpas_nt_add(mpas_nt_sub(mpas_nt_add(mpas_nt_sub(NTC(pmfuq, jk + 1),
                    NTC(pmfuq, jk)), NTC(pmfdq, jk + 1)),
                    NTC(pmfdq, jk)), NTC(pmful, jk + 1)),
                    NTC(pmful, jk)), NTC(plude, jk)),
                    NTC(pdmfup, jk)), NTC(pdmfdp, jk)));
        } else {
            const float big = mpas_nt_sub(mpas_nt_add(
                mpas_nt_add(NTC(pmfus, jk), NTC(pmfds, jk)),
                mpas_nt_mul(c.alf, NTC(pdpmel, jk))),
                mpas_nt_mul(zalv, mpas_nt_add(mpas_nt_add(
                    mpas_nt_add(NTC(pmful, jk), NTC(pdmfup, jk)),
                    NTC(pdmfdp, jk)), NTC(plude, jk))));
            zdtdt = mpas_nt_neg(mpas_nt_mul(mpas_nt_mul(zdp, c.rcpd), big));
            zdqdt = mpas_nt_neg(mpas_nt_mul(zdp, mpas_nt_add(
                mpas_nt_add(mpas_nt_add(NTC(pmfuq, jk), NTC(plude, jk)),
                          NTC(pmfdq, jk)),
                mpas_nt_add(mpas_nt_add(NTC(pmful, jk), NTC(pdmfup, jk)),
                          NTC(pdmfdp, jk)))));
        }
        /* ADD.  See the header. */
        NTC(ptent, jk) = mpas_nt_add(NTC(ptent, jk), zdtdt);
        NTC(ptenq, jk) = mpas_nt_add(NTC(ptenq, jk), zdqdt);
        NTC(pcte, jk) = mpas_nt_mul(zdp, NTC(plude, jk));
    }
#undef NTC

}
