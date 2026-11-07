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
#include "hexv842_nt_cududvn_bindings.metalh"
kernel void hexv842_nt_cududvn(constant Hexv842NtCududvnResources &r [[buffer(0)]], constant Hexv842NtCududvnParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *in = r.in;
    device const int *sf = r.sf;
    device float *out = r.out;
    int i=int(gid),ncol=p.ncol,klev=p.klev,ktopm2=p.ktopm2; if(i>=ncol) return;
    float cp=p.cp,rd=p.rd,rv=p.rv,xlv=p.xlv,xls=p.xls,xlf=p.xlf,grav=p.grav;
    ulong st=ulong(klev+2)*ulong(ncol);
    device const int *ldcum=sf,*ktype=sf+ncol,*kcbot=sf+2*ncol;
    device const float *paph=in,*puen=in+st,*pven=in+2*st,*pmfu=in+3*st,*pmfd=in+4*st;
    device const float *puu=in+5*st,*pud=in+6*st,*pvu=in+7*st,*pvd=in+8*st;
    device float *ptenu=out,*ptenv=out+st,*zmfuu=out+2*st,*zmfuv=out+3*st,*zmfdu=out+4*st,*zmfdv=out+5*st;
    if (!ldcum[i]) return;

    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);

    const int kb = kcbot[i];
    const int kt3 = (ktype[i] == 3);

    /* 1.0 the fluxes (:3191-3201) */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        const int ik = jk - 1;
        NTC(zmfuu, jk) = mpas_nt_mul(NTC(pmfu, jk),
            mpas_nt_sub(NTC(puu, jk), NTC(puen, ik)));
        NTC(zmfuv, jk) = mpas_nt_mul(NTC(pmfu, jk),
            mpas_nt_sub(NTC(pvu, jk), NTC(pven, ik)));
        NTC(zmfdu, jk) = mpas_nt_mul(NTC(pmfd, jk),
            mpas_nt_sub(NTC(pud, jk), NTC(puen, ik)));
        NTC(zmfdv, jk) = mpas_nt_mul(NTC(pmfd, jk),
            mpas_nt_sub(NTC(pvd, jk), NTC(pven, ik)));
    }

    /* linear fluxes below cloud (:3203-3215) */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        if (jk <= kb) continue;
        float zzp = mpas_nt_div(mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, jk)),
                              mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, kb)));
        if (kt3) zzp = mpas_nt_mul(zzp, zzp);
        NTC(zmfuu, jk) = mpas_nt_mul(NTC(zmfuu, kb), zzp);
        NTC(zmfuv, jk) = mpas_nt_mul(NTC(zmfuv, kb), zzp);
        NTC(zmfdu, jk) = mpas_nt_mul(NTC(zmfdu, kb), zzp);
        NTC(zmfdv, jk) = mpas_nt_mul(NTC(zmfdv, kb), zzp);
    }

    /* 2.0 and 3.0 fused (:3219-3249).  ADD, do not assign. */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        const float zdp = mpas_nt_div(
            c.g, mpas_nt_sub(NTC(paph, jk + 1), NTC(paph, jk)));
        float zdudt, zdvdt;
        if (jk < klev) {
            const int ik = jk + 1;
            zdudt = mpas_nt_mul(zdp, mpas_nt_sub(mpas_nt_add(
                mpas_nt_sub(NTC(zmfuu, ik), NTC(zmfuu, jk)), NTC(zmfdu, ik)),
                NTC(zmfdu, jk)));
            zdvdt = mpas_nt_mul(zdp, mpas_nt_sub(mpas_nt_add(
                mpas_nt_sub(NTC(zmfuv, ik), NTC(zmfuv, jk)), NTC(zmfdv, ik)),
                NTC(zmfdv, jk)));
        } else {
            zdudt = mpas_nt_neg(mpas_nt_mul(zdp,
                mpas_nt_add(NTC(zmfuu, jk), NTC(zmfdu, jk))));
            zdvdt = mpas_nt_neg(mpas_nt_mul(zdp,
                mpas_nt_add(NTC(zmfuv, jk), NTC(zmfdv, jk))));
        }
        NTC(ptenu, jk) = mpas_nt_add(NTC(ptenu, jk), zdudt);
        NTC(ptenv, jk) = mpas_nt_add(NTC(ptenv, jk), zdvdt);
    }
#undef NTC

}
