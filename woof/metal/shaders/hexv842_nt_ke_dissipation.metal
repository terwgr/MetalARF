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
#include "hexv842_nt_ke_dissipation_bindings.metalh"
kernel void hexv842_nt_ke_dissipation(constant Hexv842NtKeDissipationResources &r [[buffer(0)]], constant Hexv842NtKeDissipationParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *in = r.in;
    device const int *sf = r.sf;
    device float *out = r.out;
    int i=int(gid),ncol=p.ncol,klev=p.klev; if(i>=ncol) return;
    float cp=p.cp,rd=p.rd,rv=p.rv,xlv=p.xlv,xls=p.xls,xlf=p.xlf,grav=p.grav;
    ulong st=ulong(klev+2)*ulong(ncol);
    device const int *ldcum=sf,*kctop=sf+ncol;
    device const float *paph=in,*puen=in+st,*pven=in+2*st,*ztenu=in+3*st,*ztenv=in+4*st,*pvom=in+5*st,*pvol=in+6*st;
    device float *ptte=out,*zuv2=out+st;
    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);


    const int ld = ldcum[i];
    const int ktop = kctop[i];
    float zsum12 = 0.0f;
    float zsum22 = 0.0f;

    /* :1034-1048  the integrals */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(zuv2, jk) = 0.0f;
        if (ld && jk >= ktop - 1) {
            const float zdz =
                mpas_nt_sub(NTC(paph, jk + 1), NTC(paph, jk));
            const float zduten =
                mpas_nt_sub(NTC(pvom, jk), NTC(ztenu, jk));
            const float zdvten =
                mpas_nt_sub(NTC(pvol, jk), NTC(ztenv, jk));
            NTC(zuv2, jk) = mpas_nt_sqrt(mpas_nt_add(
                mpas_nt_mul(zduten, zduten), mpas_nt_mul(zdvten, zdvten)));
            zsum22 = mpas_nt_add(zsum22, mpas_nt_mul(NTC(zuv2, jk), zdz));
            zsum12 = mpas_nt_sub(zsum12, mpas_nt_mul(mpas_nt_add(
                mpas_nt_mul(NTC(puen, jk), zduten),
                mpas_nt_mul(NTC(pven, jk), zdvten)), zdz));
        }
    }

    /* :1049-1056  the heating.  ADD, do not assign. */
    for (int jk = 1; jk <= klev; ++jk) {
        if (ld && jk >= ktop - 1) {
            const float ztdis = mpas_nt_div(
                mpas_nt_mul(mpas_nt_mul(c.rcpd, zsum12), NTC(zuv2, jk)),
                max(1.0e-15f, zsum22));
            NTC(ptte, jk) = mpas_nt_add(ztdis, NTC(ptte, jk));
        }
    }
#undef NTC

}
