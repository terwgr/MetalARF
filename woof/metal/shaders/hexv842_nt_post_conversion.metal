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
static inline float nt_foealfa(float tt) {
    const float clamped = max(NT_RTICE, min(NT_RTWAT, tt));
    const float r = mpas_nt_div(mpas_nt_sub(clamped, NT_RTICE),
                              mpas_nt_sub(NT_RTWAT, NT_RTICE));
    return min(1.0f, mpas_nt_mul(r, r));
}
#include "hexv842_nt_post_conversion_bindings.metalh"
kernel void hexv842_nt_post_conversion(constant Hexv842NtPostConversionResources &r [[buffer(0)]], constant Hexv842NtPostConversionParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *in = r.in;
    device const float *sf = r.sf;
    device float *out = r.out;
    int i=int(gid),ncol=p.ncol,klev=p.klev; if(i>=ncol) return; float delt=p.delt;
    ulong st=ulong(klev+2)*ulong(ncol);
    device const float *pcte=in,*ztp1=in+st,*ptte=in+2*st,*ztt=in+3*st,*pqte=in+4*st,*zqq=in+5*st;
    device const float *qcf=in+6*st,*qif=in+7*st,*uf=in+8*st,*vf=in+9*st,*pvom=in+10*st,*pvol=in+11*st;
    device const float *prsfc=sf,*pssfc=sf+ncol;
    device float *zqp1=out,*pqc=out+st,*pqi=out+2*st,*pt=out+3*st,*pqv=out+4*st,*pu=out+5*st,*pv=out+6*st,*zprecc=out+7*st;
    for (int jk = 1; jk <= klev; ++jk) {
        if (NTC(pcte, jk) > 0.0f) {
            const float fliq = nt_foealfa(NTC(ztp1, jk));
            const float fice = mpas_nt_sub(1.0f, fliq);
            NTC(pqc, jk) = mpas_nt_add(
                NTC(qcf, jk),
                mpas_nt_mul(mpas_nt_mul(fliq, NTC(pcte, jk)), delt));
            NTC(pqi, jk) = mpas_nt_add(
                NTC(qif, jk),
                mpas_nt_mul(mpas_nt_mul(fice, NTC(pcte, jk)), delt));
        } else {
            NTC(pqc, jk) = NTC(qcf, jk);
            NTC(pqi, jk) = NTC(qif, jk);
        }
    }

    /* :308-314 */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(pt, jk) = mpas_nt_add(
            NTC(ztp1, jk),
            mpas_nt_mul(mpas_nt_sub(NTC(ptte, jk), NTC(ztt, jk)), delt));
        const float q = mpas_nt_add(
            NTC(zqp1, jk),
            mpas_nt_mul(mpas_nt_sub(NTC(pqte, jk), NTC(zqq, jk)), delt));
        NTC(zqp1, jk) = q;
        NTC(pqv, jk) = mpas_nt_div(q, mpas_nt_sub(1.0f, q));
    }

    /* :316-318.  amax1 clamps a negative flux product to zero. */
    zprecc[i] = max(0.0f, mpas_nt_mul(mpas_nt_add(prsfc[i], pssfc[i]), delt));

    /* :319-325, lmfdudv is .true. at :55 */
    for (int jk = 1; jk <= klev; ++jk) {
        NTC(pu, jk) = mpas_nt_add(NTC(uf, jk),
                                mpas_nt_mul(delt, NTC(pvom, jk)));
        NTC(pv, jk) = mpas_nt_add(NTC(vf, jk),
                                mpas_nt_mul(delt, NTC(pvol, jk)));
    }
#undef NTC

}
