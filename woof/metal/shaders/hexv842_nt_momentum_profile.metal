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

#define NT_CMFCMIN 1.0e-10f
#define NT_PGCOEF 0.7f

#include "hexv842_nt_momentum_profile_bindings.metalh"
kernel void hexv842_nt_momentum_profile(constant Hexv842NtMomentumProfileResources &r [[buffer(0)]], constant Hexv842NtMomentumProfileParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *in = r.in;
    device const int *sf = r.sf;
    device float *out = r.out;
    int i=int(gid),ncol=p.ncol,klev=p.klev; if(i>=ncol) return;
    ulong st=ulong(klev+2)*ulong(ncol);
    device const int *ldcum=sf,*ktype=sf+ncol,*kcbot=sf+2*ncol,*kctop=sf+3*ncol,*kdpl=sf+4*ncol,*idtop=sf+5*ncol;
    device const float *puen=in,*pven=in+st,*pmfu=in+2*st,*pmfd=in+3*st,*pmfude_rate=in+4*st,*pmfdde_rate=in+5*st;
    device float *puu=out,*pvu=out+st,*pud=out+2*st,*pvd=out+3*st;
    if (!ldcum[i]) return;


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
            const float pgf_u = mpas_nt_neg(mpas_nt_mul(
                mpas_nt_mul(NT_PGCOEF, 0.5f),
                mpas_nt_add(
                    mpas_nt_mul(NTC(pmfu, ik),
                        mpas_nt_sub(NTC(puen, ik), NTC(puen, jk))),
                    mpas_nt_mul(NTC(pmfu, jk),
                        mpas_nt_sub(NTC(puen, jk), NTC(puen, jk - 1))))));
            const float pgf_v = mpas_nt_neg(mpas_nt_mul(
                mpas_nt_mul(NT_PGCOEF, 0.5f),
                mpas_nt_add(
                    mpas_nt_mul(NTC(pmfu, ik),
                        mpas_nt_sub(NTC(pven, ik), NTC(pven, jk))),
                    mpas_nt_mul(NTC(pmfu, jk),
                        mpas_nt_sub(NTC(pven, jk), NTC(pven, jk - 1))))));
            const float zerate = mpas_nt_add(
                mpas_nt_sub(NTC(pmfu, jk), NTC(pmfu, ik)),
                NTC(pmfude_rate, jk));
            const float zderate = NTC(pmfude_rate, jk);
            const float zmfa = mpas_nt_div(
                1.0f, max(NT_CMFCMIN, NTC(pmfu, jk)));
            NTC(puu, jk) = mpas_nt_mul(mpas_nt_add(mpas_nt_sub(mpas_nt_add(
                mpas_nt_mul(NTC(puu, ik), NTC(pmfu, ik)),
                mpas_nt_mul(zerate, NTC(puen, jk))),
                mpas_nt_mul(zderate, NTC(puu, ik))), pgf_u), zmfa);
            NTC(pvu, jk) = mpas_nt_mul(mpas_nt_add(mpas_nt_sub(mpas_nt_add(
                mpas_nt_mul(NTC(pvu, ik), NTC(pmfu, ik)),
                mpas_nt_mul(zerate, NTC(pven, jk))),
                mpas_nt_mul(zderate, NTC(pvu, ik))), pgf_v), zmfa);
        }
    }

    /* the downdraft profile (:972-991), UPWARD in index */
    for (int jk = 3; jk <= klev; ++jk) {
        const int ik = jk - 1;
        if (jk == idt) {
            NTC(pud, jk) = mpas_nt_mul(0.5f,
                mpas_nt_add(NTC(puu, jk), NTC(puen, ik)));
            NTC(pvd, jk) = mpas_nt_mul(0.5f,
                mpas_nt_add(NTC(pvu, jk), NTC(pven, ik)));
        } else if (jk > idt) {
            const float zerate = mpas_nt_add(
                mpas_nt_add(mpas_nt_neg(NTC(pmfd, jk)), NTC(pmfd, ik)),
                NTC(pmfdde_rate, jk));
            const float zmfa = mpas_nt_div(
                1.0f, min(-NT_CMFCMIN, NTC(pmfd, jk)));
            NTC(pud, jk) = mpas_nt_mul(mpas_nt_add(mpas_nt_sub(
                mpas_nt_mul(NTC(pud, ik), NTC(pmfd, ik)),
                mpas_nt_mul(zerate, NTC(puen, ik))),
                mpas_nt_mul(NTC(pmfdde_rate, jk), NTC(pud, ik))), zmfa);
            NTC(pvd, jk) = mpas_nt_mul(mpas_nt_add(mpas_nt_sub(
                mpas_nt_mul(NTC(pvd, ik), NTC(pmfd, ik)),
                mpas_nt_mul(zerate, NTC(pven, ik))),
                mpas_nt_mul(NTC(pmfdde_rate, jk), NTC(pvd, ik))), zmfa);
        }
    }
#undef NTC

}
