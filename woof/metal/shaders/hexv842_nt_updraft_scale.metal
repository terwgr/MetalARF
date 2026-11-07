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
#include "hexv842_nt_updraft_scale_bindings.metalh"
kernel void hexv842_nt_updraft_scale(constant Hexv842NtUpdraftScaleResources &r [[buffer(0)]], constant Hexv842NtUpdraftScaleParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *paph = r.paph;
    device const int *sf_i = r.sf_i;
    device const float *sf = r.sf;
    device int *out_i = r.out_i;
    device float *out = r.out;
    int i=int(gid),ncol=p.ncol,klev=p.klev; if(i>=ncol) return;
    float ztmst=p.ztmst,cp=p.cp,rd=p.rd,rv=p.rv,xlv=p.xlv,xls=p.xls,xlf=p.xlf,grav=p.grav;
    ulong st=ulong(klev+2)*ulong(ncol);
    device const int *loddraf=sf_i,*kcbot=sf_i+ncol,*kctop=sf_i+2*ncol;
    device const float *zmfub1=sf,*zmfub=sf+ncol;
    device int *ldcum=out_i,*ktype=out_i+ncol,*idtop=out_i+2*ncol;
    device float *pmfu=out,*pmfus=out+st,*pmfuq=out+2*st,*pmful=out+3*st,*pdmfup=out+4*st,*plude=out+5*st;
    device float *pmfude_rate=out+6*st,*pmfd=out+7*st,*pmfds=out+8*st,*pmfdq=out+9*st,*pdmfdp=out+10*st,*pmfdde_rate=out+11*st;
    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);


    const float zcons2 = mpas_nt_div(3.0f, mpas_nt_mul(c.g, ztmst));
    const int ld0 = ldcum[i];
    const int kb = kcbot[i];
    const int ktop = kctop[i];
    const int ldd = loddraf[i];

    /* :743-746 */
    float zmfs = 1.0f;
    if (ld0) zmfs = mpas_nt_div(zmfub1[i], max(NT_CMFCMIN, zmfub[i]));

    /* :747-761  taper below cloud base, then cap */
    for (int jk = 2; jk <= klev; ++jk) {
        if (ld0 && jk >= ktop - 1) {
            if (jk > kb) {
                const float zdz = mpas_nt_div(
                    mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, jk)),
                    mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, kb)));
                NTC(pmfu, jk) = mpas_nt_mul(NTC(pmfu, kb), zdz);
            }
            const float zmfmax = mpas_nt_mul(
                mpas_nt_sub(NTC(paph, jk), NTC(paph, jk - 1)), zcons2);
            if (mpas_nt_mul(NTC(pmfu, jk), zmfs) > zmfmax)
                zmfs = min(zmfs, mpas_nt_div(zmfmax, NTC(pmfu, jk)));
        }
    }

    /* :762-774  apply */
    for (int jk = 2; jk <= klev; ++jk) {
        if (ld0 && jk <= kb && jk >= ktop - 1) {
            NTC(pmfu, jk) = mpas_nt_mul(NTC(pmfu, jk), zmfs);
            NTC(pmfus, jk) = mpas_nt_mul(NTC(pmfus, jk), zmfs);
            NTC(pmfuq, jk) = mpas_nt_mul(NTC(pmfuq, jk), zmfs);
            NTC(pmful, jk) = mpas_nt_mul(NTC(pmful, jk), zmfs);
            NTC(pdmfup, jk) = mpas_nt_mul(NTC(pdmfup, jk), zmfs);
            NTC(plude, jk) = mpas_nt_mul(NTC(plude, jk), zmfs);
            NTC(pmfude_rate, jk) =
                mpas_nt_mul(NTC(pmfude_rate, jk), zmfs);
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
