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
#include "hexv842_nt_adjust_bindings.metalh"
kernel void hexv842_nt_adjust(constant Hexv842NtAdjustResources &r [[buffer(0)]], constant Hexv842NtAdjustParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *in = r.in;
    device const int *sf = r.sf;
    device float *out = r.out;
    int i=int(gid),ncol=p.ncol,klev=p.klev; if(i>=ncol) return;
    float ztmst=p.ztmst,cp=p.cp,rd=p.rd,rv=p.rv,xlv=p.xlv,xls=p.xls,xlf=p.xlf,grav=p.grav;
    ulong st=ulong(klev+2)*ulong(ncol);
    device const int *ldcum=sf,*loddraf=sf+ncol,*idtop=sf+2*ncol,*kctop=sf+3*ncol,*kcbot=sf+4*ncol;
    device const float *paph=in,*pqen=in+st,*pmfu=in+2*st,*pmfuq=in+3*st,*pmful=in+4*st;
    device float *pmfd=out,*pmfds=out+st,*pmfdq=out+2*st,*plude=out+3*st,*pdmfup=out+4*st,*pdmfdp=out+5*st;
    device float *pmfdde_rate=out+6*st,*pmfude_rate=out+7*st,*pmflxr=out+8*st,*pmflxs=out+9*st,*prsfc=out+10*st,*pssfc=prsfc+ncol;
    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);


    const int ld = ldcum[i];
    const int ldd = loddraf[i];
    const int idt = idtop[i];
    const int ktop = kctop[i];
    const int kb = kcbot[i];

    /* :834-847 */
    float zmfs = 1.0f;
    for (int jk = 2; jk <= klev; ++jk) {
        if (ldd && jk >= idt - 1) {
            const float zmfmax = mpas_nt_mul(NTC(pmfu, jk), 0.98f);
            if (mpas_nt_add(mpas_nt_add(NTC(pmfd, jk), zmfmax), 1.0e-15f)
                    < 0.0f) {
                zmfs = min(zmfs, mpas_nt_div(mpas_nt_neg(zmfmax), NTC(pmfd, jk)));
            }
        }
    }

    /* :849-861.  zmfuub runs DOWNWARD through the column. */
    float zmfuub = 0.0f;
    for (int jk = 2; jk <= klev; ++jk) {
        if (zmfs < 1.0f && jk >= idt - 1) {
            NTC(pmfd, jk) = mpas_nt_mul(NTC(pmfd, jk), zmfs);
            NTC(pmfds, jk) = mpas_nt_mul(NTC(pmfds, jk), zmfs);
            NTC(pmfdq, jk) = mpas_nt_mul(NTC(pmfdq, jk), zmfs);
            NTC(pmfdde_rate, jk) = mpas_nt_mul(NTC(pmfdde_rate, jk), zmfs);
            zmfuub = mpas_nt_sub(zmfuub,
                mpas_nt_mul(mpas_nt_sub(1.0f, zmfs), NTC(pdmfdp, jk)));
            NTC(pmflxr, jk + 1) = mpas_nt_add(NTC(pmflxr, jk + 1), zmfuub);
            NTC(pdmfdp, jk) = mpas_nt_mul(NTC(pdmfdp, jk), zmfs);
        }
    }

    /* :863-880 */
    for (int jk = 2; jk <= klev - 1; ++jk) {
        if (ldd && jk >= idt - 1) {
            const float zerate = mpas_nt_add(
                mpas_nt_add(mpas_nt_neg(NTC(pmfd, jk)), NTC(pmfd, jk - 1)),
                NTC(pmfdde_rate, jk));
            if (zerate < 0.0f)
                NTC(pmfdde_rate, jk) =
                    mpas_nt_sub(NTC(pmfdde_rate, jk), zerate);
        }
        if (ld && jk >= ktop - 1) {
            const float zerate = mpas_nt_add(
                mpas_nt_sub(NTC(pmfu, jk), NTC(pmfu, jk + 1)),
                NTC(pmfude_rate, jk));
            if (zerate < 0.0f)
                NTC(pmfude_rate, jk) =
                    mpas_nt_sub(NTC(pmfude_rate, jk), zerate);
            NTC(pdmfup, jk) = mpas_nt_sub(mpas_nt_sub(
                mpas_nt_add(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1)),
                NTC(pmflxr, jk)), NTC(pmflxs, jk));
            NTC(pdmfdp, jk) = 0.0f;
        }
    }

    /* :883-892  the downdraft-top humidity guard */
    if (ldd) {
        const int jk = idt;
        const int ik = min(jk + 1, klev);
        if (NTC(pmfdq, jk) < mpas_nt_mul(0.3f, NTC(pmfdq, ik)))
            NTC(pmfdq, jk) = mpas_nt_mul(0.3f, NTC(pmfdq, ik));
    }

    /* :896-913  the near-cloud-top humidity guard */
    for (int jk = 2; jk <= klev; ++jk) {
        if (ld && jk >= ktop - 1 && jk < kb) {
            const float zdz = mpas_nt_div(mpas_nt_mul(ztmst, c.g),
                mpas_nt_sub(NTC(paph, jk + 1), NTC(paph, jk)));
            float zmfa = mpas_nt_add(mpas_nt_sub(mpas_nt_add(mpas_nt_sub(mpas_nt_sub(
                mpas_nt_add(NTC(pmfuq, jk + 1), NTC(pmfdq, jk + 1)),
                NTC(pmfuq, jk)), NTC(pmfdq, jk)),
                NTC(pmful, jk + 1)), NTC(pmful, jk)), NTC(pdmfup, jk));
            zmfa = mpas_nt_mul(mpas_nt_sub(zmfa, NTC(plude, jk)), zdz);
            if (mpas_nt_add(NTC(pqen, jk), zmfa) < 0.0f) {
                NTC(plude, jk) = mpas_nt_add(NTC(plude, jk), mpas_nt_div(
                    mpas_nt_mul(2.0f, mpas_nt_add(NTC(pqen, jk), zmfa)), zdz));
            }
            if (NTC(plude, jk) < 0.0f) NTC(plude, jk) = 0.0f;
        }
        if (!ld) NTC(pmfude_rate, jk) = 0.0f;
        if (fabs(NTC(pmfd, jk - 1)) < 1.0e-20f)
            NTC(pmfdde_rate, jk) = 0.0f;
    }

    prsfc[i] = NTC(pmflxr, klev + 1);
    pssfc[i] = NTC(pmflxs, klev + 1);
#undef NTC

}
