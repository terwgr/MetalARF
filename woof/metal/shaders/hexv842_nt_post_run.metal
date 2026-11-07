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

#include "hexv842_nt_post_run_bindings.metalh"
kernel void hexv842_nt_post_run(constant Hexv842NtPostRunResources &r [[buffer(0)]], constant Hexv842NtPostRunParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *in = r.in;
    device const float *rn = r.rn;
    device float *out = r.out;
    int i=int(gid),ncol=p.ncol,klev=p.klev,stepcu=p.stepcu; if(i>=ncol) return; float dt=p.dt;
    ulong st=ulong(klev+2)*ulong(ncol);
    device const float *exner=in,*qv=in+st,*qc=in+2*st,*qi=in+3*st,*t=in+4*st,*u=in+5*st,*v=in+6*st;
    device const float *tf=in+7*st,*qvf=in+8*st,*qcf=in+9*st,*qif=in+10*st,*uf=in+11*st,*vf=in+12*st;
    device float *rthcuten=out,*rqvcuten=out+st,*rqccuten=out+2*st,*rqicuten=out+3*st,*rucuten=out+4*st,*rvcuten=out+5*st,*raincv=out+6*st,*pratec=raincv+ncol;
    const float fstepcu = (float)stepcu;
    const float delt = mpas_nt_mul(dt, fstepcu);
    const float rdelt = mpas_nt_div(1.0f, delt);

    /* :506-509.  pratec divides by ONE product, stepcu*dt -- not by
     * stepcu and then by dt, which rounds twice. */
    raincv[i] = mpas_nt_div(rn[i], fstepcu);
    pratec[i] = mpas_nt_div(rn[i], mpas_nt_mul(fstepcu, dt));

    /* :511-524 */
    for (int jk = 1; jk <= klev; ++jk) {
        const int zz = klev + 1 - jk;
        NTC(rthcuten, jk) = mpas_nt_mul(
            mpas_nt_div(mpas_nt_sub(NTC(tf, zz), NTC(t, jk)), NTC(exner, jk)),
            rdelt);
        NTC(rqvcuten, jk) =
            mpas_nt_mul(mpas_nt_sub(NTC(qvf, zz), NTC(qv, jk)), rdelt);
        NTC(rqccuten, jk) =
            mpas_nt_mul(mpas_nt_sub(NTC(qcf, zz), NTC(qc, jk)), rdelt);
        NTC(rqicuten, jk) =
            mpas_nt_mul(mpas_nt_sub(NTC(qif, zz), NTC(qi, jk)), rdelt);
        NTC(rucuten, jk) =
            mpas_nt_mul(mpas_nt_sub(NTC(uf, zz), NTC(u, jk)), rdelt);
        NTC(rvcuten, jk) =
            mpas_nt_mul(mpas_nt_sub(NTC(vf, zz), NTC(v, jk)), rdelt);
    }
#undef NTC

}
