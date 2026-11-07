#include "hexv842_nt_mfub_bindings.metalh"
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

#define NTM(a,k) (a)[(ulong)(k)*(ulong)ncol+(ulong)i]
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

kernel void hexv842_nt_mfub(constant Hexv842NtMfubResources &r [[buffer(0)]], constant Hexv842NtMfubParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *inputs=r.inputs;
    device const int *surf_i=r.surf_i;
    device int *out_i=r.out_i;
    device float *out=r.out;

    int i=int(gid), ncol=p.ncol,nz=p.nz,tiedtke_closure=p.tiedtke_closure; if(i>=ncol) return;
    float ztmst=p.ztmst,cp=p.cp,rd=p.rd,rv=p.rv,xlv=p.xlv,xls=p.xls,xlf=p.xlf,grav=p.grav;
    ulong st=ulong(nz+2)*ulong(ncol);
    device const float *ptte=inputs,*pqte=inputs+st,*paph=inputs+2*st,*puen=inputs+3*st,*pven=inputs+4*st;
    device const float *ptu=inputs+5*st,*pqu=inputs+6*st,*plu=inputs+7*st,*ztenh=inputs+8*st,*zqenh=inputs+9*st;
    device const int *ktype=surf_i,*kcbot=surf_i+ncol,*lndj=surf_i+2*ncol;
    device int *ldcum=out_i;
    device float *zdhpbl=out,*upbl=out+ncol,*zmfub=out+2*ncol;
    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);
    const int klev = nz;

    /* :468-469.  zcons2 = 3/(g*ztmst); the 3 is not a typo -- zcons at :468 is
     * the same quantity with 1.  ZTMST, cumastrn's step, which is
     * dt*stepcu: this parameter was named dt and so bound the MODEL step,
     * identical at stepcu = 1 and half the reference's at stepcu = 2 (found
     * by tools/ntiedtke_column_oracle; the closure below had the same). */
    const float zcons2 = mpas_nt_div(3.0f, mpas_nt_mul(c.g, ztmst));

    float dhpbl = 0.0f;
    float zdqpbl = 0.0f;   /* cu6 :832-839 */
    float up = 0.0f;
    bool ld = ldcum[i] != 0;

    /* do jk = 2, klev */
    for (int jk = 2; jk <= klev; ++jk) {
        if (!(jk >= kcbot[i] && ld)) continue;
        const float dp = mpas_nt_sub(NTM(paph, jk + 1), NTM(paph, jk));
        dhpbl = mpas_nt_add(dhpbl, mpas_nt_mul(
            mpas_nt_add(mpas_nt_mul(c.alv, NTM(pqte, jk)),
                      mpas_nt_mul(c.cpd, NTM(ptte, jk))), dp));
        /* module_cu_tiedtke.F:839 -- MOISTURE convergence alone, the
         * quantity cu6's deep first guess divides.  Accumulated in the
         * same loop over the same dp so the two agree cell for cell. */
        zdqpbl = mpas_nt_add(zdqpbl, mpas_nt_mul(NTM(pqte, jk), dp));
        if (lndj[i] == 0) {
            const float u = NTM(puen, jk), v = NTM(pven, jk);
            const float wspeed = sqrt32(mpas_nt_add(mpas_nt_mul(u, u),
                                                 mpas_nt_mul(v, v)));
            up = mpas_nt_add(up, mpas_nt_mul(wspeed, dp));
        }
    }

    float mfub = 0.0f;
    if (ld) {
        const int ikb = kcbot[i];
        const float zmfmax = mpas_nt_mul(
            mpas_nt_sub(NTM(paph, ikb), NTM(paph, ikb - 1)), zcons2);
        if (ktype[i] == 1) {
            if (tiedtke_closure) {
                /* module_cu_tiedtke.F:855-866.  cu6's deep first guess is
                 * MOISTURE CONVERGENCE, not cu16's geometric 0.1*zmfmax.
                 * Same zqumqe/zdqmin construction as the shallow arm
                 * below; the 0.01 floor is cu6 :862 and the cap :866. */
                const float zqumqe = mpas_nt_sub(
                    mpas_nt_add(NTM(pqu, ikb), NTM(plu, ikb)), NTM(zqenh, ikb));
                const float zdqmin = max(
                    mpas_nt_mul(0.01f, NTM(zqenh, ikb)), 1.0e-10f);
                if (zdqpbl > 0.0f && zqumqe > zdqmin) {
                    mfub = mpas_nt_div(zdqpbl,
                                     mpas_nt_mul(c.g, max(zqumqe, zdqmin)));
                    mfub = (mfub < zmfmax ? mfub : zmfmax);
                } else {
                    mfub = 0.01f;
                }
            } else {
                mfub = mpas_nt_mul(0.1f, zmfmax);
            }
        } else if (ktype[i] == 2) {
            const float zqumqe = mpas_nt_sub(
                mpas_nt_add(NTM(pqu, ikb), NTM(plu, ikb)), NTM(zqenh, ikb));
            const float zdqmin = max(
                mpas_nt_mul(0.01f, NTM(zqenh, ikb)), 1.0e-10f);
            float zdh = mpas_nt_add(
                mpas_nt_mul(c.cpd, mpas_nt_sub(NTM(ptu, ikb), NTM(ztenh, ikb))),
                mpas_nt_mul(c.alv, zqumqe));
            zdh = mpas_nt_mul(c.g, max(zdh, mpas_nt_mul(1.0e5f, zdqmin)));
            if (dhpbl > 0.0f) {
                mfub = mpas_nt_div(dhpbl, zdh);
                mfub = (mfub < zmfmax ? mfub : zmfmax);
            } else {
                mfub = mpas_nt_mul(0.1f, zmfmax);
                ld = false;                    /* :536 -- clears ldcum */
            }
        }
        /* ktype 0 or 3 with ldcum true is unreachable from cumastrn --
         * cutypen produces only 0/1/2 and ldcum is false whenever ktype is
         * 0 -- and the reference assigns nothing there, leaving zmfub at
         * its initialised zero.  So does this. */
    }

    zdhpbl[i] = dhpbl;
    upbl[i] = up;
    zmfub[i] = mfub;
    ldcum[i] = ld ? 1 : 0;

}
