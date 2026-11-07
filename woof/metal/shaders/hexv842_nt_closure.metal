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
#define NTK(a,k) (a)[(ulong)(k)*(ulong)ncol+(ulong)i]
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
#include "hexv842_nt_closure_bindings.metalh"
kernel void hexv842_nt_closure(constant Hexv842NtClosureResources &r [[buffer(0)]], constant Hexv842NtClosureParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *lev = r.lev;
    device const int *sf_i = r.sf_i;
    device const float *sf = r.sf;
    device float *out = r.out;
    int i=int(gid),ncol=p.ncol,nz=p.nz,tiedtke_closure=p.tiedtke_closure; if(i>=ncol) return;
    float ztmst=p.ztmst,cp=p.cp,rd=p.rd,rv=p.rv,xlv=p.xlv,xls=p.xls,xlf=p.xlf,grav=p.grav;
    ulong st=ulong(nz+2)*ulong(ncol);
    device const int *ldcum=sf_i,*ktype=sf_i+ncol,*kcbot=sf_i+2*ncol,*kctop=sf_i+3*ncol,*kdpl=sf_i+4*ncol,*loddraf=sf_i+5*ncol,*lndj=sf_i+6*ncol;
    device const float *wup=sf,*zmfub=sf+ncol,*zdhpbl=sf+2*ncol,*scale_fac=sf+3*ncol,*scale_fac2=sf+4*ncol;
    device const float *pgeoh=lev,*paph=lev+st,*pap=lev+2*st,*pgeo=lev+3*st,*pten=lev+4*st,*pqen=lev+5*st;
    device const float *ptenh=lev+6*st,*pqenh=lev+7*st,*ptu=lev+8*st,*pqu=lev+9*st,*plu=lev+10*st,*pmfu=lev+11*st;
    device const float *ptd=lev+12*st,*pqd=lev+13*st,*ptte=lev+14*st,*pqte=lev+15*st;
    device float *pmfd=out,*pmfds=out+st,*pmfdq=out+2*st,*pdmfdp=out+3*st,*pmfdde_rate=out+4*st;
    device float *zheat=out+5*st,*zcape=zheat+ncol,*zcape1=zcape+ncol,*zcape2=zcape1+ncol,*ztauc=zcape2+ncol;
    device float *ztaubl=ztauc+ncol,*ztau_o=ztaubl+ncol,*upbl=ztau_o+ncol,*zmfub1=upbl+ncol;
    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);
    const int klev = nz;
    const float zcons2 = mpas_nt_div(3.0f, mpas_nt_mul(c.g, ztmst));
    const bool ld = ldcum[i] != 0;
    const int kt = ktype[i];
    const bool deep = ld && kt == 1;

    /* :622-641  timescales.  Written only on the deep arm. */
    if (deep) {
        const int ikb = kcbot[i], ikt = kctop[i];
        zmfub1[i] = zmfub[i];
        ztauc[i] = mpas_nt_div(
            mpas_nt_sub(NTK(pgeoh, ikt), NTK(pgeoh, ikb)),
            mpas_nt_mul(mpas_nt_add(2.0f, min(15.0f, wup[i])), c.g));
        if (lndj[i] == 0) {
            upbl[i] = mpas_nt_add(2.0f, mpas_nt_div(
                upbl[i], mpas_nt_sub(NTK(paph, klev + 1), NTK(paph, ikb))));
            ztaubl[i] = mpas_nt_div(
                mpas_nt_sub(NTK(pgeoh, ikb), NTK(pgeoh, klev + 1)),
                mpas_nt_mul(c.g, upbl[i]));
            ztaubl[i] = min(300.0f, ztaubl[i]);
        } else {
            ztaubl[i] = ztauc[i];
        }
        zheat[i] = 0.0f; zcape[i] = 0.0f;
        zcape1[i] = 0.0f; zcape2[i] = 0.0f;
    }

    /* :644-668  the CAPE and heating integrals */
    for (int jk = 1; jk <= klev; ++jk) {
        if (deep && jk <= kcbot[i] && jk > kctop[i]) {
            const float zdz = mpas_nt_sub(NTK(pgeo, jk - 1), NTK(pgeo, jk));
            const float zdp = mpas_nt_sub(NTK(pap, jk), NTK(pap, jk - 1));
            zheat[i] = mpas_nt_add(zheat[i], mpas_nt_mul(
                mpas_nt_add(
                    mpas_nt_div(mpas_nt_add(mpas_nt_sub(NTK(pten, jk - 1),
                                                  NTK(pten, jk)),
                                        mpas_nt_mul(zdz, c.rcpd)),
                              NTK(ptenh, jk)),
                    mpas_nt_mul(c.vtmpc1, mpas_nt_sub(NTK(pqen, jk - 1),
                                                  NTK(pqen, jk)))),
                mpas_nt_mul(c.g, mpas_nt_add(NTK(pmfu, jk), NTK(pmfd, jk)))));
            /* :655-656 sums LEFT TO RIGHT:
             *     (ptu-ztenh)/ztenh + vtmpc1*(pqu-zqenh) - plu
             * is ((buoyancy + moisture) - plu).  It was transcribed as
             * buoyancy + (moisture - plu), which rounds differently; the
             * analytic fixture happened to agree, real columns from the
             * column oracle (tools/ntiedtke_column_oracle) do not. */
            zcape1[i] = mpas_nt_add(zcape1[i], mpas_nt_mul(
                mpas_nt_sub(
                    mpas_nt_add(
                        mpas_nt_div(mpas_nt_sub(NTK(ptu, jk), NTK(ptenh, jk)),
                                  NTK(ptenh, jk)),
                        mpas_nt_mul(c.vtmpc1,
                                  mpas_nt_sub(NTK(pqu, jk), NTK(pqenh, jk)))),
                    NTK(plu, jk)),
                zdp));
        }
        if (deep && jk >= kcbot[i]) {
            if (mpas_nt_sub(NTK(paph, klev + 1), NTK(paph, kdpl[i]))
                    < 50.0e2f) {
                const float zdp = mpas_nt_sub(NTK(paph, jk + 1),
                                            NTK(paph, jk));
                zcape2[i] = mpas_nt_add(zcape2[i], mpas_nt_mul(
                    mpas_nt_mul(ztaubl[i], mpas_nt_add(
                        mpas_nt_mul(mpas_nt_add(1.0f,
                                            mpas_nt_mul(c.vtmpc1,
                                                      NTK(pqen, jk))),
                                  NTK(ptte, jk)),
                        mpas_nt_mul(mpas_nt_mul(c.vtmpc1, NTK(pten, jk)),
                                  NTK(pqte, jk)))),
                    zdp));
            }
        }
    }

    /* :670-694  the deep closure */
    if (deep) {
        const int ikb = kcbot[i];
        float tc = ztauc[i];
        tc = max(ztmst, tc);
        tc = max(360.0f, tc);
        tc = min(10800.0f, tc);
        ztauc[i] = tc;
        /* modC.  cu16 :676 scales the adjustment time by scale_fac
         * (11.6246 at dx=4500); cu6 uses a FIXED 2400 s
         * (module_cu_tiedtke.F:105 parameter ztau = 2400.0), which is
         * precisely the scale-awareness deletion that separates the two
         * schemes' deep closures. */
        const float ztau = tiedtke_closure
            ? 2400.0f : mpas_nt_mul(tc, scale_fac[i]);
        ztau_o[i] = ztau;
        /* nonequil is .true. (cu_ntiedtke_common:49) */
        zcape2[i] = max(0.0f, zcape2[i]);
        zcape[i] = max(0.0f, min(mpas_nt_sub(zcape1[i], zcape2[i]),
                                     5000.0f));
        zheat[i] = max(1.0e-4f, zheat[i]);
        float b1 = mpas_nt_div(mpas_nt_mul(zcape[i], zmfub[i]),
                             mpas_nt_mul(zheat[i], ztau));
        b1 = max(b1, 0.001f);
        const float zmfmax = mpas_nt_mul(
            mpas_nt_sub(NTK(paph, ikb), NTK(paph, ikb - 1)), zcons2);
        zmfub1[i] = (b1 < zmfmax ? b1 : zmfmax);
    }

    /* :696-720  the shallow closure -- the ONLY use of scale_fac2 */
    if (ld && kt == 2) {
        const int ikb = kcbot[i];
        float zeps;
        if (NTK(pmfd, ikb) < 0.0f && loddraf[i] != 0) {
            zeps = mpas_nt_div(mpas_nt_neg(NTK(pmfd, ikb)),
                             max(zmfub[i], NT_CMFCMIN));
        } else {
            zeps = 0.0f;
        }
        const float zqumqe = mpas_nt_sub(
            mpas_nt_sub(mpas_nt_add(NTK(pqu, ikb), NTK(plu, ikb)),
                      mpas_nt_mul(zeps, NTK(pqd, ikb))),
            mpas_nt_mul(mpas_nt_sub(1.0f, zeps), NTK(pqenh, ikb)));
        const float zdqmin = max(mpas_nt_mul(0.01f, NTK(pqenh, ikb)),
                                   NT_CMFCMIN);
        const float zmfmax = mpas_nt_mul(
            mpas_nt_sub(NTK(paph, ikb), NTK(paph, ikb - 1)), zcons2);
        float zdh = mpas_nt_add(
            mpas_nt_mul(c.cpd, mpas_nt_sub(
                mpas_nt_sub(NTK(ptu, ikb), mpas_nt_mul(zeps, NTK(ptd, ikb))),
                mpas_nt_mul(mpas_nt_sub(1.0f, zeps), NTK(ptenh, ikb)))),
            mpas_nt_mul(c.alv, zqumqe));
        zdh = mpas_nt_mul(c.g, max(zdh, mpas_nt_mul(1.0e5f, zdqmin)));
        /* zdhpbl is an INPUT: cumastrn built it at :506-517 from CUTYPEN's
         * kcbot and ldcum, and cuascn has since changed both.  Recomputing
         * it from closure-time state is wrong on every shallow column. */
        float b1 = (zdhpbl[i] > 0.0f) ? mpas_nt_div(zdhpbl[i], zdh)
                                      : zmfub[i];
        b1 = mpas_nt_div(b1, scale_fac2[i]);
        zmfub1[i] = (b1 < zmfmax ? b1 : zmfmax);
    }

    /* :722-724  mid-level takes NEITHER factor */
    if (ld && kt == 3) zmfub1[i] = zmfub[i];

    /* :726-740  scale the downdraft by zmfub1/zmfub */
    if (ld) {
        const float zfac = mpas_nt_div(zmfub1[i],
                                     max(zmfub[i], NT_CMFCMIN));
        for (int jk = 1; jk <= klev; ++jk) {
            NTK(pmfd, jk) = mpas_nt_mul(NTK(pmfd, jk), zfac);
            NTK(pmfds, jk) = mpas_nt_mul(NTK(pmfds, jk), zfac);
            NTK(pmfdq, jk) = mpas_nt_mul(NTK(pmfdq, jk), zfac);
            NTK(pdmfdp, jk) = mpas_nt_mul(NTK(pdmfdp, jk), zfac);
            NTK(pmfdde_rate, jk) = mpas_nt_mul(NTK(pmfdde_rate, jk), zfac);
        }
    }

}
