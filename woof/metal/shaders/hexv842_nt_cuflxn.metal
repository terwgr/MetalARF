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

#define NT_ZTAUMEL 18000.0f
#define NT_ZCUCOV 0.05f
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
static inline float nt_foeewm(float tt, thread const NtConst &c) {
    const float a = nt_foealfa(tt);
    const float el = gfk_exp(mpas_nt_div(
        mpas_nt_mul(NT_C3LES, mpas_nt_sub(tt, NT_TMELT)),
        mpas_nt_sub(tt, NT_C4LES)));
    const float ei = gfk_exp(mpas_nt_div(
        mpas_nt_mul(NT_C3IES, mpas_nt_sub(tt, NT_TMELT)),
        mpas_nt_sub(tt, NT_C4IES)));
    return mpas_nt_mul(c.c2es, mpas_nt_add(mpas_nt_mul(a, el),
                                       mpas_nt_mul(mpas_nt_sub(1.0f, a), ei)));
}
#include "hexv842_nt_cuflxn_bindings.metalh"
kernel void hexv842_nt_cuflxn(constant Hexv842NtCuflxnResources &r [[buffer(0)]], constant Hexv842NtCuflxnParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    device const float *in = r.in;
    device const int *sf_i = r.sf_i;
    device int *out_i = r.out_i;
    device float *out = r.out;
    int i=int(gid),ncol=p.ncol,klev=p.klev; if(i>=ncol) return;
    float ztmst=p.ztmst,cp=p.cp,rd=p.rd,rv=p.rv,xlv=p.xlv,xls=p.xls,xlf=p.xlf,grav=p.grav;
    ulong st=ulong(klev+2)*ulong(ncol);
    device const int *kcbot=sf_i,*kctop=sf_i+ncol,*kdtop=sf_i+2*ncol,*lndj=sf_i+3*ncol;
    device const float *pten=in,*ptenh=in+st,*pqenh=in+2*st,*paph=in+3*st,*pap=in+4*st,*pqen=in+5*st,*pgeoh=in+6*st;
    device int *ldcum=out_i,*lddraf=out_i+ncol,*ktype=out_i+2*ncol;
    device float *pmfu=out,*pmfd=out+st,*pmfus=out+2*st,*pmfds=out+3*st,*pmfuq=out+4*st,*pmfdq=out+5*st;
    device float *pmful=out+6*st,*plude=out+7*st,*plglac=out+8*st,*pdmfup=out+9*st,*pdmfdp=out+10*st;
    device float *pmfdde_rate=out+11*st,*pqsen=out+12*st,*pdpmel=out+13*st,*pmflxr=out+14*st,*pmflxs=out+15*st,*prain=out+16*st;
    const NtConst c = nt_init(cp, rd, rv, xlv, xls, xlf, grav);


    const float zcons1a = mpas_nt_div(
        c.cpd, mpas_nt_mul(mpas_nt_mul(c.alf, c.g), NT_ZTAUMEL));
    const float zcons2 = mpas_nt_div(3.0f, mpas_nt_mul(c.g, ztmst));
    const float zcpecons = mpas_nt_div(5.44e-4f, c.g);

    /* 1.0 (:2865-2875) */
    float pr = 0.0f;
    const int kb = kcbot[i];
    const int ktop = kctop[i];
    const int kdt = kdtop[i];
    const int ld = ldcum[i];
    int ldd = lddraf[i];
    if (!ld || kdt < ktop) ldd = 0;
    if (!ld) ktype[i] = 0;
    const int kt3 = (ktype[i] == 3);
    int idbas = klev;
    const float rhevap = (lndj[i] == 1) ? 0.7f : 0.9f;

    const int ktopm2 = 2;             /* :2877, unconditional.  See header. */

    for (int jk = ktopm2; jk <= klev; ++jk) {
        const int ikb = min(jk + 1, klev);
        NTC(pmflxr, jk) = 0.0f;
        NTC(pmflxs, jk) = 0.0f;
        NTC(pdpmel, jk) = 0.0f;
        if (ld && jk >= ktop) {
            NTC(pmfus, jk) = mpas_nt_sub(NTC(pmfus, jk),
                mpas_nt_mul(NTC(pmfu, jk),
                    mpas_nt_add(mpas_nt_mul(c.cpd, NTC(ptenh, jk)),
                              NTC(pgeoh, jk))));
            NTC(pmfuq, jk) = mpas_nt_sub(NTC(pmfuq, jk),
                mpas_nt_mul(NTC(pmfu, jk), NTC(pqenh, jk)));
            NTC(plglac, jk) = mpas_nt_mul(NTC(pmfu, jk), NTC(plglac, jk));
            const int llddraf = ldd && (jk >= kdt);
            if (llddraf && jk >= kdt) {
                NTC(pmfds, jk) = mpas_nt_sub(NTC(pmfds, jk),
                    mpas_nt_mul(NTC(pmfd, jk),
                        mpas_nt_add(mpas_nt_mul(c.cpd, NTC(ptenh, jk)),
                                  NTC(pgeoh, jk))));
                NTC(pmfdq, jk) = mpas_nt_sub(NTC(pmfdq, jk),
                    mpas_nt_mul(NTC(pmfd, jk), NTC(pqenh, jk)));
            } else {
                NTC(pmfd, jk) = 0.0f;
                NTC(pmfds, jk) = 0.0f;
                NTC(pmfdq, jk) = 0.0f;
                NTC(pdmfdp, jk - 1) = 0.0f;
            }
            if (llddraf && NTC(pmfd, jk) < 0.0f
                && fabs(NTC(pmfd, ikb)) < 1.0e-20f) {
                idbas = jk;
            }
        } else {
            NTC(pmfu, jk) = 0.0f;  NTC(pmfd, jk) = 0.0f;
            NTC(pmfus, jk) = 0.0f; NTC(pmfds, jk) = 0.0f;
            NTC(pmfuq, jk) = 0.0f; NTC(pmfdq, jk) = 0.0f;
            NTC(pmful, jk) = 0.0f; NTC(plglac, jk) = 0.0f;
            NTC(pdmfup, jk - 1) = 0.0f;
            NTC(pdmfdp, jk - 1) = 0.0f;
            NTC(plude, jk - 1) = 0.0f;
        }
    }

    NTC(pmflxr, klev + 1) = 0.0f;
    NTC(pmflxs, klev + 1) = 0.0f;

    /* the cloud-base taper (:2926-2938) */
    if (ld) {
        const int ikb = kb;
        const int ik = ikb + 1;
        float zzp = mpas_nt_div(
            mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, ik)),
            mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, ikb)));
        if (kt3) zzp = mpas_nt_mul(zzp, zzp);
        NTC(pmfu, ik) = mpas_nt_mul(NTC(pmfu, ikb), zzp);
        NTC(pmfus, ik) = mpas_nt_mul(
            mpas_nt_sub(NTC(pmfus, ikb),
                mpas_nt_mul(nt_foelhm(NTC(ptenh, ikb), c), NTC(pmful, ikb))),
            zzp);
        NTC(pmfuq, ik) = mpas_nt_mul(
            mpas_nt_add(NTC(pmfuq, ikb), NTC(pmful, ikb)), zzp);
        NTC(pmful, ik) = 0.0f;
    }

    for (int jk = ktopm2; jk <= klev; ++jk) {
        if (ld && jk > kb + 1) {
            const int ikb = kb + 1;
            float zzp = mpas_nt_div(
                mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, jk)),
                mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, ikb)));
            if (kt3) zzp = mpas_nt_mul(zzp, zzp);
            NTC(pmfu, jk) = mpas_nt_mul(NTC(pmfu, ikb), zzp);
            NTC(pmfus, jk) = mpas_nt_mul(NTC(pmfus, ikb), zzp);
            NTC(pmfuq, jk) = mpas_nt_mul(NTC(pmfuq, ikb), zzp);
            NTC(pmful, jk) = 0.0f;
        }
        const int ik = idbas;
        const int llddraf = ldd && (jk > ik) && (ik < klev);
        if (llddraf && ik == kb + 1) {
            float zzp = mpas_nt_div(
                mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, jk)),
                mpas_nt_sub(NTC(paph, klev + 1), NTC(paph, ik)));
            if (kt3) zzp = mpas_nt_mul(zzp, zzp);
            NTC(pmfd, jk) = mpas_nt_mul(NTC(pmfd, ik), zzp);
            NTC(pmfds, jk) = mpas_nt_mul(NTC(pmfds, ik), zzp);
            NTC(pmfdq, jk) = mpas_nt_mul(NTC(pmfdq, ik), zzp);
            NTC(pmfdde_rate, jk) =
                mpas_nt_neg(mpas_nt_sub(NTC(pmfd, jk - 1), NTC(pmfd, jk)));
        } else if (llddraf && ik != kb + 1 && jk == ik + 1) {
            NTC(pmfdde_rate, jk) =
                mpas_nt_neg(mpas_nt_sub(NTC(pmfd, jk - 1), NTC(pmfd, jk)));
        }
    }

    /* 2. melting and the rain/snow split (:2975-3011) */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        if (!(ld && jk >= ktop - 1)) continue;
        pr = mpas_nt_add(pr, NTC(pdmfup, jk));
        if (NTC(pmflxs, jk) > 0.0f && NTC(pten, jk) > NT_TMELT) {
            const float zcons1 = mpas_nt_mul(zcons1a, mpas_nt_add(1.0f,
                mpas_nt_mul(0.5f, mpas_nt_sub(NTC(pten, jk), NT_TMELT))));
            const float zfac = mpas_nt_mul(zcons1,
                mpas_nt_sub(NTC(paph, jk + 1), NTC(paph, jk)));
            const float zsnmlt = min(NTC(pmflxs, jk),
                mpas_nt_mul(zfac, mpas_nt_sub(NTC(pten, jk), NT_TMELT)));
            NTC(pdpmel, jk) = zsnmlt;
            NTC(pqsen, jk) = mpas_nt_div(
                nt_foeewm(mpas_nt_sub(NTC(pten, jk),
                                    mpas_nt_div(zsnmlt, zfac)), c),
                NTC(pap, jk));
        }
        float zalfaw = nt_foealfa(NTC(pten, jk));
        /* No liquid precipitation above the melting level. */
        if (NTC(pten, jk) < NT_TMELT && zalfaw > 0.0f) {
            NTC(plglac, jk) = mpas_nt_add(NTC(plglac, jk),
                mpas_nt_mul(zalfaw,
                    mpas_nt_add(NTC(pdmfup, jk), NTC(pdmfdp, jk))));
            zalfaw = 0.0f;
        }
        NTC(pmflxr, jk + 1) = mpas_nt_add(
            mpas_nt_add(NTC(pmflxr, jk),
                mpas_nt_mul(zalfaw,
                    mpas_nt_add(NTC(pdmfup, jk), NTC(pdmfdp, jk)))),
            NTC(pdpmel, jk));
        NTC(pmflxs, jk + 1) = mpas_nt_sub(
            mpas_nt_add(NTC(pmflxs, jk),
                mpas_nt_mul(mpas_nt_sub(1.0f, zalfaw),
                    mpas_nt_add(NTC(pdmfup, jk), NTC(pdmfdp, jk)))),
            NTC(pdpmel, jk));
        if (mpas_nt_add(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1)) < 0.0f) {
            NTC(pdmfdp, jk) = mpas_nt_neg(mpas_nt_add(
                mpas_nt_add(NTC(pmflxr, jk), NTC(pmflxs, jk)),
                NTC(pdmfup, jk)));
            NTC(pmflxr, jk + 1) = 0.0f;
            NTC(pmflxs, jk + 1) = 0.0f;
            NTC(pdpmel, jk) = 0.0f;
        } else if (NTC(pmflxr, jk + 1) < 0.0f) {
            NTC(pmflxs, jk + 1) =
                mpas_nt_add(NTC(pmflxs, jk + 1), NTC(pmflxr, jk + 1));
            NTC(pmflxr, jk + 1) = 0.0f;
        } else if (NTC(pmflxs, jk + 1) < 0.0f) {
            NTC(pmflxr, jk + 1) =
                mpas_nt_add(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1));
            NTC(pmflxs, jk + 1) = 0.0f;
        }
    }


    /* the sub-cloud evaporation (:3012-3057).
     *
     * NAMED COVERAGE GAP: measured, 48 columns reach this guard and ZERO
     * move pdmfup, because zdrfl1 carries max(0, pqsen - pqen) and the
     * fixture's soundings are saturated below cloud base wherever the block
     * runs.  So the 0.5777 power law, zrmin and the rhevap land/sea split
     * are transcribed and graded only in the sense that their guard is
     * evaluated -- and rhevap is the ONLY path lndj takes into cuflxn, so
     * land/sea is untested here too.  See docs/ntiedtke/PORT-RECORD.md section 16.
     */
    for (int jk = ktopm2; jk <= klev; ++jk) {
        if (!(ld && jk >= kb)) continue;
        const float zrfl = mpas_nt_add(NTC(pmflxr, jk), NTC(pmflxs, jk));
        if (zrfl > 1.0e-20f) {
            const float zdp = mpas_nt_sub(NTC(paph, jk + 1), NTC(paph, jk));
            /* LEFT TO RIGHT, as written at :3017-3018:
             *     sqrt(paph/paph_sfc)/5.09e-3*zrfl/zcucov
             * is ((sqrt(...)/5.09e-3)*zrfl)/zcucov.  This was transcribed as
             * (sqrt(...)/5.09e-3)*(zrfl/zcucov), a different rounding, and
             * the analytic fixture never moved pdmfup here so nothing saw it
             * (the named coverage gap below); real columns from the column
             * oracle (tools/ntiedtke_column_oracle) do. */
            const float zbase = mpas_nt_div(
                mpas_nt_mul(
                    mpas_nt_div(mpas_nt_sqrt(mpas_nt_div(NTC(paph, jk),
                                                   NTC(paph, klev + 1))),
                              5.09e-3f),
                    zrfl),
                NT_ZCUCOV);
            const float zdrfl1 = mpas_nt_mul(
                mpas_nt_mul(mpas_nt_mul(mpas_nt_mul(zcpecons,
                    max(0.0f, mpas_nt_sub(NTC(pqsen, jk), NTC(pqen, jk)))),
                    NT_ZCUCOV),
                    gfk_pow(zbase, 0.5777f)),
                zdp);
            float zrnew = mpas_nt_sub(zrfl, zdrfl1);
            /* :3021-3022, left to right as well:
             *     zcucov*max(...)*zcons2*(paph(jk+1)-paph(jk))
             * is ((zcucov*max(...))*zcons2)*dp, not (zcucov*max)*(zcons2*dp). */
            const float zrmin = mpas_nt_sub(zrfl,
                mpas_nt_mul(mpas_nt_mul(mpas_nt_mul(NT_ZCUCOV,
                    max(0.0f, mpas_nt_sub(
                        mpas_nt_mul(rhevap, NTC(pqsen, jk)), NTC(pqen, jk)))),
                    zcons2), zdp));
            const float mpas_precip_nonnegative = (zrnew > 0.0f ? zrnew : 0.0f);
            const float zrfln = (mpas_precip_nonnegative > zrmin ? mpas_precip_nonnegative : zrmin);
            const float zdrfl = min(0.0f, mpas_nt_sub(zrfln, zrfl));
            const float zdenom = mpas_nt_div(1.0f, max(1.0e-20f,
                mpas_nt_add(NTC(pmflxr, jk), NTC(pmflxs, jk))));
            float zalfaw = nt_foealfa(NTC(pten, jk));
            if (NTC(pten, jk) < NT_TMELT) zalfaw = 0.0f;
            const float zpdr = mpas_nt_mul(zalfaw, NTC(pdmfdp, jk));
            const float zpds =
                mpas_nt_mul(mpas_nt_sub(1.0f, zalfaw), NTC(pdmfdp, jk));
            NTC(pmflxr, jk + 1) = mpas_nt_add(
                mpas_nt_add(mpas_nt_add(NTC(pmflxr, jk), zpdr), NTC(pdpmel, jk)),
                mpas_nt_mul(mpas_nt_mul(zdrfl, NTC(pmflxr, jk)), zdenom));
            NTC(pmflxs, jk + 1) = mpas_nt_add(
                mpas_nt_sub(mpas_nt_add(NTC(pmflxs, jk), zpds),
                          NTC(pdpmel, jk)),
                mpas_nt_mul(mpas_nt_mul(zdrfl, NTC(pmflxs, jk)), zdenom));
            NTC(pdmfup, jk) = mpas_nt_add(NTC(pdmfup, jk), zdrfl);
            if (mpas_nt_add(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1)) < 0.0f) {
                NTC(pdmfup, jk) = mpas_nt_sub(NTC(pdmfup, jk),
                    mpas_nt_add(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1)));
                NTC(pmflxr, jk + 1) = 0.0f;
                NTC(pmflxs, jk + 1) = 0.0f;
                NTC(pdpmel, jk) = 0.0f;
            } else if (NTC(pmflxr, jk + 1) < 0.0f) {
                NTC(pmflxs, jk + 1) =
                    mpas_nt_add(NTC(pmflxs, jk + 1), NTC(pmflxr, jk + 1));
                NTC(pmflxr, jk + 1) = 0.0f;
            } else if (NTC(pmflxs, jk + 1) < 0.0f) {
                NTC(pmflxr, jk + 1) =
                    mpas_nt_add(NTC(pmflxr, jk + 1), NTC(pmflxs, jk + 1));
                NTC(pmflxs, jk + 1) = 0.0f;
            }
        } else {
            NTC(pmflxr, jk + 1) = 0.0f;
            NTC(pmflxs, jk + 1) = 0.0f;
            NTC(pdmfdp, jk) = 0.0f;
            NTC(pdpmel, jk) = 0.0f;
        }
    }

    prain[i] = pr;
    lddraf[i] = ldd;
#undef NTC

}
