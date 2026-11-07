#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "soft_gfk_log.metalh"
#include "hexv842_mpas_rrtmg_lw_setcoef_f32_bindings.metalh"
#pragma clang fp contract(off)
#define NBNDLW 16
#define MXMOL 38
#define RLW_AD(a,b) mpas_add(a,b)
#define RLW_SU(a,b) mpas_sub(a,b)
#define RLW_MU(a,b) mpas_mul(a,b)
#define RLW_DV(a,b) cu_div(a,b)
#define rlw_log(x) as_type<float>(soft_gfk_log(as_type<uint>(x)))
kernel void hexv842_mpas_rrtmg_lw_setcoef_f32(constant Hexv842MpasRrtmgLwSetcoefF32Resources &r [[buffer(0)]],constant Hexv842MpasRrtmgLwSetcoefF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int ncol=p.ncol,nlayers=p.nlayers,istart=p.istart;
 device const float* pavel=r.pavel;
 device const float* tavel=r.tavel;
 device const float* tz=r.tz;
 device const float* tbound_v=r.tbound_v;
 device const float* semiss=r.semiss;
 device const float* coldry=r.coldry;
 device const float* wkl=r.wkl;
 device const float* wbroad=r.wbroad;
 device const float* totplnk=r.totplnk;
 device const float* totplk16=r.totplk16;
 device const float* preflog=r.preflog;
 device const float* tref=r.tref;
 device const float* chi_mls=r.chi_mls;
 device int* laytrop_v=r.laytrop_v;
 device int* jp=r.jp;
 device int* jt=r.jt;
 device int* jt1=r.jt1;
 device float* planklay=r.planklay;
 device float* planklev=r.planklev;
 device float* plankbnd=r.plankbnd;
 device float* colh2o=r.colh2o;
 device float* colco2=r.colco2;
 device float* colo3=r.colo3;
 device float* coln2o=r.coln2o;
 device float* colco=r.colco;
 device float* colch4=r.colch4;
 device float* colo2=r.colo2;
 device float* colbrd=r.colbrd;
 device float* fac00=r.fac00;
 device float* fac01=r.fac01;
 device float* fac10=r.fac10;
 device float* fac11=r.fac11;
 device float* rat_h2oco2=r.rat_h2oco2;
 device float* rat_h2oco2_1=r.rat_h2oco2_1;
 device float* rat_h2oo3=r.rat_h2oo3;
 device float* rat_h2oo3_1=r.rat_h2oo3_1;
 device float* rat_h2on2o=r.rat_h2on2o;
 device float* rat_h2on2o_1=r.rat_h2on2o_1;
 device float* rat_h2och4=r.rat_h2och4;
 device float* rat_h2och4_1=r.rat_h2och4_1;
 device float* rat_n2oco2=r.rat_n2oco2;
 device float* rat_n2oco2_1=r.rat_n2oco2_1;
 device float* rat_o3co2=r.rat_o3co2;
 device float* rat_o3co2_1=r.rat_o3co2_1;
 device float* selffac=r.selffac;
 device float* selffrac=r.selffrac;
 device int* indself=r.indself;
 device float* forfac=r.forfac;
 device float* forfrac=r.forfrac;
 device int* indfor=r.indfor;
 device float* minorfrac=r.minorfrac;
 device float* scaleminor=r.scaleminor;
 device float* scaleminorn2=r.scaleminorn2;
 device int* indminor=r.indminor;

    int col = int(gid);
    if (col >= ncol) return;
    const int nl = nlayers;
    device const float* pav = pavel + (long)col * nl;
    device const float* tav_ = tavel + (long)col * nl;
    device const float* tzc = tz + (long)col * (nl + 1);
    device const float* sem = semiss + (long)col * NBNDLW;
    device const float* cdry = coldry + (long)col * nl;
    device const float* wklc = wkl + (long)col * MXMOL * nl;
    device const float* wbr = wbroad + (long)col * nl;
#define TOTPLNK(i, b) totplnk[(i - 1) + 181 * (b - 1)]
#define CHI(s, j) chi_mls[(s - 1) + 7 * (j - 1)]
#define OUT1(a) a[(long)col * nl + (lay - 1)]
#define PLLAY(lay, b) planklay[((long)col * nl + (lay - 1)) * NBNDLW + (b - 1)]
#define PLLEV(lev, b) planklev[((long)col * (nl + 1) + (lev)) * NBNDLW + (b - 1)]

    float stpfac = RLW_DV(296.0f, 1013.0f);
    float tbound = tbound_v[col];

    int indbound = (int)RLW_SU(tbound, 159.0f);
    if (indbound < 1) indbound = 1;
    else if (indbound > 180) indbound = 180;
    float tbndfrac = RLW_SU(RLW_SU(tbound, 159.0f), (float)indbound);
    int indlev0 = (int)RLW_SU(tzc[0], 159.0f);
    if (indlev0 < 1) indlev0 = 1;
    else if (indlev0 > 180) indlev0 = 180;
    float t0frac = RLW_SU(RLW_SU(tzc[0], 159.0f), (float)indlev0);
    int laytrop = 0;

    for (int lay = 1; lay <= nl; ++lay) {
        float tavl = tav_[lay - 1];
        int indlay = (int)RLW_SU(tavl, 159.0f);
        if (indlay < 1) indlay = 1;
        else if (indlay > 180) indlay = 180;
        float tlayfrac = RLW_SU(RLW_SU(tavl, 159.0f), (float)indlay);
        float tzl = tzc[lay];
        int indlev = (int)RLW_SU(tzl, 159.0f);
        if (indlev < 1) indlev = 1;
        else if (indlev > 180) indlev = 180;
        float tlevfrac = RLW_SU(RLW_SU(tzl, 159.0f), (float)indlev);

        for (int ib = 1; ib <= 15; ++ib) {
            if (lay == 1) {
                float dbdtlev = RLW_SU(TOTPLNK(indbound + 1, ib),
                                       TOTPLNK(indbound, ib));
                plankbnd[(long)col * NBNDLW + (ib - 1)] =
                    RLW_MU(sem[ib - 1],
                           RLW_AD(TOTPLNK(indbound, ib),
                                  RLW_MU(tbndfrac, dbdtlev)));
                dbdtlev = RLW_SU(TOTPLNK(indlev0 + 1, ib),
                                 TOTPLNK(indlev0, ib));
                PLLEV(0, ib) = RLW_AD(TOTPLNK(indlev0, ib),
                                      RLW_MU(t0frac, dbdtlev));
            }
            float dbdtlev = RLW_SU(TOTPLNK(indlev + 1, ib),
                                   TOTPLNK(indlev, ib));
            float dbdtlay = RLW_SU(TOTPLNK(indlay + 1, ib),
                                   TOTPLNK(indlay, ib));
            PLLAY(lay, ib) = RLW_AD(TOTPLNK(indlay, ib),
                                    RLW_MU(tlayfrac, dbdtlay));
            PLLEV(lay, ib) = RLW_AD(TOTPLNK(indlev, ib),
                                    RLW_MU(tlevfrac, dbdtlev));
        }
        {
            const int ib = 16;
            if (istart == 16) {
                if (lay == 1) {
                    float dbdtlev = RLW_SU(totplk16[indbound],
                                           totplk16[indbound - 1]);
                    plankbnd[(long)col * NBNDLW + (ib - 1)] =
                        RLW_MU(sem[ib - 1],
                               RLW_AD(totplk16[indbound - 1],
                                      RLW_MU(tbndfrac, dbdtlev)));
                    dbdtlev = RLW_SU(TOTPLNK(indlev0 + 1, ib),
                                     TOTPLNK(indlev0, ib));
                    PLLEV(0, ib) = RLW_AD(totplk16[indlev0 - 1],
                                          RLW_MU(t0frac, dbdtlev));
                }
                float dbdtlev = RLW_SU(totplk16[indlev],
                                       totplk16[indlev - 1]);
                float dbdtlay = RLW_SU(totplk16[indlay],
                                       totplk16[indlay - 1]);
                PLLAY(lay, ib) = RLW_AD(totplk16[indlay - 1],
                                        RLW_MU(tlayfrac, dbdtlay));
                PLLEV(lay, ib) = RLW_AD(totplk16[indlev - 1],
                                        RLW_MU(tlevfrac, dbdtlev));
            } else {
                if (lay == 1) {
                    float dbdtlev = RLW_SU(TOTPLNK(indbound + 1, ib),
                                           TOTPLNK(indbound, ib));
                    plankbnd[(long)col * NBNDLW + (ib - 1)] =
                        RLW_MU(sem[ib - 1],
                               RLW_AD(TOTPLNK(indbound, ib),
                                      RLW_MU(tbndfrac, dbdtlev)));
                    dbdtlev = RLW_SU(TOTPLNK(indlev0 + 1, ib),
                                     TOTPLNK(indlev0, ib));
                    PLLEV(0, ib) = RLW_AD(TOTPLNK(indlev0, ib),
                                          RLW_MU(t0frac, dbdtlev));
                }
                float dbdtlev = RLW_SU(TOTPLNK(indlev + 1, ib),
                                       TOTPLNK(indlev, ib));
                float dbdtlay = RLW_SU(TOTPLNK(indlay + 1, ib),
                                       TOTPLNK(indlay, ib));
                PLLAY(lay, ib) = RLW_AD(TOTPLNK(indlay, ib),
                                        RLW_MU(tlayfrac, dbdtlay));
                PLLEV(lay, ib) = RLW_AD(TOTPLNK(indlev, ib),
                                        RLW_MU(tlevfrac, dbdtlev));
            }
        }

        // Purity guard: an out-of-contract nonpositive pavel would make
        // plog NaN; !(NaN <= 4.56) then counts the layer into laytrop
        // and the band kernels index absa/selfref outside their tables
        // (pool-layout-dependent reads).  The clamp keeps plog finite
        // and every derived index inside its table's domain; for the
        // positive pressures the prep contract guarantees it is the
        // identity, bitwise.
        float pavl = pav[lay - 1];
        if (!soft32_greater(as_type<uint>(pavl),0u)) pavl = 1.0e-6f;
        float plog = rlw_log(pavl);
        int jpl = (int)RLW_SU(36.0f, RLW_MU(5.0f, RLW_AD(plog, 0.04f)));
        if (jpl < 1) jpl = 1;
        else if (jpl > 58) jpl = 58;
        OUT1(jp) = jpl;
        int jp1 = jpl + 1;
        float fp = RLW_MU(5.0f, RLW_SU(preflog[jpl - 1], plog));

        int jtl = (int)RLW_AD(3.0f, RLW_DV(RLW_SU(tavl, tref[jpl - 1]),
                                           15.0f));
        if (jtl < 1) jtl = 1;
        else if (jtl > 4) jtl = 4;
        OUT1(jt) = jtl;
        float ft = RLW_SU(RLW_DV(RLW_SU(tavl, tref[jpl - 1]), 15.0f),
                          (float)(jtl - 3));
        int jt1l = (int)RLW_AD(3.0f, RLW_DV(RLW_SU(tavl, tref[jp1 - 1]),
                                            15.0f));
        if (jt1l < 1) jt1l = 1;
        else if (jt1l > 4) jt1l = 4;
        OUT1(jt1) = jt1l;
        float ft1 = RLW_SU(RLW_DV(RLW_SU(tavl, tref[jp1 - 1]), 15.0f),
                           (float)(jt1l - 3));
        float water = RLW_DV(wklc[0 * nl + (lay - 1)], cdry[lay - 1]);
        float scalefac = RLW_DV(RLW_MU(pav[lay - 1], stpfac), tavl);

        if (!(plog <= 4.56f)) {
            laytrop = laytrop + 1;

            OUT1(forfac) = RLW_DV(scalefac, RLW_AD(1.0f, water));
            float factor = RLW_DV(RLW_SU(332.0f, tavl), 36.0f);
            int indforl = min(2, max(1, (int)factor));
            OUT1(indfor) = indforl;
            OUT1(forfrac) = RLW_SU(factor, (float)indforl);

            OUT1(selffac) = RLW_MU(water, OUT1(forfac));
            factor = RLW_DV(RLW_SU(tavl, 188.0f), 7.2f);
            int indselfl = min(9, max(1, (int)factor - 7));
            OUT1(indself) = indselfl;
            OUT1(selffrac) = RLW_SU(factor, (float)(indselfl + 7));

            OUT1(scaleminor) = RLW_DV(pav[lay - 1], tavl);
            OUT1(scaleminorn2) = RLW_MU(
                RLW_DV(pav[lay - 1], tavl),
                RLW_DV(wbr[lay - 1],
                       RLW_AD(cdry[lay - 1], wklc[0 * nl + (lay - 1)])));
            factor = RLW_DV(RLW_SU(tavl, 180.8f), 7.2f);
            int indminorl = min(18, max(1, (int)factor));
            OUT1(indminor) = indminorl;
            OUT1(minorfrac) = RLW_SU(factor, (float)indminorl);

            OUT1(rat_h2oco2) = RLW_DV(CHI(1, jpl), CHI(2, jpl));
            OUT1(rat_h2oco2_1) = RLW_DV(CHI(1, jpl + 1), CHI(2, jpl + 1));
            OUT1(rat_h2oo3) = RLW_DV(CHI(1, jpl), CHI(3, jpl));
            OUT1(rat_h2oo3_1) = RLW_DV(CHI(1, jpl + 1), CHI(3, jpl + 1));
            OUT1(rat_h2on2o) = RLW_DV(CHI(1, jpl), CHI(4, jpl));
            OUT1(rat_h2on2o_1) = RLW_DV(CHI(1, jpl + 1), CHI(4, jpl + 1));
            OUT1(rat_h2och4) = RLW_DV(CHI(1, jpl), CHI(6, jpl));
            OUT1(rat_h2och4_1) = RLW_DV(CHI(1, jpl + 1), CHI(6, jpl + 1));
            OUT1(rat_n2oco2) = RLW_DV(CHI(4, jpl), CHI(2, jpl));
            OUT1(rat_n2oco2_1) = RLW_DV(CHI(4, jpl + 1), CHI(2, jpl + 1));
        } else {
            OUT1(forfac) = RLW_DV(scalefac, RLW_AD(1.0f, water));
            float factor = RLW_DV(RLW_SU(tavl, 188.0f), 36.0f);
            OUT1(indfor) = 3;
            OUT1(forfrac) = RLW_SU(factor, 1.0f);

            OUT1(selffac) = RLW_MU(water, OUT1(forfac));

            OUT1(scaleminor) = RLW_DV(pav[lay - 1], tavl);
            OUT1(scaleminorn2) = RLW_MU(
                RLW_DV(pav[lay - 1], tavl),
                RLW_DV(wbr[lay - 1],
                       RLW_AD(cdry[lay - 1], wklc[0 * nl + (lay - 1)])));
            factor = RLW_DV(RLW_SU(tavl, 180.8f), 7.2f);
            int indminorl = min(18, max(1, (int)factor));
            OUT1(indminor) = indminorl;
            OUT1(minorfrac) = RLW_SU(factor, (float)indminorl);

            OUT1(rat_h2oco2) = RLW_DV(CHI(1, jpl), CHI(2, jpl));
            OUT1(rat_h2oco2_1) = RLW_DV(CHI(1, jpl + 1), CHI(2, jpl + 1));
            OUT1(rat_o3co2) = RLW_DV(CHI(3, jpl), CHI(2, jpl));
            OUT1(rat_o3co2_1) = RLW_DV(CHI(3, jpl + 1), CHI(2, jpl + 1));
        }

        OUT1(colh2o) = RLW_MU(1.e-20f, wklc[0 * nl + (lay - 1)]);
        OUT1(colco2) = RLW_MU(1.e-20f, wklc[1 * nl + (lay - 1)]);
        OUT1(colo3) = RLW_MU(1.e-20f, wklc[2 * nl + (lay - 1)]);
        OUT1(coln2o) = RLW_MU(1.e-20f, wklc[3 * nl + (lay - 1)]);
        OUT1(colco) = RLW_MU(1.e-20f, wklc[4 * nl + (lay - 1)]);
        OUT1(colch4) = RLW_MU(1.e-20f, wklc[5 * nl + (lay - 1)]);
        OUT1(colo2) = RLW_MU(1.e-20f, wklc[6 * nl + (lay - 1)]);
        if (soft32_equal(as_type<uint>(OUT1(colco2)),0u)) OUT1(colco2) = RLW_MU(1.e-32f, cdry[lay - 1]);
        if (soft32_equal(as_type<uint>(OUT1(colo3)),0u)) OUT1(colo3) = RLW_MU(1.e-32f, cdry[lay - 1]);
        if (soft32_equal(as_type<uint>(OUT1(coln2o)),0u)) OUT1(coln2o) = RLW_MU(1.e-32f, cdry[lay - 1]);
        if (soft32_equal(as_type<uint>(OUT1(colco)),0u)) OUT1(colco) = RLW_MU(1.e-32f, cdry[lay - 1]);
        if (soft32_equal(as_type<uint>(OUT1(colch4)),0u)) OUT1(colch4) = RLW_MU(1.e-32f, cdry[lay - 1]);
        OUT1(colbrd) = RLW_MU(1.e-20f, wbr[lay - 1]);

        float compfp = RLW_SU(1.0f, fp);
        OUT1(fac10) = RLW_MU(compfp, ft);
        OUT1(fac00) = RLW_MU(compfp, RLW_SU(1.0f, ft));
        OUT1(fac11) = RLW_MU(fp, ft1);
        OUT1(fac01) = RLW_MU(fp, RLW_SU(1.0f, ft1));

        OUT1(selffac) = RLW_MU(OUT1(colh2o), OUT1(selffac));
        OUT1(forfac) = RLW_MU(OUT1(colh2o), OUT1(forfac));
    }
    laytrop_v[col] = laytrop;
#undef TOTPLNK
#undef CHI
#undef OUT1
#undef PLLAY
#undef PLLEV

}
