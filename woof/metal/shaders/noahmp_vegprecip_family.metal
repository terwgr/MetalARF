#include <metal_stdlib>
#include "noahmp_vegprecip_phenology_bindings.metalh"
#include "noahmp_vegprecip_precip_heat_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

static inline float vp_min(float a, float b) { return a < b ? a : b; }
static inline float vp_max(float a, float b) { return a > b ? a : b; }

// PHENOLOGY, packed inputs: snowh,tv,lat,julian,hvt,hvb,tmin,LAIM[12],SAIM[12].
kernel void noahmp_vegprecip_phenology(
    constant NoahmpVegprecipPhenologyResources &r [[buffer(0)]],
    constant NoahmpVegprecipPhenologyParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncase)) return;
    const ulong base = ulong(tid);
    const ulong xb = base * 31ul;
    const ulong ib = base * 6ul;
    const ulong yb = base * 6ul;
    device const float *x = r.x + xb;
    device const int *ix = r.ix + ib;
    device float *y = r.y + yb;
    const float snowh = x[0], tv = x[1], lat = x[2];
    const float julian = x[3], hvt = x[4], hvb = x[5], tmin = x[6];
    const int yearlen = ix[1];

    const float rylen = float(yearlen);
    float day = julian;
    if (lat < 0.0f) day = fmod(julian + 0.5f * rylen, rylen);
    const float t = (12.0f * day) / rylen;
    int it1 = int(trunc(t + 0.5f));
    const int it2_raw = it1 + 1;
    const float wt1 = (float(it1) + 0.5f) - t;
    const float wt2 = 1.0f - wt1;
    if (it1 < 1) it1 = 12;
    const int it2 = it2_raw > 12 ? 1 : it2_raw;

    float lai = wt1 * x[7 + it1 - 1] + wt2 * x[7 + it2 - 1];
    float sai = wt1 * x[19 + it1 - 1] + wt2 * x[19 + it2 - 1];
    if (sai < 0.05f) sai = 0.0f;
    if (lai < 0.05f || sai == 0.0f) lai = 0.0f;
    if (ix[0] == ix[2] || ix[0] == ix[3] || ix[0] == ix[4] || ix[5] != 0) {
        lai = 0.0f;
        sai = 0.0f;
    }

    const float span = hvt - hvb;
    const float db = vp_min(vp_max(snowh - hvb, 0.0f), span);
    float fb = db / vp_max(1.0e-6f, span);
    if (hvt > 0.0f && hvt <= 1.0f) {
        const float snowhc = hvt * exp(-snowh / 0.2f);
        fb = vp_min(snowh, snowhc) / snowhc;
    }
    float elai = lai * (1.0f - fb);
    float esai = sai * (1.0f - fb);
    if (esai < 0.05f) esai = 0.0f;
    if (elai < 0.05f || esai == 0.0f) elai = 0.0f;
    const float igs = tv > tmin ? 1.0f : 0.0f;
    y[0] = lai; y[1] = sai; y[2] = elai; y[3] = esai; y[4] = igs; y[5] = fb;
}

// PRECIP_HEAT, packed inputs: dt,uu,vv,elai,esai,fveg,bdfall,rain,snow,fp,
// canliq,canice,tv,sfctmp,tg,ch2op. Integer ix[0] is IST.
kernel void noahmp_vegprecip_precip_heat(
    constant NoahmpVegprecipPrecipHeatResources &r [[buffer(0)]],
    constant NoahmpVegprecipPrecipHeatParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncase)) return;
    const ulong xb = ulong(tid) * 16ul;
    const ulong ib = ulong(tid);
    const ulong yb = ulong(tid) * 16ul;
    device const float *x = r.x + xb;
    device const int *ix = r.ix + ib;
    device float *y = r.y + yb;
    const int ist = ix[0];
    const float dt=x[0], uu=x[1], vv=x[2], elai=x[3], esai=x[4];
    const float fveg=x[5], bdfall=x[6], rain=x[7], snow=x[8], fp=x[9];
    float canliq=x[10], canice=x[11];
    const float tv=x[12], sfctmp=x[13], tg=x[14], ch2op=x[15];

    float qintr=0.0f, qdripr=0.0f, qthror=0.0f;
    float qints=0.0f, qdrips=0.0f, qthros=0.0f, icedrip=0.0f;
    const float lsai = elai + esai;
    const float maxliq = (fveg * ch2op) * lsai;
    if (lsai > 0.0f) {
        qintr = (fveg * rain) * fp;
        const float cap = ((maxliq - canliq) / dt)
            * (1.0f - exp(-((rain * dt) / maxliq)));
        qintr = vp_max(vp_min(qintr, cap), 0.0f);
        qdripr = (fveg * rain) - qintr;
        qthror = (1.0f - fveg) * rain;
        canliq = vp_max(0.0f, canliq + qintr * dt);
    } else {
        qthror = rain;
        if (canliq > 0.0f) { qdripr = canliq / dt; canliq = 0.0f; }
    }

    float pah_ac = ((fveg * rain) * 4188.0f) * (sfctmp - tv);
    float pah_cg = (qdripr * 4188.0f) * (tv - tg);
    float pah_ag = (qthror * 4188.0f) * (sfctmp - tg);
    const float maxsno = ((fveg * 6.6f) * (0.27f + 46.0f / bdfall)) * lsai;
    if (lsai > 0.0f) {
        qints = (fveg * snow) * fp;
        const float cap = ((maxsno - canice) / dt)
            * (1.0f - exp(-((snow * dt) / maxsno)));
        qints = vp_max(vp_min(qints, cap), 0.0f);
        const float ft = vp_max(0.0f, (tv - 270.15f) / 1.87e5f);
        const float fv = sqrt(uu * uu + vv * vv) / 1.56e5f;
        icedrip = vp_max(0.0f, canice) * (fv + ft);
        icedrip = vp_min(canice / dt + qints, icedrip);
        qdrips = ((fveg * snow) - qints) + icedrip;
        qthros = (1.0f - fveg) * snow;
        canice = vp_max(0.0f, canice + (qints - icedrip) * dt);
    } else {
        qthros = snow;
        if (canice > 0.0f) { qdrips = canice / dt; canice = 0.0f; }
    }

    float fwet;
    if (canice > 0.0f)
        fwet = vp_max(0.0f, canice) / vp_max(maxsno, 1.0e-6f);
    else
        fwet = vp_max(0.0f, canliq) / vp_max(maxliq, 1.0e-6f);
    fwet = pow(vp_min(fwet, 1.0f), 0.667f);
    const float cmc = canliq + canice;
    pah_ac += ((fveg * snow) * 2094.0f) * (sfctmp - tv);
    pah_cg += (qdrips * 2094.0f) * (tv - tg);
    pah_ag += (qthros * 2094.0f) * (sfctmp - tg);
    float pahv = pah_ac - pah_cg, pahg = pah_cg, pahb = pah_ag;
    if (fveg > 0.0f && fveg < 1.0f) {
        pahg /= fveg; pahb /= (1.0f - fveg);
    } else if (fveg <= 0.0f) {
        pahb = pahg + pahb; pahg = 0.0f; pahv = 0.0f;
    } else if (fveg >= 1.0f) {
        pahb = 0.0f;
    }
    pahv = vp_min(vp_max(pahv, -20.0f), 20.0f);
    pahg = vp_min(vp_max(pahg, -20.0f), 20.0f);
    pahb = vp_min(vp_max(pahb, -20.0f), 20.0f);
    const float qrain = qdripr + qthror;
    float qsnow = qdrips + qthros;
    float snowhin = qsnow / bdfall;
    if (ist == 2 && tg > 273.16f) { qsnow = 0.0f; snowhin = 0.0f; }
    y[0]=canliq; y[1]=canice; y[2]=qintr; y[3]=qdripr;
    y[4]=qthror; y[5]=qints; y[6]=qdrips; y[7]=qthros;
    y[8]=pahv; y[9]=pahg; y[10]=pahb; y[11]=qrain;
    y[12]=qsnow; y[13]=snowhin; y[14]=fwet; y[15]=cmc;
}
