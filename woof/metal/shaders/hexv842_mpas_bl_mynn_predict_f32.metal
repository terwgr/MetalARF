#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_predict_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float mynn_result(float a,float b,float value) {
    uint x=as_type<uint>(a),y=as_type<uint>(b),r=as_type<uint>(value);
    if ((r&0x7fffffffu)<=0x7f800000u) return value;
    if ((x&0x7fffffffu)>0x7f800000u) return as_type<float>(x|0x00400000u);
    if ((y&0x7fffffffu)>0x7f800000u) return as_type<float>(y|0x00400000u);
    return as_type<float>(0xffc00000u);
}
static inline float mynn_add(float a,float b){return mynn_result(a,b,mpas_add(a,b));}
static inline float mynn_sub(float a,float b){return mynn_result(a,b,mpas_sub(a,b));}
static inline float mynn_mul(float a,float b){return mynn_result(a,b,mpas_mul(a,b));}
static inline float mynn_div(float a,float b){return mynn_result(a,b,cu_div(a,b));}
#define mpas_add mynn_add
#define mpas_sub mynn_sub
#define mpas_mul mynn_mul
#define mpas_div mynn_div

static inline float mynn_neg(float a){return as_type<float>(as_type<uint>(a)^0x80000000u);}
static inline float mynn_min(float a,float b){return soft32_less(as_type<uint>(a),as_type<uint>(b))?a:b;}
static inline float mynn_max(float a,float b){return soft32_greater(as_type<uint>(a),as_type<uint>(b))?a:b;}
static inline float mynn_sqrt(float a){return mynn_result(a,a,cu_sqrt(a));}
static inline float mynn_square(float a){return mynn_mul(a,a);}

static inline float mynn_cube(float a){return mynn_mul(mynn_square(a),a);}

kernel void hexv842_mpas_bl_mynn_predict_f32(constant Hexv842MpasBlMynnPredictF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnPredictF32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;float closure=cfg.closure;float delt=cfg.delt;float ust=cfg.ust;float flt=cfg.flt;float flq=cfg.flq;float pmz=cfg.pmz;float phh=cfg.phh;float karman=cfg.karman;bool bl_mynn_edmf_tke=cfg.bl_mynn_edmf_tke!=0;bool bl_mynn_tkebudget=cfg.bl_mynn_tkebudget!=0;constexpr float b1=24.f,b2=15.f,sqfac=3.f;
#define dz(i) r.dz[ulong(col)*nl+((i)-1)]
#define el(i) r.el[ulong(col)*nl+((i)-1)]
#define dfq(i) r.dfq[ulong(col)*nl+((i)-1)]
#define rho(i) r.rho[ulong(col)*nl+((i)-1)]
#define s_aw(i) r.s_aw[ulong(col)*nl+((i)-1)]
#define s_awqke(i) r.s_awqke[ulong(col)*nl+((i)-1)]
#define pdk(i) r.pdk[ulong(col)*nl+((i)-1)]
#define pdt(i) r.pdt[ulong(col)*nl+((i)-1)]
#define pdq(i) r.pdq[ulong(col)*nl+((i)-1)]
#define pdc(i) r.pdc[ulong(col)*nl+((i)-1)]
#define qke(i) r.qke[ulong(col)*nl+((i)-1)]
#define tsq(i) r.tsq[ulong(col)*nl+((i)-1)]
#define qsq(i) r.qsq[ulong(col)*nl+((i)-1)]
#define cov(i) r.cov[ulong(col)*nl+((i)-1)]
#define qwt1d(i) r.qwt1d[ulong(col)*nl+((i)-1)]
#define qdiss1d(i) r.qdiss1d[ulong(col)*nl+((i)-1)]
#define tke_up(i) r.scratch[(ulong(col)*18+0)*(nl+1)+((i)-1)]
#define dzinv(i) r.scratch[(ulong(col)*18+1)*(nl+1)+((i)-1)]
#define qkw(i) r.scratch[(ulong(col)*18+2)*(nl+1)+((i)-1)]
#define bp(i) r.scratch[(ulong(col)*18+3)*(nl+1)+((i)-1)]
#define rp(i) r.scratch[(ulong(col)*18+4)*(nl+1)+((i)-1)]
#define df3q(i) r.scratch[(ulong(col)*18+5)*(nl+1)+((i)-1)]
#define dtz(i) r.scratch[(ulong(col)*18+6)*(nl+1)+((i)-1)]
#define a(i) r.scratch[(ulong(col)*18+7)*(nl+1)+((i)-1)]
#define b(i) r.scratch[(ulong(col)*18+8)*(nl+1)+((i)-1)]
#define c(i) r.scratch[(ulong(col)*18+9)*(nl+1)+((i)-1)]
#define d(i) r.scratch[(ulong(col)*18+10)*(nl+1)+((i)-1)]
#define x(i) r.scratch[(ulong(col)*18+11)*(nl+1)+((i)-1)]
#define rhoinv(i) r.scratch[(ulong(col)*18+12)*(nl+1)+((i)-1)]
#define rhoz(i) r.scratch[(ulong(col)*18+13)*(nl+1)+((i)-1)]
#define kqdz(i) r.scratch[(ulong(col)*18+14)*(nl+1)+((i)-1)]
#define kmdz(i) r.scratch[(ulong(col)*18+15)*(nl+1)+((i)-1)]
#define cp(i) r.scratch[(ulong(col)*18+16)*(nl+1)+((i)-1)]
#define dp(i) r.scratch[(ulong(col)*18+17)*(nl+1)+((i)-1)]

int k;
float vkz;
float pdk1;
float phm;
float pdt1;
float pdq1;
float pdc1;
float b1l;
float b2l;
float onoff;
// bl_mynn_subroutines.F90:2200
if ((!(bl_mynn_edmf_tke))) {
// bl_mynn_subroutines.F90:2201
onoff = 0.0f;
// bl_mynn_subroutines.F90:2202
} else {
// bl_mynn_subroutines.F90:2203
onoff = 1.0f;
// bl_mynn_subroutines.F90:2204
}
// bl_mynn_subroutines.F90:2207
vkz = mpas_mul(mpas_mul(karman, 0.5f), dz(kts));
// bl_mynn_subroutines.F90:2211
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2213
qkw(k) = mynn_sqrt(mynn_max(((float)(qke(k))), ((float)(0.0f))));
// bl_mynn_subroutines.F90:2214
df3q(k) = mpas_mul(sqfac, dfq(k));
// bl_mynn_subroutines.F90:2215
dtz(k) = mpas_div(delt, dz(k));
// bl_mynn_subroutines.F90:2216
}
// bl_mynn_subroutines.F90:2221
rhoz(kts) = rho(kts);
// bl_mynn_subroutines.F90:2222
rhoinv(kts) = mpas_div(1.f, rho(kts));
// bl_mynn_subroutines.F90:2223
kqdz(kts) = mpas_mul(rhoz(kts), df3q(kts));
// bl_mynn_subroutines.F90:2224
kmdz(kts) = mpas_mul(rhoz(kts), dfq(kts));
// bl_mynn_subroutines.F90:2225
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2226
rhoz(k) = mpas_div(mpas_add(mpas_mul(rho(k), dz(((k) - (1)))), mpas_mul(rho(((k) - (1))), dz(k))), mpas_add(dz(((k) - (1))), dz(k)));
// bl_mynn_subroutines.F90:2227
rhoz(k) = mynn_max(((float)(rhoz(k))), ((float)(1e-4f)));
// bl_mynn_subroutines.F90:2228
rhoinv(k) = mpas_div(1.f, mynn_max(((float)(rho(k))), ((float)(1e-4f))));
// bl_mynn_subroutines.F90:2229
kqdz(k) = mpas_mul(rhoz(k), df3q(k));
// bl_mynn_subroutines.F90:2230
kmdz(k) = mpas_mul(rhoz(k), dfq(k));
// bl_mynn_subroutines.F90:2231
}
// bl_mynn_subroutines.F90:2232
rhoz(((kte) + (1))) = rhoz(kte);
// bl_mynn_subroutines.F90:2233
kqdz(((kte) + (1))) = mpas_mul(rhoz(((kte) + (1))), df3q(kte));
// bl_mynn_subroutines.F90:2234
kmdz(((kte) + (1))) = mpas_mul(rhoz(((kte) + (1))), dfq(kte));
// bl_mynn_subroutines.F90:2237
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2238
kqdz(k) = mynn_max(((float)(kqdz(k))), ((float)(mpas_mul(0.5f, s_aw(k)))));
// bl_mynn_subroutines.F90:2239
kqdz(k) = mynn_max(((float)(kqdz(k))), ((float)(mpas_mul(mynn_neg(0.5f), mpas_sub(s_aw(k), s_aw(((k) + (1))))))));
// bl_mynn_subroutines.F90:2240
kmdz(k) = mynn_max(((float)(kmdz(k))), ((float)(mpas_mul(0.5f, s_aw(k)))));
// bl_mynn_subroutines.F90:2241
kmdz(k) = mynn_max(((float)(kmdz(k))), ((float)(mpas_mul(mynn_neg(0.5f), mpas_sub(s_aw(k), s_aw(((k) + (1))))))));
// bl_mynn_subroutines.F90:2242
}
// bl_mynn_subroutines.F90:2245
pdk1 = mpas_div(mpas_mul(mpas_mul(2.0f, mynn_cube(ust)), pmz), vkz);
// bl_mynn_subroutines.F90:2246
phm = mpas_div(mpas_mul(mpas_div(2.0f, ust), phh), vkz);
// bl_mynn_subroutines.F90:2247
pdt1 = mpas_mul(phm, mynn_square(flt));
// bl_mynn_subroutines.F90:2248
pdq1 = mpas_mul(phm, mynn_square(flq));
// bl_mynn_subroutines.F90:2249
pdc1 = mpas_mul(mpas_mul(phm, flt), flq);
// bl_mynn_subroutines.F90:2252
pdk(kts) = mpas_sub(pdk1, pdk(((kts) + (1))));
// bl_mynn_subroutines.F90:2257
pdt(kts) = pdt(((kts) + (1)));
// bl_mynn_subroutines.F90:2258
pdq(kts) = pdq(((kts) + (1)));
// bl_mynn_subroutines.F90:2259
pdc(kts) = pdc(((kts) + (1)));
// bl_mynn_subroutines.F90:2263
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2264
b1l = mpas_mul(mpas_mul(b1, 0.5f), mpas_add(el(((k) + (1))), el(k)));
// bl_mynn_subroutines.F90:2265
bp(k) = mpas_div(mpas_mul(2.f, qkw(k)), b1l);
// bl_mynn_subroutines.F90:2266
rp(k) = mpas_add(pdk(((k) + (1))), pdk(k));
// bl_mynn_subroutines.F90:2267
}
// bl_mynn_subroutines.F90:2275
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2287
a(k) = mpas_add(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kqdz(k)), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(k)), onoff));
// bl_mynn_subroutines.F90:2289
b(k) = mpas_add(mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kqdz(k), kqdz(((k) + (1))))), rhoinv(k))), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), mpas_sub(s_aw(k), s_aw(((k) + (1))))), onoff)), mpas_mul(bp(k), delt));
// bl_mynn_subroutines.F90:2292
c(k) = mpas_sub(mpas_mul(mpas_mul(mynn_neg(dtz(k)), kqdz(((k) + (1)))), rhoinv(k)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(0.5f, dtz(k)), rhoinv(k)), s_aw(((k) + (1)))), onoff));
// bl_mynn_subroutines.F90:2294
d(k) = mpas_add(mpas_add(mpas_mul(rp(k), delt), qke(k)), mpas_mul(mpas_mul(mpas_mul(dtz(k), rhoinv(k)), mpas_sub(s_awqke(k), s_awqke(((k) + (1))))), onoff));
// bl_mynn_subroutines.F90:2296
}
// bl_mynn_subroutines.F90:2311
a(kte) = 0.f;
// bl_mynn_subroutines.F90:2312
b(kte) = 1.f;
// bl_mynn_subroutines.F90:2313
c(kte) = 0.f;
// bl_mynn_subroutines.F90:2314
d(kte) = qke(kte);
// bl_mynn_subroutines.F90:2317
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:2319
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2321
qke(k) = mynn_max(((float)(x(k))), ((float)(1.e-4f)));
// bl_mynn_subroutines.F90:2322
qke(k) = mynn_min(((float)(qke(k))), ((float)(150.f)));
// bl_mynn_subroutines.F90:2323
}
// bl_mynn_subroutines.F90:2327
if (bl_mynn_tkebudget) {
// bl_mynn_subroutines.F90:2329
for (int mynn_element=1; mynn_element <= nl; ++mynn_element) { tke_up(mynn_element) = mpas_mul(0.5f, qke(mynn_element)); }
// bl_mynn_subroutines.F90:2330
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { dzinv(mynn_element+1) = mpas_div(1.f, dz(mynn_element+1)); }
// bl_mynn_subroutines.F90:2331
k = kts;
// bl_mynn_subroutines.F90:2332
qwt1d(k) = mpas_mul(dzinv(k), mpas_add(mpas_sub(mpas_mul(kqdz(((k) + (1))), mpas_sub(tke_up(((k) + (1))), tke_up(k))), mpas_mul(kqdz(k), tke_up(k))), mpas_mul(mpas_mul(mpas_mul(0.5f, rhoinv(k)), mpas_add(mpas_add(mpas_mul(s_aw(((k) + (1))), tke_up(((k) + (1)))), mpas_mul(mpas_sub(s_aw(((k) + (1))), s_aw(k)), tke_up(k))), mpas_sub(s_awqke(k), s_awqke(((k) + (1)))))), onoff)));
// bl_mynn_subroutines.F90:2337
for (k=((kts) + (1)); (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2338
qwt1d(k) = mpas_mul(dzinv(k), mpas_add(mpas_sub(mpas_mul(kqdz(((k) + (1))), mpas_sub(tke_up(((k) + (1))), tke_up(k))), mpas_mul(kqdz(k), mpas_sub(tke_up(k), tke_up(((k) - (1)))))), mpas_mul(mpas_mul(mpas_mul(0.5f, rhoinv(k)), mpas_add(mpas_sub(mpas_add(mpas_mul(s_aw(((k) + (1))), tke_up(((k) + (1)))), mpas_mul(mpas_sub(s_aw(((k) + (1))), s_aw(k)), tke_up(k))), mpas_mul(s_aw(k), tke_up(((k) - (1))))), mpas_sub(s_awqke(k), s_awqke(((k) + (1)))))), onoff)));
// bl_mynn_subroutines.F90:2344
}
// bl_mynn_subroutines.F90:2345
k = kte;
// bl_mynn_subroutines.F90:2346
qwt1d(k) = mpas_mul(dzinv(k), mpas_add(mpas_mul(mynn_neg(kqdz(k)), mpas_sub(tke_up(k), tke_up(((k) - (1))))), mpas_mul(mpas_mul(mpas_mul(0.5f, rhoinv(k)), mpas_add(mpas_sub(mpas_mul(mynn_neg(s_aw(k)), tke_up(k)), mpas_mul(s_aw(k), tke_up(((k) - (1))))), s_awqke(k))), onoff)));
// bl_mynn_subroutines.F90:2349
for (int mynn_element=0; mynn_element < nl; ++mynn_element) { qdiss1d(mynn_element+1) = mpas_mul(bp(mynn_element+1), tke_up(mynn_element+1)); }
// bl_mynn_subroutines.F90:2350
}
// bl_mynn_subroutines.F90:2353
if (((closure) > (2.5f))) {
// bl_mynn_subroutines.F90:2356
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2357
b2l = mpas_mul(mpas_mul(b2, 0.5f), mpas_add(el(((k) + (1))), el(k)));
// bl_mynn_subroutines.F90:2358
bp(k) = mpas_div(mpas_mul(2.f, qkw(k)), b2l);
// bl_mynn_subroutines.F90:2359
rp(k) = mpas_add(pdq(((k) + (1))), pdq(k));
// bl_mynn_subroutines.F90:2360
}
// bl_mynn_subroutines.F90:2369
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2370
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:2371
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), mpas_mul(bp(k), delt));
// bl_mynn_subroutines.F90:2372
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:2373
d(k) = mpas_add(mpas_mul(rp(k), delt), qsq(k));
// bl_mynn_subroutines.F90:2374
}
// bl_mynn_subroutines.F90:2376
a(kte) = mynn_neg(1.f);
// bl_mynn_subroutines.F90:2377
b(kte) = 1.f;
// bl_mynn_subroutines.F90:2378
c(kte) = 0.f;
// bl_mynn_subroutines.F90:2379
d(kte) = 0.f;
// bl_mynn_subroutines.F90:2382
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:2384
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2386
qsq(k) = mynn_max(((float)(x(k))), ((float)(1e-17f)));
// bl_mynn_subroutines.F90:2387
}
// bl_mynn_subroutines.F90:2388
} else {
// bl_mynn_subroutines.F90:2390
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2391
if (((qkw(k)) <= (0.0f))) {
// bl_mynn_subroutines.F90:2392
b2l = 0.0f;
// bl_mynn_subroutines.F90:2393
} else {
// bl_mynn_subroutines.F90:2394
b2l = mpas_div(mpas_mul(mpas_mul(b2, 0.25f), mpas_add(el(((k) + (1))), el(k))), qkw(k));
// bl_mynn_subroutines.F90:2395
}
// bl_mynn_subroutines.F90:2396
qsq(k) = mpas_mul(b2l, mpas_add(pdq(((k) + (1))), pdq(k)));
// bl_mynn_subroutines.F90:2397
}
// bl_mynn_subroutines.F90:2398
qsq(kte) = qsq(((kte) - (1)));
// bl_mynn_subroutines.F90:2399
}
// bl_mynn_subroutines.F90:2402
if (((closure) >= (3.0f))) {
// bl_mynn_subroutines.F90:2408
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2409
b2l = mpas_mul(mpas_mul(b2, 0.5f), mpas_add(el(k), el(((k) + (1)))));
// bl_mynn_subroutines.F90:2410
bp(k) = mpas_div(mpas_mul(2.f, qkw(k)), b2l);
// bl_mynn_subroutines.F90:2411
rp(k) = mpas_add(pdt(k), pdt(((k) + (1))));
// bl_mynn_subroutines.F90:2412
}
// bl_mynn_subroutines.F90:2422
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2428
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:2429
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), mpas_mul(bp(k), delt));
// bl_mynn_subroutines.F90:2430
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:2431
d(k) = mpas_add(mpas_mul(rp(k), delt), tsq(k));
// bl_mynn_subroutines.F90:2432
}
// bl_mynn_subroutines.F90:2441
a(kte) = mynn_neg(1.f);
// bl_mynn_subroutines.F90:2442
b(kte) = 1.f;
// bl_mynn_subroutines.F90:2443
c(kte) = 0.f;
// bl_mynn_subroutines.F90:2444
d(kte) = 0.f;
// bl_mynn_subroutines.F90:2447
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:2449
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2451
tsq(k) = x(k);
// bl_mynn_subroutines.F90:2452
}
// bl_mynn_subroutines.F90:2456
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2457
b2l = mpas_mul(mpas_mul(b2, 0.5f), mpas_add(el(k), el(((k) + (1)))));
// bl_mynn_subroutines.F90:2458
bp(k) = mpas_div(mpas_mul(2.f, qkw(k)), b2l);
// bl_mynn_subroutines.F90:2459
rp(k) = mpas_add(pdc(k), pdc(((k) + (1))));
// bl_mynn_subroutines.F90:2460
}
// bl_mynn_subroutines.F90:2470
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2476
a(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(k)), rhoinv(k));
// bl_mynn_subroutines.F90:2477
b(k) = mpas_add(mpas_add(1.f, mpas_mul(mpas_mul(dtz(k), mpas_add(kmdz(k), kmdz(((k) + (1))))), rhoinv(k))), mpas_mul(bp(k), delt));
// bl_mynn_subroutines.F90:2478
c(k) = mpas_mul(mpas_mul(mynn_neg(dtz(k)), kmdz(((k) + (1)))), rhoinv(k));
// bl_mynn_subroutines.F90:2479
d(k) = mpas_add(mpas_mul(rp(k), delt), cov(k));
// bl_mynn_subroutines.F90:2480
}
// bl_mynn_subroutines.F90:2489
a(kte) = mynn_neg(1.f);
// bl_mynn_subroutines.F90:2490
b(kte) = 1.f;
// bl_mynn_subroutines.F90:2491
c(kte) = 0.f;
// bl_mynn_subroutines.F90:2492
d(kte) = 0.f;
// bl_mynn_subroutines.F90:2495
{ int n=kte;

float m;
int i;
// bl_mynn_subroutines.F90:4397
cp(1) = mpas_div(c(1), b(1));
// bl_mynn_subroutines.F90:4398
dp(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4400
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4401
m = mpas_sub(b(i), mpas_mul(cp(((i) - (1))), a(i)));
// bl_mynn_subroutines.F90:4402
cp(i) = mpas_div(c(i), m);
// bl_mynn_subroutines.F90:4403
dp(i) = mpas_div(mpas_sub(d(i), mpas_mul(dp(((i) - (1))), a(i))), m);
// bl_mynn_subroutines.F90:4404
}
// bl_mynn_subroutines.F90:4406
x(n) = dp(n);
// bl_mynn_subroutines.F90:4408
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4409
x(i) = mpas_sub(dp(i), mpas_mul(cp(i), x(((i) + (1)))));
// bl_mynn_subroutines.F90:4410
}

}
// bl_mynn_subroutines.F90:2497
for (k=kts; (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:2499
cov(k) = x(k);
// bl_mynn_subroutines.F90:2500
}
// bl_mynn_subroutines.F90:2502
} else {
// bl_mynn_subroutines.F90:2505
for (k=kts; (1) > 0 ? k <= (((kte) - (1))) : k >= (((kte) - (1))); k += (1)) {
// bl_mynn_subroutines.F90:2506
if (((qkw(k)) <= (0.0f))) {
// bl_mynn_subroutines.F90:2507
b2l = 0.0f;
// bl_mynn_subroutines.F90:2508
} else {
// bl_mynn_subroutines.F90:2509
b2l = mpas_div(mpas_mul(mpas_mul(b2, 0.25f), mpas_add(el(((k) + (1))), el(k))), qkw(k));
// bl_mynn_subroutines.F90:2510
}
// bl_mynn_subroutines.F90:2512
tsq(k) = mpas_mul(b2l, mpas_add(pdt(((k) + (1))), pdt(k)));
// bl_mynn_subroutines.F90:2513
cov(k) = mpas_mul(b2l, mpas_add(pdc(((k) + (1))), pdc(k)));
// bl_mynn_subroutines.F90:2514
}
// bl_mynn_subroutines.F90:2516
tsq(kte) = tsq(((kte) - (1)));
// bl_mynn_subroutines.F90:2517
cov(kte) = cov(((kte) - (1)));
// bl_mynn_subroutines.F90:2519
}

#undef dz
#undef el
#undef dfq
#undef rho
#undef s_aw
#undef s_awqke
#undef pdk
#undef pdt
#undef pdq
#undef pdc
#undef qke
#undef tsq
#undef qsq
#undef cov
#undef qwt1d
#undef qdiss1d
#undef tke_up
#undef dzinv
#undef qkw
#undef bp
#undef rp
#undef df3q
#undef dtz
#undef a
#undef b
#undef c
#undef d
#undef x
#undef rhoinv
#undef rhoz
#undef kqdz
#undef kmdz
#undef cp
#undef dp
}
