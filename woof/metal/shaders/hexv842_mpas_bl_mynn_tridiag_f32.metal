#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_tridiag_f32_bindings.metalh"
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

kernel void hexv842_mpas_bl_mynn_tridiag1_f32(constant Hexv842MpasBlMynnTridiag1F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnTridiag1F32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl;int n=nl;
#define a(i) r.a[(ulong(col)*nl)+((i)-1)]
#define b(i) r.b[(ulong(col)*nl)+((i)-1)]
#define c(i) r.c[(ulong(col)*nl)+((i)-1)]
#define d(i) r.d[(ulong(col)*nl)+((i)-1)]
#define q(i) r.q[(ulong(col)*nl)+((i)-1)]

int i;
float p;
// bl_mynn_subroutines.F90:4362
c(n) = 0.f;
// bl_mynn_subroutines.F90:4363
q(1) = mpas_div(mynn_neg(c(1)), b(1));
// bl_mynn_subroutines.F90:4364
d(1) = mpas_div(d(1), b(1));
// bl_mynn_subroutines.F90:4366
for (i=2; (1) > 0 ? i <= (n) : i >= (n); i += (1)) {
// bl_mynn_subroutines.F90:4367
p = mpas_div(1.f, mpas_add(b(i), mpas_mul(a(i), q(((i) - (1))))));
// bl_mynn_subroutines.F90:4368
q(i) = mpas_mul(mynn_neg(c(i)), p);
// bl_mynn_subroutines.F90:4369
d(i) = mpas_mul(mpas_sub(d(i), mpas_mul(a(i), d(((i) - (1))))), p);
// bl_mynn_subroutines.F90:4370
}
// bl_mynn_subroutines.F90:4372
for (i=((n) - (1)); ((-(1))) > 0 ? i <= (1) : i >= (1); i += ((-(1)))) {
// bl_mynn_subroutines.F90:4373
d(i) = mpas_add(d(i), mpas_mul(q(i), d(((i) + (1)))));
// bl_mynn_subroutines.F90:4374
}

#undef a
#undef b
#undef c
#undef d
#undef q
}

kernel void hexv842_mpas_bl_mynn_tridiag2_f32(constant Hexv842MpasBlMynnTridiag2F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnTridiag2F32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl;int n=nl;
#define a(i) r.a[(ulong(col)*nl)+((i)-1)]
#define b(i) r.b[(ulong(col)*nl)+((i)-1)]
#define c(i) r.c[(ulong(col)*nl)+((i)-1)]
#define d(i) r.d[(ulong(col)*nl)+((i)-1)]
#define x(i) r.x[(ulong(col)*nl)+((i)-1)]
#define cp(i) r.cp[(ulong(col)*nl)+((i)-1)]
#define dp(i) r.dp[(ulong(col)*nl)+((i)-1)]

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

#undef a
#undef b
#undef c
#undef d
#undef x
#undef cp
#undef dp
}

kernel void hexv842_mpas_bl_mynn_tridiag3_f32(constant Hexv842MpasBlMynnTridiag3F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnTridiag3F32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl;int kte=nl;
#define a(i) r.a[(ulong(col)*nl)+((i)-1)]
#define b(i) r.b[(ulong(col)*nl)+((i)-1)]
#define c(i) r.c[(ulong(col)*nl)+((i)-1)]
#define d(i) r.d[(ulong(col)*nl)+((i)-1)]
#define x(i) r.x[(ulong(col)*nl)+((i)-1)]

constexpr int kts = 1;
int in;
// bl_mynn_subroutines.F90:4439
for (in=((kte) - (1)); ((-(1))) > 0 ? in <= (kts) : in >= (kts); in += ((-(1)))) {
// bl_mynn_subroutines.F90:4440
d(in) = mpas_sub(d(in), mpas_div(mpas_mul(c(in), d(((in) + (1)))), b(((in) + (1)))));
// bl_mynn_subroutines.F90:4441
b(in) = mpas_sub(b(in), mpas_div(mpas_mul(c(in), a(((in) + (1)))), b(((in) + (1)))));
// bl_mynn_subroutines.F90:4442
}
// bl_mynn_subroutines.F90:4444
for (in=((kts) + (1)); (1) > 0 ? in <= (kte) : in >= (kte); in += (1)) {
// bl_mynn_subroutines.F90:4445
d(in) = mpas_sub(d(in), mpas_div(mpas_mul(a(in), d(((in) - (1)))), b(((in) - (1)))));
// bl_mynn_subroutines.F90:4446
}
// bl_mynn_subroutines.F90:4448
for (in=kts; (1) > 0 ? in <= (kte) : in >= (kte); in += (1)) {
// bl_mynn_subroutines.F90:4449
x(in) = mpas_div(d(in), b(in));
// bl_mynn_subroutines.F90:4450
}
// bl_mynn_subroutines.F90:4452
return;

#undef a
#undef b
#undef c
#undef d
#undef x
}
