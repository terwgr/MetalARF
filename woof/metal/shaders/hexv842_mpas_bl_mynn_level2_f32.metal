#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_level2_f32_bindings.metalh"
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

kernel void hexv842_mpas_bl_mynn_level2_f32(constant Hexv842MpasBlMynnLevel2F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnLevel2F32Params &cfg [[buffer(1)]],uint col [[thread_position_in_grid]]) {
if(col>=uint(cfg.nc))return;int nl=cfg.nl,kts=1,kte=nl;float gtr=mynn_div(cfg.gravity,300.f),tv0=mynn_mul(cfg.ep1,300.f);
constexpr float b1 = 24.0f;
constexpr float b2 = 15.0f;
constexpr float pr = 0.74f;
constexpr float g1 = 0.235f;
constexpr float c2 = 0.729f;
constexpr float c3 = 0.340f;
constexpr float c5 = 0.2f;
constexpr float a1 = ((((b1) * (((1.0f) - (((3.0f) * (g1))))))) / (6.0f));
constexpr float c1 = ((g1) - (((1.0f) / (((((3.0f) * (a1))) * (2.88449914061481660f))))));
constexpr float a2 = ((((a1) * (((g1) - (c1))))) / (((g1) * (pr))));
constexpr float g2 = ((((((b2) / (b1))) * (((1.0f) - (c3))))) + (((((((2.0f) * (a1))) / (b1))) * (((3.0f) - (((2.0f) * (c2))))))));
constexpr float ckmod = 1.f;
#define dz(i) r.dz[ulong(col)*nl+((i)-1)]
#define u(i) r.u[ulong(col)*nl+((i)-1)]
#define v(i) r.v[ulong(col)*nl+((i)-1)]
#define thl(i) r.thl[ulong(col)*nl+((i)-1)]
#define thetav(i) r.thetav[ulong(col)*nl+((i)-1)]
#define qw(i) r.qw[ulong(col)*nl+((i)-1)]
#define ql(i) r.ql[ulong(col)*nl+((i)-1)]
#define vt(i) r.vt[ulong(col)*nl+((i)-1)]
#define vq(i) r.vq[ulong(col)*nl+((i)-1)]
#define dtl(i) r.dtl[ulong(col)*nl+((i)-1)]
#define dqw(i) r.dqw[ulong(col)*nl+((i)-1)]
#define dtv(i) r.dtv[ulong(col)*nl+((i)-1)]
#define gm(i) r.gm[ulong(col)*nl+((i)-1)]
#define gh(i) r.gh[ulong(col)*nl+((i)-1)]
#define sm(i) r.sm[ulong(col)*nl+((i)-1)]
#define sh(i) r.sh[ulong(col)*nl+((i)-1)]

int k;
float rfc;
float f1;
float f2;
float rf1;
float rf2;
float smc;
float shc;
float ri1;
float ri2;
float ri3;
float ri4;
float duz;
float dtz;
float dqz;
float vtt;
float vqq;
float dtq;
float dzk;
float afk;
float abk;
float ri;
float rf;
float a2fac;
// bl_mynn_subroutines.F90:711
rfc = mpas_div(g1, mpas_add(g1, g2));
// bl_mynn_subroutines.F90:712
f1 = mpas_add(mpas_add(mpas_mul(b1, mpas_sub(g1, c1)), mpas_mul(mpas_mul(mpas_mul(3.0f, a2), mpas_sub(1.0f, c2)), mpas_sub(1.0f, c5))), mpas_mul(mpas_mul(2.0f, a1), mpas_sub(3.0f, mpas_mul(2.0f, c2))));
// bl_mynn_subroutines.F90:714
f2 = mpas_sub(mpas_mul(b1, mpas_add(g1, g2)), mpas_mul(mpas_mul(3.0f, a1), mpas_sub(1.0f, c2)));
// bl_mynn_subroutines.F90:715
rf1 = mpas_div(mpas_mul(b1, mpas_sub(g1, c1)), f1);
// bl_mynn_subroutines.F90:716
rf2 = mpas_div(mpas_mul(b1, g1), f2);
// bl_mynn_subroutines.F90:717
smc = mpas_div(mpas_mul(mpas_div(a1, a2), f1), f2);
// bl_mynn_subroutines.F90:718
shc = mpas_mul(mpas_mul(3.0f, a2), mpas_add(g1, g2));
// bl_mynn_subroutines.F90:720
ri1 = mpas_div(0.5f, smc);
// bl_mynn_subroutines.F90:721
ri2 = mpas_mul(rf1, smc);
// bl_mynn_subroutines.F90:722
ri3 = mpas_sub(mpas_mul(mpas_mul(4.0f, rf2), smc), mpas_mul(2.0f, ri2));
// bl_mynn_subroutines.F90:723
ri4 = mynn_square(ri2);
// bl_mynn_subroutines.F90:725
for (k=((kts) + (1)); (1) > 0 ? k <= (kte) : k >= (kte); k += (1)) {
// bl_mynn_subroutines.F90:726
dzk = mpas_mul(0.5f, mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:727
afk = mpas_div(dz(k), mpas_add(dz(k), dz(((k) - (1)))));
// bl_mynn_subroutines.F90:728
abk = mpas_sub(1.0f, afk);
// bl_mynn_subroutines.F90:729
duz = mpas_add(mynn_square(mpas_sub(u(k), u(((k) - (1))))), mynn_square(mpas_sub(v(k), v(((k) - (1))))));
// bl_mynn_subroutines.F90:730
duz = mpas_div(duz, mynn_square(dzk));
// bl_mynn_subroutines.F90:731
dtz = mpas_div(mpas_sub(thl(k), thl(((k) - (1)))), dzk);
// bl_mynn_subroutines.F90:732
dqz = mpas_div(mpas_sub(qw(k), qw(((k) - (1)))), dzk);
// bl_mynn_subroutines.F90:734
vtt = mpas_add(mpas_add(1.0f, mpas_mul(vt(k), abk)), mpas_mul(vt(((k) - (1))), afk));
// bl_mynn_subroutines.F90:735
vqq = mpas_add(mpas_add(tv0, mpas_mul(vq(k), abk)), mpas_mul(vq(((k) - (1))), afk));
// bl_mynn_subroutines.F90:736
dtq = mpas_add(mpas_mul(vtt, dtz), mpas_mul(vqq, dqz));
// bl_mynn_subroutines.F90:740
dtl(k) = dtz;
// bl_mynn_subroutines.F90:741
dqw(k) = dqz;
// bl_mynn_subroutines.F90:742
dtv(k) = dtq;
// bl_mynn_subroutines.F90:747
gm(k) = duz;
// bl_mynn_subroutines.F90:748
gh(k) = mpas_mul(mynn_neg(dtq), gtr);
// bl_mynn_subroutines.F90:751
if (k == kte && (kte-kts)%2 == 0) { ri = mpas_div(mynn_neg(gh(k)), mynn_max(((float)(1.0e-10f)), ((float)(duz)))); } else { ri = mpas_div(mynn_neg(gh(k)), mynn_max(((float)(duz)), ((float)(1.0e-10f)))); }
// bl_mynn_subroutines.F90:754
if (((ckmod) == (1))) {
// bl_mynn_subroutines.F90:755
if (k == kts + 2*((kte-kts-1)/2) + 1) { a2fac = mpas_div(1.f, mpas_add(1.f, mynn_max(((float)(ri)), ((float)(0.0f))))); } else { a2fac = mpas_div(1.f, mpas_add(1.f, mynn_max(((float)(0.0f)), ((float)(ri))))); }
// bl_mynn_subroutines.F90:756
} else {
// bl_mynn_subroutines.F90:757
a2fac = 1.f;
// bl_mynn_subroutines.F90:758
}
// bl_mynn_subroutines.F90:760
rfc = mpas_div(g1, mpas_add(g1, g2));
// bl_mynn_subroutines.F90:761
f1 = mpas_add(mpas_add(mpas_mul(b1, mpas_sub(g1, c1)), mpas_mul(mpas_mul(mpas_mul(mpas_mul(3.0f, a2), a2fac), mpas_sub(1.0f, c2)), mpas_sub(1.0f, c5))), mpas_mul(mpas_mul(2.0f, a1), mpas_sub(3.0f, mpas_mul(2.0f, c2))));
// bl_mynn_subroutines.F90:763
f2 = mpas_sub(mpas_mul(b1, mpas_add(g1, g2)), mpas_mul(mpas_mul(3.0f, a1), mpas_sub(1.0f, c2)));
// bl_mynn_subroutines.F90:764
rf1 = mpas_div(mpas_mul(b1, mpas_sub(g1, c1)), f1);
// bl_mynn_subroutines.F90:765
rf2 = mpas_div(mpas_mul(b1, g1), f2);
// bl_mynn_subroutines.F90:766
smc = mpas_div(mpas_mul(mpas_div(a1, mpas_mul(a2, a2fac)), f1), f2);
// bl_mynn_subroutines.F90:767
shc = mpas_mul(mpas_mul(3.0f, mpas_mul(a2, a2fac)), mpas_add(g1, g2));
// bl_mynn_subroutines.F90:769
ri1 = mpas_div(0.5f, smc);
// bl_mynn_subroutines.F90:770
ri2 = mpas_mul(rf1, smc);
// bl_mynn_subroutines.F90:771
ri3 = mpas_sub(mpas_mul(mpas_mul(4.0f, rf2), smc), mpas_mul(2.0f, ri2));
// bl_mynn_subroutines.F90:772
ri4 = mynn_square(ri2);
// bl_mynn_subroutines.F90:775
rf = mynn_min(((float)(mpas_mul(ri1, mpas_sub(mpas_add(ri, ri2), mynn_sqrt(mpas_add(mpas_sub(mynn_square(ri), mpas_mul(ri3, ri)), ri4)))))), ((float)(rfc)));
// bl_mynn_subroutines.F90:777
sh(k) = mpas_div(mpas_mul(shc, mpas_sub(rfc, rf)), mpas_sub(1.0f, rf));
// bl_mynn_subroutines.F90:778
sm(k) = mpas_mul(mpas_div(mpas_mul(smc, mpas_sub(rf1, rf)), mpas_sub(rf2, rf)), sh(k));
// bl_mynn_subroutines.F90:779
}

#undef dz
#undef u
#undef v
#undef thl
#undef thetav
#undef qw
#undef ql
#undef vt
#undef vq
#undef dtl
#undef dqw
#undef dtv
#undef gm
#undef gh
#undef sm
#undef sh
}
