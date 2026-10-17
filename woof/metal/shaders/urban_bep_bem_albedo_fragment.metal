#include <metal_stdlib>
#include "urban_bep_bem_albedo_fragment_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_pow.metalh"
using namespace metal;

// module_sf_bep_bem.F bb_albwindow (:5397-5436), using the BEM-specific
// integrand bb_foncs (:5441-5453).
#pragma clang fp contract(off)
static inline uint soft_urban_bem_sinf(uint xbits) {
    return as_type<uint>(precise::sin(as_type<float>(xbits)));
}
static inline float ba_add(float a,float b) { return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b))); }
static inline float ba_sub(float a,float b) { return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b))); }
static inline float ba_mul(float a,float b) { return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b))); }
static inline float ba_div(float a,float b) { return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b))); }
static inline float ba_pow(float a,float b) { return as_type<float>(soft_gfk_pow_bits(as_type<uint>(a),as_type<uint>(b))); }
static inline float ba_abs(float a) { return as_type<float>(as_type<uint>(a)&0x7fffffffu); }

static inline float urban_bem_foncs(float x,float a,float b,float c,
                                    float alfa,float delt,float gam) {
    constexpr float pi=3.14159265358979323846f;
    const float terms=ba_add(ba_add(
        ba_div(ba_mul(a,ba_pow(x,alfa)),ba_pow(pi,alfa)),
        ba_div(ba_mul(b,ba_pow(x,delt)),ba_pow(pi,delt))),
        ba_div(ba_mul(c,ba_pow(x,gam)),ba_pow(pi,gam)));
    return ba_mul(terms,as_type<float>(soft_urban_bem_sinf(as_type<uint>(x))));
}

// The integral is refined one subinterval at a time until two successive
// right-endpoint sums agree (bb_albwindow). One serial thread per case made
// that a multi-second dispatch that trips the GPU watchdog, so a threadgroup
// evaluates each sum's terms in parallel and lane 0 adds them in source
// order: the same operations and summation order as the serial loop. The
// refinement is bounded; error 12 reports a sum that never converged.
constant int ALB_LANES = 256;
constant int ALB_NMAX = 2048;

kernel void urban_bep_bem_albedo_fragment(
    constant UrbanBepBemAlbedoFragmentResources &r [[buffer(0)]],
    constant UrbanBepBemAlbedoFragmentParams &p [[buffer(1)]],
    uint lane [[thread_position_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup float terms[ALB_LANES];
    threadgroup int done;
    if (group>=uint(p.ncase)) return;
    const uint tid=group;
    constexpr float pi=3.14159265358979323846f;
    const float n1=1.0f,n2=1.5f,epsilon=1.0e-7f;
    const int pn=r.p_num[tid], qn=r.q_num[tid];
    if(qn==0) { if(lane==0u) r.error[tid]=11; return; }
    const float g0=ba_div(ba_mul(ba_mul(4.0f,n1),n2),
        ba_mul(ba_add(n1,n2),ba_add(n1,n2)));
    const float a=8.0f,b=ba_div(0.25f,float(qn));
    const float c=ba_sub(ba_sub(1.0f,a),b);
    const float alfa=ba_add(5.2f,ba_mul(0.7f,float(qn)));
    const float delt=2.0f;
    const float gam=ba_add(ba_add(5.26f,ba_mul(0.06f,float(pn))),
        ba_mul(ba_add(0.73f,ba_mul(0.04f,float(pn))),float(qn)));
    float previous=0.0f;
    for(int n=1;n<=ALB_NMAX;++n) {
        float current=0.0f;
        for(int k0=1;k0<=n;k0+=ALB_LANES) {
            const int k=k0+int(lane);
            if(k<=n) {
                const float x=ba_div(ba_mul(pi,float(k)),float(n));
                terms[lane]=ba_mul(ba_div(pi,float(n)),
                    urban_bem_foncs(x,a,b,c,alfa,delt,gam));
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if(lane==0u) {
                const int count=min(ALB_LANES,n-k0+1);
                for(int j=0;j<count;++j) current=ba_add(current,terms[j]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if(lane==0u) {
            done=0;
            if(n>1&&ba_abs(ba_sub(previous,current))<epsilon) {
                r.albwin[tid]=ba_add(ba_sub(1.0f,g0),ba_mul(ba_div(g0,2.0f),current));
                r.error[tid]=0;
                done=1;
            }
            previous=current;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(done!=0) return;
    }
    if(lane==0u) r.error[tid]=12;
}
