#include "hexv842_nt_prefix_math_bindings.metalh"
#include <metal_stdlib>
#include "soft_gfk_pow.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_exp.metalh"
using namespace metal;
#pragma clang fp contract(off)

static float add32(float a,float b) { return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b))); }
static float sub32(float a,float b) { return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b))); }
static float mul32(float a,float b) { return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b))); }
static float div32(float a,float b) { return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b))); }
static float sqrt32(float a) { return as_type<float>(soft32_sqrt(as_type<uint>(a))); }
static float exp_pinned(float x) { return as_type<float>(soft_gfk_exp(as_type<uint>(x))); }
static float log_pinned(float x) { return as_type<float>(soft_gfk_log(as_type<uint>(x))); }
static float pow_pinned(float x, float y) { return as_type<float>(soft_gfk_pow_bits(as_type<uint>(x),as_type<uint>(y))); }
kernel void hexv842_nt_prefix_math(constant Hexv842NtPrefixMathResources &r [[buffer(0)]], constant Hexv842NtPrefixMathParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    device const float *density=r.density;
    device const float *level=r.level;
    device float *out=r.out;

    if (i >= uint(p.ncol)) return;
    int n=p.ncol, nz=p.nz, s=n*nz;
    float dx = div32(p.len_disp, pow_pinned(density[i], 0.25f));
    out[i] = dx;
    if (dx < 15000.0f) {
        float ratio = div32(15000.0f, dx);
        float base = add32(1.06133f, log_pinned(ratio));
        float sf = mul32(mul32(base, base), base);
        out[n+i]=sf; out[2*n+i]=sqrt32(sf);
    } else { out[n+i]=add32(1.0f,mul32(1.33e-5f,dx)); out[2*n+i]=1.0f; }
    float c2es=div32(mul32(610.78f,p.rd),p.rv);
    float vtmpc1=sub32(div32(p.rv,p.rd),1.0f);
    for (int k=0;k<nz;++k) {
        int j=k*n+i, o=(nz-1-k)*n+i;
        float t=level[j], pressure=level[s+j];
        float clamped=max(250.16f,min(273.16f,t));
        float ratio=div32(sub32(clamped,250.16f),sub32(273.16f,250.16f));
        float a=min(1.0f,mul32(ratio,ratio));
        float el=exp_pinned(div32(mul32(17.2693882f,sub32(t,273.16f)),sub32(t,35.86f)));
        float ei=exp_pinned(div32(mul32(21.875f,sub32(t,273.16f)),sub32(t,7.66f)));
        float zqs=div32(mul32(c2es,add32(mul32(a,el),mul32(sub32(1.0f,a),ei))),pressure);
        zqs=min(0.5f,zqs);
        float zcor=div32(1.0f,sub32(1.0f,mul32(vtmpc1,zqs)));
        out[3*n+o]=mul32(zqs,zcor);
    }
}
