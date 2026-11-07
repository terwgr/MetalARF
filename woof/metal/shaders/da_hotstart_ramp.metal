#include <metal_stdlib>
#include "da_hotstart_ramp_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;

// NumPy 2.5.4 FLOAT_sin SIMD path, restricted to finite angles in [0, pi].
// See numpy/_core/src/umath/loops_trigonometric.dispatch.cpp. Preserve the
// fused range reduction and polynomial operations used by that implementation.
static inline uint hotstart_numpy_sine(uint a){
    float x=as_type<float>(a);
    uint q=soft32_mul(a,as_type<uint>(0x1.45f306p-1f));
    q=soft32_sub(soft32_add(q,as_type<uint>(0x1.800000p+23f)),as_type<uint>(0x1.800000p+23f));
    float y=as_type<float>(q);
    float z=fma(y,-0x1.921fb0p+00f,x);
    z=fma(y,-0x1.5110b4p-22f,z);
    z=fma(y,-0x1.846988p-48f,z);
    float z2=as_type<float>(soft32_mul(as_type<uint>(z),as_type<uint>(z)));
    float c=fma(0x1.98e616p-16f,z2,-0x1.6c06dcp-10f);
    c=fma(c,z2,0x1.55553cp-05f);
    c=fma(c,z2,-0x1.000000p-01f);
    c=fma(c,z2,0x1.000000p+00f);
    float s=fma(0x1.7d3bbcp-19f,z2,-0x1.a06bbap-13f);
    s=fma(s,z2,0x1.11119ap-07f);
    s=fma(s,z2,-0x1.555556p-03f);
    s=fma(s,z2,0.0f);
    s=fma(s,z,z);
    uint quadrant=uint(y);
    uint v=as_type<uint>((quadrant & 1u)?c:s);
    return (quadrant & 2u)?soft32_sub(0u,v):v;
}

kernel void da_hotstart_ramp(constant DaHotstartRampResources& r [[buffer(0)]],
                             constant DaHotstartRampParams& p [[buffer(1)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    const uint h=as_type<uint>(r.height[i]);
    const uint base=as_type<uint>(p.base), top=as_type<uint>(p.top);
    const uint zero=0u, one=as_type<uint>(1.0f);
    const uint width=soft32_sub(top,base);
    if (!soft32_is_finite(h) || !soft32_is_finite(base) || !soft32_is_finite(top)
        || !soft32_is_finite(width) || !soft32_greater(width,zero)
        || p.shape < 0 || p.shape > 2) {
        r.ramp[i]=0.0f;
        r.flags[i]=1u;
        return;
    }
    const uint x=soft32_div(soft32_sub(h,base),width);
    const bool inside=soft32_greater_equal(x,zero) && soft32_less_equal(x,one);
    if (!inside) {
        r.ramp[i]=0.0f;
        r.flags[i]=0u;
        return;
    }
    uint value=one;
    if (p.shape == 0) {
        const uint angle=soft32_mul(as_type<uint>(3.141592653589793f),x);
        // Native FMA is used only in the verified range.
        if ((angle & 0x7fffffffu) != 0u && soft32_less(angle,as_type<uint>(0x1p-60f))) {
            r.ramp[i]=0.0f;
            r.flags[i]=1u;
            return;
        }
        const uint s=hotstart_numpy_sine(angle);
        value=soft32_mul(s,s);
    } else if (p.shape == 1) {
        const uint v=soft32_sub(soft32_mul(as_type<uint>(2.0f),x),one);
        value=soft32_sub(one,v & 0x7fffffffu);
    }
    r.ramp[i]=as_type<float>(value);
    r.flags[i]=0u;
}
