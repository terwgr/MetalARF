#include <metal_stdlib>
#include "da_hotstart_pointwise_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;

// NumPy clip preserves an input NaN and leaves equal signed-zero words alone.
static inline uint hs_clip(uint x, uint lo, uint hi) {
    if (soft32_is_nan(x)) return x;
    if (soft32_less(x, lo)) x = lo;
    if (soft32_greater(x, hi)) x = hi;
    return x;
}
static inline uint hs_max(uint a, uint b) {
    if (soft32_is_nan(a)) return soft32_quiet_nan(a);
    if (soft32_is_nan(b)) return soft32_quiet_nan(b);
    if (soft32_is_zero(a) && soft32_is_zero(b)) return a & b;
    return soft32_less(a,b) ? b : a;
}

// Inputs qsat and ramp are computed by the existing host policy. The status
// bit requests host fallback for unsupported non-finite ancillary inputs.
kernel void da_hotstart_pointwise(constant DaHotstartPointwiseResources& r [[buffer(0)]],
                                  constant DaHotstartPointwiseParams& p [[buffer(1)]],
                                  uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    const uint zero=0u;
    const uint obs=as_type<uint>(r.obs[i]), model=as_type<uint>(r.model[i]);
    const uint ramp=as_type<uint>(r.ramp[i]), qsat=as_type<uint>(r.qsat[i]);
    const uint qv=as_type<uint>(r.qv[i]);
    const uint at=p.ledger ? as_type<uint>(r.applied_theta[i]) : zero;
    const uint aq=p.ledger ? as_type<uint>(r.applied_qv[i]) : zero;
    uint flags=0u;
    if (!soft32_is_finite(ramp) || !soft32_is_finite(qv)
        || (p.adjust_moisture && !soft32_is_finite(qsat))
        || (p.ledger && (!soft32_is_finite(at) || !soft32_is_finite(aq)))) flags |= 4u;
    const bool valid=r.mask[i]!=0u;
    const uint cap=as_type<uint>(p.cap);
    const uint z=soft32_is_nan(obs) ? soft32_quiet_nan(obs)
        : (soft32_greater(obs,cap) ? cap : obs);
    const bool inside=soft32_greater(ramp,zero);
    const bool echo=valid && soft32_greater_equal(z,as_type<uint>(p.echo));
    const uint deficit=soft32_sub(z,model);
    const bool warm=echo && soft32_greater_equal(deficit,as_type<uint>(p.deficit)) && inside;
    uint theta=zero, moisture=zero;
    if (warm) {
        theta=hs_clip(soft32_mul(soft32_mul(as_type<uint>(p.theta_per_db),deficit),ramp),
                      zero,as_type<uint>(p.theta_max));
        if (p.adjust_moisture) {
            const uint shortfall=hs_clip(soft32_sub(soft32_mul(as_type<uint>(p.rh),qsat),qv),
                                           zero,as_type<uint>(p.qv_max));
            moisture=hs_clip(soft32_mul(shortfall,ramp),zero,as_type<uint>(p.qv_max));
        }
    }
    const bool clear=p.clear_air && valid && inside
        && soft32_less_equal(z,as_type<uint>(p.clear_obs))
        && soft32_greater_equal(model,as_type<uint>(p.clear_model));
    if (clear) {
        const uint excess=soft32_sub(model,z);
        const uint cool=hs_clip(soft32_mul(soft32_mul(as_type<uint>(p.clear_theta_per_db),excess),ramp),
                                zero,as_type<uint>(p.theta_min));
        theta=soft32_negate_bits(cool);
        if (p.adjust_moisture) {
            const uint surplus=hs_clip(soft32_sub(qv,soft32_mul(as_type<uint>(p.clear_rh),qsat)),
                                          zero,as_type<uint>(p.qv_min));
            const uint dry=hs_clip(soft32_mul(surplus,ramp),zero,as_type<uint>(p.qv_min));
            moisture=soft32_negate_bits(dry);
        }
    }
    uint lo_t=soft32_negate_bits(as_type<uint>(p.theta_min)), hi_t=as_type<uint>(p.theta_max);
    uint lo_q=soft32_negate_bits(as_type<uint>(p.qv_min)), hi_q=as_type<uint>(p.qv_max);
    if (p.ledger) {
        lo_t=soft32_sub(lo_t,at); hi_t=soft32_sub(hi_t,at);
        lo_q=soft32_sub(lo_q,aq); hi_q=soft32_sub(hi_q,aq);
    }
    theta=hs_clip(theta,lo_t,hi_t);
    moisture=hs_max(hs_clip(moisture,lo_q,hi_q),soft32_negate_bits(qv));
    flags |= uint(warm) | (uint(clear)<<1);
    r.theta[i]=as_type<float>(theta); r.moisture[i]=as_type<float>(moisture); r.flags[i]=flags;
}
