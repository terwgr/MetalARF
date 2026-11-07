#include <metal_stdlib>
#include "tke_budget_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline ulong budget_double(float value) {
    return soft64_from_f32_bits(as_type<uint>(value));
}

// Match CUDA's 256-lane strided accumulation and binary reduction tree.
// All sums and derived terms stay binary64, including gradual underflow.
kernel void tke_budget_accumulate(
    constant TkeBudgetAccumulateResources &r [[buffer(0)]],
    constant TkeBudgetAccumulateParams &p [[buffer(1)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    if (group >= uint((p.nfield + 2) * p.nz)) return;
    const uint k=group%uint(p.nz), term=group/uint(p.nz);
    const uint plane=uint(p.ny)*uint(p.nx);
    ulong sum=0ul;
    for(uint column=lane;column<plane;column+=256u) {
        const uint index=k*plane+column;
        ulong value;
        if(term<uint(p.nfield)) {
            value=budget_double(r.terms[(term*uint(p.nz))*plane+index]);
        } else {
            const ulong chm0=soft64_fma(budget_double(r.c1[k]),
                budget_double(r.mu0[column]),budget_double(r.c2[k]));
            const ulong chm=soft64_fma(budget_double(r.c1[k]),
                budget_double(r.mu[column]),budget_double(r.c2[k]));
            if(term==uint(p.nfield)) {
                value=soft64_mul(soft64_fma(chm,budget_double(r.tke[index]),
                    soft64_mul(chm0,budget_double(r.tke0[index]))^0x8000000000000000ul),
                    budget_double(p.rdt));
            } else {
                value=soft64_mul(soft64_mul(chm,soft64_sub(budget_double(r.tke[index]),
                    budget_double(r.raw[index]))),budget_double(p.rdt));
            }
        }
        sum=soft64_add(sum,value);
    }
    threadgroup ulong partial[256];
    partial[lane]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128u;stride>0u;stride>>=1u) {
        if(lane<stride) partial[lane]=soft64_add(partial[lane],partial[lane+stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0u) {
        r.acc[group]=soft64_add(r.acc[group],partial[0]);
        if(group==0u)r.steps[0]=soft64_add(r.steps[0],0x3ff0000000000000ul);
    }
}
