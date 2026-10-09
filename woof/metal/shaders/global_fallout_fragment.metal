// Sequential FP64 column fallout; each level receives the flux above it.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
#include "global_fallout_fragment_bindings.metalh"

using namespace metal;
static inline ulong gfmax(ulong a,ulong b){return soft64_greater(a,b)?a:b;}
kernel void global_fallout_fragment(
    constant GlobalFalloutFragmentResources &r [[buffer(0)]],
    constant GlobalFalloutFragmentParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if(column>=uint(p.ncolumn))return;
    const ulong gravity=r.gravity[0], fall_dt=r.fall_speed_dt[0];
    const ulong min_dz=0x4034000000000000ul; // 20 m
    const ulong min_mass=0x3d719799812dea11ul; // 1e-12
    ulong incoming=0ul, precip=0ul;
    for(int level=0;level<p.nlevel;++level) {
        const uint index=uint(level)*uint(p.ncolumn)+column;
        const ulong mass=soft64_div(r.dp[index],gravity);
        ulong z=soft64_div(r.geopotential[index],gravity);
        ulong dz;
        if(level+1<p.nlevel) {
            const uint next=index+uint(p.ncolumn);
            ulong znext=soft64_div(r.geopotential[next],gravity);
            dz=gfmax(soft64_abs_bits(soft64_sub(z,znext)),min_dz);
        } else {
            const uint previous=index-uint(p.ncolumn);
            ulong zprevious=soft64_div(r.geopotential[previous],gravity);
            dz=gfmax(soft64_abs_bits(soft64_sub(zprevious,z)),min_dz);
        }
        ulong fraction=soft64_sub(0x3ff0000000000000ul,
            soft64_plm_exp(soft64_negate_bits(soft64_div(fall_dt,dz))));
        if(soft64_less(fraction,0ul))fraction=0ul;
        if(soft64_greater(fraction,0x3ff0000000000000ul))fraction=0x3ff0000000000000ul;
        ulong current=soft64_add(r.species[index],incoming);
        ulong removed=soft64_mul(current,fraction);
        r.species_out[index]=soft64_sub(current,removed);
        ulong removed_mass=soft64_mul(removed,mass);
        if(level+1<p.nlevel) {
            const uint next=index+uint(p.ncolumn);
            ulong next_mass=soft64_div(r.dp[next],gravity);
            incoming=soft64_div(removed_mass,gfmax(next_mass,min_mass));
        } else {
            precip=soft64_add(precip,removed_mass);
            incoming=0ul;
        }
    }
    r.precip[column]=precip;
}
