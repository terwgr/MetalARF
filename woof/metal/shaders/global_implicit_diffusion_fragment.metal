// Batched per-column Thomas solve for Global vertical turbulence diffusion.
#include <metal_stdlib>
#include "soft_binary64.metalh"
#include "global_implicit_diffusion_fragment_bindings.metalh"
using namespace metal;
static inline ulong gdmax(ulong a,ulong b){return soft64_greater(a,b)?a:b;}
static inline ulong gdabs(ulong x){return soft64_abs_bits(x);}
kernel void global_implicit_diffusion_fragment(
    constant GlobalImplicitDiffusionFragmentResources &r [[buffer(0)]],
    constant GlobalImplicitDiffusionFragmentParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    if(column>=uint(p.ncolumn))return;
    const ulong zero=0ul,one=0x3ff0000000000000ul,half_bits=0x3fe0000000000000ul;
    const uint n=uint(p.nlevel), nc=uint(p.ncolumn), plane=n*nc;
    thread ulong coupling[127];
    thread ulong a[128],b[128],c[128],cp[128],dp[128],solution[128];
    for(uint k=0;k+1<n;++k){
        ulong dz=gdmax(gdabs(soft64_sub(r.height[k*nc+column],r.height[(k+1)*nc+column])),one);
        ulong ki=soft64_mul(half_bits,soft64_add(r.diffusivity[k*nc+column],r.diffusivity[(k+1)*nc+column]));
        ulong mi=soft64_mul(half_bits,soft64_add(r.dp[k*nc+column],r.dp[(k+1)*nc+column]));
        coupling[k]=soft64_div(soft64_mul(r.dt[0],soft64_mul(ki,mi)),soft64_mul(dz,dz));
    }
    for(uint k=0;k<n;++k){
        a[k]=k==0?zero:soft64_negate_bits(soft64_div(coupling[k-1],r.dp[k*nc+column]));
        c[k]=k+1==n?zero:soft64_negate_bits(soft64_div(coupling[k],r.dp[k*nc+column]));
        ulong diag=one;
        if(k>0)diag=soft64_add(diag,soft64_div(coupling[k-1],r.dp[k*nc+column]));
        if(k+1<n)diag=soft64_add(diag,soft64_div(coupling[k],r.dp[k*nc+column]));
        b[k]=diag;
    }
    for(uint field=0;field<uint(p.nfield);++field){
        const uint base=field*plane;
        cp[0]=soft64_div(c[0],b[0]);
        dp[0]=soft64_div(r.fields[base+column],b[0]);
        for(uint k=1;k<n;++k){
            ulong den=soft64_sub(b[k],soft64_mul(a[k],cp[k-1]));
            cp[k]=k+1<n?soft64_div(c[k],den):zero;
            dp[k]=soft64_div(soft64_sub(r.fields[base+k*nc+column],soft64_mul(a[k],dp[k-1])),den);
        }
        solution[n-1]=dp[n-1];
        for(int k=int(n)-2;k>=0;--k)
            solution[k]=soft64_sub(dp[k],soft64_mul(cp[k],solution[k+1]));
        for(uint k=0;k<n;++k)r.updated_fields[base+k*nc+column]=solution[k];
    }
}
