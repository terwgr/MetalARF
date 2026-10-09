#include <metal_stdlib>
#include "global_vertical_scalar_flux_fragment_bindings.metalh"
#include "soft_binary64.metalh"
using namespace metal;

// needs integration: source proposal for the NumPy MUSCL vertical scalar
// flux divergence in MoistDycore._vertical_scalar_flux_divergence.
#pragma clang fp contract(off)
static inline bool gvs_negative(ulong x) {
    return soft64_sign(x) && !soft64_is_zero(x);
}

static inline ulong gvs_gradient(
    constant GlobalVerticalScalarFluxFragmentResources &r,
    uint tracer, uint level, uint col, uint nlevel, uint ncolumn) {
    if(level==0u || level+1u>=nlevel) return 0ul;
    const ulong base=ulong(tracer)*ulong(nlevel)*ulong(ncolumn)+ulong(col);
    const ulong here=base+ulong(level)*ulong(ncolumn);
    const ulong above=here-ulong(ncolumn), below=here+ulong(ncolumn);
    const ulong ga=soft64_div(soft64_sub(r.scalar[here],r.scalar[above]),
        soft64_sub(r.pressure_full[ulong(level)*ulong(ncolumn)+col],
                   r.pressure_full[ulong(level-1u)*ulong(ncolumn)+col]));
    const ulong gb=soft64_div(soft64_sub(r.scalar[below],r.scalar[here]),
        soft64_sub(r.pressure_full[ulong(level+1u)*ulong(ncolumn)+col],
                   r.pressure_full[ulong(level)*ulong(ncolumn)+col]));
    const ulong product=soft64_mul(ga,gb);
    if(gvs_negative(product) || soft64_is_zero(product)) return 0ul;
    return soft64_div(soft64_mul(0x4000000000000000ul,product),
                      soft64_add(ga,gb));
}

static inline ulong gvs_face(
    constant GlobalVerticalScalarFluxFragmentResources &r,
    uint tracer, uint interface_level, uint col,
    uint nlevel, uint ncolumn) {
    const uint upper=interface_level-1u, lower=interface_level;
    const ulong base=ulong(tracer)*ulong(nlevel)*ulong(ncolumn)+ulong(col);
    const ulong ua=base+ulong(upper)*ulong(ncolumn), lo=base+ulong(lower)*ulong(ncolumn);
    const ulong pu=r.pressure_full[ulong(upper)*ulong(ncolumn)+col];
    const ulong pl=r.pressure_full[ulong(lower)*ulong(ncolumn)+col];
    const ulong ph=r.pressure_half[ulong(interface_level)*ulong(ncolumn)+col];
    const ulong above=soft64_add(r.scalar[ua],soft64_mul(
        gvs_gradient(r,tracer,upper,col,nlevel,ncolumn),soft64_sub(ph,pu)));
    const ulong below=soft64_add(r.scalar[lo],soft64_mul(
        gvs_gradient(r,tracer,lower,col,nlevel,ncolumn),soft64_sub(ph,pl)));
    const ulong lo_value=gvs_negative(r.omega[ulong(interface_level)*ulong(ncolumn)+col])
        ? below : above;
    const ulong minimum=gvs_negative(soft64_sub(r.scalar[ua],r.scalar[lo]))
        ? r.scalar[ua] : r.scalar[lo];
    const ulong maximum=gvs_negative(soft64_sub(r.scalar[ua],r.scalar[lo]))
        ? r.scalar[lo] : r.scalar[ua];
    if(gvs_negative(soft64_sub(lo_value,minimum))) return minimum;
    if(gvs_negative(soft64_sub(maximum,lo_value))) return maximum;
    return lo_value;
}

kernel void global_vertical_scalar_flux_fragment(
    constant GlobalVerticalScalarFluxFragmentResources &r [[buffer(0)]],
    constant GlobalVerticalScalarFluxFragmentParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.count)) return;
    const uint nlev=uint(p.nlevel), ncolumn=uint(p.ncolumn);
    const uint tracer=tid/(nlev*ncolumn);
    const uint within=tid%(nlev*ncolumn);
    const uint level=within/ncolumn, col=within%ncolumn;
    ulong lower_flux=0ul, upper_flux=0ul;
    if(level>0u) lower_flux=soft64_mul(
        r.omega[ulong(level)*ulong(ncolumn)+col],
        gvs_face(r,tracer,level,col,nlev,ncolumn));
    if(level+1u<nlev) upper_flux=soft64_mul(
        r.omega[ulong(level+1u)*ulong(ncolumn)+col],
        gvs_face(r,tracer,level+1u,col,nlev,ncolumn));
    r.divergence[tid]=soft64_sub(upper_flux,lower_flux);
}
