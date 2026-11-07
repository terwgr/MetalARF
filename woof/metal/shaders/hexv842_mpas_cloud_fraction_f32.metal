#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_cloud_fraction_f32_bindings.metalh"
#pragma clang fp contract(off)

// MPAS cloudiness_from_MPAS and cld_fraction, [level,cell] public layout.
kernel void hexv842_mpas_cloud_fraction_f32(
    constant Hexv842MpasCloudFractionF32Resources &r [[buffer(0)]],
    constant Hexv842MpasCloudFractionF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int i=int(tid), plane=p.nlev*p.ncells;
    if (i>=plane) return;
    const int col=i%p.ncells;
    r.qvrad[i]=r.qv[i]; r.qcrad[i]=r.qc[i]; r.qirad[i]=r.qi[i]; r.qsrad[i]=r.qs[i];
    if (i < p.ncells) {
        r.dx[col]=mpas_mul(mpas_div(p.len_disp,gfk_pow(r.density[col],0.25f)),0.001f);
        r.land[col]=r.xland[col];
    }
    const float temp=r.temp[i], pressure=r.pressure[i];
    const float esw=mpas_mul(1000.f*.61078f,gfk_exp(mpas_div(mpas_mul(17.2693882f,
        mpas_sub(temp,273.15f)),mpas_sub(temp,35.86f))));
    const float esi=mpas_mul(1000.f*.61078f,gfk_exp(mpas_div(mpas_mul(21.8745584f,
        mpas_sub(temp,273.15f)),mpas_sub(temp,7.66f))));
    const float ep2=287.f/461.6f;
    const float sw=mpas_div(mpas_mul(ep2,esw),mpas_sub(pressure,esw));
    const float si=mpas_div(mpas_mul(ep2,esi),mpas_sub(pressure,esi));
    const float qcld=mpas_add(mpas_add(r.qc[i],r.qi[i]),r.qs[i]);
    const float w=qcld<1.e-12f ? 0.f : mpas_div(mpas_add(r.qi[i],r.qs[i]),qcld);
    const float qvs=mpas_add(mpas_mul(mpas_sub(1.f,w),sw),mpas_mul(w,si));
    float rh=mpas_div(r.qv[i],qvs);
    float out;
    if (qcld<1.e-12f) out=0.f;
    else if (rh>=1.f) out=1.f;
    else {
        const float subsat=max(1.e-10f,mpas_sub(qvs,r.qv[i]));
        const float denom=gfk_pow(subsat,.49f);
        const float arg=max(-6.9f,mpas_div(mpas_mul(-100.f,qcld),denom));
        rh=max(1.e-10f,rh);
        out=mpas_mul(gfk_pow(rh,.25f),mpas_sub(1.f,gfk_exp(arg)));
        if (out<.01f) out=0.f;
    }
    r.cldfrac[i]=out;
    r.diag_cldfrac[i]=out;
}
