#include <metal_stdlib>
#include "exact_gfk_math.metalh"
#include "wrf_exact_noah_forcing_bindings.metalh"
#include "wrf_exact_noah_rain_clear_bindings.metalh"
#include "wrf_exact_noah_sfcdiags_bindings.metalh"
using namespace metal;
#define A(x) (reinterpret_cast<device Exact32*>(r.x))
#define C(x) (reinterpret_cast<device const Exact32*>(r.x))
kernel void wrf_exact_noah_forcing(constant WrfExactNoahForcingResources &r [[buffer(0)]], constant WrfExactNoahForcingParams &p [[buffer(1)]], uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.ncol)) return;
    A(sfcprs_a)[c] = Exact32(0.5f) * (C(p_if0_a)[c] + C(p_if1_a)[c]);
    Exact32 t = C(temperature0_a)[c];
    A(sfctmp_a)[c] = t;
    A(qv1_a)[c] = C(qv0_a)[c];
    A(dz8w1_a)[c] = C(dz0_a)[c];
    A(rib_a)[c] = C(br_a)[c];
    A(sr_a)[c] = p.use_scheme_sr ? C(sr_scheme_a)[c] :
        (soft32_less_equal(t.bits, Exact32(p.freezing).bits) ? Exact32(1.0f) : Exact32(0.0f));
    A(rainbl_a)[c] = A(rainbl_a)[c] + C(pending_a)[c];
}
kernel void wrf_exact_noah_rain_clear(constant WrfExactNoahRainClearResources &r [[buffer(0)]], constant WrfExactNoahRainClearParams &p [[buffer(1)]], uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.ncol)) return;
    A(rainbl_a)[c] = Exact32(0.0f);
    A(pending_a)[c] = Exact32(0.0f);
}
kernel void wrf_exact_noah_sfcdiags(constant WrfExactNoahSfcdiagsResources &r [[buffer(0)]], constant WrfExactNoahSfcdiagsParams &p [[buffer(1)]], uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.ncol)) return;
    Exact32 ps=C(psfc_a)[c], ts=C(tsk_a)[c];
    Exact32 rho=ps/(Exact32(p.rd)*ts), cq=C(cqs2_a)[c], ch=C(chs2_a)[c];
    bool qa=cq>=Exact32(p.active_floor), ta=ch>=Exact32(p.active_floor);
    Exact32 sq=qa?cq:Exact32(1.0f), st=ta?ch:Exact32(1.0f);
    Exact32 qs=C(qsfc_a)[c], inv=qs-C(qfx_a)[c]/(rho*sq);
    Exact32 diag=qa?inv:qs;
    A(q2_a)[c]=diag>Exact32(0.0f)?diag:C(qv1_a)[c];
    Exact32 heated=ts-C(hfx_a)[c]/((rho*Exact32(p.cp_air))*st);
    Exact32 temperature=ta?heated:ts;
    A(t2_a)[c]=temperature;
    A(th2_a)[c]=temperature*exact_gfk_pow(Exact32(p.p0)/ps,Exact32(p.rcp));
}
