// Private native MPAS aerosol writeback and dynamics-time tendencies.
#include <metal_stdlib>
#include "exact_binary32.metalh"
using namespace metal;
#include "hexv842_mpas_thompson_aerosol_to_f32_bindings.metalh"
kernel void hexv842_mpas_thompson_aerosol_to_f32(
    constant Hexv842MpasThompsonAerosolToF32Resources &r [[buffer(0)]],
    constant Hexv842MpasThompsonAerosolToF32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    uint n=uint(p.nz)*uint(p.columns); if(gid>=n) return;
    Exact32 nc=reinterpret_cast<device const Exact32*>(r.slab)[gid];
    Exact32 ni=reinterpret_cast<device const Exact32*>(r.slab)[n+gid];
    Exact32 nw=reinterpret_cast<device const Exact32*>(r.slab)[2*n+gid];
    Exact32 dt=Exact32(p.dt_dyn);
    device Exact32 *tc=reinterpret_cast<device Exact32*>(r.before_nc);
    device Exact32 *ti=reinterpret_cast<device Exact32*>(r.before_nifa);
    device Exact32 *tw=reinterpret_cast<device Exact32*>(r.before_nwfa);
    tc[gid]=(nc-tc[gid])/dt; ti[gid]=(ni-ti[gid])/dt; tw[gid]=(nw-tw[gid])/dt;
    reinterpret_cast<device Exact32*>(r.nc)[gid]=nc;
    reinterpret_cast<device Exact32*>(r.nifa)[gid]=ni;
    reinterpret_cast<device Exact32*>(r.nwfa)[gid]=nw;
    reinterpret_cast<device Exact32*>(r.prepared_nc)[gid]=nc;
    reinterpret_cast<device Exact32*>(r.prepared_nifa)[gid]=ni;
    reinterpret_cast<device Exact32*>(r.prepared_nwfa)[gid]=nw;
}
