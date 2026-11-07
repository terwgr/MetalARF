// Private native MPAS aerosol input and snapshot packing.
#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_aerosol_from_f32_bindings.metalh"
kernel void hexv842_mpas_thompson_aerosol_from_f32(
    constant Hexv842MpasThompsonAerosolFromF32Resources &r [[buffer(0)]],
    constant Hexv842MpasThompsonAerosolFromF32Params &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    uint n=uint(p.nz)*uint(p.columns); if(gid>=n) return;
    r.slab[gid]=r.nc[gid]; r.slab[n+gid]=r.nifa[gid]; r.slab[2*n+gid]=r.nwfa[gid];
    r.before_nc[gid]=r.nc[gid]; r.before_nifa[gid]=r.nifa[gid]; r.before_nwfa[gid]=r.nwfa[gid];
    r.prepared_nc[gid]=r.nc[gid]; r.prepared_nifa[gid]=r.nifa[gid]; r.prepared_nwfa[gid]=r.nwfa[gid];
    if(gid<uint(p.columns)) { r.slab[3*n+gid]=r.nifa2d[gid]; r.slab[3*n+uint(p.columns)+gid]=r.nwfa2d[gid]; }
}
