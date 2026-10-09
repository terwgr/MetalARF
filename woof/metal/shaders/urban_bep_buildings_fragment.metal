#include <metal_stdlib>
#include "urban_bep_buildings_fragment_bindings.metalh"
using namespace metal;

// needs integration: compose after urban_bep_flux_helpers_fragment.metal and
// generate the typed binding from the submit wrapper. One thread owns one
// urban column, matching the CUDA lane's serial direction/height order.
#pragma clang fp contract(off)

kernel void urban_bep_column_buildings_fragment(
    constant UrbanBepBuildingsFragmentResources &r [[buffer(0)]],
    constant UrbanBepBuildingsFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const int nz=p.nzurban;
    const ulong level_count=ulong(nz);
    const ulong levels=ulong(nz+1);
    for (int direction=0; direction<p.nd; ++direction) {
        const ulong dir=ulong(col)*2ul+ulong(direction);
        const ulong flat=ulong(col)*level_count;
        const ulong out_flat=ulong(col)*648ul;
        const ulong flat_out=out_flat+504ul+ulong(direction)*18ul;
        const ulong height=flat+ulong(0);
        float uh=0.0f,vh=0.0f,th=0.0f,eh=0.0f;
        urban_bep_flux_flat(5.0f,r.z0[dir*levels],r.ua_u[height],r.va_u[height],
            r.pt_u[height],r.pt0_u[height],r.ptg[dir],uh,vh,th,eh);
        r.column_building_x[flat_out]=uh;
        r.column_building_x[flat_out+36ul]=vh;
        r.column_building_x[flat_out+72ul]=th;
        r.column_building_x[flat_out+108ul]=eh;
        for (int k=1;k<nz;++k) {
            const ulong idx=flat+ulong(k);
            uh=vh=th=eh=0.0f;
            if (r.ss[ulong(col)*levels+ulong(k+1)]>0.0f) {
                urban_bep_flux_flat(5.0f,r.z0[dir*levels+ulong(k+1)],
                    r.ua_u[idx],r.va_u[idx],r.pt_u[idx],r.pt0_u[idx],
                    r.ptr[dir*levels+ulong(k+1)],uh,vh,th,eh);
            }
            r.column_building_x[flat_out+ulong(k)]=uh;
            r.column_building_x[flat_out+36ul+ulong(k)]=vh;
            r.column_building_x[flat_out+72ul+ulong(k)]=th;
            r.column_building_x[flat_out+108ul+ulong(k)]=eh;
        }
        for (int side=0;side<2;++side) {
            const int wall_index=2*direction+side;
            for (int k=0;k<nz;++k) {
                const ulong idx=flat+ulong(k);
                const ulong wall0=out_flat+ulong(wall_index)*18ul+ulong(k);
                urban_bep_flux_wall(r.ua_u[idx],r.va_u[idx],r.pt_u[idx],r.da_u[idx],
                    r.ptw[ulong(col)*72ul+ulong(wall_index)*18ul+ulong(k)],
                    r.drst[dir],p.dt,
                    r.column_building_x[wall0],
                    r.column_building_x[wall0+72ul],
                    r.column_building_x[wall0+144ul],
                    r.column_building_x[wall0+216ul],
                    r.column_building_x[wall0+288ul],
                    r.column_building_x[wall0+360ul],
                    r.column_building_x[wall0+432ul]);
            }
        }
    }
}
