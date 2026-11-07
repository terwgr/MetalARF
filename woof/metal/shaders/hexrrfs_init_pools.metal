#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "soft_glibc_solar_trig.metalh"
#include "hexrrfs_init_directions_bindings.metalh"
#include "hexrrfs_init_microphysics_top_bindings.metalh"
#include "hexrrfs_init_ocean_temperature_bindings.metalh"
#pragma clang fp contract(off)
using namespace ntc;
inline F startup_get(device const float *p, int i) { return f32_word(as_type<uint>(p[i])); }
inline void startup_put(device float *p, int i, F x) { p[i] = as_type<float>(x.bits); }
kernel void hexrrfs_init_directions(constant HexrrfsInitDirectionsResources &r [[buffer(0)]],
                                    constant HexrrfsInitDirectionsParams &p [[buffer(1)]],
                                    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.cells)) return;
    int c = int(gid);
    F slon=f32_word(soft_glibc_solar_sinf(as_type<uint>(r.lon[c])));
    F clon=f32_word(soft_glibc_solar_cosf(as_type<uint>(r.lon[c])));
    F slat=f32_word(soft_glibc_solar_sinf(as_type<uint>(r.lat[c])));
    F clat=f32_word(soft_glibc_solar_cosf(as_type<uint>(r.lat[c])));
    F ax=-slon, ay=clon, az=F(0.f);
    F mi=tc_div(F(1.f),tc_sqrt(tc_add(tc_add(tc_mul(ax,ax),tc_mul(ay,ay)),tc_mul(az,az))));
    startup_put(r.east,c,tc_mul(ax,mi));
    startup_put(r.east,p.extent+c,tc_mul(ay,mi));
    startup_put(r.east,2*p.extent+c,tc_mul(az,mi));
    ax=tc_mul(-clon,slat); ay=tc_mul(-slon,slat); az=clat;
    mi=tc_div(F(1.f),tc_sqrt(tc_add(tc_add(tc_mul(ax,ax),tc_mul(ay,ay)),tc_mul(az,az))));
    startup_put(r.north,c,tc_mul(ax,mi));
    startup_put(r.north,p.extent+c,tc_mul(ay,mi));
    startup_put(r.north,2*p.extent+c,tc_mul(az,mi));
}
kernel void hexrrfs_init_microphysics_top(constant HexrrfsInitMicrophysicsTopResources &r [[buffer(0)]],
                                          constant HexrrfsInitMicrophysicsTopParams &p [[buffer(1)]],
                                          uint gid [[thread_position_in_grid]]) {
    if (gid >= 1) return;
    int result=0;
    F height=tc_div(F(.5f),startup_get(r.rdzw,0));
    if (height <= F(p.top)) result=1;
    for (int k=1;k<p.levels;++k) {
        height=tc_add(height,startup_get(r.dzu,k));
        if (height <= F(p.top)) result=k+1;
    }
    r.level[0]=result;
}
kernel void hexrrfs_init_ocean_temperature(constant HexrrfsInitOceanTemperatureResources &r [[buffer(0)]],
                                           constant HexrrfsInitOceanTemperatureParams &p [[buffer(1)]],
                                           uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.solve)) return;
    int c=int(gid);
    startup_put(r.deep,c,tc_sub(startup_get(r.sst,c),F(5.f)));
}
