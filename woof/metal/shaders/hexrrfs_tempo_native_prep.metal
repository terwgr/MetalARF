#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "hexrrfs_tempo_native_from_bindings.metalh"
#include "hexrrfs_tempo_native_emissions_bindings.metalh"
#include "hexrrfs_tempo_native_pack_bindings.metalh"
using namespace ntc;
#pragma clang fp contract(off)

// The source MPAS arrays are independent allocations. The prepared and saved
// buffers use the carrier orders in tempo_v303.py FIELD_ARGUMENTS and
// TEND_ARGUMENTS. Each plane is (nz, nc), column fastest.
inline F get(device const float *p, ulong i) { return f32_word(as_type<uint>(p[i])); }
inline void put(device float *p, ulong i, F x) { p[i] = as_type<float>(x.bits); }

kernel void hexrrfs_tempo_native_from(constant HexrrfsTempoNativeFromResources &r [[buffer(0)]],
                            constant HexrrfsTempoNativeFromParams &p [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    const ulong n = ulong(uint(p.nz)) * uint(p.nc), x = gid;
    if (x >= n || x % uint(p.nc) >= uint(p.solve)) return;
    // Use safe dummy allocations for absent pointers: only enabled members
    // are dereferenced. Missing optional prepared planes retain their words.
    device const float *m[13] = {r.qv,r.qc,r.qr,r.qi,r.qs,r.qg,r.ni,r.nr,
                                  r.nc,r.nwfa,r.nifa,r.ng,r.volg};
    for (int j=0; j<13; ++j) {
        if (j < 8 || (j < 11 ? p.aerosol : p.hail)) {
            if (j < 8 || (p.presence & (1 << (j-8))))
                r.prepared[ulong(j+4)*n+x] = m[j][x];
        }
    }
    const F qv = get(r.prepared, 4*n+x);
    const F factor = tc_add(1.0f,tc_mul(tc_div(F(p.rv),F(p.rd)),tc_max(0.0f,qv)));
    put(r.prepared, 18*n+x, tc_mul(get(r.zz,x),get(r.rho_zz,x)));
    put(r.prepared, x, tc_div(get(r.theta_m,x),factor));
    r.prepared[17*n+x]=r.exner[x];
    put(r.prepared, n+x, tc_add(get(r.pressure_base,x),get(r.pressure_p,x)));
    r.prepared[19*n+x]=r.zgrid[x];
    put(r.prepared, 3*n+x, tc_sub(get(r.zgrid,x+uint(p.nc)),get(r.zgrid,x)));
    r.prepared[2*n+x]=r.w[x];
    const F initial_factor = tc_add(1.0f,tc_mul(tc_div(F(p.rv),F(p.rd)),tc_max(0.0f,get(r.qv,x))));
    put(r.saved, x, tc_div(get(r.theta_m,x),initial_factor));
    for (int j=0; j<8; ++j) r.saved[ulong(j+1)*n+x]=m[j][x];
    if (p.aerosol) {
        r.saved[9*n+x]=r.nc[x]; r.saved[10*n+x]=r.nifa[x]; r.saved[11*n+x]=r.nwfa[x];
    }
}

kernel void hexrrfs_tempo_native_emissions(constant HexrrfsTempoNativeEmissionsResources &r [[buffer(0)]],
                                 constant HexrrfsTempoNativeEmissionsParams &p [[buffer(1)]],
                                 uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.solve) || !p.aerosol) return;
    const ulong n=ulong(uint(p.nz))*uint(p.nc), at=13*n+c;
    put(r.prepared,at,tc_add(get(r.prepared,at),tc_mul(get(r.nwfa2d,c),F(p.dt))));
}

// phase=0: pack before each substep; phase=1: unpack after each substep.
// The caller executes emissions exactly once, before the first pack.
kernel void hexrrfs_tempo_native_pack(constant HexrrfsTempoNativePackResources &r [[buffer(0)]],
                            constant HexrrfsTempoNativePackParams &p [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {
    const ulong n=ulong(uint(p.nz))*uint(p.nc), x=gid;
    if (x>=n || x%uint(p.nc)>=uint(p.solve)) return;
    const F exner=get(r.prepared,17*n+x);
    for(int j=0;j<17;++j) {
        const bool present=j<12 || (p.presence & (1 << (j-12)));
        const ulong at=ulong(j)*n+x;
        if (!present) { if (p.phase == 0) r.packed[at]=0.0f; continue; }
        if (p.phase == 0)
            put(r.packed,at,j==0 ? tc_mul(get(r.prepared,x),exner) : get(r.prepared,at));
        else if (j==0 || j>=4)
            put(r.prepared,at,j==0 ? tc_div(get(r.packed,x),exner) : get(r.packed,at));
    }
}
