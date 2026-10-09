#include <metal_stdlib>
#include "close_periodic_alias_bindings.metalh"
using namespace metal;

kernel void close_periodic_alias(
    constant ClosePeriodicAliasResources &r [[buffer(0)]],
    constant ClosePeriodicAliasParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const ulong nxf=ulong(q.nx+1), nyf=ulong(q.ny+1);
    const ulong su=ulong(q.ny)*nxf, sv=nyf*ulong(q.nx);
    const ulong plane=nyf*nxf;
    if(ulong(tid)>=ulong(q.nz)*plane) return;
    const uint k=uint(ulong(tid)/plane);
    const ulong rem=ulong(tid)-ulong(k)*plane;
    const int j=int(rem/nxf), i=int(rem-ulong(j)*nxf);
    if(!q.boundary_x && i==q.nx && j<q.ny) {
        const ulong dst=ulong(k)*su+ulong(j)*nxf+ulong(q.nx);
        r.u[dst]=r.u[dst-ulong(q.nx)];
    }
    if(!q.boundary_y && j==q.ny && i<q.nx) {
        const ulong dst=ulong(k)*sv+ulong(q.ny)*ulong(q.nx)+ulong(i);
        r.v[dst]=r.v[dst-ulong(q.ny)*ulong(q.nx)];
    }
}
