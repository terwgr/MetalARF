#include <metal_stdlib>
#include "specified_relaxation_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

inline float lbc_value(float value, float tendency, float dtbc) {
    return value + dtbc * tendency;
}

kernel void specified_relaxation(
    constant SpecifiedRelaxationResources &r [[buffer(0)]],
    constant SpecifiedRelaxationParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const uint nz=uint(q.nz), ny=uint(q.ny), nx=uint(q.nx);
    const uint width=uint(q.width);
    const ulong plane=ulong(ny)*nx;
    const ulong count=ulong(nz)*plane;
    if (ulong(tid)>=count) return;
    const int k=int(ulong(tid)/plane);
    const int rem=int(ulong(tid)-ulong(k)*plane);
    const int j=rem/q.nx, i=rem-j*q.nx;
    float result=r.field_tend[tid];

    // The source processes Y sides first, which gives them corner ownership.
    const int ds=j, dn=q.ny-1-j;
    if (ds<q.spec_zone && i>=ds && i<q.nx-ds) {
        const ulong b=(ulong(k)*width+ulong(ds))*nx+ulong(i);
        r.field_tend[tid]=r.south_t[b];
        return;
    }
    if (dn<q.spec_zone && i>=dn && i<q.nx-dn) {
        const ulong b=(ulong(k)*width+ulong(dn))*nx+ulong(i);
        r.field_tend[tid]=r.north_t[b];
        return;
    }
    const int dw=i, de=q.nx-1-i;
    if (dw<q.spec_zone && j>=dw+1 && j<q.ny-dw-1) {
        const ulong b=(ulong(k)*ny+ulong(j))*width+ulong(dw);
        r.field_tend[tid]=r.west_t[b];
        return;
    }
    if (de<q.spec_zone && j>=de+1 && j<q.ny-de-1) {
        const ulong b=(ulong(k)*ny+ulong(j))*width+ulong(de);
        r.field_tend[tid]=r.east_t[b];
        return;
    }
    if (!q.apply_relax) return;

    if (ds>=q.spec_zone && ds<q.relax_zone && i>=ds && i<q.nx-ds) {
        const int d=ds;
        const ulong b0=(ulong(k)*width+ulong(d))*nx+ulong(i);
        const ulong bm=(ulong(k)*width+ulong(d))*nx+ulong(max(i-1,0));
        const ulong bp=(ulong(k)*width+ulong(d))*nx+ulong(min(i+1,q.nx-1));
        const ulong bo=(ulong(k)*width+ulong(d-1))*nx+ulong(i);
        const ulong bi=(ulong(k)*width+ulong(d+1))*nx+ulong(i);
        const ulong cell=ulong(tid), west_cell=cell-ulong(i)+ulong(max(i-1,0));
        const ulong east_cell=cell-ulong(i)+ulong(min(i+1,q.nx-1));
        const float f0=lbc_value(r.south[b0],r.south_t[b0],q.dtbc)-r.field[cell];
        const float f1=lbc_value(r.south[bm],r.south_t[bm],q.dtbc)-r.field[west_cell];
        const float f2=lbc_value(r.south[bp],r.south_t[bp],q.dtbc)-r.field[east_cell];
        const float f3=lbc_value(r.south[bo],r.south_t[bo],q.dtbc)-r.field[cell-nx];
        const float f4=lbc_value(r.south[bi],r.south_t[bi],q.dtbc)-r.field[cell+nx];
        result += r.fcx[d]*f0-r.gcx[d]*(f1+f2+f3+f4-4.0f*f0);
    } else if (dn>=q.spec_zone && dn<q.relax_zone && i>=dn && i<q.nx-dn) {
        const int d=dn;
        const ulong b0=(ulong(k)*width+ulong(d))*nx+ulong(i);
        const ulong bm=(ulong(k)*width+ulong(d))*nx+ulong(max(i-1,0));
        const ulong bp=(ulong(k)*width+ulong(d))*nx+ulong(min(i+1,q.nx-1));
        const ulong bo=(ulong(k)*width+ulong(d-1))*nx+ulong(i);
        const ulong bi=(ulong(k)*width+ulong(d+1))*nx+ulong(i);
        const ulong cell=ulong(tid), west_cell=cell-ulong(i)+ulong(max(i-1,0));
        const ulong east_cell=cell-ulong(i)+ulong(min(i+1,q.nx-1));
        const float f0=lbc_value(r.north[b0],r.north_t[b0],q.dtbc)-r.field[cell];
        const float f1=lbc_value(r.north[bm],r.north_t[bm],q.dtbc)-r.field[west_cell];
        const float f2=lbc_value(r.north[bp],r.north_t[bp],q.dtbc)-r.field[east_cell];
        const float f3=lbc_value(r.north[bo],r.north_t[bo],q.dtbc)-r.field[cell+nx];
        const float f4=lbc_value(r.north[bi],r.north_t[bi],q.dtbc)-r.field[cell-nx];
        result += r.fcx[d]*f0-r.gcx[d]*(f1+f2+f3+f4-4.0f*f0);
    } else if (dw>=q.spec_zone && dw<q.relax_zone
               && j>=dw+1 && j<q.ny-dw-1) {
        const int d=dw;
        const ulong b0=(ulong(k)*ny+ulong(j))*width+ulong(d);
        const ulong bm=(ulong(k)*ny+ulong(j-1))*width+ulong(d);
        const ulong bp=(ulong(k)*ny+ulong(j+1))*width+ulong(d);
        const ulong bo=(ulong(k)*ny+ulong(j))*width+ulong(d-1);
        const ulong bi=(ulong(k)*ny+ulong(j))*width+ulong(d+1);
        const ulong cell=ulong(tid);
        const float f0=lbc_value(r.west[b0],r.west_t[b0],q.dtbc)-r.field[cell];
        const float f1=lbc_value(r.west[bm],r.west_t[bm],q.dtbc)-r.field[cell-nx];
        const float f2=lbc_value(r.west[bp],r.west_t[bp],q.dtbc)-r.field[cell+nx];
        const float f3=lbc_value(r.west[bo],r.west_t[bo],q.dtbc)-r.field[cell-1];
        const float f4=lbc_value(r.west[bi],r.west_t[bi],q.dtbc)-r.field[cell+1];
        result += r.fcx[d]*f0-r.gcx[d]*(f1+f2+f3+f4-4.0f*f0);
    } else if (de>=q.spec_zone && de<q.relax_zone
               && j>=de+1 && j<q.ny-de-1) {
        const int d=de;
        const ulong b0=(ulong(k)*ny+ulong(j))*width+ulong(d);
        const ulong bm=(ulong(k)*ny+ulong(j-1))*width+ulong(d);
        const ulong bp=(ulong(k)*ny+ulong(j+1))*width+ulong(d);
        const ulong bo=(ulong(k)*ny+ulong(j))*width+ulong(d-1);
        const ulong bi=(ulong(k)*ny+ulong(j))*width+ulong(d+1);
        const ulong cell=ulong(tid);
        const float f0=lbc_value(r.east[b0],r.east_t[b0],q.dtbc)-r.field[cell];
        const float f1=lbc_value(r.east[bm],r.east_t[bm],q.dtbc)-r.field[cell-nx];
        const float f2=lbc_value(r.east[bp],r.east_t[bp],q.dtbc)-r.field[cell+nx];
        const float f3=lbc_value(r.east[bo],r.east_t[bo],q.dtbc)-r.field[cell+1];
        const float f4=lbc_value(r.east[bi],r.east_t[bi],q.dtbc)-r.field[cell-1];
        result += r.fcx[d]*f0-r.gcx[d]*(f1+f2+f3+f4-4.0f*f0);
    }
    r.field_tend[tid]=result;
}
