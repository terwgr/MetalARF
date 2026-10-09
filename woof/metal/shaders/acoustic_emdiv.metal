#include <metal_stdlib>
#include "apply_emdiv_bindings.metalh"
#include "update_mudf_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void apply_emdiv(constant ApplyEmdivResources &r [[buffer(0)]],
                        constant ApplyEmdivParams &q [[buffer(1)]],
                        uint tid [[thread_position_in_grid]]) {
    const int nyf=q.ny+1, nxf=q.nx+1;
    const ulong plane=ulong(nyf)*ulong(nxf);
    const ulong count=ulong(q.nz)*plane;
    if(ulong(tid)>=count) return;
    const int k=int(ulong(tid)/plane);
    const ulong rem=ulong(tid)-ulong(k)*plane;
    const int j=int(rem/ulong(nxf)), i=int(rem-ulong(j)*ulong(nxf));
    if(q.save_mu && k==0 && j<q.ny && i<q.nx)
        r.mu_prev[ulong(j)*ulong(q.nx)+ulong(i)]=r.mu_pp[ulong(j)*ulong(q.nx)+ulong(i)];

    bool do_u=false;
    int gi=i;
    if(j<q.ny) {
        if(q.specified)
            do_u=j>=q.spec_zone && j<q.ny-q.spec_zone &&
                 i>=q.spec_zone && i<=q.nx-q.spec_zone;
        else if(q.boundary_x)
            do_u=i>=1 && i<q.nx;
        else {
            do_u=true;
            if(i==q.nx) gi=0;
        }
    }
    if(do_u) {
        const int gim=gi==0?q.nx-1:gi-1;
        const ulong c=ulong(j)*ulong(q.nx)+ulong(gi);
        const ulong cm=ulong(j)*ulong(q.nx)+ulong(gim);
        float gx=r.mudf[c]-r.mudf[cm];
        gx=q.xscale*gx;
        if(q.has_msf) gx=gx/r.msfu[ulong(j)*ulong(nxf)+ulong(gi)];
        const float increment=r.c1h[k]*gx;
        const ulong ix=ulong(k)*ulong(q.ny)*ulong(nxf)+ulong(j)*ulong(nxf)+ulong(i);
        r.u_pp[ix]=r.u_pp[ix]+increment;
    }

    bool do_v=false;
    int gj=j;
    if(i<q.nx) {
        if(q.specified)
            do_v=i>=q.spec_zone && i<q.nx-q.spec_zone &&
                 j>=q.spec_zone && j<=q.ny-q.spec_zone;
        else if(q.boundary_y)
            do_v=j>=1 && j<q.ny;
        else {
            do_v=true;
            if(j==q.ny) gj=0;
        }
    }
    if(do_v) {
        const int gjm=gj==0?q.ny-1:gj-1;
        const ulong c=ulong(gj)*ulong(q.nx)+ulong(i);
        const ulong cm=ulong(gjm)*ulong(q.nx)+ulong(i);
        float gy=r.mudf[c]-r.mudf[cm];
        gy=q.yscale*gy;
        if(q.has_msf) gy=gy/r.msfv[ulong(gj)*ulong(q.nx)+ulong(i)];
        const float increment=r.c1h[k]*gy;
        const ulong ix=ulong(k)*ulong(q.ny+1)*ulong(q.nx)+ulong(j)*ulong(q.nx)+ulong(i);
        r.v_pp[ix]=r.v_pp[ix]+increment;
    }
}

kernel void update_mudf(constant UpdateMudfResources &r [[buffer(0)]],
                        constant UpdateMudfParams &q [[buffer(1)]],
                        uint column_id [[thread_position_in_grid]]) {
    const ulong count=ulong(q.ny)*ulong(q.nx);
    if(ulong(column_id)>=count) return;
    const int j=int(column_id)/q.nx, i=int(column_id)-j*q.nx;
    const ulong c=ulong(column_id);
    if((q.boundary_x && (i<q.width || i>=q.nx-q.width)) ||
       (q.boundary_y && (j<q.width || j>=q.ny-q.width))) {
        r.mudf[c]=0.0f;
        return;
    }
    const float difference=r.mu_pp[c]-r.mu_prev[c];
    r.mudf[c]=difference/q.dtau;
}
