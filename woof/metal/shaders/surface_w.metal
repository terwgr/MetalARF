#include <metal_stdlib>
#include "set_surface_w_bindings.metalh"
using namespace metal;
#pragma clang fp contract(off)

kernel void set_surface_w(constant SetSurfaceWResources &r [[buffer(0)]],
                          constant SetSurfaceWParams &q [[buffer(1)]],
                          uint column_id [[thread_position_in_grid]]) {
    const uint ny=uint(q.ny), nx=uint(q.nx);
    const ulong su=ulong(ny)*(nx+1);
    const ulong sv=ulong(ny+1)*nx;
    if (ulong(column_id)>=ulong(ny)*ulong(nx)) return;
    const int j=int(column_id)/q.nx, i=int(column_id)-j*q.nx;
    const ulong c=ulong(column_id);
    const int jm=q.boundary_y?max(j-1,0):(j-1+q.ny)%q.ny;
    const int jp=q.boundary_y?min(j+1,q.ny-1):(j+1)%q.ny;
    const int im=q.boundary_x?max(i-1,0):(i-1+q.nx)%q.nx;
    const int ip=q.boundary_x?min(i+1,q.nx-1):(i+1)%q.nx;

    const ulong u0=ulong(j)*(nx+1)+ulong(i);
    const ulong u1=u0+1;
    const ulong v0=ulong(j)*nx+ulong(i);
    const ulong v1=ulong(j+1)*nx+ulong(i);
    const float uc0=q.cf1*r.u[u0]+q.cf2*r.u[su+u0];
    const float uc=uc0+q.cf3*r.u[2*su+u0];
    const float ue0=q.cf1*r.u[u1]+q.cf2*r.u[su+u1];
    const float ue=ue0+q.cf3*r.u[2*su+u1];
    const float vs0=q.cf1*r.v[v0]+q.cf2*r.v[sv+v0];
    const float vs=vs0+q.cf3*r.v[2*sv+v0];
    const float vn0=q.cf1*r.v[v1]+q.cf2*r.v[sv+v1];
    const float vn=vn0+q.cf3*r.v[2*sv+v1];

    float dys=r.ht[c]-r.ht[ulong(jm)*nx+ulong(i)];
    float dyn=r.ht[ulong(jp)*nx+ulong(i)]-r.ht[c];
    float dxw=r.ht[c]-r.ht[ulong(j)*nx+ulong(im)];
    float dxe=r.ht[ulong(j)*nx+ulong(ip)]-r.ht[c];
    if(q.boundary_y && j==0) dys=0.0f;
    if(q.boundary_y && j==q.ny-1) dyn=0.0f;
    if(q.boundary_x && i==0) dxw=0.0f;
    if(q.boundary_x && i==q.nx-1) dxe=0.0f;

    const float vterm=(dyn*vn+dys*vs)*q.half_rdy;
    const float uterm=(dxe*ue+dxw*uc)*q.half_rdx;
    float value=vterm+uterm;
    if(q.has_msf) value=value*r.msft[c];
    r.w[c]=value;
}
