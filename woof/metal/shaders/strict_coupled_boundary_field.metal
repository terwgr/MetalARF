#include <metal_stdlib>
#include "strict_coupled_boundary_field_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;

static inline uint load_word(device const float *p, ulong i) { return as_type<uint>(p[i]); }
static inline uint column_mass(device const float *b, device const float *p, ulong i) {
    return soft32_add(load_word(b,i),load_word(p,i));
}
kernel void strict_coupled_boundary_field(
        constant StrictCoupledBoundaryFieldResources &r [[buffer(0)]],
        constant StrictCoupledBoundaryFieldParams &q [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
    ulong count=ulong(q.nz)*ulong(q.ny)*ulong(q.nx);
    if (ulong(tid)>=count) return;
    int k=int(ulong(tid)/(ulong(q.ny)*ulong(q.nx)));
    ulong rem=ulong(tid)-ulong(k)*ulong(q.ny)*ulong(q.nx);
    int j=int(rem/ulong(q.nx));
    int i=int(rem-ulong(j)*ulong(q.nx));
    ulong map_index=ulong(j)*ulong(q.nx)+ulong(i);
    // Use the rounded cell masses and replace outer faces with the adjacent cell.
    uint mu;
    if (q.kind==0) {
        ulong row=ulong(j)*ulong(q.mnx);
        if (i==0) mu=column_mass(r.mub2d,r.mup,row);
        else if (i==q.mnx) mu=column_mass(r.mub2d,r.mup,row+ulong(q.mnx-1));
        else {
            ulong x=row+ulong(i);
            mu=soft32_mul(0x3f000000u,soft32_add(column_mass(r.mub2d,r.mup,x),column_mass(r.mub2d,r.mup,x-1u)));
        }
    } else if (q.kind==1) {
        if (j==0) mu=column_mass(r.mub2d,r.mup,ulong(i));
        else if (j==q.mny) mu=column_mass(r.mub2d,r.mup,ulong(q.mny-1)*ulong(q.mnx)+ulong(i));
        else {
            ulong x=ulong(j)*ulong(q.mnx)+ulong(i);
            mu=soft32_mul(0x3f000000u,soft32_add(column_mass(r.mub2d,r.mup,x),column_mass(r.mub2d,r.mup,x-ulong(q.mnx))));
        }
    } else {
        mu=column_mass(r.mub2d,r.mup,ulong(j)*ulong(q.mnx)+ulong(i));
    }
    uint c1=q.kind==3 ? load_word(r.c1f,ulong(k)) : load_word(r.c1h,ulong(k));
    uint c2=q.kind==3 ? load_word(r.c2f,ulong(k)) : load_word(r.c2h,ulong(k));
    uint coupled=soft32_add(soft32_mul(c1,mu),c2);
    uint value=load_word(r.target,ulong(tid));
    if (q.kind==2) {
        // Form total theta before subtracting the 300 K boundary offset.
        ulong base=q.thb_3d ? (ulong(k)*ulong(q.mny)+ulong(j))*ulong(q.mnx)+ulong(i) : ulong(k);
        value=soft32_sub(soft32_add(load_word(r.thb,base),value),0x43960000u);
    }
    uint product=soft32_mul(coupled,value);
    // Snapshot momentum divides by the forward map factor on both axes.
    if (q.has_msf && (q.kind==0 || q.kind==1))
        product=soft32_div(product,q.kind==0 ? load_word(r.msfu,map_index) : load_word(r.msfv,map_index));
    r.out[tid]=as_type<float>(product);
}
