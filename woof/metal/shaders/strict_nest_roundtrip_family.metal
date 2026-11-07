#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "strict_nest_roundtrip_family_bindings.metalh"
using namespace metal;

static inline Exact32 point_mass(Exact32 c1, Exact32 c2,
    device const Exact32 *mub, device const Exact32 *mu, ulong q) {
    return exact_add(exact_add(exact_mul(c1,mub[q]),c2),exact_mul(c1,mu[q]));
}
// Sum is intentionally not pre-halved: the inverse is 2*map/sum.
static inline Exact32 face_sum(Exact32 c1, Exact32 c2,
    device const Exact32 *mub, device const Exact32 *mu,
    device const Exact32 *halo, ulong current, ulong previous, bool edge) {
    Exact32 a=exact_add(exact_mul(c1,mub[current]),c2);
    Exact32 b=edge ? a : exact_add(exact_mul(c1,mub[previous]),c2);
    Exact32 s=exact_add(a,b);
    s=exact_add(s,exact_mul(c1,mu[current]));
    s=exact_add(s,exact_mul(c1,edge ? halo[current] : mu[previous]));
    return s;
}
kernel void strict_nest_roundtrip(
    constant StrictNestRoundtripResources &r [[buffer(0)]],
    constant StrictNestRoundtripParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    ulong count=ulong(p.nz)*ulong(p.ny)*ulong(p.nx);
    if (ulong(tid)>=count) return;
    int k=int(ulong(tid)/(ulong(p.ny)*ulong(p.nx)));
    ulong rem=ulong(tid)-ulong(k)*ulong(p.ny)*ulong(p.nx);
    int j=int(rem/ulong(p.nx));int i=int(rem-ulong(j)*ulong(p.nx));
    device const Exact32 *mub=reinterpret_cast<device const Exact32*>(r.mub2d);
    device const Exact32 *mu=reinterpret_cast<device const Exact32*>(r.mup);
    device const Exact32 *halo=reinterpret_cast<device const Exact32*>(r.halo);
    device const Exact32 *c1h=reinterpret_cast<device const Exact32*>(r.c1h);
    device const Exact32 *c2h=reinterpret_cast<device const Exact32*>(r.c2h);
    device const Exact32 *c1f=reinterpret_cast<device const Exact32*>(r.c1f);
    device const Exact32 *c2f=reinterpret_cast<device const Exact32*>(r.c2f);
    device const Exact32 *msft=reinterpret_cast<device const Exact32*>(r.msft);
    device const Exact32 *msfu=reinterpret_cast<device const Exact32*>(r.msfu);
    device const Exact32 *msfv=reinterpret_cast<device const Exact32*>(r.msfv);
    device Exact32 *field=reinterpret_cast<device Exact32*>(r.field);
    Exact32 forward, inverse;ulong map=ulong(j)*ulong(p.nx)+ulong(i);
    if (p.kind==0 || p.kind==1) {
        bool x=p.kind==0;
        bool final=x ? i==p.mnx : j==p.mny;
        if (final) {
            int jj=x ? j : p.mny-1; int ii=x ? p.mnx-1 : i;
            ulong q=ulong(jj)*ulong(p.mnx)+ulong(ii);
            Exact32 mass=point_mass(c1h[k],c2h[k],mub,mu,q);
            Exact32 factor=p.has_msf ? (x ? msfu[map] : msfv[map]) : Exact32(1.0f);
            forward=p.has_msf ? exact_div(mass,factor) : mass;
            inverse=exact_div(factor,mass);
        } else {
            bool edge=x ? i==0 : j==0;
            ulong q=ulong(j)*ulong(p.mnx)+ulong(i);
            ulong prev=edge ? q : q-(x ? 1ul : ulong(p.mnx));
            Exact32 s=face_sum(c1h[k],c2h[k],mub,mu,halo,q,prev,edge);
            Exact32 factor=p.has_msf ? (x ? msfu[map] : msfv[map]) : Exact32(1.0f);
            Exact32 half_sum=exact_mul(Exact32(.5f),s);
            forward=p.has_msf ? exact_div(half_sum,factor) : half_sum;
            inverse=exact_div(exact_mul(Exact32(2.0f),factor),s);
        }
    } else {
        ulong q=ulong(j)*ulong(p.mnx)+ulong(i);
        bool full=p.kind==2 || p.kind==3;
        Exact32 mass=point_mass(full?c1f[k]:c1h[k],full?c2f[k]:c2h[k],mub,mu,q);
        if (p.kind==2 && p.has_msf) {
            forward=exact_div(mass,msft[map]); inverse=exact_div(msft[map],mass);
        } else { forward=mass;inverse=exact_div(Exact32(1.0f),mass); }
    }
    field[tid]=exact_mul(exact_mul(field[tid],forward),inverse);
}
kernel void strict_nest_halo_face(
    constant StrictNestHaloFaceResources &r [[buffer(0)]],
    constant StrictNestHaloFaceParams &p [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    ulong width=p.kind==0 ? ulong(p.mny) : ulong(p.mnx);
    if (ulong(tid)>=ulong(p.nz)*width) return;
    ulong k=ulong(tid)/width, along=ulong(tid)%width;
    ulong cell=p.kind==0 ? along*ulong(p.mnx) : along;
    ulong offset=p.kind==0 ? k*ulong(p.ny)*ulong(p.nx)+along*ulong(p.nx) : k*ulong(p.ny)*ulong(p.nx)+along;
    device const Exact32 *mub=reinterpret_cast<device const Exact32*>(r.mub2d);
    device const Exact32 *mu=reinterpret_cast<device const Exact32*>(r.mup);
    device const Exact32 *halo=reinterpret_cast<device const Exact32*>(r.halo);
    device const Exact32 *c1=reinterpret_cast<device const Exact32*>(r.c1h);
    device const Exact32 *c2=reinterpret_cast<device const Exact32*>(r.c2h);
    device const Exact32 *map=reinterpret_cast<device const Exact32*>(r.msf);
    device const Exact32 *field=reinterpret_cast<device const Exact32*>(r.field);
    device Exact32 *out=reinterpret_cast<device Exact32*>(r.out);
    Exact32 sum=face_sum(c1[k],c2[k],mub,mu,halo,cell,cell,true);
    Exact32 factor=exact_mul(Exact32(.5f),sum);
    if (p.has_msf) factor=exact_div(factor,p.kind==0 ? map[along*ulong(p.nx)] : map[along]);
    out[offset]=exact_mul(factor,field[offset]);
}
