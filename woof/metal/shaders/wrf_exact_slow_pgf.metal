#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_slow_pgf_bindings.metalh"
using namespace metal;
#define A(name) reinterpret_cast<device const Exact32 *>(r.name)

static Exact32 pp(constant WrfExactSlowPgfResources &r, constant WrfExactSlowPgfParams &q, uint k, ulong c, ulong st) {
    Exact32 v=A(p)[ulong(k)*st+c];
    return q.diagnostics ? v : v-A(pb)[q.base3d ? ulong(k)*st+c : k];
}
static Exact32 dpn(constant WrfExactSlowPgfResources &r, constant WrfExactSlowPgfParams &q, uint k, ulong c, ulong st) {
    if (k==uint(q.nz)) return q.top_lid ? Exact32(q.cfn)*pp(r,q,k-1,c,st)+Exact32(q.cfn1)*pp(r,q,k-2,c,st) : Exact32(0.0f);
    if (k==0) return (Exact32(q.cf1)*pp(r,q,0,c,st)+Exact32(q.cf2)*pp(r,q,1,c,st))+Exact32(q.cf3)*pp(r,q,2,c,st);
    return A(fnm)[k]*pp(r,q,k,c,st)+A(fnp)[k]*pp(r,q,k-1,c,st);
}
static Exact32 dpn_face(constant WrfExactSlowPgfResources &r, constant WrfExactSlowPgfParams &q, uint k, ulong a, ulong b, ulong st) {
    if (!q.bigstep) return Exact32(0.5f)*(dpn(r,q,k,a,st)+dpn(r,q,k,b,st));
    if (k==uint(q.nz) && !q.top_lid) return 0.0f;
    uint k0=k==uint(q.nz) ? k-1 : k;
    Exact32 pair0=pp(r,q,k0,b,st)+pp(r,q,k0,a,st);
    if (k==0) {
        Exact32 pair1=pp(r,q,1,b,st)+pp(r,q,1,a,st);
        Exact32 pair2=pp(r,q,2,b,st)+pp(r,q,2,a,st);
        return Exact32(0.5f)*((Exact32(q.cf1)*pair0+Exact32(q.cf2)*pair1)+Exact32(q.cf3)*pair2);
    }
    uint k1=k==uint(q.nz) ? k-2 : k-1;
    Exact32 pair1=pp(r,q,k1,b,st)+pp(r,q,k1,a,st);
    Exact32 high=(k==uint(q.nz) ? Exact32(q.cfn) : A(fnm)[k])*pair0;
    Exact32 low=(k==uint(q.nz) ? Exact32(q.cfn1) : A(fnp)[k])*pair1;
    return Exact32(0.5f)*(high+low);
}
static Exact32 half_phi(constant WrfExactSlowPgfResources &r, constant WrfExactSlowPgfParams &q, uint k, ulong c, ulong st) {
    ulong a=ulong(k)*st+c, b=a+st;
    if (!q.base3d) return Exact32(0.5f)*(A(php)[a]+A(php)[b]);
    return Exact32(0.5f)*(((A(phb)[a]+A(phb)[b])+A(php)[a])+A(php)[b]);
}
static Exact32 face(constant WrfExactSlowPgfResources &r, constant WrfExactSlowPgfParams &q, ulong ca, ulong cb, uint k, ulong st, Exact32 rd, Exact32 half_rd) {
    ulong a=ulong(k)*st+ca, b=ulong(k)*st+cb;
    Exact32 muf=Exact32(0.5f)*(((A(mup)[ca]+A(mup)[cb])+A(mub2d)[ca])+A(mub2d)[cb]);
    Exact32 dmu=Exact32(0.5f)*(A(mup)[ca]+A(mup)[cb]);
    Exact32 layer=A(c1h)[k]*muf+A(c2h)[k];
    Exact32 dhi=A(php)[a+st]-A(php)[b+st];
    Exact32 bracket=q.bigstep ? (dhi+A(php)[a])-A(php)[b] : dhi+(A(php)[a]-A(php)[b]);
    bracket=bracket+(A(alt)[a]+A(alt)[b])*(pp(r,q,k,ca,st)-pp(r,q,k,cb,st));
    Exact32 pba=A(pb)[q.base3d ? a : k], pbb=A(pb)[q.base3d ? b : k];
    bracket=bracket+(A(al)[a]+A(al)[b])*(pba-pbb);
    Exact32 left=(half_rd*layer)*bracket;
    Exact32 dphp=half_phi(r,q,k,ca,st)-half_phi(r,q,k,cb,st);
    Exact32 vertical=A(rdnw)[k]*(dpn_face(r,q,k+1,ca,cb,st)-dpn_face(r,q,k,ca,cb,st));
    Exact32 dm=q.bigstep ? Exact32(0.5f)*(A(c1h)[k]*A(mup)[cb]+A(c1h)[k]*A(mup)[ca]) : A(c1h)[k]*dmu;
    return left+(rd*dphp)*(vertical-dm);
}
kernel void wrf_exact_slow_pgf(constant WrfExactSlowPgfResources &r [[buffer(0)]], constant WrfExactSlowPgfParams &q [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    uint nx=q.nx, ny=q.ny, nz=q.nz, nxf=nx+1, nyf=ny+1;
    if (ulong(tid)>=ulong(nz)*nyf*nxf) return;
    uint k=tid/(nyf*nxf), rem=tid-k*nyf*nxf, j=rem/nxf, i=rem-j*nxf;
    ulong st=ulong(ny)*nx;
    if (j<ny && (!q.boundary_x || (i>0 && i<nx))) {
        ulong ca=ulong(j)*nx+i%nx, cb=ulong(j)*nx+(i+nx-1)%nx, out=(ulong(k)*ny+j)*nxf+i;
        Exact32 term=face(r,q,ca,cb,k,st,Exact32(q.rdx),Exact32(q.half_rdx));
        if (q.use_cq) term=A(cqu)[out]*term;
        device Exact32 *u=reinterpret_cast<device Exact32 *>(r.ru_t); u[out]=u[out]-term;
    }
    if (i<nx && (!q.boundary_y || (j>0 && j<ny))) {
        ulong ca=ulong(j%ny)*nx+i, cb=ulong((j+ny-1)%ny)*nx+i, out=(ulong(k)*nyf+j)*nx+i;
        Exact32 term=face(r,q,ca,cb,k,st,Exact32(q.rdy),Exact32(q.half_rdy));
        if (q.use_cq) term=A(cqv)[out]*term;
        device Exact32 *v=reinterpret_cast<device Exact32 *>(r.rv_t); v[out]=v[out]-term;
    }
}
#undef A
