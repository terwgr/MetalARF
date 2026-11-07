#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_acoustic_calc_coefs_bindings.metalh"
using namespace metal;
#define A(n) reinterpret_cast<device const Exact32 *>(r.n)
#define O(n) reinterpret_cast<device Exact32 *>(r.n)
kernel void wrf_exact_acoustic_calc_coefs(constant WrfExactAcousticCalcCoefsResources &r [[buffer(0)]], constant WrfExactAcousticCalcCoefsParams &q [[buffer(1)]], uint c [[thread_position_in_grid]]) {
    ulong st=ulong(q.ny)*q.nx;
    if (ulong(c)>=st) return;
    Exact32 mut=A(mub2d)[c]+A(mup)[c];
    Exact32 cof=((Exact32(0.5f)*Exact32(q.dtau))*Exact32(q.gravity))*(Exact32(1.0f)+Exact32(q.epssm));
    cof=cof*cof;
    for (int k=0;k<q.nz;++k) {
        ulong ix=ulong(k)*st+c;
        O(c2a)[ix]=(Exact32(q.gamma)*A(p)[ix])/A(alt)[ix];
    }
    O(gam)[c]=0.0f;
    O(a)[st+c]=0.0f;
    for (int k=2;k<q.nz;++k) {
        ulong ix=ulong(k)*st+c, prev=ulong(k-1)*st+c;
        Exact32 denom=(A(c1h)[k-1]*mut+A(c2h)[k-1])*(A(c1f)[k-1]*mut+A(c2f)[k-1]);
        Exact32 value=q.moist_cq ? ((((-A(cqw)[ix]*cof)*A(rdn)[k])*A(rdnw)[k-1])*O(c2a)[prev])/denom : ((((-cof)*A(rdn)[k])*A(rdnw)[k-1])*O(c2a)[prev])/denom;
        O(a)[ix]=value;
    }
    int top=q.nz-1; ulong topix=ulong(q.nz)*st+c;
    if (q.top_lid) O(a)[topix]=Exact32::from_bits(0x80000000u);
    else {
        Exact32 denom=(A(c1h)[top]*mut+A(c2h)[top])*(A(c1f)[top]*mut+A(c2f)[top]);
        O(a)[topix]=(((-Exact32(2.0f)*cof)*(A(rdnw)[top]*A(rdnw)[top]))*O(c2a)[ulong(top)*st+c])/denom;
    }
    for (int k=1;k<q.nz;++k) {
        ulong ix=ulong(k)*st+c, prev=ulong(k-1)*st+c;
        Exact32 ch=A(c1h)[k]*mut+A(c2h)[k], chm=A(c1h)[k-1]*mut+A(c2h)[k-1];
        Exact32 cf=A(c1f)[k]*mut+A(c2f)[k], cfp=A(c1f)[k+1]*mut+A(c2f)[k+1];
        Exact32 up=(A(rdnw)[k]*O(c2a)[ix])/(ch*cf);
        Exact32 lo=(A(rdnw)[k-1]*O(c2a)[prev])/(chm*cf);
        Exact32 b,cc;
        if (q.moist_cq) {
            Exact32 cq=A(cqw)[ix];
            b=Exact32(1.0f)+((cq*cof)*A(rdn)[k])*(up+lo);
            cc=((((-cq*cof)*A(rdn)[k])*A(rdnw)[k])*O(c2a)[ix])/(ch*cfp);
        } else {
            b=Exact32(1.0f)+(cof*A(rdn)[k])*(up+lo);
            cc=((((-cof)*A(rdn)[k])*A(rdnw)[k])*O(c2a)[ix])/(ch*cfp);
        }
        Exact32 al=Exact32(1.0f)/(b-O(a)[ix]*O(gam)[prev]);
        O(alpha)[ix]=al;
        O(gam)[ix]=cc*al;
    }
    Exact32 btop=Exact32(1.0f)+(((Exact32(2.0f)*cof)*(A(rdnw)[top]*A(rdnw)[top]))*O(c2a)[ulong(top)*st+c])/((A(c1h)[top]*mut+A(c2h)[top])*(A(c1f)[q.nz]*mut+A(c2f)[q.nz]));
    O(alpha)[topix]=Exact32(1.0f)/(btop-O(a)[topix]*O(gam)[ulong(top)*st+c]);
    O(gam)[topix]=0.0f;
}
#undef A
#undef O
