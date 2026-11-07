#include "hexv842_nt_cuinin_bindings.metalh"
#include <metal_stdlib>
#include "soft_gfk_exp.metalh"
using namespace metal;

static float add(float a,float b) { return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b))); }
static float sub(float a,float b) { return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b))); }
static float mul(float a,float b) { return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b))); }
static float divr(float a,float b) { return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b))); }
static float expn(float a) { return as_type<float>(soft_gfk_exp(as_type<uint>(a))); }
struct C { float cp,rcp,c2,vtmp,r5v,r5s,rv,rs,alv,als; };
static float alfa(float t) {
    float x=max(250.16f,min(273.16f,t));
    float r=divr(sub(x,250.16f),sub(273.16f,250.16f)); return min(1.0f,mul(r,r));
}
static float ewm(float t, thread const C &c) {
    float a=alfa(t);
    float el=expn(divr(mul(17.2693882f,sub(t,273.16f)),sub(t,35.86f)));
    float ei=expn(divr(mul(21.875f,sub(t,273.16f)),sub(t,7.66f)));
    return mul(c.c2,add(mul(a,el),mul(sub(1.0f,a),ei)));
}
static float dem(float t, thread const C &c) {
    float a=alfa(t), dl=sub(t,35.86f), di=sub(t,7.66f);
    return add(mul(mul(a,c.r5v),divr(1.0f,mul(dl,dl))),
               mul(mul(sub(1.0f,a),c.r5s),divr(1.0f,mul(di,di))));
}
static float ldcpm(float t, thread const C &c) {
    float a=alfa(t); return add(mul(a,c.rv),mul(sub(1.0f,a),c.rs));
}
static void adjust(thread float &t,thread float &q,float p,thread const C &c) {
    float qp=divr(1.0f,p);
    for (int pass=0;pass<2;++pass) {
        float qs=min(0.5f,mul(ewm(t,c),qp));
        float cor=divr(1.0f,sub(1.0f,mul(c.vtmp,qs)));
        qs=mul(qs,cor);
        float den=add(1.0f,mul(mul(qs,cor),dem(t,c)));
        float cond=divr(sub(q,qs),den);
        t=add(t,mul(ldcpm(t,c),cond)); q=sub(q,cond);
    }
}
// Input field order pten,pqen,pqsen,puen,pven,pverv,pgeo,paph,pgeoh;
// each is nz*ncol, top down. Output order ptenh,pqenh,pqsenh,ptu,pqu,
// ptd,pqd,puu,pvu,pud,pvd,plu, then int klab and one int klwmin per cell.
kernel void hexv842_nt_cuinin(constant Hexv842NtCuininResources &r [[buffer(0)]], constant Hexv842NtCuininParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    device const float *in=r.in;
    device float *out=r.out;
    device int *ints=r.ints;

    if (i>=uint(p.ncol)) return;
    int n=p.ncol,nz=p.nz,s=n*nz;
    float rcp=divr(1.0f,p.cp);
    float c5l=mul(17.2693882f,sub(273.16f,35.86f));
    float c5i=mul(21.875f,sub(273.16f,7.66f));
    C c={p.cp,rcp,divr(mul(610.78f,p.rd),p.rv),sub(divr(p.rv,p.rd),1.0f),
         mul(mul(c5l,p.xlv),rcp),mul(mul(c5i,p.xls),rcp),mul(p.xlv,rcp),mul(p.xls,rcp),p.xlv,p.xls};
#define IN(f,k) in[(f)*s+(k)*n+i]
#define OUT(f,k) out[(f)*s+(k)*n+i]
    OUT(0,0)=0.0f; OUT(1,0)=0.0f; OUT(2,0)=0.0f;
    for (int k=1;k<nz;++k) {
        float a=add(mul(c.cp,IN(0,k-1)),IN(6,k-1));
        float b=add(mul(c.cp,IN(0,k)),IN(6,k));
        OUT(0,k)=mul(sub(max(a,b),IN(8,k)),c.rcp);
        OUT(1,k)=IN(1,k-1); OUT(2,k)=IN(2,k-1);
        if (k>=nz-2) continue;
        float t=OUT(0,k), q=OUT(2,k); adjust(t,q,IN(7,k),c);
        OUT(0,k)=t; OUT(2,k)=q;
        float v=add(min(IN(1,k-1),IN(2,k-1)),sub(q,IN(2,k-1)));
        OUT(1,k)=max(v,0.0f);
    }
    OUT(0,nz-1)=mul(sub(add(mul(c.cp,IN(0,nz-1)),IN(6,nz-1)),IN(8,nz-1)),c.rcp);
    OUT(1,nz-1)=IN(1,nz-1); OUT(0,0)=IN(0,0); OUT(1,0)=IN(1,0);
    for (int k=nz-2;k>=1;--k) {
        float z1=add(mul(c.cp,OUT(0,k)),IN(8,k));
        float z2=add(mul(c.cp,OUT(0,k+1)),IN(8,k+1));
        OUT(0,k)=mul(sub(z2>z1?z2:z1,IN(8,k)),c.rcp);
    }
    int lwmin=nz; float zwmax=0.0f;
    for (int k=nz-1;k>=2;--k) if (IN(5,k)<zwmax) { zwmax=IN(5,k); lwmin=k+1; }
    ints[s+i]=lwmin;
    for (int k=0;k<nz;++k) {
        int ik=k==0?0:k-1;
        OUT(3,k)=OUT(0,k); OUT(5,k)=OUT(0,k);
        OUT(4,k)=OUT(1,k); OUT(6,k)=OUT(1,k); OUT(11,k)=0.0f;
        OUT(7,k)=IN(3,ik); OUT(9,k)=IN(3,ik);
        OUT(8,k)=IN(4,ik); OUT(10,k)=IN(4,ik);
        ints[k*n+i]=0;
    }
#undef IN
#undef OUT
}
