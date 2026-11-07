#include <metal_stdlib>
#include "wrf_exact_diff_opt1_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)
// WRF v4.7.1 model-coordinate horizontal diffusion.
// module_big_step_utilities_em.F:2715-3060. Explicit REAL staging keeps
// coordinate fluxes separate from the unchanged diff_opt=2 kernels.
static inline float d1add(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float d1sub(float a,float b){return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
static inline float d1mul(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float d1four(float a,float b,float c,float d){
    return d1mul(0.25f,d1add(d1add(d1add(a,b),c),d));
}
static inline int d1index(int n,int size,int boundary){
    return boundary ? max(0,min(n,size-1)) : (n%size+size)%size;
}
struct Diff1 {
    device const float *field,*km,*mu,*c1,*c2,*base,*msft,*msfu,*msfv;
    int nz,ny,nx,stag,bx,by,perturb;
    float mass(int k,int j,int i) const {
        return d1add(d1mul(c1[k],mu[d1index(j,ny,by)*nx+d1index(i,nx,bx)]),c2[k]);
    }
    float kval(int k,int j,int i) const {
        return km[(max(0,min(k,nz-1))*ny+d1index(j,ny,by))*nx+d1index(i,nx,bx)];
    }
    float fval(int k,int j,int i) const {
        int ii=stag==1 ? (bx ? max(0,min(i,nx)) : d1index(i,nx,0)) : d1index(i,nx,bx);
        int jj=stag==2 ? (by ? max(0,min(j,ny)) : d1index(j,ny,0)) : d1index(j,ny,by);
        return field[(k*(ny+(stag==2))+jj)*(nx+(stag==1))+ii];
    }
    float bval(int k,int j,int i) const {
        return base[(k*ny+d1index(j,ny,by))*nx+d1index(i,nx,bx)];
    }
    float map(int stagger,int j,int i) const {
        int width=nx+(stagger==1);
        int ii=stagger==1 && bx ? max(0,min(i,nx)) : d1index(i,nx,bx);
        int jj=stagger==2 && by ? max(0,min(j,ny)) : d1index(j,ny,by);
        return (stagger==1?msfu:stagger==2?msfv:msft)[jj*width+ii];
    }
    float delta(int k,int j1,int i1,int j0,int i0) const {
        float v=d1sub(fval(k,j1,i1),fval(k,j0,i0));
        if(perturb) v=d1add(d1sub(v,bval(k,j1,i1)),bval(k,j0,i0));
        return v;
    }
};


static inline float d1div(float a,float b){return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));}
static inline float d1sqrt(float a){return as_type<float>(soft32_sqrt(as_type<uint>(a)));}


// fminf on binary32 words, including gradual underflow and signed zero.
static inline float d1min_exact(float a,float b){
    uint x=as_type<uint>(a),y=as_type<uint>(b);
    if(soft32_is_nan(x)) return b;
    if(soft32_is_nan(y)) return a;
    if(soft32_is_zero(x) && soft32_is_zero(y)) return as_type<float>(x|y);
    return soft32_less(x,y) ? a : b;
}

kernel void wrf_exact_diff_opt1_horizontal(constant WrfExactDiffOpt1HorizontalResources &r [[buffer(0)]], constant WrfExactDiffOpt1HorizontalParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    const int sx=p.nx+(p.stag==1),sy=p.ny+(p.stag==2);
    if(gid>=uint((p.nz+(p.stag==3))*sy*sx))return;
    const int i=int(gid)%sx,j=(int(gid)/sx)%sy,k=int(gid)/(sx*sy);
    device const float *field=r.field;
    device const float *km=r.km;
    device const float *mu=r.mu;
    device const float *c1=r.c1;
    device const float *c2=r.c2;
    device const float *base=r.base;
    device const float *msft=r.msft;
    device const float *msfu=r.msfu;
    device const float *msfv=r.msfv;
    device float *tendency=r.tendency;
    float rdx=p.rdx;
    float rdy=p.rdy;
    int nz=p.nz;
    int ny=p.ny;
    int nx=p.nx;
    int stag=p.stag;
    int bx=p.bx;
    int by=p.by;
    int perturb=p.perturb;

    int nxs=nx+(stag==1),nys=ny+(stag==2),nzs=nz+(stag==3);
    if(i>=nxs||j>=nys||k>=nzs||(stag==3&&(k==0||k==nz)))return;
    if(bx&&(i==0||i>=nx-(stag!=1)))return;
    if(by&&(j==0||j>=ny-(stag!=2)))return;
    Diff1 q={field,km,mu,c1,c2,base,msft,msfu,msfv,nz,ny,nx,stag,bx,by,perturb};
    float xm,xp,ym,yp;
    if(stag==1){
        xm=d1mul(d1mul(q.mass(k,j,i-1),q.kval(k,j,i-1)),rdx);
        xp=d1mul(d1mul(q.mass(k,j,i),q.kval(k,j,i)),rdx);
        ym=d1mul(d1mul(d1four(q.mass(k,j,i),q.mass(k,j-1,i),q.mass(k,j-1,i-1),q.mass(k,j,i-1)),
                      d1four(q.kval(k,j,i),q.kval(k,j-1,i),q.kval(k,j-1,i-1),q.kval(k,j,i-1))),rdy);
        yp=d1mul(d1mul(d1four(q.mass(k,j,i),q.mass(k,j+1,i),q.mass(k,j+1,i-1),q.mass(k,j,i-1)),
                      d1four(q.kval(k,j,i),q.kval(k,j+1,i),q.kval(k,j+1,i-1),q.kval(k,j,i-1))),rdy);
    }else if(stag==2){
        xm=d1mul(d1mul(d1four(q.mass(k,j,i),q.mass(k,j-1,i),q.mass(k,j-1,i-1),q.mass(k,j,i-1)),
                      d1four(q.kval(k,j,i),q.kval(k,j-1,i),q.kval(k,j-1,i-1),q.kval(k,j,i-1))),rdx);
        xp=d1mul(d1mul(d1four(q.mass(k,j,i),q.mass(k,j-1,i),q.mass(k,j-1,i+1),q.mass(k,j,i+1)),
                      d1four(q.kval(k,j,i),q.kval(k,j-1,i),q.kval(k,j-1,i+1),q.kval(k,j,i+1))),rdx);
        // This WRF v4.7.1 coordinate-v meridional row omits dry mass.
        ym=d1mul(q.kval(k,j-1,i),rdy);
        yp=d1mul(q.kval(k,j,i),rdy);
    }else if(stag==3){
        xm=d1mul(d1mul(d1four(q.mass(k,j,i),q.mass(k,j,i-1),q.mass(k,j,i),q.mass(k,j,i-1)),
                      d1four(q.kval(k,j,i),q.kval(k,j,i-1),q.kval(k-1,j,i),q.kval(k-1,j,i-1))),rdx);
        xp=d1mul(d1mul(d1four(q.mass(k,j,i+1),q.mass(k,j,i),q.mass(k,j,i+1),q.mass(k,j,i)),
                      d1four(q.kval(k,j,i+1),q.kval(k,j,i),q.kval(k-1,j,i+1),q.kval(k-1,j,i))),rdx);
        ym=d1mul(d1mul(d1mul(q.map(2,j,i),d1div(1.0f,q.map(2,j,i))),
                      d1four(q.mass(k,j,i),q.mass(k,j-1,i),q.mass(k,j,i),q.mass(k,j-1,i))),
                      d1four(q.kval(k,j,i),q.kval(k,j-1,i),q.kval(k-1,j,i),q.kval(k-1,j-1,i)));
        ym=d1mul(ym,rdy);
        yp=d1mul(d1mul(d1mul(q.map(2,j+1,i),d1div(1.0f,q.map(2,j+1,i))),
                      d1four(q.mass(k,j+1,i),q.mass(k,j,i),q.mass(k,j+1,i),q.mass(k,j,i))),
                      d1four(q.kval(k,j+1,i),q.kval(k,j,i),q.kval(k-1,j+1,i),q.kval(k-1,j,i)));
        yp=d1mul(yp,rdy);
    }else{
        xm=d1mul(d1mul(d1mul(d1mul(0.5f,d1add(q.kval(k,j,i),q.kval(k,j,i-1))),0.5f),
                      d1add(q.mass(k,j,i),q.mass(k,j,i-1))),rdx);
        xp=d1mul(d1mul(d1mul(d1mul(0.5f,d1add(q.kval(k,j,i+1),q.kval(k,j,i))),0.5f),
                      d1add(q.mass(k,j,i+1),q.mass(k,j,i))),rdx);
        ym=d1mul(d1mul(d1mul(d1mul(d1mul(q.map(2,j,i),d1div(1.0f,q.map(2,j,i))),0.5f),
                      d1add(q.kval(k,j,i),q.kval(k,j-1,i))),0.5f),
                      d1add(q.mass(k,j,i),q.mass(k,j-1,i)));
        ym=d1mul(ym,rdy);
        yp=d1mul(d1mul(d1mul(d1mul(d1mul(q.map(2,j+1,i),d1div(1.0f,q.map(2,j+1,i))),0.5f),
                      d1add(q.kval(k,j+1,i),q.kval(k,j,i))),0.5f),
                      d1add(q.mass(k,j+1,i),q.mass(k,j,i)));
        yp=d1mul(yp,rdy);
    }
    float map=q.map(stag==3?0:stag,j,i),map2=d1mul(map,map);
    float tx=d1mul(d1mul(map2,rdx),d1sub(d1mul(xp,q.delta(k,j,i+1,j,i)),d1mul(xm,q.delta(k,j,i,j,i-1))));
    float ty=d1mul(d1mul(map2,rdy),d1sub(d1mul(yp,q.delta(k,j+1,i,j,i)),d1mul(ym,q.delta(k,j,i,j-1,i))));
    int idx=(k*nys+j)*nxs+i;
    tendency[idx]=d1add(tendency[idx],d1add(tx,ty));
}

kernel void wrf_exact_diff_opt1_km4(constant WrfExactDiffOpt1Km4Resources &r [[buffer(0)]], constant WrfExactDiffOpt1Km4Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if(gid>=uint(p.nz*p.ny*p.nx))return;
    const int i=int(gid)%p.nx,j=(int(gid)/p.nx)%p.ny,k=int(gid)/(p.nx*p.ny);
    device const float *d11=r.d11;
    device const float *d22=r.d22;
    device const float *d12=r.d12;
    device const float *msft=r.msft;
    device float *km=r.km;
    device float *kh=r.kh;
    float dx=p.dx;
    float dy=p.dy;
    float cs=p.cs;
    float pr=p.pr;
    int nz=p.nz;
    int ny=p.ny;
    int nx=p.nx;
    int bx=p.bx;
    int by=p.by;

    if(i>=nx||j>=ny||k>=nz)return;
    int idx=(k*ny+j)*nx+i;
    if((bx&&(i==0||i==nx-1))||(by&&(j==0||j==ny-1))){km[idx]=kh[idx]=0.0f;return;}
    int ip=d1index(i+1,nx,bx),jp=d1index(j+1,ny,by);
    float d=d1sub(d11[idx],d22[idx]);
    float cross=d1four(d12[idx],d12[(k*ny+jp)*nx+i],d12[(k*ny+j)*nx+ip],d12[(k*ny+jp)*nx+ip]);
    float strain=d1sqrt(d1add(d1mul(0.25f,d1mul(d,d)),d1mul(cross,cross)));
    float map=msft[j*nx+i],length=d1sqrt(d1div(d1mul(d1div(dx,map),dy),map));
    float value=d1min_exact(d1mul(d1mul(d1mul(d1mul(cs,cs),length),length),strain),d1mul(10.0f,length));
    km[idx]=value;kh[idx]=d1div(value,pr);
}

kernel void wrf_exact_diff_opt1_theta(constant WrfExactDiffOpt1ThetaResources &r [[buffer(0)]], constant WrfExactDiffOpt1ThetaParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if(gid>=uint(p.nz*p.ny*p.nx))return;
    device const float *thp=r.thp;
    device const float *thb=r.thb;
    device float *theta=r.theta;
    float t0=p.t0;
    int nz=p.nz;
    int ny=p.ny;
    int nx=p.nx;
    int thb3d=p.thb3d;
    theta[gid]=d1sub(d1add(thp[gid],thb[thb3d?gid:gid/uint(ny*nx)]),t0);
}
