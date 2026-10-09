#include <metal_stdlib>
#include "urban_bem_surface_temp_fragment_bindings.metalh"
#include "soft_binary32.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float ubmst_add(float a,float b){return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmst_sub(float a,float b){return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmst_mul(float a,float b){return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmst_div(float a,float b){return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b)));}
static inline float ubmst_pow4(float x){float x2=ubmst_mul(x,x);return ubmst_mul(x2,x2);}

// WRF bb_surf_temp/bb_soil_temp. One thread per column, direction loops in
// source order; tg is the local packed BEM ground profile and is updated.
kernel void urban_bem_surface_temp_fragment(
    constant UrbanBemSurfaceTempFragmentResources &r [[buffer(0)]],
    constant UrbanBemSurfaceTempFragmentParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if(col>=uint(p.ncol))return;
    device const int *active=r.active_cols;
    device const int *utype=r.utype;
    device const int *class_int=r.class_int;
    device const float *class_float=r.class_float;
    device const float *alag=r.alag;
    device const float *csg=r.csg;
    device const float *interp=r.interp;
    device const float *rsg=r.rsg;
    device const float *rlg=r.rlg;
    device float *tg=r.tg;
    device float *ptg=r.ptg;
    device float *sfg=r.sfg;
    device float *lfg=r.lfg;
    device float *gfg=r.gfg;
    device int *error=r.error;
    for(int id=0;id<2;++id)error[ulong(col)*2ul+uint(id)]=0;
    if(active[col]==0)return;
    int type=utype[col];
    if(p.nurbm<1||p.nurbm>11||type<1||type>p.nurbm){error[ulong(col)*2ul]=7201;return;}
    int nd=class_int[uint(type)];
    if(nd<1||nd>2){error[ulong(col)*2ul]=7202;return;}
    float em=class_float[98u+uint(type-1)];
    float alb=class_float[65u+uint(type-1)];
    const float dzg[10]={0.2f,0.12f,0.08f,0.05f,0.03f,0.02f,0.02f,0.01f,0.005f,0.0025f};
    const ulong soil=ulong(col)*20ul;
    const ulong coeff=ulong(col)*10ul;
    const ulong rad=ulong(col)*2ul;
    const ulong pressure=ulong(col)*126ul+72ul;
    for(int dir=0;dir<nd;++dir){
        float tsig=tg[soil+uint(dir*10+9)];
        float alpha=ubmst_add(ubmst_add(ubmst_sub(ubmst_add(
            ubmst_mul(ubmst_sub(1.0f,alb),rsg[rad+uint(dir)]),
            ubmst_mul(em,rlg[rad+uint(dir)])),ubmst_mul(ubmst_mul(em,p.sigma),ubmst_pow4(tsig))),
            sfg[rad+uint(dir)]),lfg[rad+uint(dir)]);
        float cddz[10];
        cddz[0]=ubmst_div(alag[coeff+0],dzg[0]);
        for(int iz=1;iz<10;++iz)cddz[iz]=ubmst_div(ubmst_mul(2.0f,alag[coeff+uint(iz)]),ubmst_add(dzg[iz],dzg[iz-1]));
        float a[30],b[10];
        for(int q=0;q<30;++q)a[q]=0.0f;
        for(int iz=0;iz<10;++iz)b[iz]=tg[soil+uint(dir*10+iz)];
        a[1]=1.0f;
        for(int iz=1;iz<9;++iz){
            float dz=dzg[iz];
            a[iz*3]=ubmst_div(ubmst_mul(-cddz[iz],p.dt),dz);
            a[iz*3+1]=ubmst_add(1.0f,ubmst_div(ubmst_mul(p.dt,ubmst_add(cddz[iz],cddz[iz+1])),dz));
            a[iz*3+2]=ubmst_div(ubmst_mul(-cddz[iz+1],p.dt),dz);
        }
        a[27]=ubmst_div(ubmst_mul(-p.dt,cddz[9]),dzg[9]);
        a[28]=ubmst_add(1.0f,ubmst_div(ubmst_mul(p.dt,cddz[9]),dzg[9]));
        b[9]=ubmst_add(b[9],ubmst_div(ubmst_div(ubmst_mul(p.dt,alpha),csg[coeff+9]),dzg[9]));
        for(int iz=8;iz>=0;--iz){
            float d=a[(iz+1)*3+1];
            if(d==0.0f){error[rad+uint(dir)]=7203;return;}
            b[iz]=ubmst_sub(b[iz],ubmst_div(ubmst_mul(a[iz*3+2],b[iz+1]),d));
            a[iz*3+1]=ubmst_sub(a[iz*3+1],ubmst_div(ubmst_mul(a[iz*3+2],a[(iz+1)*3]),d));
        }
        for(int iz=1;iz<10;++iz){
            float d=a[(iz-1)*3+1];
            if(d==0.0f){error[rad+uint(dir)]=7203;return;}
            b[iz]=ubmst_sub(b[iz],ubmst_div(ubmst_mul(a[iz*3],b[iz-1]),d));
        }
        for(int iz=0;iz<10;++iz){
            if(a[iz*3+1]==0.0f){error[rad+uint(dir)]=7203;return;}
            b[iz]=ubmst_div(b[iz],a[iz*3+1]);
        }
        for(int iz=0;iz<10;++iz)tg[soil+uint(dir*10+iz)]=b[iz];
        float pr=interp[pressure];
        ptg[rad+uint(dir)]=ubmst_mul(b[9],precise::pow(ubmst_div(pr,100000.0f),-0.28585657f));
        float net=ubmst_sub(ubmst_add(ubmst_mul(ubmst_sub(1.0f,alb),rsg[rad+uint(dir)]),ubmst_mul(em,rlg[rad+uint(dir)])),ubmst_mul(ubmst_mul(em,p.sigma),ubmst_pow4(tsig)));
        gfg[rad+uint(dir)]=ubmst_add(net,sfg[rad+uint(dir)]);
    }
}
