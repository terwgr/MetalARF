#include <metal_stdlib>
#include "flux_div_v_bindings.metalh"
using namespace metal;
#ifndef GPUWM_WRF_EXACT
#define GPUWM_WRF_EXACT 0
#endif
#ifndef GPUWM_WRF_EXACT_C_ADVECTION
#define GPUWM_WRF_EXACT_C_ADVECTION GPUWM_WRF_EXACT
#endif
static inline ulong vi(int k,int j,int i,int ny,int nx){return (ulong(k)*(ny+1)+j)*nx+i;}
static inline ulong ui(int k,int j,int i,int ny,int nx){return (ulong(k)*ny+j)*(nx+1)+i;}
static inline ulong wi(int k,int j,int i,int ny,int nx){return (ulong(k)*ny+j)*nx+i;}
static inline int per(int i,int n){int r=i%n;return r<0?r+n:r;}
static inline float f5(float a,float b,float c,float d,float e,float f,float v){
#if GPUWM_WRF_EXACT_C_ADVECTION
 float center=(37.0f*(d+c)-8.0f*(e+b)+(f+a))/60.0f;float diss=copysign(1.0f,v)*((f-a)-5.0f*(e-b)+10.0f*(d-c))/60.0f;return v*(center-diss);
#else
 return (v*(37.0f*(d+c)-8.0f*(e+b)+(f+a))-abs(v)*(10.0f*(d-c)-5.0f*(e-b)+(f-a)))/60.0f;
#endif
}
static inline float f3(float a,float b,float c,float d,float v){
#if GPUWM_WRF_EXACT_C_ADVECTION
 float center=(7.0f*(c+b)-(d+a))/12.0f;float diss=copysign(1.0f,v)*((d-a)-3.0f*(c-b))/12.0f;return v*(center-diss);
#else
 return (v*(7.0f*(c+b)-(d+a))+abs(v)*(3.0f*(c-b)-(d-a)))/12.0f;
#endif
}
static inline float f3h(float a,float b,float c,float d,float v){
#if GPUWM_WRF_EXACT_C_ADVECTION
 float center=(7.0f*(c+b)-(d+a))/12.0f;float diss=copysign(1.0f,v)*((d-a)-3.0f*(c-b))/12.0f;return v*(center+diss);
#else
 return (v*(7.0f*(c+b)-(d+a))-abs(v)*(3.0f*(c-b)-(d-a)))/12.0f;
#endif
}
static inline float f2(float a,float b,float v){return 0.5f*v*(a+b);}
static inline float zhalf(device const float*q,float v,int kf,int j,int i,int nz,int ny,int nx,device const float*fnm,device const float*fnp){
 if(kf==0||kf==nz)return 0.0f;if(kf==1||kf==nz-1)return v*(fnm[kf]*q[vi(kf,j,i,ny,nx)]+fnp[kf]*q[vi(kf-1,j,i,ny,nx)]);return f3(q[vi(kf-2,j,i,ny,nx)],q[vi(kf-1,j,i,ny,nx)],q[vi(kf,j,i,ny,nx)],q[vi(kf+1,j,i,ny,nx)],v);
}
static inline float yf_open(constant FluxDivVResources&r,constant FluxDivVParams&p,float v,int k,int m,int i){
 int d=min(m,p.ny-1-m);if(d==0){float a=r.v[vi(k,m,i,p.ny,p.nx)],b=r.v[vi(k,m+1,i,p.ny,p.nx)];if(p.spec){if(m==0){if(b<0.0f)a=b;}else if(a>0.0f)b=a;}return f2(a,b,v);}if(d==1)return f3h(r.v[vi(k,m-1,i,p.ny,p.nx)],r.v[vi(k,m,i,p.ny,p.nx)],r.v[vi(k,m+1,i,p.ny,p.nx)],r.v[vi(k,m+2,i,p.ny,p.nx)],v);return f5(r.v[vi(k,m-2,i,p.ny,p.nx)],r.v[vi(k,m-1,i,p.ny,p.nx)],r.v[vi(k,m,i,p.ny,p.nx)],r.v[vi(k,m+1,i,p.ny,p.nx)],r.v[vi(k,m+2,i,p.ny,p.nx)],r.v[vi(k,m+3,i,p.ny,p.nx)],v);
}
static inline float yf_periodic(constant FluxDivVResources&r,constant FluxDivVParams&p,float v,int k,int m,int i){return f5(r.v[vi(k,per(m-2,p.ny),i,p.ny,p.nx)],r.v[vi(k,per(m-1,p.ny),i,p.ny,p.nx)],r.v[vi(k,per(m,p.ny),i,p.ny,p.nx)],r.v[vi(k,per(m+1,p.ny),i,p.ny,p.nx)],r.v[vi(k,per(m+2,p.ny),i,p.ny,p.nx)],r.v[vi(k,per(m+3,p.ny),i,p.ny,p.nx)],v);}
static inline float xf_open(constant FluxDivVResources&r,constant FluxDivVParams&p,float v,int k,int j,int f){if(f<=0||f>=p.nx)return 0.0f;int d=min(f,p.nx-f);if(d==1)return f2(r.v[vi(k,j,f-1,p.ny,p.nx)],r.v[vi(k,j,f,p.ny,p.nx)],v);if(d==2)return f3h(r.v[vi(k,j,f-2,p.ny,p.nx)],r.v[vi(k,j,f-1,p.ny,p.nx)],r.v[vi(k,j,f,p.ny,p.nx)],r.v[vi(k,j,f+1,p.ny,p.nx)],v);return f5(r.v[vi(k,j,f-3,p.ny,p.nx)],r.v[vi(k,j,f-2,p.ny,p.nx)],r.v[vi(k,j,f-1,p.ny,p.nx)],r.v[vi(k,j,f,p.ny,p.nx)],r.v[vi(k,j,f+1,p.ny,p.nx)],r.v[vi(k,j,f+2,p.ny,p.nx)],v);}
static inline float xf_periodic(constant FluxDivVResources&r,constant FluxDivVParams&p,float v,int k,int j,int f){return f5(r.v[vi(k,j,per(f-3,p.nx),p.ny,p.nx)],r.v[vi(k,j,per(f-2,p.nx),p.ny,p.nx)],r.v[vi(k,j,per(f-1,p.nx),p.ny,p.nx)],r.v[vi(k,j,per(f,p.nx),p.ny,p.nx)],r.v[vi(k,j,per(f+1,p.nx),p.ny,p.nx)],r.v[vi(k,j,per(f+2,p.nx),p.ny,p.nx)],v);}
kernel void flux_div_v(constant FluxDivVResources&r [[buffer(0)]],constant FluxDivVParams&p [[buffer(1)]],uint tid [[thread_position_in_grid]]){
 int n=p.nz*(p.ny+1)*p.nx;if(tid>=uint(n))return;int i=int(tid%uint(p.nx)),j=int((tid/uint(p.nx))%uint(p.ny+1)),k=int(tid/uint(p.nx*(p.ny+1)));ulong out=vi(k,j,i,p.ny,p.nx);float mscale=p.has_msf?r.msf[ulong(j)*p.nx+i]:1.0f;
 if(p.open_x||p.open_y){
  bool yedge=p.open_y&&(j==0||j==p.ny);int rs=p.open_y?max(j-1,0):per(j-1,p.ny);int rn=p.open_y?min(j,p.ny-1):per(j,p.ny);int jv=p.open_y?j:rn;float tx=0.0f,ty=0.0f,th=0.0f;bool ax=false,ay=false;
  if(!yedge){float f[2];for(int side=0;side<2;side++){int m=side==0?rs:rn;float vel=0.5f*(r.rv[vi(k,m,i,p.ny,p.nx)]+r.rv[vi(k,m+1,i,p.ny,p.nx)]);f[side]=p.open_y?yf_open(r,p,vel,k,m,i):yf_periodic(r,p,vel,k,m,i);}ty=mscale*p.dy_inv*(f[1]-f[0]);
#if !GPUWM_WRF_EXACT_C_ADVECTION
   th+=-(f[1]-f[0])*p.dy_inv;
#endif
   ay=true;}
  if(!yedge){
   if(p.open_x&&(i==0||i==p.nx-1)){
    if(!p.spec){float uw0,uw1,ub,term;if(i==0){uw0=0.5f*(r.ru[ui(k,rs,0,p.ny,p.nx)]+r.ru[ui(k,rn,0,p.ny,p.nx)]);uw1=0.5f*(r.ru[ui(k,rs,1,p.ny,p.nx)]+r.ru[ui(k,rn,1,p.ny,p.nx)]);ub=min(uw0,0.0f);term=ub*(r.v[vi(k,jv,1,p.ny,p.nx)]-r.v[vi(k,jv,0,p.ny,p.nx)])+r.v[vi(k,jv,0,p.ny,p.nx)]*(uw1-uw0);}else{uw0=0.5f*(r.ru[ui(k,rs,p.nx-1,p.ny,p.nx)]+r.ru[ui(k,rn,p.nx-1,p.ny,p.nx)]);uw1=0.5f*(r.ru[ui(k,rs,p.nx,p.ny,p.nx)]+r.ru[ui(k,rn,p.nx,p.ny,p.nx)]);ub=max(uw1,0.0f);term=ub*(r.v[vi(k,jv,p.nx-1,p.ny,p.nx)]-r.v[vi(k,jv,p.nx-2,p.ny,p.nx)])+r.v[vi(k,jv,p.nx-1,p.ny,p.nx)]*(uw1-uw0);}tx=mscale*p.dx_inv*term;
#if !GPUWM_WRF_EXACT_C_ADVECTION
     th+=-p.dx_inv*term;
#endif
     ax=true;}
   }else{float f[2];for(int side=0;side<2;side++){int face=i+side;float vel=0.5f*(r.ru[ui(k,rs,face,p.ny,p.nx)]+r.ru[ui(k,rn,face,p.ny,p.nx)]);f[side]=p.open_x?xf_open(r,p,vel,k,jv,face):xf_periodic(r,p,vel,k,jv,face);}tx=mscale*p.dx_inv*(f[1]-f[0]);
#if !GPUWM_WRF_EXACT_C_ADVECTION
     th+=-(f[1]-f[0])*p.dx_inv;
#endif
     ax=true;}
  }
  float fz[2];float t=r.tend[out];if(ay)t=t-ty;if(ax)t=t-tx;
  if(!yedge){for(int side=0;side<2;side++){int kf=k+side;float vel=0.5f*(r.rw[wi(kf,rs,i,p.ny,p.nx)]+r.rw[wi(kf,rn,i,p.ny,p.nx)]);fz[side]=zhalf(r.v,vel,kf,jv,i,p.nz,p.ny,p.nx,r.fnm,r.fnp);}}
#if GPUWM_WRF_EXACT_C_ADVECTION
  if(!yedge)t=t-r.rdnw[k]*(fz[1]-fz[0]);r.tend[out]=t;
#else
  float horizontal=p.has_msf?mscale*th:th;r.tend[out]+=horizontal-(yedge?0.0f:(fz[1]-fz[0])*r.rdnw[k]);
#endif
  return;
 }
 int jc=per(j,p.ny),jm=per(j-1,p.ny);float fx[2],fy[2],fz[2];
 for(int side=0;side<2;side++){int face=i+side;float velx=0.5f*(r.ru[ui(k,jm,face,p.ny,p.nx)]+r.ru[ui(k,jc,face,p.ny,p.nx)]);fx[side]=xf_periodic(r,p,velx,k,jc,face);int m=side==0?jm:jc;float vely=0.5f*(r.rv[vi(k,m,i,p.ny,p.nx)]+r.rv[vi(k,m+1,i,p.ny,p.nx)]);fy[side]=yf_periodic(r,p,vely,k,m,i);int kf=k+side;float velz=0.5f*(r.rw[wi(kf,jm,i,p.ny,p.nx)]+r.rw[wi(kf,jc,i,p.ny,p.nx)]);fz[side]=zhalf(r.v,velz,kf,jc,i,p.nz,p.ny,p.nx,r.fnm,r.fnp);}
#if GPUWM_WRF_EXACT_C_ADVECTION
 float result=r.tend[out];result=result-(mscale*p.dy_inv)*(fy[1]-fy[0]);result=result-(mscale*p.dx_inv)*(fx[1]-fx[0]);result=result-r.rdnw[k]*(fz[1]-fz[0]);r.tend[out]=result;
#else
 if(p.has_msf)r.tend[out]+=mscale*(-(fx[1]-fx[0])*p.dx_inv-(fy[1]-fy[0])*p.dy_inv)-(fz[1]-fz[0])*r.rdnw[k];else r.tend[out]+=-(fx[1]-fx[0])*p.dx_inv-(fy[1]-fy[0])*p.dy_inv-(fz[1]-fz[0])*r.rdnw[k];
#endif
}
