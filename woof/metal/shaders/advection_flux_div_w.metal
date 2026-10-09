#include <metal_stdlib>
#include "flux_div_w_bindings.metalh"
using namespace metal;
#ifndef GPUWM_WRF_EXACT
#define GPUWM_WRF_EXACT 0
#endif
#ifndef GPUWM_WRF_EXACT_C_ADVECTION
#define GPUWM_WRF_EXACT_C_ADVECTION GPUWM_WRF_EXACT
#endif
static inline ulong wi(int k,int j,int i,int ny,int nx){return (ulong(k)*ny+j)*nx+i;}
static inline ulong ui(int k,int j,int i,int ny,int nx){return (ulong(k)*ny+j)*(nx+1)+i;}
static inline ulong vi(int k,int j,int i,int ny,int nx){return (ulong(k)*(ny+1)+j)*nx+i;}
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
static inline float xf_open(constant FluxDivWResources&r,constant FluxDivWParams&p,float v,int k,int j,int f){
 if(f<=0||f>=p.nx)return 0.0f;int d=min(f,p.nx-f);if(d==1)return f2(r.w[wi(k,j,f-1,p.ny,p.nx)],r.w[wi(k,j,f,p.ny,p.nx)],v);if(d==2)return f3h(r.w[wi(k,j,f-2,p.ny,p.nx)],r.w[wi(k,j,f-1,p.ny,p.nx)],r.w[wi(k,j,f,p.ny,p.nx)],r.w[wi(k,j,f+1,p.ny,p.nx)],v);return f5(r.w[wi(k,j,f-3,p.ny,p.nx)],r.w[wi(k,j,f-2,p.ny,p.nx)],r.w[wi(k,j,f-1,p.ny,p.nx)],r.w[wi(k,j,f,p.ny,p.nx)],r.w[wi(k,j,f+1,p.ny,p.nx)],r.w[wi(k,j,f+2,p.ny,p.nx)],v);
}
static inline float xf_periodic(constant FluxDivWResources&r,constant FluxDivWParams&p,float v,int k,int j,int f){return f5(r.w[wi(k,j,per(f-3,p.nx),p.ny,p.nx)],r.w[wi(k,j,per(f-2,p.nx),p.ny,p.nx)],r.w[wi(k,j,per(f-1,p.nx),p.ny,p.nx)],r.w[wi(k,j,per(f,p.nx),p.ny,p.nx)],r.w[wi(k,j,per(f+1,p.nx),p.ny,p.nx)],r.w[wi(k,j,per(f+2,p.nx),p.ny,p.nx)],v);}
static inline float yf_open(constant FluxDivWResources&r,constant FluxDivWParams&p,float v,int k,int g,int i){if(g<=0||g>=p.ny)return 0.0f;int d=min(g,p.ny-g);if(d==1)return f2(r.w[wi(k,g-1,i,p.ny,p.nx)],r.w[wi(k,g,i,p.ny,p.nx)],v);if(d==2)return f3h(r.w[wi(k,g-2,i,p.ny,p.nx)],r.w[wi(k,g-1,i,p.ny,p.nx)],r.w[wi(k,g,i,p.ny,p.nx)],r.w[wi(k,g+1,i,p.ny,p.nx)],v);return f5(r.w[wi(k,g-3,i,p.ny,p.nx)],r.w[wi(k,g-2,i,p.ny,p.nx)],r.w[wi(k,g-1,i,p.ny,p.nx)],r.w[wi(k,g,i,p.ny,p.nx)],r.w[wi(k,g+1,i,p.ny,p.nx)],r.w[wi(k,g+2,i,p.ny,p.nx)],v);}
static inline float yf_periodic(constant FluxDivWResources&r,constant FluxDivWParams&p,float v,int k,int g,int i){return f5(r.w[wi(k,per(g-3,p.ny),i,p.ny,p.nx)],r.w[wi(k,per(g-2,p.ny),i,p.ny,p.nx)],r.w[wi(k,per(g-1,p.ny),i,p.ny,p.nx)],r.w[wi(k,per(g,p.ny),i,p.ny,p.nx)],r.w[wi(k,per(g+1,p.ny),i,p.ny,p.nx)],r.w[wi(k,per(g+2,p.ny),i,p.ny,p.nx)],v);}
static inline float lid_velocity(device const float*flux,int j,int i,int nz,int ny,int nx,device const float*fnm,device const float*fnp){return (2.0f-fnm[nz-1])*flux[ui(nz-1,j,i,ny,nx)]-fnp[nz-1]*flux[ui(nz-2,j,i,ny,nx)];}
static inline float lid_velocity_v(device const float*flux,int j,int i,int nz,int ny,int nx,device const float*fnm,device const float*fnp){return (2.0f-fnm[nz-1])*flux[vi(nz-1,j,i,ny,nx)]-fnp[nz-1]*flux[vi(nz-2,j,i,ny,nx)];}
static inline float zflux(constant FluxDivWResources&r,constant FluxDivWParams&p,int k,int side,int j,int i){int m=k-1+side;float vel=0.5f*(r.rw[wi(m,j,i,p.ny,p.nx)]+r.rw[wi(m+1,j,i,p.ny,p.nx)]);if(m==0||m==p.nz-1)return f2(r.w[wi(m,j,i,p.ny,p.nx)],r.w[wi(m+1,j,i,p.ny,p.nx)],vel);return f3(r.w[wi(m-1,j,i,p.ny,p.nx)],r.w[wi(m,j,i,p.ny,p.nx)],r.w[wi(m+1,j,i,p.ny,p.nx)],r.w[wi(m+2,j,i,p.ny,p.nx)],vel);}
kernel void flux_div_w(constant FluxDivWResources&r [[buffer(0)]],constant FluxDivWParams&p [[buffer(1)]],uint tid [[thread_position_in_grid]]){
 int n=(p.nz+1)*p.ny*p.nx;if(tid>=uint(n))return;int i=int(tid%uint(p.nx)),j=int((tid/uint(p.nx))%uint(p.ny)),k=int(tid/uint(p.nx*p.ny));if(k==0||(k==p.nz&&p.nz<2))return;
 float mscale=p.has_msf?r.msf[ulong(j)*p.nx+i]:1.0f;ulong out=wi(k,j,i,p.ny,p.nx);
 if(k==p.nz){
  float tx=0.0f,ty=0.0f,th=0.0f,tb=0.0f;bool ax=false,ay=false;
  if(p.open_x&&(i==0||i==p.nx-1)){
   if(!p.spec){int f0=i==0?0:p.nx-1;float r0=lid_velocity(r.ru,j,f0,p.nz,p.ny,p.nx,r.fnm,r.fnp),r1=lid_velocity(r.ru,j,f0+1,p.nz,p.ny,p.nx,r.fnm,r.fnp);float ub=i==0?min(0.5f*(r0+r1),0.0f):max(0.5f*(r0+r1),0.0f);float dq=i==0?r.w[wi(k,j,1,p.ny,p.nx)]-r.w[wi(k,j,0,p.ny,p.nx)]:r.w[wi(k,j,p.nx-1,p.ny,p.nx)]-r.w[wi(k,j,p.nx-2,p.ny,p.nx)];tx=p.dx_inv*(ub*dq+r.w[out]*(r1-r0));
#if !GPUWM_WRF_EXACT_C_ADVECTION
    tb+=-p.dx_inv*(ub*dq+r.w[out]*(r1-r0));
#endif
   }
  }else{float r0=lid_velocity(r.ru,j,i,p.nz,p.ny,p.nx,r.fnm,r.fnp),r1=lid_velocity(r.ru,j,i+1,p.nz,p.ny,p.nx,r.fnm,r.fnp);float f0=p.open_x?xf_open(r,p,r0,k,j,i):xf_periodic(r,p,r0,k,j,i);float f1=p.open_x?xf_open(r,p,r1,k,j,i+1):xf_periodic(r,p,r1,k,j,i+1);tx=(mscale*p.dx_inv)*(f1-f0);
#if !GPUWM_WRF_EXACT_C_ADVECTION
   th+=-(f1-f0)*p.dx_inv;
#endif
   ax=true;}
  if(p.open_y&&(j==0||j==p.ny-1)){
   if(!p.spec){int g0=j==0?0:p.ny-1;float r0=lid_velocity_v(r.rv,g0,i,p.nz,p.ny,p.nx,r.fnm,r.fnp),r1=lid_velocity_v(r.rv,g0+1,i,p.nz,p.ny,p.nx,r.fnm,r.fnp);float vb=j==0?min(0.5f*(r0+r1),0.0f):max(0.5f*(r0+r1),0.0f);float dq=j==0?r.w[wi(k,1,i,p.ny,p.nx)]-r.w[wi(k,0,i,p.ny,p.nx)]:r.w[wi(k,p.ny-1,i,p.ny,p.nx)]-r.w[wi(k,p.ny-2,i,p.ny,p.nx)];ty=p.dy_inv*(vb*dq+r.w[out]*(r1-r0));
#if !GPUWM_WRF_EXACT_C_ADVECTION
    tb+=-p.dy_inv*(vb*dq+r.w[out]*(r1-r0));
#endif
   }
  }else{float r0=lid_velocity_v(r.rv,j,i,p.nz,p.ny,p.nx,r.fnm,r.fnp),r1=lid_velocity_v(r.rv,j+1,i,p.nz,p.ny,p.nx,r.fnm,r.fnp);float f0=p.open_y?yf_open(r,p,r0,k,j,i):yf_periodic(r,p,r0,k,j,i);float f1=p.open_y?yf_open(r,p,r1,k,j+1,i):yf_periodic(r,p,r1,k,j+1,i);ty=(mscale*p.dy_inv)*(f1-f0);
#if !GPUWM_WRF_EXACT_C_ADVECTION
   th+=-(f1-f0)*p.dy_inv;
#endif
   ay=true;}
  float top_flux=0.25f*(r.rw[out]+r.rw[wi(k-1,j,i,p.ny,p.nx)])*(r.w[out]+r.w[wi(k-1,j,i,p.ny,p.nx)]);
#if GPUWM_WRF_EXACT_C_ADVECTION
  float result=r.tend[out];if(ay)result=result-ty;if(ax)result=result-tx;r.tend[out]=result+2.0f*r.rdn[p.nz-1]*top_flux;
#else
  float horizontal=p.has_msf?mscale*th+tb:th+tb;r.tend[out]+=horizontal+2.0f*r.rdn[p.nz-1]*top_flux;
#endif
  return;
 }
 if(p.open_x||p.open_y){
  float tx=0.0f,ty=0.0f,th=0.0f,tb=0.0f;bool ax=false,ay=false;
  if(p.open_x&&(i==0||i==p.nx-1)){
   if(!p.spec){
    float ruw0,ruw1,ub,term;
    if(i==0){
     ruw0=r.fnm[k]*r.ru[ui(k,j,0,p.ny,p.nx)]+r.fnp[k]*r.ru[ui(k-1,j,0,p.ny,p.nx)];
     ruw1=r.fnm[k]*r.ru[ui(k,j,1,p.ny,p.nx)]+r.fnp[k]*r.ru[ui(k-1,j,1,p.ny,p.nx)];
     ub=min(0.5f*(ruw0+ruw1),0.0f);
     term=ub*(r.w[wi(k,j,1,p.ny,p.nx)]-r.w[wi(k,j,0,p.ny,p.nx)])+r.w[wi(k,j,0,p.ny,p.nx)]*(ruw1-ruw0);
    }else{
     ruw0=r.fnm[k]*r.ru[ui(k,j,p.nx-1,p.ny,p.nx)]+r.fnp[k]*r.ru[ui(k-1,j,p.nx-1,p.ny,p.nx)];
     ruw1=r.fnm[k]*r.ru[ui(k,j,p.nx,p.ny,p.nx)]+r.fnp[k]*r.ru[ui(k-1,j,p.nx,p.ny,p.nx)];
     ub=max(0.5f*(ruw0+ruw1),0.0f);
     term=ub*(r.w[wi(k,j,p.nx-1,p.ny,p.nx)]-r.w[wi(k,j,p.nx-2,p.ny,p.nx)])+r.w[wi(k,j,p.nx-1,p.ny,p.nx)]*(ruw1-ruw0);
    }
    tx=p.dx_inv*term;
#if !GPUWM_WRF_EXACT_C_ADVECTION
    tb+=-p.dx_inv*term;
#endif
    ax=true;
   }
  }else{
   float f[2];
   for(int side=0;side<2;side++){
    int face=i+side;
    float vel=r.fnm[k]*r.ru[ui(k,j,face,p.ny,p.nx)]+r.fnp[k]*r.ru[ui(k-1,j,face,p.ny,p.nx)];
    f[side]=p.open_x?xf_open(r,p,vel,k,j,face):xf_periodic(r,p,vel,k,j,face);
   }
   tx=(mscale*p.dx_inv)*(f[1]-f[0]);
#if !GPUWM_WRF_EXACT_C_ADVECTION
   th+=-(f[1]-f[0])*p.dx_inv;
#endif
   ax=true;
  }
  if(p.open_y&&(j==0||j==p.ny-1)){
   if(!p.spec){
    float rvw0,rvw1,vb,term;
    if(j==0){
     rvw0=r.fnm[k]*r.rv[vi(k,0,i,p.ny,p.nx)]+r.fnp[k]*r.rv[vi(k-1,0,i,p.ny,p.nx)];
     rvw1=r.fnm[k]*r.rv[vi(k,1,i,p.ny,p.nx)]+r.fnp[k]*r.rv[vi(k-1,1,i,p.ny,p.nx)];
     vb=min(0.5f*(rvw0+rvw1),0.0f);
     term=vb*(r.w[wi(k,1,i,p.ny,p.nx)]-r.w[wi(k,0,i,p.ny,p.nx)])+r.w[wi(k,0,i,p.ny,p.nx)]*(rvw1-rvw0);
    }else{
     rvw0=r.fnm[k]*r.rv[vi(k,p.ny-1,i,p.ny,p.nx)]+r.fnp[k]*r.rv[vi(k-1,p.ny-1,i,p.ny,p.nx)];
     rvw1=r.fnm[k]*r.rv[vi(k,p.ny,i,p.ny,p.nx)]+r.fnp[k]*r.rv[vi(k-1,p.ny,i,p.ny,p.nx)];
     vb=max(0.5f*(rvw0+rvw1),0.0f);
     term=vb*(r.w[wi(k,p.ny-1,i,p.ny,p.nx)]-r.w[wi(k,p.ny-2,i,p.ny,p.nx)])+r.w[wi(k,p.ny-1,i,p.ny,p.nx)]*(rvw1-rvw0);
    }
    ty=p.dy_inv*term;
#if !GPUWM_WRF_EXACT_C_ADVECTION
    tb+=-p.dy_inv*term;
#endif
    ay=true;
   }
  }else{
   float f[2];
   for(int side=0;side<2;side++){
    int row=j+side;
    float vel=r.fnm[k]*r.rv[vi(k,row,i,p.ny,p.nx)]+r.fnp[k]*r.rv[vi(k-1,row,i,p.ny,p.nx)];
    f[side]=p.open_y?yf_open(r,p,vel,k,row,i):yf_periodic(r,p,vel,k,row,i);
   }
   ty=(mscale*p.dy_inv)*(f[1]-f[0]);
#if !GPUWM_WRF_EXACT_C_ADVECTION
   th+=-(f[1]-f[0])*p.dy_inv;
#endif
   ay=true;
  }
  float fz0=zflux(r,p,k,0,j,i),fz1=zflux(r,p,k,1,j,i);
#if GPUWM_WRF_EXACT_C_ADVECTION
  float result=r.tend[out];if(ay)result=result-ty;if(ax)result=result-tx;result=result-r.rdn[k]*(fz1-fz0);r.tend[out]=result;
#else
  float horizontal=p.has_msf?mscale*th+tb:th+tb;r.tend[out]+=horizontal-(fz1-fz0)*r.rdn[k];
#endif
  return;
 }
 float fx[2],fy[2],fz[2];
 for(int side=0;side<2;side++){
  int face=i+side;
  float velx=r.fnm[k]*r.ru[ui(k,j,face,p.ny,p.nx)]+r.fnp[k]*r.ru[ui(k-1,j,face,p.ny,p.nx)];
  fx[side]=xf_periodic(r,p,velx,k,j,face);
  int row=j+side;
  float vely=r.fnm[k]*r.rv[vi(k,row,i,p.ny,p.nx)]+r.fnp[k]*r.rv[vi(k-1,row,i,p.ny,p.nx)];
  fy[side]=yf_periodic(r,p,vely,k,row,i);
  fz[side]=zflux(r,p,k,side,j,i);
 }
#if GPUWM_WRF_EXACT_C_ADVECTION
 float result=r.tend[out];result=result-(mscale*p.dy_inv)*(fy[1]-fy[0]);result=result-(mscale*p.dx_inv)*(fx[1]-fx[0]);result=result-r.rdn[k]*(fz[1]-fz[0]);r.tend[out]=result;
#else
 if(p.has_msf)r.tend[out]+=mscale*(-(fx[1]-fx[0])*p.dx_inv-(fy[1]-fy[0])*p.dy_inv)-(fz[1]-fz[0])*r.rdn[k];
 else r.tend[out]+=-(fx[1]-fx[0])*p.dx_inv-(fy[1]-fy[0])*p.dy_inv-(fz[1]-fz[0])*r.rdn[k];
#endif
}
