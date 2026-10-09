#include <metal_stdlib>
#include "flux_div_u_vorder_bindings.metalh"
using namespace metal;
#ifndef GPUWM_WRF_EXACT
#define GPUWM_WRF_EXACT 0
#endif
#ifndef GPUWM_WRF_EXACT_C_ADVECTION
#define GPUWM_WRF_EXACT_C_ADVECTION GPUWM_WRF_EXACT
#endif
static inline ulong ui(int k,int j,int i,int ny,int nx){return (ulong(k)*ny+j)*(nx+1)+i;}
static inline ulong vi(int k,int j,int i,int ny,int nx){return (ulong(k)*(ny+1)+j)*nx+i;}
static inline ulong wi(int k,int j,int i,int ny,int nx){return (ulong(k)*ny+j)*nx+i;}
static inline int per(int i,int n){int r=i%n;return r<0?r+n:r;}
static inline float f5(float a,float b,float c,float d,float e,float f,float v){
#if GPUWM_WRF_EXACT_C_ADVECTION
 float center=(37.0f*(d+c)-8.0f*(e+b)+(f+a))/60.0f;
 float diss=copysign(1.0f,v)*((f-a)-5.0f*(e-b)+10.0f*(d-c))/60.0f;
 return v*(center-diss);
#else
 return (v*(37.0f*(d+c)-8.0f*(e+b)+(f+a))-abs(v)*(10.0f*(d-c)-5.0f*(e-b)+(f-a)))/60.0f;
#endif
}
static inline float f3(float a,float b,float c,float d,float v){
#if GPUWM_WRF_EXACT_C_ADVECTION
 float center=(7.0f*(c+b)-(d+a))/12.0f;
 float diss=copysign(1.0f,v)*((d-a)-3.0f*(c-b))/12.0f;
 return v*(center-diss);
#else
 return (v*(7.0f*(c+b)-(d+a))+abs(v)*(3.0f*(c-b)-(d-a)))/12.0f;
#endif
}
static inline float f3h(float a,float b,float c,float d,float v){
#if GPUWM_WRF_EXACT_C_ADVECTION
 float center=(7.0f*(c+b)-(d+a))/12.0f;
 float diss=copysign(1.0f,v)*((d-a)-3.0f*(c-b))/12.0f;
 return v*(center+diss);
#else
 return (v*(7.0f*(c+b)-(d+a))-abs(v)*(3.0f*(c-b)-(d-a)))/12.0f;
#endif
}
static inline float flux5v(float a,float b,float c,float d,float e,float f,float v){
#if GPUWM_WRF_EXACT_C_ADVECTION
 float center=(37.0f*(d+c)-8.0f*(e+b)+(f+a))/60.0f;
 float diss=copysign(1.0f,v)*((f-a)-5.0f*(e-b)+10.0f*(d-c))/60.0f;
 return v*(center+diss);
#else
 return (v*(37.0f*(d+c)-8.0f*(e+b)+(f+a))+abs(v)*(10.0f*(d-c)-5.0f*(e-b)+(f-a)))/60.0f;
#endif
}
static inline float f2(float a,float b,float v){return 0.5f*v*(a+b);}
static inline float zhalf(device const float*q,float v,int kf,int j,int i,int nz,int ny,int nx,device const float*fnm,device const float*fnp,int vorder){
 if(kf==0||kf==nz)return 0.0f;
 if(kf==1||kf==nz-1)return v*(fnm[kf]*q[ui(kf,j,i,ny,nx)]+fnp[kf]*q[ui(kf-1,j,i,ny,nx)]);
 if(vorder==5&&kf>=3&&kf<=nz-3)return flux5v(q[ui(kf-3,j,i,ny,nx)],q[ui(kf-2,j,i,ny,nx)],q[ui(kf-1,j,i,ny,nx)],q[ui(kf,j,i,ny,nx)],q[ui(kf+1,j,i,ny,nx)],q[ui(kf+2,j,i,ny,nx)],v);return f3(q[ui(kf-2,j,i,ny,nx)],q[ui(kf-1,j,i,ny,nx)],q[ui(kf,j,i,ny,nx)],q[ui(kf+1,j,i,ny,nx)],v);
}
static inline float xf_open(constant FluxDivUVorderResources&r,constant FluxDivUVorderParams&p,float v,int k,int j,int m){
 int d=min(m,p.nx-1-m);
 if(d==0){float a=r.u[ui(k,j,m,p.ny,p.nx)],b=r.u[ui(k,j,m+1,p.ny,p.nx)];if(p.spec){if(m==0){if(b<0.0f)a=b;}else if(a>0.0f)b=a;}return f2(a,b,v);}
 if(d==1)return f3h(r.u[ui(k,j,m-1,p.ny,p.nx)],r.u[ui(k,j,m,p.ny,p.nx)],r.u[ui(k,j,m+1,p.ny,p.nx)],r.u[ui(k,j,m+2,p.ny,p.nx)],v);
 return f5(r.u[ui(k,j,m-2,p.ny,p.nx)],r.u[ui(k,j,m-1,p.ny,p.nx)],r.u[ui(k,j,m,p.ny,p.nx)],r.u[ui(k,j,m+1,p.ny,p.nx)],r.u[ui(k,j,m+2,p.ny,p.nx)],r.u[ui(k,j,m+3,p.ny,p.nx)],v);
}
static inline float xf_periodic(constant FluxDivUVorderResources&r,constant FluxDivUVorderParams&p,float v,int k,int j,int m){
 return f5(r.u[ui(k,j,per(m-2,p.nx),p.ny,p.nx)],r.u[ui(k,j,per(m-1,p.nx),p.ny,p.nx)],r.u[ui(k,j,per(m,p.nx),p.ny,p.nx)],r.u[ui(k,j,per(m+1,p.nx),p.ny,p.nx)],r.u[ui(k,j,per(m+2,p.nx),p.ny,p.nx)],r.u[ui(k,j,per(m+3,p.nx),p.ny,p.nx)],v);
}
static inline float yf_open(constant FluxDivUVorderResources&r,constant FluxDivUVorderParams&p,float v,int k,int g,int i){
 if(g<=0||g>=p.ny)return 0.0f;int d=min(g,p.ny-g);
 if(d==1)return f2(r.u[ui(k,g-1,i,p.ny,p.nx)],r.u[ui(k,g,i,p.ny,p.nx)],v);
 if(d==2)return f3h(r.u[ui(k,g-2,i,p.ny,p.nx)],r.u[ui(k,g-1,i,p.ny,p.nx)],r.u[ui(k,g,i,p.ny,p.nx)],r.u[ui(k,g+1,i,p.ny,p.nx)],v);
 return f5(r.u[ui(k,g-3,i,p.ny,p.nx)],r.u[ui(k,g-2,i,p.ny,p.nx)],r.u[ui(k,g-1,i,p.ny,p.nx)],r.u[ui(k,g,i,p.ny,p.nx)],r.u[ui(k,g+1,i,p.ny,p.nx)],r.u[ui(k,g+2,i,p.ny,p.nx)],v);
}
static inline float yf_periodic(constant FluxDivUVorderResources&r,constant FluxDivUVorderParams&p,float v,int k,int g,int i){
 return f5(r.u[ui(k,per(g-3,p.ny),i,p.ny,p.nx)],r.u[ui(k,per(g-2,p.ny),i,p.ny,p.nx)],r.u[ui(k,per(g-1,p.ny),i,p.ny,p.nx)],r.u[ui(k,per(g,p.ny),i,p.ny,p.nx)],r.u[ui(k,per(g+1,p.ny),i,p.ny,p.nx)],r.u[ui(k,per(g+2,p.ny),i,p.ny,p.nx)],v);
}
kernel void flux_div_u_vorder(constant FluxDivUVorderResources&r [[buffer(0)]],constant FluxDivUVorderParams&p [[buffer(1)]],uint tid [[thread_position_in_grid]]){
 int n=p.nz*p.ny*(p.nx+1);if(tid>=uint(n))return;
 int i=int(tid%uint(p.nx+1)),j=int((tid/uint(p.nx+1))%uint(p.ny)),k=int(tid/uint((p.nx+1)*p.ny));
 ulong out=ui(k,j,i,p.ny,p.nx);float mscale=p.has_msf?r.msf[ulong(j)*(p.nx+1)+i]:1.0f;
 if(p.open_x||p.open_y){
  bool xedge=p.open_x&&(i==0||i==p.nx);
  int cl=p.open_x?max(i-1,0):per(i-1,p.nx);
  int cr=p.open_x?min(i,p.nx-1):per(i,p.nx);
  int iu=p.open_x?i:cr;
  float tx=0.0f,ty=0.0f,th=0.0f;
  bool ax=false,ay=false;

  if(!xedge){
   float f[2];
   for(int side=0;side<2;side++){
    int m=side==0?cl:cr;
    float vel=0.5f*(r.ru[ui(k,j,m,p.ny,p.nx)]+r.ru[ui(k,j,m+1,p.ny,p.nx)]);
    f[side]=p.open_x?xf_open(r,p,vel,k,j,m):xf_periodic(r,p,vel,k,j,m);
   }
   tx=mscale*p.dx_inv*(f[1]-f[0]);
#if !GPUWM_WRF_EXACT_C_ADVECTION
   th+=-(f[1]-f[0])*p.dx_inv;
#endif
   ax=true;
  }

  if(!xedge){
   if(p.open_y&&(j==0||j==p.ny-1)){
    if(!p.spec){
     float ub,dsum,term;
     if(j==0){
      float vw=0.5f*(r.rv[vi(k,0,cl,p.ny,p.nx)]+r.rv[vi(k,0,cr,p.ny,p.nx)]);
      ub=min(vw,0.0f);
      dsum=(r.rv[vi(k,1,cl,p.ny,p.nx)]-r.rv[vi(k,0,cl,p.ny,p.nx)])
          +(r.rv[vi(k,1,cr,p.ny,p.nx)]-r.rv[vi(k,0,cr,p.ny,p.nx)]);
      term=ub*(r.u[ui(k,1,iu,p.ny,p.nx)]-r.u[ui(k,0,iu,p.ny,p.nx)])
          +0.5f*r.u[ui(k,0,iu,p.ny,p.nx)]*dsum;
     }else{
      float vw=0.5f*(r.rv[vi(k,p.ny,cl,p.ny,p.nx)]+r.rv[vi(k,p.ny,cr,p.ny,p.nx)]);
      ub=max(vw,0.0f);
      dsum=(r.rv[vi(k,p.ny,cl,p.ny,p.nx)]-r.rv[vi(k,p.ny-1,cl,p.ny,p.nx)])
          +(r.rv[vi(k,p.ny,cr,p.ny,p.nx)]-r.rv[vi(k,p.ny-1,cr,p.ny,p.nx)]);
      term=ub*(r.u[ui(k,p.ny-1,iu,p.ny,p.nx)]-r.u[ui(k,p.ny-2,iu,p.ny,p.nx)])
          +0.5f*r.u[ui(k,p.ny-1,iu,p.ny,p.nx)]*dsum;
     }
     ty=mscale*p.dy_inv*term;
#if !GPUWM_WRF_EXACT_C_ADVECTION
     th+=-p.dy_inv*term;
#endif
     ay=true;
    }
   }else{
    float f[2];
    for(int side=0;side<2;side++){
     int g=j+side;
     float vel=0.5f*(r.rv[vi(k,g,cl,p.ny,p.nx)]+r.rv[vi(k,g,cr,p.ny,p.nx)]);
     f[side]=p.open_y?yf_open(r,p,vel,k,g,iu):yf_periodic(r,p,vel,k,g,iu);
    }
    ty=mscale*p.dy_inv*(f[1]-f[0]);
#if !GPUWM_WRF_EXACT_C_ADVECTION
    th+=-(f[1]-f[0])*p.dy_inv;
#endif
    ay=true;
   }
  }

  float fz[2];
  bool apply_z=!(xedge&&p.open_y);
  if(apply_z){
   for(int side=0;side<2;side++){
    int kf=k+side;float vel;
    if(p.open_x&&i==0)vel=r.rw[wi(kf,j,0,p.ny,p.nx)];
    else if(p.open_x&&i==p.nx)vel=r.rw[wi(kf,j,p.nx-1,p.ny,p.nx)];
    else vel=0.5f*(r.rw[wi(kf,j,cl,p.ny,p.nx)]+r.rw[wi(kf,j,cr,p.ny,p.nx)]);
    fz[side]=zhalf(r.u,vel,kf,j,iu,p.nz,p.ny,p.nx,r.fnm,r.fnp,p.vorder);
   }
  }
#if GPUWM_WRF_EXACT_C_ADVECTION
  float result=r.tend[out];if(ay)result=result-ty;if(ax)result=result-tx;
  if(apply_z)result=result-r.rdnw[k]*(fz[1]-fz[0]);
  r.tend[out]=result;
#else
  float horizontal=p.has_msf?mscale*th:th;
  if(apply_z)r.tend[out]+=horizontal-(fz[1]-fz[0])*r.rdnw[k];
  else r.tend[out]+=horizontal;
#endif
  return;
 }
 int ic=per(i,p.nx),im1c=per(i-1,p.nx);float fx[2],fy[2],fz[2];
 for(int s=0;s<2;s++){int m=s==0?im1c:ic;float velx=0.5f*(r.ru[ui(k,j,m,p.ny,p.nx)]+r.ru[ui(k,j,m+1,p.ny,p.nx)]);fx[s]=xf_periodic(r,p,velx,k,j,m);int g=j+s;float vely=0.5f*(r.rv[vi(k,g,im1c,p.ny,p.nx)]+r.rv[vi(k,g,ic,p.ny,p.nx)]);fy[s]=yf_periodic(r,p,vely,k,g,ic);int kf=k+s;float velz=0.5f*(r.rw[wi(kf,j,im1c,p.ny,p.nx)]+r.rw[wi(kf,j,ic,p.ny,p.nx)]);fz[s]=zhalf(r.u,velz,kf,j,ic,p.nz,p.ny,p.nx,r.fnm,r.fnp,p.vorder);}
#if GPUWM_WRF_EXACT_C_ADVECTION
 float result=r.tend[out];result=result-(mscale*p.dy_inv)*(fy[1]-fy[0]);result=result-(mscale*p.dx_inv)*(fx[1]-fx[0]);result=result-r.rdnw[k]*(fz[1]-fz[0]);r.tend[out]=result;
#else
 if(p.has_msf)r.tend[out]+=mscale*(-(fx[1]-fx[0])*p.dx_inv-(fy[1]-fy[0])*p.dy_inv)-(fz[1]-fz[0])*r.rdnw[k];
 else r.tend[out]+=-(fx[1]-fx[0])*p.dx_inv-(fy[1]-fy[0])*p.dy_inv-(fz[1]-fz[0])*r.rdnw[k];
#endif
}
