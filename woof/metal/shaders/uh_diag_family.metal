#include <metal_stdlib>
#include "uh_columns_bindings.metalh"
#include "uh_smooth_max_bindings.metalh"
#include "uh_boundary_copy_bindings.metalh"
using namespace metal;

inline ulong i3(int k,int j,int i,int ny,int nx) { return (ulong(k)*ny+j)*nx+i; }
inline ulong i3u(int k,int j,int i,int ny,int nx) { return (ulong(k)*ny+j)*(nx+1)+i; }
inline ulong i3v(int k,int j,int i,int ny,int nx) { return (ulong(k)*(ny+1)+j)*nx+i; }
inline float z_w(device const float* ph,device const float* phb,int b3,int k,int j,int i,int ny,int nx) {
  float base=b3?phb[i3(k,j,i,ny,nx)]:phb[k]; return (ph[i3(k,j,i,ny,nx)]+base)/9.81f;
}
inline float rdzw(device const float* ph,device const float* phb,int b3,int k,int j,int i,int ny,int nx) { return 1.0f/(z_w(ph,phb,b3,k+1,j,i,ny,nx)-z_w(ph,phb,b3,k,j,i,ny,nx)); }
inline float zx(device const float* ph,device const float* phb,int b3,float rdx,int k,int j,int is,int ny,int nx) {
 if(is==0||is==nx)return 0.0f;
 float a=b3?phb[i3(k,j,is,ny,nx)]:phb[k], b=b3?phb[i3(k,j,is-1,ny,nx)]:phb[k];
 float p1=rdx*(a-b)/9.81f, p2=rdx*(ph[i3(k,j,is,ny,nx)]-ph[i3(k,j,is-1,ny,nx)])/9.81f; return p1+p2;
}
inline float zy(device const float* ph,device const float* phb,int b3,float rdy,int k,int js,int i,int ny,int nx) {
 if(js==0||js==ny)return 0.0f;
 float a=b3?phb[i3(k,js,i,ny,nx)]:phb[k], b=b3?phb[i3(k,js-1,i,ny,nx)]:phb[k];
 float p1=rdy*(a-b)/9.81f, p2=rdy*(ph[i3(k,js,i,ny,nx)]-ph[i3(k,js-1,i,ny,nx)])/9.81f; return p1+p2;
}
inline float hv(device const float*v,device const float*msfv,int k,int j,int i,int ny,int nx){return v[i3v(k,j,i,ny,nx)]/msfv[j*nx+i];}
inline float hu(device const float*u,device const float*msfu,int k,int j,int i,int ny,int nx){return u[i3u(k,j,i,ny,nx)]/msfu[j*(nx+1)+i];}
inline float havg_v(device const float*v,device const float*msfv,device const float*fnm,device const float*fnp,float c1,float c2,float c3,float t1,float t2,int k,int j,int i,int nz,int ny,int nx){
 if(k==0){float a=c1*hv(v,msfv,0,j,i-1,ny,nx);a=a+c2*hv(v,msfv,1,j,i-1,ny,nx);a=a+c3*hv(v,msfv,2,j,i-1,ny,nx);a=a+c1*hv(v,msfv,0,j,i,ny,nx);a=a+c2*hv(v,msfv,1,j,i,ny,nx);a=a+c3*hv(v,msfv,2,j,i,ny,nx);return 0.5f*a;}
 if(k==nz){float a=t1*(hv(v,msfv,nz-1,j,i,ny,nx)+hv(v,msfv,nz-1,j,i-1,ny,nx));float b=t2*(hv(v,msfv,nz-2,j,i,ny,nx)+hv(v,msfv,nz-2,j,i-1,ny,nx));return 0.5f*(a+b);}
 float a=fnm[k]*(hv(v,msfv,k,j,i-1,ny,nx)+hv(v,msfv,k,j,i,ny,nx));float b=fnp[k]*(hv(v,msfv,k-1,j,i-1,ny,nx)+hv(v,msfv,k-1,j,i,ny,nx));return 0.5f*(a+b);
}
inline float havg_u(device const float*u,device const float*msfu,device const float*fnm,device const float*fnp,float c1,float c2,float c3,float t1,float t2,int k,int j,int i,int nz,int ny,int nx){
 if(k==0){float a=c1*hu(u,msfu,0,j-1,i,ny,nx);a=a+c2*hu(u,msfu,1,j-1,i,ny,nx);a=a+c3*hu(u,msfu,2,j-1,i,ny,nx);a=a+c1*hu(u,msfu,0,j,i,ny,nx);a=a+c2*hu(u,msfu,1,j,i,ny,nx);a=a+c3*hu(u,msfu,2,j,i,ny,nx);return 0.5f*a;}
 if(k==nz){float a=t1*(hu(u,msfu,nz-1,j-1,i,ny,nx)+hu(u,msfu,nz-1,j,i,ny,nx));float b=t2*(hu(u,msfu,nz-2,j-1,i,ny,nx)+hu(u,msfu,nz-2,j,i,ny,nx));return 0.5f*(a+b);}
 float a=fnm[k]*(hu(u,msfu,k,j-1,i,ny,nx)+hu(u,msfu,k,j,i,ny,nx));float b=fnp[k]*(hu(u,msfu,k-1,j-1,i,ny,nx)+hu(u,msfu,k-1,j,i,ny,nx));return 0.5f*(a+b);
}
inline float rvort(device const float*u,device const float*v,device const float*ph,device const float*phb,int b3,device const float*msfu,device const float*msfv,device const float*fnm,device const float*fnp,float c1,float c2,float c3,float t1,float t2,float rdx,float rdy,float mm,int k,int j,int i,int nz,int ny,int nx){
 float xz=0.25f*((zx(ph,phb,b3,rdx,k,j-1,i,ny,nx)+zx(ph,phb,b3,rdx,k,j,i,ny,nx))+zx(ph,phb,b3,rdx,k+1,j-1,i,ny,nx)+zx(ph,phb,b3,rdx,k+1,j,i,ny,nx));
 float dz=(((rdzw(ph,phb,b3,k,j,i,ny,nx)+rdzw(ph,phb,b3,k,j-1,i,ny,nx))+rdzw(ph,phb,b3,k,j-1,i-1,ny,nx))+rdzw(ph,phb,b3,k,j,i-1,ny,nx));
 float dh=havg_v(v,msfv,fnm,fnp,c1,c2,c3,t1,t2,k+1,j,i,nz,ny,nx)-havg_v(v,msfv,fnm,fnp,c1,c2,c3,t1,t2,k,j,i,nz,ny,nx);
 float tx=((dh*0.25f)*xz)*dz, dvd=hv(v,msfv,k,j,i,ny,nx)-hv(v,msfv,k,j,i-1,ny,nx);float rx=mm*(rdx*dvd-tx);
 float yz=0.25f*((zy(ph,phb,b3,rdy,k,j,i-1,ny,nx)+zy(ph,phb,b3,rdy,k,j,i,ny,nx))+zy(ph,phb,b3,rdy,k+1,j,i-1,ny,nx)+zy(ph,phb,b3,rdy,k+1,j,i,ny,nx));
 dz=((rdzw(ph,phb,b3,k,j,i,ny,nx)+rdzw(ph,phb,b3,k,j,i-1,ny,nx))+rdzw(ph,phb,b3,k,j-1,i-1,ny,nx))+rdzw(ph,phb,b3,k,j-1,i,ny,nx);
 dh=havg_u(u,msfu,fnm,fnp,c1,c2,c3,t1,t2,k+1,j,i,nz,ny,nx)-havg_u(u,msfu,fnm,fnp,c1,c2,c3,t1,t2,k,j,i,nz,ny,nx);
 float ty=((dh*0.25f)*yz)*dz, duy=hu(u,msfu,k,j,i,ny,nx)-hu(u,msfu,k,j-1,i,ny,nx);return rx-mm*(rdy*duy-ty);
}
inline float wavg(device const float*w,int k,int j,int i,int ny,int nx){ulong a=i3(k,j,i,ny,nx);float x=w[a];x=x+w[a-1];x=x+w[a-nx];x=x+w[a-nx-1];x=x+w[a+ulong(ny)*nx];x=x+w[a+ulong(ny)*nx-1];x=x+w[a+ulong(ny)*nx-ulong(nx)];x=x+w[a+ulong(ny)*nx-ulong(nx)-1];return 0.125f*x;}
inline float agl(device const float*ph,device const float*phb,int b3,device const float*ht,int k,int j,int i,int ny,int nx){float x=z_w(ph,phb,b3,k,j,i,ny,nx)-ht[j*nx+i];x=x+(z_w(ph,phb,b3,k,j,i-1,ny,nx)-ht[j*nx+i-1]);x=x+(z_w(ph,phb,b3,k,j-1,i,ny,nx)-ht[(j-1)*nx+i]);x=x+(z_w(ph,phb,b3,k,j-1,i-1,ny,nx)-ht[(j-1)*nx+i-1]);return 0.25f*x;}

kernel void uh_columns(constant UhColumnsResources&r [[buffer(0)]],constant UhColumnsParams&p [[buffer(1)]],uint col [[thread_position_in_grid]]){
 if(col>=uint(p.ncol))return;int j=int(col)/p.nx,i=int(col)-j*p.nx;if(i==0||j==0){r.uh[col]=0.0f;r.usecol[col]=1.0f;return;}
 float c2=-(0.5f*r.dnw[p.nz-1]/r.dn[p.nz-1]),c1=1.0f-c2,t2=c2,t1=1.0f-t2;
 float mm=(0.25f*(r.msfu[(j-1)*(p.nx+1)+i]+r.msfu[j*(p.nx+1)+i]))*(r.msfv[j*p.nx+i-1]+r.msfv[j*p.nx+i]);
 float acc=0.0f,use=1.0f,zl=agl(r.ph,r.phb,p.phb3d,r.ht,0,j,i,p.ny,p.nx),wa=0.0f,rv=0.0f;bool have=false;
 for(int k=0;k<p.nz;k++){float zu=agl(r.ph,r.phb,p.phb3d,r.ht,k+1,j,i,p.ny,p.nx);if(zl>=2000.0f&&zu<=5000.0f){if(!have){wa=wavg(r.w,k,j,i,p.ny,p.nx);rv=rvort(r.u,r.v,r.ph,r.phb,p.phb3d,r.msfu,r.msfv,r.fnm,r.fnp,p.cf1,p.cf2,p.cf3,t1,t2,p.rdx,p.rdy,mm,k,j,i,p.nz,p.ny,p.nx);}float wa1=0.0f,rv1=0.0f;if(k+1<p.nz){wa1=wavg(r.w,k+1,j,i,p.ny,p.nx);rv1=rvort(r.u,r.v,r.ph,r.phb,p.phb3d,r.msfu,r.msfv,r.fnm,r.fnp,p.cf1,p.cf2,p.cf3,t1,t2,p.rdx,p.rdy,mm,k+1,j,i,p.nz,p.ny,p.nx);}if(wa>0.0f&&wa1>0.0f){float pair=((wa*rv)+(wa1*rv1))*0.5f;acc=acc+pair*(zu-zl);}else{use=0.0f;acc=0.0f;}wa=wa1;rv=rv1;have=true;}else have=false;zl=zu;}
 r.uh[col]=acc;r.usecol[col]=use;
}

kernel void uh_smooth_max(constant UhSmoothMaxResources&r [[buffer(0)]],constant UhSmoothMaxParams&p [[buffer(1)]],uint col [[thread_position_in_grid]]){
 if(col>=uint(p.ny*p.nx))return;int j=int(col)/p.nx,i=int(col)-j*p.nx;if(i<1||i>p.nx-2||j<1||j>p.ny-2)return;ulong c=ulong(j)*p.nx+i;float t=0.25f*r.uh[c];float e=r.uh[c+1]+r.uh[c-1];e=e+r.uh[c+p.nx];e=e+r.uh[c-p.nx];float q=r.uh[c+p.nx+1]+r.uh[c-p.nx+1];q=q+r.uh[c+p.nx-1];q=q+r.uh[c-p.nx-1];float s=(t+0.125f*e)+0.0625f*q;if(r.usecol[c]!=0.0f&&s>r.up_heli_max[c])r.up_heli_max[c]=s;
}
kernel void uh_boundary_copy(constant UhBoundaryCopyResources&r [[buffer(0)]],constant UhBoundaryCopyParams&p [[buffer(1)]],uint col [[thread_position_in_grid]]){
 if(col>=uint(p.ncol))return;int j=int(col)/p.nx,i=int(col)-j*p.nx;
 // Equivalent to WRF's ordered x-then-y boundary copy. The south-west
 // corner therefore takes the post-x value from (1,1), without a race.
 if(j==0)r.field[col]=r.field[p.nx+max(i,1)];else if(i==0)r.field[col]=r.field[j*p.nx+1];
}
