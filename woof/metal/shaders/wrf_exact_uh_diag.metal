#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_uh_columns_bindings.metalh"
#include "wrf_exact_uh_smooth_max_bindings.metalh"
#include "wrf_exact_uh_boundary_copy_bindings.metalh"
using namespace metal;
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_less_equal(a.bits,b.bits); }
static inline bool operator!=(Exact32 a, Exact32 b) { return !soft32_equal(a.bits,b.bits); }


inline ulong i3(int k,int j,int i,int ny,int nx) { return (ulong(k)*ny+j)*nx+i; }
inline ulong i3u(int k,int j,int i,int ny,int nx) { return (ulong(k)*ny+j)*(nx+1)+i; }
inline ulong i3v(int k,int j,int i,int ny,int nx) { return (ulong(k)*(ny+1)+j)*nx+i; }
inline Exact32 z_w(device const Exact32* ph,device const Exact32* phb,int b3,int k,int j,int i,int ny,int nx) {
  Exact32 base=b3?phb[i3(k,j,i,ny,nx)]:phb[k]; return (ph[i3(k,j,i,ny,nx)]+base)/9.81f;
}
inline Exact32 rdzw(device const Exact32* ph,device const Exact32* phb,int b3,int k,int j,int i,int ny,int nx) { return 1.0f/(z_w(ph,phb,b3,k+1,j,i,ny,nx)-z_w(ph,phb,b3,k,j,i,ny,nx)); }
inline Exact32 zx(device const Exact32* ph,device const Exact32* phb,int b3,Exact32 rdx,int k,int j,int is,int ny,int nx) {
 if(is==0||is==nx)return 0.0f;
 Exact32 a=b3?phb[i3(k,j,is,ny,nx)]:phb[k], b=b3?phb[i3(k,j,is-1,ny,nx)]:phb[k];
 Exact32 p1=rdx*(a-b)/9.81f, p2=rdx*(ph[i3(k,j,is,ny,nx)]-ph[i3(k,j,is-1,ny,nx)])/9.81f; return p1+p2;
}
inline Exact32 zy(device const Exact32* ph,device const Exact32* phb,int b3,Exact32 rdy,int k,int js,int i,int ny,int nx) {
 if(js==0||js==ny)return 0.0f;
 Exact32 a=b3?phb[i3(k,js,i,ny,nx)]:phb[k], b=b3?phb[i3(k,js-1,i,ny,nx)]:phb[k];
 Exact32 p1=rdy*(a-b)/9.81f, p2=rdy*(ph[i3(k,js,i,ny,nx)]-ph[i3(k,js-1,i,ny,nx)])/9.81f; return p1+p2;
}
inline Exact32 hv(device const Exact32*v,device const Exact32*msfv,int k,int j,int i,int ny,int nx){return v[i3v(k,j,i,ny,nx)]/msfv[j*nx+i];}
inline Exact32 hu(device const Exact32*u,device const Exact32*msfu,int k,int j,int i,int ny,int nx){return u[i3u(k,j,i,ny,nx)]/msfu[j*(nx+1)+i];}
inline Exact32 havg_v(device const Exact32*v,device const Exact32*msfv,device const Exact32*fnm,device const Exact32*fnp,Exact32 c1,Exact32 c2,Exact32 c3,Exact32 t1,Exact32 t2,int k,int j,int i,int nz,int ny,int nx){
 if(k==0){Exact32 a=c1*hv(v,msfv,0,j,i-1,ny,nx);a=a+c2*hv(v,msfv,1,j,i-1,ny,nx);a=a+c3*hv(v,msfv,2,j,i-1,ny,nx);a=a+c1*hv(v,msfv,0,j,i,ny,nx);a=a+c2*hv(v,msfv,1,j,i,ny,nx);a=a+c3*hv(v,msfv,2,j,i,ny,nx);return 0.5f*a;}
 if(k==nz){Exact32 a=t1*(hv(v,msfv,nz-1,j,i,ny,nx)+hv(v,msfv,nz-1,j,i-1,ny,nx));Exact32 b=t2*(hv(v,msfv,nz-2,j,i,ny,nx)+hv(v,msfv,nz-2,j,i-1,ny,nx));return 0.5f*(a+b);}
 Exact32 a=fnm[k]*(hv(v,msfv,k,j,i-1,ny,nx)+hv(v,msfv,k,j,i,ny,nx));Exact32 b=fnp[k]*(hv(v,msfv,k-1,j,i-1,ny,nx)+hv(v,msfv,k-1,j,i,ny,nx));return 0.5f*(a+b);
}
inline Exact32 havg_u(device const Exact32*u,device const Exact32*msfu,device const Exact32*fnm,device const Exact32*fnp,Exact32 c1,Exact32 c2,Exact32 c3,Exact32 t1,Exact32 t2,int k,int j,int i,int nz,int ny,int nx){
 if(k==0){Exact32 a=c1*hu(u,msfu,0,j-1,i,ny,nx);a=a+c2*hu(u,msfu,1,j-1,i,ny,nx);a=a+c3*hu(u,msfu,2,j-1,i,ny,nx);a=a+c1*hu(u,msfu,0,j,i,ny,nx);a=a+c2*hu(u,msfu,1,j,i,ny,nx);a=a+c3*hu(u,msfu,2,j,i,ny,nx);return 0.5f*a;}
 if(k==nz){Exact32 a=t1*(hu(u,msfu,nz-1,j-1,i,ny,nx)+hu(u,msfu,nz-1,j,i,ny,nx));Exact32 b=t2*(hu(u,msfu,nz-2,j-1,i,ny,nx)+hu(u,msfu,nz-2,j,i,ny,nx));return 0.5f*(a+b);}
 Exact32 a=fnm[k]*(hu(u,msfu,k,j-1,i,ny,nx)+hu(u,msfu,k,j,i,ny,nx));Exact32 b=fnp[k]*(hu(u,msfu,k-1,j-1,i,ny,nx)+hu(u,msfu,k-1,j,i,ny,nx));return 0.5f*(a+b);
}
inline Exact32 rvort(device const Exact32*u,device const Exact32*v,device const Exact32*ph,device const Exact32*phb,int b3,device const Exact32*msfu,device const Exact32*msfv,device const Exact32*fnm,device const Exact32*fnp,Exact32 c1,Exact32 c2,Exact32 c3,Exact32 t1,Exact32 t2,Exact32 rdx,Exact32 rdy,Exact32 mm,int k,int j,int i,int nz,int ny,int nx){
 Exact32 xz=0.25f*((zx(ph,phb,b3,rdx,k,j-1,i,ny,nx)+zx(ph,phb,b3,rdx,k,j,i,ny,nx))+zx(ph,phb,b3,rdx,k+1,j-1,i,ny,nx)+zx(ph,phb,b3,rdx,k+1,j,i,ny,nx));
 Exact32 dz=(((rdzw(ph,phb,b3,k,j,i,ny,nx)+rdzw(ph,phb,b3,k,j-1,i,ny,nx))+rdzw(ph,phb,b3,k,j-1,i-1,ny,nx))+rdzw(ph,phb,b3,k,j,i-1,ny,nx));
 Exact32 dh=havg_v(v,msfv,fnm,fnp,c1,c2,c3,t1,t2,k+1,j,i,nz,ny,nx)-havg_v(v,msfv,fnm,fnp,c1,c2,c3,t1,t2,k,j,i,nz,ny,nx);
 Exact32 tx=((dh*0.25f)*xz)*dz, dvd=hv(v,msfv,k,j,i,ny,nx)-hv(v,msfv,k,j,i-1,ny,nx);Exact32 rx=mm*(rdx*dvd-tx);
 Exact32 yz=0.25f*((zy(ph,phb,b3,rdy,k,j,i-1,ny,nx)+zy(ph,phb,b3,rdy,k,j,i,ny,nx))+zy(ph,phb,b3,rdy,k+1,j,i-1,ny,nx)+zy(ph,phb,b3,rdy,k+1,j,i,ny,nx));
 dz=((rdzw(ph,phb,b3,k,j,i,ny,nx)+rdzw(ph,phb,b3,k,j,i-1,ny,nx))+rdzw(ph,phb,b3,k,j-1,i-1,ny,nx))+rdzw(ph,phb,b3,k,j-1,i,ny,nx);
 dh=havg_u(u,msfu,fnm,fnp,c1,c2,c3,t1,t2,k+1,j,i,nz,ny,nx)-havg_u(u,msfu,fnm,fnp,c1,c2,c3,t1,t2,k,j,i,nz,ny,nx);
 Exact32 ty=((dh*0.25f)*yz)*dz, duy=hu(u,msfu,k,j,i,ny,nx)-hu(u,msfu,k,j-1,i,ny,nx);return rx-mm*(rdy*duy-ty);
}
inline Exact32 wavg(device const Exact32*w,int k,int j,int i,int ny,int nx){ulong a=i3(k,j,i,ny,nx);Exact32 x=w[a];x=x+w[a-1];x=x+w[a-nx];x=x+w[a-nx-1];x=x+w[a+ulong(ny)*nx];x=x+w[a+ulong(ny)*nx-1];x=x+w[a+ulong(ny)*nx-ulong(nx)];x=x+w[a+ulong(ny)*nx-ulong(nx)-1];return 0.125f*x;}
inline Exact32 agl(device const Exact32*ph,device const Exact32*phb,int b3,device const Exact32*ht,int k,int j,int i,int ny,int nx){Exact32 x=z_w(ph,phb,b3,k,j,i,ny,nx)-ht[j*nx+i];x=x+(z_w(ph,phb,b3,k,j,i-1,ny,nx)-ht[j*nx+i-1]);x=x+(z_w(ph,phb,b3,k,j-1,i,ny,nx)-ht[(j-1)*nx+i]);x=x+(z_w(ph,phb,b3,k,j-1,i-1,ny,nx)-ht[(j-1)*nx+i-1]);return 0.25f*x;}

kernel void wrf_exact_uh_columns(constant WrfExactUhColumnsResources&r [[buffer(0)]],constant WrfExactUhColumnsParams&p [[buffer(1)]],uint col [[thread_position_in_grid]]){
 if(col>=uint(p.ncol))return;int j=int(col)/p.nx,i=int(col)-j*p.nx;if(i==0||j==0){((device Exact32*)r.uh)[col]=0.0f;((device Exact32*)r.usecol)[col]=1.0f;return;}
 Exact32 c2=-(0.5f*((device const Exact32*)r.dnw)[p.nz-1]/((device const Exact32*)r.dn)[p.nz-1]),c1=1.0f-c2,t2=c2,t1=1.0f-t2;
 Exact32 mm=(0.25f*(((device const Exact32*)r.msfu)[(j-1)*(p.nx+1)+i]+((device const Exact32*)r.msfu)[j*(p.nx+1)+i]))*(((device const Exact32*)r.msfv)[j*p.nx+i-1]+((device const Exact32*)r.msfv)[j*p.nx+i]);
 Exact32 acc=0.0f,use=1.0f,zl=agl(((device const Exact32*)r.ph),((device const Exact32*)r.phb),p.phb3d,((device const Exact32*)r.ht),0,j,i,p.ny,p.nx),wa=0.0f,rv=0.0f;bool have=false;
 for(int k=0;k<p.nz;k++){Exact32 zu=agl(((device const Exact32*)r.ph),((device const Exact32*)r.phb),p.phb3d,((device const Exact32*)r.ht),k+1,j,i,p.ny,p.nx);if(zl>=2000.0f&&zu<=5000.0f){if(!have){wa=wavg(((device const Exact32*)r.w),k,j,i,p.ny,p.nx);rv=rvort(((device const Exact32*)r.u),((device const Exact32*)r.v),((device const Exact32*)r.ph),((device const Exact32*)r.phb),p.phb3d,((device const Exact32*)r.msfu),((device const Exact32*)r.msfv),((device const Exact32*)r.fnm),((device const Exact32*)r.fnp),p.cf1,p.cf2,p.cf3,t1,t2,p.rdx,p.rdy,mm,k,j,i,p.nz,p.ny,p.nx);}Exact32 wa1=0.0f,rv1=0.0f;if(k+1<p.nz){wa1=wavg(((device const Exact32*)r.w),k+1,j,i,p.ny,p.nx);rv1=rvort(((device const Exact32*)r.u),((device const Exact32*)r.v),((device const Exact32*)r.ph),((device const Exact32*)r.phb),p.phb3d,((device const Exact32*)r.msfu),((device const Exact32*)r.msfv),((device const Exact32*)r.fnm),((device const Exact32*)r.fnp),p.cf1,p.cf2,p.cf3,t1,t2,p.rdx,p.rdy,mm,k+1,j,i,p.nz,p.ny,p.nx);}if(wa>0.0f&&wa1>0.0f){Exact32 pair=((wa*rv)+(wa1*rv1))*0.5f;acc=acc+pair*(zu-zl);}else{use=0.0f;acc=0.0f;}wa=wa1;rv=rv1;have=true;}else have=false;zl=zu;}
 ((device Exact32*)r.uh)[col]=acc;((device Exact32*)r.usecol)[col]=use;
}

kernel void wrf_exact_uh_smooth_max(constant WrfExactUhSmoothMaxResources&r [[buffer(0)]],constant WrfExactUhSmoothMaxParams&p [[buffer(1)]],uint col [[thread_position_in_grid]]){
 if(col>=uint(p.ny*p.nx))return;int j=int(col)/p.nx,i=int(col)-j*p.nx;if(i<1||i>p.nx-2||j<1||j>p.ny-2)return;ulong c=ulong(j)*p.nx+i;Exact32 t=0.25f*((device Exact32*)r.uh)[c];Exact32 e=((device Exact32*)r.uh)[c+1]+((device Exact32*)r.uh)[c-1];e=e+((device Exact32*)r.uh)[c+p.nx];e=e+((device Exact32*)r.uh)[c-p.nx];Exact32 q=((device Exact32*)r.uh)[c+p.nx+1]+((device Exact32*)r.uh)[c-p.nx+1];q=q+((device Exact32*)r.uh)[c+p.nx-1];q=q+((device Exact32*)r.uh)[c-p.nx-1];Exact32 s=(t+0.125f*e)+0.0625f*q;if(((device Exact32*)r.usecol)[c]!=0.0f&&s>((device Exact32*)r.up_heli_max)[c])((device Exact32*)r.up_heli_max)[c]=s;
}
kernel void wrf_exact_uh_boundary_copy(constant WrfExactUhBoundaryCopyResources&r [[buffer(0)]],constant WrfExactUhBoundaryCopyParams&p [[buffer(1)]],uint col [[thread_position_in_grid]]){
 if(col>=uint(p.ncol))return;int j=int(col)/p.nx,i=int(col)-j*p.nx;
 // Equivalent to WRF's ordered x-then-y boundary copy. The south-west
 // corner therefore takes the post-x value from (1,1), without a race.
 if(j==0)((device Exact32*)r.field)[col]=((device Exact32*)r.field)[p.nx+max(i,1)];else if(i==0)((device Exact32*)r.field)[col]=((device Exact32*)r.field)[j*p.nx+1];
}
