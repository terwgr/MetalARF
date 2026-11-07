#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_gfk_exp.metalh"
#include "hexv842_sw_mcica_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float sw_exp(float x) {return as_type<float>(soft_gfk_exp(as_type<uint>(x)));}
static inline float mpas_kiss(thread uint& s1,thread uint& s2,thread uint& s3,thread uint& s4) {
 s1=69069u*s1+1327217885u;s2^=s2<<13;s2^=s2>>17;s2^=s2<<5;
 s3=18000u*(s3&65535u)+(s3>>16);s4=30903u*(s4&65535u)+(s4>>16);
 int kiss=as_type<int>(s1+s2+(s3<<16)+s4);
 return mpas_add(mpas_mul((float)kiss,2.328306e-10f),.5f);
}

kernel void hexv842_sw_mcica_f32(constant Hexv842SwMcicaF32Resources &r [[buffer(0)]],
 constant Hexv842SwMcicaF32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int nc=p.nc,nl=p.nl,icld=p.icld,idcor=p.idcor,julday=p.julday,irng=0,permuteseed=1;
 device const float *play=r.play,*cf=r.cf,*iw=r.iw,*lw=r.lw,*sw=r.sw,*hgt=r.hgt,*latitude=r.latitude,*uniform=r.uniform;
 device float *out=r.out;

 int c=int(tid);if(c>=nc)return;
 #define X(a,k) a[c*nl+(k)]
 #define O(f,k,g) out[(((f)*nc+c)*nl+(k))*112+(g)]
 if(icld==0)return;
 // Native seed3/seed4 read past pmid for short columns (SW:1745-1748).
 // Leave the missing storage indeterminate without issuing a CUDA OOB load.
 unsigned int seeds[4];if(irng==0)for(int j=0;j<4;j++){float pm;if(j<nl)pm=mpas_mul(X(play,j),100.f);seeds[j]=(unsigned int)(int)mpas_mul(mpas_sub(pm,(float)(int)pm),1000000000.f);}
 if(irng==0)for(int i=0;i<permuteseed;i++)mpas_kiss(seeds[0],seeds[1],seeds[2],seeds[3]);
 float zo=2500.f;
 if(idcor==1){float am3=mpas_mul(julday>181?-0.07671232521533966f:0.07671232521533966f,(float)(julday-(julday>181?272:91)));
  float d=mpas_sub(latitude[c],am3);zo=mpas_mul(mpas_add(1.4315f,mpas_mul(2.1219f,sw_exp(mpas_div(-mpas_mul(d,d),654.54107666015625f)))),1.e3f);}
 float inverse=mpas_div(1.f,zo);
 for(int g=0;g<112;g++) {
  float prev=0.f,prev_cf=0.f;
  float maximum=0.f;if(icld==3)maximum=irng==0?mpas_kiss(seeds[0],seeds[1],seeds[2],seeds[3]):uniform[(c*nl)*112+g];
  for(int k=0;k<nl;k++){
   // Native SELECT CASE has no DEFAULT: unsupported overlap leaves CDF unset.
   float r;if(icld>=1&&icld<=5)r=icld==3?maximum:(irng==0?mpas_kiss(seeds[0],seeds[1],seeds[2],seeds[3]):uniform[(c*nl+k)*112+g]);
   float frac=X(cf,k)<1.e-20f?0.f:X(cf,k);
   if(icld==2&&k>0){if(prev>mpas_sub(1.f,prev_cf))r=prev;else r=mpas_mul(r,mpas_sub(1.f,prev_cf));}
   if(icld==4||icld==5){float r2=irng==0?mpas_kiss(seeds[0],seeds[1],seeds[2],seeds[3]):uniform[((nc+c)*nl+k)*112+g];
    if(k>0){float alpha=sw_exp(mpas_mul(-mpas_sub(X(hgt,k),X(hgt,k-1)),inverse));if(icld==5&&frac==0.f&&prev_cf>0.f)alpha=0.f;if(r2<alpha)r=prev;}}
   bool cloudy=r>=mpas_sub(1.f,frac);O(0,k,g)=cloudy?1.f:0.f;
   O(1,k,g)=cloudy?X(iw,k):0.f;O(2,k,g)=cloudy?X(lw,k):0.f;O(3,k,g)=cloudy?X(sw,k):0.f;
   prev=r;prev_cf=frac;
  }
 }
 #undef X
 #undef O

}
