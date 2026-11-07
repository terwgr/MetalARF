#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "soft_glibc_solar_trig.metalh"
#include "hexv842_sw_prep_f32_bindings.metalh"
#include "hexrrfs_sw_solar_bindings.metalh"
#pragma clang fp contract(off)
static inline float sw_sin(float x) {return as_type<float>(soft_glibc_solar_sinf(as_type<uint>(x)));}
static inline float sw_cos(float x) {return as_type<float>(soft_glibc_solar_cosf(as_type<uint>(x)));}
// Binary32 remainder by positive 1440, retaining signed zero and subnormals.
static inline float sw_fmod1440(float value) {
 uint bits=as_type<uint>(value),mag=bits&0x7fffffffu,sgn=bits&0x80000000u;
 if(mag<0x44b40000u)return value;
 if(mag>=0x7f800000u)return as_type<float>(0xffc00000u);
 int ex=int(mag>>23),ey=137;uint x=(mag&0x7fffffu)|0x800000u,y=0xb40000u;
 while(ex>ey){if(x>=y)x-=y;x<<=1;--ex;}
 if(x>=y)x-=y;
 if(!x)return as_type<float>(sgn);
 while(x<0x800000u){x<<=1;--ey;}
 return as_type<float>(sgn|(uint(ey)<<23)|(x&0x7fffffu));
}
static inline float sw_solar_cosine(float xt24,float gmt,float lon,
                                    float lat,float degrad,float declin) {
 float local=mpas_add(mpas_add(gmt,mpas_div(xt24,60.f)),mpas_div(lon,15.f));
 float angle=mpas_mul(mpas_mul(15.f,mpas_sub(local,12.f)),degrad);
 float latitude=mpas_mul(lat,degrad);
 return mpas_add(mpas_mul(sw_sin(latitude),sw_sin(declin)),
                 mpas_mul(mpas_mul(sw_cos(latitude),sw_cos(declin)),sw_cos(angle)));
}

// The RRFS equation of time adjusts only the cosine used by day selection.
kernel void hexrrfs_sw_solar(constant HexrrfsSwSolarResources &r [[buffer(0)]],
                            constant HexrrfsSwSolarParams &p [[buffer(1)]],
                            uint cell [[thread_position_in_grid]]) {
 if(cell>=uint(p.nc))return;
 float da=mpas_div(mpas_mul(6.2831853071795862f,mpas_sub(p.julian,p.day_offset)),365.f);
 float two=mpas_mul(2.f,da);
 float eq=mpas_add(.000075f,mpas_mul(.001868f,sw_cos(da)));
 eq=mpas_sub(eq,mpas_mul(.032077f,sw_sin(da)));
 eq=mpas_sub(eq,mpas_mul(.014615f,sw_cos(two)));
 eq=mpas_mul(mpas_sub(eq,mpas_mul(p.second_sine,sw_sin(two))),229.18f);
 float xt24=mpas_add(sw_fmod1440(mpas_add(r.controls[0],mpas_mul(r.controls[1],.5f))),eq);
 r.coszr[cell]=sw_solar_cosine(xt24,p.gmt,r.surface[p.nc+cell],r.surface[cell],p.degrad,r.controls[2]);
}

constant float MPAS_RETAB[95] = {5.92779f,6.26422f,6.61973f,6.99539f,7.39234f,7.81177f,8.25496f,8.72323f,9.21800f,9.74075f,10.2930f,10.8765f,11.4929f,12.1440f,12.8317f,13.5581f,14.2319f,15.0351f,15.8799f,16.7674f,17.6986f,18.6744f,19.6955f,20.7623f,21.8757f,23.0364f,24.2452f,25.5034f,26.8125f,27.7895f,28.6450f,29.4167f,30.1088f,30.7306f,31.2943f,31.8151f,32.3077f,32.7870f,33.2657f,33.7540f,34.2601f,34.7892f,35.3442f,35.9255f,36.5316f,37.1602f,37.8078f,38.4720f,39.1508f,39.8442f,40.5552f,41.2912f,42.0635f,42.8876f,43.7863f,44.7853f,45.9170f,47.2165f,48.7221f,50.4710f,52.4980f,54.8315f,57.4898f,60.4785f,63.7898f,65.5604f,71.2885f,75.4113f,79.7368f,84.2351f,88.8833f,93.6658f,98.5739f,103.603f,108.752f,114.025f,119.424f,124.954f,130.630f,136.457f,142.446f,148.608f,154.956f,161.503f,168.262f,175.248f,182.473f,189.952f,197.699f,205.728f,214.055f,222.694f,231.661f,240.971f,250.639f};
constant float MPAS_O3SUM[31] = {5.297E-8f,5.852E-8f,6.579E-8f,7.505E-8f,8.577E-8f,9.895E-8f,1.175E-7f,1.399E-7f,1.677E-7f,2.003E-7f,2.571E-7f,3.325E-7f,4.438E-7f,6.255E-7f,8.168E-7f,1.036E-6f,1.366E-6f,1.855E-6f,2.514E-6f,3.240E-6f,4.033E-6f,4.854E-6f,5.517E-6f,6.089E-6f,6.689E-6f,1.106E-5f,1.462E-5f,1.321E-5f,9.856E-6f,5.960E-6f,5.960E-6f};
constant float MPAS_O3WIN[31] = {4.629E-8f,4.686E-8f,5.017E-8f,5.613E-8f,6.871E-8f,8.751E-8f,1.138E-7f,1.516E-7f,2.161E-7f,3.264E-7f,4.968E-7f,7.338E-7f,1.017E-6f,1.308E-6f,1.625E-6f,2.011E-6f,2.516E-6f,3.130E-6f,3.840E-6f,4.703E-6f,5.486E-6f,6.289E-6f,6.993E-6f,7.494E-6f,8.197E-6f,9.632E-6f,1.113E-5f,1.146E-5f,9.389E-6f,6.135E-6f,6.135E-6f};
constant float MPAS_PPSUM[31] = {955.890f,850.532f,754.599f,667.742f,589.841f,519.421f,455.480f,398.085f,347.171f,301.735f,261.310f,225.360f,193.419f,165.490f,141.032f,120.125f,102.689f,87.829f,75.123f,64.306f,55.086f,47.209f,40.535f,34.795f,29.865f,19.122f,9.277f,4.660f,2.421f,1.294f,0.647f};
constant float MPAS_PPWIN[31] = {955.747f,841.783f,740.199f,649.538f,568.404f,495.815f,431.069f,373.464f,322.354f,277.190f,237.635f,203.433f,174.070f,148.949f,127.408f,108.915f,93.114f,79.551f,67.940f,58.072f,49.593f,42.318f,36.138f,30.907f,26.362f,16.423f,7.583f,3.620f,1.807f,0.938f,0.469f};
static inline float mpas_ice_radius(float t,int bound) {
 int idx=max(1,min((int)mpas_sub(t,179.f),bound))-1;
 float corr=mpas_sub(t,(float)(int)t);
 return mpas_add(mpas_mul(MPAS_RETAB[idx],mpas_sub(1.f,corr)),mpas_mul(MPAS_RETAB[idx+1],corr));
}
// MPAS module_ra_rrtmg_lw.F:12308-12444, native diagnosed ozone.
static inline float mpas_ozone(int k,device const float* lev) {
 float ann[31],half_press[32];
 ann[0]=mpas_mul(.5f,mpas_add(MPAS_O3SUM[0],MPAS_O3WIN[0]));
 half_press[0]=1100.f;half_press[31]=0.f;
 for(int j=1;j<31;j++) {
  ann[j]=mpas_add(MPAS_O3WIN[j-1],mpas_mul(mpas_div(mpas_sub(MPAS_O3WIN[j],MPAS_O3WIN[j-1]),mpas_sub(MPAS_PPWIN[j],MPAS_PPWIN[j-1])),mpas_sub(MPAS_PPSUM[j],MPAS_PPWIN[j-1])));
  ann[j]=mpas_mul(.5f,mpas_add(ann[j],MPAS_O3SUM[j]));
  half_press[j]=mpas_div(mpas_add(MPAS_PPSUM[j],MPAS_PPSUM[j-1]),2.f);
 }
 float v=0.f;
 for(int j=0;j<31;j++) {
  float pb1=mpas_max(0.f,mpas_sub(lev[k],half_press[j]));
  float pb2=mpas_max(0.f,mpas_sub(lev[k],half_press[j+1]));
  float pt1=mpas_max(0.f,mpas_sub(lev[k+1],half_press[j]));
  float pt2=mpas_max(0.f,mpas_sub(lev[k+1],half_press[j+1]));
  v=mpas_add(v,mpas_mul(mpas_add(mpas_sub(mpas_sub(pb2,pb1),pt2),pt1),ann[j]));
 }
 return mpas_mul(mpas_div(v,mpas_sub(lev[k],lev[k+1])),.603461f);
}
// MPAS module_ra_rrtmg_vinterp.F:21-173. Bottom-to-top caller pressures.
static inline float mpas_ozone_interp(float p,int nozn,device const float* pin,device const float* o3,thread int& upper) {
 if(p<pin[0]) return mpas_div(mpas_mul(o3[0],p),pin[0]);
 if(p>pin[nozn-1])return o3[nozn-1];
 for(int j=upper;j<nozn-1;j++)if(pin[j]<p&&p<=pin[j+1]){upper=j;break;}
 float dpu=mpas_sub(p,pin[upper]),dpl=mpas_sub(pin[upper+1],p);
 return mpas_div(mpas_add(mpas_mul(o3[upper],dpl),mpas_mul(o3[upper+1],dpu)),mpas_add(dpl,dpu));
}

kernel void hexv842_sw_prep_f32(constant Hexv842SwPrepF32Resources &r [[buffer(0)]],
 constant Hexv842SwPrepF32Params &params [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int nc=params.nc,nz=params.nz,icloud=params.icloud,o3input=params.o3input,reqc=params.reqc,reqi=params.reqi,reqs=params.reqs,nozn=params.nozn;
 float xtime=r.controls[0],gmt=params.gmt,radt=r.controls[1],degrad=params.degrad,declin=r.controls[2];
 device const float *fields=r.fields,*surf=r.surf,*pw=r.pw,*tw=r.tw,*pin=r.pin,*oz=r.oz;
 device float *layers=r.layers,*interfaces=r.interfaces,*hgt=r.hgt,*coszr=r.coszr;

 int c=int(tid);if(c>=nc)return;
 int nl=nz+1;int nl1=nl+1;
 #define FI(f,k) fields[((f)*nc+c)*nz+(k)]
 #define SF(f) surf[(f)*nc+c]
 #define L(f,k) layers[((f)*nc+c)*nl+(k)]
 #define IL(f,k) interfaces[((f)*nc+c)*nl1+(k)]
 float xt24=sw_fmod1440(mpas_add(xtime,mpas_mul(radt,.5f)));
 coszr[c]=sw_solar_cosine(xt24,gmt,SF(1),SF(0),degrad,declin);
 for(int k=0;k<=nz;k++){IL(0,k)=mpas_div(pw[c*(nz+1)+k],100.f);IL(1,k)=tw[c*(nz+1)+k];}
 IL(0,nl)=1.e-5f;IL(1,nl)=mpas_add(IL(1,nz),0.f);
 float sum=0.f;
 for(int k=0;k<nl;k++) {
  float p=(k<nz?mpas_div(FI(0,k),100.f):mpas_mul(.5f,IL(0,nz)));
  float t=(k<nz?FI(1,k):mpas_add(IL(1,nz),0.f));
  L(0,k)=p;L(1,k)=t;L(2,k)=(k<nz?mpas_mul(mpas_max(FI(4,k),1.e-12f),1.607793f):L(2,nz-1));
  L(4,k)=379.e-6f;L(5,k)=1774.e-9f;L(6,k)=319.e-9f;L(7,k)=.209488f;
  float dz=FI(2,min(k,nz-1));hgt[c*nl+k]=mpas_add(sum,mpas_mul(.5f,dz));sum=mpas_add(sum,dz);
  if(k==nz){L(8,k)=0.f;L(9,k)=0.f;L(10,k)=0.f;L(11,k)=0.f;L(12,k)=10.f;L(13,k)=10.f;L(14,k)=10.f;continue;}
  float cf=icloud>0?FI(8,k):0.f;
  float qc=icloud>0?mpas_max(FI(5,k),0.f):0.f;
  float qi=icloud>0?mpas_max(FI(6,k),0.f):0.f;
  float qs=icloud>0?mpas_max(FI(7,k),0.f):0.f;
  float pdel=mpas_sub(IL(0,k),IL(0,k+1));
  L(8,k)=cf;
  float rel;
  if(reqc) {
   rel=mpas_max(2.5f,mpas_mul(FI(9,k),1.e6f));
   if(rel<=2.5f&&soft32_greater(as_type<uint>(cf),0u)){if(mpas_sub(SF(4),1.5f)>0.f)rel=10.5f;else if(mpas_sub(SF(4),1.5f)<0.f)rel=7.5f;}
  } else {
   rel=mpas_add(8.f,mpas_mul(6.f,mpas_min(1.f,mpas_max(0.f,mpas_mul(mpas_sub(273.16f,t),.05f)))));
   rel=mpas_add(rel,mpas_mul(mpas_sub(14.f,rel),mpas_min(1.f,mpas_max(0.f,mpas_mul(mpas_mul(.001f,SF(6)),10.f)))));
   rel=mpas_add(rel,mpas_mul(mpas_sub(14.f,rel),mpas_min(1.f,mpas_max(0.f,mpas_sub(1.f,mpas_sub(2.f,SF(4)))))));
   rel=mpas_add(rel,mpas_mul(mpas_sub(14.f,rel),mpas_min(1.f,mpas_max(0.f,SF(5)))));
  }
  float rei;
  if(reqi){rei=mpas_max(5.f,mpas_mul(FI(10,k),1.e6f));if(rei<=5.f&&soft32_greater(as_type<uint>(cf),0u))rei=mpas_max(5.f,mpas_ice_radius(t,75));}
  else {rei=mpas_ice_radius(t,94);rei=mpas_min(140.f,mpas_mul(rei,1.0315f));}
  float res=reqs?mpas_max(10.f,mpas_mul(FI(11,k),1.e6f)):10.f;
  float snowfactor=1.f;
  if(reqs&&res>130.f){snowfactor=mpas_mul(mpas_div(130.f,res),mpas_div(130.f,res));res=130.f;}
  float denom=mpas_max(.01f,cf);
  L(10,k)=mpas_div(mpas_mul(mpas_div(mpas_mul(mpas_mul(qc,pdel),100.f),9.80616f),1000.f),denom);
  float qice=(!reqi&&!reqs)?mpas_add(qi,qs):qi;
  L(9,k)=mpas_div(mpas_mul(mpas_div(mpas_mul(mpas_mul(qice,pdel),100.f),9.80616f),1000.f),denom);
  float sn=mpas_mul(mpas_div(mpas_mul(mpas_mul(qs,pdel),100.f),9.80616f),1000.f);
  L(11,k)=reqs?mpas_div(mpas_mul(snowfactor,sn),denom):0.f;
  L(12,k)=rei;L(13,k)=rel;L(14,k)=res;
 }
 int upper=0;
 for(int k=nl-1;k>=0;k--)L(3,k)=o3input==2?mpas_ozone_interp(L(0,k),nozn,pin,nozn<2?oz+c*3+1:oz+c*nozn,upper):mpas_ozone(k,interfaces+c*nl1);
 #undef FI
 #undef SF
 #undef L
 #undef IL

}
