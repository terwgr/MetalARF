#include <metal_stdlib>
#include "thompson_frozen_vapor_network_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
using namespace metal;
#pragma clang fp contract(off)

// Every CUDA double below is represented by its IEEE word.  Table payloads
// are physical Fortran-order words, and the offsets mirror the CUDA formulas.
static inline ulong t64(float x) { return soft64_from_f32_bits(as_type<uint>(x)); }
static inline float t32(ulong x) { return as_type<float>(soft64_to_f32_bits(x)); }
static inline ulong tadd(ulong a, ulong b) { return soft64_add(a,b); }
static inline ulong tsub(ulong a, ulong b) { return soft64_sub(a,b); }
static inline ulong tmul(ulong a, ulong b) { return soft64_mul(a,b); }
static inline ulong tdiv(ulong a, ulong b) { return soft64_div(a,b); }
static inline ulong tmin(ulong a, ulong b) { return soft64_less(a,b) ? a : b; }
static inline ulong tmax(ulong a, ulong b) { return soft64_greater(a,b) ? a : b; }
static inline ulong tneg(ulong a) { return a ^ 0x8000000000000000ul; }
static inline ulong tabs(ulong a) { return soft64_abs_bits(a); }
static inline ulong tatan(ulong x) {
 const bool neg=soft64_sign(x); ulong a=tabs(x);
 const ulong one=0x3ff0000000000000ul,halfpi=0x3ff921fb54442d18ul,quarterpi=0x3fe921fb54442d18ul;
 bool reciprocal=soft64_greater(a,one);
 if(reciprocal)a=tdiv(one,a);
 bool reduce=soft64_greater(a,0x3fe0000000000000ul);
 if(reduce)a=tdiv(tsub(a,one),tadd(a,one));
 const ulong z=tmul(a,a); ulong term=a,sum=a;
 for(int k=1;k<32;++k){term=tneg(tmul(term,z));sum=tadd(sum,tdiv(term,soft64_from_i32(2*k+1)));}
 if(reduce)sum=tadd(quarterpi,sum);
 if(reciprocal)sum=tsub(halfpi,sum);
 return neg?tneg(sum):sum;
}
static inline bool tpos(ulong a) { return soft64_greater(a,0ul); }
static inline float tp(float a,float b) { return precise::pow(a,b); }
static inline float te(float a) { return precise::exp(a); }
static inline float tl10(float a) { return precise::log10(a); }
static inline float trslf(float p,float t) {
 const float x=max(-80.f,t-273.16f);
 float e=611.583699f+x*(44.4606896f+x*(1.43177157f+x*(.0264224321f+x*(.000299291081f+x*(.00000203154182f+x*(.00000000702620698f+x*(0.379534310e-11f+x*(-.0000000000000321582393f))))))));
 e=min(e,p*.15f); return .622f*e/(p-e);
}
static inline float trsif(float p,float t) {
 const float x=max(-80.f,t-273.16f);
 float e=609.868993f+x*(49.9320233f+x*(1.84672631f+x*(.0402737184f+x*(.000565392987f+x*(.00000521693933f+x*(.0000000307839583f+x*(.000000000105785160f+x*.000000000000161444444f)))))));
 e=min(e,p*.15f); return .622f*e/max(1.e-4f,p-e);
}
static inline float tfield_a(float tc,float m) {
 const float tc2=tc*tc,m2=m*m;
 const float v=5.065339f-.062659f*tc-3.032362f*m+.029469f*tc*m-.000285f*tc2+.31255f*m2+.000204f*tc2*m+.003199f*tc*m2-.015952f*m2*m;
 return tp(10.f,v);
}
static inline float tfield_b(float tc,float m) {
 const float tc2=tc*tc,m2=m*m;
 return .476221f-.015896f*tc+.165977f*m+.007468f*tc*m-.000141f*tc2+.060366f*m2+.000079f*tc2*m+.000594f*tc*m2-.003577f*m2*m;
}
static inline int tindex(float x,int first,int n) {
 int c=int(precise::round(tl10(x))),e=c;
 for(int q=c-1;q<=c+1;++q){ float s=tp(10.f,float(q)); float m=x/s; if(m>=1.f&&m<10.f){e=q;break;} }
 float s=tp(10.f,float(e)); int digit=int(x/s); return clamp(digit+9*(e-first)-1,0,n-1);
}
static inline int tindex64(ulong x,int first,int n) {
 const ulong log10x=tdiv(soft64_plm_log(x),0x40026bb1bbb55516ul);
 int c=int(soft64_to_i32_rz(tadd(log10x,0x3fe0000000000000ul)).value),e=c;
 for(int q=c-1;q<=c+1;++q){ float s=tp(10.f,float(q)); ulong m=tdiv(x,t64(s)); if(!soft64_less(m,0x3ff0000000000000ul)&&soft64_less(m,0x4024000000000000ul)){e=q;break;} }
 float s=tp(10.f,float(e)); int digit=soft64_to_i32_rz(tdiv(x,t64(s))).value; return clamp(digit+9*(e-first)-1,0,n-1);
}
static inline float tbound_ice(float mass,float rho,float ni) {
 if(mass<=1.e-12f)return 0.f;
 const float am=3.1415926536f*890.f/6.f; float number=max(1.e-6f,ni*rho);
 ulong lambda=t64(tp(am*6.f*number/mass,1.f/3.f));float md=t32(tdiv(0x4010000000000000ul,lambda));
 if(md<5.e-6f){lambda=tdiv(0x4010000000000000ul,0x3ed4f8b588e368f1ul);number=min(999.e3f,((1.f/6.f)*mass/am)*t32(tmul(tmul(lambda,lambda),lambda)));}
 else if(md>300.e-6f){lambda=tdiv(0x4010000000000000ul,0x3f33a92a30553261ul);number=((1.f/6.f)*mass/am)*t32(tmul(tmul(lambda,lambda),lambda));}
 return min(number,999.e3f)/rho;
}
static inline float tbound_rain(float mass,float rho,float nr) {
 if(mass<=1.e-12f)return 0.f;
 const float am=3.1415926536f*1000.f/6.f; float number=max(1.e-6f,nr*rho);
 float lambda=tp(am*6.f*number/mass,1.f/3.f),mvd=3.672f/lambda;
 if(mvd>2.5e-3f)mvd=2.5e-3f;else if(mvd<37.5e-6f)mvd=37.5e-6f;else return nr;
 lambda=3.672f/mvd; number=(1.f/6.f)*mass/am*lambda*lambda*lambda; return number/rho;
}
static inline bool train(float qr,float nr,float rho,thread float &number,thread ulong &lambda,thread float &mvd) {
 number=1.e-6f;lambda=0x3ff0000000000000ul;mvd=0.f;if(qr<=1.e-12f)return false;
 float mass=qr*rho;const float am=3.1415926536f*1000.f/6.f;number=max(1.e-6f,nr*rho);
 if(number<=1.e-6f){float l=3.672f/1.e-3f;number=(1.f/6.f)*mass/am*l*l*l;}
 float lf=tp(am*6.f*number/mass,1.f/3.f);mvd=3.672f/lf;
 if(mvd>2.5e-3f){mvd=2.5e-3f;lf=3.672f/mvd;number=(1.f/6.f)*mass/am*lf*lf*lf;}
 else if(mvd<37.5e-6f){mvd=37.5e-6f;lf=3.672f/mvd;number=(1.f/6.f)*mass/am*lf*lf*lf;}
 lambda=t64(lf);return true;
}

kernel void thompson_frozen_vapor_cloud_network(
 constant ThompsonFrozenVaporCloudNetworkResources &r [[buffer(0)]],
 constant ThompsonFrozenVaporCloudNetworkParams &p [[buffer(1)]], uint idx [[thread_position_in_grid]]) {
 if(idx>=uint(p.n))return;
 if(p.include_snow_rime_conversion!=0)r.snow_velocity_boost[idx]=1.f;
 if(r.temperature[idx]>=273.15f){
   if(r.qi[idx]>0.f||r.ni[idx]>0.f){float qve=max(1.e-10f,r.qv[idx]);float re=.622f*r.pressure[idx]/(287.04f*r.temperature[idx]*(qve+.622f));r.ni[idx]=tbound_ice(r.qi[idx]*re,re,r.ni[idx]);}
   return;
 }
 const float temp0=r.temperature[idx],qv0=max(1.e-10f,r.qv[idx]);
 const float qvsi=trsif(r.pressure[idx],temp0),qvs=trslf(r.pressure[idx],temp0);
 float ssati=qv0/qvsi-1.f,ssatw=qv0/qvs-1.f;
 if(fabs(ssati)<1.e-15f)ssati=0.f;if(fabs(ssatw)<1.e-15f)ssatw=0.f;
 const bool nuc=ssati>=.25f||(ssatw>1.e-15f&&temp0<253.15f);
 if(r.qi[idx]<=1.e-12f&&r.qs[idx]<=1.e-12f&&r.qg[idx]<=1.e-12f&&r.qr[idx]<=1.e-12f&&r.qc[idx]<=1.e-12f&&!nuc)return;
 const float rho=.622f*r.pressure[idx]/(287.04f*temp0*(qv0+.622f)),orho=1.f/rho,dt=p.dt,idt=1.f/dt,tempc=temp0-273.15f,it=1.f/temp0;
 const float diff=2.11e-5f*tp(temp0/273.15f,1.94f)*(101325.f/r.pressure[idx]);
 const float visc=(1.718f+.0049f*tempc-1.2e-5f*tempc*tempc)*1.e-5f;
 const float cond=(5.69f+.0168f*tempc)*1.e-5f*418.936f;
 const float icp=1.f/(1004.f*(1.f+.887f*qv0)),sd=rho*qvsi,st=it*(2.834e6f*it*(1.f/461.5f)-1.f),slope=sd*st;
 const float curve=sd*(st*st-2.f*2.834e6f*it*it*it*(1.f/461.5f)+it*it);
 const float gamma=2.834e6f*diff/cond*slope,gr=gamma/(1.f+gamma);
 const float alpha=max(1.e-9f,.5f*gr*gr*curve/slope*sd/slope);
 float xsat=ssati;if(fabs(xsat)<1.e-9f)xsat=0.f;const float xs2=xsat*xsat,a2=alpha*alpha;
 const float vg=4.f*3.1415926536f*(1.f-alpha*xsat+2.f*a2*xs2-5.f*a2*alpha*xs2*xsat)/(1.f+gamma);
 const float vlim=(qv0-qvsi)*rho*idt*.999f;
 ulong nucRate=0ul,nucN=0ul;
 if(nuc){float oldN=r.qi[idx]>1.e-12f?max(1.e-6f,r.ni[idx]*rho):0.f;
 float target=min(250.e3f,5.f*te(.304f*(273.15f-temp0)));
 nucN=t64(max(0.f,target-oldN)*idt);nucRate=tmin(t64(vlim),tmul(0x3d719799812dea11ul,nucN));nucN=tdiv(nucRate,0x3d719799812dea11ul);}
 ulong iceRate=0ul,toSnow=0ul,iceNum=0ul,autoM=0ul,autoN=0ul,rIce=0ul,rRain=0ul,rIceN=0ul,rRainN=0ul,rPair=0ul;
 float iceParticle=1.e-12f,iceMass=r.qi[idx]*rho,iceNumber=max(1.e-6f,r.ni[idx]*rho),rainMass=r.qr[idx]*rho,rainN,mvd;ulong rainLambda;
 bool rainActive=train(r.qr[idx],r.nr[idx],rho,rainN,rainLambda,mvd); ulong rainSelf=0ul;
 if(rainActive&&mvd>50.e-6f){float eff=1.f-te(2300.f*(mvd-1950.e-6f));rainSelf=t64(eff*2.f*rainN*rainMass);}
 float flooredIceDiameter=0.f;
 if(r.qi[idx]>1.e-12f){const float ami=3.1415926536f*890.f/6.f;
 if(iceNumber<=1.e-6f){ulong l5=tdiv(0x4010000000000000ul,0x3ed4f8b588e368f1ul);ulong l53=tmul(tmul(l5,l5),l5);iceNumber=min(999.e3f,((1.f/6.f)*iceMass/ami)*t32(l53));}
 ulong lam=t64(tp(ami*6.f*iceNumber/iceMass,1.f/3.f));ulong ilam=tdiv(0x3ff0000000000000ul,lam);
 float md=t32(tmul(0x4010000000000000ul,ilam));
 if(md<5.e-6f){lam=tdiv(0x4010000000000000ul,0x3ed4f8b588e368f1ul);ilam=tdiv(0x3ff0000000000000ul,lam);iceNumber=min(999.e3f,(1.f/6.f)*iceMass/ami*t32(tmul(tmul(lam,lam),lam)));md=5.e-6f;}
 else if(md>300.e-6f){lam=tdiv(0x4010000000000000ul,0x3f33a92a30553261ul);ilam=tdiv(0x3ff0000000000000ul,lam);iceNumber=(1.f/6.f)*iceMass/ami*t32(tmul(tmul(lam,lam),lam));md=300.e-6f;}
 flooredIceDiameter=max(tp(1.e-12f/ami,1.f/3.f),md);iceParticle=ami*flooredIceDiameter*flooredIceDiameter*flooredIceDiameter;int mb=iceMass>1.e-10f?tindex(iceMass,-10,64):0,nb=iceNumber>1.f?tindex(iceNumber,0,55):0;
 ulong total=t64(.5f*vg*diff*ssati*sd*iceNumber);total=tmul(total,ilam);
 if(tpos(total)){total=tmin(total,t64(vlim));ulong frac=r.ice_deposition_partition[mb+64*nb];iceRate=tmul(frac,total);toSnow=tmul(tsub(0x3ff0000000000000ul,frac),total);}
 else {total=tmax(tmax(t64(-iceMass*idt),total),t64(vlim));float minD=tp(1.e-12f/ami,1.f/3.f),pd=max(minD,md),pm=ami*pd*pd*pd;iceNum=tmax(t64(-iceNumber*idt),tdiv(total,t64(pm)));iceRate=total;}
 int ti=mb+64*nb;
 if(mb==63||md>1500.e-6f){autoM=tmul(t64(iceMass*.99f),t64(idt));autoN=tmul(t64(iceNumber*.95f),t64(idt));}
 else if(md>=30.e-6f){autoM=tmin(tmul(t64(iceMass*.99f),t64(idt)),tmul(r.ice_to_snow_mass[ti],t64(idt)));autoN=tmin(tmul(t64(iceNumber*.95f),t64(idt)),tmul(r.ice_to_snow_number[ti],t64(idt)));}
 if(rainMass>=1.e-6f&&mvd>4.f*flooredIceDiameter){float df=precise::sqrt((101325.f/(287.05f*298.f))*orho);ulong intercept=tmul(t64(rainN),rainLambda),shift=tadd(rainLambda,t64(195.f));float me=.95f,masspre=3.1415926536f*.25f*4854.f*6.f,amr=3.1415926536f*1000.f/6.f,rmp=3.1415926536f*.25f*amr*4854.f*720.f;
 rIce=tmul(t64(df*masspre*me*iceMass),tmul(intercept,soft64_plm_pow(shift,t64(-4.f))));rRain=tmul(t64(df*rmp*me*iceNumber),tmul(intercept,soft64_plm_pow(shift,t64(-7.f))));rRain=tmin(rRain,t64(rainMass*idt));rIceN=tdiv(rIce,t64(iceParticle));rRainN=tmin(tmul(t64(df*masspre*me*iceNumber),tmul(intercept,soft64_plm_pow(shift,t64(-4.f)))),t64(rainN*idt));rPair=tadd(rIce,rRain);}
 }
 // Complete cold-rain source group.  These diagnostics share the immutable
 // entry state and are capped together below, after every source is formed.
 ulong freezeI=0ul,freezeIN=0ul,freezeG=0ul,freezeGN=0ul;
 ulong rsRain=0ul,rsSnow=0ul,rsG=0ul,rsN=0ul;
 ulong rgRain=0ul,rgG=0ul,rgN=0ul;
 const float pi=3.1415926536f,amr=pi*1000.f/6.f;
 int rmb=tindex(rainMass,-6,37);
 ulong rinter=tmul(t64(((1.f/6.f)*rainMass)/amr),
                   tmul(tmul(tmul(rainLambda,rainLambda),rainLambda),rainLambda));
 int rib=tindex64(rinter,6,37);
 if(rainActive){
   if(rainMass>1.e-6f){int tb=clamp(int(precise::round(-(temp0-273.15f)))-1,0,44),nb=27;
     int ix=rmb+37*(rib+37*(tb+45*nb));
     freezeI=tdiv(r.rain_to_ice_mass[ix],t64(dt));
     freezeIN=tdiv(r.rain_to_ice_number[ix],t64(dt));
     freezeG=tdiv(r.rain_to_graupel_mass[ix],t64(dt));
     freezeGN=tdiv(tmin(t64(rainN),r.rain_to_graupel_number[ix]),t64(dt));
   } else if(rainMass>1.e-12f&&temp0<235.16f){
     freezeI=tdiv(t64(rainMass),t64(dt));freezeIN=tdiv(t64(rainN),t64(dt));
   }
   float sm= r.qs[idx]*rho;
   if(rainMass>=1.e-6f&&sm>=1.e-6f){int sb=tindex(sm,-6,37);
     int raw=int((tempc-2.5f)/5.f)-1;
     int tb=clamp(-raw,1,9)-1,ix=sb+37*(tb+9*(rib+37*rmb));
     rsRain=tneg(tadd(tadd(r.tmr_racs2[ix],r.tcr_sacr2[ix]),tadd(r.tmr_racs1[ix],r.tcr_sacr1[ix])));
     rsSnow=tsub(tadd(r.tmr_racs2[ix],r.tcr_sacr2[ix]),tadd(r.tcs_racs1[ix],r.tms_sacr1[ix]));
     rsG=tadd(tadd(r.tmr_racs1[ix],r.tcr_sacr1[ix]),tadd(r.tcs_racs1[ix],r.tms_sacr1[ix]));
     rsN=tadd(tadd(r.tnr_racs1[ix],r.tnr_racs2[ix]),tadd(r.tnr_sacr1[ix],r.tnr_sacr2[ix]));
     rsRain=tmax(t64(-rainMass/dt),rsRain);rsSnow=tmax(t64(-sm/dt),rsSnow);
     rsG=tmin(t64((rainMass+sm)/dt),rsG);rsN=tmin(t64(rainN/dt),rsN);
   }
   float gmass=r.qg[idx]*rho;
   if(rainMass>=1.e-6f&&gmass>=1.e-6f){int gmb=tindex(gmass,-6,37);
     float ip=max(2.f,min(6.f,3.f+(2.f/7.f)*(tl10(max(1.e-9f,gmass))+8.f)));
     int gib=tindex64(t64(tp(10.f,ip)),2,37);
     int base=gib+37*(gmb+37*(0+1*(rib+37*rmb)));
     int ix=base+4*37*37,sz=37*37*1*37*37;
     if(ix<sz){rgG=tadd(r.tmr_racg[ix],r.tcr_gacr[ix]);rgG=tmin(t64(rainMass/dt),rgG);
       rgRain=tneg(rgG);rgN=tadd(r.tnr_racg[ix],r.tnr_gacr[ix]);rgN=tmin(t64(rainN/dt),rgN);}
   }
 }
 const float cloudMass=max(0.f,r.qc[idx]*rho),cloudNumber=100.e6f;
 ulong cloudAuto=0ul,cloudAutoN=0ul,cloudRain=0ul,cloudFreeze=0ul,cloudFreezeN=0ul;
 ulong snowRime=0ul,graupRime=0ul,hmN=0ul,hmM=0ul,snowHm=0ul,graupHm=0ul;
 float cloudMvd=1.e-6f,snowRimeMoment0=0.f,snowRimeDiameter=0.f;
 const float cloudAm=pi*1000.f/6.f;
 const bool lqc=r.qc[idx]>1.e-12f;
 float cloudLambda=0.f;
 if(lqc){cloudLambda=tp(cloudNumber*cloudAm*1.30767389e12f*2.08767448e-9f/cloudMass,1.f/3.f);cloudMvd=max(1.e-6f,min(15.672f/cloudLambda,50.e-6f));}
 if(cloudMass>.01e-3f){
   float xdc=max(1.f,tp(cloudMass/(cloudAm*cloudNumber),1.f/3.f)*1.e6f);
   float dcg=tp(6.40238373e15f*7.64716632e-13f,1.f/3.f)/cloudLambda*1.e6f;
   float xdc3=xdc*xdc*xdc,dcg3=dcg*dcg*dcg;
   float dcb=tp(max(0.f,xdc3*dcg3-xdc3*xdc3),1.f/6.f);
   float zt=6.25e-6f*xdc*dcb*dcb*dcb-.4f,z1=.5f*(zt+fabs(zt));
   float zeta=.027f*cloudMass*z1,td=.5f*dcb-7.5f,taud=.5f*(td+fabs(td))+1.e-12f;
   float tau=3.72f/(cloudMass*taud);
   cloudAuto=tmin(t64(cloudMass/dt),t64(zeta/tau));
   cloudAutoN=tdiv(cloudAuto,t64(cloudAm*12.f*10.f*50.e-6f*50.e-6f*50.e-6f));
 }
 if(rainActive&&lqc&&mvd>50.e-6f&&cloudMvd>1.e-6f){
   const ulong dr0=0x3f0ad333f9742941ul,dr1=0x3f740389985ae9dbul;
   ulong lf=tdiv(soft64_plm_log(tdiv(t64(mvd),dr0)),soft64_plm_log(tdiv(dr1,dr0)));
   int rb=min(1+int(soft64_to_i32_rz(tmul(t64(100.),lf)).value),100),cb=int(cloudMvd*1.e6f);
   float eff=t32(r.rain_cloud_efficiency[(rb-1)+100*(cb-1)]);
   float df=precise::sqrt((101325.f/(287.05f*298.f))/rho),coef=pi*.25f*4854.f*6.f;
   float intercept=rainN*t32(rainLambda),rate=df*coef*eff*cloudMass*intercept*tp(t32(rainLambda)+195.f,-4.f);
   cloudRain=tmin(t64(cloudMass/dt),t64(rate));
 }
 if(cloudMass>1.e-6f){
   int mb=tindex(cloudMass,-6,37),cn=65,tb=clamp(int(precise::round(-tempc))-1,0,44),nb=27;
   int ix=mb+37*(cn+100*(tb+45*nb));
   cloudFreeze=tmin(t64(cloudMass/dt),tdiv(r.cloud_to_ice_mass[ix],t64(dt)));
   cloudFreezeN=tmin(t64(cloudNumber/dt),tmin(tdiv(cloudFreeze,0x3d819799812dea11ul),tdiv(r.cloud_to_ice_number[ix],t64(dt))));
 } else if(cloudMass>1.e-12f&&temp0<235.16f){cloudFreeze=tdiv(t64(cloudMass),t64(dt));cloudFreezeN=tdiv(t64(cloudNumber),t64(dt));}
 float cloudSnow=max(0.f,r.qs[idx]*rho);
 if(lqc&&r.qs[idx]>1.e-12f&&cloudMvd>1.e-6f){
   float st=min(-.1f,tempc),sm2=cloudSnow*(1.f/.069f);
   float sm3=tfield_a(st,3.f)*tp(sm2,tfield_b(st,3.f));
   snowRimeMoment0=tfield_a(st,0.f)*tp(sm2,tfield_b(st,0.f));snowRimeDiameter=sm3/sm2;
   if(snowRimeDiameter>300.e-6f){
     float rimingMoment=tfield_a(st,2.55f)*tp(sm2,tfield_b(st,2.55f));
     ulong dr=tdiv(0x3f947ae147ae147bul,0x3f33a92a30553261ul),logr=soft64_plm_log(dr);
     ulong first=tmul(0x3f33a92a30553261ul,soft64_plm_exp(tmul(0x3f747ae147ae147bul,logr)));
     ulong last=tmul(0x3f33a92a30553261ul,soft64_plm_exp(tmul(tdiv(t64(99.5f),t64(100.f)),logr)));
     ulong binratio=tdiv(soft64_plm_log(tdiv(t64(snowRimeDiameter),first)),soft64_plm_log(tdiv(last,first)));
     int sb=min(1+int(soft64_to_i32_rz(tmul(t64(100.),binratio)).value),100);
     ulong sbfrac=tdiv(tsub(soft64_from_i32(sb),t64(.5f)),t64(100.f));
     ulong tsd=tmul(0x3f33a92a30553261ul,soft64_plm_exp(tmul(sbfrac,logr)));
     int cb=int(cloudMvd*1.e6f);ulong tcd=tmul(soft64_from_i32(cb),0x3eb0c6f7a0b5ed8dul);
     ulong cv=tmul(0x40c73e0000000000ul,tmul(tmul(tmul(0x40c3880000000000ul,tcd),tcd),0x3fd0000000000000ul));
     ulong sv=tsub(tmul(t64(40.),tmul(soft64_plm_pow(tsd,0x3fe199999999999aul),soft64_plm_exp(tneg(tmul(t64(100.),tsd))))),cv);
     ulong pi64=0x400921fb54442d18ul;
     ulong am64=tdiv(tmul(pi64,0x408f400000000000ul),0x4018000000000000ul);
     ulong meltarg=tdiv(tmul(tmul(0x3fb1a9fbe76c8b44ul,tsd),tsd),am64);
     ulong melt=soft64_plm_pow(meltarg,0x3fd5555555555555ul);
     ulong frac=tdiv(tcd,melt);float eff=0.f;
     if(!soft64_greater(frac,0x3fd0000000000000ul)&&!soft64_less(tcd,0x3ed92a737110e454ul)&&!soft64_less(sv,0x3f50624dd2f1a9fcul)){
       ulong stokes=tdiv(tmul(tmul(tmul(tcd,tcd),sv),0x408f400000000000ul),tmul(0x3ef203b89c3e39a2ul,melt));
       ulong re=tdiv(tmul(t64(9.),stokes),tmul(tmul(frac,frac),t64(1000.)));
       ulong lr=soft64_plm_log(re),k0=soft64_plm_exp(tadd(tadd(0xbfb9c779a6b50b0ful,tneg(tmul(0x3fd6e978d4fdf3b6ul,lr))),tmul(0x3f9ab9f559b3d07dul,tmul(lr,lr))));
       ulong z=soft64_plm_log(tdiv(stokes,tadd(k0,0x3cd203af9ee75616ul)));
       ulong h=tadd(tadd(tadd(0x3fc2c083126e978dul,tmul(0x3ff4d4fdf3b645a2ul,z)),tneg(tmul(0x3fe36c8b43958106ul,tmul(z,z)))),tmul(0x3fd2c083126e978dul,tmul(tmul(z,z),z)));
       ulong yc=tmul(tdiv(0x4000000000000000ul,0x400921fb54442d18ul),tatan(h));
       ulong y=tdiv(tmul(tadd(yc,frac),tadd(yc,frac)),tmul(tadd(0x3ff0000000000000ul,frac),tadd(0x3ff0000000000000ul,frac)));
       eff=max(0.f,min(t32(y),.95f));
     }
     float df=precise::sqrt((101325.f/(287.05f*298.f))/rho),pre=pi*.25f*40.f;
     snowRime=tmin(t64(df*pre*eff*cloudMass*rimingMoment),t64(cloudMass/dt));
   }
 }
 float cloudGraupel=max(0.f,r.qg[idx]*rho);
 if(lqc&&cloudGraupel>=1.e-6f&&cloudMvd>1.e-6f){
   float amg=pi*400.f/6.f,ip=max(2.f,min(6.f,3.f+(2.f/7.f)*(tl10(max(1.e-9f,cloudGraupel))+8.f)));
   float ig=tp(10.f,ip);ulong lambda=t64(tp(ig*amg*6.f/cloudGraupel,.25f));float mvdg=t32(tdiv(0x400d604189374bc7ul,lambda));
   if(mvdg>25.4e-3f)lambda=tdiv(0x400d604189374bc7ul,0x3f9a027525460aa6ul);else if(mvdg<50.e-6f)lambda=tdiv(0x400d604189374bc7ul,0x3f0a36e2eb1c432dul);
   ulong ilambda=tdiv(0x3ff0000000000000ul,lambda);float ng=((1.f/6.f)*cloudGraupel/amg)*t32(tmul(tmul(lambda,lambda),lambda));
   ulong inter=tmul(t64(ng),lambda);float df=precise::sqrt((101325.f/(287.05f*298.f))/rho),cloudVisc=(1.718f+.0049f*tempc-1.2e-5f*tempc*tempc)*1.e-5f;
   float diam=t32(tmul(t64(4.),ilambda));float vel=t32(tmul(t64(df*442.f*20.3632278f*(1.f/6.f)),soft64_plm_pow(ilambda,0x3fec7ae147ae147bul)));
   float st=cloudMvd*cloudMvd*vel*1000.f/(9.f*cloudVisc*diam),eff=0.f;
   if(st>=.4f&&st<=10.f)eff=.55f*tl10(2.51f*st);else if(st>10.f)eff=.77f;
   float pre=pi*.25f*442.f*5.23476267f;
   graupRime=tmul(t64(df*pre*eff*cloudMass),tmul(inter,soft64_plm_pow(ilambda,0x400f1eb851eb851ful)));
 }
 if(soft64_greater(graupRime,0x3cd203af9ee75616ul)&&tempc> -8.f){float f=0.f;if(tempc>=-5.f&&tempc< -3.f)f=.5f*(-3.f-tempc);else if(tempc> -8.f&&tempc< -5.f)f=.33333333f*(8.f+tempc);
   hmN=tmul(tmul(0x41b4dcd650000000ul,t64(f)),graupRime);hmM=tmul(0x3d719799812dea11ul,hmN);ulong sum=tadd(snowRime,graupRime);
   if(tpos(sum)){snowHm=tmul(tdiv(snowRime,sum),hmM);graupHm=tmul(tdiv(graupRime,sum),hmM);}
 }
 if((p.include_cold_rain!=0||p.include_cold_cloud!=0)&&nuc){float existing=iceNumber;
   float target=min(250.e3f,5.f*te(.304f*(273.15f-temp0)));
   nucN=tmul(t64(max(0.f,target-existing-t32(tmul(tadd(freezeIN,cloudFreezeN),t64(dt))))),t64(idt));
   nucRate=tmin(t64(vlim),tmul(t64(1.e-12f),nucN));nucN=tdiv(nucRate,t64(1.e-12f));}
 // The remaining rates and grouped WRF species conservation are kept in
 // source order. (Cold-rain tables use the source's documented flattened
 // Fortran address expressions.)
 ulong snowRate=0ul,snowCollect=0ul,snowCollectN=0ul,graupRate=0ul;
 float snowMass=r.qs[idx]*rho;
 if(r.qs[idx]>1.e-12f){float rf=precise::sqrt((101325.f/(287.05f*298.f))*orho),rfs=precise::sqrt(rf),vf=precise::sqrt(rho/visc),m2=snowMass*(1.f/.069f),tc=min(-.1f,tempc);
 float m1=tfield_a(tc,1.f)*tp(m2,tfield_b(tc,1.f)),vm=1.f+(1.f+.55f)*.5f;
 float mv=tfield_a(tc,vm)*tp(m2,tfield_b(tc,vm)),mr=tfield_a(tc,2.55f)*tp(m2,tfield_b(tc,2.55f));
 float cap=max(.15f,min(.5f,.15f+((tempc+1.5f)*(.5f-.15f))/(-30.f+1.5f)));
 float vc=.28f*tp(.632f,1.f/3.f)*precise::sqrt(40.f),mom=.86f*m1+vc*rfs*vf*mv;
 snowRate=t64(cap*vg*diff*ssati*sd*mom);
 if(tpos(snowRate))snowRate=tmin(snowRate,t64(vlim));else snowRate=tmax(tmax(t64(-snowMass*idt),snowRate),t64(vlim));
 if(r.qi[idx]>1.e-12f&&snowMass>=1.e-6f){float df=precise::sqrt((101325.f/(287.05f*298.f))*orho);snowCollect=t64((3.1415926536f*.25f*40.f)*df*.05f*iceMass*mr);snowCollectN=tdiv(snowCollect,t64(iceParticle));}
 }
 ulong snowGraupel=0ul,snowGraupelN=0ul;
 if(p.include_snow_rime_conversion!=0&&soft64_greater(snowRime,tmul(t64(2.f),snowRate))&&soft64_greater(snowRate,0x3cd203af9ee75616ul)){
   float rr=min(30.f,t32(tdiv(snowRime,snowRate))),gf=min(.95f,.15f+(rr-2.f)*.028f);
   r.snow_velocity_boost[idx]=min(1.5f,1.1f+(rr-2.f)*.014f);
   snowGraupel=tmul(t64(gf),snowRime);snowGraupelN=tdiv(tmul(snowGraupel,t64(snowRimeMoment0)),t64(snowMass));
   float sv=40.f*tp(snowRimeDiameter,.55f)*te(-100.f*snowRimeDiameter);
   float rp=-(cloudMvd*.5e6f)*sv/min(-.1f,tempc);rp=max(.1f,min(rp,10.f));
   float rd=(.051f+.114f*rp-.0055f*rp*rp)*1000.f;
   if(rd<150.f){gf=0.f;snowGraupel=0ul;snowGraupelN=0ul;}
   snowRime=tmul(tsub(0x3ff0000000000000ul,t64(gf)),snowRime);
 }
 float gm=r.qg[idx]*rho;
 if(r.qg[idx]>1.e-12f&&ssati< -1.e-15f){float rf=precise::sqrt((101325.f/(287.05f*298.f))*orho),rfs=precise::sqrt(rf),vf=precise::sqrt(rho/visc),amg=3.1415926536f*400.f/6.f;
 float ip=max(2.f,min(6.f,3.f+(2.f/7.f)*(tl10(max(1.e-9f,gm))+8.f))),inter=tp(10.f,ip),lambda=tp(inter*amg*6.f/gm,.25f);
 float nk=max(1.e-6f,((1.f/6.f)*gm*tp(lambda,3.f))/amg/rho),gn=max(1.e-6f,nk*rho);lambda=tp(amg*6.f*gn/gm,1.f/3.f);
 float mv=3.672f/lambda;if(mv>25.4e-3f){mv=25.4e-3f;lambda=3.672f/mv;gn=((1.f/6.f)*gm*tp(lambda,3.f))/amg;}else if(mv<50.e-6f){mv=50.e-6f;lambda=3.672f/mv;gn=((1.f/6.f)*gm*tp(lambda,3.f))/amg;}
 float il=1.f/lambda,ni=gn*lambda,vc=.28f*tp(.632f,1.f/3.f)*precise::sqrt(442.f)*1.9021706581115723f;
 float mom=ni*(.86f*tp(il,2.f)+vc*vf*rfs*tp(il,2.945f));graupRate=t64(.5f*vg*diff*ssati*sd*mom);graupRate=tmax(tmax(t64(-gm*idt),graupRate),t64(vlim));
 }
 // Shared source caps execute after every process has been diagnosed.
 ulong vapor=tadd(tadd(tadd(tadd(nucRate,iceRate),toSnow),snowRate),graupRate);
 if((soft64_greater(vapor,0x3cd203af9ee75616ul)&&soft64_greater(vapor,t64(vlim)))||(soft64_less(vapor,0xbcd203af9ee75616ul)&&soft64_less(vapor,t64(vlim)))){
   ulong ratio=tdiv(t64(vlim),vapor);nucRate=tmul(nucRate,ratio);iceRate=tmul(iceRate,ratio);toSnow=tmul(toSnow,ratio);iceNum=tmul(iceNum,ratio);snowRate=tmul(snowRate,ratio);graupRate=tmul(graupRate,ratio);vapor=t64(vlim);
 }
 ulong cloudLimit=t64(-cloudMass*idt),cloudSum=tneg(tadd(tadd(tadd(tadd(tadd(cloudAuto,cloudFreeze),cloudRain),snowRime),snowGraupel),graupRime));
 if(p.include_cold_cloud!=0&&r.qc[idx]>1.e-12f&&soft64_less(cloudSum,cloudLimit)){
   ulong ratio=tdiv(cloudLimit,cloudSum);cloudAuto=tmul(cloudAuto,ratio);cloudFreeze=tmul(cloudFreeze,ratio);cloudRain=tmul(cloudRain,ratio);snowRime=tmul(snowRime,ratio);snowGraupel=tmul(snowGraupel,ratio);graupRime=tmul(graupRime,ratio);
 }
 ulong ilim=t64(-iceMass*idt),isum=tsub(tsub(tsub(iceRate,autoM),snowCollect),rIce);
 if(r.qi[idx]>1.e-12f&&soft64_less(isum,0xbcd203af9ee75616ul)&&soft64_less(isum,ilim)){
   ulong ratio=tdiv(ilim,isum),old=iceRate;iceRate=tmul(iceRate,ratio);autoM=tmul(autoM,ratio);snowCollect=tmul(snowCollect,ratio);rIce=tmul(rIce,ratio);vapor=tadd(vapor,tsub(iceRate,old));
 }
 ulong rlim=t64(-rainMass*idt),rsum=tadd(tadd(tadd(tneg(freezeG),tneg(freezeI)),tneg(rRain)),tadd(rsRain,rgRain));
 if(rainActive&&soft64_less(rsum,rlim)){ulong ratio=tdiv(rlim,rsum);freezeG=tmul(freezeG,ratio);freezeI=tmul(freezeI,ratio);rRain=tmul(rRain,ratio);rsRain=tmul(rsRain,ratio);rgRain=tmul(rgRain,ratio);}
 ulong slim=t64(-snowMass*idt),ssum=tadd(tsub(snowRate,snowHm),rsSnow);
 if(r.qs[idx]>1.e-12f&&soft64_less(ssum,slim)){ulong ratio=tdiv(slim,ssum),old=snowRate;snowRate=tmul(snowRate,ratio);snowHm=tmul(snowHm,ratio);rsSnow=tmul(rsSnow,ratio);vapor=tadd(vapor,tsub(snowRate,old));}
 ulong glim=t64(-gm*idt),gsum=tadd(tsub(graupRate,graupHm),rgG);
 if(r.qg[idx]>1.e-12f&&soft64_less(gsum,glim)){ulong ratio=tdiv(glim,gsum),old=graupRate;graupRate=tmul(graupRate,ratio);graupHm=tmul(graupHm,ratio);rgG=tmul(rgG,ratio);vapor=tadd(vapor,tsub(graupRate,old));}
 ulong paired=tmin(tabs(rgRain),tabs(rgG));rgRain=tneg(paired);rgG=paired;
 ulong hmMass=tadd(snowHm,graupHm);
 ulong rainRate=tadd(tadd(tadd(tadd(cloudAuto,cloudRain),tneg(freezeG)),tneg(freezeI)),tadd(tadd(tneg(rRain),rsRain),rgRain));
 ulong rainNumberSink=tadd(tadd(tadd(tadd(tadd(rainSelf,freezeGN),freezeIN),rRainN),rsN),rgN);
 float latentV=2.5e6f+(2106.f-4218.f)*tempc,latentF=2.834e6f-latentV;
 ulong fusion=tadd(cloudFreeze,snowRime);fusion=tadd(fusion,snowGraupel);fusion=tadd(fusion,graupRime);fusion=tadd(fusion,freezeI);fusion=tadd(fusion,freezeG);fusion=tadd(fusion,rRain);fusion=tadd(fusion,rsSnow);fusion=tadd(fusion,rsG);fusion=tadd(fusion,rgG);
 float postQv=r.qv[idx]-t32(tmul(vapor,t64(orho)))*dt;
 ulong heat=tadd(tmul(tmul(tmul(t64(2.834e6f*icp),vapor),t64(orho)),t64(dt)),
                  tmul(tmul(tmul(t64(latentF*icp),fusion),t64(orho)),t64(dt)));
 float postT=temp0+t32(heat);
 float qrEntry=r.qr[idx]>1.e-12f?r.qr[idx]:0.f,nrEntry=r.qr[idx]>1.e-12f?r.nr[idx]:0.f;
 float qgEntry=r.qg[idx]>1.e-12f?r.qg[idx]:0.f;
 float ngEntry=(qgEntry>0.f&&p.track_graupel_number!=0)?r.graupel_number_shadow[idx]:0.f;
 ulong qir=tadd(nucRate,hmMass);qir=tadd(qir,cloudFreeze);qir=tadd(qir,freezeI);qir=tadd(qir,iceRate);qir=tsub(tsub(tsub(qir,autoM),snowCollect),rIce);
 ulong nir=tadd(nucN,hmN);nir=tadd(nir,cloudFreezeN);nir=tadd(nir,freezeIN);nir=tadd(nir,iceNum);nir=tsub(tsub(tsub(nir,autoN),snowCollectN),rIceN);
 float qi=max(0.f,r.qi[idx]+t32(tmul(qir,t64(orho)))*dt);
 float ni=max(0.f,r.ni[idx]+t32(tmul(nir,t64(orho)))*dt);
 ulong qsr=tadd(toSnow,snowRate);qsr=tadd(qsr,autoM);qsr=tadd(qsr,snowCollect);qsr=tadd(qsr,snowRime);qsr=tadd(qsr,rsSnow);qsr=tsub(qsr,snowHm);
 ulong qgr=tadd(graupRate,freezeG);qgr=tadd(qgr,snowGraupel);qgr=tadd(qgr,graupRime);qgr=tsub(qgr,graupHm);qgr=tadd(qgr,rPair);qgr=tadd(qgr,rsG);qgr=tadd(qgr,rgG);
 float qs=max(0.f,r.qs[idx]+t32(tmul(qsr,t64(orho)))*dt);
 float qg=max(0.f,r.qg[idx]+t32(tmul(qgr,t64(orho)))*dt);
 float qr=max(0.f,r.qr[idx]+t32(tmul(rainRate,t64(orho)))*dt);
 float nr=max(0.f,r.nr[idx]+t32(tmul(tsub(cloudAutoN,rainNumberSink),t64(orho)))*dt);
 ni=tbound_ice(qi*rho,rho,ni);nr=tbound_rain(qr*rho,rho,nr);
 if(p.track_graupel_number!=0){
   float initial=r.graupel_number_shadow[idx];
   float initialNumber=qgEntry>0.f?max(1.e-6f,initial*rho):0.f;
   ulong graupVapor=qgEntry>0.f?tmul(graupRate,tdiv(t64(initialNumber),t64(gm))):0ul;
   ulong gnRate=tadd(tadd(tadd(tadd(freezeGN,snowGraupelN),rRainN),rsN),graupVapor);
   r.graupel_number_shadow[idx]=initial+t32(tmul(tmul(gnRate,t64(orho)),t64(dt)));
 }
 ulong qcr=tadd(cloudAuto,cloudFreeze);qcr=tadd(qcr,cloudRain);qcr=tadd(qcr,snowRime);qcr=tadd(qcr,snowGraupel);qcr=tadd(qcr,graupRime);
 float qc=max(0.f,r.qc[idx]-t32(tmul(qcr,t64(orho)))*dt);
 if(!(qr*rho>1.e-12f)){qr=qrEntry+(-qrEntry*idt)*dt;nr=nrEntry+(-nrEntry*idt)*dt;}
 const float xrg=qg*rho;
 if(!(xrg>1.e-12f)){qg=qgEntry+(-qgEntry*idt)*dt;if(p.track_graupel_number!=0)r.graupel_number_shadow[idx]=ngEntry+(-ngEntry*idt)*dt;}
 else if(p.track_graupel_number!=0){float amg=3.1415926536f*400.f/6.f;
   float xng=max(1.e-6f,r.graupel_number_shadow[idx]*rho);
   ulong lamg=t64(tp(amg*6.f*xng/xrg,1.f/3.f));float mvdg=t32(tdiv(t64(3.672f),lamg));bool bounded=false;
   if(mvdg>25.4e-3f){lamg=t64(3.672f/25.4e-3f);bounded=true;}else if(mvdg<50.e-6f){lamg=t64(3.672f/50.e-6f);bounded=true;}
   if(bounded){ulong xnb=tdiv(tmul(t64((1.f/6.f)*xrg),soft64_plm_pow(lamg,t64(3.f))),t64(amg));float xngb=t32(xnb);float ngten=(xngb-ngEntry*rho)*idt*(1.f/rho);r.graupel_number_shadow[idx]=ngEntry+ngten*dt;}
 }
 r.qi[idx]=qi;r.ni[idx]=ni;r.qs[idx]=qs;r.qg[idx]=qg;r.qr[idx]=qr;r.nr[idx]=nr;r.qc[idx]=qc;r.qv[idx]=postQv;r.temperature[idx]=postT;
}
