#include <metal_stdlib>
#include "noahmp_vegeflux_family_bindings.metalh"

using namespace metal;

constant uint VF_ESAT[28] = {
  0x40C37319u,0x3EE32656u,0x3C6A1E55u,0x398AF867u,0x364B6C50u,0x32AEB9EAu,0x2E86F33Bu,
  0x40C37E63u,0x3F00E367u,0x3C9A8091u,0x39DAF453u,0x36C371F7u,0x334FD334u,0x2F4A2E60u,
  0x3EE33B10u,0x3CEA0BB0u,0x3A501761u,0x374BE117u,0x33DE9991u,0x2FC2326Bu,0xAB479299u,
  0x3F00C69Cu,0x3D1A8D72u,0x3AA632DDu,0x37CFD543u,0x34A15DEFu,0x3114557Bu,0x2CFAE738u
};
constant float VF_GRAV=9.80616f,VF_SB=5.67e-8f,VF_VKC=0.4f,VF_TFRZ=273.16f;
constant float VF_CPAIR=1004.64f,VF_CWAT=4.188e6f,VF_CICE=2.094e6f;
constant float VF_DENH2O=1000.0f,VF_DENICE=917.0f;
static inline float vf_min(float a,float b){return a<b?a:b;}
static inline float vf_max(float a,float b){return a>b?a:b;}
static inline float vf_p3(float x){return x*(x*x);}
static inline float vf_p4(float x){float t=x*x;return t*t;}
static inline float vf_esat(int base,float t){
  float y=as_type<float>(VF_ESAT[base+5])+t*as_type<float>(VF_ESAT[base+6]);
  y=as_type<float>(VF_ESAT[base+4])+t*y;y=as_type<float>(VF_ESAT[base+3])+t*y;
  y=as_type<float>(VF_ESAT[base+2])+t*y;y=as_type<float>(VF_ESAT[base+1])+t*y;
  return 100.0f*(as_type<float>(VF_ESAT[base])+t*y);
}
static inline float vf_tdc(float t){return vf_min(50.0f,vf_max(-50.0f,t-VF_TFRZ));}
static inline void vf_esat_all(float t,thread float &w,thread float &i,thread float &dw,thread float &di){
  w=vf_esat(0,t);i=vf_esat(7,t);dw=vf_esat(14,t);di=vf_esat(21,t);
}
static inline void vf_ragrb(int iter,float vai,float rho,float hg,float tah,float zpd,float z0mg,
 float z0hg,float hcan,float uc,float z0h,float fv,float cwp,float mpe,float dleaf,
 thread float &mozg,thread float &fhg,thread float &ramg,thread float &rahg,
 thread float &rawg,thread float &rb){
  float molg=0.0f;
  if(iter>1){float x=(((VF_VKC*(VF_GRAV/tah))*hg)/(rho*VF_CPAIR));if(fabs(x)<=mpe)x=mpe;
    molg=(-vf_p3(fv))/x;mozg=vf_min((zpd-z0mg)/molg,1.0f);}
  float fhgnew=mozg<0.0f?pow(1.0f-15.0f*mozg,-0.25f):1.0f+4.7f*mozg;
  fhg=iter==1?fhgnew:0.5f*(fhg+fhgnew);
  float cwpc=pow(((cwp*vai)*hcan)*fhg,0.5f);
  float t1=exp(-(cwpc*z0hg)/hcan),t2=exp(-(cwpc*(z0h+zpd))/hcan);
  float tmprah2=(hcan*exp(cwpc)/cwpc)*(t1-t2);
  float kh=vf_max((VF_VKC*fv)*(hcan-zpd),mpe);ramg=0.0f;rahg=tmprah2/kh;rawg=rahg;
  float tmprb=(cwpc*50.0f)/(1.0f-exp(-cwpc/2.0f));rb=vf_min(vf_max(tmprb*sqrt(dleaf/uc),5.0f),50.0f);
}
struct VFSfc{float moz,fm,fh,fm2,fh2,fv,cm,ch,ch2;int mozsgn;};
static inline void vf_sfcdif(int it,float ts,float rho,float h,float q,float zl,float zpd,float z0m,float z0h,float ur,float mpe,thread VFSfc &s){
  float old=s.moz,tc=log((zl-zpd)/z0m),th=log((zl-zpd)/z0h),tc2=log((2.0f+z0m)/z0m),th2=log((2.0f+z0h)/z0h),moz2=0.0f;
  if(it==1){s.fv=0.0f;s.moz=0.0f;}else{float tv=(1.0f+0.61f*q)*ts,x=(((VF_VKC*(VF_GRAV/tv))*h)/(rho*VF_CPAIR));if(fabs(x)<=mpe)x=mpe;
    float mol=(-vf_p3(s.fv))/x;s.moz=vf_min((zl-zpd)/mol,1.0f);moz2=vf_min((2.0f+z0h)/mol,1.0f);}
  if(old*s.moz<0.0f)s.mozsgn++;if(s.mozsgn>=2){s.moz=0.0f;s.fm=s.fh=moz2=s.fm2=s.fh2=0.0f;}
  float fm,fh,fm2,fh2;
  if(s.moz<0.0f){float x=pow(1.0f-16.0f*s.moz,0.25f),a=log((1.0f+x*x)/2.0f),b=log((1.0f+x)/2.0f);
    fm=2.0f*b+a-2.0f*atan(x)+1.5707963f;fh=2.0f*a;
    x=pow(1.0f-16.0f*moz2,0.25f);a=log((1.0f+x*x)/2.0f);b=log((1.0f+x)/2.0f);fm2=2.0f*b+a-2.0f*atan(x)+1.5707963f;fh2=2.0f*a;
  }else{fm=fh=-5.0f*s.moz;fm2=fh2=-5.0f*moz2;}
  if(it==1){s.fm=fm;s.fh=fh;s.fm2=fm2;s.fh2=fh2;}else{s.fm=0.5f*(s.fm+fm);s.fh=0.5f*(s.fh+fh);s.fm2=0.5f*(s.fm2+fm2);s.fh2=0.5f*(s.fh2+fh2);}
  s.fh=vf_min(s.fh,0.9f*th);s.fm=vf_min(s.fm,0.9f*tc);s.fh2=vf_min(s.fh2,0.9f*th2);s.fm2=vf_min(s.fm2,0.9f*tc2);
  float cmfm=tc-s.fm,chfh=th-s.fh,ch2fh2=th2-s.fh2;if(fabs(cmfm)<=mpe)cmfm=mpe;if(fabs(chfh)<=mpe)chfh=mpe;if(fabs(ch2fh2)<=mpe)ch2fh2=mpe;
  s.cm=(VF_VKC*VF_VKC)/(cmfm*cmfm);s.ch=(VF_VKC*VF_VKC)/(cmfm*chfh);s.fv=ur*sqrt(s.cm);s.ch2=(VF_VKC*s.fv)/ch2fh2;
}
static inline float vf_sf1(float a,float b){return pow(a,(b-25.0f)/10.0f);}
static inline float vf_sf2(float a){float t=a+273.16f;return 1.0f+exp((-2.2e5f+710.0f*t)/(8.314f*t));}
static inline void vf_stomata(device const float *p,float mpe,float apar,float foln,float tv,float ei,float ea,float sfctmp,float sfcprs,float fveg,float o2,float co2,float igs,float btran,float rb,thread float &rs,thread float &psn){
  float c3=p[3],apar_scale=apar/vf_max(fveg,1.0e-6f),cf=(sfcprs/(8.314f*sfctmp))*1.0e6f;rs=(1.0f/p[10])*cf;psn=0.0f;if(apar_scale<=0.0f)return;
  float fnf=vf_min(foln/vf_max(mpe,p[13]),1.0f),tc=tv-VF_TFRZ,ppf=4.6f*apar_scale,j=ppf*p[12];
  float kc=p[4]*vf_sf1(p[5],tc),ko=p[6]*vf_sf1(p[7],tc),awc=kc*(1.0f+o2/ko),cp=((0.5f*kc*ko)/o2)*0.21f;
  float vcmx=((p[9]/vf_sf2(tc))*fnf*btran)*vf_sf1(p[8],tc),ci=(0.7f*co2*c3)+(0.4f*co2*(1.0f-c3));
  float rlb=rb/cf,cea=vf_max((0.25f*ei*c3)+(0.40f*ei*(1.0f-c3)),vf_min(ea,ei));
  for(int it=0;it<3;++it){float wj=(((vf_max(ci-cp,0.0f)*j)/(ci+2.0f*cp))*c3)+(j*(1.0f-c3));
    float wc=(((vf_max(ci-cp,0.0f)*vcmx)/(ci+awc))*c3)+(vcmx*(1.0f-c3));
    float we=(0.5f*vcmx*c3)+(((4000.0f*vcmx*ci)/sfcprs)*(1.0f-c3));psn=vf_min(vf_min(wj,wc),we)*igs;
    float cs=vf_max(co2-((1.37f*rlb*sfcprs*psn)),mpe);
    float a=((p[11]*psn*sfcprs*cea)/(cs*ei))+p[10];
    float b=(((p[11]*psn*sfcprs)/cs)+p[10])*rlb-1.0f,c=-rlb;
    float disc=sqrt(b*b-4.0f*a*c),qq=b>=0.0f?-0.5f*(b+disc):-0.5f*(b-disc),r1=qq/a,r2=c/qq;
    rs=vf_max(r1,r2);ci=vf_max(cs-(psn*sfcprs*1.65f*rs),0.0f);
  }rs*=cf;
}

kernel void noahmp_vegeflux_flux(
    constant NoahmpVegefluxFluxResources &r [[buffer(0)]],
    constant NoahmpVegefluxFluxParams &q [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
  if(idx>=uint(q.n))return;
  ulong ib=ulong(idx)*50ul,pb=ulong(idx)*14ul,lb=ulong(idx)*7ul,ob=ulong(idx)*30ul;
  int isnow=r.isnow[idx];device const float *v=r.in+ib,*p=r.pin+pb;
  device const float *dz=r.dzsnso+lb,*stc=r.stc+lb,*df=r.df+lb;
  float dt=v[0],sav=v[1],sag=v[2],lwdn=v[3],ur=v[4],uu=v[5],vv=v[6],sfctmp=v[7];
  float qair=v[8],eair=v[9],rho=v[10],snowh=v[11],vai=v[12],gammav=v[13],gammag=v[14],fwet=v[15];
  float laisun=v[16],laisha=v[17],cwp=v[18],zlvl=v[19],zpd=v[20],z0m=v[21],fveg=v[22],z0mg=v[23];
  float emv=v[24],emg=v[25],canliq=v[26],canice=v[27],rsurf=v[28],latheav=v[29],parsun=v[30],parsha=v[31];
  float igs=v[32],foln=v[33],co2=v[34],o2=v[35],btran=v[36],sfcprs=v[37],rhsur=v[38],pahv=v[39],pahg=v[40];
  float eah=v[41],tah=v[42],tv=v[43],tg=v[44],cm=v[45],ch=v[46],qsfc=v[47],psfc=v[48];
  float mpe=1.0e-6f;int liter=0;float fv=0.1f,dtv=0.0f,dtg=0.0f;
  VFSfc s={0.0f,0.0f,0.0f,0.0f,0.0f,0.1f,0.0f,0.0f,0.0f,0};
  float hg=0.0f,h=0.0f,mozg=0.0f,fhg=0.0f,rssun=0.0f,rssha=0.0f,psnsun=0.0f,psnsha=0.0f;
  float vaie=vf_min(6.0f,vai),laisune=vf_min(6.0f,laisun),laishae=vf_min(6.0f,laisha);
  float esatw,esati,dsatw,dsati,t=vf_tdc(tg);vf_esat_all(t,esatw,esati,dsatw,dsati);float estg=t>0.0f?esatw:esati;
  qsfc=(0.622f*eair)/(psfc-0.378f*eair);
  float hcan=p[1],uc=(ur*log(((hcan-zpd)-z0m)/z0m))/log(zlvl/z0m);
  float air=-(emv*(1.0f+(1.0f-emv)*(1.0f-emg))*lwdn)-(emv*emg*VF_SB*vf_p4(tg));
  float cir=(2.0f-emv*(1.0f-emg)-emv)*VF_SB;
  float cah=0.0f,cvh=0.0f,rahg=1.0f,rawg=1.0f,z0h=z0m,shc=0.0f,evc=0.0f,tr=0.0f,irc=0.0f,canhs=0.0f;
  for(int it=1;it<=20;++it){
    z0h=z0m;float z0hg=z0mg;vf_sfcdif(it,sfctmp,rho,h,qair,zlvl,zpd,z0m,z0h,ur,mpe,s);
    float rahc=vf_max(1.0f,1.0f/(s.ch*ur)),rawc=rahc,ramg,rb;
    vf_ragrb(it,vaie,rho,hg,tah,zpd,z0mg,z0hg,hcan,uc,z0h,s.fv,cwp,mpe,p[0],mozg,fhg,ramg,rahg,rawg,rb);
    t=vf_tdc(tv);vf_esat_all(t,esatw,esati,dsatw,dsati);float estv=t>0.0f?esatw:esati,destv=t>0.0f?dsatw:dsati;
    if(it==1){vf_stomata(p,mpe,parsun,foln,tv,estv,eah,sfctmp,sfcprs,fveg,o2,co2,igs,btran,rb,rssun,psnsun);vf_stomata(p,mpe,parsha,foln,tv,estv,eah,sfctmp,sfcprs,fveg,o2,co2,igs,btran,rb,rssha,psnsha);}
    cah=1.0f/rahc;cvh=(2.0f*vaie)/rb;float cgh=1.0f/rahg,cond=cah+cvh+cgh;
    float ata=(sfctmp*cah+tg*cgh)/cond,bta=cvh/cond,csh=((1.0f-bta)*rho*VF_CPAIR)*cvh;
    float caw=1.0f/rawc,cew=(fwet*vaie)/rb,ctw=(1.0f-fwet)*(laisune/(rb+rssun)+laishae/(rb+rssha));
    float cgw=1.0f/(rawg+rsurf);cond=caw+cew+ctw+cgw;
    float aea=(eair*caw+estg*cgw)/cond,bea=(cew+ctw)/cond;
    float cev=(((1.0f-bea)*cew*rho)*VF_CPAIR)/gammav,ctr=(((1.0f-bea)*ctw*rho)*VF_CPAIR)/gammav;
    tah=ata+bta*tv;eah=aea+bea*estv;
    irc=fveg*(air+cir*vf_p4(tv));shc=(((fveg*rho)*VF_CPAIR)*cvh)*(tv-tah);
    evc=((((fveg*rho)*VF_CPAIR)*cew)*(estv-eah))/gammav;tr=((((fveg*rho)*VF_CPAIR)*ctw)*(estv-eah))/gammav;
    if(tv>VF_TFRZ)evc=vf_min((canliq*latheav)/dt,evc);else evc=vf_min((canice*latheav)/dt,evc);
    float hcv=((p[2]*vaie)*VF_CWAT)+(canliq*VF_CWAT)/VF_DENH2O+(canice*VF_CICE)/VF_DENICE;
    float b=sav-irc-shc-evc-tr+pahv;
    float a=fveg*((4.0f*cir*vf_p3(tv)+csh+(cev+ctr)*destv)+hcv/dt);dtv=b/a;
    irc+=(((fveg*4.0f)*cir)*vf_p3(tv))*dtv;shc+=(fveg*csh)*dtv;evc+=((fveg*cev)*destv)*dtv;tr+=((fveg*ctr)*destv)*dtv;
    canhs=(dtv*fveg*hcv)/dt;tv+=dtv;h=((rho*VF_CPAIR)*(tah-sfctmp))/rahc;hg=((rho*VF_CPAIR)*(tg-tah))/rahg;
    qsfc=(0.622f*eah)/(sfcprs-0.378f*eah);
    if(liter==1)break;if(it>=5&&fabs(dtv)<=0.01f&&liter==0)liter=1;
  }
  air=-(emg*(1.0f-emv)*lwdn)-(emg*emv*VF_SB*vf_p4(tv));cir=emg*VF_SB;
  float csh=(rho*VF_CPAIR)/rahg,cev=(rho*VF_CPAIR)/(gammag*(rawg+rsurf));
  int kk=isnow+3;float cgh=(2.0f*df[kk])/dz[kk];
  float irg=0.0f,shg=0.0f,evg=0.0f,gh=0.0f,destg=0.0f;estg=0.0f;
  for(int it=0;it<5;++it){t=vf_tdc(tg);vf_esat_all(t,esatw,esati,dsatw,dsati);estg=t>0.0f?esatw:esati;destg=t>0.0f?dsatw:dsati;
    irg=cir*vf_p4(tg)+air;shg=csh*(tg-tah);evg=cev*(estg*rhsur-eah);gh=cgh*(tg-stc[kk]);
    float b=sag-irg-shg-evg-gh+pahg,a=4.0f*cir*vf_p3(tg)+csh+cev*destg+cgh;dtg=b/a;
    irg+=(4.0f*cir*vf_p3(tg))*dtg;shg+=csh*dtg;evg+=(cev*destg)*dtg;gh+=cgh*dtg;tg+=dtg;
  }
  if(snowh>0.05f&&tg>VF_TFRZ){tg=VF_TFRZ;irg=(cir*vf_p4(tg)-emg*(1.0f-emv)*lwdn)-(emg*emv*VF_SB*vf_p4(tv));shg=csh*(tg-tah);evg=cev*(estg*rhsur-eah);gh=sag+pahg-(irg+shg+evg);}
  float taux=-((rho*s.cm*ur)*uu),tauy=-((rho*s.cm*ur)*vv);
  float cah2=(s.fv*VF_VKC)/(log((2.0f+z0h)/z0h)-s.fh2),t2mv,q2v;
  if(cah2<1.0e-5f){t2mv=tah;q2v=qsfc;}else{t2mv=tah-(((shg+shc/fveg)/(rho*VF_CPAIR))/cah2);q2v=qsfc-((((evc+tr)/fveg+evg)/(latheav*rho))/cah2);}
  r.out[ob+0]=eah;r.out[ob+1]=tah;r.out[ob+2]=tv;r.out[ob+3]=tg;r.out[ob+4]=s.cm;r.out[ob+5]=cah;
  r.out[ob+6]=taux;r.out[ob+7]=tauy;r.out[ob+8]=irg;r.out[ob+9]=irc;r.out[ob+10]=shg;r.out[ob+11]=shc;
  r.out[ob+12]=evg;r.out[ob+13]=evc;r.out[ob+14]=tr;r.out[ob+15]=gh;r.out[ob+16]=t2mv;
  r.out[ob+17]=psnsun;r.out[ob+18]=psnsha;r.out[ob+19]=canhs;r.out[ob+20]=qsfc;r.out[ob+21]=q2v;
  r.out[ob+22]=cah2;r.out[ob+23]=cvh;r.out[ob+24]=1.0f/rahg;r.out[ob+25]=rssun;r.out[ob+26]=rssha;
  r.out[ob+27]=sav;r.out[ob+28]=sag;r.out[ob+29]=v[49];
}
