#include <metal_stdlib>
#include "soft_binary32.metalh"
#include "soft_binary64.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"
#include "soft_gfk_pow.metalh"
#include "noahmp_radiation_family_bindings.metalh"
using namespace metal;

// Production batch leaf. Transcendentals and RN sqrt use shared integer-bit
// interfaces; their source operation graphs are kept in the shared helpers.

constant uint C_F32[24] = {
    0x00000000u, 0x3f800000u, 0x3f000000u, 0x40000000u,
    0x4388947bu, 0x3f0ccccdu, 0x3c23d70au, 0x45610000u,
    0x3f570a3du, 0x3de147aeu, 0x3ecccccdU, 0x3d75c28fu,
    0x3e19999au, 0x3fd9999au, 0x3a83126fu, 0xbecccccdU,
    0x3f19999au, 0x3f220c4au, 0x3ea8f5c3u, 0x3f608312u,
    0x358637bdu, 0xbc23d70au, 0x4b000000u, 0x80000000u
};

#define K_ZERO as_type<float>(C_F32[0])
#define K_ONE as_type<float>(C_F32[1])
#define K_HALF as_type<float>(C_F32[2])
#define K_TWO as_type<float>(C_F32[3])
#define K_TFRZ as_type<float>(C_F32[4])
#define K_P55 as_type<float>(C_F32[5])
#define K_P01 as_type<float>(C_F32[6])
#define K_T3600 as_type<float>(C_F32[7])
#define K_P84 as_type<float>(C_F32[8])
#define K_P11 as_type<float>(C_F32[9])
#define K_P40 as_type<float>(C_F32[10])
#define K_P06 as_type<float>(C_F32[11])
#define K_P15 as_type<float>(C_F32[12])
#define K_P17 as_type<float>(C_F32[13])
#define K_P001 as_type<float>(C_F32[14])
#define K_NP4 as_type<float>(C_F32[15])
#define K_P6 as_type<float>(C_F32[16])
#define K_P633 as_type<float>(C_F32[17])
#define K_P330 as_type<float>(C_F32[18])
#define K_P877 as_type<float>(C_F32[19])
#define K_MPE as_type<float>(C_F32[20])
#define K_NP01 as_type<float>(C_F32[21])

#define FADD(a,b) as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b)))
#define FSUB(a,b) as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b)))
#define FMUL(a,b) as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b)))
#define FDIV(a,b) as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b)))
#define FSQRT(a) as_type<float>(soft32_sqrt(as_type<uint>(a)))
#define FNEG(a) as_type<float>(as_type<uint>(a) ^ 0x80000000u)
#define FABS(a) as_type<float>(soft32_abs_bits(as_type<uint>(a)))

static inline float f_max(float a, float b) { return soft32_greater(as_type<uint>(b), as_type<uint>(a)) ? b : a; }
static inline float f_min(float a, float b) { return soft32_less(as_type<uint>(b), as_type<uint>(a)) ? b : a; }
static inline float f_sign(float a, float b) {
    return as_type<float>((as_type<uint>(a) & 0x7fffffffu) | (as_type<uint>(b) & 0x80000000u));
}
static inline float glibc_expf(float x) { return as_type<float>(soft_gfk_exp(as_type<uint>(x))); }
static inline float glibc_logf(float x) { return as_type<float>(soft_gfk_log(as_type<uint>(x))); }
static inline float glibc_powf(float x, float y) {
    uint ix=as_type<uint>(x), iy=as_type<uint>(y);
    if ((ix-0x00800000u)>=0x7f000000u || ((2u*iy-1u)>=0xfeffffffu))
        return as_type<float>(0x7fc00000u);
    return as_type<float>(soft_gfk_pow_bits(ix,iy));
}

static inline void d_snow_age(thread const float *p, thread float *o) {
    float tau0=p[0], gg=p[1], eg=p[2], ds=p[3], swemx=p[4];
    float dt=p[5], tg=p[6], sneqvo=p[7], sneqv=p[8], tauss=p[9];
    if (sneqv <= K_ZERO) tauss=K_ZERO;
    else {
        float dela0=FDIV(dt,tau0);
        float arg=FMUL(gg,FSUB(FDIV(K_ONE,K_TFRZ),FDIV(K_ONE,tg)));
        float age1=glibc_expf(arg);
        float age2=glibc_expf(f_min(K_ZERO,FMUL(eg,arg)));
        float tage=FADD(FADD(age1,age2),ds);
        float dela=FMUL(dela0,tage);
        float dels=FDIV(f_max(K_ZERO,FSUB(sneqv,sneqvo)),swemx);
        float sge=FMUL(FADD(tauss,dela),FSUB(K_ONE,dels));
        tauss=f_max(K_ZERO,sge);
    }
    o[0]=tauss; o[1]=FDIV(tauss,FADD(tauss,K_ONE));
}

static inline void d_snowalb_class(thread const float *p, thread float *o) {
    float swemx=p[0], qsnow=p[1], dt=p[2], albold=p[3];
    float alb=FADD(K_P55,FMUL(FSUB(albold,K_P55),glibc_expf(FDIV(FMUL(K_NP01,dt),K_T3600))));
    if (qsnow > K_ZERO) {
        float cap=FDIV(swemx,dt);
        alb=FADD(alb,FDIV(FMUL(f_min(qsnow,cap),FSUB(K_P84,alb)),cap));
    }
    o[0]=alb; o[1]=alb; o[2]=alb; o[3]=alb; o[4]=alb;
}

static inline void d_groundalb(thread const float *p, int ist, thread float *o) {
    float albsat[2]={p[0],p[1]}, albdry[2]={p[2],p[3]}, alblak[2]={p[4],p[5]};
    float fsno=p[6], smc1=p[7], albsnd[2]={p[8],p[9]}, albsni[2]={p[10],p[11]};
    float cosz=p[12], tg=p[13];
    for (int ib=0; ib<2; ++ib) {
        float inc=f_max(FSUB(K_P11,FMUL(K_P40,smc1)),K_ZERO), albsod, albsoi;
        if (ist==1) { albsod=f_min(FADD(albsat[ib],inc),albdry[ib]); albsoi=albsod; }
        else if (tg>K_TFRZ) { albsod=FDIV(K_P06,FADD(glibc_powf(f_max(K_P01,cosz),K_P17),K_P15)); albsoi=K_P06; }
        else { albsod=alblak[ib]; albsoi=albsod; }
        o[ib]=FADD(FMUL(albsod,FSUB(K_ONE,fsno)),FMUL(albsnd[ib],fsno));
        o[2+ib]=FADD(FMUL(albsoi,FSUB(K_ONE,fsno)),FMUL(albsni[ib],fsno));
    }
}

static inline void d_surrad(thread const float *p, thread float *o) {
    float mpe=p[0],fsun=p[1],fsha=p[2],elai=p[3],vai=p[4],laisun=p[5],laisha=p[6];
    thread const float *solad=p+7,*solai=p+9,*fabd=p+11,*fabi=p+13,*ftdd=p+15,*ftid=p+17,*ftii=p+19;
    thread const float *albgrd=p+21,*albgri=p+23,*albd=p+25,*albi=p+27,*frevd=p+29,*frevi=p+31,*fregd=p+33,*fregi=p+35;
    float sag=K_ZERO,sav=K_ZERO,fsa=K_ZERO,cad[2],cai[2];
    for(int ib=0;ib<2;++ib){
        cad[ib]=FMUL(solad[ib],fabd[ib]); cai[ib]=FMUL(solai[ib],fabi[ib]);
        sav=FADD(FADD(sav,cad[ib]),cai[ib]); fsa=FADD(FADD(fsa,cad[ib]),cai[ib]);
        float trd=FMUL(solad[ib],ftdd[ib]);
        float tri=FADD(FMUL(solad[ib],ftid[ib]),FMUL(solai[ib],ftii[ib]));
        float ab=FADD(FMUL(trd,FSUB(K_ONE,albgrd[ib])),FMUL(tri,FSUB(K_ONE,albgri[ib])));
        sag=FADD(sag,ab); fsa=FADD(fsa,ab);
    }
    float laifra=FDIV(elai,f_max(vai,mpe)),parsun,parsha;
    if(fsun>K_ZERO){
        parsun=FDIV(FMUL(FADD(cad[0],FMUL(fsun,cai[0])),laifra),f_max(laisun,mpe));
        parsha=FDIV(FMUL(FMUL(fsha,cai[0]),laifra),f_max(laisha,mpe));
    }else{parsun=K_ZERO;parsha=FDIV(FMUL(FADD(cad[0],cai[0]),laifra),f_max(laisha,mpe));}
    float rvis=FADD(FMUL(albd[0],solad[0]),FMUL(albi[0],solai[0]));
    float rnir=FADD(FMUL(albd[1],solad[1]),FMUL(albi[1],solai[1]));
    o[0]=parsun;o[1]=parsha;o[2]=sav;o[3]=sag;o[4]=fsa;o[5]=FADD(rvis,rnir);
    o[6]=FADD(FADD(FADD(FMUL(frevd[0],solad[0]),FMUL(frevi[0],solai[0])),FMUL(frevd[1],solad[1])),FMUL(frevi[1],solai[1]));
    o[7]=FADD(FADD(FADD(FMUL(fregd[0],solad[0]),FMUL(fregi[0],solai[0])),FMUL(fregd[1],solad[1])),FMUL(fregi[1],solai[1]));
}

static inline void d_twostream(thread const float *p,int ib1,int ic,thread float *o) {
    int ib=ib1-1;
    float xl=p[0],omegas[2]={p[1],p[2]},betads=p[3],betais=p[4];
    float cosz=p[5],vai=p[6],fwet=p[7],t=p[8],fveg=p[9];
    float albgrd[2]={p[10],p[11]},albgri[2]={p[12],p[13]},rho[2]={p[14],p[15]},tau[2]={p[16],p[17]};
    float fab[2]={p[18],p[19]},fre[2]={p[20],p[21]},ftd[2]={p[22],p[23]},fti[2]={p[24],p[25]};
    float gdir=p[26],frev[2]={p[27],p[28]},freg[2]={p[29],p[30]},bgap=p[31],wgap=p[32];
    float gap,kopen;
    if(vai==K_ZERO){gap=K_ONE;kopen=K_ONE;}else{gap=FSUB(K_ONE,fveg);kopen=FSUB(K_ONE,fveg);}
    float coszi=f_max(K_P001,cosz),chil=f_min(f_max(xl,K_NP4),K_P6);
    if(FABS(chil)<=K_P01)chil=K_P01;
    float phi1=FSUB(FSUB(K_HALF,FMUL(K_P633,chil)),FMUL(FMUL(K_P330,chil),chil));
    float phi2=FMUL(K_P877,FSUB(K_ONE,FMUL(K_TWO,phi1)));
    gdir=FADD(phi1,FMUL(phi2,coszi)); float ext=FDIV(gdir,coszi);
    float avmu_log_arg=FDIV(FADD(phi1,phi2),phi1);
    float avmu_log=glibc_logf(avmu_log_arg);
    float avmu=FDIV(FSUB(K_ONE,FMUL(FDIV(phi1,phi2),avmu_log)),phi2);
    float omegal=FADD(rho[ib],tau[ib]),tmp0=FADD(gdir,FMUL(phi2,coszi)),tmp1=FMUL(phi1,coszi);
    float asu_log_arg=FDIV(FADD(tmp1,tmp0),tmp1);
    float asu_log=glibc_logf(asu_log_arg);
    float asu_tail=FSUB(K_ONE,FMUL(FDIV(tmp1,tmp0),asu_log));
    float asu=FMUL(FDIV(FMUL(FMUL(K_HALF,omegal),gdir),tmp0),asu_tail);
    float betadl=FMUL(FDIV(FADD(K_ONE,FMUL(avmu,ext)),FMUL(FMUL(omegal,avmu),ext)),asu);
    float qh=FDIV(FADD(K_ONE,chil),K_TWO),q=FMUL(qh,qh);
    float betail=FDIV(FMUL(K_HALF,FADD(FADD(rho[ib],tau[ib]),FMUL(FSUB(rho[ib],tau[ib]),q))),omegal);
    float tmp2;
    if(t>K_TFRZ){tmp0=omegal;tmp1=betadl;tmp2=betail;}
    else{
        tmp0=FADD(FMUL(FSUB(K_ONE,fwet),omegal),FMUL(fwet,omegas[ib]));
        tmp1=FDIV(FADD(FMUL(FMUL(FSUB(K_ONE,fwet),omegal),betadl),FMUL(FMUL(fwet,omegas[ib]),betads)),tmp0);
        tmp2=FDIV(FADD(FMUL(FMUL(FSUB(K_ONE,fwet),omegal),betail),FMUL(FMUL(fwet,omegas[ib]),betais)),tmp0);
    }
    float omega=tmp0,betad=tmp1,betai=tmp2;
    float b=FADD(FSUB(K_ONE,omega),FMUL(omega,betai)),c=FMUL(omega,betai);
    tmp0=FMUL(avmu,ext); float d=FMUL(FMUL(tmp0,omega),betad),f=FMUL(FMUL(tmp0,omega),FSUB(K_ONE,betad));
    tmp1=FSUB(FMUL(b,b),FMUL(c,c)); float h=FDIV(FSQRT(tmp1),avmu),sigma=FSUB(FMUL(tmp0,tmp0),tmp1);
    if(FABS(sigma)<K_MPE)sigma=f_sign(K_MPE,sigma);
    float p1=FADD(b,FMUL(avmu,h)),p2=FSUB(b,FMUL(avmu,h)),p3=FADD(b,tmp0),p4=FSUB(b,tmp0);
    float s1=glibc_expf(FMUL(FNEG(h),vai)),s2=glibc_expf(FMUL(FNEG(ext),vai));
    float u1,u2,u3;
    if(ic==0){u1=FSUB(b,FDIV(c,albgrd[ib]));u2=FSUB(b,FMUL(c,albgrd[ib]));u3=FADD(f,FMUL(c,albgrd[ib]));}
    else{u1=FSUB(b,FDIV(c,albgri[ib]));u2=FSUB(b,FMUL(c,albgri[ib]));u3=FADD(f,FMUL(c,albgri[ib]));}
    tmp2=FSUB(u1,FMUL(avmu,h));float tmp3=FADD(u1,FMUL(avmu,h));
    float d1=FSUB(FDIV(FMUL(p1,tmp2),s1),FMUL(FMUL(p2,tmp3),s1));
    float tmp4=FADD(u2,FMUL(avmu,h)),tmp5=FSUB(u2,FMUL(avmu,h));
    float d2=FSUB(FDIV(tmp4,s1),FMUL(tmp5,s1));
    float h1=FSUB(FMUL(FNEG(d),p4),FMUL(c,f));
    float tmp6=FSUB(d,FDIV(FMUL(h1,p3),sigma));
    float tmp7=FMUL(FSUB(FSUB(d,c),FMUL(FDIV(h1,sigma),FADD(u1,tmp0))),s2);
    float h2=FDIV(FSUB(FDIV(FMUL(tmp6,tmp2),s1),FMUL(p2,tmp7)),d1);
    float h3=FNEG(FDIV(FSUB(FMUL(FMUL(tmp6,tmp3),s1),FMUL(p1,tmp7)),d1));
    float h4=FSUB(FMUL(FNEG(f),p3),FMUL(c,d)),tmp8=FDIV(h4,sigma);
    float tmp9=FMUL(FSUB(u3,FMUL(tmp8,FSUB(u2,tmp0))),s2);
    float h5=FNEG(FDIV(FADD(FDIV(FMUL(tmp8,tmp4),s1),tmp9),d2));
    float h6=FDIV(FADD(FMUL(FMUL(tmp8,tmp5),s1),tmp9),d2);
    float h7=FDIV(FMUL(c,tmp2),FMUL(d1,s1)),h8=FDIV(FMUL(FMUL(FNEG(c),tmp3),s1),d1);
    float h9=FDIV(tmp4,FMUL(d2,s1)),h10=FDIV(FMUL(FNEG(tmp5),s1),d2);
    float ftds,ftis;
    if(ic==0){ftds=FADD(FMUL(s2,FSUB(K_ONE,gap)),gap);ftis=FMUL(FADD(FADD(FDIV(FMUL(h4,s2),sigma),FMUL(h5,s1)),FDIV(h6,s1)),FSUB(K_ONE,gap));}
    else{ftds=K_ZERO;ftis=FADD(FMUL(FADD(FMUL(h9,s1),FDIV(h10,s1)),FSUB(K_ONE,kopen)),kopen);}
    ftd[ib]=ftds;fti[ib]=ftis;
    float fres,freveg,frebar;
    if(ic==0){fres=FADD(FMUL(FADD(FADD(FDIV(h1,sigma),h2),h3),FSUB(K_ONE,gap)),FMUL(albgrd[ib],gap));freveg=FMUL(FADD(FADD(FDIV(h1,sigma),h2),h3),FSUB(K_ONE,gap));frebar=FMUL(albgrd[ib],gap);}
    else{fres=FADD(FMUL(FADD(h7,h8),FSUB(K_ONE,kopen)),FMUL(albgri[ib],kopen));freveg=FADD(FMUL(FADD(h7,h8),FSUB(K_ONE,kopen)),FMUL(albgri[ib],kopen));frebar=K_ZERO;}
    fre[ib]=fres;frev[ib]=freveg;freg[ib]=frebar;
    fab[ib]=FSUB(FSUB(FSUB(K_ONE,fre[ib]),FMUL(FSUB(K_ONE,albgrd[ib]),ftd[ib])),FMUL(FSUB(K_ONE,albgri[ib]),fti[ib]));
    o[0]=fab[0];o[1]=fab[1];o[2]=fre[0];o[3]=fre[1];o[4]=ftd[0];o[5]=ftd[1];o[6]=fti[0];o[7]=fti[1];o[8]=gdir;
    o[9]=frev[0];o[10]=frev[1];o[11]=freg[0];o[12]=freg[1];o[13]=bgap;o[14]=wgap;
}

static inline void d_albedo(thread const float *p,int ist,thread float *o) {
    float swemx=p[4],dt=p[24],cosz=p[25],elai=p[26],esai=p[27],tg=p[28],tv=p[29];
    float fsno=p[31],fwet=p[32],sneqvo=p[33],sneqv=p[34],qsnow=p[35],fveg=p[36];
    float albold=p[41],tauss=p[42],fage=p[43],mpe=K_MPE;
    float bgap=K_ZERO,wgap=K_ZERO,albd[2]={K_ZERO,K_ZERO},albi[2]={K_ZERO,K_ZERO};
    float albgrd[2]={K_ZERO,K_ZERO},albgri[2]={K_ZERO,K_ZERO},albsnd[2]={K_ZERO,K_ZERO},albsni[2]={K_ZERO,K_ZERO};
    float fabd[2]={K_ZERO,K_ZERO},fabi[2]={K_ZERO,K_ZERO},ftdd[2]={K_ZERO,K_ZERO},ftid[2]={K_ZERO,K_ZERO},ftii[2]={K_ZERO,K_ZERO};
    float fsun=K_ZERO,frevd[2]={p[44],p[45]},frevi[2]={p[46],p[47]},fregd[2]={p[48],p[49]},fregi[2]={p[50],p[51]};
    float ftdi[2]={K_ZERO,K_ZERO},gdir=K_ZERO;
    if(cosz>K_ZERO){
        float rho[2],tau[2],vai=K_ZERO;
        for(int ib=0;ib<2;++ib){vai=FADD(elai,esai);float wl=FDIV(elai,f_max(vai,mpe)),ws=FDIV(esai,f_max(vai,mpe));
            rho[ib]=f_max(FADD(FMUL(p[11+ib],wl),FMUL(p[13+ib],ws)),mpe);tau[ib]=f_max(FADD(FMUL(p[15+ib],wl),FMUL(p[17+ib],ws)),mpe);}
        float sai[10]={p[0],p[1],p[2],p[3],swemx,dt,tg,sneqvo,sneqv,tauss},sao[2];d_snow_age(sai,sao);tauss=sao[0];fage=sao[1];
        float sci[4]={swemx,qsnow,dt,albold},sco[5];d_snowalb_class(sci,sco);albsnd[0]=sco[1];albsnd[1]=sco[2];albsni[0]=sco[3];albsni[1]=sco[4];albold=sco[0];
        float gai[14]={p[5],p[6],p[7],p[8],p[9],p[10],fsno,p[37],albsnd[0],albsnd[1],albsni[0],albsni[1],cosz,tg},gao[4];d_groundalb(gai,ist,gao);
        albgrd[0]=gao[0];albgrd[1]=gao[1];albgri[0]=gao[2];albgri[1]=gao[3];
        for(int ib=1;ib<=2;++ib){
            float tsi[33]={p[19],p[20],p[21],p[22],p[23],cosz,vai,fwet,tv,fveg,albgrd[0],albgrd[1],albgri[0],albgri[1],rho[0],rho[1],tau[0],tau[1],fabd[0],fabd[1],albd[0],albd[1],ftdd[0],ftdd[1],ftid[0],ftid[1],gdir,frevd[0],frevd[1],fregd[0],fregd[1],bgap,wgap};
            float tso[15];d_twostream(tsi,ib,0,tso);fabd[0]=tso[0];fabd[1]=tso[1];albd[0]=tso[2];albd[1]=tso[3];ftdd[0]=tso[4];ftdd[1]=tso[5];ftid[0]=tso[6];ftid[1]=tso[7];gdir=tso[8];frevd[0]=tso[9];frevd[1]=tso[10];fregd[0]=tso[11];fregd[1]=tso[12];bgap=tso[13];wgap=tso[14];
            float tii[33]={p[19],p[20],p[21],p[22],p[23],cosz,vai,fwet,tv,fveg,albgrd[0],albgrd[1],albgri[0],albgri[1],rho[0],rho[1],tau[0],tau[1],fabi[0],fabi[1],albi[0],albi[1],ftdi[0],ftdi[1],ftii[0],ftii[1],gdir,frevi[0],frevi[1],fregi[0],fregi[1],bgap,wgap};
            d_twostream(tii,ib,1,tso);fabi[0]=tso[0];fabi[1]=tso[1];albi[0]=tso[2];albi[1]=tso[3];ftdi[0]=tso[4];ftdi[1]=tso[5];ftii[0]=tso[6];ftii[1]=tso[7];gdir=tso[8];frevi[0]=tso[9];frevi[1]=tso[10];fregi[0]=tso[11];fregi[1]=tso[12];bgap=tso[13];wgap=tso[14];
        }
        float ext=FMUL(FDIV(gdir,cosz),FSQRT(FSUB(FSUB(K_ONE,rho[0]),tau[0])));
        fsun=FDIV(FSUB(K_ONE,glibc_expf(FMUL(FNEG(ext),vai))),f_max(FMUL(ext,vai),mpe));ext=fsun;fsun=ext<K_P01?K_ZERO:ext;
    }
    o[0]=fage;o[1]=albold;o[2]=tauss;o[3]=fsun;o[4]=bgap;o[5]=wgap;
    thread float *v[15]={albgrd,albgri,albd,albi,fabd,fabi,ftdd,ftid,ftii,frevd,frevi,fregd,fregi,albsnd,albsni};
    for(int k=0;k<15;++k){o[6+2*k]=v[k][0];o[7+2*k]=v[k][1];}
}

static inline void d_radiation(thread const float *p,int ist,thread float *o) {
    float alb[36];d_albedo(p,ist,alb);float elai=p[26],esai=p[27];thread const float *solad=p+52,*solai=p+54;
    float fsun=alb[3],fsha=FSUB(K_ONE,fsun),laisun=FMUL(elai,fsun),laisha=FMUL(elai,fsha),vai=FADD(elai,esai),s[37];
    s[0]=K_MPE;s[1]=fsun;s[2]=fsha;s[3]=elai;s[4]=vai;s[5]=laisun;s[6]=laisha;s[7]=solad[0];s[8]=solad[1];s[9]=solai[0];s[10]=solai[1];
    const int order[13]={4,5,6,7,8,0,1,2,3,9,10,11,12};
    for(int k=0;k<13;++k){s[11+2*k]=alb[6+2*order[k]];s[12+2*k]=alb[7+2*order[k]];}
    float sr[8];d_surrad(s,sr);o[0]=fsun;o[1]=laisun;o[2]=laisha;o[3]=sr[0];o[4]=sr[1];o[5]=sr[2];o[6]=sr[3];o[7]=sr[4];o[8]=sr[5];o[9]=sr[6];o[10]=sr[7];
    o[11]=alb[32];o[12]=alb[33];o[13]=alb[34];o[14]=alb[35];o[15]=alb[4];o[16]=alb[5];o[17]=alb[1];o[18]=alb[2];
}

kernel void noahmp_radiation_batch(
    constant NoahmpRadiationBatchResources &r [[buffer(0)]],
    constant NoahmpRadiationBatchParams &q [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(q.n)) return;
    thread float p[56], o[19];
    for (uint k = 0; k < 56; ++k) p[k] = r.in[tid * 56 + k];
    d_radiation(p, r.ist[tid], o);
    for (uint k = 0; k < 19; ++k) r.out[tid * 19 + k] = o[k];
}

#undef FADD
#undef FSUB
#undef FMUL
#undef FDIV
#undef FSQRT
#undef FNEG
#undef FABS
