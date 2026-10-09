#include <metal_stdlib>
#include "noahmp_snow_family_bindings.metalh"

using namespace metal;

// Float-only leaves from noahmp_snow.cu. CUDA's COMPACT leaf uses binary64
glibc expf intermediates; it and the snowwater composition are intentionally
not exposed here.
#pragma clang fp contract(off)

constant uint NSNOW = 3;
constant uint NSOIL = 4;
constant uint NFULL = 7;
constant uint ST_SNOWH = 0, ST_SNEQV = 1, ST_SNICE = 2, ST_SNLIQ = 5;
constant uint ST_STC = 8, ST_ZSNSO = 15, ST_DZSNSO = 22;
constant uint ST_SH2O = 29, ST_SICE = 33;
constant float K_ZERO = 0.0f, K_ONE = 1.0f, K_HALF = 0.5f;
constant float K_TWO = 2.0f, K_TFRZ = 273.16f, K_HFUS = 0.3336e6f;
constant float K_CWAT = 4.188e6f, K_CICE = 2.094e6f;
constant float K_DENH2O = 1000.0f, K_DENICE = 917.0f;
constant float K_P025 = 0.025f, K_P05 = 0.05f, K_P1 = 0.1f;
constant float K_P20 = 0.20f, K_MAXLIQ = 0.4f, K_1EM8 = 1.0e-8f;
constant float K_1EM6 = 1.0e-6f, K_500 = 500.0f, K_50 = 50.0f;
constant float DZMIN[3] = {0.025f, 0.025f, 0.1f};

struct SnowCol {
    int isnow;
    float snowh, sneqv;
    float snice[3], snliq[3];
    float stc[7], zsnso[7], dzsnso[7];
    float sh2o[4], sice[4];
};

inline uint si(int j) { return uint(j + 2); }
inline uint fi(int j) { return uint(j + 2); }
inline float get_ice(thread SnowCol &c, int j) { return c.snice[si(j)]; }
inline float get_liq(thread SnowCol &c, int j) { return c.snliq[si(j)]; }
inline float get_dz(thread SnowCol &c, int j) { return c.dzsnso[fi(j)]; }
inline float get_stc(thread SnowCol &c, int j) { return c.stc[fi(j)]; }
inline void put_ice(thread SnowCol &c, int j, float x) { c.snice[si(j)] = x; }
inline void put_liq(thread SnowCol &c, int j, float x) { c.snliq[si(j)] = x; }
inline void put_dz(thread SnowCol &c, int j, float x) { c.dzsnso[fi(j)] = x; }
inline void put_stc(thread SnowCol &c, int j, float x) { c.stc[fi(j)] = x; }

inline void load_col(device const float *p, int isnow, thread SnowCol &c) {
    c.isnow=isnow; c.snowh=p[ST_SNOWH]; c.sneqv=p[ST_SNEQV];
    for (uint k=0;k<3;k++) { c.snice[k]=p[ST_SNICE+k]; c.snliq[k]=p[ST_SNLIQ+k]; }
    for (uint k=0;k<7;k++) { c.stc[k]=p[ST_STC+k]; c.zsnso[k]=p[ST_ZSNSO+k]; c.dzsnso[k]=p[ST_DZSNSO+k]; }
    for (uint k=0;k<4;k++) { c.sh2o[k]=p[ST_SH2O+k]; c.sice[k]=p[ST_SICE+k]; }
}
inline void store_col(thread SnowCol &c, device float *p, device int *ip) {
    ip[0]=c.isnow; p[ST_SNOWH]=c.snowh; p[ST_SNEQV]=c.sneqv;
    for (uint k=0;k<3;k++) { p[ST_SNICE+k]=c.snice[k]; p[ST_SNLIQ+k]=c.snliq[k]; }
    for (uint k=0;k<7;k++) { p[ST_STC+k]=c.stc[k]; p[ST_ZSNSO+k]=c.zsnso[k]; p[ST_DZSNSO+k]=c.dzsnso[k]; }
    for (uint k=0;k<4;k++) { p[ST_SH2O+k]=c.sh2o[k]; p[ST_SICE+k]=c.sice[k]; }
}
inline float fmax_first(float a,float b) { return a < b ? b : a; }
inline float fmin_first(float a,float b) { return b < a ? b : a; }

inline void combo(thread float &dz, thread float &wliq, thread float &wice,
                  thread float &t, float dz2,float wliq2,float wice2,float t2) {
    float dzc=dz+dz2, wicec=wice+wice2, wliqc=wliq+wliq2;
    float h=((K_CICE*wice+K_CWAT*wliq)*(t-K_TFRZ))+K_HFUS*wliq;
    float h2=((K_CICE*wice2+K_CWAT*wliq2)*(t2-K_TFRZ))+K_HFUS*wliq2;
    float hc=h+h2, tc;
    if(hc<K_ZERO) tc=K_TFRZ+hc/(K_CICE*wicec+K_CWAT*wliqc);
    else if(hc<=K_HFUS*wliqc) tc=K_TFRZ;
    else tc=K_TFRZ+(hc-K_HFUS*wliqc)/(K_CICE*wicec+K_CWAT*wliqc);
    dz=dzc; wice=wicec; wliq=wliqc; t=tc;
}

inline void snowfall(thread SnowCol &c,float dt,float qsnow,float snowhin,float sfctmp) {
    int newnode=0;
    if(c.isnow==0 && qsnow>K_ZERO) { c.snowh+=snowhin*dt; c.sneqv+=qsnow*dt; }
    if(c.isnow==0 && c.snowh>=K_P025) {
        c.isnow=-1; newnode=1; put_dz(c,0,c.snowh); c.snowh=K_ZERO;
        put_stc(c,0,fmin_first(K_TFRZ,sfctmp)); put_ice(c,0,c.sneqv); put_liq(c,0,K_ZERO);
    }
    if(c.isnow<0 && newnode==0 && qsnow>K_ZERO) {
        put_ice(c,c.isnow+1,get_ice(c,c.isnow+1)+qsnow*dt);
        put_dz(c,c.isnow+1,get_dz(c,c.isnow+1)+snowhin*dt);
    }
}

inline void combine(thread SnowCol &c,thread float &ponding1,thread float &ponding2) {
    int isnow_old=c.isnow;
    for(int j=isnow_old+1;j<=0;j++) if(get_ice(c,j)<=K_P1) {
        if(j!=0) {
            put_liq(c,j+1,get_liq(c,j+1)+get_liq(c,j)); put_ice(c,j+1,get_ice(c,j+1)+get_ice(c,j));
            put_dz(c,j+1,get_dz(c,j+1)+get_dz(c,j));
        } else if(c.isnow < -1) {
            put_liq(c,j-1,get_liq(c,j-1)+get_liq(c,j)); put_ice(c,j-1,get_ice(c,j-1)+get_ice(c,j));
            put_dz(c,j-1,get_dz(c,j-1)+get_dz(c,j));
        } else {
            if(get_ice(c,j)>=K_ZERO) { ponding1=get_liq(c,j); c.sneqv=get_ice(c,j); c.snowh=get_dz(c,j); }
            else { ponding1=get_liq(c,j)+get_ice(c,j);
                if(ponding1<K_ZERO) { c.sice[0]+=ponding1/(get_dz(c,1)*K_DENH2O); ponding1=K_ZERO; }
                c.sneqv=K_ZERO; c.snowh=K_ZERO; }
            put_liq(c,j,K_ZERO); put_ice(c,j,K_ZERO); put_dz(c,j,K_ZERO);
        }
        if(j>c.isnow+1 && c.isnow < -1) for(int i=j;i>=c.isnow+2;i--) {
            put_stc(c,i,get_stc(c,i-1)); put_liq(c,i,get_liq(c,i-1));
            put_ice(c,i,get_ice(c,i-1)); put_dz(c,i,get_dz(c,i-1));
        }
        c.isnow++;
    }
    if(c.sice[0]<K_ZERO) { c.sh2o[0]+=c.sice[0]; c.sice[0]=K_ZERO; }
    if(c.isnow==0) return;
    c.sneqv=K_ZERO; c.snowh=K_ZERO; float zwice=K_ZERO,zwliq=K_ZERO;
    for(int j=c.isnow+1;j<=0;j++) { c.sneqv=(c.sneqv+get_ice(c,j))+get_liq(c,j); c.snowh+=get_dz(c,j); zwice+=get_ice(c,j); zwliq+=get_liq(c,j); }
    if(c.snowh<K_P025 && c.isnow<0) { c.isnow=0; c.sneqv=zwice; ponding2=zwliq; if(c.sneqv<=K_ZERO)c.snowh=K_ZERO; }
    if(c.isnow < -1) {
        isnow_old=c.isnow; int mssi=1;
        for(int i=isnow_old+1;i<=0;i++) if(get_dz(c,i)<DZMIN[mssi-1]) {
            int neighbor;
            if(i==c.isnow+1) neighbor=i+1;
            else if(i==0) neighbor=i-1;
            else { neighbor=i+1; if(get_dz(c,i-1)+get_dz(c,i)<get_dz(c,i+1)+get_dz(c,i))neighbor=i-1; }
            int j,l; if(neighbor>i){j=neighbor;l=i;}else{j=i;l=neighbor;}
            float dz=get_dz(c,j),liq=get_liq(c,j),ice=get_ice(c,j),temp=get_stc(c,j);
            combo(dz,liq,ice,temp,get_dz(c,l),get_liq(c,l),get_ice(c,l),get_stc(c,l));
            put_dz(c,j,dz);put_liq(c,j,liq);put_ice(c,j,ice);put_stc(c,j,temp);
            if(j-1>c.isnow+1)for(int k=j-1;k>=c.isnow+2;k--){put_stc(c,k,get_stc(c,k-1));put_ice(c,k,get_ice(c,k-1));put_liq(c,k,get_liq(c,k-1));put_dz(c,k,get_dz(c,k-1));}
            c.isnow++; if(c.isnow>=-1)break;
        } else mssi++;
    }
}

inline void divide(thread SnowCol &c) {
    float dz[3],ice[3],liq[3],temp[3];
    for(uint k=0;k<3;k++){dz[k]=as_type<float>(0x7fc00000u);ice[k]=dz[k];liq[k]=dz[k];temp[k]=dz[k];}
    int isn=c.isnow;
    for(int j=1;j<=3;j++)if(j<=abs(isn)){dz[j-1]=get_dz(c,j+isn);ice[j-1]=get_ice(c,j+isn);liq[j-1]=get_liq(c,j+isn);temp[j-1]=get_stc(c,j+isn);}
    int msno=abs(isn); float drr,propor,zwice,zwliq,dtdz;
    if(msno==1 && dz[0]>K_P05){msno=2;dz[0]/=K_TWO;ice[0]/=K_TWO;liq[0]/=K_TWO;dz[1]=dz[0];ice[1]=ice[0];liq[1]=liq[0];temp[1]=temp[0];}
    if(msno>1 && dz[0]>K_P05){drr=dz[0]-K_P05;propor=drr/dz[0];zwice=propor*ice[0];zwliq=propor*liq[0];propor=K_P05/dz[0];ice[0]*=propor;liq[0]*=propor;dz[0]=K_P05;
        combo(dz[1],liq[1],ice[1],temp[1],drr,zwliq,zwice,temp[0]);
        if(msno<=2 && dz[1]>K_P20){msno=3;dtdz=(temp[0]-temp[1])/((dz[0]+dz[1])/K_TWO);dz[1]/=K_TWO;ice[1]/=K_TWO;liq[1]/=K_TWO;dz[2]=dz[1];ice[2]=ice[1];liq[2]=liq[1];temp[2]=temp[1]-dtdz*dz[1]/K_TWO;if(temp[2]>=K_TFRZ)temp[2]=temp[1];else temp[1]+=dtdz*dz[1]/K_TWO;}}
    if(msno>2 && dz[1]>K_P20){drr=dz[1]-K_P20;propor=drr/dz[1];zwice=propor*ice[1];zwliq=propor*liq[1];propor=K_P20/dz[1];ice[1]*=propor;liq[1]*=propor;dz[1]=K_P20;combo(dz[2],liq[2],ice[2],temp[2],drr,zwliq,zwice,temp[1]);}
    c.isnow=-msno;
    for(int j=c.isnow+1;j<=0;j++){uint k=uint(j-c.isnow-1);put_dz(c,j,dz[k]);put_ice(c,j,ice[k]);put_liq(c,j,liq[k]);put_stc(c,j,temp[k]);}
}

inline float snowh2o(thread SnowCol &c,float dt,float qsnfro,float qsnsub,float qrain,float ssi,float snow_ret_fac,thread float &ponding1,thread float &ponding2) {
    float vol_liq[3]={0.0f,0.0f,0.0f},vol_ice[3]={0.0f,0.0f,0.0f},epore[3]={0.0f,0.0f,0.0f};
    if(c.sneqv==K_ZERO){c.sice[0]+=(qsnfro-qsnsub)*dt/(get_dz(c,1)*K_DENH2O);if(c.sice[0]<K_ZERO){c.sh2o[0]+=c.sice[0];c.sice[0]=K_ZERO;}}
    if(c.isnow==0&&c.sneqv>K_ZERO){float tmp=c.sneqv;c.sneqv=(c.sneqv-qsnsub*dt)+qsnfro*dt;float propor=c.sneqv/tmp;c.snowh=fmax_first(K_ZERO,propor*c.snowh);c.snowh=fmin_first(fmax_first(c.snowh,c.sneqv/K_500),c.sneqv/K_50);
        if(c.sneqv<K_ZERO){c.sice[0]+=c.sneqv/(get_dz(c,1)*K_DENH2O);c.sneqv=K_ZERO;c.snowh=K_ZERO;}if(c.sice[0]<K_ZERO){c.sh2o[0]+=c.sice[0];c.sice[0]=K_ZERO;}}
    if(c.snowh<=K_1EM8||c.sneqv<=K_1EM6){c.snowh=K_ZERO;c.sneqv=K_ZERO;}
    if(c.isnow<0){float wgdif=(get_ice(c,c.isnow+1)-qsnsub*dt)+qsnfro*dt;put_ice(c,c.isnow+1,wgdif);if(wgdif<K_1EM6&&c.isnow<0)combine(c,ponding1,ponding2);if(c.isnow<0)put_liq(c,c.isnow+1,fmax_first(K_ZERO,get_liq(c,c.isnow+1)+qrain*dt));}
    for(int j=c.isnow+1;j<=0;j++){uint k=si(j);vol_ice[k]=fmin_first(K_ONE,get_ice(c,j)/(get_dz(c,j)*K_DENICE));epore[k]=K_ONE-vol_ice[k];}
    float qin=K_ZERO,qout=K_ZERO;
    for(int j=c.isnow+1;j<=0;j++){uint k=si(j);put_liq(c,j,get_liq(c,j)+qin);vol_liq[k]=get_liq(c,j)/(get_dz(c,j)*K_DENH2O);qout=fmax_first(K_ZERO,(vol_liq[k]-ssi*epore[k])*get_dz(c,j));if(j==0)qout=fmax_first((vol_liq[k]-epore[k])*get_dz(c,j),snow_ret_fac*dt*qout);qout*=K_DENH2O;put_liq(c,j,get_liq(c,j)-qout);
        if(get_liq(c,j)/(get_ice(c,j)+get_liq(c,j))>K_MAXLIQ){qout+=get_liq(c,j)-(K_MAXLIQ/(K_ONE-K_MAXLIQ))*get_ice(c,j);put_liq(c,j,(K_MAXLIQ/(K_ONE-K_MAXLIQ))*get_ice(c,j));}qin=qout;}
    for(int j=c.isnow+1;j<=0;j++)put_dz(c,j,fmax_first(get_dz(c,j),get_liq(c,j)/K_DENH2O+get_ice(c,j)/K_DENICE));
    return qout/dt;
}

kernel void noahmp_snow_combo(constant NoahmpSnowComboResources &r [[buffer(0)]], constant NoahmpSnowComboParams &q [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(q.n))return; ulong a=ulong(tid)*53ul,b=ulong(tid)*41ul;
    float dz=r.in[a],liq=r.in[a+1],ice=r.in[a+2],t=r.in[a+3];combo(dz,liq,ice,t,r.in[a+4],r.in[a+5],r.in[a+6],r.in[a+7]);r.out[b]=dz;r.out[b+1]=liq;r.out[b+2]=ice;r.out[b+3]=t;
}
kernel void noahmp_snow_snowfall(constant NoahmpSnowSnowfallResources &r [[buffer(0)]],constant NoahmpSnowSnowfallParams &q [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(q.n))return;ulong a=ulong(tid)*53ul,b=ulong(tid)*41ul;SnowCol c;load_col(r.in+a,r.iin[ulong(tid)*4],c);snowfall(c,r.in[a+37],r.in[a+38],r.in[a+39],r.in[a+40]);store_col(c,r.out+b,r.iout+tid);
}
kernel void noahmp_snow_combine(constant NoahmpSnowCombineResources &r [[buffer(0)]],constant NoahmpSnowCombineParams &q [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(q.n))return;ulong a=ulong(tid)*53ul,b=ulong(tid)*41ul;SnowCol c;load_col(r.in+a,r.iin[ulong(tid)*4],c);float p1=r.in[a+37],p2=r.in[a+38];combine(c,p1,p2);store_col(c,r.out+b,r.iout+tid);r.out[b+37]=p1;r.out[b+38]=p2;
}
kernel void noahmp_snow_divide(constant NoahmpSnowDivideResources &r [[buffer(0)]],constant NoahmpSnowDivideParams &q [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(q.n))return;ulong a=ulong(tid)*53ul,b=ulong(tid)*41ul;SnowCol c;load_col(r.in+a,r.iin[ulong(tid)*4],c);divide(c);store_col(c,r.out+b,r.iout+tid);
}
kernel void noahmp_snow_snowh2o(constant NoahmpSnowSnowh2oResources &r [[buffer(0)]],constant NoahmpSnowSnowh2oParams &q [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(q.n))return;ulong a=ulong(tid)*53ul,b=ulong(tid)*41ul;SnowCol c;load_col(r.in+a,r.iin[ulong(tid)*4],c);float p1=r.in[a+43],p2=r.in[a+44];float qb=snowh2o(c,r.in[a+37],r.in[a+38],r.in[a+39],r.in[a+40],r.in[a+41],r.in[a+42],p1,p2);store_col(c,r.out+b,r.iout+tid);r.out[b+37]=qb;r.out[b+38]=p1;r.out[b+39]=p2;
}
