// Strict selected-source climate diagnostics; FP32 words retain gradual underflow.
#include <metal_stdlib>
#include "climate_diagnostics_exact_bindings.metalh"
#include "soft_word32.metalh"
using namespace metal;
#pragma clang fp contract(off)

// Exact finite FP32 remainder, including subnormal operands. The source clock
// uses positive finite intervals; special values follow fmod's domain rules.
static uint climate_fmod(uint x, uint y) {
    const uint sign=x & 0x80000000u;
    uint ax=x & 0x7fffffffu, ay=y & 0x7fffffffu;
    if (soft32_is_nan(x)) return soft32_quiet_nan(x);
    if (soft32_is_nan(y)) return soft32_quiet_nan(y);
    if (ax==0x7f800000u || ay==0u) return soft32_qnan();
    if (ay==0x7f800000u || ax<ay) return x;
    if (ax==ay) return sign;
    Soft32Decoded a=soft32_decode(x), b=soft32_decode(y);
    uint m=a.significand; int e=a.exponent;
    while(e>b.exponent) {
        if(m>=b.significand) m-=b.significand;
        if(m==0u) return sign;
        m <<= 1; --e;
    }
    if(m>=b.significand) m-=b.significand;
    if(m==0u) return sign;
    while(m<0x800000u) {m<<=1;--e;}
    return soft32_pack_extended(sign!=0u,e,m<<3);
}
static F divf(F x,F y){return f32_word(soft32_div(x.bits,y.bits));}
static F fsqrt(F x){return f32_word(soft32_sqrt(x.bits));}
static F maxzero(F x){return soft32_is_nan(x.bits)||soft32_sign(x.bits)?F(0.f):x;}
static void finish(thread F &mean,thread F &sd,int n){mean=divf(mean,F(float(n)));sd=fsqrt(maxzero(divf(sd,F(float(n)))-mean*mean));}
static void six(F x,F tt,bool reset,bool end,int n,device float *lo,device float *hi,device float *tlo,device float *thi,device float *avg,device float *sd,uint i){
 F l=F(lo[i]),h=F(hi[i]),tl=F(tlo[i]),th=F(thi[i]),a=F(avg[i]),s=F(sd[i]);
 if(reset){l=x;h=x;a=x;s=x*x;}else{if(x<l){l=x;tl=tt;}if(x>h){h=x;th=tt;}a=a+x;s=s+x*x;if(end)finish(a,s,n);}
 lo[i]=float(l);hi[i]=float(h);tlo[i]=float(tl);thi[i]=float(th);avg[i]=float(a);sd[i]=float(s);
}
static void four(F x,F tt,bool reset,bool end,int n,device float *hi,device float *thi,device float *avg,device float *sd,uint i){
 F h=F(hi[i]),th=F(thi[i]),a=F(avg[i]),s=F(sd[i]);
 if(reset){h=x;a=x;s=x*x;}else{if(x>h){h=x;th=tt;}a=a+x;s=s+x*x;if(end)finish(a,s,n);}
 hi[i]=float(h);thi[i]=float(th);avg[i]=float(a);sd[i]=float(s);
}
kernel void climate_diagnostics_exact(constant ClimateDiagnosticsExactResources &r [[buffer(0)]],constant ClimateDiagnosticsExactParams &p [[buffer(1)]],uint i [[thread_position_in_grid]]){
 if(i>=uint(p.count)) return;
 F dt=F(p.dt),secs=F(p.secs),period=divf(F(float(p.diag_int))*F(60.f),dt),wp=divf(F(float(p.wind_int))*F(60.f),dt);
 F zero=F(.01f);
 bool reset=f32_word(climate_fmod(divf(secs,dt).bits,period.bits)&0x7fffffffu)<zero && p.restart==0;
 bool end=f32_word(climate_fmod(divf(secs+dt,dt).bits,period.bits)&0x7fffffffu)<zero;
 bool wr=f32_word(climate_fmod(divf(secs,dt).bits,wp.bits)&0x7fffffffu)<zero && p.restart==0;
 bool wf=f32_word(climate_fmod(divf(secs+dt,dt).bits,wp.bits)&0x7fffffffu)<zero;
 int n=int(float(period)),nw=int(float(wp));r.cl_nsteps[i]=float(n);r.cl_nsteps_wind[i]=float(nw);
 F tt=F(p.xtime)+divf(dt,F(60.f));
 six(F(r.t2[i]),tt,reset,end,n,r.t2min,r.t2max,r.tt2min,r.tt2max,r.t2mean,r.t2std,i);
 six(F(r.q2[i]),tt,reset,end,n,r.q2min,r.q2max,r.tq2min,r.tq2max,r.q2mean,r.q2std,i);
 six(F(r.tsk[i]),tt,reset,end,n,r.skintempmin,r.skintempmax,r.tskintempmin,r.tskintempmax,r.skintempmean,r.skintempstd,i);
 six(F(r.swdown[i]),tt,reset,end,n,r.swradmin,r.swradmax,r.tswradmin,r.tswradmax,r.swradmean,r.swradstd,i);
 six(F(r.swddni[i]),tt,reset,end,n,r.swnormmin,r.swnormmax,r.tswnormmin,r.tswnormmax,r.swnormmean,r.swnormstd,i);
 four(divf(F(r.raincv[i]),dt),tt,reset,end,n,r.raincvmax,r.traincvmax,r.raincvmean,r.raincvstd,i);
 four(divf(F(r.rainncv[i]),dt),tt,reset,end,n,r.rainncvmax,r.trainncvmax,r.rainncvmean,r.rainncvstd,i);
 F u=F(r.u10[i]),v=F(r.v10[i]),speed2=u*u+v*v,speed=fsqrt(speed2);
 F umax=F(r.u10max[i]),vmax=F(r.v10max[i]),sm=F(r.spduv10max[i]),tm=F(r.tspduv10max[i]);
 F ua=F(r.u10mean[i]),va=F(r.v10mean[i]),sa=F(r.spduv10mean[i]);
 F us=F(r.u10std[i]),vs=F(r.v10std[i]),ss=F(r.spduv10std[i]);
 if(wr){sm=speed;ua=u;va=v;sa=speed;us=u*u;vs=v*v;ss=speed2;}
 else {if(speed>sm){umax=u;vmax=v;sm=speed;tm=tt;}sa=sa+speed;ss=ss+speed*speed;ua=ua+u;va=va+v;us=us+u*u;vs=vs+v*v;if(wf){finish(ua,us,nw);finish(va,vs,nw);finish(sa,ss,nw);}}
 r.u10max[i]=float(umax);r.v10max[i]=float(vmax);r.spduv10max[i]=float(sm);r.tspduv10max[i]=float(tm);
 r.u10mean[i]=float(ua);r.v10mean[i]=float(va);r.spduv10mean[i]=float(sa);r.u10std[i]=float(us);r.v10std[i]=float(vs);r.spduv10std[i]=float(ss);
}
