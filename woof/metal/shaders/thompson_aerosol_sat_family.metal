#include <metal_stdlib>
#include "thompson_aerosol_sat_family_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_binary32.metalh"
#include "soft_libm64.metalh"

using namespace metal;

constant float AA_PI = 3.1415926536f;
constant float AA_R_DRY = 287.04f;
constant float AA_RHO_W = 1000.0f;
constant float AA_R1 = 1.0e-12f;
constant float AA_NC_FLOOR = 2.0f;
constant float AA_NT_C_MAX = 1999.0e6f;
constant float AA_SAT_EPS = 1.0e-15f;
constant float AA_SSAT_GATE = 1.0e-6f;
constant float AA_XSAT_SNAP = 1.0e-9f;
constant uint AA_NBC = 100u;
constant uint AA_NTB_C = 37u;
constant uint AA_NTB_ARC = 7u;
constant uint AA_NTB_ARW = 9u;
constant uint AA_NTB_ART = 7u;
constant uint AA_NTB_ARR = 5u;
constant uint AA_NTB_ARK = 4u;
constant float AA_TA_NA[7] = {10.0f,31.6f,100.0f,316.0f,1000.0f,3160.0f,10000.0f};
constant float AA_TA_WW[9] = {0.01f,0.0316f,0.1f,0.316f,1.0f,3.16f,10.0f,31.6f,100.0f};

inline uint aa_bits(float x) { return as_type<uint>(x); }
inline float aa_f32(uint x) { return as_type<float>(x); }
inline float aa_narrow(ulong x);
inline ulong aa_widen(float x);
inline float aa_add(float a,float b) { return aa_f32(soft32_add(aa_bits(a),aa_bits(b))); }
inline float aa_sub(float a,float b) { return aa_f32(soft32_sub(aa_bits(a),aa_bits(b))); }
inline float aa_mul(float a,float b) { return aa_f32(soft32_mul(aa_bits(a),aa_bits(b))); }
inline float aa_div(float a,float b) { return aa_f32(soft32_div(aa_bits(a),aa_bits(b))); }
inline float aa_logf(float x) {
    return aa_narrow(soft64_plm_log(aa_widen(x)));
}
inline float aa_log10f(float x) {
    return aa_narrow(soft64_div(soft64_plm_log(aa_widen(x)),
                                 0x40026bb1bbb55516ul));
}
inline float aa_expf(float x) {
    return aa_narrow(soft64_plm_exp(aa_widen(x)));
}
inline float aa_powf(float x,float y) {
    return aa_narrow(soft64_plm_pow(aa_widen(x), aa_widen(y)));
}
inline float aa_sqrtf(float x) { return aa_f32(soft32_sqrt(aa_bits(x))); }
inline float aa_max(float a,float b) {
    uint x=aa_bits(a), y=aa_bits(b);
    if (soft32_is_nan(x)) return b;
    if (soft32_is_nan(y)) return a;
    if (soft32_is_zero(x)&&soft32_is_zero(y)) return (x&0x80000000u)?b:a;
    return soft32_greater(x,y)?a:b;
}
inline float aa_min(float a,float b) {
    uint x=aa_bits(a), y=aa_bits(b);
    if (soft32_is_nan(x)) return b;
    if (soft32_is_nan(y)) return a;
    if (soft32_is_zero(x)&&soft32_is_zero(y)) return (x&0x80000000u)?a:b;
    return soft32_less(x,y)?a:b;
}
inline float aa_abs(float x) { return aa_f32(aa_bits(x)&0x7fffffffu); }
inline float aa_neg(float x) { return aa_f32(aa_bits(x)^0x80000000u); }
inline ulong aa_d(float x) { return soft64_from_f32_bits(aa_bits(x)); }
inline float aa_f(ulong x) { return aa_f32(soft64_to_f32_bits(x)); }
// CUDA roundf: nearest integer with halfway cases away from zero. Decode the
// binary32 significand so tie handling adds no rounded floating intermediate.
// The caller's physical-domain input is finite and int32-representable.
inline int aa_roundf_away(float x) {
    const uint bits=aa_bits(x);
    const uint exponent=(bits>>23u)&0xffu;
    if(exponent==0u || exponent<126u) return 0;
    const int e=int(exponent)-127;
    const uint significand=(bits&0x007fffffu)|0x00800000u;
    uint magnitude;
    if(e<0) {
        magnitude=1u;
    } else if(e>=23) {
        magnitude=significand<<uint(e-23);
    } else {
        const uint shift=uint(23-e);
        const uint discarded=significand&((1u<<shift)-1u);
        magnitude=significand>>shift;
        if(discarded >= (1u<<(shift-1u))) magnitude+=1u;
    }
    const int signed_magnitude=int(magnitude);
    return (bits>>31u)?-signed_magnitude:signed_magnitude;
}
inline ulong aa_dadd(ulong a,ulong b) { return soft64_add(a,b); }
inline ulong aa_dsub(ulong a,ulong b) { return soft64_sub(a,b); }
inline ulong aa_dmul(ulong a,ulong b) { return soft64_mul(a,b); }
inline ulong aa_ddiv(ulong a,ulong b) { return soft64_div(a,b); }
inline ulong aa_dneg(ulong x) { return x^0x8000000000000000ul; }
inline ulong aa_dmax(ulong a,ulong b) {
    if(soft64_is_nan(a)) return b;
    if(soft64_is_nan(b)) return a;
    if(soft64_is_zero(a)&&soft64_is_zero(b)) return (a>>63u)?b:a;
    return soft64_greater(a,b)?a:b;
}
inline ulong aa_dmin(ulong a,ulong b) {
    if(soft64_is_nan(a)) return b;
    if(soft64_is_nan(b)) return a;
    if(soft64_is_zero(a)&&soft64_is_zero(b)) return (a>>63u)?a:b;
    return soft64_less(a,b)?a:b;
}
inline float aa_narrow(ulong x) { return aa_f32(soft64_to_f32_bits(x)); }
inline ulong aa_widen(float x) { return soft64_from_f32_bits(aa_bits(x)); }
inline int aa_nint_double(ulong xbits) {
    const ulong half_bits=0x3fe0000000000000ul;
    const ulong adjusted=(xbits>>63u)?soft64_sub(xbits,half_bits):soft64_add(xbits,half_bits);
    return soft64_to_i32_rz(adjusted).value;
}
struct AAEnv { float rho,qvs,ssatw,lvap,ocp,lvt2,diffu,tcond,otemp; };

inline float aa_rslf(float pressure,float temp) {
    const float c0=0.611583699e3f,c1=0.444606896e2f,c2=0.143177157e1f;
    const float c3=0.264224321e-1f,c4=0.299291081e-3f,c5=0.203154182e-5f;
    const float c6=0.702620698e-8f,c7=0.379534310e-11f,c8=-0.321582393e-13f;
    const float x=aa_max(-80.0f,aa_sub(temp,273.16f));
    float es=aa_add(c0,aa_mul(x,aa_add(c1,aa_mul(x,aa_add(c2,aa_mul(x,
      aa_add(c3,aa_mul(x,aa_add(c4,aa_mul(x,aa_add(c5,aa_mul(x,
      aa_add(c6,aa_mul(x,aa_add(c7,aa_mul(x,c8))))))))))))))));
    es=aa_min(es,aa_mul(pressure,0.15f));
    return aa_div(aa_mul(0.622f,es),aa_sub(pressure,es));
}

inline AAEnv aa_environment(float temp,float pressure,float qv) {
    AAEnv e;
    e.otemp=aa_div(1.0f,temp);
    float tc=aa_sub(temp,273.15f);
    e.rho=aa_div(aa_mul(0.622f,pressure),
      aa_mul(aa_mul(AA_R_DRY,temp),aa_add(qv,0.622f)));
    e.qvs=aa_rslf(pressure,temp);
    e.ssatw=aa_sub(aa_div(qv,e.qvs),1.0f);
    if (aa_abs(e.ssatw)<AA_SAT_EPS) e.ssatw=0.0f;
    e.diffu=aa_mul(aa_mul(2.11e-5f,aa_powf(aa_div(temp,273.15f),1.94f)),
                   aa_div(101325.0f,pressure));
    e.lvap=aa_add(2.5e6f,aa_mul(2106.0f-4218.0f,tc));
    e.tcond=aa_mul(aa_mul(aa_add(5.69f,aa_mul(0.0168f,tc)),1.0e-5f),418.936f);
    e.ocp=aa_div(1.0f,aa_mul(1004.0f,aa_add(1.0f,aa_mul(0.887f,qv))));
    e.lvt2=aa_mul(aa_mul(aa_mul(aa_mul(aa_mul(e.lvap,e.lvap),e.ocp),
                              1.0f/461.5f),e.otemp),e.otemp);
    return e;
}

inline float aa_activation(float temp,float w,float aerosol,
                           device const ulong *tnccn_act_bits) {
    float nlocal=aa_mul(aerosol,1.0e-6f);
    if(nlocal>=AA_TA_NA[6]) nlocal=aa_sub(AA_TA_NA[6],1.0f);
    else if(nlocal<=AA_TA_NA[0]) nlocal=aa_add(AA_TA_NA[0],1.0f);
    int i=7;
    for(int n=2;n<=7;++n) if(nlocal>=AA_TA_NA[n-2]&&nlocal<AA_TA_NA[n-1]) {i=n;break;}
    const float x1=aa_logf(AA_TA_NA[i-2]),x2=aa_logf(AA_TA_NA[i-1]);
    float wlocal=w;
    if(wlocal>=AA_TA_WW[8]) wlocal=aa_sub(AA_TA_WW[8],1.0f);
    else if(wlocal<=AA_TA_WW[0]) wlocal=aa_add(AA_TA_WW[0],0.001f);
    int j=9;
    for(int n=2;n<=9;++n) if(wlocal>=AA_TA_WW[n-2]&&wlocal<AA_TA_WW[n-1]) {j=n;break;}
    const float y1=aa_logf(AA_TA_WW[j-2]),y2=aa_logf(AA_TA_WW[j-1]);
    const int k=max(1,min(aa_roundf_away(aa_mul(aa_sub(temp,243.15f),0.1f))+1,7));
    const int l=3,m=2;
    const int tail=(k-1)+7*((l-1)+5*(m-1));
    const int base=7*9*tail;
    const int lo=base+7*(j-2),hi=base+7*(j-1);
    const float A=aa_f32(soft64_to_f32_bits(tnccn_act_bits[lo+i-2]));
    const float B=aa_f32(soft64_to_f32_bits(tnccn_act_bits[lo+i-1]));
    const float C=aa_f32(soft64_to_f32_bits(tnccn_act_bits[hi+i-1]));
    const float D=aa_f32(soft64_to_f32_bits(tnccn_act_bits[hi+i-2]));
    const float nx=aa_logf(nlocal),wy=aa_logf(wlocal);
    const float t=aa_div(aa_sub(nx,x1),aa_sub(x2,x1));
    const float u=aa_div(aa_sub(wy,y1),aa_sub(y2,y1));
    const float t1=aa_sub(1.0f,t),u1=aa_sub(1.0f,u);
    float f=aa_mul(aa_mul(t1,u1),A);
    f=aa_add(f,aa_mul(aa_mul(t,u1),B));
    f=aa_add(f,aa_mul(aa_mul(t,u),C));
    f=aa_add(f,aa_mul(aa_mul(t1,u),D));
    return aa_mul(aerosol,f);
}

inline int aa_decade_index(float value,int first_exponent,int table_size) {
    const int center=aa_roundf_away(aa_log10f(value));
    int exponent=center;
    for(int candidate=center-1;candidate<=center+1;++candidate) {
        const float scale=aa_powf(10.0f,float(candidate));
        const float mantissa=aa_div(value,scale);
        if(mantissa>=1.0f&&mantissa<10.0f) {exponent=candidate;break;}
    }
    const float scale=aa_powf(10.0f,float(exponent));
    const int digit=int(aa_div(value,scale));
    const int one_based=digit+9*(exponent-first_exponent);
    return max(0,min(one_based-1,table_size-1));
}

inline int aa_droplet_bin(float nc) {
    ulong ratio=aa_ddiv(aa_widen(nc),0x412fc397d2e29693ul);
    ulong raw=aa_dadd(0x3ff0000000000000ul,
      aa_ddiv(aa_dmul(0x4059000000000000ul,soft64_plm_log(ratio)),
              0x401c000000000000ul));
    return max(1,min(aa_nint_double(raw),100))-1;
}

inline ulong aa_droplet_evap(AAEnv e,float rc,float nc,float orho,float odt,float dt,
                              device const ulong *tnc,thread ulong &prw,
                              thread int &idx_d,thread int &idx_c,thread int &idx_n) {
    const float rvs=aa_mul(e.rho,e.qvs);
    const float tt=aa_sub(aa_mul(aa_mul(e.lvap,e.otemp),1.0f/461.5f),1.0f);
    const float rvsp=aa_mul(aa_mul(rvs,e.otemp),tt);
    const float ppa=aa_mul(aa_mul(aa_mul(e.otemp,tt),e.otemp),tt);
    float ppb=aa_mul(-2.0f,e.lvap); ppb=aa_mul(ppb,e.otemp); ppb=aa_mul(ppb,e.otemp);
    ppb=aa_mul(ppb,e.otemp); ppb=aa_mul(ppb,1.0f/461.5f);
    const float ppc=aa_mul(e.otemp,e.otemp);
    const float rvspp=aa_mul(rvs,aa_add(aa_add(ppa,ppb),ppc));
    const float g=aa_mul(aa_div(aa_mul(e.lvap,e.diffu),e.tcond),rvsp);
    const float gr=aa_div(g,aa_add(1.0f,g));
    float alphsc=aa_mul(0.5f,gr); alphsc=aa_mul(alphsc,gr);
    alphsc=aa_mul(alphsc,rvspp); alphsc=aa_div(alphsc,rvsp);
    alphsc=aa_mul(alphsc,rvs); alphsc=aa_div(alphsc,rvsp);
    alphsc=aa_max(1.0e-9f,alphsc);
    float xsat=e.ssatw; if(aa_abs(xsat)<AA_XSAT_SNAP) xsat=0.0f;
    const float ax=aa_mul(alphsc,xsat);
    float t2=aa_mul(2.0f,alphsc); t2=aa_mul(t2,alphsc);
    t2=aa_mul(t2,xsat); t2=aa_mul(t2,xsat);
    float t3=aa_mul(5.0f,alphsc); t3=aa_mul(t3,alphsc); t3=aa_mul(t3,alphsc);
    t3=aa_mul(t3,xsat); t3=aa_mul(t3,xsat); t3=aa_mul(t3,xsat);
    float par=aa_sub(1.0f,ax); par=aa_add(par,t2); par=aa_sub(par,t3);
    const float two_pi=aa_mul(2.0f,AA_PI);
    float t1=aa_mul(two_pi,par); t1=aa_div(t1,aa_add(1.0f,g));
    ulong arg=aa_dmul(0xc000000000000000ul,aa_widen(dt));
    arg=aa_dmul(arg,aa_widen(t1)); arg=aa_ddiv(arg,aa_widen(two_pi));
    arg=aa_dmul(arg,0x4010000000000000ul); arg=aa_dmul(arg,aa_widen(e.diffu));
    arg=aa_dmul(arg,aa_widen(e.ssatw)); arg=aa_dmul(arg,aa_widen(rvs));
    arg=aa_ddiv(arg,aa_widen(AA_RHO_W));
    const ulong dc_star=soft64_sqrt(arg);
    const ulong dmicron=aa_dmul(dc_star,0x412e848000000000ul);
    const Soft64I32Result dtrunc=soft64_to_i32_rz(dmicron);
    idx_d=max(1,min(dtrunc.value,int(AA_NBC)));
    const int idx_n0=aa_droplet_bin(nc);
    idx_n=idx_n0+1;
    const int idx_c0=(rc>1.0e-6f)?aa_decade_index(rc,-6,int(AA_NTB_C)):0;
    idx_c=idx_c0+1;
    const int flat=(idx_d-1)+int(AA_NBC)*((idx_c-1)+int(AA_NTB_C)*(idx_n-1));
    const ulong tnc_bits=tnc[flat];
    const float floor32=aa_mul(aa_mul(aa_mul(-rc,0.99f),orho),odt);
    prw=aa_dmax(aa_widen(floor32),prw);
    const float pnc32=aa_mul(aa_mul(aa_mul(-nc,0.99f),orho),odt);
    ulong table_rate=aa_dmul(aa_dneg(tnc_bits),aa_widen(orho));
    table_rate=aa_dmul(table_rate,aa_widen(odt));
    return aa_dmax(aa_widen(pnc32),table_rate);
}

kernel void thompson_aa_saturation_adjust(
    constant ThompsonAaSaturationAdjustResources &r [[buffer(0)]],
    constant ThompsonAaSaturationAdjustParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    device float *temperature = r.temperature;
    device const float *pressure = r.pressure;
    device float *qv = r.qv;
    device float *qc = r.qc;
    device const float *nc_entry = r.nc_entry;
    device float *ncten = r.ncten;
    device float *nwfaten = r.nwfaten;
    device const float *nwfa_work_m3 = r.nwfa_work_m3;
    device const float *w = r.w;
    device const ulong *tnccn_act_bits = r.tnccn_act_bits;
    device const ulong *tnc_wev_bits = r.tnc_wev_bits;
    device float *reference_density = r.reference_density;
    device float *reference_temperature = r.reference_temperature;
    device float *condensation_rate = r.condensation_rate;
    device float *cloud_presence = r.cloud_presence;
    const float dt = p.dt;
    const int size = p.size;
    const int has_reference_density = p.has_reference_density;
    const int has_reference_temperature = p.has_reference_temperature;
    const int has_condensation_rate = p.has_condensation_rate;
    const int has_cloud_presence = p.has_cloud_presence;
    if(idx>=uint(size)) return;
    const float temp0=temperature[idx],pres=pressure[idx];
    const float qv0=aa_max(1.0e-10f,qv[idx]),qc0=qc[idx];
    AAEnv e=aa_environment(temp0,pres,qv0);
    if(has_reference_density) reference_density[idx]=e.rho;
    if(has_reference_temperature) reference_temperature[idx]=temp0;
    if(has_condensation_rate) condensation_rate[idx]=0.0f;
    const bool lqc=qc0>AA_R1;
    if(has_cloud_presence) cloud_presence[idx]=lqc?1.0f:0.0f;
    const float orho=aa_div(1.0f,e.rho),odt=aa_div(1.0f,dt);
    const float rc=lqc?aa_mul(qc0,e.rho):AA_R1;
    const float nc_work=lqc?aa_max(AA_NC_FLOOR,aa_min(
      aa_mul(aa_add(nc_entry[idx],aa_mul(ncten[idx],dt)),e.rho),AA_NT_C_MAX)):AA_NC_FLOOR;
    if(!(e.ssatw>AA_SAT_EPS||(e.ssatw < -AA_SAT_EPS&&lqc))) return;
    float clap=aa_div(aa_sub(qv0,e.qvs),aa_add(1.0f,aa_mul(e.lvt2,e.qvs)));
    for(int it=0;it<3;++it) {
        const float ex=aa_expf(aa_mul(e.lvt2,clap));
        const float fcd=aa_add(aa_sub(aa_mul(e.qvs,ex),qv0),clap);
        const float dfcd=aa_add(aa_mul(aa_mul(e.qvs,e.lvt2),ex),1.0f);
        clap=aa_sub(clap,aa_div(fcd,dfcd));
    }
    const float xrc=aa_add(rc,aa_mul(clap,e.rho));
    ulong prw=0ul,pnc=0ul;
    if(xrc>AA_R1) {
        prw=aa_widen(aa_mul(clap,odt));
        if(clap>AA_SAT_EPS) {
            const float xnc=aa_max(AA_NC_FLOOR,aa_activation(temp0,w[idx],nwfa_work_m3[idx],tnccn_act_bits));
            const float diff=aa_sub(xnc,nc_work);
            float rate=aa_mul(0.5f,aa_add(diff,aa_abs(diff)));
            rate=aa_mul(rate,odt); rate=aa_mul(rate,orho); pnc=aa_widen(rate);
        } else if(clap < -AA_SAT_EPS&&e.ssatw < -AA_SSAT_GATE) {
            int id=0,ic=0,in=0;
            pnc=aa_droplet_evap(e,rc,nc_work,orho,odt,dt,tnc_wev_bits,prw,id,ic,in);
        }
    } else {
        prw=aa_widen(aa_mul(aa_mul(-rc,orho),odt));
        pnc=aa_widen(aa_mul(aa_mul(-nc_work,orho),odt));
    }
    const float prw32=aa_narrow(prw);
    const float tten=aa_narrow(aa_dmul(aa_widen(aa_mul(e.lvap,e.ocp)),prw));
    qv[idx]=aa_sub(qv[idx],aa_mul(prw32,dt));
    qc[idx]=aa_add(qc0,aa_mul(prw32,dt));
    if(has_cloud_presence&&!(aa_mul(qc[idx],e.rho)>AA_R1)) cloud_presence[idx]=0.0f;
    temperature[idx]=aa_add(temp0,aa_mul(tten,dt));
    ncten[idx]=aa_narrow(aa_dadd(aa_widen(ncten[idx]),pnc));
    nwfaten[idx]=aa_narrow(aa_dsub(aa_widen(nwfaten[idx]),pnc));
    if(has_condensation_rate) condensation_rate[idx]=prw32;
}

inline float aa_bound_rain_number(float rr,float nr_m3,thread ulong &lamr_out) {
    const float amr=523.59881591796875f;
    float nr=aa_max(1.0e-6f,nr_m3);
    float base=aa_mul(aa_mul(aa_mul(amr,6.0f),1.0f),nr);
    base=aa_div(base,rr);
    lamr_out=aa_widen(aa_powf(base,1.0f/3.0f));
    const float mvd_num=3.672f;
    float mvd=aa_narrow(aa_ddiv(aa_widen(mvd_num),lamr_out));
    const bool high=mvd>2.5e-3f;
    const bool low=mvd<aa_mul(50.0e-6f,0.75f);
    if(high||low) {
        mvd=high?2.5e-3f:aa_mul(50.0e-6f,0.75f);
        lamr_out=aa_widen(aa_div(mvd_num,mvd));
        const float prefix=aa_mul(aa_mul(1.0f,1.0f/6.0f),rr);
        ulong nr_bits=soft64_plm_pow(lamr_out,0x4008000000000000ul);
        nr_bits=aa_dmul(aa_widen(prefix),nr_bits);
        nr_bits=aa_ddiv(nr_bits,aa_widen(amr));
        nr=aa_narrow(nr_bits);
        base=aa_mul(aa_mul(aa_mul(amr,6.0f),1.0f),nr);
        base=aa_div(base,rr);
        lamr_out=aa_widen(aa_powf(base,1.0f/3.0f));
    }
    return nr;
}

inline void aa_rain_evap(
    device float *qr,device float *nr,device float *temperature,
    device const float *pressure,device float *qv,device float *nwfaten,
    device float *reference_density,device float *reference_temperature,
    device const float *graupel_melt_marker,device const float *condensation_rate,
    device const float *entry_density,bool has_nwfaten,bool has_reference_density,
    bool has_reference_temperature,bool has_graupel_marker,bool has_condensation_rate,
    bool has_entry_density,float dt,uint idx) {
    const float temp0=temperature[idx],qv0=aa_max(1.0e-10f,qv[idx]);
    const float pres=pressure[idx];
    const float rho=aa_div(aa_mul(0.622f,pres),
      aa_mul(aa_mul(AA_R_DRY,temp0),aa_add(qv0,0.622f)));
    const bool melting=has_graupel_marker&&graupel_melt_marker[idx]>0.0f;
    const float rho_entry=has_entry_density?entry_density[idx]:rho;
    if(has_reference_density) reference_density[idx]=rho_entry;
    if(has_reference_temperature) reference_temperature[idx]=temp0;
    if(qr[idx]<=AA_R1) {if(has_reference_density)reference_density[idx]=0.0f;return;}
    if(has_condensation_rate&&condensation_rate[idx]>0.0f) return;
    const float qvs=aa_rslf(pres,temp0);
    float ssatw=aa_sub(aa_div(qv0,qvs),1.0f);
    if(aa_abs(ssatw)<AA_SAT_EPS) ssatw=0.0f;
    if(ssatw>=-AA_SAT_EPS) return;
    if(has_reference_density) reference_density[idx]=-rho;
    const float orho=aa_div(1.0f,rho),odt=aa_div(1.0f,dt);
    const float rr=aa_mul(qr[idx],rho_entry);
    ulong lamr0=0ul;
    const float nrwork=aa_bound_rain_number(rr,aa_mul(nr[idx],rho_entry),lamr0);
    ulong ilamr=aa_ddiv(0x3ff0000000000000ul,lamr0);
    ulong n0r=aa_dmul(aa_widen(aa_mul(nrwork,1.0f)),lamr0);
    const ulong lamr=aa_ddiv(0x3ff0000000000000ul,ilamr);
    const float tc=aa_sub(temp0,273.15f),otemp=aa_div(1.0f,temp0);
    const float rho_not=aa_div(101325.0f,aa_mul(287.05f,298.0f));
    const float rhof=aa_sqrtf(aa_mul(rho_not,orho));
    const float rhof2=aa_sqrtf(rhof);
    const float diffu=aa_mul(aa_mul(2.11e-5f,aa_powf(aa_div(temp0,273.15f),1.94f)),
                             aa_div(101325.0f,pres));
    const float visbase=aa_add(1.718f,aa_mul(0.0049f,tc));
    const float visco=tc>=0.0f?aa_mul(visbase,1.0e-5f):
       aa_mul(aa_sub(visbase,aa_mul(aa_mul(1.2e-5f,tc),tc)),1.0e-5f);
    const float vsc2=aa_sqrtf(aa_div(rho,visco));
    const float lvap=aa_add(2.5e6f,aa_mul(2106.0f-4218.0f,tc));
    const float tcond=aa_mul(aa_mul(aa_add(5.69f,aa_mul(0.0168f,tc)),1.0e-5f),418.936f);
    const float ocp=aa_div(1.0f,aa_mul(1004.0f,aa_add(1.0f,aa_mul(0.887f,qv0))));
    const float rvs=aa_mul(rho,qvs);
    const float tt=aa_sub(aa_mul(aa_mul(lvap,otemp),1.0f/461.5f),1.0f);
    const float rvsp=aa_mul(aa_mul(rvs,otemp),tt);
    const float ppa=aa_mul(aa_mul(aa_mul(otemp,tt),otemp),tt);
    float ppb=aa_mul(-2.0f,lvap);ppb=aa_mul(ppb,otemp);ppb=aa_mul(ppb,otemp);
    ppb=aa_mul(ppb,otemp);ppb=aa_mul(ppb,1.0f/461.5f);
    const float ppc=aa_mul(otemp,otemp);
    const float rvspp=aa_mul(rvs,aa_add(aa_add(ppa,ppb),ppc));
    const float g=aa_mul(aa_div(aa_mul(lvap,diffu),tcond),rvsp);
    const float gr=aa_div(g,aa_add(1.0f,g));
    float alphsc=aa_mul(0.5f,gr);alphsc=aa_mul(alphsc,gr);alphsc=aa_mul(alphsc,rvspp);
    alphsc=aa_div(alphsc,rvsp);alphsc=aa_mul(alphsc,rvs);alphsc=aa_div(alphsc,rvsp);
    alphsc=aa_max(1.0e-9f,alphsc);
    const float xsat=aa_min(-1.0e-9f,ssatw);
    const float ax=aa_mul(alphsc,xsat);
    float t2=aa_mul(2.0f,alphsc);t2=aa_mul(t2,alphsc);t2=aa_mul(t2,xsat);t2=aa_mul(t2,xsat);
    float t3=aa_mul(5.0f,alphsc);t3=aa_mul(t3,alphsc);t3=aa_mul(t3,alphsc);
    t3=aa_mul(t3,xsat);t3=aa_mul(t3,xsat);t3=aa_mul(t3,xsat);
    float par=aa_sub(1.0f,ax);par=aa_add(par,t2);par=aa_sub(par,t3);
    float t1=aa_mul(aa_mul(2.0f,AA_PI),par);t1=aa_div(t1,aa_add(1.0f,g));
    const float sc3=aa_powf(0.632f,1.0f/3.0f);
    const float t1q=aa_mul(0.78f,1.0f);
    const float t2q=aa_mul(aa_mul(aa_mul(0.308f,sc3),aa_sqrtf(4854.0f)),2.0f);
    const float prefix=aa_mul(aa_mul(t1,diffu),-ssatw);
    ulong diff_term=soft64_plm_pow(ilamr,0x4000000000000000ul);
    diff_term=aa_dmul(aa_widen(t1q),diff_term);
    float ventpre=aa_mul(aa_mul(t2q,vsc2),rhof2);
    const ulong ventbase=aa_dadd(lamr,aa_widen(aa_mul(0.5f,195.0f)));
    ulong ventterm=soft64_plm_pow(ventbase,0xc008000000000000ul);
    ventterm=aa_dmul(aa_widen(ventpre),ventterm);
    ulong rate=aa_dmul(aa_widen(prefix),n0r);
    rate=aa_dmul(rate,aa_widen(rvs));
    rate=aa_dmul(rate,aa_dadd(diff_term,ventterm));
    ulong prv;
    if(aa_div(qv0,qvs)<0.95f&&aa_mul(rr,orho)<=1.0e-8f) {
        prv=aa_widen(aa_mul(aa_mul(rr,orho),odt));
    } else {
        const float rate_max=aa_min(aa_mul(aa_mul(rr,orho),odt),
                                    aa_mul(aa_sub(qvs,qv0),odt));
        prv=aa_dmin(aa_widen(rate_max),aa_dmul(rate,aa_widen(orho)));
        if(melting) {
            const float eva=aa_min(1.0f,aa_add(0.01f,
              aa_mul(0.99f-0.01f,aa_div(tc,20.0f))));
            prv=aa_dmul(prv,aa_widen(eva));
        }
    }
    const float pnr_cap=aa_mul(aa_mul(aa_mul(nrwork,0.99f),orho),odt);
    ulong pnr=aa_dmin(aa_widen(pnr_cap),
      aa_ddiv(aa_dmul(prv,aa_widen(nrwork)),aa_widen(rr)));
    const float qv_t=aa_narrow(prv),nr_t=aa_narrow(pnr);
    const float temp_t=aa_neg(aa_narrow(aa_dmul(aa_widen(aa_mul(lvap,ocp)),prv)));
    qr[idx]=aa_add(qr[idx],aa_mul(aa_neg(aa_narrow(prv)),dt));
    qv[idx]=aa_add(qv[idx],aa_mul(qv_t,dt));
    nr[idx]=aa_add(nr[idx],aa_mul(aa_neg(nr_t),dt));
    temperature[idx]=aa_add(temp0,aa_mul(temp_t,dt));
    if(has_nwfaten) nwfaten[idx]=aa_narrow(aa_dadd(aa_widen(nwfaten[idx]),pnr));
}

kernel void thompson_aa_rain_evaporation(
    constant ThompsonAaRainEvaporationResources &r [[buffer(0)]],
    constant ThompsonAaRainEvaporationParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    device float *qr = r.qr;
    device float *nr = r.nr;
    device float *temperature = r.temperature;
    device const float *pressure = r.pressure;
    device float *qv = r.qv;
    device float *nwfaten = r.nwfaten;
    device float *reference_density = r.reference_density;
    device float *reference_temperature = r.reference_temperature;
    device const float *graupel_melt_marker = r.graupel_melt_marker;
    device const float *condensation_rate = r.condensation_rate;
    device const float *entry_density = r.entry_density;
    const float dt = p.dt;
    const int size = p.size;
    const int has_nwfaten = p.has_nwfaten;
    const int has_reference_density = p.has_reference_density;
    const int has_reference_temperature = p.has_reference_temperature;
    const int has_graupel_marker = p.has_graupel_marker;
    const int has_condensation_rate = p.has_condensation_rate;
    const int has_entry_density = p.has_entry_density;
    if(idx>=uint(size)) return;
    aa_rain_evap(qr,nr,temperature,pressure,qv,nwfaten,reference_density,
      reference_temperature,graupel_melt_marker,condensation_rate,entry_density,
      has_nwfaten!=0,has_reference_density!=0,
      has_reference_temperature!=0,has_graupel_marker!=0,
      has_condensation_rate!=0,has_entry_density!=0,dt,idx);
}
