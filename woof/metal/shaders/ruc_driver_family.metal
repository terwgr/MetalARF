#include <metal_stdlib>
#include "ruc_driver_family_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"

using namespace metal;
#pragma clang fp contract(off)

#ifndef NAN
#define NAN as_type<float>(0x7fc00000u)
#endif

static inline float ruc_fadd_rn(float a,float b) { return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b))); }
static inline float ruc_fsub_rn(float a,float b) { return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b))); }
static inline float ruc_fmul_rn(float a,float b) { return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b))); }
static inline float ruc_fdiv_rn(float a,float b) { return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b))); }
static inline float ruc_fused_min(float a,float b) { return (isnan(a)||isnan(b)) ? as_type<float>(0x7fc00000u) : min(a,b); }
static inline float ruc_fused_max(float a,float b) { return (isnan(a)||isnan(b)) ? as_type<float>(0x7fc00000u) : max(a,b); }
static inline float ruc_qsn_lookup(float tn, device const float* table) {
    float raw = ruc_fsub_rn(tn, 173.15f);
    raw = ruc_fdiv_rn(raw, 0.05f);
    raw = ruc_fadd_rn(raw, 1.0f);
    int index = int(raw);
    if (index < 1) { index = 1; raw = 1.0f; }
    if (index > 5000) { index = 5000; raw = 5001.0f; }
    float lower = table[index - 1];
    float weight = ruc_fsub_rn(raw, float(index));
    float value = ruc_fsub_rn(table[index], lower);
    value = ruc_fmul_rn(value, weight);
    return ruc_fadd_rn(value, lower);
}
constant float ruc_soil_layer_depth[9] = {0.00f,0.01f,0.04f,0.10f,0.30f,0.60f,1.00f,1.60f,3.00f};
#define RUC_NZS 9

static inline float gfk_log(float value) { return as_type<float>(soft_gfk_log(as_type<uint>(value))); }
static inline float gfk_exp(float value) { return as_type<float>(soft_gfk_exp(as_type<uint>(value))); }
static inline float ruc_fsqrt_rn(float value) { return as_type<float>(soft32_sqrt(as_type<uint>(value))); }
static inline void ruc_mosaic_parameters(
    int i,int n,int nlcat,int nscat,int mosaic_lu,int mosaic_soil,
    device const float* landusef,device const float* soilctop,
    int soil,int iswater,bool rdlai2d,float factor,float incoming_znt,
    device const int* ifortbl,device const float* z0tbl,device const float* lemitbl,
    device const float* pctbl,device const float* laitbl,device const float* bb,
    device const float* drysmc,device const float* hc,device const float* maxsmc,
    device const float* refsmc,device const float* satpsi,device const float* satdk,
    device const float* wltsmc,device const float* qtz,
    device float& emiss,device float& pc,device float& znt,device float& lai,
    device float& qwrtz,device float& rhocs,device float& bclh,device float& dqm,
    device float& ksat,device float& psis,device float& qmin,device float& ref,device float& wilt) {
    if(mosaic_lu) {
        float area=0.0f, rough=0.0f, leaf=0.0f;
        emiss=0.0f; pc=0.0f;
        for(int k=0;k<nlcat;++k) {
            float fraction=landusef[k*n+i];
            int forest=ifortbl[k];
            float delta=0.0f,scaled=ruc_fmul_rn(0.8f,laitbl[k]);
            if(forest==1) delta=fmin(0.2f,scaled);
            if(forest==2 || forest==7) delta=fmin(0.5f,scaled);
            if(forest==3) delta=fmin(0.45f,scaled);
            if(forest==4) delta=fmin(0.75f,scaled);
            if(forest==5) delta=fmin(0.86f,scaled);
            float today_lai=k+1==iswater ? laitbl[k]:ruc_fsub_rn(laitbl[k],ruc_fmul_rn(delta,factor));
            float today_znt=k+1==iswater ? incoming_znt:(forest==7 ? ruc_fsub_rn(z0tbl[k],ruc_fmul_rn(0.125f,factor)):z0tbl[k]);
            float logarithm=gfk_log(ruc_fdiv_rn(5.0f,today_znt));
            area=ruc_fadd_rn(area,fraction);
            emiss=ruc_fadd_rn(emiss,ruc_fmul_rn(lemitbl[k],fraction));
            rough=ruc_fadd_rn(rough,ruc_fdiv_rn(fraction,ruc_fmul_rn(logarithm,logarithm)));
            if(!rdlai2d) leaf=ruc_fadd_rn(leaf,ruc_fmul_rn(today_lai,fraction));
            pc=ruc_fadd_rn(pc,ruc_fmul_rn(pctbl[k],fraction));
        }
        area=fmin(1.0f,area);
        emiss=ruc_fdiv_rn(emiss,area);
        pc=ruc_fdiv_rn(pc,area);
        znt=ruc_fdiv_rn(5.0f,gfk_exp(ruc_fsqrt_rn(ruc_fdiv_rn(1.0f,rough))));
        if(!rdlai2d) lai=ruc_fdiv_rn(leaf,area);
    }
    if(mosaic_soil) {
        float area=0.0f;
        qwrtz=0.0f; rhocs=0.0f; bclh=0.0f; dqm=0.0f; ksat=0.0f;
        psis=0.0f; qmin=0.0f; ref=0.0f; wilt=0.0f;
        for(int k=0;k<nscat;++k) {
            if(k==13) continue;
            float fraction=soilctop[k*n+i];
            area=ruc_fadd_rn(area,fraction);
            rhocs=ruc_fadd_rn(rhocs,ruc_fmul_rn(ruc_fmul_rn(hc[k],1.0e6f),fraction));
            bclh=ruc_fadd_rn(bclh,ruc_fmul_rn(bb[k],fraction));
            dqm=ruc_fadd_rn(dqm,ruc_fmul_rn(ruc_fsub_rn(maxsmc[k],drysmc[k]),fraction));
            ksat=ruc_fadd_rn(ksat,ruc_fmul_rn(satdk[k],fraction));
            psis=ruc_fsub_rn(psis,ruc_fmul_rn(satpsi[k],fraction));
            qmin=ruc_fadd_rn(qmin,ruc_fmul_rn(drysmc[k],fraction));
            ref=ruc_fadd_rn(ref,ruc_fmul_rn(refsmc[k],fraction));
            wilt=ruc_fadd_rn(wilt,ruc_fmul_rn(wltsmc[k],fraction));
            qwrtz=ruc_fadd_rn(qwrtz,ruc_fmul_rn(qtz[k],fraction));
        }
        area=fmin(1.0f,area);
        if(area<=0.0f) {
            int k=soil-1;
            rhocs=ruc_fmul_rn(hc[k],1.0e6f); bclh=bb[k];
            dqm=ruc_fsub_rn(maxsmc[k],drysmc[k]); ksat=satdk[k];
            psis=-satpsi[k]; qmin=drysmc[k]; ref=refsmc[k];
            wilt=wltsmc[k]; qwrtz=qtz[k];
        } else {
            rhocs=ruc_fdiv_rn(rhocs,area); bclh=ruc_fdiv_rn(bclh,area);
            dqm=ruc_fdiv_rn(dqm,area); ksat=ruc_fdiv_rn(ksat,area);
            psis=ruc_fdiv_rn(psis,area); qmin=ruc_fdiv_rn(qmin,area);
            ref=ruc_fdiv_rn(ref,area); wilt=ruc_fdiv_rn(wilt,area);
            qwrtz=ruc_fdiv_rn(qwrtz,area);
        }
    }
}
#define D_ADMIT for (int k=0;k<RUC_NZS;++k) if (!isfinite((*reinterpret_cast<device float *>(r.raw + (123 + uint(k)) * uint(n) + uint(i))))) { d_flag(flags,0); admitted=false; } \
for (int k=0;k<RUC_NZS;++k) if (!isfinite((*reinterpret_cast<device float *>(r.raw + (132 + uint(k)) * uint(n) + uint(i))))) { d_flag(flags,1); admitted=false; } \
for (int k=0;k<RUC_NZS;++k) if (!isfinite((*reinterpret_cast<device float *>(r.raw + (141 + uint(k)) * uint(n) + uint(i))))) { d_flag(flags,2); admitted=false; } \
for (int k=0;k<RUC_NZS;++k) if (!isfinite((*reinterpret_cast<device float *>(r.raw + (150 + uint(k)) * uint(n) + uint(i))))) { d_flag(flags,3); admitted=false; } \
for (int k=0;k<RUC_NZS;++k) if (!isfinite((*reinterpret_cast<device float *>(r.raw + (159 + uint(k)) * uint(n) + uint(i))))) { d_flag(flags,4); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 0 * uint(n) + uint(i))))) { d_flag(flags,5); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 1 * uint(n) + uint(i))))) { d_flag(flags,6); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 2 * uint(n) + uint(i))))) { d_flag(flags,7); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 3 * uint(n) + uint(i))))) { d_flag(flags,8); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 4 * uint(n) + uint(i))))) { d_flag(flags,9); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 5 * uint(n) + uint(i))))) { d_flag(flags,10); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 6 * uint(n) + uint(i))))) { d_flag(flags,11); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 7 * uint(n) + uint(i))))) { d_flag(flags,12); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 8 * uint(n) + uint(i))))) { d_flag(flags,13); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 9 * uint(n) + uint(i))))) { d_flag(flags,14); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 10 * uint(n) + uint(i))))) { d_flag(flags,15); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 11 * uint(n) + uint(i))))) { d_flag(flags,16); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))))) { d_flag(flags,17); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 13 * uint(n) + uint(i))))) { d_flag(flags,18); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 14 * uint(n) + uint(i))))) { d_flag(flags,19); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 15 * uint(n) + uint(i))))) { d_flag(flags,20); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 16 * uint(n) + uint(i))))) { d_flag(flags,21); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 17 * uint(n) + uint(i))))) { d_flag(flags,22); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 18 * uint(n) + uint(i))))) { d_flag(flags,23); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 19 * uint(n) + uint(i))))) { d_flag(flags,24); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 20 * uint(n) + uint(i))))) { d_flag(flags,25); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 21 * uint(n) + uint(i))))) { d_flag(flags,26); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 22 * uint(n) + uint(i))))) { d_flag(flags,27); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i))))) { d_flag(flags,28); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 24 * uint(n) + uint(i))))) { d_flag(flags,29); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 25 * uint(n) + uint(i))))) { d_flag(flags,30); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 26 * uint(n) + uint(i))))) { d_flag(flags,31); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 27 * uint(n) + uint(i))))) { d_flag(flags,32); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 28 * uint(n) + uint(i))))) { d_flag(flags,33); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 29 * uint(n) + uint(i))))) { d_flag(flags,34); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 30 * uint(n) + uint(i))))) { d_flag(flags,35); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 31 * uint(n) + uint(i))))) { d_flag(flags,36); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 32 * uint(n) + uint(i))))) { d_flag(flags,37); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 33 * uint(n) + uint(i))))) { d_flag(flags,38); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 34 * uint(n) + uint(i))))) { d_flag(flags,39); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 35 * uint(n) + uint(i))))) { d_flag(flags,40); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 36 * uint(n) + uint(i))))) { d_flag(flags,41); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 37 * uint(n) + uint(i))))) { d_flag(flags,42); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 38 * uint(n) + uint(i))))) { d_flag(flags,43); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 39 * uint(n) + uint(i))))) { d_flag(flags,44); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 40 * uint(n) + uint(i))))) { d_flag(flags,45); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 41 * uint(n) + uint(i))))) { d_flag(flags,46); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 42 * uint(n) + uint(i))))) { d_flag(flags,47); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 43 * uint(n) + uint(i))))) { d_flag(flags,48); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 44 * uint(n) + uint(i))))) { d_flag(flags,49); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 45 * uint(n) + uint(i))))) { d_flag(flags,50); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 46 * uint(n) + uint(i))))) { d_flag(flags,51); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 47 * uint(n) + uint(i))))) { d_flag(flags,52); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 48 * uint(n) + uint(i))))) { d_flag(flags,53); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 49 * uint(n) + uint(i))))) { d_flag(flags,54); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 50 * uint(n) + uint(i))))) { d_flag(flags,55); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 51 * uint(n) + uint(i))))) { d_flag(flags,56); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 52 * uint(n) + uint(i))))) { d_flag(flags,57); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 53 * uint(n) + uint(i))))) { d_flag(flags,58); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 54 * uint(n) + uint(i))))) { d_flag(flags,59); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 55 * uint(n) + uint(i))))) { d_flag(flags,60); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 56 * uint(n) + uint(i))))) { d_flag(flags,61); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 57 * uint(n) + uint(i))))) { d_flag(flags,62); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 58 * uint(n) + uint(i))))) { d_flag(flags,63); admitted=false; } \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 59 * uint(n) + uint(i))))) { d_flag(flags,64); admitted=false; }

#define D_CHECK_OUTPUT if (!isfinite((*reinterpret_cast<device float *>(r.raw + 0 * uint(n) + uint(i))))) d_flag(flags,1056); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 1 * uint(n) + uint(i))))) d_flag(flags,1057); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 2 * uint(n) + uint(i))))) d_flag(flags,1058); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 3 * uint(n) + uint(i))))) d_flag(flags,1059); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 4 * uint(n) + uint(i))))) d_flag(flags,1060); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 5 * uint(n) + uint(i))))) d_flag(flags,1061); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 6 * uint(n) + uint(i))))) d_flag(flags,1062); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 7 * uint(n) + uint(i))))) d_flag(flags,1063); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 8 * uint(n) + uint(i))))) d_flag(flags,1064); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 9 * uint(n) + uint(i))))) d_flag(flags,1065); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 10 * uint(n) + uint(i))))) d_flag(flags,1066); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 11 * uint(n) + uint(i))))) d_flag(flags,1067); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))))) d_flag(flags,1068); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 13 * uint(n) + uint(i))))) d_flag(flags,1069); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 14 * uint(n) + uint(i))))) d_flag(flags,1070); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 15 * uint(n) + uint(i))))) d_flag(flags,1071); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 16 * uint(n) + uint(i))))) d_flag(flags,1072); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 17 * uint(n) + uint(i))))) d_flag(flags,1073); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 18 * uint(n) + uint(i))))) d_flag(flags,1074); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 19 * uint(n) + uint(i))))) d_flag(flags,1075); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 20 * uint(n) + uint(i))))) d_flag(flags,1076); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 21 * uint(n) + uint(i))))) d_flag(flags,1077); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 22 * uint(n) + uint(i))))) d_flag(flags,1078); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i))))) d_flag(flags,1079); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 24 * uint(n) + uint(i))))) d_flag(flags,1080); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 25 * uint(n) + uint(i))))) d_flag(flags,1081); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 26 * uint(n) + uint(i))))) d_flag(flags,1082); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 27 * uint(n) + uint(i))))) d_flag(flags,1083); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 28 * uint(n) + uint(i))))) d_flag(flags,1084); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 29 * uint(n) + uint(i))))) d_flag(flags,1085); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 30 * uint(n) + uint(i))))) d_flag(flags,1086); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 31 * uint(n) + uint(i))))) d_flag(flags,1087); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 32 * uint(n) + uint(i))))) d_flag(flags,1088); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 33 * uint(n) + uint(i))))) d_flag(flags,1089); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 34 * uint(n) + uint(i))))) d_flag(flags,1090); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 35 * uint(n) + uint(i))))) d_flag(flags,1091); \
for (int k=0;k<RUC_NZS;++k) if (!isfinite((*reinterpret_cast<device float *>(r.raw + (123 + uint(k)) * uint(n) + uint(i))))) d_flag(flags,1092); \
for (int k=0;k<RUC_NZS;++k) if (!isfinite((*reinterpret_cast<device float *>(r.raw + (132 + uint(k)) * uint(n) + uint(i))))) d_flag(flags,1093); \
for (int k=0;k<RUC_NZS;++k) if (!isfinite((*reinterpret_cast<device float *>(r.raw + (141 + uint(k)) * uint(n) + uint(i))))) d_flag(flags,1094); \
for (int k=0;k<RUC_NZS;++k) if (!isfinite((*reinterpret_cast<device float *>(r.raw + (150 + uint(k)) * uint(n) + uint(i))))) d_flag(flags,1095); \
for (int k=0;k<RUC_NZS;++k) if (!isfinite((*reinterpret_cast<device float *>(r.raw + (159 + uint(k)) * uint(n) + uint(i))))) d_flag(flags,1096); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 114 * uint(n) + uint(i))))) d_flag(flags,1097); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 115 * uint(n) + uint(i))))) d_flag(flags,1098); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 116 * uint(n) + uint(i))))) d_flag(flags,1099); \
if (!isfinite((*reinterpret_cast<device float *>(r.raw + 117 * uint(n) + uint(i))))) d_flag(flags,1100);
static inline float d_min(float a,float b) {
    return (isnan(a) || isnan(b)) ? NAN : fmin(a,b);
}
static inline float d_max(float a,float b) {
    return (isnan(a) || isnan(b)) ? NAN : fmax(a,b);
}
static inline float d_npmin(float a,float b) {
    return isnan(a) ? a : (isnan(b) ? b : (a<b ? a:b));
}
static inline float d_npmax(float a,float b) {
    return isnan(a) ? a : (isnan(b) ? b : (a>b ? a:b));
}
static inline void d_flag(device uint* flags,int bit) {
    atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(flags) + bit/32, 1u<<(bit%32), memory_order_relaxed);
}
#define A(a,b) ruc_fadd_rn((a),(b))
#define M(a,b) ruc_fmul_rn((a),(b))
#define B(a,b) ruc_fsub_rn((a),(b))
#define Q(a,b) ruc_fdiv_rn((a),(b))
static inline float d_qsn(float temperature,device const float* tbq,
                                     device uint* flags,int first,thread bool* admitted) {
    if(!isfinite(temperature)) {
        d_flag(flags,first);
        if(admitted) *admitted=false;
    }
    float raw=A(Q(B(temperature,173.15f),0.05f),1.0f);
    if(!(fabs(raw)<2147483648.0f)) {
        d_flag(flags,first+1);
        if(admitted) *admitted=false;
        return tbq[0];
    }
    return ruc_qsn_lookup(temperature,tbq);
}

kernel void ruc_driver_prologue(
    constant RucDriverPrologueResources& r [[buffer(0)]],
    constant RucDriverPrologueParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int n = p.n, ktau = p.ktau, iswater = p.iswater, isice = p.isice;
    const int nv = p.nv, ns = p.ns;
    const float dt = p.dt, icealbedo = p.icealbedo, cn = p.cn;
    const int nlcat = p.nlcat, nscat = p.nscat;
    const int mosaic_lu = p.mosaic_lu, mosaic_soil = p.mosaic_soil;
    const int lakemodel = p.lakemodel, qvg_air = p.qvg_air, rdlai2d = p.rdlai2d;
    const float xice_threshold = p.xice_threshold;
    const uint forms = uint(p.forms);
    device const float* landusef = reinterpret_cast<device const float *>(
        r.raw + p.table_base + p.table_offset_landusef);
    device const float* soilctop = reinterpret_cast<device const float *>(
        r.raw + p.table_base + p.table_offset_soilctop);
    device uint* flags = reinterpret_cast<device uint *>(r.flags);
    device int* integer = reinterpret_cast<device int *>(r.raw + p.int_base);
    device uint* run = reinterpret_cast<device uint *>(r.raw + p.run_base);
    device const int* ifortbl = reinterpret_cast<device const int *>(r.raw + p.table_base + p.table_offset_ifortbl);
    device const float* z0tbl = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_z0tbl);
    device const float* lemitbl = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_lemitbl);
    device const float* pctbl = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_pctbl);
    device const float* laitbl = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_laitbl);
    device const float* bb = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_bb);
    device const float* drysmc = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_drysmc);
    device const float* hc = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_hc);
    device const float* maxsmc = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_maxsmc);
    device const float* refsmc = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_refsmc);
    device const float* satpsi = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_satpsi);
    device const float* satdk = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_satdk);
    device const float* wltsmc = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_wltsmc);
    device const float* qtz = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_qtz);
    device const float* tbq = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_tbq);


    int i = int(tid);
    if (i >= n) return;
    
    
    // xice_threshold: module_surface_driver.F:1365-1368, 0.5 or 0.02 by
    // fractional_seaice; the host hands the run's value to every ice test.
    if (p.frzfrac_mode != 0) (*reinterpret_cast<device float *>(r.raw + 43 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 38 * uint(n) + uint(i)))<=273.15f ? 1.0f:0.0f;
    (*reinterpret_cast<device float *>(r.raw + 42 * uint(n) + uint(i)))=A((*reinterpret_cast<device float *>(r.raw + 42 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + p.pending_base + uint(i))));
    bool component=(*reinterpret_cast<device float *>(r.raw + 51 * uint(n) + uint(i)))>=xice_threshold && (*reinterpret_cast<device float *>(r.raw + 51 * uint(n) + uint(i)))<=1.0f;
    if(component) {
        (*reinterpret_cast<device float *>(r.raw + 49 * uint(n) + uint(i)))=icealbedo;
        (*reinterpret_cast<device float *>(r.raw + 5 * uint(n) + uint(i)))=Q(B((*reinterpret_cast<device float *>(r.raw + 5 * uint(n) + uint(i))),M(B(1.0f,(*reinterpret_cast<device float *>(r.raw + 51 * uint(n) + uint(i)))),0.08f)),(*reinterpret_cast<device float *>(r.raw + 51 * uint(n) + uint(i))));
        (*reinterpret_cast<device float *>(r.raw + 6 * uint(n) + uint(i)))=Q(B((*reinterpret_cast<device float *>(r.raw + 6 * uint(n) + uint(i))),M(B(1.0f,(*reinterpret_cast<device float *>(r.raw + 51 * uint(n) + uint(i)))),0.98f)),(*reinterpret_cast<device float *>(r.raw + 51 * uint(n) + uint(i))));
        (*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 65 * uint(n) + uint(i)));
    }
    bool admitted=true;
    D_ADMIT
    int veg=integer[i];
    int soil=integer[n+i];
    if(veg<1 || veg>nv) { d_flag(flags,65); admitted=false; veg=1; }
    if(soil<1 || soil>ns) { d_flag(flags,65+1); admitted=false; soil=1; }
    integer[i]=veg; integer[n+i]=soil;
    if(ktau==1) {
        for(int k=0;k<RUC_NZS;++k) (*reinterpret_cast<device float *>(r.raw + (159 + uint(k)) * uint(n) + uint(i)))=0.0f;
        float blended=M(0.5f,A((*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + (141 + uint(0)) * uint(n) + uint(i)))));
        // Snow by lineage (gpuwm.core.ruc_tier RUC_SNOW_FORMS).  wrf_45, the
        // operational RAP/HRRR branch's LSMRUC:459-465: snow water with no
        // cover starts at min(1, snow/32), written as the exact snow*2**-5,
        // and the inside-snow repair blends above 32 mm of snow water.
        // GPUWM_SNOW_WRF461: WRF v4.6.1 blends wherever snowc > 0.
        bool inside_snow;
        if((forms&2u)!=0u) {
        inside_snow=(*reinterpret_cast<device float *>(r.raw + 2 * uint(n) + uint(i)))>0.0f;
        } else {
        if((*reinterpret_cast<device float *>(r.raw + 0 * uint(n) + uint(i)))>0.0f && (*reinterpret_cast<device float *>(r.raw + 2 * uint(n) + uint(i)))<=0.0f) (*reinterpret_cast<device float *>(r.raw + 2 * uint(n) + uint(i)))=d_min(1.0f,M((*reinterpret_cast<device float *>(r.raw + 0 * uint(n) + uint(i))),0.03125f));
        inside_snow=(*reinterpret_cast<device float *>(r.raw + 0 * uint(n) + uint(i)))>32.0f;
        }
        if((*reinterpret_cast<device float *>(r.raw + 29 * uint(n) + uint(i)))<170.0f || (*reinterpret_cast<device float *>(r.raw + 29 * uint(n) + uint(i)))>400.0f)
            (*reinterpret_cast<device float *>(r.raw + 29 * uint(n) + uint(i)))=inside_snow ? blended:(*reinterpret_cast<device float *>(r.raw + (141 + uint(0)) * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 30 * uint(n) + uint(i)))=B(blended,273.15f);
        (*reinterpret_cast<device float *>(r.raw + 27 * uint(n) + uint(i)))=Q(d_qsn((*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))),tbq,flags,80,&admitted),M((*reinterpret_cast<device float *>(r.raw + 37 * uint(n) + uint(i))),1.0e-2f));
        // QVG/QCG cold start by lineage (gpuwm.core.ruc_tier
        // RUC_QVG_COLD_START_FORMS).  qvg_air: the operational RAP/HRRR branch's
        // LSMRUC:479-483, the lowest-level vapour with no ground condensate.
        // Otherwise public WRF v4.6.1 LSMRUC:505-514, saturation at the
        // skin times moisture availability, condensate from the air.
        if(qvg_air) {
            if((*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i)))<=0.0f || (*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i)))>0.1f) { (*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 39 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 24 * uint(n) + uint(i)))=0.0f; }
        } else {
            if((*reinterpret_cast<device float *>(r.raw + 24 * uint(n) + uint(i)))<0.0f || (*reinterpret_cast<device float *>(r.raw + 24 * uint(n) + uint(i)))>0.1f) (*reinterpret_cast<device float *>(r.raw + 24 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 40 * uint(n) + uint(i)));
            if((*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i)))<=0.0f || (*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i)))>0.1f) (*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i)))=M((*reinterpret_cast<device float *>(r.raw + 27 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 8 * uint(n) + uint(i))));
        }
        (*reinterpret_cast<device float *>(r.raw + 26 * uint(n) + uint(i)))=Q((*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i))),A(1.0f,(*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i)))));
        (*reinterpret_cast<device float *>(r.raw + 22 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 35 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 34 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 25 * uint(n) + uint(i)))=0.0f;
        (*reinterpret_cast<device float *>(r.raw + 17 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 18 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 19 * uint(n) + uint(i)))=0.0f;
        (*reinterpret_cast<device float *>(r.raw + 33 * uint(n) + uint(i)))=-1000.0f; (*reinterpret_cast<device float *>(r.raw + 28 * uint(n) + uint(i)))=1.0f;
    }
    (*reinterpret_cast<device float *>(r.raw + 78 * uint(n) + uint(i)))=M((*reinterpret_cast<device float *>(r.raw + 37 * uint(n) + uint(i))),1.0e-5f);
    (*reinterpret_cast<device float *>(r.raw + 79 * uint(n) + uint(i)))=M((*reinterpret_cast<device float *>(r.raw + 36 * uint(n) + uint(i))),0.5f);
    float resolved_liquid=M((*reinterpret_cast<device float *>(r.raw + 56 * uint(n) + uint(i))),B(1.0f,(*reinterpret_cast<device float *>(r.raw + 43 * uint(n) + uint(i)))));
    float resolved_frozen=M((*reinterpret_cast<device float *>(r.raw + 56 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 43 * uint(n) + uint(i))));
    float convective=B((*reinterpret_cast<device float *>(r.raw + 42 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 56 * uint(n) + uint(i))));
    bool cold=(*reinterpret_cast<device float *>(r.raw + 38 * uint(n) + uint(i)))<273.0f;
    bool mixed_cold=(*reinterpret_cast<device float *>(r.raw + 43 * uint(n) + uint(i)))>0.0f && cold;
    float convective_liquid=mixed_cold ? d_max(0.0f,M(convective,B(1.0f,(*reinterpret_cast<device float *>(r.raw + 43 * uint(n) + uint(i)))))) : (cold ? 0.0f:d_max(0.0f,convective));
    float convective_frozen=mixed_cold ? d_max(0.0f,M(convective,(*reinterpret_cast<device float *>(r.raw + 43 * uint(n) + uint(i))))) : (cold ? d_max(0.0f,convective):0.0f);
    (*reinterpret_cast<device float *>(r.raw + 80 * uint(n) + uint(i)))=M(Q(A(resolved_liquid,convective_liquid),dt),1.0e-3f);
    (*reinterpret_cast<device float *>(r.raw + 81 * uint(n) + uint(i)))=M(Q(A(resolved_frozen,convective_frozen),dt),1.0e-3f);
    float frozen_total=A(resolved_frozen,convective_frozen);
    bool falling=frozen_total>0.0f;
    float denominator=falling ? frozen_total:1.0f;
    (*reinterpret_cast<device float *>(r.raw + 82 * uint(n) + uint(i)))=falling ? d_min(1.0f,d_max(0.0f,Q((*reinterpret_cast<device float *>(r.raw + 57 * uint(n) + uint(i))),denominator))):0.0f;
    (*reinterpret_cast<device float *>(r.raw + 83 * uint(n) + uint(i)))=falling ? d_min(1.0f,d_max(0.0f,Q((*reinterpret_cast<device float *>(r.raw + 58 * uint(n) + uint(i))),denominator))):0.0f;
    (*reinterpret_cast<device float *>(r.raw + 84 * uint(n) + uint(i)))=falling ? d_min(1.0f,d_max(0.0f,Q(B(B(resolved_frozen,(*reinterpret_cast<device float *>(r.raw + 57 * uint(n) + uint(i)))),(*reinterpret_cast<device float *>(r.raw + 58 * uint(n) + uint(i)))),denominator))):0.0f;
    (*reinterpret_cast<device float *>(r.raw + 85 * uint(n) + uint(i)))=falling ? d_min(1.0f,d_max(0.0f,Q(convective_frozen,denominator))):0.0f;
    (*reinterpret_cast<device float *>(r.raw + 34 * uint(n) + uint(i)))=M(M((*reinterpret_cast<device float *>(r.raw + 81 * uint(n) + uint(i))),dt),1000.0f);
    (*reinterpret_cast<device float *>(r.raw + 86 * uint(n) + uint(i)))=Q(Q((*reinterpret_cast<device float *>(r.raw + 47 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 41 * uint(n) + uint(i)))),(*reinterpret_cast<device float *>(r.raw + 8 * uint(n) + uint(i))));
    (*reinterpret_cast<device float *>(r.raw + 87 * uint(n) + uint(i)))=Q(Q((*reinterpret_cast<device float *>(r.raw + 48 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 41 * uint(n) + uint(i)))),M(1004.5f,A(1.0f,M(0.84f,(*reinterpret_cast<device float *>(r.raw + 39 * uint(n) + uint(i)))))));
    (*reinterpret_cast<device float *>(r.raw + 88 * uint(n) + uint(i)))=M((*reinterpret_cast<device float *>(r.raw + 0 * uint(n) + uint(i))),1.0e-3f); (*reinterpret_cast<device float *>(r.raw + 89 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 1 * uint(n) + uint(i)));
    (*reinterpret_cast<device float *>(r.raw + 90 * uint(n) + uint(i)))=M((*reinterpret_cast<device float *>(r.raw + 3 * uint(n) + uint(i))),1.0e-3f); (*reinterpret_cast<device float *>(r.raw + 91 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 2 * uint(n) + uint(i)));
    (*reinterpret_cast<device float *>(r.raw + 92 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 33 * uint(n) + uint(i)));
    (*reinterpret_cast<device float *>(r.raw + 93 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 0 * uint(n) + uint(i)))>0.0f && (*reinterpret_cast<device float *>(r.raw + 1 * uint(n) + uint(i)))>0.0f ? Q((*reinterpret_cast<device float *>(r.raw + 0 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 1 * uint(n) + uint(i)))):300.0f;
    // The driver version of SOILVEGIN.  The old leaf's fmin/fmax omit NaN
    // propagation.  These operands have been admitted, but derived NaNs must
    // still match the driver's CuPy minimum and maximum.
    int forest=ifortbl[veg-1];
    float range=B((*reinterpret_cast<device float *>(r.raw + 54 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 53 * uint(n) + uint(i))));
    float ratio=Q(B((*reinterpret_cast<device float *>(r.raw + 55 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 53 * uint(n) + uint(i)))),d_max(1.0f,range));
    float bounded=d_max(0.0f,d_min(1.0f,ratio));
    float factor=range<1.0f ? 1.0f:B(1.0f,bounded);
    float scaled=M(0.8f,laitbl[veg-1]), delta=0.0f;
    if(forest==1) delta=d_min(0.2f,scaled);
    if(forest==2 || forest==7) delta=d_min(0.5f,scaled);
    if(forest==3) delta=d_min(0.45f,scaled);
    if(forest==4) delta=d_min(0.75f,scaled);
    if(forest==5) delta=d_min(0.86f,scaled);
    float incoming_znt=(*reinterpret_cast<device float *>(r.raw + 11 * uint(n) + uint(i)));
    // module_sf_ruclsm.F:7075 ``if(.not.rdlai2d) LAI = LAItoday(IVGTYP)``:
    // under rdlai2d the prescribed 2-D LAI (the monthly field) stays.
    if(!rdlai2d) (*reinterpret_cast<device float *>(r.raw + 7 * uint(n) + uint(i)))=veg==iswater ? laitbl[veg-1]:B(laitbl[veg-1],M(delta,factor));
    if(veg!=iswater) (*reinterpret_cast<device float *>(r.raw + 11 * uint(n) + uint(i)))=forest==7 ? B(z0tbl[veg-1],M(0.125f,factor)):z0tbl[veg-1];
    (*reinterpret_cast<device float *>(r.raw + 94 * uint(n) + uint(i)))=lemitbl[veg-1]; (*reinterpret_cast<device float *>(r.raw + 95 * uint(n) + uint(i)))=pctbl[veg-1];
    (*reinterpret_cast<device float *>(r.raw + 96 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 97 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 98 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 99 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 100 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 101 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 103 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 104 * uint(n) + uint(i)))=0.0f;
    if(soil!=14) {
        (*reinterpret_cast<device float *>(r.raw + 96 * uint(n) + uint(i)))=qtz[soil-1]; (*reinterpret_cast<device float *>(r.raw + 97 * uint(n) + uint(i)))=M(hc[soil-1],1.0e6f);
        (*reinterpret_cast<device float *>(r.raw + 98 * uint(n) + uint(i)))=bb[soil-1]; (*reinterpret_cast<device float *>(r.raw + 99 * uint(n) + uint(i)))=B(maxsmc[soil-1],drysmc[soil-1]);
        (*reinterpret_cast<device float *>(r.raw + 100 * uint(n) + uint(i)))=satdk[soil-1]; (*reinterpret_cast<device float *>(r.raw + 101 * uint(n) + uint(i)))=-satpsi[soil-1];
        (*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i)))=drysmc[soil-1]; (*reinterpret_cast<device float *>(r.raw + 103 * uint(n) + uint(i)))=refsmc[soil-1]; (*reinterpret_cast<device float *>(r.raw + 104 * uint(n) + uint(i)))=wltsmc[soil-1];
    }
    ruc_mosaic_parameters(i,n,nlcat,nscat,mosaic_lu,mosaic_soil,
        landusef,soilctop,soil,iswater,rdlai2d!=0,factor,incoming_znt,
        ifortbl,z0tbl,lemitbl,pctbl,laitbl,bb,drysmc,hc,maxsmc,
        refsmc,satpsi,satdk,wltsmc,qtz,
        (*reinterpret_cast<device float *>(r.raw + 94 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 95 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 11 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 7 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 96 * uint(n) + uint(i))),
        (*reinterpret_cast<device float *>(r.raw + 97 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 98 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 99 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 100 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 101 * uint(n) + uint(i))),
        (*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 103 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 104 * uint(n) + uint(i))));
    bool forested=forest>2;
    (*reinterpret_cast<device float *>(r.raw + 105 * uint(n) + uint(i)))=forested ? 2.0f:0.85f;
    integer[3*n+i]=4;
    for(int k=1;k<RUC_NZS;++k) if(ruc_soil_layer_depth[k]>=(forested ? 0.4f:1.1f)) {
        integer[3*n+i]=k+1; break;
    }
    bool lake=lakemodel==1 && (*reinterpret_cast<device float *>(r.raw + 59 * uint(n) + uint(i)))==1.0f;
    bool water=B((*reinterpret_cast<device float *>(r.raw + 50 * uint(n) + uint(i))),1.5f)>=0.0f && !lake;
    bool land=!(B((*reinterpret_cast<device float *>(r.raw + 50 * uint(n) + uint(i))),1.5f)>=0.0f || lake);
    bool ice=land && (*reinterpret_cast<device float *>(r.raw + 51 * uint(n) + uint(i)))>=xice_threshold;
    (*reinterpret_cast<device float *>(r.raw + 222 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 51 * uint(n) + uint(i)))>=xice_threshold ? 1.0f:0.0f;
    integer[2*n+i]=ice ? isice:veg; integer[4*n+i]=1;
    if(water) {
        (*reinterpret_cast<device float *>(r.raw + 31 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 32 * uint(n) + uint(i)))=1.0f; (*reinterpret_cast<device float *>(r.raw + 0 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 1 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 2 * uint(n) + uint(i)))=0.0f;
        (*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i)))=Q(d_qsn((*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))),tbq,flags,82,&admitted),M((*reinterpret_cast<device float *>(r.raw + 37 * uint(n) + uint(i))),1.0e-2f));
        (*reinterpret_cast<device float *>(r.raw + 26 * uint(n) + uint(i)))=Q((*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i))),A(1.0f,(*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i))))); (*reinterpret_cast<device float *>(r.raw + 28 * uint(n) + uint(i)))=1.0f;
        for(int k=0;k<RUC_NZS;++k) {
            (*reinterpret_cast<device float *>(r.raw + (123 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + (132 + uint(k)) * uint(n) + uint(i)))=1.0f; (*reinterpret_cast<device float *>(r.raw + (141 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i)));
        }
    }
    if(ice) {
        (*reinterpret_cast<device float *>(r.raw + 11 * uint(n) + uint(i)))=0.011f; (*reinterpret_cast<device float *>(r.raw + 4 * uint(n) + uint(i)))=0.75f; (*reinterpret_cast<device float *>(r.raw + 99 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 103 * uint(n) + uint(i)))=1.0f;
        (*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 104 * uint(n) + uint(i)))=0.0f; (*reinterpret_cast<device float *>(r.raw + 94 * uint(n) + uint(i)))=0.98f;
        (*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 27 * uint(n) + uint(i)))=Q(d_qsn((*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))),tbq,flags,84,&admitted),M((*reinterpret_cast<device float *>(r.raw + 37 * uint(n) + uint(i))),1.0e-2f));
        (*reinterpret_cast<device float *>(r.raw + 26 * uint(n) + uint(i)))=Q((*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i))),A(1.0f,(*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i)))));
        for(int k=0;k<RUC_NZS;++k) {
            (*reinterpret_cast<device float *>(r.raw + (123 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + (150 + uint(k)) * uint(n) + uint(i)))=1.0f;
            (*reinterpret_cast<device float *>(r.raw + (132 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + (159 + uint(k)) * uint(n) + uint(i)))=0.0f;
            (*reinterpret_cast<device float *>(r.raw + (141 + uint(k)) * uint(n) + uint(i)))=d_min(271.4f,(*reinterpret_cast<device float *>(r.raw + (141 + uint(k)) * uint(n) + uint(i))));
        }
    }
    for(int k=0;k<RUC_NZS;++k) {
        (*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i)))=d_min(d_max(0.0f,B((*reinterpret_cast<device float *>(r.raw + (123 + uint(k)) * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i))))),(*reinterpret_cast<device float *>(r.raw + 99 * uint(n) + uint(i))));
        (*reinterpret_cast<device float *>(r.raw + (177 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + (141 + uint(k)) * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + (204 + uint(k)) * uint(n) + uint(i)))=d_min(d_max(0.0f,B((*reinterpret_cast<device float *>(r.raw + (132 + uint(k)) * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i))))),(*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i))));
        (*reinterpret_cast<device float *>(r.raw + (213 + uint(k)) * uint(n) + uint(i)))=Q(B((*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + (204 + uint(k)) * uint(n) + uint(i)))),0.9f);
        (*reinterpret_cast<device float *>(r.raw + (186 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + (150 + uint(k)) * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + (195 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + (159 + uint(k)) * uint(n) + uint(i)));
    }
    (*reinterpret_cast<device float *>(r.raw + 106 * uint(n) + uint(i)))=land ? d_max(0.00001f,d_min(1.0f,Q((*reinterpret_cast<device float *>(r.raw + (168 + uint(0)) * uint(n) + uint(i))),B((*reinterpret_cast<device float *>(r.raw + 103 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i))))))):0.0f;
    (*reinterpret_cast<device float *>(r.raw + 107 * uint(n) + uint(i)))=5.0e-4f; (*reinterpret_cast<device float *>(r.raw + 108 * uint(n) + uint(i)))=cn;
    (*reinterpret_cast<device float *>(r.raw + 109 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 110 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 111 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 112 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 113 * uint(n) + uint(i)))=0.0f;
    (*reinterpret_cast<device float *>(r.raw + 114 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 115 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 116 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 117 * uint(n) + uint(i)))=0.0f;
    run[i]=uint(land && admitted);
    atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(flags) + 36, land && !ice ? 1u:0u, memory_order_relaxed);
    atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(flags) + 37, water ? 1u:0u, memory_order_relaxed);
    atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(flags) + 38, lake ? 1u:0u, memory_order_relaxed);
    atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(flags) + 39, ice ? 1u:0u, memory_order_relaxed);
}

// NumPy float32 arithmetic retains an operand NaN's payload and quiets it.
// Invalid arithmetic without NaN operands produces its negative quiet NaN.
// CUDA arithmetic canonicalizes both, so retain these words explicitly in
// the host diagnostic transcription.  Finite arithmetic remains one IEEE op.
static inline float d_np_result(float a,float b,float result) {
    if(isnan(a)) return as_type<float>(as_type<int>(a) | 0x00400000);
    if(isnan(b)) return as_type<float>(as_type<int>(b) | 0x00400000);
    return isnan(result) ? as_type<float>(0xffc00000):result;
}
static inline float d_npadd(float a,float b) { return d_np_result(a,b,ruc_fadd_rn(a,b)); }
static inline float d_npsub(float a,float b) { return d_np_result(a,b,ruc_fsub_rn(a,b)); }
static inline float d_npmul(float a,float b) { return d_np_result(a,b,ruc_fmul_rn(a,b)); }
static inline float d_npdiv(float a,float b) { return d_np_result(a,b,ruc_fdiv_rn(a,b)); }
#define P_A(a,b) d_npadd((a),(b))
#define P_B(a,b) d_npsub((a),(b))
#define P_M(a,b) d_npmul((a),(b))
#define P_Q(a,b) d_npdiv((a),(b))
static inline float d_saturation(float pressure,float temperature) {
    float x=d_npmax(-80.0f,P_B(temperature,273.16f));
    float liquid=-.321582393e-13f;
    liquid=P_A(.379534310e-11f,P_M(x,liquid));
    liquid=P_A(.702620698e-8f,P_M(x,liquid));
    liquid=P_A(.203154182e-5f,P_M(x,liquid));
    liquid=P_A(.299291081e-3f,P_M(x,liquid));
    liquid=P_A(.264224321e-1f,P_M(x,liquid));
    liquid=P_A(.143177157e01f,P_M(x,liquid));
    liquid=P_A(.444606896e02f,P_M(x,liquid));
    liquid=P_A(.611583699e03f,P_M(x,liquid));
    float ice=.161444444e-12f;
    ice=P_A(.105785160e-9f,P_M(x,ice));
    ice=P_A(.307839583e-7f,P_M(x,ice));
    ice=P_A(.521693933e-5f,P_M(x,ice));
    ice=P_A(.565392987e-3f,P_M(x,ice));
    ice=P_A(.402737184e-1f,P_M(x,ice));
    ice=P_A(.184672631e01f,P_M(x,ice));
    ice=P_A(.499320233e02f,P_M(x,ice));
    ice=P_A(.609868993e03f,P_M(x,ice));
    return P_B(temperature,273.15f)<=0.0f ? P_Q(P_M(.622f,ice),P_B(pressure,ice)) : P_Q(P_M(.622f,liquid),P_B(pressure,liquid));
}
kernel void ruc_driver_epilogue(
    constant RucDriverEpilogueResources& r [[buffer(0)]],
    constant RucDriverEpilogueParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int n = p.n;
    const float dt = p.dt;
    device uint* flags = reinterpret_cast<device uint *>(r.flags);
    device uint* run = reinterpret_cast<device uint *>(r.raw + 665 * uint(n));
    device const float* tbq = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_tbq);
    device const float* lemitbl = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_lemitbl);
    device const float* zshalf = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_zshalf);
    device const int* integer = reinterpret_cast<device const int *>(r.raw + 660 * uint(n));
    device const float* landusef = reinterpret_cast<device const float *>(
        r.raw + p.table_base + p.table_offset_landusef);
    const int nlcat = p.nlcat, mosaic_lu = p.mosaic_lu;
    const int crop = p.crop, natural = p.natural;
    const int irrigation = p.irrigation, log_profile = p.log_profile;
    const float xice_threshold = p.xice_threshold;


    int i = int(tid);
    if (i >= n) return;
    
    
    float qsat=Q(d_qsn((*reinterpret_cast<device float *>(r.raw + 38 * uint(n) + uint(i))),tbq,flags,1024,nullptr),M((*reinterpret_cast<device float *>(r.raw + 37 * uint(n) + uint(i))),1.0e-2f));
    if(run[i]) {
        for(int k=0;k<RUC_NZS;++k) {
            (*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 243 * uint(n) + uint(k) * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + (177 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 234 * uint(n) + uint(k) * uint(n) + uint(i)));
            (*reinterpret_cast<device float *>(r.raw + (186 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 311 * uint(n) + uint(k) * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + (195 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 320 * uint(n) + uint(k) * uint(n) + uint(i)));
            (*reinterpret_cast<device float *>(r.raw + (204 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 338 * uint(n) + uint(k) * uint(n) + uint(i)));
        }
        (*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 309 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 29 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 282 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 30 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 277 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 25 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 274 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 272 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 27 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 271 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 24 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 347 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 22 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 350 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 35 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 351 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 21 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 352 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 5 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 635 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 11 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 306 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 94 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 289 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 88 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 300 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 89 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 301 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 91 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 302 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 92 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 278 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 90 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 307 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 106 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 308 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 115 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 348 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 116 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 364 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 117 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 365 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 114 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 366 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 14 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 356 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 15 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 357 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 13 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 358 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 111 * uint(n) + uint(i))) = (*reinterpret_cast<device float *>(r.raw + 359 * uint(n) + uint(i)));
        // Irrigation after SFCTMP and before soil diagnostics, by WRF
        // lineage (gpuwm.core.ruc_mosaic.IRRIGATION_FORMS; the host twin is
        // ruc_mosaic.irrigate).  irrigation==1: WRF v4.6.1 LSMRUC:985-1009
        // under mosaic_lu, a per-step relaxation.  irrigation==0: WRF v4.5.2
        // LSMRUC:970-999, a crop-fraction-scaled hard floor with no mosaic
        // gate.  It reads the LANDUSEF fractions when the run carries them
        // (nlcat>0); a run without them (mosaic_lu=0) gives the dominant
        // category the whole cell, which is WRF's arithmetic on a one-hot
        // LANDUSEF.  Its gates read the leaf area index SOILVEGIN left in
        // the scratch, table or 2-D, and the dominant category.
        if(irrigation==1) {
            if(mosaic_lu) {
                float croparea=landusef[(crop-1)*n+i];
                float naturalarea=landusef[(natural-1)*n+i];
                float factor=d_max(0.0f,d_min(1.0f,Q(B((*reinterpret_cast<device float *>(r.raw + 55 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 53 * uint(n) + uint(i)))),d_max(1.0f,B((*reinterpret_cast<device float *>(r.raw + 54 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 53 * uint(n) + uint(i))))))));
                if((croparea>0.0f || naturalarea>0.0f) && factor>0.75f) {
                    float cropsm=B(M(1.1f,(*reinterpret_cast<device float *>(r.raw + 104 * uint(n) + uint(i)))),(*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i))));
                    float cropfr=d_min(1.0f,A(croparea,M(0.4f,naturalarea)));
                    for(int k=0;k<integer[3*n+i];++k) {
                        float newsm=A(M(cropsm,cropfr),M(B(1.0f,cropfr),(*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i)))));
                        if((*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i)))<newsm) (*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i)))=newsm;
                    }
                }
            }
        } else {
            float croparea=nlcat>0 ? landusef[(crop-1)*n+i]:(integer[i]==crop ? 1.0f:0.0f);
            float naturalarea=nlcat>0 ? landusef[(natural-1)*n+i]:(integer[i]==natural ? 1.0f:0.0f);
            if(croparea>0.0f && (*reinterpret_cast<device float *>(r.raw + 7 * uint(n) + uint(i)))>1.1f) {
                float cropsm=B(M(1.1f,(*reinterpret_cast<device float *>(r.raw + 104 * uint(n) + uint(i)))),(*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i))));
                float floor=M(cropsm,croparea);
                for(int k=0;k<integer[3*n+i];++k) {
                    if((*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i)))<floor) (*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i)))=floor;
                }
            } else if(integer[i]==natural && (*reinterpret_cast<device float *>(r.raw + 7 * uint(n) + uint(i)))>0.7f) {
                float cropsm=B(M(1.2f,(*reinterpret_cast<device float *>(r.raw + 104 * uint(n) + uint(i)))),(*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i))));
                float floor=M(M(cropsm,naturalarea),0.4f);
                for(int k=0;k<integer[3*n+i];++k) {
                    if((*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i)))<floor) (*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i)))=floor;
                }
            }
        }
        float available=0.0f,maximum=0.0f;
        for(int k=0;k<RUC_NZS-1;++k) {
            float thickness=B(zshalf[k+1],zshalf[k]);
            available=A(available,M(A((*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i)))),thickness));
            maximum=A(maximum,M(A((*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 99 * uint(n) + uint(i)))),thickness));
        }
        float bottom=B(ruc_soil_layer_depth[RUC_NZS-1],zshalf[RUC_NZS-1]);
        available=A(available,M(A((*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + (168 + uint(RUC_NZS-1)) * uint(n) + uint(i)))),bottom));
        maximum=A(maximum,M(A((*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 99 * uint(n) + uint(i)))),bottom));
        float sr=M(M((*reinterpret_cast<device float *>(r.raw + 116 * uint(n) + uint(i))),dt),1000.0f),ur=M(M((*reinterpret_cast<device float *>(r.raw + 117 * uint(n) + uint(i))),dt),1000.0f);
        (*reinterpret_cast<device float *>(r.raw + 17 * uint(n) + uint(i)))=A((*reinterpret_cast<device float *>(r.raw + 17 * uint(n) + uint(i))),sr); (*reinterpret_cast<device float *>(r.raw + 18 * uint(n) + uint(i)))=A((*reinterpret_cast<device float *>(r.raw + 18 * uint(n) + uint(i))),ur);
        (*reinterpret_cast<device float *>(r.raw + 19 * uint(n) + uint(i)))=A((*reinterpret_cast<device float *>(r.raw + 19 * uint(n) + uint(i))),sr); (*reinterpret_cast<device float *>(r.raw + 31 * uint(n) + uint(i)))=M(available,1000.0f);
        (*reinterpret_cast<device float *>(r.raw + 32 * uint(n) + uint(i)))=M(maximum,1000.0f);
        for(int k=0;k<RUC_NZS;++k) {
            (*reinterpret_cast<device float *>(r.raw + (123 + uint(k)) * uint(n) + uint(i)))=A((*reinterpret_cast<device float *>(r.raw + (168 + uint(k)) * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i))));
            (*reinterpret_cast<device float *>(r.raw + (132 + uint(k)) * uint(n) + uint(i)))=d_min(A((*reinterpret_cast<device float *>(r.raw + (204 + uint(k)) * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 102 * uint(n) + uint(i)))),(*reinterpret_cast<device float *>(r.raw + (123 + uint(k)) * uint(n) + uint(i))));
            (*reinterpret_cast<device float *>(r.raw + (141 + uint(k)) * uint(n) + uint(i)))=k==RUC_NZS-1 ? (*reinterpret_cast<device float *>(r.raw + 52 * uint(n) + uint(i))):(*reinterpret_cast<device float *>(r.raw + (177 + uint(k)) * uint(n) + uint(i)));
            (*reinterpret_cast<device float *>(r.raw + (150 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + (186 + uint(k)) * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + (159 + uint(k)) * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + (195 + uint(k)) * uint(n) + uint(i)));
        }
        (*reinterpret_cast<device float *>(r.raw + 10 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 11 * uint(n) + uint(i))); (*reinterpret_cast<device float *>(r.raw + 9 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 87 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 26 * uint(n) + uint(i)))=Q((*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i))),A(1.0f,(*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i)))));
        (*reinterpret_cast<device float *>(r.raw + 28 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 39 * uint(n) + uint(i)))>=M(qsat,0.95f) && (*reinterpret_cast<device float *>(r.raw + 39 * uint(n) + uint(i)))<(*reinterpret_cast<device float *>(r.raw + 23 * uint(n) + uint(i))) ? 0.0f:1.0f;
        (*reinterpret_cast<device float *>(r.raw + 6 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 0 * uint(n) + uint(i)))==0.0f ? lemitbl[integer[i]-1]:(*reinterpret_cast<device float *>(r.raw + 94 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 0 * uint(n) + uint(i)))=M((*reinterpret_cast<device float *>(r.raw + 88 * uint(n) + uint(i))),1000.0f); (*reinterpret_cast<device float *>(r.raw + 1 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 89 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 3 * uint(n) + uint(i)))=M((*reinterpret_cast<device float *>(r.raw + 90 * uint(n) + uint(i))),1000.0f); (*reinterpret_cast<device float *>(r.raw + 8 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 106 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 16 * uint(n) + uint(i)))=A((*reinterpret_cast<device float *>(r.raw + 16 * uint(n) + uint(i))),M((*reinterpret_cast<device float *>(r.raw + 14 * uint(n) + uint(i))),dt)); (*reinterpret_cast<device float *>(r.raw + 20 * uint(n) + uint(i)))=M(-1.0f,(*reinterpret_cast<device float *>(r.raw + 111 * uint(n) + uint(i))));
        (*reinterpret_cast<device float *>(r.raw + 2 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 91 * uint(n) + uint(i)))>0.0f && (*reinterpret_cast<device float *>(r.raw + 51 * uint(n) + uint(i)))>=xice_threshold ? M((*reinterpret_cast<device float *>(r.raw + 91 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 51 * uint(n) + uint(i)))):(*reinterpret_cast<device float *>(r.raw + 91 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 33 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 92 * uint(n) + uint(i)));
        (*reinterpret_cast<device float *>(r.raw + 16 * uint(n) + uint(i)))=A((*reinterpret_cast<device float *>(r.raw + 16 * uint(n) + uint(i))),M((*reinterpret_cast<device float *>(r.raw + 14 * uint(n) + uint(i))),dt));
    }
    D_CHECK_OUTPUT
    float fraction=(*reinterpret_cast<device float *>(r.raw + 51 * uint(n) + uint(i)));
    bool component=fraction>=xice_threshold && fraction<=1.0f;
    if(component) (*reinterpret_cast<device float *>(r.raw + 5 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 5 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),0.08f)); if(component) (*reinterpret_cast<device float *>(r.raw + 6 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 6 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),0.98f));
    if(component) (*reinterpret_cast<device float *>(r.raw + 48 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 48 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),(*reinterpret_cast<device float *>(r.raw + 67 * uint(n) + uint(i))))); if(component) (*reinterpret_cast<device float *>(r.raw + 47 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 47 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),(*reinterpret_cast<device float *>(r.raw + 68 * uint(n) + uint(i)))));
    if(component) (*reinterpret_cast<device float *>(r.raw + 63 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 63 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),(*reinterpret_cast<device float *>(r.raw + 69 * uint(n) + uint(i))))); if(component) (*reinterpret_cast<device float *>(r.raw + 62 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 62 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),(*reinterpret_cast<device float *>(r.raw + 70 * uint(n) + uint(i)))));
    if(component) (*reinterpret_cast<device float *>(r.raw + 61 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 61 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),(*reinterpret_cast<device float *>(r.raw + 71 * uint(n) + uint(i))))); if(component) (*reinterpret_cast<device float *>(r.raw + 46 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 46 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),(*reinterpret_cast<device float *>(r.raw + 72 * uint(n) + uint(i)))));
    if(component) (*reinterpret_cast<device float *>(r.raw + 26 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 26 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),(*reinterpret_cast<device float *>(r.raw + 73 * uint(n) + uint(i))))); if(component) (*reinterpret_cast<device float *>(r.raw + 64 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 64 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),(*reinterpret_cast<device float *>(r.raw + 74 * uint(n) + uint(i)))));
    if(component) (*reinterpret_cast<device float *>(r.raw + 13 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 13 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),(*reinterpret_cast<device float *>(r.raw + 75 * uint(n) + uint(i))))); if(component) (*reinterpret_cast<device float *>(r.raw + 14 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 14 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),(*reinterpret_cast<device float *>(r.raw + 76 * uint(n) + uint(i))))); if(component) (*reinterpret_cast<device float *>(r.raw + 15 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 15 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),(*reinterpret_cast<device float *>(r.raw + 77 * uint(n) + uint(i)))));
    if(component) (*reinterpret_cast<device float *>(r.raw + 65 * uint(n) + uint(i)))=(*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i)));
    if(component) (*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i)))=A(M((*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))),fraction),M(B(1.0f,fraction),(*reinterpret_cast<device float *>(r.raw + 66 * uint(n) + uint(i)))));
    float cqs=Q((*reinterpret_cast<device float *>(r.raw + 47 * uint(n) + uint(i))),M((*reinterpret_cast<device float *>(r.raw + 8 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 41 * uint(n) + uint(i)))));
    (*reinterpret_cast<device float *>(r.raw + 46 * uint(n) + uint(i)))=Q((*reinterpret_cast<device float *>(r.raw + 48 * uint(n) + uint(i))),M((*reinterpret_cast<device float *>(r.raw + 63 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 41 * uint(n) + uint(i)))));
    float th2=(*reinterpret_cast<device float *>(r.raw + 61 * uint(n) + uint(i)))<1.0e-5f ? P_M((*reinterpret_cast<device float *>(r.raw + 38 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 121 * uint(n) + uint(i)))) : P_B(P_M((*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 121 * uint(n) + uint(i)))),P_Q((*reinterpret_cast<device float *>(r.raw + 13 * uint(n) + uint(i))),P_M(P_M((*reinterpret_cast<device float *>(r.raw + 41 * uint(n) + uint(i))),1004.5f),(*reinterpret_cast<device float *>(r.raw + 61 * uint(n) + uint(i))))));
    float t2=P_M(th2,(*reinterpret_cast<device float *>(r.raw + 122 * uint(n) + uint(i))));
    float lower=d_npmin((*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 38 * uint(n) + uint(i)))),upper=d_npmax((*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 38 * uint(n) + uint(i))));
    t2=d_npmin(upper,d_npmax(lower,t2));
    (*reinterpret_cast<device float *>(r.raw + 118 * uint(n) + uint(i)))=t2; (*reinterpret_cast<device float *>(r.raw + 119 * uint(n) + uint(i)))=P_M(t2,(*reinterpret_cast<device float *>(r.raw + 121 * uint(n) + uint(i))));
    float qlev=d_npmin(d_saturation((*reinterpret_cast<device float *>(r.raw + 37 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 38 * uint(n) + uint(i)))),(*reinterpret_cast<device float *>(r.raw + 39 * uint(n) + uint(i))));
    float prox=P_A(qlev,P_Q((*reinterpret_cast<device float *>(r.raw + 14 * uint(n) + uint(i))),P_M((*reinterpret_cast<device float *>(r.raw + 41 * uint(n) + uint(i))),cqs)));
    float qsfcmr=P_Q((*reinterpret_cast<device float *>(r.raw + 26 * uint(n) + uint(i))),P_B(1.0f,(*reinterpret_cast<device float *>(r.raw + 26 * uint(n) + uint(i)))));
    float q2=(*reinterpret_cast<device float *>(r.raw + 62 * uint(n) + uint(i)))<1.0e-5f ? qlev:P_B(prox,P_Q((*reinterpret_cast<device float *>(r.raw + 14 * uint(n) + uint(i))),P_M((*reinterpret_cast<device float *>(r.raw + 41 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 62 * uint(n) + uint(i))))));
    q2=d_npmin(d_npmax(qsfcmr,qlev),d_npmax(d_npmin(qsfcmr,qlev),q2));
    (*reinterpret_cast<device float *>(r.raw + 120 * uint(n) + uint(i)))=d_npmin(d_saturation((*reinterpret_cast<device float *>(r.raw + 60 * uint(n) + uint(i))),t2),q2);
    // ruc_2m_diagnostic = log_profile: the operational RAP/HRRR branch's
    // module_sf_sfcdiags_ruclsm.F:150-179 (gpuwm.core.ruc_tier
    // RUC_2M_DIAGNOSTIC_FORMS; host twin ruc_runtime._sfcdiags_ruclsm).
    // dz1 is half the lowest layer, LSMRUC's conflx.
    if(log_profile) {
        float dz1=(*reinterpret_cast<device float *>(r.raw + 79 * uint(n) + uint(i)));
        float dT=P_B((*reinterpret_cast<device float *>(r.raw + 38 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))));
        float dQ=P_B(qlev,qsfcmr);
        if(dT>0.0f) {
            float fh=d_npmin(d_npmax(P_B(1.0f,P_Q(dT,10.0f)),0.01f),1.0f);
            float fac=P_Q(gfk_log(P_Q(P_A(2.0f,0.05f),P_A(0.05f,fh))),
                          gfk_log(P_Q(P_A(dz1,0.05f),P_A(0.05f,fh))));
            float t2a=P_A((*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))),P_M(fac,P_B((*reinterpret_cast<device float *>(r.raw + 38 * uint(n) + uint(i))),(*reinterpret_cast<device float *>(r.raw + 12 * uint(n) + uint(i))))));
            (*reinterpret_cast<device float *>(r.raw + 118 * uint(n) + uint(i)))=t2a; (*reinterpret_cast<device float *>(r.raw + 119 * uint(n) + uint(i)))=P_M(t2a,(*reinterpret_cast<device float *>(r.raw + 121 * uint(n) + uint(i))));
        }
        if(dQ>0.0f) {
            float fh=d_npmin(d_npmax(P_B(1.0f,P_Q(dQ,0.003f)),0.01f),1.0f);
            float fac=P_Q(gfk_log(P_Q(P_A(2.0f,0.05f),P_A(0.05f,fh))),
                          gfk_log(P_Q(P_A(dz1,0.05f),P_A(0.05f,fh))));
            (*reinterpret_cast<device float *>(r.raw + 120 * uint(n) + uint(i)))=P_A(qsfcmr,P_M(fac,P_B(qlev,qsfcmr)));
        }
    }
}

// The fields are written only when no check anywhere in the call failed:
// the driver's own words and the fused sfctmp's bit words (sflags).
kernel void ruc_driver_commit(
    constant RucDriverCommitResources& r [[buffer(0)]],
    constant RucDriverCommitParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    uint i = tid;
    if (i >= uint(p.n)) return;
    for (uint word = 0; word < 36; ++word)
        if (reinterpret_cast<device const uint *>(r.flags)[word] != 0u) return;
    for (uint word = 0; word < uint(p.sf_flag_words); ++word)
        if (r.flags[p.sf_flag_base + word] != 0ul) return;
    r.raw[p.alive_base + i] = 1u;
}

