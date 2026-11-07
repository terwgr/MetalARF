#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_bl_mynn_adapter_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float mynn_result(float a,float b,float value) {
    uint x=as_type<uint>(a),y=as_type<uint>(b),r=as_type<uint>(value);
    if ((r&0x7fffffffu)<=0x7f800000u) return value;
    if ((x&0x7fffffffu)>0x7f800000u) return as_type<float>(x|0x00400000u);
    if ((y&0x7fffffffu)>0x7f800000u) return as_type<float>(y|0x00400000u);
    return as_type<float>(0xffc00000u);
}
static inline float mynn_add(float a,float b){return mynn_result(a,b,mpas_add(a,b));}
static inline float mynn_sub(float a,float b){return mynn_result(a,b,mpas_sub(a,b));}
static inline float mynn_mul(float a,float b){return mynn_result(a,b,mpas_mul(a,b));}
static inline float mynn_div(float a,float b){return mynn_result(a,b,cu_div(a,b));}
#define mpas_add mynn_add
#define mpas_sub mynn_sub
#define mpas_mul mynn_mul
#define mpas_div mynn_div

kernel void hexv842_mpas_bl_mynn_pre_f32(constant Hexv842MpasBlMynnPreF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnPreF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
int count=p.count;int f_qc=p.f_qc;int f_qi=p.f_qi;int f_qs=p.f_qs;device const float* qv=r.qv;device const float* qc=r.qc;device const float* qi=r.qi;device const float* qs=r.qs;device float* sqv=r.sqv;device float* sqc=r.sqc;device float* sqi=r.sqi;device float* sqs=r.sqs;
    const int n = gid;
    if (n >= count) return;
    const float denom = mpas_add(1.0f, qv[n]);
    sqv[n] = mpas_div(qv[n], denom);
    sqc[n] = f_qc ? mpas_div(qc[n], denom) : 0.0f;
    sqi[n] = f_qi ? mpas_div(qi[n], denom) : 0.0f;
    // Pinned MPAS deliberately uses snow in this denominator. When the
    // species is absent its output is left as provided by the caller.
    if (f_qs) sqs[n] = mpas_div(qs[n], mpas_add(1.0f, qs[n]));

}

kernel void hexv842_mpas_bl_mynn_pre_rp1_f32(constant Hexv842MpasBlMynnPreRp1F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnPreRp1F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
int count=p.count;int f_qc=p.f_qc;int f_qi=p.f_qi;int f_qs=p.f_qs;device const float* qv=r.qv;device const float* qc=r.qc;device const float* qi=r.qi;device const float* qs=r.qs;device float* sqv=r.sqv;device float* sqc=r.sqc;device float* sqi=r.sqi;device float* sqs=r.sqs;
    const int n = gid;
    if (n >= count) return;
    const float denom = mpas_add(1.0f, qv[n]);
    sqv[n] = mpas_div(qv[n], denom);
    sqc[n] = f_qc ? mpas_div(qc[n], denom) : 0.0f;
    sqi[n] = f_qi ? mpas_div(qi[n], denom) : 0.0f;
    if (f_qs) sqs[n] = mpas_div(qs[n], denom);

}

kernel void hexv842_mpas_bl_mynn_post_f32(constant Hexv842MpasBlMynnPostF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnPostF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
int count=p.count;int f_qc=p.f_qc;int f_qi=p.f_qi;int f_qs=p.f_qs;float delt=p.delt;device const float* qv=r.qv;device const float* qc=r.qc;device const float* qi=r.qi;device const float* qs=r.qs;device float* dqv=r.dqv;device float* dqc=r.dqc;device float* dqi=r.dqi;device float* dqs=r.dqs;
    const int n = gid;
    if (n >= count) return;
    // Preserve the original strict GNU operand destinations, including first
    // NaN selection: product first for vapor, factor first for condensates.
    // Unchanged native bl_mynn_post_run: 2cbc0, 2ccb7, 2cdb9, 2ce88;
    // witness receipts/whole-post-native-lowering-v94/original-post.txt.
    const float denom = mpas_add(qv[n], 1.0f);
    const float sqv = mpas_add(mpas_mul(dqv[n], delt), mpas_div(qv[n], denom));
    const float rqv = mpas_div(sqv, mpas_sub(1.0f, sqv));
    dqv[n] = mpas_div(mpas_sub(rqv, qv[n]), delt);
    // Preserve MPAS's literal 1+updated_specific_vapor condensate factor.
    const float factor = mpas_add(sqv, 1.0f);
    if (f_qc) {
        const float sqc = mpas_add(mpas_div(qc[n], denom), mpas_mul(dqc[n], delt));
        dqc[n] = mpas_div(mpas_sub(mpas_mul(factor, sqc), qc[n]), delt);
    }
    if (f_qi) {
        const float sqi = mpas_add(mpas_div(qi[n], denom), mpas_mul(dqi[n], delt));
        dqi[n] = mpas_div(mpas_sub(mpas_mul(factor, sqi), qi[n]), delt);
    }
    if (f_qs) {
        const float sqs = mpas_add(mpas_div(qs[n], denom), mpas_mul(dqs[n], delt));
        dqs[n] = mpas_div(mpas_sub(mpas_mul(factor, sqs), qs[n]), delt);
    }

}

kernel void hexv842_mpas_bl_mynn_post_rp1_f32(constant Hexv842MpasBlMynnPostRp1F32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnPostRp1F32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
int count=p.count;int f_qc=p.f_qc;int f_qi=p.f_qi;int f_qs=p.f_qs;float delt=p.delt;device const float* qv=r.qv;device const float* qc=r.qc;device const float* qi=r.qi;device const float* qs=r.qs;device float* dqv=r.dqv;device float* dqc=r.dqc;device float* dqi=r.dqi;device float* dqs=r.dqs;
    const int n = gid;
    if (n >= count) return;
    // Water vapor is the unchanged pristine loop in native correct mode: mynn_post's operand order.
    const float denom = mpas_add(qv[n], 1.0f);
    const float sqv = mpas_add(mpas_mul(dqv[n], delt), mpas_div(qv[n], denom));
    const float rest = mpas_sub(1.0f, sqv);
    const float rqv = mpas_div(sqv, rest);
    dqv[n] = mpas_div(mpas_sub(rqv, qv[n]), delt);
    if (f_qc) {
        const float sqc = mpas_add(mpas_div(qc[n], denom), mpas_mul(dqc[n], delt));
        dqc[n] = mpas_div(mpas_sub(mpas_div(sqc, rest), qc[n]), delt);
    }
    if (f_qi) {
        const float sqi = mpas_add(mpas_div(qi[n], denom), mpas_mul(dqi[n], delt));
        dqi[n] = mpas_div(mpas_sub(mpas_div(sqi, rest), qi[n]), delt);
    }
    if (f_qs) {
        const float sqs = mpas_add(mpas_div(qs[n], denom), mpas_mul(dqs[n], delt));
        dqs[n] = mpas_div(mpas_sub(mpas_div(sqs, rest), qs[n]), delt);
    }

}

kernel void hexv842_mpas_bl_mynn_radiation_f32(constant Hexv842MpasBlMynnRadiationF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnRadiationF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
int count=p.count;device const float* lw=r.lw;device const float* sw=r.sw;device float* result=r.result;
    const int n = gid;
    if (n < count) result[n] = mpas_add(lw[n], sw[n]);

}

kernel void hexv842_mpas_bl_mynn_dx_validate_f32(constant Hexv842MpasBlMynnDxValidateF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDxValidateF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {if(gid>=uint(p.ncol))return;uint bits=as_type<uint>(r.mesh_density[gid]);r.invalid[gid]=(bits<0x00800000u || bits>=0x7f800000u);}

kernel void hexv842_mpas_bl_mynn_dx_f32(constant Hexv842MpasBlMynnDxF32Resources &r [[buffer(0)]],constant Hexv842MpasBlMynnDxF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {if(gid>=uint(p.ncol))return;float d=r.mesh_density[gid];float power=as_type<float>(soft_gfk_pow_bits(as_type<uint>(d),0x3e800000u));r.dx[gid]=mynn_div(p.len_disp,power);}
