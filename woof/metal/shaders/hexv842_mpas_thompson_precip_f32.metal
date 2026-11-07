#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_scalar.metalh"
#include "hexv842_mpas_thompson_precip_f32_bindings.metalh"
#pragma clang fp contract(off)
kernel void hexv842_mpas_thompson_precip_f32(constant Hexv842MpasThompsonPrecipF32Resources &r [[buffer(0)]],constant Hexv842MpasThompsonPrecipF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
int nz=p.nz,columns=p.columns,phase=p.phase,l_acrain=p.l_acrain;float bucket=p.bucket;

    int col=int(gid);if(col>=columns)return;
    if(phase==0) {
        r.work_rainnc[col]=0.0f;
r.work_rainncv[col]=0.0f;
r.work_snownc[col]=0.0f;
r.work_snowncv[col]=0.0f;
r.work_graupelnc[col]=0.0f;
r.work_graupelncv[col]=0.0f;
r.work_sr[col]=0.0f;
        r.diag_rainncv[col]=0.0f;r.diag_snowncv[col]=0.0f;r.diag_graupelncv[col]=0.0f;r.diag_sr[col]=0.0f;
        return;
    }
    r.diag_precipw[col]=0.0f;
    for(int k=0;k<nz;k++) {
        int idx=k*columns+col;float rho_a=p.correct_wb3 ? r.rho[idx] : tc_div(r.rho[idx],(tc_add(1.0f,r.qv[idx])));
        r.diag_precipw[col]=tc_add(r.diag_precipw[col],tc_mul(tc_mul(r.qv[idx],rho_a),r.dz[idx]));
    }
    r.diag_rainncv[col]=r.work_rainnc[col];r.diag_rainnc[col]=tc_add(r.diag_rainnc[col],r.diag_rainncv[col]);
    if(l_acrain && soft32_less(0u,as_type<uint>(bucket)) && soft32_less(as_type<uint>(bucket),as_type<uint>(r.diag_rainnc[col]))) {
        r.i_rainnc[col]=tc_add(r.i_rainnc[col],1);r.diag_rainnc[col]=tc_sub(r.diag_rainnc[col],bucket);
    }
    r.diag_snowncv[col]=r.work_snownc[col];r.diag_graupelncv[col]=r.work_graupelnc[col];
    r.diag_sr[col]=tc_div((tc_add(r.work_snownc[col],r.work_graupelnc[col])),(tc_add(r.work_rainnc[col],1.0e-12f)));
    r.diag_snownc[col]=tc_add(r.diag_snownc[col],r.diag_snowncv[col]);r.diag_graupelnc[col]=tc_add(r.diag_graupelnc[col],r.diag_graupelncv[col]);

}
