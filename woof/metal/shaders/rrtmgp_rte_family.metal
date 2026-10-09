#include <metal_stdlib>
#include "rrtmgp_delta_scale_bindings.metalh"
#include "rrtmgp_lw_noscat_bindings.metalh"
#include "rrtmgp_lw_flux_reduce_bindings.metalh"
#include "rrtmgp_sw_2stream_bindings.metalh"
#include "rrtmgp_sw_flux_reduce_bindings.metalh"
#include "rrtmgp_fluxes_to_radiation_bindings.metalh"

using namespace metal;
#pragma clang fp contract(off)

// Float-only CUDA operation boundaries. fma is explicit where the CUDA
// source uses __fmaf_rn; contraction is disabled everywhere else.
static inline float rte_add(float a, float b) { return a + b; }
static inline float rte_sub(float a, float b) { return a - b; }
static inline float rte_mul(float a, float b) { return a * b; }
static inline float rte_div(float a, float b) { return a / b; }

kernel void rrtmgp_fluxes_to_radiation(
    constant RrtmgpFluxesToRadiationResources &r [[buffer(0)]],
    constant RrtmgpFluxesToRadiationParams &p [[buffer(1)]],
    uint item [[thread_position_in_grid]]) {
    if (item < uint(p.ncell)) {
        const uint lay = item / uint(p.ncol);
        const uint col = item - lay * uint(p.ncol);
        const uint lower = col * uint(p.nlev) + lay;
        const uint upper = lower + 1u;
        const uint cell = col * uint(p.nlay) + lay;
        const float dp = abs(r.plev[upper] - r.plev[lower]);
        const float lw_top = r.lw_dn[upper] - r.lw_up[upper];
        const float lw_bottom = r.lw_dn[lower] - r.lw_up[lower];
        const float sw_top = r.sw_dn[upper] - r.sw_up[upper];
        const float sw_bottom = r.sw_dn[lower] - r.sw_up[lower];
        const float lw_convergence = lw_top - lw_bottom;
        const float sw_convergence = sw_top - sw_bottom;
        float lw_rate = p.heating_factor * lw_convergence;
        float sw_rate = p.heating_factor * sw_convergence;
        lw_rate = lw_rate / dp;
        sw_rate = sw_rate / dp;
        lw_rate = lw_rate / r.exner[cell];
        sw_rate = sw_rate / r.exner[cell];
        r.rthratenlw[item] = lw_rate;
        r.rthratensw[item] = sw_rate;
    }
    if (item < uint(p.ncol)) {
        const uint col = item;
        const uint top = col * uint(p.nlev) + uint(p.nlev - 1);
        r.swdown[col] = r.sw_dn[col * uint(p.nlev)];
        r.glw[col] = r.lw_dn[col * uint(p.nlev)];
        r.olr[col] = r.lw_up[top];
        r.gsw[col] = r.sw_dn[col * uint(p.nlev)]
                     - r.sw_up[col * uint(p.nlev)];
    }
}

#define RRTMGP_MAX_LAYERS 128

static inline int rte_i(device const float *bits, int i) {
    return as_type<int>(bits[i]);
}
static inline int rte_pk_km_index(int jt, int je, int jp, int g,
                                  int neta, int npressk, int ngpt) {
    return (((jt * neta + je) * npressk + jp) * ngpt + g);
}
static inline float rte_pk_interp(float t, int band,
                                  device const float *table, int nplanck,
                                  int nband, float temp_min, float delta) {
    const float val = rte_div(rte_sub(t, temp_min), delta);
    const int trunc_val = int(trunc(val));
    const int idx = max(0, min(nplanck - 2, trunc_val));
    const float frac = rte_sub(val, trunc(val));
    const float lo = table[idx * nband + band];
    return fma(frac, rte_sub(table[(idx + 1) * nband + band], lo), lo);
}
static inline float rte_pk_fraction_at(
    int col, int lay, int gpt, int nlay, int ngas, int ngpt, int ntemp,
    int npres, int neta, device const float *vmr,
    device const float *iatm_meta, device const float *jt_meta,
    device const float *jp_meta, device const float *ftemp_meta,
    device const float *fpress_meta, device const float *vmr_ref,
    device const float *flavor, device const float *gpoint_flavor,
    device const float *planck_fraction) {
    const int npressk = npres + 1;
    const int cell = col * nlay + lay;
    const int iatm = rte_i(iatm_meta, cell);
    const int iflav = rte_i(gpoint_flavor, iatm * ngpt + gpt);
    const int gas1 = rte_i(flavor, iflav * 2);
    const int gas2 = rte_i(flavor, iflav * 2 + 1);
    const int jt = rte_i(jt_meta, cell), jp = rte_i(jp_meta, cell);
    const float ft = ftemp_meta[cell], fp = fpress_meta[cell];
    float pf = 0.0f;
    for (int itemp = 0; itemp < 2; ++itemp) {
        const int jtr = jt + itemp;
        const float ratio = rte_div(
            vmr_ref[(iatm * (ngas + 1) + gas1) * ntemp + jtr],
            vmr_ref[(iatm * (ngas + 1) + gas2) * ntemp + jtr]);
        const float amount1 = gas1 == 0 ? 1.0f : vmr[cell * (ngas + 1) + gas1];
        const float amount2 = gas2 == 0 ? 1.0f : vmr[cell * (ngas + 1) + gas2];
        const float mix = fma(ratio, amount2, amount1);
        const float eta = mix > 2.0f * 1.17549435e-38f
            ? rte_div(amount1, mix) : 0.5f;
        const float loce = rte_mul(eta, float(neta - 1));
        const int je = min(int(loce), neta - 2);
        const float fe = rte_sub(loce, trunc(loce));
        const float wt = itemp == 0 ? rte_sub(1.0f, ft) : ft;
        const float omfp = rte_sub(1.0f, fp), omfe = rte_sub(1.0f, fe);
        const float w00 = rte_mul(rte_mul(omfp, omfe), wt);
        const float w10 = rte_mul(rte_mul(omfp, fe), wt);
        const float w01 = rte_mul(rte_mul(fp, omfe), wt);
        const float w11 = rte_mul(rte_mul(fp, fe), wt);
        float t = rte_mul(w00, planck_fraction[rte_pk_km_index(
            jtr, je, jp, gpt, neta, npressk, ngpt)]);
        t = fma(w10, planck_fraction[rte_pk_km_index(
            jtr, je + 1, jp, gpt, neta, npressk, ngpt)], t);
        t = fma(w01, planck_fraction[rte_pk_km_index(
            jtr, je, jp + 1, gpt, neta, npressk, ngpt)], t);
        t = fma(w11, planck_fraction[rte_pk_km_index(
            jtr, je + 1, jp + 1, gpt, neta, npressk, ngpt)], t);
        pf = rte_add(pf, t);
    }
    return pf;
}

kernel void rrtmgp_delta_scale(
    constant RrtmgpDeltaScaleResources &r [[buffer(0)]],
    constant RrtmgpDeltaScaleParams &p [[buffer(1)]],
    uint idx [[thread_position_in_grid]]) {
    if (idx >= uint(p.n)) return;
    const float gg = r.g_in[idx];
    const float f = gg * gg;
    const float wf = r.ssa_in[idx] * f;
    r.tau[idx] = (1.0f - wf) * r.tau_in[idx];
    r.ssa[idx] = (r.ssa_in[idx] - wf)
        / max(3.0f * 1.17549435e-38f, 1.0f - wf);
    r.asym[idx] = (gg - f) / max(3.0f * 1.17549435e-38f, 1.0f - f);
}

kernel void rrtmgp_lw_noscat(
    constant RrtmgpLwNoscatResources &r [[buffer(0)]],
    constant RrtmgpLwNoscatParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (p.warp_fold != 0 || p.nlay > RRTMGP_MAX_LAYERS) return;
    const int col = int(tid) / p.gpt_tile;
    const int gl = int(tid) - col * p.gpt_tile;
    if (col >= p.ncol) return;
    const int gpt = p.gpt0 + gl;
    if (gpt >= p.ngpt) return;
    const int nlev = p.nlay + 1;
    const int top = p.top_at_1 ? 0 : p.nlay;
    const int sfc = p.top_at_1 ? p.nlay : 0;
    const float d = 1.0f / 0.6096748751f;
    const float pi = 3.14159265358979323846f;
    const float tau_thresh = sqrt(sqrt(0x1p-23f));
    float trans[RRTMGP_MAX_LAYERS];
    float src_up[RRTMGP_MAX_LAYERS];
    const int base = gl * nlev;
    const int pk_band = p.pk_on ? rte_i(r.pk_gpoint_bands, gpt) : 0;
    const float pk_dpl = p.pk_on
        ? rte_div(rte_sub(r.pk_temp_ref[p.pk_ntemp - 1], r.pk_temp_ref[0]),
                  float(p.pk_nplanck - 1)) : 1.0f;
    const float pk_t0 = p.pk_on ? r.pk_temp_ref[0] : 0.0f;
    const int pk_first = p.top_at_1 ? 0 : p.nlay - 1;
    float pk_prev = (p.pk_on && pk_first > 0)
        ? rte_pk_fraction_at(col, pk_first - 1, gpt, p.nlay, p.pk_ngas,
            p.ngpt, p.pk_ntemp, p.pk_npres, p.pk_neta, r.pk_vmr,
            r.pk_iatm, r.pk_jt, r.pk_jp, r.pk_ftemp, r.pk_fpress,
            r.pk_vmr_ref, r.pk_flavor, r.pk_gpoint_flavor,
            r.pk_planck_fraction) : 0.0f;
    float pk_cur = p.pk_on
        ? rte_pk_fraction_at(col, pk_first, gpt, p.nlay, p.pk_ngas,
            p.ngpt, p.pk_ntemp, p.pk_npres, p.pk_neta, r.pk_vmr,
            r.pk_iatm, r.pk_jt, r.pk_jp, r.pk_ftemp, r.pk_fpress,
            r.pk_vmr_ref, r.pk_flavor, r.pk_gpoint_flavor,
            r.pk_planck_fraction) : 0.0f;
    float pk_next = (p.pk_on && pk_first + 1 < p.nlay)
        ? rte_pk_fraction_at(col, pk_first + 1, gpt, p.nlay, p.pk_ngas,
            p.ngpt, p.pk_ntemp, p.pk_npres, p.pk_neta, r.pk_vmr,
            r.pk_iatm, r.pk_jt, r.pk_jp, r.pk_ftemp, r.pk_fpress,
            r.pk_vmr_ref, r.pk_flavor, r.pk_gpoint_flavor,
            r.pk_planck_fraction) : 0.0f;
    float dnv = rte_div(r.incident_flux[col * p.ngpt + gpt], pi);
    r.part_dn[(base + top) * p.ncol + col] = dnv;

    for (int j = 0; j < p.nlay; ++j) {
        const int lay = p.top_at_1 ? j : p.nlay - 1 - j;
        const int cell_lay = col * p.nlay + lay;
        const int cellg = cell_lay * p.ngpt + gpt;
        float tau_v = r.tau[cellg];
        if (p.fz_on) {
            const int fz_band = (col * p.nlay + lay) * p.fz_nband
                              + rte_i(r.fz_gpt_bands, gpt);
            float fz_tc = r.fz_cld_tau[fz_band];
            if (p.fz_have_mask) fz_tc = rte_mul(fz_tc, r.fz_mask[cellg]);
            const float fz_oms = rte_sub(1.0f, r.fz_cld_ssa[fz_band]);
            tau_v = rte_add(tau_v, rte_mul(fz_tc, fz_oms));
        }
        const float path = rte_mul(tau_v, d);
        const float tr = exp(-path);
        trans[lay] = tr;
        const float fact = path > tau_thresh
            ? rte_sub(rte_div(rte_sub(1.0f, tr), path), tr)
            : rte_mul(path, fma(path, fma(path, 0.125f, -1.0f / 3.0f), 0.5f));
        float lays, lo, hi;
        if (p.pk_on) {
            lays = rte_mul(pk_cur, rte_pk_interp(r.pk_tlay[cell_lay],
                pk_band, r.pk_totplnk, p.pk_nplanck, p.pk_nband, pk_t0, pk_dpl));
            const float lo_pf = lay == 0 ? pk_cur : sqrt(rte_mul(pk_prev, pk_cur));
            const float hi_pf = lay + 1 == p.nlay ? pk_cur : sqrt(rte_mul(pk_cur, pk_next));
            lo = rte_mul(lo_pf, rte_pk_interp(r.pk_tlev[col*nlev+lay],
                pk_band, r.pk_totplnk, p.pk_nplanck, p.pk_nband, pk_t0, pk_dpl));
            hi = rte_mul(hi_pf, rte_pk_interp(r.pk_tlev[col*nlev+lay+1],
                pk_band, r.pk_totplnk, p.pk_nplanck, p.pk_nband, pk_t0, pk_dpl));
        } else {
            lays = r.lay_source[cellg];
            lo = r.lev_source[(col*nlev+lay)*p.ngpt+gpt];
            hi = r.lev_source[(col*nlev+lay+1)*p.ngpt+gpt];
        }
        const float omtr = rte_sub(1.0f, tr);
        const float fact2 = rte_add(fact, fact);
        const float inc = fma(hi, omtr, rte_mul(fact2, rte_sub(lays, hi)));
        const float dec = fma(lo, omtr, rte_mul(fact2, rte_sub(lays, lo)));
        src_up[lay] = p.top_at_1 ? dec : inc;
        if (p.pk_on) {
            if (p.top_at_1) {
                pk_prev = pk_cur; pk_cur = pk_next;
                pk_next = lay + 2 < p.nlay
                    ? rte_pk_fraction_at(col, lay+2, gpt, p.nlay, p.pk_ngas,
                        p.ngpt,p.pk_ntemp,p.pk_npres,p.pk_neta,r.pk_vmr,
                        r.pk_iatm,r.pk_jt,r.pk_jp,r.pk_ftemp,r.pk_fpress,
                        r.pk_vmr_ref,r.pk_flavor,r.pk_gpoint_flavor,
                        r.pk_planck_fraction) : 0.0f;
            } else {
                pk_next = pk_cur; pk_cur = pk_prev;
                pk_prev = lay - 2 >= 0
                    ? rte_pk_fraction_at(col, lay-2, gpt, p.nlay, p.pk_ngas,
                        p.ngpt,p.pk_ntemp,p.pk_npres,p.pk_neta,r.pk_vmr,
                        r.pk_iatm,r.pk_jt,r.pk_jp,r.pk_ftemp,r.pk_fpress,
                        r.pk_vmr_ref,r.pk_flavor,r.pk_gpoint_flavor,
                        r.pk_planck_fraction) : 0.0f;
            }
        }
        dnv = fma(tr, dnv, p.top_at_1 ? inc : dec);
        const int dlev = p.top_at_1 ? lay + 1 : lay;
        r.part_dn[(base+dlev)*p.ncol+col] = dnv;
    }
    const int sg = col * p.ngpt + gpt;
    float pk_sfc;
    if (p.pk_on) {
        const int pk_sfc_lay = r.pk_play[col*p.nlay] < r.pk_play[col*p.nlay+p.nlay-1]
            ? p.nlay - 1 : 0;
        pk_sfc = rte_mul(
            rte_pk_fraction_at(col,pk_sfc_lay,gpt,p.nlay,p.pk_ngas,p.ngpt,
                p.pk_ntemp,p.pk_npres,p.pk_neta,r.pk_vmr,r.pk_iatm,r.pk_jt,
                r.pk_jp,r.pk_ftemp,r.pk_fpress,r.pk_vmr_ref,r.pk_flavor,
                r.pk_gpoint_flavor,r.pk_planck_fraction),
            rte_pk_interp(r.pk_tsfc[col],pk_band,r.pk_totplnk,p.pk_nplanck,
                p.pk_nband,pk_t0,pk_dpl));
    } else pk_sfc = r.sfc_source[sg];
    float upv = fma(rte_sub(1.0f,r.sfc_emis[sg]),dnv,
                    rte_mul(r.sfc_emis[sg],pk_sfc));
    r.part_up[(base+sfc)*p.ncol+col] = upv;
    for (int j=0;j<p.nlay;++j) {
        const int lay = p.top_at_1 ? p.nlay-1-j : j;
        upv = fma(trans[lay],upv,src_up[lay]);
        const int ulev = p.top_at_1 ? lay : lay+1;
        r.part_up[(base+ulev)*p.ncol+col] = upv;
    }
}

kernel void rrtmgp_lw_flux_reduce(
    constant RrtmgpLwFluxReduceResources &r [[buffer(0)]],
    constant RrtmgpLwFluxReduceParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncol*p.nlev)) return;
    const int lev=int(tid)/p.ncol, col=int(tid)-lev*p.ncol;
    const int o=col*p.nlev+lev; const float pi=3.14159265358979323846f;
    float su=p.gpt0==0 ? 0.0f : r.flux_up[o];
    float sd=p.gpt0==0 ? 0.0f : r.flux_dn[o];
    for(int g=0;g<p.ntile;++g){
        const int q=(g*p.nlev+lev)*p.ncol+col;
        su=fma(pi,r.part_up[q],su); sd=fma(pi,r.part_dn[q],sd);
    }
    r.flux_up[o]=su; r.flux_dn[o]=sd;
}

kernel void rrtmgp_sw_2stream(
    constant RrtmgpSw2StreamResources &r [[buffer(0)]],
    constant RrtmgpSw2StreamParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (p.warp_fold != 0 || p.nlay > RRTMGP_MAX_LAYERS) return;
    const int col=int(tid)/p.gpt_tile, gl=int(tid)-col*p.gpt_tile;
    if(col>=p.ncol) return;
    const int gpt=p.gpt0+gl; if(gpt>=p.ngpt) return;
    const int nlev=p.nlay+1, top=p.top_at_1?0:p.nlay;
    const int top_lay=p.top_at_1?0:p.nlay-1, sfc=p.top_at_1?p.nlay:0;
    const int sfc_lay=p.top_at_1?p.nlay-1:0;
    const float eps=1.1920928955078125e-7f, min_k=1.0e4f*eps;
    const float min_mu=3.4526698300124393e-4f;
    float4 optics[RRTMGP_MAX_LAYERS]; float direct[RRTMGP_MAX_LAYERS+1];
    float2 adding[RRTMGP_MAX_LAYERS+1];
    const int spec=col*p.ngpt+gpt, base=gl*nlev;
    direct[top]=r.inc_flux[spec]*r.mu0[col*p.nlay+top_lay];
    for(int j=0;j<p.nlay;++j){
        const int lay=p.top_at_1?j:p.nlay-j-1;
        const int ilev_in=p.top_at_1?lay:lay+1, ilev_out=p.top_at_1?lay+1:lay;
        const int idx=(col*p.nlay+lay)*p.ngpt+gpt;
        float ts,ws,gs;
        if(p.fz_on){
            const int fzb=(col*p.nlay+lay)*p.fz_nband+rte_i(r.fz_gpt_bands,gpt);
            float fztc=r.fz_cld_tau[fzb];
            if(p.fz_have_mask) fztc=rte_mul(fztc,r.fz_mask[idx]);
            const float fw=r.fz_cld_ssa[fzb], fg=r.fz_cld_asy[fzb];
            const float floorv=3.0f*1.17549435e-38f;
            const float total_tau=rte_add(r.tau[idx],fztc);
            const float gas_sc=rte_mul(r.tau[idx],r.ssa[idx]);
            const float cld_sc=rte_mul(fztc,fw);
            const float scatter=rte_add(gas_sc,cld_sc);
            const float total_ssa=rte_div(scatter,max(floorv,total_tau));
            const float cloud_g=rte_mul(cld_sc,fg);
            const float total_g=rte_div(cloud_g,max(floorv,scatter));
            const float f=total_g*total_g, wf=total_ssa*f;
            ts=(1.0f-wf)*total_tau;
            ws=(total_ssa-wf)/max(floorv,1.0f-wf);
            gs=(total_g-f)/max(floorv,1.0f-f);
        }else{ts=r.tau[idx];ws=r.ssa[idx];gs=p.zero_g?0.0f:r.asym[idx];}
        const float gamma1=(8.0f-ws*(5.0f+3.0f*gs))*0.25f;
        const float gamma2=3.0f*ws*(1.0f-gs)*0.25f;
        const float kval=sqrt(max((gamma1-gamma2)*(gamma1+gamma2),min_k));
        const float ex=exp(-ts*kval), ex2=ex*ex;
        float rt=1.0f/(kval*(1.0f+ex2)+gamma1*(1.0f-ex2));
        const float rdif=rt*gamma2*(1.0f-ex2), tdif=rt*2.0f*kval*ex;
        const float mu=max(min_mu,r.mu0[col*p.nlay+lay]), kmu=kval*mu;
        float dterm=1.0f-kmu*kmu; if(abs(dterm)<eps)dterm=eps;
        rt=ws*rt/dterm;
        const float gamma3=(2.0f-3.0f*mu*gs)*0.25f, gamma4=1.0f-gamma3;
        const float alpha1=gamma1*gamma4+gamma2*gamma3;
        const float alpha2=gamma1*gamma3+gamma2*gamma4;
        const float kg3=kval*gamma3,kg4=kval*gamma4,tnoscat=exp(-ts/mu);
        float rdir=rt*((1.0f-kmu)*(alpha2+kg3)
             -(1.0f+kmu)*(alpha2-kg3)*ex2
             -2.0f*(kg3-alpha2*kmu)*ex*tnoscat);
        float tdir=-rt*((1.0f+kmu)*(alpha1+kg4)*tnoscat
             -(1.0f-kmu)*(alpha1-kg4)*ex2*tnoscat
             -2.0f*(kg4+alpha1*kmu)*ex);
        rdir=max(0.0f,min(rdir,1.0f-tnoscat));
        tdir=max(0.0f,min(tdir,1.0f-tnoscat-rdir));
        float src_up=rdir*direct[ilev_in],src_dn=tdir*direct[ilev_in];
        direct[ilev_out]=tnoscat*direct[ilev_in];
        if(r.mu0[col*p.nlay+lay]<=0.0f)src_up=src_dn=0.0f;
        optics[lay]=float4(rdif,tdif,src_dn,src_up);
    }
    const float source_sfc=r.mu0[col*p.nlay+sfc_lay]>0.0f
        ? direct[sfc]*r.sfc_alb_dir[spec]:0.0f;
    float dnv,upv;
    if(p.top_at_1){
        adding[p.nlay]=float2(r.sfc_alb_dif[spec],source_sfc);
        for(int lev=p.nlay-1;lev>=0;--lev){
            const float4 c=optics[lev];const float dn_=1.0f/(1.0f-c.x*adding[lev+1].x);
            adding[lev].x=c.x+c.y*c.y*adding[lev+1].x*dn_;
            adding[lev].y=c.w+c.y*dn_*(adding[lev+1].y+adding[lev+1].x*c.z);
        }
        dnv=0.0f;upv=dnv*adding[0].x+adding[0].y;
        r.part_up[base*p.ncol+col]=upv;r.part_dn[base*p.ncol+col]=dnv+direct[0];
        r.part_dir[base*p.ncol+col]=direct[0];
        for(int lev=1;lev<nlev;++lev){const int lay=lev-1;const float4 c=optics[lay];
            const float dn_=1.0f/(1.0f-c.x*adding[lev].x);
            dnv=(c.y*dnv+c.x*adding[lev].y+c.z)*dn_;upv=dnv*adding[lev].x+adding[lev].y;
            r.part_up[(base+lev)*p.ncol+col]=upv;r.part_dn[(base+lev)*p.ncol+col]=dnv+direct[lev];
            r.part_dir[(base+lev)*p.ncol+col]=direct[lev];}
    }else{
        adding[0]=float2(r.sfc_alb_dif[spec],source_sfc);
        for(int lev=0;lev<p.nlay;++lev){const float4 c=optics[lev];const float dn_=1.0f/(1.0f-c.x*adding[lev].x);
            adding[lev+1].x=c.x+c.y*c.y*adding[lev].x*dn_;
            adding[lev+1].y=c.w+c.y*dn_*(adding[lev].y+adding[lev].x*c.z);}
        dnv=0.0f;upv=dnv*adding[p.nlay].x+adding[p.nlay].y;
        r.part_up[(base+p.nlay)*p.ncol+col]=upv;r.part_dn[(base+p.nlay)*p.ncol+col]=dnv+direct[p.nlay];
        r.part_dir[(base+p.nlay)*p.ncol+col]=direct[p.nlay];
        for(int lev=p.nlay-1;lev>=0;--lev){const float4 c=optics[lev];const float dn_=1.0f/(1.0f-c.x*adding[lev].x);
            dnv=(c.y*dnv+c.x*adding[lev].y+c.z)*dn_;upv=dnv*adding[lev].x+adding[lev].y;
            r.part_up[(base+lev)*p.ncol+col]=upv;r.part_dn[(base+lev)*p.ncol+col]=dnv+direct[lev];
            r.part_dir[(base+lev)*p.ncol+col]=direct[lev];}
    }
}

kernel void rrtmgp_sw_flux_reduce(
    constant RrtmgpSwFluxReduceResources &r [[buffer(0)]],
    constant RrtmgpSwFluxReduceParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncol*p.nlev))return;
    const int lev=int(tid)/p.ncol,col=int(tid)-lev*p.ncol,o=col*p.nlev+lev;
    float su=p.gpt0==0?0.0f:r.flux_up[o];
    float sd=p.gpt0==0?0.0f:r.flux_dn[o];
    float sr=p.gpt0==0?0.0f:r.flux_dir[o];
    for(int g=0;g<p.ntile;++g){const int q=(g*p.nlev+lev)*p.ncol+col;
        su=su+r.part_up[q];sd=sd+r.part_dn[q];sr=sr+r.part_dir[q];}
    r.flux_up[o]=su;r.flux_dn[o]=sd;r.flux_dir[o]=sr;
}
