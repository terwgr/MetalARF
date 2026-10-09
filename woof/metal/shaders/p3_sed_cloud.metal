#include <metal_stdlib>
#include "p3_sed_cloud_bindings.metalh"
#include "soft_gfk_tgamma.metalh"

using namespace metal;

// P3 cloud sedimentation, serial within each flagged column
// (p3.cu:1510-1594). Arithmetic order follows the source substep and flux
// loops. Native MSL pow and the shared soft GFK gamma stand in for P3's
// pinned libm operations; CUDA parity remains unverified.
#pragma clang fp contract(off)

constant float P3_CONS1 = 0x1.05cca60000000p+9f;
constant float P3_THRD = 0x1.5555560000000p-2f;
constant float P3_BCN = 0x1.0000000000000p+1f;
constant float P3_QSMALL = 0x1.6849b80000000p-47f;
constant float P3_NSMALL = 0x1.cd2b2a0000000p-54f;
constant float P3_PI = 0x1.921fb60000000p+1f;
constant float P3_RHOW = 0x1.f400000000000p+9f;
constant float P3_INV_RHOW = 0x1.0624de0000000p-10f;

static inline void p3_cloud_dsd(thread float &nc, float qc, float rho,
                                thread float &mu, thread float &lambda) {
    float n=fmax(nc,P3_NSMALL);
    mu=0.0005714f*(n*1.0e-6f*rho)+0.2714f;
    mu=1.0f/(mu*mu)-1.0f;
    mu=fmin(fmax(mu,2.0f),15.0f);
    lambda=pow(P3_CONS1*n*(mu+3.0f)*(mu+2.0f)*(mu+1.0f)/qc,P3_THRD);
    const float lammin=(mu+1.0f)*2.5e4f;
    const float lammax=(mu+1.0f)*1.0e6f;
    if (lambda<lammin) {
        lambda=lammin;
        n=6.0f*((lambda*lambda)*lambda)*qc
          /(P3_PI*P3_RHOW*(mu+3.0f)*(mu+2.0f)*(mu+1.0f));
    } else if (lambda>lammax) {
        lambda=lammax;
        n=6.0f*((lambda*lambda)*lambda)*qc
          /(P3_PI*P3_RHOW*(mu+3.0f)*(mu+2.0f)*(mu+1.0f));
    }
    nc=n;
}

kernel void p3_sed_cloud(
    constant P3SedCloudResources &r [[buffer(0)]],
    constant P3SedCloudParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol=uint(p.ny)*uint(p.nx);
    if (column>=ncol || r.flags[ncol+column]==0.0f) return;
    const int nz=p.nz;
    const ulong base=ulong(column);
    const ulong stride=ulong(ncol);
    int ktop=0, present=0;
    for (int k=nz-1;k>=0;--k) {
        if (r.qc[ulong(k)*stride+base]>=P3_QSMALL) {
            present=1; ktop=k; break;
        }
    }
    if (!present) return;
    int kbot=0;
    for (int k=0;k<=ktop;++k) {
        if (r.qc[ulong(k)*stride+base]>=P3_QSMALL) { kbot=k; break; }
    }
    for (int k=0;k<nz;++k) {
        const ulong idx=ulong(k)*stride+base;
        r.vq[idx]=0.0f; r.vn[idx]=0.0f; r.fq[idx]=0.0f; r.fn[idx]=0.0f;
    }

    float dt_left=p.dt, prt_accum=0.0f;
    while (dt_left>1.0e-4f) {
        float co_max=0.0f;
        for (int k=0;k<nz;++k) {
            const ulong idx=ulong(k)*stride+base;
            r.vq[idx]=0.0f; r.vn[idx]=0.0f;
        }
        for (int k=ktop;k>=kbot;--k) {
            const ulong idx=ulong(k)*stride+base;
            if (r.qc[idx]>P3_QSMALL) {
                float mu_c,lamc,nc=r.nc[idx];
                p3_cloud_dsd(nc,r.qc[idx],r.rho[idx],mu_c,lamc);
                r.nc[idx]=nc;
                const float dum=1.0f/pow(lamc,P3_BCN);
                r.vq[idx]=r.acn[idx]
                    *soft_gfk_tgammaf(4.0f+P3_BCN+mu_c)*dum
                    /soft_gfk_tgammaf(mu_c+4.0f);
                if (p.log_predict_nc!=0) {
                    r.vn[idx]=r.acn[idx]
                        *soft_gfk_tgammaf(1.0f+P3_BCN+mu_c)*dum
                        /soft_gfk_tgammaf(mu_c+1.0f);
                }
            }
            co_max=fmax(co_max,r.vq[idx]*dt_left*(1.0f/r.dz[idx]));
        }
        const int tmpint1=int(co_max+1.0f);
        const float dt_sub=fmin(dt_left,dt_left/float(tmpint1));
        const int ktemp=(kbot==0)?kbot:kbot-1;
        for (int k=ktemp;k<=ktop;++k) {
            const ulong idx=ulong(k)*stride+base;
            r.fq[idx]=r.vq[idx]*r.qc[idx]*r.rho[idx];
            if (p.log_predict_nc!=0)
                r.fn[idx]=r.vn[idx]*r.nc[idx]*r.rho[idx];
        }
        if (kbot==0) prt_accum=prt_accum+r.fq[base]*dt_sub;
        {
            const int k=ktop;
            const ulong idx=ulong(k)*stride+base;
            const float idz=1.0f/r.dz[idx];
            const float fdq=-r.fq[idx]*idz;
            r.qc[idx]=r.qc[idx]+fdq*dt_sub*r.inv_rho[idx];
            if (p.log_predict_nc!=0) {
                const float fdn=-r.fn[idx]*idz;
                r.nc[idx]=r.nc[idx]+fdn*dt_sub*r.inv_rho[idx];
            }
        }
        for (int k=ktop-1;k>=ktemp;--k) {
            const ulong idx=ulong(k)*stride+base;
            const ulong below=ulong(k+1)*stride+base;
            const float idz=1.0f/r.dz[idx];
            const float fdq=(r.fq[below]-r.fq[idx])*idz;
            r.qc[idx]=r.qc[idx]+fdq*dt_sub*r.inv_rho[idx];
            if (p.log_predict_nc!=0) {
                const float fdn=(r.fn[below]-r.fn[idx])*idz;
                r.nc[idx]=r.nc[idx]+fdn*dt_sub*r.inv_rho[idx];
            }
        }
        dt_left=dt_left-dt_sub;
        if (kbot!=0) kbot=kbot-1;
    }
    const float odt=1.0f/p.dt;
    r.prt_liq[column]=prt_accum*P3_INV_RHOW*odt;
}
