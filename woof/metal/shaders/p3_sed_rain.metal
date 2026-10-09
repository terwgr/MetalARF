#include <metal_stdlib>
#include "p3_sed_rain_bindings.metalh"

using namespace metal;

// P3 rain sedimentation (p3.cu:1596-1685). One thread owns a whole column;
// the serial Courant/substep and flux loops match the source ordering.
// Native pow/log/exp stand in for P3's pinned source libm and are not a
// CUDA-parity claim.
#pragma clang fp contract(off)

constant float P3_PI = 0x1.921fb60000000p+1f;
constant float P3_THRD = 0x1.5555560000000p-2f;
constant float P3_CONS1 = 0x1.05cca60000000p+9f;
constant float P3_QSMALL = 0x1.6849b80000000p-47f;
constant float P3_NSMALL = 0x1.cd2b2a0000000p-54f;
constant float P3_INV_DRMAX = 0x1.f3fffe0000000p+8f;
constant float P3_INV_RHOW = 0x1.0624de0000000p-10f;

static inline float p3_rain_table(device const float *table,
                                  int ii, int jj, float rii, float rjj) {
    const ulong a=ulong(ii-1)*10ul+ulong(jj-1);
    const ulong b=ulong(ii)*10ul+ulong(jj-1);
    const ulong c=ulong(ii-1)*10ul+ulong(jj);
    const ulong d=ulong(ii)*10ul+ulong(jj);
    const float dum1=table[a]+(rii-float(ii))*(table[b]-table[a]);
    const float dum2=table[c]+(rii-float(ii))*(table[d]-table[c]);
    return dum1+(rjj-float(jj))*(dum2-dum1);
}

static inline void p3_rain_dsd(thread float &nr, float qr,
                               thread float &lambda) {
    float n=fmax(nr,P3_NSMALL);
    lambda=pow(P3_CONS1*n*6.0f/qr,P3_THRD);
    const float lammin=P3_INV_DRMAX;
    const float lammax=1.0e5f;
    if (lambda<lammin || lambda>lammax) {
        lambda=clamp(lambda,lammin,lammax);
        n=exp(3.0f*log(lambda)+log(qr)-log(6.0f))/P3_CONS1;
    }
    nr=n;
}

static inline void p3_find_rain_table(float lambda, thread int &ii,
                                      thread int &jj, thread float &rii,
                                      thread float &rjj) {
    const float dum1=1.0f/lambda;
    if (dum1<=195.0e-6f) {
        rii=(dum1*1.0e6f+5.0f)*0.1f;
        rii=fmin(fmax(rii,1.0f),20.0f);
        ii=max(1,min(20,int(rii)));
    } else {
        rii=(dum1*1.0e6f-195.0f)*(P3_THRD*0.1f)+20.0f;
        rii=fmin(fmax(rii,20.0f),300.0f);
        ii=max(20,min(299,int(rii)));
    }
    rjj=1.0f;
    rjj=fmin(fmax(rjj,1.0f),10.0f);
    jj=max(1,min(9,int(rjj)));
}

kernel void p3_sed_rain(
    constant P3SedRainResources &r [[buffer(0)]],
    constant P3SedRainParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol=uint(p.ny)*uint(p.nx);
    if (column>=ncol || r.flags[ncol+column]==0.0f) return;
    const int nz=p.nz;
    const ulong base=ulong(column), stride=ulong(ncol);
    int ktop=0, present=0;
    for (int k=nz-1;k>=0;--k) {
        if (r.qr[ulong(k)*stride+base]>=P3_QSMALL) {
            present=1; ktop=k; break;
        }
    }
    if (!present) return;
    int kbot=0;
    for (int k=0;k<=ktop;++k) {
        if (r.qr[ulong(k)*stride+base]>=P3_QSMALL) { kbot=k; break; }
    }
    for (int k=0;k<nz;++k) {
        const ulong idx=ulong(k)*stride+base;
        r.vq[idx]=0.0f; r.vn[idx]=0.0f;
        r.fq[idx]=0.0f; r.fn[idx]=0.0f;
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
            if (r.qr[idx]>P3_QSMALL) {
                r.nr[idx]=fmax(r.nr[idx],P3_NSMALL);
                float nr=r.nr[idx],lamr;
                p3_rain_dsd(nr,r.qr[idx],lamr);
                r.nr[idx]=nr;
                int ii,jj; float rii,rjj;
                p3_find_rain_table(lamr,ii,jj,rii,rjj);
                r.vq[idx]=p3_rain_table(r.vm_table,ii,jj,rii,rjj)
                          *r.rhofacr[idx];
                r.vn[idx]=p3_rain_table(r.vn_table,ii,jj,rii,rjj)
                          *r.rhofacr[idx];
            }
            co_max=fmax(co_max,r.vq[idx]*dt_left*(1.0f/r.dz[idx]));
        }
        const int tmpint1=int(co_max+1.0f);
        const float dt_sub=fmin(dt_left,dt_left/float(tmpint1));
        const int ktemp=(kbot==0)?kbot:kbot-1;
        for (int k=ktemp;k<=ktop;++k) {
            const ulong idx=ulong(k)*stride+base;
            r.fq[idx]=r.vq[idx]*r.qr[idx]*r.rho[idx];
            r.fn[idx]=r.vn[idx]*r.nr[idx]*r.rho[idx];
        }
        if (kbot==0) prt_accum=prt_accum+r.fq[base]*dt_sub;
        {
            const ulong idx=ulong(ktop)*stride+base;
            const float idz=1.0f/r.dz[idx];
            const float fdq=-r.fq[idx]*idz;
            const float fdn=-r.fn[idx]*idz;
            r.qr[idx]=r.qr[idx]+fdq*dt_sub*r.inv_rho[idx];
            r.nr[idx]=r.nr[idx]+fdn*dt_sub*r.inv_rho[idx];
        }
        for (int k=ktop-1;k>=ktemp;--k) {
            const ulong idx=ulong(k)*stride+base;
            const ulong below=ulong(k+1)*stride+base;
            const float idz=1.0f/r.dz[idx];
            const float fdq=(r.fq[below]-r.fq[idx])*idz;
            const float fdn=(r.fn[below]-r.fn[idx])*idz;
            r.qr[idx]=r.qr[idx]+fdq*dt_sub*r.inv_rho[idx];
            r.nr[idx]=r.nr[idx]+fdn*dt_sub*r.inv_rho[idx];
        }
        dt_left=dt_left-dt_sub;
        if (kbot!=0) kbot=kbot-1;
    }
    const float odt=1.0f/p.dt;
    r.prt_liq[column]=r.prt_liq[column]+prt_accum*P3_INV_RHOW*odt;
}
