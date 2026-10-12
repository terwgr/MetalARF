#include <metal_stdlib>
#include "p3_sed_ice_bindings.metalh"
#include "p3_constants.metalh"
#include "p3_math.metalh"

using namespace metal;

// P3 one-category ice sedimentation (p3.cu:1687-1779). One invocation
// owns a complete column; table lookup, Courant substeps, flux completion,
// and level updates preserve source order.
#pragma clang fp contract(off)

static inline float p3_ice_tab(device const float *itab, int jj, int ii,
                               int i, int q) {
    const ulong index=(((ulong(jj)*4ul+ulong(ii))*50ul+ulong(i))*14ul
                       +ulong(q));
    return itab[index];
}

static inline float p3_ice_interp(device const float *itab, int jj, int ii,
                                  int i, int q, float d1, float d4,
                                  float d5) {
    const float a=p3_ice_tab(itab,jj-1,ii-1,i-1,q-1)
        +(d1-float(i))*(p3_ice_tab(itab,jj-1,ii-1,i,q-1)
                        -p3_ice_tab(itab,jj-1,ii-1,i-1,q-1));
    const float b=p3_ice_tab(itab,jj-1,ii,i-1,q-1)
        +(d1-float(i))*(p3_ice_tab(itab,jj-1,ii,i,q-1)
                        -p3_ice_tab(itab,jj-1,ii,i-1,q-1));
    const float tmp1=a+(d4-float(ii))*(b-a);
    const float c=p3_ice_tab(itab,jj,ii-1,i-1,q-1)
        +(d1-float(i))*(p3_ice_tab(itab,jj,ii-1,i,q-1)
                        -p3_ice_tab(itab,jj,ii-1,i-1,q-1));
    const float d=p3_ice_tab(itab,jj,ii,i-1,q-1)
        +(d1-float(i))*(p3_ice_tab(itab,jj,ii,i,q-1)
                        -p3_ice_tab(itab,jj,ii,i-1,q-1));
    const float tmp2=c+(d4-float(ii))*(d-c);
    return tmp1+(d5-float(jj))*(tmp2-tmp1);
}

static inline void p3_bulk_rime(float qi, thread float &qir,
                                thread float &qib, thread float &rhop,
                                device const float *pc) {
    float rho_rime;
    if (qib>=1.0e-15f) {
        rho_rime=qir/qib;
        if (rho_rime<P3_RHO_RIMEMIN) {
            rho_rime=P3_RHO_RIMEMIN; qib=qir/rho_rime;
        } else if (rho_rime>P3_RHO_RIMEMAX) {
            rho_rime=P3_RHO_RIMEMAX; qib=qir/rho_rime;
        }
    } else { qir=0.0f; qib=0.0f; rho_rime=0.0f; }
    if (qir>qi && rho_rime>0.0f) { qir=qi; qib=qir/rho_rime; }
    if (qir<P3_QSMALL) { qir=0.0f; qib=0.0f; }
    rhop=rho_rime;
}

static inline void p3_find_ice(float qi, float ni, float qir, float rhop,
                               thread int &i, thread int &jj,
                               thread int &ii, thread float &d1,
                               thread float &d4, thread float &d5) {
    d1=(p3_log10(qi/ni)+18.0f)*3.444606f-10.0f;
    int di=int(d1);
    d1=fmax(fmin(d1,50.0f),1.0f); di=max(1,min(49,di));
    d4=(qir/qi)*3.0f+1.0f;
    int dii=int(d4);
    d4=fmax(fmin(d4,4.0f),1.0f); dii=max(1,min(3,dii));
    if (rhop<=650.0f) d5=(rhop-50.0f)*0.005f+1.0f;
    else d5=(rhop-650.0f)*0.004f+4.0f;
    int djj=int(d5);
    d5=fmax(fmin(d5,5.0f),1.0f); djj=max(1,min(4,djj));
    i=di; ii=dii; jj=djj;
}

static inline float p3_impose_max_ni(float ni, float inv_rho,
                                     device const float *pc) {
    if (ni>=1.0e-20f) ni=ni*fmin(P3_MAX_TOTAL_NI*inv_rho/ni,1.0f);
    return ni;
}

kernel void p3_sed_ice(
    constant P3SedIceResources &r [[buffer(0)]],
    constant P3SedIceParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol=uint(p.ny)*uint(p.nx);
    if (column>=ncol || r.flags[ncol+column]==0.0f) return;
    device const float *pc = r.constants;
    const int nz=p.nz;
    const ulong base=ulong(column), stride=ulong(ncol);
    int ktop=0,present=0;
    for (int k=nz-1;k>=0;--k) {
        if (r.qi[ulong(k)*stride+base]>=P3_QSMALL) {
            present=1; ktop=k; break;
        }
    }
    if (!present) return;
    int kbot=0;
    for (int k=0;k<=ktop;++k) {
        if (r.qi[ulong(k)*stride+base]>=P3_QSMALL) { kbot=k; break; }
    }
    for (int k=0;k<nz;++k) {
        const ulong idx=ulong(k)*stride+base;
        r.vq[idx]=0.0f; r.vn[idx]=0.0f; r.fq[idx]=0.0f; r.fn[idx]=0.0f;
        r.fqir[idx]=0.0f; r.fbir[idx]=0.0f;
    }

    float dt_left=p.dt,prt_accum=0.0f;
    while (dt_left>1.0e-4f) {
        float co_max=0.0f;
        for (int k=0;k<nz;++k) {
            const ulong idx=ulong(k)*stride+base;
            r.vq[idx]=0.0f; r.vn[idx]=0.0f;
        }
        for (int k=ktop;k>=kbot;--k) {
            const ulong idx=ulong(k)*stride+base;
            if (r.qi[idx]>=P3_QSMALL) {
                r.ni[idx]=fmax(r.ni[idx],P3_NSMALL);
                float qir=r.qir[idx],qib=r.qib[idx],rhop;
                p3_bulk_rime(r.qi[idx],qir,qib,rhop,pc);
                r.qir[idx]=qir; r.qib[idx]=qib;
                int i,jj,ii; float d1,d4,d5;
                p3_find_ice(r.qi[idx],r.ni[idx],r.qir[idx],rhop,
                            i,jj,ii,d1,d4,d5);
                const float f01=p3_ice_interp(r.itab,jj,ii,i,1,d1,d4,d5);
                const float f02=p3_ice_interp(r.itab,jj,ii,i,2,d1,d4,d5);
                const float f09=p3_ice_interp(r.itab,jj,ii,i,7,d1,d4,d5);
                const float f10=p3_ice_interp(r.itab,jj,ii,i,8,d1,d4,d5);
                r.ni[idx]=fmin(r.ni[idx],f09*r.qi[idx]);
                r.ni[idx]=fmax(r.ni[idx],f10*r.qi[idx]);
                r.vq[idx]=f02*r.rhofaci[idx];
                r.vn[idx]=f01*r.rhofaci[idx];
            }
            co_max=fmax(co_max,r.vq[idx]*dt_left*(1.0f/r.dz[idx]));
        }
        const int tmpint1=int(co_max+1.0f);
        const float dt_sub=fmin(dt_left,dt_left/float(tmpint1));
        const int ktemp=(kbot==0)?kbot:kbot-1;
        for (int k=ktemp;k<=ktop;++k) {
            const ulong idx=ulong(k)*stride+base;
            r.fq[idx]=r.vq[idx]*r.qi[idx]*r.rho[idx];
            r.fn[idx]=r.vn[idx]*r.ni[idx]*r.rho[idx];
            r.fqir[idx]=r.vq[idx]*r.qir[idx]*r.rho[idx];
            r.fbir[idx]=r.vq[idx]*r.qib[idx]*r.rho[idx];
        }
        if (kbot==0) prt_accum=prt_accum+r.fq[base]*dt_sub;
        {
            const ulong idx=ulong(ktop)*stride+base;
            const float idz=1.0f/r.dz[idx], irho=r.inv_rho[idx];
            r.qi[idx]=r.qi[idx]+(-r.fq[idx]*idz)*dt_sub*irho;
            r.qir[idx]=r.qir[idx]+(-r.fqir[idx]*idz)*dt_sub*irho;
            r.qib[idx]=r.qib[idx]+(-r.fbir[idx]*idz)*dt_sub*irho;
            r.ni[idx]=r.ni[idx]+(-r.fn[idx]*idz)*dt_sub*irho;
        }
        for (int k=ktop-1;k>=ktemp;--k) {
            const ulong idx=ulong(k)*stride+base;
            const ulong below=ulong(k+1)*stride+base;
            const float idz=1.0f/r.dz[idx], irho=r.inv_rho[idx];
            r.qi[idx]=r.qi[idx]+((r.fq[below]-r.fq[idx])*idz)*dt_sub*irho;
            r.qir[idx]=r.qir[idx]+((r.fqir[below]-r.fqir[idx])*idz)*dt_sub*irho;
            r.qib[idx]=r.qib[idx]+((r.fbir[below]-r.fbir[idx])*idz)*dt_sub*irho;
            r.ni[idx]=r.ni[idx]+((r.fn[below]-r.fn[idx])*idz)*dt_sub*irho;
        }
        dt_left=dt_left-dt_sub;
        if (kbot!=0) kbot=kbot-1;
    }
    r.prt_sol[column]=r.prt_sol[column]
        +prt_accum*P3_INV_RHOW*(1.0f/p.dt);
}
