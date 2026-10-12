#include <metal_stdlib>
#include "p3_final_bindings.metalh"
#include "p3_constants.metalh"
#include "p3_math.metalh"

using namespace metal;

// P3 post-freezing cleanup and diagnostics (p3.cu:1827-1926). One thread
// owns one column and visits levels in order. Native MSL pow and the shared
// GFK gamma stand in for source-pinned r_pow/tgamma; log10 uses soft64.
// This remains ordinary-mode source only.
#pragma clang fp contract(off)

static inline ulong p3_itab_index(int jj, int ii, int i, int idx) {
    return (((ulong(jj) * 4ul + ulong(ii)) * 50ul + ulong(i)) * 14ul
            + ulong(idx));
}

static inline float p3_tab(device const float *itab, int jj, int ii, int i,
                           int idx) {
    return itab[p3_itab_index(jj, ii, i, idx)];
}

static inline float p3_tab_interp(device const float *itab, int jj, int ii,
                                  int i, int idx, float d1, float d4,
                                  float d5) {
    const float iproc1 = p3_tab(itab,jj-1,ii-1,i-1,idx-1)
        + (d1-float(i))*(p3_tab(itab,jj-1,ii-1,i,idx-1)
                         -p3_tab(itab,jj-1,ii-1,i-1,idx-1));
    const float gproc1 = p3_tab(itab,jj-1,ii,i-1,idx-1)
        + (d1-float(i))*(p3_tab(itab,jj-1,ii,i,idx-1)
                         -p3_tab(itab,jj-1,ii,i-1,idx-1));
    const float tmp1 = iproc1 + (d4-float(ii))*(gproc1-iproc1);
    const float iproc2 = p3_tab(itab,jj,ii-1,i-1,idx-1)
        + (d1-float(i))*(p3_tab(itab,jj,ii-1,i,idx-1)
                         -p3_tab(itab,jj,ii-1,i-1,idx-1));
    const float gproc2 = p3_tab(itab,jj,ii,i-1,idx-1)
        + (d1-float(i))*(p3_tab(itab,jj,ii,i,idx-1)
                         -p3_tab(itab,jj,ii,i-1,idx-1));
    const float tmp2 = iproc2 + (d4-float(ii))*(gproc2-iproc2);
    return tmp1 + (d5-float(jj))*(tmp2-tmp1);
}

static inline void p3_find_ice(float qi, float ni, float qir, float rhop,
                               thread int &i, thread int &jj,
                               thread int &ii, thread float &d1,
                               thread float &d4, thread float &d5) {
    d1 = (p3_log10(qi/ni)+18.0f)*3.444606f-10.0f;
    int di = int(d1);
    d1 = fmax(fmin(d1,50.0f),1.0f);
    di = max(1,min(49,di));
    d4 = (qir/qi)*3.0f+1.0f;
    int dii = int(d4);
    d4 = fmax(fmin(d4,4.0f),1.0f);
    dii = max(1,min(3,dii));
    if (rhop <= 650.0f) d5 = (rhop-50.0f)*0.005f+1.0f;
    else d5 = (rhop-650.0f)*0.004f+4.0f;
    int djj = int(d5);
    d5 = fmax(fmin(d5,5.0f),1.0f);
    djj = max(1,min(4,djj));
    i=di; ii=dii; jj=djj;
}

static inline float p3_impose_max_ni(float ni, float inv_rho,
                                     device const float *pc) {
    if (ni >= 1.0e-20f) {
        const float dum = P3_MAX_TOTAL_NI*inv_rho/ni;
        ni = ni*fmin(dum,1.0f);
    }
    return ni;
}

static inline void p3_calc_rime(float qi, thread float &qir,
                                thread float &qib, thread float &rhop,
                                device const float *pc) {
    float rho_rime;
    if (qib >= 1.0e-15f) {
        rho_rime = qir/qib;
        if (rho_rime < P3_RHO_RIMEMIN) {
            rho_rime=P3_RHO_RIMEMIN; qib=qir/rho_rime;
        } else if (rho_rime > P3_RHO_RIMEMAX) {
            rho_rime=P3_RHO_RIMEMAX; qib=qir/rho_rime;
        }
    } else { qir=0.0f; qib=0.0f; rho_rime=0.0f; }
    if (qir > qi && rho_rime > 0.0f) { qir=qi; qib=qir/rho_rime; }
    if (qir < P3_QSMALL) { qir=0.0f; qib=0.0f; }
    rhop=rho_rime;
}

static inline void p3_cloud_dsd(thread float &nc, float qc, float rho,
                                thread float &mu, thread float &lambda,
                                device const float *pc) {
    float n= fmax(nc,P3_NSMALL);
    mu=0.0005714f*(n*1.0e-6f*rho)+0.2714f;
    mu=1.0f/(mu*mu)-1.0f;
    mu=fmin(fmax(mu,2.0f),15.0f);
    lambda=pow(P3_CONS1*n*(mu+3.0f)*(mu+2.0f)*(mu+1.0f)/qc,P3_THRD);
    const float lammin=(mu+1.0f)*2.5e4f;
    const float lammax=(mu+1.0f)*1.0e6f;
    if (lambda < lammin) {
        lambda=lammin;
        n=6.0f*((lambda*lambda)*lambda)*qc
          /(P3_PI*P3_RHOW*(mu+3.0f)*(mu+2.0f)*(mu+1.0f));
    } else if (lambda > lammax) {
        lambda=lammax;
        n=6.0f*((lambda*lambda)*lambda)*qc
          /(P3_PI*P3_RHOW*(mu+3.0f)*(mu+2.0f)*(mu+1.0f));
    }
    nc=n;
}

static inline void p3_rain_dsd(thread float &nr, float qr,
                               thread float &lambda,
                               device const float *pc) {
    const float mu=0.0f;
    float n=fmax(nr,P3_NSMALL);
    lambda=pow(P3_CONS1*n*(mu+3.0f)*(mu+2.0f)*(mu+1.0f)/qr,P3_THRD);
    const float lammin=(mu+1.0f)*P3_INV_DRMAX;
    const float lammax=(mu+1.0f)*1.0e5f;
    if (lambda < lammin || lambda > lammax) {
        lambda=clamp(lambda,lammin,lammax);
        n=exp(3.0f*log(lambda)+log(qr)+log(1.0f)-log(6.0f))/P3_CONS1;
    }
    nr=n;
}

kernel void p3_final(
    constant P3FinalResources &r [[buffer(0)]],
    constant P3FinalParams &p [[buffer(1)]],
    uint column [[thread_position_in_grid]]) {
    const uint ncol=uint(p.ny)*uint(p.nx);
    if (column>=ncol) return;
    device const float *pc = r.constants;
    const float xxlv = p.xxlv, xxls = p.xxls;
    if (r.flags[ncol+column]==0.0f) {
        // The state-owned radii use microns; P3's entry defaults and solver
        // outputs are metres. Convert even inactive columns after entry has
        // initialized their default values.
        for (uint k=0;k<uint(p.nz);++k) {
            const ulong idx=ulong(k)*ulong(ncol)+ulong(column);
            r.effc[idx]=r.effc[idx]*1.0e6f;
            r.effi[idx]=r.effi[idx]*1.0e6f;
        }
        return;
    }
    for (uint k=0;k<uint(p.nz);++k) {
        const ulong idx=ulong(k)*ulong(ncol)+ulong(column);
        const float rho=r.rho[idx], inv_rho=r.inv_rho[idx];
        const float invexn=1.0f/r.tmparr1[idx];
        float ze_ice=1.0e-22f, ze_rain=1.0e-22f;

        if (r.qc[idx]>=P3_QSMALL) {
            float mu_c,lamc,nc=r.nc[idx];
            p3_cloud_dsd(nc,r.qc[idx],rho,mu_c,lamc,pc);
            r.nc[idx]=nc;
            r.effc[idx]=0.5f*(mu_c+3.0f)/lamc;
        } else {
            r.qv[idx]=r.qv[idx]+r.qc[idx];
            r.theta[idx]=r.theta[idx]-invexn*r.qc[idx]*xxlv*P3_INV_CP;
            r.qc[idx]=0.0f; r.nc[idx]=0.0f;
        }

        if (r.qr[idx]>=P3_QSMALL) {
            float nr=r.nr[idx],lamr;
            p3_rain_dsd(nr,r.qr[idx],lamr,pc);
            r.nr[idx]=nr;
            const float l2=lamr*lamr;
            const float lam6=(l2*l2)*l2;
            ze_rain=rho*nr*6.0f*5.0f*4.0f*3.0f*2.0f*1.0f/lam6;
            ze_rain=fmax(ze_rain,1.0e-22f);
        } else {
            r.qv[idx]=r.qv[idx]+r.qr[idx];
            r.theta[idx]=r.theta[idx]-invexn*r.qr[idx]*xxlv*P3_INV_CP;
            r.qr[idx]=0.0f; r.nr[idx]=0.0f;
        }

        r.ni[idx]=p3_impose_max_ni(r.ni[idx],inv_rho,pc);
        if (r.qi[idx]>=P3_QSMALL) {
            r.ni[idx]=fmax(r.ni[idx],P3_NSMALL);
            r.nr[idx]=fmax(r.nr[idx],P3_NSMALL);
            float qir=r.qir[idx],qib=r.qib[idx],rhop;
            p3_calc_rime(r.qi[idx],qir,qib,rhop,pc);
            r.qir[idx]=qir; r.qib[idx]=qib;
            int i,jj,ii; float d1,d4,d5;
            p3_find_ice(r.qi[idx],r.ni[idx],r.qir[idx],rhop,
                        i,jj,ii,d1,d4,d5);
            const float f02=p3_tab_interp(r.itab,jj,ii,i,2,d1,d4,d5);
            const float f06=p3_tab_interp(r.itab,jj,ii,i,6,d1,d4,d5);
            const float f09=p3_tab_interp(r.itab,jj,ii,i,7,d1,d4,d5);
            const float f10=p3_tab_interp(r.itab,jj,ii,i,8,d1,d4,d5);
            const float f13=p3_tab_interp(r.itab,jj,ii,i,9,d1,d4,d5);
            const float f15=p3_tab_interp(r.itab,jj,ii,i,11,d1,d4,d5);
            const float f16=p3_tab_interp(r.itab,jj,ii,i,12,d1,d4,d5);
            r.ni[idx]=fmin(r.ni[idx],f09*r.qi[idx]);
            r.ni[idx]=fmax(r.ni[idx],f10*r.qi[idx]);
            if (r.qir[idx]<P3_QSMALL) { r.qir[idx]=0.0f; r.qib[idx]=0.0f; }
            r.vmi[idx]=f02*r.rhofaci[idx];
            r.effi[idx]=f06; r.di[idx]=f15; r.rhopo[idx]=f16;
            ze_ice=ze_ice+0.1892f*f13*r.ni[idx]*rho;
            ze_ice=fmax(ze_ice,1.0e-22f);
        } else {
            r.qv[idx]=r.qv[idx]+r.qi[idx];
            r.theta[idx]=r.theta[idx]-invexn*r.qi[idx]*xxls*P3_INV_CP;
            r.qi[idx]=0.0f; r.ni[idx]=0.0f;
            r.qir[idx]=0.0f; r.qib[idx]=0.0f; r.di[idx]=0.0f;
        }
        r.zdbz[idx]=10.0f*p3_log10((ze_rain+ze_ice)*1.0e18f);
        if (r.qr[idx]<P3_QSMALL) r.nr[idx]=0.0f;
        r.effc[idx]=r.effc[idx]*1.0e6f;
        r.effi[idx]=r.effi[idx]*1.0e6f;
    }
}
