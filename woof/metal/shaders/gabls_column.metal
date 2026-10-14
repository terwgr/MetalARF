#include <metal_stdlib>
#include "../generated/gabls_column_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"
using namespace metal;

// Exact binary64 GABLS1 column experiment. Every prognostic value is an
// IEEE-754 word in a ulong buffer; all arithmetic stays in soft64.
#pragma clang fp contract(off)

static inline ulong gbadd(ulong a, ulong b) { return soft64_add(a,b); }
static inline ulong gbsub(ulong a, ulong b) { return soft64_sub(a,b); }
static inline ulong gbmul(ulong a, ulong b) { return soft64_mul(a,b); }
static inline ulong gbdiv(ulong a, ulong b) { return soft64_div(a,b); }
static inline ulong gbmin(ulong a, ulong b) { return soft64_less(a,b)?a:b; }
static inline ulong gbmax(ulong a, ulong b) { return soft64_greater(a,b)?a:b; }
static inline ulong gbhalf() { return 0x3fe0000000000000ul; }
static inline ulong gbzero() { return 0ul; }
static inline ulong gbone() { return 0x3ff0000000000000ul; }
static inline ulong gbsqrt(ulong a) { return soft64_sqrt(a); }
static inline ulong gbhypot(ulong a, ulong b) {
    a=soft64_abs_bits(a); b=soft64_abs_bits(b);
    ulong hi=gbmax(a,b), lo=gbmin(a,b);
    if (soft64_is_zero(hi)) return gbzero();
    ulong ratio=gbdiv(lo,hi);
    return gbmul(hi,gbsqrt(gbadd(gbone(),gbmul(ratio,ratio))));
}

static inline ulong gb_blackadar(ulong z, ulong z0, ulong kappa, ulong lambda) {
    ulong kz=gbmul(kappa,gbadd(z,z0));
    return gbdiv(kz,gbadd(gbone(),gbdiv(kz,lambda)));
}

static inline ulong gb_crossing(ulong c1, ulong c2, ulong rem, ulong h,
                                ulong tol_factor) {
    ulong nan=soft64_qnan();
    ulong disc=gbadd(gbmul(c1,c1),gbmul(gbmul(0x4010000000000000ul,c2),rem));
    if (soft64_less(disc,gbzero())) return nan;
    ulong sq=gbsqrt(disc);
    ulong signed_sq=soft64_greater_equal(c1,gbzero())?sq:soft64_negate_bits(sq);
    ulong q=gbmul(soft64_negate_bits(gbhalf()),gbadd(c1,signed_sq));
    ulong best=nan;
    ulong tol=gbmul(tol_factor,h);
    if (!soft64_equal(c2,gbzero())) {
        ulong a=gbdiv(q,c2);
        if (soft64_greater_equal(a,soft64_negate_bits(tol)) &&
            soft64_less_equal(a,gbadd(h,tol))) best=gbmax(gbzero(),gbmin(a,h));
    }
    if (!soft64_equal(c2,gbzero())) {
        ulong b=soft64_equal(q,gbzero())?gbzero():gbdiv(soft64_negate_bits(rem),q);
        if ((soft64_equal(q,gbzero()) ||
             (soft64_greater_equal(b,soft64_negate_bits(tol)) &&
              soft64_less_equal(b,gbadd(h,tol)))) &&
            (soft64_is_nan(best) || soft64_less(b,best)))
            best=gbmax(gbzero(),gbmin(b,h));
    } else if (soft64_greater(c1,gbzero())) {
        ulong lin=gbdiv(rem,c1);
        if (soft64_greater_equal(lin,soft64_negate_bits(tol)) &&
            soft64_less_equal(lin,gbadd(h,tol)) &&
            (soft64_is_nan(best) || soft64_less(lin,best)))
            best=gbmax(gbzero(),gbmin(lin,h));
    }
    return best;
}

static inline void gb_lengths(device const ulong *theta, device const ulong *e,
        device const ulong *z, ulong dz, ulong z0, ulong g, ulong kappa,
        ulong lambda, ulong lscoef, ulong emin, ulong mix_exp,
    thread ulong *lmix_out, thread ulong *leps_out, int nz) {
    ulong n2[128], up[128], dn[128];
    for (int k=0;k<nz;k++) {
        ulong dth;
        if (k==0) dth=gbdiv(gbsub(theta[1],theta[0]),dz);
        else if (k==nz-1) dth=gbdiv(gbsub(theta[nz-1],theta[nz-2]),dz);
        else dth=gbdiv(gbsub(theta[k+1],theta[k-1]),gbmul(0x4000000000000000ul,dz));
        n2[k]=gbmul(gbdiv(g,theta[k]),dth);
    }
    ulong top=gbadd(z[nz-1],gbmul(gbhalf(),dz));
    for (int k=0;k<nz;k++) {
        ulong ek=gbmax(e[k],emin);
        ulong beta=gbdiv(g,theta[k]);
        for (int side=0;side<2;side++) {
            bool down=(side==1); ulong rem=ek, acc=gbzero();
            ulong prev_z=z[k], prev_t=theta[k], got=soft64_qnan();
            int first=down?k-1:k+1, stop=down?-1:nz;
            int j=first;
            while (down ? j>=stop : j<=stop) {
                ulong nzj, ntj, h, c1, c2d;
                if (down) {
                    nzj=(j>=0)?z[j]:gbzero(); ntj=(j>=0)?theta[j]:theta[0];
                    h=gbsub(prev_z,nzj);
                    c1=gbmul(beta,gbsub(theta[k],prev_t));
                    c2d=gbmul(beta,gbsub(prev_t,ntj));
                } else {
                    nzj=(j<nz)?z[j]:top; ntj=(j<nz)?theta[j]:theta[nz-1];
                    h=gbsub(nzj,prev_z);
                    c1=gbmul(beta,gbsub(prev_t,theta[k]));
                    c2d=gbmul(beta,gbsub(ntj,prev_t));
                }
                if (soft64_greater(h,gbzero())) {
                    ulong c2=gbdiv(c2d,gbmul(0x4000000000000000ul,h));
                    ulong s=gb_crossing(c1,c2,rem,h,0x3d719799812dea11ul); // 1e-12
                    if (!soft64_is_nan(s)) { got=gbadd(acc,s); break; }
                    ulong seg=gbmul(gbadd(c1,gbmul(c2,h)),h);
                    rem=gbmax(gbsub(rem,seg),gbzero());
                    acc=gbadd(acc,h);
                }
                prev_z=nzj; prev_t=ntj; j+=down?-1:1;
            }
            if (soft64_is_nan(got)) got=down?z[k]:gbsub(top,z[k]);
            if (down) dn[k]=got; else up[k]=got;
        }
    }
    ulong p=mix_exp;
    ulong invp=gbdiv(gbone(),p);
    for (int k=0;k<nz;k++) {
        ulong lmix_bl=soft64_plm_pow(gbmul(gbhalf(),gbadd(
            soft64_plm_pow(up[k],soft64_negate_bits(p)),
            soft64_plm_pow(dn[k],soft64_negate_bits(p)))),soft64_negate_bits(invp));
        ulong leps_bl=gbmin(up[k],dn[k]);
        ulong l_b=gb_blackadar(z[k],z0,kappa,lambda);
        ulong ek=gbmax(e[k],emin);
        ulong lmix=l_b;
        if (soft64_greater(n2[k],gbzero())) {
            ulong ls=gbdiv(gbmul(lscoef,gbsqrt(ek)),gbsqrt(n2[k]));
            lmix=gbmin(lmix,ls);
        }
        lmix_out[k]=gbmin(lmix,lmix_bl);
        leps_out[k]=gbmin(l_b,leps_bl);
    }
}

static inline ulong gb_kv(ulong lmix, ulong e, ulong n2, ulong ckv,
                          ulong cks, ulong lscoef, ulong cexp, ulong emin) {
    ulong ef=gbmax(e,emin);
    ulong cr=ckv;
    if (soft64_greater(n2,gbzero())) {
        ulong ls=gbdiv(gbmul(lscoef,gbsqrt(ef)),gbsqrt(n2));
        ulong rho=gbmin(gbdiv(lmix,ls),gbone());
        ulong w=soft64_plm_pow(rho,cexp);
        cr=gbadd(ckv,gbmul(gbsub(gbdiv(cks,lscoef),ckv),w));
    }
    return gbmul(gbmul(cr,lmix),gbsqrt(ef));
}

static inline void gb_diffuse(thread ulong *phi, device const ulong *kf,
        ulong dt, ulong dz, ulong scale, bool divide_scale,
        ulong drag, bool use_drag, int nz) {
    ulong sub[128], sup[128], diag[128], cp[128], dp[128];
    for (int k=0;k<nz;k++) { sub[k]=gbzero(); sup[k]=gbzero(); }
    for (int k=0;k<nz-1;k++) {
        ulong kused=divide_scale?gbdiv(kf[k],scale):gbmul(kf[k],scale);
        ulong r=gbdiv(gbmul(dt,kused),dz);
        sub[k+1]=soft64_negate_bits(gbdiv(r,dz));
        sup[k]=soft64_negate_bits(gbdiv(r,dz));
    }
    for (int k=0;k<nz;k++) diag[k]=gbsub(gbsub(gbone(),sub[k]),sup[k]);
    if(use_drag) diag[0]=gbadd(diag[0],gbdiv(gbmul(dt,drag),dz));
    cp[0]=gbdiv(sup[0],diag[0]); dp[0]=gbdiv(phi[0],diag[0]);
    for (int k=1;k<nz;k++) {
        ulong m=gbsub(diag[k],gbmul(sub[k],cp[k-1]));
        cp[k]=gbdiv(sup[k],m);
        dp[k]=gbdiv(gbsub(phi[k],gbmul(sub[k],dp[k-1])),m);
    }
    phi[nz-1]=dp[nz-1];
    for (int k=nz-2;k>=0;k--) phi[k]=gbsub(dp[k],gbmul(cp[k],phi[k+1]));
}

// Constants slots: dt,dz,fcor,geo_u,geo_v,z0,th_surface,cooling_rate,
// kappa,g,Blackadar lambda,LS coefficient,C_KV,C_KS,CKS exponent,Pr,
// C_E,C_ES,C_ED,BL89 exponent.
kernel void gabls_column_step(
    constant GablsColumnStepResources &r [[buffer(0)]],
    constant GablsColumnStepParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    device ulong *u=r.u, *v=r.v, *theta=r.theta, *e=r.e;
    device ulong *kv=r.kv, *kface=r.kface, *skin=r.skin, *series=r.series;
    device const ulong *z=r.z, *c=r.c;
    uint nz=uint(p.nz), step_index=uint(p.step_index);
    if (tid!=0u || p.nz<2 || p.nz>128 || p.steps<1 ||
        p.step_index<0 || p.step_index>=p.steps) return;
    const ulong dt=c[0], dz=c[1], fcor=c[2], geo_u=c[3], geo_v=c[4];
    const ulong z0=c[5], kappa=c[8], g=c[9], lambda=c[10], lscoef=c[11];
    const ulong ckv=c[12], cks=c[13], cexp=c[14], pr=c[15];
    const ulong one_half=gbhalf(), e_floor=c[25];
    ulong spd=gbhypot(u[0],v[0]);
    spd=gbmax(spd,c[20]); // SFC_WSPD_FLOOR
    ulong dth=gbsub(theta[0],skin[0]);
    ulong lnm=soft64_plm_log(gbdiv(z[0],z0));
    ulong lnh=lnm;
    ulong ust=gbdiv(gbmul(kappa,spd),lnm);
    ulong tst=gbdiv(gbmul(kappa,dth),lnh);
    if (!soft64_greater(dth,gbzero())) tst=gbzero();
    else for(int it=0;it<20;it++) {
        if (!soft64_greater(tst,gbzero()) || !soft64_greater(ust,gbzero())) {
            ust=gbdiv(gbmul(kappa,spd),lnm); tst=gbzero(); break;
        }
        ulong lmo=gbdiv(gbmul(gbmul(ust,ust),theta[0]),
                        gbmul(gbmul(kappa,g),tst));
        ulong dm=gbadd(lnm,gbdiv(gbmul(c[21],gbsub(z[0],z0)),lmo));
        ulong dh=gbadd(lnh,gbdiv(gbmul(c[22],gbsub(z[0],z0)),lmo));
        if (!soft64_greater(dm,gbzero()) || !soft64_greater(dh,gbzero())) {
            ust=gbzero(); tst=gbzero(); break;
        }
        ust=gbdiv(gbmul(kappa,spd),dm);
        tst=gbdiv(gbmul(kappa,dth),dh);
    }
    const uint off=4u*step_index;
    series[off+0u]=ust; series[off+1u]=tst;
    series[off+2u]=skin[0];
    series[off+3u]=gbmul(soft64_negate_bits(ust),tst);
    ulong n2[128], lmix[128], leps[128];
    // length helper recomputes n2 internally; retain it here for K and e.
    for (uint k=0;k<nz;k++) {
        ulong grad=(k==0u)?gbdiv(gbsub(theta[1],theta[0]),dz):
            ((k==nz-1u)?gbdiv(gbsub(theta[nz-1u],theta[nz-2u]),dz):
             gbdiv(gbsub(theta[k+1u],theta[k-1u]),gbmul(0x4000000000000000ul,dz)));
        n2[k]=gbmul(gbdiv(g,theta[k]),grad);
    }
    gb_lengths(theta,e,z,dz,z0,g,kappa,lambda,lscoef,c[25],c[19],lmix,leps,int(nz));
    for (uint k=0;k<nz;k++) kv[k]=gb_kv(lmix[k],e[k],n2[k],ckv,cks,lscoef,cexp,c[25]);
    for (uint k=0;k+1u<nz;k++) kface[k]=gbmul(one_half,gbadd(kv[k],kv[k+1u]));
    ulong drag=gbdiv(gbmul(ust,ust),spd);
    // Simultaneous explicit Coriolis/PGF tendencies at the pre-update state.
    ulong unew[128], vnew[128];
    for (uint k=0;k<nz;k++) {
        ulong du=gbmul(gbmul(dt,fcor),gbsub(v[k],geo_v));
        ulong dv=gbmul(gbmul(soft64_negate_bits(dt),fcor),gbsub(u[k],geo_u));
        unew[k]=gbadd(u[k],du); vnew[k]=gbadd(v[k],dv);
    }
    gb_diffuse(unew,kface,dt,dz,gbone(),false,drag,true,int(nz));
    gb_diffuse(vnew,kface,dt,dz,gbone(),false,drag,true,int(nz));
    for(uint k=0;k<nz;k++){u[k]=unew[k];v[k]=vnew[k];}
    ulong thnew[128], enew[128];
    for(uint k=0;k<nz;k++) thnew[k]=theta[k];
    thnew[0]=gbadd(thnew[0],gbdiv(gbmul(dt,series[off+3u]),dz));
    gb_diffuse(thnew,kface,dt,dz,pr,true,gbzero(),false,int(nz));
    for(uint k=0;k<nz;k++) theta[k]=thnew[k];
    ulong pv[128]; for(uint k=0;k<nz;k++)pv[k]=gbzero();
    for(uint k=0;k+1u<nz;k++) {
        ulong du=gbsub(u[k+1u],u[k]), dv=gbsub(v[k+1u],v[k]);
        ulong eps=gbdiv(gbmul(gbmul(kface[k],du),du),dz);
        eps=gbadd(eps,gbdiv(gbmul(gbmul(kface[k],dv),dv),dz));
        eps=gbmul(one_half,eps);
        pv[k]=gbadd(pv[k],gbdiv(eps,dz)); pv[k+1u]=gbadd(pv[k+1u],gbdiv(eps,dz));
    }
    for(uint k=0;k<nz;k++) {
        ulong ek=gbmax(e[k],c[25]);
        ulong buoy=soft64_negate_bits(gbmul(gbdiv(kv[k],pr),n2[k]));
        ulong estar=gbadd(e[k],gbmul(dt,gbadd(pv[k],buoy)));
        ulong ld=gbmin(leps[k],gbdiv(gbmul(lscoef,gbsqrt(ek)),
                                      gbsqrt(soft64_greater(n2[k],gbzero())?n2[k]:gbone())));
        if (!soft64_greater(n2[k],gbzero())) ld=leps[k];
        ulong ce=c[16];
        if(c[23]!=gbzero() && soft64_greater(n2[k],gbzero())) {
            ulong ls=gbdiv(gbmul(lscoef,gbsqrt(ek)),gbsqrt(n2[k]));
            ulong rho=gbmin(gbdiv(ld,ls),gbone());
            ulong w=soft64_plm_pow(rho,cexp);
            ce=gbadd(gbmul(gbsub(gbone(),w),c[16]),gbmul(w,c[17]));
        }
        if(c[24]!=gbzero() && soft64_greater(n2[k],gbzero())) {
            ulong ls=gbdiv(gbmul(lscoef,gbsqrt(ek)),gbsqrt(n2[k]));
            ulong rho=gbmin(gbdiv(ld,ls),gbone());
            ulong w=soft64_plm_pow(rho,cexp);
            ulong lb=gb_blackadar(z[k],z0,kappa,lambda);
            ulong added=gbmul(gbmul(w,c[18]),gbdiv(ld,lb));
            ce=gbadd(ce,added);
        }
        ulong estarf=gbmax(estar,c[25]);
        ulong b=gbdiv(gbmul(ce,gbsqrt(estarf)),gbmul(0x4000000000000000ul,ld));
        ulong decay_base=gbadd(gbone(),gbmul(b,dt));
        enew[k]=gbmax(gbdiv(estar,gbmul(decay_base,decay_base)),e_floor);
    }
    gb_diffuse(enew,kface,dt,dz,0x4000000000000000ul,false,
               gbzero(),false,int(nz));
    for(uint k=0;k<nz;k++)e[k]=gbmax(enew[k],c[25]);
    skin[0]=gbsub(skin[0],gbmul(dt,c[7]));
}

kernel void gabls_column_finish(
    constant GablsColumnFinishResources &r [[buffer(0)]],
    constant GablsColumnFinishParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    device const ulong *u=r.u,*v=r.v,*theta=r.theta,*e=r.e;
    device ulong *kv=r.kv,*kface=r.kface,*stress=r.stress;
    device const ulong *z=r.z,*c=r.c;
    uint nz=uint(p.nz);
    if(tid!=0u || nz<2u || nz>128u)return;
    ulong n2[128],lmix[128],leps[128];
    for(uint k=0;k<nz;k++){
        ulong grad=(k==0u)?gbdiv(gbsub(theta[1],theta[0]),c[1]):
            ((k==nz-1u)?gbdiv(gbsub(theta[nz-1u],theta[nz-2u]),c[1]):
             gbdiv(gbsub(theta[k+1u],theta[k-1u]),gbmul(0x4000000000000000ul,c[1])));
        n2[k]=gbmul(gbdiv(c[9],theta[k]),grad);
    }
    gb_lengths(theta,e,z,c[1],c[5],c[9],c[8],c[10],c[11],c[25],c[19],lmix,leps,int(nz));
    for(uint k=0;k<nz;k++)kv[k]=gb_kv(lmix[k],e[k],n2[k],c[12],c[13],c[11],c[14],c[25]);
    for(uint k=0;k+1u<nz;k++)kface[k]=gbmul(gbhalf(),gbadd(kv[k],kv[k+1u]));
    ulong pr=c[15];
    for(uint k=0;k+1u<nz;k++){
        ulong dz=c[1];
        ulong uw=gbdiv(gbmul(soft64_negate_bits(kface[k]),gbsub(u[k+1u],u[k])),dz);
        ulong vw=gbdiv(gbmul(soft64_negate_bits(kface[k]),gbsub(v[k+1u],v[k])),dz);
        ulong wt=gbdiv(gbmul(soft64_negate_bits(gbdiv(kface[k],pr)),
                             gbsub(theta[k+1u],theta[k])),dz);
        stress[3u*k]=uw; stress[3u*k+1u]=vw; stress[3u*k+2u]=wt;
    }
}
