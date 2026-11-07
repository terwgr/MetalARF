#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_acoustic_advance_mu_th_bindings.metalh"
#include "wrf_exact_acoustic_frame_mu_t_bindings.metalh"
using namespace metal;
#define I3S(k,j,i,ny,nx) (((ulong)(k)*(ny)+(j))*(nx)+(i))
#define I3(k,j,i,ny,nx) (((ulong)(k)*(ny)+(j))*(nx)+(i))
#define IDX3(k,j,i) I3(k,j,i,ny,nx)
#define __fdiv_rn exact_div
namespace wrf_mu {

static inline
Exact32 wrf_total_divergence(int k, int j, int i, int ny, int nx,
                         device const Exact32* u_pp, device const Exact32* v_pp,
                         device const Exact32* u, device const Exact32* v,
                         device const Exact32* c1h, device const Exact32* c2h,
                         Exact32 muw, Exact32 mue, Exact32 mus, Exact32 mun,
                         Exact32 msfu_w, Exact32 msfu_e, Exact32 msfv_s, Exact32 msfv_n,
                         Exact32 msft_c, Exact32 rdx, Exact32 rdy)
{
    ulong uw = I3S(k,j,i,ny,nx+1), ue = uw+1;
    ulong vs = I3S(k,j,i,ny+1,nx), vn = vs+nx;
    Exact32 vs_full = v_pp[vs] + (c1h[k]*mus+c2h[k])*v[vs]*msfv_s;
    Exact32 vn_full = v_pp[vn] + (c1h[k]*mun+c2h[k])*v[vn]*msfv_n;
    Exact32 uw_full = u_pp[uw] + __fdiv_rn((c1h[k]*muw+c2h[k])*u[uw],msfu_w);
    Exact32 ue_full = u_pp[ue] + __fdiv_rn((c1h[k]*mue+c2h[k])*u[ue],msfu_e);
    return msft_c*msft_c*(rdy*(vn_full-vs_full)+rdx*(ue_full-uw_full));
}
static
void wrf_advance_mu_theta(device const Exact32* u_pp, device const Exact32* v_pp,
                          device const Exact32* u, device const Exact32* v, device const Exact32* mup,
                          device Exact32* mu_pp, device Exact32* mu_old, device const Exact32* rmu_t,
                          device Exact32* mudf, int write_mudf, device const Exact32* theta,
                          device Exact32* th_pp, device Exact32* th_old, device const Exact32* rth_t,
                          device Exact32* ww_pp, device const Exact32* ww_ref,
                          device const Exact32* p_pp, device Exact32* p_old,
                          device const Exact32* dnw, device const Exact32* rdnw,
                          device const Exact32* fnm, device const Exact32* fnp,
                          device const Exact32* c1h, device const Exact32* c2h, device const Exact32* mub2d,
                          device const Exact32* msft, device const Exact32* msfu, device const Exact32* msfv,
                          int mapped, Exact32 rdx, Exact32 rdy, Exact32 dtau,
                          int nz, int ny, int nx, int open_x, int open_y,
                          int spec_zone, uint tid)
{
    int c = int(tid);
    if (c >= ny*nx) return;
    int j=c/nx, i=c-j*nx;
    ulong st=(ulong)ny*nx;
    mu_old[c]=mu_pp[c];
    for (int k=0;k<nz;++k) {
        ulong h=(ulong)k*st+c;
        th_old[h]=th_pp[h]; p_old[h]=p_pp[h];
    }
    if (i<spec_zone || i>=nx-spec_zone || j<spec_zone || j>=ny-spec_zone) {
        if (write_mudf) mudf[c]=Exact32(0.0f);
        return;
    }
    int ip=open_x?min(i+1,nx-1):(i+1)%nx;
    int im=open_x?max(i-1,0):(i-1+nx)%nx;
    int jp=open_y?min(j+1,ny-1):(j+1)%ny;
    int jm=open_y?max(j-1,0):(j-1+ny)%ny;
    ulong cw=(ulong)j*nx+im, ce=(ulong)j*nx+ip;
    ulong cs=(ulong)jm*nx+i, cn=(ulong)jp*nx+i;
    Exact32 muw=Exact32(0.5f)*(mup[c]+mup[cw]+mub2d[c]+mub2d[cw]);
    Exact32 mue=Exact32(0.5f)*(mup[ce]+mup[c]+mub2d[ce]+mub2d[c]);
    Exact32 mus=Exact32(0.5f)*(mup[c]+mup[cs]+mub2d[c]+mub2d[cs]);
    Exact32 mun=Exact32(0.5f)*(mup[cn]+mup[c]+mub2d[cn]+mub2d[c]);
    Exact32 mc=mapped?msft[c]:Exact32(1.0f);
    Exact32 uxw=mapped?msfu[(ulong)j*(nx+1)+i]:Exact32(1.0f);
    Exact32 uxe=mapped?msfu[(ulong)j*(nx+1)+i+1]:Exact32(1.0f);
    Exact32 vys=mapped?msfv[(ulong)j*nx+i]:Exact32(1.0f);
    Exact32 vyn=mapped?msfv[(ulong)(j+1)*nx+i]:Exact32(1.0f);
    Exact32 dmdt=Exact32(0.0f);
    for (int k=0;k<nz;++k) {
        Exact32 dvdxi=wrf_total_divergence(k,j,i,ny,nx,u_pp,v_pp,u,v,c1h,c2h,
                              muw,mue,mus,mun,uxw,uxe,vys,vyn,mc,rdx,rdy);
        dmdt=dmdt+dnw[k]*dvdxi;
    }
    mu_pp[c]=mu_pp[c]+dtau*(dmdt+rmu_t[c]);
    if (write_mudf) mudf[c]=dmdt+rmu_t[c];
    Exact32 total_ww=Exact32(0.0f);
    ww_pp[c]=total_ww-ww_ref[c];
    for (int kk=1;kk<nz;++kk) {
        int k=kk-1;
        Exact32 dvdxi=wrf_total_divergence(k,j,i,ny,nx,u_pp,v_pp,u,v,c1h,c2h,
                              muw,mue,mus,mun,uxw,uxe,vys,vyn,mc,rdx,rdy);
        total_ww=total_ww-__fdiv_rn(dnw[k]*(c1h[k]*dmdt+dvdxi+c1h[k]*rmu_t[c]),mc);
        ww_pp[(ulong)kk*st+c]=total_ww-ww_ref[(ulong)kk*st+c];
    }
    ww_pp[(ulong)nz*st+c]=Exact32(0.0f);
    Exact32 wdtn=Exact32(0.0f);
    for (int k=0;k<nz;++k) {
        ulong h=(ulong)k*st+c;
        Exact32 wdtn_up=Exact32(0.0f);
        if (k+1<nz) wdtn_up=ww_pp[h+st]*(fnm[k+1]*theta[h+st]+fnp[k+1]*theta[h]);
        Exact32 hy=Exact32(0.5f)*rdy*(v_pp[I3S(k,j+1,i,ny+1,nx)]*(theta[IDX3(k,jp,i)]+theta[h])
                         -v_pp[I3S(k,j,i,ny+1,nx)]*(theta[h]+theta[IDX3(k,jm,i)]));
        Exact32 hx=Exact32(0.5f)*rdx*(u_pp[I3S(k,j,i+1,ny,nx+1)]*(theta[IDX3(k,j,ip)]+theta[h])
                         -u_pp[I3S(k,j,i,ny,nx+1)]*(theta[h]+theta[IDX3(k,j,im)]));
        th_pp[h]=th_pp[h]+mc*dtau*rth_t[h];
        th_pp[h]=th_pp[h]-dtau*mc*(mc*(hy+hx)+rdnw[k]*(wdtn_up-wdtn));
        wdtn=wdtn_up;
    }
}

void advance_exact_frame_mu_t(device Exact32*  mu_pp,
                              device Exact32*  th_pp,
                              device const Exact32*  rmu_t,
                              device const Exact32*  rth_t,
                              Exact32 dtau, int spec_zone, int periodic_x,
                              int nz, int ny, int nx,
                              device Exact32*  muts,
                              device const Exact32*  mub2d,
                              device const Exact32*  mup, uint tid)
{
    int c=int(tid);
    if (c>=ny*nx) return;
    int j=c/nx, i=c-j*nx;
    bool frame=j<spec_zone || j>=ny-spec_zone
                || (!periodic_x && (i<spec_zone || i>=nx-spec_zone));
    if (!frame) {
        muts[c]=(mub2d[c]+mup[c])+mu_pp[c];
        return;
    }
    muts[c]=muts[c]+dtau*rmu_t[c];
    mu_pp[c]=mu_pp[c]+dtau*rmu_t[c];
    ulong st=(ulong)ny*nx;
    for (int k=0;k<nz;++k) {
        ulong h=(ulong)k*st+c;
        th_pp[h]=th_pp[h]+dtau*rth_t[h];
    }
}

}
#define A(n) reinterpret_cast<device const Exact32 *>(r.n)
#define O(n) reinterpret_cast<device Exact32 *>(r.n)
kernel void wrf_exact_acoustic_advance_mu_th(constant WrfExactAcousticAdvanceMuThResources &r [[buffer(0)]], constant WrfExactAcousticAdvanceMuThParams &q [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
 if (ulong(tid)>=ulong(q.ny)*q.nx) return;
 wrf_mu::wrf_advance_mu_theta(A(u_pp),A(v_pp),A(u),A(v),A(mup),O(mu_pp),O(mu_pp_old),A(rmu_t),O(mudf),q.write_mudf,A(thp),O(th_pp),O(th_pp_old),A(rth_t),O(ww_pp),A(ww_ref),A(p_pp),O(p_pp_old),A(dnw),A(rdnw),A(fnm),A(fnp),A(c1h),A(c2h),A(mub2d),A(msft),A(msfu),A(msfv),q.mapped,Exact32(q.rdx),Exact32(q.rdy),Exact32(q.dtau),q.nz,q.ny,q.nx,q.open_x,q.open_y,q.spec_zone,tid);
}
kernel void wrf_exact_acoustic_frame_mu_t(constant WrfExactAcousticFrameMuTResources &r [[buffer(0)]], constant WrfExactAcousticFrameMuTParams &q [[buffer(1)]], uint tid [[thread_position_in_grid]]) {
 if (ulong(tid)>=ulong(q.ny)*q.nx) return;
 wrf_mu::advance_exact_frame_mu_t(O(mu_pp),O(th_pp),A(rmu_t),A(rth_t),Exact32(q.dtau),q.spec_zone,q.periodic_x,q.nz,q.ny,q.nx,O(muts),A(mub2d),A(mup),tid);
}
#undef A
#undef O
