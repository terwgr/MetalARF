// Source fragment included after the shared WRF grid/deformation helpers.
// Typed bindings are generated from smag2d_alternates_fragment.json. This
// fragment deliberately supports the dry profile only; the caller refuses
// moist states until the moist N2 branches are translated.
#include "smag2d_alternates_fragment_bindings.metalh"

kernel void wrf_calc_n2(constant WrfCalcN2Resources &r [[buffer(0)]],
                        constant WrfCalcN2Params &p [[buffer(1)]],
                        uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny));
    const int k=int(flat/uint(p.nx*p.ny));
    if((p.boundary_x&&(i==0||i==p.nx-1))
       ||(p.boundary_y&&(j==0||j==p.ny-1))){r.bn2[flat]=0.0f;return;}
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const int ke=(k==p.nz-1&&p.nz>1)?p.nz-2:k;
    const ulong h=sm2_i3(ke,j,i,p.ny,p.nx);
    const float base_c=p.thb3d?r.thb[h]:r.thb[ke];
    const float th_c=base_c+r.thp[h];
    float value;
    if(ke>0){
        const float dz=1.0f/wrf_sm2_rdz(q,ke,j,i)
                      +1.0f/wrf_sm2_rdz(q,ke+1,j,i);
        const ulong hp=sm2_i3(ke+1,j,i,p.ny,p.nx);
        const ulong hm=sm2_i3(ke-1,j,i,p.ny,p.nx);
        const float bp=p.thb3d?r.thb[hp]:r.thb[ke+1];
        const float bm=p.thb3d?r.thb[hm]:r.thb[ke-1];
        const float th_p=bp+r.thp[hp],th_m=bm+r.thp[hm];
        value=9.81f*((th_p-th_m)/th_c/dz);
    }else{
        const ulong h1=sm2_i3(1,j,i,p.ny,p.nx);
        const float b1=p.thb3d?r.thb[h1]:r.thb[1];
        const float th1=b1+r.thp[h1];
        value=9.81f*((th1-th_c)/th_c/wrf_sm2_rdzw(q,0,j,i));
    }
    r.bn2[flat]=value;
}

kernel void wrf_smag3d_km(constant WrfSmag3DKmResources &r [[buffer(0)]],
                          constant WrfSmag3DKmParams &p [[buffer(1)]],
                          uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny)*ulong(p.nx);
    if(ulong(flat)>=count)return;
    const int i=int(flat%uint(p.nx)),j=int((flat/uint(p.nx))%uint(p.ny));
    const int k=int(flat/uint(p.nx*p.ny));
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const float d11=wrf_sm2_d(r.d11,q,k,j,i);
    const float d22=wrf_sm2_d(r.d22,q,k,j,i);
    const float d33=2.0f*(q.w[sm2_i3(k+1,j,i,p.ny,p.nx)]
                          -q.w[sm2_i3(k,j,i,p.ny,p.nx)])
                     *wrf_sm2_rdzw(q,k,j,i);
    const float d12=0.25f*(wrf_sm2_d(r.d12,q,k,j,i)
        +wrf_sm2_d(r.d12,q,k,j+1,i)+wrf_sm2_d(r.d12,q,k,j,i+1)
        +wrf_sm2_d(r.d12,q,k,j+1,i+1));
    const float d13=0.25f*(wrf_sm2_tau_def13(q,k+1,j,i)
        +wrf_sm2_tau_def13(q,k,j,i)+wrf_sm2_tau_def13(q,k+1,j,i+1)
        +wrf_sm2_tau_def13(q,k,j,i+1));
    const float d23=0.25f*(wrf_sm2_tau_def23(q,k+1,j,i)
        +wrf_sm2_tau_def23(q,k,j,i)+wrf_sm2_tau_def23(q,k+1,j+1,i)
        +wrf_sm2_tau_def23(q,k,j+1,i));
    const float def2=0.5f*(d11*d11+d22*d22+d33*d33)
                    +d12*d12+d13*d13+d23*d23;
    const float strain=sqrt(max(0.0f,def2-r.bn2[flat]/p.prandtl));
    const float map=q.msft[ulong(j)*ulong(p.nx)+ulong(i)];
    const float dxm=p.dx/map,dym=p.dy/map,rdzw=wrf_sm2_rdzw(q,k,j,i);
    float kmh,kmv,khh,khv;
    if(p.isotropic==0){
        const float mh2=dxm*dym,mv=1.0f/rdzw,mv2=mv*mv;
        kmh=max(p.c_s*p.c_s*mh2*strain,1.0e-6f*mh2);
        kmh=min(kmh,p.mix_upper_bound*mh2/p.dt);
        kmv=max(p.c_s*p.c_s*mv2*strain,1.0e-6f*mv2);
        kmv=min(kmv,p.mix_upper_bound*mv2/p.dt);
        khh=min(kmh/p.prandtl,p.mix_upper_bound*mh2/p.dt);
        khv=min(kmv/p.prandtl,p.mix_upper_bound*mv2/p.dt);
    }else{
        const float delta=pow(dxm*dym/rdzw,0.33333333f),delta2=delta*delta;
        kmh=max(p.c_s*p.c_s*delta2*strain,1.0e-6f*delta2);
        kmh=min(kmh,p.mix_upper_bound*dxm*dym/p.dt);
        kmv=min(kmh,p.mix_upper_bound/rdzw/rdzw/p.dt);
        khh=min(kmh/p.prandtl,p.mix_upper_bound*dxm*dym/p.dt);
        khv=min(kmv/p.prandtl,p.mix_upper_bound/rdzw/rdzw/p.dt);
    }
    r.xkmh[flat]=kmh;r.xkmv[flat]=kmv;
    r.xkhh[flat]=khh;r.xkhv[flat]=khv;
}
