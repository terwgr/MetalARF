// WRF dry and moist buoyancy-frequency source fragment. Resources and
// parameters are declared in smag2d_alternates_fragment.json.
#include "smag2d_alternates_fragment_bindings.metalh"

static inline float wrf_n2_theta(constant WrfCalcN2Resources &r,
                                 constant WrfCalcN2Params &p,
                                 int k, int j, int i) {
    const ulong h=sm2_i3(k,j,i,p.ny,p.nx);
    return (p.thb3d?r.thb[h]:r.thb[k])+r.thp[h];
}

static inline float wrf_n2_qtot(constant WrfCalcN2Resources &r,
                                constant WrfCalcN2Params &p,
                                int k, int j, int i) {
    const ulong h=sm2_i3(k,j,i,p.ny,p.nx);
    float qt=p.moist?r.qv[h]:0.0f;
    if(p.has_qc) qt+=r.qc[h];
    if(p.has_qi) qt+=r.qi[h];
    return qt;
}

static inline float wrf_n2_qv(constant WrfCalcN2Resources &r,
                              constant WrfCalcN2Params &p,
                              int k, int j, int i) {
    return p.moist?r.qv[sm2_i3(k,j,i,p.ny,p.nx)]:0.0f;
}

static inline float wrf_n2_temp(float theta, float pressure) {
    return theta*pow(pressure/100000.0f,287.0f/1004.5f);
}

static inline float wrf_n2_qvs(float temperature, float pressure) {
    const float tc=temperature-273.15f;
    const float es=1000.0f*0.6112f*exp(17.67f*tc/(temperature-29.65f));
    return (287.0f/461.6f)*es/(pressure-es);
}

static inline float wrf_n2_coefa(float theta, float temperature, float qv) {
    const float xlvqv=2.5e6f*qv;
    return (1.0f+(xlvqv/287.0f)/temperature)
        /(1.0f+((2.5e6f*xlvqv/1004.5f)/461.6f)
                     /temperature/temperature)/theta;
}

static inline float wrf_n2_thetae(float theta, float temperature,
                                  float qvs) {
    return theta*(1.0f+(2.5e6f*qvs/1004.5f)/temperature);
}

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
    const float th_c=wrf_n2_theta(r,p,ke,j,i);
    const float p_c=r.p[h];
    const float t_c=wrf_n2_temp(th_c,p_c);
    const float qv_c=wrf_n2_qv(r,p,ke,j,i);
    const float qc_c=(p.moist&&p.has_qc)?r.qc[h]:0.0f;
    const float tc=t_c-273.15f;
    const float esc=1000.0f*0.6112f*exp(17.67f*tc/(t_c-29.65f));
    const float qvs_c=(287.0f/461.6f)*esc/(p_c-esc);
    const bool saturated=p.moist&&(qv_c>=qvs_c||qc_c>=1.0e-5f);
    float value;
    if(ke>0){
        const float tmpdz=1.0f/wrf_sm2_rdz(q,ke,j,i)
                         +1.0f/wrf_sm2_rdz(q,ke+1,j,i);
        const ulong hp=sm2_i3(ke+1,j,i,p.ny,p.nx);
        const ulong hm=sm2_i3(ke-1,j,i,p.ny,p.nx);
        const float th_p=wrf_n2_theta(r,p,ke+1,j,i);
        const float th_m=wrf_n2_theta(r,p,ke-1,j,i);
        const float qtot_p=wrf_n2_qtot(r,p,ke+1,j,i);
        const float qtot_m=wrf_n2_qtot(r,p,ke-1,j,i);
        if(saturated){
            const float t_p=wrf_n2_temp(th_p,r.p[hp]);
            const float t_m=wrf_n2_temp(th_m,r.p[hm]);
            const float thetaep1=wrf_n2_thetae(
                th_p,t_p,wrf_n2_qvs(t_p,r.p[hp]));
            const float thetaem1=wrf_n2_thetae(
                th_m,t_m,wrf_n2_qvs(t_m,r.p[hm]));
            value=9.81f*(wrf_n2_coefa(th_c,t_c,qv_c)
                         *(thetaep1-thetaem1)/tmpdz
                         -(qtot_p-qtot_m)/tmpdz);
        }else{
            const float qv_p=wrf_n2_qv(r,p,ke+1,j,i);
            const float qv_m=wrf_n2_qv(r,p,ke-1,j,i);
            value=9.81f*((th_p-th_m)/th_c/tmpdz
                         +1.61f*(qv_p-qv_m)/tmpdz
                         -(qtot_p-qtot_m)/tmpdz);
        }
    }else{
        const ulong h1=sm2_i3(1,j,i,p.ny,p.nx);
        const ulong h2=sm2_i3(2,j,i,p.ny,p.nx);
        const float th1=wrf_n2_theta(r,p,1,j,i);
        const float qtot_sfc=p.cf1*wrf_n2_qtot(r,p,0,j,i)
                             +p.cf2*wrf_n2_qtot(r,p,1,j,i)
                             +p.cf3*wrf_n2_qtot(r,p,2,j,i);
        const float qtot_p=wrf_n2_qtot(r,p,1,j,i);
        if(saturated){
            const float tmpdz=1.0f/wrf_sm2_rdz(q,1,j,i)
                             +0.5f/wrf_sm2_rdzw(q,0,j,i);
            const float th2=wrf_n2_theta(r,p,2,j,i);
            const float t1=wrf_n2_temp(th1,r.p[h1]);
            const float t2=wrf_n2_temp(th2,r.p[h2]);
            const float z0=wrf_sm2_phi(q,0,j,i)/9.81f;
            const float z1=0.5f*(wrf_sm2_phi(q,0,j,i)
                                 +wrf_sm2_phi(q,1,j,i))/9.81f;
            const float z2=0.5f*(wrf_sm2_phi(q,1,j,i)
                                 +wrf_sm2_phi(q,2,j,i))/9.81f;
            const float w1=(z0-z2)/(z1-z2),w2=1.0f-w1;
            const float p8w0=w1*p_c+w2*r.p[h1];
            const float t8w0=w1*t_c+w2*t1;
            const float theta_sfc=t8w0/pow(p8w0/100000.0f,287.0f/1004.5f);
            const float qvs_sfc=p.cf1*qvs_c
                +p.cf2*wrf_n2_qvs(t1,r.p[h1])
                +p.cf3*wrf_n2_qvs(t2,r.p[h2]);
            const float thetaesfc=wrf_n2_thetae(theta_sfc,t8w0,qvs_sfc);
            value=9.81f*(wrf_n2_coefa(th_c,t_c,qv_c)
                         *(wrf_n2_thetae(th1,t1,wrf_n2_qvs(t1,r.p[h1]))
                           -thetaesfc)/tmpdz
                         -(qtot_p-qtot_sfc)/tmpdz);
        }else{
            const float qv_sfc=p.cf1*wrf_n2_qv(r,p,0,j,i)
                              +p.cf2*wrf_n2_qv(r,p,1,j,i)
                              +p.cf3*wrf_n2_qv(r,p,2,j,i);
            const float tmpdz=1.0f/wrf_sm2_rdzw(q,0,j,i);
            value=9.81f*((th1-th_c)/th_c/tmpdz
                         +1.61f*(wrf_n2_qv(r,p,1,j,i)-qv_sfc)/tmpdz
                         -(qtot_p-qtot_sfc)/tmpdz);
        }
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
