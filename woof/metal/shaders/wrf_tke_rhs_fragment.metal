// WRF tke_rhs source, buoyancy, dissipation and positivity limiter.
static inline float wrf_tke_theta(thread const WrfSmagGrid &q,
                                  device const float *thp,
                                  device const float *thb, int thb3d,
                                  int k, int j, int i) {
    const int jj=wrf_sm2_iy(q,j), ii=wrf_sm2_ix(q,i);
    const ulong n=ulong(q.ny)*ulong(q.nx), c=ulong(jj)*ulong(q.nx)+ulong(ii);
    const float base=thb3d ? thb[ulong(k)*n+c] : thb[k];
    return thp[ulong(k)*n+c]+base;
}

static inline float wrf_tke_l(float tke,float bn2,float delta) {
    if (bn2>1.0e-6f) {
        const float l=0.76f*sqrt(max(tke,1.0e-6f))/sqrt(bn2);
        return max(0.001f*delta,min(l,delta));
    }
    return delta;
}

kernel void wrf_tke_rhs(constant WrfTkeRhsResources &r [[buffer(0)]],
                        constant WrfTkeRhsParams &p [[buffer(1)]],
                        uint flat [[thread_position_in_grid]]) {
    const ulong count=ulong(p.nz)*ulong(p.ny)*ulong(p.nx);
    if (ulong(flat)>=count) return;
    const int i=int(flat%uint(p.nx));
    const int j=int((flat/uint(p.nx))%uint(p.ny));
    const int k=int(flat/uint(p.nx*p.ny));
    if ((p.boundary_x&&(i==0||i==p.nx-1))
            ||(p.boundary_y&&(j==0||j==p.ny-1))) return;
    const WrfSmagGrid q=wrf_sm2_grid(r,p);
    const ulong ix=sm2_i3(k,j,i,p.ny,p.nx), col=ulong(j)*ulong(p.nx)+ulong(i);
    const float chm=r.c1[k]*r.mut[col]+r.c2[k];
    const float kmh=r.xkmh[ix], kmv=r.xkmv[ix];
    float t=0.0f;
    const float d11=wrf_sm2_d(r.d11,q,k,j,i);
    const float d22=wrf_sm2_d(r.d22,q,k,j,i);
    t+=0.5f*chm*kmh*d11*d11;
    t+=0.5f*chm*kmh*d22*d22;
    const float rdzw=wrf_sm2_rdzw(q,k,j,i);
    const float d33=2.0f*(q.w[sm2_i3(k+1,j,i,p.ny,p.nx)]
                         -q.w[sm2_i3(k,j,i,p.ny,p.nx)])*rdzw;
    t+=0.5f*chm*kmv*d33*d33;
    const float s12a=wrf_sm2_d(r.d12,q,k,j,i),s12b=wrf_sm2_d(r.d12,q,k,j+1,i);
    const float s12c=wrf_sm2_d(r.d12,q,k,j,i+1),s12d=wrf_sm2_d(r.d12,q,k,j+1,i+1);
    t+=chm*kmh*0.25f*(s12a*s12a+s12b*s12b+s12c*s12c+s12d*s12d);
    const float s13a=wrf_sm2_d13(q,k+1,j,i),s13b=wrf_sm2_d13(q,k,j,i);
    const float s13c=wrf_sm2_d13(q,k+1,j,i+1),s13d=wrf_sm2_d13(q,k,j,i+1);
    t+=chm*kmv*0.25f*(s13a*s13a+s13b*s13b+s13c*s13c+s13d*s13d);
    const float s23a=wrf_sm2_d23(q,k+1,j,i),s23b=wrf_sm2_d23(q,k,j,i);
    const float s23c=wrf_sm2_d23(q,k+1,j+1,i),s23d=wrf_sm2_d23(q,k,j+1,i);
    t+=chm*kmv*0.25f*(s23a*s23a+s23b*s23b+s23c*s23c+s23d*s23d);
    if (k==0) {
        const int il=wrf_sm2_iu(q,i),ir=wrf_sm2_iu(q,i+1);
        const int jb=wrf_sm2_jv(q,j),jt=wrf_sm2_jv(q,j+1);
        const float usum=q.u[sm2_i3(0,j,il,p.ny,p.nx+1)]+q.u[sm2_i3(0,j,ir,p.ny,p.nx+1)];
        const float vsum=q.v[sm2_i3(0,jb,i,p.ny+1,p.nx)]+q.v[sm2_i3(0,jt,i,p.ny+1,p.nx)];
        float absU=0.5f*sqrt(usum*usum+vsum*vsum);
        float cd;
        if (p.isfflx==0) cd=p.drag;
        else {
            absU+=1.0e-15f;
            const float us=p.has_ustm?r.ustm[col]:0.0f;
            cd=(us*us)/(absU*absU);
        }
        const float d13sum=0.5f*(wrf_sm2_d13(q,1,j,i)+wrf_sm2_d13(q,1,j,i+1));
        t+=chm*(0.5f*usum*cd*absU*d13sum);
        const float d23sum=0.5f*(wrf_sm2_d23(q,1,j,i)+wrf_sm2_d23(q,1,j+1,i));
        t+=chm*(0.5f*vsum*cd*absU*d23sum);
    }
    const float s_shear=t;
    const float khv=r.xkhv[ix];
    if (k>=1) t-=chm*khv*r.bn2[ix];
    else {
        float heat_flux;
        if (p.isfflx==0||p.isfflx==2) heat_flux=p.heat;
        else {
            const float vapor=q.moist?q.qv[ix]:0.0f;
            const float cpm=1004.5f*(1.0f+0.8f*vapor);
            const float hf=p.has_hfx?r.hfx[col]:0.0f;
            heat_flux=(hf/cpm)/wrf_sm2_rho(q,0,j,i);
        }
        const float theta=wrf_tke_theta(q,r.thp,r.thb,p.thb3d,0,j,i);
        t-=chm*((khv*r.bn2[ix])-(9.81f/theta)*heat_flux)*0.5f;
    }
    const float s_buoy=t-s_shear;
    const float map=q.msft[col], delta=pow((p.dx/map)*(p.dy/map)/rdzw,0.33333333f);
    const float l=wrf_tke_l(r.tke[ix],r.bn2[ix],delta);
    const float ce1=(p.c_k/0.10f)*0.19f, ce2=max(0.0f,0.93f-ce1);
    const float coefc=(k==0||k==p.nz-1)?3.9f:(ce1+ce2*l/delta);
    const float tketmp=max(r.tke[ix],1.0e-6f);
    t-=chm*coefc*tketmp*sqrt(tketmp)/l;
    const float s_diss=t-s_shear-s_buoy;
    const float limited=max(t,-chm*max(0.0f,r.tke[ix])/p.dt);
    const float s_lim=limited-t;
    r.rtke[ix]+=limited;
    if (p.budget_on!=0) {
        r.shear[ix]=s_shear;r.buoyancy[ix]=s_buoy;
        r.dissipation[ix]=s_diss;r.limiter[ix]=s_lim;
    }
}
