#include <metal_stdlib>
#include "wsm_support.metalh"
#include "soft_glibc_solar_trig.metalh"
#include "hexrrfs_ugwp_oro_run_bindings.metalh"
#include "hexrrfs_ugwp_oro_dx_bindings.metalh"
#include "hexrrfs_ugwp_oro_diff_bindings.metalh"
#pragma clang fp contract(off)
using namespace metal;
#define UG_DIV(OPTYPE) static inline Sf operator/(OPTYPE a, OPTYPE b) { return aa_div_impl(Sf(a),Sf(b)); }
UG_DIV(Sf)
static inline Sf operator/(Sf a, float b) { return aa_div_impl(a,Sf(b)); }
static inline Sf operator/(float a, Sf b) { return aa_div_impl(Sf(a),b); }
static inline Sf ug_sqrt(Sf a) { return aa_sqrtf(a); }
static inline Sf ug_sin(Sf a) { return sf_make(soft_glibc_solar_sinf(sf_bits(a))); }
static inline Sf ug_pow(Sf a,Sf b) { return sf_make(soft_gfk_pow_bits(sf_bits(a),sf_bits(b))); }
static inline Sf ug_exp(Sf a) { return sf_make(soft_gfk_exp(sf_bits(a))); }
kernel void hexrrfs_ugwp_oro_run(constant HexrrfsUgwpOroRunResources &r [[buffer(0)]], constant HexrrfsUgwpOroRunParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    device const float *layers=r.layers,*prsi0=r.prsi0,*surface=r.surface;
    device const int *kpbl0=r.kpbl0;
    device float *out=r.out,*sout=r.sout;
    device atomic_int *status=reinterpret_cast<device atomic_int *>(r.status);
    if (i >= uint(p.nc)) return;
    const int nz=p.nz,nc=p.nc,diags=p.diags;
    if (nz < 2 || nz > 2147483645) {
        atomic_store_explicit(status, 2, memory_order_relaxed);
        return;
    }
    const Sf g=p.g, cp=p.cp, rd=p.rd, rv=p.rv, fv=p.fv, pi_c=p.pi_c, dt=p.dt;
    const int kte = nz, kpblmax = nz, kpblmin = 2;
    const ulong n3 = (ulong)nz * nc;
    #define IN(m,k) Sf(layers[(ulong)(m)*n3 + (ulong)((k)-1)*nc + i])
    #define S(m) Sf(surface[(ulong)(m)*nc + i])
    #define P(k) Sf(prsi0[(ulong)((k)-1)*nc + i])
    #define O(m,k) out[(ulong)(m)*n3 + (ulong)((k)-1)*nc + i]
    #define SO(m) sout[(ulong)(m)*nc + i]
    const Sf sa = S(0), ca = S(1), hpbl = S(2), br = S(3);
    const Sf xland = S(4), dx = S(5), var = S(6), oc1 = S(7), varss = S(16);
    const int kpbl = kpbl0[i];
    const Sf inv_g = 1.0f / g, fdir = 8.0f / (2.0f*pi_c);
    // Each column owns the source's sixteen vertical scratch arrays.
    const ulong stride = ulong(nz) + 2;
    device uint *work = r.workspace + ulong(i) * 16 * stride;
    device uint *u = work + 0 * stride;
    device uint *v = work + 1 * stride;
    device uint *vtj = work + 2 * stride;
    device uint *vtk = work + 3 * stride;
    device uint *rho = work + 4 * stride;
    device uint *del = work + 5 * stride;
    device uint *za = work + 6 * stride;
    device uint *bnv2 = work + 7 * stride;
    device uint *ri = work + 8 * stride;
    device uint *velco = work + 9 * stride;
    device uint *taup = work + 10 * stride;
    device uint *taufb = work + 11 * stride;
    device uint *taud_ls = work + 12 * stride;
    device uint *taud_bl = work + 13 * stride;
    device uint *dudt = work + 14 * stride;
    device uint *dvdt = work + 15 * stride;
    Sf dusfc = 0.0f, dvsfc = 0.0f;
    Sf dus[4] = {0.0f,0.0f,0.0f,0.0f}, dvs[4] = {0.0f,0.0f,0.0f,0.0f};
    // bl_ugwp.F:275-354: constants, taper, initialization.
    Sf ls_taper;
    if (dx >= 13000.0f) ls_taper = 1.0f;
    else if (dx <= 3000.0f) ls_taper = 0.0f;
    else ls_taper = 0.5f * (ug_sin(pi_c * (dx-0.5f*(13000.0f+3000.0f))
                                    / (13000.0f-3000.0f)) + 1.0f);
    const Sf ss_taper = 1.0f;
    Sf dxy4[4] = {dx, dx, ug_sqrt(ug_pow(dx,2.0f)+ug_pow(dx,2.0f)),0.0f};
    dxy4[3] = dxy4[2];
    for (int k=1; k<=kte+1; ++k) { taup[k] = sf_bits(0.0f); taufb[k] = sf_bits(0.0f); }
    const Sf zs = IN(6,1) - 0.5f * IN(7,1);
    // bl_ugwp.F:356-380: carriers and earth-relative winds.
    for (int k=1; k<=kte; ++k) {
        vtj[k] = sf_bits(IN(2,k) * (1.0f + fv*IN(3,k)));
        vtk[k] = sf_bits(sf_make(vtj[k]) / IN(5,k));
        rho[k] = sf_bits(1.0f / rd * IN(4,k) / sf_make(vtj[k]));
        del[k] = sf_bits(P(k)-P(k+1));
        u[k] = sf_bits(IN(0,k)*ca - IN(1,k)*sa);
        v[k] = sf_bits(IN(0,k)*sa + IN(1,k)*ca);
        za[k] = sf_bits(IN(6,k)-zs);
        bnv2[k] = sf_bits(0.0f); ri[k] = sf_bits(0.0f); velco[k] = sf_bits(0.0f);
        taud_ls[k] = sf_bits(0.0f); taud_bl[k] = sf_bits(0.0f); dudt[k] = sf_bits(0.0f); dvdt[k] = sf_bits(0.0f);
        for(int m=2;m<12;++m) O(m,k)=0.0f;
    }
    // bl_ugwp.F:382-481: reference layer and direction.
    int klowtop=0;
    const Sf zlowtop=2.0f*var;
    if(zlowtop>0.0f) for(int k=2;k<=kte;++k) {
        if(IN(6,k)-IN(6,1)>=zlowtop) { klowtop=k+1; break; }
    }
    const int kbl = max(min(max(kpbl,klowtop),kpblmax),kpblmin);
    const int komax = klowtop-1;
    const Sf delks = 1.0f/(P(1)-P(kbl));
    const Sf delks1 = 1.0f/(IN(4,1)-IN(4,kbl));
    Sf ubar=0.0f,vbar=0.0f,rhobar=0.0f;
    for(int k=1;k<kbl;++k) {
        const Sf rcsks=sf_make(del[k])*delks, rdelks=sf_make(del[k])*delks;
        ubar=ubar+rcsks*sf_make(u[k]); vbar=vbar+rcsks*sf_make(v[k]); rhobar=rhobar+rdelks*sf_make(rho[k]);
    }
    // Same shared atan2 policy as terrain_drag.cu: double atan2 -> RN Sf.
    const Sf wdir=Sf(atan2(Soft8(float(ubar)),Soft8(float(vbar))))+pi_c;
    const int idir=((int)round(fdir*wdir))%8;
    const int nwdir[8]={6,7,5,8,2,3,1,4};
    const int nwd=nwdir[idir], m=(nwd-1)%4;
    const Sf oa=(Sf)(1-2*((nwd-1)/4))*S(8+m), ol=S(12+m);
    const int cross[4]={1,0,3,2};
    const Sf olp=S(12+cross[m]);
    const Sf od=aa_fmaxf(aa_fminf(olp/aa_fmaxf(ol,1.e-5f),10.0f),0.1f);
    const Sf dxy=dxy4[m];
    const bool ls_on=ls_taper>1.e-02f;
    bool ldrag=false;
    Sf ulow=0.0f, rulow=0.0f, xn=0.0f,yn=0.0f,coefm=0.0f,taub=0.0f;
    // bl_ugwp.F:487-603: mesoscale base stress, stability and propagation.
    if(ls_on) {
        for(int k=1;k<kte;++k) {
            const Sf ti=2.0f/(IN(2,k)+IN(2,k+1));
            const Sf rdz=1.0f/(IN(6,k+1)-IN(6,k));
            const Sf tem1=sf_make(u[k])-sf_make(u[k+1]),tem2=sf_make(v[k])-sf_make(v[k+1]);
            const Sf dw2=tem1*tem1+tem2*tem2;
            const Sf shr2=aa_fmaxf(dw2,1.0f)*rdz*rdz;
            const Sf bvf2=g*(g/cp+rdz*(sf_make(vtj[k+1])-sf_make(vtj[k])))*ti;
            ri[k] = sf_bits(aa_fmaxf(bvf2/shr2,-100.0f));
            bnv2[k] = sf_bits(2.0f*g*rdz*(sf_make(vtk[k+1])-sf_make(vtk[k]))/(sf_make(vtk[k+1])+sf_make(vtk[k])));
            bnv2[k] = sf_bits(aa_fmaxf(sf_make(bnv2[k]),1.e-5f));
        }
        ulow=aa_fmaxf(ug_sqrt(ubar*ubar+vbar*vbar),1.0f);
        rulow=1.0f/ulow;
        for(int k=1;k<kte;++k) {
            velco[k] = sf_bits(0.5f*((sf_make(u[k])+sf_make(u[k+1]))*ubar+(sf_make(v[k])+sf_make(v[k+1]))*vbar));
            velco[k] = sf_bits(sf_make(velco[k])*rulow);
            if(sf_make(velco[k])<1.0f && sf_make(velco[k])>0.0f) velco[k] = sf_bits(1.0f);
        }
        ldrag=sf_make(velco[1])<=0.0f;
        for(int k=2;k<kbl;++k) ldrag=ldrag || sf_make(velco[k])<=0.0f;
        const Sf wtkbj=(IN(4,1)-IN(4,2))*delks1;
        bnv2[1] = sf_bits(wtkbj*sf_make(bnv2[1])); ri[1] = sf_bits(wtkbj*sf_make(ri[1]));
        for(int k=2;k<kbl;++k) {
            const Sf rdelks=(IN(4,k)-IN(4,k+1))*delks1;
            bnv2[1] = sf_bits(sf_make(bnv2[1])+sf_make(bnv2[k])*rdelks); ri[1] = sf_bits(sf_make(ri[1])+sf_make(ri[k])*rdelks);
        }
        ldrag=ldrag || sf_make(bnv2[1])<=0.0f;
        ldrag=ldrag || ulow==1.0f;
        ldrag=ldrag || var<=0.0f;
        const Sf ksmax=4.0f*pi_c/dx;
        if(sf_make(bnv2[1])>0.0f) ldrag=ldrag || ug_sqrt(sf_make(bnv2[1]))*rulow<ksmax;
        for(int k=2;k<kbl;++k) ri[k] = sf_bits(sf_make(ri[1]));
        if(!ldrag) {
            const Sf bnv=ug_sqrt(sf_make(bnv2[1]));
            Sf fr=bnv*rulow*var*od; fr=aa_fminf(fr,10.0f);
            xn=ubar*rulow; yn=vbar*rulow;
            Sf efact=ug_pow(oa+2.0f,0.8f*fr/1.0f);
            efact=aa_fminf(aa_fmaxf(efact,0.0f),10.0f);
            coefm=ug_pow(1.0f+ol,oa+1.0f);
            const Sf xlinv=coefm*8.84e-6f;
            const Sf tem=fr*fr*oc1;
            const Sf gfobnv=1.0f*tem/((tem+0.5f)*bnv);
            taub=xlinv*rhobar*ulow*ulow*ulow*gfobnv*efact;
        }
    }
    // bl_ugwp.F:609-731: small-scale stable boundary-layer waves.
    Sf hpbl2=hpbl+10.0f;
    int kpbl2=kpbl;
    for(int k=2;k<=max(kpbl,2);++k) {
        // UGWP_INVALID_KPBL: bl_ugwp.F:634-646 reads za(k), but may
        // exit before reaching an excessive KPBL. Check the actual read.
        if(k>nz) { atomic_store_explicit(status,1,memory_order_relaxed); return; }
        if(sf_make(za[k])>300.0f) {
            kpbl2=k;
            hpbl2=(k==kpbl)?hpbl+10.0f:sf_make(za[k])+10.0f;
            break;
        }
    }
    Sf tauwavex0=0.0f,tauwavey0=0.0f;
    bool wave=false;
    // bl_ugwp.F:614-615,645-709: for kpbl<=0 the native write loop
    // is empty and initialized SS tendencies remain zero, even if the
    // preceding thvx(kpbl2) read is outside storage. Preserve those
    // observable outputs without refusing a native-completing column.
    if(xland-1.5f<=0.0f && 2.0f*varss<=hpbl && br>0.0f && kpbl>0) {
        // UGWP_INVALID_KPBL: bl_ugwp.F:645-685 dereferences kpbl2 only
        // in this branch. Native water/inactive columns may carry zero.
        if(kpbl2<1 || kpbl2>nz) { atomic_store_explicit(status,1,memory_order_relaxed); return; }
        const Sf thv1=(IN(2,1)/IN(5,1))*(1.0f+fv*IN(3,1));
        const Sf thvk=(IN(2,kpbl2)/IN(5,kpbl2))*(1.0f+fv*IN(3,kpbl2));
        if(br>0.0f && thvk-thv1>0.0f) {
            const Sf xlinv=0.001f*pi_c;
            const Sf govrth=g/(0.5f*(thvk+thv1));
            const Sf xnbv=ug_sqrt(govrth*(thvk-thv1)/hpbl2);
            if(sf_make(u[kpbl2])==0.0f || fabs(xnbv/sf_make(u[kpbl2]))>xlinv) {
                const Sf tv=2.0f*varss;
                const Sf vtemp2=0.5f*xnbv*xlinv*(tv*tv)*sf_make(rho[1]);
                tauwavex0=vtemp2*sf_make(u[1])/(1.0f+vtemp2*dt);
                tauwavex0=tauwavex0*ss_taper;
            }
            if(sf_make(v[kpbl2])==0.0f || fabs(xnbv/sf_make(v[kpbl2]))>xlinv) {
                const Sf tv=2.0f*varss;
                const Sf vtemp2=0.5f*xnbv*xlinv*(tv*tv)*sf_make(rho[1]);
                tauwavey0=vtemp2*sf_make(v[1])/(1.0f+vtemp2*dt);
                tauwavey0=tauwavey0*ss_taper;
            }
            wave=true;
        }
    }
    // UGWP_INVALID_KPBL: bl_ugwp.F:701-709 writes through kpbl when
    // waves are active. Inactive columns do not perform those writes.
    if(wave && kpbl>nz) { atomic_store_explicit(status,1,memory_order_relaxed); return; }
    for(int k=1;k<=kte;++k) {
        Sf ut=0.0f,vt=0.0f;
        if(wave && k<=kpbl) {
            ut=-1.0f*tauwavex0*2.0f*aa_fmaxf(1.0f-sf_make(za[k])/hpbl2,0.0f)/hpbl2;
            vt=-1.0f*tauwavey0*2.0f*aa_fmaxf(1.0f-sf_make(za[k])/hpbl2,0.0f)/hpbl2;
        }
        dudt[k] = sf_bits(sf_make(dudt[k])+ut); dvdt[k] = sf_bits(sf_make(dvdt[k])+vt);
        O(2,k)=O(2,k)+ut; O(3,k)=O(3,k)+vt;
        dusfc=dusfc+ut*sf_make(del[k]); dvsfc=dvsfc+vt*sf_make(del[k]);
        if(diags) { O(8,k)=ut; O(9,k)=vt; dus[2]=dus[2]+ut*sf_make(del[k]); dvs[2]=dvs[2]+vt*sf_make(del[k]); }
    }
    // bl_ugwp.F:737-807: semi-implicit Beljaars topographic form drag.
    Sf var_temp=aa_fminf(varss,500.0f)+aa_fmaxf(0.0f,0.2f*(varss-500.0f));
    const Sf a1=0.00026615161f*(var_temp*var_temp), a2=a1*0.005363f;
    bool form=xland-1.5f<=0.0f;
    for(int k=1;k<=kte;++k) {
        Sf ut=0.0f,vt=0.0f;
        if(form) {
            const Sf wsp=ug_sqrt(sf_make(u[k])*sf_make(u[k])+sf_make(v[k])*sf_make(v[k]));
            var_temp=12.0f*6.325e-3f*ug_exp(-ug_pow(sf_make(za[k])/1500.0f,1.5f))*a2
                     *ug_pow(sf_make(za[k]),-1.2f)*ss_taper;
            ut=-var_temp*wsp*sf_make(u[k])/(1.0f+var_temp*dt*wsp);
            vt=-var_temp*wsp*sf_make(v[k])/(1.0f+var_temp*dt*wsp);
            if(sf_make(za[k])/1500.0f>4.0f) form=false;
        }
        dudt[k] = sf_bits(sf_make(dudt[k])+ut); dvdt[k] = sf_bits(sf_make(dvdt[k])+vt);
        O(2,k)=O(2,k)+ut; O(3,k)=O(3,k)+vt;
        dusfc=dusfc+ut*sf_make(del[k]); dvsfc=dvsfc+vt*sf_make(del[k]);
        if(diags) { O(10,k)=ut; O(11,k)=vt; dus[3]=dus[3]+ut*sf_make(del[k]); dvs[3]=dvs[3]+vt*sf_make(del[k]); }
    }
    // bl_ugwp.F:814-878: vertical mesoscale stress and Lindzen saturation.
    if(ls_on) {
        for(int k=1;k<=kbl;++k) taup[k] = sf_bits(taub);
        bool icrilv=false;
        Sf brvf=0.0f;
        for(int k=2;k<kte;++k) {
            const int kp1=k+1;
            if(k>=kbl) {
                icrilv=icrilv || sf_make(ri[k])<0.25f || sf_make(velco[k])<=0.0f;
                brvf=ug_sqrt(aa_fmaxf(sf_make(bnv2[k]),1.e-5f));
            }
            if(k>=kbl && !ldrag) {
                if(!icrilv && sf_make(taup[k])>0.0f) {
                    const Sf temv=1.0f/sf_make(velco[k]);
                    const Sf tem1=coefm/dxy*(sf_make(rho[kp1])+sf_make(rho[k]))*brvf*sf_make(velco[k])*0.5f;
                    Sf hd=ug_sqrt(sf_make(taup[k])/tem1);
                    const Sf fro=brvf*hd*temv;
                    const Sf tem2=ug_sqrt(sf_make(ri[k])), tem=1.0f+tem2*fro;
                    const Sf rim=sf_make(ri[k])*(1.0f-fro)/(tem*tem);
                    if(rim<=0.25f) {
                        if(oa<=0.0f || kp1>=kpblmin) {
                            const Sf temc=2.0f+1.0f/tem2;
                            hd=sf_make(velco[k])*(2.0f*ug_sqrt(temc)-temc)/brvf;
                            taup[kp1] = sf_bits(tem1*hd*hd);
                        }
                    } else taup[kp1] = sf_bits(sf_make(taup[k]));
                }
            }
        }
        // bl_ugwp.F:884-940: flow-blocking layer and stress.
        if(!ldrag) {
            int kblk=0;
            Sf pe=0.0f,zblk=0.0f;
            for(int k=kte;k>=2;--k) {
                if(kblk==0 && k<=komax) {
                    pe=pe+sf_make(bnv2[k])*(IN(6,komax)-IN(6,k))*sf_make(del[k])/g/sf_make(rho[k]);
                    const Sf ke=0.5f*(ug_pow(sf_make(u[k]),2.0f)+ug_pow(sf_make(v[k]),2.0f));
                    if(pe>=ke) { kblk=min(k,kbl); zblk=sf_make(za[kblk]); }
                }
            }
            if(kblk!=0) {
                const Sf cd=aa_fmaxf(2.0f-1.0f/od,0.0f);
                taufb[1] = sf_bits(0.5f*1.92e-4f*rhobar*coefm*cd*olp*zblk*(ulow*ulow));
                const Sf tautem=sf_make(taufb[1])/(Sf)(kblk-1);
                for(int k=2;k<=kblk;++k) taufb[k] = sf_bits(sf_make(taufb[k-1])-tautem);
            }
        }
        // bl_ugwp.F:943-1025: tendencies, surface sigma and mesospheric limiters.
        taup[kte+1] = sf_bits(sf_make(taup[kte]));
        for(int k=1;k<=kte;++k) {
            taud_ls[k] = sf_bits(1.0f*(sf_make(taup[k+1])-sf_make(taup[k]))*g/sf_make(del[k]));
            taud_bl[k] = sf_bits(1.0f*(sf_make(taufb[k+1])-sf_make(taufb[k]))*g/sf_make(del[k]));
        }
        Sf dtfac=1.0f;
        for(int k=1;k<kpblmax;++k) {
            if(P(k)>=0.5f*P(1)) {
                const Sf s=sf_make(taud_ls[k])+sf_make(taud_bl[k]);
                if(s!=0.0f) dtfac=aa_fminf(dtfac,fabs(sf_make(velco[k])/(dt*s)));
            } else break;
        }
        for(int k=1;k<=kte;++k) {
            Sf dtfac_meso=1.0f;
            if(IN(4,k)<=70.0f) {
                if(sf_make(taud_ls[k])!=0.0f)
                    dtfac_meso=aa_fminf(dtfac_meso,0.5f*fabs(sf_make(velco[k])/(dt*sf_make(taud_ls[k]))));
            }
            taud_ls[k] = sf_bits(sf_make(taud_ls[k])*dtfac*dtfac_meso*ls_taper);
            taud_bl[k] = sf_bits(sf_make(taud_bl[k])*dtfac*dtfac_meso*ls_taper);
            const Sf ux=sf_make(taud_ls[k])*xn,vy=sf_make(taud_ls[k])*yn;
            const Sf ub=sf_make(taud_bl[k])*xn,vb=sf_make(taud_bl[k])*yn;
            dudt[k] = sf_bits(ux+ub+sf_make(dudt[k])); dvdt[k] = sf_bits(vy+vb+sf_make(dvdt[k]));
            O(2,k)=O(2,k)+ux+ub; O(3,k)=O(3,k)+vy+vb;
            dusfc=dusfc+(ux+ub)*sf_make(del[k]); dvsfc=dvsfc+(vy+vb)*sf_make(del[k]);
            if(diags) {
                O(4,k)=ux; O(5,k)=vy; O(6,k)=ub; O(7,k)=vb;
                dus[0]=dus[0]+ux*sf_make(del[k]); dvs[0]=dvs[0]+vy*sf_make(del[k]);
                dus[1]=dus[1]+ub*sf_make(del[k]); dvs[1]=dvs[1]+vb*sf_make(del[k]);
            }
        }
    }
    // bl_ugwp.F:1019-1059: integrate earth-relative stresses, rotate tendencies.
    SO(0)=-inv_g*dusfc; SO(1)=-inv_g*dvsfc;
    if(diags) for(int m=0;m<4;++m) { SO(2+2*m)=-inv_g*dus[m]; SO(3+2*m)=-inv_g*dvs[m]; }
    for(int k=1;k<=kte;++k) {
        O(0,k)=IN(8,k)+sf_make(dudt[k])*ca+sf_make(dvdt[k])*sa;
        O(1,k)=IN(9,k)-sf_make(dudt[k])*sa+sf_make(dvdt[k])*ca;
    }
#undef IN
#undef S
#undef P
#undef O
#undef SO
}
kernel void hexrrfs_ugwp_oro_dx(constant HexrrfsUgwpOroDxResources &r [[buffer(0)]], constant HexrrfsUgwpOroDxParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    r.dx[i] = Sf(p.length) / ug_pow(Sf(r.density[i]), Sf(0.25f));
}
kernel void hexrrfs_ugwp_oro_diff(constant HexrrfsUgwpOroDiffResources &r [[buffer(0)]], constant HexrrfsUgwpOroDiffParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    r.diff[i] = Sf(r.after[i]) - Sf(r.before[i]);
}
