#include <metal_stdlib>
using namespace metal;
#include "refl10cm_thompson_column_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"

// Source-faithful WRF 4.6.1 Morrison REFL_10CM column diagnostic. All
// binary64 arithmetic is carried as IEEE bit patterns through soft64; no
// native double or complex type is used. Table layout is documented in
// woof/metal/refl_family.py.
struct ReflComplex { ulong re; ulong im; };

static inline ulong radd(ulong a, ulong b) { return soft64_add(a,b); }
static inline ulong rsub(ulong a, ulong b) { return soft64_sub(a,b); }
static inline ulong rmul(ulong a, ulong b) { return soft64_mul(a,b); }
static inline ulong rdiv(ulong a, ulong b) { return soft64_div(a,b); }
static inline ulong rneg(ulong a) { return soft64_negate_bits(a); }
static inline ReflComplex cadd(ReflComplex a, ReflComplex b) {
    ReflComplex c; c.re=radd(a.re,b.re); c.im=radd(a.im,b.im); return c;
}
static inline ReflComplex csub(ReflComplex a, ReflComplex b) {
    ReflComplex c; c.re=rsub(a.re,b.re); c.im=rsub(a.im,b.im); return c;
}
static inline ReflComplex cmul(ReflComplex a, ReflComplex b) {
    ReflComplex c;
    c.re=rsub(rmul(a.re,b.re),rmul(a.im,b.im));
    c.im=radd(rmul(a.re,b.im),rmul(a.im,b.re)); return c;
}
static inline ReflComplex cdiv(ReflComplex a, ReflComplex b) {
    ulong d=radd(rmul(b.re,b.re),rmul(b.im,b.im));
    ReflComplex c;
    c.re=rdiv(radd(rmul(a.re,b.re),rmul(a.im,b.im)),d);
    c.im=rdiv(rsub(rmul(a.im,b.re),rmul(a.re,b.im)),d); return c;
}
static inline ulong cabs64(ReflComplex a) {
    return soft64_sqrt(radd(rmul(a.re,a.re),rmul(a.im,a.im)));
}

// atan2 is used on finite complex ratios produced by nested Maxwell-Garnett
// mixtures. Range reduction keeps the odd series rapidly convergent.
static inline ulong ratan_unit(ulong x) {
    const ulong one=0x3ff0000000000000ul;
    const ulong pi4=0x3fe921fb54442d18ul;
    const ulong tanpi8=0x3fda827999fcef32ul;
    bool shifted=x>tanpi8;
    if (shifted) x=rdiv(rsub(x,one),radd(x,one));
    ulong x2=rmul(x,x), term=x, sum=x;
    for (int n=1;n<24;++n) {
        term=rneg(rmul(term,x2));
        ulong add=rdiv(term,soft64_from_i32(2*n+1));
        sum=radd(sum,add);
    }
    return shifted ? radd(pi4,sum) : sum;
}
static inline ulong ratan2(ulong y, ulong x) {
    const ulong pi=0x400921fb54442d18ul;
    const ulong pi2=0x3ff921fb54442d18ul;
    ulong ax=soft64_abs_bits(x), ay=soft64_abs_bits(y);
    ulong a = ax>=ay ? ratan_unit(rdiv(ay,ax))
                     : rsub(pi2,ratan_unit(rdiv(ax,ay)));
    if (soft64_sign(x)) a=rsub(pi,a);
    if (soft64_sign(y)) a=rneg(a);
    return a;
}
static inline ReflComplex csqrt64(ReflComplex a) {
    const ulong half_bits=0x3fe0000000000000ul;
    ulong m=cabs64(a);
    ulong rearg=rmul(half_bits,radd(m,a.re));
    ulong imarg=rmul(half_bits,rsub(m,a.re));
    if (soft64_sign(rearg)) rearg=0ul;
    if (soft64_sign(imarg)) imarg=0ul;
    ulong re=soft64_sqrt(rearg);
    ulong im=soft64_sqrt(imarg);
    bool negative_im=soft64_less(a.im,0ul);
    ReflComplex c={re,negative_im?rneg(im):im}; return c;
}
static inline ReflComplex clog64(ReflComplex a) {
    ReflComplex c={soft64_plm_log(cabs64(a)),ratan2(a.im,a.re)}; return c;
}
static inline ulong pow_third(ulong x) {
    // Source exponent is the exact 1/3 used by the rain/graupel PSD and
    // bulk-melt diameter moments.
    return soft64_plm_exp(rmul(soft64_plm_log(x),0x3fd5555555555555ul));
}
static inline ulong pow_seven(ulong x) {
    ulong value=x;
    for (int n=1;n<7;++n) value=rmul(value,x);
    return value;
}
static inline float output_f32(ulong bits);
static inline ulong clamp64(ulong x, ulong lo, ulong hi) {
    if (soft64_sign(x) || x<lo) return lo;
    return x>hi ? hi : x;
}
static inline ReflComplex cscale(ulong s, ReflComplex a) {
    ReflComplex c={rmul(s,a.re),rmul(s,a.im)}; return c;
}
static inline ReflComplex maxwell_garnett(ulong v2, ulong v3,
                  ReflComplex m1, ReflComplex m2, ReflComplex m3) {
    const ReflComplex one={0x3ff0000000000000ul,0ul};
    ReflComplex a=cmul(m1,m1), b=cmul(m2,m2), c=cmul(m3,m3);
    ReflComplex db=csub(b,a), dc=csub(c,a);
    ReflComplex beta2=cmul(cdiv(cscale(0x4000000000000000ul,a),db),
        csub(cmul(cdiv(b,db),clog64(cdiv(b,a))),one));
    ReflComplex beta3=cmul(cdiv(cscale(0x4000000000000000ul,a),dc),
        csub(cmul(cdiv(c,dc),clog64(cdiv(c,a))),one));
    ulong v1=rsub(rsub(0x3ff0000000000000ul,v2),v3);
    ReflComplex num=cadd(cscale(v1,a),cadd(cscale(v2,cmul(beta2,b)),
                                         cscale(v3,cmul(beta3,c))));
    ReflComplex host; host.re=v1; host.im=0ul;
    ReflComplex den=cadd(host,
        cadd(cscale(v2,beta2),cscale(v3,beta3)));
    return csqrt64(cdiv(num,den));
}
static inline ReflComplex mix_ice_water_air(ReflComplex ma, ReflComplex mi,
                                             ReflComplex mw, ulong vi,
                                             ulong vw, ulong va,
                                             ulong minimum_volume) {
    ulong den=radd(vi,vw);
    if (den<minimum_volume) den=minimum_volume;
    ulong icefrac=rdiv(vi,den);
    ReflComplex inner=maxwell_garnett(0ul,icefrac,mw,ma,mi);
    return maxwell_garnett(va,0ul,inner,ma,
                           cscale(0x4000000000000000ul,ma));
}

static inline ulong soak(ReflComplex mw, ReflComplex mi, ulong xg,
                         ulong ageo, ulong fmelt, ulong meltoutside,
                         ulong pi, ulong pi5, ulong lambda4,
                         ulong minimum_volume) {
    const ulong zero=0ul, one=0x3ff0000000000000ul;
    const ulong ten=0x4024000000000000ul, ninehundred=0x408c200000000000ul;
    const ulong thousand=0x408f400000000000ul;
    ulong fm=clamp64(fmelt,zero,one), mra=clamp64(meltoutside,zero,one);
    mra=radd(mra,rmul(rsub(one,mra),fm));
    ulong xw=rmul(xg,fm);
    ulong dg=rmul(ageo,soft64_sqrt(xg));
    const ulong min_d=0x3d719799812dea11ul;
    if (dg<min_d) return 0ul;
    ulong vg=rmul(rdiv(pi,soft64_from_i32(6)),dg);
    vg=rmul(rmul(vg,dg),dg);
    ulong rhog=clamp64(rdiv(xg,vg),ten,ninehundred);
    vg=rdiv(xg,rhog);
    ulong grenz=rsub(one,rdiv(rhog,thousand));
    ulong volg;
    if (mra<=grenz) {
        volg=rmul(vg,rsub(one,rmul(mra,fm)));
    } else {
        ulong fmgrenz=rdiv(rsub(ninehundred,rhog),
            radd(rsub(rmul(mra,ninehundred),rhog),
                 rdiv(rmul(ninehundred,rhog),thousand)));
        volg=fm<=fmgrenz ? rmul(rsub(one,rmul(mra,fm)),vg)
            : radd(rdiv(rsub(xg,xw),ninehundred),rdiv(xw,thousand));
    }
    ulong dlarge=pow_third(rmul(rdiv(soft64_from_i32(6),pi),volg));
    ulong vice=rdiv(rsub(xg,xw),rmul(volg,ninehundred));
    ulong vwater=rdiv(xw,rmul(thousand,volg));
    ulong vair=rsub(rsub(one,vice),vwater);
    ReflComplex air={one,zero};
    ReflComplex core=mix_ice_water_air(air,mi,mw,vice,vwater,vair,
                                       minimum_volume);
    ReflComplex m2=cmul(core,core);
    ReflComplex two; two.re=0x4000000000000000ul; two.im=zero;
    ReflComplex ratio=cdiv(csub(m2,air),cadd(m2,two));
    ulong kfac=cabs64(ratio), d3=rmul(dlarge,dlarge);
    d3=rmul(d3,dlarge);
    ulong result=rmul(kfac,kfac);
    result=rmul(result,pi5); result=rmul(result,d3);
    result=rmul(result,d3);
    return rdiv(result,lambda4);
}

static inline float output_f32(ulong bits) {
    return as_type<float>(soft64_to_f32_bits(bits));
}



static inline float thompson_snow_field_a(float tc, float moment) {
    const float sa0=5.065339f, sa1=-0.062659f, sa2=-3.032362f;
    const float sa3=0.029469f, sa4=-0.000285f, sa5=0.31255f;
    const float sa6=0.000204f, sa7=0.003199f, sa8=0.0f, sa9=-0.015952f;
    const float tc2=tc*tc, m2=moment*moment;
    const float loga=sa0+sa1*tc+sa2*moment+sa3*tc*moment+sa4*tc2
        +sa5*m2+sa6*tc2*moment+sa7*tc*m2+sa8*tc2*tc+sa9*m2*moment;
    return pow(10.0f, loga);
}
static inline float thompson_snow_field_b(float tc, float moment) {
    const float sb0=0.476221f, sb1=-0.015896f, sb2=0.165977f;
    const float sb3=0.007468f, sb4=-0.000141f, sb5=0.060366f;
    const float sb6=0.000079f, sb7=0.000594f, sb8=0.0f, sb9=-0.003577f;
    const float tc2=tc*tc, m2=moment*moment;
    return sb0+sb1*tc+sb2*moment+sb3*tc*moment+sb4*tc2+sb5*m2
        +sb6*tc2*moment+sb7*tc*m2+sb8*tc2*tc+sb9*m2*moment;
}

// WRF classic Thompson calc_refl10cm (module_mp_thompson.F:5710-6028).
// Radar products in `tables` are radar_init()'s k_w, complex water/ice
// indices, log-bin centers/widths, and Simpson weights. Thompson's fixed
// m(D) coefficients and Field snow fit remain source-local constants.
kernel void refl10cm_thompson_column(
    constant Refl10CmThompsonColumnResources& r [[buffer(0)]],
    constant Refl10CmThompsonColumnParams& p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (p.nz<1 || p.ny<1 || p.nx<1 || col>=uint(p.ny*p.nx)) return;
    const device ulong* tab=r.tables;
    const ulong kw=tab[0], pi=tab[5], pi5=tab[6], lam4=tab[7];
    ReflComplex mw={tab[1],tab[2]}, mi={tab[3],tab[4]};
    const float tpi=3.1415926536f;
    const float am_r=tpi*1000.0f/6.0f, am_s=0.069f;
    const float am_g=tpi*400.0f/6.0f;
    const ulong one=0x3ff0000000000000ul, zero=0ul;
    const ulong e18=0x43abc16d674ec800ul;
    const ulong ten=soft64_from_i32(10);
    float rr[256],rs[256],rg[256],smob[256],smoc[256],smoz[256];
    ulong ilamr[256],n0r[256],ilamg[256],n0g[256];
    bool lqr[256],lqs[256],lqg[256];
    float zer[256],zes[256],zeg[256];
    for(int k=0;k<p.nz;++k) {
        uint ix=col+uint(k*p.ny*p.nx);
        float qvk=max(1.0e-10f,r.qv[ix]);
        float rho=0.622f*r.p[ix]/(287.04f*r.t[ix]*(qvk+0.622f));
        rr[k]=rs[k]=rg[k]=1.0e-12f;
        smob[k]=smoc[k]=smoz[k]=0.0f;
        ilamr[k]=n0r[k]=ilamg[k]=n0g[k]=zero;
        lqr[k]=lqs[k]=lqg[k]=false;
        if(r.qr[ix]>1.0e-12f) {
            rr[k]=r.qr[ix]*rho;
            float nv=max(1.0e-6f,r.nr[ix]*rho);
            float ratio=am_r*6.0f*nv/rr[k];
            ulong lambda=soft64_plm_exp(rmul(soft64_plm_log(
                soft64_from_f32_bits(as_type<uint>(ratio))),
                0x3fd5555555555555ul));
            ilamr[k]=rdiv(one,lambda);
            n0r[k]=rmul(soft64_from_f32_bits(as_type<uint>(nv)),lambda);
            lqr[k]=true;
        }
        if(r.qs[ix]>1.0e-6f) {
            rs[k]=r.qs[ix]*rho;
            float tc0=min(-0.1f,r.t[ix]-273.15f);
            smob[k]=rs[k]/am_s;
            float smo2=smob[k];
            smoc[k]=thompson_snow_field_a(tc0,3.0f)
                *pow(smo2,thompson_snow_field_b(tc0,3.0f));
            smoz[k]=thompson_snow_field_a(tc0,4.0f)
                *pow(smo2,thompson_snow_field_b(tc0,4.0f));
            lqs[k]=true;
        }
        if(r.qg[ix]>1.0e-6f) {
            rg[k]=r.qg[ix]*rho;
            float nv=max(1.0e-6f,r.graupel_number_shadow[ix]*rho);
            float ratio=am_g*6.0f*nv/rg[k];
            ulong lambda=soft64_plm_exp(rmul(soft64_plm_log(
                soft64_from_f32_bits(as_type<uint>(ratio))),
                0x3fd5555555555555ul));
            ilamg[k]=rdiv(one,lambda);
            n0g[k]=rmul(soft64_from_f32_bits(as_type<uint>(nv)),lambda);
            lqg[k]=true;
        }
    }
    bool melting=false; int k0=0;
    for(int k=p.nz-2;k>=0;--k) {
        uint ix=col+uint(k*p.ny*p.nx);
        if(r.t[ix]>273.15f&&lqr[k]&&(lqs[k+1]||lqg[k+1])) {
            k0=max(k+1,k0); melting=true; break;
        }
    }
    for(int k=0;k<p.nz;++k) {
        zer[k]=zes[k]=zeg[k]=1.0e-22f;
        if(lqr[k]) zer[k]=output_f32(rmul(rmul(n0r[k],soft64_from_i32(720)),
                                             pow_seven(ilamr[k])));
        if(lqs[k]) {
            float f=(0.176f/0.93f)*(6.0f/tpi)*(6.0f/tpi)
                *(am_s/900.0f)*(am_s/900.0f)*smoz[k];
            zes[k]=f;
        }
        if(lqg[k]) {
            ulong factor=rdiv(soft64_from_f32_bits(as_type<uint>(0.176f)),
                              soft64_from_f32_bits(as_type<uint>(0.93f)));
            ulong sixpi=rdiv(soft64_from_i32(6),soft64_from_f32_bits(as_type<uint>(tpi)));
            ulong amratio=rdiv(soft64_from_f32_bits(as_type<uint>(am_g)),soft64_from_i32(900));
            ulong ze=rmul(rmul(rmul(factor,sixpi),sixpi),rmul(amratio,amratio));
            ze=rmul(ze,n0g[k]); ze=rmul(ze,soft64_from_i32(720));
            zeg[k]=output_f32(rmul(ze,pow_seven(ilamg[k])));
        }
    }
    // WRF comments out Thompson's graupel bright-band block: only Field snow
    // contributes the wet-particle integration.
    if(melting&&k0>=1) for(int k=k0-1;k>=0;--k) {
        if(lqs[k]&&lqs[k0]) {
            float m0f=smob[k]/smoc[k];
            float mratf=smob[k]*m0f*m0f*m0f;
            ulong m0=soft64_from_f32_bits(as_type<uint>(m0f));
            ulong mrat=soft64_from_f32_bits(as_type<uint>(mratf));
            ulong slam1=rmul(m0,0x4034c7ae147ae148ul); // 20.78
            ulong slam2=rmul(m0,0x400a51eb851eb852ul); // 3.29
            ulong snowratio=rdiv(soft64_from_f32_bits(as_type<uint>(rs[k])),
                                soft64_from_f32_bits(as_type<uint>(rs[k0])));
            ulong fm=rsub(one,snowratio);
            if(soft64_less(fm,0x3fa999999999999aul)) fm=0x3fa999999999999aul;
            if(soft64_greater(fm,0x3feeb851eb851eb8ul)) fm=0x3feeb851eb851eb8ul;
            ulong eta=zero;
            for(int n=0;n<50;++n) {
                ulong d=tab[8u+uint(n)], width=tab[58u+uint(n)];
                ulong ams64=soft64_from_f32_bits(as_type<uint>(am_s));
                ulong x=rmul(ams64,rmul(d,d));
                ulong snowgeo=soft64_sqrt(rdiv(one,ams64));
                ulong cb=soak(mw,mi,x,snowgeo,
                    fm,0x3feccccccccccccdul,pi,pi5,lam4,0x3ddb7cdfd9d7bdbbul);
                ulong e1=soft64_plm_exp(rneg(rmul(slam1,d)));
                ulong md=rmul(m0,d);
                ulong pow6357=soft64_plm_exp(rmul(soft64_plm_log(md),0x3fe457a786c22681ul));
                ulong e2=soft64_plm_exp(rneg(rmul(slam2,d)));
                ulong fd=rmul(mrat,radd(rmul(0x407ea9999999999aul,e1),
                    rmul(rmul(0x403175c28f5c28f6ul,pow6357),e2)));
                ulong term=rmul(rmul(fd,cb),tab[108u+uint(n)]);
                eta=radd(eta,rmul(term,width));
            }
            ulong scale=rdiv(lam4,rmul(pi5,kw));
            zes[k]=output_f32(rmul(scale,eta));
        }
    }
    for(int k=0;k<p.nz;++k) {
        uint ix=col+uint(k*p.ny*p.nx);
        float zsum=(zer[k]+zes[k])+zeg[k];
        ulong arg=rmul(soft64_from_f32_bits(as_type<uint>(zsum)),e18);
        ulong dbz=rdiv(rmul(ten,soft64_plm_log(arg)),0x40026bb1bbb55516ul);
        float value=output_f32(dbz);
        r.refl[ix]=isnan(value)||value< -35.0f ? -35.0f : value;
    }
}
