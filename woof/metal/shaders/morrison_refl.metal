#include <metal_stdlib>
using namespace metal;
#include "refl10cm_morrison_column_bindings.metalh"
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
static inline bool f32_is_finite(float value) {
    return (as_type<uint>(value) & 0x7f800000u) != 0x7f800000u;
}
static inline bool f32_is_nan(float value) {
    return (as_type<uint>(value) & 0x7fffffffu) > 0x7f800000u;
}

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
    // Every source exponent here is the exact 1/3 from xbm=3. Avoids the
    // compiler-crashing generic soft64_plm_pow lowering.
    return soft64_plm_exp(rmul(soft64_plm_log(x),0x3fd5555555555555ul));
}
static inline ulong pow_seven(ulong x) {
    ulong value=x;
    for (int n=1;n<7;++n) value=rmul(value,x);
    return value;
}
struct ReflPsd { float mass; ulong ilam; ulong n0; bool active; bool invalid; };
static inline float output_f32(ulong bits);
static inline ReflPsd diagnose_psd(float q, float number, ulong rho,
                                    ulong xam, ulong gamma4, ulong one) {
    ReflPsd d;
    d.mass=1.0e-12f; d.ilam=0ul; d.n0=0ul;
    d.active=false; d.invalid=false;
    if (q>1.0e-9f) {
        d.mass=output_f32(rmul(soft64_from_f32_bits(as_type<uint>(q)),rho));
        d.active=true;
        if (!(number>0.0f) || !f32_is_finite(number)) {
            d.invalid=true;
        } else {
            ulong nd=rmul(soft64_from_f32_bits(as_type<uint>(number)),rho);
            ulong lambda=pow_third(rdiv(rmul(rmul(xam,gamma4),nd),
                                        soft64_from_f32_bits(as_type<uint>(d.mass))));
            d.ilam=rdiv(one,lambda);
            d.n0=rmul(nd,lambda);
        }
    }
    return d;
}
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
    ulong dg=rmul(ageo,pow_third(xg));
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

kernel void refl10cm_morrison_column(
    constant Refl10CmMorrisonColumnResources& r [[buffer(0)]],
    constant Refl10CmMorrisonColumnParams& p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (p.nz<1 || p.ny<1 || p.nx<1 || col>=uint(p.ny*p.nx)) return;
    const device ulong* t=r.tables;
    ulong kw=t[0], pi=t[8], pi5=t[9], lam4=t[10];
    ReflComplex mw={t[1],t[2]}, mi={t[3],t[4]};
    ulong xamr=t[5], xams=t[6], xamg=t[7];
    ulong rconst=t[11], r0622=t[12], minqv=t[13], gamma4=t[17];
    ulong gamma7=t[18], snow_factor=t[19], ninehundred=t[20];
    ulong thousand=t[21], e18=t[22], ten=t[23];
    ulong melt_outside=t[16], min_volume=t[15];
    const ulong one=0x3ff0000000000000ul, zero=0ul;
    bool melting=false; int k0=0;
    for (int k=p.nz-2;k>=0;--k) {
        uint ix=col+uint(k*p.ny*p.nx);
        float tempf=r.theta[ix]*r.pii[ix];
        ulong temp=soft64_from_f32_bits(as_type<uint>(tempf));
        if (temp>0x4071126666666666ul && r.qr[ix]>1.0e-9f
                && (r.qs[ix+p.ny*p.nx]>1.0e-9f
                    || r.qg[ix+p.ny*p.nx]>1.0e-9f)) {
            if (k+1>k0) k0=k+1;
            melting=true; break;
        }
    }
    float temp0f=r.theta[col+uint(k0*p.ny*p.nx)]
                *r.pii[col+uint(k0*p.ny*p.nx)];
    ulong temp0=soft64_from_f32_bits(as_type<uint>(temp0f));
    ulong qv0=soft64_from_f32_bits(as_type<uint>(r.qv[col+uint(k0*p.ny*p.nx)]));
    if (soft64_is_nan(qv0) || soft64_sign(qv0) || qv0<minqv) qv0=minqv;
    ulong p0=soft64_from_f32_bits(as_type<uint>(r.p[col+uint(k0*p.ny*p.nx)]));
    ulong den0=rmul(rconst,temp0); den0=rmul(den0,radd(qv0,r0622));
    ulong rho0=rdiv(rmul(r0622,p0),den0);
    uint idx0=col+uint(k0*p.ny*p.nx);
    ReflPsd snow0=diagnose_psd(r.qs[idx0],r.ns[idx0],rho0,xams,gamma4,one);
    ReflPsd graup0=diagnose_psd(r.qg[idx0],r.ng[idx0],rho0,xamg,gamma4,one);
    for (int k=0;k<p.nz;++k) {
        uint ix=col+uint(k*p.ny*p.nx);
        float tempf=r.theta[ix]*r.pii[ix];
        ulong temp=soft64_from_f32_bits(as_type<uint>(tempf));
        ulong qv=soft64_from_f32_bits(as_type<uint>(r.qv[ix]));
        if (soft64_is_nan(qv) || soft64_sign(qv) || qv<minqv) qv=minqv;
        ulong pressure=soft64_from_f32_bits(as_type<uint>(r.p[ix]));
        ulong denominator=rmul(rconst,temp);
        denominator=rmul(denominator,radd(qv,r0622));
        ulong rho=rdiv(rmul(r0622,pressure),denominator);
        ReflPsd rain=diagnose_psd(r.qr[ix],r.nr[ix],rho,xamr,gamma4,one);
        ReflPsd snow=diagnose_psd(r.qs[ix],r.ns[ix],rho,xams,gamma4,one);
        ReflPsd graup=diagnose_psd(r.qg[ix],r.ng[ix],rho,xamg,gamma4,one);
        float zr=1.0e-22f,zs=1.0e-22f,zg=1.0e-22f;
        bool bad=rain.invalid||snow.invalid||graup.invalid;
        if (rain.active)
            zr=output_f32(rmul(rmul(rain.n0,gamma7),pow_seven(rain.ilam)));
        if (snow.active) {
            ulong ze=snow_factor;
            ze=rmul(ze,rdiv(soft64_from_i32(6),pi));
            ze=rmul(ze,rdiv(soft64_from_i32(6),pi));
            ze=rmul(ze,rdiv(xams,ninehundred));
            ze=rmul(ze,rdiv(xams,ninehundred));
            ze=rmul(ze,snow.n0); ze=rmul(ze,gamma7);
            ze=rmul(ze,pow_seven(snow.ilam)); zs=output_f32(ze);
        }
        if (graup.active) {
            ulong ze=snow_factor;
            ze=rmul(ze,rdiv(soft64_from_i32(6),pi));
            ze=rmul(ze,rdiv(soft64_from_i32(6),pi));
            ze=rmul(ze,rdiv(xamg,ninehundred));
            ze=rmul(ze,rdiv(xamg,ninehundred));
            ze=rmul(ze,graup.n0); ze=rmul(ze,gamma7);
            ze=rmul(ze,pow_seven(graup.ilam)); zg=output_f32(ze);
        }
        if (melting && k<k0) {
            if (snow.active&&snow0.active&&!snow.invalid) {
                ulong ratio=rdiv(soft64_from_f32_bits(as_type<uint>(snow.mass)),
                                 soft64_from_f32_bits(as_type<uint>(snow0.mass)));
                ulong fm=clamp64(rsub(one,ratio),0x3f747ae147ae147bul,0x3fefae147ae147aeul);
                ulong lam=rdiv(one,snow.ilam),eta=zero;
                ulong snowgeo=pow_third(rdiv(one,xams));
                for(int n=0;n<50;++n) {
                    ulong d=t[25u+uint(n)],width=t[75u+uint(n)];
                    ulong x=rmul(xams,d);x=rmul(x,d);x=rmul(x,d);
                    ulong cb=soak(mw,mi,x,snowgeo,fm,melt_outside,pi,pi5,lam4,min_volume);
                    ulong fd=rmul(snow.n0,soft64_plm_exp(rneg(rmul(lam,d))));
                    ulong term=rmul(fd,cb);term=rmul(term,t[225u+uint(n)]);
                    term=rmul(term,width);eta=radd(eta,term);
                }
                zs=output_f32(rmul(rdiv(lam4,rmul(pi5,kw)),eta));
            }
            if (graup.active&&graup0.active&&!graup.invalid) {
                ulong ratio=rdiv(soft64_from_f32_bits(as_type<uint>(graup.mass)),
                                 soft64_from_f32_bits(as_type<uint>(graup0.mass)));
                ulong fm=clamp64(rsub(one,ratio),0x3f747ae147ae147bul,0x3fefae147ae147aeul);
                ulong lam=rdiv(one,graup.ilam),eta=zero;
                ulong graupgeo=pow_third(rdiv(one,xamg));
                for(int n=0;n<50;++n) {
                    ulong d=t[125u+uint(n)],width=t[175u+uint(n)];
                    ulong x=rmul(xamg,d);x=rmul(x,d);x=rmul(x,d);
                    ulong cb=soak(mw,mi,x,graupgeo,fm,melt_outside,pi,pi5,lam4,min_volume);
                    ulong fd=rmul(graup.n0,soft64_plm_exp(rneg(rmul(lam,d))));
                    ulong term=rmul(fd,cb);term=rmul(term,t[225u+uint(n)]);
                    term=rmul(term,width);eta=radd(eta,term);
                }
                zg=output_f32(rmul(rdiv(lam4,rmul(pi5,kw)),eta));
            }
        }
        float zsum=(zr+zs)+zg;
        ulong logarg=rmul(soft64_from_f32_bits(as_type<uint>(zsum)),e18);
        ulong dbz=rdiv(rmul(ten,soft64_plm_log(logarg)),
                       0x40026bb1bbb55516ul); // ln(10)
        float value=output_f32(dbz);
        if (bad) value=as_type<float>(0x7fc00000u);
        else if (f32_is_nan(value) || value < -35.0f) value=-35.0f;
        r.refl[ix]=value;
    }
}
