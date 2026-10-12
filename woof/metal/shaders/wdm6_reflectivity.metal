#include <metal_stdlib>
using namespace metal;
#include "refl10cm_wdm6_column_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"

struct Wdm6Complex { ulong re; ulong im; };

static inline ulong wdm6_add(ulong a, ulong b) { return soft64_add(a,b); }
static inline ulong wdm6_sub(ulong a, ulong b) { return soft64_sub(a,b); }
static inline ulong wdm6_mul(ulong a, ulong b) { return soft64_mul(a,b); }
static inline ulong wdm6_div(ulong a, ulong b) { return soft64_div(a,b); }
static inline ulong wdm6_neg(ulong a) { return soft64_negate_bits(a); }
static inline ulong wdm6_from_float(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}
static inline float wdm6_to_float(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}
static inline ulong wdm6_pow(ulong x, ulong exponent) {
    return soft64_plm_exp(wdm6_mul(soft64_plm_log(x), exponent));
}
static inline ulong wdm6_power(ulong x, ulong exponent) {
    return wdm6_pow(x, exponent);
}
static inline ulong wdm6_pow_eight(ulong x) {
    ulong x2=wdm6_mul(x,x), x4=wdm6_mul(x2,x2);
    return wdm6_mul(x4,x4);
}
static inline ulong wdm6_pow_seven(ulong x) {
    ulong x2=wdm6_mul(x,x), x4=wdm6_mul(x2,x2);
    return wdm6_mul(wdm6_mul(x4,x2),x);
}
static inline Wdm6Complex wdm6_cadd(Wdm6Complex a, Wdm6Complex b) {
    Wdm6Complex c={wdm6_add(a.re,b.re),wdm6_add(a.im,b.im)}; return c;
}
static inline Wdm6Complex wdm6_csub(Wdm6Complex a, Wdm6Complex b) {
    Wdm6Complex c={wdm6_sub(a.re,b.re),wdm6_sub(a.im,b.im)}; return c;
}
static inline Wdm6Complex wdm6_cmul(Wdm6Complex a, Wdm6Complex b) {
    Wdm6Complex c={wdm6_sub(wdm6_mul(a.re,b.re),wdm6_mul(a.im,b.im)),
                   wdm6_add(wdm6_mul(a.re,b.im),wdm6_mul(a.im,b.re))};
    return c;
}
static inline Wdm6Complex wdm6_cdiv(Wdm6Complex a, Wdm6Complex b) {
    ulong d=wdm6_add(wdm6_mul(b.re,b.re),wdm6_mul(b.im,b.im));
    Wdm6Complex c={wdm6_div(wdm6_add(wdm6_mul(a.re,b.re),
                                      wdm6_mul(a.im,b.im)),d),
                     wdm6_div(wdm6_sub(wdm6_mul(a.im,b.re),
                                      wdm6_mul(a.re,b.im)),d)};
    return c;
}
static inline ulong wdm6_cabs(Wdm6Complex a) {
    return soft64_sqrt(wdm6_add(wdm6_mul(a.re,a.re),wdm6_mul(a.im,a.im)));
}
static inline ulong wdm6_atan_unit(ulong x, ulong pi) {
    ulong one=soft64_from_i32(1);
    ulong tan_pi8=wdm6_sub(soft64_sqrt(soft64_from_i32(2)),one);
    bool shifted=x>tan_pi8;
    if (shifted) x=wdm6_div(wdm6_sub(x,one),wdm6_add(x,one));
    ulong x2=wdm6_mul(x,x), term=x, sum=x;
    for (int n=1;n<24;++n) {
        term=wdm6_neg(wdm6_mul(term,x2));
        sum=wdm6_add(sum,wdm6_div(term,soft64_from_i32(2*n+1)));
    }
    return shifted ? wdm6_add(wdm6_div(pi,soft64_from_i32(4)),sum) : sum;
}
static inline ulong wdm6_atan2(ulong y, ulong x, ulong pi) {
    ulong pi2=wdm6_div(pi,soft64_from_i32(2));
    ulong ax=soft64_abs_bits(x), ay=soft64_abs_bits(y);
    ulong angle=ax>=ay ? wdm6_atan_unit(wdm6_div(ay,ax),pi)
                       : wdm6_sub(pi2,wdm6_atan_unit(wdm6_div(ax,ay),pi));
    if (soft64_sign(x)) angle=wdm6_sub(pi,angle);
    if (soft64_sign(y)) angle=wdm6_neg(angle);
    return angle;
}
static inline Wdm6Complex wdm6_csqrt(Wdm6Complex a, ulong half) {
    ulong mag=wdm6_cabs(a);
    ulong realarg=wdm6_mul(half,wdm6_add(mag,a.re));
    ulong imagarg=wdm6_mul(half,wdm6_sub(mag,a.re));
    if (soft64_sign(realarg)) realarg=0ul;
    if (soft64_sign(imagarg)) imagarg=0ul;
    ulong im=soft64_sqrt(imagarg);
    Wdm6Complex c={soft64_sqrt(realarg),
                   soft64_less(a.im,0ul)?wdm6_neg(im):im};
    return c;
}
static inline Wdm6Complex wdm6_clog(Wdm6Complex a, ulong pi) {
    Wdm6Complex c={soft64_plm_log(wdm6_cabs(a)),
                   wdm6_atan2(a.im,a.re,pi)};
    return c;
}
static inline Wdm6Complex wdm6_cscale(ulong scale, Wdm6Complex a) {
    Wdm6Complex c={wdm6_mul(scale,a.re),wdm6_mul(scale,a.im)}; return c;
}
static inline Wdm6Complex wdm6_maxwell_garnett(
        ulong v2, ulong v3, Wdm6Complex m1, Wdm6Complex m2,
        Wdm6Complex m3, ulong one, ulong two, ulong pi) {
    Wdm6Complex a=wdm6_cmul(m1,m1), b=wdm6_cmul(m2,m2);
    Wdm6Complex c=wdm6_cmul(m3,m3), db=wdm6_csub(b,a), dc=wdm6_csub(c,a);
    Wdm6Complex beta2=wdm6_cmul(
        wdm6_cdiv(wdm6_cscale(two,a),db),
        wdm6_csub(wdm6_cmul(wdm6_cdiv(b,db),
                              wdm6_clog(wdm6_cdiv(b,a),pi)),
                   Wdm6Complex{one,0ul}));
    Wdm6Complex beta3=wdm6_cmul(
        wdm6_cdiv(wdm6_cscale(two,a),dc),
        wdm6_csub(wdm6_cmul(wdm6_cdiv(c,dc),
                              wdm6_clog(wdm6_cdiv(c,a),pi)),
                   Wdm6Complex{one,0ul}));
    ulong v1=wdm6_sub(wdm6_sub(one,v2),v3);
    Wdm6Complex numerator=wdm6_cadd(
        wdm6_cscale(v1,a),
        wdm6_cadd(wdm6_cscale(v2,wdm6_cmul(beta2,b)),
                  wdm6_cscale(v3,wdm6_cmul(beta3,c))));
    Wdm6Complex denominator=wdm6_cadd(
        Wdm6Complex{v1,0ul},
        wdm6_cadd(wdm6_cscale(v2,beta2),wdm6_cscale(v3,beta3)));
    return wdm6_csqrt(wdm6_cdiv(numerator,denominator),
                       wdm6_div(one,soft64_from_i32(2)));
}
static inline Wdm6Complex wdm6_mix_nested(
        Wdm6Complex ma, Wdm6Complex mi, Wdm6Complex mw,
        ulong vair, ulong vice, ulong vwater, ulong minimum_volume,
        ulong one, ulong two, ulong pi) {
    ulong volume=wdm6_add(vice,vwater);
    if (volume<minimum_volume) volume=minimum_volume;
    ulong icefrac=wdm6_div(vice,volume);
    Wdm6Complex inner=wdm6_maxwell_garnett(0ul,icefrac,mw,ma,mi,
                                            one,two,pi);
    return wdm6_maxwell_garnett(vair,0ul,inner,ma,
                                wdm6_cscale(two,ma),one,two,pi);
}
static inline ulong wdm6_soak(
        ulong mass, ulong ageo, ulong exponent, ulong fmelt,
        ulong meltoutside, Wdm6Complex mw, Wdm6Complex mi,
        ulong pi, ulong pi5, ulong lambda4, ulong minimum_volume,
        ulong one, ulong zero, ulong ten, ulong ninehundred,
        ulong thousand, ulong minimum_diameter, ulong six) {
    ulong fm=fmelt;
    if (soft64_sign(fm)) fm=zero;
    if (fm>one) fm=one;
    ulong mra=meltoutside;
    if (soft64_sign(mra)) mra=zero;
    if (mra>one) mra=one;
    mra=wdm6_add(mra,wdm6_mul(wdm6_sub(one,mra),fm));
    ulong watermass=wdm6_mul(mass,fm);
    ulong diameter=wdm6_mul(ageo,wdm6_power(mass,exponent));
    if (diameter<minimum_diameter) return zero;
    ulong volume=wdm6_div(pi,six);
    volume=wdm6_mul(wdm6_mul(volume,diameter),diameter);
    volume=wdm6_mul(volume,diameter);
    ulong density=wdm6_div(mass,volume);
    if (soft64_less(density,ten)) density=ten;
    if (density>ninehundred) density=ninehundred;
    volume=wdm6_div(mass,density);
    ulong boundary=wdm6_sub(one,wdm6_div(density,thousand));
    ulong meltvolume;
    if (mra<=boundary) {
        meltvolume=wdm6_mul(volume,wdm6_sub(one,wdm6_mul(mra,fm)));
    } else {
        ulong numerator=wdm6_sub(ninehundred,density);
        ulong denominator=wdm6_add(
            wdm6_sub(wdm6_mul(mra,ninehundred),density),
            wdm6_div(wdm6_mul(ninehundred,density),thousand));
        ulong fm_boundary=wdm6_div(numerator,denominator);
        meltvolume=fm<=fm_boundary
            ? wdm6_mul(wdm6_sub(one,wdm6_mul(mra,fm)),volume)
            : wdm6_add(wdm6_div(wdm6_sub(mass,watermass),ninehundred),
                       wdm6_div(watermass,thousand));
    }
    ulong large_diameter=wdm6_power(
        wdm6_mul(wdm6_div(six,pi),meltvolume),exponent);
    ulong icevolume=wdm6_div(wdm6_sub(mass,watermass),
                              wdm6_mul(meltvolume,ninehundred));
    ulong watervolume=wdm6_div(watermass,wdm6_mul(thousand,meltvolume));
    ulong airvolume=wdm6_sub(wdm6_sub(one,icevolume),watervolume);
    Wdm6Complex air={one,zero};
    Wdm6Complex core=wdm6_mix_nested(air,mi,mw,airvolume,icevolume,
        watervolume,minimum_volume,one,two,pi);
    Wdm6Complex core2=wdm6_cmul(core,core);
    Wdm6Complex numerator=wdm6_csub(core2,air);
    Wdm6Complex denominator=wdm6_cadd(core2,Wdm6Complex{two,zero});
    ulong kfac=wdm6_cabs(wdm6_cdiv(numerator,denominator));
    ulong d3=wdm6_mul(wdm6_mul(large_diameter,large_diameter),large_diameter);
    ulong result=wdm6_mul(wdm6_mul(kfac,kfac),pi5);
    result=wdm6_mul(wdm6_mul(result,d3),d3);
    return wdm6_div(result,lambda4);
}

kernel void refl10cm_wdm6_column(
        constant Refl10CmWdm6ColumnResources& r [[buffer(0)]],
        constant Refl10CmWdm6ColumnParams& p [[buffer(1)]],
        uint col [[thread_position_in_grid]]) {
    if (p.nz<1 || p.ny<1 || p.nx<1 || col>=uint(p.ny*p.nx)) return;
    const device ulong* tab=r.tables;
    Wdm6Complex mw={tab[1],tab[2]}, mi={tab[3],tab[4]};
    ulong kw=tab[0], xamr=tab[5], xams=tab[6], xamg=tab[7];
    ulong pi=tab[8], pi5=tab[9], lambda4=tab[10];
    ulong rd=tab[11], mixing_ratio=tab[12], qv_floor=tab[13];
    ulong mass_threshold=tab[14], min_volume=tab[15], meltoutside=tab[16];
    ulong gamma4=tab[17], gamma7=tab[18], snow_factor=tab[19];
    ulong ninehundred=tab[20], thousand=tab[21], e18=tab[22];
    ulong ten=tab[23], temperature0=tab[24], xcrg3=tab[25];
    ulong xcrg4=tab[26], xorg2=tab[27], n0g=tab[28];
    ulong xocms=tab[29], xocmg=tab[30], exponent=tab[31];
    ulong minimum_diameter=tab[32], minimum_mass=tab[33], minimum_z=tab[34];
    ulong tc_max=tab[35], snow_exp=tab[36], snow_prefactor=tab[37];
    ulong snow_cap=tab[38], melt_min=tab[39], melt_max=tab[40];
    ulong output_floor=tab[41], ln10=tab[42], six=tab[43], quarter=tab[44];
    ulong one=soft64_from_i32(1), zero=0ul, two=soft64_from_i32(2);
    float rr[256],rs[256],rg[256];
    ulong ilamr[256],ilams[256],ilamg[256],n0r[256],n0s[256];
    bool lqr[256],lqs[256],lqg[256];
    float zer[256],zes[256],zeg[256];
    for (int k=0;k<p.nz;++k) {
        uint ix=col+uint(k*p.ny*p.nx);
        ulong tk=wdm6_from_float(r.t[ix]);
        ulong qv=wdm6_from_float(r.qv[ix]);
        if (soft64_is_nan(qv) || soft64_less(qv,qv_floor)) qv=qv_floor;
        ulong rho=wdm6_div(wdm6_mul(mixing_ratio,
                                    wdm6_from_float(r.p[ix])),
            wdm6_mul(wdm6_mul(rd,tk),
                     wdm6_add(qv,mixing_ratio)));
        rr[k]=rs[k]=rg[k]=wdm6_to_float(minimum_mass);
        lqr[k]=lqs[k]=lqg[k]=false;
        if (r.qr[ix]>wdm6_to_float(mass_threshold)) {
            rr[k]=wdm6_to_float(wdm6_mul(wdm6_from_float(r.qr[ix]),rho));
            ulong nr=wdm6_mul(wdm6_from_float(r.nr[ix]),rho);
            ulong lambda=wdm6_power(wdm6_div(
                wdm6_mul(wdm6_mul(xamr,xcrg3),wdm6_mul(xorg2,nr)),
                wdm6_from_float(rr[k])),exponent);
            ilamr[k]=wdm6_div(one,lambda);
            n0r[k]=wdm6_mul(wdm6_mul(nr,xorg2),wdm6_mul(lambda,lambda));
            lqr[k]=true;
        }
        ulong tc=wdm6_sub(tk,temperature0);
        if (soft64_greater(tc,tc_max)) tc=tc_max;
        n0s[k]=wdm6_mul(snow_prefactor,
            soft64_plm_exp(wdm6_mul(snow_exp,tc)));
        if (n0s[k]>snow_cap) n0s[k]=snow_cap;
        if (r.qs[ix]>wdm6_to_float(mass_threshold)) {
            rs[k]=wdm6_to_float(wdm6_mul(wdm6_from_float(r.qs[ix]),rho));
            ulong lambda=wdm6_power(wdm6_div(
                wdm6_mul(wdm6_mul(xams,gamma4),n0s[k]),
                wdm6_from_float(rs[k])),quarter);
            ilams[k]=wdm6_div(one,lambda); lqs[k]=true;
        }
        if (r.qg[ix]>wdm6_to_float(mass_threshold)) {
            rg[k]=wdm6_to_float(wdm6_mul(wdm6_from_float(r.qg[ix]),rho));
            ulong lambda=wdm6_power(wdm6_div(
                wdm6_mul(wdm6_mul(xamg,gamma4),n0g),
                wdm6_from_float(rg[k])),quarter);
            ilamg[k]=wdm6_div(one,lambda); lqg[k]=true;
        }
    }
    bool melting=false; int k0=0;
    for (int k=p.nz-2;k>=0;--k) {
        uint ix=col+uint(k*p.ny*p.nx);
        if (r.t[ix]>wdm6_to_float(temperature0) && lqr[k]
                && (lqs[k+1]||lqg[k+1])) {
            k0=max(k+1,k0); melting=true; break;
        }
    }
    for (int k=0;k<p.nz;++k) {
        zer[k]=zes[k]=zeg[k]=wdm6_to_float(minimum_z);
        if (lqr[k]) zer[k]=wdm6_to_float(wdm6_mul(
            wdm6_mul(n0r[k],xcrg4),wdm6_pow_eight(ilamr[k])));
        if (lqs[k]) {
            ulong ze=wdm6_mul(wdm6_mul(snow_factor,n0s[k]),gamma7);
            ze=wdm6_mul(ze,wdm6_pow_seven(ilams[k]));
            ulong amratio=wdm6_div(xams,ninehundred);
            ulong sixpi=wdm6_div(soft64_from_i32(6),pi);
            ze=wdm6_mul(ze,wdm6_mul(amratio,amratio));
            zes[k]=wdm6_to_float(wdm6_mul(ze,wdm6_mul(sixpi,sixpi)));
        }
        if (lqg[k]) {
            ulong ze=wdm6_mul(snow_factor,n0g);
            ze=wdm6_mul(ze,gamma7);
            ze=wdm6_mul(ze,wdm6_pow_seven(ilamg[k]));
            ulong amratio=wdm6_div(xamg,ninehundred);
            ze=wdm6_mul(ze,wdm6_mul(amratio,amratio));
            ze=wdm6_mul(ze,wdm6_mul(wdm6_div(
                soft64_from_i32(6),pi),wdm6_div(soft64_from_i32(6),pi)));
            zeg[k]=wdm6_to_float(ze);
        }
    }
    if (melting && k0>=1) {
        for (int k=k0-1;k>=0;--k) {
            if (lqs[k]&&lqs[k0]) {
                ulong fraction=wdm6_sub(one,wdm6_div(
                    wdm6_from_float(rs[k]),wdm6_from_float(rs[k0])));
                if (fraction<melt_min) fraction=melt_min;
                if (fraction>melt_max) fraction=melt_max;
                ulong eta=zero, lambda=wdm6_div(one,ilams[k]);
                for (int n=0;n<50;++n) {
                    ulong d=tab[45u+uint(n)], width=tab[95u+uint(n)];
                    ulong mass=wdm6_mul(wdm6_mul(xams,d),d);
                    mass=wdm6_mul(mass,d);
                    ulong cb=wdm6_soak(mass,xocms,exponent,fraction,
                        meltoutside,mw,mi,pi,pi5,lambda4,min_volume,one,zero,
                        ten,ninehundred,thousand,minimum_diameter,six);
                    ulong term=wdm6_mul(n0s[k],soft64_plm_exp(
                        wdm6_neg(wdm6_mul(lambda,d))));
                    term=wdm6_mul(wdm6_mul(term,cb),tab[245u+uint(n)]);
                    eta=wdm6_add(eta,wdm6_mul(term,width));
                }
                zes[k]=wdm6_to_float(wdm6_mul(
                    wdm6_div(lambda4,wdm6_mul(pi5,kw)),eta));
            }
            if (lqg[k]&&lqg[k0]) {
                ulong fraction=wdm6_sub(one,wdm6_div(
                    wdm6_from_float(rg[k]),wdm6_from_float(rg[k0])));
                if (fraction<melt_min) fraction=melt_min;
                if (fraction>melt_max) fraction=melt_max;
                ulong eta=zero, lambda=wdm6_div(one,ilamg[k]);
                for (int n=0;n<50;++n) {
                    ulong d=tab[145u+uint(n)], width=tab[195u+uint(n)];
                    ulong mass=wdm6_mul(wdm6_mul(xamg,d),d);
                    mass=wdm6_mul(mass,d);
                    ulong cb=wdm6_soak(mass,xocmg,exponent,fraction,
                        meltoutside,mw,mi,pi,pi5,lambda4,min_volume,one,zero,
                        ten,ninehundred,thousand,minimum_diameter,six);
                    ulong term=wdm6_mul(n0g,soft64_plm_exp(
                        wdm6_neg(wdm6_mul(lambda,d))));
                    term=wdm6_mul(wdm6_mul(term,cb),tab[245u+uint(n)]);
                    eta=wdm6_add(eta,wdm6_mul(term,width));
                }
                zeg[k]=wdm6_to_float(wdm6_mul(
                    wdm6_div(lambda4,wdm6_mul(pi5,kw)),eta));
            }
        }
    }
    for (int k=0;k<p.nz;++k) {
        uint ix=col+uint(k*p.ny*p.nx);
        float zsum=(zer[k]+zes[k])+zeg[k];
        ulong arg=wdm6_mul(wdm6_from_float(zsum),e18);
        ulong dbz=wdm6_div(wdm6_mul(ten,soft64_plm_log(arg)),ln10);
        float value=wdm6_to_float(dbz);
        r.refl[ix]=isnan(value)||value<wdm6_to_float(output_floor)
            ? wdm6_to_float(output_floor) : value;
    }
}
