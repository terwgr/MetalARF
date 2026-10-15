#include <metal_stdlib>
using namespace metal;
#include "refl10cm_wsm6_column_bindings.metalh"
#include "soft_binary64.metalh"
#include "soft_libm64.metalh"

struct Wsm6Complex { ulong re; ulong im; };

static inline ulong wsm6_add(ulong a, ulong b) { return soft64_add(a,b); }
static inline ulong wsm6_sub(ulong a, ulong b) { return soft64_sub(a,b); }
static inline ulong wsm6_mul(ulong a, ulong b) { return soft64_mul(a,b); }
static inline ulong wsm6_div(ulong a, ulong b) { return soft64_div(a,b); }
static inline ulong wsm6_neg(ulong a) { return soft64_negate_bits(a); }
static inline ulong wsm6_from_float(float x) {
    return soft64_from_f32_bits(as_type<uint>(x));
}
static inline float wsm6_to_float(ulong x) {
    return as_type<float>(soft64_to_f32_bits(x));
}
static inline ulong wsm6_pow(ulong x, ulong exponent) {
    return soft64_plm_exp(wsm6_mul(soft64_plm_log(x), exponent));
}
static inline ulong wsm6_power(ulong x, ulong exponent) {
    return wsm6_pow(x, exponent);
}
static inline ulong wsm6_pow_eight(ulong x) {
    ulong x2=wsm6_mul(x,x), x4=wsm6_mul(x2,x2);
    return wsm6_mul(x4,x4);
}
static inline ulong wsm6_pow_seven(ulong x) {
    ulong x2=wsm6_mul(x,x), x4=wsm6_mul(x2,x2);
    return wsm6_mul(wsm6_mul(x4,x2),x);
}
static inline Wsm6Complex wsm6_cadd(Wsm6Complex a, Wsm6Complex b) {
    Wsm6Complex c={wsm6_add(a.re,b.re),wsm6_add(a.im,b.im)}; return c;
}
static inline Wsm6Complex wsm6_csub(Wsm6Complex a, Wsm6Complex b) {
    Wsm6Complex c={wsm6_sub(a.re,b.re),wsm6_sub(a.im,b.im)}; return c;
}
static inline Wsm6Complex wsm6_cmul(Wsm6Complex a, Wsm6Complex b) {
    Wsm6Complex c={wsm6_sub(wsm6_mul(a.re,b.re),wsm6_mul(a.im,b.im)),
                   wsm6_add(wsm6_mul(a.re,b.im),wsm6_mul(a.im,b.re))};
    return c;
}
static inline Wsm6Complex wsm6_cdiv(Wsm6Complex a, Wsm6Complex b) {
    ulong d=wsm6_add(wsm6_mul(b.re,b.re),wsm6_mul(b.im,b.im));
    Wsm6Complex c={wsm6_div(wsm6_add(wsm6_mul(a.re,b.re),
                                      wsm6_mul(a.im,b.im)),d),
                     wsm6_div(wsm6_sub(wsm6_mul(a.im,b.re),
                                      wsm6_mul(a.re,b.im)),d)};
    return c;
}
static inline ulong wsm6_cabs(Wsm6Complex a) {
    return soft64_sqrt(wsm6_add(wsm6_mul(a.re,a.re),wsm6_mul(a.im,a.im)));
}
static inline ulong wsm6_atan_unit(ulong x) {
    ulong one=soft64_from_i32(1);
    const ulong pi4=0x3fe921fb54442d18ul;
    const ulong tan_pi8=0x3fda827999fcef32ul;
    bool shifted=x>tan_pi8;
    if (shifted) x=wsm6_div(wsm6_sub(x,one),wsm6_add(x,one));
    ulong x2=wsm6_mul(x,x), term=x, sum=x;
    for (int n=1;n<24;++n) {
        term=wsm6_neg(wsm6_mul(term,x2));
        sum=wsm6_add(sum,wsm6_div(term,soft64_from_i32(2*n+1)));
    }
    return shifted ? wsm6_add(pi4,sum) : sum;
}
static inline ulong wsm6_atan2(ulong y, ulong x) {
    const ulong pi=0x400921fb54442d18ul;
    const ulong pi2=0x3ff921fb54442d18ul;
    ulong ax=soft64_abs_bits(x), ay=soft64_abs_bits(y);
    ulong angle=ax>=ay ? wsm6_atan_unit(wsm6_div(ay,ax))
                       : wsm6_sub(pi2,wsm6_atan_unit(wsm6_div(ax,ay)));
    if (soft64_sign(x)) angle=wsm6_sub(pi,angle);
    if (soft64_sign(y)) angle=wsm6_neg(angle);
    return angle;
}
static inline Wsm6Complex wsm6_csqrt(Wsm6Complex a, ulong half_bits) {
    ulong mag=wsm6_cabs(a);
    ulong realarg=wsm6_mul(half_bits,wsm6_add(mag,a.re));
    ulong imagarg=wsm6_mul(half_bits,wsm6_sub(mag,a.re));
    if (soft64_sign(realarg)) realarg=0ul;
    if (soft64_sign(imagarg)) imagarg=0ul;
    ulong im=soft64_sqrt(imagarg);
    Wsm6Complex c={soft64_sqrt(realarg),
                   soft64_less(a.im,0ul)?wsm6_neg(im):im};
    return c;
}
static inline Wsm6Complex wsm6_clog(Wsm6Complex a) {
    Wsm6Complex c={soft64_plm_log(wsm6_cabs(a)),
                   wsm6_atan2(a.im,a.re)};
    return c;
}
static inline Wsm6Complex wsm6_cscale(ulong scale, Wsm6Complex a) {
    Wsm6Complex c={wsm6_mul(scale,a.re),wsm6_mul(scale,a.im)}; return c;
}
static inline Wsm6Complex wsm6_maxwell_garnett(
        ulong v2, ulong v3, Wsm6Complex m1, Wsm6Complex m2,
        Wsm6Complex m3, ulong one, ulong two) {
    Wsm6Complex a=wsm6_cmul(m1,m1), b=wsm6_cmul(m2,m2);
    Wsm6Complex c=wsm6_cmul(m3,m3), db=wsm6_csub(b,a), dc=wsm6_csub(c,a);
    Wsm6Complex beta2=wsm6_cmul(
        wsm6_cdiv(wsm6_cscale(two,a),db),
        wsm6_csub(wsm6_cmul(wsm6_cdiv(b,db),
                              wsm6_clog(wsm6_cdiv(b,a))),
                   Wsm6Complex{one,0ul}));
    Wsm6Complex beta3=wsm6_cmul(
        wsm6_cdiv(wsm6_cscale(two,a),dc),
        wsm6_csub(wsm6_cmul(wsm6_cdiv(c,dc),
                              wsm6_clog(wsm6_cdiv(c,a))),
                   Wsm6Complex{one,0ul}));
    ulong v1=wsm6_sub(wsm6_sub(one,v2),v3);
    Wsm6Complex numerator=wsm6_cadd(
        wsm6_cscale(v1,a),
        wsm6_cadd(wsm6_cscale(v2,wsm6_cmul(beta2,b)),
                  wsm6_cscale(v3,wsm6_cmul(beta3,c))));
    Wsm6Complex denominator=wsm6_cadd(
        Wsm6Complex{v1,0ul},
        wsm6_cadd(wsm6_cscale(v2,beta2),wsm6_cscale(v3,beta3)));
    return wsm6_csqrt(wsm6_cdiv(numerator,denominator),
                       wsm6_div(one,soft64_from_i32(2)));
}
static inline Wsm6Complex wsm6_mix_nested(
        Wsm6Complex ma, Wsm6Complex mi, Wsm6Complex mw,
        ulong vair, ulong vice, ulong vwater, ulong minimum_volume,
        ulong one, ulong two) {
    ulong volume=wsm6_add(vice,vwater);
    if (volume<minimum_volume) volume=minimum_volume;
    ulong icefrac=wsm6_div(vice,volume);
    Wsm6Complex inner=wsm6_maxwell_garnett(0ul,icefrac,mw,ma,mi,one,two);
    return wsm6_maxwell_garnett(vair,0ul,inner,ma,
                                wsm6_cscale(two,ma),one,two);
}
static inline ulong wsm6_soak(
        ulong mass, ulong ageo, ulong exponent, ulong fmelt,
        ulong meltoutside, Wsm6Complex mw, Wsm6Complex mi,
        ulong pi, ulong pi5, ulong lambda4, ulong minimum_volume,
        ulong one, ulong two, ulong zero, ulong ten, ulong ninehundred,
        ulong thousand, ulong minimum_diameter, ulong six) {
    ulong fm=fmelt;
    if (soft64_sign(fm)) fm=zero;
    if (fm>one) fm=one;
    ulong mra=meltoutside;
    if (soft64_sign(mra)) mra=zero;
    if (mra>one) mra=one;
    mra=wsm6_add(mra,wsm6_mul(wsm6_sub(one,mra),fm));
    ulong watermass=wsm6_mul(mass,fm);
    ulong diameter=wsm6_mul(ageo,wsm6_power(mass,exponent));
    if (diameter<minimum_diameter) return zero;
    ulong volume=wsm6_div(pi,six);
    volume=wsm6_mul(wsm6_mul(volume,diameter),diameter);
    volume=wsm6_mul(volume,diameter);
    ulong density=wsm6_div(mass,volume);
    if (soft64_less(density,ten)) density=ten;
    if (density>ninehundred) density=ninehundred;
    volume=wsm6_div(mass,density);
    ulong boundary=wsm6_sub(one,wsm6_div(density,thousand));
    ulong meltvolume;
    if (mra<=boundary) {
        meltvolume=wsm6_mul(volume,wsm6_sub(one,wsm6_mul(mra,fm)));
    } else {
        ulong numerator=wsm6_sub(ninehundred,density);
        ulong denominator=wsm6_add(
            wsm6_sub(wsm6_mul(mra,ninehundred),density),
            wsm6_div(wsm6_mul(ninehundred,density),thousand));
        ulong fm_boundary=wsm6_div(numerator,denominator);
        meltvolume=fm<=fm_boundary
            ? wsm6_mul(wsm6_sub(one,wsm6_mul(mra,fm)),volume)
            : wsm6_add(wsm6_div(wsm6_sub(mass,watermass),ninehundred),
                       wsm6_div(watermass,thousand));
    }
    ulong large_diameter=wsm6_power(
        wsm6_mul(wsm6_div(six,pi),meltvolume),exponent);
    ulong icevolume=wsm6_div(wsm6_sub(mass,watermass),
                              wsm6_mul(meltvolume,ninehundred));
    ulong watervolume=wsm6_div(watermass,wsm6_mul(thousand,meltvolume));
    ulong airvolume=wsm6_sub(wsm6_sub(one,icevolume),watervolume);
    Wsm6Complex air={one,zero};
    Wsm6Complex core=wsm6_mix_nested(air,mi,mw,airvolume,icevolume,
        watervolume,minimum_volume,one,two);
    Wsm6Complex core2=wsm6_cmul(core,core);
    Wsm6Complex numerator=wsm6_csub(core2,air);
    Wsm6Complex denominator=wsm6_cadd(core2,Wsm6Complex{two,zero});
    ulong kfac=wsm6_cabs(wsm6_cdiv(numerator,denominator));
    ulong d3=wsm6_mul(wsm6_mul(large_diameter,large_diameter),large_diameter);
    ulong result=wsm6_mul(wsm6_mul(kfac,kfac),pi5);
    result=wsm6_mul(wsm6_mul(result,d3),d3);
    return wsm6_div(result,lambda4);
}

kernel void refl10cm_wsm6_column(
        constant Refl10CmWsm6ColumnResources& r [[buffer(0)]],
        constant Refl10CmWsm6ColumnParams& p [[buffer(1)]],
        uint col [[thread_position_in_grid]]) {
    if (p.nz<1 || p.ny<1 || p.nx<1 || col>=uint(p.ny*p.nx)) return;
    const device ulong* tab=r.tables;
    Wsm6Complex mw={tab[1],tab[2]}, mi={tab[3],tab[4]};
    ulong kw=tab[0], xamr=tab[5], xams=tab[6], xamg=tab[7];
    ulong pi=tab[8], pi5=tab[9], lambda4=tab[10];
    ulong rd=tab[11], mixing_ratio=tab[12], qv_floor=tab[13];
    ulong mass_threshold=tab[14], min_volume=tab[15], meltoutside=tab[16];
    ulong gamma4=tab[17], gamma7=tab[18], snow_factor=tab[19];
    ulong ninehundred=tab[20], thousand=tab[21], e18=tab[22];
    ulong ten=tab[23], temperature0=tab[24], xcrg3=tab[25];
    ulong xcrg4=tab[26], n0rain=tab[27], n0g=tab[28];
    ulong xocms=tab[29], xocmg=tab[30], exponent=tab[31];
    ulong minimum_diameter=tab[32], minimum_mass=tab[33], minimum_z=tab[34];
    ulong tc_max=tab[35], snow_exp=tab[36], snow_prefactor=tab[37];
    ulong snow_cap=tab[38], melt_min=tab[39], melt_max=tab[40];
    ulong output_floor=tab[41], ln10=tab[42], six=tab[43], quarter=tab[44];
    ulong one=soft64_from_i32(1), zero=0ul, two=soft64_from_i32(2);
    float rr[256],rs[256],rg[256];
    ulong ilamr[256],ilams[256],ilamg[256],n0s[256];
    bool lqr[256],lqs[256],lqg[256];
    float zer[256],zes[256],zeg[256];
    for (int k=0;k<p.nz;++k) {
        uint ix=col+uint(k*p.ny*p.nx);
        ulong tk=wsm6_from_float(r.t[ix]);
        ulong qv=wsm6_from_float(r.qv[ix]);
        if (soft64_is_nan(qv) || soft64_less(qv,qv_floor)) qv=qv_floor;
        ulong rho=wsm6_div(wsm6_mul(mixing_ratio,
                                    wsm6_from_float(r.p[ix])),
            wsm6_mul(wsm6_mul(rd,tk),
                     wsm6_add(qv,mixing_ratio)));
        rr[k]=rs[k]=rg[k]=wsm6_to_float(minimum_mass);
        lqr[k]=lqs[k]=lqg[k]=false;
        if (r.qr[ix]>wsm6_to_float(mass_threshold)) {
            rr[k]=wsm6_to_float(wsm6_mul(wsm6_from_float(r.qr[ix]),rho));
            ulong lambda=wsm6_power(wsm6_div(
                wsm6_mul(wsm6_mul(xamr,gamma4),n0rain),
                wsm6_from_float(rr[k])),quarter);
            ilamr[k]=wsm6_div(one,lambda);
            lqr[k]=true;
        }
        ulong tc=wsm6_sub(tk,temperature0);
        if (soft64_greater(tc,tc_max)) tc=tc_max;
        n0s[k]=wsm6_mul(snow_prefactor,
            soft64_plm_exp(wsm6_mul(snow_exp,tc)));
        if (n0s[k]>snow_cap) n0s[k]=snow_cap;
        if (r.qs[ix]>wsm6_to_float(mass_threshold)) {
            rs[k]=wsm6_to_float(wsm6_mul(wsm6_from_float(r.qs[ix]),rho));
            ulong lambda=wsm6_power(wsm6_div(
                wsm6_mul(wsm6_mul(xams,gamma4),n0s[k]),
                wsm6_from_float(rs[k])),quarter);
            ilams[k]=wsm6_div(one,lambda); lqs[k]=true;
        }
        if (r.qg[ix]>wsm6_to_float(mass_threshold)) {
            rg[k]=wsm6_to_float(wsm6_mul(wsm6_from_float(r.qg[ix]),rho));
            ulong lambda=wsm6_power(wsm6_div(
                wsm6_mul(wsm6_mul(xamg,gamma4),n0g),
                wsm6_from_float(rg[k])),quarter);
            ilamg[k]=wsm6_div(one,lambda); lqg[k]=true;
        }
    }
    bool melting=false; int k0=0;
    for (int k=p.nz-2;k>=0;--k) {
        uint ix=col+uint(k*p.ny*p.nx);
        if (r.t[ix]>wsm6_to_float(temperature0) && lqr[k]
                && (lqs[k+1]||lqg[k+1])) {
            k0=max(k+1,k0); melting=true; break;
        }
    }
    for (int k=0;k<p.nz;++k) {
        zer[k]=zes[k]=zeg[k]=wsm6_to_float(minimum_z);
        if (lqr[k]) zer[k]=wsm6_to_float(wsm6_mul(
            wsm6_mul(n0rain,gamma7),wsm6_pow_seven(ilamr[k])));
        if (lqs[k]) {
            ulong sixpi=wsm6_div(soft64_from_i32(6),pi);
            ulong amratio=wsm6_div(xams,ninehundred);
            ulong ze=wsm6_mul(snow_factor,wsm6_mul(sixpi,sixpi));
            ze=wsm6_mul(ze,wsm6_mul(amratio,amratio));
            ze=wsm6_mul(ze,n0s[k]);
            ze=wsm6_mul(ze,gamma7);
            ze=wsm6_mul(ze,wsm6_pow_seven(ilams[k]));
            zes[k]=wsm6_to_float(ze);
        }
        if (lqg[k]) {
            ulong sixpi=wsm6_div(soft64_from_i32(6),pi);
            ulong amratio=wsm6_div(xamg,ninehundred);
            ulong ze=wsm6_mul(snow_factor,wsm6_mul(sixpi,sixpi));
            ze=wsm6_mul(ze,wsm6_mul(amratio,amratio));
            ze=wsm6_mul(ze,n0g);
            ze=wsm6_mul(ze,gamma7);
            ze=wsm6_mul(ze,wsm6_pow_seven(ilamg[k]));
            zeg[k]=wsm6_to_float(ze);
        }
    }
    if (melting && k0>=1) {
        for (int k=k0-1;k>=0;--k) {
            if (lqs[k]&&lqs[k0]) {
                ulong fraction=wsm6_sub(one,wsm6_div(
                    wsm6_from_float(rs[k]),wsm6_from_float(rs[k0])));
                if (soft64_less(fraction,melt_min)) fraction=melt_min;
                if (soft64_greater(fraction,melt_max)) fraction=melt_max;
                ulong eta=zero, lambda=wsm6_div(one,ilams[k]);
                for (int n=0;n<50;++n) {
                    ulong d=tab[45u+uint(n)], width=tab[95u+uint(n)];
                    ulong mass=wsm6_mul(wsm6_mul(xams,d),d);
                    mass=wsm6_mul(mass,d);
                    ulong cb=wsm6_soak(mass,xocms,exponent,fraction,
                        meltoutside,mw,mi,pi,pi5,lambda4,min_volume,one,two,zero,
                        ten,ninehundred,thousand,minimum_diameter,six);
                    ulong term=wsm6_mul(n0s[k],soft64_plm_exp(
                        wsm6_neg(wsm6_mul(lambda,d))));
                    term=wsm6_mul(wsm6_mul(term,cb),tab[245u+uint(n)]);
                    eta=wsm6_add(eta,wsm6_mul(term,width));
                }
                zes[k]=wsm6_to_float(wsm6_mul(
                    wsm6_div(lambda4,wsm6_mul(pi5,kw)),eta));
            }
            if (lqg[k]&&lqg[k0]) {
                ulong fraction=wsm6_sub(one,wsm6_div(
                    wsm6_from_float(rg[k]),wsm6_from_float(rg[k0])));
                if (soft64_less(fraction,melt_min)) fraction=melt_min;
                if (soft64_greater(fraction,melt_max)) fraction=melt_max;
                ulong eta=zero, lambda=wsm6_div(one,ilamg[k]);
                for (int n=0;n<50;++n) {
                    ulong d=tab[145u+uint(n)], width=tab[195u+uint(n)];
                    ulong mass=wsm6_mul(wsm6_mul(xamg,d),d);
                    mass=wsm6_mul(mass,d);
                    ulong cb=wsm6_soak(mass,xocmg,exponent,fraction,
                        meltoutside,mw,mi,pi,pi5,lambda4,min_volume,one,two,zero,
                        ten,ninehundred,thousand,minimum_diameter,six);
                    ulong term=wsm6_mul(n0g,soft64_plm_exp(
                        wsm6_neg(wsm6_mul(lambda,d))));
                    term=wsm6_mul(wsm6_mul(term,cb),tab[245u+uint(n)]);
                    eta=wsm6_add(eta,wsm6_mul(term,width));
                }
                zeg[k]=wsm6_to_float(wsm6_mul(
                    wsm6_div(lambda4,wsm6_mul(pi5,kw)),eta));
            }
        }
    }
    for (int k=0;k<p.nz;++k) {
        uint ix=col+uint(k*p.ny*p.nx);
        float zsum=(zer[k]+zes[k])+zeg[k];
        ulong arg=wsm6_mul(wsm6_from_float(zsum),e18);
        ulong dbz=wsm6_div(wsm6_mul(ten,soft64_plm_log(arg)),ln10);
        float value=wsm6_to_float(dbz);
        r.refl[ix]=isnan(value)||value<wsm6_to_float(output_floor)
            ? wsm6_to_float(output_floor) : value;
    }
}
