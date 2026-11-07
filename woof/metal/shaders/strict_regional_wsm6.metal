#include <metal_stdlib>
#include "wsm_support.metalh"
#pragma clang fp contract(off)

#include "hexv842_mpas_thompson_column_math.metalh"
static inline Sf __fdiv_rn(Sf a, Sf b) { return aa_div_impl(a,b); }
static inline Sf gfk_exp(Sf a) { return sf_make(native_tc_math::thompson_aa_expf(f32_word(sf_bits(a))).bits); }
static inline Sf gfk_log(Sf a) { return sf_make(soft_gfk_log(sf_bits(a))); }
static inline Sf gfk_pow(Sf a, Sf b) { return sf_make(soft_gfk_pow_bits(sf_bits(a),sf_bits(b))); }
struct Wsm6ColumnResources {
 device float *theta [[id(0)]];
 device float *qv [[id(1)]];
 device float *qc [[id(2)]];
 device float *qi [[id(3)]];
 device float *qr [[id(4)]];
 device float *qs [[id(5)]];
 device float *qg [[id(6)]];
 device const float *den [[id(7)]];
 device const float *p [[id(8)]];
 device const float *pii [[id(9)]];
 device const float *dz [[id(10)]];
 device float *rainnc [[id(11)]];
 device float *rainncv [[id(12)]];
 device float *snownc [[id(13)]];
 device float *snowncv [[id(14)]];
 device float *graupelnc [[id(15)]];
 device float *graupelncv [[id(16)]];
 device float *sr [[id(17)]];
 device float *effc [[id(18)]];
 device float *effi [[id(19)]];
 device float *effs [[id(20)]];
 device const float *coef [[id(21)]];
};
struct Wsm6ColumnParams {
float delt;
int nz;
int ny;
int nx;
};

namespace t64 {
enum Wsm6Coef {
C_PI , C_XLV1 , C_QC0 , C_QCK1 , C_PIDNC , C_PIDN0R , C_PIDN0S , C_PIDN0G , C_PVTR , C_PVTS , C_PVTG , C_PACRR , C_PACRC , C_PACRG , C_PRECR1 , C_PRECR2 , C_PRECS1 , C_PRECS2 , C_PRECG1 , C_PRECG2 , C_G6PBR , C_ROQIMAX , C_N0G , C_BVTG , C_RSLOPERMAX , C_RSLOPESMAX , C_RSLOPEGMAX , C_RSLOPERBMAX , C_RSLOPESBMAX , C_RSLOPEGBMAX , C_RSLOPER2MAX , C_RSLOPES2MAX , C_RSLOPEG2MAX , C_RSLOPER3MAX , C_RSLOPES3MAX , C_RSLOPEG3MAX , WSM6_NCOEF }
;

static inline Sf wmax ( Sf a , Sf b ) {
return ( b > a ) ? b : a ;
}

static inline Sf wmin ( Sf a , Sf b ) {
return ( b < a ) ? b : a ;
}

static inline Sf st_diffus ( Sf x , Sf y ) {
return aa_div ( ( Sf(8.794e-5f) * gfk_exp ( gfk_log ( x ) * Sf(1.81f) ) ) , y ) ;
}

static inline Sf st_viscos ( Sf x , Sf y ) {
return aa_div ( ( aa_div ( ( Sf(1.496e-6f) * ( x * aa_sqrtf ( x ) ) ) , ( x + Sf(120.0f) ) ) ) , y ) ;
}

static inline Sf st_xka ( Sf x , Sf y ) {
return ( Sf(1.414e3f) * st_viscos ( x , y ) ) * y ;
}

static inline Sf st_diffac ( Sf a , Sf b , Sf c , Sf d , Sf e ) {
return aa_div ( ( ( d * a ) * a ) , ( ( ( st_xka ( c , d ) * Sf(461.6f) ) * c ) * c ) ) + aa_div ( Sf(1.0f) , ( e * st_diffus ( c , b ) ) ) ;
}

static inline Sf st_venfac ( Sf a , Sf b , Sf c ) {
return ( aa_div ( gfk_exp ( gfk_log ( aa_div ( st_viscos ( b , c ) , st_diffus ( b , a ) ) ) * Sf(0.3333333f) ) , aa_sqrtf ( st_viscos ( b , c ) ) ) ) * aa_sqrtf ( aa_sqrtf ( aa_div ( Sf(1.28f) , c ) ) ) ;
}

static inline Sf st_conden ( Sf a , Sf b , Sf c , Sf d , Sf e ) {
return aa_div ( ( wmax ( b , Sf(1.0e-15f) ) - c ) , ( Sf(1.0f) + aa_div ( ( ( aa_div ( ( d * d ) , ( Sf(461.6f) * e ) ) ) * c ) , ( a * a ) ) ) ) ;
}

struct Wsm6Fpvs {
Sf ttp , xa , xb , xai , xbi ;
}
;

static inline Wsm6Fpvs wsm6_fpvs_consts ( ) {
Wsm6Fpvs f ;
f . ttp = Sf(273.15f) + Sf(0.01f) ;
const Sf dldt = Sf(1846.4f) - Sf(4190.0f) ;
f . xa = __fdiv_rn ( - dldt , Sf(461.6f) ) ;
f . xb = f . xa + __fdiv_rn ( Sf(2.5e6f) , Sf(461.6f) * f . ttp ) ;
const Sf dldti = Sf(1846.4f) - Sf(2106.0f) ;
f . xai = __fdiv_rn ( - dldti , Sf(461.6f) ) ;
f . xbi = f . xai + __fdiv_rn ( Sf(2.85e6f) , Sf(461.6f) * f . ttp ) ;
return f ;
}

static inline Sf wsm6_qsat ( thread const Wsm6Fpvs & f , Sf t , Sf p , bool ice ) {
const Sf tr = aa_div ( f . ttp , t ) ;
Sf q ;
if ( ice && t < f . ttp ) q = ( Sf(610.78f) * gfk_exp ( gfk_log ( tr ) * f . xai ) ) * gfk_exp ( f . xbi * ( Sf(1.0f) - tr ) ) ;
else q = ( Sf(610.78f) * gfk_exp ( gfk_log ( tr ) * f . xa ) ) * gfk_exp ( f . xb * ( Sf(1.0f) - tr ) ) ;
q = wmin ( q , Sf(0.99f) * p ) ;
q = aa_div ( ( __fdiv_rn ( Sf(287.0f) , Sf(461.6f) ) * q ) , ( p - q ) ) ;
return wmax ( q , Sf(1.0e-15f) ) ;
}

struct Wsm6Slope {
Sf r , rb , r2 , r3 , vt ;
}
;

template < typename P_c > static inline Wsm6Slope wsm6_slope_rain ( P_c c , Sf q , Sf den , Sf denfac ) {
Wsm6Slope s ;
if ( q <= Sf(1.0e-9f) ) {
s . r = c [ C_RSLOPERMAX ] ;
s . rb = c [ C_RSLOPERBMAX ] ;
s . r2 = c [ C_RSLOPER2MAX ] ;
s . r3 = c [ C_RSLOPER3MAX ] ;
}
else {
s . r = aa_div ( Sf(1.0f) , aa_sqrtf ( aa_sqrtf ( aa_div ( c [ C_PIDN0R ] , ( q * den ) ) ) ) ) ;
s . rb = gfk_pow ( s . r , Sf(0.8f) ) ;
s . r2 = s . r * s . r ;
s . r3 = s . r2 * s . r ;
}
s . vt = ( c [ C_PVTR ] * s . rb ) * denfac ;
if ( Sf(q) <= Sf(Sf(0.0f)) ) s . vt = Sf(0.0f) ;
return s ;
}

template < typename P_c > static inline Wsm6Slope wsm6_slope_snow ( P_c c , Sf q , Sf den , Sf denfac , Sf t ) {
Wsm6Slope s ;
const Sf supcol = Sf(273.15f) - t ;
const Sf n0sfac = wmax ( wmin ( gfk_exp ( Sf(0.12f) * supcol ) , Sf(5.0e4f) ) , Sf(1.0f) ) ;
if ( q <= Sf(1.0e-9f) ) {
s . r = c [ C_RSLOPESMAX ] ;
s . rb = c [ C_RSLOPESBMAX ] ;
s . r2 = c [ C_RSLOPES2MAX ] ;
s . r3 = c [ C_RSLOPES3MAX ] ;
}
else {
s . r = aa_div ( Sf(1.0f) , aa_sqrtf ( aa_sqrtf ( aa_div ( ( c [ C_PIDN0S ] * n0sfac ) , ( q * den ) ) ) ) ) ;
s . rb = gfk_pow ( s . r , Sf(0.41f) ) ;
s . r2 = s . r * s . r ;
s . r3 = s . r2 * s . r ;
}
s . vt = ( c [ C_PVTS ] * s . rb ) * denfac ;
if ( Sf(q) <= Sf(Sf(0.0f)) ) s . vt = Sf(0.0f) ;
return s ;
}

template < typename P_c > static inline Wsm6Slope wsm6_slope_graup ( P_c c , Sf q , Sf den , Sf denfac ) {
Wsm6Slope s ;
if ( q <= Sf(1.0e-9f) ) {
s . r = c [ C_RSLOPEGMAX ] ;
s . rb = c [ C_RSLOPEGBMAX ] ;
s . r2 = c [ C_RSLOPEG2MAX ] ;
s . r3 = c [ C_RSLOPEG3MAX ] ;
}
else {
s . r = aa_div ( Sf(1.0f) , aa_sqrtf ( aa_sqrtf ( aa_div ( c [ C_PIDN0G ] , ( q * den ) ) ) ) ) ;
s . rb = gfk_pow ( s . r , c [ C_BVTG ] ) ;
s . r2 = s . r * s . r ;
s . r3 = s . r2 * s . r ;
}
s . vt = ( c [ C_PVTG ] * s . rb ) * denfac ;
if ( Sf(q) <= Sf(Sf(0.0f)) ) s . vt = Sf(0.0f) ;
return s ;
}

static inline Sf wsm6_denfac ( Sf den ) {
return aa_sqrtf ( ( aa_div ( Sf(1.0f) , den ) ) * Sf(1.28f) ) ;
}

template < typename P_c , typename P_den , typename P_dz , typename P_tk , typename P_wwl , typename P_rq , typename P_rq2 , typename P_precip1 , typename P_precip2 > static inline void wsm6_nislfv ( int km , P_c c , P_den den , P_dz dz , P_tk tk , P_wwl wwl , P_rq rq , P_rq2 rq2 , P_precip1 precip1 , P_precip2 precip2 , Sf dt , int iter , int mode , int stride ) {
Sf ww [ 65 ] ;
Sf qn [ 65 ] ;
const int k1 = km + 1 , r0 = 2 * km ;
Sf wi [ 65 ] ;
Sf zi [ 65 ] ;
Sf za [ 65 ] ;
Sf dza [ 65 ] ;
Sf qa [ 65 ] ;
Sf qa2 [ 65 ] ;
Sf qmi [ 65 ] ;
Sf qpi [ 65 ] ;
const bool two = ( rq2 != nullptr ) ;
* precip1 = Sf(0.0f) ;
if ( two ) * precip2 = Sf(0.0f) ;
Sf allold = Sf(0.0f) ;
for ( int k = 0 ;
k < km ;
++ k ) {
allold = allold + rq [ k ] ;
if ( two ) allold = allold + rq2 [ k ] ;
}
if ( Sf(allold) <= Sf(Sf(0.0f)) ) return ;
zi [ 0 ] = Sf(0.0f) ;
for ( int k = 0 ;
k < km ;
++ k ) zi [ k + 1 ] = zi [ k ] + dz [ k * stride ] ;
for ( int k = 0 ;
k < km ;
++ k ) ww [ k ] = wwl [ k ] ;
int n = 1 ;
for ( ;
;
) {
const Sf fa1 = aa_div ( Sf(9.0f) , Sf(16.0f) ) , fa2 = aa_div ( Sf(1.0f) , Sf(16.0f) ) ;
wi [ 0 ] = ww [ 0 ] ;
wi [ 1 ] = Sf(0.5f) * ( ww [ 1 ] + ww [ 0 ] ) ;
for ( int k = 2 ;
k <= km - 2 ;
++ k ) wi [ k ] = fa1 * ( ww [ k ] + ww [ k - 1 ] ) - fa2 * ( ww [ k + 1 ] + ww [ k - 2 ] ) ;
wi [ km - 1 ] = Sf(0.5f) * ( ww [ km - 1 ] + ww [ km - 2 ] ) ;
wi [ km ] = ww [ km - 1 ] ;
for ( int k = 1 ;
k < km ;
++ k ) if ( ww [ k ] == Sf(0.0f) ) wi [ k ] = ww [ k - 1 ] ;
const Sf con1 = Sf(0.05f) ;
for ( int k = km - 1 ;
k >= 0 ;
-- k ) {
const Sf dzk = dz [ k * stride ] ;
const Sf decfl = aa_div ( ( ( wi [ k + 1 ] - wi [ k ] ) * dt ) , dzk ) ;
if ( decfl > con1 ) wi [ k ] = wi [ k + 1 ] - aa_div ( ( con1 * dzk ) , dt ) ;
}
for ( int k = 0 ;
k <= km ;
++ k ) za [ k ] = zi [ k ] - wi [ k ] * dt ;
for ( int k = 0 ;
k < km ;
++ k ) dza [ k ] = za [ k + 1 ] - za [ k ] ;
dza [ km ] = zi [ km ] - za [ km ] ;
for ( int k = 0 ;
k < km ;
++ k ) {
const Sf dzk = dz [ k * stride ] ;
qa [ k ] = aa_div ( ( rq [ k ] * dzk ) , dza [ k ] ) ;
if ( two ) qa2 [ k ] = aa_div ( ( rq2 [ k ] * dzk ) , dza [ k ] ) ;
}
qa [ km ] = Sf(0.0f) ;
qa2 [ km ] = Sf(0.0f) ;
if ( n > iter ) break ;
for ( int k = 0 ;
k < km ;
++ k ) {
const Sf denk = den [ k * stride ] ;
const Sf denfac = wsm6_denfac ( denk ) ;
const Sf qr = aa_div ( qa [ k ] , denk ) ;
Sf wa ;
if ( mode == 0 ) {
wa = wsm6_slope_rain ( c , qr , denk , denfac ) . vt ;
}
else {
const Sf qr2 = aa_div ( qa2 [ k ] , denk ) ;
const Sf was = wsm6_slope_snow ( c , qr , denk , denfac , tk [ k ] ) . vt ;
const Sf wa2 = wsm6_slope_graup ( c , qr2 , denk , denfac ) . vt ;
const Sf tmp = wmax ( qr + qr2 , Sf(1.0e-15f) ) ;
wa = ( tmp > Sf(1.0e-15f) ) ? aa_div ( ( was * qr + wa2 * qr2 ) , tmp ) : Sf(0.0f) ;
}
ww [ k ] = Sf(0.5f) * ( wwl [ k ] + wa ) ;
}
n = n + 1 ;
}
for ( int ist = 0 ;
ist < ( two ? 2 : 1 ) ;
++ ist ) {
auto q = ( ist == 0 ) ? qa : qa2 ;
for ( int k = 1 ;
k < km ;
++ k ) {
const Sf dip = aa_div ( ( q [ k + 1 ] - q [ k ] ) , ( dza [ k + 1 ] + dza [ k ] ) ) ;
const Sf dim = aa_div ( ( q [ k ] - q [ k - 1 ] ) , ( dza [ k - 1 ] + dza [ k ] ) ) ;
if ( dip * Sf(dim) <= Sf(Sf(0.0f)) ) {
qmi [ k ] = q [ k ] ;
qpi [ k ] = q [ k ] ;
}
else {
qpi [ k ] = q [ k ] + ( Sf(0.5f) * ( dip + dim ) ) * dza [ k ] ;
qmi [ k ] = Sf(2.0f) * q [ k ] - qpi [ k ] ;
if ( Sf(qpi [ k ]) < Sf(Sf(0.0f)) || Sf(qmi [ k ]) < Sf(Sf(0.0f)) ) {
qpi [ k ] = q [ k ] ;
qmi [ k ] = q [ k ] ;
}
}
}
qpi [ 0 ] = q [ 0 ] ;
qmi [ 0 ] = q [ 0 ] ;
qmi [ km ] = q [ km ] ;
qpi [ km ] = q [ km ] ;
for ( int k = 0 ;
k < km ;
++ k ) qn [ k ] = Sf(0.0f) ;
int kb = 0 , kt = 0 ;
for ( int k = 0 ;
k < km ;
++ k ) {
kb = ( kb - 1 > 0 ) ? kb - 1 : 0 ;
kt = ( kt - 1 > 0 ) ? kt - 1 : 0 ;
if ( zi [ k ] >= za [ km ] ) break ;
for ( int kk = kb ;
kk < km ;
++ kk ) if ( zi [ k ] <= za [ kk + 1 ] ) {
kb = kk ;
break ;
}
for ( int kk = kt ;
kk < km ;
++ kk ) if ( zi [ k + 1 ] <= za [ kk ] ) {
kt = kk ;
break ;
}
kt = kt - 1 ;
if ( kt == kb ) {
const Sf tl = aa_div ( ( zi [ k ] - za [ kb ] ) , dza [ kb ] ) ;
const Sf th = aa_div ( ( zi [ k + 1 ] - za [ kb ] ) , dza [ kb ] ) ;
const Sf tl2 = tl * tl ;
const Sf th2 = th * th ;
const Sf qqd = Sf(0.5f) * ( qpi [ kb ] - qmi [ kb ] ) ;
const Sf qqh = qqd * th2 + qmi [ kb ] * th ;
const Sf qql = qqd * tl2 + qmi [ kb ] * tl ;
qn [ k ] = aa_div ( ( qqh - qql ) , ( th - tl ) ) ;
}
else if ( kt > kb ) {
const Sf tl = aa_div ( ( zi [ k ] - za [ kb ] ) , dza [ kb ] ) ;
const Sf tl2 = tl * tl ;
Sf qqd = Sf(0.5f) * ( qpi [ kb ] - qmi [ kb ] ) ;
const Sf qql = qqd * tl2 + qmi [ kb ] * tl ;
const Sf dql = q [ kb ] - qql ;
Sf zsum = ( Sf(1.0f) - tl ) * dza [ kb ] ;
Sf qsum = dql * dza [ kb ] ;
if ( kt - kb > 1 ) {
for ( int m = kb + 1 ;
m <= kt - 1 ;
++ m ) {
zsum = zsum + dza [ m ] ;
qsum = qsum + q [ m ] * dza [ m ] ;
}
}
const Sf th = aa_div ( ( zi [ k + 1 ] - za [ kt ] ) , dza [ kt ] ) ;
const Sf th2 = th * th ;
qqd = Sf(0.5f) * ( qpi [ kt ] - qmi [ kt ] ) ;
const Sf dqh = qqd * th2 + qmi [ kt ] * th ;
zsum = zsum + th * dza [ kt ] ;
qsum = qsum + dqh * dza [ kt ] ;
qn [ k ] = aa_div ( qsum , zsum ) ;
}
}
Sf precip = Sf(0.0f) ;
for ( int k = 0 ;
k < km ;
++ k ) {
if ( Sf(za [ k ]) < Sf(Sf(0.0f)) && Sf(za [ k + 1 ]) < Sf(Sf(0.0f)) ) {
precip = precip + q [ k ] * dza [ k ] ;
continue ;
}
else if ( Sf(za [ k ]) < Sf(Sf(0.0f)) && Sf(za [ k + 1 ]) >= Sf(Sf(0.0f)) ) {
precip = precip + q [ k ] * ( Sf(0.0f) - za [ k ] ) ;
break ;
}
break ;
}
auto out = ( ist == 0 ) ? rq : rq2 ;
for ( int k = 0 ;
k < km ;
++ k ) out [ k ] = qn [ k ] ;
if ( ist == 0 ) * precip1 = precip ;
else * precip2 = precip ;
}
}

template < typename P_theta , typename P_qv , typename P_qc , typename P_qi , typename P_qr , typename P_qs , typename P_qg , typename P_den , typename P_p , typename P_pii , typename P_dz , typename P_rainnc , typename P_rainncv , typename P_snownc , typename P_snowncv , typename P_graupelnc , typename P_graupelncv , typename P_sr , typename P_effc , typename P_effi , typename P_effs , typename P_coef > static inline void wsm6_column_impl ( P_theta theta , P_qv qv , P_qc qc , P_qi qi , P_qr qr , P_qs qs , P_qg qg , P_den den , P_p p , P_pii pii , P_dz dz , P_rainnc rainnc , P_rainncv rainncv , P_snownc snownc , P_snowncv snowncv , P_graupelnc graupelnc , P_graupelncv graupelncv , P_sr sr , P_effc effc , P_effi effi , P_effs effs , P_coef coef , Sf delt , int nz , int stride , int col ) {
Sf c [ WSM6_NCOEF ] ;
for ( int n = 0 ;
n < WSM6_NCOEF ;
++ n ) c [ n ] = coef [ n ] ;
const Sf pi = c [ C_PI ] ;
const Wsm6Fpvs fp = wsm6_fpvs_consts ( ) ;
Sf t [ 65 ] , q [ 65 ] ;
Sf qcl [ 65 ] , qil [ 65 ] ;
Sf qrl [ 65 ] , qsl [ 65 ] ;
Sf qgl [ 65 ] , cpm [ 65 ] ;
Sf xl [ 65 ] , xni [ 65 ] ;
Sf v1 [ 65 ] , v2 [ 65 ] ;
Sf rq1 [ 65 ] , rq2 [ 65 ] ;
Sf rq3 [ 65 ] ;
auto denc = den + col ;
auto dzc = dz + col ;
auto pc = p + col ;
for ( int k = 0 ;
k < nz ;
++ k ) {
const int id = col + k * stride ;
t [ k ] = Sf(theta [ id ]) * Sf(pii [ id ]) ;
q [ k ] = qv [ id ] ;
qcl [ k ] = wmax ( qc [ id ] , Sf(0.0f) ) ;
qrl [ k ] = wmax ( qr [ id ] , Sf(0.0f) ) ;
qil [ k ] = wmax ( qi [ id ] , Sf(0.0f) ) ;
qsl [ k ] = wmax ( qs [ id ] , Sf(0.0f) ) ;
qgl [ k ] = wmax ( qg [ id ] , Sf(0.0f) ) ;
const Sf qq = wmax ( q [ k ] , Sf(1.0e-15f) ) ;
cpm [ k ] = Sf(1004.5f) * ( Sf(1.0f) - qq ) + qq * Sf(1846.4f) ;
xl [ k ] = Sf(2.5e6f) - c [ C_XLV1 ] * ( t [ k ] - Sf(273.15f) ) ;
}
Sf rainncv_c = Sf(0.0f) , snowncv_c = Sf(0.0f) , graupelncv_c = Sf(0.0f) , sr_c = Sf(0.0f) ;
Sf rain_c = rainnc [ col ] , snow_c = snownc [ col ] , graupel_c = graupelnc [ col ] ;
Sf loopsf = round ( __fdiv_rn ( delt , Sf(120.0f) ) ) ;
int loops = ( int ) loopsf ;
if ( loops < 1 ) loops = 1 ;
Sf dtcld = aa_div ( delt , ( Sf ) loops ) ;
if ( delt <= Sf(120.0f) ) dtcld = delt ;
for ( int loop = 0 ;
loop < loops ;
++ loop ) {
for ( int k = 0 ;
k < nz ;
++ k ) {
const Sf denk = denc [ k * stride ] ;
const Sf denfac = wsm6_denfac ( denk ) ;
Sf temp = denk * wmax ( qil [ k ] , Sf(1.0e-15f) ) ;
temp = aa_sqrtf ( aa_sqrtf ( ( temp * temp ) * temp ) ) ;
xni [ k ] = wmin ( wmax ( Sf(5.38e7f) * temp , Sf(1.0e3f) ) , Sf(1.0e6f) ) ;
const Wsm6Slope sr_ = wsm6_slope_rain ( c , qrl [ k ] , denk , denfac ) ;
const Wsm6Slope ss_ = wsm6_slope_snow ( c , qsl [ k ] , denk , denfac , t [ k ] ) ;
const Wsm6Slope sg_ = wsm6_slope_graup ( c , qgl [ k ] , denk , denfac ) ;
Sf workr = sr_ . vt ;
const Sf qsum = wmax ( qsl [ k ] + qgl [ k ] , Sf(1.0e-15f) ) ;
v2 [ k ] = ( qsum > Sf(1.0e-15f) ) ? aa_div ( ( ss_ . vt * qsl [ k ] + sg_ . vt * qgl [ k ] ) , qsum ) : Sf(0.0f) ;
rq1 [ k ] = denk * qrl [ k ] ;
rq2 [ k ] = denk * qsl [ k ] ;
rq3 [ k ] = denk * qgl [ k ] ;
if ( Sf(qrl [ k ]) <= Sf(Sf(0.0f)) ) workr = Sf(0.0f) ;
v1 [ k ] = workr ;
}
Sf delqrs1 , delqrs2 , delqrs3 ;
wsm6_nislfv ( nz , c , denc , dzc , t , v1 , rq1 , ( thread Sf * ) nullptr , & delqrs1 , ( thread Sf * ) nullptr , dtcld , 1 , 0 , stride ) ;
wsm6_nislfv ( nz , c , denc , dzc , t , v2 , rq2 , rq3 , & delqrs2 , & delqrs3 , dtcld , 1 , 1 , stride ) ;
for ( int k = 0 ;
k < nz ;
++ k ) {
const Sf denk = denc [ k * stride ] ;
qrl [ k ] = wmax ( aa_div ( rq1 [ k ] , denk ) , Sf(0.0f) ) ;
qsl [ k ] = wmax ( aa_div ( rq2 [ k ] , denk ) , Sf(0.0f) ) ;
qgl [ k ] = wmax ( aa_div ( rq3 [ k ] , denk ) , Sf(0.0f) ) ;
}
const Sf dz0 = dzc [ 0 ] ;
const Sf fall1 = aa_div ( ( aa_div ( delqrs1 , dz0 ) ) , dtcld ) ;
const Sf fall2 = aa_div ( ( aa_div ( delqrs2 , dz0 ) ) , dtcld ) ;
const Sf fall3 = aa_div ( ( aa_div ( delqrs3 , dz0 ) ) , dtcld ) ;
for ( int k = 0 ;
k < nz ;
++ k ) {
const Sf denk = denc [ k * stride ] ;
if ( Sf(qil [ k ]) <= Sf(Sf(0.0f)) ) {
v1 [ k ] = Sf(0.0f) ;
}
else {
const Sf xmi = aa_div ( ( denk * qil [ k ] ) , xni [ k ] ) ;
const Sf diameter = wmax ( wmin ( Sf(11.9f) * aa_sqrtf ( xmi ) , Sf(500.0e-6f) ) , Sf(1.0e-25f) ) ;
v1 [ k ] = Sf(1.49e4f) * gfk_exp ( gfk_log ( diameter ) * Sf(1.31f) ) ;
}
rq1 [ k ] = denk * qil [ k ] ;
}
Sf delqi ;
wsm6_nislfv ( nz , c , denc , dzc , t , v1 , rq1 , ( thread Sf * ) nullptr , & delqi , ( thread Sf * ) nullptr , dtcld , 0 , 2 , stride ) ;
for ( int k = 0 ;
k < nz ;
++ k ) qil [ k ] = wmax ( aa_div ( rq1 [ k ] , denc [ k * stride ] ) , Sf(0.0f) ) ;
const Sf fallc = aa_div ( ( aa_div ( delqi , dz0 ) ) , dtcld ) ;
{
const Sf fallsum = ( ( fall1 + fall2 ) + fall3 ) + fallc ;
const Sf fallsum_qsi = fall2 + fallc ;
const Sf fallsum_qg = fall3 ;
if ( Sf(fallsum) > Sf(Sf(0.0f)) ) {
rainncv_c = ( __fdiv_rn ( fallsum * dz0 , Sf(1000.0f) ) * dtcld ) * Sf(1000.0f) + rainncv_c ;
rain_c = ( __fdiv_rn ( fallsum * dz0 , Sf(1000.0f) ) * dtcld ) * Sf(1000.0f) + rain_c ;
}
if ( Sf(fallsum_qsi) > Sf(Sf(0.0f)) ) {
snowncv_c = ( __fdiv_rn ( fallsum_qsi * dz0 , Sf(1000.0f) ) * dtcld ) * Sf(1000.0f) + snowncv_c ;
snow_c = ( __fdiv_rn ( fallsum_qsi * dz0 , Sf(1000.0f) ) * dtcld ) * Sf(1000.0f) + snow_c ;
}
if ( Sf(fallsum_qg) > Sf(Sf(0.0f)) ) {
graupelncv_c = ( __fdiv_rn ( fallsum_qg * dz0 , Sf(1000.0f) ) * dtcld ) * Sf(1000.0f) + graupelncv_c ;
graupel_c = ( __fdiv_rn ( fallsum_qg * dz0 , Sf(1000.0f) ) * dtcld ) * Sf(1000.0f) + graupel_c ;
}
if ( Sf(fallsum) > Sf(Sf(0.0f)) ) sr_c = aa_div ( ( snowncv_c + graupelncv_c ) , ( rainncv_c + Sf(1.0e-12f) ) ) ;
}
for ( int k = 0 ;
k < nz ;
++ k ) {
const Sf denk = denc [ k * stride ] ;
const Sf pk = pc [ k * stride ] ;
const Sf denfac = wsm6_denfac ( denk ) ;
Sf tk = t [ k ] , qk = q [ k ] ;
Sf qck = qcl [ k ] , qik = qil [ k ] , qrk = qrl [ k ] , qsk = qsl [ k ] , qgk = qgl [ k ] ;
const Sf cpmk = cpm [ k ] , xlk = xl [ k ] ;
const Sf qsat1 = wsm6_qsat ( fp , tk , pk , false ) ;
const Sf rh1 = wmax ( aa_div ( qk , qsat1 ) , Sf(1.0e-15f) ) ;
const Sf qsat2 = wsm6_qsat ( fp , tk , pk , true ) ;
const Sf rh2 = wmax ( aa_div ( qk , qsat2 ) , Sf(1.0e-15f) ) ;
Wsm6Slope sr_ = wsm6_slope_rain ( c , qrk , denk , denfac ) ;
Wsm6Slope ss_ = wsm6_slope_snow ( c , qsk , denk , denfac , tk ) ;
Wsm6Slope sg_ = wsm6_slope_graup ( c , qgk , denk , denfac ) ;
Sf supcol = Sf(273.15f) - tk ;
Sf n0sfac = wmax ( wmin ( gfk_exp ( Sf(0.12f) * supcol ) , Sf(5.0e4f) ) , Sf(1.0f) ) ;
if ( tk > Sf(273.15f) ) {
const Sf xlf = Sf(3.5e5f) ;
const Sf work2 = st_venfac ( pk , tk , denk ) ;
if ( Sf(qsk) > Sf(Sf(0.0f)) ) {
const Sf coeres = ss_ . r2 * aa_sqrtf ( ss_ . r * ss_ . rb ) ;
Sf psmlt = aa_div ( aa_div ( __fdiv_rn ( st_xka ( tk , denk ) , xlf ) * ( Sf(273.15f) - tk ) * pi , Sf(2.0f) ) * n0sfac * ( c [ C_PRECS1 ] * ss_ . r2 + c [ C_PRECS2 ] * work2 * coeres ) , denk ) ;
psmlt = wmin ( wmax ( psmlt * dtcld , - qsk ) , Sf(0.0f) ) ;
qsk = qsk + psmlt ;
qrk = qrk - psmlt ;
tk = tk + ( aa_div ( xlf , cpmk ) ) * psmlt ;
}
if ( Sf(qgk) > Sf(Sf(0.0f)) ) {
const Sf coeres = sg_ . r2 * aa_sqrtf ( sg_ . r * sg_ . rb ) ;
Sf pgmlt = aa_div ( __fdiv_rn ( st_xka ( tk , denk ) , xlf ) * ( Sf(273.15f) - tk ) * ( c [ C_PRECG1 ] * sg_ . r2 + c [ C_PRECG2 ] * work2 * coeres ) , denk ) ;
pgmlt = wmin ( wmax ( pgmlt * dtcld , - qgk ) , Sf(0.0f) ) ;
qgk = qgk + pgmlt ;
qrk = qrk - pgmlt ;
tk = tk + ( aa_div ( xlf , cpmk ) ) * pgmlt ;
}
}
{
const Sf supc = Sf(273.15f) - tk ;
Sf xlf = Sf(2.85e6f) - xlk ;
if ( Sf(supc) < Sf(Sf(0.0f)) ) xlf = Sf(3.5e5f) ;
if ( Sf(supc) < Sf(Sf(0.0f)) && Sf(qik) > Sf(Sf(0.0f)) ) {
qck = qck + qik ;
tk = tk - ( aa_div ( xlf , cpmk ) ) * qik ;
qik = Sf(0.0f) ;
}
if ( supc > Sf(40.0f) && Sf(qck) > Sf(Sf(0.0f)) ) {
qik = qik + qck ;
tk = tk + ( aa_div ( xlf , cpmk ) ) * qck ;
qck = Sf(0.0f) ;
}
if ( Sf(supc) > Sf(Sf(0.0f)) && qck > Sf(1.0e-15f) ) {
const Sf supcolt = wmin ( supc , Sf(50.0f) ) ;
const Sf pfrzdtc = wmin ( ( ( __fdiv_rn ( __fdiv_rn ( ( Sf(100.0f) * ( gfk_exp ( Sf(0.66f) * supcolt ) - Sf(1.0f) ) ) * denk , Sf(1000.0f) ) , Sf(3.0e8f) ) * qck ) * qck ) * dtcld , qck ) ;
qik = qik + pfrzdtc ;
tk = tk + ( aa_div ( xlf , cpmk ) ) * pfrzdtc ;
qck = qck - pfrzdtc ;
}
if ( Sf(supc) > Sf(Sf(0.0f)) && Sf(qrk) > Sf(Sf(0.0f)) ) {
Sf temp = sr_ . r3 ;
temp = ( temp * temp ) * sr_ . r ;
const Sf supcolt = wmin ( supc , Sf(50.0f) ) ;
const Sf pfrzdtr = wmin ( ( ( ( aa_div ( ( ( ( ( Sf(20.0f) * ( pi * pi ) ) * Sf(100.0f) ) * Sf(8.0e6f) ) * Sf(1000.0f) ) , denk ) ) * ( gfk_exp ( Sf(0.66f) * supcolt ) - Sf(1.0f) ) ) * temp ) * dtcld , qrk ) ;
qgk = qgk + pfrzdtr ;
tk = tk + ( aa_div ( xlf , cpmk ) ) * pfrzdtr ;
qrk = qrk - pfrzdtr ;
}
}
sr_ = wsm6_slope_rain ( c , qrk , denk , denfac ) ;
ss_ = wsm6_slope_snow ( c , qsk , denk , denfac , tk ) ;
sg_ = wsm6_slope_graup ( c , qgk , denk , denfac ) ;
const Sf work1_1 = st_diffac ( xlk , pk , tk , denk , qsat1 ) ;
const Sf work1_2 = st_diffac ( Sf(2.85e6f) , pk , tk , denk , qsat2 ) ;
const Sf work2 = st_venfac ( pk , tk , denk ) ;
Sf prevp = Sf(0.f) , psdep = Sf(0.f) , pgdep = Sf(0.f) , praut = Sf(0.f) , psaut = Sf(0.f) ;
Sf pgaut = Sf(0.f) , pracw = Sf(0.f) , praci = Sf(0.f) , piacr = Sf(0.f) , psaci = Sf(0.f) ;
Sf psacw = Sf(0.f) , pracs = Sf(0.f) , psacr = Sf(0.f) , pgacw = Sf(0.f) , paacw = Sf(0.f) ;
Sf pgaci = Sf(0.f) , pgacr = Sf(0.f) , pgacs = Sf(0.f) , pigen = Sf(0.f) , pidep = Sf(0.f) ;
Sf pseml = Sf(0.f) , pgeml = Sf(0.f) , psevp = Sf(0.f) , pgevp = Sf(0.f) ;
{
const Sf supsat = wmax ( qk , Sf(1.0e-15f) ) - qsat1 ;
const Sf satdt = aa_div ( supsat , dtcld ) ;
if ( qck > c [ C_QC0 ] ) {
praut = c [ C_QCK1 ] * gfk_pow ( qck , __fdiv_rn ( Sf(7.0f) , Sf(3.0f) ) ) ;
praut = wmin ( praut , aa_div ( qck , dtcld ) ) ;
}
if ( qrk > Sf(1.0e-9f) && qck > Sf(1.0e-15f) ) {
pracw = wmin ( ( ( ( c [ C_PACRR ] * sr_ . r3 ) * sr_ . rb ) * qck ) * denfac , aa_div ( qck , dtcld ) ) ;
}
if ( Sf(qrk) > Sf(Sf(0.0f)) ) {
const Sf coeres = sr_ . r2 * aa_sqrtf ( sr_ . r * sr_ . rb ) ;
prevp = aa_div ( ( ( rh1 - Sf(1.0f) ) * ( c [ C_PRECR1 ] * sr_ . r2 + ( c [ C_PRECR2 ] * work2 ) * coeres ) ) , work1_1 ) ;
if ( Sf(prevp) < Sf(Sf(0.0f)) ) {
prevp = wmax ( prevp , aa_div ( - qrk , dtcld ) ) ;
prevp = wmax ( prevp , aa_div ( satdt , Sf(2.0f) ) ) ;
}
else {
prevp = wmin ( prevp , aa_div ( satdt , Sf(2.0f) ) ) ;
}
}
}
supcol = Sf(273.15f) - tk ;
n0sfac = wmax ( wmin ( gfk_exp ( Sf(0.12f) * supcol ) , Sf(5.0e4f) ) , Sf(1.0f) ) ;
const Sf supsat = wmax ( qk , Sf(1.0e-15f) ) - qsat2 ;
const Sf satdt = aa_div ( supsat , dtcld ) ;
int ifsat = 0 ;
{
Sf temp = denk * wmax ( qik , Sf(1.0e-15f) ) ;
temp = aa_sqrtf ( aa_sqrtf ( ( temp * temp ) * temp ) ) ;
xni [ k ] = wmin ( wmax ( Sf(5.38e7f) * temp , Sf(1.0e3f) ) , Sf(1.0e6f) ) ;
}
const Sf xnik = xni [ k ] ;
const Sf eacrs = gfk_exp ( Sf(0.07f) * ( - supcol ) ) ;
const Sf xmi = aa_div ( ( denk * qik ) , xnik ) ;
const Sf diameter = wmin ( Sf(11.9f) * aa_sqrtf ( xmi ) , Sf(500.0e-6f) ) ;
const Sf vt2i = Sf(1.49e4f) * gfk_pow ( diameter , Sf(1.31f) ) ;
const Sf vt2r = ( c [ C_PVTR ] * sr_ . rb ) * denfac ;
const Sf vt2s = ( c [ C_PVTS ] * ss_ . rb ) * denfac ;
const Sf vt2g = ( c [ C_PVTG ] * sg_ . rb ) * denfac ;
const Sf qsum = wmax ( qsk + qgk , Sf(1.0e-15f) ) ;
const Sf vt2ave = ( qsum > Sf(1.0e-15f) ) ? aa_div ( ( vt2s * qsk + vt2g * qgk ) , qsum ) : Sf(0.0f) ;
if ( Sf(supcol) > Sf(Sf(0.0f)) && qik > Sf(1.0e-15f) ) {
if ( qrk > Sf(1.0e-9f) ) {
Sf acrfac = ( Sf(2.0f) * sr_ . r3 + ( Sf(2.0f) * diameter ) * sr_ . r2 ) + ( diameter * diameter ) * sr_ . r ;
praci = aa_div ( ( ( ( ( pi * qik ) * Sf(8.0e6f) ) * fabs ( vt2r - vt2i ) ) * acrfac ) , Sf(4.0f) ) ;
Sf m = wmin ( wmax ( Sf(0.0f) , aa_div ( qrk , qik ) ) , Sf(1.0f) ) ;
praci = praci * ( m * m ) ;
praci = wmin ( praci , aa_div ( qik , dtcld ) ) ;
piacr = aa_div ( __fdiv_rn ( ( pi * pi ) * Sf(841.9f) * Sf(8.0e6f) * Sf(1000.0f) * xnik * denfac * c [ C_G6PBR ] * sr_ . r3 * sr_ . r3 * sr_ . rb , Sf(24.0f) ) , denk ) ;
m = wmin ( wmax ( Sf(0.0f) , aa_div ( qik , qrk ) ) , Sf(1.0f) ) ;
piacr = piacr * ( m * m ) ;
piacr = wmin ( piacr , aa_div ( qrk , dtcld ) ) ;
}
if ( qsk > Sf(1.0e-9f) ) {
const Sf acrfac = ( Sf(2.0f) * ss_ . r3 + ( Sf(2.0f) * diameter ) * ss_ . r2 ) + ( diameter * diameter ) * ss_ . r ;
psaci = aa_div ( ( ( ( ( ( ( ( pi * qik ) * eacrs ) * Sf(2.0e6f) ) * n0sfac ) * fabs ( vt2ave - vt2i ) ) * acrfac ) ) , Sf(4.0f) ) ;
psaci = wmin ( psaci , aa_div ( qik , dtcld ) ) ;
}
if ( qgk > Sf(1.0e-9f) ) {
const Sf egi = gfk_exp ( Sf(0.07f) * ( - supcol ) ) ;
const Sf acrfac = ( Sf(2.0f) * sg_ . r3 + ( Sf(2.0f) * diameter ) * sg_ . r2 ) + ( diameter * diameter ) * sg_ . r ;
pgaci = aa_div ( ( ( ( ( ( pi * egi ) * qik ) * c [ C_N0G ] ) * fabs ( vt2ave - vt2i ) ) * acrfac ) , Sf(4.0f) ) ;
pgaci = wmin ( pgaci , aa_div ( qik , dtcld ) ) ;
}
}
if ( qsk > Sf(1.0e-9f) && qck > Sf(1.0e-15f) ) {
const Sf m = wmin ( wmax ( Sf(0.0f) , aa_div ( qsk , qck ) ) , Sf(1.0f) ) ;
psacw = wmin ( ( ( ( ( ( c [ C_PACRC ] * n0sfac ) * ss_ . r3 ) * ss_ . rb ) * ( m * m ) ) * qck ) * denfac , aa_div ( qck , dtcld ) ) ;
}
if ( qgk > Sf(1.0e-9f) && qck > Sf(1.0e-15f) ) {
const Sf m = wmin ( wmax ( Sf(0.0f) , aa_div ( qgk , qck ) ) , Sf(1.0f) ) ;
pgacw = wmin ( ( ( ( ( c [ C_PACRG ] * sg_ . r3 ) * sg_ . rb ) * ( m * m ) ) * qck ) * denfac , aa_div ( qck , dtcld ) ) ;
}
if ( qsum > Sf(1.0e-15f) ) paacw = aa_div ( ( qsk * psacw + qgk * pgacw ) , qsum ) ;
if ( qsk > Sf(1.0e-9f) && qrk > Sf(1.0e-9f) ) {
if ( Sf(supcol) > Sf(Sf(0.0f)) ) {
const Sf acrfac = ( ( ( Sf(5.0f) * ss_ . r3 ) * ss_ . r3 ) * sr_ . r + ( ( Sf(2.0f) * ss_ . r3 ) * ss_ . r2 ) * sr_ . r2 ) + ( ( Sf(0.5f) * ss_ . r2 ) * ss_ . r2 ) * sr_ . r3 ;
pracs = ( ( ( ( ( ( ( pi * pi ) * Sf(8.0e6f) ) * Sf(2.0e6f) ) * n0sfac ) * fabs ( vt2r - vt2ave ) ) * ( aa_div ( Sf(100.0f) , denk ) ) ) * acrfac ) ;
const Sf m = wmin ( wmax ( Sf(0.0f) , aa_div ( qrk , qsk ) ) , Sf(1.0f) ) ;
pracs = pracs * ( m * m ) ;
pracs = wmin ( pracs , aa_div ( qsk , dtcld ) ) ;
}
const Sf acrfac = ( ( ( Sf(5.0f) * sr_ . r3 ) * sr_ . r3 ) * ss_ . r + ( ( Sf(2.0f) * sr_ . r3 ) * sr_ . r2 ) * ss_ . r2 ) + ( ( Sf(0.5f) * sr_ . r2 ) * sr_ . r2 ) * ss_ . r3 ;
psacr = ( ( ( ( ( ( ( pi * pi ) * Sf(8.0e6f) ) * Sf(2.0e6f) ) * n0sfac ) * fabs ( vt2ave - vt2r ) ) * ( aa_div ( Sf(1000.0f) , denk ) ) ) * acrfac ) ;
const Sf m = wmin ( wmax ( Sf(0.0f) , aa_div ( qsk , qrk ) ) , Sf(1.0f) ) ;
psacr = psacr * ( m * m ) ;
psacr = wmin ( psacr , aa_div ( qrk , dtcld ) ) ;
}
if ( qgk > Sf(1.0e-9f) && qrk > Sf(1.0e-9f) ) {
const Sf acrfac = ( ( ( Sf(5.0f) * sr_ . r3 ) * sr_ . r3 ) * sg_ . r + ( ( Sf(2.0f) * sr_ . r3 ) * sr_ . r2 ) * sg_ . r2 ) + ( ( Sf(0.5f) * sr_ . r2 ) * sr_ . r2 ) * sg_ . r3 ;
pgacr = ( ( ( ( ( pi * pi ) * Sf(8.0e6f) ) * c [ C_N0G ] ) * fabs ( vt2ave - vt2r ) ) * ( aa_div ( Sf(1000.0f) , denk ) ) ) * acrfac ;
const Sf m = wmin ( wmax ( Sf(0.0f) , aa_div ( qgk , qrk ) ) , Sf(1.0f) ) ;
pgacr = pgacr * ( m * m ) ;
pgacr = wmin ( pgacr , aa_div ( qrk , dtcld ) ) ;
}
if ( qgk > Sf(1.0e-9f) && qsk > Sf(1.0e-9f) ) pgacs = Sf(0.0f) ;
if ( Sf(supcol) <= Sf(Sf(0.0f)) ) {
const Sf xlf = Sf(3.5e5f) ;
if ( Sf(qsk) > Sf(Sf(0.0f)) ) pseml = wmin ( wmax ( __fdiv_rn ( ( Sf(4190.0f) * supcol ) * ( paacw + psacr ) , xlf ) , aa_div ( - qsk , dtcld ) ) , Sf(0.0f) ) ;
if ( Sf(qgk) > Sf(Sf(0.0f)) ) pgeml = wmin ( wmax ( __fdiv_rn ( ( Sf(4190.0f) * supcol ) * ( paacw + pgacr ) , xlf ) , aa_div ( - qgk , dtcld ) ) , Sf(0.0f) ) ;
}
if ( Sf(supcol) > Sf(Sf(0.0f)) ) {
if ( Sf(qik) > Sf(Sf(0.0f)) && ifsat != 1 ) {
pidep = aa_div ( ( ( ( Sf(4.0f) * diameter ) * xnik ) * ( rh2 - Sf(1.0f) ) ) , work1_2 ) ;
const Sf supice = satdt - prevp ;
if ( Sf(pidep) < Sf(Sf(0.0f)) ) {
pidep = wmax ( wmax ( pidep , aa_div ( satdt , Sf(2.0f) ) ) , supice ) ;
pidep = wmax ( pidep , aa_div ( - qik , dtcld ) ) ;
}
else {
pidep = wmin ( wmin ( pidep , aa_div ( satdt , Sf(2.0f) ) ) , supice ) ;
}
if ( fabs ( prevp + pidep ) >= fabs ( satdt ) ) ifsat = 1 ;
}
if ( Sf(qsk) > Sf(Sf(0.0f)) && ifsat != 1 ) {
const Sf coeres = ss_ . r2 * aa_sqrtf ( ss_ . r * ss_ . rb ) ;
psdep = aa_div ( ( ( ( rh2 - Sf(1.0f) ) * n0sfac ) * ( c [ C_PRECS1 ] * ss_ . r2 + ( c [ C_PRECS2 ] * work2 ) * coeres ) ) , work1_2 ) ;
const Sf supice = ( satdt - prevp ) - pidep ;
if ( Sf(psdep) < Sf(Sf(0.0f)) ) {
psdep = wmax ( psdep , aa_div ( - qsk , dtcld ) ) ;
psdep = wmax ( wmax ( psdep , aa_div ( satdt , Sf(2.0f) ) ) , supice ) ;
}
else {
psdep = wmin ( wmin ( psdep , aa_div ( satdt , Sf(2.0f) ) ) , supice ) ;
}
if ( fabs ( ( prevp + pidep ) + psdep ) >= fabs ( satdt ) ) ifsat = 1 ;
}
if ( Sf(qgk) > Sf(Sf(0.0f)) && ifsat != 1 ) {
const Sf coeres = sg_ . r2 * aa_sqrtf ( sg_ . r * sg_ . rb ) ;
pgdep = aa_div ( ( ( rh2 - Sf(1.0f) ) * ( c [ C_PRECG1 ] * sg_ . r2 + ( c [ C_PRECG2 ] * work2 ) * coeres ) ) , work1_2 ) ;
const Sf supice = ( ( satdt - prevp ) - pidep ) - psdep ;
if ( Sf(pgdep) < Sf(Sf(0.0f)) ) {
pgdep = wmax ( pgdep , aa_div ( - qgk , dtcld ) ) ;
pgdep = wmax ( wmax ( pgdep , aa_div ( satdt , Sf(2.0f) ) ) , supice ) ;
}
else {
pgdep = wmin ( wmin ( pgdep , aa_div ( satdt , Sf(2.0f) ) ) , supice ) ;
}
if ( fabs ( ( ( prevp + pidep ) + psdep ) + pgdep ) >= fabs ( satdt ) ) ifsat = 1 ;
}
if ( Sf(supsat) > Sf(Sf(0.0f)) && ifsat != 1 ) {
const Sf supice = ( ( ( satdt - prevp ) - pidep ) - psdep ) - pgdep ;
const Sf xni0 = Sf(1.0e3f) * gfk_exp ( Sf(0.1f) * supcol ) ;
const Sf roqi0 = Sf(4.92e-11f) * gfk_pow ( xni0 , Sf(1.33f) ) ;
pigen = wmax ( Sf(0.0f) , aa_div ( ( aa_div ( roqi0 , denk ) - wmax ( qik , Sf(0.0f) ) ) , dtcld ) ) ;
pigen = wmin ( wmin ( pigen , satdt ) , supice ) ;
}
if ( Sf(qik) > Sf(Sf(0.0f)) ) {
const Sf qimax = aa_div ( c [ C_ROQIMAX ] , denk ) ;
psaut = wmax ( Sf(0.0f) , aa_div ( ( qik - qimax ) , dtcld ) ) ;
}
if ( Sf(qsk) > Sf(Sf(0.0f)) ) {
const Sf alpha2 = Sf(1.0e-3f) * gfk_exp ( Sf(0.09f) * ( - supcol ) ) ;
pgaut = wmin ( wmax ( Sf(0.0f) , alpha2 * ( qsk - Sf(6.0e-4f) ) ) , aa_div ( qsk , dtcld ) ) ;
}
}
if ( Sf(supcol) < Sf(Sf(0.0f)) ) {
if ( Sf(qsk) > Sf(Sf(0.0f)) && rh1 < Sf(1.0f) ) {
const Sf coeres = ss_ . r2 * aa_sqrtf ( ss_ . r * ss_ . rb ) ;
psevp = aa_div ( ( ( ( rh1 - Sf(1.0f) ) * n0sfac ) * ( c [ C_PRECS1 ] * ss_ . r2 + ( c [ C_PRECS2 ] * work2 ) * coeres ) ) , work1_1 ) ;
psevp = wmin ( wmax ( psevp , aa_div ( - qsk , dtcld ) ) , Sf(0.0f) ) ;
}
if ( Sf(qgk) > Sf(Sf(0.0f)) && rh1 < Sf(1.0f) ) {
const Sf coeres = sg_ . r2 * aa_sqrtf ( sg_ . r * sg_ . rb ) ;
pgevp = aa_div ( ( ( rh1 - Sf(1.0f) ) * ( c [ C_PRECG1 ] * sg_ . r2 + ( c [ C_PRECG2 ] * work2 ) * coeres ) ) , work1_1 ) ;
pgevp = wmin ( wmax ( pgevp , aa_div ( - qgk , dtcld ) ) , Sf(0.0f) ) ;
}
}
{
Sf delta2 = Sf(0.0f) , delta3 = Sf(0.0f) ;
if ( qrk < Sf(1.0e-4f) && qsk < Sf(1.0e-4f) ) delta2 = Sf(1.0f) ;
if ( qrk < Sf(1.0e-4f) ) delta3 = Sf(1.0f) ;
Sf value , source , factor ;
if ( tk <= Sf(273.15f) ) {
value = wmax ( Sf(1.0e-15f) , qck ) ;
source = ( praut + pracw + paacw + paacw ) * dtcld ;
if ( source > value ) {
factor = aa_div ( value , source ) ;
praut = praut * factor ;
pracw = pracw * factor ;
paacw = paacw * factor ;
}
value = wmax ( Sf(1.0e-15f) , qik ) ;
source = ( psaut - pigen - pidep + praci + psaci + pgaci ) * dtcld ;
if ( source > value ) {
factor = aa_div ( value , source ) ;
psaut = psaut * factor ;
pigen = pigen * factor ;
pidep = pidep * factor ;
praci = praci * factor ;
psaci = psaci * factor ;
pgaci = pgaci * factor ;
}
value = wmax ( Sf(1.0e-15f) , qrk ) ;
source = ( - praut - prevp - pracw + piacr + psacr + pgacr ) * dtcld ;
if ( source > value ) {
factor = aa_div ( value , source ) ;
praut = praut * factor ;
prevp = prevp * factor ;
pracw = pracw * factor ;
piacr = piacr * factor ;
psacr = psacr * factor ;
pgacr = pgacr * factor ;
}
value = wmax ( Sf(1.0e-15f) , qsk ) ;
source = - ( ( psdep + psaut - pgaut + paacw + piacr * delta3 + praci * delta3 - pracs * ( Sf(1.0f) - delta2 ) + psacr * delta2 + psaci - pgacs ) * dtcld ) ;
if ( source > value ) {
factor = aa_div ( value , source ) ;
psdep = psdep * factor ;
psaut = psaut * factor ;
pgaut = pgaut * factor ;
paacw = paacw * factor ;
piacr = piacr * factor ;
praci = praci * factor ;
psaci = psaci * factor ;
pracs = pracs * factor ;
psacr = psacr * factor ;
pgacs = pgacs * factor ;
}
value = wmax ( Sf(1.0e-15f) , qgk ) ;
source = - ( ( pgdep + pgaut + piacr * ( Sf(1.0f) - delta3 ) + praci * ( Sf(1.0f) - delta3 ) + psacr * ( Sf(1.0f) - delta2 ) + pracs * ( Sf(1.0f) - delta2 ) + pgaci + paacw + pgacr + pgacs ) * dtcld ) ;
if ( source > value ) {
factor = aa_div ( value , source ) ;
pgdep = pgdep * factor ;
pgaut = pgaut * factor ;
piacr = piacr * factor ;
praci = praci * factor ;
psacr = psacr * factor ;
pracs = pracs * factor ;
paacw = paacw * factor ;
pgaci = pgaci * factor ;
pgacr = pgacr * factor ;
pgacs = pgacs * factor ;
}
const Sf work2u = - ( prevp + psdep + pgdep + pigen + pidep ) ;
qk = qk + work2u * dtcld ;
qck = wmax ( qck - ( praut + pracw + paacw + paacw ) * dtcld , Sf(0.0f) ) ;
qrk = wmax ( qrk + ( praut + pracw + prevp - piacr - pgacr - psacr ) * dtcld , Sf(0.0f) ) ;
qik = wmax ( qik - ( psaut + praci + psaci + pgaci - pigen - pidep ) * dtcld , Sf(0.0f) ) ;
qsk = wmax ( qsk + ( psdep + psaut + paacw - pgaut + piacr * delta3 + praci * delta3 + psaci - pgacs - pracs * ( Sf(1.0f) - delta2 ) + psacr * delta2 ) * dtcld , Sf(0.0f) ) ;
qgk = wmax ( qgk + ( pgdep + pgaut + piacr * ( Sf(1.0f) - delta3 ) + praci * ( Sf(1.0f) - delta3 ) + psacr * ( Sf(1.0f) - delta2 ) + pracs * ( Sf(1.0f) - delta2 ) + pgaci + paacw + pgacr + pgacs ) * dtcld , Sf(0.0f) ) ;
const Sf xlf = Sf(2.85e6f) - xlk ;
const Sf xlwork2 = - ( Sf(2.85e6f) * ( psdep + pgdep + pidep + pigen ) ) - xlk * prevp - xlf * ( piacr + paacw + paacw + pgacr + psacr ) ;
tk = tk - ( aa_div ( xlwork2 , cpmk ) ) * dtcld ;
}
else {
value = wmax ( Sf(1.0e-15f) , qck ) ;
source = ( praut + pracw + paacw + paacw ) * dtcld ;
if ( source > value ) {
factor = aa_div ( value , source ) ;
praut = praut * factor ;
pracw = pracw * factor ;
paacw = paacw * factor ;
}
value = wmax ( Sf(1.0e-15f) , qrk ) ;
source = ( - paacw - praut + pseml + pgeml - pracw - paacw - prevp ) * dtcld ;
if ( source > value ) {
factor = aa_div ( value , source ) ;
praut = praut * factor ;
prevp = prevp * factor ;
pracw = pracw * factor ;
paacw = paacw * factor ;
pseml = pseml * factor ;
pgeml = pgeml * factor ;
}
value = wmax ( Sf(1.0e-9f) , qsk ) ;
source = ( pgacs - pseml - psevp ) * dtcld ;
if ( source > value ) {
factor = aa_div ( value , source ) ;
pgacs = pgacs * factor ;
psevp = psevp * factor ;
pseml = pseml * factor ;
}
value = wmax ( Sf(1.0e-9f) , qgk ) ;
source = - ( ( pgacs + pgevp + pgeml ) * dtcld ) ;
if ( source > value ) {
factor = aa_div ( value , source ) ;
pgacs = pgacs * factor ;
pgevp = pgevp * factor ;
pgeml = pgeml * factor ;
}
const Sf work2u = - ( prevp + psevp + pgevp ) ;
qk = qk + work2u * dtcld ;
qck = wmax ( qck - ( praut + pracw + paacw + paacw ) * dtcld , Sf(0.0f) ) ;
qrk = wmax ( qrk + ( praut + pracw + prevp + paacw + paacw - pseml - pgeml ) * dtcld , Sf(0.0f) ) ;
qsk = wmax ( qsk + ( psevp - pgacs + pseml ) * dtcld , Sf(0.0f) ) ;
qgk = wmax ( qgk + ( pgacs + pgevp + pgeml ) * dtcld , Sf(0.0f) ) ;
const Sf xlf = Sf(2.85e6f) - xlk ;
const Sf xlwork2 = - ( xlk * ( prevp + psevp + pgevp ) ) - xlf * ( pseml + pgeml ) ;
tk = tk - ( aa_div ( xlwork2 , cpmk ) ) * dtcld ;
}
}
{
const Sf qsatn = wsm6_qsat ( fp , tk , pk , false ) ;
const Sf work1 = st_conden ( tk , qk , qsatn , xlk , cpmk ) ;
Sf pcond = wmin ( wmax ( aa_div ( work1 , dtcld ) , Sf(0.0f) ) , aa_div ( wmax ( qk , Sf(0.0f) ) , dtcld ) ) ;
if ( Sf(qck) > Sf(Sf(0.0f)) && Sf(work1) < Sf(Sf(0.0f)) ) pcond = aa_div ( wmax ( work1 , - qck ) , dtcld ) ;
qk = qk - pcond * dtcld ;
qck = wmax ( qck + pcond * dtcld , Sf(0.0f) ) ;
tk = tk + ( ( aa_div ( ( pcond * xlk ) , cpmk ) ) * dtcld ) ;
}
if ( qck <= Sf(1.0e-15f) ) qck = Sf(0.0f) ;
if ( qik <= Sf(1.0e-15f) ) qik = Sf(0.0f) ;
t [ k ] = tk ;
q [ k ] = qk ;
qcl [ k ] = qck ;
qil [ k ] = qik ;
qrl [ k ] = qrk ;
qsl [ k ] = qsk ;
qgl [ k ] = qgk ;
}
}
rainnc [ col ] = rain_c ;
rainncv [ col ] = rainncv_c ;
snownc [ col ] = snow_c ;
snowncv [ col ] = snowncv_c ;
graupelnc [ col ] = graupel_c ;
graupelncv [ col ] = graupelncv_c ;
sr [ col ] = sr_c ;
for ( int k = 0 ;
k < nz ;
++ k ) {
const int id = col + k * stride ;
const Sf th = aa_div ( t [ k ] , pii [ id ] ) ;
theta [ id ] = th ;
qv [ id ] = q [ k ] ;
qc [ id ] = qcl [ k ] ;
qi [ id ] = qil [ k ] ;
qr [ id ] = qrl [ k ] ;
qs [ id ] = qsl [ k ] ;
qg [ id ] = qgl [ k ] ;
const Sf tt = th * pii [ id ] ;
const Sf rho = denc [ k * stride ] ;
Sf re_qc = Sf(2.49e-6f) , re_qi = Sf(4.99e-6f) , re_qs = Sf(9.99e-6f) ;
const Sf rqc = wmax ( Sf(1.0e-12f) , qcl [ k ] * rho ) ;
const Sf rqi = wmax ( Sf(1.0e-12f) , qil [ k ] * rho ) ;
Sf temp = rho * wmax ( qil [ k ] , Sf(1.0e-15f) ) ;
temp = aa_sqrtf ( aa_sqrtf ( ( temp * temp ) * temp ) ) ;
const Sf ni = wmin ( wmax ( Sf(5.38e7f) * temp , Sf(1.0e3f) ) , Sf(1.0e6f) ) ;
const Sf rni = wmax ( Sf(1.0e-6f) , ni * rho ) ;
const Sf rqs = wmax ( Sf(1.0e-12f) , qsl [ k ] * rho ) ;
if ( rqc > Sf(1.0e-12f) ) {
const Sf lamdac = gfk_pow ( aa_div ( ( c [ C_PIDNC ] * Sf(3.0e8f) ) , rqc ) , __fdiv_rn ( Sf(1.0f) , Sf(3.0f) ) ) ;
re_qc = wmax ( Sf(2.51e-6f) , wmin ( Sf(1.5f) * ( aa_div ( Sf(1.0f) , lamdac ) ) , Sf(50.0e-6f) ) ) ;
}
if ( rqi > Sf(1.0e-12f) && rni > Sf(1.0e-6f) ) {
const Sf diai = Sf(11.9f) * aa_sqrtf ( aa_div ( rqi , ni ) ) ;
re_qi = wmax ( Sf(10.01e-6f) , wmin ( ( Sf(0.75f) * Sf(0.163f) ) * diai , Sf(125.0e-6f) ) ) ;
}
if ( rqs > Sf(1.0e-12f) ) {
const Sf supc = Sf(273.15f) - tt ;
const Sf nsf = wmax ( wmin ( gfk_exp ( Sf(0.12f) * supc ) , Sf(5.0e4f) ) , Sf(1.0f) ) ;
const Sf lamdas = aa_sqrtf ( aa_sqrtf ( aa_div ( ( c [ C_PIDN0S ] * nsf ) , rqs ) ) ) ;
re_qs = wmax ( Sf(25.0e-6f) , wmin ( Sf(0.5f) * ( aa_div ( Sf(1.0f) , lamdas ) ) , Sf(999.0e-6f) ) ) ;
}
re_qc = wmax ( Sf(2.49e-6f) , wmin ( re_qc , Sf(50.0e-6f) ) ) ;
re_qi = wmax ( Sf(4.99e-6f) , wmin ( re_qi , Sf(125.0e-6f) ) ) ;
re_qs = wmax ( Sf(9.99e-6f) , wmin ( re_qs , Sf(999.0e-6f) ) ) ;
effc [ id ] = re_qc * Sf(1.0e6f) ;
effi [ id ] = re_qi * Sf(1.0e6f) ;
effs [ id ] = re_qs * Sf(1.0e6f) ;
}
}

}
kernel void strict_regional_wsm6(
    constant Wsm6ColumnResources &wsm_r [[buffer(0)]],
    constant Wsm6ColumnParams &wsm_p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
 using namespace t64;
    device float *theta = wsm_r.theta;
    device float *qv = wsm_r.qv;
    device float *qc = wsm_r.qc;
    device float *qi = wsm_r.qi;
    device float *qr = wsm_r.qr;
    device float *qs = wsm_r.qs;
    device float *qg = wsm_r.qg;
    device const float *den = wsm_r.den;
    device const float *p = wsm_r.p;
    device const float *pii = wsm_r.pii;
    device const float *dz = wsm_r.dz;
    device float *rainnc = wsm_r.rainnc;
    device float *rainncv = wsm_r.rainncv;
    device float *snownc = wsm_r.snownc;
    device float *snowncv = wsm_r.snowncv;
    device float *graupelnc = wsm_r.graupelnc;
    device float *graupelncv = wsm_r.graupelncv;
    device float *sr = wsm_r.sr;
    device float *effc = wsm_r.effc;
    device float *effi = wsm_r.effi;
    device float *effs = wsm_r.effs;
    device const float *coef = wsm_r.coef;
    const float delt = wsm_p.delt;
    const int nz = wsm_p.nz;
    const int ny = wsm_p.ny;
    const int nx = wsm_p.nx;
{
int col = int ( gid ) ;
int stride = ny * nx ;
if ( col >= stride || nz > 64 ) return ;
wsm6_column_impl ( theta , qv , qc , qi , qr , qs , qg , den , p , pii , dz , rainnc , rainncv , snownc , snowncv , graupelnc , graupelncv , sr , effc , effi , effs , coef , delt , nz , stride , col ) ;
}

}
