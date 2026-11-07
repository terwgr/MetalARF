#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "thompson_column_helpers.metalh"
using namespace ntc;
struct Wsm6State {
F dtcldcr ;
F n0r ;
F avtr ;
F bvtr ;
F r0 ;
F peaut ;
F xncr ;
F xmyu ;
F avts ;
F bvts ;
F lamdarmax ;
F lamdasmax ;
F dicon ;
F dimax ;
F pfrz1 ;
F pfrz2 ;
F qcrmin ;
F eacrc ;
F dens ;
F qs0 ;
F n0smax ;
F n0s ;
F alpha ;
F qc0 ;
F qck1 ;
F bvtr1 ;
F bvtr2 ;
F bvtr3 ;
F bvtr4 ;
F g1pbr ;
F g3pbr ;
F g4pbr ;
F g5pbro2 ;
F pvtr ;
F eacrr ;
F pacrr ;
F bvtr6 ;
F g6pbr ;
F precr1 ;
F precr2 ;
F roqimax ;
F bvts1 ;
F bvts2 ;
F bvts3 ;
F bvts4 ;
F g1pbs ;
F g3pbs ;
F g4pbs ;
F n0g ;
F avtg ;
F bvtg ;
F deng ;
F lamdagmax ;
F g5pbso2 ;
F pvts ;
F pacrs ;
F precs1 ;
F precs2 ;
F pidn0r ;
F xlv1 ;
F pacrc ;
F pi ;
F bvtg1 ;
F bvtg2 ;
F bvtg3 ;
F bvtg4 ;
F g1pbg ;
F g3pbg ;
F g4pbg ;
F g5pbgo2 ;
F pvtg ;
F pacrg ;
F precg1 ;
F precg2 ;
F pidn0g ;
F rslopermax ;
F rslopesmax ;
F rslopegmax ;
F rsloperbmax ;
F rslopesbmax ;
F rslopegbmax ;
F rsloper2max ;
F rslopes2max ;
F rslopeg2max ;
F rsloper3max ;
F rslopes3max ;
F rslopeg3max ;
F pidn0s ;
F pidnc ;
int nrbins ;
int slen ;
tc44::cd m_w_0 ;
tc44::cd m_i_0 ;
Soft8 xxdx [ ( 50 + 1 ) ] ;
Soft8 xxds [ ( 50 ) ] ;
Soft8 xdts [ ( 50 ) ] ;
Soft8 xxdg [ ( 50 ) ] ;
Soft8 xdtg [ ( 50 ) ] ;
Soft8 lamda_radar ;
Soft8 k_w ;
Soft8 pi5 ;
Soft8 lamda4 ;
Soft8 simpson [ ( 50 + 1 ) ] ;
Soft8 basis [ ( 3 ) ] ;
F xcre [ ( 4 ) ] ;
F xcse [ ( 4 ) ] ;
F xcge [ ( 4 ) ] ;
F xcrg [ ( 4 ) ] ;
F xcsg [ ( 4 ) ] ;
F xcgg [ ( 4 ) ] ;
F xam_r ;
F xbm_r ;
F xmu_r ;
F xobmr ;
F xam_s ;
F xbm_s ;
F xmu_s ;
F xoams ;
F xobms ;
F xocms ;
F xam_g ;
F xbm_g ;
F xmu_g ;
F xoamg ;
F xobmg ;
F xocmg ;
F xorg2 ;
F xosg2 ;
F xogg2 ;
Soft8 melt_outside_s ;
Soft8 melt_outside_g ;
}
;

static inline F gammln ( thread Wsm6State & S , F xx ) {
int j ;
Soft8 stp = SD8 ( 0x40040d931ff62706ul ) ;
Soft8 cof [ ( 6 - 1 + 1 ) ] = {
SD8 ( 0x40530b869f76170aul ) , ( - SD8 ( 0x4055a0572b143656ul ) ) , SD8 ( 0x4038039bf13b4cc6ul ) , ( - SD8 ( 0x3ff3b5348dcdcc8cul ) ) , SD8 ( 0x3f53cd7319ec12d8ul ) , ( - SD8 ( 0x3ed6a1181c0e9888ul ) ) }
;
Soft8 ser ;
Soft8 tmp ;
Soft8 x ;
Soft8 y ;
F result ;
x = xx ;
y = x ;
tmp = tc_add( x , SD8 ( 0x4016000000000000ul ) ) ;
tmp = tc_sub( tc_mul( tc_add( x , SD8 ( 0x3fe0000000000000ul ) ) , tc_log( tmp ) ) , tmp ) ;
ser = SD8 ( 0x3ff00000000d0ec7ul ) ;
for ( j = 1 ;
j <= 6 ;
j += 1 ) {
y = tc_add( y , SD8 ( 0x3ff0000000000000ul ) ) ;
ser = tc_add( ser , tc_div( cof [ ( ( j ) - ( 1 ) ) * ( 1 ) ] , y ) ) ;
}
result = tc_add( tmp , tc_log( tc_div( tc_mul( stp , ser ) , x ) ) ) ;
return result ;
}

static inline F wgamma ( thread Wsm6State & S , F y ) {
F result ;
result = tc_exp( gammln ( S , y ) ) ;
return result ;
}

static inline void state_parameters ( thread Wsm6State & S ) {
S . nrbins = 50 ;
S . slen = 20 ;
S . lamda_radar = 0.10f ;
S . basis [ 0 ] = tc_div( SD8 ( 0x3ff0000000000000ul ) , SD8 ( 0x4008000000000000ul ) ) ;
S . basis [ 1 ] = tc_div( SD8 ( 0x4010000000000000ul ) , SD8 ( 0x4008000000000000ul ) ) ;
S . basis [ 2 ] = tc_div( SD8 ( 0x3ff0000000000000ul ) , SD8 ( 0x4008000000000000ul ) ) ;
S . melt_outside_s = SD8 ( 0x3feccccccccccccdul ) ;
S . melt_outside_g = SD8 ( 0x3feccccccccccccdul ) ;
}

static inline void radar_init ( thread Wsm6State & S ) {
int n ;
S . pi5 = tc_mul( tc_mul( tc_mul( tc_mul( 3.14159f , 3.14159f ) , 3.14159f ) , 3.14159f ) , 3.14159f ) ;
S . lamda4 = tc_mul( tc_mul( tc_mul( S . lamda_radar , S . lamda_radar ) , S . lamda_radar ) , S . lamda_radar ) ;
S . m_w_0 = tc44::cd_make( SD8 ( 0x402213d1a0000000ul ) , - SD8 ( 0x3ff634d6c0000000ul ) ) ;
S . m_i_0 = tc44::cd_make( SD8 ( 0x3ffc91dae74d3b24ul ) , - SD8 ( 0x3f21acb622ae704dul ) ) ;
S . k_w = SD8 ( 0x3fede4b5a79d9de5ul ) ;
for ( n = 1 ;
n <= ( S . nrbins + 1 ) ;
n += 1 ) {
S . simpson [ ( ( n ) - ( 1 ) ) * ( 1 ) ] = SD8 ( 0x0000000000000000ul ) ;
}
for ( n = 1 ;
n <= ( S . nrbins - 1 ) ;
n += 2 ) {
S . simpson [ ( ( n ) - ( 1 ) ) * ( 1 ) ] = tc_add( S . simpson [ ( ( n ) - ( 1 ) ) * ( 1 ) ] , S . basis [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] ) ;
S . simpson [ ( ( ( n + 1 ) ) - ( 1 ) ) * ( 1 ) ] = tc_add( S . simpson [ ( ( ( n + 1 ) ) - ( 1 ) ) * ( 1 ) ] , S . basis [ ( ( 2 ) - ( 1 ) ) * ( 1 ) ] ) ;
S . simpson [ ( ( ( n + 2 ) ) - ( 1 ) ) * ( 1 ) ] = tc_add( S . simpson [ ( ( ( n + 2 ) ) - ( 1 ) ) * ( 1 ) ] , S . basis [ ( ( 3 ) - ( 1 ) ) * ( 1 ) ] ) ;
}
for ( n = 1 ;
n <= S . slen ;
n += 1 ) {
}
S . xxdx [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] = SD8 ( 0x3f1a36e2eb1c432dul ) ;
S . xxdx [ ( ( ( S . nrbins + 1 ) ) - ( 1 ) ) * ( 1 ) ] = SD8 ( 0x3f947ae147ae147bul ) ;
for ( n = 2 ;
n <= S . nrbins ;
n += 1 ) {
S . xxdx [ ( ( n ) - ( 1 ) ) * ( 1 ) ] = tc_exp( tc_add( tc_mul( tc_div( ( Soft8 ) ( ( n - 1 ) ) , ( Soft8 ) ( S . nrbins ) ) , tc_log( tc_div( S . xxdx [ ( ( ( S . nrbins + 1 ) ) - ( 1 ) ) * ( 1 ) ] , S . xxdx [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] ) ) ) , tc_log( S . xxdx [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] ) ) ) ;
}
for ( n = 1 ;
n <= S . nrbins ;
n += 1 ) {
S . xxds [ ( ( n ) - ( 1 ) ) * ( 1 ) ] = tc_sqrt( tc_mul( S . xxdx [ ( ( n ) - ( 1 ) ) * ( 1 ) ] , S . xxdx [ ( ( ( n + 1 ) ) - ( 1 ) ) * ( 1 ) ] ) ) ;
S . xdts [ ( ( n ) - ( 1 ) ) * ( 1 ) ] = tc_sub( S . xxdx [ ( ( ( n + 1 ) ) - ( 1 ) ) * ( 1 ) ] , S . xxdx [ ( ( n ) - ( 1 ) ) * ( 1 ) ] ) ;
}
S . xxdx [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] = SD8 ( 0x3f1a36e2eb1c432dul ) ;
S . xxdx [ ( ( ( S . nrbins + 1 ) ) - ( 1 ) ) * ( 1 ) ] = SD8 ( 0x3fa999999999999aul ) ;
for ( n = 2 ;
n <= S . nrbins ;
n += 1 ) {
S . xxdx [ ( ( n ) - ( 1 ) ) * ( 1 ) ] = tc_exp( tc_add( tc_mul( tc_div( ( Soft8 ) ( ( n - 1 ) ) , ( Soft8 ) ( S . nrbins ) ) , tc_log( tc_div( S . xxdx [ ( ( ( S . nrbins + 1 ) ) - ( 1 ) ) * ( 1 ) ] , S . xxdx [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] ) ) ) , tc_log( S . xxdx [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] ) ) ) ;
}
for ( n = 1 ;
n <= S . nrbins ;
n += 1 ) {
S . xxdg [ ( ( n ) - ( 1 ) ) * ( 1 ) ] = tc_sqrt( tc_mul( S . xxdx [ ( ( n ) - ( 1 ) ) * ( 1 ) ] , S . xxdx [ ( ( ( n + 1 ) ) - ( 1 ) ) * ( 1 ) ] ) ) ;
S . xdtg [ ( ( n ) - ( 1 ) ) * ( 1 ) ] = tc_sub( S . xxdx [ ( ( ( n + 1 ) ) - ( 1 ) ) * ( 1 ) ] , S . xxdx [ ( ( n ) - ( 1 ) ) * ( 1 ) ] ) ;
}
S . xcre [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] = tc_add( 1.f , S . xbm_r ) ;
S . xcre [ ( ( 2 ) - ( 1 ) ) * ( 1 ) ] = tc_add( 1.f , S . xmu_r ) ;
S . xcre [ ( ( 3 ) - ( 1 ) ) * ( 1 ) ] = tc_add( 4.f , S . xmu_r ) ;
S . xcre [ ( ( 4 ) - ( 1 ) ) * ( 1 ) ] = tc_add( 7.f , S . xmu_r ) ;
for ( n = 1 ;
n <= 4 ;
n += 1 ) {
S . xcrg [ ( ( n ) - ( 1 ) ) * ( 1 ) ] = wgamma ( S , S . xcre [ ( ( n ) - ( 1 ) ) * ( 1 ) ] ) ;
}
S . xorg2 = tc_div( 1.f , S . xcrg [ ( ( 2 ) - ( 1 ) ) * ( 1 ) ] ) ;
S . xcse [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] = tc_add( 1.f , S . xbm_s ) ;
S . xcse [ ( ( 2 ) - ( 1 ) ) * ( 1 ) ] = tc_add( 1.f , S . xmu_s ) ;
S . xcse [ ( ( 3 ) - ( 1 ) ) * ( 1 ) ] = tc_add( 4.f , S . xmu_s ) ;
S . xcse [ ( ( 4 ) - ( 1 ) ) * ( 1 ) ] = tc_add( 7.f , S . xmu_s ) ;
for ( n = 1 ;
n <= 4 ;
n += 1 ) {
S . xcsg [ ( ( n ) - ( 1 ) ) * ( 1 ) ] = wgamma ( S , S . xcse [ ( ( n ) - ( 1 ) ) * ( 1 ) ] ) ;
}
S . xosg2 = tc_div( 1.f , S . xcsg [ ( ( 2 ) - ( 1 ) ) * ( 1 ) ] ) ;
S . xcge [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] = tc_add( 1.f , S . xbm_g ) ;
S . xcge [ ( ( 2 ) - ( 1 ) ) * ( 1 ) ] = tc_add( 1.f , S . xmu_g ) ;
S . xcge [ ( ( 3 ) - ( 1 ) ) * ( 1 ) ] = tc_add( 4.f , S . xmu_g ) ;
S . xcge [ ( ( 4 ) - ( 1 ) ) * ( 1 ) ] = tc_add( 7.f , S . xmu_g ) ;
for ( n = 1 ;
n <= 4 ;
n += 1 ) {
S . xcgg [ ( ( n ) - ( 1 ) ) * ( 1 ) ] = wgamma ( S , S . xcge [ ( ( n ) - ( 1 ) ) * ( 1 ) ] ) ;
}
S . xogg2 = tc_div( 1.f , S . xcgg [ ( ( 2 ) - ( 1 ) ) * ( 1 ) ] ) ;
S . xobmr = tc_div( 1.f , S . xbm_r ) ;
S . xoams = tc_div( 1.f , S . xam_s ) ;
S . xobms = tc_div( 1.f , S . xbm_s ) ;
S . xocms = tc_pow( S . xoams , S . xobms ) ;
S . xoamg = tc_div( 1.f , S . xam_g ) ;
S . xobmg = tc_div( 1.f , S . xbm_g ) ;
S . xocmg = tc_pow( S . xoamg , S . xobmg ) ;
}

template < typename P_qv1d , typename P_qr1d , typename P_qs1d , typename P_qg1d , typename P_t1d , typename P_p1d , typename P_dbz >
static inline  void refl10cm_wsm6 ( thread Wsm6State & S , P_qv1d qv1d , P_qr1d qr1d , P_qs1d qs1d , P_qg1d qg1d , P_t1d t1d , P_p1d p1d , P_dbz dbz , int kts , int kte, int nz, device F *wf, device Soft8 *wd, device bool *wb ) {
bool melti ;
device bool *l_qr=wb; wb+=nz+3;
device bool *l_qs=wb; wb+=nz+3;
device bool *l_qg=wb; wb+=nz+3;
int i ;
int k ;
int k_0 ;
int kbot ;
int n ;
F r = 287.f ;
F temp_c ;
device F *temp=wf; wf+=nz+3;
device F *pres=wf; wf+=nz+3;
device F *qv=wf; wf+=nz+3;
device F *rho=wf; wf+=nz+3;
device F *rr=wf; wf+=nz+3;
device F *rs=wf; wf+=nz+3;
device F *rg=wf; wf+=nz+3;
device F *ze_rain=wf; wf+=nz+3;
device F *ze_snow=wf; wf+=nz+3;
device F *ze_graupel=wf; wf+=nz+3;
Soft8 fmelt_s ;
Soft8 fmelt_g ;
Soft8 cback ;
Soft8 x ;
Soft8 eta ;
Soft8 f_d ;
device Soft8 *ilamr=wd; wd+=nz+3;
device Soft8 *ilams=wd; wd+=nz+3;
device Soft8 *ilamg=wd; wd+=nz+3;
device Soft8 *n0_r=wd; wd+=nz+3;
device Soft8 *n0_s=wd; wd+=nz+3;
device Soft8 *n0_g=wd; wd+=nz+3;
Soft8 lamr ;
Soft8 lams ;
Soft8 lamg ;
for ( k = kts ;
k <= kte ;
k += 1 ) {
dbz [ ( ( k ) - ( kts ) ) * ( 1 ) ] = ( - 35.0f ) ;
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
temp [ ( ( k ) - ( kts ) ) * ( 1 ) ] = t1d [ ( ( k ) - ( kts ) ) * ( 1 ) ] ;
temp_c = tc_min( tc_sub( temp [ ( ( k ) - ( kts ) ) * ( 1 ) ] , 273.15f ) , ( - 0.001f ) ) ;
qv [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_max( qv1d [ ( ( k ) - ( kts ) ) * ( 1 ) ] , 1.e-10f ) ;
pres [ ( ( k ) - ( kts ) ) * ( 1 ) ] = p1d [ ( ( k ) - ( kts ) ) * ( 1 ) ] ;
rho [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_div( tc_mul( 0.622f , pres [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) , tc_mul( tc_mul( r , temp [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) , tc_add( qv [ ( ( k ) - ( kts ) ) * ( 1 ) ] , 0.622f ) ) ) ;
if ( ( qr1d [ ( ( k ) - ( kts ) ) * ( 1 ) ] > 1.e-9f ) ) {
rr [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_mul( qr1d [ ( ( k ) - ( kts ) ) * ( 1 ) ] , rho [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) ;
n0_r [ ( ( k ) - ( kts ) ) * ( 1 ) ] = S . n0r ;
lamr = tc_pow( tc_div( tc_mul( tc_mul( S . xam_r , S . xcrg [ ( ( 3 ) - ( 1 ) ) * ( 1 ) ] ) , n0_r [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) , rr [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) , tc_div( 1.f , S . xcre [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] ) ) ;
ilamr [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_div( 1.f , lamr ) ;
l_qr [ ( ( k ) - ( kts ) ) * ( 1 ) ] = true ;
}
else {
rr [ ( ( k ) - ( kts ) ) * ( 1 ) ] = 1.e-12f ;
l_qr [ ( ( k ) - ( kts ) ) * ( 1 ) ] = false ;
}
if ( ( qs1d [ ( ( k ) - ( kts ) ) * ( 1 ) ] > 1.e-9f ) ) {
rs [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_mul( qs1d [ ( ( k ) - ( kts ) ) * ( 1 ) ] , rho [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) ;
n0_s [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_min( S . n0smax , tc_mul( S . n0s , tc_exp( tc_mul( ( - S . alpha ) , temp_c ) ) ) ) ;
lams = tc_pow( tc_div( tc_mul( tc_mul( S . xam_s , S . xcsg [ ( ( 3 ) - ( 1 ) ) * ( 1 ) ] ) , n0_s [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) , rs [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) , tc_div( 1.f , S . xcse [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] ) ) ;
ilams [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_div( 1.f , lams ) ;
l_qs [ ( ( k ) - ( kts ) ) * ( 1 ) ] = true ;
}
else {
rs [ ( ( k ) - ( kts ) ) * ( 1 ) ] = 1.e-12f ;
l_qs [ ( ( k ) - ( kts ) ) * ( 1 ) ] = false ;
}
if ( ( qg1d [ ( ( k ) - ( kts ) ) * ( 1 ) ] > 1.e-9f ) ) {
rg [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_mul( qg1d [ ( ( k ) - ( kts ) ) * ( 1 ) ] , rho [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) ;
n0_g [ ( ( k ) - ( kts ) ) * ( 1 ) ] = S . n0g ;
lamg = tc_pow( tc_div( tc_mul( tc_mul( S . xam_g , S . xcgg [ ( ( 3 ) - ( 1 ) ) * ( 1 ) ] ) , n0_g [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) , rg [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) , tc_div( 1.f , S . xcge [ ( ( 1 ) - ( 1 ) ) * ( 1 ) ] ) ) ;
ilamg [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_div( 1.f , lamg ) ;
l_qg [ ( ( k ) - ( kts ) ) * ( 1 ) ] = true ;
}
else {
rg [ ( ( k ) - ( kts ) ) * ( 1 ) ] = 1.e-12f ;
l_qg [ ( ( k ) - ( kts ) ) * ( 1 ) ] = false ;
}
}
melti = false ;
k_0 = kts ;
for ( k = ( kte - 1 ) ;
k >= kts ;
k += ( - 1 ) ) {
if ( ( ( ( temp [ ( ( k ) - ( kts ) ) * ( 1 ) ] > 273.15f ) && l_qr [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) && ( l_qs [ ( ( ( k + 1 ) ) - ( kts ) ) * ( 1 ) ] || l_qg [ ( ( ( k + 1 ) ) - ( kts ) ) * ( 1 ) ] ) ) ) {
k_0 = tc_max( ( k + 1 ) , k_0 ) ;
melti = true ;
break ;
}
}
for ( k = kts ;
k <= kte ;
k += 1 ) {
ze_rain [ ( ( k ) - ( kts ) ) * ( 1 ) ] = 1.e-22f ;
ze_snow [ ( ( k ) - ( kts ) ) * ( 1 ) ] = 1.e-22f ;
ze_graupel [ ( ( k ) - ( kts ) ) * ( 1 ) ] = 1.e-22f ;
if ( l_qr [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) {
ze_rain [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_mul( tc_mul( n0_r [ ( ( k ) - ( kts ) ) * ( 1 ) ] , S . xcrg [ ( ( 4 ) - ( 1 ) ) * ( 1 ) ] ) , tc_pow( ilamr [ ( ( k ) - ( kts ) ) * ( 1 ) ] , S . xcre [ ( ( 4 ) - ( 1 ) ) * ( 1 ) ] ) ) ;
}
if ( l_qs [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) {
ze_snow [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_mul( tc_mul( tc_mul( tc_mul( tc_mul( tc_mul( tc_mul( tc_div( 0.176f , 0.93f ) , tc_div( 6.0f , S . pi ) ) , tc_div( 6.0f , S . pi ) ) , tc_div( S . xam_s , 900.0f ) ) , tc_div( S . xam_s , 900.0f ) ) , n0_s [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) , S . xcsg [ ( ( 4 ) - ( 1 ) ) * ( 1 ) ] ) , tc_pow( ilams [ ( ( k ) - ( kts ) ) * ( 1 ) ] , S . xcse [ ( ( 4 ) - ( 1 ) ) * ( 1 ) ] ) ) ;
}
if ( l_qg [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) {
ze_graupel [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_mul( tc_mul( tc_mul( tc_mul( tc_mul( tc_mul( tc_mul( tc_div( 0.176f , 0.93f ) , tc_div( 6.0f , S . pi ) ) , tc_div( 6.0f , S . pi ) ) , tc_div( S . xam_g , 900.0f ) ) , tc_div( S . xam_g , 900.0f ) ) , n0_g [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) , S . xcgg [ ( ( 4 ) - ( 1 ) ) * ( 1 ) ] ) , tc_pow( ilamg [ ( ( k ) - ( kts ) ) * ( 1 ) ] , S . xcge [ ( ( 4 ) - ( 1 ) ) * ( 1 ) ] ) ) ;
}
}
if ( ( melti && ( k_0 >= ( kts + 1 ) ) ) ) {
for ( k = ( k_0 - 1 ) ;
k >= kts ;
k += ( - 1 ) ) {
if ( ( l_qs [ ( ( k ) - ( kts ) ) * ( 1 ) ] && l_qs [ ( ( k_0 ) - ( kts ) ) * ( 1 ) ] ) ) {
fmelt_s = tc_max( tc_min( tc_sub( SD8 ( 0x3ff0000000000000ul ) , tc_div( rs [ ( ( k ) - ( kts ) ) * ( 1 ) ] , rs [ ( ( k_0 ) - ( kts ) ) * ( 1 ) ] ) ) , SD8 ( 0x3fefae147ae147aeul ) ) , SD8 ( 0x3f747ae147ae147bul ) ) ;
eta = SD8 ( 0x0000000000000000ul ) ;
lams = tc_div( 1.f , ilams [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) ;
for ( n = 1 ;
n <= S . nrbins ;
n += 1 ) {
x = tc_mul( S . xam_s , tc_pow( S . xxds [ ( ( n ) - ( 1 ) ) * ( 1 ) ] , S . xbm_s ) ) ;
cback = tc44 :: refl_rayleigh_soak_wetgraupel ( x , ( Soft8 ) ( S . xocms ) , ( Soft8 ) ( S . xobms ) , fmelt_s , S . melt_outside_s , S . m_w_0 , S . m_i_0 , S . pi5 , S . lamda4 ) ;
f_d = tc_mul( tc_mul( n0_s [ ( ( k ) - ( kts ) ) * ( 1 ) ] , tc_pow( S . xxds [ ( ( n ) - ( 1 ) ) * ( 1 ) ] , S . xmu_s ) ) , tc_exp( tc_mul( ( - lams ) , S . xxds [ ( ( n ) - ( 1 ) ) * ( 1 ) ] ) ) ) ;
eta = tc_add( eta , tc_mul( tc_mul( tc_mul( f_d , cback ) , S . simpson [ ( ( n ) - ( 1 ) ) * ( 1 ) ] ) , S . xdts [ ( ( n ) - ( 1 ) ) * ( 1 ) ] ) ) ;
}
ze_snow [ ( ( k ) - ( kts ) ) * ( 1 ) ] = ( F ) ( tc_mul( tc_div( S . lamda4 , tc_mul( S . pi5 , S . k_w ) ) , eta ) ) ;
}
if ( ( l_qg [ ( ( k ) - ( kts ) ) * ( 1 ) ] && l_qg [ ( ( k_0 ) - ( kts ) ) * ( 1 ) ] ) ) {
fmelt_g = tc_max( tc_min( tc_sub( SD8 ( 0x3ff0000000000000ul ) , tc_div( rg [ ( ( k ) - ( kts ) ) * ( 1 ) ] , rg [ ( ( k_0 ) - ( kts ) ) * ( 1 ) ] ) ) , SD8 ( 0x3fefae147ae147aeul ) ) , SD8 ( 0x3f747ae147ae147bul ) ) ;
eta = SD8 ( 0x0000000000000000ul ) ;
lamg = tc_div( 1.f , ilamg [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) ;
for ( n = 1 ;
n <= S . nrbins ;
n += 1 ) {
x = tc_mul( S . xam_g , tc_pow( S . xxdg [ ( ( n ) - ( 1 ) ) * ( 1 ) ] , S . xbm_g ) ) ;
cback = tc44 :: refl_rayleigh_soak_wetgraupel ( x , ( Soft8 ) ( S . xocmg ) , ( Soft8 ) ( S . xobmg ) , fmelt_g , S . melt_outside_g , S . m_w_0 , S . m_i_0 , S . pi5 , S . lamda4 ) ;
f_d = tc_mul( tc_mul( n0_g [ ( ( k ) - ( kts ) ) * ( 1 ) ] , tc_pow( S . xxdg [ ( ( n ) - ( 1 ) ) * ( 1 ) ] , S . xmu_g ) ) , tc_exp( tc_mul( ( - lamg ) , S . xxdg [ ( ( n ) - ( 1 ) ) * ( 1 ) ] ) ) ) ;
eta = tc_add( eta , tc_mul( tc_mul( tc_mul( f_d , cback ) , S . simpson [ ( ( n ) - ( 1 ) ) * ( 1 ) ] ) , S . xdtg [ ( ( n ) - ( 1 ) ) * ( 1 ) ] ) ) ;
}
ze_graupel [ ( ( k ) - ( kts ) ) * ( 1 ) ] = ( F ) ( tc_mul( tc_div( S . lamda4 , tc_mul( S . pi5 , S . k_w ) ) , eta ) ) ;
}
}
}
for ( k = kte ;
k >= kts ;
k += ( - 1 ) ) {
dbz [ ( ( k ) - ( kts ) ) * ( 1 ) ] = tc_mul( 10.f , tc_log10( tc_mul( tc_add( tc_add( ze_rain [ ( ( k ) - ( kts ) ) * ( 1 ) ] , ze_snow [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) , ze_graupel [ ( ( k ) - ( kts ) ) * ( 1 ) ] ) , SD8 ( 0x43abc16d674ec800ul ) ) ) ) ;
}
}


#include "hexv842_wsm6_radar_f32_bindings.metalh"
kernel void hexv842_wsm6_radar_f32(constant Hexv842Wsm6RadarF32Resources &r [[buffer(0)]], constant Hexv842Wsm6RadarF32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 if(gid>=uint(p.count))return;
 int col=p.first+int(gid),nz=p.nz,nc=p.nc;
 Wsm6State S{};
S.dtcldcr=((device const F*)r.state)[0*p.state_stride];
S.n0r=((device const F*)r.state)[1*p.state_stride];
S.avtr=((device const F*)r.state)[2*p.state_stride];
S.bvtr=((device const F*)r.state)[3*p.state_stride];
S.r0=((device const F*)r.state)[4*p.state_stride];
S.peaut=((device const F*)r.state)[5*p.state_stride];
S.xncr=((device const F*)r.state)[6*p.state_stride];
S.xmyu=((device const F*)r.state)[7*p.state_stride];
S.avts=((device const F*)r.state)[8*p.state_stride];
S.bvts=((device const F*)r.state)[9*p.state_stride];
S.lamdarmax=((device const F*)r.state)[10*p.state_stride];
S.lamdasmax=((device const F*)r.state)[11*p.state_stride];
S.dicon=((device const F*)r.state)[12*p.state_stride];
S.dimax=((device const F*)r.state)[13*p.state_stride];
S.pfrz1=((device const F*)r.state)[14*p.state_stride];
S.pfrz2=((device const F*)r.state)[15*p.state_stride];
S.qcrmin=((device const F*)r.state)[16*p.state_stride];
S.eacrc=((device const F*)r.state)[17*p.state_stride];
S.dens=((device const F*)r.state)[18*p.state_stride];
S.qs0=((device const F*)r.state)[19*p.state_stride];
S.n0smax=((device const F*)r.state)[20*p.state_stride];
S.n0s=((device const F*)r.state)[21*p.state_stride];
S.alpha=((device const F*)r.state)[22*p.state_stride];
S.qc0=((device const F*)r.state)[23*p.state_stride];
S.qck1=((device const F*)r.state)[24*p.state_stride];
S.bvtr1=((device const F*)r.state)[25*p.state_stride];
S.bvtr2=((device const F*)r.state)[26*p.state_stride];
S.bvtr3=((device const F*)r.state)[27*p.state_stride];
S.bvtr4=((device const F*)r.state)[28*p.state_stride];
S.g1pbr=((device const F*)r.state)[29*p.state_stride];
S.g3pbr=((device const F*)r.state)[30*p.state_stride];
S.g4pbr=((device const F*)r.state)[31*p.state_stride];
S.g5pbro2=((device const F*)r.state)[32*p.state_stride];
S.pvtr=((device const F*)r.state)[33*p.state_stride];
S.eacrr=((device const F*)r.state)[34*p.state_stride];
S.pacrr=((device const F*)r.state)[35*p.state_stride];
S.bvtr6=((device const F*)r.state)[36*p.state_stride];
S.g6pbr=((device const F*)r.state)[37*p.state_stride];
S.precr1=((device const F*)r.state)[38*p.state_stride];
S.precr2=((device const F*)r.state)[39*p.state_stride];
S.roqimax=((device const F*)r.state)[40*p.state_stride];
S.bvts1=((device const F*)r.state)[41*p.state_stride];
S.bvts2=((device const F*)r.state)[42*p.state_stride];
S.bvts3=((device const F*)r.state)[43*p.state_stride];
S.bvts4=((device const F*)r.state)[44*p.state_stride];
S.g1pbs=((device const F*)r.state)[45*p.state_stride];
S.g3pbs=((device const F*)r.state)[46*p.state_stride];
S.g4pbs=((device const F*)r.state)[47*p.state_stride];
S.n0g=((device const F*)r.state)[48*p.state_stride];
S.avtg=((device const F*)r.state)[49*p.state_stride];
S.bvtg=((device const F*)r.state)[50*p.state_stride];
S.deng=((device const F*)r.state)[51*p.state_stride];
S.lamdagmax=((device const F*)r.state)[52*p.state_stride];
S.g5pbso2=((device const F*)r.state)[53*p.state_stride];
S.pvts=((device const F*)r.state)[54*p.state_stride];
S.pacrs=((device const F*)r.state)[55*p.state_stride];
S.precs1=((device const F*)r.state)[56*p.state_stride];
S.precs2=((device const F*)r.state)[57*p.state_stride];
S.pidn0r=((device const F*)r.state)[58*p.state_stride];
S.xlv1=((device const F*)r.state)[59*p.state_stride];
S.pacrc=((device const F*)r.state)[60*p.state_stride];
S.pi=((device const F*)r.state)[61*p.state_stride];
S.bvtg1=((device const F*)r.state)[62*p.state_stride];
S.bvtg2=((device const F*)r.state)[63*p.state_stride];
S.bvtg3=((device const F*)r.state)[64*p.state_stride];
S.bvtg4=((device const F*)r.state)[65*p.state_stride];
S.g1pbg=((device const F*)r.state)[66*p.state_stride];
S.g3pbg=((device const F*)r.state)[67*p.state_stride];
S.g4pbg=((device const F*)r.state)[68*p.state_stride];
S.g5pbgo2=((device const F*)r.state)[69*p.state_stride];
S.pvtg=((device const F*)r.state)[70*p.state_stride];
S.pacrg=((device const F*)r.state)[71*p.state_stride];
S.precg1=((device const F*)r.state)[72*p.state_stride];
S.precg2=((device const F*)r.state)[73*p.state_stride];
S.pidn0g=((device const F*)r.state)[74*p.state_stride];
S.rslopermax=((device const F*)r.state)[75*p.state_stride];
S.rslopesmax=((device const F*)r.state)[76*p.state_stride];
S.rslopegmax=((device const F*)r.state)[77*p.state_stride];
S.rsloperbmax=((device const F*)r.state)[78*p.state_stride];
S.rslopesbmax=((device const F*)r.state)[79*p.state_stride];
S.rslopegbmax=((device const F*)r.state)[80*p.state_stride];
S.rsloper2max=((device const F*)r.state)[81*p.state_stride];
S.rslopes2max=((device const F*)r.state)[82*p.state_stride];
S.rslopeg2max=((device const F*)r.state)[83*p.state_stride];
S.rsloper3max=((device const F*)r.state)[84*p.state_stride];
S.rslopes3max=((device const F*)r.state)[85*p.state_stride];
S.rslopeg3max=((device const F*)r.state)[86*p.state_stride];
S.pidn0s=((device const F*)r.state)[87*p.state_stride];
S.pidnc=((device const F*)r.state)[88*p.state_stride];

 state_parameters(S);
 F denr=((device const F*)r.constants)[16],dens=100.f;

S . xam_r = tc_div( tc_mul( S . pi , denr ) , 6.f ) ;
S . xbm_r = 3.f ;
S . xmu_r = 0.f ;
S . xam_s = tc_div( tc_mul( S . pi , dens ) , 6.f ) ;
S . xbm_s = 3.f ;
S . xmu_s = 0.f ;
S . xam_g = tc_div( tc_mul( S . pi , S . deng ) , 6.f ) ;
S . xbm_g = 3.f ;
S . xmu_g = 0.f ;

 radar_init(S);
 device F *wf=(device F*)r.ws_f+ulong(gid)*18*(nz+3);
 device Soft8 *wd=(device Soft8*)r.ws_d+ulong(gid)*6*(nz+3);
 device bool *wb=(device bool*)r.ws_b+ulong(gid)*3*(nz+3);
 device F *t=wf,*qv=t+nz+3,*qr=qv+nz+3,*qs=qr+nz+3,*qg=qs+nz+3,*pr=qg+nz+3,*dbz=pr+nz+3,*zp=dbz+nz+3;
 wf=zp+nz+3;
 for(int k=0;k<nz;++k){int x=k*nc+col;
 t[k]=tc_mul(((device const F*)r.th)[x],((device const F*)r.pii)[x]);
 qv[k]=((device const F*)r.qv)[x];qr[k]=((device const F*)r.qr)[x];
 qs[k]=((device const F*)r.qs)[x];qg[k]=((device const F*)r.qg)[x];pr[k]=((device const F*)r.p)[x];
 F z=((device const F*)r.zgrid)[x],z0=((device const F*)r.zgrid)[col],z1=((device const F*)r.zgrid)[x+nc];
 zp[k]=tc_add(tc_sub(z,z0),tc_mul(.5f,tc_sub(z1,z)));
 }
 refl10cm_wsm6(S,qv,qr,qs,qg,t,pr,dbz,1,nz,nz,wf,wd,wb);
 F maximum=-35.f;int kp=0;
 for(int k=0;k<nz;++k){dbz[k]=tc_max(dbz[k],-35.f);((device F*)r.refl)[k*nc+col]=dbz[k];maximum=tc_max(maximum,dbz[k]);if(zp[k]<1000.f)kp=k;}
 ((device F*)r.refl_max)[col]=maximum;
 F value=dbz[kp];
 if(kp+1<nz){F w1=tc_div(tc_sub(zp[kp+1],1000.f),tc_sub(zp[kp+1],zp[kp]));value=tc_add(tc_mul(w1,dbz[kp]),tc_mul(tc_sub(1.f,w1),dbz[kp+1]));}
 ((device F*)r.refl_1km)[col]=value;
 ((device F*)r.refl_1km_max)[col]=p.running_max?tc_max(((device F*)r.refl_1km_max)[col],value):value;
}
static inline F wsm6_rslf ( F p , F t ) {
const F c [ 9 ] = {
.611583699E03f , .444606896E02f , .143177157E01f , .264224321E-1f , .299291081E-3f , .203154182E-5f , .702620698E-8f , .379534310E-11f , - .321582393E-13f }
;
F x = tc_max( - 80.0f , tc_sub( ( t ) , ( 273.16f ) ) ) , es = c [ 8 ] ;
for ( int i = 7 ;
i >= 0 ;
-- i ) es = tc_add( ( c [ i ] ) , ( tc_mul( ( x ) , ( es ) ) ) ) ;
es = tc_min( es , tc_mul( ( p ) , ( 0.15f ) ) ) ;
return tc_div( ( tc_mul( ( 0.622f ) , ( es ) ) ) , ( tc_sub( ( p ) , ( es ) ) ) ) ;
}

static inline F wsm6_rsif ( F p , F t ) {
const F c [ 9 ] = {
.609868993E03f , .499320233E02f , .184672631E01f , .402737184E-1f , .565392987E-3f , .521693933E-5f , .307839583E-7f , .105785160E-9f , .161444444E-12f }
;
F x = tc_max( - 80.0f , tc_sub( ( t ) , ( 273.16f ) ) ) , es = c [ 8 ] ;
for ( int i = 7 ;
i >= 0 ;
-- i ) es = tc_add( ( c [ i ] ) , ( tc_mul( ( x ) , ( es ) ) ) ) ;
es = tc_min( es , tc_mul( ( p ) , ( 0.15f ) ) ) ;
return tc_div( ( tc_mul( ( 0.622f ) , ( es ) ) ) , ( tc_sub( ( p ) , ( es ) ) ) ) ;
}

#include "hexv842_wsm6_relhum_f32_bindings.metalh"
kernel void hexv842_wsm6_relhum_f32(constant Hexv842Wsm6RelhumF32Resources &r [[buffer(0)]],constant Hexv842Wsm6RelhumF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if(gid>=uint(p.nz*p.nc))return;
 int i=int(gid)%p.nc;
 if(i>=p.ns){r.relhum[gid]=0.f;return;}
 F t=tc_mul(((device const F*)r.th)[gid],((device const F*)r.pii)[gid]);
 F pressure=((device const F*)r.pres)[gid],qvs=wsm6_rslf(pressure,t);
 if(tc_sub(t,273.16f)<=0.f)qvs=wsm6_rsif(pressure,t);
 ((device F*)r.relhum)[gid]=tc_mul(tc_div(((device const F*)r.qv)[gid],qvs),100.f);
}
