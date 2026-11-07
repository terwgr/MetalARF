// Translated from the CUDA turbine equations by tools/wind_farm_metal/generate.py.

#include "wind_power_bindings.metalh"

#include "wind_gad_bindings.metalh"

#include "wind_source_bindings.metalh"

#include "wind_wake_bindings.metalh"

#include "wind_sample_bindings.metalh"

#include "wind_fitch_sparse_bindings.metalh"

#include "wind_locate_bindings.metalh"

#include "wind_locate_cells_bindings.metalh"

#include "wind_farm_support.metalh"

template < typename P_ll , typename P_lat , typename P_lon , typename P_indices > static inline void wind_locate_cells_body ( int nt , int nc , P_ll ll , P_lat lat , P_lon lon , P_indices indices , uint gid ) {
int a = 0 * 1 + gid ;
if ( a >= nt ) return ;
Soft8 la = ( Soft8 ) ll [ 2 * a ] * SD8(0x400921fb54442d18ul) / SD8(0x4066800000000000ul) , lo = ( Soft8 ) ll [ 2 * a + 1 ] * SD8(0x400921fb54442d18ul) / SD8(0x4066800000000000ul) ;
Soft8 sl = fd64_sin ( la ) , cl = fd64_cos ( la ) , best = - SD8(0x4000000000000000ul) ;
int cell = 0 ;
for ( int c = 0 ;
c < nc ;
++ c ) {
Soft8 dot = sl * fd64_sin ( ( Soft8 ) lat [ c ] ) + cl * fd64_cos ( ( Soft8 ) lat [ c ] ) * fd64_cos ( ( Soft8 ) lon [ c ] - lo ) ;
if ( dot > best ) {
best = dot ;
cell = c ;
}
}
indices [ a ] = cell ;
}


kernel void wind_locate_cells(constant WindLocateCellsResources &r [[buffer(0)]], constant WindLocateCellsParams &p [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    wind_locate_cells_body(p.nt, p.nc, (const device F *)r.ll, (const device F *)r.lat, (const device F *)r.lon, r.indices, gid);
}


static inline F clampw ( F x , F lo , F hi ) {
return fminf ( hi , fmaxf ( lo , x ) ) ;
}


template < typename P_ll , typename P_lat , typename P_lon , typename P_xy > static inline void wind_locate_body ( int nt , int nx , int ny , P_ll ll , P_lat lat , P_lon lon , P_xy xy , uint gid ) {
int a = 0 * 1 + gid ;
if ( a >= nt ) return ;
F la = ll [ 2 * a ] , lo = ll [ 2 * a + 1 ] , scale = wf_cosf ( __fdiv_rn ( la * SF(0x40490fdbu) , SF(0x43340000u) ) ) , best = SF(0x7149f2cau) ;
int c = 0 ;
for ( int q = 0 ;
q < nx * ny ;
++ q ) {
F dl = wf_remainderf ( lon [ q ] - lo , SF(0x43b40000u) ) * scale , dlat = lat [ q ] - la ;
F d = dl * dl + dlat * dlat ;
if ( d < best ) {
best = d ;
c = q ;
}
}
int i = c % nx , j = c / nx , il = max ( i - 1 , 0 ) , ir = min ( i + 1 , nx - 1 ) , jl = max ( j - 1 , 0 ) , jr = min ( j + 1 , ny - 1 ) ;
F ax = wf_remainderf ( lon [ j * nx + ir ] - lon [ j * nx + il ] , SF(0x43b40000u) ) * scale / ( ir - il ) ;
F ay = ( lat [ j * nx + ir ] - lat [ j * nx + il ] ) / ( ir - il ) ;
F bx = wf_remainderf ( lon [ jr * nx + i ] - lon [ jl * nx + i ] , SF(0x43b40000u) ) * scale / ( jr - jl ) ;
F by = ( lat [ jr * nx + i ] - lat [ jl * nx + i ] ) / ( jr - jl ) , det = ax * by - ay * bx ;
F x = - wf_remainderf ( lon [ c ] - lo , SF(0x43b40000u) ) * scale , y = la - lat [ c ] ;
xy [ 2 * a ] = i + ( x * by - y * bx ) / det ;
xy [ 2 * a + 1 ] = j + ( y * ax - x * ay ) / det ;
}


kernel void wind_locate(constant WindLocateResources &r [[buffer(0)]], constant WindLocateParams &p [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    wind_locate_body(p.nt, p.nx, p.ny, (const device F *)r.ll, (const device F *)r.lat, (const device F *)r.lon, (device F *)r.xy, gid);
}


template < typename P_c > static inline F curve ( P_c c , int start , int n , F u , int field ) {
if ( u < c [ 5 * start ] || u > c [ 5 * ( start + n - 1 ) ] ) return SF(0x00000000u) ;
int j = 0 ;
while ( j < n - 2 && c [ 5 * ( start + j + 1 ) ] < u ) ++ j ;
auto a = c + 5 * ( start + j ) ;
auto b = a + 5 ;
F t = ( u - a [ 0 ] ) / ( b [ 0 ] - a [ 0 ] ) ;
return a [ field ] + t * ( b [ field ] - a [ field ] ) ;
}


template < typename P_c , typename P_limits > static inline F fitch_curve ( P_c c , int start , int n , F u , int field , P_limits limits ) {
if ( limits [ 0 ] >= SF(0x00000000u) ) {
if ( u <= limits [ 0 ] ) return field == 1 ? limits [ 2 ] : SF(0x00000000u) ;
if ( u >= limits [ 1 ] ) return field == 1 ? limits [ 3 ] : SF(0x00000000u) ;
}
return curve ( c , start , n , u , field ) ;
}


static inline F diskprimitive ( F z , F r ) {
z = clampw ( z , - r , r ) ;
return z * sqrtf ( fmaxf ( SF(0x00000000u) , r * r - z * z ) ) + r * r * wf_asinf ( z / r ) ;
}


static inline F overlap ( F r , F R , F d ) {
if ( d >= r + R ) return SF(0x00000000u) ;
if ( d <= fabsf ( R - r ) ) {
F a = fminf ( r , R ) ;
return SF(0x40490fdbu) * a * a ;
}
F x = clampw ( ( d * d + r * r - R * R ) / ( 2 * d * r ) , - SF(0x3f800000u) , SF(0x3f800000u) ) ;
F y = clampw ( ( d * d + R * R - r * r ) / ( 2 * d * R ) , - SF(0x3f800000u) , SF(0x3f800000u) ) ;
return r * r * wf_acosf ( x ) + R * R * wf_acosf ( y ) - SF(0x3f000000u) * sqrtf ( fmaxf ( SF(0x00000000u) , ( - d + r + R ) * ( d + r - R ) * ( d - r + R ) * ( d + r + R ) ) ) ;
}


kernel void wind_fitch_sparse(constant WindFitchSparseResources &r [[buffer(0)]], constant WindFitchSparseParams &prm [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
if (gid >= uint(prm.n)) return;
int nc = int(prm.nc);
int nz = int(prm.nz);
device const int *cols = (device const int *)r.cols;
device const int *ptr = (device const int *)r.ptr;
device const int *ids = (device const int *)r.ids;
device const F *t = (device const F *)r.t;
device const int *meta = (device const int *)r.meta;
device const F *curves = (device const F *)r.curves;
device const F *limits = (device const F *)r.limits;
device const F *zi = (device const F *)r.zi;
device const F *u = (device const F *)r.u;
device const F *v = (device const F *)r.v;
device const F *rho = (device const F *)r.rho;
device const F *msf = (device const F *)r.msf;
F area = F(prm.area);
F dt = F(prm.dt);
F factor = F(prm.factor);
int hubnormalize = int(prm.hubnormalize);
device F *du = (device F *)r.du;
device F *dv = (device F *)r.dv;
device F *qke = (device F *)r.qke;
device F *adv = (device F *)r.adv;
device F *hub = (device F *)r.hub;
device F *tke_rate = (device F *)r.tke_rate;
int c = cols [ ( gid / 128 ) ] , k = ( gid % 128 ) , p = k * nc + c ;
threadgroup uint sums_bits[128]; threadgroup F *sums = (threadgroup F *)sums_bits;
threadgroup uint hs_bits[7]; threadgroup F *hs = (threadgroup F *)hs_bits;
F su = SF(0x00000000u) , sv = SF(0x00000000u) , st = SF(0x00000000u) ;
F U = k < nz ? wf_hypotf ( u [ p ] , v [ p ] ) : SF(0x00000000u) ;
for ( int j = ptr [ c ] ;
j < ptr [ c + 1 ] ;
++ j ) {
int a = ids [ j ] ;
F R = t [ 5 * a + 3 ] , h = t [ 5 * a + 2 ] + zi [ c ] ;
if ( k == 0 ) {
int z = 0 ;
while ( z < nz - 2 && SF(0x3f000000u) * ( zi [ ( z + 1 ) * nc + c ] + zi [ ( z + 2 ) * nc + c ] ) < h ) ++ z ;
int q = z * nc + c ;
F z0 = SF(0x3f000000u) * ( zi [ q ] + zi [ q + nc ] ) , z1 = SF(0x3f000000u) * ( zi [ q + nc ] + zi [ q + 2 * nc ] ) ;
F f = clampw ( ( h - z0 ) / ( z1 - z0 ) , SF(0x00000000u) , SF(0x3f800000u) ) ;
F uh = u [ q ] + f * ( u [ q + nc ] - u [ q ] ) , vh = v [ q ] + f * ( v [ q + nc ] - v [ q ] ) ;
F sp = wf_hypotf ( u [ q ] , v [ q ] ) + f * ( wf_hypotf ( u [ q + nc ] , v [ q + nc ] ) - wf_hypotf ( u [ q ] , v [ q ] ) ) ;
F mag = wf_hypotf ( uh , vh ) ;
if ( mag > SF(0x00000000u) ) {
uh *= sp / mag ;
vh *= sp / mag ;
}
hs [ 0 ] = sp ;
hs [ 1 ] = fitch_curve ( curves , meta [ 5 * a ] , meta [ 5 * a + 1 ] , sp , 1 , limits + 5 * a ) * t [ 5 * a + 4 ] ;
hs [ 2 ] = fitch_curve ( curves , meta [ 5 * a ] , meta [ 5 * a + 1 ] , sp , 2 , limits + 5 * a ) ;
hs [ 3 ] = limits [ 5 * a + 4 ] > SF(0x00000000u) ? limits [ 5 * a + 4 ] : rho [ q ] + f * ( rho [ q + nc ] - rho [ q ] ) ;
hs [ 4 ] = uh ;
hs [ 5 ] = vh ;
}
F A = k < nz ? diskprimitive ( zi [ p + nc ] - h , R ) - diskprimitive ( zi [ p ] - h , R ) : SF(0x00000000u) ;
sums [ k ] = A * U * U * U ;
threadgroup_barrier ( mem_flags :: mem_threadgroup ) ;
for ( int stride = 64 ;
stride > 0 ;
stride >>= 1 ) {
if ( k < stride ) sums [ k ] = sums [ k ] + (sums [ k + stride ] );
threadgroup_barrier ( mem_flags :: mem_threadgroup ) ;
}
F ec = ( hs [ 2 ] > SF(0x00000000u) && sums [ 0 ] > SF(0x00000000u) ) ? SF(0x40490fdbu) * R * R * hs [ 0 ] * hs [ 0 ] * hs [ 0 ] / sums [ 0 ] : SF(0x3f800000u) ;
if ( k == 0 ) {
auto o = hub + 10 * a ;
o [ 0 ] = hs [ 4 ] ;
o [ 1 ] = hs [ 5 ] ;
o [ 2 ] = hs [ 0 ] ;
o [ 3 ] = hs [ 1 ] ;
o [ 4 ] = hs [ 3 ] ;
o [ 5 ] = SF(0x00000000u) ;
o [ 6 ] = h ;
o [ 7 ] = SF(0x3f000000u) * hs [ 3 ] * SF(0x40490fdbu) * R * R * hs [ 2 ] * hs [ 0 ] * hs [ 0 ] * hs [ 0 ] * t [ 5 * a + 4 ] ;
o [ 8 ] = hs [ 2 ] ;
o [ 9 ] = ec ;
}
if ( k < nz && A > SF(0x00000000u) && t [ 5 * a + 4 ] != SF(0x00000000u) ) {
F ct = hubnormalize ? hs [ 1 ] : fitch_curve ( curves , meta [ 5 * a ] , meta [ 5 * a + 1 ] , U , 1 , limits + 5 * a ) ;
F cp = hubnormalize ? hs [ 2 ] : fitch_curve ( curves , meta [ 5 * a ] , meta [ 5 * a + 1 ] , U , 2 , limits + 5 * a ) ;
if ( ! hubnormalize ) ec = SF(0x3f800000u) ;
F coeff = SF(0x3f000000u) * A * msf [ c ] * msf [ c ] / area / ( zi [ p + nc ] - zi [ p ] ) ;
F drag = ct * U * ec ;
su -= coeff * drag * u [ p ] ;
sv -= coeff * drag * v [ p ] ;
st += coeff * factor * fmaxf ( SF(0x00000000u) , ct - cp ) * U * U * U * ec ;
}
threadgroup_barrier ( mem_flags :: mem_threadgroup ) ;
}
if ( k < nz ) {
du [ p ] = du [ p ] + (su );
dv [ p ] = dv [ p ] + (sv );
qke [ p ] = qke [ p ] + (SF(0x40000000u) * dt * st );
if ( adv != qke ) adv [ p ] = adv [ p ] + (SF(0x40000000u) * dt * st );
tke_rate [ p ] = st ;
}

}


template < typename P_col , typename P_t , typename P_meta , typename P_curves , typename P_limits , typename P_zi , typename P_u , typename P_v , typename P_rho , typename P_km , typename P_hub > static inline void wind_sample_body ( int nt , int nc , int nz , P_col col , P_t t , P_meta meta , P_curves curves , P_limits limits , P_zi zi , P_u u , P_v v , P_rho rho , P_km km , P_hub hub , uint gid ) {
int a = 1 * 0 + gid ;
if ( a >= nt ) return ;
int c = col [ a ] ;
F h = t [ 5 * a + 2 ] + zi [ c ] ;
int k = 0 ;
while ( k < nz - 2 && SF(0x3f000000u) * ( zi [ ( k + 1 ) * nc + c ] + zi [ ( k + 2 ) * nc + c ] ) < h ) ++ k ;
F z0 = SF(0x3f000000u) * ( zi [ k * nc + c ] + zi [ ( k + 1 ) * nc + c ] ) ;
F z1 = SF(0x3f000000u) * ( zi [ ( k + 1 ) * nc + c ] + zi [ ( k + 2 ) * nc + c ] ) ;
F f = clampw ( ( h - z0 ) / ( z1 - z0 ) , SF(0x00000000u) , SF(0x3f800000u) ) ;
int p = k * nc + c ;
F uh = u [ p ] + f * ( u [ p + nc ] - u [ p ] ) , vh = v [ p ] + f * ( v [ p + nc ] - v [ p ] ) ;
F mag = wf_hypotf ( uh , vh ) ;
F sp = wf_hypotf ( u [ p ] , v [ p ] ) + f * ( wf_hypotf ( u [ p + nc ] , v [ p + nc ] ) - wf_hypotf ( u [ p ] , v [ p ] ) ) ;
if ( mag > SF(0x00000000u) ) {
uh *= sp / mag ;
vh *= sp / mag ;
}
auto o = hub + 10 * a ;
o [ 0 ] = uh ;
o [ 1 ] = vh ;
o [ 2 ] = sp ;
o [ 3 ] = fitch_curve ( curves , meta [ 5 * a ] , meta [ 5 * a + 1 ] , sp , 1 , limits + 5 * a ) * t [ 5 * a + 4 ] ;
o [ 4 ] = limits [ 5 * a + 4 ] > SF(0x00000000u) ? limits [ 5 * a + 4 ] : rho [ p ] + f * ( rho [ p + nc ] - rho [ p ] ) ;
o [ 5 ] = fmaxf ( SF(0x00000000u) , km [ p ] + f * ( km [ p + nc ] - km [ p ] ) ) ;
o [ 6 ] = zi [ c ] + t [ 5 * a + 2 ] ;
o [ 8 ] = fitch_curve ( curves , meta [ 5 * a ] , meta [ 5 * a + 1 ] , sp , 2 , limits + 5 * a ) ;
F integral = SF(0x00000000u) , R = t [ 5 * a + 3 ] ;
for ( int kk = 0 ;
kk < nz ;
++ kk ) {
F A = diskprimitive ( zi [ ( kk + 1 ) * nc + c ] - h , R ) - diskprimitive ( zi [ kk * nc + c ] - h , R ) ;
F U = wf_hypotf ( u [ kk * nc + c ] , v [ kk * nc + c ] ) ;
integral += A * U * U * U ;
}
o [ 9 ] = ( o [ 8 ] > SF(0x00000000u) && integral > SF(0x00000000u) ) ? SF(0x40490fdbu) * R * R * sp * sp * sp / integral : SF(0x3f800000u) ;
o [ 7 ] = SF(0x3f000000u) * o [ 4 ] * SF(0x40490fdbu) * t [ 5 * a + 3 ] * t [ 5 * a + 3 ] * o [ 8 ] * sp * sp * sp * t [ 5 * a + 4 ] ;
}


kernel void wind_sample(constant WindSampleResources &r [[buffer(0)]], constant WindSampleParams &p [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    wind_sample_body(p.nt, p.nc, p.nz, r.col, (const device F *)r.t, r.meta, (const device F *)r.curves, (const device F *)r.limits, (const device F *)r.zi, (const device F *)r.u, (const device F *)r.v, (const device F *)r.rho, (const device F *)r.km, (device F *)r.hub, gid);
}


template < typename P_t , typename P_h , typename P_speed > static inline void wind_wake_body ( int nt , P_t t , P_h h , F kw , F maxd , F sd , int superposition , P_speed speed , uint gid ) {
int index = 1 * 0 + gid ;
if ( index >= nt * 7 ) return ;
int a = index / 7 , s = index % 7 ;
const F offsets [ 7 ] = {
- SF(0x40200000u) , - SF(0x3fc00000u) , - SF(0x3f000000u) , SF(0x00000000u) , SF(0x3f000000u) , SF(0x3fc00000u) , SF(0x40200000u) }
;
F base = h [ 10 * a + 2 ] , sum = SF(0x00000000u) ;
int count = 0 ;
F deltaangle = __fdiv_rn ( offsets [ s ] * SF(0x40490fdbu) , SF(0x43340000u) ) ;
for ( int b = 0 ;
b < nt ;
++ b ) {
if ( b == a || t [ 5 * b + 4 ] == SF(0x00000000u) ) continue ;
F theta = wf_atan2f ( h [ 10 * a + 1 ] + h [ 10 * b + 1 ] , h [ 10 * a ] + h [ 10 * b ] ) + deltaangle ;
F cs = wf_pair_trig ( theta, true ) , sn = wf_pair_trig ( theta, false ) , dx = t [ 5 * a ] - t [ 5 * b ] , dy = t [ 5 * a + 1 ] - t [ 5 * b + 1 ] ;
F x = dx * cs + dy * sn , y = - dx * sn + dy * cs , R = t [ 5 * b + 3 ] ;
if ( x <= SF(0x00000000u) || x > SF(0x40000000u) * R * maxd || fabsf ( y ) > x * SF(0x3f13cd3au) ) continue ;
F r = t [ 5 * a + 3 ] , d = wf_hypotf ( y , h [ 10 * a + 6 ] - h [ 10 * b + 6 ] ) ;
F fraction = overlap ( r , R + kw * x , d ) / ( SF(0x40490fdbu) * r * r ) ;
if ( fraction <= SF(0x00000000u) ) continue ;
F def = ( SF(0x3f800000u) - sqrtf ( fmaxf ( SF(0x00000000u) , SF(0x3f800000u) - h [ 10 * b + 3 ] ) ) ) / ( ( SF(0x3f800000u) + kw * x / R ) * ( SF(0x3f800000u) + kw * x / R ) ) ;
F loss = h [ 10 * b + 2 ] * def * fraction ;
if ( superposition == 1 ) sum += loss ;
else if ( superposition == 2 ) sum += loss * loss ;
else {
F mix = base * ( SF(0x3f800000u) - fraction ) + h [ 10 * b + 2 ] * ( SF(0x3f800000u) - def ) * fraction ;
sum += mix * mix ;
}
++ count ;
}
speed [ index ] = fmaxf ( SF(0x00000000u) , count == 0 ? base : superposition == 1 ? base - sum : superposition == 2 ? base - sqrtf ( sum ) : sqrtf ( sum / count ) ) ;
}


kernel void wind_wake(constant WindWakeResources &r [[buffer(0)]], constant WindWakeParams &p [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    wind_wake_body(p.nt, (const device F *)r.t, (const device F *)r.h, F(p.kw), F(p.maxd), F(p.sd), p.superposition, (device F *)r.speed, gid);
}


template < typename P_ptr , typename P_ids , typename P_t , typename P_meta , typename P_curves , typename P_limits , typename P_hub , typename P_wake , typename P_zi , typename P_u , typename P_v , typename P_msf , typename P_du , typename P_dv , typename P_qke , typename P_adv , typename P_tke_rate > static inline void wind_source_body ( int scheme , int nc , int nz , int nx , int ny , P_ptr ptr , P_ids ids , P_t t , P_meta meta , P_curves curves , P_limits limits , P_hub hub , P_wake wake , P_zi zi , P_u u , P_v v , P_msf msf , F area , F dx , F dt , F factor , F sigma0factor , F sd , int hubnormalize , P_du du , P_dv dv , P_qke qke , P_adv adv , P_tke_rate tke_rate , uint gid ) {
int p = 1 * 0 + gid ;
if ( p >= nc * nz ) return ;
int c = p % nc ;
F su = SF(0x00000000u) , sv = SF(0x00000000u) , st = SF(0x00000000u) ;
F z0 = zi [ p ] , z1 = zi [ p + nc ] , dz = z1 - z0 , U = wf_hypotf ( u [ p ] , v [ p ] ) ;
const F offsets [ 7 ] = {
- SF(0x40200000u) , - SF(0x3fc00000u) , - SF(0x3f000000u) , SF(0x00000000u) , SF(0x3f000000u) , SF(0x3fc00000u) , SF(0x40200000u) }
;
F weights [ 7 ] , wt = SF(0x00000000u) ;
for ( int s = 0 ;
s < 7 ;
++ s ) {
weights [ s ] = sd > SF(0x00000000u) ? wf_expf ( - SF(0x3f000000u) * offsets [ s ] * offsets [ s ] / ( sd * sd ) ) : ( s == 3 ? SF(0x3f800000u) : SF(0x00000000u) ) ;
wt += weights [ s ] ;
}
for ( int j = ptr [ c ] ;
j < ptr [ c + 1 ] ;
++ j ) {
int a = ids [ j ] ;
if ( t [ 5 * a + 4 ] == SF(0x00000000u) ) continue ;
F R = t [ 5 * a + 3 ] , h = hub [ 10 * a + 6 ] , Uh = hub [ 10 * a + 2 ] ;
F A = diskprimitive ( z1 - h , R ) - diskprimitive ( z0 - h , R ) ;
F inv = msf [ c ] * msf [ c ] / area ;
int off = meta [ 5 * a ] , n = meta [ 5 * a + 1 ] ;
if ( scheme == 2 ) {
F s0 = sigma0factor * R , b = hub [ 10 * a + 5 ] * dx / fmaxf ( Uh , SF(0x358637bdu) ) ;
F s1 = sqrtf ( s0 * s0 + b ) , sig = ( SF(0x40000000u) / SF(0x40400000u) ) * ( s1 * s1 + s1 * s0 + s0 * s0 ) / ( s1 + s0 ) ;
F z = SF(0x3f000000u) * ( z0 + z1 ) - h ;
F wf_force = sqrtf ( SF(0x40490fdbu) / SF(0x41000000u) ) * hub [ 10 * a + 3 ] * R * R * Uh * Uh * inv / sig * wf_expf ( - SF(0x3f000000u) * z * z / ( sig * sig ) ) ;
if ( U > SF(0x00000000u) ) {
su -= wf_force * u [ p ] / U ;
sv -= wf_force * v [ p ] / U ;
}
continue ;
}
if ( A <= SF(0x00000000u) ) continue ;
F drag = SF(0x00000000u) , tk = SF(0x00000000u) ;
if ( scheme == 3 ) {
for ( int s = 0 ;
s < 7 ;
++ s ) {
F ws = wake [ 7 * a + s ] , r = Uh > SF(0x00000000u) ? ws / Uh : SF(0x00000000u) ;
F ct = curve ( curves , off , n , ws , 1 ) , cp = curve ( curves , off , n , ws , 2 ) , uk = r * U ;
drag += weights [ s ] * ct * uk ;
tk += weights [ s ] * factor * ( ct - cp ) * uk * uk * uk ;
}
drag /= wt ;
tk /= wt ;
}
else {
F ct = hubnormalize ? hub [ 10 * a + 3 ] : fitch_curve ( curves , off , n , U , 1 , limits + 5 * a ) , cp = hubnormalize ? hub [ 10 * a + 8 ] : fitch_curve ( curves , off , n , U , 2 , limits + 5 * a ) ;
F ec = hubnormalize ? hub [ 10 * a + 9 ] : SF(0x3f800000u) ;
drag = ct * U * ec ;
tk = factor * fmaxf ( SF(0x00000000u) , ct - cp ) * U * U * U * ec ;
}
F coeff = SF(0x3f000000u) * A * inv / dz ;
su -= coeff * drag * u [ p ] ;
sv -= coeff * drag * v [ p ] ;
st += coeff * tk ;
}
du [ p ] = du [ p ] + (su );
dv [ p ] = dv [ p ] + (sv );
tke_rate [ p ] = st ;
qke [ p ] = qke [ p ] + (SF(0x40000000u) * dt * st );
if ( adv != qke ) adv [ p ] = adv [ p ] + (SF(0x40000000u) * dt * st );
}


kernel void wind_source(constant WindSourceResources &r [[buffer(0)]], constant WindSourceParams &p [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    wind_source_body(p.scheme, p.nc, p.nz, p.nx, p.ny, r.ptr, r.ids, (const device F *)r.t, r.meta, (const device F *)r.curves, (const device F *)r.limits, (const device F *)r.hub, (const device F *)r.wake, (const device F *)r.zi, (const device F *)r.u, (const device F *)r.v, (const device F *)r.msf, F(p.area), F(p.dx), F(p.dt), F(p.factor), F(p.sigma0factor), F(p.sd), p.hubnormalize, (device F *)r.du, (device F *)r.dv, (device F *)r.qke, (device F *)r.adv, (device F *)r.tke_rate, gid);
}


template < typename P_rows , typename P_pm > static inline F polar ( P_rows rows , P_pm pm , int id , F alpha , int field ) {
int o = pm [ 2 * id ] , n = pm [ 2 * id + 1 ] , j = 0 ;
if ( alpha <= rows [ 3 * o ] ) return rows [ 3 * o + field ] ;
if ( alpha >= rows [ 3 * ( o + n - 1 ) ] ) return rows [ 3 * ( o + n - 1 ) + field ] ;
while ( j < n - 2 && rows [ 3 * ( o + j + 1 ) ] < alpha ) ++ j ;
auto p = rows + 3 * ( o + j ) ;
F f = ( alpha - p [ 0 ] ) / ( p [ 3 ] - p [ 0 ] ) ;
return p [ field ] + f * ( p [ 3 + field ] - p [ field ] ) ;
}


template < typename P_ptr , typename P_ids , typename P_t , typename P_meta , typename P_curves , typename P_blades , typename P_polars , typename P_pm , typename P_hub , typename P_zi , typename P_u , typename P_v , typename P_du , typename P_dv , typename P_dw , typename P_failures > static inline void wind_gad_body ( int nc , int nz , int nx , F dx , F dy , F sigma , P_ptr ptr , P_ids ids , P_t t , P_meta meta , P_curves curves , P_blades blades , P_polars polars , P_pm pm , P_hub hub , P_zi zi , P_u u , P_v v , F tol , int maxiter , P_du du , P_dv dv , P_dw dw , P_failures failures , uint gid ) {
int p = 1 * 0 + gid ;
if ( p >= nc * nz ) return ;
int c = p % nc ;
F x = ( c % nx ) * dx , y = ( c / nx ) * dy , z = SF(0x3f000000u) * ( zi [ p ] + zi [ p + nc ] ) ;
F fx = SF(0x00000000u) , fy = SF(0x00000000u) , fz = SF(0x00000000u) ;
int failed = 0 ;
for ( int j = ptr [ c ] ;
j < ptr [ c + 1 ] ;
++ j ) {
int a = ids [ j ] ;
if ( t [ 5 * a + 4 ] == SF(0x00000000u) ) continue ;
F sp = hub [ 10 * a + 2 ] ;
if ( sp <= SF(0x358637bdu) ) continue ;
F cs = hub [ 10 * a ] / sp , sn = hub [ 10 * a + 1 ] / sp ;
F dn = ( x - t [ 5 * a ] ) * cs + ( y - t [ 5 * a + 1 ] ) * sn ;
F cross = - ( x - t [ 5 * a ] ) * sn + ( y - t [ 5 * a + 1 ] ) * cs ;
F vertical = z - hub [ 10 * a + 6 ] , r = wf_hypotf ( cross , vertical ) , R = t [ 5 * a + 3 ] ;
int bo = meta [ 5 * a + 2 ] , bn = meta [ 5 * a + 3 ] ;
F rhub = blades [ 4 * bo ] * R ;
if ( r <= rhub || r >= R || fabsf ( dn ) > SF(0x40800000u) * sigma ) continue ;
F speed = fmaxf ( SF(0x00000000u) , u [ p ] * cs + v [ p ] * sn ) , rpm = curve ( curves , meta [ 5 * a ] , meta [ 5 * a + 1 ] , sp , 3 ) ;
if ( rpm <= SF(0x00000000u) || speed <= SF(0x00000000u) ) continue ;
F pitch = curve ( curves , meta [ 5 * a ] , meta [ 5 * a + 1 ] , sp , 4 ) , omega = rpm * ( SF(0x40000000u) * SF(0x40490fdbu) / SF(0x42700000u) ) ;
int b = 0 ;
while ( b < bn - 2 && blades [ 4 * ( bo + b + 1 ) ] * R < r ) ++ b ;
auto bb = blades + 4 * ( bo + b ) ;
F f = clampw ( ( r / R - bb [ 0 ] ) / ( bb [ 4 ] - bb [ 0 ] ) , SF(0x00000000u) , SF(0x3f800000u) ) ;
F chord = bb [ 1 ] + f * ( bb [ 5 ] - bb [ 1 ] ) , twist = bb [ 2 ] + f * ( bb [ 6 ] - bb [ 2 ] ) ;
F B = ( F ) meta [ 5 * a + 4 ] , sol = B * chord / ( SF(0x40000000u) * SF(0x40490fdbu) * r ) , at = SF(0x00000000u) , cn = SF(0x00000000u) , ct = SF(0x00000000u) , vr2 = SF(0x00000000u) ;
bool converged = false ;
for ( int it = 0 ;
it < maxiter ;
++ it ) {
F vn = speed * SF(0x3f7ae148u) , vt = omega * r * ( SF(0x3f800000u) + at ) , psi = wf_atan2f ( vn , vt ) , s = wf_pair_trig ( psi, false ) , co = wf_pair_trig ( psi, true ) ;
F alpha = __fdiv_rn ( psi * SF(0x43340000u) , SF(0x40490fdbu) ) - twist + pitch ;
F cl0 = polar ( polars , pm , ( int ) bb [ 3 ] , alpha , 1 ) , cd0 = polar ( polars , pm , ( int ) bb [ 3 ] , alpha , 2 ) ;
F cl = cl0 + f * ( polar ( polars , pm , ( int ) bb [ 7 ] , alpha , 1 ) - cl0 ) ;
F cd = cd0 + f * ( polar ( polars , pm , ( int ) bb [ 7 ] , alpha , 2 ) - cd0 ) ;
cn = cl * co + cd * s ;
ct = cl * s - cd * co ;
vr2 = vn * vn + vt * vt ;
F wf_force = SF(0x40000000u) / SF(0x40490fdbu) * ( wf_acosf ( wf_expf ( - B * ( R - r ) / ( SF(0x40000000u) * r * s ) ) ) + wf_acosf ( wf_expf ( - B * ( r - rhub ) / ( SF(0x40000000u) * r * s ) ) ) ) ;
F den = SF(0x40800000u) * wf_force * s * co - sol * ct ;
F next = fabsf ( den ) > SF(0x322bcc77u) ? sol * ct / den : at ;
if ( ! isfinite ( next ) || fabsf ( next ) > SF(0x40000000u) ) {
break ;
}
if ( fabsf ( next - at ) < tol ) {
converged = true ;
break ;
}
at = SF(0x3f000000u) * ( next + at ) ;
}
if ( ! converged ) ++ failed ;
F gaussian = wf_expf ( - SF(0x3f000000u) * dn * dn / ( sigma * sigma ) ) / ( sqrtf ( SF(0x40000000u) * SF(0x40490fdbu) ) * sigma ) ;
F coeff = SF(0x3f000000u) * vr2 * B * chord / ( SF(0x40000000u) * SF(0x40490fdbu) * r ) * gaussian ;
F fn = coeff * cn , ft = coeff * ct ;
fx -= fn * cs + ft * ( vertical / r ) * sn ;
fy -= fn * sn - ft * ( vertical / r ) * cs ;
fz += ft * ( cross / r ) ;
}
du [ p ] = fx ;
dv [ p ] = fy ;
dw [ p ] = fz ;
failures [ p ] = failures [ p ] + (failed );
}


kernel void wind_gad(constant WindGadResources &r [[buffer(0)]], constant WindGadParams &p [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    wind_gad_body(p.nc, p.nz, p.nx, F(p.dx), F(p.dy), F(p.sigma), r.ptr, r.ids, (const device F *)r.t, r.meta, (const device F *)r.curves, (const device F *)r.blades, (const device F *)r.polars, r.pm, (const device F *)r.hub, (const device F *)r.zi, (const device F *)r.u, (const device F *)r.v, F(p.tol), p.maxiter, (device F *)r.du, (device F *)r.dv, (device F *)r.dw, r.failures, gid);
}


template < typename P_t , typename P_meta , typename P_curves , typename P_wake , typename P_hub > static inline void wind_power_body ( int nt , P_t t , P_meta meta , P_curves curves , P_wake wake , F sd , P_hub hub , uint gid ) {
int a = 0 * 1 + gid ;
if ( a >= nt ) return ;
const F angles [ 7 ] = {
- SF(0x40200000u) , - SF(0x3fc00000u) , - SF(0x3f000000u) , SF(0x00000000u) , SF(0x3f000000u) , SF(0x3fc00000u) , SF(0x40200000u) }
;
F power = SF(0x00000000u) , total = SF(0x00000000u) ;
for ( int s = 0 ;
s < 7 ;
++ s ) {
F w = sd > SF(0x00000000u) ? wf_expf ( - SF(0x3f000000u) * angles [ s ] * angles [ s ] / ( sd * sd ) ) : ( s == 3 ? SF(0x3f800000u) : SF(0x00000000u) ) ;
F u = wake [ 7 * a + s ] ;
power += w * curve ( curves , meta [ 5 * a ] , meta [ 5 * a + 1 ] , u , 2 ) * u * u * u ;
total += w ;
}
hub [ 10 * a + 7 ] = SF(0x3f000000u) * hub [ 10 * a + 4 ] * SF(0x40490fdbu) * t [ 5 * a + 3 ] * t [ 5 * a + 3 ] * power / total * t [ 5 * a + 4 ] ;
}


kernel void wind_power(constant WindPowerResources &r [[buffer(0)]], constant WindPowerParams &p [[buffer(1)]],
        uint gid [[thread_position_in_grid]]) {
    wind_power_body(p.nt, (const device F *)r.t, r.meta, (const device F *)r.curves, (const device F *)r.wake, F(p.sd), (device F *)r.hub, gid);
}
