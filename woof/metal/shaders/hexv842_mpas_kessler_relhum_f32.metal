#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
using namespace ntc;
#pragma clang fp contract(off)
#define A(a,b) tc_add((a),(b))
#define S(a,b) tc_sub((a),(b))
#define M(a,b) tc_mul((a),(b))
#define D(a,b) tc_div((a),(b))
#define MAX(a,b) tc_max((a),(b))
#define MIN(a,b) tc_min((a),(b))

// MPAS-Model-v8.4.2/src/core_atmosphere/physics/physics_wrf/module_mp_thompson.F:4597-4627
static inline F mpas_rslf(F p,F t) {
 const F coefficients[9]={.611583699E03f,.444606896E02f,.143177157E01f,
 .264224321E-1f,.299291081E-3f,.203154182E-5f,.702620698E-8f,
 .379534310E-11f,-.321582393E-13f};
 F x=MAX(-80.0f,S(t,273.16f)), es=coefficients[8];
 for(int i=7;i>=0;--i) es=A(coefficients[i],M(x,es));
 es=MIN(es,M(p,0.15f));
 return D(M(0.622f,es),S(p,es));
}
// MPAS-Model-v8.4.2/src/core_atmosphere/physics/physics_wrf/module_mp_thompson.F:4632-4658
static inline F mpas_rsif(F p,F t) {
 const F coefficients[9]={.609868993E03f,.499320233E02f,.184672631E01f,
 .402737184E-1f,.565392987E-3f,.521693933E-5f,.307839583E-7f,
 .105785160E-9f,.161444444E-12f};
 F x=MAX(-80.0f,S(t,273.16f)), es=coefficients[8];
 for(int i=7;i>=0;--i) es=A(coefficients[i],M(x,es));
 es=MIN(es,M(p,0.15f));
 return D(M(0.622f,es),S(p,es));
}
// MPAS-Model-v8.4.2/src/core_atmosphere/physics/mpas_atmphys_driver_microphysics.F:798-850
#include "hexv842_mpas_kessler_relhum_f32_bindings.metalh"
kernel void hexv842_mpas_kessler_relhum_f32(constant Hexv842MpasKesslerRelhumF32Resources &r [[buffer(0)]],constant Hexv842MpasKesslerRelhumF32Params &params [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
device F* t=(device F*)r.t;
device F* pii=(device F*)r.pii;
device F* qvp=(device F*)r.qvp;
device F* pressure_b=(device F*)r.pressure_b;
device F* pressure_p=(device F*)r.pressure_p;
device F* relhum=(device F*)r.relhum;
int n=params.n;
int nc=params.nc;
int ns=params.ns;

 int p=int(gid);
 if(p>=n || p%nc>=ns) return;
 F pressure=tc_add(pressure_b[p],pressure_p[p]);
 F temp=tc_mul(t[p],pii[p]);
 F tempc=tc_sub(temp,273.16f);
 F qvs=mpas_rslf(pressure,temp);
 if(tempc<=0.0f) qvs=mpas_rsif(pressure,temp);
 relhum[p]=tc_mul(tc_div(qvp[p],qvs),100.0f);

}
