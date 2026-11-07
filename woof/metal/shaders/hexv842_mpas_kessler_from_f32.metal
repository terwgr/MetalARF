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
#include "hexv842_mpas_kessler_from_f32_bindings.metalh"
kernel void hexv842_mpas_kessler_from_f32(constant Hexv842MpasKesslerFromF32Resources &r [[buffer(0)]],constant Hexv842MpasKesslerFromF32Params &params [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
device F* theta_m=(device F*)r.theta_m;
device F* qv=(device F*)r.qv;
device F* qc=(device F*)r.qc;
device F* qr=(device F*)r.qr;
device F* rho_zz=(device F*)r.rho_zz;
device F* zz=(device F*)r.zz;
device F* exner=(device F*)r.exner;
device F* zgrid=(device F*)r.zgrid;
device F* t=(device F*)r.t;
device F* qvp=(device F*)r.qvp;
device F* qcp=(device F*)r.qcp;
device F* qrp=(device F*)r.qrp;
device F* rho=(device F*)r.rho;
device F* pii=(device F*)r.pii;
device F* z=(device F*)r.z;
device F* dz=(device F*)r.dz;
F rvord=F(params.rvord);
int nz=params.nz;
int nc=params.nc;
int ns=params.ns;

 int p=int(gid);
 if(p>=nz*nc || p%nc>=ns) return;
 qvp[p]=qv[p]; qcp[p]=qc[p]; qrp[p]=qr[p];
 rho[p]=M(zz[p],rho_zz[p]);
 t[p]=D(theta_m[p],A(1.0f,M(rvord,MAX(0.0f,qvp[p]))));
 pii[p]=exner[p]; z[p]=zgrid[p]; dz[p]=S(zgrid[p+nc],zgrid[p]);

}
