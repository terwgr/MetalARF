#include <metal_stdlib>
using namespace metal;
#include "cumulus_support.metalh"
#include "hexv842_mpas_surface_diagnostics_f32_bindings.metalh"
#pragma clang fp contract(off)
static inline float native_diag_mul(float a,float b) {return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b)));}
static inline float native_diag_sub(float a,float b) {return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b)));}
struct MM5Constants { float cp; float g; float rovcp; float r; float xlv; float svp1; float svp2; float svp3; float svpt0; float ep1; float ep2; float karman; float p1000mb; };
// MPAS mpas_atmphys_sfc_diagnostics.F:79-91,101-116 and
// physics_wrf/module_sf_sfcdiags.F:47-64: GNU scalar arithmetic preserves
// an input NaN's payload and quiets signaling NaNs. CUDA arithmetic instead
// canonicalizes NaNs. Retain the source operand word; finite arithmetic
// continues to use the shared rounded primitives, without a tolerance.
static inline float mm5_diag_quiet(float a) {
return as_type<float>(as_type<uint>(a) | 0x00400000u);
}
// MPAS physics_wrf/module_sf_sfcdiags.F:47-64 scalar product semantics.
static inline float mm5_diag_mul(float a, float b) {
if(a!=a) return mm5_diag_quiet(a);
if(b!=b) return mm5_diag_quiet(b);
return native_diag_mul(a,b);
}
// MPAS physics_wrf/module_sf_sfcdiags.F:47-64 scalar division semantics.
static inline float mm5_diag_div(float a, float b) {
if(a!=a) return mm5_diag_quiet(a);
if(b!=b) return mm5_diag_quiet(b);
return cu_div(a,b);
}
// MPAS physics_wrf/module_sf_sfcdiags.F:47-64 scalar subtraction semantics.
static inline float mm5_diag_sub(float a, float b) {
if(a!=a) return mm5_diag_quiet(a);
if(b!=b) return mm5_diag_quiet(b);
return native_diag_sub(a,b);
}
// MPAS mpas_atmphys_sfc_diagnostics.F:27-123 atmphys_sfc_diagnostics
// The default LSM arm performs no stores, exactly as MPAS's SELECT CASE.

kernel void hexv842_mpas_surface_diagnostics_f32(
 constant Hexv842MpasSurfaceDiagnosticsF32Resources &r [[buffer(0)]],
 constant Hexv842MpasSurfaceDiagnosticsF32Params &p [[buffer(1)]],
 uint i [[thread_position_in_grid]]) {

if(i>=uint(p.n) || p.lsm==0) return;
int n=p.n,lsm=p.lsm; float xice_threshold=p.threshold;
MM5Constants c;
c.cp=r.constants[0];
c.g=r.constants[1];
c.rovcp=r.constants[2];
c.r=r.constants[3];
c.xlv=r.constants[4];
c.svp1=r.constants[5];
c.svp2=r.constants[6];
c.svp3=r.constants[7];
c.svpt0=r.constants[8];
c.ep1=r.constants[9];
c.ep2=r.constants[10];
c.karman=r.constants[11];
c.p1000mb=r.constants[12];
float psfcpa=r.a[40*n+i];
float tsk=r.a[41*n+i];
float xland=r.a[44*n+i];
float qsfc=r.a[6*n+i];
float qfx=r.a[5*n+i];
float hfx=r.a[4*n+i];
float cqs2=r.a[21*n+i];
float chs2=r.a[20*n+i];
float rho=mm5_diag_div(psfcpa,mm5_diag_mul(c.r,tsk));
float q2,t2;
if(lsm==1 || (soft32_less(0u,as_type<uint>(mm5_diag_sub(xland,1.5f))) || (soft32_less_equal(as_type<uint>(mm5_diag_sub(xland,1.5f)),0u) && soft32_less_equal(as_type<uint>(xice_threshold),as_type<uint>(r.xice[i]))))) {
q2=soft32_less(as_type<uint>(cqs2),as_type<uint>(1.e-5f)) ? qsfc : mm5_diag_sub(qsfc,mm5_diag_div(qfx,mm5_diag_mul(rho,cqs2)));
t2=soft32_less(as_type<uint>(chs2),as_type<uint>(1.e-5f)) ? tsk : mm5_diag_sub(tsk,mm5_diag_div(hfx,mm5_diag_mul(mm5_diag_mul(rho,c.cp),chs2)));
} else { q2=r.q2mxy[i]; t2=r.t2mxy[i]; }
r.a[18*n+i]=q2;
r.a[17*n+i]=t2;
r.a[16*n+i]=mm5_diag_mul(t2,gfk_pow(mm5_diag_div(c.p1000mb,psfcpa),c.rovcp));

}
