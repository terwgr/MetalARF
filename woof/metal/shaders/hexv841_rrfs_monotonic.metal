#include <metal_stdlib>
using namespace metal;
#include "hex_fct_ieee32.metalh"
#pragma clang fp contract(off)
#define C(k,c) ((k)*nc+(c))
#define E(k,e) ((k)*ne+(e))
#define S(k,j,c) (((k)*2+(j))*nc+(c))
#define Q(s,k,c) (((s)*nl+(k))*nc+(c))

#include "hexv841_rrfs_monotonic_mono_source_rrfs_bindings.metalh"
kernel void hexv841_rrfs_monotonic_mono_source_rrfs(constant Hexv841RrfsMonotonicMonoSourceRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsMonotonicMonoSourceRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int ns = p.ns;
    int nl = p.nl;
    int nc = p.nc;
    int lo = p.lo;
    int hi = p.hi;
    float dt = p.dt;
    device const float *rho = r.rho;
    device float *old = r.old;
    device float *tend = r.tend;
 int c=int(gid % p.dispatch_x);if(c<lo||c>=hi)return;
 int k=int((gid / p.dispatch_x) % p.dispatch_levels),s=int(gid / (p.dispatch_x * p.dispatch_levels));{int q=Q(s,k,c);
 old[q]=mpas_add(old[q],mpas_div(mpas_mul(dt,tend[q]),rho[C(k,c)]));tend[q]=0.0f;}
}

#include "hexv841_rrfs_monotonic_mono_density_rrfs_bindings.metalh"
kernel void hexv841_rrfs_monotonic_mono_density_rrfs(constant Hexv841RrfsMonotonicMonoDensityRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsMonotonicMonoDensityRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int nl = p.nl;
    int nc = p.nc;
    int ne = p.ne;
    int me = p.me;
    int lo = p.lo;
    int hi = p.hi;
    float dt = p.dt;
    device const int *counts = r.counts;
    device const int *edges = r.edges;
    device const float *sign = r.sign;
    device const float *dv = r.dv;
    device const float *area = r.area;
    device const float *uh = r.uh;
    device const float *ww = r.ww;
    device const float *rd = r.rd;
    device const float *rho = r.rho;
    device float *out = r.out;
 int c=int(gid % p.dispatch_x);if(c<lo||c>=hi)return;
 int k=int((gid / p.dispatch_x) % p.dispatch_levels);{float v=0.0f;
 for(int j=0;j<counts[c];++j){int e=edges[j*nc+c];v=mpas_sub(v,mpas_mul(mpas_mul(mpas_mul(sign[j*nc+c],uh[E(k,e)]),dv[e]),area[c]));}
 out[C(k,c)]=mpas_add(rho[C(k,c)],mpas_mul(dt,mpas_sub(v,mpas_mul(rd[k],mpas_sub(ww[C(k+1,c)],ww[C(k,c)])))));}
}

#include "hexv841_rrfs_monotonic_mono_carriers_rrfs_bindings.metalh"
kernel void hexv841_rrfs_monotonic_mono_carriers_rrfs(constant Hexv841RrfsMonotonicMonoCarriersRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsMonotonicMonoCarriersRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int s = p.s;
    int nl = p.nl;
    int nc = p.nc;
    int lo = p.lo;
    int hi = p.hi;
    int dummy = p.dummy;
    device const float *old = r.old;
    device const float *stage = r.stage;
    device float *o = r.o;
    device float *n = r.n;
 s+=int(gid / (p.dispatch_x * p.dispatch_levels));o+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;n+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;
 int c=int(gid % p.dispatch_x);if(c>=nc)return;
 int k=int((gid / p.dispatch_x) % p.dispatch_levels);
 if(c==dummy){o[C(k,c)]=0.0f;n[C(k,c)]=0.0f;}
 else if(c>=lo&&c<hi){o[C(k,c)]=old[Q(s,k,c)];n[C(k,c)]=stage[Q(s,k,c)];}
}

#include "hexv841_rrfs_monotonic_mono_vertical_rrfs_bindings.metalh"
kernel void hexv841_rrfs_monotonic_mono_vertical_rrfs(constant Hexv841RrfsMonotonicMonoVerticalRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsMonotonicMonoVerticalRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int nl = p.nl;
    int nc = p.nc;
    int me = p.me;
    int lo = p.lo;
    int hi = p.hi;
    float coef = p.coef;
    device const int *counts = r.counts;
    device const int *neighbors = r.neighbors;
    device const float *o = r.o;
    device const float *n = r.n;
    device const float *ww = r.ww;
    device const float *fnm = r.fnm;
    device const float *fnp = r.fnp;
    device float *mn = r.mn;
    device float *mx = r.mx;
    device float *w = r.w;
 o+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;n+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;mn+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;mx+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;w+=int(gid / (p.dispatch_x * p.dispatch_levels))*(nl+1)*nc;
 int c=int(gid % p.dispatch_x);if(c<lo||c>=hi)return;
 if(int((gid / p.dispatch_x) % p.dispatch_levels)==0){w[C(0,c)]=0.0f;w[C(nl,c)]=0.0f;}
 int k=int((gid / p.dispatch_x) % p.dispatch_levels);{float lower=o[C(k? k-1:1,c)],current=o[C(k,c)];
 float maxv=fct_max(lower,current),minv=fct_min(lower,current);
 if(k>0&&k<nl-1){maxv=fct_max(maxv,o[C(k+1,c)]);minv=fct_min(minv,o[C(k+1,c)]);}
 for(int j=0;j<counts[c];++j){float v=o[C(k,neighbors[c*me+j])];maxv=fct_max(maxv,v);minv=fct_min(minv,v);}
 mx[C(k,c)]=maxv;mn[C(k,c)]=minv;
 if(k==0)return;
 float qi=n[C(k,c)],qm=n[C(k-1,c)],u=ww[C(k,c)];
 if(k==1||k==nl-1)w[C(k,c)]=mpas_mul(u,mpas_add(mpas_mul(fnm[k],qi),mpas_mul(fnp[k],qm)));
 else {float qp=n[C(k+1,c)],qmm=n[C(k-2,c)];
 float centered=mpas_div(mpas_mul(u,mpas_sub(mpas_mul(7.0f,mpas_add(qi,qm)),mpas_add(qp,qmm))),12.0f);
 float up=mpas_div(mpas_mul(mpas_mul(coef,mpas_abs(u)),mpas_sub(mpas_sub(qp,qmm),mpas_mul(3.0f,mpas_sub(qi,qm)))),12.0f);
 w[C(k,c)]=mpas_add(centered,up);}}
}

#include "hexv841_rrfs_monotonic_mono_high_edge_rrfs_bindings.metalh"
kernel void hexv841_rrfs_monotonic_mono_high_edge_rrfs(constant Hexv841RrfsMonotonicMonoHighEdgeRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsMonotonicMonoHighEdgeRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int nl = p.nl;
    int nc = p.nc;
    int ne = p.ne;
    int width = p.width;
    int lo = p.lo;
    int hi = p.hi;
    int solved = p.solved;
    device const int *coe = r.coe;
    device const int *counts = r.counts;
    device const int *cells = r.cells;
    device const float *coef = r.coef;
    device const float *coef3 = r.coef3;
    device const float *uh = r.uh;
    device const float *q = r.q;
    device float *flux = r.flux;
 q+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;flux+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*ne;
 int e=int(gid % p.dispatch_x);if(e<lo||e>=hi)return;
 int k=int((gid / p.dispatch_x) % p.dispatch_levels);{float v=0.0f,u=uh[E(k,e)];
 if(coe[2*e]<solved||coe[2*e+1]<solved){
 for(int j=0;j<counts[e];++j){float weight=mpas_add(coef[j*ne+e],mpas_mul(mpas_copysign(1.0f,u),coef3[j*ne+e]));
 if(counts[e]==10){float term=mpas_mul(weight,q[C(k,cells[j*ne+e])]);v=j?mpas_add(v,term):term;}
 else v=mpas_add(v,mpas_mul(mpas_mul(u,weight),q[C(k,cells[j*ne+e])])) ;}
 if(counts[e]==10)v=mpas_mul(u,v);}
 flux[E(k,e)]=v;}
}

#include "hexv841_rrfs_monotonic_mono_low_vertical_rrfs_bindings.metalh"
kernel void hexv841_rrfs_monotonic_mono_low_vertical_rrfs(constant Hexv841RrfsMonotonicMonoLowVerticalRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsMonotonicMonoLowVerticalRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int nl = p.nl;
    int nc = p.nc;
    int lo = p.lo;
    int hi = p.hi;
    float dt = p.dt;
    device const float *o = r.o;
    device const float *rho = r.rho;
    device const float *ww = r.ww;
    device const float *rd = r.rd;
    device float *n = r.n;
    device float *w = r.w;
    device float *scale = r.scale;
 o+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;n+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;w+=int(gid / (p.dispatch_x * p.dispatch_levels))*(nl+1)*nc;scale+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*2*nc;
 int c=int(gid % p.dispatch_x);if(c<lo||c>=hi)return;
 for(int k=0;k<nl;++k){float mass=mpas_mul(o[C(k,c)],rho[C(k,c)]);
 if(k<nl-1){float u=ww[C(k+1,c)];float f=mpas_mul(dt,mpas_add(mpas_mul(fct_max(0.0f,u),o[C(k,c)]),mpas_mul(fct_min(0.0f,u),o[C(k+1,c)])));mass=mpas_sub(mass,mpas_mul(f,rd[k]));}
 if(k>0){float u=ww[C(k,c)];float f=mpas_mul(dt,mpas_add(mpas_mul(fct_max(0.0f,u),o[C(k-1,c)]),mpas_mul(fct_min(0.0f,u),o[C(k,c)])));mass=mpas_add(mass,mpas_mul(f,rd[k]));w[C(k,c)]=mpas_sub(mpas_mul(dt,w[C(k,c)]),f);}
 n[C(k,c)]=mass;}
 for(int k=0;k<nl;++k){scale[S(k,0,c)]=mpas_mul(-rd[k],mpas_sub(fct_min(0.0f,w[C(k+1,c)]),fct_max(0.0f,w[C(k,c)])));
 scale[S(k,1,c)]=mpas_mul(-rd[k],mpas_sub(fct_max(0.0f,w[C(k+1,c)]),fct_min(0.0f,w[C(k,c)])));}
}

#include "hexv841_rrfs_monotonic_mono_low_edge_rrfs_bindings.metalh"
kernel void hexv841_rrfs_monotonic_mono_low_edge_rrfs(constant Hexv841RrfsMonotonicMonoLowEdgeRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsMonotonicMonoLowEdgeRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int nl = p.nl;
    int nc = p.nc;
    int ne = p.ne;
    int lo = p.lo;
    int hi = p.hi;
    int regional = p.regional;
    float dt = p.dt;
    device const int *coe = r.coe;
    device const int *mask = r.mask;
    device const float *dv = r.dv;
    device const float *uh = r.uh;
    device const float *o = r.o;
    device float *flux = r.flux;
    device float *upwind = r.upwind;
    device float *residual = r.residual;
 o+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;flux+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*ne;upwind+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*ne;residual+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*ne;
 int e=int(gid % p.dispatch_x);if(e<lo||e>=hi)return;
 int k=int((gid / p.dispatch_x) % p.dispatch_levels);{float u=uh[E(k,e)];float f=mpas_mul(mpas_mul(dv[e],dt),mpas_add(mpas_mul(fct_max(0.0f,u),o[C(k,coe[2*e])]),mpas_mul(fct_min(0.0f,u),o[C(k,coe[2*e+1])])));
 upwind[E(k,e)]=f;residual[E(k,e)]=mpas_sub(mpas_mul(dt,flux[E(k,e)]),f);
 if((regional&&mask[e]==5)||mask[e]==4){residual[E(k,e)]=0.0f;flux[E(k,e)]=f;}}
}

#include "hexv841_rrfs_monotonic_mono_scale_rrfs_bindings.metalh"
kernel void hexv841_rrfs_monotonic_mono_scale_rrfs(constant Hexv841RrfsMonotonicMonoScaleRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsMonotonicMonoScaleRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int nl = p.nl;
    int nc = p.nc;
    int ne = p.ne;
    int me = p.me;
    int lo = p.lo;
    int hi = p.hi;
    device const int *counts = r.counts;
    device const int *edges = r.edges;
    device const float *sign = r.sign;
    device const float *area = r.area;
    device const float *upwind = r.upwind;
    device const float *residual = r.residual;
    device const float *mn = r.mn;
    device const float *mx = r.mx;
    device const float *rho = r.rho;
    device float *n = r.n;
    device float *scale = r.scale;
 upwind+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*ne;residual+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*ne;mn+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;mx+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;n+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;scale+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*2*nc;
 int c=int(gid % p.dispatch_x);if(c<lo||c>=hi)return;
 int k=int((gid / p.dispatch_x) % p.dispatch_levels);{float mass=n[C(k,c)],si=scale[S(k,0,c)],so=scale[S(k,1,c)];
 for(int j=0;j<counts[c];++j){int e=edges[j*nc+c];float sg=sign[j*nc+c];
 mass=mpas_sub(mass,mpas_mul(mpas_mul(sg,upwind[E(k,e)]),area[c]));
 so=mpas_sub(so,mpas_mul(fct_max(0.0f,mpas_mul(sg,residual[E(k,e)])),area[c]));
 si=mpas_sub(si,mpas_mul(fct_min(0.0f,mpas_mul(sg,residual[E(k,e)])),area[c]));}
 n[C(k,c)]=mass;
 scale[S(k,0,c)]=fct_min(1.0f,fct_max(0.0f,mpas_div(mpas_sub(mpas_mul(mx[C(k,c)],rho[C(k,c)]),mass),mpas_add(si,1.e-20f))));
 scale[S(k,1,c)]=fct_min(1.0f,fct_max(0.0f,mpas_div(mpas_sub(mpas_mul(mn[C(k,c)],rho[C(k,c)]),mass),mpas_sub(so,1.e-20f))));}
}

#include "hexv841_rrfs_monotonic_mono_limit_edge_rrfs_bindings.metalh"
kernel void hexv841_rrfs_monotonic_mono_limit_edge_rrfs(constant Hexv841RrfsMonotonicMonoLimitEdgeRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsMonotonicMonoLimitEdgeRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int nl = p.nl;
    int nc = p.nc;
    int ne = p.ne;
    int lo = p.lo;
    int hi = p.hi;
    int solved = p.solved;
    int regional = p.regional;
    float dt = p.dt;
    device const int *coe = r.coe;
    device const int *mask = r.mask;
    device const float *dv = r.dv;
    device const float *uh = r.uh;
    device const float *o = r.o;
    device const float *scale = r.scale;
    device float *flux = r.flux;
 o+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;scale+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*2*nc;flux+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*ne;
 int e=int(gid % p.dispatch_x);if(e<lo||e>=hi)return;int c0=coe[2*e],c1=coe[2*e+1];
 if(c0>=solved&&c1>=solved)return;
 int k=int((gid / p.dispatch_x) % p.dispatch_levels);{float u=uh[E(k,e)];float f=mpas_mul(mpas_mul(dv[e],dt),mpas_add(mpas_mul(fct_max(0.0f,u),o[C(k,c0)]),mpas_mul(fct_min(0.0f,u),o[C(k,c1)])));
 float v=mpas_sub(mpas_mul(dt,flux[E(k,e)]),f);if((regional&&mask[e]==5)||mask[e]==4)v=0.0f;
 flux[E(k,e)]=mpas_add(mpas_mul(fct_max(0.0f,v),fct_min(scale[S(k,1,c0)],scale[S(k,0,c1)])),mpas_mul(fct_min(0.0f,v),fct_min(scale[S(k,0,c0)],scale[S(k,1,c1)])));}
}

#include "hexv841_rrfs_monotonic_mono_finish_rrfs_bindings.metalh"
kernel void hexv841_rrfs_monotonic_mono_finish_rrfs(constant Hexv841RrfsMonotonicMonoFinishRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsMonotonicMonoFinishRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int s = p.s;
    int nl = p.nl;
    int nc = p.nc;
    int ne = p.ne;
    int me = p.me;
    int lo = p.lo;
    int hi = p.hi;
    int copylo = p.copylo;
    int copyhi = p.copyhi;
    device const int *counts = r.counts;
    device const int *edges = r.edges;
    device const int *mask = r.mask;
    device const float *sign = r.sign;
    device const float *area = r.area;
    device const float *rd = r.rd;
    device const float *rho = r.rho;
    device const float *scale = r.scale;
    device const float *flux = r.flux;
    device float *w = r.w;
    device float *n = r.n;
    device float *stage = r.stage;
 s+=int(gid / (p.dispatch_x * p.dispatch_levels));scale+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*2*nc;flux+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*ne;w+=int(gid / (p.dispatch_x * p.dispatch_levels))*(nl+1)*nc;n+=int(gid / (p.dispatch_x * p.dispatch_levels))*nl*nc;
 int c=int(gid % p.dispatch_x);if(c>=nc)return;
 if(c>=lo&&c<hi){for(int k=1;k<nl;++k){float v=w[C(k,c)];w[C(k,c)]=mpas_add(mpas_mul(fct_max(0.0f,v),fct_min(scale[S(k-1,1,c)],scale[S(k,0,c)])),mpas_mul(fct_min(0.0f,v),fct_min(scale[S(k,1,c)],scale[S(k-1,0,c)])));}
 for(int k=0;k<nl;++k){float mass=n[C(k,c)];for(int j=0;j<counts[c];++j){int e=edges[j*nc+c];mass=mpas_sub(mass,mpas_mul(mpas_mul(sign[j*nc+c],flux[E(k,e)]),area[c]));}
 n[C(k,c)]=mpas_div(mpas_add(mass,mpas_mul(-rd[k],mpas_sub(w[C(k+1,c)],w[C(k,c)]))),rho[C(k,c)]);}}
 if(c>=copylo&&c<copyhi&&mask[c]<=2)for(int k=0;k<nl;++k)stage[Q(s,k,c)]=fct_max(0.0f,n[C(k,c)]);
}
