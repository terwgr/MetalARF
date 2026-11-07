#include <metal_stdlib>
using namespace metal;
#include "hex_fct_ieee32.metalh"
#pragma clang fp contract(off)
#define Q(s,k,c,nl,nc) (((s)*(nl)+(k))*(nc)+(c))


#include "hexv841_rrfs_transport_gsl_scalar_laplacian_rrfs_bindings.metalh"
kernel void hexv841_rrfs_transport_gsl_scalar_laplacian_rrfs(constant Hexv841RrfsTransportGslScalarLaplacianRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsTransportGslScalarLaplacianRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int ns = p.ns;
    int nl = p.nl;
    int nc = p.nc;
    int ne = p.ne;
    int me = p.me;
    int cells = p.cells;
    device const int *counts = r.counts;
    device const int *edges = r.edges;
    device const int *coe = r.coe;
    device const float *sign = r.sign;
    device const float *inv_area = r.inv_area;
    device const float *dv = r.dv;
    device const float *inv_dc = r.inv_dc;
    device const float *scale2 = r.scale2;
    device const float *km = r.km;
    device const float *rho = r.rho;
    device const float *old = r.old;
    float prandtl_inv = p.prandtl_inv;
    device float *lap = r.lap;
    device float *tend = r.tend;
 
 int id=int(gid % p.dispatch_x);
 if(id>=nl*cells)return;
 int c=(int)(id%cells), k=(int)(id/cells);
 for(int j=0;j<counts[c];++j){
  int e=edges[c*me+j], c0=coe[2*e], c1=coe[2*e+1];
  float edge_sign=mpas_mul(mpas_mul(mpas_mul(inv_area[c],sign[c*me+j]),dv[e]),inv_dc[e]);
  float pr_scale=mpas_mul(prandtl_inv,scale2[e]);
  {
   float avg=mpas_mul(mpas_mul(mpas_mul(0.5f,mpas_add(km[k*nc+c0],km[k*nc+c1])),pr_scale),rho[k*ne+e]);
   for(int s=0;s<ns;++s){
    int q=Q(s,k,c,nl,nc);
    float flux=mpas_mul(edge_sign,mpas_sub(old[Q(s,k,c1,nl,nc)],old[Q(s,k,c0,nl,nc)]));
    tend[q]=mpas_add(tend[q],mpas_mul(flux,avg));
    lap[q]=mpas_add(lap[q],flux);
   }
  }
 }
}


#include "hexv841_rrfs_transport_gsl_scalar_filter4_rrfs_bindings.metalh"
kernel void hexv841_rrfs_transport_gsl_scalar_filter4_rrfs(constant Hexv841RrfsTransportGslScalarFilter4RrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsTransportGslScalarFilter4RrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int ns = p.ns;
    int nl = p.nl;
    int nc = p.nc;
    int ne = p.ne;
    int me = p.me;
    int solved = p.solved;
    device const int *counts = r.counts;
    device const int *edges = r.edges;
    device const int *coe = r.coe;
    device const float *sign = r.sign;
    device const float *inv_area = r.inv_area;
    device const float *dv = r.dv;
    device const float *inv_dc = r.inv_dc;
    device const float *scale4 = r.scale4;
    device const float *rho = r.rho;
    device const float *lap = r.lap;
    float filter4 = p.filter4;
    device float *tend = r.tend;
 int id=int(gid % p.dispatch_x);
 if(id>=nl*solved)return;
 int c=(int)(id%solved), k=(int)(id/solved);
 float area=mpas_mul(filter4,inv_area[c]);
 for(int j=0;j<counts[c];++j){
  int e=edges[c*me+j], c0=coe[2*e], c1=coe[2*e+1];
  float edge_sign=mpas_mul(mpas_mul(mpas_mul(mpas_mul(scale4[e],area),dv[e]),sign[c*me+j]),inv_dc[e]);
  {
   
   
   area=mpas_mul(mpas_mul(filter4,inv_area[c]),rho[(nl-1)*ne+e]);
   for(int s=0;s<ns;++s){
    int q=Q(s,k,c,nl,nc);
    tend[q]=mpas_sub(tend[q],mpas_mul(mpas_mul(edge_sign,rho[k*ne+e]),
       mpas_sub(lap[Q(s,k,c1,nl,nc)],lap[Q(s,k,c0,nl,nc)])));
   }
  }
 }
}



#include "hexv841_rrfs_transport_scalar_edge_rrfs_bindings.metalh"
kernel void hexv841_rrfs_transport_scalar_edge_rrfs(constant Hexv841RrfsTransportScalarEdgeRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsTransportScalarEdgeRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int ns = p.ns;
    int nl = p.nl;
    int nc = p.nc;
    int ne = p.ne;
    int width = p.width;
    int regional = p.regional;
    const int edge_start = p.edge_start;
    const int edge_end = p.edge_end;
    device const int *mask = r.mask;
    device const int *coe = r.coe;
    device const float *dv = r.dv;
    device const float *stage = r.stage;
    device const float *uh = r.uh;
    device const int *counts = r.counts;
    device const int *cells = r.cells;
    device const float *coef = r.coef;
    device const float *coef3 = r.coef3;
    device float *edge = r.edge;
 
 
 int id=int(gid % p.dispatch_x);if(id>=nl*ne)return;
 int e=(int)(id%ne);if(e<edge_start || e>=edge_end)return;
 if(regional && mask[e]>5)return;
 {int k=(int)(id/ne);float velocity=uh[k*ne+e];float direction=mpas_copysign(1.0f,velocity);
  for(int t=0;t<ns;++t){float value=0.0f;
   if(!regional || mask[e]<4){
    for(int j=0;j<counts[e];++j){int c=cells[j*ne+e];
     float weight=mpas_add(coef[j*ne+e],mpas_mul(direction,coef3[j*ne+e]));
     float term=mpas_mul(weight,stage[Q(t,k,c,nl,nc)]);
     value=(counts[e]==10 && j==0)?term:mpas_add(value,term);}
   }else if(mask[e]<=5){
    float ud=mpas_copysign(.5f,velocity);
    float positive=mpas_mul(dv[e],mpas_abs(mpas_add(ud,.5f)));
    float negative=mpas_mul(dv[e],mpas_abs(mpas_sub(ud,.5f)));
    value=mpas_add(mpas_mul(positive,stage[Q(t,k,coe[2*e],nl,nc)]),mpas_mul(negative,stage[Q(t,k,coe[2*e+1],nl,nc)]));
   }
   edge[(t*nl+k)*ne+e]=value;
  }
 }
}

 float scalar_vertical_rrfs(int s,int k,int c,int nl,int nc,float coef,
 device const float *q,device const float *ww,device const float *fnm,device const float *fnp){
 if(k==0 || k==nl)return 0.0f;
 float u=ww[k*nc+c];float qi=q[Q(s,k,c,nl,nc)],qm=q[Q(s,k-1,c,nl,nc)];
 if(k==1 || k==nl-1)return mpas_mul(u,mpas_add(mpas_mul(fnm[k],qi),mpas_mul(fnp[k],qm)));
 float qp=q[Q(s,k+1,c,nl,nc)],qmm=q[Q(s,k-2,c,nl,nc)];
 float centered=mpas_div(mpas_mul(u,mpas_sub(mpas_mul(7.0f,mpas_add(qi,qm)),mpas_add(qp,qmm))),12.0f);
 float up=mpas_div(mpas_mul(mpas_mul(coef,mpas_abs(u)),mpas_sub(mpas_sub(qp,qmm),mpas_mul(3.0f,mpas_sub(qi,qm)))),12.0f);
 return mpas_add(centered,up);
}

#include "hexv841_rrfs_transport_scalar_finish_rrfs_bindings.metalh"
kernel void hexv841_rrfs_transport_scalar_finish_rrfs(constant Hexv841RrfsTransportScalarFinishRrfsResources &r [[buffer(0)]],
    constant Hexv841RrfsTransportScalarFinishRrfsParams &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.dispatch_x * p.dispatch_levels * p.dispatch_species)) return;
    int ns = p.ns;
    int nl = p.nl;
    int nc = p.nc;
    int ne = p.ne;
    int me = p.me;
    int solved = p.solved;
    device const int *mask = r.mask;
    device const int *counts = r.counts;
    device const int *edges = r.edges;
    device const float *sign = r.sign;
    device const float *inv_area = r.inv_area;
    device const float *uh = r.uh;
    device const float *edge = r.edge;
    device const float *ww = r.ww;
    device const float *fnm = r.fnm;
    device const float *fnp = r.fnp;
    device const float *rdzw = r.rdzw;
    device const float *old = r.old;
    device const float *stage = r.stage;
    device const float *rho0 = r.rho0;
    device const float *rho1 = r.rho1;
    device const float *source = r.source;
    float dt = p.dt;
    float weight = p.weight;
    float coef = p.coef;
    device float *out = r.out;
    device float *target = r.target;
 
 int id=int(gid % p.dispatch_x);if(id>=nl*nc)return;
 int c=(int)(id%nc);
 float old_weight=mpas_sub(1.0f,weight);
 {int k=(int)(id/nc);float density=mpas_add(mpas_mul(old_weight,rho0[k*nc+c]),mpas_mul(weight,rho1[k*nc+c]));
  target[k*nc+c]=density;
  for(int t=0;t<ns;++t){int q=Q(t,k,c,nl,nc);out[q]=stage[q];
   if(c>=solved || mask[c]>5)continue;
   float tend=0.0f;
   for(int j=0;j<counts[c];++j){int e=edges[j*nc+c];
    tend=mpas_sub(tend,mpas_mul(mpas_mul(sign[j*nc+c],uh[k*ne+e]),edge[(t*nl+k)*ne+e]));}
   tend=mpas_add(mpas_mul(tend,inv_area[c]),source[q]);
   float lower=scalar_vertical_rrfs(t,k,c,nl,nc,coef,stage,ww,fnm,fnp);
   float upper=scalar_vertical_rrfs(t,k+1,c,nl,nc,coef,stage,ww,fnm,fnp);
   float update=mpas_add(mpas_mul(old[q],rho0[k*nc+c]),mpas_mul(dt,mpas_sub(tend,mpas_mul(rdzw[k],mpas_sub(upper,lower)))));
   out[q]=mpas_mul(update,mpas_div(1.0f,density));
  }
 }
}
