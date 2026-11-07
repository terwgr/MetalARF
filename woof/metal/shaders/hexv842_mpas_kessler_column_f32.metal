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
#include "hexv842_mpas_kessler_column_f32_bindings.metalh"
kernel void hexv842_mpas_kessler_column_f32(constant Hexv842MpasKesslerColumnF32Resources &r [[buffer(0)]],constant Hexv842MpasKesslerColumnF32Params &params [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
device F* t=(device F*)r.t;
device F* qv=(device F*)r.qv;
device F* qc=(device F*)r.qc;
device F* qr=(device F*)r.qr;
device F* rho=(device F*)r.rho;
device F* pii=(device F*)r.pii;
device F* z=(device F*)r.z;
device F* dz8w=(device F*)r.dz8w;
device F* rainnc=(device F*)r.rainnc;
device F* rainncv=(device F*)r.rainncv;
device F* work=(device F*)r.work;
F dt=F(params.dt);
F xlv=F(params.xlv);
F cp=F(params.cp);
F ep2=F(params.ep2);
F svp1=F(params.svp1);
F svp2=F(params.svp2);
F svp3=F(params.svp3);
F svpt0=F(params.svpt0);
F rhowater=F(params.rhowater);
int nz=params.nz;
int nc=params.nc;
int ns=params.ns;

 int c=int(gid);
 if(c>=ns) return;
 // Each work plane is [level, cell]. No shared column or vertical cap.
 device F *prodk=work, *vt=work+nz*nc, *vtden=work+2*nz*nc;
 device F *rhok=work+3*nz*nc, *rdzk=work+4*nz*nc, *rdzw=work+5*nz*nc;
 F f5=D(M(M(svp2,S(svpt0,svp3)),xlv),cp);
 F crmax=0.0f;
 for(int k=0;k<nz;++k) {
  int p=k*nc+c;
  prodk[p]=qr[p]; rhok[p]=rho[p];
  F qrr=MAX(0.0f,M(M(qr[p],0.001f),rhok[p]));
  vtden[p]=tc_sqrt(D(rhok[c],rhok[p]));
  vt[p]=M(M(36.34f,tc_pow(qrr,0.1364f)),vtden[p]);
  rdzw[p]=D(1.0f,dz8w[p]);
  crmax=MAX(M(M(vt[p],dt),rdzw[p]),crmax);
 }
 for(int k=0;k<nz-1;++k) rdzk[k*nc+c]=D(1.0f,S(z[(k+1)*nc+c],z[k*nc+c]));
 int top=(nz-1)*nc+c;
 rdzk[top]=D(1.0f,S(z[top],z[top-nc]));
 // NINT is ties away from zero; the argument is positive.
 int nfall=max(1,(int)A(A(0.5f,D(crmax,0.75f)),0.5f));
 F dtfall=D(dt,(F)nfall), time_sediment=dt;
 while(nfall>0) {
  time_sediment=S(time_sediment,dtfall);
  F ppt=D(M(M(M(rhok[c],prodk[c]),vt[c]),dtfall),rhowater);
  rainncv[c]=M(ppt,1000.0f);
  rainnc[c]=A(rainnc[c],M(ppt,1000.0f));
  for(int k=0;k<nz-1;++k) {
   int p=k*nc+c, u=p+nc;
   F factor=D(M(dtfall,rdzk[p]),rhok[p]);
   prodk[p]=S(prodk[p],M(factor,S(M(M(rhok[p],prodk[p]),vt[p]),
                                 M(M(rhok[u],prodk[u]),vt[u]))));
  }
  prodk[top]=S(prodk[top],M(M(M(dtfall,rdzk[top]),prodk[top]),vt[top]));
  if(nfall>1) {
   --nfall; crmax=0.0f;
   for(int k=0;k<nz;++k) {
    int p=k*nc+c;
    F qrr=MAX(0.0f,M(M(prodk[p],0.001f),rhok[p]));
    vt[p]=M(M(36.34f,tc_pow(qrr,0.1364f)),vtden[p]);
    crmax=MAX(M(M(vt[p],time_sediment),rdzw[p]),crmax);
   }
   int nnew=max(1,(int)A(A(0.5f,D(crmax,0.75f)),0.5f));
   if(nnew!=nfall) { nfall=nnew; dtfall=D(time_sediment,(F)nfall); }
  } else { nfall=0; }
 }
 for(int k=0;k<nz;++k) {
  int p=k*nc+c;
  F factorn=D(1.0f,A(1.0f,M(M(2.2f,dt),tc_pow(MAX(0.0f,qr[p]),0.875f))));
  F qrprod=A(M(qc[p],S(1.0f,factorn)),
                 M(M(M(factorn,0.001f),dt),MAX(S(qc[p],0.001f),0.0f)));
  F rcgs=M(0.001f,rho[p]);
  qc[p]=MAX(S(qc[p],qrprod),0.0f);
  qr[p]=S(A(qr[p],prodk[p]),qr[p]);
  qr[p]=MAX(A(qr[p],qrprod),0.0f);
  F temp=M(pii[p],t[p]);
  F pressure=M(1.0e5f,tc_pow(pii[p],F(1004.0f/287.0f)));
  F gam=D(2.5e6f,M(1004.0f,pii[p]));
  F es=M(M(1000.0f,svp1),tc_exp(D(M(svp2,S(temp,svpt0)),S(temp,svp3))));
  es=MIN(es,M(0.99f,pressure));
  F qvs=D(M(ep2,es),S(pressure,es));
  F delta=S(temp,svp3);
  F prod=D(S(qv[p],qvs),A(1.0f,D(M(M(D(pressure,S(pressure,es)),qvs),f5),M(delta,delta))));
  F rq=M(rcgs,qr[p]);
  F ern=MIN(MIN(M(M(dt,D(M(A(1.6f,M(124.9f,tc_pow(rq,0.2046f))),
                                tc_pow(rq,0.525f)),
                              A(D(2.55e8f,M(pressure,qvs)),5.4e5f))),
                          D(MAX(S(qvs,qv[p]),0.0f),M(rcgs,qvs))),
                    MAX(S(-prod,qc[p]),0.0f)),qr[p]);
  F product=MAX(prod,-qc[p]);
  t[p]=A(t[p],M(gam,S(product,ern)));
  qv[p]=MAX(A(S(qv[p],product),ern),0.0f);
  qc[p]=A(qc[p],product); qr[p]=S(qr[p],ern);
 }

}
