#include <metal_stdlib>
#include "soft_binary32.metalh"
using namespace metal;
struct Resources {
 device const uint *plev [[id(0)]];
 device uint *wkl [[id(1)]];
 device uint *wx [[id(2)]];
 device uint *coldry [[id(3)]];
 device uint *wbrodl [[id(4)]];
 device uint *pwvcm [[id(5)]];
};
struct Params { uint ncol, nl; float grav, avogad; };
#define AD(a,b) soft32_add((a),(b))
#define SU(a,b) soft32_sub((a),(b))
#define MU(a,b) soft32_mul((a),(b))
#define DV(a,b) soft32_div((a),(b))
#define C(x) as_type<uint>(x##f)
kernel void wrf_exact_rrtmg_lw_inatm(constant Resources& r [[buffer(0)]],
                            constant Params& p [[buffer(1)]],
                            uint col [[thread_position_in_grid]]) {
 if (col >= p.ncol) return;
 uint nl=p.nl, base=col*nl, wb=col*38u*nl, xb=col*4u*nl, pb=col*(nl+1u);
 uint amttl=C(0.0), wvttl=C(0.0);
 for(uint l=0;l<nl;++l) {
  uint w1=r.wkl[wb+l];
  uint amm=AD(MU(SU(C(1.0),w1),C(28.9660)),MU(w1,C(18.0160)));
  uint num=MU(MU(SU(r.plev[pb+l],r.plev[pb+l+1u]),C(1.e3)),as_type<uint>(p.avogad));
  uint den=MU(MU(MU(C(1.e2),as_type<uint>(p.grav)),amm),AD(C(1.0),w1));
  r.coldry[base+l]=DV(num,den);
 }
 for(uint l=0;l<nl;++l) {
  uint cd=r.coldry[base+l], summol=C(0.0);
  for(uint imol=1;imol<7;++imol)summol=AD(summol,r.wkl[wb+imol*nl+l]);
  r.wbrodl[base+l]=MU(cd,SU(C(1.0),summol));
  for(uint imol=0;imol<7;++imol) { uint j=wb+imol*nl+l; r.wkl[j]=MU(cd,r.wkl[j]); }
  amttl=AD(AD(amttl,cd),r.wkl[wb+l]);
  wvttl=AD(wvttl,r.wkl[wb+l]);
  for(uint ix=0;ix<4;++ix) { uint j=xb+ix*nl+l; r.wx[j]=MU(MU(cd,r.wx[j]),C(1.e-20)); }
 }
 uint wvsh=DV(MU(C(18.0160),wvttl),MU(C(28.9660),amttl));
 r.pwvcm[col]=DV(MU(wvsh,MU(C(1.e3),r.plev[pb])),MU(C(1.e2),as_type<uint>(p.grav)));
}
