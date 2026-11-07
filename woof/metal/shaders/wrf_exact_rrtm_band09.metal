// Generated from classic RRTM GASABS band 9; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_rrtm_lw_taumol_09_bindings.metalh"
using namespace metal;
static inline bool operator==(Exact32 a, Exact32 b) { return soft32_equal(a.bits,b.bits); }
static inline bool operator<=(Exact32 a, Exact32 b) { return soft32_greater_equal(b.bits,a.bits); }
static inline int rrtm_trunc(Exact32 x) { return int(trunc(as_type<float>(x.bits))); }
static inline Exact32 rrtm_floor(Exact32 x) { return Exact32(floor(as_type<float>(x.bits))); }
inline Exact32 rrtm_lw_interp4(device const Exact32 *tab, int nrow, int ng,
                              int g, int i0, int i1,
                              Exact32 f00, Exact32 f01, Exact32 f10, Exact32 f11) {
    i0 = clamp(i0, 0, nrow - 1); i1 = clamp(i1, 0, nrow - 1);
    const int i0n = clamp(i0 + 1, 0, nrow - 1);
    const int i1n = clamp(i1 + 1, 0, nrow - 1);
    return f00 * tab[ulong(i0) * ulong(ng) + ulong(g)]
         + f10 * tab[ulong(i0n) * ulong(ng) + ulong(g)]
         + f01 * tab[ulong(i1) * ulong(ng) + ulong(g)]
         + f11 * tab[ulong(i1n) * ulong(ng) + ulong(g)];
}

inline Exact32 rrtm_lw_interp8(device const Exact32 *tab, int nrow, int ng,
                              int g, int i0, int i1, int step,
                              Exact32 fs, Exact32 f00, Exact32 f01,
                              Exact32 f10, Exact32 f11) {
    const Exact32 f000 = (1.0f - fs) * f00;
    const Exact32 f100 = fs * f00;
    const Exact32 f010 = (1.0f - fs) * f10;
    const Exact32 f110 = fs * f10;
    const Exact32 f001 = (1.0f - fs) * f01;
    const Exact32 f101 = fs * f01;
    const Exact32 f011 = (1.0f - fs) * f11;
    const Exact32 f111 = fs * f11;
    i0 = clamp(i0, 0, nrow - 1); i1 = clamp(i1, 0, nrow - 1);
    const int a0 = clamp(i0 + 1, 0, nrow - 1);
    const int a1 = clamp(i0 + step, 0, nrow - 1);
    const int a2 = clamp(i0 + step + 1, 0, nrow - 1);
    const int b0 = clamp(i1 + 1, 0, nrow - 1);
    const int b1 = clamp(i1 + step, 0, nrow - 1);
    const int b2 = clamp(i1 + step + 1, 0, nrow - 1);
    return f000 * tab[ulong(i0) * ulong(ng) + ulong(g)]
         + f100 * tab[ulong(a0) * ulong(ng) + ulong(g)]
         + f010 * tab[ulong(a1) * ulong(ng) + ulong(g)]
         + f110 * tab[ulong(a2) * ulong(ng) + ulong(g)]
         + f001 * tab[ulong(i1) * ulong(ng) + ulong(g)]
         + f101 * tab[ulong(b0) * ulong(ng) + ulong(g)]
         + f011 * tab[ulong(b1) * ulong(ng) + ulong(g)]
         + f111 * tab[ulong(b2) * ulong(ng) + ulong(g)];
}

inline void rrtm_lw_spec(Exact32 a, Exact32 b, Exact32 ratio, Exact32 nsub,
                         thread Exact32 &comb, thread int &js,
                         thread Exact32 &fs) {
    comb = a + ratio * b;
    Exact32 parm = a / comb;
    parm = parm >= 0.999999f ? 0.999999f : parm;
    const Exact32 mult = nsub * parm;
    js = 1 + rrtm_trunc(mult);
    fs = mult - rrtm_floor(mult);
}

inline Exact32 rrtm_lw_ref_ratio(device const Exact32 *ref,
                               int nref, int jp, Exact32 fp,
                               Exact32 d0, Exact32 d1) {
    const int j0 = clamp(jp - 1, 0, nref - 1);
    const int j1 = clamp(jp, 0, nref - 1);
    const Exact32 a = ref[j0] / d0;
    const Exact32 b = ref[j1] / d1;
    return a + fp * (b - a);
}

inline int3 rrtm_lw_table_info(device const int *index, int table_id) {
    const ulong i = ulong(table_id) * 3ul;
    return int3(index[i], index[i + 1ul], index[i + 2ul]);
}

inline Exact32 rrtm_lw_table_value(device const Exact32 *values,
                                  device const int *index,
                                  int table_id, int row, int col) {
    const int3 meta = rrtm_lw_table_info(index, table_id);
    const int r = clamp(row, 0, meta.y - 1);
    const int c = clamp(col, 0, meta.z - 1);
    return values[ulong(meta.x) + ulong(r) * ulong(meta.z) + ulong(c)];
}

inline Exact32 rrtm_lw_blob_ratio(device const Exact32 *values,
                                 device const int *index, int table_id,
                                 int jp, Exact32 fp, Exact32 d0, Exact32 d1) {
    const Exact32 a = rrtm_lw_table_value(values, index, table_id, jp - 1, 0) / d0;
    const Exact32 b = rrtm_lw_table_value(values, index, table_id, jp, 0) / d1;
    return a + fp * (b - a);
}

kernel void wrf_exact_rrtm_lw_taumol_09(
    constant WrfExactRrtmLwTaumol09Resources &r [[buffer(0)]],
    constant WrfExactRrtmLwTaumol09Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncell))return;
    const int col=int(tid)/p.nlay,lay=int(tid)-col*p.nlay;
    const ulong i=ulong(tid),outbase=i*140ul+96ul;
    const int jp=r.jp[i],jt=r.jt[i],jt1=r.jt1[i];
    const bool trop=lay+1<=r.laytrop[col];
    Exact32 comb=0.0f,fs=0.0f;int js=1;
    rrtm_lw_spec(reinterpret_cast<device const Exact32*>(r.colh2o)[i],reinterpret_cast<device const Exact32*>(r.colch4)[i],21.6282f,8.0f,comb,js,fs);
    int jfrac=js;Exact32 ffrac=fs;
    int jsl=js;Exact32 fsl=fs;
    if(js==8){
      if(fs<=0.68f) fsl=fs/0.68f;
      else if(fs<=0.92f){jsl=9;fsl=(fs-0.68f)/0.24f;}
      else {jsl=10;fsl=(fs-0.92f)/0.08f;}
    } else if(js==9){jsl=10;fsl=1.0f;jfrac=8;ffrac=1.0f;}
    const int ns=jsl+rrtm_trunc(fsl+0.5f);
    Exact32 eta=1.0f-rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_eta,ns-1,0);
    if(eta==0.0f)eta=1.0f;
    const Exact32 h0=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_h2oref,jp-1,0);
    const Exact32 h1=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_h2oref,jp,0);
    const Exact32 c0=21.6282f*rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_ch4ref,jp-1,0)/eta;
    const Exact32 c1=21.6282f*rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_ch4ref,jp,0)/eta;
    const Exact32 den0=ns==11?h0:c0,den1=ns==11?h1:c1;
    const Exact32 nref0=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_n2oref,jp-1,0);
    const Exact32 nref1=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_n2oref,jp,0);
    const Exact32 nratio=nref0/den0;
    const Exact32 nratio1=nref1/den1;
    const Exact32 n2omult=reinterpret_cast<device const Exact32*>(r.coln2o)[i]-comb*(nratio+(reinterpret_cast<device const Exact32*>(r.fac01)[i]+reinterpret_cast<device const Exact32*>(r.fac11)[i])*(nratio1-nratio));
    int ioff=0;
    if(r.laylow[col]>=1 && lay+1>=r.laylow[col])ioff=p.ng;
    if(r.layswtch[col]>=1 && lay+1>=r.layswtch[col] && (r.laylow[col]<1 || lay+1<r.laylow[col] || r.layswtch[col]>=r.laylow[col]))ioff=2*p.ng;
    const int row=ioff/p.ng;
    const int3 ma=rrtm_lw_table_info(r.table_index,p.id_absa);
    const int3 mb=rrtm_lw_table_info(r.table_index,p.id_absb);
    const int lo0=((jp-1)*5+(jt-1))*p.nspa+jsl-1;
    const int lo1=(jp*5+(jt1-1))*p.nspa+jsl-1;
    const int hi0=((jp-13)*5+(jt-1))*p.nspb;
    const int hi1=((jp-12)*5+(jt1-1))*p.nspb;
    for(int g=0;g<p.ng;++g){
    const Exact32 s0=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_self,r.indself[i]-1,g);
    const Exact32 s1=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_self,r.indself[i],g);
    const Exact32 continuum=(s0+reinterpret_cast<device const Exact32*>(r.selffrac)[i]*(s1-s0));
      const Exact32 low=comb*rrtm_lw_interp8(reinterpret_cast<device const Exact32*>(r.tables)+ma.x,p.nrowa,p.ng,g,lo0,lo1,11,fsl,
        reinterpret_cast<device const Exact32*>(r.fac00)[i],reinterpret_cast<device const Exact32*>(r.fac01)[i],reinterpret_cast<device const Exact32*>(r.fac10)[i],reinterpret_cast<device const Exact32*>(r.fac11)[i])
        +(reinterpret_cast<device const Exact32*>(r.colh2o)[i]*reinterpret_cast<device const Exact32*>(r.selffac)[i])*continuum+n2omult*rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_absn2o,row,g);
      const Exact32 high=reinterpret_cast<device const Exact32*>(r.colch4)[i]*rrtm_lw_interp4(reinterpret_cast<device const Exact32*>(r.tables)+mb.x,p.nrowb,p.ng,g,hi0,hi1,
        reinterpret_cast<device const Exact32*>(r.fac00)[i],reinterpret_cast<device const Exact32*>(r.fac01)[i],reinterpret_cast<device const Exact32*>(r.fac10)[i],reinterpret_cast<device const Exact32*>(r.fac11)[i]);
      const Exact32 pl0=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_fraca,jfrac-1,g);
      const Exact32 pl1=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_fraca,jfrac,g);
      reinterpret_cast<device Exact32*>(r.taug)[outbase+ulong(g)]=trop?low:high;
      reinterpret_cast<device Exact32*>(r.pfrac)[outbase+ulong(g)]=trop?pl0+ffrac*(pl1-pl0):
        rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_fracb,0,g);
    }
}
