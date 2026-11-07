// Generated from classic RRTM GASABS bands 12–16; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_rrtm_lw_taumol_12_16_bindings.metalh"
using namespace metal;
static inline bool operator==(Exact32 a, Exact32 b) { return soft32_equal(a.bits,b.bits); }
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

kernel void wrf_exact_rrtm_lw_taumol_12_16(
    constant WrfExactRrtmLwTaumol1216Resources &r [[buffer(0)]],
    constant WrfExactRrtmLwTaumol1216Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col=int(tid)/p.nlay, lay=int(tid)-col*p.nlay;
    const ulong i=ulong(tid), outbase=i*140ul+ulong(p.offset);
    const int jp=r.jp[i], jt=r.jt[i], jt1=r.jt1[i];
    const bool trop=lay+1<=r.laytrop[col];
    const int3 ma=rrtm_lw_table_info(r.table_index,p.id_absa);
    const int3 mb=rrtm_lw_table_info(r.table_index,p.id_absb);
    const Exact32 ga = p.gas_a==1 ? reinterpret_cast<device const Exact32*>(r.colh2o)[i] : p.gas_a==2 ? reinterpret_cast<device const Exact32*>(r.colco2)[i]
                   : p.gas_a==3 ? reinterpret_cast<device const Exact32*>(r.coln2o)[i] : reinterpret_cast<device const Exact32*>(r.colch4)[i];
    const Exact32 gb = p.gas_b==1 ? reinterpret_cast<device const Exact32*>(r.colh2o)[i] : p.gas_b==2 ? reinterpret_cast<device const Exact32*>(r.colco2)[i]
                   : p.gas_b==3 ? reinterpret_cast<device const Exact32*>(r.coln2o)[i] : reinterpret_cast<device const Exact32*>(r.colch4)[i];
    const int lo0=((jp-1)*5+(jt-1))*p.nspa;
    const int lo1=(jp*5+(jt1-1))*p.nspa;
    const int hi0=((jp-13)*5+(jt-1))*p.nspb;
    const int hi1=((jp-12)*5+(jt1-1))*p.nspb;
    Exact32 comb=0.0f, fs=0.0f; int js=1;
    if(p.mode==0) rrtm_lw_spec(ga,gb,Exact32(p.strrat),8.0f,comb,js,fs);
    for(int g=0;g<p.ng;++g){
    Exact32 continuum=0.0f;
    if(p.have_self!=0){
      const Exact32 s0=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_self,r.indself[i]-1,g);
      const Exact32 s1=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_self,r.indself[i],g);
      continuum=(s0+reinterpret_cast<device const Exact32*>(r.selffrac)[i]*(s1-s0));
    }
      Exact32 low=0.0f, high=0.0f, pl=0.0f, ph=0.0f;
      if(p.mode==0){
        low=comb*rrtm_lw_interp8(reinterpret_cast<device const Exact32*>(r.tables)+ma.x,p.nrowa,p.ng,g,lo0+js-1,
          lo1+js-1,9,fs,reinterpret_cast<device const Exact32*>(r.fac00)[i],reinterpret_cast<device const Exact32*>(r.fac01)[i],reinterpret_cast<device const Exact32*>(r.fac10)[i],reinterpret_cast<device const Exact32*>(r.fac11)[i])
          +(reinterpret_cast<device const Exact32*>(r.colh2o)[i]*reinterpret_cast<device const Exact32*>(r.selffac)[i])*continuum;
        pl=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_fraca,js-1,g)
          +fs*(rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_fraca,js,g)
             -rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_fraca,js-1,g));
      } else {
        low=ga*(rrtm_lw_interp4(reinterpret_cast<device const Exact32*>(r.tables)+ma.x,p.nrowa,p.ng,g,lo0,lo1,
          reinterpret_cast<device const Exact32*>(r.fac00)[i],reinterpret_cast<device const Exact32*>(r.fac01)[i],reinterpret_cast<device const Exact32*>(r.fac10)[i],reinterpret_cast<device const Exact32*>(r.fac11)[i])+reinterpret_cast<device const Exact32*>(r.selffac)[i]*continuum);
        high=ga*rrtm_lw_interp4(reinterpret_cast<device const Exact32*>(r.tables)+mb.x,p.nrowb,p.ng,g,hi0,hi1,
          reinterpret_cast<device const Exact32*>(r.fac00)[i],reinterpret_cast<device const Exact32*>(r.fac01)[i],reinterpret_cast<device const Exact32*>(r.fac10)[i],reinterpret_cast<device const Exact32*>(r.fac11)[i]);
        pl=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_fraca,0,g);
        ph=rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables),r.table_index,p.id_fracb,0,g);
      }
      reinterpret_cast<device Exact32*>(r.taug)[outbase+ulong(g)]=p.mode==0 ? (trop?low:0.0f):(trop?low:high);
      reinterpret_cast<device Exact32*>(r.pfrac)[outbase+ulong(g)]=p.mode==0 ? (trop?pl:0.0f):(trop?pl:ph);
    }
}
