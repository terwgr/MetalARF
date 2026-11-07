// Generated from classic RRTM GASABS band 3; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "wrf_exact_rrtm_lw_taumol_03_bindings.metalh"
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

kernel void wrf_exact_rrtm_lw_taumol_03(
    constant WrfExactRrtmLwTaumol03Resources &r [[buffer(0)]],
    constant WrfExactRrtmLwTaumol03Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col = int(tid) / p.nlay;
    const int lay = int(tid) - col * p.nlay;
    const ulong i = ulong(tid), outbase = i * 140ul + 22ul;
    const int jp = r.jp[i], jt = r.jt[i], jt1 = r.jt1[i];
    const Exact32 a = reinterpret_cast<device const Exact32*>(r.colh2o)[i], b = reinterpret_cast<device const Exact32*>(r.colco2)[i];
    const Exact32 fp = reinterpret_cast<device const Exact32*>(r.fac01)[i] + reinterpret_cast<device const Exact32*>(r.fac11)[i];
    const bool trop = lay + 1 <= r.laytrop[col];
    Exact32 comb, fs; int js;
    rrtm_lw_spec(a, b, 1.19268f, 8.0f, comb, js, fs);
    int jsl = js; Exact32 fsl = fs;
    if (js == 8) {
        jsl = fs >= 0.9f ? 9 : js;
        fsl = fs >= 0.9f ? 10.0f * (fs - 0.9f) : fs / 0.9f;
    }
    const int ns = jsl + rrtm_trunc(fsl + 0.5f);
    Exact32 eta = 1.0f - rrtm_lw_table_value(
        reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_etaref, ns - 1, 0);
    if (eta == 0.0f) eta = 1.0f;
    const Exact32 h1 = rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables), r.table_index,
                                         p.id_h2oref, jp - 1, 0);
    const Exact32 h2 = rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables), r.table_index,
                                         p.id_h2oref, jp, 0);
    const Exact32 c1 = 1.19268f * rrtm_lw_table_value(
        reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_co2ref, jp - 1, 0) / eta;
    const Exact32 c2 = 1.19268f * rrtm_lw_table_value(
        reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_co2ref, jp, 0) / eta;
    const Exact32 w1 = ns == 10 ? h1 : c1;
    const Exact32 w2 = ns == 10 ? h2 : c2;
    const Exact32 n2omult = reinterpret_cast<device const Exact32*>(r.coln2o)[i] - comb * rrtm_lw_blob_ratio(
        reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_n2oref, jp, fp, w1, w2);
    const int i0 = ((jp - 1) * 5 + (jt - 1)) * p.nspa3 + jsl - 1;
    const int i1 = (jp * 5 + (jt1 - 1)) * p.nspa3 + jsl - 1;
    const Exact32 f00 = reinterpret_cast<device const Exact32*>(r.fac00)[i], f01 = reinterpret_cast<device const Exact32*>(r.fac01)[i];
    const Exact32 f10 = reinterpret_cast<device const Exact32*>(r.fac10)[i], f11 = reinterpret_cast<device const Exact32*>(r.fac11)[i];
    Exact32 combu, fsu; int jsu;
    rrtm_lw_spec(a, b, 1.19268f, 4.0f, combu, jsu, fsu);
    const int nsu = jsu + rrtm_trunc(fsu + 0.5f);
    Exact32 etau = 1.0f - rrtm_lw_table_value(
        reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_etaref, nsu - 1, 0);
    if (etau == 0.0f) etau = 1.0f;
    const Exact32 hu1 = rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables), r.table_index,
        p.id_h2oref, jp - 1, 0);
    const Exact32 hu2 = rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables), r.table_index,
        p.id_h2oref, jp, 0);
    const Exact32 cu1 = 1.19268f * rrtm_lw_table_value(
        reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_co2ref, jp - 1, 0) / etau;
    const Exact32 cu2 = 1.19268f * rrtm_lw_table_value(
        reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_co2ref, jp, 0) / etau;
    const Exact32 n2omu = reinterpret_cast<device const Exact32*>(r.coln2o)[i] - combu * rrtm_lw_blob_ratio(
        reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_n2oref, jp, fp, nsu == 5 ? hu1 : cu1,
        nsu == 5 ? hu2 : cu2);
    const int hi0u = ((jp - 13) * 5 + (jt - 1)) * p.nspb3 + jsu - 1;
    const int hi1u = ((jp - 12) * 5 + (jt1 - 1)) * p.nspb3 + jsu - 1;
    const int3 meta_a = rrtm_lw_table_info(r.table_index, p.id_absa);
    const int3 meta_b = rrtm_lw_table_info(r.table_index, p.id_absb);
    for (int g = 0; g < 16; ++g) {
    const Exact32 self0 = rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables), r.table_index,
        p.id_self3, r.indself[i] - 1,g);
    const Exact32 self1 = rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables), r.table_index,
        p.id_self3, r.indself[i],g);
    const Exact32 continuum = reinterpret_cast<device const Exact32*>(r.selffac)[i]
        * (self0 + reinterpret_cast<device const Exact32*>(r.selffrac)[i] * (self1 - self0));
        const Exact32 low = comb * rrtm_lw_interp8(
            reinterpret_cast<device const Exact32*>(r.tables) + meta_a.x, p.nrowa3, 16, g, i0, i1, 10,
            fsl, f00, f01, f10, f11)
          + a * (continuum + reinterpret_cast<device const Exact32*>(r.forfac)[i] * rrtm_lw_table_value(
              reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_forref, 0, g))
          + n2omult * rrtm_lw_table_value(
              reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_absn2oa, 0, g);
        const Exact32 high = combu * rrtm_lw_interp8(
            reinterpret_cast<device const Exact32*>(r.tables) + meta_b.x, p.nrowb3, 16, g, hi0u, hi1u, 5, fsu,
            f00, f01, f10, f11)
          + (a * reinterpret_cast<device const Exact32*>(r.forfac)[i]) * rrtm_lw_table_value(
              reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_forref, 0, g)
          + n2omu * rrtm_lw_table_value(
              reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_absn2ob, 0, g);
        const Exact32 pl = rrtm_lw_table_value(
            reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_fraca, jsl - 1, g)
          + fsl * (rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables), r.table_index,
              p.id_fraca, jsl, g) - rrtm_lw_table_value(
              reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_fraca, jsl - 1, g));
        const Exact32 ph = rrtm_lw_table_value(
            reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_fracb, jsu - 1, g)
          + fsu * (rrtm_lw_table_value(reinterpret_cast<device const Exact32*>(r.tables), r.table_index,
              p.id_fracb, jsu, g) - rrtm_lw_table_value(
              reinterpret_cast<device const Exact32*>(r.tables), r.table_index, p.id_fracb, jsu - 1, g));
        reinterpret_cast<device Exact32*>(r.taug)[outbase + ulong(g)] = trop ? low : high;
        reinterpret_cast<device Exact32*>(r.pfrac)[outbase + ulong(g)] = trop ? pl : ph;
    }
    (void)meta_a;
}
