#include <metal_stdlib>
#include "rrtm_lw_pack_top_down_bindings.metalh"
#include "rrtm_lw_t8w_bindings.metalh"
#include "rrtm_lw_mm5atm_bindings.metalh"
#include "rrtm_lw_setcoef_bindings.metalh"
#include "rrtm_lw_taumol_01_02_bindings.metalh"
#include "rrtm_lw_taumol_03_bindings.metalh"
#include "rrtm_lw_taumol_06_bindings.metalh"
#include "rrtm_lw_taumol_04_05_bindings.metalh"
#include "rrtm_lw_taumol_07_bindings.metalh"
#include "rrtm_lw_taumol_08_bindings.metalh"
#include "rrtm_lw_taumol_09_bindings.metalh"
#include "rrtm_lw_taumol_10_11_bindings.metalh"
#include "rrtm_lw_taumol_12_16_bindings.metalh"
#include "rrtm_lw_make_itr_bindings.metalh"
#include "rrtm_lw_rtrn_down_bindings.metalh"
#include "rrtm_lw_rtrn_up_bindings.metalh"
#include "rrtm_lw_rtrn_fluxes_bindings.metalh"

using namespace metal;

#pragma clang fp contract(off)

inline float rrtm_lw_planck(device const float *totplnk,
                            device const float *delwave,
                            float temperature, int band) {
    const int whole = int(trunc(temperature));
    const int raw = int(trunc(temperature - 159.0f));
    const int row = clamp(raw, 1, 180) - 1;
    const float frac = temperature - float(whole);
    const ulong lo = ulong(row) * 16ul + ulong(band);
    const float lower = totplnk[lo];
    const float upper = totplnk[lo + 16ul];
    return delwave[band] * (lower + frac * (upper - lower));
}

kernel void rrtm_lw_make_itr(
    constant RrtmLwMakeItrResources &r [[buffer(0)]],
    constant RrtmLwMakeItrParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.npoint))return;
    const float od=1.66f*r.taug[tid];
    float tf=od/(p.bpade+od);
    if(od<=0.0f)tf=0.0f;
    r.itr[tid]=clamp(int(trunc(5000.0f*tf+0.5f)),0,5000);
}

inline float rrtm_lw_o3_layer(device const float *o3wrk,
                              device const float *ppwrkh,
                              float bottom, float top) {
    float weighted = 0.0f;
    for (int i = 0; i < 31; ++i) {
        const float pb1 = max(bottom - ppwrkh[i], 0.0f);
        const float pb2 = max(bottom - ppwrkh[i + 1], 0.0f);
        const float pt1 = max(top - ppwrkh[i], 0.0f);
        const float pt2 = max(top - ppwrkh[i + 1], 0.0f);
        weighted += (pb2 - pb1 - pt2 + pt1) * o3wrk[i];
    }
    return weighted / (bottom - top);
}

inline float rrtm_lw_standard_temperature(device const float *pprof,
                                          device const float *tprof,
                                          int nprof, float pressure) {
    int above = 0;
    for (int i = 0; i < nprof; ++i) above += pprof[i] >= pressure;
    int klev = clamp(above, 1, nprof);
    if (!(pressure > pprof[nprof - 1])) klev = nprof;
    const bool last = klev == nprof;
    const int k0 = clamp(klev - 1, 0, nprof - 1);
    const int k1 = clamp(klev, 0, nprof - 1);
    const float vark = tprof[k0];
    const float vark1 = last ? vark : tprof[k1];
    const float span = last ? 1.0f : pprof[k1] - pprof[k0];
    const float weight = last ? 0.0f : (pressure - pprof[k0]) / span;
    return weight * (vark1 - vark) + vark;
}

kernel void rrtm_lw_pack_top_down(
    constant RrtmLwPackTopDownResources &r [[buffer(0)]],
    constant RrtmLwPackTopDownParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= uint(p.n)) return;
    const uint col = index / uint(p.nlev);
    const uint lev = index - col * uint(p.nlev);
    const uint y = col / uint(p.nx), x = col - y * uint(p.nx);
    r.output[index] = r.input[((uint(p.nlev) - 1u - lev) * uint(p.ny) + y)
                             * uint(p.nx) + x];
}

kernel void rrtm_lw_t8w(
    constant RrtmLwT8WResources &r [[buffer(0)]],
    constant RrtmLwT8WParams &p [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= uint(p.ncell)) return;
    const int col = int(index) / p.nlev;
    const int lev = int(index) - col * p.nlev;
    const ulong layer_base = ulong(col) * ulong(p.nz);
    const ulong level_base = ulong(col) * ulong(p.nlev);
    if (lev > 0 && lev < p.nz) {
        const int j = p.nz - lev;
        r.tw[level_base + ulong(lev)] = r.fnm[j] * r.t[layer_base + ulong(lev - 1)]
            + r.fnp[j] * r.t[layer_base + ulong(lev)];
    } else if (lev == p.nz) {
        const float z0 = 0.5f * (r.z_interface[level_base + ulong(p.nz)]
                                + r.z_interface[level_base + ulong(p.nz - 1)]);
        const float z1 = 0.5f * (r.z_interface[level_base + ulong(p.nz - 1)]
                                + r.z_interface[level_base + ulong(p.nz - 2)]);
        const float w = (r.z_interface[level_base + ulong(p.nz)] - z1)
                      / (z0 - z1);
        r.tw[level_base + ulong(lev)] = w * r.t[layer_base + ulong(p.nz - 1)]
                                      + (1.0f - w) * r.t[layer_base + ulong(p.nz - 2)];
    } else {
        const float z0 = 0.5f * (r.z_interface[level_base]
                                + r.z_interface[level_base + 1ul]);
        const float z1 = 0.5f * (r.z_interface[level_base + 1ul]
                                + r.z_interface[level_base + 2ul]);
        const float w = (r.z_interface[level_base] - z1) / (z0 - z1);
        r.tw[level_base] = w * r.t[layer_base]
                         + (1.0f - w) * r.t[layer_base + 1ul];
    }
}

kernel void rrtm_lw_mm5atm(
    constant RrtmLwMm5AtmResources &r [[buffer(0)]],
    constant RrtmLwMm5AtmParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong c = ulong(col);
    const ulong nlay = ulong(p.nlay);
    const ulong nz = ulong(p.nz);
    const ulong levbase = c * (nlay + 1ul);
    const ulong laybase = c * nlay;
    const ulong inlay = c * nz;
    const ulong inlev = c * (nz + 1ul);
    for (int lev = 0; lev <= p.nz; ++lev) {
        r.pz[levbase + ulong(lev)] = r.pw_pa[inlev + ulong(p.nz - lev)] * 0.01f;
        r.tz[levbase + ulong(lev)] = r.tw[inlev + ulong(p.nz - lev)];
    }
    for (int lay = 0; lay < p.nz; ++lay) {
        const ulong dst = laybase + ulong(lay);
        const ulong src = inlay + ulong(p.nz - 1 - lay);
        r.pavel[dst] = r.p_pa[src] * 0.01f;
        r.tavel[dst] = r.t[src];
    }
    for (int lev = p.nz + 1; lev < p.nlay; ++lev) {
        const float pressure = r.pz[levbase + ulong(lev - 1)] - 4.0f;
        r.pz[levbase + ulong(lev)] = pressure;
        r.pavel[laybase + ulong(lev - 1)] = 0.5f
            * (pressure + r.pz[levbase + ulong(lev - 1)]);
    }
    r.pz[levbase + nlay] = 0.0f;
    r.pavel[laybase + nlay - 1ul] = 0.5f
        * r.pz[levbase + nlay - 1ul];
    if (p.nlay > p.nz) {
        const float offset = r.tz[levbase + nz]
            - rrtm_lw_standard_temperature(
                r.pprof, r.tprof, p.nprof, r.pz[levbase + nz]);
        for (int lev = p.nz + 1; lev <= p.nlay; ++lev) {
            const float standard = rrtm_lw_standard_temperature(
                r.pprof, r.tprof, p.nprof, r.pz[levbase + ulong(lev)]);
            r.tz[levbase + ulong(lev)] = standard + offset;
            r.tavel[laybase + ulong(lev - 1)] = 0.5f
                * (r.tz[levbase + ulong(lev)]
                   + r.tz[levbase + ulong(lev - 1)]);
        }
    }
    const float tbound = r.tsfc[c] > 340.0f ? 339.99f : r.tsfc[c];
    r.tbound[c] = tbound;
    for (int band = 0; band < 16; ++band)
        r.semiss[c * 16ul + ulong(band)] = r.emiss[c];

    const float grav = p.g * 100.0f;
    for (int lay = 0; lay < p.nlay; ++lay) {
        const ulong dst = laybase + ulong(lay);
        float o3;
        if (p.nlay == p.nz) {
            o3 = rrtm_lw_o3_layer(r.o3wrk, r.ppwrkh,
                r.pz[levbase + ulong(lay)], r.pz[levbase + ulong(lay + 1)]);
        } else if (lay == p.nlay - 1) {
            o3 = 6.135e-6f;
        } else {
            o3 = rrtm_lw_o3_layer(r.o3wrk, r.ppwrkh,
                r.pz[levbase + ulong(lay + 1)],
                r.pz[levbase + ulong(lay + 2)]);
        }
        const ulong src = inlay + ulong(p.nz - 1 - min(lay, p.nz - 1));
        const float q = lay < p.nz ? max(r.qv[src], 1.0e-12f) : 5.0e-6f;
        const float co2 = p.co2vmr;
        const float n2o = p.n2ovmr;
        const float ch4 = p.ch4vmr;
        const float amm = (1.0f - q * 1.607758f) * 28.9644f
                        + q * 1.607758f * 18.0154f;
        const float coldry = (r.pz[levbase + ulong(lay)]
                             - r.pz[levbase + ulong(lay + 1)])
                           * 1.0e3f * 6.022e23f
                           / (grav * amm * (1.0f + q * 1.607758f));
        r.coldry[dst] = coldry;
        float cloud_tau = 0.0f;
        if (lay < p.nz) {
            const ulong src = inlay + ulong(p.nz - 1 - lay);
            const float temp = r.t[src];
            const float rho = (r.p_pa[src] * 0.01f) / (p.r * temp) * 100.0f;
            const float clwp = rho * max(r.qc[src], 0.0f) * r.dz[src] * 1000.0f;
            const float ciwp = rho * max(r.qi[src], 0.0f) * r.dz[src] * 1000.0f;
            const float plwp = pow(max(rho * max(r.qr[src], 0.0f), 0.0f), 0.75f)
                               * r.dz[src] * 1000.0f;
            const float piwp = pow(max(rho * max(r.qs[src], 0.0f), 0.0f), 0.75f)
                               * r.dz[src] * 1000.0f;
            cloud_tau = 0.144f * clwp + 0.0735f * ciwp
                      + 0.000330f * plwp + 0.00234f * piwp;
            r.taucloud[dst] = cloud_tau;
            r.cldfra_out[dst] = cloud_tau > 0.01f ? 1.0f : r.cldfra[src];
        } else {
            r.taucloud[dst] = 0.0f;
            r.cldfra_out[dst] = 0.0f;
        }
        for (int m = 0; m < 35; ++m) r.wkl[dst * 35ul + ulong(m)] = 0.0f;
        for (int x = 0; x < 4; ++x) r.wx[dst * 4ul + ulong(x)] = 0.0f;
        r.wkl[dst * 35ul] = q * 1.607758f * coldry;
        r.wkl[dst * 35ul + 1ul] = co2 * coldry;
        r.wkl[dst * 35ul + 2ul] = o3 * 0.603461f * coldry;
        r.wkl[dst * 35ul + 3ul] = n2o * coldry;
        r.wkl[dst * 35ul + 5ul] = ch4 * coldry;
    }
}

kernel void rrtm_lw_setcoef(
    constant RrtmLwSetcoefResources &r [[buffer(0)]],
    constant RrtmLwSetcoefParams &p [[buffer(1)]],
    uint col [[thread_position_in_grid]]) {
    if (col >= uint(p.ncol)) return;
    const ulong base = ulong(col) * ulong(p.nlay);
    int laytrop = 0, layswtch = 0, laylow = 0;
    for (int lay = 0; lay < p.nlay; ++lay) {
        const ulong i = base + ulong(lay);
        const float plog = log(r.pavel[i]);
        const int jp = clamp(int(trunc(36.0f - 5.0f * (plog + 0.04f))), 1, 58);
        const float fp = 5.0f * (r.preflog[jp - 1] - plog);
        const float tdiff = r.tavel[i] - r.tref[jp - 1];
        const int jt = clamp(int(trunc(3.0f + tdiff / 15.0f)), 1, 4);
        const float ft = tdiff / 15.0f - float(jt - 3);
        const float tdiff1 = r.tavel[i] - r.tref[jp];
        const int jt1 = clamp(int(trunc(3.0f + tdiff1 / 15.0f)), 1, 4);
        const float ft1 = tdiff1 / 15.0f - float(jt1 - 3);
        const float water = r.wkl[i * 35ul] / r.coldry[i];
        const float scalefac = r.pavel[i] * (296.0f / 1013.0f) / r.tavel[i];
        const float forfac = scalefac / (1.0f + water);
        const float selffac = water * forfac;
        const float factor = (r.tavel[i] - 188.0f) / 7.2f;
        const int indself = clamp(int(trunc(factor)) - 7, 1, 9);
        const float selffrac = factor - float(indself + 7);
        r.jp[i] = jp; r.jt[i] = jt; r.jt1[i] = jt1;
        r.indself[i] = indself;
        r.fac00[i] = (1.0f - fp) * (1.0f - ft);
        r.fac01[i] = fp * (1.0f - ft1);
        r.fac10[i] = (1.0f - fp) * ft;
        r.fac11[i] = fp * ft1;
        r.forfac[i] = forfac;
        r.selffac[i] = selffac;
        r.selffrac[i] = selffrac;
        if (plog > 4.56f) ++laytrop;
        if (plog > 5.76f) ++layswtch;
        if (plog >= 6.62f) ++laylow;
    }
    if (laylow == 0) laylow = 1;
    const int probe = clamp(layswtch, 0, p.nlay - 1);
    if (r.jp[base + ulong(probe)] <= 6) ++layswtch;
    r.laytrop[col] = laytrop;
    r.layswtch[col] = layswtch;
    r.laylow[col] = laylow;
    for (int lay = 0; lay < p.nlay; ++lay) {
        const ulong i = base + ulong(lay);
        const float coldry = r.coldry[i];
        const float floor = 1.0e-32f * coldry;
        r.colh2o[i] = 1.0e-20f * r.wkl[i * 35ul];
        r.colco2[i] = 1.0e-20f * r.wkl[i * 35ul + 1ul];
        r.colo3[i] = 1.0e-20f * r.wkl[i * 35ul + 2ul];
        r.coln2o[i] = 1.0e-20f * r.wkl[i * 35ul + 3ul];
        r.colch4[i] = 1.0e-20f * r.wkl[i * 35ul + 5ul];
        r.colo2[i] = 1.0e-20f * r.wkl[i * 35ul + 6ul];
        if (r.colco2[i] == 0.0f) r.colco2[i] = floor;
        if (r.coln2o[i] == 0.0f) r.coln2o[i] = floor;
        if (r.colch4[i] == 0.0f) r.colch4[i] = floor;
        const float co2reg = 3.55e-24f * coldry;
        r.co2mult[i] = ((r.colco2[i] - co2reg) * 272.63f
            * exp(-1919.4f / r.tavel[i]))
            / (8.7604e-4f * r.tavel[i]);
    }
}

inline float rrtm_lw_interp4(device const float *tab, int nrow, int ng,
                              int g, int i0, int i1,
                              float f00, float f01, float f10, float f11) {
    i0 = clamp(i0, 0, nrow - 1); i1 = clamp(i1, 0, nrow - 1);
    const int i0n = clamp(i0 + 1, 0, nrow - 1);
    const int i1n = clamp(i1 + 1, 0, nrow - 1);
    return f00 * tab[ulong(i0) * ulong(ng) + ulong(g)]
         + f10 * tab[ulong(i0n) * ulong(ng) + ulong(g)]
         + f01 * tab[ulong(i1) * ulong(ng) + ulong(g)]
         + f11 * tab[ulong(i1n) * ulong(ng) + ulong(g)];
}

inline float rrtm_lw_interp8(device const float *tab, int nrow, int ng,
                              int g, int i0, int i1, int step,
                              float fs, float f00, float f01,
                              float f10, float f11) {
    const float f000 = (1.0f - fs) * f00;
    const float f100 = (1.0f - fs) * f10;
    const float f010 = fs * f00;
    const float f110 = fs * f10;
    const float f001 = (1.0f - fs) * f01;
    const float f101 = (1.0f - fs) * f11;
    const float f011 = fs * f01;
    const float f111 = fs * f11;
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

inline void rrtm_lw_spec(float a, float b, float ratio, float nsub,
                         thread float &comb, thread int &js,
                         thread float &fs) {
    comb = a + ratio * b;
    float parm = a / comb;
    parm = parm >= 0.999999f ? 0.999999f : parm;
    const float mult = nsub * parm;
    js = 1 + int(trunc(mult));
    fs = mult - floor(mult);
}

inline float rrtm_lw_ref_ratio(device const float *ref,
                               int nref, int jp, float fp,
                               float d0, float d1) {
    const int j0 = clamp(jp - 1, 0, nref - 1);
    const int j1 = clamp(jp, 0, nref - 1);
    const float a = ref[j0] / d0;
    const float b = ref[j1] / d1;
    return a + fp * (b - a);
}

inline int3 rrtm_lw_table_info(device const int *index, int table_id) {
    const ulong i = ulong(table_id) * 3ul;
    return int3(index[i], index[i + 1ul], index[i + 2ul]);
}

inline float rrtm_lw_table_value(device const float *values,
                                  device const int *index,
                                  int table_id, int row, int col) {
    const int3 meta = rrtm_lw_table_info(index, table_id);
    const int r = clamp(row, 0, meta.y - 1);
    const int c = clamp(col, 0, meta.z - 1);
    return values[ulong(meta.x) + ulong(r) * ulong(meta.z) + ulong(c)];
}

inline float rrtm_lw_blob_ratio(device const float *values,
                                 device const int *index, int table_id,
                                 int jp, float fp, float d0, float d1) {
    const float a = rrtm_lw_table_value(values, index, table_id, jp - 1, 0) / d0;
    const float b = rrtm_lw_table_value(values, index, table_id, jp, 0) / d1;
    return a + fp * (b - a);
}

kernel void rrtm_lw_taumol_01_02(
    constant RrtmLwTaumol0102Resources &r [[buffer(0)]],
    constant RrtmLwTaumol0102Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col = int(tid) / p.nlay;
    const int lay = int(tid) - col * p.nlay;
    const ulong i = ulong(tid);
    const int jp = r.jp[i], jt = r.jt[i], jt1 = r.jt1[i];
    const int lo1 = ((jp - 1) * 5 + (jt - 1)) * p.nspa1;
    const int lo11 = (jp * 5 + (jt1 - 1)) * p.nspa1;
    const int hi1 = ((jp - 13) * 5 + (jt - 1)) * p.nspb1;
    const int hi11 = ((jp - 12) * 5 + (jt1 - 1)) * p.nspb1;
    const int self = clamp(r.indself[i] - 1, 0, p.nself1 - 2);
    const float selfterm = r.selffac[i] * (r.self1[ulong(self) * 8ul]
        + r.selffrac[i] * (r.self1[ulong(self + 1) * 8ul]
                           - r.self1[ulong(self) * 8ul]));
    const bool trop = lay + 1 <= r.laytrop[col];
    for (int g = 0; g < 8; ++g) {
        const float low = r.colh2o[i] * (rrtm_lw_interp4(
            r.absa1, p.nabsa1, 8, g, lo1, lo11,
            r.fac00[i], r.fac01[i], r.fac10[i], r.fac11[i])
            + selfterm + r.forfac[i] * r.forref1[g]);
        const float high = r.colh2o[i] * (rrtm_lw_interp4(
            r.absb1, p.nabsb1, 8, g, hi1, hi11,
            r.fac00[i], r.fac01[i], r.fac10[i], r.fac11[i])
            + r.forfac[i] * r.forref1[g]);
        const ulong out = i * 140ul + ulong(g);
        r.taug[out] = trop ? low : high;
        r.pfrac[out] = trop ? r.fraca1[g] : r.fracb1[g];
    }

    const float water = 1.0e20f * r.colh2o[i] / r.coldry[i];
    const float h2oparam = water / (water + 0.002f);
    int ifrac = 13;
    for (int candidate = 12; candidate >= 2; --candidate)
        if (h2oparam >= r.refparam2[candidate - 1]) ifrac = candidate;
    const float refat = r.refparam2[ifrac - 1];
    const float refprev = r.refparam2[ifrac - 2];
    const float fracint = (h2oparam - refat) / (refprev - refat);
    const float fp2 = r.fac11[i] + r.fac01[i];
    int ifp = int(trunc(200.0f * fp2 + 0.5f));
    if (ifp <= 0) ifp = 0;
    ifp = clamp(ifp, 0, p.ncorr - 1);
    const float fc00 = r.fac00[i] * r.corr2[ifp];
    const float fc10 = r.fac10[i] * r.corr2[ifp];
    const float fc01 = r.fac01[i] * r.corr1[ifp];
    const float fc11 = r.fac11[i] * r.corr1[ifp];
    const int lo2 = ((jp - 1) * 5 + (jt - 1)) * p.nspa2;
    const int lo21 = (jp * 5 + (jt1 - 1)) * p.nspa2;
    const int hi2 = ((jp - 13) * 5 + (jt - 1)) * p.nspb2;
    const int hi21 = ((jp - 12) * 5 + (jt1 - 1)) * p.nspb2;
    const int self2 = clamp(r.indself[i] - 1, 0, p.nself2 - 2);
    const float selfterm2 = r.selffac[i] * (r.self2[ulong(self2) * 14ul]
        + r.selffrac[i] * (r.self2[ulong(self2 + 1) * 14ul]
                           - r.self2[ulong(self2) * 14ul]));
    for (int g = 0; g < 14; ++g) {
        const float low = r.colh2o[i] * (rrtm_lw_interp4(
            r.absa2, p.nabsa2, 14, g, lo2, lo21,
            fc00, fc01, fc10, fc11) + selfterm2 + r.forfac[i] * r.forref2[g]);
        const float high = r.colh2o[i] * (rrtm_lw_interp4(
            r.absb2, p.nabsb2, 14, g, hi2, hi21,
            fc00, fc01, fc10, fc11) + r.forfac[i] * r.forref2[g]);
        const ulong out = i * 140ul + ulong(8 + g);
        const float f0 = r.fraca2[ulong(ifrac - 1) * 14ul + ulong(g)];
        const float f1 = r.fraca2[ulong(ifrac - 2) * 14ul + ulong(g)];
        r.taug[out] = trop ? low : high;
        r.pfrac[out] = trop ? f0 + fracint * (f1 - f0) : r.fracb2[g];
    }
}

kernel void rrtm_lw_taumol_03(
    constant RrtmLwTaumol03Resources &r [[buffer(0)]],
    constant RrtmLwTaumol03Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col = int(tid) / p.nlay;
    const int lay = int(tid) - col * p.nlay;
    const ulong i = ulong(tid), outbase = i * 140ul + 22ul;
    const int jp = r.jp[i], jt = r.jt[i], jt1 = r.jt1[i];
    const float a = r.colh2o[i], b = r.colco2[i];
    const float fp = r.fac01[i] + r.fac11[i];
    const bool trop = lay + 1 <= r.laytrop[col];
    float comb, fs; int js;
    rrtm_lw_spec(a, b, 1.19268f, 8.0f, comb, js, fs);
    int jsl = js; float fsl = fs;
    if (js == 8) {
        jsl = fs >= 0.9f ? 9 : js;
        fsl = fs >= 0.9f ? 10.0f * (fs - 0.9f) : fs / 0.9f;
    }
    const int ns = jsl + int(trunc(fsl + 0.5f));
    float eta = 1.0f - rrtm_lw_table_value(
        r.tables, r.table_index, p.id_etaref, ns - 1, 0);
    if (eta == 0.0f) eta = 1.0f;
    const float h1 = rrtm_lw_table_value(r.tables, r.table_index,
                                         p.id_h2oref, jp - 1, 0);
    const float h2 = rrtm_lw_table_value(r.tables, r.table_index,
                                         p.id_h2oref, jp, 0);
    const float c1 = 1.19268f * rrtm_lw_table_value(
        r.tables, r.table_index, p.id_co2ref, jp - 1, 0) / eta;
    const float c2 = 1.19268f * rrtm_lw_table_value(
        r.tables, r.table_index, p.id_co2ref, jp, 0) / eta;
    const float w1 = ns == 10 ? h1 : c1;
    const float w2 = ns == 10 ? h2 : c2;
    const float n2omult = r.coln2o[i] - comb * rrtm_lw_blob_ratio(
        r.tables, r.table_index, p.id_n2oref, jp, fp, w1, w2);
    const int i0 = ((jp - 1) * 5 + (jt - 1)) * p.nspa3 + jsl - 1;
    const int i1 = (jp * 5 + (jt1 - 1)) * p.nspa3 + jsl - 1;
    const float self0 = rrtm_lw_table_value(r.tables, r.table_index,
        p.id_self3, r.indself[i] - 1, 0);
    const float self1 = rrtm_lw_table_value(r.tables, r.table_index,
        p.id_self3, r.indself[i], 0);
    const float continuum = r.selffac[i]
        * (self0 + r.selffrac[i] * (self1 - self0));
    const float f00 = r.fac00[i], f01 = r.fac01[i];
    const float f10 = r.fac10[i], f11 = r.fac11[i];
    float combu, fsu; int jsu;
    rrtm_lw_spec(a, b, 1.19268f, 4.0f, combu, jsu, fsu);
    const int nsu = jsu + int(trunc(fsu + 0.5f));
    float etau = 1.0f - rrtm_lw_table_value(
        r.tables, r.table_index, p.id_etaref, nsu - 1, 0);
    if (etau == 0.0f) etau = 1.0f;
    const float hu1 = rrtm_lw_table_value(r.tables, r.table_index,
        p.id_h2oref, jp - 1, 0);
    const float hu2 = rrtm_lw_table_value(r.tables, r.table_index,
        p.id_h2oref, jp, 0);
    const float cu1 = 1.19268f * rrtm_lw_table_value(
        r.tables, r.table_index, p.id_co2ref, jp - 1, 0) / etau;
    const float cu2 = 1.19268f * rrtm_lw_table_value(
        r.tables, r.table_index, p.id_co2ref, jp, 0) / etau;
    const float n2omu = r.coln2o[i] - combu * rrtm_lw_blob_ratio(
        r.tables, r.table_index, p.id_n2oref, jp, fp, nsu == 5 ? hu1 : cu1,
        nsu == 5 ? hu2 : cu2);
    const int hi0u = ((jp - 13) * 5 + (jt - 1)) * p.nspb3 + jsu - 1;
    const int hi1u = ((jp - 12) * 5 + (jt1 - 1)) * p.nspb3 + jsu - 1;
    const int3 meta_a = rrtm_lw_table_info(r.table_index, p.id_absa);
    const int3 meta_b = rrtm_lw_table_info(r.table_index, p.id_absb);
    for (int g = 0; g < 16; ++g) {
        const float low = comb * rrtm_lw_interp8(
            r.tables + meta_a.x, p.nrowa3, 16, g, i0, i1, 10,
            fsl, f00, f01, f10, f11)
          + a * (continuum + r.forfac[i] * rrtm_lw_table_value(
              r.tables, r.table_index, p.id_forref, 0, g))
          + n2omult * rrtm_lw_table_value(
              r.tables, r.table_index, p.id_absn2oa, 0, g);
        const float high = combu * rrtm_lw_interp8(
            r.tables + meta_b.x, p.nrowb3, 16, g, hi0u, hi1u, 5, fsu,
            f00, f01, f10, f11)
          + (a * r.forfac[i]) * rrtm_lw_table_value(
              r.tables, r.table_index, p.id_forref, 0, g)
          + n2omu * rrtm_lw_table_value(
              r.tables, r.table_index, p.id_absn2ob, 0, g);
        const float pl = rrtm_lw_table_value(
            r.tables, r.table_index, p.id_fraca, jsl - 1, g)
          + fsl * (rrtm_lw_table_value(r.tables, r.table_index,
              p.id_fraca, jsl, g) - rrtm_lw_table_value(
              r.tables, r.table_index, p.id_fraca, jsl - 1, g));
        const float ph = rrtm_lw_table_value(
            r.tables, r.table_index, p.id_fracb, jsu - 1, g)
          + fsu * (rrtm_lw_table_value(r.tables, r.table_index,
              p.id_fracb, jsu, g) - rrtm_lw_table_value(
              r.tables, r.table_index, p.id_fracb, jsu - 1, g));
        r.taug[outbase + ulong(g)] = trop ? low : high;
        r.pfrac[outbase + ulong(g)] = trop ? pl : ph;
    }
    (void)meta_a;
}

kernel void rrtm_lw_taumol_06(
    constant RrtmLwTaumol06Resources &r [[buffer(0)]],
    constant RrtmLwTaumol06Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col = int(tid) / p.nlay;
    const int lay = int(tid) - col * p.nlay;
    const ulong i = ulong(tid), outbase = i * 140ul + 68ul;
    const int jp = r.jp[i], jt = r.jt[i], jt1 = r.jt1[i];
    const int i0 = ((jp - 1) * 5 + (jt - 1)) * p.nspa6;
    const int i1 = (jp * 5 + (jt1 - 1)) * p.nspa6;
    const float self0 = rrtm_lw_table_value(r.tables,r.table_index,p.id_self6,
                                             r.indself[i] - 1,0);
    const float self1 = rrtm_lw_table_value(r.tables,r.table_index,p.id_self6,
                                             r.indself[i],0);
    const float continuum = r.selffac[i]
        * (self0 + r.selffrac[i] * (self1 - self0));
    const bool trop = lay + 1 <= r.laytrop[col];
    const int3 meta = rrtm_lw_table_info(r.table_index,p.id_absa6);
    for (int g=0; g<8; ++g) {
        const float cfc = r.wx[i * 4ul + 1ul] * rrtm_lw_table_value(
            r.tables,r.table_index,p.id_cfc11,0,g)
          + r.wx[i * 4ul + 2ul] * rrtm_lw_table_value(
            r.tables,r.table_index,p.id_cfc12,0,g);
        const float gas = r.colh2o[i] * (rrtm_lw_interp4(
            r.tables+meta.x,p.nrowa6,8,g,i0,i1,r.fac00[i],r.fac01[i],
            r.fac10[i],r.fac11[i]) + continuum)
          + cfc + r.co2mult[i] * rrtm_lw_table_value(
              r.tables,r.table_index,p.id_co2,0,g);
        r.taug[outbase+ulong(g)] = trop ? gas : cfc;
        r.pfrac[outbase+ulong(g)] = rrtm_lw_table_value(
            r.tables,r.table_index,p.id_frac,0,g);
    }
}

kernel void rrtm_lw_taumol_04_05(
    constant RrtmLwTaumol0405Resources &r [[buffer(0)]],
    constant RrtmLwTaumol0405Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col=int(tid)/p.nlay, lay=int(tid)-col*p.nlay;
    const ulong i=ulong(tid), outbase=i*140ul+ulong(p.offset);
    const int jp=r.jp[i], jt=r.jt[i], jt1=r.jt1[i];
    const bool trop=lay+1<=r.laytrop[col];
    float comb,fs; int js;
    rrtm_lw_spec(r.colh2o[i],r.colco2[i],p.strrat_low,8.0f,comb,js,fs);
    const int i0=((jp-1)*5+(jt-1))*p.nspa+js-1;
    const int i1=(jp*5+(jt1-1))*p.nspa+js-1;
    const int3 ma=rrtm_lw_table_info(r.table_index,p.id_absa);
    const int3 ms=rrtm_lw_table_info(r.table_index,p.id_self);
    const float s0=rrtm_lw_table_value(r.tables,r.table_index,p.id_self,
                                        r.indself[i]-1,0);
    const float s1=rrtm_lw_table_value(r.tables,r.table_index,p.id_self,
                                        r.indself[i],0);
    const float continuum=r.selffac[i]*(s0+r.selffrac[i]*(s1-s0));
    float combu,fsu; int jsu;
    rrtm_lw_spec(r.colo3[i],r.colco2[i],p.strrat_high,4.0f,combu,jsu,fsu);
    float js_adj=float(jsu), fs_adj=fsu;
    if (p.special_high != 0) {
        if (jsu > 1) js_adj=float(jsu+1);
        else if (fsu >= 0.0024f) { js_adj=2.0f; fs_adj=(fsu-0.0024f)/0.9976f; }
        else { js_adj=1.0f; fs_adj=fsu/0.0024f; }
    }
    const int ju=int(js_adj);
    const int hi0=((jp-13)*5+(jt-1))*p.nspb+ju-1;
    const int hi1=((jp-12)*5+(jt1-1))*p.nspb+ju-1;
    const int3 mb=rrtm_lw_table_info(r.table_index,p.id_absb);
    const float f00=r.fac00[i],f01=r.fac01[i],f10=r.fac10[i],f11=r.fac11[i];
    for(int g=0;g<p.ng;++g){
        float low=comb*rrtm_lw_interp8(r.tables+ma.x,p.nrowa,p.ng,g,
            i0,i1,p.step_low,fs,f00,f01,f10,f11)+r.colh2o[i]*continuum;
        if(p.band==5) low+=r.wx[i*4ul]*rrtm_lw_table_value(
            r.tables,r.table_index,p.id_ccl4,0,g);
        const float high=combu*rrtm_lw_interp8(r.tables+mb.x,p.nrowb,p.ng,g,
            hi0,hi1,p.step_high,fs_adj,f00,f01,f10,f11)
            +(p.band==5?r.wx[i*4ul]*rrtm_lw_table_value(
              r.tables,r.table_index,p.id_ccl4,0,g):0.0f);
        const float pl=rrtm_lw_table_value(r.tables,r.table_index,p.id_fraca,
                                            js-1,g)+fs*(rrtm_lw_table_value(
            r.tables,r.table_index,p.id_fraca,js,g)-rrtm_lw_table_value(
            r.tables,r.table_index,p.id_fraca,js-1,g));
        const float ph=rrtm_lw_table_value(r.tables,r.table_index,p.id_fracb,
                                            ju-1,g)+fs_adj*(rrtm_lw_table_value(
            r.tables,r.table_index,p.id_fracb,ju,g)-rrtm_lw_table_value(
            r.tables,r.table_index,p.id_fracb,ju-1,g));
        r.taug[outbase+ulong(g)]=trop?low:high;
        r.pfrac[outbase+ulong(g)]=trop?pl:ph;
    }
    (void)ms;
}

kernel void rrtm_lw_taumol_07(
    constant RrtmLwTaumol07Resources &r [[buffer(0)]],
    constant RrtmLwTaumol07Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncell))return;
    const int col=int(tid)/p.nlay,lay=int(tid)-col*p.nlay;
    const ulong i=ulong(tid),outbase=i*140ul+76ul;
    const int jp=r.jp[i],jt=r.jt[i],jt1=r.jt1[i];
    const bool trop=lay+1<=r.laytrop[col];
    float comb,fs;int js;
    rrtm_lw_spec(r.colh2o[i],r.colo3[i],82110.4f,8.0f,comb,js,fs);
    const int lo0=((jp-1)*5+(jt-1))*p.nspa+js-1;
    const int lo1=(jp*5+(jt1-1))*p.nspa+js-1;
    const int3 ma=rrtm_lw_table_info(r.table_index,p.id_absa);
    const int3 mb=rrtm_lw_table_info(r.table_index,p.id_absb);
    const int3 ms=rrtm_lw_table_info(r.table_index,p.id_self);
    const float s0=rrtm_lw_table_value(r.tables,r.table_index,p.id_self,r.indself[i]-1,0);
    const float s1=rrtm_lw_table_value(r.tables,r.table_index,p.id_self,r.indself[i],0);
    const float self=r.selffac[i]*(s0+r.selffrac[i]*(s1-s0));
    const float f00=r.fac00[i],f01=r.fac01[i],f10=r.fac10[i],f11=r.fac11[i];
    const int hi0=((jp-13)*5+(jt-1))*p.nspb;
    const int hi1=((jp-12)*5+(jt1-1))*p.nspb;
    for(int g=0;g<12;++g){
      const float low=comb*rrtm_lw_interp8(r.tables+ma.x,p.nrowa,12,g,lo0,lo1,9,fs,
          f00,f01,f10,f11)+r.colh2o[i]*self+r.co2mult[i]*rrtm_lw_table_value(
          r.tables,r.table_index,p.id_co2,0,g);
      const float high=r.colo3[i]*rrtm_lw_interp4(r.tables+mb.x,p.nrowb,12,g,
          hi0,hi1,f00,f01,f10,f11)+r.co2mult[i]*rrtm_lw_table_value(
          r.tables,r.table_index,p.id_co2,0,g);
      const float pl=rrtm_lw_table_value(r.tables,r.table_index,p.id_fraca,js-1,g)
        +fs*(rrtm_lw_table_value(r.tables,r.table_index,p.id_fraca,js,g)
        -rrtm_lw_table_value(r.tables,r.table_index,p.id_fraca,js-1,g));
      r.taug[outbase+ulong(g)]=trop?low:high;
      r.pfrac[outbase+ulong(g)]=trop?pl:rrtm_lw_table_value(
          r.tables,r.table_index,p.id_fracb,0,g);
    }
    (void)ms;
}

kernel void rrtm_lw_taumol_08(
    constant RrtmLwTaumol08Resources &r [[buffer(0)]],
    constant RrtmLwTaumol08Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncell))return;
    const int col=int(tid)/p.nlay,lay=int(tid)-col*p.nlay;
    const ulong i=ulong(tid),outbase=i*140ul+88ul;
    const int jp=r.jp[i],jt=r.jt[i],jt1=r.jt1[i];
    const float fp=r.fac01[i]+r.fac11[i];
    const bool lowbranch=lay+1<=r.layswtch[col];
    const int3 ma=rrtm_lw_table_info(r.table_index,p.id_absa);
    const int3 mb=rrtm_lw_table_info(r.table_index,p.id_absb);
    const int3 ms=rrtm_lw_table_info(r.table_index,p.id_self);
    const int i0=((jp-1)*5+(jt-1))*p.nspa;
    const int i1=(jp*5+(jt1-1))*p.nspa;
    const int hi0=((jp-7)*5+(jt-1))*p.nspb;
    const int hi1=((jp-6)*5+(jt1-1))*p.nspb;
    const float s0=rrtm_lw_table_value(r.tables,r.table_index,p.id_self,r.indself[i]-1,0);
    const float s1=rrtm_lw_table_value(r.tables,r.table_index,p.id_self,r.indself[i],0);
    const float self=r.selffac[i]*(s0+r.selffrac[i]*(s1-s0));
    const float nref0=rrtm_lw_table_value(r.tables,r.table_index,p.id_n2oref,jp-1,0);
    const float nref1=rrtm_lw_table_value(r.tables,r.table_index,p.id_n2oref,jp,0);
    const float h0=rrtm_lw_table_value(r.tables,r.table_index,p.id_h2oref,jp-1,0);
    const float h1=rrtm_lw_table_value(r.tables,r.table_index,p.id_h2oref,jp,0);
    const float o0=rrtm_lw_table_value(r.tables,r.table_index,p.id_o3ref,jp-1,0);
    const float o1=rrtm_lw_table_value(r.tables,r.table_index,p.id_o3ref,jp,0);
    const float n2ol=r.coln2o[i]-r.colh2o[i]*(nref0/h0+fp*(nref1/h1-nref0/h0));
    const float n2ou=r.coln2o[i]-r.colo3[i]*(nref0/o0+fp*(nref1/o1-nref0/o0));
    const float cfc=r.wx[i*4ul+2ul]*rrtm_lw_table_value(r.tables,r.table_index,p.id_cfc12,0,0);
    for(int g=0;g<8;++g){
      const float cfcg=r.wx[i*4ul+2ul]*rrtm_lw_table_value(r.tables,r.table_index,p.id_cfc12,0,g)
        +r.wx[i*4ul+3ul]*rrtm_lw_table_value(r.tables,r.table_index,p.id_cfc22,0,g);
      const float lo=r.colh2o[i]*rrtm_lw_interp4(r.tables+ma.x,p.nrowa,8,g,i0,i1,
          r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i])+r.colh2o[i]*self+cfcg
          +r.co2mult[i]*rrtm_lw_table_value(r.tables,r.table_index,p.id_co2a,0,g)
          +n2ol*rrtm_lw_table_value(r.tables,r.table_index,p.id_n2oa,0,g);
      const float hi=r.colo3[i]*rrtm_lw_interp4(r.tables+mb.x,p.nrowb,8,g,hi0,hi1,
          r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i])+cfcg
          +r.co2mult[i]*rrtm_lw_table_value(r.tables,r.table_index,p.id_co2b,0,g)
          +n2ou*rrtm_lw_table_value(r.tables,r.table_index,p.id_n2ob,0,g);
      r.taug[outbase+ulong(g)]=lowbranch?lo:hi;
      r.pfrac[outbase+ulong(g)]=rrtm_lw_table_value(r.tables,r.table_index,
          lowbranch?p.id_fraca:p.id_fracb,0,g);
    }
    (void)ms;(void)cfc;
}

kernel void rrtm_lw_taumol_10_11(
    constant RrtmLwTaumol1011Resources &r [[buffer(0)]],
    constant RrtmLwTaumol1011Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncell))return;
    const int col=int(tid)/p.nlay,lay=int(tid)-col*p.nlay;
    const ulong i=ulong(tid),outbase=i*140ul+ulong(p.offset);
    const int jp=r.jp[i],jt=r.jt[i],jt1=r.jt1[i];
    const int lo0=((jp-1)*5+(jt-1))*p.nspa;
    const int lo1=(jp*5+(jt1-1))*p.nspa;
    const int hi0=((jp-13)*5+(jt-1))*p.nspb;
    const int hi1=((jp-12)*5+(jt1-1))*p.nspb;
    const int3 ma=rrtm_lw_table_info(r.table_index,p.id_absa);
    const int3 mb=rrtm_lw_table_info(r.table_index,p.id_absb);
    const bool trop=lay+1<=r.laytrop[col];
    float self=0.0f;
    if(p.have_self!=0){
      const float s0=rrtm_lw_table_value(r.tables,r.table_index,p.id_self,r.indself[i]-1,0);
      const float s1=rrtm_lw_table_value(r.tables,r.table_index,p.id_self,r.indself[i],0);
      self=r.selffac[i]*(s0+r.selffrac[i]*(s1-s0));
    }
    for(int g=0;g<p.ng;++g){
      const float low=r.colh2o[i]*(rrtm_lw_interp4(r.tables+ma.x,p.nrowa,p.ng,g,
          lo0,lo1,r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i])+self);
      const float high=r.colh2o[i]*rrtm_lw_interp4(r.tables+mb.x,p.nrowb,p.ng,g,
          hi0,hi1,r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i]);
      r.taug[outbase+ulong(g)]=trop?low:high;
      r.pfrac[outbase+ulong(g)]=rrtm_lw_table_value(r.tables,r.table_index,
          trop?p.id_fraca:p.id_fracb,0,g);
    }
}

kernel void rrtm_lw_taumol_12_16(
    constant RrtmLwTaumol1216Resources &r [[buffer(0)]],
    constant RrtmLwTaumol1216Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col=int(tid)/p.nlay, lay=int(tid)-col*p.nlay;
    const ulong i=ulong(tid), outbase=i*140ul+ulong(p.offset);
    const int jp=r.jp[i], jt=r.jt[i], jt1=r.jt1[i];
    const bool trop=lay+1<=r.laytrop[col];
    const int3 ma=rrtm_lw_table_info(r.table_index,p.id_absa);
    const int3 mb=rrtm_lw_table_info(r.table_index,p.id_absb);
    const float ga = p.gas_a==1 ? r.colh2o[i] : p.gas_a==2 ? r.colco2[i]
                   : p.gas_a==3 ? r.coln2o[i] : r.colch4[i];
    const float gb = p.gas_b==1 ? r.colh2o[i] : p.gas_b==2 ? r.colco2[i]
                   : p.gas_b==3 ? r.coln2o[i] : r.colch4[i];
    const int lo0=((jp-1)*5+(jt-1))*p.nspa;
    const int lo1=(jp*5+(jt1-1))*p.nspa;
    const int hi0=((jp-13)*5+(jt-1))*p.nspb;
    const int hi1=((jp-12)*5+(jt1-1))*p.nspb;
    float continuum=0.0f;
    if(p.have_self!=0){
      const float s0=rrtm_lw_table_value(r.tables,r.table_index,p.id_self,r.indself[i]-1,0);
      const float s1=rrtm_lw_table_value(r.tables,r.table_index,p.id_self,r.indself[i],0);
      continuum=r.selffac[i]*(s0+r.selffrac[i]*(s1-s0));
    }
    float comb=0.0f, fs=0.0f; int js=1;
    if(p.mode==0) rrtm_lw_spec(ga,gb,p.strrat,8.0f,comb,js,fs);
    for(int g=0;g<p.ng;++g){
      float low=0.0f, high=0.0f, pl=0.0f, ph=0.0f;
      if(p.mode==0){
        low=comb*rrtm_lw_interp8(r.tables+ma.x,p.nrowa,p.ng,g,lo0+js-1,
          lo1+js-1,9,fs,r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i])
          +r.colh2o[i]*continuum;
        pl=rrtm_lw_table_value(r.tables,r.table_index,p.id_fraca,js-1,g)
          +fs*(rrtm_lw_table_value(r.tables,r.table_index,p.id_fraca,js,g)
             -rrtm_lw_table_value(r.tables,r.table_index,p.id_fraca,js-1,g));
      } else {
        low=ga*rrtm_lw_interp4(r.tables+ma.x,p.nrowa,p.ng,g,lo0,lo1,
          r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i])+r.colh2o[i]*continuum;
        high=ga*rrtm_lw_interp4(r.tables+mb.x,p.nrowb,p.ng,g,hi0,hi1,
          r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i]);
        pl=rrtm_lw_table_value(r.tables,r.table_index,p.id_fraca,0,g);
        ph=rrtm_lw_table_value(r.tables,r.table_index,p.id_fracb,0,g);
      }
      r.taug[outbase+ulong(g)]=p.mode==0 ? (trop?low:0.0f):(trop?low:high);
      r.pfrac[outbase+ulong(g)]=p.mode==0 ? (trop?pl:0.0f):(trop?pl:ph);
    }
}

kernel void rrtm_lw_taumol_09(
    constant RrtmLwTaumol09Resources &r [[buffer(0)]],
    constant RrtmLwTaumol09Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid>=uint(p.ncell))return;
    const int col=int(tid)/p.nlay,lay=int(tid)-col*p.nlay;
    const ulong i=ulong(tid),outbase=i*140ul+108ul;
    const int jp=r.jp[i],jt=r.jt[i],jt1=r.jt1[i];
    const bool trop=lay+1<=r.laytrop[col];
    float comb=0.0f,fs=0.0f;int js=1;
    rrtm_lw_spec(r.colh2o[i],r.colch4[i],21.6282f,8.0f,comb,js,fs);
    int jfrac=js;float ffrac=fs;
    int jsl=js;float fsl=fs;
    if(js==8){
      if(fs<=0.68f) fsl=fs/0.68f;
      else if(fs<=0.92f){jsl=9;fsl=(fs-0.68f)/0.24f;}
      else {jsl=10;fsl=(fs-0.92f)/0.08f;}
    } else if(js==9){jsl=10;fsl=0.0f;jfrac=8;ffrac=1.0f;}
    const int ns=jsl+int(trunc(fsl+0.5f));
    float eta=1.0f-rrtm_lw_table_value(r.tables,r.table_index,p.id_eta,ns-1,0);
    if(eta==0.0f)eta=1.0f;
    const float h0=rrtm_lw_table_value(r.tables,r.table_index,p.id_h2oref,jp-1,0);
    const float h1=rrtm_lw_table_value(r.tables,r.table_index,p.id_h2oref,jp,0);
    const float c0=21.6282f*rrtm_lw_table_value(r.tables,r.table_index,p.id_ch4ref,jp-1,0)/eta;
    const float c1=21.6282f*rrtm_lw_table_value(r.tables,r.table_index,p.id_ch4ref,jp,0)/eta;
    const float den0=ns==11?h0:c0,den1=ns==11?h1:c1;
    const float nref0=rrtm_lw_table_value(r.tables,r.table_index,p.id_n2oref,jp-1,0);
    const float nref1=rrtm_lw_table_value(r.tables,r.table_index,p.id_n2oref,jp,0);
    const float nratio=nref0/den0;
    const float nratio1=nref1/den1;
    const float n2omult=r.coln2o[i]-comb*(nratio+(r.fac01[i]+r.fac11[i])*(nratio1-nratio));
    int ioff=0;
    if(r.laylow[col]>=1 && lay+1>=r.laylow[col])ioff=p.ng;
    if(r.layswtch[col]>=1 && lay+1>=r.layswtch[col])ioff=2*p.ng;
    const int row=ioff/p.ng;
    const int3 ma=rrtm_lw_table_info(r.table_index,p.id_absa);
    const int3 mb=rrtm_lw_table_info(r.table_index,p.id_absb);
    const int lo0=((jp-1)*5+(jt-1))*p.nspa+jsl-1;
    const int lo1=(jp*5+(jt1-1))*p.nspa+jsl-1;
    const int hi0=((jp-13)*5+(jt-1))*p.nspb;
    const int hi1=((jp-12)*5+(jt1-1))*p.nspb;
    const float s0=rrtm_lw_table_value(r.tables,r.table_index,p.id_self,r.indself[i]-1,0);
    const float s1=rrtm_lw_table_value(r.tables,r.table_index,p.id_self,r.indself[i],0);
    const float continuum=r.selffac[i]*(s0+r.selffrac[i]*(s1-s0));
    for(int g=0;g<p.ng;++g){
      const float low=comb*rrtm_lw_interp8(r.tables+ma.x,p.nrowa,p.ng,g,lo0,lo1,11,fsl,
        r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i])
        +r.colh2o[i]*continuum+n2omult*rrtm_lw_table_value(r.tables,r.table_index,p.id_absn2o,row,g);
      const float high=r.colch4[i]*rrtm_lw_interp4(r.tables+mb.x,p.nrowb,p.ng,g,hi0,hi1,
        r.fac00[i],r.fac01[i],r.fac10[i],r.fac11[i]);
      const float pl0=rrtm_lw_table_value(r.tables,r.table_index,p.id_fraca,jfrac-1,g);
      const float pl1=rrtm_lw_table_value(r.tables,r.table_index,p.id_fraca,jfrac,g);
      r.taug[outbase+ulong(g)]=trop?low:high;
      r.pfrac[outbase+ulong(g)]=trop?pl0+ffrac*(pl1-pl0):
        rrtm_lw_table_value(r.tables,r.table_index,p.id_fracb,0,g);
    }
}

kernel void rrtm_lw_rtrn_down(
    constant RrtmLwRtrnDownResources &r [[buffer(0)]],
    constant RrtmLwRtrnDownParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col = int(tid) / p.ngpt;
    const int ig = int(tid) - col * p.ngpt;
    const int band = int(r.ngb0[ig]);
    const ulong profile = ulong(col) * ulong(p.nlay);
    const ulong spectral = ulong(col) * ulong(p.nlay) * ulong(p.ngpt)
                         + ulong(ig);
    float bglev = r.pfrac[spectral + ulong(p.nlay - 1) * ulong(p.ngpt)]
        * rrtm_lw_planck(r.totplnk, r.delwave,
                         r.tz[ulong(col) * ulong(p.nlay + 1)
                              + ulong(p.nlay)], band);
    float radld = 0.0f;
    float radclrd = 0.0f;
    bool cloud_seen = false;
    for (int lev = p.nlay - 1; lev >= 0; --lev) {
        const ulong cell = profile + ulong(lev);
        const ulong idx = spectral + ulong(lev) * ulong(p.ngpt);
        const float cfrac = r.cldfrac[cell];
        const float odcld = 1.66f * r.taucloud[cell];
        const float abscld = 1.0f - exp(-odcld);
        const float efclfrac = abscld * cfrac;
        const int pade = clamp(r.itr[idx], 0, 5000);
        const float odclr = r.tau_table[pade];
        const float tauf = r.tf_table[pade];
        const float abss = 1.0f - r.trans_table[pade];
        const float bglay = r.pfrac[idx]
            * rrtm_lw_planck(r.totplnk, r.delwave, r.tavel[cell], band);
        const float delbgup = bglev - bglay;
        const float bbu = bglay + tauf * delbgup;
        const float next_bglev = r.pfrac[idx]
            * rrtm_lw_planck(r.totplnk, r.delwave,
                r.tz[ulong(col) * ulong(p.nlay + 1) + ulong(lev)], band);
        const float delbgdn = next_bglev - bglay;
        const float bbd = bglay + tauf * delbgdn;
        const float odsm = odclr + odcld;
        const float factot = odsm / (p.bpade + odsm);
        const float bbutot = bglay + factot * delbgup;
        const float bbdlevd = bglay + factot * delbgdn;
        const float atot = abss + abscld - abss * abscld;
        const float gassrc = bbd * abss;
        const bool cloudy = cfrac > 0.0f;
        cloud_seen = cloud_seen || cloudy;
        const float radld_cloudy = radld - radld * (abss + efclfrac * (1.0f - abss))
            + gassrc + cfrac * (bbdlevd * atot - gassrc);
        const float radld_clear = radld + (bbd - radld) * abss;
        const float clear_step = radclrd + (bbd - radclrd) * abss;
        radld = cloudy ? radld_cloudy : radld_clear;
        radclrd = (cloudy || cloud_seen) ? clear_step : radld;
        r.bbu[idx] = bbu;
        r.bbutot[idx] = bbutot;
        r.atot[idx] = atot;
        r.abss[idx] = abss;
        r.down[ulong(col) * ulong(p.nlay + 1) * ulong(p.ngpt)
               + ulong(lev) * ulong(p.ngpt) + ulong(ig)] = radld;
        r.clear_down[ulong(col) * ulong(p.nlay + 1) * ulong(p.ngpt)
                     + ulong(lev) * ulong(p.ngpt) + ulong(ig)] = radclrd;
        bglev = next_bglev;
    }
}

kernel void rrtm_lw_rtrn_up(
    constant RrtmLwRtrnUpResources &r [[buffer(0)]],
    constant RrtmLwRtrnUpParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col = int(tid) / p.ngpt;
    const int ig = int(tid) - col * p.ngpt;
    const int band = int(r.ngb0[ig]);
    const ulong profile = ulong(col) * ulong(p.nlay);
    const ulong spectral = ulong(col) * ulong(p.nlay) * ulong(p.ngpt)
                         + ulong(ig);
    const ulong levels = ulong(col) * ulong(p.nlay + 1) * ulong(p.ngpt);
    const float semis = r.semiss[ulong(col) * 16ul + ulong(band)];
    const float surface_emit = r.pfrac[spectral]
        * semis * rrtm_lw_planck(r.totplnk, r.delwave, r.tbound[col], band);
    float radlu = surface_emit + (1.0f - semis) * r.down[levels + ulong(ig)];
    r.up[levels + ulong(ig)] = radlu;
    float radclru = surface_emit + (1.0f - semis)
        * r.clear_down[levels + ulong(ig)];
    for (int lay = 0; lay < p.nlay; ++lay) {
        const ulong cell = profile + ulong(lay);
        const ulong idx = spectral + ulong(lay) * ulong(p.ngpt);
        const float cfrac = r.cldfrac[cell];
        const float odcld = 1.66f * r.taucloud[cell];
        const float abscld = 1.0f - exp(-odcld);
        const float efclfrac = abscld * cfrac;
        const float bbu = r.bbu[idx];
        const float bbutot = r.bbutot[idx];
        const float atot = r.atot[idx];
        const float layer_abss = r.abss[idx];
        const float gassrc = bbu * layer_abss;
        const float cloudy = radlu - radlu * (layer_abss + efclfrac * (1.0f - layer_abss))
            + gassrc + cfrac * (bbutot * atot - gassrc);
        const float clear = radlu + (bbu - radlu) * layer_abss;
        radlu = cfrac > 0.0f ? cloudy : clear;
        radclru = radclru + (bbu - radclru) * layer_abss;
        r.up[levels + ulong(lay + 1) * ulong(p.ngpt) + ulong(ig)] = radlu;
    }
}

kernel void rrtm_lw_rtrn_fluxes(
    constant RrtmLwRtrnFluxesResources &r [[buffer(0)]],
    constant RrtmLwRtrnFluxesParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col = int(tid) / (p.nlay + 1);
    const int lev = int(tid) - col * (p.nlay + 1);
    const ulong base = ulong(col) * ulong(p.nlay + 1) * 140ul
                     + ulong(lev) * 140ul;
    float up_flux = 0.0f;
    float down_flux = 0.0f;
    for (int ig = 0; ig < 140; ++ig) {
        up_flux += r.up[base + ulong(ig)] * 0.5f;
        down_flux += r.down[base + ulong(ig)] * 0.5f;
    }
    up_flux *= p.fluxfac;
    down_flux *= p.fluxfac;
    if (lev == 0) r.glw[col] = down_flux;
    if (lev == p.nmodel) r.olr[col] = up_flux;
    if (lev < p.nmodel) {
        const float fnet0 = up_flux - down_flux;
        const ulong next_base = base + 140ul;
        float next_up = 0.0f;
        float next_down = 0.0f;
        for (int ig = 0; ig < 140; ++ig) {
            next_up += r.up[next_base + ulong(ig)] * 0.5f;
            next_down += r.down[next_base + ulong(ig)] * 0.5f;
        }
        const float fnet1 = (next_up - next_down) * p.fluxfac;
        const float dp = r.pz[ulong(col) * ulong(p.nlay + 1) + ulong(lev)]
                       - r.pz[ulong(col) * ulong(p.nlay + 1) + ulong(lev + 1)];
        const float htr = p.heatfac * (fnet0 - fnet1) / dp / 86400.0f;
        const int iy = col / p.nx;
        const int ix = col - iy * p.nx;
        r.rthratenlw[(ulong(lev) * ulong(p.ny) + ulong(iy))
                     * ulong(p.nx) + ulong(ix)] = htr / r.exner[ulong(col)
                                                               * ulong(p.nmodel)
                                                               + ulong(lev)];
    }
}
