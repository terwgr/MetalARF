// Generated from classic RRTM RTRN; do not edit.
#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "wrf_exact_rrtm_lw_rtrn_down_bindings.metalh"
#include "wrf_exact_rrtm_lw_rtrn_up_bindings.metalh"
#include "wrf_exact_rrtm_lw_rtrn_fluxes_bindings.metalh"
using namespace metal;

static inline Exact32 exact_rrtm_exp(Exact32 x) { return Exact32::from_bits(soft_gfk_exp(x.bits)); }
static inline Exact32 exact_rrtm_planck(device const Exact32 *table, device const Exact32 *wave, Exact32 temperature, int band) {
    const int whole = int(trunc(as_type<float>(temperature.bits)));
    const int row = clamp(int(trunc(as_type<float>((temperature - 159.0f).bits))), 1, 181) - 1;
    const Exact32 fraction = temperature - Exact32(float(whole));
    const ulong lo = ulong(row) * 16ul + ulong(band);
    const ulong hi = ulong(min(row + 1, 180)) * 16ul + ulong(band);
    return wave[band] * (table[lo] + fraction * (table[hi] - table[lo]));
}
kernel void wrf_exact_rrtm_lw_rtrn_down(
    constant WrfExactRrtmLwRtrnDownResources &r [[buffer(0)]],
    constant WrfExactRrtmLwRtrnDownParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col = int(tid) / p.ngpt;
    const int ig = int(tid) - col * p.ngpt;
    const int band = int(r.ngb0[ig]);
    const ulong profile = ulong(col) * ulong(p.nlay);
    const ulong spectral = ulong(col) * ulong(p.nlay) * ulong(p.ngpt)
                         + ulong(ig);
    Exact32 bglev = reinterpret_cast<device const Exact32*>(r.pfrac)[spectral + ulong(p.nlay - 1) * ulong(p.ngpt)]
        * exact_rrtm_planck(reinterpret_cast<device const Exact32*>(r.totplnk), reinterpret_cast<device const Exact32*>(r.delwave),
                         reinterpret_cast<device const Exact32*>(r.tz)[ulong(col) * ulong(p.nlay + 1)
                              + ulong(p.nlay)], band);
    Exact32 radld = 0.0f;
    Exact32 radclrd = 0.0f;
    bool cloud_seen = false;
    const ulong top = ulong(col) * ulong(p.nlay + 1) * ulong(p.ngpt) + ulong(p.nlay) * ulong(p.ngpt) + ulong(ig);
    reinterpret_cast<device Exact32*>(r.down)[top] = 0.0f;
    reinterpret_cast<device Exact32*>(r.clear_down)[top] = 0.0f;
    for (int lev = p.nlay - 1; lev >= 0; --lev) {
        const ulong cell = profile + ulong(lev);
        const ulong idx = spectral + ulong(lev) * ulong(p.ngpt);
        const Exact32 cfrac = reinterpret_cast<device const Exact32*>(r.cldfrac)[cell];
        const Exact32 odcld = 1.66f * reinterpret_cast<device const Exact32*>(r.taucloud)[cell];
        const Exact32 abscld = 1.0f - exact_rrtm_exp(-odcld);
        const Exact32 efclfrac = abscld * cfrac;
        const int pade = clamp(r.itr[idx], 0, 5000);
        const Exact32 odclr = reinterpret_cast<device const Exact32*>(r.tau_table)[pade];
        const Exact32 tauf = reinterpret_cast<device const Exact32*>(r.tf_table)[pade];
        const Exact32 abss = 1.0f - reinterpret_cast<device const Exact32*>(r.trans_table)[pade];
        const Exact32 bglay = reinterpret_cast<device const Exact32*>(r.pfrac)[idx]
            * exact_rrtm_planck(reinterpret_cast<device const Exact32*>(r.totplnk), reinterpret_cast<device const Exact32*>(r.delwave), reinterpret_cast<device const Exact32*>(r.tavel)[cell], band);
        const Exact32 delbgup = bglev - bglay;
        const Exact32 bbu = bglay + tauf * delbgup;
        const Exact32 next_bglev = reinterpret_cast<device const Exact32*>(r.pfrac)[idx]
            * exact_rrtm_planck(reinterpret_cast<device const Exact32*>(r.totplnk), reinterpret_cast<device const Exact32*>(r.delwave),
                reinterpret_cast<device const Exact32*>(r.tz)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)], band);
        const Exact32 delbgdn = next_bglev - bglay;
        const Exact32 bbd = bglay + tauf * delbgdn;
        const Exact32 odsm = odclr + odcld;
        const Exact32 factot = odsm / (Exact32(p.bpade) + odsm);
        const Exact32 bbutot = bglay + factot * delbgup;
        const Exact32 bbdlevd = bglay + factot * delbgdn;
        const Exact32 atot = abss + abscld - abss * abscld;
        const Exact32 gassrc = bbd * abss;
        const bool cloudy = cfrac > 0.0f;
        cloud_seen = cloud_seen || cloudy;
        const Exact32 radld_cloudy = radld - radld * (abss + efclfrac * (1.0f - abss))
            + gassrc + cfrac * (bbdlevd * atot - gassrc);
        const Exact32 radld_clear = radld + (bbd - radld) * abss;
        const Exact32 clear_step = radclrd + (bbd - radclrd) * abss;
        radld = cloudy ? radld_cloudy : radld_clear;
        radclrd = (cloudy || cloud_seen) ? clear_step : radld;
        reinterpret_cast<device Exact32*>(r.bbu)[idx] = bbu;
        reinterpret_cast<device Exact32*>(r.bbutot)[idx] = bbutot;
        reinterpret_cast<device Exact32*>(r.atot)[idx] = atot;
        reinterpret_cast<device Exact32*>(r.abss)[idx] = abss;
        reinterpret_cast<device Exact32*>(r.down)[ulong(col) * ulong(p.nlay + 1) * ulong(p.ngpt)
               + ulong(lev) * ulong(p.ngpt) + ulong(ig)] = radld;
        reinterpret_cast<device Exact32*>(r.clear_down)[ulong(col) * ulong(p.nlay + 1) * ulong(p.ngpt)
                     + ulong(lev) * ulong(p.ngpt) + ulong(ig)] = radclrd;
        bglev = next_bglev;
    }
}

kernel void wrf_exact_rrtm_lw_rtrn_up(
    constant WrfExactRrtmLwRtrnUpResources &r [[buffer(0)]],
    constant WrfExactRrtmLwRtrnUpParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col = int(tid) / p.ngpt;
    const int ig = int(tid) - col * p.ngpt;
    const int band = int(r.ngb0[ig]);
    const ulong profile = ulong(col) * ulong(p.nlay);
    const ulong spectral = ulong(col) * ulong(p.nlay) * ulong(p.ngpt)
                         + ulong(ig);
    const ulong levels = ulong(col) * ulong(p.nlay + 1) * ulong(p.ngpt);
    const Exact32 semis = reinterpret_cast<device const Exact32*>(r.semiss)[ulong(col) * 16ul + ulong(band)];
    const Exact32 surface_emit = reinterpret_cast<device const Exact32*>(r.pfrac)[spectral]
        * (semis * exact_rrtm_planck(reinterpret_cast<device const Exact32*>(r.totplnk), reinterpret_cast<device const Exact32*>(r.delwave), reinterpret_cast<device const Exact32*>(r.tbound)[col], band));
    Exact32 radlu = surface_emit + (1.0f - semis) * reinterpret_cast<device const Exact32*>(r.down)[levels + ulong(ig)];
    reinterpret_cast<device Exact32*>(r.up)[levels + ulong(ig)] = radlu;
    Exact32 radclru = surface_emit + (1.0f - semis)
        * reinterpret_cast<device const Exact32*>(r.clear_down)[levels + ulong(ig)];
    reinterpret_cast<device Exact32*>(r.clear_up)[levels + ulong(ig)] = radclru;
    for (int lay = 0; lay < p.nlay; ++lay) {
        const ulong cell = profile + ulong(lay);
        const ulong idx = spectral + ulong(lay) * ulong(p.ngpt);
        const Exact32 cfrac = reinterpret_cast<device const Exact32*>(r.cldfrac)[cell];
        const Exact32 odcld = 1.66f * reinterpret_cast<device const Exact32*>(r.taucloud)[cell];
        const Exact32 abscld = 1.0f - exact_rrtm_exp(-odcld);
        const Exact32 efclfrac = abscld * cfrac;
        const Exact32 bbu = reinterpret_cast<device const Exact32*>(r.bbu)[idx];
        const Exact32 bbutot = reinterpret_cast<device const Exact32*>(r.bbutot)[idx];
        const Exact32 atot = reinterpret_cast<device const Exact32*>(r.atot)[idx];
        const Exact32 layer_abss = reinterpret_cast<device const Exact32*>(r.abss)[idx];
        const Exact32 gassrc = bbu * layer_abss;
        const Exact32 cloudy = radlu - radlu * (layer_abss + efclfrac * (1.0f - layer_abss))
            + gassrc + cfrac * (bbutot * atot - gassrc);
        const Exact32 clear = radlu + (bbu - radlu) * layer_abss;
        radlu = cfrac > 0.0f ? cloudy : clear;
        radclru = radclru + (bbu - radclru) * layer_abss;
        reinterpret_cast<device Exact32*>(r.clear_up)[levels + ulong(lay + 1) * ulong(p.ngpt) + ulong(ig)] = radclru;
        reinterpret_cast<device Exact32*>(r.up)[levels + ulong(lay + 1) * ulong(p.ngpt) + ulong(ig)] = radlu;
    }
}

kernel void wrf_exact_rrtm_lw_rtrn_fluxes(
    constant WrfExactRrtmLwRtrnFluxesResources &r [[buffer(0)]],
    constant WrfExactRrtmLwRtrnFluxesParams &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= uint(p.ncell)) return;
    const int col = int(tid) / (p.nlay + 1);
    const int lev = int(tid) - col * (p.nlay + 1);
    const ulong base = ulong(col) * ulong(p.nlay + 1) * 140ul
                     + ulong(lev) * 140ul;
    Exact32 up_flux = 0.0f;
    Exact32 down_flux = 0.0f;
    for (int ig = 0; ig < 140; ++ig) {
        up_flux += reinterpret_cast<device const Exact32*>(r.up)[base + ulong(ig)];
        down_flux += reinterpret_cast<device const Exact32*>(r.down)[base + ulong(ig)];
    }
    up_flux = (up_flux * 0.5f) * Exact32(p.fluxfac);
    down_flux = (down_flux * 0.5f) * Exact32(p.fluxfac);
    Exact32 clear_up_flux = 0.0f, clear_down_flux = 0.0f;
    for (int ig = 0; ig < 140; ++ig) {
        clear_up_flux += reinterpret_cast<device const Exact32*>(r.clear_up)[base + ulong(ig)];
        clear_down_flux += reinterpret_cast<device const Exact32*>(r.clear_down)[base + ulong(ig)];
    }
    clear_up_flux = (clear_up_flux * 0.5f) * Exact32(p.fluxfac);
    clear_down_flux = (clear_down_flux * 0.5f) * Exact32(p.fluxfac);
    reinterpret_cast<device Exact32*>(r.clear_up_flux)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)] = clear_up_flux;
    reinterpret_cast<device Exact32*>(r.clear_down_flux)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)] = clear_down_flux;
    if (lev == p.nlay) {
        reinterpret_cast<device Exact32*>(r.htr)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)] = 0.0f;
        reinterpret_cast<device Exact32*>(r.clear_htr)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)] = 0.0f;
    }
    if (lev == 0) reinterpret_cast<device Exact32*>(r.glw)[col] = down_flux;
    if (lev == p.nmodel) reinterpret_cast<device Exact32*>(r.olr)[col] = up_flux;
    if (lev < p.nlay) {
        const Exact32 fnet0 = up_flux - down_flux;
        const ulong next_base = base + 140ul;
        Exact32 next_up = 0.0f;
        Exact32 next_down = 0.0f;
        for (int ig = 0; ig < 140; ++ig) {
            next_up += reinterpret_cast<device const Exact32*>(r.up)[next_base + ulong(ig)];
            next_down += reinterpret_cast<device const Exact32*>(r.down)[next_base + ulong(ig)];
        }
        const Exact32 fnet1 = (next_up * 0.5f) * Exact32(p.fluxfac) - (next_down * 0.5f) * Exact32(p.fluxfac);
        Exact32 clear_next_up = 0.0f, clear_next_down = 0.0f;
        for (int ig = 0; ig < 140; ++ig) {
            clear_next_up += reinterpret_cast<device const Exact32*>(r.clear_up)[next_base + ulong(ig)];
            clear_next_down += reinterpret_cast<device const Exact32*>(r.clear_down)[next_base + ulong(ig)];
        }
        const Exact32 clear_fnet1 = (clear_next_up * 0.5f) * Exact32(p.fluxfac) - (clear_next_down * 0.5f) * Exact32(p.fluxfac);
        const Exact32 dp = reinterpret_cast<device const Exact32*>(r.pz)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)]
                       - reinterpret_cast<device const Exact32*>(r.pz)[ulong(col) * ulong(p.nlay + 1) + ulong(lev + 1)];
        const Exact32 htr_day = Exact32(p.heatfac) * (fnet0 - fnet1) / dp;
        reinterpret_cast<device Exact32*>(r.htr)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)] = htr_day;
        reinterpret_cast<device Exact32*>(r.clear_htr)[ulong(col) * ulong(p.nlay + 1) + ulong(lev)] = Exact32(p.heatfac) * ((clear_up_flux - clear_down_flux) - clear_fnet1) / dp;
        const Exact32 htr = htr_day / 86400.0f;
        const int iy = col / p.nx;
        const int ix = col - iy * p.nx;
        if (lev < p.nmodel) reinterpret_cast<device Exact32*>(r.rthratenlw)[(ulong(lev) * ulong(p.ny) + ulong(iy))
                     * ulong(p.nx) + ulong(ix)] = htr / reinterpret_cast<device const Exact32*>(r.exner)[ulong(col)
                                                               * ulong(p.nmodel)
                                                               + ulong(lev)];
    }
}
