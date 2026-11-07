#include <metal_stdlib>
#include "exact_binary32.metalh"
#include "rrtmgp_exact_cloud_optics_bindings.metalh"
using namespace metal;

static inline Exact32 cloud_lookup(device const Exact32 *table, int lo,
                                   int stride, Exact32 fraction) {
    const Exact32 value = table[lo];
    return value + fraction * (table[lo + stride] - value);
}

static inline int cloud_size(Exact32 pos, int count) {
    if (!(pos > Exact32(0.0f))) return 0;
    if (pos >= Exact32(float(count - 2))) return count - 2;
    return int(as_type<float>(pos.bits));
}

kernel void rrtmgp_exact_cloud_optics(
    constant RrtmgpExactCloudOpticsResources &r [[buffer(0)]],
    constant RrtmgpExactCloudOpticsParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.ncell * p.nband)) return;
    const int cell = int(gid) / p.nband;
    const int band = int(gid) - cell * p.nband;
    device const Exact32 *clwp = reinterpret_cast<device const Exact32 *>(r.clwp);
    device const Exact32 *ciwp = reinterpret_cast<device const Exact32 *>(r.ciwp);
    device const Exact32 *reliq = reinterpret_cast<device const Exact32 *>(r.reliq);
    device const Exact32 *dgice = reinterpret_cast<device const Exact32 *>(r.dgice);
    Exact32 ltau(0.0f), lscatter(0.0f), lgscatter(0.0f);
    Exact32 itau(0.0f), iscatter(0.0f), igscatter(0.0f);
    if (clwp[cell] > Exact32(0.0f)) {
        const Exact32 pos = (reliq[cell] - Exact32(p.liq_offset)) / Exact32(p.liq_step);
        const int size = cloud_size(pos, p.nliq);
        const Exact32 fraction = pos - Exact32(float(size));
        const int lo = size * p.nband + band;
        ltau = clwp[cell] * cloud_lookup(reinterpret_cast<device const Exact32 *>(r.extliq), lo, p.nband, fraction);
        lscatter = ltau * cloud_lookup(reinterpret_cast<device const Exact32 *>(r.ssaliq), lo, p.nband, fraction);
        lgscatter = lscatter * cloud_lookup(reinterpret_cast<device const Exact32 *>(r.asyliq), lo, p.nband, fraction);
    }
    if (ciwp[cell] > Exact32(0.0f)) {
        const Exact32 pos = (dgice[cell] - Exact32(p.ice_offset)) / Exact32(p.ice_step);
        const int size = cloud_size(pos, p.nice);
        const Exact32 fraction = pos - Exact32(float(size));
        const int lo = (size * p.nband + band) * p.nrgh + 1;
        const int stride = p.nband * p.nrgh;
        itau = ciwp[cell] * cloud_lookup(reinterpret_cast<device const Exact32 *>(r.extice), lo, stride, fraction);
        iscatter = itau * cloud_lookup(reinterpret_cast<device const Exact32 *>(r.ssaice), lo, stride, fraction);
        igscatter = iscatter * cloud_lookup(reinterpret_cast<device const Exact32 *>(r.asyice), lo, stride, fraction);
    }
    const Exact32 total = ltau + itau;
    const Exact32 scatter = lscatter + iscatter;
    reinterpret_cast<device Exact32 *>(r.tau)[gid] = total;
    reinterpret_cast<device Exact32 *>(r.ssa)[gid] = scatter / exact_max(Exact32(0x1p-23f), total);
    reinterpret_cast<device Exact32 *>(r.asym)[gid] = (lgscatter + igscatter) / exact_max(Exact32(0x1p-23f), scatter);
}
