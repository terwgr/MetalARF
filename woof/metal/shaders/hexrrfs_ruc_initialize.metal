#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "hexrrfs_ruc_initialize_bindings.metalh"
using namespace ntc;

// Packed module_ruc_land table offsets from the retained TABLE_LAYOUT.
constant int ruc_lucats = 0, ruc_z0tbl = 356, ruc_slcats = 610;
constant int ruc_bb = 612, ruc_drysmc = 642;
constant int ruc_maxsmc = 702, ruc_refsmc = 732, ruc_satpsi = 762;

kernel void hexrrfs_ruc_initialize(constant HexrrfsRucInitializeResources &r [[buffer(0)]],
                                  constant HexrrfsRucInitializeParams &p [[buffer(1)]],
                                  uint c [[thread_position_in_grid]]) {
    if (c >= uint(p.solve))
        return;
    int soil = r.isltyp[c], vegetation = r.ivgtyp[c];
    int soil_count = as_type<int>(r.tables[ruc_slcats]);
    int vegetation_count = as_type<int>(r.tables[ruc_lucats]);
    if (soil < 1 || soil > soil_count || soil_count > 30
            || vegetation < 1 || vegetation > vegetation_count || vegetation_count > 50) {
        atomic_fetch_or_explicit(reinterpret_cast<device atomic_int *>(r.error), 1,
                                 memory_order_relaxed);
        return;
    }
    int s = soil - 1;
    r.znt[c] = r.tables[ruc_z0tbl + vegetation - 1];
    if (F(r.xice[c]) > F(0.f) || soil == 14) {
        bool ice = F(r.xice[c]) > F(0.f);
        for (int l = 0; l < p.nsl; ++l) {
            int i = l * p.nc + int(c);
            r.smfr3d[i] = ice ? 1.f : 0.f;
            r.sh2o[i] = ice ? 0.f : 1.f;
        }
        r.mavail[c] = 1.f;
        return;
    }
    F qmin = F(r.tables[ruc_drysmc + s]);
    F dqm = tc_sub(F(r.tables[ruc_maxsmc + s]), qmin);
    F ref = F(r.tables[ruc_refsmc + s]);
    F psis = -F(r.tables[ruc_satpsi + s]);
    F bclh = F(r.tables[ruc_bb + s]);
    F available = tc_div(tc_sub(F(r.smois[c]), qmin), tc_sub(ref, qmin));
    available = F(1.f) < available ? F(1.f) : available;
    available = F(.00001f) > available ? F(.00001f) : available;
    r.mavail[c] = as_type<float>(available.bits);
    for (int l = 0; l < p.nsl; ++l) {
        int i = l * p.nc + int(c);
        F t = F(r.tslb[i]), moist = F(r.smois[i]);
        F unfrozen = moist, frozen = F(0.f);
        if (gfk_log(tc_div(t, F(273.15f))) < F(0.f)) {
            F cooling = tc_div(tc_mul(F(3.35e5f), tc_sub(t, F(273.15f))), t);
            F suction = tc_div(tc_div(cooling, F(9.81f)), psis);
            unfrozen = tc_mul(tc_add(dqm, qmin), gfk_pow(suction, tc_div(F(-1.f), bclh)));
            unfrozen = F(0.f) > unfrozen ? F(0.f) : unfrozen;
            unfrozen = unfrozen < moist ? unfrozen : moist;
            frozen = tc_div(tc_sub(moist, unfrozen), F(.9f));
        }
        r.sh2o[i] = as_type<float>(unfrozen.bits);
        r.smfr3d[i] = as_type<float>(frozen.bits);
    }
}
