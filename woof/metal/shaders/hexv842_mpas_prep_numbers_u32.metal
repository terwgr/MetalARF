#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_prep_numbers_u32_bindings.metalh"
kernel void hexv842_mpas_prep_numbers_u32(
    constant Hexv842MpasPrepNumbersU32Resources &r [[buffer(0)]],
    constant Hexv842MpasPrepNumbersU32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int cell = int(tid);
    if (cell >= p.solved) return;
    for (int k=0; k<p.nlev; ++k) {
        const int i=k*p.ncells+cell;
        if (p.aerosol) { r.out_nifa[i]=r.nifa[i]; r.out_nwfa[i]=r.nwfa[i]; }
        if (p.mynn) {
            r.out_nc[i]=0u; r.out_ni[i]=0u; r.out_nifa[i]=0u; r.out_nwfa[i]=0u;
            if (p.ice) r.out_ni[i]=r.ni[i];
            if (p.trio) { r.out_nc[i]=r.nc[i]; r.out_nifa[i]=r.nifa[i]; r.out_nwfa[i]=r.nwfa[i]; }
        }
    }
}
