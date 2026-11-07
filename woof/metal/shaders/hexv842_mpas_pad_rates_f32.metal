#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_pad_rates_f32_bindings.metalh"
kernel void hexv842_mpas_pad_rates_f32(
    constant Hexv842MpasPadRatesF32Resources &r [[buffer(0)]],
    constant Hexv842MpasPadRatesF32Params &p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    int i=int(tid), width=p.ncells+1;
    if(i>=p.nlev*width) return;
    int cell=i%width, level=i/width;
    r.out[i]=cell==p.ncells ? 0.0f : r.rates[level*p.ncells+cell];
}
