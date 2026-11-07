#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_sw_kiss_guard_i32_bindings.metalh"
kernel void hexv842_sw_kiss_guard_i32(constant Hexv842SwKissGuardI32Resources &r [[buffer(0)]],
 constant Hexv842SwKissGuardI32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int c=int(tid);if(c>=p.nc)return;
 float a=mpas_mul(r.play[c*p.nl],100.f),b=mpas_mul(r.play[c*p.nl+1],100.f);
 r.invalid[c]=soft32_less(as_type<uint>(a),as_type<uint>(b))?1:0;
}
