"""Generate the native KISS bottom-to-top pressure guard."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
s='''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "hexv842_sw_kiss_guard_i32_bindings.metalh"
kernel void hexv842_sw_kiss_guard_i32(constant Hexv842SwKissGuardI32Resources &r [[buffer(0)]],
 constant Hexv842SwKissGuardI32Params &p [[buffer(1)]],uint tid [[thread_position_in_grid]]) {
 int c=int(tid);if(c>=p.nc)return;
 float a=mpas_mul(r.play[c*p.nl],100.f),b=mpas_mul(r.play[c*p.nl+1],100.f);
 r.invalid[c]=soft32_less(as_type<uint>(a),as_type<uint>(b))?1:0;
}
'''
(root/'woof/metal/shaders/hexv842_sw_kiss_guard_i32.metal').write_text(s)
res=[('play','float','read'),('invalid','int32','read_write')]
k=dict(name='hexv842_sw_kiss_guard_i32',module='hexv842_sw_kiss_guard_i32',source='woof/metal/shaders/hexv842_sw_kiss_guard_i32.metal',cuda_source='woof/hex/mpasphys/rrtmg_sw_v842.py',cuda_symbol='_check_kiss_pressure',resources=res,shapes={r[0]:[['any_1d']] for r in res},parameters=[('nc','int32'),('nl','int32')],threads_per_group=128,layout_version=1,arithmetic_policies=['default'])
(root/'woof/metal/signature_fragments/hexv842_sw_kiss_guard_i32.json').write_text(json.dumps({'kernels':[k]},indent=2)+'\n')
