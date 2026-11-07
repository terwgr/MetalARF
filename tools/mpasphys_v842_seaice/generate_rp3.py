"""Generate the native sea-ice fractional temperature correction kernel."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
name = 'hexv842_mpas_seaice_rp3_f32'
shader = '''#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_mpas_seaice_rp3_f32_bindings.metalh"
#pragma clang fp contract(off)

kernel void hexv842_mpas_seaice_rp3_f32(
    constant Hexv842MpasSeaiceRp3F32Resources &r [[buffer(0)]],
    constant Hexv842MpasSeaiceRp3F32Params &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= uint(p.n)) return;
    float x = r.ice[i], blended = r.skin[i];
    if (x >= p.threshold && x <= 1.0f) {
        float sea = fmax(r.sst[i], 271.4f);
        float ice = cu_div(mpas_sub(blended, mpas_mul(mpas_sub(1.0f, x), sea)), x);
        ice = fmax(ice, 221.4f);
        if (x < 0.2f && blended < 253.15f) ice = 253.15f;
        if (x < 0.1f && blended < 263.15f) ice = 263.15f;
        r.fields[p.sea_index * p.n + i] = sea;
        r.fields[p.ice_index * p.n + i] = ice;
    }
}
'''
resources = [('fields', 'float', 'read_write'), ('sst', 'float', 'read'),
             ('skin', 'float', 'read'), ('ice', 'float', 'read')]
row = dict(name=name, module=name, source=f'woof/metal/shaders/{name}.metal',
           cuda_source='woof/hex/mpasphys/sf_mynn_v842.py', cuda_symbol='sf_mynn_sea',
           resources=resources, shapes={r[0]: [['any_1d']] for r in resources},
           parameters=[('n','int32'), ('sea_index','int32'), ('ice_index','int32'),
                       ('threshold','float32')], threads_per_group=128,
           layout_version=1, arithmetic_policies=['default'])
(root / f'woof/metal/shaders/{name}.metal').write_text(shader)
(root / f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps({'kernels':[row]}, indent=2) + '\n')
