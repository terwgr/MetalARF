"""Bind the native SW solar-flux body with source upper-layer guards."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
family=(root/'woof/metal/shaders/rrtmg_legacy_sw_family.metal').read_text()
bodies=(root/'woof/metal/shaders/rrtmg_legacy_sw_bodies.metalh').read_text()
body=bodies[:bodies.index('// ---------------------------------------------------------------------------\n// cldprmc_sw')]
start=body.index('static inline void rsw_sfluxzen_body(')
front,flux=body[:start],body[start:]
for marker in ('case 17: {','case 28: {','default:   // 29'):
    assert flux.count(marker)==1
    flux=flux.replace(marker,marker+'\n        if (lsf <= laytrop) break;')
body=(front+flux+'\n#endif\n').replace('WOOF_RRTMG_LEGACY_SW_BODIES_METALH','HEXV842_SW_SFLUX_BODY_METALH')
(root/'woof/metal/shaders/hexv842_sw_sflux_body.metalh').write_text(body)
prefix=family[:family.index('#include "rrtmg_legacy_sw_bodies.metalh"')]
prefix+='\n#include "hexv842_sw_sflux_body.metalh"\n#include "hexv842_sw_sfluxzen_f32_bindings.metalh"\n'
a=family.index('kernel void rrtmg_legacy_sw_sfluxzen(');b=family.index('kernel void rrtmg_legacy_sw_spcvmc(',a)
entry=family[a:b].replace('rrtmg_legacy_sw_sfluxzen','hexv842_sw_sfluxzen_f32').replace('RrtmgLegacySwSfluxzen','Hexv842SwSfluxzenF32')
(root/'woof/metal/shaders/hexv842_sw_sfluxzen_f32.metal').write_text((prefix+entry).rstrip()+chr(10))
fragment=json.loads((root/'woof/metal/signature_fragments/rrtmg_legacy_sw_family.json').read_text())
k=next(k for k in fragment['kernels'] if k['name']=='rrtmg_legacy_sw_sfluxzen').copy()
k.update(name='hexv842_sw_sfluxzen_f32',module='hexv842_sw_sfluxzen_f32',source='woof/metal/shaders/hexv842_sw_sfluxzen_f32.metal',cuda_source='woof/hex/mpasphys/rrtmg_sw_v842.py',cuda_symbol='rsw_sfluxzen_b')
(root/'woof/metal/signature_fragments/hexv842_sw_sfluxzen_f32.json').write_text(json.dumps({'kernels':[k]},indent=2)+'\n')
