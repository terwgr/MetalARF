"""Generate the native RP7-corrected MYNN tendency kernel."""
from pathlib import Path
import json
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_TEXT_CORRECTIONS
root=Path(__file__).resolve().parents[2]
old='hexv842_mpas_bl_mynn_tendencies_f32';name='hexv842_mpas_bl_mynn_tendencies_correct_f32'
text=(root/f'woof/metal/shaders/{old}.metal').read_text()
text=text[:text.index('\nkernel void hexv842_mpas_bl_mynn_tendencies_pack0_f32')]
for suffix in ('pack0','pack1','scatter'):
    text=text.replace(f'#include "hexv842_mpas_bl_mynn_tendencies_{suffix}_f32_bindings.metalh"\n','')
text=text.replace(old+'_bindings.metalh',name+'_bindings.metalh').replace('Hexv842MpasBlMynnTendenciesF32','Hexv842MpasBlMynnTendenciesCorrectF32').replace('kernel void '+old+'(', 'kernel void '+name+'(')
original,corrected,count=MYNN_TEXT_CORRECTIONS['RP7'][0];assert count==1 and text.count(original)==1
text=text.replace(original,corrected).replace('float delt=cfg.delt;','float grav=cfg.gravity;\nfloat delt=cfg.delt;')
spec=json.loads((root/f'woof/metal/signature_fragments/{old}.json').read_text())['kernels'][0]
spec.update(name=name,module=name,source=f'woof/metal/shaders/{name}.metal');spec['parameters'].append(['gravity','float32'])
(root/f'woof/metal/shaders/{name}.metal').write_text(text)
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps({'kernels':[spec]},indent=2)+'\n')
