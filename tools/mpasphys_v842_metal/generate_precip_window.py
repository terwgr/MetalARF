"""Translate native interval surface precipitation arithmetic without contraction."""
from pathlib import Path
import json,re
from woof.hex.mpasphys.precip_window_v842 import SOURCE
ROOT=Path(__file__).resolve().parents[2]
fields=('rainncv','raincv','snowncv','graupelncv','sr')
windows=('rainncv_bl','raincv_bl','snowncv_bl','graupelncv_bl','frozen_bl','sr_bl')
for phase in ('add','reset'):
    original='precip_window_'+phase
    start=SOURCE.index('{',SOURCE.index('void '+original+'('));end=SOURCE.index('\n}',start)
    body=SOURCE[start+1:end]
    body=body.replace('blockIdx.x * blockDim.x + threadIdx.x','int(gid)')
    if phase=='add':
        replacements={
            'rainncv_bl[i] + rainncv[i]':'mpas_add(rainncv_bl[i],rainncv[i])',
            'sr[i] * rainncv[i]':'mpas_mul(sr[i],rainncv[i])',
            'frozen_bl[i] + frozen':'mpas_add(frozen_bl[i],frozen)',
            'raincv_bl[i] + raincv[i]':'mpas_add(raincv_bl[i],raincv[i])',
            'snowncv_bl[i] + snowncv[i]':'mpas_add(snowncv_bl[i],snowncv[i])',
            'graupelncv_bl[i] + graupelncv[i]':'mpas_add(graupelncv_bl[i],graupelncv[i])',
            'rainncv_bl[i] > 0.0f':'soft32_less(0u,as_type<uint>(rainncv_bl[i]))',
            'frozen_bl[i] / rainncv_bl[i]':'mpas_div(frozen_bl[i],rainncv_bl[i])',
            'q < 1.0f':'soft32_less(as_type<uint>(q),as_type<uint>(1.0f))',
        }
        for old,new in replacements.items():
            assert body.count(old)==1,old
            body=body.replace(old,new)
    names=(*fields,*windows,'nsteps_bl') if phase=='add' else (*windows[:-1],'nsteps_bl')
    parameters=('n','have_nc','have_c','have_sg') if phase=='add' else ('n',)
    for name in names:body=re.sub(r'\b'+name+r'\[','r.'+name+'[',body)
    for name in parameters:body=re.sub(r'\b'+name+r'\b','p.'+name,body)
    entry='hexv842_mpas_precip_window_'+phase+'_f32';title=''.join(word.title() for word in entry.split('_'))
    shader='#include <metal_stdlib>\nusing namespace metal;\n#include "hex_ieee32.metalh"\n#include "'+entry+'_bindings.metalh"\n#pragma clang fp contract(off)\n'
    shader+='kernel void '+entry+'(constant '+title+'Resources &r [[buffer(0)]],constant '+title+'Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {\n'+body+'\n}\n'
    (ROOT/'woof/metal/shaders'/Path(entry+'.metal')).write_text(shader)
    resources=[[name,'int32' if name=='nsteps_bl' else 'float32','read' if name in fields else 'read_write'] for name in names]
    spec={'kernels':[dict(name=entry,module=entry,source='woof/metal/shaders/'+entry+'.metal',cuda_source='woof/hex/mpasphys/precip_window_v842.py',cuda_symbol=original,resources=resources,shapes={name:[['any_1d']] for name in names},parameters=[[name,'int32'] for name in parameters],threads_per_group=128,layout_version=1,arithmetic_policies=['default'])]}
    (ROOT/'woof/metal/signature_fragments'/Path(entry+'.json')).write_text(json.dumps(spec,indent=2)+'\n')
