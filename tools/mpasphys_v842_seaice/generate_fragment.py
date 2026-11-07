"""Generate the signature fragment for the isolated native sea-ice kernels."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[2]
params = [(x, 'int32') for x in ('n', 'nsoil')] + [(x, 'float32') for x in ('dt', 'threshold', 'albedo_default', 'thickness_default', 'snowdepth_max', 'snowdepth_min')] + [(x, 'int32') for x in ('albedo_opt', 'thickness_opt', 'snowdepth_opt', 'frpcpn', 'optional_noah', 'microp', 'convection')]
kernels=[]
for op, resources in [('from', [('fields','float','read_write')]), ('column', [('fields','float','read_write'),('soil','float','read_write'),('p8w2','float','read'),('errors','int32','read_write'),('ran','int32','read_write')]), ('to', [('fields','float','read_write')]), ('state', [('ran','int32','read'),('state','int32','read_write')])]:
    name = f'hexv842_seaice_{op}_' + ('i32' if op == 'state' else 'f32')
    kernels.append(dict(name=name, module='hexv842_seaice_f32', source='woof/metal/shaders/hexv842_seaice_f32.metal', cuda_source='woof/hex/mpasphys/seaice_v842.py', cuda_symbol={'from':'seaice_from_MPAS','column':'seaice_noah','to':'seaice_to_MPAS','state':'seaice_module_state'}[op], resources=resources, shapes={r[0]: [['any_1d']] for r in resources}, parameters=([('n','int32'),('i_start','int32'),('j_index','int32')] if op == 'state' else params), threads_per_group=128, layout_version=1, arithmetic_policies=['default']))
(root/'woof/metal/signature_fragments/hexv842_seaice_f32.json').write_text(json.dumps({'kernels':kernels},indent=2)+'\n')
