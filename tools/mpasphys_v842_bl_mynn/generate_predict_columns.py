"""Generate native MYNN predictor with per-column surface quantities."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[2]
old='hexv842_mpas_bl_mynn_predict_f32';name='hexv842_mpas_bl_mynn_predict_columns_f32'
s=(root/f'woof/metal/shaders/{old}.metal').read_text().replace(old,name).replace('Hexv842MpasBlMynnPredictF32','Hexv842MpasBlMynnPredictColumnsF32')
for n in ('ust','flt','flq','pmz','phh'):s=s.replace(f'float {n}=cfg.{n};',f'float {n}=r.{n}[col];')
for n in ('s_aw','s_awqke'):s=s.replace(f'#define {n}(i) r.{n}[ulong(col)*nl+((i)-1)]',f'#define {n}(i) r.{n}[ulong(col)*(nl+1)+((i)-1)]')
fragment=json.loads((root/f'woof/metal/signature_fragments/{old}.json').read_text())['kernels'][0]
fragment.update(name=name,module=name,source=f'woof/metal/shaders/{name}.metal')
fragment['resources'] += [[n,'float','read'] for n in ('ust','flt','flq','pmz','phh')]
fragment['shapes'].update({n:[['any_1d']] for n in ('ust','flt','flq','pmz','phh')})
fragment['parameters']=[x for x in fragment['parameters'] if x[0] not in ('ust','flt','flq','pmz','phh')]
(root/f'woof/metal/shaders/{name}.metal').write_text(s)
(root/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps({'kernels':[fragment]},indent=2)+'\n')
