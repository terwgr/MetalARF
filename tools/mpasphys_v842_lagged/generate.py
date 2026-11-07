"""Generate native lagged-tendency kernels and binding metadata."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[2]
names=['hexv842_mpas_add_lagged_f32','hexv842_mpas_zero_lagged_f32','hexv842_mpas_recover_cells_f32']
base=json.loads((root/'woof/metal/signature_fragments/hexv841_recover_cells_f32.json').read_text())['kernels'][0]
base.update(name=names[2],module=names[2],source='woof/metal/shaders/hexv842_mpas_lagged_f32.metal')
base['resources'].insert(4,['diabatic_theta','float','read'])
base['resources'].append(['invalid','int32','read_write'])
base['shapes']['invalid']=[['any_1d']]
base['shapes']['diabatic_theta']=[['any_1d']]
base['parameters'].extend([['has_diabatic','int32'],['dt','float32']])
def entry(name,symbol,resources,parameters):
 return dict(name=name,module=name,source='woof/metal/shaders/hexv842_mpas_lagged_f32.metal',cuda_source='woof/hex/cuda_driver.py',cuda_symbol=symbol,resources=resources,shapes={r[0]:[['any_1d']] for r in resources},parameters=parameters,threads_per_group=128,layout_version=1,arithmetic_policies=['default'])
kernels=[entry(names[0],'add_lagged_diabatic_theta_v842_f32', [['rho','float','read'],['diabatic_theta','float','read'],['tend_rt','float','read_write']], [['nlev','int32'],['ncells','int32']]),entry(names[1],'zero_speczone_lagged_diabatic_theta_v842_f32',[['spec_cells','int32','read'],['diabatic_theta','float','read_write']],[['nlev','int32'],['ncells','int32'],['nspec','int32']]),base]
(root/'woof/metal/signature_fragments/hexv842_mpas_lagged_f32.json').write_text(json.dumps({'kernels':kernels},indent=2)+'\n')
