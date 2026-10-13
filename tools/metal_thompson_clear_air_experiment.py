"""Bounded device replay of classic Thompson with identical clear-air inputs."""
from pathlib import Path
import argparse,json,sys
import numpy as np
SRC=Path(__file__).resolve().parents[1];sys.path.insert(0,str(SRC))
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
from woof.core.thompson_runtime import load_classic_device_tables
from woof.metal.thompson_frozen_vapor_network import submit_frozen_vapor_network

parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--table-root',type=Path,required=True);parser.add_argument('--initial-audit',type=Path)
args=parser.parse_args()
audit=json.loads(args.initial_audit.read_text()) if args.initial_audit else {'extra_nucleation_rows':[]}
cases=[{'name':f"saved_initial_{r['i']}_{r['j']}_{r['k']}",'temperature':r['temperature_k'],'pressure':r['pressure_pa'],'qv':r['qv']} for r in audit['extra_nucleation_rows']]
cases += [{'name':'ice_supersaturation_below_correct_nucleation_threshold','temperature':230.,'pressure':26000.,'qv':.00025},
          {'name':'cold_dry_control','temperature':240.,'pressure':30000.,'qv':.0001},
          {'name':'warm_dry_control','temperature':300.,'pressure':96000.,'qv':.01}]
names=['qi','ni','qs','qg','qr','nr','qc','temperature','pressure','qv','graupel_number_shadow','snow_velocity_boost']
inputs={name:np.array([case.get(name,0.) for case in cases],np.float32) for name in names}
with MetalContext() as ctx:
    xp=MetalArrayModule(ctx)
    owner=load_classic_device_tables(args.table_root,backend=xp,cache=False)
    arrays={name:xp.asarray(value) for name,value in inputs.items()}
    submit_frozen_vapor_network(arrays['qi'],arrays['ni'],arrays['qs'],arrays['qg'],arrays['qr'],arrays['nr'],arrays['temperature'],arrays['pressure'],arrays['qv'],owner,15.,qc=arrays['qc'],graupel_number_shadow=arrays['graupel_number_shadow'],snow_velocity_boost=arrays['snow_velocity_boost'])
    outputs={name:value.to_numpy() for name,value in arrays.items()}
manifest=json.loads((SRC/'woof/libexec/metal/METAL_BUNDLE.json').read_text())
rows=[]
for index,case in enumerate(cases):
    row={'case':case['name'],'input':{n:float(v[index]) for n,v in inputs.items()},'output':{n:float(v[index]) for n,v in outputs.items()}}
    row['unchanged_physical_fields']=all(np.array_equal(inputs[n][index:index+1],outputs[n][index:index+1]) for n in names if n!='snow_velocity_boost')
    rows.append(row)
assert all(row['unchanged_physical_fields'] for row in rows), 'Clear-air CUDA early-exit contract violated'
result={'method':'One production classic Thompson cold-source launch at dt=15s, original canonical tables, identical FP32 temperature/pressure/qv inputs and zero hydrometeors. Saved initial inputs selected because the bad coefficient changes the nucleation gate. Correct CUDA gate returns before process arithmetic for every selected cell, so unchanged physical fields are the exact reference; snow velocity boost is deliberately reset to 1. No forecast integration.','bundle':manifest,'cases':len(rows),'changed_cases':sum(not row['unchanged_physical_fields'] for row in rows),'expect_clear_unchanged':True,'rows':rows}
args.out.write_text(json.dumps(result,indent=2)+'\n')
np.savez(args.out.with_suffix('.npz'),**{f'input_{n}':v for n,v in inputs.items()},**{f'output_{n}':v for n,v in outputs.items()})
print('Cases',len(rows),'changed',result['changed_cases'],flush=True)
for row in rows[-3:]:print(json.dumps(row),flush=True)
