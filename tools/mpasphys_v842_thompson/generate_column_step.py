"""Generate the native Thompson column step with depth-sized workspaces."""
from pathlib import Path
import argparse,json,re,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tools/mp28_metal'),str(Path(__file__).resolve().parent)]
from parse import split_top,def_name
from translate import convert_table
from generate import convert_kernel,POLICIES
from column_translation import translated_struct
from woof.hex.mpasphys import thompson_v842 as classic
from woof.hex.mpasphys import thompson_aero_v842 as aerosol
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--aerosol', action='store_true', help='generate the aerosol-aware specialization')
native = aerosol if parser.parse_args().aerosol else classic
name='hexv842_mpas_thompson_column_step_f32' if native is classic else 'hexv842_mpas_thompson_aerosol_column_step_f32'
source=native.cuda_source(device_math_source='// supplied by Metal support',max_levels=256,numerics='strict')
source=source.replace('#define THOMPSON_COLUMN_KMAX 256','#define THOMPSON_COLUMN_KMAX TCKMAX',1)
pre=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=source.encode(),capture_output=True,check=True).stdout.decode()
items=split_top(pre);tables='\n'.join(convert_table(i) for i in items if '__constant__' in i)
constants,body,counts=translated_struct(native)
constants=constants.replace('constant F ','constant float ')
body=re.sub(r'(constant F \* \w+ = )(tc_\w+) ;',r'\1(constant F*)\2 ;',body)
body=body.replace('device Soft8 * tc_wsd ;','device Soft8 * tc_wsd ;\ndevice bool * tc_wsb ;',1)
item=next(i for i in items if def_name(i)=='mpas_thompson_step')
code,params=convert_kernel(item,name,'ntc',[])
code=code.replace('&p [[buffer(1)]]','&params [[buffer(1)]]').replace(' = p.',' = params.')
code=re.sub(r'\bfloat\b','F',code)
code=re.sub(r'(device (?:const )?F \*\w+ = )(r\.\w+);',r'\1(device F *)\2;',code)
code=code.replace('int col = int ( gid ) ;','int slot = int ( gid ) ; int col = params.first + slot ; if(slot>=params.count)return;')
code=code.replace('ThompsonColumn m ;','ThompsonColumn m ; m.nz=nz; m.tc_wsf=(device F*)r.ws_f+ulong(slot)*93*(nz+1); m.tc_wsd=(device Soft8*)r.ws_d+ulong(slot)*64*(nz+1); m.tc_wsb=(device bool*)r.ws_b+ulong(slot)*7*(nz+1);')
code=code.replace('m . r ,','f32_word(1133479199u) ,').replace('m . nain1','f32_word(1223959552u)')
def declarations(match):
    names=re.findall(r'(\w+) \[ TCKMAX \]',match.group())
    return ' '.join('TCKF ( %s , TCKMAX ) ;'%n for n in names)
code=re.sub(r'^F (?:\w+ \[ TCKMAX \] ,? ?)+;$',declarations,code,flags=re.M)
if code.count('TCKF (')!=20 or counts!=(73,64,7):raise ValueError('native column workspace layout changed')
resources=[];parameters=[]
for p in params:
    if p['pointer']:
        resources.append([p['name'],'uint64' if p['base']=='double' else 'float','read' if p['const'] else 'read_write'])
    else:parameters.append([p['name'],'float32' if p['base']=='float' else 'int32'])
if native is aerosol:
    packed_names=('nc','nifa','nwfa','nifa2d','nwfa2d')
    offsets=('0ul','ulong(params.nz)*ulong(params.columns)',
             '2ul*ulong(params.nz)*ulong(params.columns)',
             '3ul*ulong(params.nz)*ulong(params.columns)',
             '3ul*ulong(params.nz)*ulong(params.columns)+ulong(params.columns)')
    for field,offset in zip(packed_names,offsets):
        pattern=rf'(device (?:const )?F \*{field} = )\(device F \*\)r\.{field};'
        code,n=re.subn(pattern,rf'\1(device F *)r.aerosol + ({offset});',code)
        if n!=1:raise ValueError('aerosol pointer translation changed: '+field)
    resources=[r for r in resources if r[0] not in packed_names]
    resources.append(['aerosol','float','read_write'])
resources += [['ws_f','float','read_write'],['ws_d','uint64','read_write'],['ws_b','uint32','read_write']]
parameters += [['first','int32'],['count','int32']]
header=f'''#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
using namespace ntc;
#define TCKMAX nz
#define TCSPACE device
#define TCF(n,sz) device F * n=tc_wsf;tc_wsf+=(sz)
#define TCD(n,sz) device Soft8 * n=tc_wsd;tc_wsd+=(sz)
#define TCB(n,sz) device bool * n=tc_wsb;tc_wsb+=(sz)
#define TCKF(n,sz) device F * n=m.tc_wsf;m.tc_wsf+=(sz)
static inline bool tc_any(device bool* a,int lo,int hi){{for(int k=lo;k<=hi;k++)if(a[k-lo])return true;return false;}}
'''
shader=header+tables+constants+'\n'+body+f'\n#include "{name}_bindings.metalh"\n'+code
(ROOT/f'woof/metal/shaders/{name}.metal').write_text(shader)
spec={'kernels':[{'name':name,'module':name,'source':f'woof/metal/shaders/{name}.metal','cuda_source':('woof/hex/mpasphys/thompson_v842.py' if native is classic else 'woof/hex/mpasphys/thompson_aero_v842.py'),'cuda_symbol':'mpas_thompson_step','resources':resources,'shapes':{r[0]:[['any_1d']] for r in resources},'parameters':parameters,'threads_per_group':32,'layout_version':1,'arithmetic_policies':POLICIES}]}
(ROOT/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(spec,indent=2)+'\n')
print('native Thompson resources',len(resources),'workspace planes',93,64,7)
