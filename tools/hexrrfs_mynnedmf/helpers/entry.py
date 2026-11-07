import os
import ast,re,json
from pathlib import Path
p=Path(__file__).resolve().parent
pin=os.environ.get('WOOF_MYNN_PIN', 'c6d08709')
source=Path(os.environ.get('WOOF_SOURCE_ROOT', Path.cwd())) / f'woof/hex/mpasphys/rrfs/bl_mynnedmf_{pin}.py'
s=next(ast.literal_eval(n.value) for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='MODULE_CUDA')
a=s.index('void rrfs_mynnedmf_column('); b=s.index(') {',a)
params=[]
for x in s[a+len('void rrfs_mynnedmf_column('):b].split(','):
 m=re.fullmatch(r'(unsigned long long|unsigned char|float|int)\s*(\*?)\s*(\w+)',x.strip());assert m,x
 params.append(m.groups())
end=s.index('\n}',b)
body=s[s.index('MynnText message;int error_flag=0;',b):end]
body=re.sub(r'\bfloat\b','Exact32',body)
resources=[(t,n) for t,star,n in params if star]
scalars=[(t,n) for t,star,n in params if not star]
text='#include "column_support.metalh"\n#include "column_typed.metalh"\nstruct Resources {\n'
for t,n in resources:text+=f'    device { {"float":"Exact32","int":"int","unsigned char":"uchar"}[t] } *{n};\n'
text+='};\nstruct Parameters {\n'
for t,n in scalars:text+=f'    { {"float":"float","int":"int","unsigned long long":"ulong"}[t] } {n};\n'
text+='};\nkernel void rrfs_mynnedmf_l1(constant Resources &r [[buffer(0)]],constant Parameters &parameters [[buffer(1)]],uint gid [[thread_position_in_grid]]) {\n'
for t,n in resources:text+=f'    device { {"float":"Exact32","int":"int","unsigned char":"uchar"}[t] } *{n}=r.{n};\n'
for t,n in scalars:text+=f'    { {"float":"Exact32","int":"int","unsigned long long":"ulong"}[t] } {n}=parameters.{n};\n'
text+='''    if(gid>=uint(batch_count)) return;
    int column_index=int(gid),index=batch_first+column_index;
    MynnArena arena{workspace+ulong(column_index)*workspace_stride,0};
    MynnColumn column{arena};
    column.active_i=its+index%(ite-its+1);
    column.active_j=jts+index/(ite-its+1);
'''+body+'\n}\n'
(p/'column_entry.metal').write_text(text)

arguments=next(ast.literal_eval(n.value) for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign) and getattr(n.targets[0],"id",None)=="ARGUMENTS")
resource_map=[]
for i,(t,n) in enumerate(resources):
 meta=arguments.get(n,{})
 resource_map.append({"index":i,"name":n,"kind":t,"dimensions":meta.get("dimensions"),"intent":meta.get("intent"),"optional":meta.get("optional")})
(p/"argument_map.json").write_text(json.dumps({"resources":resource_map,"parameters":[{"name":n,"kind":t} for t,n in scalars],"dispatch":"one thread per column; scratch offset = local gid * workspace_stride"},indent=2)+"\n")


"""Generate compact batch lane entry retaining original descriptor bounds."""
import json,re
from pathlib import Path
p=Path(__file__).resolve().parent
schema=json.loads((p/'argument_map.json').read_text())
s=(p/'column_entry.metal').read_text()
fields=[r for r in schema['resources'] if r['name'] not in ('workspace','error_flags')]
assert len(fields)==(87 if pin=='c942418' else 92)
start=s.index('struct Resources {');end=s.index('struct Parameters {',start)
s=s[:start]+'''struct Resources {
    device uint *fields [[id(0)]];
    device uint *workspace [[id(1)]];
    device int *error_flags [[id(2)]];
    device int *offsets [[id(3)]];
};
'''+s[end:]
s=s.replace('    ulong workspace_stride;','    int workspace_stride;')
pos=s.index('};\nkernel void rrfs_mynnedmf_l1',s.index('struct Parameters'))
s=s[:pos]+'    int presence0; int presence1; int presence2;\n'+s[pos:]
s=s.replace('kernel void rrfs_mynnedmf_l1(', 'kernel void rrfs_mynnedmf_packed_l1(')
a=s.index('    device uchar *workspace=r.workspace;')
b=s.index('    int batch_first=parameters.batch_first;',a)
code='''    device uchar *workspace=reinterpret_cast<device uchar *>(r.workspace);
    device int *error_flags=r.error_flags;
'''
for i,entry in enumerate(fields):
 n=entry['name'];t='int' if entry['kind']=='int' else 'Exact32';mask=f'uint(parameters.presence{i//32})'
 code+=f'    device {t} *{n}=({mask}&(1u<<{i%32})) ? reinterpret_cast<device {t} *>(r.fields+r.offsets[{i}]+gid) : nullptr;\n'
s=s[:a]+code+s[b:]
s=s.replace('    ulong workspace_stride=parameters.workspace_stride;', '    int workspace_stride=parameters.workspace_stride;')
s=s.replace('    if(gid>=uint(batch_count)) return;','')
needle='uint gid [[thread_position_in_grid]]) {\n'
assert s.count(needle)==1
s=s.replace(needle,needle+'    if(gid>=uint(parameters.batch_count)) return;\n')
for entry in fields:
 n=entry['name']; dims=entry['dimensions']; assert dims,n
 strides=[]
 for dim in dims:
  if dim=='kms:kme':strides.append('batch_count')
  elif dim in ('nchem','ndvel'):
   strides.append('batch_count*(kme-kms+1)' if 'kms:kme' in dims else 'batch_count')
  else:strides.append('0')
 for i,x in enumerate(strides):
  pattern=rf'(a_{re.escape(n)}\.stride\[{i}\]=)[^;]+;'
  s,count=re.subn(pattern,lambda m:m.group(1)+x+';',s)
  assert count==1,(n,i,count)
(p/'column_packed.metal').write_text(s)
(p/'packed_abi.json').write_text(json.dumps({'resources':[['fields','uint32','read_write'],['workspace','uint32','read_write'],['error_flags','int32','read_write'],['offsets','int32','read']], 'fields':[r['name'] for r in fields], 'parameters':[{'name':x['name'],'kind':('int' if x['name']=='workspace_stride' else x['kind'])} for x in schema['parameters']]+[{'name':'presence'+str(i),'kind':'int'} for i in range(3)]},indent=2)+'\n')


"""Emit named source products; ABI remains generated from the registry."""
import json,re
from pathlib import Path
p=Path(__file__).resolve().parent;d=p/'bundle';d.mkdir(exist_ok=True)
renames={
 'column_packed.metal':'hexrrfs_mynnedmf_columns.metal',
 'column_typed.metalh':'hexrrfs_mynnedmf_column.metalh',
 'column_support.metalh':'hexrrfs_mynnedmf_support.metalh',
 'array_support.metalh':'hexrrfs_mynnedmf_array.metalh',
 'mynn64.metalh':'hexrrfs_mynnedmf_real64.metalh',
 'mynn_math32.metalh':'hexrrfs_mynnedmf_math32.metalh',
 'mynn_pow_fma.metalh':'hexrrfs_mynnedmf_pow.metalh',
 'mynn_erf.metalh':'hexrrfs_mynnedmf_erf.metalh',
}
if pin=='c942418':
 renames['column_packed.metal']='hexrrfs_mynnedmf_old_columns.metal'
 renames['column_typed.metalh']='hexrrfs_mynnedmf_old_column.metalh'
for source,target in renames.items():
 if pin=='c942418' and source not in ('column_packed.metal', 'column_typed.metalh'):
  continue
 s=(p/source).read_text()
 if pin=='c942418':
  s=s.replace('MynnColumn', 'MynnOldColumn')
 for a,b in renames.items():s=s.replace(f'#include "{a}"',f'#include "{b}"')
 if source=='column_packed.metal':
  a=s.index('struct Resources {');b=s.index('kernel void rrfs_mynnedmf_packed_l1',a)
  binding = 'hexrrfs_mynnedmf_old_columns' if pin=='c942418' else 'hexrrfs_mynnedmf_columns'
  s=s[:a]+f'#include "{binding}_bindings.metalh"\n'+s[b:]
  s=s.replace('rrfs_mynnedmf_packed_l1',binding)
  type_prefix = 'HexrrfsMynnedmfOldColumns' if pin=='c942418' else 'HexrrfsMynnedmfColumns'
  s=s.replace('constant Resources &r',f'constant {type_prefix}Resources &r')
  s=s.replace('constant Parameters &parameters',f'constant {type_prefix}Params &parameters')
 (d/target).write_text(s)
# Scratch-only copy of the ABI types the registry generator would emit. This
# is used to compile the emitted source before any repository fragment exists.
abi=json.loads((p/'packed_abi.json').read_text());td=p/'standalone_bindings';td.mkdir(exist_ok=True)
s=f'struct {type_prefix}Resources {{\n'
for i,(name,kind,mode) in enumerate(abi['resources']):
 ty={'uint32':'uint','int32':'int'}[kind];const='const ' if mode=='read' else ''
 s+=f'    device {const}{ty} *{name} [[id({i})]];\n'
s+=f'}};\nstruct {type_prefix}Params {{\n'
for x in abi['parameters']:s+=f'    {"float" if x["kind"]=="float" else "int"} {x["name"]};\n'
s+='};\n'
s+=f'static_assert(sizeof({type_prefix}Params) == {4*len(abi["parameters"])}, "parameter ABI size changed");\n'
(td/f'{binding}_bindings.metalh').write_text(s)
