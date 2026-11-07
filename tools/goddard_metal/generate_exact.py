"""Generate the separate strict Goddard source-closure and grid kernels."""
from pathlib import Path
import importlib, json, re, runpy, struct, tempfile
m=importlib.import_module('tools.goddard_metal.generate')
root=Path(__file__).resolve().parents[2]
temp=tempfile.TemporaryDirectory(prefix='woof-goddard-exact-')
out=Path(temp.name)
(out/'woof/metal/shaders').mkdir(parents=True,exist_ok=True)
(out/'woof/metal/signature_fragments').mkdir(parents=True,exist_ok=True)
m.ROOT=out; m.SHADERS=out/'woof/metal/shaders'
m.PROVIDED |= {'gfk_log','gfk_d2f_rn','gfk_exp2_core','gfk_exp',
               'gfk_powf_log2','gfk_checkint','gfk_zeroinfnan','gfk_pow'}
old_table=m.global_table
def table(item):
    return '' if (m.def_name(item) or '').startswith('GFK_') else old_table(item)
m.global_table=table
m.main()
fragment=(m.SHADERS/'goddard_fragment.metalh').read_text().replace(
    'tools/goddard_metal/generate.py', 'tools/goddard_metal/generate_exact.py')
shader=(m.SHADERS/'goddard_family.metal').read_text().replace(
    'tools/goddard_metal/generate.py', 'tools/goddard_metal/generate_exact.py')
# Carry binary32 values as F words; do not alter the bound ABI declarations.
fragment=re.sub(r'\bfloat\b','F',fragment)
shader=re.sub(r'\bfloat\b','F',shader)
# Resource members retain their generated float pointer ABI.
shader=re.sub(r'(device (?:const )?F \*(\w+))=r\.(\w+);',
              lambda x:x[1]+'=reinterpret_cast<device '+('const ' if 'const' in x[1] else '')+'F *>(r.'+x[3]+');',shader)
# Constant floating-point tables must be raw words, not F constructors.
for name in ('gdd_sounding','gdd_radius_points'):
    pattern=r'constant F '+name+r'(\s*\[[^;=]+)=\s*\{(.*?)\}\s*;'
    def encode(match):
        def word(n):
            value=float(n.group(0)[:-1]); return '0x%08xu'%struct.unpack('<I',struct.pack('<f',value))[0]
        return 'constant uint '+name+match[1]+'={'+re.sub(r'(?<![\w.])(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?f\b',word,match[2])+'};'
    fragment,n=re.subn(pattern,encode,fragment,flags=re.S)
    if n!=1:raise ValueError((name,'constant table count',n))
    dims=3 if name=='gdd_sounding' else 1
    def accesses(text):
        pattern=re.compile(r'\b'+name+r'\b')
        positions=list(pattern.finditer(text))
        for match in reversed(positions):
            if 'constant uint' in text[max(0,match.start()-20):match.start()]:
                continue
            end=match.end()
            for _ in range(dims):
                while text[end].isspace():end+=1
                if text[end]!='[':raise ValueError((name,'index',text[end:end+40]))
                depth=1;end+=1
                while depth:
                    depth += (text[end]=='[')-(text[end]==']');end+=1
            text=text[:match.start()]+'f32_word('+text[match.start():end]+')'+text[end:]
        return text
    fragment=accesses(fragment);shader=accesses(shader)
shader=shader.replace('#include "goddard_support.metalh"','#include "goddard_support.metalh"\n#include "fast_sbm_support.metalh"\nstatic inline Soft8 g_div(Soft8 x, float y) { return x / Soft8(y); }\nstatic inline Soft8 g_pow(Soft8 x, float y) { return plm_pow(x, Soft8(y)); }\nstatic inline F g_div(F x, float y) { return aa_div_impl(x, F(y)); }\nstatic inline F g_pow(F x, float y) { return gfk_pow(x, F(y)); }')
sigs=json.loads((out/'woof/metal/signature_fragments/goddard_family.json').read_text())['kernels']
for row in sigs:
    old=row['name']; new='wrf_exact_'+old
    old_title=''.join(x.title() for x in old.split('_'))
    new_title=''.join(x.title() for x in new.split('_'))
    shader=shader.replace('kernel void '+old+'(', 'kernel void '+new+'(')
    shader=shader.replace(old_title+'Resources',new_title+'Resources').replace(old_title+'Params',new_title+'Params')
    row.update(name=new,module='wrf_exact_goddard_family',
               source='woof/metal/shaders/wrf_exact_goddard_family.metal',
               arithmetic_policies=['wrf-exact'])
shader=shader.replace('goddard_family_bindings.metalh','wrf_exact_goddard_family_bindings.metalh')
shader=shader.replace('goddard_fragment.metalh','wrf_exact_goddard_fragment.metalh')
shaders=root/'woof/metal/shaders'
(shaders/'wrf_exact_goddard_fragment.metalh').write_text(fragment)
(shaders/'wrf_exact_goddard_family.metal').write_text(shader)
fragments=root/'woof/metal/signature_fragments'
(fragments/'wrf_exact_goddard_family.json').write_text(json.dumps({'kernels':sigs},indent=2)+'\n')
# The grid entries copy bit patterns and perform only the source's two
# ordered radius-conversion multiplications.
grid=(shaders/'goddard_grid.metal').read_text().replace(
    'tools/goddard_metal/generate_grid.py', 'tools/goddard_metal/generate_exact.py')
grid=grid.replace('goddard_grid_bindings.metalh','wrf_exact_goddard_grid_bindings.metalh')
grid=grid.replace('#include <metal_stdlib>','#include <metal_stdlib>\n#include "soft_binary32.metalh"')
grid=grid.replace('float value;', 'uint value;').replace('value=0.0f;', 'value=0u;')
grid=grid.replace('value=(value*1.e-6f)*1.e6f;',
 'value=soft32_mul(soft32_mul(value,as_type<uint>(1.e-6f)),as_type<uint>(1.e6f));')
grid=re.sub(r'r\.(\w+)\[([^\]]+)\]',
 lambda x:'(reinterpret_cast<device '+('' if x[1] in ('raw','heat_lw','heat_sw','surface') else 'const ')+'uint *>(r.'+x[1]+'))['+x[2]+']',grid)
grid_sigs=json.loads((fragments/'goddard_grid.json').read_text())['kernels']
for row in grid_sigs:
    old=row['name']; new='wrf_exact_'+old
    old_title=''.join(x.title() for x in old.split('_'))
    new_title=''.join(x.title() for x in new.split('_'))
    grid=grid.replace('kernel void '+old+'(', 'kernel void '+new+'(')
    grid=grid.replace(old_title+'Resources',new_title+'Resources').replace(old_title+'Params',new_title+'Params')
    row.update(name=new,module='wrf_exact_goddard_grid',source='woof/metal/shaders/wrf_exact_goddard_grid.metal',arithmetic_policies=['wrf-exact'])
(shaders/'wrf_exact_goddard_grid.metal').write_text(grid)
(fragments/'wrf_exact_goddard_grid.json').write_text(json.dumps({'kernels':grid_sigs},indent=2)+'\n')
runpy.run_path(str(root/'tools/goddard_metal/generate_tier.py'),run_name='__main__')
temp.cleanup()
print('Generated strict Goddard column and grid entries')
