import os
"""Extract the pinned column and apply only address-space and lane-one rewrites.

The emitted candidate deliberately retains CUDA real kinds and CUDA arithmetic
helpers: replacing those requires a type-aware numeric lowering, not token swaps.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

root = Path(os.environ.get('WOOF_SOURCE_ROOT', Path.cwd()))
pin = os.environ.get('WOOF_MYNN_PIN', 'c6d08709')
src = root / f'woof/hex/mpasphys/rrfs/bl_mynnedmf_{pin}.py'
out = Path(__file__).resolve().parent
parsed = ast.parse(src.read_text())
values = {n.targets[0].id: ast.literal_eval(n.value) for n in parsed.body
          if isinstance(n, ast.Assign) and len(n.targets) == 1 and
          isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('ARRAY_SUPPORT', 'MODULE_CUDA')}
support, module = values['ARRAY_SUPPORT'], values['MODULE_CUDA']
start = module.index('template<int MYNN_LANES> struct MynnColumn')
# Brace counting ignores comments and quoted strings, including braces in diagnostics.
def close_brace(text, start):
    depth = 0
    token = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]')
    for m in token.finditer(text, start):
        t = m.group()
        if t == '{': depth += 1
        elif t == '}':
            depth -= 1
            if depth == 0:
                assert text[m.end():].lstrip().startswith(';')
                return m.end()+text[m.end():].index(';')+1
    raise ValueError('unclosed struct')
column = module[start:close_brace(module, module.index('{', start))]
# Preserve original binary32/binary64/int inventory before any transformation.
report = {'source_sha256': hashlib.sha256(src.read_bytes()).hexdigest(),
          'column_sha256': hashlib.sha256(column.encode()).hexdigest(),
          'column_bytes': len(column),
          'type_tokens': dict(Counter(re.findall(r'\b(?:float|double|int|bool)\b', column))),
          'array_types': dict(Counter(f'{t},{r}' for t,r in re.findall(r'MynnArray<\s*(\w+)\s*,\s*(\d+)', column))),
          'cuda_tokens': dict(Counter(re.findall(r'__\w+', column)))}
# Extract representation code separately from arithmetic helpers. Device arena storage
# is intentionally device address space; all descriptors and scalar references are thread.
array = support[support.index('template<int R> struct MynnIndices'):support.index('// RRFS', support.index('template<class T> struct MynnOptional'))]
# Include complete optional type, ending immediately before intrinsic helpers.
optional_start = support.index('template<class T> struct MynnOptional')
optional_end = support.index('template<class T> __device__ T mynn_min', optional_start)
array = support[support.index('template<int R> struct MynnIndices'):optional_end]
array = re.sub(r'//[^\n]*', '', array)
array = array.replace('__device__ ', '')
array = array.replace('unsigned char* data;', 'device unsigned char* data;')
array = array.replace('void* allocate(', 'device void* allocate(').replace('void* p = data', 'device void* p = data')
array = array.replace('MynnArena&', 'thread MynnArena&')
array = array.replace('T* data = nullptr;', 'device T* data = nullptr;')
array = array.replace('T& linear(', 'device T& linear(').replace('T& operator()(', 'device T& operator()(')
array = re.sub(r'(?<!\w)(MynnArray<T,N>|int)& (out|d|od|offset)', r'thread \1& \2', array)
array = array.replace('static_cast<T*>(arena.allocate', 'static_cast<device T*>(arena.allocate')
array = array.replace('const char*', 'constant char*')
array = array.replace('MynnText& operator=', 'thread MynnText& operator=').replace('const MynnText& b', 'thread const MynnText& b')
column = re.sub(r'//[^\n]*', '', column)
column = column.replace('template<int MYNN_LANES> struct MynnColumn', 'struct MynnColumn')
column = column.replace('MynnArena& arena;', 'thread MynnArena& arena;')
column = column.replace('static constexpr int lane_width = MYNN_LANES; static constexpr int mynn_tile = 32 / MYNN_LANES;', 'static constexpr int lane_width = 1; static constexpr int mynn_tile = 1;')
column = re.sub(r'__device__ __forceinline__ void mynn_lane_sync\(\) const \{[^}]*\}', 'void mynn_lane_sync() const {}', column)
column = re.sub(r'__device__ __forceinline__ (int|float|double|bool) mynn_lane_bcast\(\1 v, int src\) const \{[^}]*\}', r'\1 mynn_lane_bcast(\1 v, int src) const { return v; }', column)
column = re.sub(r'__(?:device|forceinline|noinline)__\s*', '', column)
# The generated methods use references exclusively for scalar/optional/text thread locals.
column = re.sub(r'\b(float|double|int|bool|MynnText|MynnOptional<[^>]+>)\s*&', r'thread \1&', column)
(out/'candidate.metal').write_text('#include <metal_stdlib>\nusing namespace metal;\n\n'+array+'\n'+column+'\n')
(out/'inventory.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
print(json.dumps(report, indent=2, sort_keys=True))

# All frame declarations are the first statement of a column method. The
# source only returns void or a single local identifier; preserve the latter's
# value before rewind, then return it. Reject any shape not audited here.
tokens = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{}]|\breturn\s+\w+\s*;|\breturn\s*;|MynnFrame mynn_frame\(arena\);')
stack=[]
frames={}
edits=[]
for m in tokens.finditer(column):
    t=m.group()
    if t=='{': stack.append(m.start())
    elif t=='}':
        opened=stack.pop()
        if opened in frames:
            edits.append((m.start(),m.start(),'\narena.offset=mynn_saved_offset;\n'))
    elif t=='MynnFrame mynn_frame(arena);':
        assert len(stack)==2 and stack[-1] not in frames
        frames[stack[-1]]=True
        edits.append((m.start(),m.end(),'ulong mynn_saved_offset=arena.offset;'))
    elif t.startswith('return') and any(o in frames for o in stack):
        value=t[len('return'):].strip()[:-1].strip()
        # Framed value-returning methods all return float local variables.
        replacement = ('{ auto mynn_return_value='+value+'; arena.offset=mynn_saved_offset; return mynn_return_value; }'
                       if value else '{ arena.offset=mynn_saved_offset; return; }')
        edits.append((m.start(),m.end(),replacement))
assert not stack and len(frames)==37
for a,b,replacement in reversed(sorted(edits)):
    column=column[:a]+replacement+column[b:]
(out/'column_lane1.metalh').write_text(column+'\n')


"""Create typed whole-column diagnostic candidate, preserving literal kinds."""
import re,struct
from pathlib import Path
p=Path(__file__).resolve().parent
s=(p/'column_lane1.metalh').read_text()
# Unsuffixed source floating literals are all binary64 cast operands.
pat=r'\(double\)\((\d+\.\d*(?:e[-+]?\d+)?)\)'
def dconst(m):
 bits=struct.unpack('>Q',struct.pack('>d',float(m.group(1))))[0]
 return f'Mynn64::from_bits(0x{bits:016x}ul)'
s,n=re.subn(pat,dconst,s,flags=re.I)
assert n==11,n
# Existing static float members are constexpr-only source constants; retaining
# those native constants permits C++14 constant evaluation and performs no runtime math.
lines=[]
for line in s.splitlines(keepends=True):
 if not line.startswith('static constexpr float '):
  line=re.sub(r'\bfloat\b','Exact32',line)
  line=re.sub(r'\bdouble\b','Mynn64',line)
 lines.append(line)
s=''.join(lines)
s=s.replace('constexpr Exact32 ', 'const Exact32 ')
constants=re.findall(r'^static constexpr float [^\n]+',s,re.M)
assert len(constants)==(102 if pin == 'c942418' else 103)
other_constants=re.findall(r'static constexpr (?:int|bool) \w+ = [^;]+;',s)
assert len(other_constants)==8, other_constants
s=re.sub(r'static constexpr (?:int|bool) \w+ = [^;]+;','',s)
s='\n'.join(x.replace('static constexpr','constant constexpr',1) for x in other_constants)+'\n'+s
s=re.sub(r'^static constexpr float [^\n]+\n','',s,flags=re.M)
s='\n'.join(x.replace('static constexpr float','constant constexpr float',1) for x in constants)+'\n'+s
needle='(mynn_ddiv(q3sq, dlsq)) < (r6_neg(gh(k)))'
assert s.count(needle)==2
s=s.replace(needle, '(mynn_ddiv(q3sq, dlsq)) < (Mynn64(r6_neg(gh(k))))')
assert s.count('goto mynn_label_86;')==1 and s.count('mynn_label_86: ;')==1
s=s.replace('goto mynn_label_86;', 'break;').replace('mynn_label_86: ;', '')
assert not re.search(r'\bdouble\b',s)
(p/'column_typed.metalh').write_text(s)

# Source scans may reach the top interface; refuse before entering a scan
# body that needs mass-level data above kte. Preserve all valid iterations.
column_path = p / 'column_typed.metalh'
column = column_path.read_text()
needle = 'struct MynnColumn {\nthread MynnArena& arena;\n'
assert column.count(needle) == 1
column = column.replace(needle, needle + 'int geometry_error = 0;\n')
for pattern, expected in ((r'^while \(\(\(zwk\) <= .*\{', 3),
                          (r'^while \(\(\(zw1\(k\)\) <= \(200\.f\)\)\) \{', 1)):
    column, count = re.subn(
        pattern, lambda match: match.group() +
        '\nif (k > kte) { geometry_error = -1; arena.offset = mynn_saved_offset; return; }',
        column, flags=re.M)
    assert count == expected, (pattern, count)
for name, expected in (('f_mym_length', 2), ('f_get_pblh', 2),
                       ('f_mym_initialize', 1), ('f_mym_turbulence', 1)):
    column, count = re.subn(
        r'^' + name + r'\([^\n]*;', lambda match: match.group() +
        '\nif (geometry_error) { arena.offset = mynn_saved_offset; return; }',
        column, flags=re.M)
    assert count == expected, (name, count)
column, count = re.subn(
    r'^f_mynnedmf\([^\n]*;', lambda match: match.group() +
    '\nif (geometry_error) { errflg = geometry_error; arena.offset = mynn_saved_offset; return; }',
    column, flags=re.M)
assert count == 1
column_path.write_text('\n'.join(line.rstrip() for line in column.splitlines()).rstrip() + '\n')
