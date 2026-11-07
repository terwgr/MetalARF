"""Translate the selected NSSL NUCOND scalar source to the Metal word type."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT/'woof/core/kernels/nssl2_nucond.cu').read_text()
a = source.index('namespace {'); b = source.index('}  // namespace',a)+len('}  // namespace')
c = source.index('extern "C" __global__ void nssl2_nucond_default(')
brace = source.index('{',c); depth=1; end=brace+1
while depth:
    depth += (source[end]=='{')-(source[end]=='}'); end+=1
helpers=source[a:b]
body=source[brace+1:end-1]
fields={'full_theta':'full_theta','air_density':'air_density','pressure_pa':'pressure_pa',
 'exner':'exner','w_interface':'w_interface','ssfilt':'initial_supersaturation_percent',
 'qv_field':'qv','qc_field':'qc','qr_field':'qr','qi_field':'qi','qs_field':'qs',
 'qndrop':'qndrop','qnr':'qnr','qni':'qni','qns':'qns','qnn':'qnn',
 'qg':'qg','qh':'qh','qng':'qng','qnh':'qnh','qvolg':'qvolg','qvolh':'qvolh'}
def convert(text):
    text=text.replace('__device__ __forceinline__','static inline').replace('__device__','static')
    text=re.sub(r'\bconstexpr (float|int)\b',r'constant constexpr \1',text)
    text=re.sub(r'(?<!constexpr )\bfloat\b','F',text)
    text=re.sub(r'\(F\)(\([^()]*\)|[a-zA-Z_]\w*)',r'F(\1)',text)
    text=text.replace('__int_as_float(', 'f32_word(').replace('sqrtf(', 'nucond_sqrt(').replace('fabsf(', 'nucond_abs(').replace('lroundf(', 'nucond_lround(')
    return text
helpers=convert(helpers)
body=re.sub(r'\bconst int idx = blockDim\.x \* blockIdx\.x \+ threadIdx\.x;', 'const int idx = int(thread_position_in_grid);',body)
for source_name,target in fields.items():
    body=re.sub(r'(?m)^(\s*)'+source_name+r'\[idx\] = ([^;]+);',lambda m: m.group(1)+'__store_'+target+'('+m.group(2)+');',body)
for source_name,target in fields.items():
    body=re.sub(r'\b'+source_name+r'\[([^\]]+)\]',lambda m:'F(r.'+target+'['+m.group(1)+'])',body)
# Stores were rewritten above and must not be rewritten as reads.
# Stores are markers until all reads have been converted.
for param in ('n','nz','horizontal_size','predicted_ccn','concentration_space','has_frozen'):
    body=re.sub(r'\b'+param+r'\b','p.'+param,body)
body=re.sub(r'\bdtp\b','F(p.dt)',body)
body=convert(body)
# The transformation also touches the explicit output float casts.
body=body.replace(' = F(', ' = F(')  # declarations remain word operations
body=re.sub(r'(?m)^(\s*)__store_(\w+)\(([^;]+)\);',lambda m:m.group(1)+'r.'+m.group(2)+'[idx] = float('+m.group(3)+');',body)
macros='''\n#define NSSL2_WRF_CWMASN f32_word(0x2916eacau)\n#define NSSL2_WRF_CWMASX f32_word(0x3078b42bu)\n#define NSSL2_WRF_XVRMN f32_word(0x2a96eac9u)\n#define NSSL2_WRF_XVRMX0 f32_word(0x33f2dff3u)\n'''
preamble='''\n// Generated from the selected strict NUCOND source by tools/generate_nssl2_nucond_exact.py.\nstatic inline F nucond_sqrt(F x) { return f32_word(soft32_sqrt(x.bits)); }\nstatic inline F nucond_abs(F x) { return f32_word(x.bits & 0x7fffffffu); }\nstatic inline int nucond_lround(F x) {\n    const uint magnitude=x.bits & 0x7fffffffu; const int exponent=int(magnitude>>23)-127;\n    if (exponent < -1) return 0;\n    const uint sig=(magnitude & 0x7fffffu)|0x800000u;\n    uint value;\n    if (exponent < 0) value=1u;\n    else if (exponent >= 23) value=sig << uint(exponent-23);\n    else { const uint shift=uint(23-exponent); value=(sig >> shift)+uint((sig & ((1u << shift)-1u)) >= (1u << (shift-1u))); }\n    return (x.bits & 0x80000000u) != 0u ? -int(value) : int(value);\n}\n'''
shader=preamble+macros+helpers+'''\n\nkernel void nssl2_nucond_stage_exact(\n    constant Nssl2NucondStageExactResources &r [[buffer(0)]],\n    constant Nssl2NucondStageExactParams &p [[buffer(1)]],\n    uint thread_position_in_grid [[thread_position_in_grid]]) {\n'''+body+'\n}\n'
for name in ('NSSL2_WRF_CWMASN','NSSL2_WRF_CWMASX','NSSL2_WRF_XVRMN','NSSL2_WRF_XVRMX0'):
    shader+='\n#undef '+name
shader+='\n'
path=ROOT/'woof/metal/shaders/nssl2_reduce_exact.metal'
content=path.read_text();marker='\n// Generated from the selected strict NUCOND source by tools/generate_nssl2_nucond_exact.py.'
if marker in content: content=content[:content.index(marker)]
path.write_text(content+shader)
