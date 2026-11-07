"""Translate the pinned native MPAS surface-layer column source to Metal."""
from pathlib import Path
import re
from woof.hex.mpasphys.sf_mynn_v842 import SOURCE, FIELDS, _BLEND, CORRECT_MODE_SOURCE

ROOT=Path(__file__).resolve().parents[2]
source=SOURCE[:SOURCE.index('// MPAS physics_mmm/sf_mynn.F90:130-996\nextern "C" __global__')]
source=re.sub(r'__device__ float psi\w+\[1001\];\n','',source)
for name in ('m_psi_init','m_zolri','m_zolri2'):
    source=re.sub(r'__device__ __noinline__ (?:void|float) '+name+r'\([^;{]*\);\n','',source)
    m=re.search(r'__device__ __noinline__ (?:void|float) '+name+r'\([^;{]*\) \{',source)
    assert m,name
    depth=1;i=m.end()
    while depth:
        if source[i]=='{': depth+=1
        elif source[i]=='}': depth-=1
        i+=1
    source=source[:m.start()]+source[i:]

# CUDA device-global tables become one explicit immutable buffer passed to
# every call that reads them; state is initialized once per Metal context.
for name in ('m_sf_mynn_run','m_zolrib','m_psim_stable','m_psih_stable','m_psim_unstable','m_psih_unstable'):
    matches=list(re.finditer(r'\b'+name+r'\s*\(',source))
    for m in reversed(matches):
        pos=m.end();depth=1;i=pos
        while depth:
            if source[i]=='(':depth+=1
            elif source[i]==')':depth-=1
            i+=1
        before=source[max(0,m.start()-50):m.start()]
        declaration=bool(re.search(r'__device__ __noinline__ (?:void|float)\s+$',before))
        arg=', device const float *tables' if declaration else ', tables'
        source=source[:i-1]+arg+source[i-1:]
for name,off in (('psim_stab',0),('psih_stab',1001),('psim_unstab',2002),('psih_unstab',3003)):
    source=re.sub(r'\b'+name+r'\[([^\]]+)\]',lambda m:'tables['+str(off)+' + ('+m.group(1)+')]',source)
source=source.replace('__device__ __noinline__ ','static inline ')
source=re.sub(r'\bfloat\s*&', 'thread float &',source)
for old,new in {'__fadd_rn':'mpas_add','__fsub_rn':'mpas_sub','__fmul_rn':'mpas_mul',
                '__fdiv_rn':'cu_div','SL_SQRT':'cu_sqrt','SL_LOG':'gfk_log','SL_EXP':'gfk_exp',
                'SL_POW':'gfk_pow','SL_ATAN':'native_mpas_atan','fmaxf':'fmax','fminf':'fmin','fabsf':'fabs'}.items():
    source=re.sub(r'\b'+old+r'\b',new,source)
match=re.search(r'm_sf_mynn_run\((fields\[0\*n\+i\].*?)\);',SOURCE)
assert match
call=match.group(1)
call=re.sub(r'fields\[(\d+)\*n\+i\]',lambda m:'v'+m.group(1),call)
call=re.sub(r'constants\[(\d+)\]',lambda m:'r.constants['+m.group(1)+']',call)
for name in ('itimestep','isfflx','spp_pbl','isftcflx','iz0tlnd','present'):
    call=re.sub(r'\b'+name+r'\b','p.'+name,call)
loads='\n'.join(f'    float v{i}=r.fields[{i}*p.n+i];' for i in range(len(FIELDS)))
stores='\n'.join(f'    r.fields[{i}*p.n+i]=v{i};' for i in range(len(FIELDS)))
shader='''#include <metal_stdlib>\nusing namespace metal;\n#include "hex_ieee32.metalh"\n#include "cumulus_support.metalh"\n#include "soft_urban_trig.metalh"\n#include "hexv842_mpas_sf_mynn_f32_bindings.metalh"\n#pragma clang fp contract(off)\nstatic inline float native_mpas_atan(float x) { return as_type<float>(soft_urban_atanf(as_type<uint>(x))); }\n'''+source+'''\nkernel void hexv842_mpas_sf_mynn_tables_f32(constant Hexv842MpasSfMynnTablesF32Resources &r [[buffer(0)]], constant Hexv842MpasSfMynnTablesF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n    if (i>=1001) return;\n    float z=mpas_mul(float(i),0x1.47ae140000000p-7f);\n    r.tables[i]=m_psim_stable_full(z); r.tables[1001+i]=m_psih_stable_full(z);\n    z=mpas_mul(-float(i),0x1.47ae140000000p-7f);\n    r.tables[2002+i]=m_psim_unstable_full(z); r.tables[3003+i]=m_psih_unstable_full(z);\n}\nkernel void hexv842_mpas_sf_mynn_columns_f32(constant Hexv842MpasSfMynnColumnsF32Resources &r [[buffer(0)]], constant Hexv842MpasSfMynnColumnsF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n    if (i>=uint(p.n)) return;\n'''+loads+'\n    m_sf_mynn_run('+call+',r.tables);\n'+stores+'\n}\n'
idx={name:FIELDS.index(name) for name in FIELDS}
old,corrected,count=CORRECT_MODE_SOURCE['RP3'][0]
assert count==1 and corrected.startswith(old)
rp3=corrected[len(old):]
for original,replacement in {
    'F(tsk)':f"r.fields[{idx['tsk']}*p.n+i]",
    'sea[IDX_tsk*n+i]':f"r.sea[{idx['tsk']}*p.n+i]",
    'ice[i]':'r.ice[i]', '__fsub_rn':'mpas_sub', '__fmul_rn':'mpas_mul',
    '__fdiv_rn':'cu_div', 'fmaxf':'fmax',
}.items():rp3=rp3.replace(original,replacement)
shader+='''\nkernel void hexv842_mpas_sf_mynn_prepare_f32(constant Hexv842MpasSfMynnPrepareF32Resources &r [[buffer(0)]], constant Hexv842MpasSfMynnPrepareF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {\n    if(i>=uint(p.n)) return;\n'''
for target,profile,level in (('u1d',0,0),('v1d',1,0),('t1d',2,0),('qv1d',3,0),('p1d',5,0),('rho1d',6,0),('dz8w1d',7,0),('u1d2',0,1),('v1d2',1,1),('dz2w1d',7,1)):
    shader+=f'    r.fields[{idx[target]}*p.n+i]=r.profiles[{2*profile+level}*p.n+i];\n'
shader+=f'''    r.fields[{idx['rstoch1d']}*p.n+i]=0.0f;
    r.fields[{idx['dx']}*p.n+i]=cu_div(p.len_disp,gfk_pow(r.density[i],0.25f));
    if(p.first) {{
        float u=r.profiles[i],v=r.profiles[2*p.n+i],q=r.profiles[6*p.n+i];
        r.fields[{idx['ust']}*p.n+i]=fmax(mpas_mul(0.04f,cu_sqrt(mpas_add(mpas_mul(u,u),mpas_mul(v,v)))),0.001f);
        r.fields[{idx['mol']}*p.n+i]=0.0f;
        r.fields[{idx['qsfc']}*p.n+i]=cu_div(q,mpas_add(1.0f,q));
        r.fields[{idx['qstar']}*p.n+i]=0.0f;
    }}
}}
kernel void hexv842_mpas_sf_mynn_sea_f32(constant Hexv842MpasSfMynnSeaF32Resources &r [[buffer(0)]], constant Hexv842MpasSfMynnSeaF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {{
    if(i>=uint(p.n)) return;
    for(int j=0;j<{len(FIELDS)};++j) r.sea[j*p.n+i]=r.fields[j*p.n+i];
    r.sea[{idx['qsfc']}*p.n+i]=p.first ? r.fields[{idx['qsfc']}*p.n+i] : 0.0f;
    if(r.ice[i]>=p.threshold && r.ice[i]<=1.0f) {{
        r.sea[{idx['xland']}*p.n+i]=2.0f; r.sea[{idx['mavail']}*p.n+i]=1.0f;
        r.sea[{idx['znt']}*p.n+i]=0.0001f; r.sea[{idx['tsk']}*p.n+i]=fmax(r.sst[i],271.4f);
        if(p.rp3) {{{rp3}
        }}
    }}
}}
kernel void hexv842_mpas_sf_mynn_blend_f32(constant Hexv842MpasSfMynnBlendF32Resources &r [[buffer(0)]], constant Hexv842MpasSfMynnBlendF32Params &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {{
    if(i>=uint(p.n) || r.ice[i]<p.threshold || r.ice[i]>1.0f) return;
    float x=r.ice[i],y=mpas_sub(1.0f,x);
'''
for name in _BLEND:
    j=idx[name];shader+=f'    r.fields[{j}*p.n+i]=mpas_add(mpas_mul(r.fields[{j}*p.n+i],x),mpas_mul(y,r.sea[{j}*p.n+i]));\n'
shader+=f'''    if(x>=0.5f) r.fields[{idx['regime']}*p.n+i]=r.regime_hold[i];\n}}\n'''
(ROOT/'woof/metal/shaders/hexv842_mpas_sf_mynn_f32.metal').write_text(shader)
