from pathlib import Path
import argparse,re,json,ast
root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description='Generate the original native MM5 shader and ABI fragments.')
parser.add_argument('--output-root', type=Path, default=root)
destination=parser.parse_args().output_root
shader_dir=destination/'woof/metal/shaders'
fragment_dir=destination/'woof/metal/signature_fragments'
shader_dir.mkdir(parents=True, exist_ok=True)
fragment_dir.mkdir(parents=True, exist_ok=True)
src=next(ast.literal_eval(n.value) for n in ast.parse((root/'woof/hex/mpasphys/sfclay_mm5_v842.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SOURCE' for t in n.targets))
s=src[src.index('struct MM5State'):src.index('// MPAS physics_mmm/sf_sfclayrev.F90:79-920')]+src[src.index('// MPAS physics_wrf/module_sf_sfclay.F:265-946'):src.index('// MPAS physics_mmm/sf_sfclayrev.F90:27-55;')]
s=s.replace('if ((s.br < 0x0.0p+0f)) goto L310; // MPAS:562\nif ((s.br < 0x1.99999a0000000p-3f)) goto L270; // MPAS:566', 'if (!(s.br < 0x0.0p+0f) && !(s.br < 0x1.99999a0000000p-3f)) {')
s=s.replace('goto L320; // MPAS:588\nL270:;\nif ((s.br == 0x0.0p+0f)) goto L280; // MPAS:592', '} else if (!(s.br < 0x0.0p+0f) && !(s.br == 0x0.0p+0f)) {')
s=s.replace('goto L320; // MPAS:624\nL280:;', '} else if (!(s.br < 0x0.0p+0f)) {')
s=s.replace('goto L320; // MPAS:645\nL310:;', '} else {')
s=s.replace('L320:;', '}')
s=s.replace('if ((optional_mask & 4u) == 4u) { // MPAS:826\nif ((scm_force_flux == 0x1.0000000000000p+0f)) goto L350; // MPAS:827\n} // MPAS:828', 'if (!((optional_mask & 4u) == 4u && scm_force_flux == 1)) {')
s=s.replace('L350:;', '}')
s=s.replace('if ((isfflx == 0x0.0p+0f)) goto L410; // MPAS:835', 'if (!isfflx) return;')
s=s.replace('if ((optional_mask & 4u) == 4u) { // MPAS:889\nif ((scm_force_flux == 0x1.0000000000000p+0f)) goto L405; // MPAS:890\n} // MPAS:891', 'if (!((optional_mask & 4u) == 4u && scm_force_flux == 1)) {')
s=s.replace('L405:;', '}')
s=re.sub(r'^L\d+:;\s*$', '', s, flags=re.M)
s=s.replace('const float *tables','device const float *tables').replace('const MM5Constants &c','thread const MM5Constants &c').replace('MM5State &s','thread MM5State &s').replace('float &zol2','thread float &zol2')
s=re.sub(r'\bfloat\b(?! \*tables)', 'F',s)
s=s.replace('device const F *tables','device const float *tables')
s=s.replace('__device__','static __attribute__((noinline))').replace('unsigned int optional_mask = 255','unsigned int optional_mask')
s=s.replace('iter = FADD(iter, 0x1.0000000000000p+0f);', 'iter = int(FADD(iter, 0x1.0000000000000p+0f));')
s=s.replace('fminf(', 'sf_min(').replace('fmaxf(', 'sf_max(').replace('fabsf(', 'fabs(')
# F/int casts must be explicit for table indices.
s=re.sub(r'\(int\)\((FADD\(nzol(?:2|10)?, 0x1\.0000000000000p\+0f\))\)',r'int(\1)',s)
# Convert loads from device float table via implicit F construction.
header='''#include <metal_stdlib>
#include "hexv842_mm5_classic_tables_bindings.metalh"
#include "hexv842_mm5_classic_bindings.metalh"
#include "soft_word32.metalh"
#include "soft_glibc_solar_trig.metalh"
using namespace metal;
static inline F operator/(F a,F b) { return aa_div_impl(a,b); }
static inline F sf_min(F a,F b) { if (soft32_is_nan(a.bits)) return b; if (soft32_is_nan(b.bits)) return a; if (((a.bits|b.bits)&0x7fffffffu)==0) return f32_word(a.bits|b.bits); return a<b?a:b; }
static inline F sf_max(F a,F b) { if (soft32_is_nan(a.bits)) return b; if (soft32_is_nan(b.bits)) return a; if (((a.bits|b.bits)&0x7fffffffu)==0) return f32_word(a.bits&b.bits); return a>b?a:b; }
static inline F mm5_log(F a) { return gfk_log(a); }
static inline F mm5_exp(F a) { return gfk_exp(a); }
static inline F mm5_pow(F a,F b) { return gfk_pow(a,b); }
static inline F mm5_atan(F a) { return f32_word(soft_glibc_solar_atanf(a.bits)); }
#define FADD(a,b) (F(a)+F(b))
#define FSUB(a,b) (F(a)-F(b))
#define FMUL(a,b) (F(a)*F(b))
#define FDIV(a,b) (F(a)/F(b))
#define FSQRT(a) f32_word(soft32_sqrt(F(a).bits))
'''
fields=re.search(r'struct MM5State \{(.*?)\};',src).group(1)
fields=re.findall(r'float (\w+);',fields)
consts=re.findall(r'float (\w+);',re.search(r'struct MM5Constants \{(.*?)\};',src).group(1))
table='''kernel void hexv842_mm5_classic_tables(constant Hexv842Mm5ClassicTablesResources &r [[buffer(0)]], constant Hexv842Mm5ClassicTablesParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.n)) return;
 F z=FMUL(-F(float(i)),0.01f);
 F x=mm5_pow(FSUB(1.f,FMUL(16.f,z)),0.25f);
 r.tables[4004+i]=float(FADD(FSUB(FADD(FMUL(2.f,mm5_log(FMUL(0.5f,FADD(1.f,x)))),mm5_log(FMUL(0.5f,FADD(1.f,FMUL(x,x))))),FMUL(2.f,mm5_atan(x))),1.57079637050628662109375f));
 F y=mm5_pow(FSUB(1.f,FMUL(16.f,z)),0.5f);
 r.tables[5005+i]=float(FMUL(2.f,mm5_log(FMUL(0.5f,FADD(1.f,y)))));
}
'''
column='''kernel void hexv842_mm5_classic(constant Hexv842Mm5ClassicResources &r [[buffer(0)]], constant Hexv842Mm5ClassicParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.n)) return;
 MM5State s; MM5Constants c;
'''+''.join(f' s.{f}=r.a[{j}*p.n+i];\n' for j,f in enumerate(fields))+''.join(f' c.{f}=r.constants[{j}];\n' for j,f in enumerate(consts))+''' mm5_classic(s,c,r.tables,p.isfflx,p.isftcflx,p.iz0tlnd,p.scm_force_flux,p.shalwater_z0,uint(p.optional_mask));
'''+''.join(f' r.a[{j}*p.n+i]=float(s.{f});\n' for j,f in enumerate(fields[:34]))+'}\n'
(shader_dir/'hexv842_mm5_classic.metal').write_text(header+s+table+column)
def spec(name,symbol,resources,params):
 return dict(name=name,module='hexv842_mm5_classic',source='woof/metal/shaders/hexv842_mm5_classic.metal',cuda_source='woof/hex/mpasphys/sfclay_mm5_v842.py',cuda_symbol=symbol,resources=resources,shapes={r[0]:[['any_1d']] for r in resources},parameters=[[v,'int32'] for v in params],threads_per_group=128,layout_version=1,arithmetic_policies=['default'])
items=[spec('hexv842_mm5_classic_tables','mm5_init_tables',[['tables','float','read_write']],['n']),spec('hexv842_mm5_classic','mm5_columns',[['a','float','read_write'],['constants','float','read'],['tables','float','read']],['n','isfflx','isftcflx','iz0tlnd','scm_force_flux','shalwater_z0','optional_mask'])]
items[0]['shapes']={'tables':[['6006']]}
items[1]['shapes']={'a':[['48','n']],'constants':[['13']],'tables':[['6006']]}
for item in items:
 (fragment_dir/(item['name']+'.json')).write_text(json.dumps({'kernels':[item]},indent=2)+'\n')
assert len(fields) == 48 and len(consts) == 13 and 'goto' not in s
