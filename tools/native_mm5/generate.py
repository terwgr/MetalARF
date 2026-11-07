from pathlib import Path
import argparse,re,json,ast
root=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description='Generate the native revised MM5 shader and ABI fragments.')
parser.add_argument('--output-root', type=Path, default=root)
destination=parser.parse_args().output_root
shader_dir=destination/'woof/metal/shaders'
fragment_dir=destination/'woof/metal/signature_fragments'
shader_dir.mkdir(parents=True, exist_ok=True)
fragment_dir.mkdir(parents=True, exist_ok=True)
src=next(ast.literal_eval(n.value) for n in ast.parse((root/'woof/hex/mpasphys/sfclay_mm5_v842.py').read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_SOURCE' for t in n.targets))
s=src[src.index('struct MM5State'):src.index('// MPAS physics_wrf/module_sf_sfclay.F:265-946')]
# Rewrite only the revised routine's forward jumps into structured control flow.
s=s.replace('if ((s.br < 0x0.0p+0f)) goto L310; // MPAS:416\nif ((s.br == 0x0.0p+0f)) goto L280; // MPAS:417','if (!(s.br < 0x0.0p+0f) && !(s.br == 0x0.0p+0f)) {')
s=s.replace('goto L320; // MPAS:443\nL280:;', '} else if (s.br == 0x0.0p+0f) {')
s=s.replace('goto L320; // MPAS:464\nL310:;', '} else {')
s=s.replace('L320:;', '}')
s=s.replace('if ((optional_mask & 4u) == 4u) { // MPAS:783\nif (scm_force_flux) goto L350; // MPAS:784\n} // MPAS:785','if (!((optional_mask & 4u) == 4u && scm_force_flux)) {')
s=s.replace('L350:;', '}')
s=s.replace('if (!(isfflx)) goto L410; // MPAS:792','if (!isfflx) return;')
s=s.replace('if ((optional_mask & 4u) == 4u) { // MPAS:852\nif (scm_force_flux) goto L405; // MPAS:853\n} // MPAS:854','if (!((optional_mask & 4u) == 4u && scm_force_flux)) {')
s=s.replace('L405:;', '}')
s=re.sub(r'^L\d+:;\s*$', '', s, flags=re.M)
s=s.replace('const float *tables','device const float *tables').replace('const MM5Constants &c','thread const MM5Constants &c').replace('MM5State &s','thread MM5State &s').replace('float &zol2','thread float &zol2')
s=re.sub(r'\bfloat\b(?! \*tables)', 'F',s)
s=s.replace('device const F *tables','device const float *tables')
s=s.replace('thread F &zol2','thread F &zol2')
s=s.replace('__device__','static __attribute__((noinline))').replace('unsigned int optional_mask = 255','unsigned int optional_mask')
s=s.replace('iter = FADD(iter, 0x1.0000000000000p+0f);', 'iter = int(FADD(iter, 0x1.0000000000000p+0f));')
s=s.replace('fminf(', 'sf_min(').replace('fmaxf(', 'sf_max(').replace('fabsf(', 'fabs(')
# F/int casts must be explicit for table indices.
s=re.sub(r'\(int\)\((FADD\(nzol, 0x1\.0000000000000p\+0f\))\)',r'int(\1)',s)
# Convert loads from device float table via implicit F construction.
header='''#include <metal_stdlib>
#include "hexv842_mm5_tables_bindings.metalh"
#include "hexv842_mm5_revised_bindings.metalh"
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
table='''kernel void hexv842_mm5_tables(constant Hexv842Mm5TablesResources &r [[buffer(0)]], constant Hexv842Mm5TablesParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.n)) return;
 F z=FMUL(F(float(i)),0.01f);
 r.tables[i]=float(mm5_psim_stable_full(z,r.tables));
 r.tables[1001+i]=float(mm5_psih_stable_full(z,r.tables));
 z=FMUL(-F(float(i)),0.01f);
 r.tables[2002+i]=float(mm5_psim_unstable_full(z,r.tables));
 r.tables[3003+i]=float(mm5_psih_unstable_full(z,r.tables));
}
'''
column='''kernel void hexv842_mm5_revised(constant Hexv842Mm5RevisedResources &r [[buffer(0)]], constant Hexv842Mm5RevisedParams &p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
 if(i>=uint(p.n)) return;
 MM5State s; MM5Constants c;
'''+''.join(f' s.{f}=r.a[{j}*p.n+i];\n' for j,f in enumerate(fields))+''.join(f' c.{f}=r.constants[{j}];\n' for j,f in enumerate(consts))+''' mm5_revised(s,c,r.tables,p.isfflx,p.isftcflx,p.iz0tlnd,p.scm_force_flux,p.shalwater_z0,uint(p.optional_mask));
'''+''.join(f' r.a[{j}*p.n+i]=float(s.{f});\n' for j,f in enumerate(fields[:34]))+'}\n'
(shader_dir/'hexv842_mm5_revised.metal').write_text(header+s+table+column)
def spec(name,symbol,resources,params):
 return dict(name=name,module='hexv842_mm5_revised',source='woof/metal/shaders/hexv842_mm5_revised.metal',cuda_source='woof/hex/mpasphys/sfclay_mm5_v842.py',cuda_symbol=symbol,resources=resources,shapes={r[0]:[['any_1d']] for r in resources},parameters=[[v,'int32'] for v in params],threads_per_group=128,layout_version=1,arithmetic_policies=['default'])
items=[spec('hexv842_mm5_tables','mm5_init_tables',[['tables','float','read_write']],['n']),spec('hexv842_mm5_revised','mm5_columns',[['a','float','read_write'],['constants','float','read'],['tables','float','read']],['n','isfflx','isftcflx','iz0tlnd','scm_force_flux','shalwater_z0','optional_mask'])]
items[0]['shapes']={'tables':[['4004']]}
items[1]['shapes']={'a':[['48','n']],'constants':[['13']],'tables':[['4004']]}
for item in items:
 (fragment_dir/(item['name']+'.json')).write_text(json.dumps({'kernels':[item]},indent=2)+'\n')
print(len(fields),len(consts),len(s), 'gotos',s.count('goto'))
