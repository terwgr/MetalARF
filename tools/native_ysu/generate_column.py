"""Lower the selected MPAS v8.4.2 YSU column into Metal source.

The specialization is the native MPAS slot: nmix=0, flag_bep=false, optional
urban/tracer arrays absent. Species and top-down controls remain parameters.
The caller must provide 61*(nz+2)*count float workspace words.
"""
import argparse
import ast
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-root", type=Path, default=root)
destination = parser.parse_args().output_root
shader_dir = destination / "woof/metal/shaders"
fragment_dir = destination / "woof/metal/signature_fragments"
shader_dir.mkdir(parents=True, exist_ok=True)
fragment_dir.mkdir(parents=True, exist_ok=True)
source = (root / 'woof/hex/mpasphys/ysu_v842.py').read_text()
module_ast = ast.parse(source)
def literal(name):
    return next(ast.literal_eval(node.value) for node in module_ast.body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets))
body = literal('_CUDA_BODY')
array_names = literal('ARRAY_ARGUMENTS')
float_names = literal('FLOAT_ARGUMENTS')
name = 'hexv842_mpas_ysu_column_f32'
# MYNN and YSU share the pinned powf/expf/tanhf source routines.
math = (root/'woof/metal/shaders/hexv842_mpas_bl_mynn_edmf_condensation_f32.metal').read_text().split('constant float j0')[0]
math = math.replace('hexv842_mpas_bl_mynn_edmf_condensation_f32_bindings.metalh', name+'_bindings.metalh')
math += '''
static inline float ysu_add(float a,float b) { return mynn_add(a,b); }
static inline float ysu_sub(float a,float b) { return mynn_sub(a,b); }
static inline float ysu_mul(float a,float b) { return mynn_mul(a,b); }
static inline float ysu_div(float a,float b) { return mynn_div(a,b); }
#define mpas_ysu_powf mynn_pow
#define mpas_ysu_expf mynn_exp
#define mpas_ysu_tanhf mynn_tanh
#define mpas_ysu_sqrtf mynn_sqrt
static inline float ysu_abs(float a) { return as_type<float>(as_type<uint>(a)&0x7fffffffu); }
#define mpas_ysu_absf ysu_abs
#define YSU_DEVICE static inline
'''
body = body[body.index('YSU_DEVICE float ysu_square'):]
body = body.replace('return a < b ? a : b;', 'return soft32_less(as_type<uint>(a), as_type<uint>(b)) ? a : b;', 1)
body = body.replace('return a > b ? a : b;', 'return soft32_greater(as_type<uint>(a), as_type<uint>(b)) ? a : b;', 1)
body = body.replace('float *p;', 'device float *p;').replace('int *p;', 'device int *p;')
body = body.replace('YSU_DEVICE float &operator', 'device float &operator').replace('YSU_DEVICE int &operator', 'device int &operator')
body = body.replace('float *workspace', 'device float *workspace').replace('int *status', 'thread int *status')
body = body.replace('float & zi', 'thread float & zi')
# The MPAS driver supplies these, plus the output views. BEP/tracers remain absent.
active = {'dz8w2d','pi2d','tx','qvx','qcx','qix','utnp','vtnp','ttnp','qvtnp','qctnp','qitnp',
          'p2di','p2d','hpbl','ust','znt','xland','hfx','qfx','wspd','br','psim','psih','psfcpa',
          'kpbl1d','ux','vx','rthraten','ctopo','ctopo2','exch_hx','exch_mx','u10','v10','uox','vox','wstar','delta'}
assert active <= set(array_names)
resources = [[n,'int32' if n == 'kpbl1d' else 'float','read_write'] for n in array_names if n in active]
resources.append(['workspace','float','read_write'])
params = [['ncol','int32'],['offset','int32'],['count','int32'],['nz','int32'],
          ['f_qc','int32'],['f_qi','int32'],['topdown','int32']] + [[n,'float32'] for n in float_names]
typename = ''.join(x.title() for x in name.split('_'))
launch = f'\nkernel void {name}(constant {typename}Resources &r [[buffer(0)]], constant {typename}Params &cfg [[buffer(1)]], uint lane [[thread_position_in_grid]]) {{\n'
launch += 'if (lane >= uint(cfg.count)) return;\nint col=cfg.offset+int(lane), nz=cfg.nz;\nYsuArgs p{};\n'
for n in array_names:
    if n not in active:
        continue
    view = 'YsuIntView' if n == 'kpbl1d' else 'YsuView'
    launch += f'p.{n} = {view}{{r.{n}, col, cfg.ncol, nz}};\n'
for n in float_names:
    launch += f'p.{n}=cfg.{n};\n'
launch += '''p.its=1; p.ite=1; p.kte=nz; p.kme=nz+1; p.nmix=0; p.flag_bep=false;
p.f_qc=cfg.f_qc!=0; p.f_qi=cfg.f_qi!=0; p.ysu_topdown_pblmix=cfg.topdown==1;
int status=0;
bl_ysu_run(p, r.workspace+lane, cfg.count, &status);
}
'''
(shader_dir/(name+'.metal')).write_text(math+body+launch)
fragment={'kernels':[dict(name=name,module=name,source='woof/metal/shaders/'+name+'.metal',cuda_source='woof/hex/mpasphys/ysu_v842.py',cuda_symbol='ysu_v842_columns',resources=resources,shapes={x[0]:[['any_1d']] for x in resources},parameters=params,threads_per_group=32,layout_version=1,arithmetic_policies=['soft32-rn-nonftz','soft64-rn-bit-ops','portable-libm64'])]}
(fragment_dir/(name+'.json')).write_text(json.dumps(fragment,indent=2)+'\n')
print('Generated',name,'resources',len(resources),'source lines',len((math+body+launch).splitlines()))
