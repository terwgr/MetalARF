"""Lower native WSM6 reflectivity and its radar initializer to Metal."""
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools/mp28_metal')]
from translate import convert_function, convert_struct, untok
from woof.hex.mpasphys import wsm6_v842 as native

source = subprocess.run(['clang++', '-E', '-P', '-x', 'c++', '-'],
    input=native.CUDA_MODULES, capture_output=True, text=True, check=True).stdout

def function(name):
    match = re.search(r'__device__\s+(?:void|float)\s+'+name+r'\([^;]*?\)\s*\{', source)
    start = match.end(); end = start; depth = 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[match.start():end]

raw_struct = source.split('struct Wsm6State {', 1)[1].split('};', 1)[0]
core_fields = re.findall(r'^float (\w+);$', raw_struct.split('int nrbins;', 1)[0], flags=re.M)
struct = '\n'.join(line for line in raw_struct.splitlines()
                   if 'char *' not in line and 'Wsm6Workspace *' not in line)
struct = convert_struct('struct Wsm6State {'+struct+'};').replace('float', 'F').replace('cx', 'tc44::cd')
renames = dict(add='tc_add', sub='tc_sub', mul='tc_mul', div='tc_div', mp_pow='tc_pow',
               mp_exp='tc_exp', mp_log='tc_log', mp_log10='tc_log10', mp_sqrt='tc_sqrt',
               mp_min='tc_min', mp_max='tc_max')

def lower(text):
    text = untok(convert_function(text))
    text = re.sub(r'\bfloat\b', 'F', text)
    text = text.replace('Wsm6State & S', 'thread Wsm6State & S')
    text = re.sub(r'\bcx\s*\(', 'tc44::cd_make(', text)
    for old,new in renames.items():
        text = re.sub(r'\b'+old+r'\s*\(', new+'(', text)
    text = text.replace('thread thread ', 'thread ')
    if text.startswith('template '):
        pos = text.index('>') + 1
        return text[:pos]+'\nstatic inline '+text[pos:]
    return 'static inline '+text

init = function('radar_init')
init = '\n'.join(line for line in init.splitlines() if not re.search(r'S\.\w*string_\w*\s*=', line))
params = function('state_parameters')
# Core initialization is already source-derived on Metal. Radar adds only its
# own state parameters and the three mass laws from mp_wsm6_init.
params = '\n'.join(line for line in params.splitlines() if 'S.' not in line or
                  any('S.'+name in line for name in ('nrbins', 'slen', 'lamda_radar', 'basis', 'melt_outside')))
init_mass = function('mp_wsm6_init')
init_mass = '\n'.join(line for line in init_mass.splitlines() if re.search(r'S\.(?:xam|xbm|xmu)_', line))
radar = function('refl10cm_wsm6').replace('goto label195;', 'break;').replace('label195:;', '')
# The same fixed Maxwell-Garnett branch is used by the Thompson port.
radar = re.sub(r'rayleigh_soak_wetgraupel\(S, (.*?), cback, S\.mixingrulestring_[sg],.*?\);',
    lambda m: 'cback = tc44::refl_rayleigh_soak_wetgraupel('+','.join(m.group(1).split(',')[:-1])+', S.pi5, S.lamda4);', radar)
radar = lower(radar)
planes = {'F': [], 'Soft8': [], 'bool': []}
for typ in planes:
    pattern = r'\b'+typ+r' (\w+) \[ WSM6_KMAX \+ 3 \] ;'
    def plane(match):
        name=match.group(1); planes[typ].append(name)
        workspace={'F':'wf','Soft8':'wd','bool':'wb'}[typ]
        return f'device {typ} *{name}={workspace}; {workspace}+=nz+3;'
    radar = re.sub(pattern, plane, radar)
assert tuple(map(len, planes.values())) == (10,6,3), planes
radar = radar.replace('int kts , int kte )', 'int kts , int kte, int nz, device F *wf, device Soft8 *wd, device bool *wb )')
name = 'hexv842_wsm6_radar_f32'
title = 'Hexv842Wsm6RadarF32'
header = '''#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
#include "thompson_column_helpers.metalh"
using namespace ntc;
'''
# The source initializer's 89 core words are stored in the existing state ABI.
load = ''.join(f'S.{field}=((device const F*)r.state)[{i}*p.state_stride];\n' for i,field in enumerate(core_fields))
kernel = '''
#include "hexv842_wsm6_radar_f32_bindings.metalh"
kernel void hexv842_wsm6_radar_f32(constant Hexv842Wsm6RadarF32Resources &r [[buffer(0)]], constant Hexv842Wsm6RadarF32Params &p [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
 if(gid>=uint(p.count))return;
 int col=p.first+int(gid),nz=p.nz,nc=p.nc;
 Wsm6State S{};
'''+load+'''
 state_parameters(S);
 F denr=((device const F*)r.constants)[16],dens=100.f;
'''+lower('__device__ void mass(Wsm6State &S) {'+init_mass+'}').split('{',1)[1].rsplit('}',1)[0]+'''
 radar_init(S);
 device F *wf=(device F*)r.ws_f+ulong(gid)*18*(nz+3);
 device Soft8 *wd=(device Soft8*)r.ws_d+ulong(gid)*6*(nz+3);
 device bool *wb=(device bool*)r.ws_b+ulong(gid)*3*(nz+3);
 device F *t=wf,*qv=t+nz+3,*qr=qv+nz+3,*qs=qr+nz+3,*qg=qs+nz+3,*pr=qg+nz+3,*dbz=pr+nz+3,*zp=dbz+nz+3;
 wf=zp+nz+3;
 for(int k=0;k<nz;++k){int x=k*nc+col;
 t[k]=tc_mul(((device const F*)r.th)[x],((device const F*)r.pii)[x]);
 qv[k]=((device const F*)r.qv)[x];qr[k]=((device const F*)r.qr)[x];
 qs[k]=((device const F*)r.qs)[x];qg[k]=((device const F*)r.qg)[x];pr[k]=((device const F*)r.p)[x];
 F z=((device const F*)r.zgrid)[x],z0=((device const F*)r.zgrid)[col],z1=((device const F*)r.zgrid)[x+nc];
 zp[k]=tc_add(tc_sub(z,z0),tc_mul(.5f,tc_sub(z1,z)));
 }
 refl10cm_wsm6(S,qv,qr,qs,qg,t,pr,dbz,1,nz,nz,wf,wd,wb);
 F maximum=-35.f;int kp=0;
 for(int k=0;k<nz;++k){dbz[k]=tc_max(dbz[k],-35.f);((device F*)r.refl)[k*nc+col]=dbz[k];maximum=tc_max(maximum,dbz[k]);if(zp[k]<1000.f)kp=k;}
 ((device F*)r.refl_max)[col]=maximum;
 F value=dbz[kp];
 if(kp+1<nz){F w1=tc_div(tc_sub(zp[kp+1],1000.f),tc_sub(zp[kp+1],zp[kp]));value=tc_add(tc_mul(w1,dbz[kp]),tc_mul(tc_sub(1.f,w1),dbz[kp+1]));}
 ((device F*)r.refl_1km)[col]=value;
 ((device F*)r.refl_1km_max)[col]=p.running_max?tc_max(((device F*)r.refl_1km_max)[col],value):value;
}
'''
body = '\n'.join((struct, lower(function('gammln')), lower(function('wgamma')), lower(params), lower(init), radar, kernel))
(ROOT/f'woof/metal/shaders/{name}.metal').write_text(header+body)
resources = [[n,'float','read'] for n in ('qv','qr','qs','qg','th','pii','p','zgrid','state','constants')]
resources += [[n,'float','read_write'] for n in ('refl','refl_max','refl_1km','refl_1km_max','ws_f')]
resources += [['ws_d','uint64','read_write'],['ws_b','uint32','read_write']]
parameters = [[n,'int32'] for n in ('nz','nc','first','count','state_stride','running_max')]
fragment={'kernels':[dict(name=name,module=name,source=f'woof/metal/shaders/{name}.metal',cuda_source='woof/hex/mpasphys/wsm6_refl_start_v842.py',cuda_symbol='wsm6_start_radar',resources=resources,shapes={n:[['any_1d']] for n,_,_ in resources},parameters=parameters,threads_per_group=32,layout_version=1,arithmetic_policies=['default'])]}
(ROOT/f'woof/metal/signature_fragments/{name}.json').write_text(json.dumps(fragment,indent=2)+'\n')

rh_source = subprocess.run(['clang++','-E','-P','-x','c++','-'],
    input=native.WSM6_RELHUM_SOURCE,capture_output=True,text=True,check=True).stdout
renames.update(__fadd_rn='tc_add',__fsub_rn='tc_sub',__fmul_rn='tc_mul',__fdiv_rn='tc_div',
               aa_fmaxf='tc_max',aa_fminf='tc_min')
rh_functions=[]
for symbol in ('wsm6_rslf','wsm6_rsif'):
    match=re.search(r'__device__ float '+symbol+r'\([^;]*?\)\s*\{',rh_source)
    end=match.end();depth=1
    while depth:
        depth+=(rh_source[end]=='{')-(rh_source[end]=='}');end+=1
    rh_functions.append(lower(rh_source[match.start():end]))
rh_name='hexv842_wsm6_relhum_f32'
rh_kernel='''
#include "hexv842_wsm6_relhum_f32_bindings.metalh"
kernel void hexv842_wsm6_relhum_f32(constant Hexv842Wsm6RelhumF32Resources &r [[buffer(0)]],constant Hexv842Wsm6RelhumF32Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if(gid>=uint(p.nz*p.nc))return;
 int i=int(gid)%p.nc;
 if(i>=p.ns){r.relhum[gid]=0.f;return;}
 F t=tc_mul(((device const F*)r.th)[gid],((device const F*)r.pii)[gid]);
 F pressure=((device const F*)r.pres)[gid],qvs=wsm6_rslf(pressure,t);
 if(tc_sub(t,273.16f)<=0.f)qvs=wsm6_rsif(pressure,t);
 ((device F*)r.relhum)[gid]=tc_mul(tc_div(((device const F*)r.qv)[gid],qvs),100.f);
}
'''
with (ROOT/f'woof/metal/shaders/{name}.metal').open('a') as stream:
    stream.write('\n'.join(rh_functions)+rh_kernel)
rh_resources=[[n,'float','read'] for n in ('th','pii','qv','pres')]+[['relhum','float','read_write']]
rh_fragment={'kernels':[dict(name=rh_name,module=name,source=f'woof/metal/shaders/{name}.metal',cuda_source='woof/hex/mpasphys/wsm6_v842.py',cuda_symbol='wsm6_relhum',resources=rh_resources,shapes={n:[['any_1d']] for n,_,_ in rh_resources},parameters=[[n,'int32'] for n in ('nz','nc','ns')],threads_per_group=128,layout_version=1,arithmetic_policies=['default'])]}
(ROOT/f'woof/metal/signature_fragments/{rh_name}.json').write_text(json.dumps(rh_fragment,indent=2)+'\n')
