"""Compare real dry-periodic dycore steps with host-compiled selected CUDA.

The host backend adapts CUDA kernel execution to C++; its bookkeeping uses
bit copies and its diagnostics dispatches the selected CUDA source. The
Metal step uses production admission and dispatch unmodified by the harness.
This is host-source agreement, not NVIDIA-device or benchmark certification.
"""
from __future__ import annotations

import argparse
import ctypes
import copy
from dataclasses import replace
import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from types import MappingProxyType
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.mp28_metal.parse import closure,split_top
from tools.smag_wrf_exact.compare import compare_words


def compare_float64(actual, reference):
    a,b=(np.ascontiguousarray(v,dtype=np.float64) for v in (actual,reference))
    if a.shape != b.shape:
        raise ValueError(f'comparison shape mismatch: {a.shape} != {b.shape}')
    unequal=a.view(np.uint64) != b.view(np.uint64)
    return {'words':int(a.size),'different_words':int(unequal.sum()),
            'first':np.argwhere(unequal)[:3].tolist()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--steps',type=int,default=2)
    ap.add_argument('--nwp-diagnostics',action='store_true',help='compare strict UH with source and checkpoint windows')
    ap.add_argument('--base3d',action='store_true')
    ap.add_argument('--terrain',action='store_true',help='probe strict moist nonuniform terrain')
    ap.add_argument('--private-terrain',action='store_true',help='probe unadmitted nonuniform terrain base')
    ap.add_argument('--terrain-height',type=float,default=80.)
    ap.add_argument('--terrain-relief',type=float,default=40.)
    ap.add_argument('--igw-sounding',action='store_true')
    ap.add_argument('--igw-2d',action='store_true',help='probe the stock one-row IGW source chain')
    ap.add_argument('--igw-v-seed',action='store_true',help='seed matching nonzero periodic V faces in the IGW probe')
    ap.add_argument('--open',action='store_true',help='probe admitted open lateral boundaries')
    ap.add_argument('--private-open',action='store_true',help='probe both open lateral boundaries before admission')
    ap.add_argument('--specified',action='store_true')
    ap.add_argument('--boundary-crossover',action='store_true')
    ap.add_argument('--moist',action='store_true')
    ap.add_argument('--km',type=int,choices=(2,3),help='probe admitted TKE or 3-D mixing')
    ap.add_argument('--private-km',type=int,choices=(2,3),help='probe unadmitted TKE or 3-D mixing')
    ap.add_argument('--zero-tke',action='store_true',help='start TKE at zero')
    ap.add_argument('--seed-flux',action='store_true',help='select unforced surface and TKE seed')
    ap.add_argument('--isotropic',action='store_true',help='use isotropic mixing lengths')
    ap.add_argument('--tke-budget',action='store_true',help='compare TKE budget source fields')
    ap.add_argument('--km4',action='store_true', help='probe strict km_opt=4 mixing')
    ap.add_argument('--private-km4',action='store_true', help='omit km4 admission for private probing')
    ap.add_argument('--coordinate2',action='store_true',help='probe strict coordinate TKE mixing')
    ap.add_argument('--private-coordinate2',action='store_true',help='probe unadmitted coordinate TKE mixing')
    ap.add_argument('--private-coordinate4',action='store_true',help='probe unadmitted coordinate km4 mixing')
    ap.add_argument('--coordinate4',action='store_true',help='probe strict coordinate km4 mixing')
    ap.add_argument('--km1',action='store_true', help='probe strict constant-K mixing')
    ap.add_argument('--private-km1',action='store_true', help='probe unadmitted strict constant-K mixing')
    ap.add_argument('--surface-only',action='store_true',help='constant-K with zero diffusivities and prescribed surface flux')
    ap.add_argument('--specified-moist',action='store_true',help='probe specified moist physics-off without mixing')
    ap.add_argument('--private-specified-moist',action='store_true',help='omit specified moist admission for private probing')
    ap.add_argument('--surface-flux',type=int,choices=(0,2),help='exercise prescribed mixing surface flux')
    ap.add_argument('--dudhia',action='store_true', help='strict Dudhia physics/dycore source comparison')
    ap.add_argument('--goddard',action='store_true', help='compare strict paired Goddard radiation and restart')
    ap.add_argument('--goddard-levels',type=int,help='probe tiered Goddard at 50 through 128 levels')
    ap.add_argument('--classic',action='store_true', help='compare strict RRTM–Dudhia radiation and restart')
    ap.add_argument('--goddard-capture-band',type=Path,help='capture the first actual host and Metal band call')
    ap.add_argument('--noah',action='store_true', help='compose revised MM5 and warm Noah with Dudhia')
    ap.add_argument('--climate-diagnostics',action='store_true',help='compare native one-minute climate statistic windows')
    ap.add_argument('--precip-buckets',action='store_true',help='exercise both precipitation accumulation windows')
    ap.add_argument('--solar-generation',choices=('wrf_461','wrf_480'),default='wrf_461')
    ap.add_argument('--checkpoint',action='store_true')
    ap.add_argument('--tree-checkpoint',action='store_true')
    ap.add_argument('--stock-coordinates',action='store_true', help='hybrid and hypsometric option 2')
    ap.add_argument('--checkpoint-after',type=int,default=3)
    ap.add_argument('--swrad-scat',type=float,default=1.0)
    ap.add_argument('--moist-adv-opt',type=int,choices=(0,1),default=1)
    ap.add_argument('--vscalar5',action='store_true',help='select fifth-order vertical scalar advection')
    ap.add_argument('--private-vscalar5',action='store_true',help='probe unadmitted fifth-order vertical scalar advection')
    ap.add_argument('--upper-wind',action='store_true',help='exercise NOAA upper-wind limiter')
    ap.add_argument('--upper-wind-control',action='store_true',help='seed same winds without NOAA limiter')
    ap.add_argument('--rayleigh',action='store_true',help='exercise implicit upper damping')
    ap.add_argument('--rayleigh-control',action='store_true',help='run the same case without implicit upper damping')
    ap.add_argument('--w-damp',action='store_true',help='exercise vertical velocity limiter')
    ap.add_argument('--w-damp-control',action='store_true',help='run the same stretched grid without the limiter')
    ap.add_argument('--fork-factor2',type=float,default=.04,help='NOAA moisture filter factor')
    ap.add_argument('--diff6-fork',action='store_true',help='exercise the NOAA sixth-order form')
    ap.add_argument('--diff6-slope',action='store_true',help='exercise the sixth-order terrain taper')
    ap.add_argument('--diff6-unlimited',action='store_true',help='exercise dry unlimited sixth-order filter')
    ap.add_argument('--diff6',action='store_true',help='exercise strict sixth-order filter')
    ap.add_argument('--emdiv',action='store_true',help='exercise external-mode damping')
    ap.add_argument('--open-top',action='store_true',help='exercise the open top boundary')
    ap.add_argument('--vmom5',action='store_true',help='select fifth-order vertical momentum advection')
    ap.add_argument('--vertical-pattern',action='store_true',help='exercise vertically varying fields on nine levels')
    ap.add_argument('--vertical-min-grid',action='store_true',help='use the minimum five-level vertical grid')
    ap.add_argument('--h5',action='store_true',help='select fifth-order geopotential advection')
    ap.add_argument('--private-h5',action='store_true',help='probe unadmitted fifth-order geopotential')
    modes=ap.add_mutually_exclusive_group()
    modes.add_argument('--mapped',action='store_true')
    modes.add_argument('--maps-only',action='store_true')
    modes.add_argument('--rotation-only',action='store_true');args=ap.parse_args()
    if args.goddard or args.classic:
        args.dudhia=True
    if args.goddard:
        args.solar_generation='wrf_480'
    if args.goddard_levels is not None and (not args.goddard or not 50 <= args.goddard_levels <= 128):
        ap.error('--goddard-levels requires --goddard and a depth from 50 through 128')
    if args.goddard and args.classic:
        ap.error('--goddard and --classic are mutually exclusive')
    if args.igw_2d:
        args.igw_sounding = True
    if args.igw_v_seed and not args.igw_2d:
        ap.error('--igw-v-seed requires --igw-2d')
    if args.open and args.private_open: ap.error('choose one open-boundary probe')
    args.open_mode=args.open or args.private_open
    if args.nwp_diagnostics and not (args.specified and args.tree_checkpoint and args.vertical_pattern and args.steps >= 7):
        ap.error('--nwp-diagnostics requires specified tree checkpoint, vertical pattern and at least seven steps')
    if args.diff6_fork and not (args.diff6 or args.diff6_unlimited): ap.error('--diff6-fork requires a sixth-order filter')
    if args.diff6_unlimited and (args.diff6 or args.moist or args.dudhia): ap.error('unlimited sixth-order probe requires dry state and excludes --diff6')
    args.diff6=args.diff6 or args.diff6_unlimited
    if args.upper_wind and args.upper_wind_control: ap.error('choose one upper-wind mode')
    if (args.upper_wind or args.upper_wind_control) and not (args.rayleigh and args.mapped): ap.error('upper-wind probe requires --rayleigh --mapped')
    if args.rayleigh and args.rayleigh_control: ap.error('choose one upper-damping mode')
    if args.w_damp and args.w_damp_control: ap.error('choose one w-damping mode')
    args.damp_grid=args.w_damp or args.w_damp_control
    if args.vscalar5 and args.private_vscalar5: ap.error('choose one vertical-order probe')
    args.vertical5=args.vscalar5 or args.private_vscalar5
    args.vertical_pattern=args.vertical_pattern or args.vertical5 or args.vmom5
    if args.h5 and args.private_h5: ap.error('choose one horizontal-order probe')
    args.horizontal5=args.h5 or args.private_h5
    if args.terrain and args.private_terrain: ap.error('choose one terrain probe')
    args.terrain_mode=args.terrain or args.private_terrain
    if args.terrain_mode:
        args.base3d=True
    if args.coordinate2:
        if args.km or args.private_km or args.private_coordinate2: ap.error('choose one alternate closure')
        args.km=2
    if args.private_coordinate2:
        if args.km or args.private_km: ap.error('choose one alternate closure')
        args.private_km=2
    if args.km and args.private_km: ap.error('choose one alternate closure')
    args.selected_km=args.km or args.private_km
    if (args.zero_tke or args.seed_flux) and args.selected_km != 2:
        ap.error('TKE options require km2')
    if args.tke_budget and args.selected_km != 2:
        ap.error('--tke-budget requires --private-km 2')
    args.coordinate4 = args.coordinate4 or args.private_coordinate4
    args.mixing = args.selected_km or args.km4 or args.private_km4 or args.km1 or args.private_km1 or args.coordinate4
    if args.open_mode and (args.specified or args.dudhia or args.mixing or args.base3d or args.diff6):
        ap.error('open probe requires flat physics-off configuration with mixing and filter disabled')
    if args.terrain_mode and (not args.moist or (args.mixing and not (args.km4 or args.km or args.coordinate4 or args.private_terrain))):
        ap.error('terrain requires moisture and a supported mixing profile')
    if args.selected_km and (args.km1 or args.private_km1 or args.km4 or args.private_km4 or args.coordinate4):
        ap.error('choose one mixing package')
    if (args.km1 or args.private_km1) and (args.km4 or args.private_km4 or args.coordinate4):
        ap.error('choose one mixing package')
    if args.surface_only and not ((args.km1 or args.private_km1) and args.surface_flux is not None):
        ap.error('--surface-only requires --km1 and --surface-flux')
    args.private_mixing = args.private_km or args.private_km4 or args.private_km1 or args.private_coordinate4
    args.specified_moist = args.specified_moist or args.private_specified_moist
    if args.specified_moist and not (args.specified and args.moist and not args.mixing and not args.dudhia):
        ap.error('--specified-moist requires --specified --moist and physics/mixing off')
    if args.surface_flux is not None and not args.mixing:
        ap.error('--surface-flux requires a mixing probe')
    if args.mixing and not args.mapped:
        ap.error('mixing probes require --mapped')
    if args.stock_coordinates and not (args.specified and (args.dudhia or args.mixing or args.specified_moist or args.terrain_mode)):
        ap.error('--stock-coordinates requires a specified physics or mixing probe')
    if args.tree_checkpoint:
        if not (args.igw_2d or args.open_mode or (args.specified and args.boundary_crossover and (args.dudhia or args.mixing or args.specified_moist))
                or (args.mixing and args.dudhia and not args.specified)
                or (args.terrain_mode and args.specified and args.boundary_crossover)):
            ap.error('--tree-checkpoint requires specified crossover or periodic km4 Dudhia')
        args.checkpoint=True
    if args.checkpoint and not 1 <= args.checkpoint_after < args.steps:
        ap.error('--checkpoint-after must name a step before the final step')
    if args.noah and not args.dudhia: ap.error('--noah requires --dudhia')
    if args.dudhia and args.steps < 6:
        ap.error("--dudhia requires at least six steps to exercise radiation cadence")
    if args.checkpoint and not (args.igw_2d or args.open_mode or args.terrain_mode or args.mixing or args.specified_moist or (args.dudhia and args.steps >= 7)):
        ap.error("--checkpoint requires --km4 or --dudhia with at least seven steps")
    if args.dudhia:
        args.moist = True
        if (args.base3d and not args.terrain_mode) or args.igw_sounding:
            ap.error("--dudhia currently covers the flat sounding or private terrain")
    if args.boundary_crossover and (not args.specified or args.steps < 2):
        ap.error('--boundary-crossover requires --specified and at least two steps')
    from woof.wrf_exact import ENABLED,DIAGNOSTICS_ENABLED,ADVECTION_ENABLED,BIGSTEP_ENABLED,DIFFUSION_ENABLED
    if not all((ENABLED,DIAGNOSTICS_ENABLED,ADVECTION_ENABLED,BIGSTEP_ENABLED,DIFFUSION_ENABLED)):
        raise ValueError('the paired scope requires strict root and all four controls')
    from tools.thompson_real_column_parity import host_backend
    host_backend.install()
    from woof.core import kernels as K
    from woof.core.kernel_shape import shape_defines
    from woof.core import dycore
    old_get,old_get_def=K.get_kernel,K.get_kernel_int_defines
    loaded={};records={}
    # Compile the actual runtime-generated CUDA elementwise bodies on host.
    tempdir=tempfile.TemporaryDirectory(prefix='woof-strict-dry-elementwise-')
    prior_elementwise=dycore.cp.ElementwiseKernel
    ptr=ctypes.POINTER(ctypes.c_float)
    def address(value): return value.ctypes.data_as(ptr)
    def elementwise(inputs,outputs,body,name,**options):
        if not name.startswith('gpuwm_pd_fold_sources_') and name not in ('gpuwm_couple_momentum','gpuwm_omega_column_scan','gpuwm_update_scalar','gpuwm_compose_pbl_radiation_cumulus'):
            return prior_elementwise(inputs,outputs,body,name,**options)
        mapped='msf' in inputs
        prelude = ("#include <stddef.h>\ntypedef float T;\n"
            "static float __fadd_rn(float a,float b){volatile float r=a+b;return r;}\n"
            "static float __fsub_rn(float a,float b){volatile float r=a-b;return r;}\n"
            "static float __fmul_rn(float a,float b){volatile float r=a*b;return r;}\n"
            "static float __fdiv_rn(float a,float b){volatile float r=a/b;return r;}\n")
        if name.startswith('gpuwm_pd_fold_sources_'):
            count=int(name.rsplit('_',1)[1])
            names=['q0']+[f's{j}' for j in range(count)]+['chm0']
            params=''.join(f'const float* {n}_ptr,' for n in names)
            loads=''.join(f'T {n}={n}_ptr[i];' for n in names)
            unit=prelude+f'extern "C" void reference({params}float dt,float* result,int count){{for(int i=0;i<count;++i){{{loads} T out;'+body+'result[i]=out;}}\n'
        elif name == 'gpuwm_compose_pbl_radiation_cumulus':
            unit=prelude+('extern "C" void reference(const float* ap,const float* bp,'
                'const float* cp,float* op,int count){for(int i=0;i<count;++i){'
                'T a=ap[i],b=bp[i],c=cp[i],out;'+body+'op[i]=out;}}\n')
        elif name == 'gpuwm_update_scalar':
            # Strict install normalizes the final ElementwiseKernel compile
            # to no-FTZ. This scope never selects the default clamp branch.
            if 'isnan' in body:
                raise ValueError('strict reference does not select scalar clamp')
            optional=[n for n in ('msft','physics','fixed') if re.search(r'\b'+n+r'\b',inputs)]
            extras=''.join('const float* '+n+',' for n in optional)
            loads=''.join('T '+n+'='+n+'_ptr[i];' for n in optional if n != 'msft')
            # Raw map uses its name; value operands load at the current index.
            for n in optional:
                if n != 'msft': extras=extras.replace('* '+n+',','* '+n+'_ptr,')
            unit=(prelude+'extern "C" void reference(const float* q0_ptr,const float* tend_ptr,'
                'const float* c1h,const float* c2h,const float* mu0,const float* mu,'+extras+
                'float dt_eff,int ncol,float* out,int count){'
                'for(size_t i=0;i<(size_t)count;++i){T q0=q0_ptr[i],tend=tend_ptr[i],q;'
                +loads+body+'out[i]=q;}}\n')
        elif name == 'gpuwm_couple_momentum':
            unit=prelude+('extern "C" void reference(const float* winds,const float* c1h,'
                'const float* c2h,const float* muface,'
                + ('const float* msf,' if mapped else '') + 'float* out,int ncol,int count){'
                'for(size_t i=0;i<(size_t)count;++i){T wind=winds[i],flux;'+body+'out[i]=flux;}}\n')
        else:
            unit=prelude+('extern "C" void reference(const float* ru,const float* rv,const float* dnw,'
                'const float* c1h,' + ('const float* msft,' if mapped else '')
                + 'float rdx,float rdy,int nz,int ny,int nx,float* ww){'
                'for(size_t i=0;i<(size_t)(ny*nx);++i){'+body+'}}\n')
        digest=hashlib.sha256(unit.encode()).hexdigest()
        path=Path(tempdir.name)/digest
        cpp=path.with_suffix('.cpp');lib=path.with_suffix('.dylib')
        cpp.write_text(unit)
        flags=['-O2','-std=c++17','-ffp-contract=off','-fno-unsafe-math-optimizations','-dynamiclib']
        subprocess.run(['clang++',*flags,str(cpp),'-o',str(lib)],check=True,capture_output=True,text=True)
        dll=ctypes.CDLL(str(lib));fn=dll.reference;fn.restype=None
        records['elementwise:'+name]={'body_sha256':hashlib.sha256(body.encode()).hexdigest(),
            'host_unit_sha256':digest,'input_declaration':inputs,'output_declaration':outputs,
            'cuda_options':list(options.get('options',())),
            'effective_cuda_options':list(__import__('woof.wrf_exact',fromlist=['effective_options']).effective_options(options.get('options',()))),
            'host_flags':flags}
        if name.startswith('gpuwm_pd_fold_sources_'):
            fn.argtypes=[ptr]*len(names)+[ctypes.c_float,ptr,ctypes.c_int]
            def invoke(*args,**kwargs):
                *arrays,dt,out=args
                fn(*map(address,arrays),float(dt),address(out),int(out.size))
                return out
        elif name == 'gpuwm_compose_pbl_radiation_cumulus':
            fn.argtypes=[ptr]*4+[ctypes.c_int]
            def invoke(a,b,c,out):
                fn(*map(address,(a,b,c,out)),int(out.size))
                return out
        elif name == 'gpuwm_update_scalar':
            fn.argtypes=[ptr]*(6+len(optional))+[ctypes.c_float,ctypes.c_int,ptr,ctypes.c_int]
            def invoke(*args,**kwargs):
                *arrays,dt,ncol,out=args
                fn(*map(address,arrays),float(dt),int(ncol),address(out),int(out.size))
                return out
        elif name == 'gpuwm_couple_momentum':
            fn.argtypes=[ptr]*(6 if mapped else 5)+[ctypes.c_int]*2
            def invoke(*args,**kwargs):
                *arrays,ncol,out=args
                fn(*map(address,(*arrays,out)),int(ncol),int(out.size))
                return out
        else:
            fn.argtypes=[ptr]*(5 if mapped else 4)+[ctypes.c_float]*2+[ctypes.c_int]*3+[ptr]
            def invoke(*args,**kwargs):
                *arrays,rdx,rdy,nz,ny,nx,ww=args
                fn(*map(address,arrays),float(rdx),float(rdy),int(nz),int(ny),int(nx),address(ww))
                return ww
        invoke._host_library=dll
        if name == 'gpuwm_update_scalar':
            seed=np.array([1,0x007fffff],dtype=np.uint32).view(np.float32)
            zero=np.zeros(2,np.float32); one=np.ones(2,np.float32)
            arrays=[seed,zero,np.zeros(1,np.float32),np.ones(1,np.float32),zero,zero]
            arrays += [one if n == 'msft' else zero for n in optional]
            out=np.empty(2,np.float32)
            invoke(*arrays,np.float32(1),np.int32(2),out)
            if not np.array_equal(out.view(np.uint32),seed.view(np.uint32)):
                raise AssertionError('captured scalar host reference flushed subnormals')
            records['elementwise:'+name]['subnormal_words_checked']=2
        return invoke
    dycore.cp.ElementwiseKernel=elementwise
    prefix=('\n'.join(f'#define {name} 1' for name in (
        'GPUWM_WRF_EXACT','GPUWM_WRF_EXACT_D_DIAGNOSTICS',
        'GPUWM_WRF_EXACT_C_ADVECTION','GPUWM_WRF_EXACT_C_BIGSTEP',
        'GPUWM_WRF_EXACT_C_DIFFUSION'))+'\n')

    def selected(name,func,defines=()):
        key=(name,func,tuple(defines))
        if key not in loaded:
            assembled=prefix+(K.module_source_int_defines(name,tuple(defines)) if defines else K.module_source(name))
            if name == 'goddard':
                # The Goddard closure includes a templated GArray type, so
                # keep the complete selected source as the column probe does.
                source=assembled
            else:
                expanded=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],
                    input=assembled,text=True,capture_output=True,check=True).stdout
                items,_=closure(split_top(expanded),(func,))
                source='\n'.join(items)
            # Host compilers cannot assemble PTX RN instructions. Keep the
            # source's per-operation boundaries with volatile binary32 stores.
            pattern = r'asm\("(add|sub|mul|div)\.rn\.f32 %0, %1, %2;"\s*:\s*"=f"\(r\)\s*:\s*"f"\(a\),\s*"f"\(b\)\);'
            operators = {'add': '+', 'sub': '-', 'mul': '*', 'div': '/'}
            host_source, rewrites = re.subn(pattern, lambda m:
                'volatile real host_rounded = a ' + operators[m.group(1)] + ' b; r = host_rounded;', source)
            renames = 0
            if (name == 'surface_w' and func == 'set_surface_w') or name == 'diff_opt1':
                # The generic host launcher uses bx/by as launch dimensions;
                # rename the source boundary arguments to avoid C++ collisions.
                host_source, renames = re.subn(r'\b(bx|by)\b',
                    lambda m: 'boundary_' + m.group(), host_source)
            records[str(key)]={'assembled_sha256':hashlib.sha256(assembled.encode()).hexdigest(),
                'closure_sha256':hashlib.sha256(source.encode()).hexdigest(),
                'host_closure_sha256':hashlib.sha256(host_source.encode()).hexdigest(),
                'ptx_rn_rewrites':rewrites, 'host_boundary_identifier_renames':renames}
            loaded[key]=host_backend.HostModule('strict_dry_'+name+'_'+func,
                'struct alignas(16) float4 { float x,y,z,w; };\n'+host_source).get_function(func)
        return loaded[key]

    def get(name,func,shape=()):
        if args.noah and name == 'noah':
            if name not in loaded:
                source = prefix + '#define __launch_bounds__(...)\n' + K.module_source(name)
                loaded[name] = host_backend.HostModule('strict_noah_chain', source)
                records['noah_source'] = {'assembled_sha256': hashlib.sha256(source.encode()).hexdigest()}
            return loaded[name].get_function(func)
        defs=shape_defines(name,func,shape) if shape else ()
        return selected(name,func,defs)

    goddard_host_stages={};goddard_metal_stages={}
    def get_def(name,func,defines,shape=()):
        kernel=selected(name,func,tuple(defines)+(shape_defines(name,func,shape) if shape else ()))
        if not args.goddard or name != 'goddard': return kernel
        def capture(grid,block,values,*rest,**kwargs):
            result=kernel(grid,block,values,*rest,**kwargs)
            if func not in goddard_host_stages:
                goddard_host_stages[func]={param.name:value.copy() for param,value in zip(kernel.params,values)
                    if isinstance(value,np.ndarray) and param.name!='lut'}
            return result
        return capture

    for module in tuple(sys.modules.values()):
        if module is None or not getattr(module,'__name__','').startswith('woof'):
            continue
        for attr,value in tuple(vars(module).items()):
            if value is old_get: setattr(module,attr,get)
            elif value is old_get_def: setattr(module,attr,get_def)
    K.get_kernel,K.get_kernel_int_defines=get,get_def

    if args.mixing:
        original_add_arrays=dycore.prepare_add_arrays
        def host_add_arrays(state,pairs):
            if not getattr(state,'_host_setup_state',False):
                return original_add_arrays(state,pairs)
            # Both CUDA entrypoints call the same glue_add_chunk.  Host
            # arrays lack CuPy device pointers, so launch its single-row
            # entrypoint per pair without changing the arithmetic body.
            kernel=selected('bandwidth_glue','glue_add')
            def launch():
                for src,dst in pairs:
                    kernel(((dst.size+511)//512,),(128,),
                           (src,dst,np.uint64(dst.size)))
            return launch
        dycore.prepare_add_arrays=host_add_arrays

    original_book=dycore._prepare_bookkeeping
    def bookkeeping(state,pairs,*,zero=False,field_names=None):
        if not getattr(state,'_host_setup_state',False):
            return original_book(state,pairs,zero=zero,field_names=field_names)
        def launch():
            for src,dst in pairs:
                if zero: dst.fill(np.float32(0))
                else: np.copyto(dst,src)
        return launch
    dycore._prepare_bookkeeping=bookkeeping

    original_diag=dycore.update_diagnostics
    def diagnostic(state,hypsometric_opt=1,window=None,muts=None):
        if not getattr(state,'_host_setup_state',False):
            return original_diag(state,hypsometric_opt,window=window,muts=muts)
        nz,ny,nx=state.p.shape
        j0,i0,nyw,nxw=(0,0,ny,nx) if window is None else window
        name='calc_p_alpha_carried_muts' if muts is not None else 'calc_p_alpha'
        kernel=get('diagnostics',name,(nz,ny,nx) if muts is None else ())
        values={name:getattr(state,name) for name in (
            'thp','php','mup','thb','phb','alb','pb','rdnw','c1h','c2h',
            'c3h','c4h','c3f','c4f','dc3f','dc4f','p','al','alt','p_perturbation')}
        values.update(mub=state.mub2d,dphbr=state.dphb_resid,
            muts=muts,qv=state.qv if state.qv is not None else state.thp,p_top=np.float32(0 if state.p_top is None else state.p_top),
            hypso=np.int32(hypsometric_opt),moist=np.int32(state.qv is not None),base3d=np.int32(state.thb.ndim==3),
            nz=np.int32(nz),ny=np.int32(ny),nx=np.int32(nx),j0=np.int32(j0),i0=np.int32(i0),
            nyw=np.int32(nyw),nxw=np.int32(nxw))
        kernel(((nyw*nxw+255)//256,),(256,),tuple(
            values[p.name.removeprefix('gpuwm_runtime_')] for p in kernel.params))
    dycore.update_diagnostics=diagnostic

    from woof.config import RunConfig
    from woof.core.grid import make_vertical_coord,make_base_state
    from woof.core.state import init_theta_perturbation,adopt_wrf_theta_carrier
    from woof.metal.array import MetalArrayModule
    from woof.metal.runtime import MetalContext
    cfg=RunConfig(nx=12 if args.specified else 8,ny=11 if args.specified else 7,
                  nz=args.goddard_levels or (9 if args.vertical_pattern and not args.vertical_min_grid else 5),dx=1000.,dy=1000.,ztop=12000. if args.goddard else 9000. if args.nwp_diagnostics else 5000.,dt=1.,run_seconds=2.,
                  nwp_diagnostics=int(args.nwp_diagnostics),
                  terrain_opt=int(args.base3d),specified=args.specified,
                  moist=args.moist,moist_adv_opt=args.moist_adv_opt)
    if args.igw_2d:
        from woof.verify.cases import igw
        cfg=igw.default_config()
    if args.mixing:
        cfg=replace(cfg,diff_opt=2,km_opt=1,khdif=75.,kvdif=2.) if (args.km1 or args.private_km1) else replace(cfg,diff_opt=2,km_opt=4)
        if args.selected_km: cfg=replace(cfg,km_opt=args.selected_km,tke_budget=int(args.tke_budget),mix_isotropic=int(args.isotropic))
        if args.seed_flux: cfg=replace(cfg,isfflx=0)
        if args.coordinate4 or args.private_coordinate2 or args.coordinate2: cfg=replace(cfg,diff_opt=1)
        if args.surface_only: cfg=replace(cfg,khdif=0.,kvdif=0.)
        if args.surface_flux is not None:
            cfg=replace(cfg,isfflx=args.surface_flux,tke_heat_flux=0.05,
                        tke_drag_coefficient=0.002)
    if args.open_mode:
        cfg=replace(cfg,open_x=True,open_y=True)
    if args.open_mode or args.specified_moist or (args.terrain_mode and not args.mixing):
        cfg=replace(cfg,km_opt=0,km_opt_zero_acknowledgement='no-horizontal-mixing-operator-v1')
    if args.dudhia:
        cfg=replace(cfg, ra_lw_physics=5 if args.goddard else 1 if args.classic else 0, ra_sw_physics=5 if args.goddard else 1,
                    radt=5.0/60, solar_generation=args.solar_generation, ra_sw_eclipse=0, swrad_scat=args.swrad_scat)
    if args.noah:
        cfg=replace(cfg,sf_sfclay_physics=1,sf_surface_physics=2,seaice_prognostic=False)
    if args.stock_coordinates:
        cfg=replace(cfg,hybrid_opt=2,hypsometric_opt=2)
    if args.rayleigh or args.rayleigh_control:
        cfg=replace(cfg,damp_opt=3 if args.rayleigh else 0,zdamp=1800.,dampcoef=.2)
    if args.upper_wind: cfg=replace(cfg,upper_wind_limiter_form="noaa_wrf39")
    if args.damp_grid: cfg=replace(cfg,w_damping=int(args.w_damp),eta_levels=(1., .9, .8, .7, .6, .5999, .5, .4, .2, 0.))
    if args.diff6_slope and not args.diff6: ap.error('--diff6-slope requires --diff6')
    if args.diff6: cfg=replace(cfg,diff_6th_opt=1 if args.diff6_unlimited else 2,diff_6th_factor=0.12,diff_6th_slopeopt=int(args.diff6_slope))
    if args.diff6_fork: cfg=replace(cfg,diff_6th_form="noaa_wrf39",diff_6th_factor2=args.fork_factor2)
    if args.emdiv: cfg=replace(cfg,emdiv=0.01)
    if args.open_top: cfg=replace(cfg,top_lid=False)
    if args.horizontal5: cfg=replace(cfg,h_sca_adv_order=5)
    if args.vertical5: cfg=replace(cfg,v_sca_adv_order=5)
    if args.vmom5: cfg=replace(cfg,v_mom_adv_order=5)
    if args.damp_grid and cfg.nz != 9: ap.error('w-damping probe requires nine vertical levels')
    if args.precip_buckets: cfg=replace(cfg,prec_acc_dt=.05,prec_acc_dt1=.1)
    if args.climate_diagnostics: cfg=replace(cfg,output_diagnostics=1,diag_int=1,wind_int=1)
    coord=make_vertical_coord(cfg.nz,hybrid_opt=cfg.hybrid_opt,etac=cfg.etac,eta_levels=cfg.eta_levels)
    if args.igw_sounding:
        from woof.verify.cases.igw import sounding
    else:
        sounding=lambda z:np.full_like(np.asarray(z,dtype=np.float64),300.)
    y,x=np.indices((cfg.ny,cfg.nx))
    terrain=(args.terrain_height+args.terrain_relief*np.cos(2*np.pi*x/cfg.nx)*np.cos(2*np.pi*y/cfg.ny)
             if args.terrain_mode else np.zeros((cfg.ny,cfg.nx)))
    base=make_base_state(coord,sounding,
                         p_surf=cfg.p_surf,ztop=cfg.ztop,
                         terrain_z=terrain if args.base3d else None)
    def perturb(x,z):
        v=(.02*np.cos(np.pi*x/(cfg.nx*cfg.dx)))[None,None,:]
        if args.vertical_pattern:
            zv=np.asarray(z);zv=zv[:,None,None] if zv.ndim == 1 else zv
            v=v*(1+.3*np.sin(zv*2*np.pi/cfg.ztop))
        return np.broadcast_to(v,(cfg.nz,cfg.ny,cfg.nx))
    def setup_map(state,xp):
        if not (args.mapped or args.maps_only or args.rotation_only): return
        y,x=np.indices((cfg.ny,cfg.nx))
        field=(np.ones_like(x,dtype=np.float64) if args.rotation_only else
               1.0+0.02*np.cos(2*np.pi*x/cfg.nx)*np.cos(2*np.pi*y/cfg.ny))
        u=np.concatenate((field,field[:,:1]),axis=1)
        v=np.concatenate((field,field[:1,:]),axis=0)
        state.set_map_coriolis(msft=field,msfu=u,msfv=v,
            f=np.full_like(field,0 if args.maps_only else 1e-4),
            e=np.full_like(field,0 if args.maps_only else 3e-5),
            sina=np.full_like(field,0.6),cosa=np.full_like(field,0.8))
        state.u[...] = xp.asarray(np.broadcast_to(2.0+0.3*np.cos(2*np.pi*np.arange(cfg.nx+1)/cfg.nx),state.u.shape),dtype=np.float32)
        state.v[...] = xp.asarray(np.broadcast_to((1.0+0.2*np.sin(2*np.pi*np.arange(cfg.ny+1)/cfg.ny))[None,:,None],state.v.shape),dtype=np.float32)
        if args.vertical_pattern:
            vertical=1+.15*np.sin(2*np.pi*np.arange(cfg.nz)/cfg.nz)
            state.u[...] = xp.asarray(np.broadcast_to((2+.3*np.cos(2*np.pi*np.arange(cfg.nx+1)/cfg.nx))[None,None,:]*vertical[:,None,None],state.u.shape),dtype=np.float32)
            state.v[...] = xp.asarray(np.broadcast_to((1+.2*np.sin(2*np.pi*np.arange(cfg.ny+1)/cfg.ny))[None,:,None]*vertical[:,None,None],state.v.shape),dtype=np.float32)
            state.w[...] = xp.asarray(np.broadcast_to((.3*np.sin(np.pi*np.arange(cfg.nz+1)/cfg.nz))[:,None,None],state.w.shape),dtype=np.float32)
    def setup_igw_v(state,xp):
        if args.igw_v_seed:
            values=.5+.2*np.sin(2*np.pi*np.arange(cfg.nx)/cfg.nx)
            state.v[...] = xp.asarray(np.broadcast_to(values[None,None,:],state.v.shape).copy(),dtype=np.float32)
    def setup_open(state,xp):
        if not args.open_mode: return
        u=np.asarray(state.u if xp is np else state.u.to_numpy()).copy()
        v=np.asarray(state.v if xp is np else state.v.to_numpy()).copy()
        sy=np.where(np.arange(cfg.ny)%2, -1., 1.)
        sx=np.where(np.arange(cfg.nx)%2, -1., 1.)
        u[:,:,0]=sy[None,:]*2.; u[:,:,-1]=-sy[None,:]*2.
        v[:,0,:]=sx[None,:]*2.; v[:,-1,:]=-sx[None,:]*2.
        state.u[...] = xp.asarray(u,dtype=np.float32)
        state.v[...] = xp.asarray(v,dtype=np.float32)
    def setup_top(state,xp):
        if not args.open_top: return
        x=np.arange(cfg.nx)[None,:]
        w=np.broadcast_to((.3*np.sin(np.pi*np.arange(cfg.nz+1)/cfg.nz))[:,None,None],state.w.shape).copy()
        w[-1,...]=.12+.03*np.cos(2*np.pi*x/cfg.nx)
        state.w[...] = xp.asarray(w,dtype=np.float32)
    def setup_upper_wind(state,xp):
        if not (args.upper_wind or args.upper_wind_control): return
        vertical=1+.15*np.sin(2*np.pi*np.arange(cfg.nz)/cfg.nz)
        wind=np.broadcast_to((2+.3*np.cos(2*np.pi*np.arange(cfg.nx+1)/cfg.nx))[None,None,:]*vertical[:,None,None],state.u.shape).copy()
        wind[-3:]+=125.
        state.u[...] = xp.asarray(wind,dtype=np.float32)
    def setup_tke(state,xp):
        if args.selected_km != 2 or args.zero_tke: return
        z,y,x=np.indices((cfg.nz,cfg.ny,cfg.nx))
        state.tke[...] = xp.asarray(0.4+0.03*z+0.02*np.cos(2*np.pi*x/cfg.nx),dtype=np.float32)
    if args.nwp_diagnostics:
        from woof.core import uh_diag
        from tools.strict_uh_diag.compare import host_library
        uh_reference, uh_fold = host_library(tempdir.name)
        def host_uh_columns(u,v,w,ph,phb,msfu,msfv,ht,dn,dnw,fnm,fnp,cf1,cf2,cf3,rdx,rdy):
            arrays=[np.ascontiguousarray(a,dtype=np.float32) for a in (u,v,w,ph,phb,msfu,msfv,ht,dn,dnw,fnm,fnp)]
            nz,ny,nx=u.shape[0],w.shape[1],w.shape[2]
            uh=np.zeros((ny,nx),np.float32);use=np.zeros_like(uh);unused=np.zeros_like(uh)
            uh_reference(*map(address,arrays[:5]),int(phb.ndim==3),*map(address,arrays[5:]),
                         *map(float,(cf1,cf2,cf3,rdx,rdy)),address(uh),address(use),address(unused),nz,ny,nx)
            return uh,use.astype(bool)
        def host_uh_fold(uh,use,target):
            mask=np.ascontiguousarray(use,dtype=np.float32)
            uh_fold(address(uh),address(mask),address(target),*target.shape)
        uh_diag.cal_helicity_uh_columns_np=host_uh_columns
        uh_diag.apply_uh_smoother_max_np=host_uh_fold
        records['uh_diag']={'source_sha256':hashlib.sha256((ROOT/'woof/core/kernels/uh_diag.cu').read_bytes()).hexdigest(),
                            'reference':'selected CUDA unit compiled for host; boundary uses ordered byte copies'}
    def setup_uh(state,xp):
        if not args.nwp_diagnostics:return
        u=np.broadcast_to(-.04*np.arange(cfg.ny,dtype=np.float32)[None,:,None],state.u.shape).copy()
        v=np.broadcast_to(.07*np.arange(cfg.nx,dtype=np.float32)[None,None,:],state.v.shape).copy()
        if args.upper_wind or args.upper_wind_control: u[-3:]+=125.
        w=np.broadcast_to((2.*np.sin(np.pi*np.arange(cfg.nz+1)/cfg.nz))[:,None,None],state.w.shape).copy().astype(np.float32)
        state.u[...] = xp.asarray(u);state.v[...] = xp.asarray(v);state.w[...] = xp.asarray(w)
        for slot in uh_diag.TRACKER_WINDOW_SLOTS: state.scratch((cfg.ny,cfg.nx),slot)
    host=(igw.build(cfg,coord,base,array_module=np) if args.igw_2d else
          init_theta_perturbation(cfg,coord,base,perturb,array_module=np))
    setup_igw_v(host,np)
    setup_tke(host,np)
    if args.igw_sounding: adopt_wrf_theta_carrier(host)
    setup_map(host,np)
    setup_open(host,np)
    setup_upper_wind(host,np)
    setup_top(host,np)
    setup_uh(host,np)
    def setup_moist(state,xp):
        if not args.moist: return
        z,y,x=np.indices((cfg.nz,cfg.ny,cfg.nx))
        for name,base in (('qv',0.008),('qc',0.0003),('qr',0.0001)):
            values=base*(1.0+0.1*np.cos(2*np.pi*x/cfg.nx)+0.05*np.sin(2*np.pi*y/cfg.ny))
            if args.vertical_pattern: values*=1+.15*np.sin(2*np.pi*z/cfg.nz)
            getattr(state,name)[...] = xp.asarray(values,dtype=np.float32)
        dycore.update_diagnostics(state,cfg.hypsometric_opt)
    setup_moist(host,np)
    clocks=[]
    if args.specified:
        from woof.ingest.lateral_bc import (domain_boundary_snapshot,build_lateral_boundaries,
            attach_lateral_boundaries,bind_lateral_boundary_clock)
        from woof.core.clock import DomainTicks,DomainClock
        first=dict(domain_boundary_snapshot(host));first['w']=np.asarray(host.w,dtype=np.float64).copy()
        second={name: np.asarray(value,dtype=np.float64)+0.01 for name,value in first.items()}
        snapshots=(first,second)
        times=(0.,20.)
        if args.boundary_crossover:
            snapshots=(first,second)+tuple(
                {name: np.asarray(value,dtype=np.float64)-0.03-0.001*(index-2)
                 for name,value in first.items()} for index in range(2,args.steps+1))
            times=tuple(float(index) for index in range(args.steps+1))
        boundaries=build_lateral_boundaries(snapshots,times,
            spec_bdy_width=cfg.spec_bdy_width,spec_zone=cfg.spec_zone,relax_zone=cfg.relax_zone)
        def attach(state):
            attach_lateral_boundaries(state,boundaries)
            spec=DomainTicks(1,0,1,1,np.float32(cfg.dt),1,None,None,None,None,None,None,None,
                             lbc_interval_ticks=1 if args.boundary_crossover else 20)
            clock=DomainClock(spec,1,2 if args.boundary_crossover else 20)
            bind_lateral_boundary_clock(state,clock)
            clocks.append(clock)
        attach(host)
    elif args.tree_checkpoint:
        from woof.core.clock import DomainTicks,DomainClock
        def periodic_clock():
            step_ticks=int(cfg.dt) if args.igw_2d else 1
            spec=DomainTicks(1,0,1,step_ticks,np.float32(cfg.dt),step_ticks,None,None,None,None,None,None,None,
                             lbc_interval_ticks=20*step_ticks)
            return DomainClock(spec,1,20*step_ticks)
        clocks.extend((periodic_clock(),periodic_clock()))
    fields=('u','v','w','thp','php','mup','p','al','alt','p_perturbation')
    if args.moist: fields += ('qv','qc','qr')
    if args.selected_km == 2: fields += ('tke',)
    inputs=('thb','phb','pb','alb','dphb_resid','mub2d','ht',
            'msft','msfu','msfv','f','e','sina','cosa',
            'c1h','c2h','c1f','c2f','c3h','c4h','c3f','c4f','dc3f','dc4f',
            'rdnw','rdn','fnm','fnp')
    if args.dudhia:
        from woof.core import physics, dudhia, tendency_coupling
        if args.private_terrain:
            strict_dudhia=physics._metal_strict_physics_profile
            def private_dudhia(candidate,**kwargs):
                return strict_dudhia(replace(candidate,terrain_opt=0),**kwargs)
            physics._metal_strict_physics_profile=private_dudhia
            dycore._metal_strict_physics_profile=private_dudhia
        if args.solar_generation == 'wrf_480':
            from woof.core import solar
            from tools.metal_solar_word_probe import build_reference
            from woof.metal.array import MetalArray
            solar_library,solar_reference=build_reference(Path(tempdir.name))
            original_solar=solar.solar_geometry_device
            def host_solar(julian,xtime,gmt,latitude,longitude,*,generation='wrf_480',declination=None):
                if isinstance(latitude,MetalArray) or isinstance(longitude,MetalArray):
                    return original_solar(julian,xtime,gmt,latitude,longitude,
                        generation=generation,declination=declination)
                from woof.core.rrtmg_legacy import _DEGRAD,_DPD
                lat=np.ascontiguousarray(latitude,dtype=np.float32)
                lon=np.ascontiguousarray(longitude,dtype=np.float32)
                pair=np.empty(2,np.float32);cosine=np.empty_like(lat);hrang=np.empty_like(lat)
                solar_reference(lat.size,lat.ctypes.data,lon.ctypes.data,
                    julian,xtime,gmt,_DEGRAD,_DPD,int(generation=='wrf_480'),
                    int(declination is not None),0. if declination is None else declination,
                    pair.ctypes.data,cosine.ctypes.data,hrang.ctypes.data)
                return pair,cosine,hrang
            solar.solar_geometry_device=host_solar
            original_geometry=dudhia.wrf_solar_geometry
            def host_geometry(valid_time,latitude,longitude,**kwargs):
                if isinstance(latitude,MetalArray) or isinstance(longitude,MetalArray):
                    return original_geometry(valid_time,latitude,longitude,**kwargs)
                pair,cosine,_=solar.geometry_at(valid_time,latitude,longitude,**kwargs)
                # NumPy scalar readback replaces only CuPy scalar.get().
                return cosine,float(pair[1])
            dudhia.wrf_solar_geometry=host_geometry
            records['solar_geometry']={'source_sha256':hashlib.sha256(
                K.module_source('rrtmg_legacy_solar').encode()).hexdigest(),
                'reference':'selected pinned source, shared call calendar; host scalar readback adapter'}
        tendency_coupling._MODULE=host_backend.HostModule(
            'strict_physics_coupling', prefix+tendency_coupling._SOURCE)
        records['physics_coupling']={'source_sha256':hashlib.sha256(
            tendency_coupling._SOURCE.encode()).hexdigest(),
            'effective_cuda_options':list(__import__('woof.wrf_exact',fromlist=['effective_options']).effective_options(()))}
        records['dudhia_sw']={'source_sha256':hashlib.sha256(
            K.module_source('dudhia_sw').encode()).hexdigest(),
            'effective_cuda_options':list(__import__('woof.wrf_exact',fromlist=['effective_options']).effective_options(
                K.module_options('dudhia_sw')))}
        if args.terrain_mode:
            atmosphere_original=physics._prepare_atmosphere
            atmosphere_snapshots={}
            def capture_atmosphere(state):
                result=atmosphere_original(state)
                atmosphere_snapshots.setdefault(id(state),[]).append({name:
                    value.to_numpy() if hasattr(value,'to_numpy') else np.array(value,copy=True)
                    for name,value in result.items()})
                return result
            physics._prepare_atmosphere=capture_atmosphere
        # Host numpy arrays emulate CUDA device buffers, so choose the actual
        # CUDA column path rather than the NumPy approximate mirror.
        dudhia._array_namespace=lambda *arrays: dycore.cp
        geometry=(np.broadcast_to(np.linspace(-60.,60.,cfg.ny,dtype=np.float32)[:,None],(cfg.ny,cfg.nx)).copy(),
                  np.broadcast_to(np.linspace(-150.,150.,cfg.nx,dtype=np.float32)[None,:],(cfg.ny,cfg.nx)).copy())
        if args.noah:
            from woof.core import noah_forcing, noah_sfcdiags
            noah_forcing._MODULE=host_backend.HostModule('strict_noah_forcing_chain', noah_forcing._SOURCE)
            noah_sfcdiags._KERNEL=host_backend.HostModule('strict_noah_sfcdiags_chain', noah_sfcdiags._source()).get_function('noah_sfcdiags')
        if args.goddard:
            from woof.core.radiation_composition import ColumnTableRadiation
            radiation_snapshots={}
            from woof.metal import goddard_family
            original_submit=goddard_family._submit
            def capture_submit(name,resources,parameters,count):
                result=original_submit(name,resources,parameters,count)
                if name not in goddard_metal_stages and name not in ('goddard_pack_grid','goddard_scatter_grid'):
                    goddard_metal_stages[name]={key:value.to_numpy() for key,value in resources.items()
                        if key not in ('lut','workspace')}
                    if args.goddard_capture_band and name=='goddard_bands_columns':
                        pairs={}
                        for label,values in (('host',goddard_host_stages[name]),('metal',goddard_metal_stages[name])):
                            pairs.update({label+'_'+key:value for key,value in values.items()})
                        pairs['levels']=np.array(parameters['np'],np.int32)
                        pairs['columns']=np.array(parameters['columns'],np.int32)
                        np.savez(args.goddard_capture_band,**pairs)
                        print('captured actual source and Metal band calls:',args.goddard_capture_band)
                        raise SystemExit(0)
                return result
            goddard_family._submit=capture_submit
            original_goddard=ColumnTableRadiation.__call__
            def capture_goddard(adapter,*,atmosphere,fields,state,cfg):
                def host_copy(value):
                    return value.to_numpy() if hasattr(value,'to_numpy') else np.array(value,copy=True)
                snapshot={name:host_copy(atmosphere[name]) for name in
                    ('temperature','pressure','p_interface','z_interface','dz','rho','exner')}
                snapshot.update({'state_'+name:host_copy(getattr(state,name)) for name in ('qv','qc','qr','qi','qs','qg','qh') if getattr(state,name,None) is not None})
                snapshot.update({'field_'+name:host_copy(fields[name]) for name in ('tsk','emiss','albedo')})
                output=original_goddard(adapter,atmosphere=atmosphere,fields=fields,state=state,cfg=cfg)
                snapshot.update({name:host_copy(getattr(output,name)) for name in
                    ('coszen','cldfra','swdown','glw','rthratenlw','rthratensw')})
                radiation_snapshots.setdefault(id(state),[]).append(snapshot)
                return output
            ColumnTableRadiation.__call__=capture_goddard
        def attach_physics(state):
            if args.goddard:
                from woof.core.radiation_composition import ColumnTableRadiation
                radiation=ColumnTableRadiation(datetime(2026,1,1,12,tzinfo=timezone.utc),*geometry,column_chunk=32)
            elif args.classic:
                from woof.core.rrtm_lw import RRTMDudhiaRadiation
                radiation=RRTMDudhiaRadiation(datetime(2026,1,1,12,tzinfo=timezone.utc),*geometry,
                                             p_top=float(state.p_top),icloud=1,swrad_scat=cfg.swrad_scat,
                                             column_chunk=32)
            else:
                radiation=dudhia.DudhiaShortwaveRadiation(
                    datetime(2026,1,1,12,tzinfo=timezone.utc),*geometry, swrad_scat=cfg.swrad_scat)
            saved=physics.cp
            if getattr(state,'_metal_setup_state',False): physics.cp=None
            try: physics.initialize_physics(state,cfg,glw=300.,radiation=radiation)
            finally: physics.cp=saved
            if args.noah and getattr(state,'_host_setup_state',False):
                state.physics._noah_seaice_checked=True  # NumPy host fatal carrier has no CuPy .get().
        attach_physics(host)
    budget_host=[]
    if args.tke_budget:
        from woof.core import tke_budget
        accumulate_original=tke_budget.accumulate
        # Execute the actual CUDA per-thread body and reduction tree serially.
        source=K.module_source('tke_budget')
        body=source.split('    double sum = 0.0;',1)[1].split('    sdata[tid] = sum;',1)[0]
        unit=("#include <stddef.h>\nextern \"C\" void reference(const float* terms,const float* tke0,"
              "const float* tke,const float* raw,const float* mu0,const float* mu,"
              "const float* c1,const float* c2,float rdt,double* acc,int nfield,int nz,int ny,int nx){"
              "const size_t plane=(size_t)ny*nx;const int nthreads=256;"
              "for(int t=0;t<nfield+2;++t)for(int k=0;k<nz;++k){double sdata[256];"
              "for(int tid=0;tid<256;++tid){double sum=0.0;"+body+"sdata[tid]=sum;}"
              "for(int s=128;s>0;s>>=1)for(int tid=0;tid<s;++tid)sdata[tid]+=sdata[tid+s];"
              "acc[(size_t)t*nz+k]+=sdata[0];}}\n")
        cpp=Path(tempdir.name)/'budget.cpp'; lib=cpp.with_suffix('.dylib');cpp.write_text(unit)
        flags=['-O2','-std=c++17','-ffp-contract=off','-fno-unsafe-math-optimizations','-dynamiclib']
        subprocess.run(['clang++',*flags,str(cpp),'-o',str(lib)],check=True,capture_output=True,text=True)
        budget_dll=ctypes.CDLL(str(lib)); budget_reference=budget_dll.reference
        budget_reference.argtypes=[ptr]*8+[ctypes.c_float,ctypes.POINTER(ctypes.c_double)]+[ctypes.c_int]*4
        budget_acc=np.zeros((len(tke_budget.TERMS),cfg.nz),dtype=np.float64)
        budget_resume_acc=np.zeros_like(budget_acc)
        records['tke_budget']={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
            'host_unit_sha256':hashlib.sha256(unit.encode()).hexdigest(),'host_flags':flags,
            'effective_cuda_options':list(__import__('woof.wrf_exact',fromlist=['effective_options']).effective_options(K.module_options('tke_budget')))}
        def capture_budget(state,candidate):
            if getattr(state,'_host_setup_state',False):
                terms=tke_budget.term_buffer(state,candidate).copy()
                raw=tke_budget.raw_carrier(state,candidate).copy()
                mu0=state.mub2d+state.mup0; mu=state.mub2d+state.mup
                budget_reference(*map(address,(terms,state.tke0,state.tke,raw,mu0,mu,state.c1h,state.c2h)),
                    float(np.float32(1.0/candidate.dt)),budget_acc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                    len(tke_budget.TERM_FIELDS),cfg.nz,cfg.ny,cfg.nx)
                if args.checkpoint and len(budget_host) >= args.checkpoint_after:
                    budget_reference(*map(address,(terms,state.tke0,state.tke,raw,mu0,mu,state.c1h,state.c2h)),
                        float(np.float32(1.0/candidate.dt)),budget_resume_acc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        len(tke_budget.TERM_FIELDS),cfg.nz,cfg.ny,cfg.nx)
                budget_host.append((terms,raw,budget_acc.copy(),budget_resume_acc.copy()))
            else:
                accumulate_original(state,candidate)
        tke_budget.accumulate=capture_budget
    result={'steps':args.steps,'compiler_flags':list(host_backend.CXX_FLAGS),'cases':{},'sources':records}
    precip_slots=()
    if args.precip_buckets:
        from woof.core.output_diagnostics import PRECIP_HISTORY,boundary_due
        precip_slots=tuple(PRECIP_HISTORY)
        result['precipitation_buckets']={'periods_seconds':[3,6],
            'reset_steps':{str(p):[n+1 for n in range(args.steps) if boundary_due(n*cfg.dt,p)] for p in (3,6)},
            'producers':'zero in this physics-off/Dudhia chain; nonzero inputs have separate selected-source caller proof'}
    climate_slots=()
    if args.climate_diagnostics:
        from woof.core.climate_diagnostics import PERSISTENT
        climate_slots=PERSISTENT
        result['climate_statistics']={'interval_minutes':1,
            'reset_steps':[n+1 for n in range(args.steps) if n % 60 == 0],
            'finish_steps':[n+1 for n in range(args.steps) if (n+1) % 60 == 0],
            'producers':'actual driver fields with shared missing-producer zero fallback; no injected producers'}
    diagnostic_slots=precip_slots+climate_slots
    if args.terrain_mode:
        variation={name:bool(np.any(getattr(host,name) != (getattr(host,name).flat[0]
                   if getattr(host,name).ndim == 2 else getattr(host,name)[:, :1, :1])))
                   for name in ('ht','mub2d','phb','pb','alb','dphb_resid')}
        assert all(variation.values()), 'terrain and base fields must vary'
        if args.mapped and not np.any(host.msft != np.float32(1)):
            raise AssertionError('terrain map factors were uniform')
        result['terrain_field_variation']=variation
        result['terrain_range']=[float(np.min(host.ht)),float(np.max(host.ht))]
        if args.private_terrain:
            result['private_admission_omissions']='terrain and mixing admission checks' + (' plus Dudhia physics terrain gate' if args.dudhia else '')
    if args.mixing:
        result['host_add_adapter']='single-row glue_add per pair; same CUDA glue_add_chunk as glue_add_arrays'
    if args.surface_flux is not None:
        surface_effects=[]
        vertical_original=dycore.launch_wrf_smag2d_vertical
        def capture_surface(state, candidate, km, **kwargs):
            if not getattr(state,'_host_setup_state',False) or any(kwargs.get(name) is None for name in ('ru','rv','rth')):
                return vertical_original(state,candidate,km,**kwargs)
            before={name:np.array(kwargs[name],copy=True) for name in ('ru','rv','rth')}
            vertical_original(state,candidate,km,**kwargs)
            surface_effects.append({name:bool(np.any(kwargs[name][0] != before[name][0])) for name in before})
        dycore.launch_wrf_smag2d_vertical=capture_surface
    with MetalContext() as context:
        context.batch_kernel_calls=False
        submit=context._submit_kernel_batch
        def sync(*a,**kw):
            result=submit(*a,**kw);result.wait();return result
        context._submit_kernel_batch=sync
        xp=MetalArrayModule(context)
        if args.tke_budget:
            # One cell discriminates fused from separate multiply/subtract;
            # another exercises the FP32 subnormal load and 256-lane stride.
            plane=257
            synthetic={name:np.zeros(shape,dtype=np.float32) for name,shape in (
                ('terms',(len(tke_budget.TERM_FIELDS),1,1,plane)),
                ('tke0',(1,1,plane)),('tke',(1,1,plane)),('raw',(1,1,plane)),
                ('mu0',(1,plane)),('mu',(1,plane)),('c1',(1,)),('c2',(1,)))}
            synthetic['c1'][0]=np.float32(0.3717438280582428)
            synthetic['c2'][0]=np.float32(41.64615249633789)
            synthetic['mu'][0,0]=np.float32(74967.6328125)
            synthetic['mu0'][0,0]=np.float32(96908.1796875)
            synthetic['tke'][0,0,0]=np.float32(0.6626277565956116)
            synthetic['tke0'][0,0,0]=np.float32(0.540581226348877)
            synthetic['terms'].view(np.uint32)[0,0,0,256]=1
            c1,c2=map(float,(synthetic['c1'][0],synthetic['c2'][0]))
            ch=c1*float(synthetic['mu'][0,0])+c2
            ch0=c1*float(synthetic['mu0'][0,0])+c2
            t=float(synthetic['tke'][0,0,0]);t0=float(synthetic['tke0'][0,0,0])
            fma=ctypes.CDLL(None).fma;fma.argtypes=[ctypes.c_double]*3;fma.restype=ctypes.c_double
            assert ch*t-ch0*t0 != fma(ch,t,-ch0*t0), 'FMA control was not discriminating'
            expected=np.zeros((len(tke_budget.TERMS),1),np.float64)
            budget_reference(*map(address,(synthetic[n] for n in ('terms','tke0','tke','raw','mu0','mu','c1','c2'))),
                1.0,expected.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                len(tke_budget.TERM_FIELDS),1,1,plane)
            arrays={name:xp.asarray(value) for name,value in synthetic.items()}
            arrays['acc']=xp.asarray(np.zeros_like(expected));arrays['steps']=xp.asarray(np.zeros(1,np.float64))
            context.submit_kernel('wrf_exact_tke_budget_accumulate',arrays,
                dict(rdt=np.float32(1),nfield=len(tke_budget.TERM_FIELDS),nz=1,ny=1,nx=plane),
                exact_policy=True,grid_size=len(tke_budget.TERMS)*256,threads_per_group=256)
            result['cases']['budget_fma_subnormal_stride']=compare_float64(arrays['acc'].to_numpy(),expected)
            assert expected[0,0] != 0 and arrays['steps'].to_numpy()[0] == 1
        metal=(igw.build(cfg,coord,base,array_module=xp) if args.igw_2d else
               init_theta_perturbation(cfg,coord,base,perturb,array_module=xp))
        setup_igw_v(metal,xp)
        if args.igw_sounding: adopt_wrf_theta_carrier(metal)
        setup_map(metal,xp)
        setup_open(metal,xp)
        setup_upper_wind(metal,xp)
        setup_top(metal,xp)
        setup_uh(metal,xp)
        setup_tke(metal,xp)
        setup_moist(metal,xp)
        if args.specified: attach(metal)
        for field in fields + inputs:
            result['cases']['initial_'+field]=compare_words(getattr(metal,field).to_numpy(),getattr(host,field))
        if any(x['different_words'] for x in result['cases'].values()):
            raise AssertionError('host and Metal initial state differ')
        supported=dycore._strict_metal_dry_periodic_supported
        if args.private_open or args.private_mixing or args.private_specified_moist or args.private_terrain or args.private_h5 or args.private_vscalar5:
            def private_scope(state,candidate,**kwargs):
                if args.private_open:
                    if not (candidate.open_x and candidate.open_y): return False
                    return supported(state,replace(candidate,open_x=False,open_y=False),**kwargs)
                if args.private_vscalar5 and candidate.v_sca_adv_order == 5:
                    candidate=replace(candidate,v_sca_adv_order=3)
                if args.private_h5:
                    if candidate.h_sca_adv_order == 5: candidate=replace(candidate,h_sca_adv_order=2)
                    if not (args.private_mixing or args.private_specified_moist or args.private_terrain):
                        return supported(state,candidate,**kwargs)
                if args.private_vscalar5 and not (args.private_h5 or args.private_mixing or args.private_specified_moist or args.private_terrain):
                    return supported(state,candidate,**kwargs)
                if args.private_terrain:
                    shadow=copy.copy(state)
                    if args.private_km == 2: shadow.tke=None
                    shadow.ht=xp.asarray(np.zeros_like(host.ht))
                    for name in ('mub2d','phb','pb','alb','dphb_resid'):
                        values=np.array(getattr(host,name),copy=True)
                        values[...] = values.flat[0] if values.ndim == 2 else values[:, :1, :1]
                        setattr(shadow,name,xp.asarray(values))
                    # The shadow only bypasses admission; the real configuration
                    # still runs the coordinate operator and terrain fields.
                    return supported(shadow,replace(candidate,km_opt=0,
                        diff_opt=2 if args.private_coordinate4 and candidate.diff_opt == 1 else candidate.diff_opt),**kwargs)
                if (args.private_coordinate4 or args.private_coordinate2) and candidate.diff_opt == 1:
                    candidate=replace(candidate,diff_opt=2)
                if args.specified and args.moist and not args.dudhia:
                    from woof.core.dycore import SPECIES, extra_moist_species
                    if (not candidate.moist or candidate.mp_physics != 0 or not candidate.moist_cq
                            or candidate.moist_adv_opt not in (0,1)
                            or any(getattr(state,name,None) is None for name in SPECIES)
                            or extra_moist_species(state)):
                        return False
                    shadow=copy.copy(state);shadow.qv=None
                    if args.private_km == 2: shadow.tke=None
                    return supported(shadow,replace(candidate,km_opt=0,moist=False),**kwargs)
                shadow=copy.copy(state)
                if args.private_km == 2: shadow.tke=None
                return supported(shadow,replace(candidate,km_opt=0),**kwargs)
            dycore._strict_metal_dry_periodic_supported=private_scope
            if args.private_vscalar5: result['private_admission_omissions']='vertical scalar order gate only; actual fifth-order caller executes'
            elif args.private_h5: result['private_admission_omissions']='horizontal order gate only; actual fifth-order caller executes'
            elif not args.private_terrain: result['private_admission_omissions']=('coordinate mixing admission only; actual caller executes'
                if args.private_coordinate4 else 'specified moist gate; actual moisture executes'
                if args.private_specified_moist else 'strict km_opt=4 and specified moist gates; actual mixing and moisture execute'
                if args.specified and args.moist and not args.dudhia else
                'strict mixing gate only; actual mixing executes')
        if args.dudhia:
            attach_physics(metal)
            result['solar']=('selected pinned-source wrf_480 device geometry with shared call calendar'
                if args.solar_generation=='wrf_480' else 'shared upstream host wrf_461 geometry, not independently verified')
        def admitted(candidate=cfg, state=metal, **kw):
            scope=private_scope if args.private_open or args.private_mixing or args.private_specified_moist or args.private_terrain or args.private_h5 or args.private_vscalar5 else supported
            return scope(state,candidate,acoustic=kw.get('acoustic',True),
                mass_flux_observer=kw.get('observer'),mass_flux_accumulator=kw.get('accumulator'))
        negatives={}
        for name,value in ((('moist',False) if args.moist else ('moist',True)),('open_x',False if args.open_mode else True),('open_y',False if args.open_mode else True),
                ('nested',True),('terrain_opt',2),
                ('zadvect_implicit',True),('diff_6th_opt',3),('emdiv',-0.1),
                ('w_damping',2),('damp_opt',1),('diff_opt',3),('h_sca_adv_order',3),
                ('v_sca_adv_order',7),('v_mom_adv_order',7),
                ('nx',6),('ny',6),('nz',4)):
            altered=replace(cfg,**{name:value})
            negatives[name]=not admitted(candidate=altered)
        if args.diff6_unlimited:
            shadow=copy.copy(metal)
            for name in ('qv','qc','qr'):
                setattr(shadow,name,xp.asarray(np.zeros((cfg.nz,cfg.ny,cfg.nx),np.float32)))
            negatives['unlimited_moist']=not admitted(state=shadow,candidate=replace(cfg,moist=True))
        if args.diff6:
            for name,value in (('diff_6th_form','invalid'),('diff_6th_factor',float('nan')),
                               ('diff_6th_slopeopt',2),('diff_6th_thresh',0.0)):
                negatives[name]=not admitted(candidate=replace(cfg,**{name:value}))
        if args.diff6_fork:
            checks=(('diff_6th_factor2',-1.),('diff_6th_factor2',float('nan')))
            for name,value in checks:
                negatives[f'fork_{name}_{value}']=not admitted(candidate=replace(cfg,**{name:value}))
        if args.w_damp:
            for value in (0.0, 1.01, float('nan')):
                negatives[f'w_crit_cfl_{value}']=not admitted(candidate=replace(cfg,w_crit_cfl=value))
        if args.rayleigh:
            for name,value in (('damp_opt',2),('dampcoef',float('nan')),('dampcoef',-1.),
                               ('zdamp',0.),('zdamp',cfg.ztop),('upper_wind_limiter_form','invalid')):
                negatives[f'rayleigh_{name}_{value}']=not admitted(candidate=replace(cfg,**{name:value}))
        negatives['emdiv_nonfinite']=not admitted(candidate=replace(cfg,emdiv=float('nan')))
        if args.igw_2d:
            for name,value in (('isfflx',0),('top_lid',False),('hybrid_opt',2),
                               ('hypsometric_opt',2),('khdif',1.),('diff_6th_opt',1),
                               ('output_diagnostics',1)):
                negatives['one_row_'+name]=not admitted(candidate=replace(cfg,**{name:value}))
        if args.precip_buckets:
            negatives['precip_nonfinite']=not admitted(candidate=replace(cfg,prec_acc_dt=float('nan')))
            negatives['precip_negative']=not admitted(candidate=replace(cfg,prec_acc_dt1=-1.))
        if args.climate_diagnostics:
            for name,value in (('diag_int',0),('wind_int',2**31),('dt',61.)):
                negatives['climate_'+name]=not admitted(candidate=replace(cfg,**{name:value}))
        if args.open_mode:
            for name,value in (('specified',True),('km_opt',4),('diff_6th_opt',2),
                               ('w_damping',1),('damp_opt',3),('output_diagnostics',1),
                               ('prec_acc_dt',1.0),('prec_acc_dt1',1.0)):
                negatives['open_'+name]=not admitted(candidate=replace(cfg,**{name:value}))
        negatives['acoustic_off']=not admitted(acoustic=False)
        negatives['observer']=not admitted(observer=object())
        negatives['accumulator']=not admitted(accumulator=object())
        if args.terrain and not args.private_terrain:
            for name,value in (('km_opt',1),('bl_pbl_physics',1)):
                negatives['terrain_'+name]=not admitted(candidate=replace(cfg,**{name:value}))
            if not args.dudhia:
                negatives['terrain_radiation']=not admitted(candidate=replace(cfg,ra_lw_physics=0,ra_sw_physics=1))
            if args.mixing and not args.coordinate4:
                negatives['terrain_coordinate_mixing']=not admitted(candidate=replace(cfg,diff_opt=1))
            if args.coordinate4:
                for name,value in (('km_opt',2),('specified',False),('hybrid_opt',0),
                                   ('hypsometric_opt',1),('ra_lw_physics',5),('ra_sw_physics',5),
                                   ('diff_6th_opt',2),('h_sca_adv_order',5),
                                   ('v_sca_adv_order',5),('v_mom_adv_order',5)):
                    negatives['terrain_coordinate_'+name]=not admitted(candidate=replace(cfg,**{name:value}))
            if args.dudhia:
                negatives['terrain_dudhia_km1']=not dycore._metal_strict_physics_profile(
                    replace(cfg,km_opt=1),radiation=metal.physics.radiation_callable)
            with_driver=copy.copy(metal);with_driver.physics=object()
            negatives['terrain_driver']=not admitted(state=with_driver)
            for name in ('ht','mub2d','phb','pb','alb','dphb_resid'):
                invalid=copy.copy(metal)
                values=np.array(getattr(host,name),copy=True);values.flat[-1]=np.nan
                setattr(invalid,name,xp.asarray(values))
                negatives['terrain_nonfinite_'+name]=not admitted(state=invalid)
        if args.mixing and not args.private_mixing and not args.private_terrain:
            terrain_check=() if args.terrain and args.coordinate4 else (
                ('terrain_opt',0) if args.terrain else ('terrain_opt',1),)
            for name,value in (terrain_check + (
                               ('bl_pbl_physics',1),('sf_surface_physics',0 if args.noah else 2),
                               ('isfflx',3),('tke_heat_flux',float('nan')),
                               ('tke_drag_coefficient',-0.001))):
                altered=replace(cfg,**{name:value})
                if args.surface_only and name == 'isfflx': altered=replace(altered,khdif=1.)
                negatives['km4_'+name+str(value)]=not admitted(candidate=altered)
            if args.km == 2:
                missing=copy.copy(metal);missing.tke=None
                negatives['tke_missing']=not admitted(state=missing)
            else:
                extra=copy.copy(metal);extra.tke=xp.zeros(host.p.shape,dtype=np.float32)
                negatives['unexpected_tke']=not admitted(state=extra)
            if args.km1:
                for name,value in (('khdif',-1.),('kvdif',float('nan'))):
                    negatives[name+'_invalid']=not admitted(candidate=replace(cfg,**{name:value}))
            if args.coordinate4 or args.coordinate2:
                for name,value in (('isfflx',0),('isfflx',2),('khdif',1.),('kvdif',1.)):
                    negatives['coordinate_'+name+str(value)]=not admitted(candidate=replace(cfg,**{name:value}))
            if not args.specified:
                negatives['km4_specified_without_forcing']=not admitted(candidate=replace(cfg,specified=True))
            if not args.dudhia:
                negatives['km4_radiation']=not admitted(candidate=replace(cfg,ra_lw_physics=0,ra_sw_physics=1))
            with_driver=copy.copy(metal);with_driver.physics=object()
            negatives['km4_driver']=not admitted(state=with_driver)
        for name,value in ((('thb',301.0),) if args.terrain_mode else (('ht',1.0),('thb',301.0))):
            altered=copy.copy(metal)
            setattr(altered,name,xp.asarray(np.full(getattr(host,name).shape,value,dtype=np.float32)))
            negatives[name+'_values']=not admitted(state=altered)
        for name in (() if args.terrain_mode else (('phb','pb','alb','dphb_resid','mub2d') if args.base3d else ('mub2d',))):
            altered=copy.copy(metal)
            values=np.array(getattr(host,name),copy=True)
            values.flat[-1] += np.float32(1.0)
            setattr(altered,name,xp.asarray(values))
            negatives[name+'_nonuniform']=not admitted(state=altered)
        if args.moist:
            for name,value in (('mp_physics',1),('moist_cq',False),('moist_adv_opt',2)):
                negatives['moist_'+name]=not admitted(candidate=replace(cfg,**{name:value}))
            if not args.dudhia and not args.specified:
                negatives['moist_specified']=not admitted(candidate=replace(cfg,specified=True))
        if args.noah:
            for name,value in (('seaice_prognostic',True),('sf_surface_mosaic',1),('sf_urban_physics',1),('usemonalb',True),('rdlai2d',True)):
                negatives['noah_'+name]=not admitted(candidate=replace(cfg,**{name:value}))
        if args.dudhia:
            missing=copy.copy(metal);missing.physics=None
            negatives['radiation_missing_driver']=not admitted(state=missing)
            for name,value in (('icloud',0),('solar_generation','unknown'),
                               ('ra_sw_eclipse',1),('alb_sol',1),('sf_lake_physics',1),
                               ('sf_ocean_physics',1),('cu_physics',1),
                               ('sf_surface_physics',0 if args.noah else 2),('bl_pbl_physics',1),
                               ('sf_sfclay_physics',91),('slope_rad',1),
                               ('ra_lw_physics',0 if args.classic else 1),('ra_sw_physics',0)):
                negatives['radiation_'+name]=not admitted(candidate=replace(cfg,**{name:value}))
        if args.specified:
            from woof.ingest.lateral_bc import RationalTimeLaw
            empty=copy.copy(metal);empty.lateral_boundaries=None
            negatives['specified_missing_forcing']=not admitted(state=empty)
            interval=boundaries.intervals[0]; boundary_fields=dict(interval.fields)
            field=boundary_fields['u']; side=field.west
            law=RationalTimeLaw(np.zeros_like(side.value),np.zeros_like(side.value))
            boundary_fields['u']=replace(field,west=replace(side,time_law=law))
            nonlinear=copy.copy(metal)
            nonlinear.lateral_boundaries=replace(boundaries,intervals=(replace(interval,fields=boundary_fields),))
            negatives['specified_nonlinear']=not admitted(state=nonlinear)
            for name,value in (('relax_w',True),('spec_exp',0.1),
                               ('relax_timescale_s',10.0),('spec_bdy_width',6),
                               ('spec_zone',2),('relax_zone',3)):
                negatives['specified_'+name]=not admitted(candidate=replace(cfg,**{name:value}))
            unbound=copy.copy(metal)
            unbound._lateral_boundary_device=copy.copy(metal._lateral_boundary_device)
            unbound._lateral_boundary_device.clock=None
            negatives['specified_unbound_clock']=not admitted(state=unbound)
            wrong_clock=copy.copy(metal)
            wrong_clock._lateral_boundary_device=copy.copy(metal._lateral_boundary_device)
            wrong_clock._lateral_boundary_device.clock=copy.copy(metal._lateral_boundary_device.clock)
            wrong_clock._lateral_boundary_device.clock.dt_fp32=np.float32(2.0)
            negatives['specified_clock_dt_mismatch']=not admitted(state=wrong_clock)
        result['scope']={'positive':bool(admitted()),'negative':negatives}
        if not result['scope']['positive'] or not all(negatives.values()):
            raise AssertionError(f"strict scope admission check failed: {result['scope']}")
        from woof.state_serialization_contract import WRF_MUTS_SLOT
        served=[]
        stage_snapshots={}
        if args.igw_2d or args.open_mode or args.dudhia or args.mixing or args.specified_moist or args.terrain_mode:
            capture_original=dycore._capture_wrf_muts
            def capture_stage(state, candidate):
                capture_original(state,candidate)
                def host_value(value):
                    return value.to_numpy() if getattr(state,'_metal_setup_state',False) else np.array(value,copy=True)
                record={name:host_value(getattr(state,name)) for name in ('u','v','w','thp','php','mup')}
                if args.moist:
                    record.update({name:host_value(getattr(state,name)) for name in ('qv','qc','qr')})
                record['muts']=host_value(state.existing_scratch(WRF_MUTS_SLOT))
                if args.emdiv:
                    mudf=host_value(state.existing_scratch('acoustic_mudf'))
                    if not np.any(np.diff(mudf,axis=1) != 0): raise AssertionError('external-mode damping is inactive')
                    record['mudf']=mudf
                if args.mixing:
                    for name in ('u0','v0','w0','thp0','qv0','qc0','qr0','tke','tke0'):
                        value=getattr(state,name,None)
                        if value is not None: record[name]=host_value(value)
                    for name in ('smag_km','smag_kh','smag_kmv','smag_khv','smag_ru','smag_rv','smag_rw',
                                 'smag_rth','smag_rqv','smag_rqc','smag_rqr','smag_rtke','diff1_theta_initial'):
                        value=state.existing_scratch(name)
                        if value is not None: record[name]=host_value(value)
                stage_snapshots.setdefault(id(state),[]).append(record)
            dycore._capture_wrf_muts=capture_stage
        open_records={}
        if args.open_mode:
            radiative_original=dycore.apply_open_radiative_bc
            gradient_original=dycore.apply_open_zero_gradient
            def capture_open(state,candidate,original,names,label):
                def read(name):
                    value=getattr(state,name)
                    return value.to_numpy() if getattr(state,'_metal_setup_state',False) else np.array(value,copy=True)
                before={name:read(name) for name in names}
                original(state,candidate)
                after={name:read(name) for name in names}
                changed=sum(np.count_nonzero(before[name].view(np.uint32)!=after[name].view(np.uint32)) for name in names)
                open_records.setdefault((id(state),label),[]).append((after,int(changed)))
            dycore.apply_open_radiative_bc=lambda state,candidate: capture_open(state,candidate,radiative_original,('ru_t','rv_t'),'radiative')
            dycore.apply_open_zero_gradient=lambda state,candidate: capture_open(state,candidate,gradient_original,('u','v','w','thp')+(('qv','qc','qr') if args.moist else ()),'gradient')
            initial_u=np.asarray(host.u);initial_v=np.asarray(host.v)
            for face in (initial_u[:,:,0],initial_u[:,:,-1],initial_v[:,0,:],initial_v[:,-1,:]):
                assert np.any(face < 0) and np.any(face > 0), 'both inflow and outflow must be seeded on each face'
        acoustic_records={}
        upper_records={}
        flux_records={}
        if args.rayleigh:
            original_acoustic=dycore.prepare_acoustic_substep_launch
            def capture_acoustic(state,candidate,*launch_args,**kwargs):
                launch=original_acoustic(state,candidate,*launch_args,**kwargs)
                def capture(*,first):
                    def read(v): return v.to_numpy() if getattr(state,'_metal_setup_state',False) else np.array(v,copy=True)
                    before=read(state.u) if args.upper_wind else None
                    launch(first=first)
                    if args.upper_wind:
                        upper_records.setdefault(id(state),[]).append((before,read(state.u),read(state.v)))
                    acoustic_records.setdefault(id(state),[]).append((read(state.w_pp),read(state.ph_pp)))
                capture.absorbs=getattr(launch,'absorbs',frozenset())
                return capture
            dycore.prepare_acoustic_substep_launch=capture_acoustic
        if args.upper_wind:
            original_refresh=dycore.refresh_saved_wind_fluxes
            def capture_refresh(state,candidate,ru,rv):
                original_refresh(state,candidate,ru,rv)
                def read(v): return v.to_numpy() if getattr(state,'_metal_setup_state',False) else np.array(v,copy=True)
                flux_records.setdefault(id(state),[]).append((read(ru),read(rv)))
            dycore.refresh_saved_wind_fluxes=capture_refresh
        damping_records={}
        if args.w_damp:
            original_damp=dycore.apply_w_damping
            def capture_damp(state,candidate,ww):
                def read(value): return value.to_numpy() if getattr(state,'_metal_setup_state',False) else np.array(value,copy=True)
                before=read(state.rw_t)
                original_damp(state,candidate,ww)
                after=read(state.rw_t)
                damping_records.setdefault(id(state),[]).append((before,after))
            dycore.apply_w_damping=capture_damp
        resumed=None
        atmosphere_seen=0
        atmosphere_resume_offset=None
        checkpoint_dir=tempfile.TemporaryDirectory(prefix='strict-dudhia-restart-') if args.checkpoint else None
        if args.tree_checkpoint:
            from woof.core.clock import TickClock, Schedule
            from woof.core.model import DomainNode, ExperimentState, ModelRuntimeStatus
            def tree(state, clock):
                tick=TickClock(clock.tick_den,args.steps*clock.step_ticks,
                               datetime(2026,1,1,12,tzinfo=timezone.utc),(clock.spec,))
                schedule=Schedule(tick,(),())
                domain=SimpleNamespace(grid_id=1,parent_id=0,run=cfg,start_time=None)
                node=DomainNode(domain,None,state,clock,None,[],None)
                model=ExperimentState(node,MappingProxyType({1:node}),schedule,None,
                                      'synthetic-strict-radiation-tree')
                model._runtime_status=ModelRuntimeStatus()
                if args.nwp_diagnostics:
                    from woof.core.nest_spawn import SpawnConfig, SpawnController, SpawnWatch
                    from woof.core.spawn_runner import SpawnRunner
                    from woof.core.storm_tracking import NestFootprint
                    watch=SpawnWatch(SpawnConfig(trigger='time',at_s=100.0),
                        NestFootprint(2,4,4,7,7,3,parent_id=1),keepout_cells=1)
                    controller=SpawnController({2:watch},{2:1})
                    experiment=SimpleNamespace(domains=(SimpleNamespace(grid_id=2,retire=None),))
                    model._spawn_runner=SpawnRunner(experiment=experiment,
                        on_child_built=lambda *_:None,controller=controller,array_module=xp)
                return model
            continuous_tree=tree(metal,clocks[1])
            result['tree_scope']='synthetic one-root model with dormant lifecycle; direct dycore steps, not the tree executor' if args.nwp_diagnostics else 'synthetic one-root model; direct dycore steps, not the tree executor'
        for n in range(args.steps):
            if args.specified or args.tree_checkpoint:
                for clock in clocks:
                    if args.boundary_crossover and n>0: clock.mark_force()
                    clock.prepare_step()
                if args.boundary_crossover:
                    from woof.ingest.lateral_bc import _served_external_interval
                    served.append(tuple(_served_external_interval(s,s._lateral_boundary_device)[0].start_seconds
                                        for s in (host,metal)))
            dycore.step(host,cfg)
            dycore.step(metal,cfg)
            if args.goddard:
                for stage in goddard_host_stages.keys() & goddard_metal_stages.keys():
                    aa=goddard_host_stages[stage];bb=goddard_metal_stages[stage]
                    for name in aa.keys() & bb.keys():
                        result['cases'][f'goddard_stage_{stage}_{name}']=compare_words(bb[name].reshape(-1),aa[name].reshape(-1))
                a=radiation_snapshots.get(id(host),[]);b=radiation_snapshots.get(id(metal),[])
                for j in range(min(len(a),len(b))):
                    for name in a[j]:
                        result['cases'][f'goddard_call{j+1}_{name}']=compare_words(b[j][name],a[j][name])
            if args.nwp_diagnostics:
                for slot in ('up_heli_max','uh_diag_col','uh_diag_use',*uh_diag.TRACKER_WINDOW_SLOTS):
                    got=metal.existing_scratch(slot).to_numpy();want=host.existing_scratch(slot)
                    result['cases'][f'step{n+1}_uh_{slot}']=compare_words(got,want)
                active=int(np.count_nonzero(host.existing_scratch('up_heli_max')))
                result.setdefault('uh_active_cells',[]).append(active)
                assert active>0, 'UH diagnostic was inactive'
                from woof.io.history_layout import live_state_history_fields
                assert live_state_history_fields(metal)['UP_HELI_MAX'] is metal.existing_scratch('up_heli_max')
                if n+1==args.checkpoint_after:
                    saved=[metal.existing_scratch(slot).to_numpy().copy() for slot in uh_diag.TRACKER_WINDOW_SLOTS]
                    uh_diag.reset_up_heli_max(host);uh_diag.reset_up_heli_max(metal)
                    assert not np.count_nonzero(metal.existing_scratch('up_heli_max').to_numpy())
                    assert all(np.array_equal(saved[i].view(np.uint32),metal.existing_scratch(slot).to_numpy().view(np.uint32)) for i,slot in enumerate(uh_diag.TRACKER_WINDOW_SLOTS))
            for name in diagnostic_slots:
                result['cases'][f'step{n+1}_'+name]=compare_words(
                    metal.existing_scratch(name).to_numpy(),host.existing_scratch(name))
            if args.tke_budget:
                terms,raw,acc,resume_acc=budget_host[-1]
                if (args.coordinate2 or args.private_coordinate2) and np.any(terms[:4] != 0):
                    raise AssertionError('coordinate TKE unexpectedly ran tke_rhs')
                if args.diff6 and not np.any(terms[tke_budget.TERM_FIELDS.index('diffusion_6th')] != 0):
                    raise AssertionError('sixth-order TKE budget was zero')
                transport=terms[tke_budget.TERM_FIELDS.index('transport')]
                if not np.any(transport != 0):
                    raise AssertionError('budget transport was zero')
                if args.mapped and not np.any(transport != transport / host.msft[None]):
                    raise AssertionError('budget map factor had no effect')
                result.setdefault('budget_transport_nonzero',[]).append(int(np.count_nonzero(transport)))
                for name,value,expected in (
                    ('terms',tke_budget.term_buffer(metal,cfg),terms),
                    ('raw',tke_budget.raw_carrier(metal,cfg),raw),
                    ('acc',tke_budget.accumulator(metal,cfg),acc)):
                    result['cases'][f'step{n+1}_budget_{name}']=(compare_float64 if name=='acc' else compare_words)(value.to_numpy(),expected)
            if args.specified or args.tree_checkpoint:
                for clock in clocks: clock.advance()
            if args.checkpoint and n+1==args.checkpoint_after:
                from woof.io.restart import (write_restart,restore_restart,
                                             write_tree_restart,restore_tree_restart)
                if args.tree_checkpoint:
                    path=write_tree_restart(Path(checkpoint_dir.name),continuous_tree,
                        continuous_tree.schedule.clock.start_time+timedelta(seconds=(n+1)*cfg.dt))
                else:
                    path=Path(checkpoint_dir.name)/'checkpoint.npz'
                    write_restart(path,metal,cfg)
                resumed=(igw.build(cfg,coord,base,array_module=xp) if args.igw_2d else
                         init_theta_perturbation(cfg,coord,base,perturb,array_module=xp))
                setup_igw_v(resumed,xp)
                if args.igw_2d: adopt_wrf_theta_carrier(resumed)
                setup_map(resumed,xp);setup_open(resumed,xp);setup_moist(resumed,xp)
                setup_uh(resumed,xp)
                if args.specified:
                    attach(resumed)
                    resume_clock=clocks.pop()
                elif args.tree_checkpoint:
                    resume_clock=periodic_clock()
                if args.dudhia:
                    attach_physics(resumed)
                    if args.terrain_mode:
                        atmosphere_resume_offset=len(atmosphere_snapshots.get(id(metal),[]))
                if args.tree_checkpoint:
                    resumed_tree=tree(resumed,resume_clock)
                    info=restore_tree_restart(path,resumed_tree)
                    assert info.elapsed_ticks==(n+1)*resume_clock.step_ticks
                    source_clock=clocks[1]
                    assert (resume_clock.ticks,resume_clock.step_count,resume_clock.dtbc_fp32.tobytes()) == (
                        source_clock.ticks,source_clock.step_count,source_clock.dtbc_fp32.tobytes())
                    result['restart_clock']='restored by write_tree_restart/restore_tree_restart'
                    result['checkpoint_after_step']=n+1
                else:
                    restore_restart(path,resumed,cfg)
                if args.specified and not args.tree_checkpoint:
                    # Standalone checkpoints do not carry the tree clock.
                    # Hand it forward explicitly; this is not a tree restart test.
                    source_clock=clocks[1]
                    resume_clock.ticks=source_clock.ticks
                    resume_clock.step_count=source_clock.step_count
                    resume_clock.dtbc_fp32=source_clock.dtbc_fp32
                    result['restart_clock']='explicit handoff; standalone restart does not restore tree clock'
            elif resumed is not None:
                if args.specified or args.tree_checkpoint:
                    if args.boundary_crossover: resume_clock.mark_force()
                    resume_clock.prepare_step()
                dycore.step(resumed,cfg)
                if args.specified or args.tree_checkpoint: resume_clock.advance()
            if resumed is not None:
                if args.nwp_diagnostics:
                    for slot in ('up_heli_max',*uh_diag.TRACKER_WINDOW_SLOTS):
                        result['cases'][f'step{n+1}_restart_uh_{slot}']=compare_words(
                            metal.existing_scratch(slot).to_numpy(),resumed.existing_scratch(slot).to_numpy())
                if args.tke_budget and n+1>args.checkpoint_after:
                    result['cases'][f'step{n+1}_restart_budget_acc']=compare_float64(
                        tke_budget.accumulator(resumed,cfg).to_numpy(),resume_acc)
                    report=tke_budget.drain(resumed,cfg,reset_window=False)
                    assert report['steps']==n+1-args.checkpoint_after
                    for row,name in enumerate(tke_budget.TERMS):
                        result['cases'][f'step{n+1}_restart_budget_report_'+name]=compare_float64(
                            report['profiles'][name],resume_acc[row]/report['steps'])
                for field in fields:
                    result['cases'][f'step{n+1}_restart_'+field]=compare_words(
                        getattr(metal,field).to_numpy(),getattr(resumed,field).to_numpy())
                for name in diagnostic_slots:
                    result['cases'][f'step{n+1}_restart_'+name]=compare_words(
                        metal.existing_scratch(name).to_numpy(),resumed.existing_scratch(name).to_numpy())
                if args.dudhia:
                    for field in ('rthratenlw','rthratensw','rthraten','_wrf_rthraten'):
                        result['cases'][f'step{n+1}_restart_'+field]=compare_words(
                            getattr(metal.physics,field).to_numpy(),getattr(resumed.physics,field).to_numpy())
                result['cases'][f'step{n+1}_restart_muts']=compare_words(
                    metal.existing_scratch(WRF_MUTS_SLOT).to_numpy(),
                    resumed.existing_scratch(WRF_MUTS_SLOT).to_numpy())
                if args.coordinate4 or args.private_coordinate2 or args.coordinate2:
                    result['cases'][f'step{n+1}_restart_theta_initial']=compare_words(
                        metal.existing_scratch('diff1_theta_initial').to_numpy(),
                        resumed.existing_scratch('diff1_theta_initial').to_numpy())
                if args.dudhia:
                    result['cases'][f'step{n+1}_restart_radiation_tendency']=compare_words(
                        metal.physics.radiation_tendencies.rtheta.to_numpy(),
                        resumed.physics.radiation_tendencies.rtheta.to_numpy())
                    if n+1>args.checkpoint_after:
                        # The composed work buffer is rebuilt on compute, not serialized.
                        result['cases'][f'step{n+1}_restart_tendency']=compare_words(
                            metal.physics.tendencies.rtheta.to_numpy(),resumed.physics.tendencies.rtheta.to_numpy())
                    surface_fields=('swdown','glw')
                    if args.noah:
                        from woof.core.noah import _F2D, _F3D
                        surface_fields=tuple(dict.fromkeys((*surface_fields,*_F2D,*_F3D,'ebal','q2','t2','th2')))
                    for field in surface_fields:
                        result['cases'][f'step{n+1}_restart_'+field]=compare_words(
                            metal.physics.fields[field].to_numpy(),resumed.physics.fields[field].to_numpy())
                if args.tree_checkpoint:
                    assert (resume_clock.ticks,resume_clock.step_count,resume_clock.dtbc_fp32.tobytes()) == (
                        clocks[1].ticks,clocks[1].step_count,clocks[1].dtbc_fp32.tobytes())
                    if args.dudhia:
                        assert resumed.physics.call_counts['radiation']==metal.physics.call_counts['radiation']==1+n//5
            metal_muts=metal.existing_scratch(WRF_MUTS_SLOT)
            host_muts=host.existing_scratch(WRF_MUTS_SLOT)
            if metal_muts is None or host_muts is None:
                raise AssertionError('carried MUTS missing after step')
            if args.igw_2d or args.open_mode or args.dudhia or args.mixing or args.specified_moist or args.terrain_mode:
                hst=stage_snapshots[id(host)][-3:];mst=stage_snapshots[id(metal)][-3:]
                assert len(hst)==len(mst)==3
                if args.igw_v_seed:
                    for stage in hst:
                        v=stage['v']
                        assert np.any(v != 0) and np.array_equal(v[:,0,:].view(np.uint32),v[:,1,:].view(np.uint32))
                if args.mixing:
                    if not args.surface_only:
                        assert np.any(hst[0]['smag_km'] != 0), 'mixing coefficient was zero'
                    assert any(np.any(hst[0][key] != 0) for key in ('smag_ru','smag_rv','smag_rth')), 'mixing tendencies were zero'
                for stage,(hstage,mstage) in enumerate(zip(hst,mst),1):
                    assert hstage.keys()==mstage.keys()
                    for field in hstage:
                        result['cases'][f'step{n+1}_stage{stage}_'+field]=compare_words(
                            mstage[field],hstage[field])
                if (args.open_mode or args.mixing or args.specified_moist or args.terrain_mode) and resumed is not None and n+1>args.checkpoint_after:
                    rst=stage_snapshots[id(resumed)][-3:]
                    assert len(rst)==3
                    for stage,(mstage,rstage) in enumerate(zip(mst,rst),1):
                        assert mstage.keys()==rstage.keys()
                        for field in mstage:
                            result['cases'][f'step{n+1}_restart_stage{stage}_'+field]=compare_words(
                                mstage[field],rstage[field])
            if args.dudhia:
                if args.terrain_mode:
                    host_at=atmosphere_snapshots.get(id(host),[])
                    metal_at=atmosphere_snapshots.get(id(metal),[])
                    assert len(host_at)==len(metal_at) and len(host_at)>0
                    for index in range(atmosphere_seen,len(host_at)):
                        assert host_at[index].keys()==metal_at[index].keys()
                        for name in host_at[index]:
                            result['cases'][f'atmosphere{index}_'+name]=compare_words(
                                metal_at[index][name],host_at[index][name])
                    atmosphere_seen=len(host_at)
                    if resumed is not None and n+1>args.checkpoint_after:
                        restored_at=atmosphere_snapshots.get(id(resumed),[])
                        expected_at=metal_at[atmosphere_resume_offset:]
                        assert len(restored_at)==len(expected_at)
                        for index,(expected,restored) in enumerate(zip(expected_at,restored_at)):
                            assert expected.keys()==restored.keys()
                            for name in expected:
                                result['cases'][f'restart_atmosphere{index}_'+name]=compare_words(
                                    restored[name],expected[name])
                for field in ('rthratenlw','rthratensw','rthraten','_wrf_rthraten'):
                    result['cases'][f'step{n+1}_physics_'+field]=compare_words(
                        getattr(metal.physics,field).to_numpy(),getattr(host.physics,field))
                result['cases'][f'step{n+1}_physics_tendency']=compare_words(
                    metal.physics.tendencies.rtheta.to_numpy(),host.physics.tendencies.rtheta)
                surface_fields=('swdown','glw')
                if args.noah:
                    from woof.core.noah import _F2D, _F3D
                    surface_fields=tuple(dict.fromkeys((*surface_fields,*_F2D,*_F3D,'ebal','q2','t2','th2')))
                for field in surface_fields:
                    result['cases'][f'step{n+1}_physics_'+field]=compare_words(
                        metal.physics.fields[field].to_numpy(),host.physics.fields[field])
                assert metal.physics.call_counts['radiation']==host.physics.call_counts['radiation']==(1+n//5)
                heating=metal.physics.rthratensw.to_numpy() if (args.goddard or args.classic) else metal.physics.rthraten.to_numpy()
                assert np.any(heating!=0) and np.any(heating==0), 'day and night not both exercised'
            result['cases'][f'step{n+1}_muts']=compare_words(metal_muts.to_numpy(),host_muts)
            for field in fields:
                reference=getattr(host,field)
                if (args.terrain_mode or args.vertical_pattern) and not np.all(np.isfinite(reference)):
                    raise AssertionError(f'nonfinite state: {field}')
                result['cases'][f'step{n+1}_'+field]=compare_words(
                    getattr(metal,field).to_numpy(),reference)
            if (args.horizontal5 or args.vertical5) and args.moist and args.moist_adv_opt == 1 and not args.open_mode:
                if any(np.any(getattr(host,name) < 0) for name in ('qv','qc','qr')):
                    raise AssertionError('positive-definite moisture became negative')
            if args.moist and (args.terrain_mode or args.vertical_pattern) and not all(np.any(getattr(host,name) != 0)
                                             for name in ('qv','qc','qr')):
                raise AssertionError('moisture fields were inactive')
        if args.open_mode:
            result['open_activity']={}
            for label in ('radiative','gradient'):
                h=open_records[(id(host),label)];m=open_records[(id(metal),label)]
                assert len(h)==len(m)==3*args.steps
                changed=sum(x[1] for x in h)
                assert changed, f'open {label} inactive'
                result['open_activity'][label+'_changed_words']=changed
                for index,((ha,_),(ma,_)) in enumerate(zip(h,m)):
                    for name in ha:
                        result['cases'][f'open_{label}{index}_{name}']=compare_words(ma[name],ha[name])
                if resumed is not None:
                    r=open_records[(id(resumed),label)]; ref=m[-len(r):]
                    assert len(r)>0
                    for index,((ma,_),(ra,_)) in enumerate(zip(ref,r)):
                        for name in ma:
                            result['cases'][f'restart_open_{label}{index}_{name}']=compare_words(ra[name],ma[name])
        if args.rayleigh:
            h=acoustic_records[id(host)]; m=acoustic_records[id(metal)]
            assert len(h)==len(m) and len(h)>3*args.steps
            for index,((hw,hp),(mw,mp)) in enumerate(zip(h,m)):
                result['cases'][f'acoustic{index}_w']=compare_words(mw,hw)
                result['cases'][f'acoustic{index}_phi']=compare_words(mp,hp)
            if resumed is not None:
                r=acoustic_records[id(resumed)]; ref=m[-len(r):]
                assert len(r)>0
                for index,((mw,mp),(rw,rp)) in enumerate(zip(ref,r)):
                    result['cases'][f'restart_acoustic{index}_w']=compare_words(rw,mw)
                    result['cases'][f'restart_acoustic{index}_phi']=compare_words(rp,mp)
        if args.upper_wind:
            h=upper_records[id(host)]; m=upper_records[id(metal)]
            assert len(h)==len(m) and len(h)>3*args.steps
            changed=sum(int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))) for a,b,_ in h)
            if not changed: raise AssertionError('upper-wind limiter inactive')
            result['upper_wind_changed_words']=changed
            for index,((_,hu,hv),(_,mu,mv)) in enumerate(zip(h,m)):
                result['cases'][f'upper{index}_u']=compare_words(mu,hu)
                result['cases'][f'upper{index}_v']=compare_words(mv,hv)
            hf=flux_records[id(host)]; mf=flux_records[id(metal)]
            assert len(hf)==len(mf)==3*args.steps
            for index,((hu,hv),(mu,mv)) in enumerate(zip(hf,mf)):
                result['cases'][f'refresh{index}_ru']=compare_words(mu,hu)
                result['cases'][f'refresh{index}_rv']=compare_words(mv,hv)
            if resumed is not None:
                r=upper_records[id(resumed)]; ref=m[-len(r):]
                assert len(r)>0
                for index,((_,mu,mv),(_,ru,rv)) in enumerate(zip(ref,r)):
                    result['cases'][f'restart_upper{index}_u']=compare_words(ru,mu)
                    result['cases'][f'restart_upper{index}_v']=compare_words(rv,mv)
                rf=flux_records[id(resumed)]; ref=mf[-len(rf):]
                assert len(rf)>0
                for index,((mu,mv),(ru,rv)) in enumerate(zip(ref,rf)):
                    result['cases'][f'restart_refresh{index}_ru']=compare_words(ru,mu)
                    result['cases'][f'restart_refresh{index}_rv']=compare_words(rv,mv)
        if args.w_damp:
            h=damping_records[id(host)]; m=damping_records[id(metal)]
            assert len(h)==len(m)==3*args.steps
            if not any(np.any(a.view(np.uint32)!=b.view(np.uint32)) for a,b in h):
                raise AssertionError('vertical damping was inactive')
            result['damping_changed_words']=sum(int(np.count_nonzero(a.view(np.uint32)!=b.view(np.uint32))) for a,b in h)
            for index,((ha,hb),(ma,mb)) in enumerate(zip(h,m)):
                result['cases'][f'damping{index}_before']=compare_words(ma,ha)
                result['cases'][f'damping{index}_after']=compare_words(mb,hb)
            if resumed is not None:
                r=damping_records[id(resumed)]; ref=m[-len(r):]
                for index,((ma,mb),(ra,rb)) in enumerate(zip(ref,r)):
                    result['cases'][f'restart_damping{index}_before']=compare_words(ra,ma)
                    result['cases'][f'restart_damping{index}_after']=compare_words(rb,mb)
        if args.open_top and not np.any(host.w[-1] != 0):
            raise AssertionError('open top vertical velocity is inactive')
        if args.terrain_mode or args.mixing or args.horizontal5 or args.vertical_pattern:
            result['terrain_final_digest']=hashlib.sha256(b''.join(
                np.ascontiguousarray(getattr(host,name)).tobytes() for name in fields)).hexdigest()
        if args.moist and args.diff6:
            result['moisture_final_digest']=hashlib.sha256(b''.join(
                np.ascontiguousarray(getattr(host,name)).tobytes() for name in ('qv','qc','qr'))).hexdigest()
        if args.tke_budget:
            report=tke_budget.drain(metal,cfg,reset_window=False)
            assert report['steps']==args.steps
            for row,name in enumerate(tke_budget.TERMS):
                result['cases']['budget_report_'+name]=compare_float64(
                    report['profiles'][name],budget_acc[row]/report['steps'])
            assert tke_budget.drain(metal,cfg)['steps']==args.steps
            assert tke_budget.drain(metal,cfg) is None
            if resumed is not None:
                assert tke_budget.drain(resumed,cfg)['steps']==args.steps-args.checkpoint_after
                assert tke_budget.drain(resumed,cfg) is None
        if args.boundary_crossover:
            result['served_intervals']=served
            if served[:2] != [(0.,0.),(1.,1.)]:
                raise AssertionError('boundary crossover was not exercised')
        if args.climate_diagnostics:
            from woof.core.climate_diagnostics import HISTORY
            result['climate_statistics']['nonzero_history_words']={name:int(np.count_nonzero(metal.existing_scratch(name).to_numpy())) for name in HISTORY}
    if args.noah:
        result['noah_nonzero_words']={name:int(np.count_nonzero(host.physics.fields[name]))
            for name in ('hfx','qfx','grdflx','tsk','smois','sh2o')}
        assert result['noah_nonzero_words']['hfx'], 'Noah sensible heat flux was inactive'
    result['words']=sum(x['words'] for x in result['cases'].values())
    result['different_words']=sum(x['different_words'] for x in result['cases'].values())
    if args.surface_flux is not None:
        result['surface_effects']=surface_effects
        assert len(surface_effects)==args.steps and all(row['rth'] for row in surface_effects)
        if args.surface_flux == 0:
            assert all(row['ru'] and row['rv'] for row in surface_effects)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if result['different_words']: raise SystemExit(1)


if __name__=='__main__': main()
