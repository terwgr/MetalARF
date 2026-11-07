from pathlib import Path
import ctypes, importlib.util, json, subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import tempfile
d=Path(tempfile.mkdtemp(prefix='woof-lake-exp-'))
spec=importlib.util.spec_from_file_location('trig',ROOT/'tools/metal_trig_word_probe.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
r=mod.RUNNER.replace('uint32_t','uint64_t').replace('bytes / 4','bytes / 8').replace('bytes*6','bytes').replace('function < 6','function < 1').replace('{"sinf", "cosf", "tanf", "asinf", "acosf", "atanf"}','{"exp"}')
(d/'run.mm').write_text(r)
(d/'probe.metal').write_text('''#include <metal_stdlib>\nusing namespace metal;\n#include "lake_wcr_exp.metalh"\nkernel void probe_exp(device const ulong *x [[buffer(0)]],device ulong *out [[buffer(1)]],constant ulong &n [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<n)out[i]=lake_wcr_exp_bits(x[i]);}\n''')
rng=np.random.default_rng(31)
x=np.concatenate((rng.uniform(-746,710,20000),rng.uniform(-.004,.004,10000),np.array([0.,-0.,2.**-54,-2.**-54,2.**-8,-2.**-8,710.,-746.,-745.,-708.],np.float64)))
k=np.arange(-4096,4097,dtype=np.float64)*2.**-53
threshold=np.array([-746.,-745.,-708.,-2.**-8,-2.**-54,0.,2.**-54,2.**-8,710.],np.float64)
x=np.concatenate((x,k,np.nextafter(threshold,-np.inf),np.nextafter(threshold,np.inf),np.array([np.inf,-np.inf])))
x=np.concatenate((x,np.array([0x7ff8000000001234,0xfff8000000001234,0x7ff0000000001234,0xfff0000000001234],np.uint64).view(np.float64)))
lat=x.astype(np.float32);x.tofile(d/'input.bin')
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.1','-fmetal-math-mode=safe','-ffp-contract=off','-I',str(ROOT/'woof/metal/shaders'),'-c',str(d/'probe.metal'),'-o',str(d/'probe.air')],check=True)
subprocess.run(['xcrun','-sdk','macosx','metallib',str(d/'probe.air'),'-o',str(d/'probe.metallib')],check=True)
subprocess.run(['clang++','-std=c++17','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',str(d/'run.mm'),'-o',str(d/'run')],check=True)
subprocess.run([str(d/'run'),str(d/'probe.metallib'),str(d/'input.bin'),str(d/'output.bin')],check=True)
(d/'host.cpp').write_text('#include "f64cr.cuh"\nextern "C" double s(double x){return wcr_exp(x);}\n')
subprocess.run(['clang++','-std=c++17','-O0','-fPIC','-fno-fast-math','-ffp-contract=off','-shared','-I',str(ROOT/'woof/core/kernels'),str(d/'host.cpp'),'-o',str(d/'host.dylib')],check=True)
h=ctypes.CDLL(str(d/'host.dylib'));h.s.argtypes=[ctypes.c_double];h.s.restype=ctypes.c_double
reference=np.array([h.s(float(v)) for v in x],np.float64).view(np.uint64);actual=np.fromfile(d/'output.bin',dtype=np.uint64);bad=np.flatnonzero(reference!=actual)
print(json.dumps({'samples':int(x.size),'mismatches':int(bad.size),'first':[{'input':float(x[k]).hex(),'reference':hex(int(reference[k])),'metal':hex(int(actual[k]))} for k in bad[:8]]},indent=2))

if bad.size: raise SystemExit(1)
