from pathlib import Path
import ctypes, importlib.util, json, subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import tempfile
d=Path(tempfile.mkdtemp(prefix='woof-lake-sine-'))
spec=importlib.util.spec_from_file_location('trig',ROOT/'tools/metal_trig_word_probe.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
r=mod.RUNNER.replace('uint32_t','uint64_t').replace('bytes / 4','bytes / 8').replace('bytes*6','bytes').replace('function < 6','function < 1').replace('{"sinf", "cosf", "tanf", "asinf", "acosf", "atanf"}','{"sin"}')
(d/'run.mm').write_text(r)
(d/'probe.metal').write_text('''#include <metal_stdlib>\nusing namespace metal;\n#include "lake_wcr_sin.metalh"\nkernel void probe_sin(device const ulong *x [[buffer(0)]],device ulong *out [[buffer(1)]],constant ulong &n [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<n)out[i]=lake_wcr_sin_bits(x[i]);}\n''')
lat=np.concatenate([np.linspace(-90,90,18001,dtype=np.float32),np.random.default_rng(19).uniform(-90,90,10000).astype(np.float32)])
x=lat.astype(np.float64)*np.float64(np.pi)/180.
extra=np.array([0.,-0.,np.nextafter(0.,1.),np.nextafter(0.,-1.),np.pi/4.,-np.pi/4.,np.pi/2.,-np.pi/2.],np.float64)
x=np.concatenate((x,extra));lat=np.concatenate((lat,np.rad2deg(extra).astype(np.float32)));x.tofile(d/'input.bin')
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.1','-fmetal-math-mode=safe','-ffp-contract=off','-I',str(ROOT/'woof/metal/shaders'),'-c',str(d/'probe.metal'),'-o',str(d/'probe.air')],check=True)
subprocess.run(['xcrun','-sdk','macosx','metallib',str(d/'probe.air'),'-o',str(d/'probe.metallib')],check=True)
subprocess.run(['clang++','-std=c++17','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',str(d/'run.mm'),'-o',str(d/'run')],check=True)
subprocess.run([str(d/'run'),str(d/'probe.metallib'),str(d/'input.bin'),str(d/'output.bin')],check=True)
(d/'host.cpp').write_text('#include "f64cr.cuh"\nextern "C" double s(double x){return wcr_sin(x);}\n')
subprocess.run(['clang++','-std=c++17','-O0','-fPIC','-fno-fast-math','-ffp-contract=off','-shared','-I',str(ROOT/'woof/core/kernels'),str(d/'host.cpp'),'-o',str(d/'host.dylib')],check=True)
h=ctypes.CDLL(str(d/'host.dylib'));h.s.argtypes=[ctypes.c_double];h.s.restype=ctypes.c_double
reference=np.array([h.s(float(v)) for v in x],np.float64).view(np.uint64);actual=np.fromfile(d/'output.bin',dtype=np.uint64);bad=np.flatnonzero(reference!=actual)
print(json.dumps({'samples':int(x.size),'mismatches':int(bad.size),'first':[{'latitude':float(lat[k]),'input':float(x[k]).hex(),'reference':hex(int(reference[k])),'metal':hex(int(actual[k]))} for k in bad[:8]]},indent=2))

if bad.size: raise SystemExit(1)
