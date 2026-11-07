from pathlib import Path
import ctypes, importlib.util, json, subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
import tempfile
d=Path(tempfile.mkdtemp(prefix='woof-lake-log-'))
spec=importlib.util.spec_from_file_location('trig',ROOT/'tools/metal_trig_word_probe.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
r=mod.RUNNER.replace('uint32_t','uint64_t').replace('bytes / 4','bytes / 8').replace('bytes*6','bytes*2').replace('function < 6','function < 2').replace('{"sinf", "cosf", "tanf", "asinf", "acosf", "atanf"}','{"log", "log10"}')
(d/'run.mm').write_text(r)
(d/'probe.metal').write_text('''#include <metal_stdlib>\nusing namespace metal;\n#include "lake_wcr_log.metalh"\nkernel void probe_log(device const ulong *x [[buffer(0)]],device ulong *out [[buffer(1)]],constant ulong &n [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<n)out[i]=lake_wcr_log_bits(x[i]);}\nkernel void probe_log10(device const ulong *x [[buffer(0)]],device ulong *out [[buffer(1)]],constant ulong &n [[buffer(2)]],uint i [[thread_position_in_grid]]){if(i<n)out[i]=lake_wcr_log10_bits(x[i]);}\n''')
rng=np.random.default_rng(71);bits=rng.integers(1,0x7ff0000000000000,25000,dtype=np.uint64)
near=1.+np.arange(-4096,4097,dtype=np.float64)*2.**-52
x=np.concatenate((bits.view(np.float64),near,np.array([0.,-0.,-1.,np.inf,1.,np.nextafter(0.,1.),np.finfo(np.float64).tiny,np.nextafter(1.,0.),np.nextafter(1.,2.)])));extra=np.array([0x7ff8000000001234,0xfff8000000001234,0x7ff0000000001234,0xfff0000000001234,0xfff0000000000000,0xbff8000000000000,0x8000000000000001],np.uint64).view(np.float64)
x=np.concatenate((x,extra));x.tofile(d/'input.bin')
subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.1','-fmetal-math-mode=safe','-ffp-contract=off','-I',str(ROOT/'woof/metal/shaders'),'-c',str(d/'probe.metal'),'-o',str(d/'probe.air')],check=True)
subprocess.run(['xcrun','-sdk','macosx','metallib',str(d/'probe.air'),'-o',str(d/'probe.metallib')],check=True)
subprocess.run(['clang++','-std=c++17','-O2','-fobjc-arc','-framework','Foundation','-framework','Metal',str(d/'run.mm'),'-o',str(d/'run')],check=True)
subprocess.run([str(d/'run'),str(d/'probe.metallib'),str(d/'input.bin'),str(d/'output.bin')],check=True)
(d/'host.cpp').write_text('#include "f64cr.cuh"\nextern "C" double l(double x){return wcr_log(x);}\nextern "C" double t(double x){return wcr_log10(x);}\n')
subprocess.run(['clang++','-std=c++17','-O0','-fPIC','-fno-fast-math','-ffp-contract=off','-shared','-I',str(ROOT/'woof/core/kernels'),str(d/'host.cpp'),'-o',str(d/'host.dylib')],check=True)
h=ctypes.CDLL(str(d/'host.dylib'));actual=np.fromfile(d/'output.bin',dtype=np.uint64).reshape(2,-1);results=[]
for j,name in enumerate(('l','t')):
 fn=getattr(h,name);fn.argtypes=[ctypes.c_double];fn.restype=ctypes.c_double;ref=np.array([fn(float(v)) for v in x],np.float64).view(np.uint64);bad=np.flatnonzero(ref!=actual[j]);results.append({'function':name,'samples':int(x.size),'mismatches':int(bad.size),'first':[{'input':float(x[k]).hex(),'reference':hex(int(ref[k])),'metal':hex(int(actual[j,k]))} for k in bad[:5]]})
print(json.dumps(results,indent=2));
if any(z['mismatches'] for z in results):raise SystemExit(1)
