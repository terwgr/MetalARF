#!/usr/bin/env python3
"""Bounded synchronous Metal glibc-helper comparison with the NumPy authority.

Needs Xcode/Metal Toolchain. Compiles a temporary probe, not production code.
Binary64 polynomial words are checked as well as FP32 outputs: FP32 rounding
can conceal an incorrect fused intermediate. No CUDA device is needed.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, subprocess, sys, tempfile
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from woof.core import noahmp_libm as ref

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=pathlib.Path, required=True)
    p.add_argument('--shader-root', type=pathlib.Path, default=ROOT/'woof/metal/shaders')
    p.add_argument('--record-only', action='store_true', help='retain a failing baseline without accepting it')
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261012)
    centers = [0,1,0x007fffff,0x00800000,0x3f330000,0x3f800000,0x40000000,
               0x42b17218,0x42cff1b4,0x7f7fffff,0x7f800000,0x7fc00000,
               0x80000000,0x80000001,0xbf800000,0xff800000]
    centers += [0x3f330000+(i<<19) for i in range(17)]
    boundary = np.array([(v+d)&0xffffffff for v in centers for d in range(-3,4)],np.uint32)
    dense = np.arange(0x3f800000-4096,0x3f800000+4096,dtype=np.uint32)
    x = np.concatenate([boundary,dense,rng.integers(0,2**32,16384,dtype=np.uint32)])
    y = rng.uniform(-8,8,len(x)).astype(np.float32).view(np.uint32)
    # Explicit exceptional pow branches (signed zero, infinity, NaN, integer/noninteger exponents).
    specials = np.array([0,0x80000000,0x3f800000,0xbf800000,0x3f000000,0x40000000,0x7f800000,0xff800000,0x7fc00000],np.uint32)
    sx,sy=np.meshgrid(np.array(centers,np.uint32),specials,indexing='ij')
    inputs=np.column_stack([np.concatenate([x,sx.ravel()]),np.concatenate([y,sy.ravel()])]).astype('<u4');inputs.tofile(out/'inputs.bin')
    expected=np.empty((len(inputs),5),np.uint64)
    with np.errstate(all='ignore'):
        for i,(xb,yb) in enumerate(inputs):
            xv,yv=np.array([xb,yb],np.uint32).view(np.float32)
            expected[i,:3]=[int(np.asarray(fn(*args),np.float32).view(np.uint32)) for fn,args in [(ref.logf,(xv,)),(ref.expf,(xv,)),(ref.powf,(xv,yv))]]
            normal=0x3f000000|(int(xb)&0x00ffffff)
            expected[i,3]=ref._asuint64(ref._powf_log2(normal))
            xd=float(ref._asfloat(normal))*ref._EXP2F_INVLN2_SCALED
            expected[i,4]=ref._asuint64(ref._exp2_core(xd,ref._EXP2F_SHIFT,ref._EXP2F_POLY_SCALED,0))
    expected.tofile(out/'reference.bin')
    with tempfile.TemporaryDirectory(prefix='metal-gfk-') as temp:
        t=pathlib.Path(temp); source=ROOT/'tools/metal_gfk_probe'
        commands=[['xcrun','-sdk','macosx','metal','-std=metal3.1','-fno-fast-math','-I',a.shader_root,'-c',source/'probe.metal','-o',t/'probe.air'],
                  ['xcrun','-sdk','macosx','metallib',t/'probe.air','-o',t/'probe.metallib'],
                  ['xcrun','--sdk','macosx','clang++','-std=c++17','-fobjc-arc',source/'driver.mm','-framework','Foundation','-framework','Metal','-o',t/'driver'],
                  [t/'driver',t/'probe.metallib',out/'inputs.bin',out/'metal.bin']]
        for cmd in commands:subprocess.run(list(map(str,cmd)),check=True,timeout=240)
    got=np.fromfile(out/'metal.bin',dtype='<u8').reshape(expected.shape)
    names=['logf_fp32','expf_fp32','powf_fp32','pow_log2_binary64','exp2_core_binary64']
    counts={n:int(np.count_nonzero(got[:,i]!=expected[:,i])) for i,n in enumerate(names)}
    witnesses={}
    for col,name in enumerate(names):
        ids=np.flatnonzero(got[:,col]!=expected[:,col])[:8]
        witnesses[name]=[{'x_bits':hex(int(inputs[i,0])),'y_bits':hex(int(inputs[i,1])),'expected':hex(int(expected[i,col])),'metal':hex(int(got[i,col]))} for i in ids]
    receipt={'rows':len(inputs),'seed':20261012,'synchronous':True,'mismatches':counts,'witnesses':witnesses,
             'input_sha256':hashlib.sha256((out/'inputs.bin').read_bytes()).hexdigest(),
             'shader_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.shader_root.glob('soft*.metalh')}}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'rows':len(inputs),'mismatches':counts},indent=2))
    return 0 if not any(counts.values()) or a.record_only else 1
if __name__=='__main__':raise SystemExit(main())
