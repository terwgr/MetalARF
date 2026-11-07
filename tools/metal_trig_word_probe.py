#!/usr/bin/env python3
"""Bounded Metal word comparison against the retained pinned trig source.

Run through the project's GPU queue. This checks the actual software-arithmetic
header, including large reduction and subnormal inputs. The reference compiles
the retained CUDA bodies for the host; it is not a CUDA-device parity claim.
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("sinf", "cosf", "tanf", "asinf", "acosf", "atanf")

RUNNER = r"""
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdint>
#include <vector>
int main(int argc, char **argv) {
  @autoreleasepool {
    if (argc != 4) return 2;
    FILE *f = fopen(argv[2], "rb");
    if (!f) return 3;
    fseek(f, 0, SEEK_END); size_t bytes = ftell(f); rewind(f);
    std::vector<uint32_t> data(bytes / 4);
    if (fread(data.data(), 1, bytes, f) != bytes) return 4;
    fclose(f);
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) return 5;
    NSError *error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:
      [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&error];
    if (!library) { fprintf(stderr, "%s\n", [[error description] UTF8String]); return 6; }
    id<MTLBuffer> input = [device newBufferWithBytes:data.data() length:bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:bytes*6 options:MTLResourceStorageModeShared];
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLCommandBuffer> command = [queue commandBuffer];
    const char *names[] = {"sinf", "cosf", "tanf", "asinf", "acosf", "atanf"};
    uint32_t n = uint32_t(data.size());
    for (uint32_t function = 0; function < 6; ++function) {
      NSString *name = [@"probe_" stringByAppendingString:[NSString stringWithUTF8String:names[function]]];
      id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:name] error:&error];
      if (!pipeline) { fprintf(stderr, "%s\n", [[error description] UTF8String]); return 7; }
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:input offset:0 atIndex:0];
      [encoder setBuffer:output offset:bytes*function atIndex:1];
      [encoder setBytes:&n length:sizeof(n) atIndex:2];
      [encoder dispatchThreads:MTLSizeMake(n,1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];
      [encoder endEncoding];
    }
    [command commit]; [command waitUntilCompleted];
    if ([command status] == MTLCommandBufferStatusError) {
      fprintf(stderr, "%s\n", [[[command error] description] UTF8String]); return 8;
    }
    f = fopen(argv[3], "wb");
    if (!f) return 9;
    const size_t written = fwrite([output contents], 1, bytes*6, f); fclose(f);
    return written == bytes*6 ? 0 : 10;
  }
}
"""


def main():
    # Neighbours of source range boundaries and every exceptional argument.
    boundaries = [0, 1, 0x007fffff, 0x00800000, 0x7f7fffff, 0x7f800000,
        0x7f800001, 0x7fc00000, 0x7fffffff, 0x3f800000, 0x3f2ab445,
        0x3f083a1a, 0x328885a3, 0x39826222, 0x4c700518, 0x3f8a1f62,
        0x4d56d355, 0x57d7b0ed, 0x5980445e, 0x63fc86fe, 0x6a662711,
        0x6ad36709, 0x72b505bb, 0x3f400000, 0x42f00000, 0x4d800000]
    boundaries += [e << 23 for e in (102, 114, 115, 127, 155, 214)]
    bits = np.array(sorted({((b+d)&0xffffffff)^sign
        for b in boundaries for d in (-2,-1,0,1,2)
        for sign in (0,0x80000000)}), np.uint32)
    inputs = np.concatenate((bits,
        np.random.default_rng(412).integers(0,2**32,4096,dtype=np.uint32),
        np.linspace(-1,1,4097,dtype=np.float32).view(np.uint32),
        np.linspace(-4*np.pi,4*np.pi,4097,dtype=np.float32).view(np.uint32)))
    with tempfile.TemporaryDirectory(prefix="woof-metal-trig-") as temp:
        directory = Path(temp)
        shader = directory/'probe.metal'
        shader.write_text('#include <metal_stdlib>\nusing namespace metal;\n'
            '#include "soft_glibc_solar_trig.metalh"\n'
            + ''.join(f'kernel void probe_{name}(device const uint *x [[buffer(0)]], '
                'device uint *out [[buffer(1)]], constant uint &n [[buffer(2)]], '
                f'uint i [[thread_position_in_grid]]) {{ if(i<n) out[i]=soft_glibc_solar_{name}(x[i]); }}\n'
                for name in NAMES))
        subprocess.run(['xcrun','-sdk','macosx','metal','-std=metal3.1',
            '-fmetal-math-mode=safe','-fmetal-math-fp32-functions=precise',
            '-ffp-contract=off','-I',str(ROOT/'woof/metal/shaders'),
            '-c',str(shader),'-o',str(directory/'probe.air')],check=True)
        subprocess.run(['xcrun','-sdk','macosx','metallib',str(directory/'probe.air'),
                        '-o',str(directory/'probe.metallib')],check=True)
        (directory/'run.mm').write_text(RUNNER)
        subprocess.run(['clang++','-std=c++17','-O2','-fobjc-arc',
            '-framework','Foundation','-framework','Metal',str(directory/'run.mm'),
            '-o',str(directory/'run')],check=True)
        inputs.tofile(directory/'input.bin')
        subprocess.run([str(directory/'run'),str(directory/'probe.metallib'),
            str(directory/'input.bin'),str(directory/'output.bin')],check=True)
        received = np.fromfile(directory/'output.bin',dtype=np.uint32).reshape(6,-1)
        reference = directory/'reference.cpp'
        reference.write_text(f'#include "{ROOT / "tools/glibc_trig_flt32_proof/host_shim.hpp"}"\n'
            f'#include "{ROOT / "woof/core/kernels/glibc_trig_flt32.cuh"}"\n'
            + 'extern "C" void reference(int f,const unsigned *in,unsigned *out,int n) {'
            + 'for(int i=0;i<n;++i) { float x=__uint_as_float(in[i]); float y; switch(f) {'
            + ''.join(f'case {i}: y=glibc_{name}(x); break;' for i,name in enumerate(NAMES))
            + 'default: y=0; } out[i]=__float_as_uint(y); }}\n')
        subprocess.run(['clang++','-std=c++17','-O2','-ffp-contract=off','-fno-fast-math',
            '-shared','-fPIC',str(reference),'-o',str(directory/'reference.dylib')],check=True)
        lib = ctypes.CDLL(str(directory/'reference.dylib'))
        lib.reference.argtypes = [ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int]
        records = []
        for i,name in enumerate(NAMES):
            expected = np.empty_like(inputs)
            lib.reference(i,inputs.ctypes.data,expected.ctypes.data,inputs.size)
            bad = np.flatnonzero(expected != received[i])
            records.append({'function':name,'words':int(inputs.size),'mismatches':int(bad.size),
                'examples':[{'input':hex(int(inputs[k])),'expected':hex(int(expected[k])),
                    'actual':hex(int(received[i,k]))} for k in bad[:5]]})
    report = {'reference':'retained pinned source compiled for host',
        'words':sum(r['words'] for r in records),
        'mismatches':sum(r['mismatches'] for r in records),'functions':records}
    print(json.dumps(report,indent=2))
    return bool(report['mismatches'])


if __name__ == '__main__':
    raise SystemExit(main())
