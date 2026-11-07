"""Compare Metal Thompson vapour fill with the unmodified CUDA source on host.

Run under Anvil after registering the fragment and rebuilding the bundle.
"""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from woof.core.microphysics import _THOMPSON_VAPOUR_FILL_SOURCE, thompson_negative_vapour_fill
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext

PRELUDE = r'''
#include <cmath>
#include <cstring>
#define __global__
#define __restrict__
struct Dim { unsigned x; }; Dim blockIdx, blockDim, threadIdx;
static unsigned __float_as_uint(float x) { unsigned bits; std::memcpy(&bits,&x,4); return bits; }
static float __fadd_rn(float a,float b) { volatile float r=a+b; return r; }
static float __fmul_rn(float a,float b) { volatile float r=a*b; return r; }
'''
WRAPPER = r'''
extern "C" void run(float* q, int nz, long long ncol) {
    blockDim.x=1; threadIdx.x=0;
    for (blockIdx.x=0; blockIdx.x<(unsigned)ncol; ++blockIdx.x)
        thompson_negative_vapour_fill(q,nz,ncol);
}
'''


def main():
    with tempfile.TemporaryDirectory() as tmp:
        source, lib = Path(tmp)/'ref.cpp', Path(tmp)/'ref.dylib'
        source.write_text(PRELUDE + _THOMPSON_VAPOUR_FILL_SOURCE + WRAPPER)
        subprocess.run(['clang++','-std=c++17','-O0','-ffp-contract=off',
                        '-shared','-fPIC',str(source),'-o',str(lib)], check=True)
        ref = ctypes.CDLL(str(lib)).run
        ref.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_longlong]
        ref.restype = None
        rng = np.random.default_rng(34390)
        words = 0
        with MetalContext() as context:
            xp = MetalArrayModule(context)
            for nz in (1, 6, 10):
                shape = (nz, 3, 5)
                data = rng.uniform(-2e-5, 2e-5, shape).astype(np.float32)
                bits = data.view(np.uint32).reshape(nz, -1)
                bits[0, :6] = [0x80000000, 0x80000001, 0xff800000,
                               0xffc00001, 0x7fc00001, 0xbf800000]
                if nz >= 10:
                    bits[2, 0] = 0xbf800000
                    bits[3, 0] = 0xbf800000
                    bits[1, 0] = 0x7fc00001
                    bits[4, 1] = 0xbf800000
                    bits[5, 1] = 0x7f800000
                    bits[6, 1] = 0xbf800000
                    bits[-1, 2] = 0xbf800000
                expected = data.copy()
                ref(expected.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), nz, 15)
                qv = xp.asarray(data)
                thompson_negative_vapour_fill(qv)
                actual = qv.to_numpy()
                np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32),
                                              err_msg=f'nz={nz}')
                words += expected.size
    print(f'0/{words} different words: boundary, NaN, signed zero, subnormal, adjacent repairs')


if __name__ == '__main__':
    main()
