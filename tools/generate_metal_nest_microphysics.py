"""Lower the column-local nest microphysics closures to Metal resources."""

from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "woof/core/kernels/nest_microphysics.cu"
SHADER = ROOT / "woof/metal/shaders/nest_microphysics_family.metal"
FRAGMENT = ROOT / "woof/metal/signature_fragments/nest_microphysics_family.json"


def generate():
    source = SOURCE.read_text()
    kernels = []

    def entry(match):
        symbol, signature = match.groups()
        name = "nest_" + symbol
        title = "".join(part.title() for part in name.split("_"))
        signature = re.sub(r"//[^\n]*", "", signature)
        declarations, resources, parameters = [], [], []
        for argument in signature.split(","):
            argument = " ".join(argument.split()).replace(" __restrict__", "")
            field = argument.split()[-1]
            if "*" in argument:
                access = "read" if argument.startswith("const ") else "read_write"
                resources.append([field, "float", access])
                declarations.append(f"    device {argument} = r.{field};")
            else:
                dtype, field = argument.split()
                parameters.append([field, "float32" if dtype == "float" else "int32"])
                declarations.append(f"    const {dtype} {field} = p.{field};")
        kernels.append({
            "name": name, "module": "nest_microphysics_family",
            "source": str(SHADER.relative_to(ROOT)),
            "cuda_source": str(SOURCE.relative_to(ROOT)), "cuda_symbol": symbol,
            "resources": resources,
            "shapes": {field: [["any_1d"]] for field, _, _ in resources},
            "parameters": parameters, "threads_per_group": 128,
            "layout_version": 1, "arithmetic_policies": ["default"],
        })
        return (f"kernel void {name}(\n"
                f"    constant {title}Resources &r [[buffer(0)]],\n"
                f"    constant {title}Params &p [[buffer(1)]],\n"
                "    uint tid [[thread_position_in_grid]])\n{\n"
                + "\n".join(declarations))

    source, count = re.subn(
        r'extern "C" __global__\s+void (\w+)\((.*?)\)\s*\{',
        entry, source, flags=re.DOTALL)
    if count != 2:
        raise ValueError("expected both nest microphysics entry points")
    source = source.replace("static __device__ __forceinline__", "static inline")
    source = source.replace("const float* __restrict__", "device const float*")
    source = re.sub(r"(?<!device )const float\*", "device const float*", source)
    source = source.replace("const NsslTransitionState&", "thread const NsslTransitionState&")
    source = source.replace("float* qi_rim", "thread float* qi_rim")
    source = source.replace("float* bi_rim", "thread float* bi_rim")
    source = source.replace("float* qi, float* qs, float* qg", "thread float* qi, thread float* qs, thread float* qg")
    source = source.replace("blockDim.x * blockIdx.x + threadIdx.x", "int(tid)")
    source = re.sub(r"\bdouble\b", "Nssl2Soft64", source)
    source = re.sub(r"(?<![\w.])(1\.0|0\.25)(?![\w.])", r"\1f", source)
    for operation in ("add", "sub", "mul", "div"):
        source = source.replace(f"__f{operation}_rn", f"nest_edge_f{operation}")
    source = source.replace("fminf(", "fmin(").replace("fmaxf(", "fmax(")
    source = source.replace("powf(", "nest_edge_powf(").replace("expf(", "nest_edge_expf(")
    source = source.replace("cbrtf(", "nest_edge_cbrtf_positive(")
    header = '''#include <metal_stdlib>
#include "nssl2_soft64.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_pow.metalh"
#include "soft_gfk_exp.metalh"
using namespace metal;
#pragma clang fp contract(off)

static inline float nest_edge_fadd(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float nest_edge_fsub(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float nest_edge_fmul(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float nest_edge_fdiv(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float nest_edge_powf(float a, float b) { return as_type<float>(soft_gfk_pow_bits(as_type<uint>(a), as_type<uint>(b))); }
static inline float nest_edge_expf(float a) { return as_type<float>(soft_gfk_exp(as_type<uint>(a))); }
// The ice-number closure supplies a nonnegative argument. Refine the seed
// in software binary64, then round the cube root once to binary32.
static inline float nest_edge_cbrtf_positive(float value) {
    if (value == 0.0f || !isfinite(value)) return value;
    Nssl2Soft64 x(value), root(nest_edge_powf(value, 1.0f / 3.0f));
    for (int iteration = 0; iteration < 4; ++iteration)
        root = (2.0f * root + x / (root * root)) / 3.0f;
    return float(root);
}
'''
    includes = "".join(f'#include "{kernel["name"]}_bindings.metalh"\n'
                       for kernel in kernels)
    SHADER.write_text(includes + header + "\n" + source)
    FRAGMENT.write_text(json.dumps({"kernels": kernels}, indent=2) + "\n")


if __name__ == "__main__":
    generate()
