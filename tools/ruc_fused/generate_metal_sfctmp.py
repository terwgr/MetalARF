"""Lower the generated RUC SFCTMP CUDA chain to its packed MSL ABI.

The CUDA file uses one pointer alias per generated array. This source
generator keeps its operation sequence and folds each alias to an offset into
the RUC-owned 32-bit workspace, leaving two Metal resources (workspace and
flag words). CUDA compiles one module per RUC lineage (GPUWM_SNOW_WRF461,
GPUWM_SOILPROP_WRF461); the Metal kernels read the same choice from the
`forms` parameter bits defined in ruc_sfctmp_helpers.metalh.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "woof/core/kernels/ruc_fused_sfctmp.cuh"
OUTPUT = ROOT / "woof/metal/shaders/ruc_sfctmp_family.metal"
LAYOUT_PATH = ROOT / "woof/metal/ruc_metal_layout.py"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("ruc_metal_layout", LAYOUT_PATH)
assert SPEC is not None and SPEC.loader is not None
LAYOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAYOUT)


def _aliases() -> str:
    lines = []
    for index, (pointer_type, offset) in enumerate(LAYOUT.sfctmp_aliases()):
        lines.append(
            f"#define a{index} (reinterpret_cast<device {pointer_type} *>(r.raw + {offset}))")
    return "\n".join(lines)


def _rewrite_atomic_min(text: str) -> str:
    """Drop diagnostic payload minima; the public runtime consumes check bits only."""
    needle = "atomicMin(r.flags + "
    while needle in text:
        start = text.index(needle)
        open_paren = text.index("(", start)
        depth = 0
        comma = None
        end = None
        for position in range(open_paren, len(text)):
            char = text[position]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = position + 1
                    break
            elif char == "," and depth == 1 and comma is None:
                comma = position
        if comma is None or end is None:
            raise RuntimeError("could not parse RUC SFCTMP atomicMin expression")
        # The payload is a 64-bit (cell, offending value) pair. The public
        # runtime reports the corresponding check bit and never reads it.
        text = text[:start] + "/* unused diagnostic payload */" + text[end:]
    return text


def _rewrite_atomic_or(text: str) -> str:
    """Lower uint64 status ORs to two supported 32-bit atomic lanes."""
    pattern = re.compile(
        r"atomicOr\((?:r\.)?flags\s*\+\s*(\d+)\s*,\s*1ull\s*<<\s*(\d+)\s*\)")

    def replace(match):
        word, bit = map(int, match.groups())
        if not 0 <= bit < 64:
            raise RuntimeError(f"unexpected RUC SFCTMP status bit {bit}")
        lane = 2 * word + bit // 32
        return ("atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(r.flags) "
                f"+ {lane}, 1u << {bit % 32}, memory_order_relaxed)")

    text, count = pattern.subn(replace, text)
    if count == 0:
        raise RuntimeError("no RUC SFCTMP status atomic ORs were converted")
    return text


# Helpers whose arithmetic depends on the RUC lineage also take `forms`.
_FORM_HELPERS = ("ruc_soil_properties", "ruc_snow_temperature_step",
                 "ruc_snow_preparation")


def _append_column(text: str) -> str:
    """Pass the packed column extent/index to the extracted RUC helpers."""
    names = (
        "ruc_transpiration", "ruc_soil_temperature_step",
        "ruc_soil_phase_partition", "ruc_soil_prepare_moisture",
        "ruc_soil_moisture_step", "ruc_soil_finalize",
        "ruc_tanhf_glibc_array", "ruc_soil_canopy_setup",
        "ruc_soil_properties", "ruc_sea_ice_step",
        "ruc_snow_preparation", "ruc_snow_sea_ice_step",
        "ruc_snow_temperature_step", "ruc_snow_soil_canopy_setup",
        "ruc_snow_layer_thresholds", "ruc_snow_soil_prepare_moisture",
        "ruc_snow_soil_finalize",
    )
    for name in names:
        needle = name + "("
        start = 0
        while (position := text.find(needle, start)) >= 0:
            open_paren = position + len(name)
            depth = 0
            end = None
            for index in range(open_paren, len(text)):
                if text[index] == "(":
                    depth += 1
                elif text[index] == ")":
                    depth -= 1
                    if depth == 0:
                        end = index
                        break
            if end is None:
                raise RuntimeError(f"unterminated call to {name}")
            suffix = ", c, forms" if name in _FORM_HELPERS else ", c"
            text = text[:end] + suffix + text[end:]
            start = end + len(suffix) + 1
    return text


def _insert_spp_hydraulic_operator(text: str) -> str:
    """Apply the member's hydraulic SPP immediately after stage-0 soilprop."""
    start = text.index("kernel void ruc_sfctmp_stage0")
    end = text.index("kernel void ruc_sfctmp_stage1", start)
    stage0 = text[start:end]
    call_start = stage0.find("ruc_soil_properties(")
    if call_start < 0 or stage0.find("ruc_soil_properties(", call_start + 1) >= 0:
        raise RuntimeError("expected exactly one stage-0 RUC soil-properties call")
    call_end = stage0.find(");", call_start)
    if call_end < 0:
        raise RuntimeError("unterminated stage-0 RUC soil-properties call")
    arguments = stage0[call_start + len("ruc_soil_properties("):call_end].split(",")
    if len(arguments) < 7:
        raise RuntimeError("RUC soil-properties call has no packed hydro output")
    hydro = arguments[-5].strip()
    operator = (
        f" if (p.spp_lsm != 0) {{ for (int k = 0; k < RUC_NZS; ++k) {{ "
        f"const int index = k * n + c; const float original = {hydro}[index]; "
        f"const float pattern = r.pattern_spp_lsm[index]; "
        "if (p.has_field_sf != 0) "
        "r.field_sf[index] = ruc_fmul_rn(original, pattern); "
        f"{hydro}[index] = ruc_fmul_rn(original, ruc_fadd_rn(1.0f, pattern)); "
        "} }"
    )
    call_end += 2  # insert after the call's closing parenthesis and semicolon
    stage0 = stage0[:call_end] + operator + stage0[call_end:]
    return text[:start] + stage0 + text[end:]


def generate() -> str:
    source = SOURCE.read_text(encoding="utf-8")
    aliases = re.sub(r"^#define a\d+ .*\n", "", source, flags=re.MULTILINE)
    source = aliases
    source = source.replace(
        "#define RUC_SFCTMP_FLAG_WORDS 15\n", "")
    source = source.replace(
        "#ifndef NAN\n#define NAN __int_as_float(0x7fc00000)\n#endif\n", "")
    source = source.replace(
        "__device__ __forceinline__", "static inline")
    source = re.sub(
        r"static inline (?:real|float) ruc_fused_(?:min|max)\((?:real|float) a, (?:real|float) b\) \{[^\n]+\}\n",
        "", source)
    source = source.replace("real ", "float ").replace("real*", "float*")
    source = source.replace("__int_as_float(", "as_type<float>(")
    source = source.replace("__fadd_rn(", "ruc_fadd_rn(")
    source = source.replace("__fsub_rn(", "ruc_fsub_rn(")
    source = source.replace("__fmul_rn(", "ruc_fmul_rn(")
    source = source.replace("__fdiv_rn(", "ruc_fdiv_rn(")
    source = source.replace("unsigned long long", "ulong")
    source = source.replace("unsigned int", "uint")
    source = source.replace("ruc_sfctmp_zero_outputs(const ulong* ptrs, int n, int c)",
                            "ruc_sfctmp_zero_outputs(constant RucSfctmpResources& r, constant RucSfctmpParams& p, int n, int c)")
    source = source.replace("ruc_sfctmp_zero_outputs(ptrs, n, c)",
                            "ruc_sfctmp_zero_outputs(r, p, n, c)")
    signature = re.compile(
        r'extern "C" __global__ void ruc_sfctmp_stage([012])\(\s*'
        r'const ulong\* ptrs, const bool\* run, ulong\* flags, bool\* alive,\s*'
        r'float delt, float c1sn, float c2sn, float rsmax, int isice, int urban, '
        r'int ncategory, int limit, int n\) \{')

    def replace_signature(match):
        stage = match.group(1)
        return (
            f"kernel void ruc_sfctmp_stage{stage}(\n"
            "    constant RucSfctmpResources& r [[buffer(0)]],\n"
            "    constant RucSfctmpParams& p [[buffer(1)]],\n"
            "    uint tid [[thread_position_in_grid]]) {\n"
            "    const int n = p.n;\n"
            "    const float delt = p.delt;\n"
            "    const float c1sn = p.c1sn;\n"
            "    const float c2sn = p.c2sn;\n"
            "    const float rsmax = p.rsmax;\n"
            "    const int isice = p.isice;\n"
            "    const int urban = p.urban;\n"
            "    const int ncategory = p.ncategory;\n"
            "    const int limit = p.limit;\n"
            "    const uint forms = uint(p.forms);\n"
            "    const int c = int(tid);")

    source, count = signature.subn(replace_signature, source)
    if count != 3:
        raise RuntimeError(f"expected three SFCTMP kernels, converted {count}")
    source = source.replace("const int c = int(tid);", "const int c = int(tid);")
    source = source.replace("blockIdx.x * blockDim.x + threadIdx.x", "tid")
    source = source.replace("int c = tid;", "")
    source = source.replace("!run[c]", "r.raw[p.run_base + uint(c)] == 0u")
    source = source.replace("run[c]", "r.raw[p.run_base + uint(c)] != 0u")
    source = source.replace("alive[c]", "r.raw[p.alive_base + uint(c)]")
    source = source.replace("flags + ", "r.flags + ")
    source = _rewrite_atomic_or(source)
    source = _rewrite_atomic_min(source)
    source = _append_column(source)
    source = _insert_spp_hydraulic_operator(source)
    source = source.replace("return;", "return;")
    source = source.replace("__device__", "")
    source = source.replace("__forceinline__", "")
    header = '''#include <metal_stdlib>
#include "ruc_sfctmp_family_bindings.metalh"
#include "soft_binary32.metalh"

using namespace metal;
using RucSfctmpResources = RucSfctmpStage0Resources;
using RucSfctmpParams = RucSfctmpStage0Params;
#pragma clang fp contract(off)

static inline float ruc_fadd_rn(float a, float b) { return as_type<float>(soft32_add(as_type<uint>(a), as_type<uint>(b))); }
static inline float ruc_fsub_rn(float a, float b) { return as_type<float>(soft32_sub(as_type<uint>(a), as_type<uint>(b))); }
static inline float ruc_fmul_rn(float a, float b) { return as_type<float>(soft32_mul(as_type<uint>(a), as_type<uint>(b))); }
static inline float ruc_fdiv_rn(float a, float b) { return as_type<float>(soft32_div(as_type<uint>(a), as_type<uint>(b))); }
static inline float ruc_fused_min(float a, float b) { return (isnan(a) || isnan(b)) ? as_type<float>(0x7fc00000u) : min(a,b); }
static inline float ruc_fused_max(float a, float b) { return (isnan(a) || isnan(b)) ? as_type<float>(0x7fc00000u) : max(a,b); }
'''
    macros = '''
#define RUC_NZS 9
#define RUC_SFCTMP_FLAG_WORDS 15
#define RUC_SFCTMP_NZS 9
#define RUC_RUN (reinterpret_cast<device uint *>(r.raw + p.run_base))
#define RUC_ALIVE (reinterpret_cast<device uint *>(r.raw + p.alive_base))
'''
    return (header + macros + '\n#include "ruc_sfctmp_helpers.metalh"\n'
            + _aliases() + "\n" + source)


def main() -> None:
    OUTPUT.write_text(generate(), encoding="utf-8")


if __name__ == "__main__":
    main()
