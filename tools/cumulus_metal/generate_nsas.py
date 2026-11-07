#!/usr/bin/env python3
"""Translate the CUDA New SAS unit (cu_physics=96) to a Metal kernel.

Statements are carried over unchanged: address spaces, the float division
and square root helpers and the kernel boundary change and nothing else.
Run from the repository root with the data companion on PYTHONPATH.
"""
from pathlib import Path
import json
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/mp28_metal"))
sys.path.insert(0, str(ROOT))
from parse import split_top, def_name  # noqa: E402
from translate import (tokenize, untok, map_types, rewrite_division,  # noqa: E402
                       convert_table)

SHADERS = ROOT / "woof/metal/shaders"
KEEP = ("NSAS_PCRIT_TABLE", "NSAS_ACRITT_TABLE", "S_DEL", "static_assert",
        "NsasPhys", "nsas_phys", "nsas_fpvs", "nsas2d", "nscv2d")
RESOURCES = (
    ("t3d", "float", "read"), ("qv3d", "float", "read"),
    ("qc3d", "float", "read"), ("qi3d", "float", "read"),
    ("p3d", "float", "read"), ("pi3d", "float", "read"),
    ("rho3d", "float", "read"), ("dz8w", "float", "read"),
    ("u3d", "float", "read"), ("v3d", "float", "read"),
    ("p3di", "float", "read"), ("w", "float", "read"),
    ("sc", "float", "read"), ("ic", "int32", "read"),
    ("rthcuten", "float", "read_write"), ("rqvcuten", "float", "read_write"),
    ("rqccuten", "float", "read_write"), ("rqicuten", "float", "read_write"),
    ("rucuten", "float", "read_write"), ("rvcuten", "float", "read_write"),
    ("sfc", "float", "read_write"), ("ws", "float", "read_write"),
    ("row_limits", "int32", "read"))
COLUMN_PARAMS = ("col0", "ntile", "ncol", "nz", "row_width", "has_qc", "has_qi")


def preprocess():
    from woof.core import kernels
    result = subprocess.run(
        ["clang++", "-E", "-P", "-std=c++17", "-x", "c++", "-"],
        input=kernels.module_source("nsas").encode(),
        capture_output=True, check=True)
    return split_top(result.stdout.decode())


def translate(text):
    text = re.sub(r"\bconst NsasCol&", "const thread NsasCol &", text)
    text = re.sub(r"\bconst NsasPhys&", "const thread NsasPhys &", text)
    text = re.sub(r"\b(float|int)\* (rain|kbot|ktop|icps)\b", r"thread \1 *\2", text)
    text = text.replace("const int* row_limits = nullptr",
                        "device const int *row_limits = nullptr")
    text = re.sub(r"const float\* (pcrit|acritt) =", r"constant const float *\1 =", text)
    head, brace, body = text.partition("{")
    body = re.sub(r"\*\s*(rain|kbot|ktop|icps)\b", r"\1[0]", body)
    text = head + brace + body
    text = text.replace("sqrtf(", "cu_sqrt(").replace("fmaxf(", "fmax(").replace(
        "fminf(", "fmin(")
    return untok(rewrite_division(map_types(tokenize(text))))


def remove_block(text, opener):
    start = text.index(opener)
    depth = 0
    for j in range(text.index("{", start), len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            return text[:start] + text[j + 1:]
    raise ValueError(opener)


def column_kernel(items):
    src = next(i for i in items if def_name(i) == "nsas_column")
    body = src[src.index("{", src.index(")")) + 1:src.rindex("}")]
    body = remove_block(body, "if (prep != nullptr)")
    body = body.replace("row_limits == nullptr ? nullptr : row_limits + 3*(i/row_width)",
                        "row_limits + 3*(i/row_width)")
    body = body.replace("if (rqicuten != nullptr)", "if (has_qi)")
    body = body.replace("if (rqccuten != nullptr)", "if (has_qc)")
    body = re.sub(r"const int t = blockIdx\.x \* blockDim\.x \+ threadIdx\.x;",
                  "const int t = int(gid);", body)
    head = ["kernel void nsas_column(",
            "    constant NsasColumnResources &r [[buffer(0)]],",
            "    constant NsasColumnParams &p [[buffer(1)]],",
            "    uint gid [[thread_position_in_grid]])", "{"]
    for name, kind, access in RESOURCES:
        ctype = "int" if kind == "int32" else "float"
        const = "const " if access == "read" else ""
        head.append(f"    device {const}{ctype} *{name} = r.{name};")
    for name in COLUMN_PARAMS:
        head.append(f"    const int {name} = p.{name};")
    return "\n".join(head) + "\n" + untok(rewrite_division(map_types(
        tokenize(translate_kernel_body(body))))) + "\n}\n"


def translate_kernel_body(body):
    return body.replace("sqrtf(", "cu_sqrt(").replace("fmaxf(", "fmax(").replace(
        "fminf(", "fmin(")


def row_limits_kernel(items):
    src = next(i for i in items if def_name(i) == "nsas_row_limits")
    body = src[src.index("{", src.index(")")) + 1:src.rindex("}")]
    body = body.replace("const int row = blockIdx.x * blockDim.x + threadIdx.x;",
                        "const int row = int(gid);")
    head = ("kernel void nsas_row_limits(\n"
            "    constant NsasRowLimitsResources &r [[buffer(0)]],\n"
            "    constant NsasRowLimitsParams &p [[buffer(1)]],\n"
            "    uint gid [[thread_position_in_grid]])\n{\n"
            "    device const float *p_in = r.p;\n"
            "    device const float *p8w = r.p8w;\n"
            "    device int *limits = r.limits;\n"
            "    const int ncol = p.ncol;\n    const int nz = p.nz;\n"
            "    const int row_width = p.row_width;\n")
    body = re.sub(r"\bp\[", "p_in[", body)
    return head + untok(rewrite_division(map_types(tokenize(body)))) + "\n}\n"


HELPERS = r"""
kernel void nsas_pack_columns(
    constant NsasPackColumnsResources &r [[buffer(0)]],
    constant NsasPackColumnsParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= p.ncol) return;
    const ulong i = ulong(gid);
    r.sc[i * 7 + 0] = r.xland[i];
    r.sc[i * 7 + 1] = r.hpbl[i];
    r.sc[i * 7 + 2] = r.hfx[i];
    r.sc[i * 7 + 3] = r.qfx[i];
    r.sc[i * 7 + 4] = p.dx;
    r.sc[i * 7 + 5] = p.dt;
    r.sc[i * 7 + 6] = p.pgcon;
    r.ic[i * 3 + 0] = p.stepcu;
    r.ic[i * 3 + 1] = p.ncloud;
    r.ic[i * 3 + 2] = p.dx_factor;
}

kernel void nsas_clock_rain(
    constant NsasClockRainResources &r [[buffer(0)]],
    constant NsasClockRainParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= p.ncol) return;
    r.raincv[gid] = r.pratec[gid] * p.dt;
}
"""


def signatures():
    common = dict(module="nsas_family", source="woof/metal/shaders/nsas_family.metal",
                  cuda_source="woof/core/kernels/nsas.cu", layout_version=1,
                  arithmetic_policies=["soft32-rn-nonftz"])
    column = dict(
        name="nsas_column", cuda_symbol="nsas_column",
        resources=[list(r) for r in RESOURCES],
        shapes={r[0]: [["any_1d"]] for r in RESOURCES},
        parameters=[[n, "int32"] for n in COLUMN_PARAMS],
        threads_per_group=64, **common)
    rows = dict(
        name="nsas_row_limits", cuda_symbol="nsas_row_limits",
        resources=[["p", "float", "read"], ["p8w", "float", "read"],
                   ["limits", "int32", "read_write"]],
        shapes={"p": [["any_1d"]], "p8w": [["any_1d"]], "limits": [["any_1d"]]},
        parameters=[["ncol", "int32"], ["nz", "int32"], ["row_width", "int32"]],
        threads_per_group=64, **common)
    pack = dict(
        name="nsas_pack_columns", cuda_symbol="nsas_column",
        resources=[["xland", "float", "read"], ["hpbl", "float", "read"],
                   ["hfx", "float", "read"], ["qfx", "float", "read"],
                   ["sc", "float", "read_write"], ["ic", "int32", "read_write"]],
        shapes={n: [["any_1d"]] for n in ("xland", "hpbl", "hfx", "qfx", "sc", "ic")},
        parameters=[["dx", "float32"], ["dt", "float32"], ["pgcon", "float32"],
                    ["stepcu", "int32"], ["ncloud", "int32"],
                    ["dx_factor", "int32"], ["ncol", "int32"]],
        threads_per_group=64, **common)
    rain = dict(
        name="nsas_clock_rain", cuda_symbol="nsas_column",
        resources=[["pratec", "float", "read"], ["raincv", "float", "read_write"]],
        shapes={"pratec": [["any_1d"]], "raincv": [["any_1d"]]},
        parameters=[["dt", "float32"], ["ncol", "int32"]],
        threads_per_group=64, **common)
    return [column, rows, pack, rain]


def main():
    items = preprocess()
    out = []
    for item in items:
        name = def_name(item)
        if name not in KEEP:
            continue
        if item.startswith("static_assert") or item.startswith("enum"):
            out.append(item)
        elif "__constant__" in item:
            out.append(convert_table(item))
        elif item.startswith("struct NsasPhys") or item.startswith("enum"):
            out.append(item)
        else:
            out.append(translate(item))
    text = ("// Generated by tools/cumulus_metal/generate_nsas.py from nsas.cu; "
            "do not edit.\n"
            '#pragma clang diagnostic ignored "-Wunused-variable"\n'
            '#pragma clang diagnostic ignored "-Wparentheses-equality"\n'
            + "\n".join(out) + "\n")
    (SHADERS / "nsas_fragment.metalh").write_text(text)
    (SHADERS / "nsas_family.metal").write_text(
        "// Generated by tools/cumulus_metal/generate_nsas.py; do not edit.\n"
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        '#include "nsas_family_bindings.metalh"\n'
        '#include "nsas_support.metalh"\n'
        '#include "nsas_fragment.metalh"\n'
        + row_limits_kernel(items) + column_kernel(items) + HELPERS)
    (ROOT / "woof/metal/signature_fragments/nsas_family.json").write_text(
        json.dumps({"kernels": signatures()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
