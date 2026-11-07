#!/usr/bin/env python3
"""Translate the pinned CCPP TKE-EDMF CUDA unit (satmedmf.cu) to Metal.

    python tools/satmedmf_metal/generate.py

The unit is preprocessed as the loader hands it to nvrtc. Its column functions
and its three kernels are translated statement for statement: binary64 values
are Soft8 on the soft64 helpers, the workspace views and the sm_* vocabulary
come from woof/metal/shaders/satmedmf_support.metalh, and the GFS constants
are folded here from gfs_physcons.cuh.
"""
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/mp28_metal"))
sys.path.insert(0, str(ROOT))
from parse import def_name, split_top  # noqa: E402
from translate import map_types, match_fwd, split_params, tokenize, untok  # noqa: E402

SHADERS = ROOT / "woof/metal/shaders"
KERNELS = ("satmedmf_fpvs_table", "satmedmf_exner", "satmedmf_column")
POLICIES = ["soft32-rn-nonftz", "soft64-rn-bit-ops", "portable-libm64"]


def preprocessed():
    from woof.core import kernels
    done = subprocess.run(["clang++", "-E", "-P", "-std=c++17", "-x", "c++", "-"],
                          input=kernels.module_source("satmedmf").encode(),
                          capture_output=True, check=True)
    return split_top(done.stdout.decode())


def constants(items):
    """The sm_* aliases of the GFS physical constants, as binary64 words."""
    values = {}
    source = (ROOT / "woof/core/kernels/gfs_physcons.cuh").read_text()
    for line in re.findall(r"constexpr double ([^;]*);", source):
        for name, expression in (part.split("=") for part in line.split(",")):
            values[name.strip()] = float(eval(expression, {}, dict(values)))
    out = []
    for item in items:
        if item.startswith("constexpr double sm_"):
            for alias, target in re.findall(r"(sm_\w+)=gfs_physcons::(\w+)", item):
                bits = struct.unpack("<Q", struct.pack("<d", values[target]))[0]
                out.append("constant Soft8 %s = SD8(0x%016xul);" % (alias, bits))
    return out


def int_literals(text):
    """Integer variables the source initialises with a binary64 literal."""
    names = set()
    for decl in re.findall(r"\bint\s+([^;(]+);", text):
        names.update(n.strip() for n in decl.split(","))

    def repl(m):
        if m[1] not in names:
            return m[0]
        value = float.fromhex(m[2].rstrip("f"))
        if value != int(value):
            raise ValueError(m[0])
        return "%s = %d;" % (m[1], int(value))
    return re.sub(r"\b(\w+)\s*=\s*(0x[0-9a-fA-F.]+p[+-]?\d+f?)\s*;", repl, text)


def convert_function(item):
    toks = map_types(tokenize(int_literals(item)))
    first = next(k for k, t in enumerate(toks) if t[1] in ("{", ";"))
    paren = next(k for k, t in enumerate(toks[:first]) if t[1] == "(")
    close = match_fwd(toks, paren)
    params = []
    for p in split_params(toks, paren, close):
        words = [t for _, t in p]
        if "*" in words:
            p = [("id", "thread" if words[0] == "char" else "device")] + p
        elif "&" in words:
            p = [("id", "thread")] + p
        params.append(p)
    mid = []
    for k, p in enumerate(params):
        if k:
            mid.append(("op", ","))
        mid.extend(p)
    out = [("id", "static"), ("id", "inline")] + toks[:paren + 1] + mid + toks[close:]
    text = untok(out)
    # Workspace pointers keep the device address space.
    return re.sub(r"(?m)^(Soft8|int|bool) \* (\w+) = ws", r"device \1 * \2 = ws", text)


def kernel_info(item):
    toks = tokenize(item)
    paren = next(k for k, t in enumerate(toks) if t[1] == "(")
    close = match_fwd(toks, paren)
    params = []
    for p in split_params(toks, paren, close):
        words = [t for _, t in p if t != "const"]
        params.append({"name": words[-1], "pointer": "*" in words,
                       "base": words[0], "const": any(t == "const" for _, t in p)})
    return toks[paren - 1][1], params, untok(toks[close + 1:])


def convert_kernel(item):
    name, params, body = kernel_info(item)
    body = re.sub(r"blockIdx\s*\.\s*x\s*\*\s*blockDim\s*\.\s*x\s*\+\s*threadIdx\s*\.\s*x",
                  "int(gid)", body)
    # Scratch strides are binary64 words; the Metal scratch is typed to match.
    body = untok(map_types(tokenize(body)))
    title = "".join(w.title() for w in name.split("_"))
    lines = ["kernel void %s(" % name,
             "    constant %sResources &r [[buffer(0)]]," % title,
             "    constant %sParams &params [[buffer(1)]]," % title,
             "    uint gid [[thread_position_in_grid]]) {"]
    resources, parameters = [], []
    for p in params:
        n, base = p["name"], p["base"]
        if p["pointer"]:
            cv = "const " if p["const"] else ""
            if base == "double":
                lines.append("    device %sSoft8 *%s = (device %sSoft8 *)r.%s;" % (cv, n, cv, n))
                resources.append([n, "float64_bits", "read" if p["const"] else "read_write"])
            else:
                lines.append("    device %s%s *%s = r.%s;" % (cv, base, n, n))
                resources.append([n, "int32", "read" if p["const"] else "read_write"])
        elif base == "double":
            lines.append("    const Soft8 %s = Soft8::fromBits(r.%s_bits[0]);" % (n, n))
            resources.append([n + "_bits", "float64_bits", "read"])
        else:
            lines.append("    const int %s = params.%s;" % (n, n))
            parameters.append([n, "int32"])
    if not parameters:
        # A Metal parameter block cannot be empty; the launch passes its thread count.
        lines.append("    (void)params.threads;")
        parameters.append(["threads", "int32"])
    lines.append(body)
    lines.append("}")
    shapes = {r[0]: [["any_1d"]] for r in resources}
    signature = {"name": name, "cuda_symbol": name, "module": "satmedmf_family",
                 "source": "woof/metal/shaders/satmedmf_family.metal",
                 "cuda_source": "woof/core/kernels/satmedmf.cu", "resources": resources,
                 "shapes": shapes, "parameters": parameters, "threads_per_group": 32,
                 "layout_version": 1, "arithmetic_policies": POLICIES}
    return "\n".join(lines), signature


def main():
    items = preprocessed()
    functions, kernels, signatures, declarations = [], [], [], []
    for item in items:
        name = def_name(item)
        if name in KERNELS:
            text, signature = convert_kernel(item)
            kernels.append((KERNELS.index(name), text))
            signatures.append((KERNELS.index(name), signature))
        elif item.startswith("__device__ void sm_"):
            (functions if "{" in item else declarations).append(convert_function(item))
    header = "// Generated by tools/satmedmf_metal/generate.py; do not edit.\n"
    (SHADERS / "satmedmf_fragment.metalh").write_text(
        header + '#include "satmedmf_support.metalh"\n#pragma clang fp contract(off)\n'
        + "\n".join(constants(items)) + "\n" + "\n".join(declarations)
        + "\n" + "\n".join(functions) + "\n")
    (SHADERS / "satmedmf_family.metal").write_text(
        header + '#include <metal_stdlib>\n#include "satmedmf_fragment.metalh"\n'
        '#include "satmedmf_family_bindings.metalh"\n\n'
        + "\n\n".join(t for _, t in sorted(kernels)) + "\n")
    (ROOT / "woof/metal/signature_fragments/satmedmf_family.json").write_text(
        json.dumps({"kernels": [s for _, s in sorted(signatures)]}, indent=2) + "\n")


if __name__ == "__main__":
    main()
