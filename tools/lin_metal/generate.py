#!/usr/bin/env python3
"""Translate the Purdue Lin CUDA column to Metal, per column-storage tier.

    python tools/lin_metal/generate.py

The unit is preprocessed as the loader hands it to nvrtc (one pass per
LIN_KMAX tier) and the definitions lin_column reaches are translated
statement for statement. The soft32 arithmetic helpers come from
woof/metal/shaders/lin_support.metalh.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/mp28_metal"))
sys.path.insert(0, str(ROOT))
from parse import closure, def_name, split_top  # noqa: E402
from translate import convert_function_item, map_types, rewrite_division, tokenize, untok  # noqa: E402
import generate as g  # noqa: E402
from gotos import rewrite_function  # noqa: E402

SHADERS = ROOT / "woof/metal/shaders"
# CUDA's 256 and 512 tiers size only the kernel's own arrays: the headers are
# included before the tier define, so the helpers keep 128 levels and overrun.
TIERS = ((128, "lin_column"),)
PROVIDED = {"lin_add", "lin_sub", "lin_mul", "lin_div", "lin_min", "lin_max",
            "lin_powi", "gfk_exp", "gfk_pow"}
# The wrapper's bare float arithmetic goes through the same soft32 operations.
KERNEL_REWRITES = (
    ("sqrtf(1.29f*orhoz[k])", "sqrtf(lin_mul(1.29f,orhoz[k]))"),
    ("sr[col]=(pptice+pptsnow+pptgraul)/(pptice+pptsnow+pptgraul+pptrain+1.e-12f);",
     "sr[col]=lin_div(lin_add(lin_add(pptice,pptsnow),pptgraul),"
     "lin_add(lin_add(lin_add(lin_add(pptice,pptsnow),pptgraul),pptrain),1.e-12f));"),
    ("rainncv[col]=pptrain+pptsnow+pptgraul+pptice;",
     "rainncv[col]=lin_add(lin_add(lin_add(pptrain,pptsnow),pptgraul),pptice);"),
    ("rainnc[col]=rainnc[col]+pptrain+pptsnow+pptgraul+pptice;",
     "rainnc[col]=lin_add(lin_add(lin_add(lin_add(rainnc[col],pptrain),pptsnow),pptgraul),pptice);"),
    ("snowncv[col]=pptsnow+pptice; snownc[col]=snownc[col]+pptsnow+pptice;",
     "snowncv[col]=lin_add(pptsnow,pptice); snownc[col]=lin_add(lin_add(snownc[col],pptsnow),pptice);"),
    ("graupelnc[col]=graupelnc[col]+pptgraul;",
     "graupelnc[col]=lin_add(graupelnc[col],pptgraul);"),
)


def as_words(text):
    """The unit's binary32 values travel as F, the soft32 word type."""
    return re.sub(r"\bfloat\b", "F", text)


def preprocessed(kmax):
    from woof.core import kernels
    text = kernels.module_source_int_defines("lin", (("LIN_KMAX", kmax),))
    done = subprocess.run(["clang++", "-E", "-P", "-std=c++17", "-x", "c++", "-"],
                          input=text.encode(), capture_output=True, check=True)
    return [i for i in split_top(done.stdout.decode()) if def_name(i) not in PROVIDED]


def translate_tier(kmax, kernel_name):
    items = preprocessed(kmax)
    reached, _ = closure(items, ["lin_column"])
    helpers, kernel = [], None
    for item in reached:
        kind = g.kind_of(item)
        if kind == "kernel":
            kernel = item
        elif kind == "function":
            # Casts that drop const only matter to CUDA; the template parameters
            # carry each pointer's own address space.
            item = re.sub(r"\(\s*(?:int|float)\s*\*\s*\)\s*", "", item)
            helpers.append(as_words(convert_function_item(
                rewrite_function(item) if "goto" in item else item)))
        elif kind == "table":
            helpers.append(as_words(g.convert_table(item)))
    for old, new in KERNEL_REWRITES:
        if old not in kernel:
            raise ValueError("lin.cu changed: " + old)
        kernel = kernel.replace(old, new)
    # The pressure parameter would shadow the Params block in the wrapper.
    kernel = re.sub(r"\bp\b", "pressure", kernel)
    namespace = "lin%d" % kmax
    text, params = g.convert_kernel(kernel, kernel_name, namespace, [])
    text = text.replace("lin_column(", kernel_name + "(", 1)
    text = re.sub(r"device (const )?float \*(\w+) = r\.(\w+);",
                  r"device \1F *\2 = (device \1F *)r.\3;", text)
    text = re.sub(r"const float (\w+) = p\.(\w+);", r"F \1 = F(p.\2);", text)
    text = as_words(text)
    return namespace, helpers, text, params


def signature(name, params):
    resources, shapes, parameters = [], {}, []
    for p in params:
        if p["pointer"]:
            resources.append([p["name"], "float", "read" if p["const"] else "read_write"])
            shapes[p["name"]] = [["any_1d"]]
        else:
            parameters.append([p["name"], "float32" if p["base"] == "float" else "int32"])
    return {"name": name, "cuda_symbol": "lin_column", "module": "lin_family",
            "source": "woof/metal/shaders/lin_family.metal",
            "cuda_source": "woof/core/kernels/lin.cu", "resources": resources,
            "shapes": shapes, "parameters": parameters, "threads_per_group": 32,
            "layout_version": 1, "arithmetic_policies": ["soft32-rn-nonftz"]}


def main():
    sections, kernels, signatures = [], [], []
    for kmax, name in TIERS:
        namespace, helpers, text, params = translate_tier(kmax, name)
        sections.append("namespace %s {\n%s\n}" % (namespace, "\n".join(helpers)))
        kernels.append(text)
        signatures.append(signature(name, params))
    (SHADERS / "lin_fragment.metalh").write_text(
        "// Generated from woof/core/kernels/lin_wrf.cuh by tools/lin_metal/generate.py; do not edit.\n"
        '#include "lin_support.metalh"\n#pragma clang fp contract(off)\n'
        + "\n".join(sections) + "\n")
    (SHADERS / "lin_family.metal").write_text(
        "// Generated by tools/lin_metal/generate.py from the CUDA source; do not edit.\n"
        '#include <metal_stdlib>\n#include "lin_fragment.metalh"\n'
        '#include "lin_family_bindings.metalh"\n\n' + "\n\n".join(kernels) + "\n")
    (ROOT / "woof/metal/signature_fragments/lin_family.json").write_text(
        json.dumps({"kernels": signatures}, indent=2) + "\n")


if __name__ == "__main__":
    main()
