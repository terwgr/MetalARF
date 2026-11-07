#!/usr/bin/env python3
"""Generate the mp=28 source-network Metal shaders and their signatures from
the CUDA translation units, for both Thompson generations.

    python tools/mp28_metal/generate.py

The CUDA units are preprocessed exactly as the loader hands them to nvrtc
(wrf_461) or with THOMPSON_AA_WRF39 defined (wrf_39_noaa); the definitions
each kernel reaches are translated statement for statement.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from parse import closure, def_name, split_top  # noqa: E402
from translate import (convert_function_item, convert_struct,  # noqa: E402
                       convert_table, map_types, rewrite_division,
                       split_params, template_pointer_params, tokenize, untok)

SHADERS = ROOT / "woof/metal/shaders"
FRAGMENTS = ROOT / "woof/metal/signature_fragments"

PROVIDED = {"glibc_exp", "glibc_log", "glibc_pow"}

#: stem -> [(cuda unit, kernel, versions, threads)]
FAMILIES = {
    "cold": ("thompson_aerosol_cold", [
        ("thompson_aa_cold_network", ("461", "39"), 32)]),
    "warm": ("thompson_aerosol_warm", [
        ("thompson_aa_warm_source_network", ("461", "39"), 32),
        ("thompson_aa_ncten_balance", ("461",), 32)]),
    "sat": ("thompson_aerosol_sat", [
        ("thompson_aa_saturation_adjust", ("461",), 32),
        ("thompson_aa_rain_evaporation", ("461",), 32)]),
}
POLICIES = ["soft32-rn-nonftz", "soft64-rn-bit-ops", "portable-libm64"]


def preprocessed(unit, version):
    from woof.core import kernels
    if version == "39":
        text = kernels.module_source_int_defines(
            unit, (("THOMPSON_AA_WRF39", 1),))
    else:
        text = kernels.module_source(unit)
    done = subprocess.run(
        ["clang++", "-E", "-P", "-std=c++17", "-x", "c++", "-"],
        input=text.encode(), capture_output=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.decode()[-3000:])
    text = done.stdout.decode()
    # The CUDA overloads differ only in their pointer pointee type. Pointer
    # address-space templates erase that distinction, so retain a distinct
    # name for the saturation helper returning the distribution slope.
    if unit == "thompson_aerosol_sat":
        text = re.sub(r"(float\s+)thompson_aa_bound_rain_number(\s*\(\s*float\s+rr)",
                      r"\1thompson_aa_bound_rain_distribution\2", text)
        text = re.sub(r"(const float nr_work\s*=\s*)thompson_aa_bound_rain_number",
                      r"\1thompson_aa_bound_rain_distribution", text)
    # The shared CUDA helper carries the same range reduction and binary64
    # polynomial needed by the source network's plain atan call.
    text = re.sub(r"\batan\s*\(", "thompson_aa_atan_reduced(", text)
    return text


def items_of(unit, version):
    items = split_top(preprocessed(unit, version))
    keep = []
    for it in items:
        if (it.startswith("namespace") or it.startswith("typedef")
                or it.startswith("#pragma")):
            continue
        if def_name(it) in PROVIDED:
            continue
        keep.append(it)
    return keep


def typed_nulls(text, items):
    """Keep a pointee type when a null argument reaches a pointer template."""
    from translate import match_fwd
    declarations = {}
    for item in items:
        head = item.split("{", 1)[0]
        if "(" not in head:
            continue
        toks = tokenize(head)
        op = next(i for i, t in enumerate(toks) if t[1] == "(")
        cl = match_fwd(toks, op)
        declarations[toks[op - 1][1]] = split_params(toks, op, cl)
    toks = tokenize(text)
    for i in range(len(toks) - 2, -1, -1):
        if toks[i][1] not in declarations or toks[i + 1][1] != "(":
            continue
        cl = match_fwd(toks, i + 1)
        args = split_params(toks, i + 1, cl)
        params = declarations[toks[i][1]]
        if len(args) != len(params):
            continue
        changed = False
        for j, (arg, prm) in enumerate(zip(args, params)):
            if [t[1] for t in arg] != ["nullptr"] or not any(t[1] == "*" for t in prm):
                continue
            base = [t for t in prm[:-1] if t[1] not in ("__restrict__",)]
            args[j] = tokenize("(thread ") + base + tokenize(")nullptr")
            changed = True
        if changed:
            middle = []
            for j, arg in enumerate(args):
                if j:
                    middle.append(("op", ","))
                middle.extend(arg)
            toks[i + 2:cl] = middle
    return untok(toks)


def kind_of(item):
    if item.startswith('extern "C"'):
        return "kernel"
    if item.startswith("struct"):
        return "struct"
    head = item.split("{")[0].split("=")[0]
    if "(" in head:
        return "function"
    return "table"


def kernel_info(item):
    toks = tokenize(item)
    paren = next(k for k, t in enumerate(toks) if t[1] == "(")
    from translate import match_fwd
    close = match_fwd(toks, paren)
    name = toks[paren - 1][1]
    params = []
    for p in split_params(toks, paren, close):
        words = [t for _, t in p if t not in ("__restrict__", "const")]
        pointer = "*" in words
        pname = words[-1]
        base = " ".join(w for w in words[:-1] if w != "*")
        params.append({"name": pname, "pointer": pointer, "base": base,
                       "const": any(t == "const" for _, t in p)})
    return name, params, toks[close + 1:]


def convert_kernel(item, metal_name, namespace, optional):
    name, params, body_toks = kernel_info(item)
    body_text = untok(body_toks)
    body_text = re.sub(
        r"blockIdx\s*\.\s*x\s*\*\s*blockDim\s*\.\s*x\s*\+\s*threadIdx\s*\.\s*x"
        r"|blockDim\s*\.\s*x\s*\*\s*blockIdx\s*\.\s*x\s*\+\s*threadIdx\s*\.\s*x",
        "int ( gid )", body_text)
    body = rewrite_division(map_types(tokenize(body_text)))
    body_text = untok(body)
    title = "".join(p.title() for p in metal_name.split("_"))
    lines = ["kernel void %s(" % metal_name,
             "    constant %sResources &r [[buffer(0)]]," % title,
             "    constant %sParams &p [[buffer(1)]]," % title,
             "    uint gid [[thread_position_in_grid]]) {",
             "    using namespace %s;" % namespace]
    for prm in params:
        if prm["pointer"]:
            cv = "const " if prm["const"] else ""
            if prm["base"] == "double":
                ptr = "device const Soft8 *%s = (device const Soft8 *)r.%s;" % (
                    prm["name"], prm["name"])
            else:
                ptr = "device %sfloat *%s = r.%s;" % (cv, prm["name"], prm["name"])
            if prm["name"] in optional:
                typ, rest = ptr.split(" *%s = " % prm["name"])
                ptr = "%s *%s = p.has_%s != 0 ? %s : nullptr;" % (
                    typ, prm["name"], prm["name"], rest.rstrip(";"))
            lines.append("    " + ptr)
        else:
            lines.append("    const %s %s = p.%s;" % (
                "Soft8" if prm["base"] == "double" else prm["base"],
                prm["name"], prm["name"]))
    lines.append(body_text)
    lines.append("}")
    return "\n".join(lines), params


def optional_pointers(items, kernel_params):
    text = "\n".join(items)
    names = set(re.findall(r"(\w+)\s*(?:!=|==)\s*nullptr", text))
    # Several optional arguments are tested by an aliased helper parameter
    # or by the presence of a paired accumulator. Preserve nullness for all
    # optional source inputs at the public boundary.
    names.update(("qcten", "qrten", "nrten", "qiten", "niten", "qvten",
                  "tten", "qsten", "qgten", "ngten", "theta", "exner",
                  "tau1_temperature"))
    return [p["name"] for p in kernel_params if p["pointer"] and p["name"] in names]


def translate_family(stem):
    unit, kernels = FAMILIES[stem]
    fragments = {}
    kernel_texts = []
    signatures = []
    for kname, versions, threads in kernels:
        for version in versions:
            items = items_of(unit, version)
            reached, _ = closure(items, [kname])
            reached = [typed_nulls(it, reached) for it in reached]
            ns = "aa" + version
            frag_items = []
            kernel_item = None
            for it in reached:
                kind = kind_of(it)
                if kind == "kernel":
                    if def_name(it) == kname:
                        kernel_item = it
                    continue
                if kind == "struct":
                    frag_items.append(convert_struct(it))
                elif kind == "table":
                    frag_items.append(convert_table(it))
                else:
                    text = convert_function_item(it)
                    frag_items.append(text)
            if kernel_item is None:
                raise RuntimeError(kname)
            _, params, _ = kernel_info(kernel_item)
            optional = optional_pointers(reached, params)
            metal_name = kname if version == "461" else kname + "_wrf39"
            ktext, params = convert_kernel(kernel_item, metal_name, ns, optional)
            frag_key = (stem, version)
            fragments.setdefault(frag_key, [])
            for t in frag_items:
                if t not in fragments[frag_key]:
                    fragments[frag_key].append(t)
            kernel_texts.append(ktext)
            signatures.append(signature(stem, kname, metal_name, params,
                                        optional, threads))
    return fragments, kernel_texts, signatures


def signature(stem, cuda_name, metal_name, params, optional, threads):
    size_name = next(p["name"] for p in params if not p["pointer"]
                     and p["name"] in ("size", "n"))
    resources, shapes, parameters = [], {}, []
    for p in params:
        if p["pointer"]:
            storage = "float64_bits" if p["base"] == "double" else "float"
            access = "read" if p["const"] else "read_write"
            resources.append([p["name"], storage, access])
            shapes[p["name"]] = ([["any_1d"]] if p["base"] == "double"
                                 else [[size_name]])
        else:
            parameters.append([p["name"], "float32" if p["base"] == "float" else "int32"])
    for name in optional:
        parameters.append(["has_" + name, "int32"])
    return {
        "name": metal_name,
        "cuda_symbol": cuda_name,
        "nullable_resources": list(optional),
        "shapes": shapes,
        "parameters": parameters,
        "module": "thompson_aerosol_%s_family" % stem,
        "source": "woof/metal/shaders/thompson_aerosol_%s_family.metal" % stem,
        "cuda_source": "woof/core/kernels/thompson_aerosol_%s.cu" % stem,
        "threads_per_group": threads,
        "layout_version": 1,
        "arithmetic_policies": POLICIES,
        "optional_flag_parameters": ["has_" + n for n in optional],
        "nullable_binding_policy": (
            "Bind a harmless same-context dummy resource for each absent "
            "optional buffer and set its int32 presence flag to zero."),
        "resources": resources,
    }


def fold_table_initializer(text):
    """Evaluate scalar constant expressions before wrapping binary64 words."""
    import ast
    import operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.LShift: operator.lshift,
           ast.RShift: operator.rshift, ast.BitOr: operator.or_,
           ast.BitAnd: operator.and_, ast.BitXor: operator.xor}
    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp):
            v = value(node.operand)
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BinOp):
            a, b = value(node.left), value(node.right)
            if isinstance(node.op, ast.Div):
                return int(a / b) if isinstance(a, int) and isinstance(b, int) else a / b
            return ops[type(node.op)](a, b)
        raise ValueError(ast.dump(node))
    before, initializer = text.split("=", 1)
    result = []
    for chunk in re.split(r"([{},;])", initializer):
        if not chunk.strip() or chunk in "{},;":
            result.append(chunk)
            continue
        toks = tokenize(chunk)
        if not any(k == "num" and ("p" in s.lower() or "." in s) for k, s in toks):
            result.append(chunk)
            continue
        expr = []
        for kind, token in toks:
            if kind == "num":
                raw = token.rstrip("uUlL")
                expr.append(repr(float.fromhex(raw)) if raw.lower().startswith("0x")
                            and ("p" in raw.lower() or "." in raw) else raw)
            else:
                expr.append(token)
        result.append(float(value(ast.parse(" ".join(expr), mode="eval").body)).hex())
    return before + "=" + "".join(result)


def generate_libm():
    """Translate the CUDA binary64 exp/log helpers with their exact tables."""
    items = split_top(preprocessed("thompson_aerosol_cold", "461"))
    sections = []
    for stem in ("exp", "log"):
        ns = "g64_" + stem
        block = next(it for it in items if it.startswith("namespace " + ns))
        nested = split_top(block[block.index("{") + 1:block.rfind("}")])
        reached, _ = closure(nested, [stem])
        translated = []
        for item in reached:
            name = def_name(item)
            if name == "asuint64":
                translated.append("static inline ulong asuint64(Soft8 x) { return x.bits; }")
            elif name == "asdouble":
                translated.append("static inline Soft8 asdouble(ulong x) { return Soft8::fromBits(x); }")
            elif kind_of(item) == "struct":
                translated.append(convert_struct(item))
            elif kind_of(item) == "table":
                translated.append(convert_table(fold_table_initializer(item)))
            else:
                translated.append(convert_function_item(item))
        sections.append("namespace aa_%s {\n%s\n}" % (ns, "\n".join(translated)))
    inf = next(it for it in items if def_name(it) == "g64_inf")
    sections.insert(0, convert_function_item(inf))
    for name in ("glibc_exp", "glibc_log"):
        wrapper = next(it for it in items if def_name(it) == name)
        wrapper = wrapper.replace("g64_exp::", "aa_g64_exp::").replace("g64_log::", "aa_g64_log::")
        sections.append(convert_function_item(wrapper))
    source = (ROOT / "woof/core/kernels/glibc_flt64.cuh").read_text()
    grant = source[source.index("// Permission is hereby granted"):source.index("namespace g64_exp")]
    (SHADERS / "thompson_aerosol_libm64_fragment.metalh").write_text(
        "// Generated from the exp/log helpers in woof/core/kernels/glibc_flt64.cuh\n"
        "// by tools/mp28_metal/generate.py; do not edit.\n"
        "// Copyright (c) 2017-2026, Arm Limited.\n"
        "// SPDX-License-Identifier: MIT\n" + grant +
        "#ifndef WOOF_THOMPSON_AEROSOL_LIBM64_FRAGMENT_METALH\n"
        "#define WOOF_THOMPSON_AEROSOL_LIBM64_FRAGMENT_METALH\n" +
        "\n".join(sections) + "\n#endif\n")


def main():
    generate_libm()
    for stem in FAMILIES:
        fragments, kernel_texts, signatures = translate_family(stem)
        includes = []
        for (s, version), frag_items in fragments.items():
            ns = "aa" + version
            path = SHADERS / ("thompson_aerosol_%s_%s_fragment.metalh" % (s, ns))
            path.write_text(
                "// Generated from woof/core/kernels/thompson_aerosol_%s.cu "
                "(%s) by tools/mp28_metal/generate.py; do not edit.\n"
                "#pragma clang fp contract(off)\nnamespace %s {\n%s\n}\n" % (
                    s, "WRF 3.9 fork" if version == "39" else "WRF 4.6.1",
                    ns, "\n".join(frag_items)))
            includes.append('#include "%s"' % path.name)
        shader = SHADERS / ("thompson_aerosol_%s_family.metal" % stem)
        shader.write_text(
            "// Generated by tools/mp28_metal/generate.py from the CUDA "
            "source; do not edit.\n#include <metal_stdlib>\n"
            '#include "thompson_aerosol_soft8.metalh"\n%s\n'
            '#include "thompson_aerosol_%s_family_bindings.metalh"\n\n%s\n' % (
                "\n".join(includes), stem, "\n\n".join(kernel_texts)))
        (FRAGMENTS / ("thompson_aerosol_%s_family.json" % stem)).write_text(
            json.dumps({"kernels": signatures}, indent=2) + "\n")


if __name__ == "__main__":
    main()
