#!/usr/bin/env python3
"""Translate a WSM-family CUDA unit to its Metal shader and signatures.

    python tools/wsm_metal/generate.py wsm5 [wsm3 wsm6r wsm7]

The unit is preprocessed exactly as the loader hands it to nvrtc, once per
level tier; the definitions each kernel reaches are translated statement for
statement with tools/mp28_metal's translator (binary64 on Soft8, float
division on the soft32 divider, address-space templates).  Each tier lives in
its own namespace and its kernels carry the tier in their name.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/mp28_metal"))
sys.path.insert(0, str(ROOT))

import translate  # noqa: E402
from parse import closure, def_name, split_top  # noqa: E402
from translate import (convert_function_item, convert_struct,  # noqa: E402
                       convert_table, map_types, rewrite_division,
                       split_params, tokenize, untok, match_fwd)

SHADERS = ROOT / "woof/metal/shaders"
FRAGMENTS = ROOT / "woof/metal/signature_fragments"

translate.FN_RENAME.update({"roundf": "round", "fabs": "fabs"})

#: Definitions supplied by wsm_support.metalh instead of translated.
PROVIDED = {"gfk_exp", "gfk_log", "gfk_pow", "glibc_exp", "glibc_log",
            "glibc_pow"}

SURFACE = {"rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
           "graupelncv", "hailnc", "hailncv", "sr", "xland"}
WORDS = {"coef", "rf", "rd", "cf"}

#: unit -> (KMAX macro, [(tier, suffix)], [(cuda kernel, metal base)])
SCHEMES = {
    "wsm6": ("WSM6_KMAX", ((64, ""), (80, "_deep")),
             (("wsm6_column", "wsm6_column"),)),
    "wdm6": ("WDM6_KMAX", ((64, ""), (80, "_deep")),
             (("wdm6_column", "wdm6_column"),)),
    "wsm5": ("WSM5_KMAX", ((64, ""), (80, "_deep")),
             (("wsm5_column", "wsm5_column"),
              ("wsm5_refl10cm_column", "wsm5_refl10cm_column"))),
    "wsm3": ("WSM3_KMAX", ((64, ""), (80, "_deep")),
             (("wsm3_column", "wsm3_column"),)),
    "wsm6r": (None, ((0, ""),),
              (("wsm6r_column", "wsm6r_column"),
               ("wsm6r_effective_radii", "wsm6r_effective_radii"))),
    "wsm7": ("WSM7_KMAX", ((64, ""), (80, "_deep")),
             (("wsm7_column", "wsm7_column"),
              ("wsm7_refl10cm_column", "wsm7_refl10cm_column"))),
}


#: Correct-mode finding WB5 compiles the column kernel with this define; the
#: result is a second entry point per tier, suffixed ``_cs``.
SURFACE_DEFINE = {"wsm5": "WSM5_PLM_SURFACE", "wsm3": "WSM3_PLM_SURFACE",
                  "wsm7": "WSM7_PLM_SURFACE", "wsm6": "WSM6_PLM_SURFACE",
                  "wdm6": "WDM6_PLM_SURFACE"}


def local_wsm6_columns(text, tier):
    """Move CUDA's strided workspace into fixed Metal thread arrays."""
    # The radar entry point is compiled separately by wsm6_reflectivity.
    text = text.split('extern "C" __global__ void wsm6_refl10cm_column')[0]
    text = re.sub(r"template <typename T>\s*struct WColT.*?__device__ __forceinline__ WCol wsm6_wnull\(\)\s*\{.*?\n\}",
                  "", text, flags=re.S)
    def arrays(match):
        names = re.findall(r"(\w+)\s*=\s*wsm6_wcol\([^)]*\)", match.group(1))
        return "float " + ", ".join(f"{n}[{tier + 1}]" for n in names) + ";"
    text = re.sub(r"const WCol ([^;{}]*wsm6_wcol\([^;{}]*);", arrays, text)
    text = re.sub(r"float \*const sedb = [^;]*;", "", text)
    text = re.sub(r"float \*wsb = ws [^;]*;", "", text)
    text = re.sub(r",\s*float \*wsb, int ws", "", text)
    text = re.sub(r",\s*sedb, ws", "", text)
    text = re.sub(r",\s*wsb, \(int\)blockDim\.x", "", text)
    text = re.sub(r",\s*float \*ws, int col0", "", text)
    text = text.replace("col0 + ", "").replace("rq2.p", "rq2")
    text = text.replace("wsm6_wnull()", "(thread float *)nullptr")
    return re.sub(r"\bconst WCol\b", "float *const", text)


def preprocessed(unit, macro, tier, surface=False):
    from woof.core import kernels
    defines = ((macro, tier),) if macro else ()
    if surface:
        defines += ((SURFACE_DEFINE[unit], 1),)
    text = (kernels.module_source_int_defines(unit, defines) if defines
            else kernels.module_source(unit))
    done = subprocess.run(
        ["clang++", "-E", "-P", "-std=c++17", "-x", "c++", "-"],
        input=text.encode(), capture_output=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.decode()[-3000:])
    text = done.stdout.decode()
    if unit == "wsm6":
        text = local_wsm6_columns(text, tier)
    # The radar code names the binary64 helpers through their namespaces; the
    # shader carries them as glibc_exp/glibc_log/glibc_pow.
    text = re.sub(r"\bg64_(exp|log|pow)\s*::\s*\1\s*\(", r"glibc_\1(", text)
    # An absent optional array reaches a templated pointer as a typed null.
    text = re.sub(r",\s*nullptr\b", ", (thread float *)nullptr", text)
    # A local pointer into a templated pointer parameter keeps its address
    # space only when deduced.
    return re.sub(r"\b(?:const\s+)?(?:float|double|int)\s*\*\s*(?:const\s+)?(\w+)\s*=",
                  r"auto \1 =", text)


#: unit -> names of the functions (and kernels) whose binary32 add, subtract
#: and multiply run on the subnormal-preserving Sf type, or "all".  The host
#: keeps binary32 subnormals; Metal's native float operations flush them.
SOFT = {"wsm6r": {"r6_accret2_level"}, "wsm5": {"wsm5_nislfv", "w5max", "w5min"},
        "wsm3": {"wsm3_nislfv", "wmax", "wmin"},
        "wsm7": {"wsm7_nislfv", "wmax", "wmin"},
        "wsm6": {"wsm6_nislfv", "wmax", "wmin"},
        "wdm6": "all"}


#: unit -> line prefixes of single statements whose arithmetic runs on Sf.
STMT = {"wsm7": ["qhk = wmax ( qhk + ( phdep", "rq4 [ k ] = denk * qhl",
                 "hailncv_c = "]}


def _make_line_softener():
    tok = re.compile(r"(?<![\w.])(\d+\.\d*(?:[eE][+-]?\d+)?f|\.\d+(?:[eE][+-]?\d+)?f"
                     r"|\d+[eE][+-]?\d+f)(?!\w)|\b[A-Za-z_]\w*\s*\[[^\]]*\]"
                     r"|(?<!\.)(?<!\. )\b([A-Za-z_]\w*)\b(?!\s*[(\[])")
    keep = {"float", "const", "int", "uint", "Sf"}

    def one(line):
        if re.search(r"= [^;()]* , [A-Za-z_]\w* =", line):
            return line
        head, sep, rest = line.partition(" = ")
        rest = tok.sub(lambda m: m.group(0) if (m.group(2) in keep or (
            m.group(2) and m.group(2).startswith(("sh_", "t64", "t80")))) else
            "Sf(%s)" % m.group(0), rest)
        return head + sep + rest

    return one


_soften_line = _make_line_softener()


def soften_statements(unit, text):
    prefixes = STMT.get(unit, ())
    if not prefixes:
        return text
    return "\n".join(_soften_line(l) if any(l.startswith(p) for p in prefixes)
                     else l for l in text.split("\n"))


#: Metal's native float comparisons flush subnormals too: a mass of 1e-41 that
#: the host sees as positive reads as zero.  Comparisons against zero in
#: these units run on the soft32 compare.
ZERO_COMPARE = ("wsm3", "wsm5", "wsm7", "wsm6", "wdm6")
ZERO_COMPARE_RE = re.compile(
    r"(\b[A-Za-z_]\w*(?: \[ [^\]\[]+ \])?) (<=|>=|<|>) 0\.0f\b(?!\s*[\w(])")


def softened(unit, name, text):
    selected = SOFT.get(unit, ())
    text = soften_statements(unit, text)
    if unit in ZERO_COMPARE:
        text = ZERO_COMPARE_RE.sub(r"Sf(\1) \2 Sf(0.0f)", text)
    if selected == "all" or name in selected:
        text = re.sub(r"\bfloat\b", "Sf", text)
        # Literals become Sf too, so a conditional of Sf and a literal has
        # one type.
        return re.sub(r"(?<![\w.])(\d+\.\d*(?:[eE][+-]?\d+)?f|\.\d+(?:[eE][+-]?\d+)?f"
                      r"|\d+[eE][+-]?\d+f)(?!\w)", r"Sf(\1)", text)
    return text


def kind_of(item):
    if item.startswith('extern "C"'):
        return "kernel"
    if item.startswith("struct"):
        return "struct"
    if item.startswith("enum"):
        return "enum"
    head = item.split("{")[0].split("=")[0]
    if "(" in head:
        return "function"
    return "table"


def kernel_info(item):
    toks = tokenize(item)
    paren = next(k for k, t in enumerate(toks) if t[1] == "(")
    close = match_fwd(toks, paren)
    name = toks[paren - 1][1]
    params = []
    for p in split_params(toks, paren, close):
        words = [t for _, t in p if t not in ("__restrict__", "const")]
        pointer = "*" in words
        params.append({"name": words[-1], "pointer": pointer,
                       "base": " ".join(w for w in words[:-1] if w != "*"),
                       "const": any(t == "const" for _, t in p)})
    return name, params, toks[close + 1:]


def convert_kernel(item, metal_name, unit=""):
    name, params, body_toks = kernel_info(item)
    body_text = re.sub(
        r"blockIdx\s*\.\s*x\s*\*\s*blockDim\s*\.\s*x\s*\+\s*threadIdx\s*\.\s*x",
        "int ( gid )", untok(body_toks))
    body_text = softened(unit, name, untok(
        rewrite_division(map_types(tokenize(body_text)))))
    if unit == "wdm6":
        body_text = body_text.replace(
            "atomicOr ( count_status , flags )",
            "atomic_fetch_or_explicit((device atomic_uint *)count_status, flags, memory_order_relaxed)")
    title = "".join(p.title() for p in metal_name.split("_"))
    lines = ["kernel void %s(" % metal_name,
             "    constant %sResources &wsm_r [[buffer(0)]]," % title,
             "    constant %sParams &wsm_p [[buffer(1)]]," % title,
             "    uint gid [[thread_position_in_grid]]) {"]
    for prm in params:
        if prm["pointer"]:
            cv = "const " if prm["const"] else ""
            if prm["name"] == "count_status":
                lines.append("    device int *count_status = wsm_r.count_status;")
            elif prm["base"] == "double":
                lines.append("    device const Soft8 *%s = (device const Soft8 *)wsm_r.%s;"
                             % (prm["name"], prm["name"]))
            else:
                lines.append("    device %sfloat *%s = wsm_r.%s;"
                             % (cv, prm["name"], prm["name"]))
        else:
            lines.append("    const %s %s = wsm_p.%s;" % (
                "Soft8" if prm["base"] == "double" else prm["base"],
                prm["name"], prm["name"]))
    lines.append(body_text)
    lines.append("}")
    return "\n".join(lines), params


def signature(unit, cuda_name, metal_name, params, threads):
    resources, shapes, parameters = [], {}, []
    for p in params:
        if p["pointer"]:
            double = p["base"] == "double"
            resources.append([p["name"], "int32" if p["name"] == "count_status" else "float64_bits" if double else "float",
                              "read" if p["const"] else "read_write"])
            shapes[p["name"]] = ([[3, "nz", "ny", "nx"]] if p["name"] == "work"
                                 else [[1]] if p["name"] == "count_status"
                                 else [["any_1d"]] if p["name"] in WORDS
                                 else [["ny", "nx"]] if p["name"] in SURFACE
                                 else [["nz", "ny", "nx"]])
        else:
            parameters.append([p["name"], "float32" if p["base"] == "float"
                               else "int32"])
    return {
        "name": metal_name, "cuda_symbol": cuda_name,
        "module": unit + "_family",
        "source": "woof/metal/shaders/%s_family.metal" % unit,
        "cuda_source": "woof/core/kernels/%s.cu" % unit,
        "shapes": shapes, "parameters": parameters,
        "threads_per_group": threads, "layout_version": 1,
        "arithmetic_policies": ["soft32-rn-nonftz", "soft64-rn-bit-ops",
                                "portable-libm64"],
        "resources": resources,
    }


def translate_tier(unit, macro, tier, wanted, surface=False):
    items = []
    for it in split_top(preprocessed(unit, macro, tier, surface)):
        if it.startswith(("namespace", "typedef", "#pragma", "using")):
            continue
        if def_name(it) in PROVIDED:
            continue
        items.append(it)
    enums = [it for it in items if kind_of(it) == "enum"]
    rest = [it for it in items if kind_of(it) != "enum"]
    reached, _ = closure(rest, [w[0] for w in wanted])
    frag, kernels = [], {}
    for it in reached:
        kind = kind_of(it)
        if kind == "kernel":
            kernels[def_name(it)] = it
        elif kind == "struct":
            frag.append(convert_struct(it))
        elif kind == "table":
            frag.append(convert_table(it))
        else:
            frag.append(softened(unit, def_name(it), convert_function_item(it)))
    used = "\n".join(frag) + "\n".join(kernels.values())
    enum_text = [untok(map_types(tokenize(e))) for e in enums
                 if any(re.search(r"\b" + m + r"\b", used)
                        for m in re.findall(r"\b([A-Z]\w*[A-Z0-9_])\b",
                                            e.split("{", 1)[1]))]
    return enum_text, frag, kernels


def generate(unit):
    macro, tiers, wanted = SCHEMES[unit]
    shader_parts, kernel_texts, sigs = [], [], []
    variants = [(tier, suffix, False) for tier, suffix in tiers]
    if unit in SURFACE_DEFINE:
        variants += [(tier, suffix + "_cs", True) for tier, suffix in tiers]
    for tier, suffix, surface in variants:
        used_wanted = wanted[:1] if surface else wanted
        enums, frag, kernels = translate_tier(unit, macro, tier, used_wanted,
                                              surface)
        ns = "t%d%s" % (tier, "cs" if surface else "")
        shader_parts.append("namespace %s {\n%s\n%s\n}" % (
            ns, "\n".join(enums), "\n".join(frag)))
        for cuda_name, base in used_wanted:
            metal_name = base + suffix
            text, params = convert_kernel(kernels[cuda_name], metal_name, unit)
            kernel_texts.append(text.replace(
                ") {\n", ") {\n    using namespace %s;\n" % ns, 1))
            sigs.append(signature(unit, cuda_name, metal_name, params, 32))
    suffix = ""
    if unit == "wsm6":
        previous = (SHADERS / "wsm6_family.metal").read_text()
        marker = "// WRF's moist_physics_prep_em contract:"
        suffix = '#include "pressure_power.metalh"\n' + re.sub(
            r"struct Wsm6(?:Prepare|Finish)(?:Resources|Params) \{.*?\};\s*",
            "", previous[previous.index(marker):], flags=re.S)
        previous_signatures = json.loads((FRAGMENTS / "wsm6_family.json").read_text())["kernels"]
        sigs.extend(k for k in previous_signatures if not k["name"].startswith("wsm6_column"))
    (SHADERS / ("%s_family.metal" % unit)).write_text(
        "// Generated by tools/wsm_metal/generate.py from woof/core/kernels/"
        "%s.cu; do not edit.\n#include <metal_stdlib>\n"
        '#include "wsm_support.metalh"\n'
        "#pragma clang fp contract(off)\n"
        '#pragma clang diagnostic ignored "-Wparentheses-equality"\n'
        '#pragma clang diagnostic ignored "-Wmissing-braces"\n%s\n'
        '#include "%s_family_bindings.metalh"\n\n%s\n' % (
            unit, "\n".join(shader_parts), unit, "\n\n".join(kernel_texts)) + suffix)
    (FRAGMENTS / ("%s_family.json" % unit)).write_text(
        json.dumps({"kernels": sigs}, indent=2) + "\n")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["wsm5"]:
        generate(name)
