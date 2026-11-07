#!/usr/bin/env python3
"""Generate the Metal translation of the generated Thompson column
(woof/core/kernels/thompson_column*.cu*, WRF v4.4 classic generation).

    python tools/thompson_column_metal/generate.py

The CUDA unit is preprocessed exactly as the loader hands it to nvrtc, the
definitions the kernel reaches are kept, and the statements are translated
unchanged with the mp=28 machinery in tools/mp28_metal: binary64 is Soft8 on
the soft64 helpers, float division runs on the soft32 divider, and the
rounded intrinsics are the plain IEEE operations of a contraction-off build.
The column body is emitted once with its level capacity left symbolic
(TCKMAX) and instantiated per capacity tier, as the CUDA loader does.
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

from parse import closure, def_name, split_top  # noqa: E402
import generate as g  # noqa: E402
from translate import (map_types, rewrite_division, tokenize,  # noqa: E402
                       untok)

SHADERS = ROOT / "woof/metal/shaders"
FRAGMENTS = ROOT / "woof/metal/signature_fragments"
TIERS = (64, 128, 256)
#: Tiers above this keep their level arrays in a device workspace: Metal
#: refuses the larger per-thread frame.
WORKSPACE_FROM = 64
KERNEL = "thompson_column_step"
FAMILY = "thompson_column_family"


def _matching_open(text, close, closer, opener):
    depth, i = 0, close
    while True:
        if text[i] == closer:
            depth += 1
        elif text[i] == opener:
            depth -= 1
            if depth == 0:
                return i
        i -= 1


def _loop_header(text, brace):
    """``for(...)`` or ``while(...)`` text in front of a block, else ''."""
    j = brace - 1
    while text[j].isspace():
        j -= 1
    if text[j] != ")":
        return ""
    k = _matching_open(text, j, ")", "(")
    m = re.search(r"\b(for|while|switch)\s*$", text[:k])
    return m.group(1) if m else ""


def eliminate_gotos(text):
    """Turn each forward ``goto`` out of a loop into ``break``.

    The generated Fortran EXIT idiom is ``for(...){ ... goto L; ... } L: ;``
    with the label directly after the loop.  Metal has no goto, so the jump
    becomes a break, provided no loop or switch sits between the two.
    """
    for label in sorted(set(re.findall(r"\bgoto\s+(label_\w+)\s*;", text))):
        decl = re.search(r"\b%s\s*:\s*;" % label, text)
        before = text[:decl.start()].rstrip()
        if not before.endswith("}"):
            raise RuntimeError(label + " does not follow a block")
        close = len(before) - 1
        open_ = _matching_open(text, close, "}", "{")
        if _loop_header(text, open_) not in ("for", "while"):
            raise RuntimeError(label + " does not follow a loop")
        body = text[open_ + 1:close]
        for jump in re.finditer(r"\bgoto\s+%s\s*;" % label, body):
            stack = []
            for m in re.finditer(r"[{}]", body[:jump.start()]):
                if m.group() == "{":
                    stack.append(_loop_header(body, m.start()) != "")
                else:
                    stack.pop()
            if any(stack):
                raise RuntimeError(label + " leaves a nested loop")
        body = re.sub(r"\bgoto\s+%s\s*;" % label, "break;", body)
        text = (text[:open_ + 1] + body + text[close:decl.start()]
                + text[decl.end():])
    return text


def preprocessed():
    from woof.core import kernels
    text = kernels.module_source_int_defines(
        "thompson_column", (("THOMPSON_COLUMN_KMAX", "TCKMAX"),))
    done = subprocess.run(
        ["clang++", "-E", "-P", "-std=c++17", "-x", "c++", "-"],
        input=text.encode(), capture_output=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.decode()[-3000:])
    text = eliminate_gotos(done.stdout.decode())
    # The host's binary64 hypot, atan2 and log10 have no Metal counterpart;
    # the mp=28 sources carry reduced-form words for them.
    text = re.sub(r"(?<![\w.])hypot\s*\(", "thompson_aa_hypot(", text)
    text = re.sub(r"(?<![\w.])atan2\s*\(", "thompson_aa_atan2(", text)
    text = re.sub(r"(?<![\w.])log10\s*\(", "thompson_aa_log10(", text)
    return text


def thread_params(line):
    """Give pointer and reference parameters of a method the thread space."""
    open_i = line.index("(")
    close_i = line.rindex(")")
    params, depth, cur = [], 0, ""
    for ch in line[open_i + 1:close_i]:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            params.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        params.append(cur)
    out = []
    for p in params:
        p = p.strip()
        if ("*" in p or "&" in p) and not re.match(r"(thread|device|constant)\b", p):
            array = re.match(r"(const\s+)?(float|Soft8)\s*\*", p)
            p = ("TCSPACE " if array else "thread ") + p
        out.append(p)
    return line[:open_i + 1] + ", ".join(out) + line[close_i:]


def convert_column_struct(text):
    """The struct with its member functions, constants and table pointers."""
    toks = rewrite_division(map_types(tokenize(text)))
    src = untok(toks)
    hoisted = []
    lines = []
    depth = 0
    for line in src.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if depth == 1 and stripped.startswith("static constexpr"):
            hoisted.append("constant " + stripped[len("static constexpr "):])
            continue
        if depth == 1 and re.match(r"const (float|Soft8) \* \w+ = tc_\w+ ;", stripped):
            stripped = re.sub(r"^const (float|Soft8) \*", r"constant \1 *", stripped)
        elif depth == 1 and re.match(r"const (float|Soft8) \* \w+ ;", stripped):
            stripped = "device " + stripped
        elif (depth == 1 and stripped.endswith("{") and "(" in stripped
              and not re.match(r"(if|for|while|else|switch|struct)\b", stripped)):
            stripped = thread_params(stripped)
        m = re.match(r"(float|Soft8) (\w+) \[ TCKMAX( \+ 1)? \] ;$", stripped)
        if m:
            if depth != 2:
                raise RuntimeError("level array outside a method's top block: " + stripped)
            stripped = "%s ( %s , TCKMAX%s ) ;" % (
                "TCF" if m.group(1) == "float" else "TCD", m.group(2), m.group(3) or "")
        depth += stripped.count("{") - stripped.count("}")
        lines.append(stripped)
        if depth == 1 and stripped.startswith("struct ThompsonColumn"):
            lines.append("device float * tc_wsf ;")
            lines.append("device Soft8 * tc_wsd ;")
    return "\n".join(hoisted), "\n".join(lines)


def build():
    items = split_top(preprocessed())
    keep = []
    for it in items:
        if (it.startswith(("namespace", "typedef", "#pragma"))
                or def_name(it) in g.PROVIDED):
            continue
        keep.append(it)
    reached, _ = closure(keep, [KERNEL])
    reached = [g.typed_nulls(it, reached) for it in reached]
    pieces, kernel_item, struct_parts = [], None, None
    for it in reached:
        kind = g.kind_of(it)
        if kind == "kernel":
            kernel_item = it
        elif kind == "struct" and def_name(it) == "ThompsonColumn":
            struct_parts = convert_column_struct(it)
        elif kind == "struct":
            pieces.append(g.convert_struct(it))
        elif kind == "table":
            pieces.append(g.convert_table(it))
        else:
            pieces.append(g.convert_function_item(it))
    hoisted, struct_text = struct_parts
    return pieces, hoisted, struct_text, kernel_item


WORKSPACE_MACROS = """\
#if TCWS
#define TCF(n, sz) device float * n = tc_wsf; tc_wsf += (sz)
#define TCD(n, sz) device Soft8 * n = tc_wsd; tc_wsd += (sz)
#define TCKF(n, sz) device float * n = m.tc_wsf; m.tc_wsf += (sz)
#define TCSPACE device
#else
#define TCF(n, sz) float n[sz]
#define TCD(n, sz) Soft8 n[sz]
#define TCKF(n, sz) float n[sz]
#define TCSPACE thread
#endif
"""


def kernel_text(kernel_item, name, tier):
    text, params = g.convert_kernel(kernel_item, name, "tcK%d" % tier, [])
    # One column per thread: ``first`` offsets a launch chunk, and the chunk
    # slot indexes the level-array workspace.
    text = text.replace("const int col = int ( gid ) ;",
                        "const int slot = int ( gid ) ;\n    const int col = p.first + slot ;")
    if "const int slot" not in text:
        raise RuntimeError("kernel column index changed")

    def declarators(match):
        names = re.findall(r"(\w+) \[ TCKMAX \]", match.group(0))
        return " ".join("TCKF ( %s , TCKMAX ) ;" % n for n in names)
    text = re.sub(r"^float (?:\w+ \[ TCKMAX \] ,? ?)+;$", declarators, text, flags=re.M)
    anchor = re.search(r"^TCKF", text, flags=re.M)
    setup = ("m.tc_wsf = r.ws_f + slot * TC_STRIDE_F;\n"
             "m.tc_wsd = (device Soft8 *)r.ws_d + slot * TC_STRIDE_D;\n")
    text = text[:anchor.start()] + setup + text[anchor.start():]
    return text, params


def strides(*texts):
    text = "\n".join(texts)
    out = []
    for macro in ("TCF", "TCKF"), ("TCD",):
        sizes = []
        for m in macro:
            sizes += re.findall(r"\b%s \( \w+ , TCKMAX( \+ 1)? \)" % m, text)
        out.append("((%d) * TCKMAX + %d)" % (len(sizes), sum(1 for z in sizes if z)))
    return out


def main():
    pieces, hoisted, struct_text, kernel_item = build()
    shared = "\n".join(pieces)
    body = hoisted + "\n" + struct_text
    (SHADERS / "thompson_column_helpers.metalh").write_text(
        "// Generated from woof/core/kernels/thompson_column*.cu* by "
        "tools/thompson_column_metal/generate.py; do not edit.\n"
        "#pragma clang fp contract(off)\n"
        "namespace tc44 {\n" + shared + "\n}\n")
    (SHADERS / "thompson_column_body.metalh").write_text(
        "// Generated by tools/thompson_column_metal/generate.py; do not edit.\n"
        "// Included once per level-capacity tier with TCKMAX and TCWS defined.\n"
        "#pragma clang fp contract(off)\n"
        "using namespace tc44;\n" + body + "\n")
    kernels = []
    params = None
    for tier in TIERS:
        name = "%s_k%d" % (KERNEL, tier)
        text, params = kernel_text(kernel_item, name, tier)
        stride_f, stride_d = strides(body, text)
        kernels.append(
            "#define TCKMAX %d\n#define TCWS %d\n%s"
            "#define TC_STRIDE_F %s\n#define TC_STRIDE_D %s\n"
            "namespace tcK%d {\n#include \"thompson_column_body.metalh\"\n}\n%s\n"
            "#undef TCKMAX\n#undef TCWS\n#undef TCF\n#undef TCD\n#undef TCKF\n"
            "#undef TCSPACE\n#undef TC_STRIDE_F\n#undef TC_STRIDE_D"
            % (tier, int(tier > WORKSPACE_FROM), WORKSPACE_MACROS, stride_f, stride_d,
               tier, text))
    (SHADERS / (FAMILY + ".metal")).write_text(
        "// Generated by tools/thompson_column_metal/generate.py from the CUDA "
        "source; do not edit.\n#include <metal_stdlib>\n"
        '#include "thompson_aerosol_soft8.metalh"\n'
        '#include "thompson_column_helpers.metalh"\n'
        '#include "%s_bindings.metalh"\n\n%s\n' % (FAMILY, "\n\n".join(kernels)))
    resources, shapes, parameters = [], {}, []
    for p in params:
        if p["pointer"]:
            storage = "float64_bits" if p["base"] == "double" else "float"
            resources.append([p["name"], storage,
                              "read" if p["const"] else "read_write"])
            shapes[p["name"]] = [["any_1d"]]
        else:
            parameters.append([p["name"], "float32" if p["base"] == "float"
                               else "int32"])
    resources += [["ws_f", "float", "read_write"], ["ws_d", "float64_bits", "read_write"]]
    shapes["ws_f"] = shapes["ws_d"] = [["any_1d"]]
    parameters.append(["first", "int32"])
    sigs = []
    for tier in TIERS:
        name = "%s_k%d" % (KERNEL, tier)
        sigs.append({
            "name": name, "cuda_symbol": KERNEL, "module": FAMILY,
            "source": "woof/metal/shaders/%s.metal" % FAMILY,
            "cuda_source": "woof/core/kernels/thompson_column.cu",
            "resources": resources, "shapes": shapes, "parameters": parameters,
            "threads_per_group": 32, "layout_version": 1,
            "arithmetic_policies": g.POLICIES})
    (FRAGMENTS / (FAMILY + ".json")).write_text(
        json.dumps({"kernels": sigs}, indent=2) + "\n")


if __name__ == "__main__":
    main()
