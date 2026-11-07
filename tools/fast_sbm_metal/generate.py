#!/usr/bin/env python3
"""Translate the FAST_SBM (mp_physics=30) CUDA unit to Metal.

    python tools/fast_sbm_metal/generate.py

The unit is preprocessed as the loader hands it to nvrtc and the definitions
its three production kernels reach are translated statement for statement.
Binary32 values travel as soft32 words (F) and binary64 values as Soft8; the
unit's own float and double mixing is kept through the implicit conversions and
mixed operators of woof/metal/shaders/fast_sbm_support.metalh. Pointer and
reference parameters become address-space templates; the packaged tables and
the thirty state fields reach the kernels as separate resources.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/mp28_metal"))
sys.path.insert(0, str(ROOT))
import parse  # noqa: E402
from parse import split_top  # noqa: E402

_plain_name = parse.def_name
import translate  # noqa: E402
from translate import map_types, match_fwd, rewrite_division, split_params, tokenize, untok  # noqa: E402
from woof.core.fast_sbm_contract import STATE_FIELDS  # noqa: E402

translate.TYPE_WORDS.update(('R', 'F'))
SHADERS = ROOT / "woof/metal/shaders"
KERNEL_DIR = ROOT / "woof/core/kernels"
KERNELS = {"fsbm_grid_fields_f32": 3, "fsbm_prepare_columns_f32": 4, "fsbm_columns_f32": 1}
POLICIES = ["soft32-rn-nonftz", "soft64-rn-bit-ops", "portable-libm64"]
# Supplied by the support headers: the pinned float and binary64 math, the
# typedefs, and the literal helper that picks the build's REAL kind.
# By-reference parameters the callers bind to state-array elements.
DEVICE_REFERENCES = {"tin", "t_new", "qv", "rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv"}
PROVIDED = ("GFK_", "gfk_", "g64_", "glibc_")
PROVIDED_NAMES = {"uint64_t", "int64_t", "uint32_t", "value", "fsbm_rl", "fsbm_is_double"}


# The dsupice neighbour reader the driver builds as a lambda.
NEIGHBOUR_READER = """
template <class R, class PT, class PQ, class PP, class PU, class PV> struct FsbmNb {
    PT t_old; PQ qvo; PP pcgs; PU u; PV v;
    long ims, kms, jms, ni, nk;
    int i, k, j;
    FsbmNbr<R> operator()(int di, int dj) const thread {
        long q = (long)(i + di - ims) + (long)(k - kms) * ni + (long)(j + dj - jms) * ni * nk;
        FsbmNbr<R> r = {t_old[q], qvo[q], pcgs[q], u[q], v[q]};
        return r;
    }
};"""


def preprocessed():
    text = (KERNEL_DIR / "fast_sbm.cu").read_text()
    done = subprocess.run(
        ["clang++", "-E", "-P", "-std=c++17", "-x", "c++", "-D__CUDACC_RTC__",
         "-I" + str(KERNEL_DIR), "-"], input=text.encode(), capture_output=True, check=True)
    out = done.stdout.decode()
    return split_top(out)


def def_name(item):
    """parse.def_name, except that a template struct is named by its type."""
    m = re.match(r"template\s*<[^>]*>\s*struct\s+(\w+)", item)
    return m.group(1) if m else _plain_name(item)


def closure(items, roots):
    parse.def_name = def_name
    return parse.closure(items, roots)


def provided(name):
    return name is None or name in PROVIDED_NAMES or name.startswith(PROVIDED)


def header_split(toks):
    """Indexes of the function's parameter parentheses."""
    brace = next(k for k, t in enumerate(toks) if t[1] == "{")
    paren = next(k for k, t in enumerate(toks[:brace]) if t[1] == "(")
    return paren, match_fwd(toks, paren)


def template_params(toks):
    """The leading ``template <...>`` clause, split off."""
    if toks and toks[0][1] == "template":
        depth, k = 0, 1
        while True:
            if toks[k][1] == "<":
                depth += 1
            elif toks[k][1] == ">":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        inner = toks[2:k]
        return [p for p in split_params([("op", "(")] + inner + [("op", ")")], 0, len(inner) + 1)], toks[k + 1:]
    return [], toks


def convert_function(item):
    toks = map_types(tokenize(item))
    templ, toks = template_params(toks)
    paren, close = header_split(toks)
    extra, params = [], []
    for k, p in enumerate(split_params(toks, paren, close)):
        words = [t for _, t in p]
        name = words[-1] if words[-1] != "]" else None
        default = "=" in words
        if "*" in words and not default:
            cut = words.index(name)
            extra.append("P_" + name)
            params.append([("id", "P_" + name), ("id", name)])
        elif "*" in words:
            params.append([("id", "device" if name == "ledger" else "thread")] + p)
        elif "&" in words:
            space = "device" if name in DEVICE_REFERENCES else "thread"
            params.append([("id", space)] + p)
        else:
            params.append(p)
    mid = []
    for k, p in enumerate(params):
        if k:
            mid.append(("op", ","))
        mid.extend(p)
    clause = [untok(p).strip() for p in templ] + ["typename " + e for e in extra]
    head = tokenize("template <" + ", ".join(clause) + ">") if clause else []
    body = toks[close:]
    return untok(head + [("id", "static"), ("id", "inline")] + toks[:paren + 1] + mid + body)


def convert_struct(item):
    toks = map_types(tokenize(item))
    text = untok(toks)
    out = []
    for line in re.split(r"(?<=[;{}])", text):
        stripped = line.strip()
        if re.match(r"(const\s+)?[\w<>\s]+?\s*\*", stripped) and stripped.endswith(";") and "(" not in stripped:
            space = "thread " if "FsbmTab" in stripped else "device "
            line = " " + space + stripped
        out.append(line)
    return "".join(out)


def words32(text):
    text = re.sub(r"as_type\s*<\s*float\s*>", "AS_TYPE_FLOAT", text)
    return re.sub(r"\bfloat\b", "F", text).replace("AS_TYPE_FLOAT", "as_type<float>")


def body_fixes(text):
    # The unit's flag struct is named F, which is the word type here.
    text = re.sub(r"\bF\b", "FSBM_FLAGS", text)
    return text


def split_declarators(text):
    """``R *a = x, *b = y;`` becomes one declaration per pointer."""
    pattern = re.compile(r"(?m)^(\s*)((?:const\s+)?(?:R|double|int|float|Soft8))\s*\*\s*(\w+)\s*=\s*([^,;]+)((?:,\s*\*\s*\w+\s*=\s*[^,;]+)+);")

    def split(m):
        rest = re.findall(r",\s*\*\s*(\w+)\s*=\s*([^,;]+)", m[5])
        decls = [(m[3], m[4])] + rest
        return "\n".join("%s%s *%s = %s;" % (m[1], m[2], n, v.strip()) for n, v in decls)
    return pattern.sub(split, text)


def local_pointers(text):
    """Local pointer and reference declarations take the type of their value."""
    text = re.sub(r"(?m)^(\s*)(?:const\s+)?(?:R|double|int|float|Soft8)\s*\*\s*(\w+)\s*=", r"\1auto \2 =", text)
    text = re.sub(r"(?m)^(\s*)(?:const\s+)?(?:R|double|int|float|Soft8)\s*&\s*(\w+)\s*=", r"\1auto &\2 =", text)
    return text


def convert(item):
    item = split_declarators(body_fixes(item))
    # A default-REAL literal is a binary32 literal of this build's R.
    item = re.sub(r"fsbm_rl\s*<\s*R\s*>\s*\(\s*\(double\)\s*\(([^()]*)\)\s*,\s*([^()]*?)\s*\)", r"R(\2)", item)
    item = re.sub(r"fsbm_is_double\s*<\s*R\s*>\s*::\s*value", "false", item)
    item = item.replace("(const R*)0", "(device const R*)0").replace("(const double*)0", "(device const double*)0")
    kind = def_name(item)
    head = item.split("{", 1)[0]
    is_struct = re.match(r"(template\s*<[^>]*>\s*)?struct\b", item) and "(" not in head.replace("template", "")
    text = convert_struct(item) if is_struct else convert_function(item) if "(" in head else untok(map_types(tokenize(item)))
    text = local_pointers(text)
    text = re.sub(r": \( (\w+) \* \) 0", r": ( thread \1 * ) 0", text)
    text = untok(rewrite_division(tokenize(text)))
    # Spell the binary64 zero explicitly when the conditional mixes kinds.
    text = re.sub(r"(fsbm_gmass \( (?:g1|gdumb|g3) , 33 \) :) 0", r"\1 SD8 ( 0x0000000000000000ul )", text)
    # Preserve the host-source breakup result in the full Metal caller.
    if kind == "coll_breakup_ks":
        text = text.replace("static inline int coll_breakup_ks", "static __attribute__((noinline)) int coll_breakup_ks", 1)
    return words32(lambdas(references(text)))


def references(text):
    """A local reference into an array becomes the array element itself."""
    pattern = re.compile(r"auto & (\w+) = ([^;]*) ;")
    while True:
        m = pattern.search(text)
        if not m:
            return text
        depth, k = 0, m.end()
        while depth >= 0:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        scope = re.sub(r"(?<![\w.])%s(?!\w)" % m[1], "( %s )" % m[2], text[m.end():k])
        text = text[:m.start()] + scope + text[k:]


def ws(pattern):
    return pattern.replace(" ", r"\s*")


def lambdas(text):
    """Index lambdas become macros; the neighbour reader becomes a functor."""
    text = re.sub(ws(r"auto (\w+) = \[ & \] \( ([^)]*) \) -> long \{ return ([^;]*) ; \} ;"),
                  lambda m: "\n#undef %s\n#define %s(%s) (%s)\n" % (
                      m[1], m[1], ", ".join(a.split()[-1] for a in m[2].split(",")), m[3].replace("\n", " ")), text)
    text = re.sub(ws(r"auto nb = \[ & \] \( int di , int dj \) \{ long q = ix \( i \+ di , k , j \+ dj \) ; "
                     r"FsbmNbr < R > r = \{ t_old \[ q \] , qvo \[ q \] , pcgs \[ q \] , S \. u \[ q \] , S \. v \[ q \] \} ; return r ; \} ;"),
                  "FsbmNb < R , decltype ( t_old ) , decltype ( qvo ) , decltype ( pcgs ) , decltype ( S.u ) , decltype ( S.v ) > "
                  "nb = { t_old , qvo , pcgs , S.u , S.v , ims , kms , jms , ni , nk , i , k , j } ;", text)
    return text


def launch_body(text):
    """fsbm_launch with the tables and state arriving prebuilt."""
    pre = text.index("FsbmTab<R> T;")
    end = text.index("T.ikr_spon_break = ikr;") + len("T.ikr_spon_break = ikr;")
    tables = re.findall(r"T\.(\w+) = reinterpret_cast<const (R|double|int)\*>\(tp\[(\d+)\]\);", text[pre:end])
    text = text[:pre] + text[end:]
    state = re.search(r"State<R> S = \{[^}]*\};", text)
    count = state.group().count("ap[")
    text = text.replace(state.group(), "")
    text = text.replace("const unsigned long long* tp, int ikr, const int* flags,\n const unsigned long long* ap,",
                        "const FsbmTab<R>& T, const int* flags, const State<R>& S,")
    text = text.replace("(long)blockIdx.x * blockDim.x + threadIdx.x", "(long)gid_")
    text = text.replace("R dt, R dx, R dy,", "R dt, R dx, R dy, long gid_,")
    return text, tables, count


def main():
    items = preprocessed()
    roots = list(KERNELS)
    kept, _ = closure([i for i in items if not provided(def_name(i))] + [i for i in items if def_name(i) in KERNELS], roots)
    helpers, launch, tables, count = [], None, None, 0
    for item in kept:
        name = def_name(item)
        if name in KERNELS:
            continue
        if name == "fsbm_launch":
            item, tables, count = launch_body(body_fixes(item))
        helpers.append(convert(item))
        if name == "FsbmNbr":
            helpers.append(NEIGHBOUR_READER)
    assert count == len(STATE_FIELDS), count
    kernels, signatures = [], []
    resources_common = [["dims", "int32", "read"], ["flags", "int32", "read"],
                        ["tab_offsets", "int32", "read"], ["tabf", "float", "read"],
                        ["tabd", "float64_bits", "read"], ["tabi", "int32", "read"],
                        ["work", "float", "read_write"], ["kz", "int32", "read_write"],
                        ["tally", "float64_bits", "read_write"], ["status", "int32", "read_write"]]
    resources_common += [[n, "float", "read_write"] for n in STATE_FIELDS]
    parameters = [["step", "int32"], ["dt", "float32"], ["dx", "float32"],
                  ["dy", "float32"], ["ikr", "int32"]]
    base = {"tabf": "(device const F *)r.tabf", "tabd": "(device const Soft8 *)r.tabd",
            "tabi": "r.tabi"}
    table_lines = []
    for name, kind, index in tables:
        src = {"R": "tabf", "double": "tabd", "int": "tabi"}[kind]
        table_lines.append("    T.%s = %s + r.tab_offsets[%s];" % (name, base[src], index))
    state_items = ", ".join("(device F *)r.%s" % n for n in STATE_FIELDS)
    for name, mode in KERNELS.items():
        title = "".join(w.title() for w in name.split("_"))
        kernels.append("\n".join([
            "kernel void %s(" % name,
            "    constant %sResources &r [[buffer(0)]]," % title,
            "    constant %sParams &params [[buffer(1)]]," % title,
            "    uint gid [[thread_position_in_grid]]) {",
            "    if (gid >= uint(%s)) return;" % (
                "(min(r.dims[1] - 1, r.dims[13] + 1) - max(r.dims[12] - 1, 1) + 1) * (min(r.dims[3] - 1, r.dims[15] + 1) - max(r.dims[14] - 1, 1) + 1)"
                if mode == 3 else "(r.dims[13] - r.dims[12] + 1) * (r.dims[15] - r.dims[14] + 1)"),
            "    FsbmTab<F> T;",
            *table_lines,
            "    T.ikr_spon_break = params.ikr;",
            "    State<F> S = {%s};" % state_items,
            "    fsbm_launch<F>(params.step, F(params.dt), F(params.dx), F(params.dy), long(gid),",
            "        r.dims, T, r.flags, S, (device F *)r.work, r.kz, (device Soft8 *)r.tally, r.status, %d);" % mode,
            "}"]))
        signatures.append({
            "name": name, "cuda_symbol": name, "module": "fast_sbm_family",
            "source": "woof/metal/shaders/fast_sbm_family.metal",
            "cuda_source": "woof/core/kernels/fast_sbm.cu", "resources": resources_common,
            "shapes": {r[0]: [["any_1d"]] for r in resources_common}, "parameters": parameters,
            "threads_per_group": 32, "layout_version": 1, "arithmetic_policies": POLICIES})
    header = "// Generated by tools/fast_sbm_metal/generate.py; do not edit.\n"
    (SHADERS / "fast_sbm_fragment.metalh").write_text(
        header + '#include "fast_sbm_support.metalh"\n#pragma clang fp contract(off)\n'
        + '#pragma clang diagnostic ignored "-Wparentheses-equality"\n'
        + "\n".join(helpers).rstrip() + "\n")
    (SHADERS / "fast_sbm_family.metal").write_text(
        header + '#include <metal_stdlib>\n#include "fast_sbm_fragment.metalh"\n'
        '#include "fast_sbm_family_bindings.metalh"\n\n' + "\n\n".join(kernels) + "\n")
    (ROOT / "woof/metal/signature_fragments/fast_sbm_family.json").write_text(
        json.dumps({"kernels": signatures}, indent=2) + "\n")


if __name__ == "__main__":
    main()
