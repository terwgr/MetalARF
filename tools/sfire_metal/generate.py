#!/usr/bin/env python3
"""Generate the SFIRE Metal shaders and signature fragments from the CUDA units.

    python tools/sfire_metal/generate.py core phys ...

Each CUDA translation unit is preprocessed exactly as the loader hands it to
nvrtc and the definitions its kernels reach are translated statement for
statement: binary32 is ``F`` (soft32 operators), binary64 is ``Soft8``,
pointer parameters become address-space templates and the glibc float libm
functions are the shared Metal ports in sfire_support.metalh.
"""
import json
import re
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "mp28_metal"))

from parse import closure, def_name, split_top  # noqa: E402
from translate import (TOKEN, match_fwd, split_params, tokenize,  # noqa: E402
                       template_pointer_params, untok)

SHADERS = ROOT / "woof/metal/shaders"
FRAGMENTS = ROOT / "woof/metal/signature_fragments"

PROVIDED = {"gfk_exp", "gfk_log", "gfk_pow", "glibc_sinf", "glibc_cosf",
            "glibc_asinf", "glibc_pow"}
RENAME = {"__int_as_float": "sf_int_as_float", "__double2float_rn": "sf_double2float_rn"}
DROP = {"__device__", "__forceinline__", "__restrict__", "__global__",
        "__noinline__", "__host__"}


def f32_bits(text):
    """Binary32 word of a C float literal, rounded once from its exact value."""
    low = text.lower()
    value = Fraction(float.fromhex(text)) if low.startswith("0x") else Fraction(text)
    if value == 0:
        return 0
    num, den = value.numerator, value.denominator
    e = num.bit_length() - den.bit_length()
    if Fraction(2) ** e > value:
        e -= 1
    e = max(e, -126)
    scaled = value / Fraction(2) ** (e - 23)
    m = scaled.numerator // scaled.denominator
    rest = scaled - m
    if rest > Fraction(1, 2) or (rest == Fraction(1, 2) and m & 1):
        m += 1
    if m == 1 << 24:
        m >>= 1
        e += 1
    if e > 127:
        return 0x7f800000
    if m < 1 << 23:
        return m
    return ((e + 127) << 23) | (m & 0x7fffff)


def convert_number(text):
    low = text.lower()
    is_hex = low.startswith("0x")
    floaty = (("p" in low) or ("." in low)) if is_hex else (("." in low) or ("e" in low))
    if floaty:
        if low.endswith("f"):
            return "SF(0x%08xu)" % f32_bits(text[:-1])
        t = text.rstrip("lL")
        value = float.fromhex(t) if is_hex else float(t)
        return "SD8(0x%016xul)" % struct.unpack("<Q", struct.pack("<d", value))[0]
    m = re.match(r"(.*?)([uUlL]*)$", text)
    body, suffix = m.group(1), m.group(2).lower()
    u, ll = "u" in suffix, suffix.count("l")
    if ll:
        return body + ("ul" if u else "l")
    return body + ("u" if u else "")


def map_types(tokens):
    out = []
    i, n = 0, len(tokens)
    while i < n:
        kind, text = tokens[i]
        if text in DROP:
            i += 1
            continue
        if text == "extern" and i + 1 < n and tokens[i + 1][0] == "str":
            i += 2
            continue
        if text == "unsigned":
            j = i + 1
            words = []
            while j < n and tokens[j][1] in ("long", "int", "short", "char"):
                words.append(tokens[j][1])
                j += 1
            out.append(("id", "ulong" if "long" in words else "uint"))
            i = j
            continue
        if text == "long":
            j = i + 1
            while j < n and tokens[j][1] in ("long", "int"):
                j += 1
            out.append(("id", "long"))
            i = j
            continue
        if text == "float":
            out.append(("id", "F"))
        elif text == "double":
            out.append(("id", "Soft8"))
        elif kind == "num":
            out.append(("num", convert_number(text)))
        elif text in RENAME:
            out.append(("id", RENAME[text]))
        else:
            out.append((kind, text))
        i += 1
    return out


def rewrite_pointer_locals(tokens):
    """``const float *a = e, *b = e2;`` -> ``auto a = e; auto b = e2;``."""
    out = []
    i, n = 0, len(tokens)
    starts = {";", "{", "}", ")"}
    while i < n:
        j = i
        if tokens[j][1] == "const":
            j += 1
        if (j + 2 < n and tokens[j][1] in ("F", "int", "uint", "ulong")
                and tokens[j + 1][1] == "*" and tokens[j + 2][0] == "id"
                and (i == 0 or tokens[i - 1][1] in starts)
                and any(t[1] == "=" for t in tokens[j + 3:j + 4])):
            end = j
            depth = 0
            while True:
                t = tokens[end][1]
                if t in "([{":
                    depth += 1
                elif t in ")]}":
                    depth -= 1
                elif t == ";" and depth == 0:
                    break
                end += 1
            decls, cur, depth = [], [], 0
            for t in tokens[j + 1:end]:
                if t[1] in "([{":
                    depth += 1
                elif t[1] in ")]}":
                    depth -= 1
                if t[1] == "," and depth == 0:
                    decls.append(cur)
                    cur = []
                else:
                    cur.append(t)
            decls.append(cur)
            for d in decls:
                assert d[0][1] == "*", d
                out.extend([("id", "auto")] + d[1:] + [("op", ";")])
            i = end + 1
            continue
        out.append(tokens[i])
        i += 1
    return out


def normalise(text):
    return untok(tokenize(text))


def apply_overrides(text, overrides):
    text = normalise(text)
    for old, new in overrides:
        old, new = normalise(old), normalise(new)
        if text.count(old) != 1:
            raise ValueError("override matches %d times: %s" % (text.count(old), old[:80]))
        text = text.replace(old, new)
    return text


def preprocessed(text):
    done = subprocess.run(["clang++", "-E", "-P", "-std=c++17", "-x", "c++", "-"],
                          input=text.encode(), capture_output=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.decode()[-3000:])
    return done.stdout.decode()


def kind_of(item):
    if item.startswith('extern "C"'):
        return "kernel"
    if item.startswith("struct"):
        return "struct"
    return "function"


def parse_header(toks):
    paren = next(k for k, t in enumerate(toks) if t[1] == "(")
    close = match_fwd(toks, paren)
    return toks[paren - 1][1], paren, close


def kernel_info(item):
    toks = tokenize(item)
    name, paren, close = parse_header(toks)
    params = []
    for p in split_params(toks, paren, close):
        words = [t[1] for t in p if t[1] not in DROP]
        pointer = "*" in words
        ident = words[-1]
        base = [w for w in words[:-1] if w not in ("const", "*")]
        params.append({"name": ident, "pointer": pointer, "const": "const" in words,
                       "base": " ".join(base)})
    return name, params


def convert_item(item, kernel=False):
    toks = map_types(tokenize(item))
    if kernel:
        out, i = [], 0
        while i < len(toks):
            if toks[i][1] in ("blockIdx", "blockDim", "threadIdx") and toks[i + 1][1] == ".":
                out.append(("num", {"blockIdx": "0", "blockDim": "1", "threadIdx": "gid"}[toks[i][1]]))
                i += 3
                continue
            out.append(toks[i])
            i += 1
        toks = out
    toks = rewrite_pointer_locals(toks)
    return toks


def convert_function(item):
    toks = template_pointer_params(convert_item(item))
    if toks[0][1] == "template":
        close = next(k for k, t in enumerate(toks) if t[1] == ">")
        return untok(toks[:close + 1]) + " static inline " + untok(toks[close + 1:])
    return "static inline " + untok(toks)


def convert_struct(item):
    return untok(map_types(tokenize(item)))


def convert_kernel(item, name):
    toks = convert_item(item, kernel=True)
    paren = next(k for k, t in enumerate(toks) if t[1] == "(")
    close = match_fwd(toks, paren)
    toks[paren - 1] = ("id", name + "_body")
    toks[close:close] = [("op", ","), ("id", "uint"), ("id", "gid")]
    toks = template_pointer_params(toks)
    close = next(k for k, t in enumerate(toks) if t[1] == ">")
    if toks[0][1] == "template":
        return untok(toks[:close + 1]) + " static inline " + untok(toks[close + 1:])
    return "static inline " + untok(toks)


def title(name):
    return "".join(part.title() for part in name.split("_"))


def wrapper(name, params, extras):
    args, resources, scalars = [], [], []
    for p in params:
        if p["pointer"]:
            access = "read" if p["const"] else "read_write"
            storage = {"float": "float", "int": "int32"}[p["base"]]
            resources.append([p["name"], storage, access])
            if p["base"] == "float":
                args.append("(%sdevice F *)r.%s" % ("const " if p["const"] else "", p["name"]))
            else:
                args.append("r.%s" % p["name"])
        else:
            kind = {"float": "float32", "int": "int32"}[p["base"]]
            scalars.append([p["name"], kind])
            args.append("F(p.%s)" % p["name"] if kind == "float32" else "p.%s" % p["name"])
    for extra in extras:
        scalars.append([extra, "int32"])
        args.append("p.%s" % extra)
    pad = not scalars
    if pad:
        scalars.append(["unused", "int32"])
    text = ("kernel void %s(constant %sResources &r [[buffer(0)]], constant %sParams &p [[buffer(1)]],\n"
            "        uint gid [[thread_position_in_grid]]) {\n    %s_body(%s);\n}\n"
            % (name, title(name), title(name), name, ", ".join(args + ["gid"])))
    return text, resources, scalars


def generate(stem, source_text, cuda_source, overrides=(), extras=None, single=(),
             policies=("soft32-rn-nonftz",)):
    extras = extras or {}
    text = apply_overrides(preprocessed(source_text), overrides)
    items = []
    for it in split_top(text):
        if it.startswith(("namespace", "typedef", "#pragma")):
            continue
        if def_name(it) in PROVIDED:
            continue
        items.append(it)
    kernels = {def_name(it): it for it in items if kind_of(it) == "kernel"}
    kept, _ = closure([it for it in items if kind_of(it) != "kernel"],
                      sorted({w for k in kernels.values() for w in re.findall(r"[A-Za-z_]\w*", k)}))
    body = ['// Generated by tools/sfire_metal/generate.py from %s.' % cuda_source,
            '// Statements follow the CUDA unit; binary32 is soft32 (F), binary64 Soft8.',
            '#include <metal_stdlib>',
            '#include "sfire_support.metalh"',
            '#include "%s_bindings.metalh"' % stem,
            'using namespace metal;', '']
    for it in kept:
        body.append(convert_struct(it) if kind_of(it) == "struct" else convert_function(it))
        body.append("")
    entries = []
    for name, it in kernels.items():
        _, params = kernel_info(it)
        body.append(convert_kernel(it, name))
        wrap, resources, scalars = wrapper(name, params, extras.get(name, ()))
        body.append(wrap)
        entries.append({
            "name": name, "cuda_symbol": name, "module": stem,
            "source": "woof/metal/shaders/%s.metal" % stem, "cuda_source": cuda_source,
            "resources": resources,
            "shapes": {r[0]: [["any_1d"]] for r in resources},
            "parameters": scalars,
            "threads_per_group": 128, "layout_version": 1,
            "arithmetic_policies": list(policies)})
    (SHADERS / ("%s.metal" % stem)).write_text("\n".join(body) + "\n", encoding="utf-8")
    FRAGMENTS.mkdir(exist_ok=True)
    (FRAGMENTS / ("%s.json" % stem)).write_text(
        json.dumps({"kernels": entries}, indent=2) + "\n", encoding="utf-8")
    return [e["name"] for e in entries]


def units():
    from woof.core import (sfire_atm, sfire_core, sfire_coupler, sfire_ideal,
                           sfire_ideal_atmos, sfire_moisture, sfire_phys,
                           sfire_spotting, sfire_wind)
    nearest = ("if(q<n) sf_nearest(d[q],t[q],ax[q],ay[q],sx,sy,st,ex,ey,et,cx2,cy2);",
               "if(q<n) { float dd,tt; sf_nearest(dd,tt,ax[q],ay[q],sx,sy,st,ex,ey,et,cx2,cy2); d[q]=dd; t[q]=tt; }")
    fuel = ("if(q<n) sf_fuel_cell(fuel[q],area[q],l+4*q,t+4*q,now,tau[q]);",
            "if(q<n) { float ff,af; sf_fuel_cell(ff,af,l+4*q,t+4*q,now,tau[q]); fuel[q]=ff; area[q]=af; }")
    ros = ("sfire_ros_device(base[idx],wind[idx],slope[idx],px[idx],py[idx],vx[idx],vy[idx],\n"
           "                     dzdx[idx],dzdy[idx],bbb[idx],betafl[idx],phiwc[idx],r0[idx],ischap[idx],advection);",
           "float rb,rw,rs; sfire_ros_device(rb,rw,rs,px[idx],py[idx],vx[idx],vy[idx],\n"
           "                     dzdx[idx],dzdy[idx],bbb[idx],betafl[idx],phiwc[idx],r0[idx],ischap[idx],advection);\n"
           "base[idx]=rb; wind[idx]=rw; slope[idx]=rs;")
    termvel = ("if(mode!=0)sfire_brand_termvel(p,height[q],dt,pres[q],aird[q],density,density_char);",
               "if(mode!=0){float hq=height[q];sfire_brand_termvel(p,hq,dt,pres[q],aird[q],density,density_char);height[q]=hq;}")
    sample = ("sfire_brand_uvw(x,y,zp,u,v,w,nx,ny,ml,mb,out[n+q],out[2*n+q],out[3*n+q]);\n"
              "    sfire_brand_met(x,y,zp,p,t,d,nx,ny,ml,mb,ihs,ihe,jhs,jhe,out[4*n+q],out[5*n+q],out[6*n+q]);",
              "{float ru,rv,rw,lp,lt,ld;\n"
              "    sfire_brand_uvw(x,y,zp,u,v,w,nx,ny,ml,mb,ru,rv,rw);\n"
              "    sfire_brand_met(x,y,zp,p,t,d,nx,ny,ml,mb,ihs,ihe,jhs,jhe,lp,lt,ld);\n"
              "    out[n+q]=ru;out[2*n+q]=rv;out[3*n+q]=rw;out[4*n+q]=lp;out[5*n+q]=lt;out[6*n+q]=ld;}")
    stage = ("int plane,int mapped) {", "int plane,int mapped,int wrf_exact) {")
    stage2 = ("rt[p]=mapped?__fdiv_rn(th[p],msft[p%plane]):th[p];",
              "rt[p]=(mapped&&!wrf_exact)?__fdiv_rn(th[p],msft[p%plane]):th[p];")
    return {
        "sfire_core": (sfire_core.source(), "woof/core/kernels/sfire_core.cu", [nearest, fuel], {}, ()),
        "sfire_phys": (sfire_phys.module_source(), "woof/core/kernels/sfire_phys.cu", [ros], {}, ()),
        "sfire_moisture": (sfire_moisture.module_source(), "woof/core/kernels/sfire_moisture.cu", [], {}, ()),
        "sfire_wind": (sfire_wind.module_source(), "woof/core/kernels/sfire_wind.cu", [], {}, ()),
        "sfire_atm": (sfire_atm.module_source(), "woof/core/kernels/sfire_atm.cu", [], {}, ()),
        "sfire_coupling": (sfire_coupler.module_source(), "woof/core/kernels/sfire_coupling.cu",
                           [stage2, stage], {}, ()),
        "sfire_spotting": (sfire_spotting.module_source(), "woof/core/kernels/sfire_spotting.cu",
                           [termvel, sample], {}, ()),
        "sfire_ideal": (sfire_ideal.module_source(), "woof/core/kernels/sfire_ideal.cu", [], {}, ()),
        "sfire_ideal_atmos": (sfire_ideal_atmos.module_source(),
                              "woof/core/kernels/sfire_ideal_atmos.cu", [], {}, ()),
    }


def main(argv):
    table = units()
    for stem in argv or sorted(table):
        src, cuda, overrides, extras, single = table[stem]
        names = generate(stem, src, cuda, overrides, extras, single,
                         policies=("soft32-rn-nonftz", "soft64-rn-bit-ops", "glibc-exp"))
        print(stem, len(names), "kernels")


if __name__ == "__main__":
    main(sys.argv[1:])
