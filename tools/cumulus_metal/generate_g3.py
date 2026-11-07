#!/usr/bin/env python3
"""Translate the CUDA Grell 3D unit (cu_physics=5) to Metal kernels.

Statements are carried over unchanged: address spaces, the lambdas that
metal cannot express, the float division and square root helpers and the
kernel boundary change and nothing else.  Run from the repository root with
the data companion on PYTHONPATH.
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
from translate import tokenize, untok, map_types, rewrite_division  # noqa: E402

SHADERS = ROOT / "woof/metal/shaders"
DEVICE = ("g3max", "g3min", "g3imax", "g3imin", "g3p2", "g3p3", "g3p4",
          "G3Params", "g3_maximi", "g3_minimi", "g3_massflx_stats")
COLUMN_RESOURCES = (
    ("lev_in", "float", "read"), ("w_in", "float", "read"),
    ("sfc_in", "float", "read"), ("kpbl_in", "int32", "read"),
    ("apr", "float", "read_write"), ("lev_out", "float", "read_write"),
    ("sfc_out", "float", "read_write"), ("isfc_out", "int32", "read_write"),
    ("xf_ens", "float", "read_write"), ("pr_ens", "float", "read_write"),
    ("ws", "float", "read_write"))
COLUMN_PARAMS = (
    ("nt", "int32"), ("col0", "int32"), ("nx", "int32"), ("ny", "int32"),
    ("nz", "int32"), ("ibegc", "int32"), ("iendc", "int32"),
    ("jbegc", "int32"), ("jendc", "int32"), ("ishallow", "int32"),
    ("avedx", "int32"), ("ichoice", "int32"), ("fqi", "int32"),
    ("dt", "float32"), ("dx", "float32"), ("xlv", "float32"),
    ("cp", "float32"), ("g", "float32"), ("rv", "float32"))
SPREAD_RESOURCES = (
    ("cugd", "float", "read"), ("pi_phy", "float", "read"),
    ("qv", "float", "read"), ("rthcuten", "float", "read_write"),
    ("rqvcuten", "float", "read_write"), ("rqccuten", "float", "read_write"),
    ("rqicuten", "float", "read_write"), ("raincv", "float", "read_write"),
    ("pratec", "float", "read_write"), ("ws", "float", "read_write"))
SPREAD_PARAMS = (
    ("nt", "int32"), ("col0", "int32"), ("nx", "int32"), ("ny", "int32"),
    ("nz", "int32"), ("ibegc", "int32"), ("iendc", "int32"),
    ("jbegc", "int32"), ("jendc", "int32"), ("avedx", "int32"),
    ("fqi", "int32"), ("dt", "float32"))

AVE9 = """
static inline float g3_ave9(device const float *f, int i, int j, int k,
                            const thread G3Params &P)
{
    long plane = long(P.nx) * long(P.ny);
    float s[9];
    int n = 0;
    for (int di = -1; di <= 1; ++di) {
        for (int dj = -1; dj <= 1; ++dj) {
            int ii = i + di, jj = j + dj;
            if (P.ibegc == 0 && P.iendc == P.nx - 1) ii = (ii + P.nx) % P.nx;
            if (P.jbegc == 0 && P.jendc == P.ny - 1) jj = (jj + P.ny) % P.ny;
            float v = 0.f;
            if (!(ii < 0 || ii >= P.nx || jj < 0 || jj >= P.ny))
                v = f[long(k) * plane + long(jj) * P.nx + ii];
            s[n++] = v;
        }
    }
    return aa_div(s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7] + s[8], 9.f);
}
"""


def preprocess():
    from woof.core import kernels
    result = subprocess.run(
        ["clang++", "-E", "-P", "-std=c++17", "-x", "c++", "-"],
        input=kernels.module_source("g3").encode(),
        capture_output=True, check=True)
    return split_top(result.stdout.decode())


def remove_lambda(text, name):
    start = text.index(f"auto {name} = [")
    depth = 0
    for j in range(text.index("{", start), len(text)):
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        if depth == 0:
            end = j + 1
            if text[end] == ";":
                end += 1
            return text[:start] + text[end:]
    raise ValueError(name)


def spaces(text):
    text = re.sub(r"\bfloat& pr = ", "device float &pr = ", text)
    text = re.sub(r"(?<![\w&])(float|int)\s*&(?!&)", r"thread \1 &", text)
    text = re.sub(r"\bG3Col\*\s*(\w+)", r"thread G3Col *\1", text)
    text = re.sub(r"\bconst G3Params&", "const thread G3Params &", text)
    text = re.sub(r"\bconst float\*\s*(\w+)\s*=", r"device const float *\1 =", text)
    text = re.sub(r"\bfloat\*\s*(\w+)\s*=", r"device float *\1 =", text)
    return text


def common(text):
    text = text.replace("sqrtf(", "cu_sqrt(").replace("fabsf(", "fabs(")
    return text


def device_item(item):
    item = re.sub(r"\bconst float\* ens\b", "device const float *ens", item)
    item = re.sub(r"\bfloat\* (edtc|axx|apr|prv)\b", r"thread float *\1", item)
    item = re.sub(r"\b(omeg|tx|qx)\[([^\]]+)\]\(", r"g3_colat(\1[\2], ", item)
    item = spaces(item)
    return untok(rewrite_division(map_types(tokenize(common(item)))))


def kernel_body(src):
    return src[src.index("{", src.index(")")) + 1:src.rindex("}")]


def column_kernel(items):
    body = kernel_body(next(i for i in items if def_name(i) == "g3_column"))
    for name in ("slot", "lin"):
        body = remove_lambda(body, name)
    body = re.sub(r"auto XF = \[&\]\(int m\) -> float& \{ return [^;]*; \};",
                  "XFDEF", body)
    body = re.sub(r"auto PR = \[&\]\(int m\) -> float \{ return [^;]*; \};",
                  "PRDEF", body)
    body = body.replace("const int tid = blockIdx.x * blockDim.x + threadIdx.x;",
                        "const int tid = int(gid);")
    body = re.sub(r"\bslot\(([^()]*)\)\(", r"g3_slot(ws, nz, nt, tid, \1).at(", body)
    body = re.sub(r"\b(omeg|tx|qx)\[([^\]]+)\]\(", r"g3_colat(\1[\2], ", body)
    body = re.sub(r"\bslot\(", "g3_slot(ws, nz, nt, tid, ", body)
    body = re.sub(r"\blin\(", "g3_lin(lev_in, lev_stride, plane, P.nx, ", body)
    body = spaces(body)
    body = untok(rewrite_division(map_types(tokenize(common(body)))))
    body = body.replace("XFDEF", "\n#define XF(m) xfc[(nall + (m) - 1) * es]\n")
    body = body.replace("PRDEF", "\n#define PR(m) prc[(nall + (m) - 1) * es]\n")
    head = ["kernel void g3_column(",
            "    constant G3ColumnResources &r [[buffer(0)]],",
            "    constant G3ColumnParams &p [[buffer(1)]],",
            "    uint gid [[thread_position_in_grid]])", "{"]
    for name, kind, access in COLUMN_RESOURCES:
        ctype = "int" if kind == "int32" else "float"
        const = "const " if access == "read" else ""
        head.append(f"    device {const}{ctype} *{name} = r.{name};")
    renames = {"nz": "nz_", "dt": "dt_", "cp": "cp_", "g": "g_", "rv": "rv_"}
    for name, kind in COLUMN_PARAMS:
        ctype = "int" if kind == "int32" else "float"
        head.append(f"    const {ctype} {renames.get(name, name)} = p.{name};")
    return "\n".join(head) + "\n" + body + "\n}\n"


def spread_kernel(items):
    body = kernel_body(next(i for i in items if def_name(i) == "g3_spread"))
    for name in ("at", "rcut"):
        body = remove_lambda(body, name)
    body = body.replace("const int tid = blockIdx.x * blockDim.x + threadIdx.x;",
                        "const int tid = int(gid);")
    body = re.sub(r"\brcut\(", "g3_rcut(plane, P.nx, P.ny, cugd_spread, ", body)
    body = re.sub(r"\bat\(", "g3_at(plane, P.nx, ", body)
    body = spaces(body)
    body = untok(rewrite_division(map_types(tokenize(common(body)))))
    head = ["kernel void g3_spread(",
            "    constant G3SpreadResources &r [[buffer(0)]],",
            "    constant G3SpreadParams &p [[buffer(1)]],",
            "    uint gid [[thread_position_in_grid]])", "{"]
    for name, kind, access in SPREAD_RESOURCES:
        const = "const " if access == "read" else ""
        head.append(f"    device {const}float *{name} = r.{name};")
    renames = {"nz": "nz_", "dt": "dt_"}
    for name, kind in SPREAD_PARAMS:
        ctype = "int" if kind == "int32" else "float"
        head.append(f"    const {ctype} {renames.get(name, name)} = p.{name};")
    return "\n".join(head) + "\n" + body + "\n}\n"


DIAG_RESOURCES = tuple(
    (name, "float", "read") for name in
    ("cloud", "ice", "rain", "t", "p", "qv", "qc", "qi")) + tuple(
    (name, "float", "read_write") for name in
    ("ca", "cb", "ia", "ib", "frac", "ra", "rb")) + (("clock", "int32", "read"),)
DIAG_PARAMS = (("fqi", "int32"), ("nvol", "int32"), ("ncol", "int32"))


def diag_kernel(items):
    body = kernel_body(next(i for i in items if def_name(i) == "g3_diag"))
    body = body.replace("long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;",
                        "long c = long(gid);")
    body = untok(rewrite_division(map_types(tokenize(body))))
    head = ["kernel void g3_diag(",
            "    constant G3DiagResources &r [[buffer(0)]],",
            "    constant G3DiagParams &prm [[buffer(1)]],",
            "    uint gid [[thread_position_in_grid]])", "{"]
    for name, kind, access in DIAG_RESOURCES:
        ctype = "int" if kind == "int32" else "float"
        const = "const " if access == "read" else ""
        head.append(f"    device {const}{ctype} *{name} = r.{name};")
    head += ["    const int fqi = prm.fqi;", "    const long nvol = prm.nvol;",
             "    const long ncol = prm.ncol;"]
    return "\n".join(head) + "\n" + body + "\n}\n"


HELPERS = r"""
kernel void g3_pack_levels(
    constant G3PackLevelsResources &r [[buffer(0)]],
    constant G3PackLevelsParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= p.ncell) return;
    const ulong c = ulong(gid);
    const ulong n = ulong(p.ncell);
    r.lev_in[0 * n + c] = r.u[c];
    r.lev_in[1 * n + c] = r.v[c];
    r.lev_in[2 * n + c] = r.t[c];
    r.lev_in[3 * n + c] = r.q[c];
    r.lev_in[4 * n + c] = r.pressure[c];
    r.lev_in[5 * n + c] = r.exner[c];
    r.lev_in[6 * n + c] = r.rho[c];
    float rthraten = r.rthlw[c] + r.rthsw[c];
    float thften = (r.thdyn[c] + rthraten + r.thbl[c]) * r.exner[c];
    float qvften = r.qvdyn[c] + r.qvbl[c];
    r.lev_in[7 * n + c] = thften;
    r.lev_in[8 * n + c] = qvften;
    r.lev_in[9 * n + c] = r.thbl[c];
    r.lev_in[10 * n + c] = r.qvbl[c];
}

kernel void g3_diag_finish(
    constant G3DiagFinishResources &r [[buffer(0)]],
    constant G3DiagFinishParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid == 0 && r.clock[3]) r.clock[1] = 0;
}

kernel void g3_pack_surface(
    constant G3PackSurfaceResources &r [[buffer(0)]],
    constant G3PackSurfaceParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]])
{
    if (int(gid) >= p.ncol) return;
    const ulong c = ulong(gid);
    const ulong n = ulong(p.ncol);
    r.sfc_in[0 * n + c] = r.p8w[c];
    r.sfc_in[1 * n + c] = r.ht[c];
    r.sfc_in[2 * n + c] = r.xland[c];
    r.sfc_in[3 * n + c] = r.gsw[c];
    r.sfc_out[2 * n + c] = 1.f;
    r.sfc_out[3 * n + c] = float(p.nz);
}
"""


def signatures():
    common_ = dict(module="g3_family", source="woof/metal/shaders/g3_family.metal",
                   cuda_source="woof/core/kernels/g3.cu", layout_version=1,
                   arithmetic_policies=["soft32-rn-nonftz"], threads_per_group=64)

    def entry(name, symbol, resources, params):
        return dict(
            name=name, cuda_symbol=symbol,
            resources=[list(r) for r in resources],
            shapes={r[0]: [["any_1d"]] for r in resources},
            parameters=[list(p) for p in params], **common_)
    names = ("u", "v", "t", "q", "pressure", "exner", "rho", "thdyn", "qvdyn",
             "rthlw", "rthsw", "thbl", "qvbl")
    return [
        entry("g3_column", "g3_column", COLUMN_RESOURCES, COLUMN_PARAMS),
        entry("g3_spread", "g3_spread", SPREAD_RESOURCES, SPREAD_PARAMS),
        entry("g3_diag", "g3_diag", DIAG_RESOURCES, DIAG_PARAMS),
        entry("g3_diag_finish", "g3_diag_finish",
              [("clock", "int32", "read_write")], [("unused", "int32")]),
        entry("g3_pack_levels", "g3_column",
              [(n, "float", "read") for n in names]
              + [("lev_in", "float", "read_write")], [("ncell", "int32")]),
        entry("g3_pack_surface", "g3_column",
              [("p8w", "float", "read"), ("ht", "float", "read"),
               ("xland", "float", "read"), ("gsw", "float", "read"),
               ("sfc_in", "float", "read_write"),
               ("sfc_out", "float", "read_write")],
              [("ncol", "int32"), ("nz", "int32")])]


def main():
    items = preprocess()
    out = []
    for item in items:
        name = def_name(item)
        if name in DEVICE or name and name.startswith("g3_cup_"):
            out.append(device_item(item))
        elif name is None and item.startswith("enum"):
            out.append(item)
        elif name == "G3S_tx":
            out.append(item)
        elif name == "g3_ave9":
            out.append(AVE9)
    text = ("// Generated by tools/cumulus_metal/generate_g3.py from g3.cu; "
            "do not edit.\n"
            '#pragma clang diagnostic ignored "-Wunused-variable"\n'
            '#pragma clang diagnostic ignored "-Wparentheses-equality"\n'
            + "\n".join(out) + "\n")
    (SHADERS / "g3_fragment.metalh").write_text(text)
    (SHADERS / "g3_family.metal").write_text(
        "// Generated by tools/cumulus_metal/generate_g3.py; do not edit.\n"
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        '#include "g3_family_bindings.metalh"\n'
        '#include "g3_support.metalh"\n'
        '#include "g3_fragment.metalh"\n'
        + column_kernel(items) + spread_kernel(items) + diag_kernel(items) + HELPERS)
    (ROOT / "woof/metal/signature_fragments/g3_family.json").write_text(
        json.dumps({"kernels": signatures()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
