"""Translate the strict regional km_opt=4 coefficient source to integer FP32.

Only types, address spaces and the thread-index ABI change. The exact CUDA
preprocessor branches retain their expression trees and binary32 temporaries.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.mp28_metal.parse import closure, def_name, split_top
from tools.thompson_real_column_parity.host_backend import parse_kernels

DEFINES = (("GPUWM_WRF_EXACT", 1), ("GPUWM_WRF_EXACT_C_DIFFUSION", 1))
KERNELS = ("wrf_smag_deform", "wrf_smag2d_km", "wrf_smag_km_bc")
STEM = "wrf_exact_smag_coefficients"
TENDENCY_KERNELS = ("wrf_smag_hd_u", "wrf_smag_hd_v", "wrf_smag_hd_w",
                    "wrf_smag_flux_s", "wrf_smag_hd_s")
TENDENCY_STEM = "wrf_exact_smag_tendencies"
VERTICAL_KERNELS = ("wrf_smag_vd_u", "wrf_smag_vd_v", "wrf_smag_vd_w", "wrf_smag_vd_s",
                    "wrf_smag_surface_u", "wrf_smag_surface_v",
                    "wrf_smag_surface_u_cd0", "wrf_smag_surface_v_cd0",
                    "wrf_smag_surface_scalars", "wrf_smag_surface_heat_const")
VERTICAL_STEM = "wrf_exact_smag_vertical"
TURBULENCE_KERNELS = ("wrf_calc_n2", "wrf_smag3d_km", "wrf_tke_km", "wrf_tke_rhs")
TURBULENCE_STEM = "wrf_exact_smag_turbulence"
EXTERNAL_MATH = {"gfk_pow", "gfk_exp", "gfk_log"}
ZERO_STRIPS = "wrf_exact_smag_zero_boundary_strips"
# Launch extents of each tendency kernel in (z, y, x) cells.
EXTENTS = {"wrf_smag_hd_u": ("nz", "ny", "nx + 1"),
           "wrf_smag_hd_v": ("nz", "ny + 1", "nx"),
           "wrf_smag_hd_w": ("nz + 1", "ny", "nx"),
           "wrf_smag_flux_s": ("nz", "ny + 1", "nx + 1"),
           "wrf_smag_hd_s": ("nz", "ny", "nx"),
           "wrf_smag_vd_u": ("nz", "ny", "nx + 1"),
           "wrf_smag_vd_v": ("nz", "ny + 1", "nx"),
           "wrf_smag_vd_w": ("nz + 1", "ny", "nx"),
           "wrf_smag_surface_u": ("1", "ny", "nx + 1"),
           "wrf_smag_surface_v": ("1", "ny + 1", "nx"),
           "wrf_smag_surface_u_cd0": ("1", "ny", "nx + 1"),
           "wrf_smag_surface_v_cd0": ("1", "ny + 1", "nx"),
           "wrf_smag_surface_scalars": ("1", "ny", "nx"),
           "wrf_smag_surface_heat_const": ("1", "ny", "nx")}


def source():
    from woof.core.kernels import module_source_int_defines
    return module_source_int_defines("smag2d", DEFINES)


def retained_source(kernels=KERNELS, *, external_math=False):
    """The source closure also serves the independently compiled host oracle."""
    done = subprocess.run(
        ["clang++", "-E", "-P", "-std=c++17", "-x", "c++", "-"],
        input=source(), text=True, capture_output=True, check=True)
    items = split_top(done.stdout)
    if external_math:
        # Keep the science expression trees, resolving only these pinned
        # scalar function boundaries through their integer translations.
        items = [item for item in items if def_name(item) not in EXTERNAL_MATH]
    items, _ = closure(items, kernels)
    return items


def exact_name(name):
    return "wrf_exact_bound_tke" if name == "bound_tke" else name.replace("wrf_", "wrf_exact_", 1)


def level_name(body):
    """Return the vertical thread index, or None for a surface kernel."""
    match = re.search(r"int (k|kw) = blockIdx\.z", body)
    return match.group(1) if match else None


def translate(item, kernels=KERNELS):
    name = def_name(item)
    if name in kernels:
        item = re.sub(r'extern\s+"C"\s+__global__\s*', "", item)
        pos = item.index("{")
        head, body = item[:pos].rstrip(), item[pos:]
        level = level_name(body)
        head = head[:-1] + ", int i, int j" + (", int " + level if level else "") + ")\n"
        body = re.sub(r"int (?:i|j|k|kw) = blockIdx\.[xyz] \* blockDim\.[xyz] \+ threadIdx\.[xyz];", "", body)
        item = head + body
    item = re.sub(r"\b(__device__|__forceinline__|__restrict__)\b", "", item)
    item = re.sub(r"\bconst real\s*\*", "device const Exact32*", item)
    item = re.sub(r"\breal\s*\*", "device Exact32*", item)
    item = re.sub(r"\breal\b", "Exact32", item)
    item = re.sub(r"\bsize_t\b", "ulong", item)
    item = re.sub(r"\bconst (WrfSmagGrid|WrfScalarField|WrfN2Column)&", r"thread const \1&", item)
    for before, after in {
            "__fadd_rn": "exact_add", "__fsub_rn": "exact_sub",
            "__fmul_rn": "exact_mul", "__fdiv_rn": "exact_div",
            "sqrtf": "exact_sqrt", "fabsf": "exact_abs",
            "fminf": "exact_min", "fmaxf": "exact_max",
            "gfk_pow": "exact_gfk_pow", "gfk_exp": "exact_gfk_exp",
            "gfk_log": "exact_gfk_log"}.items():
        item = re.sub(r"\b" + before + r"\b", after, item)
    return item


ZERO_STRIPS_SOURCE = """kernel void wrf_exact_smag_zero_boundary_strips(
    constant WrfExactSmagZeroBoundaryStripsResources &r [[buffer(0)]],
    constant WrfExactSmagZeroBoundaryStripsParams &p [[buffer(1)]],
    uint flat [[thread_position_in_grid]]) {
    if (int(flat) >= p.size) return;
    const int i = int(flat) % p.nx;
    const int j = (int(flat) / p.nx) % p.ny;
    const bool zero_x = p.boundary_x &&
        (i < p.width || p.x_hi == 0 || i >= p.nx - p.x_hi);
    const bool zero_y = p.boundary_y &&
        (j < p.width || p.y_hi == 0 || j >= p.ny - p.y_hi);
    if (zero_x || zero_y) reinterpret_cast<device uint *>(r.buf)[flat] = 0u;
}"""


# The production bound_tke caller uses CuPy clip's ordered comparisons.
# These retain the input word for signed zero and unordered values.
BOUND_TKE_SOURCE = """kernel void wrf_exact_bound_tke(
    constant WrfExactBoundTkeResources &r [[buffer(0)]],
    constant WrfExactBoundTkeParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    device uint *words = reinterpret_cast<device uint *>(r.tke);
    const uint value = words[gid], lower = 0u, upper = as_type<uint>(p.upper);
    words[gid] = soft32_greater(lower, upper) ? upper
        : (soft32_less(value, lower) ? lower
           : (soft32_greater(value, upper) ? upper : value));
}"""


# The TKE caller assembles mut with one CuPy add ufunc before the source
# kernel. Match that boundary's flush-to-zero operands and result.
TKE_MASS_SOURCE = """kernel void wrf_exact_tke_mass(
    constant WrfExactTkeMassResources &r [[buffer(0)]],
    constant WrfExactTkeMassParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= uint(p.n)) return;
    uint a = as_type<uint>(r.mub[gid]), b = as_type<uint>(r.mup[gid]);
    if ((a & 0x7f800000u) == 0u) a &= 0x80000000u;
    if ((b & 0x7f800000u) == 0u) b &= 0x80000000u;
    uint value = soft32_add(a, b);
    if ((value & 0x7f800000u) == 0u) value &= 0x80000000u;
    reinterpret_cast<device uint *>(r.mut)[gid] = value;
}"""


def wrap(expression):
    return "(" + expression + ")" if "+" in expression else expression


def emit(stem, kernels, extents=None, strips=False, *, external_math=False, bound=False):
    """Write one exact module: translated closure, wrappers and signatures."""
    items = retained_source(kernels, external_math=external_math)
    params = parse_kernels("\n".join(items))
    definitions = [translate(item, kernels) for item in items if def_name(item) != "real"]
    if re.search(r"\b(float|double)\b", "\n".join(definitions)):
        raise ValueError("untranslated floating type in the strict source closure")
    base = json.loads((ROOT / "woof/metal/signature_fragments/smag2d_family.json").read_text())
    registry = json.loads((ROOT / "woof/metal/kernel_signatures.json").read_text())
    by_name = {k["name"]: k for k in registry["kernels"]}
    by_name.update({k["name"]: k for k in base["kernels"]})
    namespace = "wrf_exact_" + stem.rsplit("_", 1)[1][:5]
    signatures = []
    wrappers = []
    for name in kernels:
        sig = copy.deepcopy(by_name[name])
        sig.update(name=exact_name(name), module=stem,
                   source="woof/metal/shaders/" + stem + ".metal",
                   arithmetic_policies=["wrf-exact"])
        signatures.append(sig)
        title = "".join(part.title() for part in sig["name"].split("_"))
        aliases = {"d11a": "d11", "d22a": "d22", "d12a": "d12"}
        if name == "wrf_tke_km":
            aliases["tke_seed"] = "seed"
        if name == "wrf_tke_rhs":
            aliases.update(kmh_a="xkmh", kmv_a="xkmv", khv_a="xkhv",
                           use_ustm="has_ustm", use_hfx="has_hfx", cd0="drag",
                           heat_flux_const="heat", tend="rtke", budget="budget_on",
                           b_shear="shear", b_buoy="buoyancy", b_diss="dissipation",
                           b_lim="limiter")
        args = []
        for param in params[name]:
            field = aliases.get(param.name, param.name)
            if param.pointer:
                const = "const " if "const" in param.text.split() else ""
                args.append("reinterpret_cast<device " + const + "Exact32*>(r." + field + ")")
            else:
                args.append(("Exact32(p." + field + ")") if param.base in ("float", "real") else "p." + field)
        ez, ey, ex = (re.sub(r"\b(n[xyz])\b", r"p.\1", e) for e in (extents or {}).get(name, ("nz", "ny", "nx")))
        level = level_name(next(item for item in items if def_name(item) == name))
        wrappers.append("\n".join([
            "kernel void " + sig["name"] + "(",
            "    constant " + title + "Resources &r [[buffer(0)]],",
            "    constant " + title + "Params &p [[buffer(1)]],",
            "    uint gid [[thread_position_in_grid]]) {",
            "    if (ulong(gid) >= ulong(" + ez + ") * ulong(" + ey + ") * ulong(" + ex + ")) return;",
            "    const int i = int(gid % uint(" + ex + "));",
            "    const int j = int((gid / uint(" + ex + ")) % uint(" + ey + "));",
            *(["    const int " + level + " = int(gid / uint(" + wrap(ex) + " * " + wrap(ey) + "));"] if level else []),
            "    " + namespace + "::" + name + "(" + ", ".join(args) + ", i, j" + (", " + level if level else "") + ");",
            "}"]))
    if strips:
        sig = copy.deepcopy(next(k for k in base["kernels"] if k["name"] == "smag_zero_boundary_strips"))
        sig.update(name=ZERO_STRIPS, module=stem,
                   source="woof/metal/shaders/" + stem + ".metal",
                   arithmetic_policies=["wrf-exact"])
        signatures.append(sig)
        wrappers.append(ZERO_STRIPS_SOURCE)
    if bound:
        sig = copy.deepcopy(by_name["bound_tke"])
        sig.update(name=exact_name("bound_tke"), module=stem,
                   source="woof/metal/shaders/" + stem + ".metal",
                   arithmetic_policies=["wrf-exact"])
        signatures.append(sig)
        wrappers.append(BOUND_TKE_SOURCE)
        signatures.append(dict(
            name="wrf_exact_tke_mass", module=stem,
            source="woof/metal/shaders/" + stem + ".metal",
            cuda_source="woof/core/dycore.py", cuda_symbol="launch_wrf_tke_km",
            resources=[["mub", "float", "read"], ["mup", "float", "read"],
                       ["mut", "float", "read_write"]],
            shapes={name: [["n"]] for name in ("mub", "mup", "mut")},
            parameters=[["n", "int32"]], threads_per_group=128,
            layout_version=1, arithmetic_policies=["wrf-exact"]))
        wrappers.append(TKE_MASS_SOURCE)
    generated = "// Generated by tools/smag_wrf_exact/generate.py; do not edit.\n"
    shader = (generated + '#include <metal_stdlib>\n#include "exact_binary32.metalh"\n'
              + ('#include "exact_gfk_math.metalh"\n' if external_math else '')
              + '#include "' + stem + '_bindings.metalh"\n'
              'using namespace metal;\nnamespace ' + namespace + ' {\n'
              + "\n".join(definitions) + "\n}\n" + "\n".join(wrappers) + "\n")
    shader = "\n".join(line.rstrip() for line in shader.splitlines()) + "\n"
    (ROOT / "woof/metal/shaders" / (stem + ".metal")).write_text(shader)
    (ROOT / "woof/metal/signature_fragments" / (stem + ".json")).write_text(
        json.dumps({"kernels": signatures}, indent=2) + "\n")


def main():
    emit(STEM, KERNELS)
    emit(TENDENCY_STEM, TENDENCY_KERNELS, EXTENTS, strips=True)
    emit(VERTICAL_STEM, VERTICAL_KERNELS, EXTENTS)
    emit(TURBULENCE_STEM, TURBULENCE_KERNELS, external_math=True, bound=True)


if __name__ == "__main__":
    main()
