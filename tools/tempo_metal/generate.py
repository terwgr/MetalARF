#!/usr/bin/env python3
"""Generate the Metal translation of the TEMPO column (mp_physics=900).

    python tools/tempo_metal/generate.py

The CUDA unit is the one woof/core/tempo_runtime.py hands to NVRTC (its
quoted includes expanded, the device branches of tempo_common.cuh selected),
preprocessed, and the definitions the column kernel reaches are translated
statement for statement with the mp=28 machinery in tools/mp28_metal: binary64
is Soft8 on the soft64 helpers, float division runs on the soft32 divider and
the rounded intrinsics are plain IEEE operations.  The lookup blob stays one
flat array of binary64 words; TEMPO_FIELDS gives each named field its offset.
The pack, column, unpack and size kernels are written here.  The unit is
instantiated per level-capacity tier, as NVRTC compiles it per level count.
"""
import ast
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
from translate import tokenize, untok  # noqa: E402

KERNEL_DIR = ROOT / "woof/core/kernels/tempo"
SHADERS = ROOT / "woof/metal/shaders"
FRAGMENTS = ROOT / "woof/metal/signature_fragments"
#: Shader files and the level capacities each one instantiates.
FAMILIES = {"tempo_family": (64, 128, 256), "tempo_family_deep": (512,)}


def family_of(tier):
    return next(f for f, tiers in FAMILIES.items() if tier in tiers)
ROOTS = ["tempo_main_column", "tempo_subgrid_cloud_radii"]
QUOTED = re.compile(r'^[ \t]*#[ \t]*include[ \t]+"([^"]+)"[ \t]*$', re.M)
NFLD = 17
STATE_FIELDS = ("t", "p", "w", "dz", "qv", "qc", "qr", "qi", "qs", "qg", "ni",
                "nr", "nc", "nwfa", "nifa", "ng", "qb")
UPDATED_FIELDS = ("t", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc",
                  "nwfa", "nifa", "ng", "qb")


DEVICE_OPS = "".join(
    "static inline device Soft8 &operator%s=(device Soft8 &a, Soft8 b) "
    "{ a = a %s b; return a; }\n" % (o, o) for o in "+-*/")


def kernel_text():
    tree = ast.parse((ROOT / "woof/core/tempo_runtime.py").read_text())
    node = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", "") == "_KERNEL")
    # hypot and atan2 reach TEMPO through the complex radar helpers; the
    # mp=28 sources carry reduced-form binary64 words for both.
    return '#include "../thompson_aerosol_libm.cuh"\n' + node


def composed(text):
    seen = set()

    def expand(source):
        def one(match):
            name = match.group(1)
            if name in seen:
                return ""
            seen.add(name)
            return expand((KERNEL_DIR / name).read_text(encoding="utf-8"))
        return QUOTED.sub(one, source)
    return expand(text)


def blob_layout():
    header = (KERNEL_DIR / "tempo_common.cuh").read_text()
    body = header.split("#define TEMPO_FIELDS(TX)", 1)[1].split("struct TempoBlob", 1)[0]
    out, offset = [], 0
    for m in re.finditer(r"TX\((\w+),(\d+),(\d+),(\d+),(\d+),(\d+)\)", body.replace(" ", "")):
        n = 1
        for d in m.groups()[1:]:
            n *= int(d)
        out.append((m.group(1), offset))
        offset += n
    return out, offset


def rewrite_blob_access(text):
    """``P.L->name[i]`` becomes ``P.L[TOFF_name + (i)]`` on the flat blob."""
    out, pos = [], 0
    for m in re.finditer(r"P\.L->(\w+)\s*\[", text):
        if m.start() < pos:
            continue
        depth, j = 1, m.end()
        while depth:
            depth += {"[": 1, "]": -1}.get(text[j], 0)
            j += 1
        out.append(text[pos:m.start()])
        out.append("P.L[TOFF_%s + (%s)]" % (
            m.group(1), rewrite_blob_access(text[m.end():j - 1])))
        pos = j
    out.append(text[pos:])
    return "".join(out)


def preprocessed():
    done = subprocess.run(
        ["clang++", "-E", "-P", "-std=c++17", "-x", "c++", "-D__CUDACC_RTC__=1",
         "-D__CUDA_ARCH__=800", "-DTEMPO_NZ_MAX=TNZMAX",
         "-DTEMPO_SOURCE_FINGERPRINT=0", "-"],
        input=composed(kernel_text()).encode(), capture_output=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(done.stderr.decode()[-3000:])
    text = done.stdout.decode()
    text = re.sub(r"(?<![\w.])hypot\s*\(", "thompson_aa_hypot(", text)
    text = re.sub(r"(?<![\w.])atan2\s*\(", "thompson_aa_atan2(", text)
    return rewrite_blob_access(text)


def strip_inline(text):
    return re.sub(r"^(?:inline\s+)+", "", text)


def convert(item):
    kind = g.kind_of(item)
    if item.startswith("enum"):
        names = re.findall(r"(\w+)\s*=\s*(\([^)]*\))", item)
        return "\n".join("constant constexpr long %s = %s;" % n for n in names)
    if item.startswith("struct TempoParams"):
        return "struct TempoParams { device const Soft8 * L ; } ;"
    if kind == "struct":
        return g.convert_struct(item)
    if kind == "table":
        return g.convert_table(item)
    text = g.convert_function_item(strip_inline(item))
    # The column and diagnostic work areas live in device memory.
    text = re.sub(r"\bTempoWork &", "device TempoWork &", text)
    text = re.sub(r"\bTempoTend & T =", "device TempoTend & T =", text)
    text = re.sub(r"(?<!device )thread (device TempoWork)", r"\1", text)
    text = re.sub(r"\b(?:inline\s+)+inline\b", "inline", text)
    text = re.sub(r"'\s(\w)\s'", r"'\1'", text)
    text = re.sub(r"\bthread (const )?TempoTend &", r"device \1TempoTend &", text)
    text = re.sub(r"(?<![\w*])(Soft8)\s*\*\s*(\w+)\s*=\s*(?=[wW]\s*\.)",
                  r"device \1 * \2 = ", text)
    text = re.sub(r"(?<![\w*])(Soft8)\s*\*\s*(\w+)\s*=\s*nullptr",
                  r"device \1 * \2 = nullptr", text)
    return text


def build():
    items = split_top(preprocessed())
    keep = []
    for it in items:
        if (it.startswith(("namespace", "typedef", "#pragma", "struct TempoBlob"))
                or def_name(it) in g.PROVIDED):
            continue
        keep.append(it)
    reached, _ = closure(keep, ROOTS)
    reached = [g.typed_nulls(it, reached) for it in reached]
    return [convert(it) for it in reached if g.kind_of(it) != "kernel"]


def sizes_kernel(tier):
    return """\
kernel void tempo_sizes_k%(t)d(
    constant TempoSizesK%(t)dResources &r [[buffer(0)]],
    constant TempoSizesK%(t)dParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid != 0u) return;
    r.sizes[0] = ulong(sizeof(tcT%(t)d::TempoWork));
    r.sizes[1] = ulong(sizeof(tcT%(t)d::ColState));
}
""" % {"t": tier}


def pack_kernel(tier):
    fields = ", ".join("r.f_%s" % n for n in STATE_FIELDS)
    return """\
kernel void tempo_pack_k%(t)d(
    constant TempoPackK%(t)dResources &r [[buffer(0)]],
    constant TempoPackK%(t)dParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    using namespace tcT%(t)d;
    const int c = int(gid);
    if (c >= p.n) return;
    device ColState * S = (device ColState *)r.cols;
    device const float * fld[%(nf)d] = {%(fields)s};
    const long g = long(p.c0) + long(c);
    for (int j = 0; j < %(nf)d; ++j) {
        S[c].f[j][0] = Soft8(0);
        for (int k = 1; k <= p.nz; ++k)
            S[c].f[j][k] = Soft8(fld[j][long(k - 1) * long(p.nt) + g]);
        for (int k = p.nz + 1; k < TNZMAX + 3; ++k) S[c].f[j][k] = Soft8(0);
    }
    for (int k = 1; k <= p.nz; ++k) {
        const long o = long(k - 1) * long(p.nt) + g;
        S[c].f[0][k] = S[c].f[0][k] * Soft8(r.pii[o]);
        S[c].f[16][k] = S[c].f[16][k] * Soft8(1000.0f);
    }
    if (p.has_emission != 0) {
        const Soft8 dt = Soft8::fromBits(ulong(uint(p.dt_lo)) | (ulong(uint(p.dt_hi)) << 32));
        S[c].f[13][1] = S[c].f[13][1] + Soft8(r.emission[g]) * dt;
    }
}
""" % {"t": tier, "nf": NFLD, "fields": fields}


def unpack_kernel(tier):
    which = [STATE_FIELDS.index(n) for n in UPDATED_FIELDS]
    outs = ", ".join("r.o_%s" % n for n in UPDATED_FIELDS)
    return """\
kernel void tempo_unpack_k%(t)d(
    constant TempoUnpackK%(t)dResources &r [[buffer(0)]],
    constant TempoUnpackK%(t)dParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    using namespace tcT%(t)d;
    const int c = int(gid);
    if (c >= p.n) return;
    device ColState * S = (device ColState *)r.cols;
    device TempoWork * W = (device TempoWork *)r.work;
    device float * out[%(nw)d] = {%(outs)s};
    const int which[%(nw)d] = {%(which)s};
    const long g = long(p.c0) + long(c);
    for (int k = 1; k <= p.nz; ++k) {
        const long o = long(k - 1) * long(p.nt) + g;
        S[c].f[0][k] = S[c].f[0][k] / Soft8(r.pii[o]);
        S[c].f[16][k] = S[c].f[16][k] * Soft8::fromBits(0x3f50624dd2f1a9fcul);
    }
    if (p.has_emission != 0) {
        const Soft8 dt = Soft8::fromBits(ulong(uint(p.dt_lo)) | (ulong(uint(p.dt_hi)) << 32));
        S[c].f[13][1] = S[c].f[13][1] + Soft8(r.emission[g]) * dt;
    }
    for (int m = 0; m < %(nw)d; ++m)
        for (int k = 1; k <= p.nz; ++k)
            out[m][long(k - 1) * long(p.nt) + g] = float(S[c].f[which[m]][k]);
    const device TempoWork & w = W[c];
    Soft8 mh = w.max_hail_diameter[1];
    for (int k = 2; k <= p.nz; ++k) mh = w.max_hail_diameter[k] > mh ? w.max_hail_diameter[k] : mh;
    // Surface products as woof/core/tempo.py forms them, in binary64 and
    // rounded once to the float32 host fields.
    const Soft8 rain = w.rain_precip, ice = w.ice_liquid_equiv_precip,
        snow = w.snow_liquid_equiv_precip, graupel = w.graupel_liquid_equiv_precip;
    const Soft8 total = rain + ice + snow + graupel;
    const float rainncv = float(total);
    const float snowncv = float(snow + ice);
    const float graupelncv = float(graupel);
    r.rainncv[g] = rainncv;
    r.snowncv[g] = snowncv;
    r.graupelncv[g] = graupelncv;
    r.rainnc[g] = r.rainnc[g] + rainncv;
    r.snownc[g] = r.snownc[g] + snowncv;
    r.graupelnc[g] = r.graupelnc[g] + graupelncv;
    r.sr[g] = float((snow + ice + graupel) / (total + SD8(0x3d719799812dea11ul)));
    if (p.has_hail != 0) r.hail[g] = float(mh);
    const Soft8 micron = Soft8::fromBits(0x412e848000000000ul);
    for (int k = 1; k <= p.nz; ++k) {
        const long o = long(k - 1) * long(p.nt) + g;
        r.effc[o] = float(w.re_cloud[k] * micron);
        r.effi[o] = float(w.re_ice[k] * micron);
        r.effs[o] = float(w.re_snow[k] * micron);
        if (p.refl_due != 0) r.refl[o] = float(w.refl10cm[k]);
    }
}
""" % {"t": tier, "nw": len(UPDATED_FIELDS), "outs": outs,
       "which": ", ".join(map(str, which))}


def columns_kernel(tier):
    return """\
kernel void tempo_columns_k%(t)d(
    constant TempoColumnsK%(t)dResources &r [[buffer(0)]],
    constant TempoColumnsK%(t)dParams &p [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
    using namespace tcT%(t)d;
    const int i = int(gid);
    if (i >= p.n) return;
    TempoParams P;
    P.L = (device const Soft8 *)r.blob;
    TempoCtx c;
    c.global_dt = Soft8::fromBits(ulong(uint(p.dt_lo)) | (ulong(uint(p.dt_hi)) << 32));
    c.global_inverse_dt = Soft8::fromBits(ulong(uint(p.idt_lo)) | (ulong(uint(p.idt_hi)) << 32));
    c.aerosolaware_flag = p.aerosolaware;
    c.hailaware_flag = p.hailaware;
    c.semi_sedi_flag = p.semi_sedi;
    c.refl10cm_from_melting_flag = p.refl10cm_from_melting;
    c.turn_off_micro_flag = 0;
    c.cloud_condensation_flag = p.cloud_condensation;
    c.refl10cm_flag = p.refl10cm;
    c.re_cloud_flag = p.re_cloud;
    c.re_ice_flag = p.re_ice;
    c.re_snow_flag = p.re_snow;
    c.max_hail_diameter_flag = p.max_hail_diameter;
    c.first_call_main = p.first_call;
    if (p.c0 + i != 0) c.first_call_main = 0;
    device ColState * S = (device ColState *)r.cols;
    device TempoWork * W = (device TempoWork *)r.work;
    device ColState & s = S[i];
    tempo_main_column(P, c, p.nz, W[i], s.f[4], s.f[5], s.f[7], s.f[6], s.f[8],
                      s.f[9], s.f[16], s.f[10], s.f[11], s.f[12], s.f[15],
                      s.f[13], s.f[14], s.f[0], s.f[1], s.f[2], s.f[3]);
    if (p.ml_for_bl_nc != 0)
        tempo_subgrid_cloud_radii(P, c, p.nz, W[i], s.f[5], s.f[6], s.f[7], s.f[8],
                                  s.f[12], s.f[2], r.qc_bl, r.fraction_bl, p.nt, p.c0 + i);
}
""" % {"t": tier}


def sig(name, resources, parameters, threads=64):
    kernel = {
        "name": name, "cuda_symbol": name.rsplit("_k", 1)[0],
        "module": family_of(int(name.rsplit("_k", 1)[1])),
        "source": "woof/metal/shaders/%s.metal" % family_of(int(name.rsplit("_k", 1)[1])),
        "cuda_source": "woof/core/tempo_runtime.py",
        "resources": resources,
        "shapes": {n: [["any_1d"]] for n, _, _ in resources},
        "parameters": parameters, "threads_per_group": threads,
        "layout_version": 1, "arithmetic_policies": g.POLICIES}
    return kernel


def signatures(tier):
    f32 = lambda n, a="read": [n, "float", a]
    bits = lambda n, a="read": [n, "float64_bits", a]
    dtp = [["dt_lo", "int32"], ["dt_hi", "int32"]]
    out = []
    out.append(sig("tempo_sizes_k%d" % tier, [bits("sizes", "read_write")], [["pad", "int32"]], 1))
    out.append(sig("tempo_pack_k%d" % tier,
                   [f32("f_" + n) for n in STATE_FIELDS] + [f32("pii"), f32("emission"),
                                                           bits("cols", "read_write")],
                   dtp + [["nz", "int32"], ["nt", "int32"], ["c0", "int32"],
                          ["n", "int32"], ["has_emission", "int32"]]))
    out.append(sig("tempo_columns_k%d" % tier,
                   [bits("blob"), bits("cols", "read_write"), bits("work", "read_write"),
                    f32("qc_bl"), f32("fraction_bl")],
                   dtp + [["idt_lo", "int32"], ["idt_hi", "int32"], ["nz", "int32"],
                          ["nt", "int32"], ["c0", "int32"], ["n", "int32"],
                          ["aerosolaware", "int32"], ["hailaware", "int32"],
                          ["semi_sedi", "int32"], ["refl10cm_from_melting", "int32"],
                          ["cloud_condensation", "int32"], ["refl10cm", "int32"],
                          ["re_cloud", "int32"], ["re_ice", "int32"], ["re_snow", "int32"],
                          ["max_hail_diameter", "int32"], ["first_call", "int32"],
                          ["ml_for_bl_nc", "int32"]]))
    surface = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv", "sr")
    out.append(sig("tempo_unpack_k%d" % tier,
                   [f32("o_" + n, "read_write") for n in UPDATED_FIELDS]
                   + [f32("pii"), f32("emission"), bits("cols", "read_write"),
                      bits("work", "read_write")]
                   + [f32(n, "read_write") for n in surface]
                   + [f32(n, "read_write") for n in ("hail", "effc", "effi", "effs", "refl")],
                   dtp + [["nz", "int32"], ["nt", "int32"], ["c0", "int32"],
                          ["n", "int32"], ["has_emission", "int32"],
                          ["has_hail", "int32"], ["refl_due", "int32"]]))
    return out


def main():
    pieces = build()
    layout, total = blob_layout()
    offsets = "\n".join("constant constexpr long TOFF_%s = %d;" % (n, o) for n, o in layout)
    helpers = (
        "// Generated from woof/core/kernels/tempo and tempo_runtime.py by "
        "tools/tempo_metal/generate.py; do not edit.\n"
        "#pragma clang fp contract(off)\n"
        "namespace tc0 {\n%s\n}\n" % ("\n".join(pieces)))
    # Free helpers that carry no level capacity are shared; the structs and
    # everything that touches them are instantiated per tier.
    (SHADERS / "tempo_body.metalh").write_text(
        "// Generated by tools/tempo_metal/generate.py; do not edit.\n"
        "// Included once per level-capacity tier with TNZMAX defined.\n"
        "#pragma clang fp contract(off)\n" + offsets
        + "\n" + DEVICE_OPS
        + "\nstruct ColState { Soft8 f[%d][TNZMAX + 3]; };\n" % NFLD
        + "\n".join(pieces) + "\n")
    for family, family_tiers in FAMILIES.items():
        tiers, sigs = [], []
        for tier in family_tiers:
            tiers.append(
                "#define TNZMAX %d\nnamespace tcT%d {\n#include \"tempo_body.metalh\"\n}\n%s\n%s\n%s\n%s\n#undef TNZMAX"
                % (tier, tier, sizes_kernel(tier), pack_kernel(tier),
                   columns_kernel(tier), unpack_kernel(tier)))
            sigs += signatures(tier)
        (SHADERS / (family + ".metal")).write_text(
            "// Generated by tools/tempo_metal/generate.py from the CUDA source; do not edit.\n"
            "#include <metal_stdlib>\n"
            '#include "thompson_aerosol_soft8.metalh"\n'
            '#include "%s_bindings.metalh"\n\n%s\n' % (family, "\n\n".join(tiers)))
        (FRAGMENTS / (family + ".json")).write_text(
            json.dumps({"kernels": sigs}, indent=2) + "\n")
    print("blob words:", total)


if __name__ == "__main__":
    main()
