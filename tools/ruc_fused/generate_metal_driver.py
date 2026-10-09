"""Lower the six-stage RUC driver to the packed Metal ABI.

The driver consumes the same raw word workspace as the three SFCTMP stages.
Its generated CUDA pointer tables are intentionally collapsed to typed row
offsets here, while arithmetic and source order remain pinned to the CUDA
driver transcription. The CUDA snow lineage define (GPUWM_SNOW_WRF461) is
the `forms` parameter's bit 1, as in the SFCTMP stages.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "woof/core/kernels/ruc_fused_driver.cuh"
SF_LAYOUT = ROOT / "woof/core/ruc_sfctmp_layout.py"
OUTPUT = ROOT / "woof/metal/shaders/ruc_driver_family.metal"
LAYOUT_PATH = ROOT / "woof/metal/ruc_metal_layout.py"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("ruc_metal_layout", LAYOUT_PATH)
assert SPEC is not None and SPEC.loader is not None
LAYOUT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAYOUT)
SF_SPEC = importlib.util.spec_from_file_location("ruc_sfctmp_layout", SF_LAYOUT)
assert SF_SPEC is not None and SF_SPEC.loader is not None
SF_LAYOUT = importlib.util.module_from_spec(SF_SPEC)
SF_SPEC.loader.exec_module(SF_LAYOUT)


def _source_macro_block(source: str) -> str:
    start = source.index("#define D_ADMIT")
    end = source.index("#define D_COMMIT", start)
    return source[start:end]


def _source_functions(source: str) -> str:
    start = source.index("__device__ __forceinline__ float d_min")
    end = source.index("#undef A", start)
    return source[start:end]


def _output_units(name: str) -> int:
    offset = LAYOUT.sfctmp_output_word_offset(name, 1)
    return offset


def _row_expression(name: str, index: str, *, profile: bool = False) -> str:
    row, is_profile = LAYOUT.driver_field_row(name)
    if profile and not is_profile:
        raise RuntimeError(f"RUC driver profile row missing for {name}")
    return f"(*reinterpret_cast<device float *>(r.raw + ({row} + uint({index})) * uint(n) + uint(i)))" if profile else f"(*reinterpret_cast<device float *>(r.raw + {row} * uint(n) + uint(i)))"


def _replace_rows(source: str) -> str:
    def profile(match):
        name, layer = match.group(1), match.group(2)
        row, is_profile = LAYOUT.driver_field_row(name)
        if not is_profile:
            raise RuntimeError(f"D_P used on scalar RUC field {name}")
        return ("(*reinterpret_cast<device float *>(r.raw + "
                f"({row} + uint({layer})) * uint(n) + uint(i)))")

    def scalar(match):
        return _row_expression(match.group(1), "0")

    source = re.sub(r"D_P\((\w+)\s*,\s*([^\)]+)\)", profile, source)
    source = re.sub(r"D_F\((\w+)\)", scalar, source)
    return source


def _replace_outputs(source: str) -> str:
    # Epilogue output arrays are the generated temporary slots already
    # resident in the SFCTMP slab. Their slot map is emitted beside the CUDA
    # pointer layout and stays independent of the public field rows.
    def profile(match):
        name, layer = match.group(1), match.group(2)
        units = _output_units(name)
        return ("(*reinterpret_cast<device float *>(r.raw + "
                f"{units} * uint(n) + uint({layer}) * uint(n) + uint(i)))")

    source = re.sub(r"o_(\w+)\[([^\]]+)\*n\s*\+\s*i\]", profile, source)

    def from_result(match):
        dest, output = match.group(1), match.group(2)
        units = _output_units(output)
        dtype = "int" if SF_LAYOUT._SFCTMP_ARRAYS[SF_LAYOUT._SFCTMP_OUTPUTS[output]][1] == "int32" else "float"
        return (f"{_row_expression(dest, '0')} = "
                f"(*reinterpret_cast<device {dtype} *>(r.raw + {units} * uint(n) + uint(i)))")

    source = re.sub(r"^#define FROM\(name,out\).*\n", "", source, flags=re.MULTILINE)
    source = re.sub(r"^#define REBLEND\(name,sea\).*\n", "", source, flags=re.MULTILINE)
    source = re.sub(r"FROM\((\w+)\s*,\s*(\w+)\)", from_result, source)
    needle = "REBLEND("
    while needle in source:
        start = source.index(needle)
        open_paren = start + len("REBLEND")
        depth = 0
        comma = None
        end = None
        for position in range(open_paren, len(source)):
            char = source[position]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = position + 1
                    break
            elif char == "," and depth == 1 and comma is None:
                comma = position
        if comma is None or end is None:
            raise RuntimeError("could not parse RUC REBLEND macro call")
        name = source[open_paren + 1:comma].strip()
        sea = source[comma + 1:end - 1].strip()
        replacement = (
            f"if(component) D_F({name})=A(M(D_F({name}),fraction),"
            f"M(B(1.0f,fraction),{sea}))")
        source = source[:start] + replacement + source[end:]
    source = source.replace("D_DECLARE_OUTPUT", "")
    return source


def _replace_cuda_ops(source: str) -> str:
    replacements = {
        "__device__ __forceinline__": "static inline",
        "__int_as_float(": "as_type<float>(",
        "__float_as_int(": "as_type<int>(",
        "__fadd_rn(": "ruc_fadd_rn(",
        "__fsub_rn(": "ruc_fsub_rn(",
        "__fmul_rn(": "ruc_fmul_rn(",
        "__fdiv_rn(": "ruc_fdiv_rn(",
        "fminf(": "fmin(",
        "fmaxf(": "fmax(",
        "fabsf(": "fabs(",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    source = source.replace("unsigned* flags", "device uint* flags")
    source = source.replace("const float* tbq", "device const float* tbq")
    source = source.replace("const float* half", "device const float* half")
    source = source.replace("bool* admitted", "thread bool* admitted")
    source = source.replace("thread bool* admitted=nullptr", "thread bool* admitted")
    source = source.replace("nullptr", "nullptr")
    source = re.sub(
        r"atomicOr\(flags\s*\+\s*(\w+),\s*([^\)]+)\)",
        r"atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(flags) + (\1), \2, memory_order_relaxed)",
        source)
    source = source.replace(
        "atomicOr(flags+bit/32,1u<<(bit%32));",
        "atomic_fetch_or_explicit(reinterpret_cast<device atomic_uint *>(flags) + bit/32, 1u<<(bit%32), memory_order_relaxed);")
    source = re.sub(
        r"atomicAdd\(flags\s*\+\s*(\d+),\s*([^\)]+)\)",
        r"atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint *>(flags) + \1, \2, memory_order_relaxed)",
        source)
    source = source.replace("D_DECLARE_SCRATCH", "")
    source = source.replace("D_COPY_INPUT", "")
    source = source.replace("((const int*)ip[D_CAT_INPUT])[i]", "integer[i]")
    source = source.replace("((const int*)ip[D_CAT_INPUT+1])[i]", "integer[n+i]")
    source = source.replace("run[i]=land && admitted;", "run[i]=uint(land && admitted);")
    source = source.replace("D_CAT_FLAG", "65")
    source = re.sub(r"(?m)^#define 65 65[ \t]*$", "", source)
    source = source.replace("int i=blockIdx.x*blockDim.x+threadIdx.x;", "int i = int(tid);")
    source = source.replace("if(i>=n) return;", "if (i >= n) return;")
    source = source.replace("#define A(a,b) __fadd_rn((a),(b))", "#define A(a,b) ruc_fadd_rn((a),(b))")
    source = source.replace("#define M(a,b) __fmul_rn((a),(b))", "#define M(a,b) ruc_fmul_rn((a),(b))")
    source = source.replace("#define B(a,b) __fsub_rn((a),(b))", "#define B(a,b) ruc_fsub_rn((a),(b))")
    source = source.replace("#define Q(a,b) __fdiv_rn((a),(b))", "#define Q(a,b) ruc_fdiv_rn((a),(b))")
    return source


def _replace_function(source: str, name: str, replacement: str) -> str:
    match = re.search(rf'extern "C" __global__ void {name}\(', source)
    if match is None:
        raise RuntimeError(f"could not convert RUC driver kernel {name}")
    open_brace = source.index("{", match.start())
    depth = 0
    close_brace = None
    for index in range(open_brace, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                close_brace = index + 1
                break
    if close_brace is None:
        raise RuntimeError(f"unterminated RUC driver kernel {name}")
    return source[:match.start()] + replacement + source[close_brace:]


def _replace_signature(source: str, name: str, replacement: str) -> str:
    match = re.search(rf'extern "C" __global__ void {name}\(', source)
    if match is None:
        raise RuntimeError(f"could not convert RUC driver kernel {name}")
    open_brace = source.index("{", match.start())
    return source[:match.start()] + replacement + source[open_brace + 1:]


_SNOW_DEFINE = re.compile(
    r"#ifdef GPUWM_SNOW_WRF461\n(?P<v461>.*?)#else\n(?P<v45>.*?)#endif\n", re.S)


def _snow_lineage(source: str) -> str:
    def replace(match):
        return ("        if((forms&2u)!=0u) {\n" + match.group("v461")
                + "        } else {\n" + match.group("v45") + "        }\n")

    source, count = _SNOW_DEFINE.subn(replace, source)
    if count != 1:
        raise RuntimeError(f"expected one RUC driver snow lineage block, found {count}")
    # The 461 arm declares the flag both arms set; hoist it out of the scopes.
    source = source.replace("        if((forms&2u)!=0u) {\n        bool inside_snow=",
                            "        bool inside_snow;\n        if((forms&2u)!=0u) {\n        inside_snow=", 1)
    source = source.replace("        bool inside_snow=D_F(snow)>32.0f;\n",
                            "        inside_snow=D_F(snow)>32.0f;\n", 1)
    if "inside_snow;\n" not in source:
        raise RuntimeError("could not hoist the RUC driver inside_snow flag")
    return source


# module_sf_ruclsm.F SOILVEGIN mosaic loops; ruc.cu ruc_mosaic_parameters.
_MOSAIC = '''
static inline float gfk_log(float value) { return as_type<float>(soft_gfk_log(as_type<uint>(value))); }
static inline float gfk_exp(float value) { return as_type<float>(soft_gfk_exp(as_type<uint>(value))); }
static inline float ruc_fsqrt_rn(float value) { return as_type<float>(soft32_sqrt(as_type<uint>(value))); }
static inline void ruc_mosaic_parameters(
    int i,int n,int nlcat,int nscat,int mosaic_lu,int mosaic_soil,
    device const float* landusef,device const float* soilctop,
    int soil,int iswater,bool rdlai2d,float factor,float incoming_znt,
    device const int* ifortbl,device const float* z0tbl,device const float* lemitbl,
    device const float* pctbl,device const float* laitbl,device const float* bb,
    device const float* drysmc,device const float* hc,device const float* maxsmc,
    device const float* refsmc,device const float* satpsi,device const float* satdk,
    device const float* wltsmc,device const float* qtz,
    device float& emiss,device float& pc,device float& znt,device float& lai,
    device float& qwrtz,device float& rhocs,device float& bclh,device float& dqm,
    device float& ksat,device float& psis,device float& qmin,device float& ref,device float& wilt) {
    if(mosaic_lu) {
        float area=0.0f, rough=0.0f, leaf=0.0f;
        emiss=0.0f; pc=0.0f;
        for(int k=0;k<nlcat;++k) {
            float fraction=landusef[k*n+i];
            int forest=ifortbl[k];
            float delta=0.0f,scaled=ruc_fmul_rn(0.8f,laitbl[k]);
            if(forest==1) delta=fmin(0.2f,scaled);
            if(forest==2 || forest==7) delta=fmin(0.5f,scaled);
            if(forest==3) delta=fmin(0.45f,scaled);
            if(forest==4) delta=fmin(0.75f,scaled);
            if(forest==5) delta=fmin(0.86f,scaled);
            float today_lai=k+1==iswater ? laitbl[k]:ruc_fsub_rn(laitbl[k],ruc_fmul_rn(delta,factor));
            float today_znt=k+1==iswater ? incoming_znt:(forest==7 ? ruc_fsub_rn(z0tbl[k],ruc_fmul_rn(0.125f,factor)):z0tbl[k]);
            float logarithm=gfk_log(ruc_fdiv_rn(5.0f,today_znt));
            area=ruc_fadd_rn(area,fraction);
            emiss=ruc_fadd_rn(emiss,ruc_fmul_rn(lemitbl[k],fraction));
            rough=ruc_fadd_rn(rough,ruc_fdiv_rn(fraction,ruc_fmul_rn(logarithm,logarithm)));
            if(!rdlai2d) leaf=ruc_fadd_rn(leaf,ruc_fmul_rn(today_lai,fraction));
            pc=ruc_fadd_rn(pc,ruc_fmul_rn(pctbl[k],fraction));
        }
        area=fmin(1.0f,area);
        emiss=ruc_fdiv_rn(emiss,area);
        pc=ruc_fdiv_rn(pc,area);
        znt=ruc_fdiv_rn(5.0f,gfk_exp(ruc_fsqrt_rn(ruc_fdiv_rn(1.0f,rough))));
        if(!rdlai2d) lai=ruc_fdiv_rn(leaf,area);
    }
    if(mosaic_soil) {
        float area=0.0f;
        qwrtz=0.0f; rhocs=0.0f; bclh=0.0f; dqm=0.0f; ksat=0.0f;
        psis=0.0f; qmin=0.0f; ref=0.0f; wilt=0.0f;
        for(int k=0;k<nscat;++k) {
            if(k==13) continue;
            float fraction=soilctop[k*n+i];
            area=ruc_fadd_rn(area,fraction);
            rhocs=ruc_fadd_rn(rhocs,ruc_fmul_rn(ruc_fmul_rn(hc[k],1.0e6f),fraction));
            bclh=ruc_fadd_rn(bclh,ruc_fmul_rn(bb[k],fraction));
            dqm=ruc_fadd_rn(dqm,ruc_fmul_rn(ruc_fsub_rn(maxsmc[k],drysmc[k]),fraction));
            ksat=ruc_fadd_rn(ksat,ruc_fmul_rn(satdk[k],fraction));
            psis=ruc_fsub_rn(psis,ruc_fmul_rn(satpsi[k],fraction));
            qmin=ruc_fadd_rn(qmin,ruc_fmul_rn(drysmc[k],fraction));
            ref=ruc_fadd_rn(ref,ruc_fmul_rn(refsmc[k],fraction));
            wilt=ruc_fadd_rn(wilt,ruc_fmul_rn(wltsmc[k],fraction));
            qwrtz=ruc_fadd_rn(qwrtz,ruc_fmul_rn(qtz[k],fraction));
        }
        area=fmin(1.0f,area);
        if(area<=0.0f) {
            int k=soil-1;
            rhocs=ruc_fmul_rn(hc[k],1.0e6f); bclh=bb[k];
            dqm=ruc_fsub_rn(maxsmc[k],drysmc[k]); ksat=satdk[k];
            psis=-satpsi[k]; qmin=drysmc[k]; ref=refsmc[k];
            wilt=wltsmc[k]; qwrtz=qtz[k];
        } else {
            rhocs=ruc_fdiv_rn(rhocs,area); bclh=ruc_fdiv_rn(bclh,area);
            dqm=ruc_fdiv_rn(dqm,area); ksat=ruc_fdiv_rn(ksat,area);
            psis=ruc_fdiv_rn(psis,area); qmin=ruc_fdiv_rn(qmin,area);
            ref=ruc_fdiv_rn(ref,area); wilt=ruc_fdiv_rn(wilt,area);
            qwrtz=ruc_fdiv_rn(qwrtz,area);
        }
    }
}
'''


def generate() -> str:
    full = SOURCE.read_text(encoding="utf-8")
    source = _source_macro_block(full) + _source_functions(full)
    source = _snow_lineage(source)
    source = source.replace(
        "bool component=D_F(xice)>=xice_threshold",
        "if (p.frzfrac_mode != 0) D_F(frzfrac)=D_F(t3d)<=273.15f ? 1.0f:0.0f;\n"
        "    D_F(rainbl)=A(D_F(rainbl),(*reinterpret_cast<device float *>(r.raw + p.pending_base + uint(i))));\n"
        "    bool component=D_F(xice)>=xice_threshold",
        1)
    source = source.replace(
        "d_qsn(D_F(t3d),tbq,flags,1024)",
        "d_qsn(D_F(t3d),tbq,flags,1024,nullptr)")
    source = _replace_outputs(source)
    source = _replace_rows(source)
    source = _replace_cuda_ops(source)

    prologue = '''kernel void ruc_driver_prologue(
    constant RucDriverPrologueResources& r [[buffer(0)]],
    constant RucDriverPrologueParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int n = p.n, ktau = p.ktau, iswater = p.iswater, isice = p.isice;
    const int nv = p.nv, ns = p.ns;
    const float dt = p.dt, icealbedo = p.icealbedo, cn = p.cn;
    const int nlcat = p.nlcat, nscat = p.nscat;
    const int mosaic_lu = p.mosaic_lu, mosaic_soil = p.mosaic_soil;
    const int lakemodel = p.lakemodel, qvg_air = p.qvg_air, rdlai2d = p.rdlai2d;
    const float xice_threshold = p.xice_threshold;
    const uint forms = uint(p.forms);
    device const float* landusef = reinterpret_cast<device const float *>(
        r.raw + p.table_base + p.table_offset_landusef);
    device const float* soilctop = reinterpret_cast<device const float *>(
        r.raw + p.table_base + p.table_offset_soilctop);
    device uint* flags = reinterpret_cast<device uint *>(r.flags);
    device int* integer = reinterpret_cast<device int *>(r.raw + p.int_base);
    device uint* run = reinterpret_cast<device uint *>(r.raw + p.run_base);
'''
    for name, dtype in (
        ("ifortbl", "int"), ("z0tbl", "float"), ("lemitbl", "float"),
        ("pctbl", "float"), ("laitbl", "float"), ("bb", "float"),
        ("drysmc", "float"), ("hc", "float"), ("maxsmc", "float"),
        ("refsmc", "float"), ("satpsi", "float"), ("satdk", "float"),
        ("wltsmc", "float"), ("qtz", "float"), ("tbq", "float"),
    ):
        prologue += (
            f"    device const {dtype}* {name} = reinterpret_cast<device const {dtype} *>("
            f"r.raw + p.table_base + p.table_offset_{name});\n")
    prologue += "\n"
    source = _replace_signature(source, "ruc_driver_prologue", prologue)

    epilogue = '''kernel void ruc_driver_epilogue(
    constant RucDriverEpilogueResources& r [[buffer(0)]],
    constant RucDriverEpilogueParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    const int n = p.n;
    const float dt = p.dt;
    device uint* flags = reinterpret_cast<device uint *>(r.flags);
    device uint* run = reinterpret_cast<device uint *>(r.raw + __RUN_BASE__ * uint(n));
    device const float* tbq = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_tbq);
    device const float* lemitbl = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_lemitbl);
    device const float* zshalf = reinterpret_cast<device const float *>(r.raw + p.table_base + p.table_offset_zshalf);
    device const int* integer = reinterpret_cast<device const int *>(r.raw + __INTEGER_BASE__ * uint(n));
    device const float* landusef = reinterpret_cast<device const float *>(
        r.raw + p.table_base + p.table_offset_landusef);
    const int nlcat = p.nlcat, mosaic_lu = p.mosaic_lu;
    const int crop = p.crop, natural = p.natural;
    const int irrigation = p.irrigation, log_profile = p.log_profile;
    const float xice_threshold = p.xice_threshold;
'''
    epilogue = epilogue.replace("__INTEGER_BASE__", str(LAYOUT.INTEGER_BASE_UNITS))
    epilogue = epilogue.replace("__RUN_BASE__", str(LAYOUT.RUN_BASE_UNITS))
    epilogue += "\n"
    source = _replace_signature(source, "ruc_driver_epilogue", epilogue)

    # The final kernel records that all device-side checks passed. Host code
    # reads the common flag slab once, raises on any set bit, then performs the
    # scatter blits into public state. That preserves WRF's all-or-nothing
    # commit without binding dozens of field buffers to one shader.
    commit = '''kernel void ruc_driver_commit(
    constant RucDriverCommitResources& r [[buffer(0)]],
    constant RucDriverCommitParams& p [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    uint i = tid;
    if (i >= uint(p.n)) return;
    for (uint word = 0; word < 36; ++word)
        if (reinterpret_cast<device const uint *>(r.flags)[word] != 0u) return;
    for (uint word = 0; word < uint(p.sf_flag_words); ++word)
        if (r.flags[p.sf_flag_base + word] != 0ul) return;
    r.raw[p.alive_base + i] = 1u;
}
'''
    source = _replace_function(source, "ruc_driver_commit", commit)
    source = source.replace("half[k+1]", "zshalf[k+1]")
    source = source.replace("half[k]", "zshalf[k]")
    source = source.replace("half[RUC_NZS-1]", "zshalf[RUC_NZS-1]")
    header = '''#include <metal_stdlib>
#include "ruc_driver_family_bindings.metalh"
#include "soft_binary32.metalh"
#include "soft_gfk_exp.metalh"
#include "soft_gfk_log.metalh"

using namespace metal;
#pragma clang fp contract(off)

#ifndef NAN
#define NAN as_type<float>(0x7fc00000u)
#endif

static inline float ruc_fadd_rn(float a,float b) { return as_type<float>(soft32_add(as_type<uint>(a),as_type<uint>(b))); }
static inline float ruc_fsub_rn(float a,float b) { return as_type<float>(soft32_sub(as_type<uint>(a),as_type<uint>(b))); }
static inline float ruc_fmul_rn(float a,float b) { return as_type<float>(soft32_mul(as_type<uint>(a),as_type<uint>(b))); }
static inline float ruc_fdiv_rn(float a,float b) { return as_type<float>(soft32_div(as_type<uint>(a),as_type<uint>(b))); }
static inline float ruc_fused_min(float a,float b) { return (isnan(a)||isnan(b)) ? as_type<float>(0x7fc00000u) : min(a,b); }
static inline float ruc_fused_max(float a,float b) { return (isnan(a)||isnan(b)) ? as_type<float>(0x7fc00000u) : max(a,b); }
static inline float ruc_qsn_lookup(float tn, device const float* table) {
    float raw = ruc_fsub_rn(tn, 173.15f);
    raw = ruc_fdiv_rn(raw, 0.05f);
    raw = ruc_fadd_rn(raw, 1.0f);
    int index = int(raw);
    if (index < 1) { index = 1; raw = 1.0f; }
    if (index > 5000) { index = 5000; raw = 5001.0f; }
    float lower = table[index - 1];
    float weight = ruc_fsub_rn(raw, float(index));
    float value = ruc_fsub_rn(table[index], lower);
    value = ruc_fmul_rn(value, weight);
    return ruc_fadd_rn(value, lower);
}
constant float ruc_soil_layer_depth[9] = {0.00f,0.01f,0.04f,0.10f,0.30f,0.60f,1.00f,1.60f,3.00f};
#define RUC_NZS 9
''' + _MOSAIC
    source = source.replace("#define D_CAT_INPUT 83\n", "")
    source = source.replace("#define D_CAT_FLAG 65\n", "")
    source = source.replace("#undef A\n", "")
    return header + source


def main() -> None:
    OUTPUT.write_text(generate(), encoding="utf-8")


if __name__ == "__main__":
    main()
