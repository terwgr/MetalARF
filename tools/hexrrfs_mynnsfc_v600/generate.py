"""Translate the retained RRFS MYNN-SFC column to its Metal source."""
import ast
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
source_path = root / "woof/hex/mpasphys/rrfs/sf_mynnsfc_v600.py"
tree = ast.parse(source_path.read_text(encoding="utf-8"))
values = {
    node.targets[0].id: ast.literal_eval(node.value)
    for node in tree.body
    if isinstance(node, ast.Assign) and len(node.targets) == 1
    and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id in ("SOURCE", "FIELDS")
}
SOURCE, FIELDS = values["SOURCE"], values["FIELDS"]

s = SOURCE[:SOURCE.index('extern "C" __global__ void mynnsfc_columns')]
# Keep the source locations for review without copying revision annotations.
s = re.sub(r'(?m)(^// RRFS MPAS-Model-rrfs/[^\n]*) @[0-9a-f]{8}(?= |$)', r'\1', s)
s = re.sub(r'__device__ float psi\w+\[1001\];\n', '', s)
for name in ('m_drv_mynnsfc_init', 'm_drv_psi_init'):
    s = re.sub(r'__device__ __noinline__ (?:void|float) '+name+r'\([^;{]*\);\n', '', s)
    m = re.search(r'__device__ __noinline__ (?:void|float) '+name+r'\([^;{]*\) \{', s)
    assert m
    depth, i = 1, m.end()
    while depth:
        depth += (s[i] == '{') - (s[i] == '}')
        i += 1
    s = s[:m.start()] + s[i:]
# The sole use of the former device-global tables is in the top-level driver.
name = 'm_drv_mynnsfc_driver'
for m in reversed(list(re.finditer(r'\b'+name+r'\s*\(', s))):
    depth, i = 1, m.end()
    while depth:
        depth += (s[i] == '(') - (s[i] == ')')
        i += 1
    s=s[:i-1]+', device const float *tables'+s[i-1:]
# Table parameters already propagate through all other scheme routines.
s=s.replace('const float *psi', 'device const float *psi')
s=s.replace('__device__ __noinline__ ', 'static inline ')
s=re.sub(r'\b(float|int|bool)\s*&', r'thread \1 &', s)
# Top-level driver supplies the four global table pointers.
m=re.search(r'(static inline void m_drv_mynnsfc_driver\([^;{]*\) \{)',s)
assert m
s=s[:m.end()]+'''\n    device const float *psim_stab=tables, *psim_unstab=tables+1001;
    device const float *psih_stab=tables+2002, *psih_unstab=tables+3003;\n'''+s[m.end():]
for old,new in {'__fadd_rn':'rr_add','__fsub_rn':'rr_sub','__fmul_rn':'rr_mul','__fdiv_rn':'rr_div','SL_SQRT':'rr_sqrt','SL_LOG':'rr_log','SL_EXP':'rr_exp','SL_POW':'rr_pow','SL_ATAN':'rr_atan','fmaxf':'rr_max','fminf':'rr_min','fabsf':'rr_abs'}.items():
    s=re.sub(r'\b'+old+r'\b',new,s)
call=re.search(r'm_drv_mynnsfc_driver\((f\[0\*n\+c\].*?)\);',SOURCE).group(1)
call=re.sub(r'f\[(\d+)\*n\+c\]',lambda m:'v'+m.group(1),call)
call=call.replace('vegtype[c]','r.vegtype[c]').replace('xice_threshold','p.xice_threshold')
loads='\n'.join(f'    float v{i}=r.fields[{i}*p.ncol+c];' for i in range(len(FIELDS)))
stores='\n'.join(f'    r.fields[{i}*p.ncol+c]=v{i};' for i in range(len(FIELDS)))
header = """#include <metal_stdlib>
using namespace metal;
#include "hex_ieee32.metalh"
#include "cumulus_support.metalh"
#include "hexv842_seaice_math.metalh"
#include "soft_glibc_solar_trig.metalh"
#include "exact_binary32.metalh"
#include "hexrrfs_mynnsfc_v600_bindings.metalh"
#pragma clang fp contract(off)
static inline bool operator<(Exact32 a,Exact32 b){return soft32_less(a.bits,b.bits);}
static inline bool operator<=(Exact32 a,Exact32 b){return soft32_less_equal(a.bits,b.bits);}
static inline bool operator==(Exact32 a,Exact32 b){return soft32_equal(a.bits,b.bits);}
static inline bool operator!=(Exact32 a,Exact32 b){return !soft32_equal(a.bits,b.bits);}
static inline Exact32 rr_add(Exact32 a,Exact32 b){return exact_add(a,b);}
static inline Exact32 rr_sub(Exact32 a,Exact32 b){return exact_sub(a,b);}
static inline Exact32 rr_mul(Exact32 a,Exact32 b){return exact_mul(a,b);}
static inline Exact32 rr_div(Exact32 a,Exact32 b){return exact_div(a,b);}
static inline Exact32 rr_sqrt(Exact32 a){return exact_sqrt(a);}
static inline Exact32 rr_abs(Exact32 a){return exact_abs(a);}
static inline Exact32 rr_min(Exact32 a,Exact32 b){return exact_min(a,b);}
static inline Exact32 rr_max(Exact32 a,Exact32 b){return exact_max(a,b);}
static inline Exact32 rr_atan(Exact32 a){return Exact32::from_bits(soft_glibc_solar_atanf(a.bits));}
static inline Exact32 rr_pow(Exact32 a,Exact32 b){return Exact32::from_bits(seaice_pow_bits(a.bits,b.bits));}
static inline Exact32 rr_exp(Exact32 a){return Exact32(seaice_exp(as_type<float>(a.bits)));}
static inline Exact32 rr_log(Exact32 x){
    uint ix=x.bits;
    if(ix==0x3f800000u)return Exact32::from_bits(0u);
    if(ix-0x00800000u>=0x7f800000u-0x00800000u){
        if(ix*2u==0u)return Exact32::from_bits(0xff800000u);
        if(ix==0x7f800000u)return x;
        if((ix&0x80000000u)||ix*2u>=0xff000000u)return Exact32::from_bits(0x7fc00000u);
        ix=soft32_mul(ix,0x4b000000u)-(23u<<23);
    }
    uint tmp=ix-0x3f330000u,i=(tmp>>19)&15u,iz=ix-(tmp&0xff800000u);
    int k=as_type<int>(tmp)>>23;
    ulong r=soft64_fma(soft64_from_f32_bits(iz),SOFT_GFK_LOGF_INVC[i],0xbff0000000000000ul);
    ulong y0=soft64_fma(soft64_from_i32(k),SOFT_GFK_LOGF_LN2,SOFT_GFK_LOGF_LOGC[i]);
    ulong r2=soft64_mul(r,r),y=soft64_fma(SOFT_GFK_LOGF_A1,r,SOFT_GFK_LOGF_A2);
    y=soft64_fma(r2,SOFT_GFK_LOGF_A0,y);
    return Exact32::from_bits(soft64_to_f32_bits(soft64_fma(r2,y,soft64_add(y0,r))));
}
// Retained interpolation indices are converted only for finite in-range inputs.
static inline int rr_int(Exact32 a){
    uint bits=a.bits, magnitude=bits&0x7fffffffu;
    int exponent=int(magnitude>>23)-127;
    if(exponent<0)return 0;
    uint significand=(magnitude&0x7fffffu)|0x800000u;
    uint integer=exponent<23?significand>>uint(23-exponent):significand<<uint(exponent-23);
    return (bits&0x80000000u)?-int(integer):int(integer);
}
"""
footer='''
kernel void hexrrfs_mynnsfc_tables(constant HexrrfsMynnsfcTablesResources &r [[buffer(0)]],constant HexrrfsMynnsfcTablesParams &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
    if(c>=uint(p.ntable)) return;
    float z=rr_mul(float(c),0x1.47ae140000000p-7f);
    r.tables[c]=m_drv_psim_stable_full(z); r.tables[2002+c]=m_drv_psih_stable_full(z);
    z=rr_mul(-float(c),0x1.47ae140000000p-7f);
    r.tables[1001+c]=m_drv_psim_unstable_full(z); r.tables[3003+c]=m_drv_psih_unstable_full(z);
}
kernel void hexrrfs_mynnsfc_columns(constant HexrrfsMynnsfcColumnsResources &r [[buffer(0)]],constant HexrrfsMynnsfcColumnsParams &p [[buffer(1)]],uint c [[thread_position_in_grid]]) {
    if(c>=uint(p.ncol)) return;
    int error=0; ulong present=ulong(uint(p.present_lo))|(ulong(uint(p.present_hi))<<32);
    int o_ivegsrc=r.options[0]; bool o_redrag=r.options[1]; int o_sfc_z0_type=r.options[2];
    int o_spp_pbl=r.options[3],o_sf_mynn_sfcflux_water=r.options[4],o_sf_mynn_sfcflux_land=r.options[5];
    int o_shalwater_z0=r.options[6],o_isfflx=r.options[7]; bool o_restart=r.options[8],o_cycling=r.options[9];
    int o_initflag=p.init_override<0?r.options[10]:p.init_override; bool o_flag_iter=r.options[11];
    int o_flagc_lsm=r.options[12],o_itimestep=p.step_override<0?r.options[13]:p.step_override;
'''+loads+'\n    m_drv_mynnsfc_driver('+call+',r.tables);\n'+stores+'\n    r.errors[c]=error;\n}\n'
# Keep device table elements and the public ABI as raw floats; all scheme
# values and comparisons use Exact32 so subnormals do not depend on hardware.
body=s+footer
body=body.replace('device const float *','device const __RAW_FLOAT__ *')
body=re.sub(r'\bfloat\b','Exact32',body).replace('__RAW_FLOAT__','float')
body=body.replace('(int)(rr_mul(', 'rr_int(rr_mul(')
body=re.sub(r'(r\.tables\[[^\]]+\]\s*=\s*)(m_drv_\w+\(z\))(;)',r'\1as_type<float>(\2.bits)\3',body)
body=re.sub(r'(r\.fields\[\d+\*p\.ncol\+c\]\s*=)v(\d+);',r'\1as_type<float>(v\2.bits);',body)
(root / 'woof/metal/shaders/hexrrfs_mynnsfc_v600.metal').write_text(
    header + body, encoding="utf-8")
