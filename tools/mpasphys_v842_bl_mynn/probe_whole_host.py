"""Compile selected native MYNN for a host oracle using equivalent host shims."""
import re
from tools.thompson_real_column_parity.host_backend import HostModule
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import MYNN_ARRAY_SUPPORT,MYNN_MODULE_CUDA,MYNN_TEXT_CORRECTIONS
from woof.hex.mpasphys.resources_build import mynn_maths,_vendored_maths

support=MYNN_ARRAY_SUPPORT
for kind,operation in (('float','__uint_as_float(__float_as_uint(a)^0x80000000u)'),('double','__longlong_as_double((unsigned long long)__double_as_longlong(a)^0x8000000000000000ull)')):
    pattern=r'__device__ __forceinline__ '+kind+r' mynn_neg\('+kind+r' a\) \{.*?\n\}'
    support,count=re.subn(pattern,'__device__ __forceinline__ '+kind+' mynn_neg('+kind+' a) { return '+operation+'; }',support,count=1,flags=re.S);assert count==1
for kind in ('float','double'):
    for name,op in (('min','<'),('max','>')):
        pattern=r'__device__ (?:MYNN_MINMAX_ATTR )?'+kind+r' mynn_'+name+r'\('+kind+r' a,'+kind+r' b\) \{.*?\n\}'
        support,count=re.subn(pattern,'__device__ '+kind+' mynn_'+name+'('+kind+' a,'+kind+' b) { return a'+op+'b ? a:b; }',support,count=1,flags=re.S);assert count==1
support=support.replace('__float2int_ru(a)','(int)ceilf(a)').replace('__float2int_rd(a)','(int)floorf(a)')
support=support.replace('offset += (i-lo[d])*stride[d++];','offset += (i-lo[d])*stride[d]; ++d;')
prefix='#pragma clang diagnostic ignored "-Wparentheses-equality"\n#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+mynn_maths(_vendored_maths())+support
def host_module(rp7=False):
    text=MYNN_MODULE_CUDA
    if rp7:
        before,after,count=MYNN_TEXT_CORRECTIONS['RP7'][0];assert text.count(before)==count
        text=text.replace(before,after)
    return HostModule('mpas_v842_mynn_whole_host_rp7' if rp7 else 'mpas_v842_mynn_whole_host',prefix+text).get_function('mynn_mmm_run')

if __name__=='__main__':
    host_module()
    print('Selected whole native MYNN module compiles for host source oracle',flush=True)
