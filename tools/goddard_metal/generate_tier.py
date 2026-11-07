"""Generate source-ordered strict Goddard longwave band tiers."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
src=(root/'woof/metal/shaders/wrf_exact_goddard_fragment.metalh').read_text()
needle='static inline void g_lwrad ( thread GddArena & arena'
start=src.index(needle,src.index(needle)+1)
start=src.rfind('template <',0,start)
body=src.index('{',start)
def close(opening):
 depth=0
 for i in range(opening,len(src)):
  if src[i]=='{':depth+=1
  elif src[i]=='}':
   depth-=1
   if depth==0:return i
 raise ValueError('unclosed')
end=close(body)
head=src[start:body].replace('void g_lwrad (','void g_lwrad_tier (').replace('int irestrict )','int irestrict , int tier_mode , int tier_band )')
text=src[body+1:end]
marker='bool b10bnd ;'
decl_end=text.index(marker)+len(marker)
band_start=text.index('for ( ibn = 1 ;',decl_end)
band_open=text.index('{',band_start)
depth=1;i=band_open+1
while depth:
 if text[i]=='{':depth+=1
 elif text[i]=='}':depth-=1
 i+=1
band_close=i-1
pre=text[:decl_end]
init=text[decl_end:band_start]
band=text[band_open+1:band_close]
final=text[band_close+1:]
assert 'sfcem' not in final
result=head+'{\n'+pre+'\ndevice Soft8 * tier_sfcem = arena.alloc<Soft8>(1);\nif (tier_mode == 0) {\n'+init+'\ntier_sfcem[0]=sfcem; return;\n}\nif (tier_mode == 1) {\nsfcem=tier_sfcem[0]; ibn=tier_band;\n'+band+'\ntier_sfcem[0]=sfcem; return;\n}\n'+final+'\n}\n'
(root/'woof/metal/shaders/wrf_exact_goddard_tier_fragment.metalh').write_text(result)
print('generated tier function:',len(result),'characters')
import json
family=(root/'woof/metal/shaders/wrf_exact_goddard_family.metal').read_text()
name='kernel void wrf_exact_goddard_bands_columns'
kstart=family.index(name);kopen=family.index('{',kstart)
depth=1;kend=kopen+1
while depth:
 if family[kend]=='{':depth+=1
 elif family[kend]=='}':depth-=1
 kend+=1
kernel=family[kstart:kend]
kernel=kernel.replace('wrf_exact_goddard_bands_columns','wrf_exact_goddard_tiers_columns').replace('WrfExactGoddardBandsColumns','WrfExactGoddardTiersColumns')
marker='int ict = flags [ col ] , icb = flags [ col + columns ] ;'
cut=kernel.index(marker);prefix=kernel[:cut]+marker+'\n'
swstart=cut+len(marker);lwstart=kernel.index('g_lwrad ( arena',swstart)
swcall=kernel[swstart:lwstart].strip()
lwend=kernel.index(';',lwstart)+1
lwcall=kernel[lwstart:lwend].replace('g_lwrad (','g_lwrad_tier (').replace(' , 1 ) ;',' , 1 , params.tier_mode , params.tier_band ) ;')
assert 'tier_mode' in lwcall
output=kernel[lwend:kernel.rfind('}')]
cutout=output.index('at += columns * 4 ;')+len('at += columns * 4 ;')
swout=output[:cutout];lwout=output[cutout:]
wrapper=prefix+'if (params.tier_mode == 0) {\n'+swcall+'\n'+lwcall+'\n'+swout+'\nreturn;\n}\n'+lwcall+'\nif (params.tier_mode == 1) return;\nat = col + columns * (3 * (np + 1) + 4);\n'+lwout+'\n}\n'
headers=family[:kstart].replace('wrf_exact_goddard_family_bindings.metalh','wrf_exact_goddard_tiers_columns_bindings.metalh')+'#include "wrf_exact_goddard_tier_fragment.metalh"\n'
(root/'woof/metal/shaders/wrf_exact_goddard_tiers.metal').write_text(headers+wrapper)
rows=json.loads((root/'woof/metal/signature_fragments/wrf_exact_goddard_family.json').read_text())['kernels']
row=next(r for r in rows if r['name']=='wrf_exact_goddard_bands_columns')
row['name']='wrf_exact_goddard_tiers_columns';row['module']='wrf_exact_goddard_tiers';row['source']='woof/metal/shaders/wrf_exact_goddard_tiers.metal'
row['parameters'].extend([['tier_mode','int32'],['tier_band','int32']])
(root/'woof/metal/signature_fragments/wrf_exact_goddard_tiers.json').write_text(json.dumps({'kernels':[row]},indent=2)+'\n')
print('generated tier wrapper and ABI')
