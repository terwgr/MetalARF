#!/usr/bin/env python3
"""Translate the Goddard CUDA operation graph and dynamic column storage."""
from pathlib import Path
import json,re,struct,subprocess,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/mp28_metal'))
sys.path.insert(0,str(ROOT))
from parse import split_top,def_name,closure
from translate import tokenize,untok,map_types,rewrite_division,match_fwd,split_params,convert_function_item,convert_table
SHADERS=ROOT/'woof/metal/shaders'
NAMES=('goddard_interface_temperature','goddard_prepare_columns',
       'goddard_cloud_columns','goddard_bands_columns','goddard_publish_columns')
MATH={'g_max','g_min','g_abs','g_exp','g_log','g_log10','g_sqrt','g_pow','g_div'}
PROVIDED={'plm_exp','plm_log','plm_pow'}
ALLOCS={};CALLS={}

def preprocess():
 from woof.core import kernels
 text='#define GDD_MAX_LEVELS np\n'+kernels.module_source('goddard')
 result=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=text.encode(),capture_output=True,check=True)
 return split_top(result.stdout.decode())

def split_arguments(text):
 toks=tokenize(text);return [untok(p).strip() for p in split_params([('op','(')]+toks+[('op',')')],0,len(toks)+1)]

def decl_info(item):
 head=item.split('{',1)[0].rstrip(';').strip()
 # Collapse array types to one token while reading comma-separated parameters.
 arrays={}
 def hide(m):
  key='GddType'+str(len(arrays));arrays[key]=m.group();return key
 head=re.sub(r'GArray\s*<\s*(GReal|float|int)\s*,\s*\d+\s*>',hide,head)
 toks=tokenize(head);op=next(i for i,t in enumerate(toks) if t[1]=='(');cl=match_fwd(toks,op)
 name=toks[op-1][1]
 params=[untok(p).strip() for p in split_params(toks,op,cl)]
 params=[re.sub(r'\bGddType\d+\b',lambda m:arrays[m[0]],p) for p in params]
 return name,params


def call_references(text,refs):
 toks=tokenize(text)
 for i in range(len(toks)-2,-1,-1):
  name=toks[i][1]
  if name not in refs or toks[i+1][1]!='(':continue
  end=match_fwd(toks,i+1);args=split_params(toks,i+1,end)
  for ix in refs[name]:args[ix]=tokenize('& (')+args[ix]+tokenize(')')
  mid=[]
  for j,a in enumerate(args):
   if j:mid.append(('op',','))
   mid.extend(a)
  toks[i+2:end]=mid
 return untok(toks)


def arrays_and_calls(name,body,functions):
 alloc=[];dynamic=set()
 def split_descriptors(m):
  typ,decl=m.groups()
  return ';'.join(typ+' '+part for part in split_arguments(decl))+';'
 body=re.sub(r'(GArray\s*<\s*(?:GReal|float|int)\s*,\s*\d+\s*>)\s+(\w+\s*=\s*\{[^;]+);',split_descriptors,body)
 def expand(m):
  typ,decl=m.groups()
  return ';'.join(typ+' '+part for part in split_arguments(decl))+';'
 body=re.sub(r'\b(GReal|float|int)\s+(\w+\s*\[[^;]+);',expand,body)
 pattern=r'\b(GReal|float|int)\s+(\w+)\s*\[([^\[\];]+)\]\s*(=\s*\{([^{}]*)\})?\s*;'
 def arena(m):
  typ,var,count,init,values=m.groups()
  for symbol,value in (('nband_sw','11'),('nband_lw','10'),('chunk','1')):
   count=re.sub(r'\b'+symbol+r'\b',value,count)
  if not re.search(r'\bnp\b',count):return m.group()
  dynamic.add(var);alloc.append((8 if typ=='GReal' else 4,count))
  init_code=''
  if init:
   if values.strip() not in ('0','0.f','0.0f'):raise ValueError((name,var,values))
   init_code='for(int gdd_i=0;gdd_i<('+count+');++gdd_i)'+var+'[gdd_i]=0;'
  return 'device '+typ+' *'+var+'=arena.alloc<'+typ+'>('+count+');'+init_code
 body=re.sub(pattern,arena,body)
 # Keep each GArray's actual pointer address space across helper calls.
 pattern=r'GArray\s*<\s*(GReal|float|int)\s*,\s*(\d+)\s*>\s+(\w+)\s*=\s*\{\s*(\w+)'
 def descriptor(m):
  typ,rank,var,data=m.groups()
  address='constant' if data.startswith('gdd_') else 'device' if data in dynamic else 'thread'
  return 'GArray<'+typ+','+rank+','+address+' '+typ+'*> '+var+'={'+data
 body=re.sub(pattern,descriptor,body)
 CALLS[name]=sorted({f for f in functions if re.search(r'\b'+f+r'\s*\(',body)})
 ALLOCS[name]=alloc
 return body


def translate_function(item,functions,refs):
 name,params=decl_info(item)
 brace=item.find('{');body=item[brace+1:item.rfind('}')]
 body=re.sub(r'\[&\]\(\)\{\s*GReal\s+(\w+)=0;\s*for\(int\s+(\w+)=([^;]+);[^;]+;[^)]+\)\s*\1\+=(\w+)\(\2\);\s*return\s+\1;\s*\}\(\)',
             lambda m:'gdd_sum_range('+m[4]+','+m[3]+',np)',body)
 body=call_references(body,refs)
 for label in ('11','21','31'):
  body=re.sub(r'if\s*\(\s*\(\s*ii\s*==\s*1\s*\)\s*\)\s*goto\s+g_label_'+label+r'\s*;','if(ii!=1){',body)
  body=re.sub(r'\bgoto\s+g_label_'+label+r'\s*;','break;',body)
  body=re.sub(r'\bg_label_'+label+r'\s*:\s*;','}',body)
 types=[];new=[];casts=[]
 for param in params:
  pname=re.findall(r'\b\w+\b',param)[-1]
  if 'GArray' in param or '&' in param:
   t='P_'+pname;types.append(t)
   if 'GArray' in param:
    rank=re.search(r',\s*(\d+)\s*>',param)[1]
    new.append(t+' '+pname+'_arg')
    casts.append('auto '+pname+'=gdd_rank<'+rank+'>('+pname+'_arg);')
   else:new.append(t+' '+pname)
   if '&' in param:body=re.sub(r'\b'+pname+r'\b','(*'+pname+')',body)
  else:new.append(param)
 body=arrays_and_calls(name,body,functions)
 for f in functions:body=re.sub(r'\b'+f+r'\s*\(',f+'(arena,',body)
 result='template<'+','.join('class '+t for t in types)+'> ' if types else ''
 result+='static inline void '+name+'(thread GddArena &arena,'+','.join(new)+'){GddScope gdd_scope(arena);'+''.join(casts)+body+'}'
 return untok(rewrite_division(map_types(tokenize(result))))


def global_table(item):
 # GReal tables initialize from REAL(4) literal values in the CUDA source.
 # Encode that float32 rounding before constructing the software binary64.
 if re.search(r'\bGReal\b',item):
  ts=tokenize(item)
  out=[]
  for k,s in ts:
   if k=='num' and s.lower().endswith('f'):
    v=float.fromhex(s[:-1]) if s.lower().startswith('0x') else float(s[:-1])
    v=struct.unpack('<f',struct.pack('<f',v))[0]
    s='SD8(0x%016xul)'%struct.unpack('<Q',struct.pack('<d',v))[0]
    k='id'
   out.append((k,s))
  item=untok(out)
 text=convert_table(item.replace('constexpr','').replace('GReal','Soft8'))
 return text


def signature(name,params):
 resources=[];parameters=[]
 for p in params:
  tokens=re.findall(r'\w+|\*',p);field=tokens[-1]
  if '*' in tokens:resources.append([field,'int32' if 'int' in tokens else 'float','read' if 'const' in tokens else 'read_write'])
  else:parameters.append([field,'float32' if 'float' in tokens else 'int32'])
 if name=='goddard_bands_columns':
  resources.append(['workspace','uint64','read_write']);parameters.append(['workspace_words','int32'])
 return dict(name=name,cuda_symbol=name,module='goddard_family',
  source='woof/metal/shaders/goddard_family.metal',cuda_source='woof/core/kernels/goddard.cu',
  resources=resources,shapes={r[0]:[['any_1d']] for r in resources},parameters=parameters,
  threads_per_group=32,layout_version=1,
  arithmetic_policies=['soft32-rn-nonftz','soft64-rn-bit-ops','portable-libm64'])


def kernel(item,functions,refs):
 name,params=decl_info(item)
 body=item[item.index('{')+1:item.rfind('}')]
 body=re.sub(r'blockIdx\.x\s*\*\s*blockDim\.x\s*\+\s*threadIdx\.x','int(gid)',body)
 body=call_references(body,refs)
 body=body.replace(' || np>np','').replace(' || np > np','')
 body=arrays_and_calls(name,body,functions)
 for f in functions:body=re.sub(r'\b'+f+r'\s*\(',f+'(arena,',body)
 title=''.join(w.title() for w in name.split('_'))
 head='kernel void '+name+'(constant '+title+'Resources&r [[buffer(0)]],constant '+title+'Params&params [[buffer(1)]],uint gid [[thread_position_in_grid]]){\n'
 for param in params:
  tokens=re.findall(r'\w+|\*',param);var=tokens[-1]
  if '*' in tokens:head+='device '+('const ' if 'const' in tokens else '')+('int' if 'int' in tokens else 'float')+' *'+var+'=r.'+var+';\n'
  else:head+='const '+('float' if 'float' in tokens else 'int')+' '+var+'=params.'+var+';\n'
 if name=='goddard_bands_columns':
  head+='if(gid>=uint(columns))return;\nGddArena arena={r.workspace+ulong(gid)*ulong(params.workspace_words),0ul};\n'
 text=head+untok(rewrite_division(map_types(tokenize(body.replace('GReal','Soft8')))))+'\n}'
 return text,signature(name,params)


def main():
 items=preprocess()
 reached,_=closure(items,NAMES)
 # GArray's templated declaration is a supplied address-space aware view.
 functions={def_name(x) for x in reached if x.startswith('__device__ void g_') and '{' in x}
 refs={}
 for x in reached:
  if def_name(x) in functions and '{' in x:
   name,params=decl_info(x);refs[name]=[i for i,p in enumerate(params) if '&' in p]
 helpers=[];kernels=[];signatures=[]
 for item in reached:
  name=def_name(item)
  if name is None or name in PROVIDED or item.startswith(('using','template<class T,int R>')):continue
  if name in NAMES:
   text,sig=kernel(item,functions,refs);kernels.append(text);signatures.append(sig)
  elif name in functions:
   if '{' in item:helpers.append(translate_function(item,functions,refs))
  elif item.startswith('template'):
   helpers.append(untok(map_types(tokenize(item))).replace('__device__',''))
  elif '{' in item and '(' in item.split('{',1)[0] and not item.startswith('struct'):
   if name in ('gdd_ozone',):
    helpers.append(convert_function_item(item))
   elif name in MATH or name.startswith('gfk_'):
    helpers.append(convert_function_item(item))
  elif '=' in item:
   helpers.append(global_table(item))
 # Definitions in CUDA follow forward declarations; template argument types
 # remain deduced in the same order, so emit matching forward declarations.
 declarations=[]
 for text in helpers:
  if re.search(r'\b(?:'+('|'.join(functions))+r')\s*\(',text):
   declarations.append(text.split('{',1)[0]+';')
 fragment='// Generated from the CUDA Goddard source by tools/goddard_metal/generate.py.\n'
 fragment+='#pragma clang diagnostic ignored \"-Wparentheses-equality\"\n#pragma clang diagnostic ignored \"-Wmissing-braces\"\n'
 fragment+='\n'.join(declarations+helpers)
 # Keep the model's portable binary64 implementation and rounding contract.
 fragment=fragment.replace('GReal','Soft8')
 (SHADERS/'goddard_fragment.metalh').write_text(fragment+'\n')
 (SHADERS/'goddard_family.metal').write_text('// Generated by tools/goddard_metal/generate.py; do not edit.\n#include <metal_stdlib>\n#include "goddard_support.metalh"\n#include "goddard_family_bindings.metalh"\n#include "goddard_fragment.metalh"\n'+'\n\n'.join(kernels)+'\n')
 (ROOT/'woof/metal/signature_fragments/goddard_family.json').write_text(json.dumps({'kernels':signatures},indent=2)+'\n')
 lines=['"""Generated Goddard per-column device storage requirements."""',
        'def workspace_words(levels):','    np_ = int(levels)',
        '    if np_ < 2:','        raise ValueError("Goddard needs at least two levels")']
 emitted=set()
 def emit(name,active=()):
  if name in emitted:return
  if name in active:raise ValueError('recursive Goddard workspace graph')
  children=CALLS.get(name,())
  for child in children:emit(child,active+(name,))
  terms=['(%d * (%s) + 7) // 8'%(size,re.sub(r'\bnp\b','np_',expr)) for size,expr in ALLOCS.get(name,())]
  own=' + '.join(terms) or '0'
  tail=' + max(['+', '.join('_'+child for child in children)+'])' if children else ''
  lines.append('    _'+name+' = '+own+tail)
  emitted.add(name)
 emit('goddard_bands_columns')
 lines.append('    return _goddard_bands_columns')
 (ROOT/'woof/metal/goddard_workspace.py').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
