"""Source-only CUDA/Metal floating-literal census and function comparison.

This flags review candidates, not proven defects. No C/C++ preprocessing or
semantic evaluation is claimed. Unmatched functions and globals remain in the
inventory; ambiguous pairing and software-FP64 words are explicit.
"""
from __future__ import annotations
import argparse,collections,csv,difflib,json,math,re,struct
from functools import lru_cache
from decimal import Decimal
from pathlib import Path

TOKEN = re.compile(r'(?<![\w.])(?:0[xX](?:[0-9a-fA-F]+\.?[0-9a-fA-F]*|\.[0-9a-fA-F]+)[pP][+-]?\d+[fFlL]?|(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?[fFlL]?|\d+[eE][+-]?\d+[fFlL]?|\d+[fF]|0[xX][0-9a-fA-F]{16}(?:[uUlL]+)?)(?![\w.])')
STRIP=re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')
SIGNATURE=re.compile(r'\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?\{')
NAME_OPEN=re.compile(r'\b([A-Za-z_]\w*)\s*\(')
GENERIC={0.,.5,1.,2.,3.,4.,5.,6.,8.,10.,16.,32.,64.,100.,256.,1000.}
BRACES=re.compile(r'[{}]|^[ \t]*#[ \t]*(?:if\b|ifdef\b|ifndef\b|elif\b|else\b|endif\b)[^\n]*',re.M)
KEY_CACHE={};INFO_CACHE={}
OVERRIDES={
    ('ysu_column.metal','ysu_column'):('ysu.cu','ysu_column_body'),
    ('ysu_column.metal','ydiagnose'):('ysu.cu','ysu_diagnose'),
    ('ysu_column.metal','ythomas'):('ysu.cu','ysu_thomas'),
    ('thompson_frozen_vapor_network.metal','trslf'):('thompson.cu','thompson_rslf'),
    ('thompson_warm_frozen_source.metal','trslf'):('thompson.cu','thompson_rslf'),
    ('thompson_aerosol_cold_family.metal','cold_rslf'):('thompson_aerosol_common.cuh','thompson_rslf'),
    ('thompson_aerosol_warm_family.metal','thompson_aa_rslf'):('thompson_aerosol_common.cuh','thompson_rslf'),
    ('thompson_aerosol_wrf39_sed_family.metal','wrf39_ice_column'):('thompson_aerosol_sed.cu','thompson_aa_wrf39_ice_sediment_impl'),
}

def stripped(source):
    return STRIP.sub(lambda m:re.sub(r'[^\n]',' ',m.group()),source)

def function_headers(code):
    """Balance argument parentheses, permitting aggregate default arguments."""
    excluded={'if','for','while','switch','catch','sizeof','alignof','defined'}
    for m in NAME_OPEN.finditer(code):
        name=m.group(1)
        if name in excluded:continue
        depth=1;pos=m.end()
        while pos<len(code) and depth:
            ch=code[pos]
            if ch==';':break
            if ch=='(':depth+=1
            elif ch==')':depth-=1
            pos+=1
        if depth:continue
        tail=re.match(r'\s*(?:const\s*)?\{',code[pos:])
        if tail:yield name,m.start(),pos+tail.end()-1

def occurrences(code,begin,end):
    out=[]
    for m in TOKEN.finditer(code,begin,end):
        raw=m.group();hex_word=raw.lower().startswith('0x') and 'p' not in raw.lower()
        if hex_word:
            word=re.sub(r'[uUlL]+$','',raw)
            value=struct.unpack('>d',int(word,16).to_bytes(8,'big'))[0]
            if not math.isfinite(value):continue
            kind='possible_software_fp64_word'
        else:
            token=re.sub(r'[fFlL]$','',raw)
            value=float.fromhex(token) if token.lower().startswith('0x') else float(token)
            kind='floating_literal'
            prefix=code[begin:m.start()].rstrip()
            if prefix.endswith('-') and (len(prefix)==1 or prefix[:-1].rstrip()[-1:] in '=([{,:+*/?'):
                value=-value
        if not math.isfinite(value):continue
        written_value=value
        if kind=='floating_literal' and raw[-1:] in 'fF':
            value=struct.unpack('f',struct.pack('f',value))[0]
        key=str(value) if value else '0.0'
        out.append({'literal':raw,'value':key,'written_value':str(written_value),'kind':kind,'line':code.count('\n',0,m.start())+1,'offset':m.start()})
    return out

def scan(path,root,backend):
    raw=path.read_text(errors='replace');code=stripped(raw)
    funcs=[];covered=[]
    for name,start,opening in function_headers(code):
        depth=1;end=opening+1;branches=[]
        for token in BRACES.finditer(code,opening+1):
            value=token.group();directive=re.match(r'\s*#\s*(\w+)',value)
            if directive:
                kind=directive.group(1)
                if kind in {'if','ifdef','ifndef'}:branches.append([depth,[]])
                elif kind in {'else','elif'} and branches:
                    branches[-1][1].append(depth);depth=branches[-1][0]
                elif kind=='endif' and branches:
                    baseline,alternatives=branches.pop();depth=max([depth,*alternatives])
            elif value=='{':depth+=1
            elif value=='}':depth-=1
            end=token.end()
            if depth==0:break
        if depth:continue
        # Ignore signature-like constructs inside an already parsed function.
        if any(a<=opening<b for a,b in covered):continue
        covered.append((opening,end))
        funcs.append({'backend':backend,'file':str(path.relative_to(root)),'function':name,'line':code.count('\n',0,start)+1,'literals':occurrences(code,opening,end)})
    outside=list(code)
    for a,b in covered:
        outside[a:b]=['\n' if c=='\n' else ' ' for c in outside[a:b]]
    funcs.append({'backend':backend,'file':str(path.relative_to(root)),'function':'<file_scope>','line':1,'literals':occurrences(''.join(outside),0,len(code))})
    return funcs

def keys(f):
    if id(f) not in KEY_CACHE:KEY_CACHE[id(f)]=set(x['value'] for x in f['literals'])
    return KEY_CACHE[id(f)]
def informative(f):
    if id(f) not in INFO_CACHE:INFO_CACHE[id(f)]={x for x in keys(f) if abs(float(x)) not in GENERIC}
    return INFO_CACHE[id(f)]
@lru_cache(None)
def family(path):
    stem=Path(path).stem
    return re.sub(r'_(?:family|fragment|device|common)$','',stem)

@lru_cache(None)
def text_similarity(a,b):return difflib.SequenceMatcher(None,a,b).ratio()

def similarity(a,b):
    aa,bb=informative(a),informative(b);shared=len(aa&bb)
    jac=shared/max(1,len(aa|bb))
    name=text_similarity(a['function'],b['function'])
    file=text_similarity(family(a['file']),family(b['file']))
    return jac*.6+name*.15+file*.25,shared,jac

def power_ten(a,b):
    if not a or not b or a*b<=0:return None
    ratio=Decimal(str(b))/Decimal(str(a));n=round(math.log10(float(ratio)))
    return n if n and abs(n)<=18 and abs(ratio/(Decimal(10)**n)-1)<Decimal('1e-6') else None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();args.out.mkdir(parents=True,exist_ok=True)
    cuda=[];metal=[];files=[]
    for path in sorted((root/'woof').rglob('*')):
        if not path.is_file():continue
        if path.suffix in {'.cu','.cuh'}:backend='cuda';target=cuda
        elif path.suffix in {'.metal','.metalh'} and 'metal/shaders' in path.as_posix():backend='metal';target=metal
        else:continue
        fs=scan(path,root,backend);target.extend(fs);files.append({'backend':backend,'file':str(path.relative_to(root)),'functions':len(fs)-1,'float_literals':sum(x['kind']=='floating_literal' for f in fs for x in f['literals']),'possible_fp64_words':sum(x['kind']!='floating_literal' for f in fs for x in f['literals'])})
    byname=collections.defaultdict(list)
    for c in cuda:byname[c['function']].append(c)
    pairs=[];unmatched=[];flags=[];used=set()
    for m in metal:
        override=OVERRIDES.get((Path(m['file']).name,m['function']))
        direct=([c for c in cuda if Path(c['file']).name==override[0] and c['function']==override[1]] if override else byname[m['function']]) if m['function']!='<file_scope>' else [c for c in cuda if c['function']=='<file_scope>' and family(c['file'])==family(m['file'])]
        method='same_function_name' if m['function']!='<file_scope>' else 'same_file_family_globals'
        if override:method='source_verified_wrapper_or_helper_mapping'
        if direct:
            ranked=sorted(((similarity(c,m),c) for c in direct),key=lambda x:x[0],reverse=True)
        elif len(informative(m))>=4:
            method='heuristic_shared_constants_and_names'
            ranked=sorted(((similarity(c,m),c) for c in cuda if c['function']!='<file_scope>' and len(informative(c))>=4),key=lambda x:x[0],reverse=True)[:2]
            if not ranked or ranked[0][0][1]<4 or ranked[0][0][2]<.35 or ranked[0][0][0]<.45:ranked=[]
        else:ranked=[]
        if not ranked:
            unmatched.append({**m,'reason':'no established counterpart'});continue
        score,c=ranked[0];used.add((c['file'],c['function'],c['line']))
        ambiguous=len(ranked)>1 and abs(score[0]-ranked[1][0][0])<.015
        pair={'cuda_file':c['file'],'cuda_function':c['function'],'cuda_line':c['line'],'metal_file':m['file'],'metal_function':m['function'],'metal_line':m['line'],'matching':method,'score':score[0],'ambiguous':ambiguous}
        ca,ma=keys(c),keys(m);left=ca-ma;right=ma-ca
        pair.update(cuda_only_values=sorted(left,key=float),metal_only_values=sorted(right,key=float),counts_equal=collections.Counter(x['value'] for x in c['literals'])==collections.Counter(x['value'] for x in m['literals']))
        pairs.append(pair)
        def flag(kind,x,y=None,aligned=False):
            co=next((v for v in c['literals'] if v['value']==x),None)
            mo=next((v for v in m['literals'] if v['value']==y),None)
            flags.append({**pair,'kind':kind,'cuda_value':x,'metal_value':y,'power10_exponent':power_ten(float(x),float(y)) if x is not None and y is not None else None,'aligned_single_replacement':aligned,'cuda_literal':co['literal'] if co else None,'cuda_literal_line':co['line'] if co else None,'metal_literal':mo['literal'] if mo else None,'metal_literal_line':mo['line'] if mo else None})
        for x in left:flag('cuda_only_value',x)
        for y in right:flag('metal_only_value',None,y)
        if len(c['literals'])*len(m['literals'])<2000000:
            matcher=difflib.SequenceMatcher(None,[v['value'] for v in c['literals']],[v['value'] for v in m['literals']],autojunk=False)
            for op,i,j,a,b in matcher.get_opcodes():
                if op=='replace' and j-i==1 and b-a==1:
                    x=c['literals'][i]['value'];y=m['literals'][a]['value']
                    if power_ten(float(x),float(y)) is not None:flag('factor_of_ten',x,y,True)
        for x in left:
            if abs(float(x)) in GENERIC:continue
            for y in right:
                if power_ten(float(x),float(y)) is not None and not any(f['kind']=='factor_of_ten' and f['cuda_file']==c['file'] and f['metal_file']==m['file'] and f['cuda_function']==c['function'] and f['metal_function']==m['function'] and f['cuda_value']==x and f['metal_value']==y for f in flags):flag('factor_of_ten',x,y)
    unmatched += [{**c,'reason':'no Metal function paired to this CUDA scope'} for c in cuda if (c['file'],c['function'],c['line']) not in used]
    summary={'method':'Comment/string-stripped source census. f-suffixed literals normalized to stored float32; unsuffixed literals to float64. 16-hex-digit words additionally decoded as possible software FP64. Factor-of-ten tolerance permits float32 rounding. Preprocessor branch braces balanced without choosing one configuration. Function names/source-verified overrides matched first; other renamed functions paired heuristically and marked. Flags are review candidates, not correctness failures. File-scope literals, multiplicity differences and unmatched functions are retained. No integer-expression evaluation, operator/control-flow validation or proof of every semantic pairing.','files':files,'cuda_scopes':len(cuda),'metal_scopes':len(metal),'paired_metal_scopes':len(pairs),'unmatched_cuda_scopes':sum(x['backend']=='cuda' for x in unmatched),'unmatched_metal_scopes':sum(x['backend']=='metal' for x in unmatched),'ambiguous_pairs':sum(x['ambiguous'] for x in pairs),'flag_counts':dict(collections.Counter(x['kind'] for x in flags))}
    for name,data in [('summary',summary),('function_pairs',pairs),('unmatched_scopes',unmatched),('literal_inventory',cuda+metal),('flags',flags)]:
        (args.out/(name+'.json')).write_text(json.dumps(data,indent=2)+'\n')
    columns=['kind','cuda_file','cuda_function','cuda_literal_line','cuda_literal','cuda_value','metal_file','metal_function','metal_literal_line','metal_literal','metal_value','power10_exponent','aligned_single_replacement','matching','ambiguous']
    with (args.out/'flags.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(flags)
    with (args.out/'unmatched_scopes.csv').open('w') as f:
        cols=['backend','file','function','line','literal_occurrences','reason']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        w.writerows({**{k:v for k,v in s.items() if k!='literals'},'literal_occurrences':len(s['literals'])} for s in unmatched)
    with (args.out/'files.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(files[0]));w.writeheader();w.writerows(files)
    print(json.dumps({k:v for k,v in summary.items() if k!='files'},indent=2))

if __name__=='__main__':main()
