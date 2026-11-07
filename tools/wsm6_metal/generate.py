"""Generate the native WSM6 Metal shaders and ABI fragments from WOOF source."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]

def generate_state(ROOT, OUT):
    """Source-derived native WSM6 core state initialization, private prototype."""
    import ast,json,re

    module=ast.parse((ROOT/'woof/hex/mpasphys/wsm6_v842.py').read_text())
    source=next(ast.literal_eval(x.value) for x in module.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_MODULES' for t in x.targets))
    def function(name):
        matches=list(re.finditer(r'__device__\s+(?:void|float)\s+'+name+r'\([^;]*?\)\s*\{',source))
        if len(matches)!=1:raise ValueError((name,len(matches)))
        m=matches[0];i=m.end();depth=1
        while depth:depth+=(source[i]=='{')-(source[i]=='}');i+=1
        return source[m.start():i]
    params=function('state_parameters');init=function('mp_wsm6_init');gamma=function('rgmma')
    core='\n'.join(function(n) for n in ('mp_wsm6_run','rgmma','fpvs','slope_wsm6','slope_rain','slope_snow','slope_graup','nislfv_rain_plm','nislfv_rain_plm6'))
    core_fields=set(re.findall(r'\bS\.(\w+)',core)) - {'workspace','last_message'}
    struct=source.split('struct Wsm6State {',1)[1].split('};',1)[0]
    fields=re.findall(r'^float (\w+);$',struct.split('int nrbins;',1)[0],flags=re.M)
    needed=set(fields)
    missing=core_fields-set(fields)
    if missing:raise ValueError(('non-float state needed by core',missing))
    fields=[n for n in fields if n in needed]
    # Remove radar-only assignments and the radar initializer for this core-only replay.
    def prune(text):
        lines=[]
        for line in text.splitlines():
            if 'radar_init(S)' in line or 'S.last_message =' in line:continue
            m=re.match(r'\s*S\.(\w+)(?:\[.*?\])?\s*=',line)
            if m and m.group(1) not in needed:continue
            lines.append(line)
        return '\n'.join(lines)
    params=prune(params);init=prune(init)
    assert 'mp_atan(1.f)' in init
    # Selected glibc atanf(1.f) is exactly this binary32 word; host oracle checks it.
    init=init.replace('mp_atan(1.f)','f32_word(0x3f490fdbu)')
    init=init.replace('S.deng = 500;', 'S.deng = 500.f;')
    def lower(text):
        text=text.replace('__device__','static inline').replace('Wsm6State &S','thread Wsm6State &S').replace('int &errflg','thread int &errflg')
        text=re.sub(r'\bfloat\b','F',text)
        text=text.replace('(F)(i)','F(float(i))')
        for old,new in [('add','tc_add'),('sub','tc_sub'),('mul','tc_mul'),('div','tc_div'),('mp_pow','tc_pow'),('mp_exp','tc_exp')]:text=re.sub(r'\b'+old+r'\(',new+'(',text)
        return text
    params,init,gamma=map(lower,(params,init,gamma))
    entry='hexv842_wsm6_state_private';title='Hexv842Wsm6StatePrivate'
    header='''#include <metal_stdlib>
using namespace metal;
#include "hexv842_mpas_thompson_column_math.metalh"
using namespace ntc;
// This reproduces the source's explicit operation order for the two powers used by init.
template<int N> static inline F ipow(F x) {
 if constexpr (N==3) return tc_mul(tc_mul(x,x),x);
 if constexpr (N==8) {F x2=tc_mul(x,x),x4=tc_mul(x2,x2);return tc_mul(x4,x4);}
}
'''
    structure='struct Wsm6State {\n'+''.join(' F '+n+';\n' for n in fields)+'};\n'
    body='\n'.join((params,gamma,init))
    kernel='''\n#include "hexv842_wsm6_state_private_bindings.metalh"
kernel void hexv842_wsm6_state_private(constant Hexv842Wsm6StatePrivateResources &r [[buffer(0)]],constant Hexv842Wsm6StatePrivateParams &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 if (gid >= uint(p.count)) return;
 Wsm6State S{};int err=0;state_parameters(S);
 device const F *c=(device const F *)r.constants;
 mp_wsm6_init(S,c[15],c[16],100.f,c[17],c[2],r.hail[gid],err);
'''+''.join(' r.state['+str(i)+'*p.count+gid]=as_type<float>(S.'+name+'.bits);\n' for i,name in enumerate(fields))+''' r.status[gid]=err;
}
'''
    (OUT/(entry+'.metal')).write_text(header+structure+body+kernel)
    resources=[['constants','float','read'],['hail','int32','read'],['state','float','read_write'],['status','int32','read_write']]
    spec={'kernels':[dict(name=entry,module=entry,source='woof/metal/shaders/'+entry+'.metal',cuda_source='woof/hex/mpasphys/wsm6_v842.py',cuda_symbol='wsm6_state_template',resources=resources,shapes={n:[['any_1d']] for n,_,_ in resources},parameters=[['count','int32']],threads_per_group=32,layout_version=1,arithmetic_policies=['default'])]}
    (OUT/(entry+'.json')).write_text(json.dumps(spec,indent=2)+'\n')
    (OUT/'state_fields.json').write_text(json.dumps(fields,indent=2)+'\n')
    binding='struct '+title+'Resources {\n'+''.join('device '+('const ' if mode=='read' else '')+('int' if typ=='int32' else 'float')+' *'+n+' [[id('+str(i)+')]];\n' for i,(n,typ,mode) in enumerate(resources))+'};\nstruct '+title+'Params {int count;};\n'
    (OUT/(entry+'_bindings.metalh')).write_text(binding)
    # Preserve every source-declared scratch plane and exact padding expression.
    ws=source.split('struct Wsm6Workspace {',1)[1].split('};',1)[0]
    planes=[]
    for typ,name,expr in re.findall(r'^\s*(float|bool|double)\s+(\w+)\[([^\]]+)\];',ws,flags=re.M):
        if name.startswith(('mp_wsm6_run_','slope_','nislfv_rain_plm')):planes.append(dict(name=name,dtype=typ,elements=expr))
    if not planes or any(row['dtype']!='float' for row in planes):raise ValueError('unexpected core scratch')
    (OUT/'workspace.json').write_text(json.dumps(planes,indent=2)+'\n')
    print('core state fields',len(fields),'scratch arrays',len(planes))


def corrected_source(source, replacements, finding):
    for old, new, expected in replacements[finding]:
        if source.count(old) != expected:
            raise ValueError(f"Native WSM6 source anchors changed for {finding}")
        source = source.replace(old, new)
    return source


def generate_column(ROOT, OUT, *, wb5=False):
    """Lower the selected WSM6 column, with the optional WB5 transport fix."""
    import ast,json,re
    O=OUT
    nodes=ast.parse((ROOT/'woof/hex/mpasphys/wsm6_v842.py').read_text()).body
    src=next(ast.literal_eval(x.value) for x in nodes if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CUDA_MODULES' for t in x.targets))
    if wb5:
        replacements=next(ast.literal_eval(x.value) for x in nodes if isinstance(x,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='CORRECT_MODE_SOURCE' for t in x.targets))
        src=corrected_source(src,replacements,'WB5')
    fields=json.loads((O/'state_fields.json').read_text());planes=json.loads((O/'workspace.json').read_text())
    base=(O/'hexv842_wsm6_state_private.metal').read_text().split('#include "hexv842_wsm6_state_private_bindings.metalh"')[0]
    base=base.replace('if constexpr (N==3)', 'if constexpr (N==2) return tc_mul(x,x);\n if constexpr (N==3)')
    ws='struct Wsm6Workspace {\n'+''.join(' device F *'+p['name']+';\n' for p in planes)+'};\n'
    base=base.replace('struct Wsm6State {',ws+'struct Wsm6State {\n thread Wsm6Workspace *workspace;')
    def function(name):
        matches=list(re.finditer(r'__device__\s+(?:void|float)\s+'+name+r'\([^;]*?\)\s*\{',src));assert len(matches)==1,(name,len(matches))
        m=matches[0];i=m.end();depth=1
        while depth:depth+=(src[i]=='{')-(src[i]=='}');i+=1
        return src[m.start():i]
    def lower(name):
        text=function(name)
        text=re.sub(r'#ifdef __CUDA_ARCH__\n(.*?)#else\n.*?#endif',lambda m:m.group(1),text,flags=re.S)
        if '#if' in text:raise ValueError('unexpected preprocessor in '+name)
        if name in ('nislfv_rain_plm','nislfv_rain_plm6'):
            close='1890' if name=='nislfv_rain_plm' else '2153'
            token='} // mp_wsm6.F90:'+close
            assert text.count('label100:;')==1 and text.count('goto label100;')==1 and text.count(token)==1
            text=text.replace('label100:;','while (true) {').replace('goto label100;','continue;').replace(token,token+'\n  break; }')
            for label in ('i_loop_cycle','find_kb_cycle','find_kt_cycle','intp_cycle','sum_precip_cycle'):
                text=text.replace('goto '+label+';','continue;')
            for label in ('find_kb_exit','find_kt_exit','intp_exit','sum_precip_exit'):
                text=text.replace('goto '+label+';','break;')
            text=re.sub(r'\b(?:i_loop|ist_loop|find_kb|find_kt|intp|sum_precip)_(?:cycle|exit):;', '',text)
            if re.search(r'\bgoto\b|\b\w+:;',text):raise ValueError('untranslated control flow in '+name)
        pos=text.index('{');sig,body=text[:pos],text[pos:]
        pointers=[]
        def parameter(m):
            n='P'+str(len(pointers));pointers.append(n);return n+' '+m.group(1)
        sig=re.sub(r'(?:const )?float\s*\*\s*(\w+)',parameter,sig)
        sig=sig.replace('__device__','static inline').replace('Wsm6State &S','thread Wsm6State &S').replace('int &errflg','thread int &errflg')
        if pointers:sig='template<'+','.join('class '+n for n in pointers)+'>\n'+sig
        body=re.sub(r'float \*(\w+)=S\.workspace->',r'device F *\1=S.workspace->',body)
        text=sig+body
        text=re.sub(r'\bfloat\b','F',text)
        text=re.sub(r'\(F\)\((\w+)\)',r'F(float(\1))',text)
        text=text.replace('S.last_message = "mp_wsm6_run OK";', '')
        for old,new in [('add','tc_add'),('sub','tc_sub'),('mul','tc_mul'),('div','tc_div'),('mp_pow','tc_pow'),('mp_exp','tc_exp'),('mp_log','tc_log'),('mp_sqrt','tc_sqrt'),('mp_abs','tc_abs'),('mp_min','tc_min'),('mp_max','tc_max'),('mp_nint','wsm_nint')]:text=re.sub(r'\b'+old+r'\(',new+'(',text)
        return text
    names=('fpvs','slope_wsm6','slope_rain','slope_snow','slope_graup','nislfv_rain_plm','nislfv_rain_plm6','mp_wsm6_run')
    body='static inline int wsm_nint(F x){return int(round(as_type<float>(x.bits)));}\n'+'\n'.join(lower(n) for n in names)
    # Lower source statement-function lambdas to explicit helpers; Metal 2.4
    # does not support lambdas. Each capture becomes a value/reference argument.
    captures={
     'lamdar':('thread Wsm6State &S','S'), 'lamdas':('thread Wsm6State &S','S'),
     'lamdag':('thread Wsm6State &S','S'), 'cpmcal':('F cpd,F qmin,F cpv','cpd,qmin,cpv'),
     'xlcal':('thread Wsm6State &S,F xlv0,F t0c','S,xlv0,t0c'),
     'diffus':('',''), 'viscos':('',''), 'xka':('',''),
     'diffac':('F rv','rv'), 'venfac':('F den0','den0'), 'conden':('F qmin,F rv','qmin,rv'),
    }
    found={};pattern=r'auto\s+(\w+)\s*=\s*\[&\]\(([^)]*)\)\s*\{\s*return\s*(.*?);\s*\};'
    def remove_lambda(m):
        n,params,expr=m.groups()
        if n not in captures:raise ValueError('unknown source lambda '+n)
        definition=(params,expr)
        if n in found and found[n]!=definition:raise ValueError('conflicting source lambda '+n)
        found[n]=definition
        return ''
    body,count=re.subn(pattern,remove_lambda,body)
    if count!=14 or set(found)!=set(captures):raise ValueError(('source lambda count or names changed',count,found.keys()))
    def calls(text):
        for n,(_,args) in captures.items():
            text=re.sub(r'\b'+n+r'\(', 'wsm_'+n+'('+((args+',') if args else ''),text)
        return text
    helpers=[]
    for n,(capture,_) in captures.items():
        params,expr=found[n]
        args=','.join(x for x in (capture,params) if x)
        helpers.append('static inline F wsm_'+n+'('+args+') { return '+calls(expr)+'; }')
    body='\n'.join(helpers)+'\n'+calls(body)
    workspace=' Wsm6Workspace ws; device F *cursor=(device F*)r.workspace+ulong(gid)*ulong(p.workspace_words);\n'
    for p in planes:
        expression=p['elements'].replace('WSM6_KMAX','nz')
        workspace+=' ws.'+p['name']+'=cursor;cursor+=('+expression+');\n'
    workspace+=' device F *local=cursor;cursor+=10*nz;device F *surface=cursor;\n'
    load='\n'.join(' S.'+n+'=((device const F*)r.state)['+str(i)+'*nc+gid];' for i,n in enumerate(fields))
    entry='hexv842_wsm6_column_private';title='Hexv842Wsm6ColumnPrivate'
    kernel='''\n#include "hexv842_wsm6_column_private_bindings.metalh"
kernel void hexv842_wsm6_column_private(constant Hexv842Wsm6ColumnPrivateResources &r [[buffer(0)]],constant Hexv842Wsm6ColumnPrivateParams &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {
 int nz=p.nz,nc=p.nc;if(gid>=uint(p.ns))return;
'''+workspace+''' Wsm6State S{};S.workspace=&ws;
'''+load+'''
 device F *v=(device F*)r.v,*s=(device F*)r.s;device const F *c=(device const F*)r.constants,*pi=(device const F*)r.pi;
 for(int n=0;n<10;++n)for(int k=0;k<nz;++k)local[n*nz+k]=v[(n*nz+k)*nc+gid];
 for(int n=0;n<7;++n)surface[n]=s[n*nc+gid];
 if(p.wrapper)for(int k=0;k<nz;++k)local[k]=tc_mul(local[k],pi[k*nc+gid]);
 int err=0;
 mp_wsm6_run(S,local,local+nz,local+2*nz,local+3*nz,local+4*nz,local+5*nz,local+6*nz,local+7*nz,local+8*nz,local+9*nz,F(p.dt),
 c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[11],c[12],c[13],c[14],c[15],c[16],c[17],c[18],c[19],
 surface,surface+1,surface+6,surface+2,surface+3,surface+4,surface+5,(device F*)nullptr,(device F*)nullptr,1,1,1,nz,err);
 if(p.wrapper)for(int k=0;k<nz;++k){F theta=tc_div(local[k],pi[k*nc+gid]);v[k*nc+gid]=theta;local[k]=tc_mul(theta,pi[k*nc+gid]);}
 for(int n=p.wrapper?1:0;n<10;++n)for(int k=0;k<nz;++k)v[(n*nz+k)*nc+gid]=local[n*nz+k];
 for(int n=0;n<7;++n)s[n*nc+gid]=surface[n];r.status[gid]=err;
}
'''
    if wb5:
        column_entry='hexv842_wsm6_column_wb5_private'
        kernel=kernel.replace(entry,column_entry).replace(title,'Hexv842Wsm6ColumnWb5Private')
        entry=column_entry
    (O/(entry+'.metal')).write_text(base+body+kernel)
    res=[['v','float','read_write'],['s','float','read_write'],['pi','float','read'],['constants','float','read'],['state','float','read'],['workspace','float','read_write'],['status','int32','read_write']]
    params=[['nz','int32'],['nc','int32'],['ns','int32'],['wrapper','int32'],['workspace_words','int32'],['dt','float32']]
    spec={'kernels':[dict(name=entry,module=entry,source='woof/metal/shaders/'+entry+'.metal',cuda_source='woof/hex/mpasphys/wsm6_v842.py',cuda_symbol='wsm6_columns',resources=res,shapes={n:[['any_1d']] for n,_,_ in res},parameters=params,threads_per_group=32,layout_version=1,arithmetic_policies=['default'])]}
    (O/(entry+'.json')).write_text(json.dumps(spec,indent=2)+'\n')
    binding='struct '+title+'Resources {\n'+''.join('device '+('const ' if mode=='read' else '')+('int' if typ=='int32' else 'float')+' *'+n+' [[id('+str(i)+')]];\n' for i,(n,typ,mode) in enumerate(res))+'};\nstruct '+title+'Params {\n'+''.join(('int ' if typ=='int32' else 'float ')+n+';\n' for n,typ in params)+'};\n'
    (O/(entry+'_bindings.metalh')).write_text(binding)
    print('generated column, workspace arrays',len(planes))


def generate_seam(ROOT, OUT, *, wb3=False):
    """Lower the selected native WSM6 boundary and optional WB3 writeback."""
    import json,re

    import ast, types
    tree=ast.parse((ROOT/"woof/hex/mpasphys/wsm6_v842.py").read_text())
    values={node.targets[0].id:ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ("FIELD_NAMES","CUDA_DRIVER","CORRECT_MODE_SOURCE")}
    w=types.SimpleNamespace(**values)
    out=OUT
    source=w.CUDA_DRIVER
    if wb3:
        source=corrected_source(source,w.CORRECT_MODE_SOURCE,'WB3')
    for phase in (('to',) if wb3 else ('from','to')):
        name='wsm6_from_mpas' if phase=='from' else 'wsm6_to_mpas'
        start=source.index('extern "C" __global__ void '+name+'(')
        brace=source.index('{',start); depth=1; end=brace+1
        while depth:
            depth+=(source[end]=='{')-(source[end]=='}'); end+=1
        body=source[brace+1:end-1]
        body=body.replace('int i=blockIdx.x*blockDim.x+threadIdx.x;', 'int i=int(gid);')
        for old,new in [('add','tc_add'),('sub','tc_sub'),('mul','tc_mul'),('div','tc_div'),('mp_max','tc_max'),('mp_pow','tc_pow')]:
            body=re.sub(r'\b'+old+r'\(',new+'(',body)
        resources=[[n,'float','read_write'] for n in w.FIELD_NAMES]
        resources += [[n,'float','read_write'] for n in ('v','s','r','prod')]
        if phase=='from': resources += [['pi','float','read_write']]
        else: resources += [['c','float','read'],['i_rainnc','int32','read_write']]
        params=[['nz','int32'],['nc','int32'],['ns','int32']]
        if phase=='from': params += [['rv_rd','float32']]
        else: params += [['top','int32'],['dt_dyn','float32'],*[ [n,'float32'] for n in ('rd','rv_rd','rvord','rcv','p0','g','bucket')],['acrain','int32']]
        entry='hexv842_wsm6_'+phase+('_wb3_private' if wb3 else '_private')
        title=''.join(x.title() for x in entry.split('_'))
        macros='\n'.join('#define FIELD_'+n+' '+str(i) for i,n in enumerate(w.FIELD_NAMES))
        macros+='\n#define F(name,k) fields[FIELD_##name][(k)*nc+i]\n#define SF(name) fields[FIELD_##name][i]\n'
        aliases='\n'.join(('int ' if t=='int32' else 'float ')+n+'=p.'+n+';' for n,t in params)
        aliases+='\n'+'\n'.join(('device '+('const ' if mode=='read' else '')+('int *' if typ=='int32' else 'float *'))+n+'=res.'+n+';' for n,typ,mode in resources if n not in w.FIELD_NAMES)
        aliases+='\ndevice float *fields[] = {'+','.join('res.'+n for n in w.FIELD_NAMES)+'};\n'
        constants='\n'.join('constant int '+key+'[] = {'+','.join('FIELD_'+x for x in vals)+'};' for key,vals in {'SPECIES':['qv','qc','qi','qr','qs','qg'],'RADII':['re_cloud','re_ice','re_snow'],'PRODUCTION':['rainprod','evapprod'],'TENDENCIES':['rqvmpten','rqcmpten','rqimpten','rqrmpten','rqsmpten','rqgmpten']}.items())
        shader='#include <metal_stdlib>\nusing namespace metal;\n#include "hexv842_mpas_thompson_scalar.metalh"\n#include "'+entry+'_bindings.metalh"\n#pragma clang fp contract(off)\n'+macros+constants+'\nkernel void '+entry+'(constant '+title+'Resources &res [[buffer(0)]],constant '+title+'Params &p [[buffer(1)]],uint gid [[thread_position_in_grid]]) {\n'+aliases+'\n'+body+'\n}\n'
        (out/(entry+'.metal')).write_text(shader)
        fragment={'kernels':[dict(name=entry,module=entry,source='woof/metal/shaders/'+entry+'.metal',cuda_source='woof/hex/mpasphys/wsm6_v842.py',cuda_symbol=name,resources=resources,shapes={n:[['any_1d']] for n,_,_ in resources},parameters=params,threads_per_group=32,layout_version=1,arithmetic_policies=['default'])]}
        (out/(entry+'.json')).write_text(json.dumps(fragment,indent=2)+'\n')
        binding='struct '+title+'Resources {\n'+''.join('device '+('const ' if mode=='read' else '')+('int' if typ=='int32' else 'float')+' *'+n+' [[id('+str(i)+')]];\n' for i,(n,typ,mode) in enumerate(resources))+'};\nstruct '+title+'Params {\n'+''.join(('int ' if typ=='int32' else 'float ')+n+';\n' for n,typ in params)+'};\n'
        (out/(entry+'_bindings.metalh')).write_text(binding)
        print(entry,len(resources),'resources')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT,
                        help='repository-shaped destination (default: repository root)')
    args = parser.parse_args()
    with TemporaryDirectory() as temp:
        output = Path(temp)
        generate_state(ROOT, output)
        generate_column(ROOT, output)
        generate_column(ROOT, output, wb5=True)
        generate_seam(ROOT, output)
        generate_seam(ROOT, output, wb3=True)
        for phase in ('state', 'column', 'from', 'to', 'column_wb5', 'to_wb3'):
            old = 'hexv842_wsm6_' + phase + '_private'
            new = 'hexv842_wsm6_' + phase + '_f32'
            for ext, directory in (('.metal', 'shaders'), ('.json', 'signature_fragments')):
                text = (output / (old + ext)).read_text()
                title = ''.join(word.title() for word in phase.split('_'))
                text = text.replace(old, new).replace('Hexv842Wsm6' + title + 'Private',
                                                       'Hexv842Wsm6' + title + 'F32')
                destination = args.output_root / 'woof' / 'metal' / directory
                destination.mkdir(parents=True, exist_ok=True)
                (destination / (new + ext)).write_text(text)

if __name__ == '__main__':
    main()
