"""Lower native GF procedures whose jumps are checked direct loop exits."""
import re
from woof.hex.mpasphys import gf_mpas_v842 as native
from tools.thompson_column_metal.generate import eliminate_gotos

# These source procedures contain only direct loop-exit jumps. The other GF
# procedures need separate, source-specific handling of their branch targets.
DIRECT_EXIT = ('gdinit','cup_output_ens_3d','neg_check','g3init','cup_minimi',
               'cup_maximi','cup_env_clev','cup_env','cup_dd_moisture_new',
               'cu_grell_freitas')


def procedure(name):
    source=native._SCHEME_SOURCE
    pattern=r'__device__\s+(?:void|GFReal)\s+'+re.escape(name)+r'\s*\([^;{]*\)\s*\{'
    match=re.search(pattern,source)
    if match is None:raise ValueError('native GF procedure not found: '+name)
    start=match.end()-1;depth=0
    for end in range(start,len(source)):
        depth+=(source[end]=='{')-(source[end]=='}')
        if depth==0:return source[match.start():end+1]
    raise ValueError('unclosed native GF procedure: '+name)


def lower(name):
    if name not in DIRECT_EXIT:raise ValueError('native GF procedure requires separate branch handling: '+name)
    source=procedure(name)
    before=len(re.findall(r'\bgoto\s+L_\d+\s*;',source))
    if len(re.findall(r'\bgoto\b',source))!=before:raise ValueError('unexpected native GF branch in '+name)
    source=re.sub(r'\bL_(\d+)\b',r'label_\1',source)
    lowered=eliminate_gotos(source)
    if re.search(r'\bgoto\b',lowered):raise ValueError('unlowered native GF jump in '+name)
    lowered=re.sub(r'\b(?:label_\d+|__label_\d+)\s*:\s*;', '',lowered)
    return lowered,before


if __name__=='__main__':
    for name in DIRECT_EXIT:
        source,count=lower(name)
        print(name,'checked loop exits',count,'lowered bytes',len(source))


def lower_partial(name):
    """Rewrite only verified direct loop exits, retaining other source gotos."""
    from tools.thompson_column_metal.generate import _matching_open,_loop_header
    source=procedure(name);count=0
    labels=sorted(set(re.findall(r'\bgoto\s+(\w+)\s*;',source)))
    for label in labels:
        jumps=list(re.finditer(r'\bgoto\s+'+re.escape(label)+r'\s*;',source))
        matches=list(re.finditer(r'\b'+re.escape(label)+r'\s*:\s*;',source))
        if len(matches)!=1:continue
        target=matches[0];before=source[:target.start()].rstrip()
        if not before.endswith('}'):continue
        close=len(before)-1;opening=_matching_open(source,close,'}','{')
        if _loop_header(source,opening) not in ('while','for'):continue
        if any(not(opening<j.start()<close) for j in jumps):continue
        block=source[opening+1:close];nested=False
        for jump in re.finditer(r'\bgoto\s+'+re.escape(label)+r'\s*;',block):
            stack=[]
            for t in re.finditer(r'[{}]',block[:jump.start()]):
                if t.group()=='{':stack.append(_loop_header(block,t.start())!='')
                else:stack.pop()
            if any(stack):nested=True;break
        if nested:continue
        temporary='label_port_'+label
        source=re.sub(r'\b'+re.escape(label)+r'\b',temporary,source)
        source=eliminate_gotos(source);count+=len(jumps)
    return source,count


def lower_kbcon():
    """Structure the native cup_kbcon search/restart branch without gotos."""
    source,count=lower_partial('cup_kbcon')
    guard='if (ierr[(long long) i + offset_1575] != 0) goto __label_000027;'
    entry='goto __label_000032;';increment='__label_000031:;';test='__label_000032:;';done='__label_000027:;'
    for token,expected in ((guard,1),(entry,2),(increment,1),(test,1),(done,1)):
        if source.count(token)!=expected:raise ValueError('native GF kbcon control changed: '+token)
    before,rest=source.split(guard,1)
    setup,rest=rest.split(entry,1)
    _,rest=rest.split(increment,1)
    step,rest=rest.split(test,1)
    iteration,after=rest.split(done,1)
    step=step.replace('goto __label_000027;','break;')
    iteration=iteration.replace('goto __label_000027;','break;')
    iteration=iteration.replace('goto __label_000031;','continue;')
    iteration=iteration.replace(entry,'gf_first=true; continue;')
    out=(before+'if (ierr[(long long) i + offset_1575] == 0) {'+setup+
         'bool gf_first=true; while (true) { if (!gf_first) {'+step+
         '} gf_first=false;'+iteration+'break; } }'+after)
    if re.search(r'\bgoto\b',out):raise ValueError('unlowered native GF kbcon jump')
    return out,count


def lower_aa0():
    """Structure the native updraft loop's skip-to-next-column branches."""
    source,count=lower_partial('cup_up_aa0');jump='goto __label_000100;';label='__label_000100:;'
    if source.count(jump)!=4 or source.count(label)!=1:raise ValueError('native GF updraft branch changed')
    first=source.index('if (ierr[(long long) i + offset_1442] != 0) '+jump)
    last=source.index(label,first);segment=source[first:last]
    if re.search(r'\b(?:while|for)\s*\(',segment):raise ValueError('native GF updraft skip crosses a loop')
    segment=segment.replace(jump,'break;')
    out=source[:first]+'do {\n'+segment+'} while(false);'+source[last+len(label):]
    if re.search(r'\bgoto\b',out):raise ValueError('unlowered native GF updraft jump')
    return out,count


def lower_ktop():
    """Structure the native cloud-top found/not-found branch."""
    source,count=lower_partial('cup_ktop')
    anchor='if (ierr[(long long) i + offset_1533] == 0)'
    found='goto __label_000041;';join='goto __label_000042;\n              __label_000041:;'
    for token in (anchor,found,join,'L_372:;','__label_000042:;'):
        if source.count(token)!=1:raise ValueError('native GF cloud-top control changed: '+token)
    start=source.index(anchor);brace=source.index('{',start)
    source=source[:brace+1]+'\n bool gf_found=false;'+source[brace+1:]
    source=source.replace(found,'gf_found=true; break;')
    index=source.index('if (*ilo == 1)',brace)
    source=source[:index]+'if (!gf_found) {\n'+source[index:]
    source=source.replace(join,'} else {')
    source=source.replace('L_372:;','L_372:;\n }')
    source=source.replace('__label_000042:;','')
    if re.search(r'\bgoto\b',source):raise ValueError('unlowered native GF cloud-top jump')
    return source,count


def lower_up_moisture():
    """Structure the native moisture loop's skip-to-next-column branches."""
    source,count=lower_partial('cup_up_moisture');jump='goto __label_000100;';label='__label_000100:;'
    if source.count(jump)!=3 or source.count(label)!=1:raise ValueError('native GF moisture branch changed')
    first=source.index('if (ierr[(long long) i + offset_842] != 0) '+jump)
    last=source.index(label,first);segment=source[first:last]
    if re.search(r'\b(?:while|for)\s*\(',segment):raise ValueError('native GF moisture skip crosses a loop')
    out=source[:first]+'do {\n'+segment.replace(jump,'break;')+'} while(false);'+source[last+len(label):]
    if re.search(r'\bgoto\b',out):raise ValueError('unlowered native GF moisture jump')
    return out,count


def lower_dd_edt():
    """Structure the native downdraft loop's skip-to-next-column branch."""
    source,count=lower_partial('cup_dd_edt');jump='goto __label_000062;';label='__label_000062:;'
    if source.count(jump)!=1 or source.count(label)!=1:raise ValueError('native GF downdraft branch changed')
    first=source.index('if (ierr[(long long) i + offset_2063] != 0) '+jump)
    last=source.index(label,first);segment=source[first:last]
    if re.search(r'\b(?:while|for)\s*\(',segment):raise ValueError('native GF downdraft skip crosses a loop')
    out=source[:first]+'do {\n'+segment.replace(jump,'break;')+'} while(false);'+source[last+len(label):]
    if re.search(r'\bgoto\b',out):raise ValueError('unlowered native GF downdraft jump')
    return out,count


def lower_forcing():
    """Remove the native forcing branch's redundant jump past its else arm."""
    source,count=lower_partial('cup_forcing_ens_3d');jump='goto __label_000100;';label='__label_000100:;'
    pattern=r'goto __label_000100;\s*}\s*L_404:;\s*}\s*else\s*\{'
    if source.count(jump)!=1 or source.count(label)!=1 or len(re.findall(pattern,source))!=1:
        raise ValueError('native GF forcing branch changed')
    source=source.replace(jump,'').replace(label,'')
    if re.search(r'\bgoto\b',source):raise ValueError('unlowered native GF forcing jump')
    return source,count


def lower_wrapped_exit(name):
    """Lower GF loop exits whose target follows only closing scopes and labels."""
    if name not in ('cup_gf','cup_gf_sh'):raise ValueError('GF wrapped exits require the named deep/shallow source')
    from tools.thompson_column_metal.generate import _loop_header
    source,count=lower_partial(name);changed=0
    for label in sorted(set(re.findall(r'\bgoto\s+(\w+)\s*;',source))):
        jumps=list(re.finditer(r'\bgoto\s+'+re.escape(label)+r'\s*;',source));targets=list(re.finditer(r'\b'+re.escape(label)+r'\s*:\s*;',source))
        if len(targets)!=1:continue
        target=targets[0];valid=True
        for jump in jumps:
            stack=[]
            for token in re.finditer(r'[{}]',source[:jump.start()]):
                if token.group()=='{':stack.append((token.start(),_loop_header(source,token.start())))
                else:stack.pop()
            loops=[pos for pos,kind in stack if kind in ('while','for')]
            if not loops:valid=False;break
            opening=loops[-1];depth=0;close=None
            for i in range(opening,len(source)):
                depth+=(source[i]=='{')-(source[i]=='}')
                if depth==0:close=i;break
            if close is None or not jump.start()<close<target.start():valid=False;break
            trail=source[close+1:target.start()]
            trail=re.sub(r'\b(?:L_\d+|__label_\d+)\s*:\s*;','',trail)
            if re.sub(r'[\s{}]','',trail):valid=False;break
        if not valid:continue
        source=re.sub(r'\bgoto\s+'+re.escape(label)+r'\s*;','break;',source)
        source=re.sub(r'\b'+re.escape(label)+r'\s*:\s*;','',source);changed+=len(jumps)
    return source,count,changed


def lower_shallow():
    """Structure native shallow error skips and first-valid ensemble search."""
    source,count,_=lower_wrapped_exit('cup_gf_sh')
    for label,expected in (('__label_000887',1),('__label_000042',2)):
        jump='goto '+label+';';target=label+':;'
        if source.count(jump)!=expected or source.count(target)!=1:raise ValueError('native GF shallow branch changed: '+label)
        first=source.index(jump)
        anchor=('if ((GFReal) ierr[(long long) i + offset_205] == 0.0f)' if label=='__label_000887' else 'if (ierr[(long long) i + offset_205] == 0)')
        start=source.rfind(anchor,0,first)
        end=source.index(target,first)
        if start<0:raise ValueError('native GF shallow skip scope missing')
        segment=source[start:end]
        if segment.count('{')!=segment.count('}'):raise ValueError('native GF shallow skip scope unbalanced')
        from tools.thompson_column_metal.generate import _loop_header
        for match in re.finditer(re.escape(jump),segment):
            stack=[]
            for token in re.finditer(r'[{}]',segment[:match.start()]):
                if token.group()=='{':stack.append(_loop_header(segment,token.start())!='')
                else:stack.pop()
            if any(stack):raise ValueError('native GF shallow skip crosses an inner loop')
        source=source[:start]+'do {\n'+segment.replace(jump,'break;')+'} while(false);'+source[end+len(target):]
    start='S_645 = 1;';jump='goto L_140;';finish='limit_644 =  gf_nan();\n                                        L_140:;'
    for token in (start,jump,finish):
        if source.count(token)!=1:raise ValueError('native GF shallow ensemble search changed')
    source=source.replace(start,start+'\n bool gf_found=false;')
    source=source.replace(jump,'gf_found=true; break;')
    source=source.replace(finish,'if (!gf_found) limit_644 = gf_nan();')
    if re.search(r'\bgoto\b',source):raise ValueError('unlowered native GF shallow jump')
    return source,count
