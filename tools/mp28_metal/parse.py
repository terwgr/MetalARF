"""Split preprocessed CUDA text into top-level definitions and find the
definitions a set of kernels reaches."""
import re

def split_top(text):
    items = []
    n = len(text)
    start = 0
    depth = 0
    paren = 0
    j = 0
    while j < n:
        c = text[j]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                k = j + 1
                while k < n and text[k] in ' \t\n':
                    k += 1
                is_def = re.search(r'(\)|\bconst|\bnoexcept)\s*$', text[start:j].rstrip().rsplit('{', 1)[0]) or text[start:j].lstrip().startswith(('namespace', 'struct', 'extern'))
                if k < n and text[k] == ';':
                    j = k
                items.append(text[start:j + 1])
                start = j + 1
        elif c == ';' and depth == 0:
            items.append(text[start:j + 1])
            start = j + 1
        j += 1
    return [i.strip() for i in items if i.strip()]

NAME_RE = re.compile(r'([A-Za-z_]\w*)\s*(?:\(|\[|=|\{|;)')

def def_name(item):
    head = item
    if item.startswith('struct'):
        return re.match(r'struct\s+(\w+)', item).group(1)
    if item.startswith('namespace'):
        return re.match(r'namespace\s+(\w+)', item).group(1)
    # function or variable: name before first '(' (function) or '[' / '='
    m = re.search(r'([A-Za-z_]\w*)\s*\(', item.split('{')[0].split('=')[0]) if '(' in item.split('{')[0].split('=')[0] else None
    if m:
        return m.group(1)
    m = re.search(r'([A-Za-z_]\w*)\s*(\[|=|;)', item)
    return m.group(1) if m else None

IDENT = re.compile(r'[A-Za-z_]\w*')

def closure(items, roots):
    names = {}
    for it in items:
        nm = def_name(it)
        if nm:
            names.setdefault(nm, []).append(it)
    keep = set()
    stack = list(roots)
    while stack:
        nm = stack.pop()
        if nm in keep or nm not in names:
            continue
        keep.add(nm)
        for it in names[nm]:
            for ident in set(IDENT.findall(it)):
                if ident != nm and ident in names and ident not in keep:
                    stack.append(ident)
    out = []
    for it in items:
        nm = def_name(it)
        if nm in keep:
            out.append(it)
    return out, keep

if __name__ == '__main__':
    import sys
    t = open(sys.argv[1]).read()
    items = split_top(t)
    out, keep = closure(items, sys.argv[2:])
    for it in out:
        print(len(it), def_name(it))
