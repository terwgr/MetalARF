"""CUDA -> Metal statement translator for the mp=28 source networks.

The statements stay untouched.  What changes: address spaces (pointer
parameters become template parameters), binary64 (``Soft8`` on the integer
soft64 helpers), the float libm and rounding intrinsics (see the prelude),
float division (``aa_div`` on the soft32 divider) and the CUDA keywords.
"""
import re
import struct

TOKEN = re.compile(r'''
    (?P<ws>\s+)
  | (?P<str>"[^"\n]*")
  | (?P<num>0[xX][0-9a-fA-F]*\.?[0-9a-fA-F]*(?:[pP][+-]?\d+)?[uUlLfF]*
          |(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?[uUlLfF]*)
  | (?P<id>[A-Za-z_]\w*)
  | (?P<op>->|\+\+|--|<<=|>>=|<<|>>|<=|>=|==|!=|&&|\|\||\+=|-=|\*=|/=|%=|&=|\|=|\^=|::|.)
''', re.X | re.S)


def tokenize(text):
    out = []
    for m in TOKEN.finditer(text):
        if m.lastgroup == 'ws':
            continue
        out.append((m.lastgroup, m.group()))
    return out


def untok(tokens):
    out = []
    prev = None
    for kind, text in tokens:
        if prev is not None and (
                (prev[0] in ('id', 'num') and kind in ('id', 'num'))
                or (prev[1] in ('+', '-') and text[:1] == prev[1])
                or (prev[1] == '/' and text[:1] in '/*')
                or (prev[1] == '&' and text[:1] == '&')):
            out.append(' ')
        elif prev is not None:
            out.append(' ')
        out.append(text)
        if text in (';', '{', '}'):
            out.append('\n')
            prev = None
            continue
        prev = (kind, text)
    return ''.join(out)


def double_literal(text):
    t = text.rstrip('lL')
    if t.lower().startswith('0x'):
        value = float.fromhex(t)
    else:
        value = float(t)
    bits = struct.unpack('<Q', struct.pack('<d', value))[0]
    return 'SD8(0x%016xul)' % bits


def convert_number(text):
    low = text.lower()
    is_hex = low.startswith('0x')
    if is_hex:
        floaty = ('p' in low) or ('.' in low)
    else:
        floaty = ('.' in low) or ('e' in low)
    if floaty:
        if low.endswith('f'):
            return text[:-1] + 'f' if not is_hex else 'as_type<float>(%du)' % struct.unpack(
                '<I', struct.pack('<f', float.fromhex(text[:-1])))[0]
        return double_literal(text)
    if low.endswith('f') and not is_hex:
        return text  # e.g. 1f is not valid C; leave to the compiler
    suffix = ''
    body = text
    m = re.match(r'(.*?)([uUlL]*)$', text)
    body, suffix = m.group(1), m.group(2).lower()
    u = 'u' in suffix
    ll = suffix.count('l')
    if ll:
        return body + ('ul' if u else 'l')
    return body + ('u' if u else '')


TYPE_WORDS = {'float', 'int', 'uint', 'ulong', 'long', 'bool', 'unsigned',
              'Soft8', 'const', 'char', 'short', 'half', 'size_t',
              'uint32_t', 'uint64_t', 'int32_t', 'int64_t'}

FN_RENAME = {
    'fmaxf': 'aa_fmaxf', 'fminf': 'aa_fminf', 'sqrtf': 'aa_sqrtf',
    'floorf': 'floor', 'ceilf': 'ceil', 'roundf': 'round',
    'fabsf': 'fabs', 'copysignf': 'copysign', 'truncf': 'trunc',
    'lroundf': 'aa_lroundf', 'lround': 'aa_lround',
}

DROP = {'__device__', '__forceinline__', '__restrict__', '__global__',
        '__noinline__', '__host__'}


def map_types(tokens):
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        kind, text = tokens[i]
        if text in DROP:
            i += 1
            continue
        if text == 'extern' and i + 1 < n and tokens[i + 1][0] == 'str':
            i += 2
            continue
        if text == 'unsigned':
            # unsigned long long [int] / unsigned long / unsigned int / unsigned
            j = i + 1
            words = []
            while j < n and tokens[j][1] in ('long', 'int', 'short', 'char'):
                words.append(tokens[j][1])
                j += 1
            if words.count('long') >= 1:
                out.append(('id', 'ulong'))
            elif words and words[0] == 'char':
                out.append(('id', 'uchar'))
            elif words and words[0] == 'short':
                out.append(('id', 'ushort'))
            else:
                out.append(('id', 'uint'))
            i = j
            continue
        if text == 'long':
            j = i + 1
            while j < n and tokens[j][1] in ('long', 'int'):
                j += 1
            out.append(('id', 'long'))
            i = j
            continue
        if text == 'double':
            out.append(('id', 'Soft8'))
        elif text == 'size_t':
            out.append(('id', 'ulong'))
        elif text == 'uint64_t':
            out.append(('id', 'ulong'))
        elif text == 'int64_t':
            out.append(('id', 'long'))
        elif text == 'uint32_t':
            out.append(('id', 'uint'))
        elif text == 'int32_t':
            out.append(('id', 'int'))
        elif kind == 'num':
            out.append(('num', convert_number(text)))
        elif kind == 'id' and text in FN_RENAME and i + 1 < n and tokens[i + 1][1] == '(':
            out.append(('id', FN_RENAME[text]))
        else:
            out.append((kind, text))
        i += 1
    return out


# ---------------------------------------------------------------- division
def match_back(tokens, i):
    """Index of the opener matching the closer at tokens[i]."""
    close = tokens[i][1]
    opener = {')': '(', ']': '[', '}': '{'}[close]
    depth = 0
    while i >= 0:
        t = tokens[i][1]
        if t == close:
            depth += 1
        elif t == opener:
            depth -= 1
            if depth == 0:
                return i
        i -= 1
    raise ValueError('unbalanced')


def match_fwd(tokens, i):
    opener = tokens[i][1]
    close = {'(': ')', '[': ']', '{': '}'}[opener]
    depth = 0
    n = len(tokens)
    while i < n:
        t = tokens[i][1]
        if t == opener:
            depth += 1
        elif t == close:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError('unbalanced')


NONOPERAND_PREV = set('+-*/%=<>!&|^~?:,;({[') | {
    '==', '!=', '<=', '>=', '&&', '||', '<<', '>>', '+=', '-=', '*=', '/=',
    'return', 'case'}


def is_cast_group(tokens, open_i, close_i):
    inner = [t for _, t in tokens[open_i + 1:close_i]]
    if not inner:
        return False
    ok = all(t in TYPE_WORDS or t in ('thread', 'device', 'constant', '*')
             for t in inner)
    if not ok:
        return False
    if open_i > 0:
        p = tokens[open_i - 1]
        if p[0] == 'id' and p[1] not in ('return', 'case'):
            return False
        if p[1] in (')', ']'):
            return False
    return True


def left_extent(tokens, i):
    """Start index of the multiplicative-chain operand left of tokens[i]."""
    j = i - 1
    while True:
        # one factor, backwards: postfix chain then prefix operators
        if j < 0:
            return j + 1
        t = tokens[j]
        # postfix chain
        while True:
            if tokens[j][1] in (')', ']'):
                j = match_back(tokens, j) - 1
                if tokens[j + 1][1] == '(' and j >= 0 and tokens[j][0] == 'id' \
                        and tokens[j][1] not in ('return', 'case', 'if', 'while', 'for'):
                    j -= 1
                    # qualified / member chain
                    while j >= 1 and tokens[j][1] in ('.', '->', '::') and tokens[j - 1][0] == 'id':
                        j -= 2
                    break
                if tokens[j + 1][1] == '[':
                    continue
                # parenthesised expression or cast group as primary
                break
            elif tokens[j][0] in ('id', 'num'):
                j -= 1
                while j >= 1 and tokens[j][1] in ('.', '->', '::') and tokens[j - 1][0] in ('id',):
                    j -= 2
                break
            else:
                raise ValueError('operand scan failed at %r' % (tokens[max(j - 3, 0):j + 3],))
        # array subscripts after identifier handled by loop above: continue if '[' group preceded
        # prefix operators and casts
        while j >= 0:
            t = tokens[j]
            if t[1] in ('-', '+', '!', '~', '*', '&') and (j == 0 or tokens[j - 1][1] in NONOPERAND_PREV):
                j -= 1
                continue
            if t[1] == ')':
                o = match_back(tokens, j)
                if is_cast_group(tokens, o, j):
                    j = o - 1
                    continue
            break
        if j >= 0 and tokens[j][1] in ('*', '%'):
            j -= 1
            continue
        return j + 1


def right_extent(tokens, i):
    """End index (inclusive) of the operand right of tokens[i]."""
    j = i + 1
    n = len(tokens)
    while j < n and (tokens[j][1] in ('-', '+', '!', '~', '*', '&')):
        j += 1
    while j < n and tokens[j][1] == '(':
        c = match_fwd(tokens, j)
        if is_cast_group(tokens, j, c):
            j = c + 1
            while j < n and tokens[j][1] in ('-', '+', '!', '~', '*', '&'):
                j += 1
            continue
        break
    if j >= n:
        raise ValueError('no right operand')
    if tokens[j][1] == '(':
        end = match_fwd(tokens, j)
    else:
        end = j
    while end + 1 < n:
        nxt = tokens[end + 1][1]
        if nxt in ('(', '['):
            end = match_fwd(tokens, end + 1)
        elif nxt in ('.', '->', '::') and end + 2 < n:
            end += 2
        else:
            break
    return end


def rewrite_division(tokens):
    toks = list(tokens)
    guard = 0
    while True:
        idx = next((k for k, t in enumerate(toks) if t == ('op', '/')), None)
        if idx is None:
            return toks
        a = left_extent(toks, idx)
        b = right_extent(toks, idx)
        new = [('id', 'aa_div'), ('op', '(')] + toks[a:idx] + [('op', ',')] \
            + toks[idx + 1:b + 1] + [('op', ')')]
        toks[a:b + 1] = new
        guard += 1
        if guard > 5000:
            raise ValueError('division rewrite did not terminate')


# --------------------------------------------------------------- functions
def split_params(tokens, open_i, close_i):
    params = []
    cur = []
    depth = 0
    for t in tokens[open_i + 1:close_i]:
        if t[1] in ('(', '[', '{'):
            depth += 1
        elif t[1] in (')', ']', '}'):
            depth -= 1
        if t[1] == ',' and depth == 0:
            params.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur:
        params.append(cur)
    return params


def template_pointer_params(tokens):
    """Rewrite ``T* x`` function parameters as template parameters."""
    # find function header: first '(' after name at depth 0, before '{'
    first_brace = next(k for k, t in enumerate(tokens) if t[1] == '{')
    paren = next(k for k, t in enumerate(tokens[:first_brace]) if t[1] == '(')
    close = match_fwd(tokens, paren)
    params = split_params(tokens, paren, close)
    tmpl = []
    new_params = []
    for p in params:
        if any(t[1] == '*' for t in p):
            name = p[-1][1]
            pn = 'P_' + name
            tmpl.append(pn)
            new_params.append([('id', pn), ('id', name)])
        else:
            if any(t[1] == "&" for t in p):
                p = [("id", "thread")] + p
            new_params.append(p)
    head = tokens[:paren + 1]
    body = tokens[close:]
    mid = []
    for k, p in enumerate(new_params):
        if k:
            mid.append(('op', ','))
        mid.extend(p)
    out = head + mid + body
    prefix = []
    if tmpl:
        prefix = tokenize('template <' + ', '.join('typename ' + t for t in tmpl) + '>')
    return prefix + out


def convert_function(text, kernel=False):
    toks = tokenize(text)
    toks = map_types(toks)
    toks = template_pointer_params(toks)
    toks = rewrite_division(toks)
    # Local coefficient arrays live in thread storage. Function pointer
    # parameters have already become address-space templates above.
    text = re.sub(r"\bconst Soft8 \* (\w+) =",
                  r"thread const Soft8 * \1 =", untok(toks))
    return tokenize(text)


def convert_struct(text):
    toks = map_types(tokenize(text))
    # header up to '{', members, closing
    ob = next(k for k, t in enumerate(toks) if t[1] == '{')
    out = list(toks[:ob + 1])
    stmt = []
    for t in toks[ob + 1:]:
        stmt.append(t)
        if t[1] in (';', '}'):
            declarator = stmt[:next((i for i, x in enumerate(stmt)
                                     if x[1] == '['), len(stmt))]
            if any(x[1] == '*' for x in declarator):
                stmt = [('id', 'device')] + stmt
            out.extend(stmt)
            stmt = []
    return untok(out)


def convert_table(text):
    """Global constant tables (__constant__ / __device__ const)."""
    toks = [t for t in tokenize(text) if t[1] not in ('__constant__', '__device__', 'const', 'static')]
    toks = map_types(toks)
    return 'constant ' + untok(toks)


def convert_function_item(text):
    toks = convert_function(text)
    if toks and toks[0][1] == 'template':
        close = next(k for k, t in enumerate(toks) if t[1] == '>')
        return untok(toks[:close + 1]) + ' static inline ' + untok(toks[close + 1:])
    return 'static inline ' + untok(toks)
