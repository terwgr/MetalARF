"""Rewrite the forward gotos of a gfortran-lowered CUDA function as structured
control flow (Metal has neither goto nor labels).

Every goto in these units jumps forward to a label in an enclosing statement
list. A goto out of a loop to the label right after it becomes ``break``; any
other goto sets a guard flag (and breaks out of its loop) and the statements it
skips run under ``if (!flag)``; a jump to the end of a void function returns.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mp28_metal"))
from translate import match_fwd, tokenize, untok  # noqa: E402


class Node:
    def __init__(self, kind, **kw):
        self.kind = kind
        self.__dict__.update(kw)


def parse_list(toks, i, end):
    out = []
    while i < end:
        node, i = parse_stmt(toks, i)
        out.append(node)
    return out


def parse_stmt(toks, i):
    kind, text = toks[i]
    if text == "{":
        close = match_fwd(toks, i)
        return Node("block", kids=parse_list(toks, i + 1, close)), close + 1
    if text == "if":
        close = match_fwd(toks, i + 1)
        cond = toks[i + 1:close + 1]
        then, j = parse_stmt(toks, close + 1)
        other = None
        if j < len(toks) and toks[j][1] == "else":
            other, j = parse_stmt(toks, j + 1)
        return Node("if", cond=cond, then=then, other=other), j
    if text in ("while", "for"):
        close = match_fwd(toks, i + 1)
        body, j = parse_stmt(toks, close + 1)
        return Node("while", head=toks[i:close + 1], body=body), j
    if text == "goto":
        return Node("goto", name=toks[i + 1][1]), i + 3
    if kind == "id" and i + 1 < len(toks) and toks[i + 1][1] == ":":
        return Node("label", name=text), i + 2
    depth, j = 0, i
    while True:
        t = toks[j][1]
        if t in "({[":
            depth += 1
        elif t in ")}]":
            depth -= 1
        elif t == ";" and depth == 0:
            return Node("simple", toks=toks[i:j + 1]), j + 1
        j += 1


def emit(node, out):
    k = node.kind
    if k == "block":
        out.append(("op", "{"))
        for kid in node.kids:
            emit(kid, out)
        out.append(("op", "}"))
    elif k == "if":
        out.append(("id", "if"))
        out.extend(node.cond)
        emit(node.then, out)
        if node.other is not None:
            out.append(("id", "else"))
            emit(node.other, out)
    elif k == "while":
        out.extend(node.head)
        emit(node.body, out)
    elif k == "simple":
        out.extend(node.toks)
    elif k == "label":
        pass
    else:
        raise ValueError("goto left behind: " + node.name)


def children(node):
    if node.kind == "block":
        return node.kids
    if node.kind == "if":
        return [node.then] + ([node.other] if node.other is not None else [])
    if node.kind == "while":
        return [node.body]
    return []


def find_paths(node, path, found):
    path = path + [node]
    if node.kind == "goto":
        found.append(path)
    for kid in children(node):
        find_paths(kid, path, found)


def inert(node):
    return node.kind == "label" or (node.kind == "simple" and node.toks[0][1] == ";")


def last_in_blocks(path, start):
    """The goto must end every block on its way up to the statement list."""
    for parent, child in zip(path[start:], path[start + 1:]):
        if parent.kind == "block":
            rest = parent.kids[parent.kids.index(child) + 1:]
            if not all(inert(n) for n in rest):
                raise ValueError("goto is followed by statements in its block: "
                                 + untok(rest[0].toks if rest[0].kind == "simple" else [])[:80])


def simple(text):
    return Node("simple", toks=tokenize(text))


def eliminate(body, returns_void):
    """body: the function's top-level block Node."""
    names = sorted({p[-1].name for p in _gotos(body)})
    for name in names:
        paths = [p for p in _gotos(body) if p[-1].name == name]
        flagged = []
        for path in paths:
            sl = next(n for n in reversed(path[:-1]) if n.kind == "block"
                      and any(isinstance(k, Node) and k.kind == "label" and k.name == name
                              for k in n.kids))
            where = path.index(sl)
            a = sl.kids.index(path[where + 1])
            label_at = next(i for i, k in enumerate(sl.kids) if k.kind == "label" and k.name == name)
            if label_at < a:
                raise ValueError("backward goto " + name)
            loops = [n for n in path[where + 1:-1] if n.kind == "while"]
            tail = all(inert(k) for k in sl.kids[label_at + 1:])
            goto = path[-1]
            if not loops:
                try:
                    last_in_blocks(path, where + 1)
                except ValueError as exc:
                    raise ValueError(name + ": " + str(exc) + " kinds=" + str([k.kind for k in sl.kids[a:label_at+1]]))
                if sl is body and tail and returns_void:
                    replacement = simple("return;")
                else:
                    replacement = simple("g_%s = true;" % name)
                    flagged.append((sl, path[where + 1]))
            else:
                if len(loops) != 1:
                    raise ValueError("goto leaves nested loops: " + name)
                follows = (loops[0] is sl.kids[a]
                           and all(inert(k) for k in sl.kids[a + 1:label_at]))
                if follows:
                    replacement = simple("break;")
                else:
                    if loops[0] is not sl.kids[a]:
                        raise ValueError("goto leaves a loop below its statement list: " + name)
                    replacement = Node("block", kids=[simple("g_%s = true;" % name), simple("break;")])
                    flagged.append((sl, loops[0]))
            _replace(path[-2], goto, replacement)
        if flagged:
            _guard(name, flagged)
    _drop_labels(body)
    out = []
    emit(body, out)
    return out


def _gotos(body):
    found = []
    find_paths(body, [], found)
    return found


def _replace(parent, old, new):
    if parent.kind == "block":
        parent.kids[parent.kids.index(old)] = new
    elif parent.kind == "if":
        if parent.then is old:
            parent.then = new
        else:
            parent.other = new
    else:
        parent.body = new


def _guard(name, flagged):
    sl = flagged[0][0]
    kids = sl.kids
    label_at = next(i for i, k in enumerate(kids) if k.kind == "label" and k.name == name)
    heads = sorted({kids.index(n) for _, n in flagged})
    out = kids[:heads[0]] + [simple("bool g_%s = false;" % name), kids[heads[0]]]
    for i, h in enumerate(heads):
        end = heads[i + 1] + 1 if i + 1 < len(heads) else label_at
        region = kids[h + 1:end]
        if any(not inert(r) for r in region):
            out.append(Node("if", cond=tokenize("(!g_%s)" % name),
                            then=Node("block", kids=region), other=None))
        else:
            out.extend(region)
    out.extend(kids[label_at:])
    sl.kids[:] = out


def _drop_labels(node):
    if node.kind == "block":
        node.kids[:] = [k for k in node.kids if k.kind != "label"]
    for kid in children(node):
        _drop_labels(kid)


def rewrite_function(text):
    """Rewrite one function definition's text; returns text without gotos."""
    toks = tokenize(text)
    open_ = next(i for i, t in enumerate(toks) if t[1] == "{")
    close = match_fwd(toks, open_)
    head = toks[:open_]
    void = any(t[1] == "void" for t in head[:3])
    body, _ = parse_stmt(toks, open_)
    return untok(head + eliminate(body, void))
