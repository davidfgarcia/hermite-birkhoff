"""Shared helpers for the exactness experiments."""
import numpy as np
import mpmath as mp

import hbmls as hb
from hbmls.core import multi_indices

EXACT = 1e-9          # |lambda(u) - y| below this counts as reproduced exactly


def symbol(f):
    """Symbol of a functional at its point: dict exponent -> coefficient."""
    q = {}
    for c, beta in f.terms:
        q[tuple(beta)] = q.get(tuple(beta), 0.0) + c
    return q


def not_d_invariant(funs):
    """Functionals at one node whose symbol has a first partial derivative outside the
    span of the node's symbols.  Empty list <=> the symbol space is D-invariant
    (closed under differentiation), i.e. the node data are of Hermite type."""
    dim = len(funs[0].point)
    top = max(f.order for f in funs)
    exps = [tuple(e) for e in multi_indices(dim, top)]
    vec = lambda q: np.array([q.get(e, 0.0) for e in exps])
    S = np.array([vec(symbol(f)) for f in funs]).T
    bad = []
    for f in funs:
        for j in range(dim):
            dq = {}
            for e, c in symbol(f).items():
                if e[j] > 0:
                    e2 = list(e)
                    e2[j] -= 1
                    dq[tuple(e2)] = dq.get(tuple(e2), 0.0) + c * e[j]
            if not dq:
                continue
            v = vec(dq)
            res = v - S @ np.linalg.lstsq(S, v, rcond=None)[0]
            if np.linalg.norm(res) > 1e-9 * np.linalg.norm(v):
                bad.append(f)
                break
    return bad


def check_node(op, y, node_index, dps=60, r=1e-6, extra=5):
    """lambda(u) - y for every datum at one node (high-precision stencils)."""
    mpop = hb.MPSingularMLS(op, dps=dps)
    p = op.nodes[node_index]
    cache = {}
    out = []
    for i in np.nonzero(op.node_of_row == node_index)[0]:
        f = op.fun[i]
        val = mp.mpf(0)
        for c, beta in f.terms:
            if beta not in cache:
                cache[beta] = mpop.derivative(p, beta, y, r=r, extra=extra)
            val += c * cache[beta]
        out.append((f, float(val), float(y[i]), abs(float(val) - y[i])))
    return out


def describe(f):
    names = []
    for c, beta in f.terms:
        s = "f" if sum(beta) == 0 else "D" + "".join("xyz"[j] * b for j, b in enumerate(beta))
        names.append(s if c == 1 else f"{c:g}{s}")
    return " + ".join(names)
