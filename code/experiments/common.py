"""Shared helpers for the exactness experiments: the predicted rule and the
high-precision check of lambda(u) at a node."""
import mpmath as mp
import numpy as np

import hbmls as hb
from hbmls.core import multi_indices

EXACT_REL = 1e-12     # |lambda(u) - y| <= EXACT_REL * max(1, |y|)  counts as exact


# --------------------------------------------------------------------------- #
# symbols
# --------------------------------------------------------------------------- #
def symbol(f):
    """Symbol of a functional: dict exponent -> coefficient."""
    q = {}
    for c, beta in f.terms:
        q[tuple(beta)] = q.get(tuple(beta), 0.0) + c
    return q


def d_dxi(q, j):
    out = {}
    for e, c in q.items():
        if e[j] > 0:
            e2 = list(e)
            e2[j] -= 1
            out[tuple(e2)] = out.get(tuple(e2), 0.0) + c * e[j]
    return {e: c for e, c in out.items() if c != 0}


def laplacian_symbol(q, dim):
    out = {}
    for j in range(dim):
        for e, c in d_dxi(d_dxi(q, j), j).items():
            out[e] = out.get(e, 0.0) + c
    return {e: c for e, c in out.items() if abs(c) > 1e-12}


def closed_under_derivatives(f, node_funs):
    """(i) every derivative of f's symbol lies in the span of the node's symbols."""
    dim = len(f.point)
    top = max(g.order for g in node_funs)
    exps = [tuple(e) for e in multi_indices(dim, top)]
    vec = lambda q: np.array([q.get(e, 0.0) for e in exps])
    S = np.array([vec(symbol(g)) for g in node_funs]).T
    frontier = [symbol(f)]
    while frontier:
        q = frontier.pop()
        for j in range(dim):
            dq = d_dxi(q, j)
            if not dq:
                continue
            v = vec(dq)
            if np.linalg.norm(v - S @ np.linalg.lstsq(S, v, rcond=None)[0]) > 1e-9 * np.linalg.norm(v):
                return False
            frontier.append(dq)
    return True


def polyharmonic(f, alpha):
    """(ii) Delta^(alpha/2) of f's symbol vanishes."""
    q = symbol(f)
    for _ in range(alpha // 2):
        q = laplacian_symbol(q, len(f.point))
        if not q:
            return True
    return not q


def predicted(f, node_funs, alpha):
    """Conjectured rule: lambda = q(D)(.)(x_i) is reproduced iff (i) and (ii)."""
    return closed_under_derivatives(f, node_funs) and polyharmonic(f, alpha)


# --------------------------------------------------------------------------- #
# high-precision check
# --------------------------------------------------------------------------- #
def lambda_u(mpop, f, y, r, cache):
    p = f.point
    val = mp.mpf(0)
    for c, beta in f.terms:
        key = (beta, r)
        if key not in cache:
            cache[key] = mpop.derivative(p, beta, y, r=r)
        val += c * cache[key]
    return val


def check_node(op, y, node_index, radii=(1e-6, 1e-4)):
    """lambda(u) - y for every datum at one node, at two stencil radii.

    Returns tuples (functional, error at radii[0], error at radii[1], verdict) where
    verdict is 'exact', 'not exact', or 'UNSTABLE' if the two radii disagree."""
    mpop = hb.MPSingularMLS(op)
    cache = {}
    out = []
    for i in np.nonzero(op.node_of_row == node_index)[0]:
        f = op.fun[i]
        errs = []
        for r in radii:
            with mp.workdps(200):
                errs.append(float(abs(lambda_u(mpop, f, y, r, cache) - mp.mpf(y[i]))))
        tol = EXACT_REL * max(1.0, abs(y[i]))
        flags = [e <= tol for e in errs]
        verdict = "exact" if all(flags) else ("not exact" if not any(flags) else "UNSTABLE")
        out.append((f, errs[0], errs[1], verdict))
    return out


def describe(f):
    names = []
    for c, beta in f.terms:
        s = "f" if sum(beta) == 0 else "D" + "".join("xyz"[j] * b for j, b in enumerate(beta))
        names.append(s if c == 1 else f"{c:+g}{s}")
    return " ".join(names).replace("+1D", "+D").replace("-1D", "-D")
