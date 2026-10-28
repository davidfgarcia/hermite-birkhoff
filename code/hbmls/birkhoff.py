"""Completion of Birkhoff (gapped) data to Hermite type.

SWIMLS reproduces the data at a node exactly when the span of the node's symbols is
D-invariant (closed under differentiation).  For a node whose data are not, we add the
missing functionals - a basis of  closure(span) / span  - with values estimated by a
smooth (non-singular) moving least-squares fit at that node.  The completed node is
D-invariant, so every *given* datum is then reproduced exactly; the estimated
values only influence accuracy, never exactness.
"""
from __future__ import annotations

import numpy as np

from .core import Functional, SingularMLS, multi_indices


def _symbol_vec(f: Functional, exps):
    idx = {tuple(e): i for i, e in enumerate(exps)}
    v = np.zeros(len(exps))
    for c, beta in f.terms:
        v[idx[tuple(beta)]] += c
    return v


def _differentiate(v, exps, j):
    """Coefficient vector of d/d xi_j of the symbol with coefficients v."""
    idx = {tuple(e): i for i, e in enumerate(exps)}
    out = np.zeros_like(v)
    for i, e in enumerate(exps):
        if v[i] != 0 and e[j] > 0:
            e2 = list(e)
            e2[j] -= 1
            out[idx[tuple(e2)]] += v[i] * e[j]
    return out


def _orth(M, tol=1e-10):
    if M.size == 0:
        return M.reshape(M.shape[0], 0)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, s > tol * max(s[0], 1.0)]


def missing_functionals(node_funs: list[Functional]) -> list[Functional]:
    """Functionals spanning closure(span)/span for the data at one node."""
    p = node_funs[0].point
    dim = len(p)
    top = max(f.order for f in node_funs)
    exps = [tuple(e) for e in multi_indices(dim, top)]
    span = _orth(np.array([_symbol_vec(f, exps) for f in node_funs]).T)
    closure = span.copy()
    frontier = [closure[:, i] for i in range(closure.shape[1])]
    while frontier:
        v = frontier.pop()
        for j in range(dim):
            dv = _differentiate(v, exps, j)
            if np.linalg.norm(dv) == 0:
                continue
            res = dv - closure @ (closure.T @ dv)
            if np.linalg.norm(res) > 1e-10 * np.linalg.norm(dv):
                closure = _orth(np.hstack([closure, res[:, None]]))
                frontier.append(res)
    # complement of span inside the closure
    proj = closure - span @ (span.T @ closure)
    extra = _orth(proj)
    out = []
    for k in range(extra.shape[1]):
        v = extra[:, k]
        v = v / np.max(np.abs(v))
        terms = tuple((float(v[i]), exps[i]) for i in range(len(exps)) if abs(v[i]) > 1e-12)
        out.append(Functional(p, terms))
    return out


def complete(functionals, values, degree: int, delta: float, length: float = 1.0):
    """Return (functionals, values) with every node completed to D-invariant data.

    Missing values are estimated by a smooth Gaussian-weighted MLS fit of the given
    data (degree `degree`, scale `delta`) evaluated at the node.
    """
    functionals = list(functionals)
    values = np.asarray(values, dtype=float)
    pts = list(dict.fromkeys(f.point for f in functionals))
    add_f, add_v = [], []
    for p in pts:
        node_funs = [f for f in functionals if f.point == p]
        miss = missing_functionals(node_funs)
        if not miss:
            continue
        a = _smooth_fit(functionals, values, np.array(p), degree, delta, length)
        exps = multi_indices(len(p), degree)
        for g in miss:
            row = g.basis_row(exps, np.array(p, dtype=float), delta)
            add_f.append(g)
            add_v.append(float(row @ a))
    return functionals + add_f, np.concatenate([values, add_v])


def _smooth_fit(functionals, values, x, degree, delta, length):
    """Coefficients (basis ((y-x)/delta)^e) of a Gaussian-weighted LS fit at x."""
    op = SingularMLS(functionals, degree=degree, alpha=2, delta=delta, length=length)
    rows = np.arange(len(functionals))
    d = np.linalg.norm(op._pts - x, axis=1)
    sw = np.exp(-0.5 * (d / delta) ** 2)
    keep = sw > 1e-8
    A = (sw[keep, None] * op.system(x, rows[keep])) * op.row_scale[keep, None]
    b = sw[keep] * values[keep] * op.row_scale[keep]
    a, *_ = np.linalg.lstsq(A, b, rcond=None)
    return a
