"""Singular-weight moving least squares (MLS) for generalized Hermite-Birkhoff data.

A data set is a list of linear differential functionals
    lambda(f) = sum_t c_t * D^{beta_t} f(p),
each attached to a point p (a "node"), together with prescribed values y.
At an evaluation point x the method fits a polynomial of total degree <= m by
weighted least squares, every functional at node p getting the weight
    w(x, p) = |x - p|^(-alpha) * exp(-(|x - p| / delta)^2)          (alpha even),
and returns the value of that local polynomial at x.

Numerics:
  * the local basis is ((y - x) / h)^e, centred at the evaluation point, so u(x)
    is simply the constant coefficient (MLS is invariant under change of basis);
  * weights are normalised in log space and the weighted problem is solved by
    an SVD-based least-squares solver, never by forming normal equations;
  * exactly at a node the singular limit is evaluated in closed form, as an
    equality-constrained least-squares problem (no epsilon regularisation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# multi-indices and the polynomial basis
# --------------------------------------------------------------------------- #
def compositions(n: int, k: int) -> list[tuple[int, ...]]:
    """All k-tuples of non-negative integers summing to n (reverse lexicographic)."""
    if k == 1:
        return [(n,)]
    out = []
    for first in range(n, -1, -1):
        for rest in compositions(n - first, k - 1):
            out.append((first,) + rest)
    return out


def multi_indices(dim: int, degree: int) -> np.ndarray:
    """Exponents of all monomials of total degree <= degree, graded order."""
    out: list[tuple[int, ...]] = []
    for total in range(degree + 1):
        out.extend(compositions(total, dim))
    return np.array(out, dtype=int)


def falling(e: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Elementwise falling factorial e (e-1) ... (e-a+1); zero when a > e >= 0."""
    e = np.asarray(e)
    a = np.asarray(a)
    out = np.ones(np.broadcast(e, a).shape)
    amax = int(np.max(a)) if np.size(a) else 0
    for t in range(amax):
        out = np.where(a > t, out * (e - t), out)
    return out


# --------------------------------------------------------------------------- #
# linear differential functionals
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Functional:
    """lambda(f) = sum_t coef_t * D^{beta_t} f(point)."""

    point: tuple[float, ...]
    terms: tuple[tuple[float, tuple[int, ...]], ...]

    @property
    def dim(self) -> int:
        return len(self.point)

    @property
    def order(self) -> int:
        return max(sum(beta) for _, beta in self.terms)

    def basis_row(self, exps: np.ndarray, center: np.ndarray, h: float) -> np.ndarray:
        """Apply the functional to every basis monomial ((y - center)/h)^e."""
        z = (np.asarray(self.point, dtype=float) - center) / h
        row = np.zeros(len(exps))
        for coef, beta in self.terms:
            beta = np.asarray(beta, dtype=int)
            ff = np.prod(falling(exps, beta[None, :]), axis=1)
            pw = exps - beta[None, :]
            ok = np.all(pw >= 0, axis=1)
            mon = np.prod(np.where(pw >= 0, z[None, :] ** np.maximum(pw, 0), 0.0), axis=1)
            row += coef * np.where(ok, ff * mon, 0.0) * h ** (-int(beta.sum()))
        return row

    def apply(self, derivative) -> float:
        """Evaluate on a function given by derivative(beta, point) -> float."""
        return sum(c * derivative(beta, self.point) for c, beta in self.terms)


def _pt(p) -> tuple[float, ...]:
    return tuple(float(v) for v in np.atleast_1d(p))


def value(p) -> Functional:
    p = _pt(p)
    return Functional(p, ((1.0, (0,) * len(p)),))


def partial(p, beta: Sequence[int]) -> Functional:
    p = _pt(p)
    return Functional(p, ((1.0, tuple(int(b) for b in beta)),))


def derivative_1d(p, r: int) -> Functional:
    return partial([p], (r,))


def directional(p, v) -> Functional:
    p = _pt(p)
    d = len(p)
    terms = tuple((float(v[j]), tuple(int(i == j) for i in range(d))) for j in range(d) if v[j] != 0)
    return Functional(p, terms)


def laplacian(p) -> Functional:
    p = _pt(p)
    d = len(p)
    return Functional(p, tuple((1.0, tuple(2 * int(i == j) for i in range(d))) for j in range(d)))


def hermite_set(p, k: int) -> list[Functional]:
    """All partial derivatives D^beta, |beta| <= k, at p (a classical Hermite node)."""
    p = _pt(p)
    return [partial(p, beta) for beta in multi_indices(len(p), k)]


# --------------------------------------------------------------------------- #
# the singular-weight MLS operator
# --------------------------------------------------------------------------- #
class SingularMLS:
    """Interpolating MLS operator for generalized Hermite-Birkhoff data.

    Parameters
    ----------
    functionals : the data functionals (any order, any mix, any dimension).
    degree      : total degree m of the local polynomial space P_m.
    alpha       : even exponent of the singular weight |x - p|^(-alpha).
    delta       : decay length of the Gaussian factor, or support radius of the bump
                  (None = pure power weight).
    weight      : "gauss" -> |x-p|^(-alpha) exp(-|x-p|^2/delta^2);
                  "bump"  -> |x-p|^(-alpha) psi(|x-p|^2/delta^2) with the C-infinity bump
                  psi(s) = exp(1 - 1/(1-s)) for s < 1 and 0 for s >= 1 (compact support).
    length      : derivative rows are scaled by length**order (dimensionless data);
                  this changes accuracy, never exactness at the nodes.
    cutoff      : rows whose normalised weight is below cutoff are dropped.
    """

    def __init__(self, functionals: Iterable[Functional], degree: int, alpha: int = 2,
                 delta: float | None = None, length: float = 1.0, cutoff: float = 1e-30,
                 weight: str = "gauss"):
        self.fun = list(functionals)
        if alpha <= 0 or alpha % 2:
            raise ValueError("alpha must be a positive even integer (smoothness of the weight)")
        self.dim = self.fun[0].dim
        self.degree = degree
        self.alpha = alpha
        self.delta = delta
        if weight not in ("gauss", "bump"):
            raise ValueError("weight must be 'gauss' or 'bump'")
        if weight == "bump" and delta is None:
            raise ValueError("the bump weight needs a support radius delta")
        self.weight = weight
        self.length = length
        self.cutoff = cutoff
        self.exps = multi_indices(self.dim, degree)
        pts = [f.point for f in self.fun]
        uniq = list(dict.fromkeys(pts))
        index = {p: i for i, p in enumerate(uniq)}
        self.nodes = np.array(uniq, dtype=float)
        self.node_of_row = np.array([index[p] for p in pts])
        self.row_scale = np.array([length ** f.order for f in self.fun])
        self.h = delta if delta is not None else 1.0
        span = np.ptp(self.nodes, axis=0).max() if len(self.nodes) > 1 else 1.0
        self.node_tol = 1e-13 * max(span, 1.0)
        # term tables for vectorised assembly: (row, term) -> coef, beta
        T = max(len(f.terms) for f in self.fun)
        self._coef = np.zeros((len(self.fun), T))
        self._beta = np.zeros((len(self.fun), T, self.dim), dtype=int)
        for i, f in enumerate(self.fun):
            for t, (c, beta) in enumerate(f.terms):
                self._coef[i, t] = c
                self._beta[i, t] = beta
        self._pts = np.array(pts, dtype=float)

    # -- pieces ---------------------------------------------------------------
    @property
    def M(self) -> int:
        return len(self.exps)

    def node_log_weights(self, x: np.ndarray, d: np.ndarray) -> np.ndarray:
        lw = -self.alpha * np.log(d)
        if self.delta is not None:
            s2 = (d / self.delta) ** 2
            if self.weight == "gauss":
                lw = lw - s2
            else:
                with np.errstate(divide="ignore"):
                    lw = lw + np.where(s2 < 1, 1.0 - 1.0 / np.maximum(1.0 - s2, 1e-300), -np.inf)
        return lw

    def system(self, x: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Matrix [lambda_i(q_e)] for the given rows, basis q_e = ((y - x)/h)^e."""
        Z = (self._pts[rows] - x) / self.h                        # (R, d)
        E = self.exps[None, :, :]                                 # (1, M, d)
        out = np.zeros((len(rows), self.M))
        for t in range(self._coef.shape[1]):
            c = self._coef[rows, t]
            if not np.any(c):
                continue
            B = self._beta[rows, t][:, None, :]                   # (R, 1, d)
            pw = E - B
            ff = np.prod(falling(E, B), axis=2)
            mon = np.prod(np.where(pw >= 0, Z[:, None, :] ** np.maximum(pw, 0), 0.0), axis=2)
            out += (c * self.h ** (-B.sum(axis=2)[:, 0]))[:, None] * ff * mon
        return out

    # -- solve ----------------------------------------------------------------
    def coefficients(self, x, y) -> np.ndarray:
        """Local-polynomial coefficients at x (basis centred at x), for data y.

        y may be (N,) or (N, r); the result is (M,) or (M, r).
        """
        x = np.asarray(x, dtype=float).reshape(self.dim)
        y = np.asarray(y, dtype=float)
        ys = y * (self.row_scale if y.ndim == 1 else self.row_scale[:, None])
        d = np.linalg.norm(self.nodes - x, axis=1)
        k = int(np.argmin(d))
        if d[k] <= self.node_tol:
            return self._at_node(x, ys, k, d)
        lw = self.node_log_weights(x, d)
        lw = lw - lw.max()
        # far-field truncation, measured against the strongest node *other than* the
        # nearest one (near a node every other weight is tiny relative to it)
        ref = np.max(np.delete(lw, k)) if len(lw) > 1 else 0.0
        keep_nodes = lw > ref + np.log(self.cutoff)
        rows = np.nonzero(keep_nodes[self.node_of_row])[0]
        sw = np.exp(0.5 * lw[self.node_of_row[rows]])
        A = sw[:, None] * self.system(x, rows) * self.row_scale[rows, None]
        b = (sw * ys[rows]) if ys.ndim == 1 else (sw[:, None] * ys[rows])
        heavy = self.node_of_row[rows] == k
        s_light = sw[~heavy].max() if np.any(~heavy) else 0.0
        if s_light < 1e-3:
            return self._two_block(A, b, heavy, s_light)
        a, *_ = np.linalg.lstsq(A, b, rcond=None)
        return a

    @staticmethod
    def _two_block(A, b, heavy, s):
        """Weighted LS  min ||R a - bh||^2 + ||L a - bl||^2  when the nearest node's rows R
        dominate the light rows L (|L| ~ s << 1).  Exact reformulation, no approximation.

        Write a = Q1 c + Q2 z (Q1: row space of R, Q2: its null space) and
        c = c0 + s*d with R Q1 c0 = bh in the least-squares sense.  Dividing the objective
        by s^2 gives the well-scaled problem
            min || R Q1 d ||^2 + || (L Q1) d + (L Q2 / s) z - (bl - L Q1 c0) / s ||^2,
        whose blocks are all O(1) or smaller.  A plain solver applied to the original
        rows would discard the light data once s^2 < machine epsilon.
        """
        R = A[heavy]
        _, sv, Vt = np.linalg.svd(R)
        r = int(np.sum(sv > sv[0] * 1e-12)) if sv.size else 0
        Q1, Q2 = Vt[:r].T, Vt[r:].T
        if s == 0.0:
            s = np.finfo(float).tiny
        L, bh, bl = A[~heavy], b[heavy], b[~heavy]
        RQ1 = R @ Q1
        c0, *_ = np.linalg.lstsq(RQ1, bh, rcond=None)
        e = (bl - L @ Q1 @ c0) / s
        top = np.hstack([RQ1, np.zeros((R.shape[0], Q2.shape[1]))])
        bot = np.hstack([L @ Q1, (L @ Q2) / s])
        zero = np.zeros((R.shape[0],) + bh.shape[1:])
        sol, *_ = np.linalg.lstsq(np.vstack([top, bot]), np.concatenate([zero, e]), rcond=None)
        return Q1 @ (c0 + s * sol[:r]) + Q2 @ sol[r:]

    def _at_node(self, x, ys, k, d) -> np.ndarray:
        """Exact singular limit at node k: node-k data as equality constraints,
        remaining data fitted by least squares with their (finite) weights."""
        at = np.nonzero(self.node_of_row == k)[0]
        R = self.system(x, at) * self.row_scale[at, None]
        yR = ys[at]
        aP, *_ = np.linalg.lstsq(R, yR, rcond=None)
        U, s, Vt = np.linalg.svd(R)
        rank = int(np.sum(s > s[0] * 1e-12)) if s.size else 0
        N = Vt[rank:].T                              # null space of the node rows
        if N.shape[1] == 0:
            return aP
        dd = d.copy()
        dd[k] = np.inf
        lw = self.node_log_weights(x, np.where(np.isfinite(dd), dd, 1.0))
        lw[k] = -np.inf
        lw = lw - lw.max()
        keep_nodes = lw > np.log(self.cutoff)
        rows = np.nonzero(keep_nodes[self.node_of_row])[0]
        sw = np.exp(0.5 * lw[self.node_of_row[rows]])
        PO = self.system(x, rows) * self.row_scale[rows, None]
        rhs = ys[rows] - PO @ aP
        A = sw[:, None] * (PO @ N)
        b = (sw * rhs) if rhs.ndim == 1 else (sw[:, None] * rhs)
        z, *_ = np.linalg.lstsq(A, b, rcond=None)
        return aP + N @ z

    # -- user interface -------------------------------------------------------
    def __call__(self, X, y) -> np.ndarray:
        """Evaluate the interpolant at the rows of X (shape (n, dim))."""
        X = np.asarray(X, dtype=float).reshape(-1, self.dim)
        return np.array([self.coefficients(x, y)[0] for x in X])

    def shape_functions(self, x) -> np.ndarray:
        """phi(x) with u(x) = phi(x) . y  (one shape function per datum)."""
        return self.coefficients(x, np.eye(len(self.fun)))[0]

    def data_from(self, derivative) -> np.ndarray:
        """Exact data for a function known through derivative(beta, point)."""
        return np.array([f.apply(derivative) for f in self.fun])
