"""SWIMLS2D with exact third derivatives of the approximant (Gaussian singular weight).

Same construction as `swimls2d.SWIMLS2D` (whose docstring derives b, b_k, b_kl).  One more
differentiation of  G b = A^T W y  (basis fixed at the evaluation point x0) gives, with
r = y - A b and W_k = W g_k, W_kl = W h_kl, W_klm = W h_klm,
    G b_klm = A^T W_klm r
              - A^T W_kl A b_m - A^T W_km A b_l - A^T W_lm A b_k
              - A^T W_k A b_lm - A^T W_l A b_km - A^T W_m A b_kl,
    h_kl  = g_k g_l + L_kl,
    h_klm = g_k g_l g_m + g_k L_lm + g_l L_km + g_m L_kl + L_klm,
where g_k = d_k log w and L_.. are the higher derivatives of log w.  In the Q1/Q2 form
(W^1/2 A = Q1 R, W^1/2 r = Q2 Q2^T W^1/2 y) this is
    B_klm = R^-1 [ (Q2^T diag(h_klm) Q1)^T (Q2^T W^1/2)
                   - sum_{3 pairs} Q1^T diag(h_kl) Q1 R B_m
                   - sum_{3}       Q1^T diag(g_k)  Q1 R B_lm ],
again without subtracting nearly equal terms.  The derivatives of the approximant
u(x) = sum_e b_e(x) ((x - x0)/delta)^e at x0 are, for a multi-index beta,
    d^beta u = sum_{gamma <= beta} binom(beta, gamma) gamma! delta^-|gamma| (B_{beta-gamma})_gamma,
which reproduces the gradient and Hessian formulas of swimls2d.

For the Gaussian weight  log w = -(alpha/2) log r2 - r2/delta^2,  t = x - x_j:
    L_k   = -alpha t_k / r2 - 2 t_k / delta^2
    L_kl  = -alpha (d_kl / r2 - 2 t_k t_l / r2^2) - 2 d_kl / delta^2
    L_klm = -alpha (-2 (d_kl t_m + d_km t_l + d_lm t_k) / r2^2 + 8 t_k t_l t_m / r2^3).
The compact bump weight is not implemented here (raises).
"""
from __future__ import annotations

from itertools import combinations_with_replacement as cwr
from math import comb, factorial

import numpy as np
from scipy.linalg import lapack, solve_triangular

from .swimls2d import SWIMLS2D

THIRD = ((0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1))     # xxx, xxy, xyy, yyy
SECOND = ((0, 0), (0, 1), (1, 1))                        # xx, xy, yy


def _mi(idx):
    """multi-index (a, b) of a tuple of coordinate indices"""
    return (sum(1 for i in idx if i == 0), sum(1 for i in idx if i == 1))


class SWIMLS2D3(SWIMLS2D):
    """SWIMLS2D plus `phi3(x)`: value, gradient, Hessian and third derivatives of phi."""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if self.weight != "gauss":
            raise NotImplementedError("third derivatives: Gaussian weight only")
        if self.degree < 3:
            raise ValueError("degree >= 3 is needed for third derivatives")

    def _gauss_width(self, x):
        return self.delta

    def _gauss_part(self, x, tk, r2k):
        """first, second, third derivatives (in x) of r2 / delta^2 for a constant width."""
        d2 = self.delta ** 2
        P1 = {k: 2 * tk[:, k] / d2 for k in range(2)}
        P2 = {(k, l): np.full(len(r2k), 2.0 * (k == l) / d2) for k, l in SECOND}
        P3 = {klm: np.zeros(len(r2k)) for klm in THIRD}
        return P1, P2, P3

    def phi3(self, x):
        """(phi (R,), grad (2, R), hess (3, R) [xx, xy, yy], third (4, R) [xxx, xxy, xyy, yyy])."""
        x = np.asarray(x, dtype=float)
        t = x[None, :] - self.nodes
        r2 = np.sum(t * t, axis=1)
        if np.any(r2 < (1e-14 * max(1.0, np.ptp(self.nodes))) ** 2):
            raise ValueError("evaluation at a node: use the nodal data directly")
        a, d = self.alpha, self.delta
        dw = self._gauss_width(x)                    # Gaussian width (= delta unless overridden)
        lw = -0.5 * a * np.log(r2) - r2 / dw ** 2
        lw -= lw.max()
        w = np.exp(lw)
        lev = (1.0 + np.sqrt(r2) / dw) ** (2 * self.degree)
        keep = np.nonzero((w * lev > self.cutoff)[self.row_node])[0]
        nd, sy = self.row_node[keep], self.row_sym[keep]
        tk, r2k, W = t[nd], r2[nd], w[nd]

        g, L2, L3 = {}, {}, {}
        for k in range(2):
            g[k] = -a * tk[:, k] / r2k
        for k, l in SECOND:
            dkl = float(k == l)
            L2[(k, l)] = -a * (dkl / r2k - 2 * tk[:, k] * tk[:, l] / r2k ** 2)
        for k, l, m in THIRD:
            dkl, dkm, dlm = float(k == l), float(k == m), float(l == m)
            L3[(k, l, m)] = -a * (-2 * (dkl * tk[:, m] + dkm * tk[:, l] + dlm * tk[:, k]) / r2k ** 2
                                  + 8 * tk[:, k] * tk[:, l] * tk[:, m] / r2k ** 3)
        P1, P2, P3 = self._gauss_part(x, tk, r2k)   # derivatives of r2 / width(x)^2
        for k in range(2):
            g[k] = g[k] - P1[k]
        for kl in SECOND:
            L2[kl] = L2[kl] - P2[kl]
        for klm in THIRD:
            L3[klm] = L3[klm] - P3[klm]
        g = np.stack([g[0], g[1]], axis=1)
        L2f = lambda k, l: L2[tuple(sorted((k, l)))]
        h2 = {kl: g[:, kl[0]] * g[:, kl[1]] + L2[kl] for kl in SECOND}
        h3 = {}
        for k, l, m in THIRD:
            h3[(k, l, m)] = (g[:, k] * g[:, l] * g[:, m] + g[:, k] * L2f(l, m) + g[:, l] * L2f(k, m)
                             + g[:, m] * L2f(k, l) + L3[(k, l, m)])

        A = self._rows(-tk / d, sy)
        S = 1.0 / np.maximum(np.sqrt(W @ A ** 2), 1e-300)
        A = A * S[None, :]
        sw = np.sqrt(W)
        Rk, M = A.shape
        if Rk < M:
            raise np.linalg.LinAlgError("fewer local rows than basis functions")
        qr, tau, _, info = lapack.dgeqrf(sw[:, None] * A)
        if info:
            raise np.linalg.LinAlgError("dgeqrf failed")
        Rf = np.triu(qr[:M, :M])
        if np.min(np.abs(np.diag(Rf))) < 1e-13 * np.max(np.abs(np.diag(Rf))):
            raise np.linalg.LinAlgError("local least-squares problem is rank deficient")
        Q1, _, info = lapack.dorgqr(qr[:, :M].copy(), tau)

        def QT(X):
            out, _, info = lapack.dormqr("L", "T", qr, tau, np.asfortranarray(X),
                                         64 * max(1, X.shape[1]))
            if info:
                raise np.linalg.LinAlgError("dormqr failed")
            return out

        Q2sw = QT(np.diag(sw))[M:]
        first = lambda hv: QT(hv[:, None] * Q1)[M:].T @ Q2sw            # (Q2^T diag(h) Q1)^T Q2^T sw
        QgQ = {k: Q1.T @ (g[:, k][:, None] * Q1) for k in range(2)}
        QhQ = {kl: Q1.T @ (h2[kl][:, None] * Q1) for kl in SECOND}
        solve = lambda Y: solve_triangular(Rf, Y)
        B = {(): solve(Q1.T * sw[None, :])}
        for k in range(2):
            B[(k,)] = solve(first(g[:, k]))
        for k, l in SECOND:
            B[(k, l)] = solve(first(h2[(k, l)]) - QgQ[k] @ (Rf @ B[(l,)]) - QgQ[l] @ (Rf @ B[(k,)]))
        Bs = lambda idx: B[tuple(sorted(idx))]
        hs = lambda kl: QhQ[tuple(sorted(kl))]
        for k, l, m in THIRD:
            rhs = first(h3[(k, l, m)])
            rhs -= hs((k, l)) @ (Rf @ B[(m,)]) + hs((k, m)) @ (Rf @ B[(l,)]) + hs((l, m)) @ (Rf @ B[(k,)])
            rhs -= (QgQ[k] @ (Rf @ Bs((l, m))) + QgQ[l] @ (Rf @ Bs((k, m)))
                    + QgQ[m] @ (Rf @ Bs((k, l))))
            B[(k, l, m)] = solve(rhs)
        B = {key: S[:, None] * v for key, v in B.items()}

        def deriv(beta):
            """d^beta u (beta a multi-index) as a row over the kept data."""
            out = 0.0
            for g0 in range(beta[0] + 1):
                for g1 in range(beta[1] + 1):
                    gam = (g0, g1)
                    rest = (0,) * (beta[0] - g0) + (1,) * (beta[1] - g1)
                    c = comb(beta[0], g0) * comb(beta[1], g1) * factorial(g0) * factorial(g1)
                    out = out + c / d ** (g0 + g1) * B[rest][self.idx[gam]]
            return out

        sc = self.row_scale[keep]

        def full(v):
            o = np.zeros(self.R)
            o[keep] = v * sc
            return o

        u = full(deriv((0, 0)))
        grad = np.array([full(deriv(b)) for b in ((1, 0), (0, 1))])
        hess = np.array([full(deriv(_mi(kl))) for kl in SECOND])
        third = np.array([full(deriv(_mi(klm))) for klm in THIRD])
        return u, grad, hess, third


class SWIMLS2D3V(SWIMLS2D3):
    """SWIMLS2D3 with a position-dependent Gaussian width delta(x) and EXACT derivatives of the
    resulting approximant (the d delta / dx terms included).  The polynomial basis and the row
    scaling keep the constant `delta` given to the constructor (a change of basis and of row units,
    which does not change the local fit), so only the weight depends on delta(x):
        log w = -(alpha/2) log r2 - r2 u(x),   u = delta(x)^-2,
        d(r2 u) = r2_k u + r2 u_k, ...  (r2_k = 2 t_k, r2_kl = 2 d_kl, r2_klm = 0).
    width_fn(x) -> (d, d1 (2,), d2 (3,) [xx, xy, yy], d3 (4,) [xxx, xxy, xyy, yyy])."""

    def __init__(self, *args, width_fn=None, **kw):
        super().__init__(*args, **kw)
        if width_fn is None:
            raise ValueError("width_fn is required")
        self.width_fn = width_fn
        self._cache = None

    def _w(self, x):
        key = (float(x[0]), float(x[1]))
        if self._cache is None or self._cache[0] != key:
            self._cache = (key, self.width_fn(np.asarray(x, float)))
        return self._cache[1]

    def _gauss_width(self, x):
        return self._w(x)[0]

    def _gauss_part(self, x, tk, r2k):
        dd, d1, d2v, d3v = self._w(x)
        D2 = {(0, 0): d2v[0], (0, 1): d2v[1], (1, 1): d2v[2]}
        D3 = dict(zip(THIRD, d3v))
        dl = lambda *ix: D2[tuple(sorted(ix))] if len(ix) == 2 else D3[tuple(sorted(ix))]
        u = dd ** -2.0
        uk = {k: -2 * dd ** -3 * d1[k] for k in range(2)}
        ukl = {(k, l): 6 * dd ** -4 * d1[k] * d1[l] - 2 * dd ** -3 * dl(k, l) for k, l in SECOND}
        uf = lambda k, l: ukl[tuple(sorted((k, l)))]
        uklm = {}
        for k, l, m in THIRD:
            uklm[(k, l, m)] = (-24 * dd ** -5 * d1[k] * d1[l] * d1[m]
                               + 6 * dd ** -4 * (dl(k, l) * d1[m] + dl(k, m) * d1[l] + dl(l, m) * d1[k])
                               - 2 * dd ** -3 * dl(k, l, m))
        r1 = {k: 2 * tk[:, k] for k in range(2)}
        r2d = lambda k, l: 2.0 * (k == l)
        P1 = {k: r1[k] * u + r2k * uk[k] for k in range(2)}
        P2 = {(k, l): r2d(k, l) * u + r1[k] * uk[l] + r1[l] * uk[k] + r2k * ukl[(k, l)]
              for k, l in SECOND}
        P3 = {}
        for k, l, m in THIRD:
            P3[(k, l, m)] = (r2d(k, l) * uk[m] + r2d(k, m) * uk[l] + r2d(l, m) * uk[k]
                             + r1[k] * uf(l, m) + r1[l] * uf(k, m) + r1[m] * uf(k, l)
                             + r2k * uklm[(k, l, m)])
        return P1, P2, P3

    def phi(self, x):
        u, g, H, _ = self.phi3(x)
        return u, g, H
