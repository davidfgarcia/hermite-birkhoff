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

    def phi3(self, x):
        """(phi (R,), grad (2, R), hess (3, R) [xx, xy, yy], third (4, R) [xxx, xxy, xyy, yyy])."""
        x = np.asarray(x, dtype=float)
        t = x[None, :] - self.nodes
        r2 = np.sum(t * t, axis=1)
        if np.any(r2 < (1e-14 * max(1.0, np.ptp(self.nodes))) ** 2):
            raise ValueError("evaluation at a node: use the nodal data directly")
        a, d = self.alpha, self.delta
        lw = -0.5 * a * np.log(r2) - r2 / d ** 2
        lw -= lw.max()
        w = np.exp(lw)
        lev = (1.0 + np.sqrt(r2) / d) ** (2 * self.degree)
        keep = np.nonzero((w * lev > self.cutoff)[self.row_node])[0]
        nd, sy = self.row_node[keep], self.row_sym[keep]
        tk, r2k, W = t[nd], r2[nd], w[nd]

        g = -a * tk / r2k[:, None] - 2 * tk / d ** 2                     # L_k
        L2, L3 = {}, {}
        for k, l in SECOND:
            dkl = float(k == l)
            L2[(k, l)] = (-a * (dkl / r2k - 2 * tk[:, k] * tk[:, l] / r2k ** 2) - 2 * dkl / d ** 2)
        for k, l, m in THIRD:
            dkl, dkm, dlm = float(k == l), float(k == m), float(l == m)
            L3[(k, l, m)] = -a * (-2 * (dkl * tk[:, m] + dkm * tk[:, l] + dlm * tk[:, k]) / r2k ** 2
                                  + 8 * tk[:, k] * tk[:, l] * tk[:, m] / r2k ** 3)
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
