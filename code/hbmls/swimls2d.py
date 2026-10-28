"""Two-dimensional SWIMLS with exact gradient and Hessian of the approximant.

Data: every node carries the same set of functionals, chosen from
    "u"   value          f(x_i)
    "dx"  derivative     f_x(x_i)
    "dy"  derivative     f_y(x_i)
    "lap" Laplacian      (f_xx + f_yy)(x_i)
At an evaluation point x the method fits a polynomial of total degree <= m by weighted
least squares, datum at x_j weighted by
    w(x, x_j) = |x - x_j|^(-alpha) exp(-|x - x_j|^2 / delta^2),      alpha even,
(weight="gauss", the default) or, with weight="bump", the compactly supported C-infinity
weight of the convergence theorem (paper, Theorem 6.1),
    w(x, x_j) = |x - x_j|^(-alpha) exp(1 - 1/(1 - |x - x_j|^2/delta^2))  for |x - x_j| < delta,
and returns u(x) = phi(x) . data.  By the exactness rule (paper, Corollary 4.5) a datum with
symbol q is reproduced if its derivatives stay in the node's symbol space and
Delta^(alpha/2) q = 0: so {u, dx, dy} needs alpha >= 2 and {u, dx, dy, lap} alpha >= 4
(Delta^2 |xi|^2 = 0).

Written in swimls-bvps (step 8, 2-D pilot) and moved here, unchanged apart from the weight
option, so that both projects use one copy (as SWIMLS1D).

Derivatives.  As in swimls1d, u(x) = P(x)^T b(x) with the basis P fixed (centred at the
evaluation point x0), G = A^T W A and, with g_k = d_k log w and h_kl = W_kl / W,
    G b_k  = A^T W_k (y - A b)
    G b_kl = A^T W_kl (y - A b) - A^T W_k A b_l - A^T W_l A b_k .
With W^1/2 A = Q1 R and the complement Q2 (W^1/2 (I - A B0) = Q2 Q2^T W^1/2), the operators
mapping data to coefficients are
    B0   = R^-1 Q1^T W^1/2
    B_k  = R^-1 (Q2^T diag(g_k) Q1)^T (Q2^T W^1/2)
    B_kl = R^-1 [(Q2^T diag(h_kl) Q1)^T (Q2^T W^1/2) - Q1^T diag(g_k) Q1 R B_l
                                                     - Q1^T diag(g_l) Q1 R B_k]
(no subtraction of nearly equal terms, see swimls1d).  Q2 is never formed: the
products Q2^T X are taken from Q^T X, applied with the Householder reflectors (LAPACK
?ormqr), at O(R^2 M) cost instead of O(R^3).
    u      = b_0
    d_k u  = b_{e_k} / delta + (b_k)_0
    d_kl u = c_kl b_{e_k + e_l} / delta^2 + ((b_l)_{e_k} + (b_k)_{e_l}) / delta + (b_kl)_0,
with c_kk = 2, c_kl = 1 (k != l).

Numerics: basis ((y - x0)/delta)^e with equilibrated columns, rows scaled by delta^order,
weights normalised by their maximum, leverage-aware cutoff as in swimls1d.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import lapack, solve_triangular

SYMBOLS = {"u": 0, "dx": 1, "dy": 1, "lap": 2}       # symbol -> order


def exponents(m):
    return np.array([(i, t - i) for t in range(m + 1) for i in range(t, -1, -1)], dtype=int)


def falling(e, r):
    out = np.ones(np.broadcast(e, r).shape)
    for t in range(int(np.max(r)) if np.size(r) else 0):
        out = np.where(r > t, out * (e - t), out)
    return out


class SWIMLS2D:
    """phi, grad phi, Hessian of phi at points, for nodes carrying the same symbols.

    Parameters
    ----------
    nodes   : (N, 2) node coordinates.
    symbols : functionals at every node, e.g. ("u", "dx", "dy") or ("u", "dx", "dy", "lap").
    degree  : total degree m of the local polynomial space (m >= 2).
    alpha   : even exponent of the singular weight.
    delta   : Gaussian decay length, or support radius of the bump (also the basis and row scale).
    cutoff  : a node's rows are dropped when w (1 + |z|)^(2m) < cutoff, z = |x_j - x|/delta.
    weight  : "gauss" (default) or "bump" (compact support, see the module docstring).
    """

    def __init__(self, nodes, symbols=("u", "dx", "dy"), degree=4, alpha=2, delta=0.1,
                 cutoff=1e-32, weight="gauss"):
        if weight not in ("gauss", "bump"):
            raise ValueError("weight must be 'gauss' or 'bump'")
        self.weight = weight
        if alpha <= 0 or alpha % 2:
            raise ValueError("alpha must be a positive even integer")
        if degree < 2:
            raise ValueError("degree >= 2 is needed for the Hessian")
        self.nodes = np.asarray(nodes, dtype=float).reshape(-1, 2)
        self.symbols = tuple(symbols)
        self.degree, self.alpha, self.delta, self.cutoff = degree, alpha, float(delta), cutoff
        K, N = len(self.symbols), len(self.nodes)
        self.row_node = np.repeat(np.arange(N), K)
        self.row_sym = np.tile(np.arange(K), N)
        self.row_order = np.array([SYMBOLS[s] for s in self.symbols])[self.row_sym]
        self.R = N * K
        self.exps = exponents(degree)
        self.M = len(self.exps)
        self.idx = {tuple(e): i for i, e in enumerate(self.exps)}
        self.row_scale = self.delta ** self.row_order.astype(float)

    def index(self, i, symbol):
        return i * len(self.symbols) + self.symbols.index(symbol)

    def _rows(self, z, sym):
        """Rows of the symbols `sym` at offsets z = (x_j - x)/delta, scaled by delta^order."""
        E = self.exps[None, :, :]
        out = np.zeros((len(z), self.M))
        for s_name, s in ((n, k) for k, n in enumerate(self.symbols)):
            sel = sym == s
            if not sel.any():
                continue
            terms = {"u": [(1.0, (0, 0))], "dx": [(1.0, (1, 0))], "dy": [(1.0, (0, 1))],
                     "lap": [(1.0, (2, 0)), (1.0, (0, 2))]}[s_name]
            zs = z[sel][:, None, :]
            acc = np.zeros((sel.sum(), self.M))
            for c, beta in terms:
                b = np.array(beta)[None, None, :]
                pw = E - b
                ff = np.prod(falling(E, b), axis=2)
                mon = np.prod(np.where(pw >= 0, zs ** np.maximum(pw, 0), 0.0), axis=2)
                acc += c * np.where(np.all(pw >= 0, axis=2), ff * mon, 0.0)
            out[sel] = acc
        return out

    def phi(self, x):
        """(phi (R,), grad (2, R), hess (3, R) as [xx, xy, yy]) at the point x."""
        x = np.asarray(x, dtype=float)
        t = x[None, :] - self.nodes                              # (N, 2)
        r2 = np.sum(t * t, axis=1)
        if np.any(r2 < (1e-14 * max(1.0, np.ptp(self.nodes))) ** 2):
            raise ValueError("evaluation at a node: use the nodal data directly")
        a, d = self.alpha, self.delta
        if self.weight == "gauss":
            lw = -0.5 * a * np.log(r2) - r2 / d ** 2
        else:
            s = r2 / d ** 2
            inside = s < 1
            one = np.where(inside, 1.0 - s, 1.0)
            lw = np.where(inside, -0.5 * a * np.log(r2) + 1.0 - 1.0 / one, -np.inf)
            if not inside.any():
                raise np.linalg.LinAlgError("no node within the support radius delta")
        lw -= lw.max()
        w = np.exp(lw)
        lev = (1.0 + np.sqrt(r2) / d) ** (2 * self.degree)
        keep = np.nonzero((w * lev > self.cutoff)[self.row_node])[0]
        nd, sy = self.row_node[keep], self.row_sym[keep]
        tk = t[nd]
        r2k = r2[nd]
        W = w[nd]
        # d_k log w and d_kl log w; the bump factor exp(1 - 1/(1 - s)) replaces
        # exp(-s), i.e. the factor 1/d^2 becomes 1/(d^2 (1 - s)^2) plus a curvature term
        if self.weight == "gauss":
            c1 = np.full(len(r2k), 1.0 / d ** 2)
            c2 = np.zeros(len(r2k))
        else:
            onek = 1.0 - r2k / d ** 2
            c1 = 1.0 / (d ** 2 * onek ** 2)
            c2 = 4.0 / (d ** 4 * onek ** 3)
        g = -a * tk / r2k[:, None] - 2 * c1[:, None] * tk         # (Rk, 2): d_k log w
        hess_lw = {}
        for k in range(2):
            for l in range(k, 2):
                dkl = 1.0 if k == l else 0.0
                hess_lw[(k, l)] = (-a * (dkl / r2k - 2 * tk[:, k] * tk[:, l] / r2k ** 2)
                                   - 2 * dkl * c1 - 2 * c2 * tk[:, k] * tk[:, l])
        hkl = {kl: g[:, kl[0]] * g[:, kl[1]] + v for kl, v in hess_lw.items()}

        A = self._rows(-tk / d, sy)                              # z = (x_j - x)/delta
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

        def QT(X):                                               # Q^T X (full Q)
            out, _, info = lapack.dormqr("L", "T", qr, tau, np.asfortranarray(X),
                                         64 * max(1, X.shape[1]))
            if info:
                raise np.linalg.LinAlgError("dormqr failed")
            return out

        Q2sw = QT(np.diag(sw))[M:]                               # (Rk-M, Rk)
        Q2g = {k: QT(g[:, k][:, None] * Q1)[M:] for k in range(2)}  # Q2^T diag(g_k) Q1
        Q1gQ1 = {k: Q1.T @ (g[:, k][:, None] * Q1) for k in range(2)}
        solve = lambda Y: solve_triangular(Rf, Y)
        B0 = solve(Q1.T * sw[None, :])
        Bk = {k: solve(Q2g[k].T @ Q2sw) for k in range(2)}
        Bkl = {}
        for (k, l), h in hkl.items():
            Q2h = QT(h[:, None] * Q1)[M:]
            Bkl[(k, l)] = solve(Q2h.T @ Q2sw - Q1gQ1[k] @ (Rf @ Bk[l]) - Q1gQ1[l] @ (Rf @ Bk[k]))
        B0 = S[:, None] * B0
        Bk = {k: S[:, None] * v for k, v in Bk.items()}
        Bkl = {kl: S[:, None] * v for kl, v in Bkl.items()}

        i0 = self.idx[(0, 0)]
        ek = [self.idx[(1, 0)], self.idx[(0, 1)]]
        u = B0[i0]
        grad = [B0[ek[k]] / d + Bk[k][i0] for k in range(2)]
        hess = []
        for k, l in ((0, 0), (0, 1), (1, 1)):
            e = (int(k == 0) + int(l == 0), int(k == 1) + int(l == 1))
            c = 2.0 if k == l else 1.0
            hess.append(c * B0[self.idx[e]] / d ** 2
                        + (Bk[l][ek[k]] + Bk[k][ek[l]]) / d + Bkl[(k, l)][i0])
        sc = self.row_scale[keep]

        def full(v):
            out = np.zeros(self.R)
            out[keep] = v * sc
            return out

        return full(u), np.array([full(v) for v in grad]), np.array([full(v) for v in hess])

    def evaluate(self, X, data):
        """u, grad u (2,), hess u (3,) at points X (none a node)."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        U, Gr, H = [], [], []
        for x in X:
            p0, p1, p2 = self.phi(x)
            U.append(p0 @ data)
            Gr.append(p1 @ data)
            H.append(p2 @ data)
        return np.array(U), np.array(Gr), np.array(H)
