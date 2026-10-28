"""One-dimensional SWIMLS (singular-weight interpolating moving least squares) for
Hermite-Birkhoff data, with exact first and second derivatives of the approximant.

Data: a list of (node index i, derivative order r), meaning the datum f^(r)(x_i).
At an evaluation point x the method fits a polynomial of degree <= m by weighted least
squares, datum (i, r) getting the weight

    w(x, x_i) = |x - x_i|^(-alpha) * exp(-((x - x_i)/delta)^2),     alpha even,

(weight="gauss", the default) or, with weight="bump", the compactly supported C-infinity
weight of the convergence theorem,

    w(x, x_i) = |x - x_i|^(-alpha) * psi(((x - x_i)/delta)^2),  psi(s) = exp(1 - 1/(1-s)), s < 1,

and returns the value of that local polynomial at x:  u(x) = phi(x) . data.

Derivatives.  u(x) = P(x)^T b(x), where b(x) = G(x)^{-1} A^T W(x) y is the coefficient
vector in a basis P that is *fixed* while differentiating (centred at the evaluation
point x0, then evaluated at x = x0), so A does not depend on x and

    b'  = G^{-1} A^T W' (y - A b),
    b'' = G^{-1} [A^T W'' (y - A b) - 2 A^T W' A b'],

and since G^{-1} A^T W = (W^{1/2} A)^+ W^{1/2}, both are weighted least-squares solves
with the ratios W'/W, W''/W folded into the right-hand side (no G^{-1}, so the condition
number of the local problem is not squared).
    u'  = P'^T b + P^T b',   u'' = P''^T b + 2 P'^T b' + P^T b''.

These are the derivatives of the approximant u itself (not the "diffuse" derivatives of
the local fit).  The formulas hold off the nodes.

Origin: written for the boundary-value-problem work (repo swimls-bvps, swimls_bvp/swimls1d.py)
and moved here, unchanged apart from the weight option, so that both projects use one copy.

Numerics: the basis is ((y - x0)/delta)^e with equilibrated columns, derivative rows
are scaled by delta^r (dimensionless data), weights are normalised by their maximum,
and all solves go through one QR factorisation of W^{1/2} A.
"""
from __future__ import annotations

import numpy as np


def falling(e: np.ndarray, r: np.ndarray) -> np.ndarray:
    """e (e-1) ... (e-r+1), elementwise; zero when r > e >= 0."""
    e = np.asarray(e)
    r = np.asarray(r)
    out = np.ones(np.broadcast(e, r).shape)
    for t in range(int(np.max(r)) if np.size(r) else 0):
        out = np.where(r > t, out * (e - t), out)
    return out


class SWIMLS1D:
    """phi(x), phi'(x), phi''(x) for the data rows (node, order).

    Parameters
    ----------
    nodes  : node coordinates, shape (n,).
    orders : derivative orders carried by every node, e.g. (0, 1) or (0, 1, 2).
    degree : degree m of the local polynomial space.
    alpha  : even exponent of the singular weight.
    delta  : Gaussian decay length, or support radius of the bump (also the basis/row scale).
    cutoff : a node's rows are dropped when w * (1 + |z|)^(2m) < cutoff, z = (x_j - x)/delta.
             This is (up to constants) the influence of the row on the local fit:
             weight times squared leverage of a degree-m polynomial at distance z.
             A cutoff on the weight alone is badly miscalibrated - dropping rows of
             weight 1e-18 perturbs phi'' at the 1e-9 level for m = 6.
    weight : "gauss" (default) or "bump" (compact support, see the module docstring).
    """

    def __init__(self, nodes, orders=(0, 1), degree=3, alpha=2, delta=0.1, cutoff=1e-32,
                 weight="gauss"):
        if alpha <= 0 or alpha % 2:
            raise ValueError("alpha must be a positive even integer")
        if weight not in ("gauss", "bump"):
            raise ValueError("weight must be 'gauss' or 'bump'")
        self.weight = weight
        self.nodes = np.asarray(nodes, dtype=float)
        self.orders = tuple(orders)
        self.degree = degree
        self.alpha = alpha
        self.delta = float(delta)
        self.cutoff = cutoff
        K = len(self.orders)
        n = len(self.nodes)
        # data row j <-> (node j // K, order orders[j % K])
        self.row_node = np.repeat(np.arange(n), K)
        self.row_order = np.tile(np.array(self.orders), n)
        self.R = n * K
        self.exps = np.arange(degree + 1)
        self.row_scale = self.delta ** self.row_order.astype(float)

    def index(self, i: int, r: int) -> int:
        """Position of datum f^(r)(x_i) in the data vector."""
        return i * len(self.orders) + self.orders.index(r)

    def _weights(self, x):
        t = x - self.nodes
        if np.any(np.abs(t) < 1e-14 * max(1.0, np.ptp(self.nodes))):
            raise ValueError("evaluation at a node: use the nodal data directly")
        a, d = self.alpha, self.delta
        if self.weight == "gauss":
            lw = -a * np.log(np.abs(t)) - (t / d) ** 2
            g = -a / t - 2 * t / d ** 2            # w'/w  = (log w)'
            dg = a / t ** 2 - 2 / d ** 2            # (log w)''
        else:
            s = (t / d) ** 2
            inside = s < 1
            one = np.where(inside, 1.0 - s, 1.0)  # 1 - s, guarded outside the support
            lw = np.where(inside, -a * np.log(np.abs(t)) + 1.0 - 1.0 / one, -np.inf)
            g = np.where(inside, -a / t - 2 * t / (d ** 2 * one ** 2), 0.0)
            dg = np.where(inside, a / t ** 2 - 2 / (d ** 2 * one ** 2)
                          - 8 * t ** 2 / (d ** 4 * one ** 3), 0.0)
        lw = lw - lw.max()
        w = np.exp(lw)
        return w, g, g * g + dg                 # w, w'/w, w''/w

    def phi(self, x: float):
        """(phi, phi', phi'') at x, each of shape (R,): u^(k)(x) = phi^(k)(x) . data."""
        w, g1, g2 = self._weights(x)
        lev = (1.0 + np.abs(x - self.nodes) / self.delta) ** (2 * self.degree)
        keep_node = w * lev > self.cutoff
        rows = np.nonzero(keep_node[self.row_node])[0]
        nd, rr = self.row_node[rows], self.row_order[rows]
        z = (self.nodes[nd] - x) / self.delta
        E = self.exps[None, :]
        pw = E - rr[:, None]
        A = np.where(pw >= 0, falling(E, rr[:, None]) * z[:, None] ** np.maximum(pw, 0), 0.0)
        W, g1, g2 = w[nd], g1[nd], g2[nd]
        # equilibrate the columns (basis change b = S c); undone at the end
        S = 1.0 / np.maximum(np.sqrt(W @ A**2), 1e-300)
        A = A * S[None, :]

        sw = np.sqrt(W)
        M = A.shape[1]
        Q, Rf = np.linalg.qr(sw[:, None] * A, mode="complete")
        Q1, Q2, Rf = Q[:, :M], Q[:, M:], Rf[:M]
        if np.min(np.abs(np.diag(Rf))) < 1e-13 * np.max(np.abs(np.diag(Rf))):
            raise np.linalg.LinAlgError("local least-squares problem is rank deficient")

        # G b' = A^T W1 (y - A b)             ->  B1 = B0 diag(g1) (I - A B0)
        # G b'' = A^T W2 (y - A b) - 2 G1 b'  ->  B2 = B0 diag(g2) (I - A B0) - 2 B0 diag(g1) A B1
        # With W^1/2 A = Q1 R and the residual projector taken from the orthogonal
        # complement, W^1/2 (I - A B0) = Q2 Q2^T W^1/2, these become
        #   B0 = R^-1 Q1^T W^1/2
        #   B1 = R^-1 Q1^T diag(g1) Q2 Q2^T W^1/2
        #   B2 = R^-1 [Q1^T diag(g2) Q2 Q2^T W^1/2 - 2 Q1^T diag(g1) Q1 R B1]
        # with no subtraction of nearly equal terms.  The earlier form
        # B1 = B0 g1 - (B0 g1 A) B0 lost 1-3 digits in phi', phi'' to that cancellation
        # (RESULTS step 4); this one is within a few ulps of correctly rounded.
        # Products are ordered (M x R)(R x (R-M))((R-M) x R), never forming the R x R
        # projector, which kept the cost of the fix to ~1.2-1.6x of the old formulas.
        Q2sw = Q2.T * sw[None, :]
        Q1g1 = Q1.T * g1[None, :]
        B0 = np.linalg.solve(Rf, Q1.T * sw[None, :])
        B1 = np.linalg.solve(Rf, (Q1g1 @ Q2) @ Q2sw)
        B2 = np.linalg.solve(Rf, ((Q1.T * g2[None, :]) @ Q2) @ Q2sw
                             - 2 * (Q1g1 @ Q1) @ (Rf @ B1))
        B0, B1, B2 = (S[:, None] * B for B in (B0, B1, B2))

        d = self.delta
        u0 = B0[0]
        u1 = B0[1] / d + B1[0] if self.degree >= 1 else B1[0]
        u2 = (2 * B0[2] / d ** 2 if self.degree >= 2 else 0) \
            + (2 * B1[1] / d if self.degree >= 1 else 0) + B2[0]
        out = []
        for v in (u0, u1, u2):
            full = np.zeros(self.R)
            full[rows] = v * self.row_scale[rows]
            out.append(full)
        return tuple(out)

    def evaluate(self, X, data):
        """u, u', u'' at the points X (none of them a node)."""
        X = np.atleast_1d(np.asarray(X, dtype=float))
        res = np.array([[p @ data for p in self.phi(x)] for x in X])
        return res[:, 0], res[:, 1], res[:, 2]
