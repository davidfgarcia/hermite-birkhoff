"""High-precision reference evaluator (mpmath).

Used to verify lambda(u) at the nodes: derivatives of the interpolant are
computed from tensor Chebyshev finite-difference stencils of tiny radius, in
enough working precision that both truncation and rounding errors are far below
the double-precision level.
"""
from __future__ import annotations

import itertools
from math import factorial

import mpmath as mp

from .core import SingularMLS


class MPSingularMLS:
    """Same operator as SingularMLS, in arbitrary precision (evaluated off the nodes)."""

    def __init__(self, op: SingularMLS, dps: int = 60):
        self.op = op
        self.dps = dps

    def _row(self, f, x):
        """The functional applied to the basis (y - x)^e, exactly, in mp."""
        row = [mp.mpf(0)] * len(self.op.exps)
        p = [mp.mpf(v) for v in f.point]
        for coef, beta in f.terms:
            for j, e in enumerate(self.op.exps):
                if any(e[i] < beta[i] for i in range(len(beta))):
                    continue
                v = mp.mpf(coef)
                for i in range(len(beta)):
                    v *= mp.mpf(factorial(int(e[i]))) / factorial(int(e[i] - beta[i]))
                    v *= (p[i] - x[i]) ** int(e[i] - beta[i])
                row[j] += v
        return row

    def value(self, x, y):
        """u(x) for x not a node."""
        with mp.workdps(self.dps):
            x = [mp.mpf(v) for v in x]
            op = self.op
            rows, rhs = [], []
            for i, f in enumerate(op.fun):
                p = op.nodes[op.node_of_row[i]]
                d2 = sum((mp.mpf(p[j]) - x[j]) ** 2 for j in range(op.dim))
                w = d2 ** (-(op.alpha // 2))
                if op.delta is not None:
                    w *= mp.exp(-d2 / mp.mpf(op.delta) ** 2)
                s = mp.sqrt(w) * mp.mpf(op.length) ** f.order
                rows.append([s * v for v in self._row(f, x)])
                rhs.append(s * mp.mpf(y[i]))
            return self._lstsq(rows, rhs)[0]

    @staticmethod
    def _lstsq(rows, rhs):
        """min ||A a - b|| via the augmented system [[I, A], [A^T, 0]] [r; a] = [b; 0],
        solved by LU with partial pivoting (mpmath's QR does not pivot)."""
        n, m = len(rows), len(rows[0])
        K = mp.zeros(n + m, n + m)
        f = mp.zeros(n + m, 1)
        for i in range(n):
            K[i, i] = 1
            f[i] = rhs[i]
            for j in range(m):
                K[i, n + j] = rows[i][j]
                K[n + j, i] = rows[i][j]
        sol = mp.lu_solve(K, f)
        return [sol[n + j] for j in range(m)]

    @staticmethod
    def _stencil(order: int, extra: int):
        """Chebyshev points s_i in [-1,1] (even count, so 0 is never sampled) and
        weights c_i with sum c_i p(s_i) = p^(order)(0) for all p of degree < n."""
        n = order + extra + 1
        n += n % 2
        s = [mp.cos((2 * i + 1) * mp.pi / (2 * n)) for i in range(n)]
        V = mp.matrix([[si ** q for si in s] for q in range(n)])
        rhs = mp.matrix([mp.mpf(factorial(order)) if q == order else 0 for q in range(n)])
        return s, mp.lu_solve(V, rhs)

    def derivative(self, x0, beta, y, r: float = 1e-6, extra: int = 5):
        """D^beta u(x0) from a tensor stencil of radius r around x0."""
        with mp.workdps(self.dps):
            dim = self.op.dim
            active = [j for j in range(dim) if beta[j] > 0] or [0]
            rr = mp.mpf(r)
            sten = {j: self._stencil(beta[j], extra) for j in active}
            total = mp.mpf(0)
            for idx in itertools.product(*[range(len(sten[j][0])) for j in active]):
                x = [mp.mpf(v) for v in x0]
                wgt = mp.mpf(1)
                for j, i in zip(active, idx):
                    s, c = sten[j]
                    x[j] += rr * s[i]
                    wgt *= c[i] / rr ** beta[j]
                total += wgt * self.value(x, y)
            return total
