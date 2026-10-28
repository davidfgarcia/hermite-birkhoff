"""E8 - convergence with compactly supported C-infinity weights (the setting of the
convergence theorem): w_j = |x - x_j|^(-alpha) psi(|x - x_j|^2 / delta^2), delta = c*h,
derivative rows scaled by delta^order.  Max error of u and of its gradient on
[0.1, 0.9]^2, Halton nodes, smooth test function.  The theorem predicts O(h^(m+1)) for u
and O(h^m) for the gradient.

Run from code/:  python3 -m experiments.e8_compact
"""
import csv
import time
from multiprocessing import Pool

import numpy as np

import hbmls as hb
from experiments.e3_convergence import GRIDPTS, deriv, halton

NS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
CS = [2.0, 3.0, 4.0]
CONFIGS = [(0, 2, 2), (0, 3, 2), (0, 5, 2), (1, 3, 2), (1, 4, 2), (2, 3, 4), (2, 5, 4)]
F = "smooth"
EX = np.array([deriv((0, 0), p, F) for p in GRIDPTS])
GX = np.array([deriv((1, 0), p, F) for p in GRIDPTS])
GY = np.array([deriv((0, 1), p, F) for p in GRIDPTS])


def grad(op, y, h, j):
    """Fourth-order central difference with step h/100 (truncation and rounding errors
    far below the errors measured)."""
    e = np.zeros(2)
    e[j] = 0.01 * h
    u = lambda t: op(GRIDPTS + t * e, y)
    return (-u(2) + 8 * u(1) - 8 * u(-1) + u(-2)) / (12 * e[j])


def run(args):
    k, m, alpha, c, n = args
    pts = halton(n, 2)
    fun = [f for p in pts for f in hb.hermite_set(p, k)]
    h = 1 / np.sqrt(n)
    delta = c * h
    op = hb.SingularMLS(fun, degree=m, alpha=alpha, delta=delta, length=delta, cutoff=1e-16, weight="bump")
    y = op.data_from(lambda b, p: deriv(b, p, F))
    e = op(GRIDPTS, y) - EX
    gx = grad(op, y, h, 0) - GX
    gy = grad(op, y, h, 1) - GY
    return dict(k=k, degree=m, alpha=alpha, c=c, nodes=n, h=h, max_err=float(np.max(np.abs(e))),
                rms_err=float(np.sqrt(np.mean(e ** 2))),
                grad_max_err=float(np.max(np.hypot(gx, gy))))


def rate(hs, es):
    return np.polyfit(np.log(hs[-3:]), np.log(es[-3:]), 1)[0]


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(k, m, a, c, n) for (k, m, a) in CONFIGS for c in CS for n in NS]
    with Pool(2) as pool:
        rows = pool.map(run, jobs)
    with open("../results/e8_compact.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(" k  m  α   c   max err of u, N = 64 ... 8192                                rate u (m+1)  rate grad (m)")
    for (k, m, a) in CONFIGS:
        for c in CS:
            rs = [r for r in rows if (r["k"], r["degree"], r["alpha"], r["c"]) == (k, m, a, c)]
            hs = np.array([r["h"] for r in rs])
            es = np.array([r["max_err"] for r in rs])
            gs = np.array([r["grad_max_err"] for r in rs])
            print(f" {k}  {m}  {a}  {c:3.1f}  " + " ".join(f"{e:.1e}" for e in es)
                  + f"   {rate(hs, es):5.2f} ({m + 1})   {rate(hs, gs):5.2f} ({m})")
    print(f"{time.time() - t0:.0f}s")
