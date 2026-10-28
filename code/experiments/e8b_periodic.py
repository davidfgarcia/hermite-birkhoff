"""E8b - like E8, but on periodically jittered grids: node (i, j) sits at
h*(i + xi_{i mod 2, j mod 2}) with four fixed offsets, so that the scaled local
configurations repeat exactly at every resolution.  The constants of the convergence
theorem are then the same for all h, and the observed rates should be m+1 (u) and m (grad).

Run from code/:  python3 -m experiments.e8b_periodic
"""
import csv
from multiprocessing import Pool

import numpy as np

import hbmls as hb
from experiments.e3_convergence import GRIDPTS, deriv
from experiments.e8_compact import EX, GX, GY, F, grad, rate

OFF = np.array([[[0.13, -0.21], [-0.27, 0.08]], [[0.05, 0.24], [-0.18, -0.12]]])
NN = [8, 12, 16, 24, 32, 48, 64, 96]
CONFIGS = [(0, 3, 2, 3.0), (1, 3, 2, 2.0), (1, 4, 2, 2.0), (2, 3, 4, 2.0), (2, 5, 4, 2.0)]


def nodes(n):
    h = 1.0 / n
    pts = [h * (np.array([i, j]) + OFF[i % 2, j % 2]) for i in range(-3, n + 4) for j in range(-3, n + 4)]
    return np.array(pts), h


def run(args):
    k, m, alpha, c, n = args
    pts, h = nodes(n)
    fun = [f for p in pts for f in hb.hermite_set(p, k)]
    delta = c * h
    op = hb.SingularMLS(fun, degree=m, alpha=alpha, delta=delta, length=delta, cutoff=1e-16, weight="bump")
    y = op.data_from(lambda b, p: deriv(b, p, F))
    e = op(GRIDPTS, y) - EX
    gx = grad(op, y, h, 0) - GX
    gy = grad(op, y, h, 1) - GY
    g = np.hypot(gx, gy)
    return dict(k=k, degree=m, alpha=alpha, c=c, n=n, h=h, max_err=float(np.max(np.abs(e))),
                rms_err=float(np.sqrt(np.mean(e ** 2))), grad_max_err=float(g.max()),
                grad_rms_err=float(np.sqrt(np.mean(g ** 2))))


if __name__ == "__main__":
    jobs = [cfg + (n,) for cfg in CONFIGS for n in NN]
    with Pool(2) as pool:
        rows = pool.map(run, jobs)
    with open("../results/e8b_periodic.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(" k  m  α   c   max err of u, h = 1/8 ... 1/96                             rate u (m+1)  rate grad (m)")
    for cfg in CONFIGS:
        rs = [r for r in rows if (r["k"], r["degree"], r["alpha"], r["c"]) == cfg]
        hs = np.array([r["h"] for r in rs])
        es = np.array([r["max_err"] for r in rs])
        gs = np.array([r["grad_max_err"] for r in rs])
        k, m, a, c = cfg
        print(f" {k}  {m}  {a}  {c:3.1f}  " + " ".join(f"{e:.1e}" for e in es)
              + f"   {rate(hs, es):5.2f} ({m + 1})   {rate(hs, gs):5.2f} ({m})")
