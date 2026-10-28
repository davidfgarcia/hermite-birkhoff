"""E7 - parameter study for h-convergence (2D, Halton nodes, smooth test function).

For data order k (0: values, 1: + gradients, 2: + Hessians), local degree m, singularity
alpha and Gaussian width delta = c*h, measure the max error on [0.1, 0.9]^2 for
N = 64 ... 4096 nodes and the observed rate over the last three refinements.  The
reference rate for moving least squares with local degree m is h^(m+1).

Run from code/:  python3 -m experiments.e7_parameters
"""
import csv
import time
from multiprocessing import Pool

import numpy as np

import hbmls as hb
from experiments.e3_convergence import GRIDPTS, deriv, halton

NS = [64, 128, 256, 512, 1024, 2048, 4096]
CS = [1.5, 2.0, 3.0, 4.0]
# (k, m, alpha)
CONFIGS = [(0, 2, 2), (0, 3, 2), (1, 2, 2), (1, 3, 2), (1, 3, 4), (1, 4, 2), (2, 3, 4), (2, 4, 4), (2, 5, 4)]
EX = np.array([deriv((0, 0), p, "smooth") for p in GRIDPTS])


def run(args):
    k, m, alpha, c, n = args
    pts = halton(n, 2)
    fun = [f for p in pts for f in hb.hermite_set(p, k)]
    h = 1 / np.sqrt(n)
    op = hb.SingularMLS(fun, degree=m, alpha=alpha, delta=c * h, length=h, cutoff=1e-14)
    yv = op.data_from(lambda b, p: deriv(b, p, "smooth"))
    e = op(GRIDPTS, yv) - EX
    return dict(k=k, degree=m, alpha=alpha, c=c, nodes=n, h=h, max_err=float(np.max(np.abs(e))),
                rms_err=float(np.sqrt(np.mean(e ** 2))))


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(k, m, a, c, n) for (k, m, a) in CONFIGS for c in CS for n in NS]
    with Pool(2) as pool:
        rows = pool.map(run, jobs)
    with open("../results/e7_parameters.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(" k  m  α    c   max err at N=64 ... 4096                                   rate(last 3)  rms rate")
    for (k, m, a) in CONFIGS:
        for c in CS:
            rs = [r for r in rows if (r["k"], r["degree"], r["alpha"], r["c"]) == (k, m, a, c)]
            hs = np.array([r["h"] for r in rs])
            es = np.array([r["max_err"] for r in rs])
            rm = np.array([r["rms_err"] for r in rs])
            rate = np.polyfit(np.log(hs[-3:]), np.log(es[-3:]), 1)[0]
            rrate = np.polyfit(np.log(hs[-3:]), np.log(rm[-3:]), 1)[0]
            print(f" {k}  {m}  {a}  {c:3.1f}   " + " ".join(f"{e:.1e}" for e in es) + f"   h^{rate:4.2f}      h^{rrate:4.2f}")
    print(f"{time.time() - t0:.0f}s")
