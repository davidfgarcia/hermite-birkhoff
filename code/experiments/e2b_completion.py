"""E2b - Birkhoff data made exact by completion.

(a) Exactness: the gapped patterns of E2 that SWIMLS fails to reproduce are completed
    (missing lower-order functionals estimated by a smooth MLS fit); every ORIGINAL
    datum is then checked with high-precision stencils.
(b) Accuracy, Komargodski-Levin setting: values at some points, gradients at other
    points, Franke function.  Plain SWIMLS vs completed SWIMLS: error of u on a grid
    and mismatch of the gradient at the gradient-only points.

Run from code/:  python3 -m experiments.e2b_completion
"""
import csv
import time
from multiprocessing import Pool

import numpy as np

import hbmls as hb
from experiments.common import EXACT, check_node, describe
from experiments.e3_convergence import GRIDPTS, EXACT as FEXACT, deriv, halton

PATTERNS_1D = {"{1}": (1,), "{2}": (2,), "{0,2}": (0, 2), "{1,2}": (1, 2), "{0,1,3}": (0, 1, 3)}
PATTERNS_2D = {
    "Dx only": lambda p: [hb.partial(p, (1, 0))],
    "f, Dxx": lambda p: [hb.value(p), hb.partial(p, (2, 0))],
    "f, Dx, Dxy": lambda p: [hb.value(p), hb.partial(p, (1, 0)), hb.partial(p, (1, 1))],
    "f, Laplacian": lambda p: [hb.value(p), hb.laplacian(p)],
}


def exactness(spec):
    kind, name = spec
    if kind == "1d":
        others = [0.7, 1.5, 2.6]
        given = [hb.derivative_1d(0.0, r) for r in PATTERNS_1D[name]]
        fun = given + [f for q in others for f in hb.hermite_set([q], 3)]
        rng = np.random.default_rng(7)
        m, alpha, delta = 5, 8, 2.0
    else:
        rng = np.random.default_rng(11)
        pts = rng.random((8, 2))
        given = PATTERNS_2D[name](pts[0])
        fun = given + [f for q in pts[1:] for f in hb.hermite_set(q, 2)]
        m, alpha, delta = 3, 6, 1.0
    y = rng.normal(size=len(fun))
    fc, yc = hb.complete(fun, y, degree=m, delta=delta)
    added = fc[len(fun):]
    op = hb.SingularMLS(fc, degree=m, alpha=alpha, delta=delta)
    out = []
    for f, got, want, err in check_node(op, yc, 0):
        if f in given:
            out.append(dict(dim=1 if kind == "1d" else 2, pattern=name,
                            added=", ".join(describe(g) for g in added), datum=describe(f), err=err))
    return out


def accuracy(n):
    vp = halton(n, 2, skip=31)
    gp = halton(n, 2, skip=31 + n)                       # different points
    fun = [hb.value(p) for p in vp] + [f for p in gp for f in (hb.partial(p, (1, 0)), hb.partial(p, (0, 1)))]
    h = 1 / np.sqrt(2 * n)
    y = np.array([f.apply(deriv) for f in fun])
    res = {}
    for label in ("plain", "completed"):
        if label == "plain":
            F, Y = fun, y
        else:
            F, Y = hb.complete(fun, y, degree=3, delta=2.0 * h, length=h)
        op = hb.SingularMLS(F, degree=3, alpha=2, delta=2.0 * h, length=h, cutoff=1e-14)
        err_u = np.max(np.abs(op(GRIDPTS, Y) - FEXACT))
        # gradient mismatch at the gradient-only points (central differences, step 1e-6)
        e = 1e-6
        mism = 0.0
        for p in gp[:20]:
            for j, beta in enumerate([(1, 0), (0, 1)]):
                dv = np.zeros(2)
                dv[j] = e
                g = (op([p + dv], Y)[0] - op([p - dv], Y)[0]) / (2 * e)
                mism = max(mism, abs(g - deriv(beta, p)))
        res[label] = (err_u, mism)
    return dict(n_value_pts=n, n_gradient_pts=n,
                plain_max_err=res["plain"][0], plain_grad_mismatch=res["plain"][1],
                completed_max_err=res["completed"][0], completed_grad_mismatch=res["completed"][1])


if __name__ == "__main__":
    t0 = time.time()
    jobs = [("1d", n) for n in PATTERNS_1D] + [("2d", n) for n in PATTERNS_2D]
    with Pool(2) as pool:
        ex = [r for block in pool.map(exactness, jobs) for r in block]
        acc = pool.map(accuracy, [64, 128, 256, 512, 1024])
    print("(a) exactness of the ORIGINAL data after completion")
    for r in ex:
        print(f"  {r['dim']}D {r['pattern']:<12} added [{r['added']}]  {r['datum']:<10} err={r['err']:.1e}"
              f"  {'exact' if r['err'] < EXACT else 'NOT EXACT'}")
    print(f"  {sum(r['err'] < EXACT for r in ex)}/{len(ex)} original data reproduced exactly")
    print("\n(b) values at n points + gradients at n other points (Franke)")
    print("     n   plain: max err  grad mismatch   completed: max err  grad mismatch")
    for r in acc:
        print(f"  {r['n_value_pts']:5d}      {r['plain_max_err']:.2e}     {r['plain_grad_mismatch']:.2e}"
              f"            {r['completed_max_err']:.2e}     {r['completed_grad_mismatch']:.2e}")
    for name, rows in (("e2b_exactness", ex), ("e2b_accuracy", acc)):
        with open(f"../results/{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{time.time() - t0:.0f}s")
