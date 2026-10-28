"""E2b - Birkhoff data made exact by completion.

(a) Exactness: for EVERY failing pattern of E2 (1D gaps, 2D gaps and combinations, and
    the Komargodski-Levin gradient-only nodes), each original datum is checked before
    and after completion (missing lower-order functionals added, values estimated by a
    smooth MLS fit).
(b) Accuracy, Komargodski-Levin setting: values at n points, gradients at n other points,
    Franke function.  Plain vs completed: error of u and gradient mismatch at the
    gradient-only points (central differences with two step sizes).

Run from code/:  python3 -m experiments.e2b_completion
"""
import csv
import time
from multiprocessing import Pool

import numpy as np

import hbmls as hb
from experiments.common import check_node, describe
from experiments.e3_convergence import GRIDPTS, EXACT as FEXACT, deriv, halton

PATTERNS_1D = {"{0,2}": (0, 2), "{0,3}": (0, 3), "{1}": (1,), "{2}": (2,), "{1,2}": (1, 2),
               "{0,1,3}": (0, 1, 3), "{0,2,3}": (0, 2, 3)}
PATTERNS_2D = {
    "Dx only": lambda p: [hb.partial(p, (1, 0))],
    "f, Dxx": lambda p: [hb.value(p), hb.partial(p, (2, 0))],
    "f, Dx, Dxy": lambda p: [hb.value(p), hb.partial(p, (1, 0)), hb.partial(p, (1, 1))],
    "f, Laplacian": lambda p: [hb.value(p), hb.laplacian(p)],
}


def setup(kind, name):
    if kind == "1d":
        rng = np.random.default_rng(7)
        given = [hb.derivative_1d(0.0, r) for r in PATTERNS_1D[name]]
        fun = given + [f for q in [0.7, 1.5, 2.6] for f in hb.hermite_set([q], 3)]
        return fun, given, rng.normal(size=len(fun)), dict(degree=5, alpha=8, delta=2.0), [0]
    if kind == "2d":
        rng = np.random.default_rng(11)
        pts = rng.random((8, 2))
        given = PATTERNS_2D[name](pts[0])
        fun = given + [f for q in pts[1:] for f in hb.hermite_set(q, 2)]
        return fun, given, rng.normal(size=len(fun)), dict(degree=3, alpha=6, delta=1.0), [0]
    rng = np.random.default_rng(21)                          # Komargodski-Levin setting
    vpts, gpts = rng.random((8, 2)), rng.random((8, 2))
    fun = [hb.value(p) for p in vpts] + [f for p in gpts for f in (hb.partial(p, (1, 0)), hb.partial(p, (0, 1)))]
    given = fun[8:14]                                        # gradients at the first 3 gradient points
    return fun, given, rng.normal(size=len(fun)), dict(degree=2, alpha=2, delta=1.0), [8, 9, 10]


def exactness(spec):
    kind, name = spec
    fun, given, y, kw, nodes = setup(kind, name)
    before = hb.SingularMLS(fun, **kw)
    fc, yc = hb.complete(fun, y, degree=kw["degree"], delta=kw["delta"])
    after = hb.SingularMLS(fc, **kw)
    added = [describe(g) for g in fc[len(fun):] if g.point == given[0].point]
    out = []
    for node in nodes:
        b = {f: v for f, *_, v in check_node(before, y, node)}
        a = {f: (e, v) for f, e, _, v in check_node(after, yc, node)}
        for f in given:
            if f in b:
                out.append(dict(pattern=f"{kind}: {name}", added=", ".join(added), datum=describe(f),
                                before=b[f], after=a[f][1], err_after=a[f][0]))
    return out


def accuracy(n):
    vp = halton(n, 2, skip=31)
    gp = halton(n, 2, skip=31 + n)
    fun = [hb.value(p) for p in vp] + [f for p in gp for f in (hb.partial(p, (1, 0)), hb.partial(p, (0, 1)))]
    h = 1 / np.sqrt(2 * n)
    y = np.array([f.apply(deriv) for f in fun])
    res = {}
    for label in ("plain", "completed"):
        F, Y = (fun, y) if label == "plain" else hb.complete(fun, y, degree=3, delta=2.0 * h, length=h)
        op = hb.SingularMLS(F, degree=3, alpha=2, delta=2.0 * h, length=h, cutoff=1e-14)
        err_u = np.max(np.abs(op(GRIDPTS, Y) - FEXACT))
        mism = {}
        for step in (1e-5, 1e-6):
            m = 0.0
            for p in gp:
                for j, beta in enumerate([(1, 0), (0, 1)]):
                    dv = np.zeros(2)
                    dv[j] = step
                    g = (op([p + dv], Y)[0] - op([p - dv], Y)[0]) / (2 * step)
                    m = max(m, abs(g - deriv(beta, p)))
            mism[step] = m
        res[label] = (err_u, mism[1e-5], mism[1e-6])
    return dict(n=n, plain_max_err=res["plain"][0], plain_grad_mismatch=res["plain"][2],
                completed_max_err=res["completed"][0], completed_grad_mismatch_h1e5=res["completed"][1],
                completed_grad_mismatch_h1e6=res["completed"][2])


if __name__ == "__main__":
    t0 = time.time()
    jobs = [("1d", n) for n in PATTERNS_1D] + [("2d", n) for n in PATTERNS_2D] + [("kl", "gradient-only nodes")]
    with Pool(2) as pool:
        ex = [r for block in pool.map(exactness, jobs) for r in block]
        acc = pool.map(accuracy, [64, 128, 256, 512, 1024])
    print("(a) original data at the patched nodes, before -> after completion")
    for r in ex:
        print(f"  {r['pattern']:<28} added [{r['added']}]  {r['datum']:<10} {r['before']:<10} -> {r['after']:<10}"
              f" (err {r['err_after']:.1e})")
    failing = [r for r in ex if r["before"] != "exact"]
    print(f"  {len(failing)} data failing before; {sum(r['after'] == 'exact' for r in failing)} of them exact after; "
          f"{sum(r['after'] == 'exact' for r in ex)}/{len(ex)} exact after in total")
    print("\n(b) values at n points + gradients at n other points (Franke)")
    print("      n  plain: max err  grad mismatch | completed: max err  grad mismatch (FD step 1e-5 / 1e-6)")
    for r in acc:
        print(f"  {r['n']:5d}      {r['plain_max_err']:.2e}     {r['plain_grad_mismatch']:.2e} |"
              f"         {r['completed_max_err']:.2e}     {r['completed_grad_mismatch_h1e5']:.1e} / "
              f"{r['completed_grad_mismatch_h1e6']:.1e}")
    for name, rows in (("e2b_exactness", ex), ("e2b_accuracy", acc)):
        with open(f"../results/{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{time.time() - t0:.0f}s")
