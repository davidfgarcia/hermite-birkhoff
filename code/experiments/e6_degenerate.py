"""E6 - configurations where condition (i) fails but the datum is still reproduced
(Theorem 4.8 / Corollary 4.9), their generic perturbations, and the error identities of
Corollary 4.4 at alpha = 2.

Every verdict uses the 2-radius high-precision check of experiments.common.

Run from code/:  python3 -m experiments.e6_degenerate
"""
import csv
from multiprocessing import Pool

import mpmath as mp
import numpy as np

import hbmls as hb
from experiments.common import check_node, describe, lambda_u


def case(name):
    """Returns (functionals, op kwargs, nodes to check, what Theorem 4.8 predicts)."""
    if name == "1D interpolatory: f'(0), f(1), f'(2.5)":
        fun = [hb.derivative_1d(0.0, 1), hb.derivative_1d(1.0, 0), hb.derivative_1d(2.5, 1)]
        return fun, dict(degree=1, alpha=2), [0, 2], "exact"
    if name == "  + value f(4)":
        fun = [hb.derivative_1d(0.0, 1), hb.derivative_1d(1.0, 0), hb.derivative_1d(2.5, 1),
               hb.derivative_1d(4.0, 0)]
        return fun, dict(degree=1, alpha=2), [0, 2], "generic"
    if name == "2D, K=2: grad at x0; Hermite k=2 at x1":
        p0, p1 = (0.2, 0.3), (0.7, 0.6)
        fun = [hb.partial(p0, (1, 0)), hb.partial(p0, (0, 1))] + hb.hermite_set(p1, 2)
        return fun, dict(degree=2, alpha=2), [0], "exact"
    if name == "  + Hermite k=2 at a third node":
        p0, p1, p2 = (0.2, 0.3), (0.7, 0.6), (0.4, 0.9)
        fun = [hb.partial(p0, (1, 0)), hb.partial(p0, (0, 1))] + hb.hermite_set(p1, 2) + hb.hermite_set(p2, 2)
        return fun, dict(degree=2, alpha=2), [0], "generic"
    if name == "2D, K=2: f, Dxx at x0; Hermite k=2 at x1 (alpha=4)":
        p0, p1 = (0.2, 0.3), (0.7, 0.6)
        fun = [hb.value(p0), hb.partial(p0, (2, 0))] + hb.hermite_set(p1, 2)
        return fun, dict(degree=2, alpha=4), [0], "exact"
    if name == "  + Hermite k=2 at a third node (alpha=4)":
        p0, p1, p2 = (0.2, 0.3), (0.7, 0.6), (0.4, 0.9)
        fun = [hb.value(p0), hb.partial(p0, (2, 0))] + hb.hermite_set(p1, 2) + hb.hermite_set(p2, 2)
        return fun, dict(degree=2, alpha=4), [0], "generic"
    if name == "2D symmetric: Dx at 0; Hermite k=1 on the x2-axis":
        fun = [hb.partial((0.0, 0.0), (1, 0))] + [f for q in [(0.0, 0.7), (0.0, -0.4), (0.0, 1.3)]
                                                  for f in hb.hermite_set(q, 1)]
        return fun, dict(degree=1, alpha=2), [0], "exact (Prop4.11)"
    if name == "  nodes moved 0.05 off the axis":
        fun = [hb.partial((0.0, 0.0), (1, 0))] + [f for q in [(0.05, 0.7), (-0.05, -0.4), (0.05, 1.3)]
                                                  for f in hb.hermite_set(q, 1)]
        return fun, dict(degree=1, alpha=2), [0], "generic"
    if name == "2D, K=2: Hermite k=3 at x0 and x1 (alpha=2)":
        p0, p1 = (0.2, 0.3), (0.7, 0.6)
        fun = hb.hermite_set(p0, 3) + hb.hermite_set(p1, 3)
        return fun, dict(degree=3, alpha=2), [0], "conjecture"
    raise KeyError(name)


CASES = ["1D interpolatory: f'(0), f(1), f'(2.5)", "  + value f(4)",
         "2D, K=2: grad at x0; Hermite k=2 at x1", "  + Hermite k=2 at a third node",
         "2D, K=2: f, Dxx at x0; Hermite k=2 at x1 (alpha=4)", "  + Hermite k=2 at a third node (alpha=4)",
         "2D symmetric: Dx at 0; Hermite k=1 on the x2-axis", "  nodes moved 0.05 off the axis",
         "2D, K=2: Hermite k=3 at x0 and x1 (alpha=2)"]


def run(name):
    fun, kw, nodes, pred = case(name)
    op = hb.SingularMLS(fun, **kw)
    out = []
    for seed in (1, 2):
        y = np.random.default_rng(seed).normal(size=len(fun))
        for node in nodes:
            for f, e1, e2, v in check_node(op, y, node):
                if f.order == 0 and name.startswith("2D, K=2: f"):
                    continue           # the value datum satisfies (i); not of interest
                out.append(dict(case=name, seed=seed, datum=f"{describe(f)} at {tuple(round(c, 2) for c in f.point)}",
                                theorem=pred, err=e1, err_r1e5=e2, verdict=v))
    return out


def identities():
    """Signed errors at a node with full Hermite data of order 3 (2D), alpha = 2."""
    rng = np.random.default_rng(5)
    pts = rng.random((7, 2))
    fun = hb.hermite_set(pts[0], 3) + [f for q in pts[1:] for f in hb.hermite_set(q, 1)]
    op = hb.SingularMLS(fun, degree=3, alpha=2)
    mpop = hb.MPSingularMLS(op)
    y = rng.normal(size=len(fun))
    err = {}
    for r in (1e-6, 1e-5):
        cache = {}
        for i in np.nonzero(op.node_of_row == 0)[0]:
            f = op.fun[i]
            with mp.workdps(200):
                err[(describe(f), r)] = float(lambda_u(mpop, f, y, r, cache) - mp.mpf(y[i]))
    rows = []
    for r in (1e-6, 1e-5):
        e = lambda s: err[(s, r)]
        rows.append(dict(radius=r, Dxx=e("Dxx"), Dyy=e("Dyy"), Dxy=e("Dxy"),
                         Dxxx=e("Dxxx"), three_Dxyy=3 * e("Dxyy"), Dyyy=e("Dyyy"), three_Dxxy=3 * e("Dxxy"),
                         lower_max=max(abs(e(s)) for s in ("f", "Dx", "Dy"))))
    return rows


if __name__ == "__main__":
    with Pool(2) as pool:
        rows = [r for block in pool.map(run, CASES) for r in block]
    print("(a) degenerate configurations: condition (i) fails at the checked data")
    for r in rows:
        print(f"  {r['case']:<52} seed {r['seed']}  {r['datum']:<22} Theory: {r['theorem']:<16}"
              f" err {r['err']:.1e} / {r['err_r1e5']:.1e}  -> {r['verdict']}")
    ids = identities()
    print("\n(b) error identities, Hermite k=3 node in 2D, alpha=2 (signed errors)")
    for r in ids:
        print("  r={radius:.0e}: Dxx {Dxx:+.12e}  Dyy {Dyy:+.12e}  Dxy {Dxy:+.1e}\n"
              "           Dxxx {Dxxx:+.12e}  3Dxyy {three_Dxyy:+.12e}\n"
              "           Dyyy {Dyyy:+.12e}  3Dxxy {three_Dxxy:+.12e}   max |err| f,Dx,Dy {lower_max:.1e}".format(**r))
    for name, data in (("e6_degenerate", rows), ("e6_identities", ids)):
        with open(f"../results/{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
