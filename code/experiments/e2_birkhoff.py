"""E2 - Birkhoff (gapped) data and general differential functionals.

One node carries a test pattern of functionals; the other nodes carry full Hermite
data.  alpha is large enough that the order condition (order < alpha) holds, so any
failure is due to the *structure* of the node's data.  Each datum is compared with
the prediction of the D-invariance rule: a functional q(D) at x_i is reproduced iff
every derivative of its symbol q lies in the span of the node's symbols.

Run from code/:  python3 -m experiments.e2_birkhoff
"""
import csv
import time
from multiprocessing import Pool

import numpy as np

import hbmls as hb
from experiments.common import check_node, describe, predicted


def pattern_1d(p, orders):
    return [hb.derivative_1d(p, r) for r in orders]


P1 = {
    "{0,1,2,3}": (0, 1, 2, 3), "{0,1,2}": (0, 1, 2), "{0,2}": (0, 2), "{0,3}": (0, 3),
    "{1}": (1,), "{2}": (2,), "{1,2}": (1, 2), "{0,1,3}": (0, 1, 3), "{0,2,3}": (0, 2, 3),
}

V = (0.6, 0.8)
P2 = {
    "f, Dx": lambda p: [hb.value(p), hb.partial(p, (1, 0))],
    "f, Dx, Dy": lambda p: [hb.value(p), hb.partial(p, (1, 0)), hb.partial(p, (0, 1))],
    "f, Dv (directional)": lambda p: [hb.value(p), hb.directional(p, V)],
    "Dx only": lambda p: [hb.partial(p, (1, 0))],
    "f, Dxx": lambda p: [hb.value(p), hb.partial(p, (2, 0))],
    "f, Dx, Dxy": lambda p: [hb.value(p), hb.partial(p, (1, 0)), hb.partial(p, (1, 1))],
    "f, Dx, Dy, Dxy": lambda p: [hb.value(p), hb.partial(p, (1, 0)), hb.partial(p, (0, 1)), hb.partial(p, (1, 1))],
    "f, Laplacian": lambda p: [hb.value(p), hb.laplacian(p)],
    "f, Dx, Dy, Laplacian": lambda p: [hb.value(p), hb.partial(p, (1, 0)), hb.partial(p, (0, 1)), hb.laplacian(p)],
    "full Hermite k=2": lambda p: hb.hermite_set(p, 2),
}


def run_1d(name):
    others = [0.7, 1.5, 2.6]
    fun = pattern_1d(0.0, P1[name]) + [f for q in others for f in hb.hermite_set([q], 3)]
    op = hb.SingularMLS(fun, degree=5, alpha=8, delta=2.0)
    y = np.random.default_rng(7).normal(size=len(fun))
    node_funs = [f for f in fun if f.point == (0.0,)]
    return [dict(dim=1, pattern=name, datum=describe(f), predicted=predicted(f, node_funs, op.alpha),
                 err=e1, err_r1e5=e2, verdict=v) for f, e1, e2, v in check_node(op, y, 0)]


def run_2d(name):
    rng = np.random.default_rng(11)
    pts = rng.random((8, 2))
    fun = P2[name](pts[0]) + [f for q in pts[1:] for f in hb.hermite_set(q, 2)]
    op = hb.SingularMLS(fun, degree=3, alpha=6, delta=1.0)
    y = rng.normal(size=len(fun))
    node_funs = P2[name](pts[0])
    return [dict(dim=2, pattern=name, datum=describe(f), predicted=predicted(f, node_funs, op.alpha),
                 err=e1, err_r1e5=e2, verdict=v) for f, e1, e2, v in check_node(op, y, 0)]


P2_ALPHA2 = {
    "a2: f, Dx, Dy, Dxy": lambda p: [hb.value(p), hb.partial(p, (1, 0)), hb.partial(p, (0, 1)), hb.partial(p, (1, 1))],
    "a2: f, Dx, Dy, Dxx": lambda p: [hb.value(p), hb.partial(p, (1, 0)), hb.partial(p, (0, 1)), hb.partial(p, (2, 0))],
    "a2: f, Dx, Dy, Dxx-Dyy": lambda p: [hb.value(p), hb.partial(p, (1, 0)), hb.partial(p, (0, 1)),
                                        hb.Functional(tuple(p), ((1.0, (2, 0)), (-1.0, (0, 2))))],
    "a2: f, Dx, Dy, Laplacian": lambda p: [hb.value(p), hb.partial(p, (1, 0)), hb.partial(p, (0, 1)), hb.laplacian(p)],
}


def run_2d_alpha2(name):
    rng = np.random.default_rng(13)
    pts = rng.random((8, 2))
    node_funs = P2_ALPHA2[name](pts[0])
    fun = node_funs + [f for q in pts[1:] for f in hb.hermite_set(q, 2)]
    op = hb.SingularMLS(fun, degree=3, alpha=2, delta=1.0)
    y = rng.normal(size=len(fun))
    return [dict(dim=2, pattern=name, datum=describe(f), predicted=predicted(f, node_funs, op.alpha),
                 err=e1, err_r1e5=e2, verdict=v) for f, e1, e2, v in check_node(op, y, 0)]


def run_kl(same_points):
    """Komargodski-Levin Sec. 3 setting: values at some points, gradients at (other) points."""
    rng = np.random.default_rng(21)
    vpts = rng.random((8, 2))
    gpts = vpts if same_points else rng.random((8, 2))
    fun = [hb.value(p) for p in vpts] + [f for p in gpts for f in (hb.partial(p, (1, 0)), hb.partial(p, (0, 1)))]
    op = hb.SingularMLS(fun, degree=2, alpha=2, delta=1.0)
    y = rng.normal(size=len(fun))
    tag = "KL: values + gradients at the SAME points" if same_points else "KL: gradients at points WITHOUT values"
    out = []
    check = (0, 1, 2) if same_points else (0, 1, 8, 9, 10)     # nodes 8.. carry gradients only
    for node in check:
        node_funs = [f for f in fun if f.point == tuple(op.nodes[node])]
        for f, e1, e2, v in check_node(op, y, node):
            out.append(dict(dim=2, pattern=tag, datum=f"node {node}: {describe(f)}",
                            predicted=predicted(f, node_funs, op.alpha), err=e1, err_r1e5=e2, verdict=v))
    return out


def job(spec):
    kind, name = spec
    return {"1d": run_1d, "2d": run_2d, "2da2": run_2d_alpha2, "kl": run_kl}[kind](name)


if __name__ == "__main__":
    t0 = time.time()
    jobs = ([("1d", n) for n in P1] + [("2d", n) for n in P2] + [("2da2", n) for n in P2_ALPHA2]
            + [("kl", True), ("kl", False)])
    with Pool(2) as pool:
        out = pool.map(job, jobs)
    rows = [r for block in out for r in block]
    agree = 0
    for r in rows:
        r["predicted"] = "exact" if r["predicted"] else "not exact"
        ok = r["verdict"] == r["predicted"]
        agree += ok
        print(f"{r['dim']}D {r['pattern']:<42} {r['datum']:<22} err={r['err']:.1e}/{r['err_r1e5']:.1e}  "
              f"{r['verdict']:<10} predicted {r['predicted']:<10}{'' if ok else '   <-- MISMATCH'}")
    print(f"\nprediction agrees with observation for {agree}/{len(rows)} data; "
          f"unstable: {sum(r['verdict'] == 'UNSTABLE' for r in rows)};  {time.time() - t0:.0f}s")
    with open("../results/e2_birkhoff.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
