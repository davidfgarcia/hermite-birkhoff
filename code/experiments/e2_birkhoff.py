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
from experiments.common import EXACT, check_node, describe, symbol
from hbmls.core import multi_indices


def predicted(f, node_funs):
    """True if every (iterated) derivative of f's symbol lies in span of node symbols."""
    dim = len(f.point)
    top = max(g.order for g in node_funs)
    exps = [tuple(e) for e in multi_indices(dim, top)]
    vec = lambda q: np.array([q.get(e, 0.0) for e in exps])
    S = np.array([vec(symbol(g)) for g in node_funs]).T
    frontier = [symbol(f)]
    while frontier:
        q = frontier.pop()
        for j in range(dim):
            dq = {}
            for e, c in q.items():
                if e[j] > 0:
                    e2 = list(e)
                    e2[j] -= 1
                    dq[tuple(e2)] = dq.get(tuple(e2), 0.0) + c * e[j]
            if not dq:
                continue
            v = vec(dq)
            if np.linalg.norm(v - S @ np.linalg.lstsq(S, v, rcond=None)[0]) > 1e-9 * np.linalg.norm(v):
                return False
            frontier.append(dq)
    return True


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
    return [dict(dim=1, pattern=name, datum=describe(f), predicted=predicted(f, node_funs),
                 err=err) for f, *_, err in check_node(op, y, 0)]


def run_2d(name):
    rng = np.random.default_rng(11)
    pts = rng.random((8, 2))
    fun = P2[name](pts[0]) + [f for q in pts[1:] for f in hb.hermite_set(q, 2)]
    op = hb.SingularMLS(fun, degree=3, alpha=6, delta=1.0)
    y = rng.normal(size=len(fun))
    node_funs = P2[name](pts[0])
    return [dict(dim=2, pattern=name, datum=describe(f), predicted=predicted(f, node_funs),
                 err=err) for f, *_, err in check_node(op, y, 0)]


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
        for f, *_, err in check_node(op, y, node):
            out.append(dict(dim=2, pattern=tag, datum=f"node {node}: {describe(f)}",
                            predicted=predicted(f, node_funs), err=err))
    return out


def job(spec):
    kind, name = spec
    return {"1d": run_1d, "2d": run_2d, "kl": run_kl}[kind](name)


if __name__ == "__main__":
    t0 = time.time()
    jobs = [("1d", n) for n in P1] + [("2d", n) for n in P2] + [("kl", True), ("kl", False)]
    with Pool(2) as pool:
        out = pool.map(job, jobs)
    rows = [r for block in out for r in block]
    agree = 0
    for r in rows:
        obs = r["err"] < EXACT
        agree += obs == r["predicted"]
        r["observed"] = obs
        print(f"{r['dim']}D {r['pattern']:<42} {r['datum']:<22} err={r['err']:.1e}  "
              f"reproduced={'yes' if obs else 'NO ':<3}  predicted={'yes' if r['predicted'] else 'NO'}"
              f"{'' if obs == r['predicted'] else '   <-- MISMATCH'}")
    print(f"\nprediction agrees with observation for {agree}/{len(rows)} data;  {time.time() - t0:.0f}s")
    with open("../results/e2_birkhoff.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
