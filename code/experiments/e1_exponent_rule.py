"""E1 - exponent rule.  Every node carries classical Hermite data (all derivatives of
order <= k).  For each weight exponent alpha we record, per derivative order r, the
worst error |D^beta u(x_i) - y| over the checked nodes.

Run from code/:  python3 -m experiments.e1_exponent_rule [1d|2d|3d]
"""
import csv
import sys
import time
from multiprocessing import Pool

import numpy as np

import hbmls as hb
from experiments.common import check_node


def case(args):
    dim, k, alpha, seed = args
    rng = np.random.default_rng(seed)
    if dim == 1:
        pts = np.array([[0.0], [0.7], [1.5], [2.6]])
        m, delta, check = k + 2, 2.0, (0, 1)
    else:
        pts = rng.random(({2: 7, 3: 9}[dim], dim))
        m, delta, check = k + 1, 1.0, (0,)
    fun = [f for p in pts for f in hb.hermite_set(p, k)]
    op = hb.SingularMLS(fun, degree=m, alpha=alpha, delta=delta)
    y = rng.normal(size=len(fun))
    worst = {}
    for node in check:
        for f, got, want, err in check_node(op, y, node):
            worst[f.order] = max(worst.get(f.order, 0.0), err)
    return [dict(dim=dim, k=k, alpha=alpha, degree=m, order=r, max_err=e) for r, e in sorted(worst.items())]


CASES = {
    "1d": [(1, k, a, 10 * k + a) for k in range(1, 6) for a in (2, 4, 6, 8)],
    "2d": [(2, k, a, 3) for k in (1, 2, 3) for a in (2, 4, 6)],
    "3d": [(3, k, a, 5) for k in (1, 2) for a in (2, 4)],
}

if __name__ == "__main__":
    part = sys.argv[1]
    t0 = time.time()
    with Pool(2) as pool:
        out = pool.map(case, CASES[part])
    rows = [r for block in out for r in block]
    for block in out:
        b0 = block[0]
        print(f"{b0['dim']}D k={b0['k']} alpha={b0['alpha']}: " +
              "  ".join(f"r={r['order']}:{r['max_err']:.0e}" for r in block), flush=True)
    with open(f"../results/e1_{part}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"done in {time.time() - t0:.0f}s")
