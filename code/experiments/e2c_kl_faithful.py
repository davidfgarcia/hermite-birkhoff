"""E2c - the Komargodski-Levin setting exactly: pure power weights |x - x_i|^(-alpha),
alpha in {2, 4}, local space P_1 or P_2, values at 8 random points and gradients at 8
other random points.  Checks the values and the gradients at the first three gradient
points.

Run from code/:  python3 -m experiments.e2c_kl_faithful
"""
from multiprocessing import Pool

import numpy as np

import hbmls as hb
from experiments.common import check_node, describe


def run(args):
    m, alpha = args
    rng = np.random.default_rng(21)
    vpts, gpts = rng.random((8, 2)), rng.random((8, 2))
    fun = [hb.value(p) for p in vpts] + [f for p in gpts for f in (hb.partial(p, (1, 0)), hb.partial(p, (0, 1)))]
    y = rng.normal(size=len(fun))
    op = hb.SingularMLS(fun, degree=m, alpha=alpha)
    out = []
    for node in (0, 8, 9, 10):
        for f, e1, e2, v in check_node(op, y, node):
            out.append((m, alpha, node, describe(f), e1, e2, v))
    return out


if __name__ == "__main__":
    with Pool(2) as pool:
        res = [r for block in pool.map(run, [(1, 2), (1, 4), (2, 2), (2, 4)]) for r in block]
    for m, a, node, d, e1, e2, v in res:
        print(f"m={m} alpha={a} node {node:2d} {d:<3} err {e1:.1e} / {e2:.1e}  {v}")
    g = [r for r in res if r[2] >= 8]
    print(f"gradient data: {sum(r[6] == 'exact' for r in g)}/{len(g)} reproduced; "
          f"errors {min(r[4] for r in g):.2g} ... {max(r[4] for r in g):.2g}")
