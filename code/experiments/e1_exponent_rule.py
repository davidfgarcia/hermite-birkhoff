"""E1 - which data are reproduced exactly, PER DATUM, against the conjectured rule

    q(D)f(x_i) is reproduced  <=>  (i) every derivative of q lies in span Q_i
                                   (ii) Delta^(alpha/2) q = 0.

Cases: classical Hermite nodes (all D^beta, |beta| <= k) in 1D/2D/3D for several alpha,
and a node whose data are the harmonic polynomials of degree <= 3 (alpha = 2).

Run from code/:  python3 -m experiments.e1_exponent_rule
"""
import csv
import time
from multiprocessing import Pool

import numpy as np

import hbmls as hb
from experiments.common import check_node, describe, predicted


def harmonic_node(p):
    """Symbols 1, x, y, xy, x^2-y^2, x^3-3xy^2, 3x^2y-y^3: harmonic and D-invariant."""
    F = hb.Functional
    p = tuple(float(v) for v in p)
    return [F(p, ((1.0, (0, 0)),)), F(p, ((1.0, (1, 0)),)), F(p, ((1.0, (0, 1)),)),
            F(p, ((1.0, (1, 1)),)), F(p, ((1.0, (2, 0)), (-1.0, (0, 2)))),
            F(p, ((1.0, (3, 0)), (-3.0, (1, 2)))), F(p, ((3.0, (2, 1)), (-1.0, (0, 3))))]


def case(spec):
    name, dim, k, alpha, m, seed = spec
    rng = np.random.default_rng(seed)
    if dim == 1:
        pts = np.array([[0.0], [0.7], [1.5], [2.6], [3.4]])
    else:
        pts = rng.random(({2: 7, 3: 8}[dim], dim))
    if name == "harmonic node":
        first = harmonic_node(pts[0])
    else:
        first = hb.hermite_set(pts[0], k)
    fun = first + [f for p in pts[1:] for f in hb.hermite_set(p, k)]
    op = hb.SingularMLS(fun, degree=m, alpha=alpha, delta=2.0 if dim == 1 else 1.0)
    y = rng.normal(size=len(fun))
    rows = []
    for f, e1, e2, verdict in check_node(op, y, 0):
        rows.append(dict(case=name, dim=dim, k=k, alpha=alpha, degree=m, datum=describe(f),
                         order=f.order, err_r1e6=e1, err_r1e5=e2, verdict=verdict,
                         predicted="exact" if predicted(f, first, alpha) else "not exact"))
    return rows


CASES = [
    ("Hermite", 1, 7, 6, 9, 1),
    ("Hermite", 1, 5, 4, 7, 2),
    ("Hermite", 2, 2, 2, 3, 3),
    ("Hermite", 2, 3, 2, 4, 3),
    ("Hermite", 2, 3, 4, 4, 3),
    ("Hermite", 2, 4, 4, 5, 4),
    ("harmonic node", 2, 3, 2, 4, 5),
    ("Hermite", 3, 2, 2, 3, 6),
    ("Hermite", 3, 3, 2, 4, 6),
]

if __name__ == "__main__":
    t0 = time.time()
    with Pool(2) as pool:
        out = pool.map(case, CASES)
    rows = [r for block in out for r in block]
    agree = 0
    for r in rows:
        ok = r["verdict"] == r["predicted"]
        agree += ok
        print(f"{r['dim']}D {r['case']:<13} k={r['k']} alpha={r['alpha']} m={r['degree']}  {r['datum']:<16}"
              f" err {r['err_r1e6']:.1e} / {r['err_r1e5']:.1e}   {r['verdict']:<10} predicted {r['predicted']:<10}"
              f"{'' if ok else '  <-- MISMATCH'}")
    print(f"\nprediction agrees for {agree}/{len(rows)} data;  "
          f"unstable verdicts: {sum(r['verdict'] == 'UNSTABLE' for r in rows)};  {time.time() - t0:.0f}s")
    with open("../results/e1_per_datum.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
