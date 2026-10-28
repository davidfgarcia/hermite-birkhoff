"""Correctness checks for hbmls.  Run from code/:   python3 -m tests.test_core"""
from math import comb, factorial as F

import numpy as np

import hbmls as hb
from hbmls.core import multi_indices

rng = np.random.default_rng(0)
results = []


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)
    results.append(bool(cond))


# 1. dimension of the polynomial space
for d, m in [(1, 4), (2, 3), (3, 2)]:
    ok(len(multi_indices(d, m)) == comb(m + d, d), f"dim P_{m}(R^{d}) = {comb(m + d, d)}")

# 2. polynomial reproduction with mixed Hermite-Birkhoff data in 2D
m = 3
coef = {tuple(e): rng.normal() for e in multi_indices(2, m)}


def dpoly(beta, p):
    s = 0.0
    for e, c in coef.items():
        if e[0] >= beta[0] and e[1] >= beta[1]:
            s += (c * F(e[0]) / F(e[0] - beta[0]) * F(e[1]) / F(e[1] - beta[1])
                  * p[0] ** (e[0] - beta[0]) * p[1] ** (e[1] - beta[1]))
    return s


pts = rng.random((12, 2))
fun = []
for i, p in enumerate(pts):
    fun += [hb.value(p)]
    if i % 3 == 0:
        fun += [hb.partial(p, (1, 0)), hb.partial(p, (0, 1))]
    if i % 4 == 1:
        fun += [hb.laplacian(p)]
    if i % 5 == 2:
        fun += [hb.directional(p, (0.6, 0.8))]
for alpha in (2, 4):
    op = hb.SingularMLS(fun, degree=m, alpha=alpha, delta=0.5)
    y = op.data_from(dpoly)
    X = rng.random((40, 2))
    err = np.max(np.abs(op(X, y) - np.array([dpoly((0, 0), x) for x in X])))
    ok(err < 1e-10, f"P_3 reproduced off the nodes, alpha={alpha}: err={err:.1e}")
    errn = np.max(np.abs(op(pts, y) - np.array([dpoly((0, 0), x) for x in pts])))
    ok(errn < 1e-10, f"P_3 reproduced at the nodes, alpha={alpha}: err={errn:.1e}")

# 3. exact interpolation of values at the nodes for random data
y = rng.normal(size=len(fun))
op = hb.SingularMLS(fun, degree=2, alpha=2, delta=0.5)
vals = [i for i, f in enumerate(fun) if len(f.terms) == 1 and f.terms[0][1] == (0, 0)]
got = op([fun[i].point for i in vals], y)
ok(np.max(np.abs(got - y[vals])) < 1e-12, "values interpolated exactly at the nodes (closed-form limit)")

# 4. continuity: approaching a node, |u - y| = O(t)
i0 = vals[0]
p0 = np.array(fun[i0].point)
gaps = [abs(op([p0 + t * np.array([0.3, -0.7])], y)[0] - y[i0]) for t in (1e-3, 1e-5, 1e-7, 1e-9)]
ratios = [gaps[j] / gaps[j + 1] for j in range(3)]
ok(all(95 < q < 105 for q in ratios[1:]), f"continuity at a node, |u(x0+tv)-y| = O(t): ratios {np.round(ratios, 1)}")

# 5. shape functions
x = np.array([0.41, 0.37])
phi = op.shape_functions(x)
ok(abs(phi @ y - op([x], y)[0]) < 1e-12, "u(x) = phi(x) . y")
yc = np.array([1.0 if (len(f.terms) == 1 and f.terms[0][1] == (0, 0)) else 0.0 for f in fun])
ok(abs(phi @ yc - 1) < 1e-12, "constants reproduced: value shape functions sum to 1")

# 6. float and high-precision evaluators agree off the nodes
mpop = hb.MPSingularMLS(op, dps=40)
ok(abs(float(mpop.value(x, y)) - op([x], y)[0]) < 1e-11, "float vs mpmath evaluators agree")

# 7. the derivative stencils are exact on the reproduced polynomial (to 1e-14)
op3 = hb.SingularMLS(fun, degree=3, alpha=4, delta=0.5)
y3 = op3.data_from(dpoly)
mp3 = hb.MPSingularMLS(op3, dps=60)
errs = [abs(float(mp3.derivative(pts[0], b, y3)) - dpoly(b, pts[0])) for b in [(0, 0), (1, 0), (1, 1), (0, 2), (3, 0)]]
ok(max(errs) < 1e-10, f"stencil derivatives exact on P_3 at a node (data rounded to double): max err {max(errs):.1e}")

# 8. near a node, with gapped (Birkhoff) data, the float evaluator keeps the other
#    nodes in play and matches 100-digit arithmetic at every distance
others = [0.7, 1.5, 2.6]
for pat in [(1,), (0, 2)]:
    fb = [hb.derivative_1d(0.0, r) for r in pat] + [f for q in others for f in hb.hermite_set([q], 3)]
    opb = hb.SingularMLS(fb, degree=5, alpha=8, delta=2.0)
    yb = np.random.default_rng(7).normal(size=len(fb))
    mpb = hb.MPSingularMLS(opb, dps=100)
    e = max(abs(float(mpb.value([t], yb)) - opb([[t]], yb)[0]) for t in (1e-2, 1e-4, 1e-8, 1e-12))
    ok(e < 1e-13, f"float = 100-digit reference near a gapped node {pat}, t in [1e-12, 1e-2]: err {e:.1e}")

# 9. compactly supported (bump) weights: polynomial reproduction, interpolation, and
#    agreement with 100-digit arithmetic near a node
xs = np.linspace(0.0, 1.0, 9)
fz = [f for q in xs for f in hb.hermite_set([q], 1)]
opz = hb.SingularMLS(fz, degree=3, alpha=2, delta=0.4, length=0.4, weight="bump")
cub = lambda b, p: [p[0] ** 3 - p[0], 3 * p[0] ** 2 - 1][b[0]]
yz = opz.data_from(cub)
Xz = np.linspace(0.01, 0.99, 37)
ok(max(abs(opz([[t]], yz)[0] - cub((0,), [t])) for t in Xz) < 1e-12, "bump weights: P_3 reproduced")
yr = np.random.default_rng(3).normal(size=len(fz))
ok(abs(opz([[0.5]], yr)[0] - yr[2 * 4]) < 1e-13, "bump weights: node value interpolated")
mpz = hb.MPSingularMLS(opz, dps=100)
e = max(abs(float(mpz.value([0.5 + t], yr)) - opz([[0.5 + t]], yr)[0]) for t in (0.3, 1e-2, 1e-6, 1e-12))
ok(e < 1e-13, f"bump weights: float = 100-digit reference near a node: err {e:.1e}")

print(f"\n{sum(results)}/{len(results)} checks passed")
