"""Exact (rational-arithmetic) counterexample to the separate-points claim of
Komargodski & Levin (2006, Sec. 3): values at x=0, 2; slopes at x=1, 3;
weights |x - z|^(-2); local basis {1, y}.  We compute u(x) in closed form and
its derivative at the slope nodes, and compare with the prescribed slopes."""
import sympy as sp

x, y = sp.symbols("x y", real=True)
vals = {0: 0, 2: 0}          # f(0) = 0, f(2) = 0
slopes = {1: 1, 3: 1}        # f'(1) = 1, f'(3) = 1
basis = [sp.Integer(1), y]
rows, data, wts = [], [], []
for z, v in vals.items():
    rows.append([b.subs(y, z) for b in basis]); data.append(v); wts.append(1 / (x - z) ** 2)
for z, s in slopes.items():
    rows.append([sp.diff(b, y).subs(y, z) for b in basis]); data.append(s); wts.append(1 / (x - z) ** 2)
P = sp.Matrix(rows); W = sp.diag(*wts); Y = sp.Matrix(data)
a = (P.T * W * P).LUsolve(P.T * W * Y)
u = sp.simplify(sum(c * b for c, b in zip(a, basis)).subs(y, x))
print("u(x) =", sp.factor(u))
for z, v in vals.items():
    print(f"u({z}) = {sp.limit(u, x, z)}   (prescribed {v})")
du = sp.diff(u, x)
for z, s in slopes.items():
    print(f"u'({z}) = {sp.nsimplify(sp.limit(du, x, z))}   (prescribed {s})")
