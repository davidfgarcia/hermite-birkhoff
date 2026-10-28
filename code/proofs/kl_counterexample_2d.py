"""Exact 2-D counterexample to Komargodski-Levin (2006, Sec. 3): values at three points,
a gradient at a fourth point (no value there), weights |x - z|^(-2) (their alpha = 2,
rho = 1), local polynomials a + b*y1 + c*y2.  Computed in exact rational arithmetic."""
import sympy as sp

x1, x2, h = sp.symbols("x1 x2 h", real=True)
Y1, Y2 = sp.symbols("Y1 Y2")
basis = [sp.Integer(1), Y1, Y2]
vals = {(0, 0): 0, (2, 0): 0, (0, 2): 0}            # f = 0 at three points
grad_pt, grad = (1, 1), (1, 1)                         # grad f(1,1) = (1, 1)
rows, data, wts = [], [], []
d2 = lambda p: (x1 - p[0]) ** 2 + (x2 - p[1]) ** 2
for p, v in vals.items():
    rows.append([b.subs({Y1: p[0], Y2: p[1]}) for b in basis]); data.append(v); wts.append(1 / d2(p))
for j, var in enumerate((Y1, Y2)):
    rows.append([sp.diff(b, var).subs({Y1: grad_pt[0], Y2: grad_pt[1]}) for b in basis])
    data.append(grad[j]); wts.append(1 / d2(grad_pt))
P = sp.Matrix(rows); W = sp.diag(*wts); Y = sp.Matrix(data)
coef = (P.T * W * P).LUsolve(P.T * W * Y)
u = sp.together(sp.simplify(coef[0] + coef[1] * x1 + coef[2] * x2))
print("u(x1,x2) =", sp.factor(u))
for p, v in vals.items():
    print(f"u{p} = {sp.simplify(u.subs({x1: p[0], x2: p[1]}))}  (prescribed {v})")
ux = sp.diff(u, x1); uy = sp.diff(u, x2)
gx = sp.limit(ux.subs({x1: 1 + h, x2: 1 + 2 * h}), h, 0)
gy = sp.limit(uy.subs({x1: 1 + h, x2: 1 + 2 * h}), h, 0)
print(f"grad u(1,1) = ({sp.nsimplify(gx)}, {sp.nsimplify(gy)})  (prescribed {grad})")
