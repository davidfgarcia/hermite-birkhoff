# Generalized Hermite–Birkhoff interpolation via singular-weight moving least squares

Research code, experiments and paper draft for **SWIMLS** (singular-weight interpolating
moving least squares) with derivative data of arbitrary order in any dimension.

- `code/hbmls/` – reference implementation (numerically stable float evaluator with the
  exact singular limit at the nodes, plus a high-precision mpmath reference)
- `code/tests/` – correctness checks: `cd code && python3 -m tests.test_core`
- `code/experiments/` – the numerical experiments behind the paper (E1–E5)
- `results/`, `figures/` – their outputs
- `paper/` – the LaTeX source (opens in LyX via File → Import → LaTeX)

Requirements: Python ≥ 3.10, numpy, mpmath, matplotlib, sympy.
