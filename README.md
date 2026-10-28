# Generalized Hermite–Birkhoff interpolation via singular-weight moving least squares

Research code, experiments and paper draft for **SWIMLS** (singular-weight interpolating
moving least squares) with derivative data of arbitrary order in any dimension.

- `code/hbmls/` – reference implementation (numerically stable float evaluator with the
  exact singular limit at the nodes, plus a high-precision mpmath reference)
- `code/tests/` – correctness checks: `cd code && python3 -m tests.test_core`
- `code/experiments/` – the numerical experiments behind the paper (E1–E8); summary in `RESULTS.md`
- `results/`, `figures/` – their outputs
- `paper/` – the LaTeX source and PDF of the draft (`main.tex`, `main.pdf`); it opens in LyX via
  File → Import → LaTeX

Requirements: Python ≥ 3.10, numpy, mpmath, matplotlib, sympy.

## Action items

Open work on the paper. This section stays here until every item is closed.

### Literature and references

- [x] **Systematic literature search** on Hermite-type and interpolating MLS since 2006.
  Done Oct 28: nothing found that pre-empts the results. The closest prior work is Shepard-type
  Hermite–Birkhoff interpolation (Farwig 1986, Coman 1998, Allasia–Bracco 2012,
  Dell'Accio–Di Tommaso 2016), now cited as related work.
- [ ] **J. Approx. Theory 2026**, "Multivariate Hermite interpolation with interpolation conditions
  induced by harmonic polynomials" (ScienceDirect PII S0021904526000201): read it. Its
  harmonic/Laplacian angle may touch the exactness criterion (Δ^(α/2) q = 0). It was paywalled
  during the search.
- [ ] **"Cited by" check** in Google Scholar or Scopus for Komargodski–Levin (2006) and Levin (1998).
  The citation databases were unreachable during the search.
- [ ] **HAL hal-01007945**, "Blended Hermite MLS Approximation for Discretizing Biharmonic PDEs":
  take a look (blocked during the search).
- [ ] **Mirzaei (2015)**, J. Comput. Appl. Math.: confirm volume, pages and DOI.
- [ ] **Shepard (1968)**: confirm the DOI against the ACM page.
- [ ] **Coman (1998)**, Calcolo 35, and the Dell'Accio–Di Tommaso survey (Dolomites Res. Notes
  Approx. 9, 2016): confirm the bibliographic data against the journals.

### Before and after submission

- [ ] **Courtesy email to Prof. David Levin** (Tel Aviv University) once the draft is final. The
  paper corrects one of his claims and builds on two of his results, so he should hear it from us
  first:
  - it gives exact counterexamples to the statement in Komargodski–Levin (2006, §3) that
    derivatives prescribed at points without function values are interpolated, and proposes
    completion as the fix;
  - it proves, and for k ≥ 3 sharpens, their conjecture that α = 2k suffices for derivatives up
    to order k (α > k is enough, and sharp);
  - the convergence theorem extends his Theorem 5 in "The approximation power of moving
    least-squares" (Math. Comp. 1998), and Remark 4.2 credits his Backus–Gilbert smoothness
    argument.

  Attach the PDF, keep it short, and invite comments before submission.
