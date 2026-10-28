# Generalized Hermite–Birkhoff interpolation via singular-weight moving least squares

Research code, experiments and paper draft for **SWIMLS** (singular-weight interpolating
moving least squares) with derivative data of arbitrary order in any dimension.

- `code/hbmls/` – reference implementation (numerically stable float evaluator with the
  exact singular limit at the nodes, plus a high-precision mpmath reference); Gaussian or
  compactly supported C∞ ("bump") weights; `SWIMLS1D` gives exact u′ and u″ of the 1-D
  interpolant (projector form, leverage-aware cutoff), the evaluator used by swimls-bvps
- `code/tests/` – correctness checks: `cd code && python3 -m tests.test_core`
- `code/experiments/` – the numerical experiments behind the paper (E1–E8); summary in `RESULTS.md`
- `results/`, `figures/` – their outputs
- `paper/` – the LaTeX source and PDF of the draft (`main.tex`, `main.pdf`); it opens in LyX via
  File → Import → LaTeX

Requirements: Python ≥ 3.10, numpy, mpmath, matplotlib, sympy.

## Action items

Open work on paper 1 (this repo: the interpolation method on its own). Paper 2, the application
to boundary value problems, lives in `swimls-bvps` and will cite this one, so paper 1 goes to arXiv
(and ideally to a journal) first. This is the single list for paper 1; it stays here until every
item is closed. Closed items are marked "Done <date>".

### Paper content

- [ ] **Harmonic stencils (optional).** Toan et al., Thm 4.1, build regular schemes from harmonic
  data only (node k carries every harmonic symbol of degree ≤ d − 2k + 2, k = 1..⌊d/2⌋ + 1). With
  α = 2 SWIMLS reproduces all of them: a remark or example next to the "harmonic node" of E1.
- [ ] **Data defined by an operator equation (optional).** A datum whose value comes from a
  differential equation (e.g. y″ or Δu at a node) is just another Hermite–Birkhoff datum, and the
  exactness rule says when it is reproduced (α = 4 for y″ in 1-D and for Δu in d-D, since
  Δ|ξ|² ≠ 0). At most one sentence of outlook; the BVP work belongs to paper 2.
- [ ] **Full independent review of the whole draft.** `paper/REVIEW.md` covered only Sections 3–5,
  when Sections 1, 2 and 6–8 were placeholders. Re-check that the revised conjecture addresses
  all four kinds of counterexample in that review.

### Literature and references

- [x] **Systematic literature search** on Hermite-type and interpolating MLS since 2006.
  Done Oct 28: nothing found that pre-empts the results. The closest prior work is Shepard-type
  Hermite–Birkhoff interpolation (Farwig 1986, Coman 1998, Allasia–Bracco 2012,
  Dell'Accio–Di Tommaso 2016), now cited as related work.
- [x] **J. Approx. Theory 2026**, "Multivariate Hermite interpolation with interpolation conditions
  induced by harmonic polynomials" (Toan, Mai, Long, Phung; JAT 318, 106303). Done Oct 28 (read
  by the theory session of swimls-bvps; David supplied the PDF). It does not pre-empt anything.
  It is about the regularity of global polynomial interpolation: no weights, no MLS, no error
  analysis. It uses the same tool as our Lemma (Fischer adjointness), though, and it has the
  ℓ = 1 case of the characterization behind condition (ii).
  - Their Thm 2.4: P ∈ Π_d is divisible by ‖x − a‖² iff q(D)P(a) = 0 for every harmonic q of
    degree ≤ d. Its homogeneous form is Calderón–Zygmund, *Studia Math.* 24 (1964) 211–226,
    Ch. 2, Lemma 1. Through our Fischer adjointness it is one line, and it extends to every ℓ:
    Π_d = ker Δ^ℓ ⊕ |z|^(2ℓ)Π_(d−2ℓ), an orthogonal sum, because multiplication by |z|^(2ℓ) is
    the adjoint of Δ^ℓ. So the data that annihilate every multiple of |z|^α are exactly the
    (α/2)-polyharmonic symbols, i.e. condition (ii).
  - Their Thm 4.1 gives regular schemes with harmonic data only. Node k carries every harmonic
    symbol of degree ≤ d − 2k + 2, for k = 1..⌊d/2⌋ + 1, and the count equals dim Π_d. With
    α = 2 these are data that SWIMLS reproduces at every node. They are candidate unisolvent
    stencils for the "harmonic node" of E1 and for the 2-D work in swimls-bvps.
  - Their "Taylor type polynomial" is the harmonic projection under the same apolar (Fischer)
    inner product, and they note that this product is also called Bombieri.
- [ ] **Cite Toan et al. (JAT 2026) and Calderón–Zygmund (1964)** next to the Fischer adjointness
  lemma and the condition-(ii) discussion. One option is to state the decomposition above as a
  remark: it explains why (ii) is the natural condition. *(Paper text: whoever owns
  `paper/main.tex`.)*
- [ ] **"Cited by" check** in Google Scholar or Scopus for Komargodski–Levin (2006) and Levin (1998).
  The citation databases were unreachable during the search.
- [ ] **HAL hal-01007945**, "Blended Hermite MLS Approximation for Discretizing Biharmonic PDEs":
  take a look (blocked during the search).
- [ ] **Mirzaei (2015)**, J. Comput. Appl. Math.: confirm volume, pages and DOI.
- [ ] **Shepard (1968)**: confirm the DOI against the ACM page.
- [ ] **Coman (1998)**, Calcolo 35, and the Dell'Accio–Di Tommaso survey (Dolomites Res. Notes
  Approx. 9, 2016): confirm the bibliographic data against the journals.

### Code

- [x] **Derivative evaluator from swimls-bvps.** Done Oct 28: `hbmls.SWIMLS1D` (same API as
  `swimls_bvp/swimls1d.py`, plus `weight="bump"`), with tests against `SingularMLS`, polynomial
  reproduction and high-precision derivatives. swimls-bvps can now import it from the submodule
  and drop its own copy.
- [ ] **Tag a release** of hermite-birkhoff; swimls-bvps pins its submodule to that tag.

### Before and after submission

- [ ] **Venue.** J. Approx. Theory, Constructive Approximation or Adv. Comput. Math.
- [ ] **arXiv**, then journal submission (before paper 2).

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
