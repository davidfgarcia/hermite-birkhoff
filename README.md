# Generalized Hermite–Birkhoff interpolation via singular-weight moving least squares

Research code, experiments and paper draft for **SWIMLS** (singular-weight interpolating
moving least squares) with derivative data of arbitrary order in any dimension.

- `code/hbmls/` – reference implementation (numerically stable float evaluator with the
  exact singular limit at the nodes, plus a high-precision mpmath reference); Gaussian or
  compactly supported C∞ ("bump") weights; `SWIMLS1D` gives exact u′ and u″ of the 1-D
  interpolant (projector form, leverage-aware cutoff), and `SWIMLS2D` the exact gradient and
  Hessian in 2-D (data u, ∂x, ∂y, Δ); both are the evaluators used by swimls-bvps
- `code/tests/` – correctness checks: `cd code && python3 -m tests.test_core` (or `PYTHONPATH=. python3 tests/test_core.py`)
- `code/experiments/` – the numerical experiments behind the paper (E1–E8); summary in `RESULTS.md`
- `results/`, `figures/` – their outputs
- `paper/` – the LaTeX source and PDF of the draft (`main.tex`, `main.pdf`); it opens in LyX via
  File → Import → LaTeX

Requirements: Python ≥ 3.10, numpy, scipy (for `SWIMLS2D`), mpmath, matplotlib, sympy.

## Results cited by paper 2 (keep stable)

Paper 2 (`swimls-bvps`) cites these results of paper 1. Their LaTeX labels must not change; if
their numbers change, record the old and new numbers here.

| Result | Label | Number |
|---|---|---|
| Fischer adjointness lemma | `lem:fischer` | Lemma 3.1 |
| Exactness rule (sufficient condition), conditions (i)–(ii) | `cor:exact`, `c1`, `c2` | Corollary 4.5 (i)–(ii) |
| Hermite data | `cor:hermite` | Corollary 4.6 |
| Convergence theorem | `thm:convergence` | Theorem 6.1 |

Renumbering log: none since the arXiv package of Oct 28. (Before that, Proposition 4.6 in
swimls-bvps notes is now Proposition 4.9, `prop:necessity`.)

arXiv number of paper 1: *not yet posted* (to be filled in here when it is).

## License

The software in this repository is licensed under the
[PolyForm Noncommercial License 1.0.0](LICENSE) (SPDX: `PolyForm-Noncommercial-1.0.0`).

- **Non-commercial use is free, with no permission needed:** personal study, research,
  experiment and teaching, and use by educational institutions, public research
  organizations, charities and government bodies. If you use it, a short note to
  david@figueiras-garcia.net is appreciated; this is a courtesy request, not a condition of
  the license.
- **Commercial use requires a separate commercial license**, which may involve fees. Contact
  david@figueiras-garcia.net.

The paper sources (`paper/`) are not covered by this license: © 2026 David Figueiras-Garcia,
all rights reserved.

## AI assistance

Developed with the assistance of Claude (Anthropic); see the declaration at the end of the paper (`paper/main.tex`).

## Action items

Open work on paper 1 (this repo: the interpolation method on its own). Paper 2, the application
to boundary value problems, lives in `swimls-bvps` and will cite this one, so paper 1 goes to arXiv
(and ideally to a journal) first. This is the single list for paper 1; it stays here until every
item is closed. Closed items are marked "Done <date>".

### Release sequence (decided by David, Oct 29) — keep this order

Step 1 done Oct 29, 2026: steps 2–4 may now go ahead ("patent pending").

1. [x] **Patent:** US provisional application **64/187,586**, filed **Oct 29, 2026** (receipt
   11:34 PM ET; micro entity, fee 3005, $65 paid). Specification: front
   section + implementation appendix, paper 1 (App. A), paper 2 draft (App. B), 67 pages;
   SHA-512 digests on the receipt match the files sent. Receipts kept by David.
   Inventor and applicant: David Figueiras Garcia.
   - **Deadline: Fri Oct 29, 2027** for the US non-provisional (and PCT, if wanted) claiming
     this date; otherwise the provisional lapses. Patent attorney for that step.
   - Material added to paper 2 after Oct 29, 2026 is not covered; if substantial, file another
     provisional before publishing it.
   - The correspondence address on file is temporary: when David moves, file form AIA/122
     in Patent Center.
   - The official filing receipt arrives in Patent Center in a few weeks: check names and title.
2. [ ] **Levin email** (`hermite-birkhoff/paper/levin_email.md`): send on or after the filing day.
3. [ ] **Paper 1 to arXiv:** upload `paper/arxiv/hermite-birkhoff-arxiv.tar.gz`; affiliation
   "Independent researcher"; arXiv's default non-exclusive license. Then make
   `hermite-birkhoff` public and add "Patent pending" to the README License section.
4. [ ] **Zenodo:** drafts and reserved DOIs for `swimls-bvps` and `quadrature-bvps` may be
   prepared now (license: PolyForm Noncommercial 1.0.0), but publish them, and make those
   repos public, only when paper 2 is ready to submit. Put the reserved DOIs in paper 2's
   bibliography.

#### Patent prior art: first pass (Oct 29, keyword search on Google Patents)

Not a professional search: keyword queries in English and Chinese on Google Patents. To be
redone by a patent attorney (classification search, e.g. CPC G06F17/13, G06F17/17, G06F30/23)
before the non-provisional. **Nothing found claims MLS with singular (interpolating) weights and
Hermite–Birkhoff data, or collocation with nodal derivative unknowns on such a trial space.**
Nearest documents:
- US6718291B1, Shapiro & Tsukanov (Intact Solutions), priority 1999: mesh-free solution
  structures with implicit (R-function) geometry; boundary conditions incl. Neumann interpolated
  by inverse-distance-type functions. Expired (2022). Relevant to the R-function future-work item.
- US7499050B2 / US20060103648A1, Livermore Software (now Ansys), priority 2004: mesh-free
  (MLS, Galerkin-type) analysis of 3-D shells. No derivative nodal data. Expired.
- US20070165025A1 / US20080246767A1, IBM, priority 2006: re-meshing via implicit MLS fields
  that interpolate the surface and match normals (gradient data). Expired (adjusted 2026-07-31).
- CN115803744A, Oxford University Innovation, priority 2020, pending: hybrid grid/meshfree
  "algebraic volume" finite-volume method; flux derivatives at nodes from weighted least squares.
  Finite-volume based, no SWIMLS, no collocation with Hermite data.
- CN102831280A: MLS-based meshless deformation simulation (graphics). CN101944144A: meshless
  cloth simulation. CN108520146A, CN113779802A: element-free Galerkin variants. Not close.
- Searched without relevant hits: "interpolating/Hermite moving least squares", MLS +
  collocation + Gauss points / orthogonal spline collocation, RBF collocation with Hermite data,
  scattered-data interpolation with gradients and compact-support weights.
Papers: the "cited by" check is done (Oct 29); see Literature and references. Closest paper:
Alexa–Adamson 2009 (singular-weight point set surfaces with Hermite/normal data).

### Paper content

- [x] **Harmonic stencils (optional).** Done Oct 28: Example 4.8. Toan et al., Thm 4.1, build regular schemes from harmonic
  data only (node k carries every harmonic symbol of degree ≤ d − 2k + 2, k = 1..⌊d/2⌋ + 1). With
  α = 2 SWIMLS reproduces all of them: a remark or example next to the "harmonic node" of E1.
- [x] **Data defined by an operator equation (optional).** Done Oct 28: one paragraph of outlook at
  the end of the conclusions. A datum whose value comes from a
  differential equation (e.g. y″ or Δu at a node) is just another Hermite–Birkhoff datum, and the
  exactness rule says when it is reproduced (α = 4 for y″ in 1-D and for Δu in d-D, since
  Δ|ξ|² ≠ 0). At most one sentence of outlook; the BVP work belongs to paper 2.
- [x] **Full independent review of the whole draft.** Done Oct 28: `paper/REVIEW2.md` (minor
  revision, no false theorem). All must-fix items applied: frozen-minimiser theorem generalised
  to every datum with Δ^(α/2)q = 0 (new frozen form of the error representation), dichotomy
  proposition and precise conjecture (covering the four counterexample families of `REVIEW.md`),
  harmonic-data example made non-vacuous, Shepard comparison corrected, counts and wording.
- [x] **Explain the extra hypothesis of Proposition 4.6** (now Proposition 4.9, necessity). Done Oct 28: Remark 4.10 after the proof
  (Schur complement; the d = 1, m = 3 example). (`prop:necessity`, "the data at the nodes
  other than x_i are unisolvent on Π_m by themselves"). From the swimls-bvps numerics session's
  resolution check of `REVIEW.md` (item G3, only partly resolved): the proof uses the
  hypothesis only to make the Schur complement S = K_RR − K_RN K_NN⁻¹ K_NR (blocks of K with
  respect to Π_m = R ⊕ N_i; K_NN is invertible by (A3) alone) injective. It cannot simply be
  dropped: with d = 1, m = 3, Hermite {f, f′, f″} at 0 and a single value f(1), the data are
  unisolvent and dim Π_3 in number, so u is the cubic interpolant and f″(0) is reproduced
  although Δξ² = 2 ≠ 0 at α = 2 (here S = 0). Suggested: a short remark after the proof saying
  both things. Full report available from the numerics session on request.
- [x] **Conjecture: what to do.** Done Oct 28 (decision with David): keep it as a clearly labelled
  open problem, do not hold the paper for a full proof. Added Proposition 4.18 (two nodes,
  deg q < 2α, every node position), so the conjecture is proved for deg q = α and for two nodes
  with deg q < 2α; open: three or more nodes, and deg q ≥ 2α. A general proof could be a short
  follow-up note.
- [x] **Declaration of generative AI use** before the references (Elsevier-style wording). Done
  Oct 28.
- [x] **Notation clean-up (main clashes).** Done Oct 28: operator K → Θ in Prop 4.9; degree n → μ
  in Remark 3.2 and Example 4.8; coordinates z = (z₁, z₂) and a, b, c in Prop 5.2; Appendix A
  rows A_h, A_l, bases U₁, U₂, scale θ. Remaining minor overloads (s, r, ℓ, h) left as they are.
- [ ] **Practical motivation (to revisit with David).** A referee may ask what derivative data are
  good for, since in our own test (§7.4) values alone were more accurate *per number of data*,
  while derivative data reduced the error *on a fixed node set*. The question to settle: in which
  situations is one node with value and derivatives cheaper than several nodes with values only
  (e.g. adjoint or automatic-differentiation gradients, measured slopes, physical constraints),
  and should the paper show one such example.
- [ ] **Optional polish:** a table for the rates in §7.4; a comparison with a competing
  Hermite–Birkhoff method.

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
- [x] **Cite Toan et al. (JAT 2026) and Calderón–Zygmund (1964)** next to the Fischer adjointness
  lemma and the condition-(ii) discussion. Done Oct 28: Remark 3.2 states the orthogonal
  decomposition Π_n = ker Δ^ℓ ⊕ |z|^(2ℓ) Π_(n−2ℓ) and cites both; the condition-(ii) discussion
  refers to it.
- [x] **"Cited by" check** for Komargodski–Levin (2006) and Levin (1998). Done Oct 29 through the
  built-in browser on David's Mac: Google Scholar blocks it ("unusual traffic"), so OpenAlex and
  Semantic Scholar were used instead. Komargodski–Levin: 19 citing works in OpenAlex, 22 in
  Semantic Scholar (Scholar says 33; the rest are probably theses and non-indexed venues). Levin
  (1998), 811 citing works: filtered in OpenAlex by title/abstract and full text for hermite,
  birkhoff, "derivative data", "singular weight", "interpolating moving least", collocation
  (60 distinct hits, all read). **Nothing does SWIMLS with general Hermite–Birkhoff data, its
  exactness theory, or collocation on it.** Alexa–Adamson, Tokmakov et al. and Kraaijpoel–van
  Leeuwen are now cited in paper 1's introduction (Oct 29); the rest are candidates:
  - *Paper 1, closest:* M. Alexa, A. Adamson, "Interpolatory point set surfaces — convexity and
    Hermite data", ACM Trans. Graph. 28(2) (2009), doi:10.1145/1516522.1516531. Point set
    surfaces interpolate with singular weights; they report that the extension to Hermite
    (normal) data is numerically unstable and propose a "Hermite point set surface" instead.
    Position: implicit surfaces with normals only; ours is polynomial MLS with arbitrary
    differential data, exactness conditions and a stable evaluator.
  - *Paper 1:* I. V. Tokmakov, A. F. Wagner, M. Minkoff, "Gradient incorporation in
    one-dimensional applications of interpolating moving least-squares methods for fitting
    potential energy surfaces", Theor. Chem. Acc. (2007), doi:10.1007/s00214-007-0358-7 (IMLS with
    gradient data, chemistry; check its weight: the chemistry IMLS usually regularises the
    singularity). Context: the Dawes–Thompson IMLS line (J. Chem. Phys. 2007, 2008).
  - *Paper 1:* D. Kraaijpoel, T. van Leeuwen, "Raising the order of multivariate approximation
    schemes using supplementary derivative data", Procedia Comput. Sci. 1 (2010),
    doi:10.1016/j.procs.2010.04.034 (generic order raising with derivative data, applies to MLS).
  - *Paper 1/2:* G. C. Bridhani, U. Saravanan, Hermite-type MLS bases with six dof per node,
    PARTICLES 2023 (doi:10.23967/c.particles.2023.001) and Acta Mech. (2026,
    doi:10.1007/s00707-026-04746-x): Hermite-type MLS as a meshfree (Galerkin) basis.
  - *Paper 2:* N. Trask, M. Maxey, X. Hu, "Compact moving least squares: an optimization framework
    for generating high-order compact meshless discretizations", J. Comput. Phys. (2016),
    doi:10.1016/j.jcp.2016.08.045 (compact meshless stencils; compare with the embedded variant).
  - *Paper 2:* M. Shirzadi, M. Dehghan, A. F. Bastani, "Optimal uniform error estimates for moving
    least-squares collocation…", Numer. Methods PDE (2020), doi:10.1002/num.22520 (MLS point
    collocation error analysis).
  - Also seen, lower priority: Ghorbani–Garshasbi 2021 (already noted), Ritt–Harker–O'Leary 2019
    (value + derivative fusion, polynomials), Gambaruto 2026 (meshfree local polynomial
    regression solver), Wang–Hao–Cheng 2014 (interpolating EFG for two-point BVPs), Schaback 2025
    (greedy local recovery), Bayona 2019 (MLS vs RBF+poly derivatives).
- [x] **HAL hal-01007945**, "Blended Hermite MLS Approximation for Discretizing Biharmonic PDEs".
  Done Oct 28: Joyot, Chinesta, Villon, Khoshnoudirad, WCCM-VIII (Venice, 2008), conference
  abstract; Hermite MLS for fourth-order PDEs in structural mechanics. Tangential, no overlap.
- [x] **Mirzaei (2015)**, J. Comput. Appl. Math. Done Oct 28: vol. 282, pp. 237–250 (from two
  citing papers); the DOI could not be confirmed and is left out.
- [x] **Shepard (1968)**. Done Oct 28: DOI 10.1145/800186.810616 matches the ACM DL record
  (pages 517–524 from secondary sources).
- [x] **Coman (1998)** and the **Dell'Accio–Di Tommaso survey**. Done Oct 28: Calcolo 35(4)
  197–203, doi:10.1007/s100920050016 (Springer); DRNA 9 (2016) Special Issue, 32–44,
  doi:10.14658/pupj-drna-2016-Special_Issue-5. Toan et al. and Calderón–Zygmund verified too.

### Code

- [x] **Derivative evaluator from swimls-bvps.** Done Oct 28: `hbmls.SWIMLS1D` (same API as
  `swimls_bvp/swimls1d.py`, plus `weight="bump"`), with tests against `SingularMLS`, polynomial
  reproduction and high-precision derivatives. swimls-bvps can now import it from the submodule
  and drop its own copy.
- [x] **Tag a release** of hermite-birkhoff; swimls-bvps pins its submodule to that tag. Done
  Oct 28: tag `v0.1.0` (commit 9c75392) is on GitHub.
- [x] **2-D evaluator from swimls-bvps** (request from paper 2, Oct 28). Decision: it belongs in
  `hbmls` (the 2-D companion of `SWIMLS1D`, one copy for both projects). Done Oct 28:
  `hbmls.SWIMLS2D`, same API as `swimls_bvp/swimls2d.py` plus `weight="bump"` (agrees with that
  file to rounding for Gaussian weights); 12 new checks (agreement with `SingularMLS`, P_m
  reproduction with gradient and Hessian, high-precision derivatives, both weights, data
  {u, ∂x, ∂y} at α = 2 and {u, ∂x, ∂y, Δ} at α = 4). New dependency: scipy.
- [ ] **Tag `v0.2.0`** (contains `SWIMLS2D`), so swimls-bvps can move its pin from v0.1.0 and drop
  its own copy. The tag is created on David's Mac clone; David pushes it
  (`git push origin v0.2.0`), since this cloud session cannot push tags.

### Before and after submission

- [ ] **Venue** (David's decision). Recommendation: J. Approx. Theory first (approximation theory
  with numerical support; Toan et al. 2026 appeared there), then Adv. Comput. Math. Constructive
  Approximation would want more theory (a proof of Conjecture 4.19).
- [ ] **arXiv** (David submits; needs his arXiv account), then journal submission, before paper 2.
  When posted, write the arXiv number in "Results cited by paper 2" above, for paper 2 to cite.
  The source package is built by `bash paper/make_arxiv.sh` → `paper/arxiv/hermite-birkhoff-arxiv.tar.gz`
  (flat figures, verification comments stripped, compiles with pdflatex). Suggested categories:
  math.NA (primary), math.CA. Before posting, replace "Draft of \today" handling if a fixed date
  is wanted, and decide whether the paper will use the target journal's class.

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

  Attach the PDF, keep it short, and invite comments before submission. Draft: `paper/levin_email.md`
  (confirm his current address first).
