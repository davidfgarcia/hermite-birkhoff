# Courtesy email to Prof. David Levin (draft)

Send once the draft is final, before the journal submission; attach the PDF.
Confirm his current address first (Tel Aviv University, School of Mathematical Sciences).

---

**Subject:** Hermite–Birkhoff interpolation by singular-weight MLS: a draft building on your work

Dear Professor Levin,

I am writing to share a draft paper, "Generalized Hermite–Birkhoff interpolation of arbitrary
order in any dimension via singular-weight moving least squares" (PDF attached), which builds
directly on your work, and I would value your comments before I submit it.

The paper studies interpolating moving least squares with singular weights |x − x_i|^(−α), α even,
when each node carries arbitrary linear differential data. Three points concern your papers:

1. Komargodski–Levin (2006), §3: the paper proves the conjecture that α = 2k suffices for data up to
   order k, and shows that α > k is enough and sharp, which improves it for k ≥ 3.
2. Komargodski–Levin (2006), §3: for directional derivatives prescribed at points that carry no
   function value, the paper gives exact counterexamples in one and two dimensions in which the
   derivative data are not interpolated, explains when they are, and proposes a completion of the
   data that restores exactness. I have tried to present this constructively, as a refinement of the
   statement rather than a criticism.
3. Your 1998 paper "The approximation power of moving least-squares": the convergence theorem
   extends your Theorem 5 to Hermite–Birkhoff data and to derivatives of the interpolant, with the
   same scaling and continuity argument, and the Backus–Gilbert formulation of your Propositions 1–2
   is credited for the smoothness of the interpolant.

If anything in the paper misrepresents your results, or if you know of related work I have missed,
I would be very grateful to hear about it. The code and experiments are available at
https://github.com/davidfgarcia/hermite-birkhoff.

With best regards,

David Figueiras Garcia
Seattle, Washington, USA
