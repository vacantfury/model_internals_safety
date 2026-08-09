# AS-6 — evidence and story

*Paper-level synthesis for AS-6, the guard-internals paper. Sibling of
`text_docs/as5/evidence_and_story.md`, founded 2026-08-09 when the title
settled and there was no home for a paper-level decision.*

**This file holds the STORY. It holds no numbers of its own.** The results of
record are `as6/phase1_map.md`; anything that is a property of the measurement
rather than of this paper is `shared/instrument_layer.md`. A number appearing
here first has no provenance — the same rule the LaTeX carries.

---

## 1. ✅ THE TITLE — settled 2026-08-09 (owner go)

> **Unread or Unenforced? Separating Representation from Enforcement Failure in
> Content Guards**

### The criterion that chose it

Inherited from AS-5, where the title survived two refutations of its own stated
rationale. The reason it survived is the criterion: **it names a state, not a
mechanism.** In this repo mechanisms are what keep getting refuted, so a title
pinned to one gets retired with it.

Applied here, the question is only: *what has never been overturned?*

**Overturned, each of which would have made a dead title:**

| claim | fate |
|---|---|
| the guards decode the cipher ladder and let it through | artefact — base-model ability 0.00 (`instrument_layer.md` §2.7, §2.7.1) |
| the cross-guard cipher dissociation | dissolved — the result is symmetric (§2.7.1) |
| `combining_marks` 0.69 is unreportable | reversed — it survives its floor (§2.7.1) |
| the matched null is bin-stable at 5/10/20 | false on WildGuard: 5/7/4 (`phase1_map.md` §0) |
| `caesar3` is "the one genuine cipher that survives" | withdrawn (§0.6) |

Every *specific cell* in this study has moved at least once, and several moved
by more than they were first reported to be worth.

**Never overturned:** the capability/policy decomposition, and the fact that
end-to-end ASR conflates the two. That is what the title names, and it is the
only layer of this paper stable enough to name.

### Why not the two candidates the skeleton carried

- **(a) "Never Decoded or Never Blocked?"** is *factually wrong*, which is worth
  recording because it read fine for a week. The guards block **65–92%** of
  every condition that survives the screens. The cell is *decoded but not
  blocked*; "never blocked" describes no condition in the study except the
  cipher band, where the decode axis is unmeasured and there is no finding.
- **(b) "What a Guard Missed and Why"** promises a mechanism ("why"), which is
  exactly the thing this paper cannot yet deliver: the measurement is
  correlational, the causal test is named and not run (`Scope`, `Limitations`).

### The alternative that was weighed and not taken

*"Significance Is Not Sufficiency: Controlling Internal Measurements of
Content-Guard Failure."* A real option, and the case for it is honest: the
**measurement contribution is the part that got STRONGER at every check**, while
the map shrank to four conditions per guard. Rejected on two grounds — it buries
the safety result under a methods framing, and *significance is not sufficiency*
is this repo's own phrase rather than an established one, so a title resting on
it has to earn it in the abstract before the reader reaches the claim.

**Revisit trigger, named rather than left implicit:** if the surviving map
shrinks further — if either guard drops below ~3 reportable conditions, or the
policy-failure cell stops being populated at a defensible operating point — then
the paper *is* a methods paper and the alternative title becomes correct. That
is a re-bet on evidence, not a re-opening on taste.

---

## 2. The legs, as the evidence currently supports them

Numbers live in `phase1_map.md`; this is the argument structure.

**Leg 1 — the decomposition.** A guard that passes an encoded attack failed in
one of two disjoint ways, and ASR reports one number for both. They have
opposite remedies: a representation failure is not reachable by more preference
data over verdicts, an enforcement failure is exactly what that data addresses.
No prior work separates them for a guard, and the closest prior studies the same
two capture positions in a *target* model.

**Leg 2 — the measurement, and it is the contribution that keeps strengthening.**
Licensing the decode read honestly costs 17→4 conditions on one guard and 12→4
on the other. The signature of a false positive is stated rather than left for a
reader to infer: **high apparent decode, block rate exactly zero** — which is
also the signature of the most publishable finding. Two screens sharing no input
(the ability-inherited control floor; bin-stability of the matched null) agree
exactly on WildGuard and disagree on Llama Guard, so the floor is the stronger
screen and the agreement is evidence rather than construction.

⚠️ **The asymmetry to state explicitly, because it is the same shape as AS-5 leg
3 with the sign flipped.** In AS-5 every defect touching the behaviour axis
inflated apparent *safety*. Here every defect touching the decode axis inflates
apparent *failure* — structurally, because the reported cell is
`read_rate × (1 − block_rate)` and artefacts concentrate where the block rate is
lowest, i.e. in the cells a paper most wants to report
(`instrument_layer.md` §2.8). **Both directions flatter whoever is reporting**,
and that is the general lesson neither paper can draw alone.

**Leg 3 — what survives.** A real but smaller policy-failure map on the
surface/comprehension band only, plus a clean negative: *blocked without
decoding* is near zero on all 38 guard–condition pairs, retiring the
format-detector account on this corpus. The cipher band is `(U)`, never "did not
decode".

---

## 3. Open

- **The causal test.** Everything here is presence, not use. Ablate the recovered
  direction and read the verdict — specified in `Limitations`, not run.
- **Two guards, one corpus.** The dissociations are properties of two
  checkpoints. §3.6.1's lesson from the AS-5 side applies: a conclusion from one
  model is not a conclusion, and two is the minimum that makes a dissociation
  one — not the number that makes either guard's behaviour general.
- **The citation gap.** The Llama Guard *3* checkpoint cannot be cited: the Herd
  paper is absent from science's master bib and the venue bib is a generated
  subset of it (TODO 63). The draft cites the family and names the checkpoint.
