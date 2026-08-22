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

- **The causal test — RAN 2026-08-09 (job `9033528`), and it does NOT move
  `Limitations`.** Still open, for a better-understood reason than before. The
  gate returned `n_eligible: 0` on Llama Guard, which read like leg-2's second
  branch (content present, causally unused). It cannot be read that way: the
  **same gate returns the same empty set on Llama-3.1-8B-Instruct while
  discarding a direction that removes ~78% of that model's refusal**
  (`instrument_layer.md` §6.3.2). An instrument returning its negative answer on
  a positive control has not measured anything, so **no causal sentence goes in
  the paper**, and the `Limitations` item stating the claim is correlational
  stands exactly as written. What the run did buy: the guard's own numbers
  (`behaviour_before` 0.974, bypass +0.023 over the matched-norm null at
  p = 0.048) are on disk and cost $0 to re-read once the gate is trusted, and the
  attrition instrumentation added the same day means the next run diagnoses
  itself rather than costing a fourth queue cycle.
- **✅ CLOSED 2026-08-21 — the map in `phase1_map.md` no longer contradicts the
  draft.** §0.6's table predated the operating-point change and carried no
  marker, so the doc of record disagreed with the paper it grounds; it now
  states both reads and points at §2 for the reported one. Same guard, same four rungs, same block rates
  (0.92 / 0.83 / 0.85 / 0.65) — but `reading_percentile` moved 50 → 75 this
  session, and `decoded_not_blocked` moved with it: homoglyph 0.08 → **0.07**,
  zero_width 0.17 → **0.17**, fullwidth 0.12 → **0.10**, reverse_words 0.25 →
  **0.08**. The change lands almost entirely on `reverse_words`, the weakest
  decode probe of the four (AUROC 0.796 against 0.985 / 0.969 / 0.880), which is
  the direction a stricter read should move things and is a check on the knob
  rather than a coincidence. **Leg 3's conclusions are unaffected** — the
  ordering is unchanged and `blocked_without_decoding` stays at 0.00–0.11 — but
  any quoted cell must now name which operating point produced it.
- **⚠️ THE CAUSAL ARM IS BLOCKED BY A DESIGN QUESTION, not a config number
  (measured 2026-08-21, $0).** The capture-spine fix that followed AS-5's
  `n_eligible: 0` derives the end-of-instruction span from the live template and
  said in terms that AS-6 must derive its guards' spans the same way. Done:
  **Llama Guard 3 is 55 tokens and WildGuard is 25**, against 5 for a chat
  model — because a guard's template puts the *classification task* after the
  payload rather than an assistant header, so its span is task text and sweeping
  it is not the experiment Arditi et al. run. `eoi_position_names` refuses both
  (it enumerates to `last_minus_6`), and the refusal is the correct state:
  extending it to 55 would make an ill-posed sweep expressible. **No published
  number moves** — `instruction_final` is `-(span + 1)`, so it still lands on the
  payload's final token on both guards, which is the right site for the content
  probe; the verdict read never consults a swept position. So `Limitations`'
  "specified but not run" stands, now for a better-understood reason.
  `instrument_layer.md` §6.3.5.
- **Two guards, one corpus.** The dissociations are properties of two
  checkpoints. §3.6.1's lesson from the AS-5 side applies: a conclusion from one
  model is not a conclusion, and two is the minimum that makes a dissociation
  one — not the number that makes either guard's behaviour general.
- **✅ CLOSED 2026-08-18 — the citation gap.** The Herd paper landed in science's
  `llm-security` master as `grattafiori2024llama3`, and the draft now cites the
  family *and* the version-3 8B checkpoint. The key was nearly minted twice; see
  §5.3.
- **✅ CLOSED 2026-08-18 — Related work is written**, 2 → 12 citations, and the
  scoop check that blocked it cleared: arXiv 2608.03201 classifies prompt–
  *response* pairs with zero encoded inputs, so the Level-3 floor stays SIREN
  alone and the paper became a citation that helps rather than a threat. A
  closer prior surfaced from its reference list and is now cited: Tasawong et
  al., LLMSEC 2025, prompt-side keyword bias in safeguards — the wrapper
  screen's published motivation. Full positioning of record in §5.

## 4. Corrected

- **✅ FIXED 2026-08-18 — `Limitations` withheld numbers `Results` was
  reporting, in both kits, for nine days.** Its first paragraph read *"One
  guard's decode map is screened; the other's is not, yet … We therefore
  withhold the unscreened guard's decode-axis numbers"* — written 2026-08-08,
  correct that day. Job `9031680` derived WildGuard's floor on 2026-08-09
  (`phase1_map.md` §0.6), the `Results` section was drafted the same day *from
  that job*, and Table 1 has reported WildGuard's decode-axis AUROCs and cells
  ever since. So the paper asserted the absence of its own table. **The comment
  standing directly above the paragraph had predicted the rewrite in terms** —
  *"if it lands, this paragraph shrinks to 'both guards screened, and here is
  what it cost'"* — which makes this the second instance in this estate of *a
  note predicting a defect is not a guard against it* (the first: the paper-kit
  parity defect, `CLAUDE.md`, 2026-08-12). Replaced with the limitation that
  actually survives, and which `Method` already sets up: **the control set is
  selected by a PROXY** — the guard's base model, because a format-locked
  classifier cannot be asked to restate a payload — so the residual assumption
  is that a safety fine-tune acquires no decoding its base lacks, and the
  direction of that error is stated (it inflates the floor, which can only
  *remove* conditions from the reported map, never add them: conservative for
  the policy failure we report). Enforcement question filed as TODO 72 rather
  than settled here: three candidate guards were weighed and none is clean, so
  the form is a design decision, not a mechanical follow-on.

---

## 5. Related work — the positioning of record

*Written 2026-08-21, after the scoop check TODO 73 named as its blocker. This
section is the SOURCE; the `Related Work` section in both LaTeX kits renders it.
A positioning claim appearing in the `.tex` first has no provenance, same rule
as a number.*

### 5.0 ✅ The scoop check ran — arXiv 2608.03201 is NOT a scoop, and it helps

**"When Refusal Looks Safe: The Refusal-Cue Shortcut in Safety Guard Models"**
(Yu Feng et al., Sydney + Alibaba, 2026-08-04, no venue, tier (C)). Filed
2026-08-05 as *"squarely on the AS-6 guard-internals line — scoop-check it
before AS-6 scopes further"* and open for sixteen days. Digest and bundle now
live in science: `literature/model-internals/notes/refusal-cue-shortcut-2608.03201.md`.

**Three separations, each checked against the full text rather than the
abstract** (the abstract alone would have supported the first two but not the
third):

| axis | 2608.03201 | AS-6 |
|---|---|---|
| what is classified | prompt–**response** pairs; the perturbation is in the response | **prompts** |
| what breaks the guard | a **training-data label imbalance** — refusal expressions co-occur almost only with unharmful labels | an **encoding** the guard may or may not decode |
| internals method | learned **sparse gates** over attention heads and MLP neurons (SafeSeek, 2603.23268) | **linear probes** on the residual stream, asking what is *represented* |

Zero encoded or obfuscated inputs anywhere in it; zero linear probing of
residual streams. The decode axis — *did the guard recover the payload at all* —
is untouched, which is the axis AS-6's Level-3 delta was always argued on. **The
scoop floor is unchanged and still set by SIREN alone.**

**It strengthens AS-6 in three specific ways, and all three belong in the
paper.** (a) It is independent evidence that **a guard's verdict can ride on a
surface cue rather than on content** — the exact worry that made us build the
wrapper arm, arrived at on a different cue, on the response side. (b) Its
contribution 4 — shortcut reliance is *partly separable* from legitimate refusal
recognition — is a **functional-separability result inside a guard**, the same
shape as represent-vs-enforce on a different axis; a neighbour to position
against, not a threat. (c) Its audit finds the imbalance **in WildGuardMix**,
which is one of our two guards' training data, so it is a citable caveat on our
WildGuard results specifically — and it compounds a caveat `model_slate.md` §2.3
already carries, that Tülu 3's safety mix contains 50k WildGuardMix prompts.

**The check also surfaced a prior we did not have, and it is closer to us than
2608.03201 is:** its reference [21], **Tasawong et al., LLMSEC 2025** (ACL
Anthology `2025.llmsec-1.14`) — safeguards leaning on **prompt-side** keywords
spuriously correlated with training labels rather than on input semantics,
degrading under keyword-distribution shift. That is the wrapper screen's
motivation, published, on our side of the prompt/response line. **Our delta over
it is a control-design one and 2608.03201 states it for us:** their keyword
manipulations use word-level associations that *may alter prompt semantics*,
whereas our wrapper arm holds the content fixed as plaintext and varies only the
wrapper — a clean factorial cell rather than a correlational shift.

### 5.1 The four paragraphs, and what each is for

1. **Guards and how they are evaluated.** Llama Guard `inan2023…`, Llama Guard 3
   `grattafiori2024llama3herd`, WildGuard `NEURIPS2024_0f69b4b9`. The framing
   move: prior evaluation reports an end-to-end block rate, which cannot say
   *why* a prompt got through.
2. **Encoded prompts against guards.** `wei2023jailbroken` (mismatched
   generalization) and DecipherGuard `yang2025decipherguard…` — the latter owns
   the behavioural result *guards fail on encoded prompts and decoding repairs
   them*, which is **never claimable here** and must be cited as the prior it is.
   Our delta is the decomposition, not the phenomenon.
3. **Guard internals.** SIREN `jiao2026siren` (the scoop floor — builds a
   detector from internal representations, never encoded inputs), Gamma-Guard
   `lv-etal-2025-gamma` (qualitative, zero encoding rungs, no decode-vs-flag
   distinction), and the two surface-cue papers above,
   `feng2026refusalcueshortcut` and `tasawong-etal-2025-shortcut`.
4. **Reading content out of activations.** `arditi2024refusal` (the estimator we
   ported) and `zhao2025llmsencode` (harmfulness at the instruction-final token
   vs refusal at the post-instruction token — the published interpretation of
   our two capture positions, and it must be attributed rather than re-derived).
   `youstra2025cifr` sits here too: ciphers plus probe monitors on internal
   activations, the nearest prior to the whole AS-5/AS-6 thesis.

### 5.2 ⚠️ Four of the seven entries were ALREADY in the corpus — under keys I would have duplicated

The intake sweep first reported seven missing bib entries. **Four already
existed** in science's `model-internals/references.bib`: `arditi2024refusal`,
`zhao2025llmsencode`, `youstra2025cifr`, `jiao2026siren`. The first check had
grepped only `llm-security/references.bib` — **the corpus has one master bib per
DIRECTION, not one master bib** — and adding the four under fresh keys would
have produced exactly the failure `paper/literature/README.md` names: a citation
key that drifts between two bibs and compiles cleanly while pointing somewhere
else. Third instance in this repo of *absence measured against a narrow index,
reported as absence in general*; the first two are recorded in `CLAUDE.md`
(the 494/119 coverage-sweep correction) and in the venue-tier sweep. **Genuinely
added: three** — `grattafiori2024llama3herd` and `tasawong-etal-2025-shortcut`
to `llm-security`, `feng2026refusalcueshortcut` to `model-internals`.

`zhao2025llmsencode` also turned out to already carry **NeurIPS 2025**, which
this session verified independently (OpenReview `zLkpt30ngy`, NeurIPS virtual
poster 115056) — and which shows 2608.03201's own citation of it, "Advances in
NeurIPS 38:140283–140318, **2026**", to be mis-yeared. TODO 63 closes with
`grattafiori2024llama3herd`: **the trap it predicted was real** — the Herd
paper's first author differs by arXiv version (v1 Dubey, v3 Grattafiori), so the
entry records both and cites the current one.

### 5.3 The venue bib is now GENERATED — and generating it found AS-5's bib is not

`paper/literature/README.md` has always said the venue bib is *"the CITED SUBSET
of science's `references.bib` … build output rather than a source of truth"*.
Nothing generated it, so both papers' bibs were hand-written. `scripts/build_venue_bib.py`
now does (keyless, no network, seconds; `uv run python scripts/build_venue_bib.py as-6`),
and it **fails loudly on an unresolved key** — a cited key with no entry renders
as a bare `?` and compiles, which is the failure a silent generator would ship.

Running it on AS-6 resolved 12 of 12. Pointing its check at **AS-5** returned
**eight keys that exist in no direction bib** — `souly2024strongreject`,
`jiang2023mistral`, `yuan2024cipherchat`, `rottger2024xstest`,
`grattafiori2024llama3`, `chao2024jailbreakbench`, `lambert2025tulu3`,
`qwen2024qwen25` — so AS-5's `paper.bib` is a second source of truth for eight
citations of an arXiv-published paper. Filed as **TODO 74**; it is not a
rename-and-go, since each entry needs its venue verified at the primary source
(the Tülu 3 COLM miss is the standing example).

**One of the eight collided with this session's own work, which is the part
worth keeping.** AS-5 already cites the Llama 3 Herd paper as
`grattafiori2024llama3`; TODO 63 was closed by minting `grattafiori2024llama3herd`
in the corpus — a **second key for one paper**, created by the very rule that
exists to prevent it, and invisible until the generator's cross-check ran. The
corpus entry was renamed onto the in-use key before anything was committed:
**keys are stable, and the published one wins over the one minutes old.**

Two guards, both mutation-verified: every key a kit cites must resolve in the
corpus, and regenerating a kit's bib must be a **no-op** (a one-character edit to
a generated entry fails it). Coverage is scoped by the generator's own header
marker, so a kit still on a hand-written bib is visibly *out* of coverage rather
than silently passing, and joins it the moment it is migrated — AS-5's kits are
out today. `tests/test_build_venue_bib.py`.

### 5.4 The review pass on the generator — one finding real, one refuted, and the refuted one was the useful half

The testing standard's independent review pass ran on `build_venue_bib.py` and
returned two findings. Both were tested before either was acted on, which is the
only reason the second one produced anything.

**Real, and fixed.** The comment strip was whole-line only, so a *trailing*
comment leaked: `\citep{live} % superseded, was \citep{parked}` cited `parked`.
Harmless while `parked` is still in the corpus (BibTeX prints only cited entries)
and a loud abort once it is not — the build fails on a key nothing cites. The
strip is now an unescaped `%` through end of line, which has two constraints
pulling against each other and both are pinned: it must not eat `\%` in prose
("refusal fell by 40\%"), and it must **leave the newline**, or a wrapped
`\citep{a,b%⏎c}` never closes its brace group and loses every key in it.

**Refuted as reported.** The second finding said a key defined twice inside one
`references.bib` would trip the cross-direction clash check and misreport itself
as a one-master violation across directions. It cannot: `parse_entries` returns a
**dict**, so the repeat collapses before `resolve_corpus` ever sees it, and the
check returns clean. Verified by running it.

**But the refutation is the finding.** The repeat collapses *silently, with the
last copy winning* — nobody is told, and that is the same one-master violation
the check exists to catch, one scope down. This corpus has had exactly it: a
`zou2024circuitbreakers` present twice and disagreeing with itself about a venue,
found by hand days earlier. So the fix is to make the reviewer's assumed
mechanism true — `entry_keys()` keeps the repeats, `resolve_corpus` counts
occurrences rather than files, and the error message names the two scopes apart
(`2x within model-internals` vs `across llm-security, model-internals`) so the
operator does not go looking in the wrong file. The real corpus is clean under
the widened check, across all three direction bibs.

**The transferable part:** *a refuted finding is not a discharged one.* The
reviewer read a mechanism that was not there and was wrong about the failure;
the reason they thought to look was a gap that was real, one level down from
where they pointed. Testing the claim rather than accepting or dismissing it is
what separated the two — and dismissing it on the correct grounds ("`parse_entries`
returns a dict, so this cannot happen") would have closed the ticket and left the
silent overwrite in place.

---

## 6. ⚠️ The reviewer-lens self-review ran 2026-08-21, and it says the third guard is NOT the next spend

Full review (gitignored, reviewer text): `text_docs/reviews/as6_selfreview_20260821.md`.
Verdict **Borderline** at a general top-tier track, **Weak Accept** at a
safety/alignment one; contribution **Significant** on the method axis, **Moderate**
on the empirical. It was commissioned to test one belief and it refuted it.

**The belief tested: "two guards, one corpus" is the binding weakness, so TODO 75's
third guard is the right ~10 GPU-hours.** It is roughly the FIFTH weakness, and the
purchase is aimed at the axis the paper is least exposed on. Three findings, none of
which needs a GPU, and the ordering matters more than any of them individually.

- **⚠️ The Method describes our own rigor as the flaw a referee rejects on.** It reads
  *"the transfer AUROC at the best licensed layer--position cell"*, which to anyone who
  reviews probing papers is an uncorrected grid search over layers × positions with the
  max reported. **We do not do that.** `deployment.py:85–99` tests the max against a
  null of maxima under shuffled TRAIN labels, so the search sits inside the null;
  `contract.py` enforces it as `selection_inside_null` with the withhold reason
  *"layer/position selection was not inside the null (P7)"*; and `probes/linear.py:139`
  fits on a held-out split with a shuffled-label control refitted on the SAME split.
  None of it is in the paper. **Two sentences convert the most likely reject reason into
  a stated strength** — the highest value-per-word edit available, and it is free.
- **⚠️ Not one confidence interval appears anywhere.** At n=100 the Wilson intervals are
  7 → [3.4, 13.7], 10 → [5.5, 17.4], 17 → [10.9, 25.5], 23 → [15.8, 32.2]. The
  EXISTENCE claim survives — the smallest cell clears zero — but every ORDERING the text
  leans on does not: LG `homoglyph` 7 and LG `fullwidth` 10 overlap almost entirely, and
  `Limitations` ¶3's *"shrinks every cell without reordering them"* is an
  ordering-stability claim about quantities whose order was never established. Arithmetic
  on cells already on disk.
- **⚠️ The screens collapsed the CONDITION axis, not the guard axis, and the paper never
  says so.** `tab:map` is 7 rows minus 2 self-labelled controls = **five rows over three
  encodings**, of which the two that replicate across guards — `homoglyph`, `zero_width` —
  are the same family (Unicode substitution/invisible characters). A 19-rung ladder
  yielded two encodings of one kind. **A third guard adds a third column on those same
  two encodings**, so it leaves the objection a referee computes in ten seconds from the
  table completely untouched.

**Where that leaves TODO 75.** The counter-argument is not weak and is recorded rather
than dismissed: the abstract nominates the SCREENING METHODOLOGY as the main
contribution, and cross-family portability of the inherited-ability selector is that
contribution's own generalisation — so the third guard is coherent *on the method axis*.
The split is clean: **third guard strengthens the METHOD claim; a second CORPUS
strengthens the EMPIRICAL claim.** The empirical claim is the weaker of the two, and it
is the one whose thinness is visible from a table. The tiebreaker is the paper's own
cautionary tale: a **corpus** property (length) faked the entire result once and its
control removed 11 of 17 conditions on one guard, while `Limitations` already argues the
inherited selector's error direction is conservative. **So if exactly one run is bought,
a second corpus buys more than a third guard** — filed as TODO 76 beside 75, both
gated, neither authorised.

**Two smaller findings worth carrying.** The abstract states *"80 per cent for a target
model in the same design"* with no citation — an uncited companion-study number doing
comparative work in an abstract, unsupported outright if AS-5 is not citable at review
time. And the abstract is 403 words with **17 numerals** against a house rule of 1–3;
the structural problem is ordering, since ~250 words of attrition precede the positive
finding, which risks reading as instrument fragility rather than as rigor.

**✅ ALL FIVE APPLIED 2026-08-21 (owner go), both kits, TODO 76 closed.** What landed:

- **Method now states the selection correction and the held-out fit.** Two sentences:
  the maximum transfer AUROC over the grid is compared against a null of maxima drawn
  under shuffled TRAIN labels, so licensing is one test rather than one per cell; and
  each probe is fitted on a held-out split with its shuffled-label control refitted on
  the SAME split. Both were already true in code and absent from the paper.
- **`tab:map` carries 95% Wilson intervals** (7 [3–14], 17 [11–26], 10 [6–17],
  8 [4–15], 23 [16–32], 23 [16–32], 9 [5–16]) and the caption says what they mean:
  *the table establishes that the cell is populated on every surviving condition and
  does not order them.* `Limitations` ¶3's ordering claim is rewritten to match.
- **The uncited "80 per cent for a target model" is CUT from both the abstract and the
  body.** What replaces it is self-contained: the wrapper accounts for 8–16% and 7–33%
  of the two guards' plaintext discrimination, *the remainder is a response to
  content*, and whether a generating model behaves the same under the identical control
  is named as a question this design can ask and this paper does not answer.
- **Abstract rewritten: 403 words / 17 numerals → 347 / 4**, decomposition first. Two
  things were cut rather than shortened and should not come back — the target-model
  figure above, and the per-screen attrition counts (they are the first result and they
  live in Table 1, where they have controls). ⚠️ n is now stated in Method (one hundred
  harmful and one hundred benign per condition), which the intervals need.
- **`Scope` says what `tab:map` implied.** After the decode screens the band is four
  conditions per guard, one of WildGuard's is then removed by the block-axis screen, and
  the two that replicate across both guards belong to one family: *demonstrated on
  surface Unicode substitution, not on encoded prompts in general.*

**Two process notes worth keeping.** The second kit was **DERIVED, not hand-ported** —
the body from `\begin{abstract}` to `\end{document}` was copied whole, since that is the
only range the parity guard compares and the only range that legitimately matches. Hand-
porting is exactly what produced the defect the guard was founded on. And the edit
matcher was made **whitespace-insensitive** after a literal match failed on a line wrap:
a wrap-sensitive edit script has its strictness set by line width, which is the same
defect `test_public_repo_hygiene.py` already paid for. Both kits build clean: 0 errors,
**0 overfull and 0 underfull boxes** — the intervals widened `tab:map` 10.9pt
past the margin and `tabcolsep` went 3pt → 1.5pt to absorb it, found in `build.log`
rather than by eye, since a 0.9pt overhang is invisible on screen.

**The run question is now the live one**, with the paper in its repaired state: item 75
(third guard, METHOD claim) vs item 77 (second corpus, EMPIRICAL claim). Neither is
authorised and neither has an approval-gate triple yet; the first deliverable for either
is a committed `conf/experiment/` preset plus `scripts/cost_model.py --preset`.

---

## 7. ⚠️ An external automated desk check failed the paper on MINIMUM QUALITY, and the internal self-review had missed it (2026-08-21)

The draft was run through an external automated desk check. Length passed, topic
passed, **minimum quality failed: the paper had no Discussion and no Conclusion
section.** It ran Introduction → Scope → Related work → Method → Results →
Limitations → bibliography and stopped. The science was never reached.

**The lesson is about the review, not the section.** §6's reviewer-lens pass had
run the same day, graded every section, and never asked whether the SET of
sections was complete — because it took *structurally complete* from this repo's
own status line, which enumerates those same six. That is the lit-search rule one
level up: **a failed lookup is not evidence of absence, and neither is an index
that was never built to answer the question being asked of it.** The estate has
now committed this shape three times (the coverage sweep measured against a
narrow index; the paper asserting the absence of its own table; this). So the fix
is mechanical rather than another instruction to reviewers: **`tests/test_paper_skeleton.py`**
asserts every kit has an opening and a concluding section, discovers kits by glob
so a new paper is covered without editing it, and matches the concluding section
by INTENT rather than by one title (AS-5 says `Conclusion`, AS-6 says `Discussion
and conclusion`; a guard pinned to one spelling would fail the wrong paper).
Verified by mutation: removing the section reddens exactly that kit's case.

**⚠️ AND THE SAME BUILD SURFACED A WORSE DEFECT THE SELF-REVIEW ALSO MISSED,
BECAUSE BOTH READ THE `.tex` AND NEITHER READ THE PDF.** Seven of the twelve
entries in AS-6's generated bib carried `note` fields, and natbib's styles
**render `note`** — so the typeset bibliography contained our internal curation
sentences: *"CANDIDATE — verify+download"*, *"was ABSENT from this bib until
2026-08-07"*, *"Entry taken verbatim from the ACL Anthology .bib"*. AS-5's
hand-written bib leaked one the same way. The masters are a curation record and
their notes are the point; **a venue bib is a citation artifact and carries no
curation metadata.** Fixed at the generator, not at the output:
`build_venue_bib.py` now strips `note` and `abstract`, top-level only (a `title`
containing `, note = …` is untouched), brace- and quote-aware, with ten tests.
AS-5's bib was stripped directly, since it is hand-written and the generator's
protections do not reach it — which makes TODO 74's migration more urgent, not
less.

**One more thing that fix broke, and the break was informative.**
`test_a_generated_bib_is_not_stale` asserted that the *verbatim master entry*
appears in the kit's `paper.bib`. True while the generator was a pure copy; red
the first time it legitimately transformed an entry. **A staleness check that
re-derives what a generator does only tests the generator it was written
against.** `render()` is now split out of `build()` and the test calls it, so the
check follows the generator by construction.

**State: all four kits (AS-5 and AS-6, both kits each) build with 0 errors, 0
undefined references, and 0 curation text in the rendered PDF. Suite 2219 green.**

---

## 8. The full external review came back 4/reject — and 9 of its 10 objections cost $0 (2026-08-21)

Desk check now **passes** on all four criteria (length, topic, minimum quality,
and prompt-injection screening). What follows is the substantive review:
**rating 4 "ok but not good enough, rejection", confidence 4.**

**The headline, and it is not what either of our own run proposals assumed:
almost nothing the reviewer objects to is the science being wrong. It is that a
METHODOLOGY paper does not specify its method.** Cons 2, 3, 5 and 6 are one
sentence four times over: the paper never states the probe estimator, the
layer/position grid, split sizes, seeds, permutation count, the control-floor
protocol (which conditions are controls, the floor values, the required margin),
the threshold-selection criterion, the corpus, the 19 encodings, the wrappers, or
checkpoint revisions. **We have every one of those written down already** in
`instrument_layer.md` §2.7, `conf/`, and the code. That is transcription, not
research.

**⚠️ NEITHER TODO 75 NOR TODO 77 IS WHAT THIS REVIEWER ASKED FOR.** Guard count
appears **nowhere** in the ten objections; the corpus appears only as *name it*
(con 6) and *your conclusion is corpus-specific* (con 9), both wording. §6's
self-review concluded the third guard was the wrong spend; this review says
**neither run is the bottleneck.** Both remain filed and gated, and the case for
either is now weaker than the case for the free work.

**Cost triage, verified against local artifacts rather than estimated.**
`outputs/runs/as6_phase1/*/scores-b10/cells.jsonl` holds **1900 cells per guard,
all 19 conditions, with per-prompt `decode_score` AND `decode_threshold`**, plus
`blocked`, `p_unsafe`/`p_safe` and the tri-state `decoded`. So the operating-point
sweep, every confidence interval, and the joint factorial are **offline and $0**.

| # | Objection | Cost |
|---|---|---|
| 1 | Causal claim overstated; recast as probe separability | $0 prose (the test is design-blocked, §3) |
| 2 | Probe spec absent | $0 prose; multi-seed refit alone needs GPU |
| 3 | Control-floor protocol absent | $0 prose |
| 4 | "Capability failure accounts for the rest" contradicts our own tri-state | $0 prose |
| 5 | Sweep not shown; post-hoc operating point | **$0** — `decode_score` is local |
| 6 | Corpus/encodings/wrappers/checkpoints unnamed | $0 prose |
| 7 | CIs only on one quantity | **$0** — from local cells |
| 8 | Wrapper screen → report the full factorial + interaction | **$0** — arms are local |
| 9 | "Retires the format-detector account" too broad | $0 prose |
| 10 | No baselines | **half $0** — two of the four are already ours |

**Con 4 is a real self-contradiction and con 1 quotes a sentence WE ADDED
YESTERDAY.** §5.4 says *"The capability failure accounts for the rest of the
ladder"* one clause before saying those cells are *"unmeasured, not not
decoded"*. And §7's new Conclusion says *"the payload's content is present in the
residual stream and the verdict does not act on it"* — the second clause is
exactly the causal claim `Limitations` disclaims. **The conclusion written to fix
a desk reject reintroduced the overclaim the rest of the paper is careful about.**
That is this repo's recurring failure mode landing in the newest prose, and it
argues for reading a new section against the paper's own disclaimers before it
ships.

**Con 10 is half-answered by material already in the paper.** The reviewer wants
an uncontrolled transfer probe (that is Table 1's first row) and a length-only
classifier (that is the length null). Both exist as *screens* and neither is
presented as a *baseline*. Reframing costs nothing; the multi-position model and
the decoder-plus-guard pipeline are genuine new work.

**⚠️ THE NINE SUGGESTED CITATIONS ARE UNVERIFIED AND SOME MAY NOT EXIST.** An LLM
reviewer's reading list is not evidence of existence, which is the exact mirror
of the error this repo already made in the other direction (the coverage sweep's
three false "misses"). **No suggested reference lands without primary-source
verification and an evidence tier**, per the config-cites-its-paper rule and the
lit-search handbook. Filed as its own task rather than folded into the writing.

**Two pros worth carrying, because they credit work from item 76 done hours
earlier:** the reviewer singles out Table 2's Wilson intervals and its refusal to
rank conditions ("the authors do not rank conditions or guards based on small
apparent differences"), and Table 1's attrition as "perhaps the strongest
empirical contribution". The free-fix pass was load-bearing for the review it had
not yet seen.

### 8.1 ⚠️ CORRECTION to §8's cost triage: the local per-prompt records are the RETIRED operating point

§8 said the operating-point sweep and every confidence interval were offline and
$0, on the strength of finding `decode_score` and `decode_threshold` per prompt
in `outputs/runs/as6_phase1/*/scores-b10/cells.jsonl`. **The records are real,
complete and load cleanly, and they are the wrong run.** `results.json` there
records `probes.reading_percentile = 50.0`, retired 2026-08-08; the paper's
Table 2 comes from run `9033528` at 75, whose records are **not local**.

Recomputing from them reproduces the SUPERSEDED map exactly — Llama Guard
8 / 17 / 12 / 25, which is `phase1_map.md` §0.6's retired table to the cell — and
disagrees with the paper by up to 17 points on `reverse_words`. **Attaching a
confidence interval to that would put an interval on a number the paper does not
contain**, which is worse than having none. The same trap as every other
instance in this repo: *a measurement is only as current as the code it timed*,
here one level over — an artifact is only as current as the knob it was produced
under.

**The split that actually holds**, and it is a clean one:

- **Free and valid now — threshold-INDEPENDENT quantities.** A block rate never
  consults the decode read. The computed block-rate intervals match Table 2's
  point estimates exactly on all seven reported conditions (Llama Guard
  0.92 / 0.83 / 0.85 / 0.65, WildGuard 0.71 / 0.75 / 0.73), which is itself the
  check that the local records are the right *guard runs* even at the wrong
  operating point. Artifact: `outputs/analysis/review_statistics_20260821.json`.
- **Needs run `9033528` down-synced** — every threshold-DEPENDENT quantity: the
  sweep (con 5), `decoded_not_blocked` recounts, and the `blocked_without_decoding`
  interval (con 7).
- **Needs the wrapper/benign run records down-synced** — AUROC intervals, benign-arm
  rates, wrapper margins, and the joint factorial with its interaction term
  (con 8). Only WildGuard's benign arm is local; `phase1_map.md` §2 carries the
  summary numbers but a summary cannot yield an interaction interval.

None of this is a re-run. It is an rsync, and sessions do not self-pull.

**`scripts/review_statistics.py` FAILS CLOSED on this rather than documenting it.**
It reads the source run's own `reading_percentile`, compares it against the
settled value, and withholds every threshold-dependent quantity with a stated
reason when they disagree — the instrument contract's rule applied to an offline
analysis script. The settled value is duplicated there as a literal on purpose:
a guard that reads the same config the run would have read cannot detect that the
run predates it.

**One table edit was tried and reverted.** Block-rate intervals in `tab:map`
cost 25.5pt of overhang that neither `\tabcolsep` nor `\footnotesize` absorbs
(the overhang is font-independent). They are computed and stored; they belong in
TODO 78's appendix of full per-condition rates, not wedged into the main table.

## 9. ⛔ AS-6's DECODE AXIS RUNS ON THE PROCEDURE THAT WAS REFUTED FOR AS-5 THE SAME DAY (2026-08-21)

**Do not write TODO 78's specification block until this is settled.** 78 asks the
paper to state its method precisely, and the method's central measurement is
under an unresolved defect. Specifying it now would produce a careful, citable
description of something that may not survive the week — and a reviewer reading
a precise method is a reviewer equipped to find the leak faster than we did.

### What is CONFIRMED, by reading the code, not by measurement

AS-5's internals leg was withdrawn (`as5/evidence_and_story.md` §4k) because
`probe_transfer` fits on **all 100 plaintext harmful + all 100 plaintext benign**
and evaluates on the encoded versions of **those same 200 items**. The condition
is held out; the ITEMS are not. Leakage at fixed *n* measured **+0.19 to +0.38**.

AS-6's decode axis is the same call. `scripts/as6_guard_probe.py` builds
`encoded_harmful` from the same `harmful` prompt list that produced
`plain_harmful_batch`, hands both to `measure_deployment`, which loops the
layer × position grid calling `probe_transfer` — the identical function. There is
no guard-side variant and no item split anywhere in that path.

The false belief is in AS-6's path too, in writing: `TransferScores`' docstring
says *"The scores are out-of-sample by construction: the probe never sees the
test condition, so every encoded example is held out without needing a split."*
The first clause is true and the second does not follow from it.

### The free evidence, which is suggestive and not decisive

Llama Guard's 19 transfer AUROCs from `scores-b10` fall into two groups:

| group | rungs | transfer AUROC |
|---|---|---|
| the four the paper reports | homoglyph, zero_width, fullwidth, reverse_words | 0.796 – **0.985** |
| the base model cannot decode | 13 rungs, base64 … morse | **0.592 – 0.678** |

Three things line up, and none of them is a measurement of AS-6:

1. **The reported band matches the refuted band.** AS-5's withdrawn readings were
   0.938–0.995; AS-6's top two are 0.985 and 0.969, on the same encoders, the
   same 200 items, the same probe function.
2. **The floor cluster matches where AS-5 landed after the split.** AS-5's Llama
   homoglyph went 0.981 → **0.618** with items held out. AS-6's can't-decode
   cluster sits at 0.592–0.678. If the guard's probe leaks like the model's, the
   reported rungs land in that cluster.
3. **Llama Guard 3 8B is a Llama-3.1-8B fine-tune**, and Llama was AS-5's *worst*
   case (0.618, band floor 0.524) while Qwen held at 0.811. The relevant prior is
   the bad one.

### ⚠️ The control floor does NOT neutralise this, and the reason is structural

The floor (0.7098, 12 controls) is derived from rungs whose **base-model ability
is 0.00**. That is the right control for surface and length confounds, which are
present at zero ability — and it is the wrong control for item memory, because
**item memory is ability-dependent by mechanism**: the encoded activation of item
*i* sits near its plaintext activation *because the model decoded it*. The
control rungs, chosen precisely for being undecodable, carry the least of exactly
the confound the reported rungs carry the most of.

So the screen subtracts a floor built where the confound is weakest and applies
it where the confound is strongest. **The margin over floor is inflated by an
unknown amount in a known direction.** This is an argument from mechanism, not a
measurement; it is stated here so that nobody re-derives it, and it is the reason
the test below is worth a queue slot.

Note that this is a *different* failure from the two already recorded on this
screen. §0.6's `caesar3` demotion was *significance is not sufficiency*, and the
sigma-window violation was *the calibration constant does not port*. Both were
caught by the floor. This one is invisible to the floor by construction.

### What is NOT known

The magnitude for guards. It could be smaller than AS-5's: a guard's template
puts ~55 tokens of classification task after the payload, so its representation
at `instruction_final` may be less item-specific than a chat model's. It could
also be larger. **Nothing here licenses a claim in either direction** — and the
free diagnostic AS-5 used (transfer beating a directly-fitted in-condition probe)
is unavailable for AS-6, because a guard cannot be asked to restate a payload, so
there is no in-condition reading to compare against.

### The test, and the instrument is BUILT

`scripts/split_half_transfer.py` now reads **both run schemas** — AS-5's single
`readings` cell and AS-6's per-family `summaries` cell — so the test that settled
AS-5 runs on the guard records unchanged. `--family` restricts it to the cells the
paper reports. `tests/test_split_half_transfer.py` (17 tests, mutation-verified)
pins the instrument in both directions: it fires when the labels carry only item
identity (A 0.997 → B 0.460) and stays quiet on a direction that genuinely
transfers (A 0.923 → B 0.883). Without the second half the screen would be a
verdict with a script attached.

**Cost: 0 GPU, $0, CPU on `short`.** The guard activations are already cached on
the cluster (`.../activations/llama_guard_3_8b/`, `.../wildguard/`) and every path
is recorded in the run records, so nothing is captured and nothing is downloaded.
The plain-condition pair is resolved by glob because AS-6's schema records only
the encoded paths; an ambiguous match is a refusal, not a pick.

**It cannot run locally.** Only `llama3_1_8b_instruct` activations are cached on
this machine; the guard captures live on `/scratch`. The script fails loud off
the cluster rather than substituting anything.

### What it gates — the run is a GATE, not a measurement

- **B holds above the floor** → the decode axis is real, the map stands, TODO 78
  specifies a sound method, and the paper gains a control it currently lacks.
- **B collapses into the 0.59–0.68 cluster** → every `decoded_not_blocked` cell is
  unmeasured, the results table is withdrawn, and the paper is re-planned around
  what survives (block rates are untouched — they never consult the decode read).

Both branches change what gets built next, which is the test this repo requires
before a run is allowed to exist.

### ✅ IT RAN — the map SURVIVES, one cell falls, and every margin thins (2026-08-21)

Two CPU jobs on `short`, 6 min 37 s and 30 s, 0 GPU, $0. Artifacts
`outputs/analysis/as6_split_half_transfer.json` and
`as6_floor_{llamaguard,wildguard}_{unsplit,split}.json`; 53 cells across both
guards, all 19 conditions each, plus both paper records.

**The mechanism argument above was right, and that is the most reusable part of
this result.** Leakage at fixed *n* is **ability-dependent, sharply**:

| condition class | leakage at fixed *n* (F − B) |
|---|---|
| the base model decodes (ability ≥ 0.82) | **+0.147 to +0.239** |
| the base model cannot (ability 0.00) | **−0.011 to +0.017** |

So the control rungs barely leak, the floor barely moves, and the candidates
fall alone. That is the *differential* collapse — a proportional one would drag
the floor down with the candidate and demote nothing.

**Llama Guard 3 8B** — floor 0.7098 → **0.7066**:

| condition | unsplit | item-split | 95% band | margin before → after | verdict |
|---|---|---|---|---|---|
| homoglyph | 0.9849 | **0.8453** | [0.780, 0.905] | +0.275 → **+0.139** | holds |
| zero_width | 0.9687 | **0.7671** | [0.689, 0.840] | +0.259 → **+0.061** | holds |
| reverse_words | 0.7964 | **0.7584** | [0.679, 0.834] | +0.087 → **+0.052** | holds |
| **fullwidth** | 0.8802 | **0.7057** | [0.627, 0.788] | +0.170 → **−0.001** | ⛔ **DEMOTED** |
| caesar3 *(control)* | 0.7168 | 0.7107 | [0.617, 0.781] | — | stays demoted |

**WildGuard** — floor 0.6852 → **0.6617** on the 11 controls this table was computed with; **both values were superseded on 2026-08-22 by the 14-control floor, 0.6803 → 0.6605 (§25)**, which changes no verdict and moves each margin below by about +0.001:

| condition | unsplit | item-split | 95% band | margin before → after | verdict |
|---|---|---|---|---|---|
| homoglyph | 0.9482 | **0.7889** | [0.717, 0.862] | +0.263 → **+0.127** | holds |
| zero_width | 0.9537 | **0.7718** | [0.689, 0.846] | +0.269 → **+0.110** | holds |
| reverse_words | 0.8182 | **0.7639** | [0.689, 0.832] | +0.133 → **+0.102** | holds |
| combining_marks | 0.8537 | **0.6644** | [0.574, 0.751] | +0.169 → **+0.003** | holds, barely |
| fullwidth | 0.6412 | **0.4811** | [0.384, 0.581] | — | below chance |

**Six of the seven cells the paper reports survive.** The results table is not
withdrawn and TODO 78 unblocks. Four things must change with it:

1. ⛔ **Llama Guard `fullwidth` is withdrawn** — 0.7057 against a floor of
   0.7066. Do NOT report that as a demotion by 0.0009: the band is [0.627,
   0.788] and straddles the floor, so the honest label is `(U)` unmeasured, the
   same status `combining_marks` has. A cell that fails by a thousandth is not
   distinguishable from one that passes by a thousandth, and WildGuard's
   `combining_marks` clears by +0.003 on the other side of the same line.
2. **Every decode AUROC in the paper is the unsplit number and must be replaced**
   by its item-split value. The paper's largest, 0.985, becomes 0.845.
3. **The margins are the real casualty, not the cells.** Llama Guard
   `zero_width` goes from +0.259 over its floor to +0.061; the screen still
   passes it, but a claim resting on how *comfortably* it passed no longer holds.
4. ⚠️ **The sigma window has tightened and is now a live fragility.** Llama Guard
   [2.214, 4.645] → **[2.113, 3.440]**, and the configured 2.0 is still *below*
   the lower bound, so a control clears its own floor — the pre-existing
   §2.7 defect, unchanged. WildGuard [1.156, 5.686] → **[1.577, 2.082]**: the
   configured 2.0 now sits 0.08 under the ceiling, so at sigma 2.1
   `combining_marks` fails. Report the window, never just the constant.

**What the run also settles, cheaply.** `E`, the within-plaintext
cross-validated AUROC at each selected cell, is 0.90–0.98 on both guards — the
denominator AS-5's internals leg never had. So unlike AS-5, the split reading
here is **degraded but not floorbound**: Llama Guard `homoglyph` reads 0.845
against a 0.935 plaintext baseline. The guard's representation of the payload
survives the encoding at ~0.09 AUROC of loss, and that is now a measured
statement rather than an artefact.

⚠️ **`reverse_words` survives by leaking least (+0.037 / +0.067), which is not a
point in its favour.** It permutes word order and leaves the words intact, so
its signal was never item memory *or* decoding — it is lexical surface, which is
why AS-5 demoted it to a control. It should be read as a positive control here,
not as a fourth finding.

⚠️ **`caesar3` stays demoted, now for two independent reasons** (control status,
and a floor it still does not clear). AS-6's most striking cell — 0.77
decoded-not-blocked — remains an artefact under every treatment tried.

**A stale note found on the way, filed rather than fixed:** `CLAUDE.md` and
§0.6 still say WildGuard's floor is a **3-control BOUND**. The Mistral-7B-v0.3
ability job that closes it has since run, so WildGuard's floor is an
**11-control distribution** on both treatments. The warning "do not use it" is
obsolete; the numbers above are the distributional floors.

---

## 10. ✅ THE SPECIFICATION BLOCK LANDED, WITH ITEM 83's FOUR REVISIONS INSIDE IT (2026-08-21)

TODO 78. Both kits rebuilt clean: **exit 0, 0 overfull boxes, 0 undefined
references or citations**, parity guard green, full suite 2546 passed.
`paper/` is gitignored, so this section is the only record of what the kits now
say.

### 10.1 The four revisions from §9, as they appear in the paper

1. **Llama Guard `fullwidth` is out of Table 2.** Its row is deleted, the
   `\multirow` drops 4 → 3, and the caption states that it appeared there before
   the holdout and is now unmeasured on the decode axis. The Results text
   justifies the withdrawal by the comparison that does NOT depend on a
   thousandth: under the holdout it reads below `caesar3`, a condition whose
   base model decodes nothing at all. *It is the cell rather than the margin
   that has to go.*
2. **Every decode AUROC in Table 2 is the item-split value** (0.985 → 0.845 and
   so on down), against floors of 0.707 and 0.662 named in the caption. The two
   surviving appearances of 0.985 are both explicitly labelled as the
   *uncontrolled* value.
3. **The margins are stated as a limitation, not buried.** A new Limitations
   paragraph gives the post-holdout range (smallest 0.052, largest 0.139,
   against a pre-holdout smallest of 0.087) and says in terms that no claim in
   the paper rests on how comfortably a condition passed a screen.
4. **The sigma window is reported, never the constant.** Same paragraph:
   $[2.11, 3.44]$ and $[1.58, 2.08]$, with 2.0 configured — below the first
   range, 0.08 inside the second — and the reason the first is survivable
   (control membership is decided before the comparison).

Plus the two additions §9 named: the item holdout is a `Controls` paragraph in
Method and a Results subsection of its own, and `E` (0.90–0.98, `homoglyph`
0.935 against a held-out 0.845) is reported as the denominator, which turns
"the signal survives" into "degraded by ~0.09 AUROC, not floorbound".

Downstream sites that moved with them: the attrition table gains a fourth screen
row (**3** and **4**); the abstract, the attrition prose and the Conclusion all
carry the new counts; and the `fullwidth` dissociation subsection now says
**neither** guard's decode survives there, which strengthens rather than weakens
it, since that subsection was always framed as a behavioural result.

### 10.2 The specification itself — Appendix A, twelve subsections

Corpus · conditions · guards, rendering and the verdict read · activation
capture · the content probe · licensing · the control floor · the item-level
holdout · the per-prompt read and operating point · base-model ability ·
per-condition block rates · reproducibility. It is transcription from `conf/`
and the instrument modules, and it answers cons 2/3/5/6 — *a methodology paper
that does not specify its method* — in one place.

**Two defects the transcription itself found, which is the argument for writing
specifications out rather than pointing at code:**

- ⛔ **The Method described the wrong null.** It said the licensing null was
  drawn "under shuffled training labels". The code permutes the **evaluation**
  labels with the fitted direction held fixed, and
  `permutation_null_max_transfer_auroc`'s docstring says in terms that shuffling
  the training labels was tried first and is WRONG here — a logistic fit on
  shuffled labels is near-constant, so its transfer AUROC collapses onto 0.5
  with almost no variance and the null licenses noise. The paper had been
  describing the rejected version of its own screen. Corrected, with the reason
  kept, because the reason is the interesting half.
- ⚠️ **"Neither guard blocks any genuine cipher condition" was false by two
  prompts.** Llama Guard blocks 2/100 `rot13`. Found by generating the
  per-condition table from `review_statistics_20260821.json` instead of quoting
  the four numbers the story needed. Now "no more than 2 per 100", with the
  table to check it against.

**A law found unenforced on both papers, filed as TODO 87:** the no-dash-line
rule (global, 2026-08-19) binds papers explicitly, and both kits predate it —
40 `---` in AS-6's body, 51 in AS-5's. New prose written from 2026-08-21
conforms; the sweep is a register change per paper, not a find-and-replace, so
it is one deliberate pass with the PDF read afterwards rather than something to
do incrementally across two kits that must stay identical.

**One gap declared rather than papered over:** the guard checkpoints are pinned
by repository identifier and not by revision hash, because nothing in
`provenance.py` records a resolved revision. The appendix says so. Filing the
code change rather than claiming it, since a claim about our harness that is not
true is the same class of defect as the null above.

**Citations added:** JailbreakBench (`NEURIPS2024_63092d79`, NeurIPS 2024 D&B)
for the corpus, which was named in the paper and cited nowhere. Mistral is
deliberately **named by identifier and not cited** — `model_slate.md` §2.1
establishes that 2310.06825 describes v0.1 while we run v0.3, so a citation
there would assert provenance the paper does not have.

### 10.3 What item 78 asked for and did NOT land

**The operating-point sweep as a figure.** It needs run `9033528`, which is not
local; the local `scores-b10` records are at the retired `reading_percentile`
50 and would reproduce the superseded map (§8.1). Still gated on the owner's
down-sync, same gate as the rest of TODO 79.

## 11. THE REFEREE'S NINE REFERENCES ARE ADJUDICATED — seven real, one absent, one a fabricated title over a real paper we already owned (2026-08-21)

TODO 82. Offline, `$0`, no GPU. The rule this pass exists to obey: **a generated
citation is not evidence of existence**, which is the mirror of the error this
repo already made in the other direction, where a coverage sweep reported three
papers as missed that were cited in three places each. Both halves cost a check.

### 11.1 The verdicts

Every one resolved at PRIMARY SOURCE (ACL Anthology, the USENIX program, the
DBLP journal record, the arXiv listing), never at an aggregator alone.

| # | as the referee wrote it | resolves? | source of record | tier |
|---|---|---|---|---|
| 1 | Xin 2025, safety arms race | yes | **Findings of ACL 2026**, `2026.findings-acl.20`, pp. 421–445 | **A** |
| 2 | Zhao 2026, in-decoding probing | yes | arXiv 2601.10543, no venue | C |
| 3 | Liu 2026, model cognition | yes | **USENIX Security 2026**, SJTU + Microsoft | **A** |
| 4 | Mu 2025, benign data mirroring | yes | **NAACL 2025 long**, pp. 1784–1799 | **A** |
| 5 | Sun 2026, unified framework | **NO** | absent from arXiv full text, from S2 title match, from our corpus | — |
| 6 | Jalan 2026, survey | yes | **Machine Learning 115(6)**, Springer | **A** |
| 7 | Wong 2025, responsible AI | yes | arXiv 2511.18933, self-described technical report | C |
| 8 | Constitutional Classifiers 2025 | yes | arXiv 2501.18837, no venue confirmed, heavy uptake | B |
| 9 | Fairoze 2026, security/privacy/provenance | **title NO** | the described work is **USENIX Security 2026**, arXiv 2510.01529 | **A** |

⚠️ **#1 is the Tülu-3 lesson running backwards.** Semantic Scholar asserted
"Annual Meeting of the ACL" while the paper's own arXiv comments field named no
venue at all. The aggregator was right in substance and unsupported by the
primary source; the Anthology is what settled it. Neither direction of an
aggregator claim is evidence.

⚠️ **#5 does not exist under that title.** Three independent lookups fail. The
nearest real papers are EasyJailbreak (2403.12171, a tooling framework by
different authors) and `xu2026sok` (IEEE S&P 2026), and the second already fills
the taxonomy-anchor role the referee wanted #5 for. Recorded as not resolving.

### 11.2 ⛔ #9 IS THE ONE THAT MATTERED, AND THE CORPUS ALREADY HELD IT

"Fairoze et al., *Security, Privacy, and Provenance for Generative AI*, 2026" is
not a paper. The author is real, and the work the referee described is
**Fairoze, Garg, Lee and Wang, *Bypassing Prompt Guards in Production with
Controlled-Release Prompting*, USENIX Security 2026** (arXiv 2510.01529). It
starts from an impossibility result, that no filter running materially faster
than the model it protects can universally separate adversarial from benign
prompts, and turns it into a working attack on four production systems and
fourteen open-weight guards.

**That is AS-6's *blocked without decoding* cell constructed on purpose, with a
lower bound behind it, and the paper was not citing it.** The gap was ours, not
the corpus's: the entry has sat in science `literature/llm-security/` under
`fairoze2026bypassingpromptguardsproduction` since before this paper had a
related-work section. Grep the corpus before believing a paper is missing.

Its venue was also under-recorded, as `@misc` arXiv. Confirmed on the official
USENIX Security 2026 program and in the paper's own v4 comments, and upgraded in
the master with the key unchanged.

**What the paper now says, and what it must not say.** We claim no novelty for
the mechanism. The complementary question is the one we can still claim: under
ordinary surface encodings that nobody designed to be undecodable, how often
does a guard fail because it never recovered the payload, and how often does it
recover the payload and let it through? Their construction guarantees the first
case by design; ours has to measure which case occurred.

### 11.3 What landed in the paper, and what did not

Cited: #1 (end-to-end verdict, motivating decomposition), #3 (closest instrument
prior, and a direct answer to the referee's single-layer question), #4 (in
Limitations, as the adversarial corpus that would test the decode read from the
side our floor cannot), #8 and #6 (guard-evaluation context), #9 (above).
Recorded in the master and NOT cited: #2 and #7, both tier C, both weak enough
that a citation would be decoration. #5 recorded as not resolving.

⚠️ **#3 cuts both ways and the paper says so.** \citet{liu2026cognition} report
safety features separable in intermediate states at up to 99 per cent,
persisting where behaviour complies, which is the claim AS-5 WITHDREW under an
item-level holdout. Theirs is a fitted probe rather than a transferred one, so
our holdout does not directly indict it, and the paper states that we make no
claim about which side of that control their number falls on. Asserting leakage
in a published result we have not read the split protocol of would be the
generated-citation error wearing the opposite hat.

### 11.4 ✅ #24 FROM THE AS-5 METHODS REVIEW WAS LIVE HERE, AND IS FIXED

Handed over by the peer session. `permutation_p_value` applies the $+1$
correction of Phipson & Smyth (2010), so with `n_permutations = 200` the
smallest attainable value is $1/201 = 0.004975$. The paper reported the
Caesar-shift artefact as licensing "at $p = 0.005$", **which is that floor**, not
an estimate. It means no null draw reached the observed value.

**The shape of the error is worth keeping.** The correction was added precisely
so a finite permutation set could not report $p = 0$ and overstate the evidence
in a paper table; the docstring says so. The prose then quoted the corrected
floor as a measurement, putting the overstatement back one level up. A guard
that fixes a number does not fix the sentence that reports it. Both kits now say
"the smallest p-value 200 draws can produce", and the appendix states the floor
where the null is specified, citing the correction.

### 11.5 Two checks that did NOT find a defect, recorded so they are not re-run

- **Rate nulled against an AUROC (the peer's P3 finding).** Does not reach AS-6.
  `as6_guard_probe.py` calls `measure_length_null` once, on
  `observed_max_transfer_auroc`, which is AUROC-valued, so the comparison is
  like with like. AS-6 builds no `Reading` objects and its block rates never
  touch `LengthNull`. ⚠️ The residue is the opposite one and is filed: the block
  rates have **no length control at all**, and length is exactly what an
  encoding inflates, so `blocked_without_decoding` could in principle be a guard
  reacting to length. The benign and wrapper arms hold content and template
  fixed but not length.
- **Unpaired null across conditions measured on the same items (the peer's #3).**
  AS-6 does not make AS-5's error. It uses overlapping Wilson intervals to
  REFUSE an ordering, never to assert equality, and says so. ⚠️ But the
  mis-specification is still there in the conservative direction: the conditions
  share the same 100 harmful prompts, so a paired test would be strictly more
  powerful than independent intervals, and declining to order on overlapping
  independent intervals discards real information. Filed rather than fixed,
  because the per-prompt verdicts are on the cluster.

## 12. THE SCREENS ARE NOW SCORED AGAINST A CRITERION THAT DOES NOT CONSULT THEM (2026-08-21)

TODO 81's free half, external review con 10: *does the licensing framework
materially improve validity, or does it only change your own results?* Offline,
`$0`, no GPU, no new run.

### 12.1 The criterion

A screen that only ever rejects cannot be evaluated by its own attrition curve.
The one criterion available here that is **independent of every screen in the
paper** is the guard's base model's DECODING ABILITY, measured by generation:
if the base model cannot decode a condition at all, nothing was decoded, so a
licence granted there is a false positive, and the judgement never consults the
probe, the null, or the floor.

| | Llama Guard 3 (base Llama-3.1-8B) | WildGuard (base Mistral-7B-Instruct-v0.3) |
|---|---|---|
| conditions with ability 0.00 | **12 of 19** | **14 of 19** |
| conditions that could carry a true positive | 7 | 5 |

That second row is the whole argument. **A method admitting 17 of 19 conditions
admits at least 17 − 7 = 10 on which nothing could have been decoded**, and 12 of
19 admits at least 12 − 5 = 7. No new measurement is needed; it is arithmetic on
the marginals.

### 12.2 The result, and the row that does NOT count

| screen | LG admits | LG undecodable | WG admits | WG undecodable |
|---|---|---|---|---|
| free permutation null | 17 | **≥ 10** | 12 | **≥ 7** |
| length-matched null | 6 | 1 (`caesar3`) | 7 | 3 (`rot13`, `base64`, `caesar7`) |
| + control floor | 4 | 0 \* | 4 | 0 \* |
| + item-level holdout | **3** | 0 \* | **4** | 0 \* |

⚠️ **\* The control floor's zero is NOT an independent score, and the paper says
so in its own voice.** The floor's control set IS the ability-0.00 conditions, so
scoring it against ability is the criterion applied to itself. Reporting "the
floor achieves zero false positives" would be circular, and it is exactly the
kind of self-certifying number the paper spends its Results section removing.
What the floor genuinely adds over a plain decodability cut is **coverage of
PARTIAL ability**, where a 0.00-vs-nonzero rule gives no answer at all
(`combining_marks` on Llama, ability 0.82, is the live case: below floor, `(U)`,
and a decodability cut would have admitted it).

**The item-level holdout is the screen with the cleanest claim in the paper.** It
consults neither ability nor the control set, it runs last, and it still removes
a condition that had passed everything before it.

### 12.3 The two baselines were already in the paper under other names

Con 10 asked for a comparison against alternatives. Two of the four are
present and were merely not presented as baselines:

- **The uncontrolled transfer probe** — what a study with no licensing step at
  all would report — is Table 1's FIRST ROW, and it is now scored: it buys at
  least 10 and 7 verifiable false positives before anything else runs.
- **A length-only classifier** is the null the second row tests against, and it
  is a strong baseline rather than a straw one. **Recomputed locally on this
  paper's own corpus rather than imported from AS-5's record: character length
  alone separates JailbreakBench harmful from its topic-matched benign set at
  AUROC 0.6544** (mean 86.0 characters against 73.8, n = 100 + 100, exact
  Mann-Whitney). Every encoder in the ladder is monotone in length, so the
  separation survives into every condition.

The remaining two of the four are real new work and stay costed separately: a
multi-position/multi-layer representation model (which `liu2026cognition` argues
for, §11.3) and an external decoder-plus-guard pipeline. Neither is free, and
the free half is what tells us whether they are worth buying.

**Typesetting note kept because it recurs:** the 5-column table overflows the
AAAI column by 17.86pt at default `tabcolsep`; 3pt fits with the headers intact.
Found in `build.log`, not by eye, which is the standing rule after the `tab:arms`
incident.

## 13. THE DASH-LAW PASS ON THE AS-6 KIT (2026-08-21)

TODO 87, AS-6 half only. Global law 2026-08-19: in prose written for the owner
or on his behalf, papers included, two clauses are never joined by a dash line.
**39 em-dash joiners in the body, now zero**, in one deliberate pass rather than
incrementally, with the built PDF read afterwards.

**The distribution is the useful fact for the AS-5 half:** 36 of the 39 were
**paired appositives** (`X---the gloss---and`) and only 3 were single
clause-joiners. That ratio decides the method. A paired appositive is not a
sentence-splitting problem, and reaching for a full stop on one produces a worse
sentence than the dash did.

**The rule that emerged, worth reusing:** the appositive's OWN internal
punctuation picks the replacement. If it contains a comma or a semicolon,
commas are unavailable and it takes parentheses (13 of 18 pairs here, e.g.
*invisible characters, homoglyph and full-width substitution, word-order
permutation*). If it is short and comma-free, commas read better and avoid a
page of parentheses (*Causal tests, ablating the direction and observing the
verdict, are the natural next step*). Two cases took neither: an imperative
inside a parenthesis reads badly, so *The direct test (ablate the direction and
observe whether the verdict moves) is well-established* became a colon with the
imperative after it.

⚠️ **A mechanical pass leaves a second defect behind, and only reading catches
it.** Converting `X---gloss---Y` to `X (gloss) Y` silently drops the comma that
the interrupted clause needs, so five sentences came out running on
(*...under free permutation licensing (the natural significance test) 17 of 19
conditions...*). The dashes were carrying two jobs, and replacing them
one-for-one restores only one. Fixed by reading every one of the 20
parentheticals in context.

⚠️ **Do not touch `--`.** 51 en dashes in this body are ranges and compounds
(`0.29--0.53`, `guard--condition`, `layer--position`) and are correct LaTeX. The
law governs clause joiners, not typography. A regex written against `-{2,}`
would have destroyed all 51.

Both kits build 0 overfull, 0 undefined. AS-5's 51 remain and stay TODO 87.

## 14. ⚠️ THE ITEM HOLDOUT REACHED THE AUROCs AND NOT THE COUNTS, AND BOTH HEADLINE CELLS ARE BIASED TOWARD OUR OWN RESULT (2026-08-21)

Found by running the peer session's AS-5 lesson against this paper: *a screen
adopted in one section does not retroactively reach a table written before it
existed, and nothing flags that.* AS-5's instance was the echo screen against
its pipeline table. **AS-6 has one, and it is worse, because it lands on the two
cells the paper reports as findings.**

### 14.1 What is true

`scripts/split_half_transfer.py` computes CONDITION-LEVEL statistics (A, B, C,
D, E, F, G are all AUROCs over 200 splits). It never emits a per-prompt label
and never touches the operating point. So:

- every **AUROC** in the paper is the held-out one ✅
- every **count** in the paper (`D&¬B` in Table 2, `blocked_without_decoding` in
  §5.8) still comes from the **unsplit** probe's per-prompt read ⛔

### 14.2 The direction is not neutral, and it favours us twice

A prompt reads *decoded* when its score exceeds the 75th percentile of the same
condition's BENIGN scores. Item memory raises a seen harmful item's score and
lowers a seen benign one's, so it both raises the numerator and lowers the
threshold. More prompts read *decoded*. Therefore:

| cell | what the paper claims | leak moves it |
|---|---|---|
| `decoded_not_blocked` | populated, this is the finding | **up** |
| `blocked_without_decoding` | "at most 5 per 100", the emptiness is the finding | **down** |

**Both headline cells are biased toward the result we state.** This is the same
asymmetry AS-5 reports on its behaviour axis (every defect inflates apparent
safety), reappearing here as *every defect inflates the reported finding*, and
it is structural for the same reason: a confound in the decode read cannot help
being aligned with the claim, because the claim is about the decode read.

### 14.3 What we did NOT do, deliberately

**No estimate of the size.** The condition-level statistic fell 0.04–0.20 AUROC,
and converting that into a count at a fixed percentile requires refitting the
per-prompt read on held-out items, not arithmetic. Inventing a correction factor
here would be the third instance of deriving a number from another number
because the real one was expensive. Disclosed in Limitations, filed as TODO 96,
gated on the same down-sync as 88/89 (the split read needs cached activations,
and only summaries are local).

⚠️ **The general check this came from is worth running on any paper here, and it
is not the same as re-running the pipeline.** Run each screen against the TABLE
it governs, not against the run that produced the table. A screen adopted after
a table exists will pass its own tests, appear in the Method, be described in
Limitations as required for validity, and still never have touched the numbers.
Nothing in the build catches it, because there is no artifact that records which
screens a table's numbers went through.

## 15. §5.8's FORMAT-DETECTOR BOUND WAS STATED OVER THE WRONG SET, SURVIVES FOR A BETTER REASON, AND HAD AN UNSTATED COUNTEREXAMPLE (2026-08-21)

Second pass of the screen-against-the-table check (§14's method, applied to a
CLAIM rather than to a table). Local, `$0`, from the runs of record
`as6_phase1/*/as6p1-full-20260805/results.json` at `reading_percentile = 50`,
which is the permissive end of the sweep and therefore the read §5.8 says it
uses.

### 15.1 What the paper said, and what the data says

The claim was: *"Blocked without decoding is at most 5 per 100 across all 38
guard--condition pairs, and exactly 0 on most of them."*

| | Llama Guard 3 | WildGuard |
|---|---|---|
| pairs with a MEASURABLE cell | 17 of 19 | 12 of 19 |
| unlicensed, cell is `null` | 2 (`base64`, `base32`) | 7 (`base32`, `hex`, `binary`, `ascii_decimal`, `unicode_escape`, `morse`, `tag_block`) |
| max over the FLOOR-CLEARING set | **0.05** (`fullwidth`) | **0.01** (`reverse_words`) |
| max over ALL 19 | **0.08** (`combining_marks`) | 0.01 |

**Two defects, opposite in sign.**

⛔ **The bound was quoted over all 38 while 9 of the 38 have a `null` cell.** As
written it reads exactly like this repo's signature failure, unmeasured folded
into a negative, which is the defect the `(U)` band exists to prevent and which
the pilot's `deployment` tri-state fix was the most consequential instance of. A
referee checking the artifact finds nine nulls and reasonably concludes we
counted them as zeros.

✅ **But the claim is TRUE, and for a stronger reason than the paper gave.**
`blocked_without_decoding` is a SUBSET of `blocked`, and **the block rate on
every one of those nine pairs is 0.00**. So the cell is empty by CONTAINMENT, no
decode read required. The unmeasured decode axis does not weaken the bound at
all on those conditions; it is irrelevant to it.

⛔ **And one pair genuinely exceeds the bound: Llama Guard `combining_marks` at
0.08.** It went unnoticed because it is not in Table 2 (the control floor demotes
it, 0.7007 against 0.7098) — so the claim's set and the table's set had drifted
apart, and only the claim's set included it. On a condition where decoding
cannot be read, the *decoded* / *not decoded* split is exactly what is
unavailable, so that cell is not evidence for a format detector either way. Now
stated in the paper rather than dropped along with the condition.

### 15.2 The general lesson, sharper than §14's

§14 found a screen that reached a statistic and not a count. **This is a screen
that reached a TABLE and not a CLAIM.** The control floor governs Table 2's
membership, and §5.8's sentence quantifies over all 19 conditions per guard, so
the floor silently changed one set and not the other. Nothing connects them:
`combining_marks` is absent from Table 2 for a reason that never propagated to a
sentence three subsections later which still ranges over it.

**The check that finds this class is cheap and mechanical: for every bound or
"across all N" claim in a paper, recompute N from the artifact and confirm the
set matches the set the screens produce.** Both of this section's defects are
visible in one pass over `results.json`, and neither is visible from the prose.

## 16. THE TWO-SCREEN CONVERGENCE WAS TRUE AND MISDESCRIBED: ONE SCREEN'S VERDICT IS LARGELY FIXED BY ABILITY (2026-08-21)

Found by the peer session's method (`paper_claims.py` / `claim_sets.py`, `dadbe6a`):
recompute a counted claim's SET from the record rather than tracing the number
forward. Applied to AS-6's `\subsection{Two independent screens converge}`. The
counts survive; what the agreement means does not.

### The claim as it stood

> On WildGuard, the conditions whose licensing is unstable under the number of
> length strata (5, 10, 20) are exactly the three the control floor rejects, and
> the conditions that are stable are exactly the four it keeps. The two screens
> share no input [...] so the agreement is evidence rather than construction.

### Three defects, none of which moves a number

**(a) The universe was unstated.** "The three the control floor rejects" is true
only inside the set WildGuard licenses at ten strata, which is seven conditions.
Over all 19 the floor rejects 15. A referee recomputing the set from the record
gets 15 and reads a contradiction. Now stated as "of the seven conditions
WildGuard licenses at ten strata". Same shape as §15's "all 38 pairs".

**(b) ⛔ The floor's verdict on those three is largely fixed once ability is
known, so the surprise value is entirely on the other screen.** The three are
`base64`, `rot13`, `caesar7`. All three have Mistral-7B-v0.3 ability 0.00, and
**two of them are inputs to the floor itself** (the floor of record is 11
controls = the ability-0.00 rungs of job `9031680`; `rot13` and `caesar7` are
both in it), where `clears()`'s control special-case makes clearing impossible
by construction. `base64` is ability-0.00 too and is outside the 11 only by the
bookkeeping accident §0.6 records (measured in an earlier Mistral run the floor
job did not read).

Arithmetic that pins it, and it closes exactly: the 19-condition ladder minus
the five rungs Mistral decodes (`homoglyph`, `zero_width`, `reverse_words`,
`combining_marks`, `fullwidth`) is 14 ability-0.00 rungs; minus the three
excluded (`base64`, `reverse_characters`, `tag_block`) is **11**, and `rot13`
and `caesar7` are in that eleven while `base64` is not.

So "the two screens share no input" is TRUE as stated about inputs and
MISLEADING as an argument, because one screen's output on the disputed
conditions is near-deterministic in ability. The claim that survives is the
better one and is now what the paper says: **bin-stability never consults
ability and recovers the ability partition anyway**, so instability under the
null tracks the base model's inability to decode.

**(c) The fourth "keep" is the paper's own unresolvable cell.** WildGuard
`combining_marks` clears the post-holdout floor by **+0.003** (0.6644 against
0.6617) with a 95% band [0.574, 0.751] that straddles the floor completely. §9
point 1 uses exactly that cell to argue that Llama Guard `fullwidth` at −0.0009
must be `(U)` and never "demoted by 0.0009", because "a cell that fails by a
thousandth is not distinguishable from one that passes by a thousandth". **The
paper applied that symmetry to `fullwidth` and not to `combining_marks`**, and a
counted claim depended on which way it went. Now stated: the partition is exact
on six of the seven and unresolved on the seventh.

### The lesson, which is the peer's with one more turn on it

§15 found a bound whose SET had gone stale. This is the same check finding a
claim whose set is exactly right and whose **warrant** is not: every count in the
sentence recomputes correctly, and the inference the sentence draws from those
counts does not hold. A set-recomputation guard catches (a) and cannot catch (b),
because (b) is about where a verdict comes from rather than what it is. The
mechanical half of the peer's design is not weakened by this; it is the reason
to keep a human read on the warrant after the counts reconcile.

No number in Table 1, Table 2 or Table 3 moves. Both kits patched and rebuilt,
0 overfull / 0 undefined, 588 paper guards green.

## 17. ⛔ FIVE ITEMS WERE BLOCKED ON A MISIDENTIFIED JOB ID, AND THE DATA WAS LOCAL THE WHOLE TIME (2026-08-21)

The down-sync ran. Five artifacts landed (`as6_split_half_transfer.json` and the
four `as6_floor_*`). **Run `9033528` was not among them, because it is not the
run anyone was waiting for.**

### The misidentification

`instrument_layer.md` §6.3.2 records `9033528` as the **causal-intervention run**
on Llama Guard 3 8B, 2026-08-09, the one that returned `n_eligible: 0`. Its row
in that section's table reads `9033528 | Llama Guard 3 8B | 0.974 | +0.023 | 0`.

`evidence_and_story.md` §8.1 recorded the same id as *"the paper's run `9033528`
at 75, whose records are not local"*, and TODO items 78, 79, 88, 89 and 96
inherited that sentence. Both statements could not be true. The causal run is the
correct reading, and every AS-6 run record on disk carries
`reading_percentile = 50.0`, so there is no percentile-75 run to sync.

### Where Table 2 actually comes from, verified rather than inferred

`outputs/analysis/operating_point_{llama_guard_3_8b,wildguard}_20260808.json`,
**local since 2026-08-08**, carry a per-condition sweep over six operating
points (50, 75, 90, 95, 99, 99.5) with `decoded_not_blocked`,
`blocked_without_decoding`, `blocked_on_content`, `never_decoded` and
`unmeasured` at each. All six of Table 2's `D&¬B` counts reproduce **exactly** at
percentile 75:

| guard | condition | paper | artefact p75 | p50 |
|---|---|---|---|---|
| Llama Guard | homoglyph | 7 | **7** | 8 |
| Llama Guard | zero_width | 17 | **17** | 17 |
| Llama Guard | reverse_words | 8 | **8** | 25 |
| WildGuard | homoglyph | 23 | **23** | 23 |
| WildGuard | zero_width | 23 | **23** | 28 |
| WildGuard | reverse_words | 9 | **9** | 16 |

So the numbers were never unprovenanced; the pointer to them was wrong.

### The sweep, which is external review con 5, computed at $0

| guard | condition | 50 | 75 | 90 | 95 | 99 | 99.5 |
|---|---|---|---|---|---|---|---|
| Llama Guard | homoglyph | 8 | 7 | 6 | 5 | 3 | 3 |
| Llama Guard | zero_width | 17 | 17 | 17 | 13 | 8 | 1 |
| Llama Guard | reverse_words | 25 | 8 | 1 | 0 | 0 | 0 |
| WildGuard | homoglyph | 23 | 23 | 13 | 10 | 3 | 0 |
| WildGuard | zero_width | 28 | 23 | 21 | 13 | 7 | 7 |
| WildGuard | reverse_words | 16 | 9 | 2 | 0 | 0 | 0 |

**The two substantive cells are robust and the control is not.** `homoglyph` and
`zero_width` stay populated on both guards out to the 99th percentile;
`reverse_words` collapses to 0 by the 95th on both. That is the expected
signature for a condition whose signal is lexical surface rather than decoding,
so the sweep independently supports its `\dagger` control label.

**And it confirms §5.8's operating-point reasoning as a measurement.** Max
`blocked_without_decoding` across all conditions climbs 8 → 31 → 54 → 56 → 71 →
72 (Llama Guard) and 1 → 4 → 16 → 39 → 46 → 46 (WildGuard) as the read tightens.
§5.8 asserts that the permissive point is the conservative choice for that
particular claim, on the argument that tightening moves prompts out of
*decoded*. The sweep shows exactly that, by a factor of nine.

### What is genuinely still gated, stated precisely this time

Only the wrapper/benign factorial (con 8). WildGuard's benign arm is local
(`guard-benign-wildguard_…9010639`); Llama Guard's is not, and no local run
record contains a wrapper arm at all. That is a real gap and it needs its own
job id identified rather than a guessed one.

### The lesson

This is §15 and §16's check pointed at a POINTER instead of a count: a blocking
claim is a claim, and it recomputes. Five items sat behind one id for a day, the
sentence naming it was contradicted by another canonical doc in the same repo,
and the resolution cost one grep. **Before recording work as blocked on an
artefact, confirm the artefact is the one the work needs** by reproducing one
number the work depends on.

## 18. THE HOUSE NAME FOR THE QUANTITY IS THE HARM GAP, IN BOTH PAPERS (2026-08-21)

AS-5 retired the word *discrimination* (peer session, owner go, `8a4fd1c`); its
quantity is **the harm gap**. AS-6 was using *discrimination* for the identical
quantity, defined in its own text as harmful block rate minus benign block rate
(WildGuard 0.99 − 0.45 = 0.54 against Llama Guard's 0.75).

Two occurrences, both AS-6's own measurement rather than a citation, and *gap*
appeared nowhere else in the paper, so the rename was two lines per kit and
load-bearing nowhere. Now zero occurrences of `discriminat` in either AS-6 kit.

**Checked and clear: AS-6 does not cite AS-5 at all.** So the sibling's retitle
and its restructure from three legs to one claim with four demonstrations do not
reach this paper. The vocabulary was the only coupling, which is the sort of
coupling that survives precisely because nothing references it.

Standing rule for both papers: **harmful rate minus benign rate is the HARM GAP.**
*Discrimination* is retired estate-wide, and it was the worse word anyway, since
it collides with the fairness sense in a safety paper.

## 19. THE LEDGER NOW CHECKS THE POINTER, AND FILLING AS-6's ENTRIES FOUND ONE MORE STALE BOUND (2026-08-21)

Owner delegated the call ("you decide"). Built as an extension of the peer's
`scripts/claim_sets.py`, not as a new tool: the improvement loop executes at
existing levels.

### The provenance field

A ledger entry may now carry **`source:`**, a glob under `outputs/`, and the
recomputation opens THAT artefact. `_resolve_source` raises when the field is
missing or matches nothing, so a provenance claim with no artefact to open is a
hard failure rather than a skip. Two new suite invariants stop the exemption from
becoming a hiding place: an entry with `check: internal` is excused from `locate`
**only if** it declares a `source`, and a `source` may not be absolute or repeat
the `outputs/` prefix.

`as6_table2_provenance` is the first consumer and it is the direct fix for §17:
it parses Table 2 **from the kit**, opens the named artefact at the named
percentile, and compares. 12 cells across both kits, all agree. Had it existed
yesterday, "Table 2 comes from run `9033528`" would have failed at `pytest` time
instead of costing a day.

### ⛔ Filling the AS-6 entries found §5.8's bound sourced from a withdrawn cell

The ledger forced the question *which set is "clears our screens"?* and the two
readings differ:

- Under the **control floor** alone, the max includes Llama Guard `fullwidth` at
  5 per 100, which is where the paper's "at most 5" came from.
- Under **all screens**, `fullwidth` is `(U)`, withdrawn by the item holdout, and
  the max over the six pairs the paper actually reports is **1 per 100**
  (WildGuard `reverse_words`).

So the bound was quoted from a condition the paper reports as unmeasured. It was
true and loose in the safe direction, which is why nothing caught it, and a
referee asking "which pair hits 5?" would have found the withdrawn one.
**Corrected to 1 per 100 over the reported pairs**, and the exceedance sentence
now names both pairs above it, `combining_marks` at 8 and `fullwidth` at 5,
with the observation that **every exceedance sits in the unmeasured band** —
which is the stronger statement, since that is exactly the set where the
decoded/not-decoded split is unavailable.

`as6_bwd_max_reported` parses the reported set from Table 2 rather than
restating it, so the bound now tracks the table. That is the general defence
against this whole family: **derive the set from the artefact the claim ranges
over, never from a list written beside it.**

### Verified by mutation, both directions

Wrong percentile → 8 mismatches, exit 1. Missing `source` → hard failure naming
the field. Wrong source artefact → initially a raw `KeyError` traceback, which is
a defect in a checker whose entire purpose is wrong pointers, now a diagnostic
naming the artefact kind. The two new suite invariants each fail on their own
mutation and pass restored. A screen that only ever passes is a verdict with a
script attached.

Recomputed values, all agreeing with both kits: `D&¬B` provenance 0 mismatches ·
max reported bound 1 · zero cells 26 · measurable pairs 29 · unlicensed pairs 9,
the last raising unless every one of their block rates is 0.00, because §5.8's
containment argument rests on precisely that.

## 20. THE WRAPPER SCREEN'S RECORDS ARE ON SCRATCH, WHICH NO SYNC PATH REACHES (2026-08-21)

§17 asked for the wrapper/benign job id to be found by opening records rather
than trusting a sentence. Done, and the answer is worse than "not synced yet".

**The jobs are `9049076` / `9049077`** (2026-08-10, 12 and 11 conditions x 100 x
3 arms, one H200 each, $0), and `phase1_map.md` §0.6.2 records in its own second
line: *"Records on `/scratch`; this section is the result."*

**The standard down-sync covers `~/projects/model_internals_safety/outputs/`.
`/scratch` is not under it.** So these records were never going to arrive from
any sync anyone ran, and TODO 79's "needs the wrapper/benign records
down-synced" was unachievable as written, for a second and different reason than
§17's misidentified id. Scratch filesystems are periodically purged; the run is
eleven days old.

### What in the paper depends on them, enumerated rather than estimated

Verified by checking each against local artefacts:

| claim | local? |
|---|---|
| Table 2 `D&notB` + AUROC | ✅ `operating_point_*` + `as6_split_half_transfer` |
| Table 3 block rates | ✅ `as6p1-full-*/results.json` |
| §5.8's bound, 26/29/9 pairs | ✅ `as6p1-full-*/results.json` |
| WildGuard benign ENCODED arm (0.05–0.29) | ✅ `guard-benign-wildguard_…9010639` |
| **the whole wrapper subsection** (0.44 vs 0.25; the 8–16% and 7–33% wrapper terms; "every other surviving condition clears") | ⛔ scratch |
| **plaintext BENIGN ceilings** (Llama Guard 0.23, WildGuard 0.45) and the harm gaps 0.75 / 0.54 built on them | ⛔ scratch |
| **Llama Guard benign encoded arm** (0.29–0.53) | ⛔ scratch |
| Table 2 caption's "excluded by the wrapper screen below" | ⛔ scratch |

`plain_block_rate` in the local records is the HARMFUL arm only (0.98 / 0.99,
both matching the paper), so the ceilings look locally backed and their benign
halves are not.

### Why this is the sharper version of §17's lesson

§17 was a pointer naming the wrong artefact. This is a pointer naming a REAL
artefact **in a location nothing collects**, recorded honestly at the time by a
session that wrote "records on /scratch" and moved on. A provenance field that
only checks *does the named artefact reproduce the number* would pass here by
raising "not found", which is correct behaviour and still leaves the paper
carrying a subsection whose evidence is one purge away from unreproducible.

**⛔ THE RULE FIRST WRITTEN HERE WAS WRONG AND IS CORRECTED BELOW (§21).** It
read: *a run whose records are written outside the collected path is not a
completed run*. The records were not in a wrong place. **The collector was
pointed at the wrong place**, and the true rule is the opposite in direction:
when a record is missing, check what the collector actually covers before
concluding the record is not there.


## 21. THE COLLECTOR WAS POINTED AT THE WRONG TREE, AND THE WRAPPER SUBSECTION IS NOW FULLY BACKED (2026-08-21)

The owner ran the scratch listing. **Everything survived**, and the cause is one
fact that explains all three provenance defects of the evening.

### The cause

The cluster's outputs live at `/scratch/<user>/internals_safety_outputs/`. The
configured down-sync site points at `~/projects/model_internals_safety/outputs/`.
**Both trees exist and both hold content**, so the sync kept succeeding and kept
delivering a partial mirror. Every GPU run wrote to the scratch tree and was
invisible to it; the CPU jobs that wrote to the project tree arrived fine, which
is why nothing ever looked broken.

Two sync sites added (scratch `runs/` and `analysis/`, excluding `activations/`),
so future GPU runs land automatically.

### What arrived, including two records nobody knew were missing

`guard-scaffold-{llama-guard,wildguard}_…{9049076,9049077}` (the wrapper runs),
`guard-benign-llama-guard_…9012160` (Llama Guard's benign arm), a SECOND
WildGuard benign run `…9012159`, and `guard-causal-llama-guard_…9033528` —
whose directory name confirms §17 against the cluster rather than against two
docs disagreeing.

### Every wrapper claim reproduces

Verified against the arrived records, not against `phase1_map.md`:
Llama Guard plaintext 0.98 harmful / **0.23** benign, harm gap **0.75**;
WildGuard 0.99 / **0.45**, gap **0.54**; WildGuard `combining_marks`
scaffold-benign **0.44** against an encoded-harmful block of **0.25**. The two
missing fields were `plain_benign_block_rate` and `scaffold_arm`, absent from
every record we had held.

### ⛔ And the fourth set-membership defect, which is also a circularity

> The two guards also differ systematically on benign encoded content (one
> blocks 0.29--0.53 of it, the other 0.05--0.29)

**The two ranges are over different sets.** Llama Guard's `0.29–0.53` includes
`fullwidth` (0.53, its maximum); WildGuard's `0.05–0.29` excludes `fullwidth`
(0.00, which would be its minimum). Recomputed over identical membership:

| set | Llama Guard | WildGuard |
|---|---|---|
| all five surface conditions | 0.29–0.53 | **0.00**–0.29 |
| excluding `fullwidth` | **0.29–0.42** | 0.05–0.29 |
| the three reported conditions | 0.29–0.39 | 0.23–0.29 |

WildGuard's published number matches the fullwidth-excluded row and Llama
Guard's matches the all-five row, so one guard's figure was never recomputed
under the other's exclusion. **Excluding `fullwidth` is the correct set for a
second reason the paper had not stated: `fullwidth` is the condition the
sentence exists to explain, so including it makes the argument rest on its own
conclusion.** Corrected to 0.29–0.42 against 0.05–0.29, with the exclusion and
its reason stated in the text. The contrast survives.

### Where this leaves the evening

Four defects of one class, all in claims whose individual numbers were correct:
a bound over a partly unmeasured set (§15), a warrant that did not follow from
its counts (§16), a bound sourced from a withdrawn condition (§19), and now two
ranges over different sets (§21). The mechanical ledger catches the first and
third. It cannot catch the second, and it would not have caught this one either,
because both endpoints are real numbers from the real artefact and only the
MEMBERSHIP differs. **The check that finds this family is not a test, it is the
habit of asking which set a number ranges over, every time one is written.**

## 22. CON 8 IS DONE, AND THE FACTORIAL MAKES THE WRAPPER CLAIM BOTH WEAKER AND MORE USEFUL (2026-08-21)

`scripts/guard_factorial.py`, keyless, GPU-free, seconds. Artifact
`outputs/analysis/guard_factorial_20260821.json`. Source: the `guard-scaffold-*`
runs that arrived in §21.

### The result

| guard | condition | gap plain | wrapped | encoded | Δ wrapper | Δ encoding beyond wrapper |
|---|---|---|---|---|---|---|
| Llama Guard | homoglyph | 0.75 | 0.63 | 0.53 | 0.12 [−0.01, 0.25] | 0.10 [−0.05, 0.25] |
| Llama Guard | zero_width | 0.75 | 0.67 | 0.54 | 0.08 [−0.05, 0.21] | 0.13 [−0.02, 0.28] |
| Llama Guard | reverse_words | 0.75 | 0.68 | 0.36 | 0.07 [−0.06, 0.20] | **0.32 [0.16, 0.48]** |
| WildGuard | homoglyph | 0.54 | 0.36 | 0.52 | **0.18 [0.04, 0.32]** | **−0.16 [−0.31, −0.01]** |
| WildGuard | zero_width | 0.54 | 0.43 | 0.48 | 0.11 [−0.03, 0.25] | −0.05 [−0.21, 0.11] |
| WildGuard | reverse_words | 0.54 | 0.50 | 0.44 | 0.04 [−0.10, 0.18] | 0.06 [−0.10, 0.22] |

### ⛔ The published percentages were point estimates straddling zero

The paper said *"the wrapper accounts for 8 to 16 per cent of Llama Guard's
plaintext harm gap and 7 to 33 per cent of WildGuard's."* Those are the Δ wrapper
column as fractions of each guard's plain gap, and **five of the six intervals
include zero.** Only WildGuard `homoglyph` separates. A percentage range reads as
a measured decomposition and this one is mostly not distinguishable from no
effect at all.

**The correction makes the paper's own argument stronger.** Its conclusion is
that neither guard is principally a format detector, and "the wrapper term is
small and mostly indistinguishable from zero" says that far better than a
percentage range that invites the reader to take 33% seriously.

### The surprising cell, which a marginal comparison could not show

**WildGuard `homoglyph`'s encoding term is NEGATIVE and separates: −0.16
[−0.31, −0.01].** Relative to the same content wearing the same wrapper, the
encoding *recovers* discrimination (wrapped gap 0.36, encoded gap 0.52). The
wrapper hurts this guard more than the encoding does. Reporting a difference of
marginal rates against a bar would have shown neither this nor Llama Guard
`reverse_words`' +0.32, which is the referee's methodological point landing on
real cells rather than in principle.

### ⚠️ The intervals are CONSERVATIVE, and the reason is a data limitation

Conditions within an arm are the same prompts rendered differently, so they are
item-paired, but **the wrapper runs persisted per-item verdicts for the encoded
harmful arm only** (`benign_cells.jsonl` is empty in every one of them). Five of
six cells survive as aggregate rates, so `unpaired_interaction_interval` treats
all four as independent and inflates the width by the shared item difficulty.
Consequence, stated in the module docstring, the script header and the table
caption: **a cell that separates here does so; a cell that does not is NOT shown
to be null.** Recovering the pairing needs a re-run with per-item persistence on
every arm, not a different formula.

The estimator went to `intervals.py` rather than into the script, per the spine
rule, and is pinned by four tests including one asserting it is genuinely wider
than the paired bootstrap on identical data. That property is its whole
justification, so it is tested rather than claimed.

## 23. THE BLOCK AXIS NOW HAS A LENGTH CONTROL, AND IT HAD TO BE MEASURED IN TOKENS (2026-08-21)

TODO 88, answered without the run it was filed as needing, and the answer is
clean: **length cannot account for any of the six harm gaps the paper reports.**

### The gap the existing controls left

AS-5 measured raw character length separating the harmful from the benign corpus
at AUROC 0.6544 (86.0 characters against 73.8) and every encoder in the ladder is
monotone in length, so the separation survives into every condition. AS-6 reports
a block-rate gap between the arms and reads it as harm sensitivity. Nothing in
the paper had constrained that reading for length:

* the **length-matched permutation null** licenses *probes*, and an AUROC is the
  only quantity it reaches;
* the **benign arm** holds the encoding fixed and varies harm;
* the **wrapper arm** holds the content fixed and varies the template.

Every one of those leaves the two arms' lengths free to differ exactly as the
bare corpora do. This is the same shape as the repo's recurring defect: a rule
that reaches some callers and not the one that matters.

### ⛔ The filed blocker was wrong, and the first correction of it was also wrong

Item 88 said the fix was gated on a down-sync of run `9033528`. After the scratch
tree arrived I recorded it as unblocked. **Both were wrong, and the second one
was written from the file listing rather than from the files.** Opening every
AS-6 record shows there are **zero per-prompt benign verdicts on disk**: all
three `benign_cells.jsonl` are 0 bytes, and every `cells.jsonl` in both guards'
19 run directories is the harmful arm, `guard-benign-*/cells.jsonl` included, so
the name records the run's PURPOSE and not its file's contents. The benign arm
survives only as the `benign_arm.benign_block_rate` summary. `measure_rate_length_null`
needs per-prompt flags on both arms, so it needs a re-run.

**The lesson is item 99's, applied in the direction it was not written for.** Its
rule is *before recording work as BLOCKED on an artefact, reproduce one number
from the artefact you claim to need*. The same rule binds in reverse: before
recording work as UNBLOCKED, open the artefact. A directory listing is a claim
about names.

### The bound that IS computable, and why it is stronger where it counts

The corpora and the encoders are local and deterministic, so both arms'
ciphertexts regenerate exactly. That gives the two length distributions for free.
Fix the blocking budget to the number of prompts the real guard blocked across
both arms, and hand that budget to the best **monotone** length-only rule (block
the longest, or the shortest, whichever separates more, ties resolved in favour
of the bound). Its harm gap is the ceiling on what a guard reacting only to
length could show at this guard's own operating rate:

* observed gap **above** the bound: length cannot account for it. Conclusive.
* observed gap **below** the bound: inconclusive, and it needs the re-run.

One-directional on purpose. A control that can only ever fail to reject is a
verdict with a script attached.

**Monotone is a real restriction and it is stated rather than hidden.** The
unrestricted optimum over all functions of length fits a 200-item sample almost
perfectly (lengths are near-unique, so "block the lengths where the harmful
fraction is highest" separates nearly completely) and would bound near 1.0 on any
corpus whatsoever. That number would describe the corpus's memorability, not a
guard. The confound reported in the literature is monotone. A test pins that a
non-monotone rule CAN beat this bound, so the scope lives in the suite and not
only in a docstring.

### ⚠️ CHARACTERS WERE THE WRONG UNIT ON HALF THE CELLS

The referee's worry and AS-5's 0.6544 both name character length, but a guard
processes tokens, and the published confound (arXiv 2605.00269) is stated over
sequence length. The two come apart precisely where it matters: `fullwidth` has
the same character count as its plaintext and roughly 2.5x the tokens. So the
bound is computed in both units and the **larger** is binding.

It changed the answer on three of the six reported cells:

| Guard | Condition | Gap | Char bnd | Token bnd | Margin | Binds |
|---|---|---|---|---|---|---|
| Llama Guard 3 | `homoglyph` | 0.53 | 0.21 | 0.21 | **+0.32** | char |
| Llama Guard 3 | `zero_width` | 0.54 | 0.20 | 0.22 | **+0.32** | token |
| Llama Guard 3 | `reverse_words` | 0.36 | 0.22 | 0.22 | **+0.14** | char |
| WildGuard | `homoglyph` | 0.52 | 0.22 | 0.26 | **+0.26** | token |
| WildGuard | `zero_width` | 0.48 | 0.22 | 0.22 | **+0.26** | char |
| WildGuard | `reverse_words` | 0.44 | 0.22 | **0.34** | **+0.10** | token |

**All six clear.** WildGuard `reverse_words` is the case that justifies the whole
token arm: its token bound exceeds its character bound by 0.12, which is *more
than the 0.10 margin that survives it*, so a character-only control would have
reported the thinnest cell in the table as comfortable.

⚠️ **The two units disagree because a rate gap at a fixed budget is not a
monotone function of the separation AUROC.** WildGuard `reverse_words` moves
0.654 to 0.663 in AUROC between units and 0.22 to 0.34 in bound: what matters is
the shape of the two distributions *at the cut*, not their overall overlap. This
is a further reason not to report a length AUROC as the control for a rate, which
is what the probe-side null does and what this replaces.

### The budget-matching choice is not load-bearing

Fixing k to the guard's own blocked count is the right comparison, because the
confound at issue is that THIS guard's decisions are partly length and its rate
is observed. A referee can still ask what happens without the constraint, so the
strictly more conservative bound is computed and recorded rather than argued
about: the maximum over EVERY budget in either unit is **0.28 on five of the six
cells and 0.38 on WildGuard `reverse_words`**, and all six still clear. The
tightest case stays WildGuard `reverse_words`, at 0.44 against 0.38.

### What licenses regenerating an arm at all

"The encoders are deterministic" is a claim about code, and this repo has been
wrong about such claims before. So the script re-encodes the **harmful** arm too
and requires it to reproduce the on-disk ciphertext for every prompt byte for
byte before reporting anything, and it fails closed on a corpus-digest mismatch
against the run record. The harmful arm is the only arm whose ground truth is on
disk, so it is the only one that can carry the check, and passing it is what
makes the benign regeneration trustworthy by the same code path.

### One thin cell worth watching

WildGuard `combining_marks` clears by **+0.02**, the thinnest in the run, and it
is the same cell that clears the control floor by +0.003. It is not in the
reported set, and two independent screens agreeing that it is marginal is worth
more than either.

Instrument: `scripts/guard_arm_length_bound.py` (keyless, GPU-free, seconds),
artifact `outputs/analysis/guard_arm_length_bound_20260821.json`, pinned by
`tests/test_guard_arm_length_bound.py` (11 tests, ceiling property checked over
every threshold rule in both directions) and by the ledger entry
`as6_length_bound_clearing_cells`, which RAISES rather than quietly returning a
smaller number if any reported cell stops clearing. Paper: a new Controls
paragraph and a new specification subsection with the table, both kits.

## 24. THE ABSTENTION WAS UNDERPOWERED, AND IT SURVIVES ANYWAY FOR A BETTER REASON (2026-08-21)

TODO 89. Item 89's own instruction was the right one and is worth restating: *do
not relax the abstention by assuming a paired test would be stronger, run one.*
Running it produced a two-part answer, and only the first part was anticipated.

### Part 1: the stated reason was wrong

Table 2's caption refused to order the surviving conditions because their Wilson
intervals overlap. The conditions are the same 100 harmful prompts wearing
different transformations, so they are item-paired and an independent interval
throws the pairing away. Under exact McNemar with Holm adjustment within each
guard, **3 of the 6 pairs separate, and none of the 6 separates under the
independent intervals**, so every separation is one the published reasoning could
not see:

| Guard | Pair | Counts | Discordant | Difference [95%] | Holm $p$ |
|---|---|---|---|---|---|
| Llama Guard 3 | `homoglyph` vs `zero_width` | 7/17 | 1/11 | −0.10 [−0.17, −0.04] | **0.019** |
| WildGuard | `homoglyph` vs `reverse_words` | 23/9 | 20/6 | +0.14 [+0.04, +0.24] | **0.019** |
| WildGuard | `reverse_words` vs `zero_width` | 9/23 | 4/18 | −0.14 [−0.23, −0.05] | **0.013** |

The three that do not separate are genuine: Llama Guard `homoglyph` vs
`reverse_words` is 7 against 8, and WildGuard `homoglyph` vs `zero_width` is 23
against 23 at McNemar $p = 1.0$.

### ⛔ Part 2: the ordering it produces is the BLOCK-RATE ordering

`decoded_not_blocked` is a conjunction, and whether an ordering of it says
anything about decoding depends on how close the decode term is to one. At the
reported operating point it is:

| Guard | Condition | Decode rate | Block rate | Unblocked | D&¬B recovers |
|---|---|---|---|---|---|
| Llama Guard 3 | `homoglyph` | 0.99 | 0.92 | 8 | 88% |
| Llama Guard 3 | `zero_width` | 0.99 | 0.83 | 17 | 100% |
| Llama Guard 3 | `reverse_words` | 0.69 | 0.65 | 35 | 23% |
| WildGuard | `homoglyph` | 0.96 | 0.75 | 25 | 92% |
| WildGuard | `zero_width` | 0.94 | 0.71 | 29 | 79% |
| WildGuard | `reverse_words` | 0.78 | 0.73 | 27 | 33% |

**On the four surface cells the decode term is not the binding one.** The count
recovers 79 to 100 per cent of the unblocked prompts, so separating two of them
largely restates Table 3's block rates. `reverse_words` at 23 and 33 per cent is
the contrast proving this is a property of those cells and not of the statistic.

**So the abstention stays, on the different ground.** The paper now says the
paired test separates three pairs and that we still decline to order the
conditions by this count, because it is not a quantity that can rank them.

⚠️ **This is good news for the claim and bad news only for ordering.** A decode
term of 0.94 to 0.99 means the failure to block cannot be attributed to
comprehension on *any* prompt at those conditions, which is the strongest form
the paper's actual thesis can take. What it cannot do is discriminate between
conditions, and the old caption was reaching for exactly that.

### ⚠️ The paper's first draft of this overstated it, and a test caught it

I wrote "the count is within one or two prompts of the unblocked count there".
True on three of the four cells and false on WildGuard `zero_width` (23 of 29).
`tests/test_paired_cell_ordering.py` asserted the claim as a number and went red
before the sentence shipped. The published form is the range, 79 to 100 per cent.
*A counted claim written as a qualitative phrase is still a counted claim*, and
this one was checkable only because the marginals were persisted to the artifact
rather than printed.

### What licenses the comparison at all

`scores-b10` was produced at `reading_percentile` 50, which is retired; the paper
reports 75. The per-prompt read is reconstructed as `decode_score > threshold_75`
from `operating_point_*_20260808.json`, and **the script refuses to compare
anything until that reconstruction reproduces all six of Table 2's published
counts exactly.** It does. The check is against the PAPER's numbers, held as a
literal, rather than against a second recomputation, because a self-check that
recomputes both sides agrees with itself at any threshold.

Instrument: `scripts/paired_cell_ordering.py`, artifact
`outputs/analysis/paired_cell_ordering_20260821.json`, estimators
`paired_bootstrap_rate_difference` and `holm_adjusted` in `intervals.py` (spine,
not the script; the first is pinned by a test asserting it is narrower than
treating the arms as independent, which is its whole justification). Ledger
entries `as6_paired_separating_pairs` and `as6_paired_total_pairs` cover the
numerator and denominator separately, since a sentence reading "three of the six"
goes stale from either end; the first also raises if an independent interval ever
starts separating, which would make the paper's contrast between the two tests
false. Paper: Table 2's caption and a new specification subsection with the
pairwise table, both kits.

## 25. WILDGUARD'S FLOOR WAS BUILT FROM 11 OF ITS 14 CONTROLS, AND THE THREE MISSING ONES WERE ALREADY MEASURED (2026-08-22)

TODO 86, closed, and it turned out to be doable locally rather than on the
cluster as filed.

### What was wrong

`instrument_layer.md` §2.7.1 recorded 14 controls at 0.6803, while the screen of
record (`as6_floor_wildguard_split/unsplit`) used **11**, because the cluster
invocation passed a single `--ability-cells` file. `base64`,
`reverse_characters` and `tag_block` all have measured Mistral-7B-v0.3 ability
**0.00** in `dissociation-mistral_…9010897`, which has been on disk since
2026-08-08. The floor was internally consistent with the controls it was given,
so nothing in the build could notice.

### The adopted values, and what does not change

| Treatment | 11 controls | 14 controls | Delta |
|---|---|---|---|
| unsplit | 0.6852 | **0.6803** | −0.0049 |
| item-split (screen of record) | 0.6617 | **0.6605** | −0.0011 |

**No verdict differs on either treatment**, and that is verified rather than
assumed: no rung's AUROC lies in the band between the two floors, and the
per-condition `clears_floor` maps are identical. The 14-control floor is the
more permissive one, so the previous screen was the stricter of the two, which
is why this is bookkeeping and not a correction.

⚠️ **It is not free of consequence for the paper's printed numbers, and three
had to move**: the caption's WildGuard floor 0.662 → **0.661**; the control
count 11 → **14**; `combining_marks` clears by 0.003 → **0.004**; and the
floor's own movement under the holdout, 0.024 → **0.020** on WildGuard. A change
that "changes no verdict" still changed four figures in two kits.

⚠️ **One thing improves.** WildGuard's sigma window widens from [1.577, 2.082]
to **[1.580, 2.124]**, so the configured 2.0 sits further inside it. Llama
Guard's remains [2.113, 3.440] with 2.0 **below the lower bound**, which is the
pre-existing defect and is untouched by this.

### Llama Guard needed nothing, and that was checked rather than assumed

Its 12 controls are already every condition Llama-3.1-8B decodes at 0.00: the
ten inert ciphers plus `reverse_characters` and `tag_block`. `hex` and
`unicode_escape` are Llama-readable (0.84 and 0.54) and are correctly excluded.
Fixing one guard and leaving the other asymmetric would have been the more
likely error here.

### The guard against a third occurrence

The control count is now a ledger claim, `as6_wildguard_floor_controls`, which
recomputes it from the artifact and raises if the floor stops being a
distribution. This is the class the ledger exists for: a number that was
internally consistent, printed for days, and wrong only relative to data already
sitting on disk.

### ⚠️ That guard was HALF a sentence, and the supersession did not reach the mirrors (2026-08-22, same day)

Three things found by sweeping the AS-6 kit for counted claims with no ledger
entry, rather than by reading the sentences again.

**(a) One sentence, one half checked.** The Method reads "distributions, over 12
controls for Llama Guard 3 and 14 for WildGuard". The ledger located `and (\w+)
for WildGuard` and nothing else, so **half a checked sentence was reading as a
checked sentence**. Llama Guard's 12 is correct and was unverified; the recipe
is now `as6_guard_floor_controls`, source-driven, with one entry per guard.

**(b) The count can hold still while the VALUE drifts.** 11 controls and 14
controls are different numbers and a count check sees that. A floor value can
move with the control set unchanged, and the value is what a screen actually
compares against, so `as6_guard_floor_value` covers Table 2's caption for both
guards. It compares at **the paper's own printed precision**, read from the
captured token rather than from a setting: `0.6617` against `0.6605` rounds away
at three places and not at four, and a tolerance knob here would be a knob on
how wrong a printed number may be.

**(c) The mirrors kept the superseded pair for a day.** The paper and this
record were both correct within the hour. `CLAUDE.md` was not: it carried "an
**11-control distribution at 0.6852**" and "0.6852→**0.6617** (WildGuard)", plus
two figures downstream of the same supersession (`combining_marks` clearing by
0.003 rather than 0.004, and WildGuard's sigma window as [1.577, 2.082] rather
than [1.580, 2.124]). `NOW.md` carried the same floor pair, and this document's
sibling `phase1_map.md` still had a "reconciled rather than left to be found"
block asserting the screen of record used 0.6852 over 11 and that re-deriving on
fourteen was "filed, not done" — written before the re-derivation it was
describing had run.

**Why this matters more than the paper case, and it is the reason the fix went
into code rather than into another warning.** A paper has referees. `CLAUDE.md`
is loaded into every session in this repo and has none, so a superseded number
there does not sit in one document waiting to be caught, it propagates into
whatever the next session reasons about. The ledger therefore takes an optional
`mirrors:` list per claim — a repo file and its own locating regex, since a
mirror phrases the claim its own way — and checks it alongside the kits, at
whatever precision that surface printed.

**One thing the mutation test caught about the guard itself.** The first mirror
anchors read `floors moved only 0.7098→**(...)** (Llama Guard)`. That put a
second copy of a value in the ledger, which rule 2 forbids, and the consequence
showed up immediately under mutation: restoring the historical error made the
guard report *locate matched 0* rather than a value mismatch. It fired, for the
wrong reason, and the reason would have disappeared the moment the copied value
was the thing that drifted. Anchored on the guard name alone, the same mutation
now reports `CLAUDE.md (mirror): asserts 0.6617`. **A guard that fires is not
yet a guard that fires for the right reason**, and only mutation separates them.

### The rest of the sweep: two more encoded, one refused, one trap

**Encoded, both from `guard_factorial_20260821.json`.** *Five of the six
reported cells have a wrapper-alone interval including zero* — a count that
falsifies the paper's "the wrapper term is small" by falling and merely
strengthens it by rising. And *the encoding term beyond the wrapper separates on
two cells in opposite directions*, where the recipe **raises unless the two
disagree in sign**: a cardinality check alone would pass two same-direction
cells, leaving a correct number under a false claim. Mutation-verified by
flipping WildGuard `homoglyph`'s term positive, which holds the count at two and
makes the sentence false; the guard fires.

⛔ **Refused: "six of the seven conditions we would otherwise have reported still
clear their re-derived floor."** The set is defined by an EDITORIAL decision, not
by the artefact. Rebuilding it from disk gives nine unique cells clearing the
unsplit floor, which becomes eight once `caesar3` is dropped as the control it
is, and eight is not seven: the ninth-to-eighth step is a screen and the
eighth-to-seventh step is the judgement that WildGuard `combining_marks`, which
clears by 0.004 with a band straddling the floor, is reported *as passing rather
than as established* and therefore is not among the cells we "would otherwise
have reported". A recipe would have to hard-code that judgement, which is a
second copy of an editorial choice living where rule 2 forbids a second copy of
a value. **The count is correct** — the one cell that falls is Llama Guard
`fullwidth`, 0.7057 against 0.7066 — and it stays checked by hand.

⚠️ **The trap, for whoever writes the next recipe over that artefact.**
`as6_split_half_transfer.json` holds **53 records over 38 unique (model, family)
pairs**, from three run sources including a `scores-b10` bin variant. All 15
duplicated keys agree to six decimals, so nothing there is contradictory — and a
recipe that counted records rather than distinct cells would still report 53
where the answer is 38, silently and with every value correct.

## 26. TODO 81's TWO "COSTED BASELINES" COST MINUTES, AND ONE OF THEM NEEDS NO GENERATION AT ALL (2026-08-22)

The item said "neither is $0", which was true and read as though these were the
expensive follow-ups. Costed properly against measured wall clock, they are the
cheapest substantial work left on this paper.

### The anchor is measurement, not the estimator

Nineteen completed guard runs are on disk with `elapsed_seconds` and a condition
count. Per condition per guard: **0.42 to 0.60 minutes**, and the spread is
tight across both guards and every run shape. A full 19-condition sweep is **8
to 11 minutes**. That is an empirical rate for exactly the job shape in
question, so it does not need the cost model's fitted token rate, which is
calibrated on generating runs and would over-predict here.

Money is **$0** for both. Guard verdicts are read from logits, so there is no
judge call and no API spend, and the partition is free. `as6_guard_probe
--dry-run` confirms it independently: 15,400 forward passes, 0 generations, 0
judge calls, $0.00.

### Baseline B is cheaper than filed, because the decode is already on disk

DecipherGuard's pipeline decodes a prompt before classifying it. Ours would need
a decoder, and the phase-0 cells **already persist `restate_response`** for
every encoded prompt: the base model's own decode attempt, generated and stored
back in August. So the pipeline needs **no generation at all**, only guard
forward passes over cached text.

**~15 minutes per guard, two jobs, $0.**

It is also the more valuable of the two. Con 10 asked whether our licensing
framework improves validity or only moves our own numbers, and this answers it
by an **independent external route**: if the external decoder repairs exactly
the conditions our internal read calls decoded, and fails exactly where we
report `(U)`, the partition is corroborated by a method that shares none of our
machinery. If it repairs a condition we call `(U)`, that is a real finding
against us and worth knowing.

### Baseline A is a refit whose run is not the expensive part

The guard capture is `layers: all` at two positions, so a multi-depth fused
probe needs **no new forward pass** if the activation cache is warm on the
cluster; if cold, re-capture is the same ~10 minutes per guard.

⛔ **The run is not what makes it expensive.** The paper already states, in its
own related-work section, that we adopt the single-cell read because the
permutation null and the control floor are constructed to correct for a selected
maximum, and that a fused multi-depth statistic **would need both rebuilt**. So
baseline A is a screens rebuild with a short job attached, not a job.

⚠️ **It shares its blocker with TODO 96.** Both need the guard activations.
Measured from the local cache, one `(arm, condition)` file is 100 MB, so ~3.8 GB
per guard and **~7.6 GB for both**. One transfer unblocks the item-level
per-prompt recount and the multi-depth baseline together.

### Recommendation

**Do B, defer A.** B is minutes, needs no generation, answers the referee's
actual question by an independent route, and requires no change to any screen. A
requires rebuilding the two screens that give this paper its contribution, which
is a larger decision than its 10-minute job suggests, and it is gated on a
transfer in any case.

## 27. BASELINE B IS BUILT, TESTED AND BLOCKED ON A RUN IN THE OTHER LANE (2026-08-22)

Con 10's second baseline, `scripts/decoder_guard_pipeline.py`, is written and
pinned. It cannot launch yet, and the reason is worth more than the script.

### What it measures

DecipherGuard repairs the encoded-prompt failure by decoding a prompt before
classifying it. We run that pipeline on our ladder and ask whether its
per-condition repair agrees with our internal partition. **It generates
nothing**: `phase0_regime_map` persists `restate_response`, the base model's own
restatement of every encoded prompt, so the decode already happened in August and
this reads guard verdicts over cached text.

Six cells per condition, three payload kinds on each of two arms:

| | harmful arm | benign arm |
|---|---|---|
| **ciphertext** | what the guard sees today | |
| **restatement** | the pipeline | the pipeline's false-positive cost |
| **plaintext** | the ceiling a perfect decoder reaches | |

The statistic is repair against the **headroom** the plaintext ceiling leaves, so
a condition the guard already blocks at its plaintext rate cannot post a large
repair merely by having nowhere to fall. It is `None`, never `0.0`, when the
headroom is zero, and a test pins that: "the pipeline did not help here" is a
measurement, and the truth at zero headroom is that the quantity does not exist.

### ⛔ It refuses to run without the benign arm, and there is no flag

**No benign restatements exist on disk.** Every `benign_cells.jsonl` in the
estate carries `arm`, `refused` and `attack_response` and no `restate_response`.
That is defect (11), which the peer session closed in the entrypoint on
2026-08-21; no run has used the new code yet.

A harmful-only version would be cheap and would measure a repair **that a guard
blocking everything would also produce**. That is the same defect this repo has
found three times on its own block rates. So the script fails closed, names the
missing arm and the preset that would produce it, and exits. A benign arm that
can be switched off is a benign arm that was off when it mattered.

### The dependency, stated so it does not cost a queue cycle

`conf/experiment/comprehension_gap_{llama3_1_8b_instruct,mistral_7b_instruct}.yaml`
are already written, target exactly the two guards' base models, and cover
exactly the four rungs the pipeline presets name. **Those runs must land first.**
A test asserts each pipeline preset's rungs are a subset of ITS OWN decoder's gap
preset, checked per decoder rather than once, because the fail-closed check trips
for the whole run and one uncovered rung wastes the job rather than degrading it.

### What the end-to-end test found before any GPU

`args.seed` did not exist. `add_common_arguments` deliberately does not add
`--seed`, since the seed is a config knob and `as6_guard_probe` reads it from
`measurements.probes.seed`. The script would have loaded an 8B guard, run 2400
forward passes, and died on the record write. **That is the whole argument for
writing the end-to-end test before the run rather than after it**, and it is the
fourth time this repo has paid or nearly paid for it.

### One structural fix taken rather than deferred

`PresetConfig.tasks` renders `--guard` for guard entrypoints and
`tests/test_presets.py` skips the model-config lookup for them, and the two
enumerated that set separately. Adding a second guard entrypoint made the test
reject a preset `tasks()` had rendered correctly. Both now read
`GUARD_TARGET_ENTRYPOINTS`, and a test forbids a second literal listing. Same
pattern as the length null and the control floor: the fix is not threading the
rule into another caller, it is leaving only one place the rule can live.

### The costed ask

Per guard: **2400 forward passes** (3 kinds x 2 arms x 100 prompts x 4 rungs), 0
generations, 0 judge calls, **$0.00**. Measured rate from 19 completed guard runs
is 0.42-0.60 min per condition per guard, so the work is minutes; the presets
declare a 1-hour ceiling for model load plus margin against an 8-hour wall.
