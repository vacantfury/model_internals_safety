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
**0 overfull and 0 underfull boxes**, 6 pages — the intervals widened `tab:map` 10.9pt
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
