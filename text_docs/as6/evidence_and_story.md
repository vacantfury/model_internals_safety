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
- **⚠️ The map in `phase1_map.md` predates the operating-point change and must be
  re-quoted from run `9033528`.** Same guard, same four rungs, same block rates
  (0.92 / 0.83 / 0.85 / 0.65) — but `reading_percentile` moved 50 → 75 this
  session, and `decoded_not_blocked` moved with it: homoglyph 0.08 → **0.07**,
  zero_width 0.17 → **0.17**, fullwidth 0.12 → **0.10**, reverse_words 0.25 →
  **0.08**. The change lands almost entirely on `reverse_words`, the weakest
  decode probe of the four (AUROC 0.796 against 0.985 / 0.969 / 0.880), which is
  the direction a stricter read should move things and is a check on the knob
  rather than a coincidence. **Leg 3's conclusions are unaffected** — the
  ordering is unchanged and `blocked_without_decoding` stays at 0.00–0.11 — but
  any quoted cell must now name which operating point produced it.
- **Two guards, one corpus.** The dissociations are properties of two
  checkpoints. §3.6.1's lesson from the AS-5 side applies: a conclusion from one
  model is not a conclusion, and two is the minimum that makes a dissociation
  one — not the number that makes either guard's behaviour general.
- **The citation gap.** The Llama Guard *3* checkpoint cannot be cited: the Herd
  paper is absent from science's master bib and the venue bib is a generated
  subset of it (TODO 63). The draft cites the family and names the checkpoint.
- **⚠️ THE PAPER HAS NO RELATED WORK SECTION AND TWO REFERENCES** (`inan2023…`
  = Llama Guard *1*, and WildGuard's NeurIPS D&B entry — both of them the
  objects of study, neither of them prior work). Sections run Introduction →
  Scope → Method → Results → Limitations → bibliography. This is the largest
  structural gap AS-6 has and it is entirely offline work: the priors are
  already identified and deep-read, in `text_docs/as6/s1_idea_check.md` and
  TODO items 12/15. The delta argument is **already settled and must be
  written, not re-derived** — the Level-3 floor is set by SIREN (2604.18519)
  alone; Gamma-Guard (EMNLP 2025) is qualitative-only with zero encoding rungs;
  DecipherGuard (2509.16870) owns the *behavioural* "guards fail on encoded
  prompts" result and is never claimable here; Zhao et al. (2507.11878) is the
  nearest capture-position prior; Youstra et al. (2508.17158) is the nearest
  prior to the whole AS-5/AS-6 thesis. **One prior is still unchecked and
  blocks the section: arXiv 2608.03201** (refusal-cue shortcut in
  LlamaGuard3/Qwen3Guard, published 2026-08-04), filed in TODO 15 as
  "scoop-check it before AS-6 scopes further" and never run. Scoop-check it
  first; a Related Work written around an unread nearest-neighbour is the
  expensive kind of rewrite.

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
