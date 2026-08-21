# The arm survey — which cell of the 2x2 the literature actually occupies

*Founded 2026-08-21. This is AS-5's framing evidence: the premise that the effect
we measure sits in a cell the field does not evaluate. It is a claim of ABSENCE,
so every row below is adjudicated against the paper's own full text and carries a
quote. Nothing here rests on recollection.*

**⚠️ Read §3 before quoting §2.** The clean version of this claim — "nobody has
ever done it" — is FALSE, and the founding paper of the class is the
counterexample. The true claim is narrower, better evidenced, and a stronger
motivation than the false one.

---

## 1. The design

Two literatures, crossed by what they send and in what form:

|                     | plain form                          | attack form                       |
|---------------------|-------------------------------------|-----------------------------------|
| **harmful** content | safety / capability benchmarks      | jailbreak + encoded-attack papers |
| **benign** content  | over-refusal benchmarks             | **the cell in question**          |

The survey question, applied to one paper at a time:

> Does this paper send BENIGN content through its own transformation and report a
> REFUSAL or false-positive RATE on that arm?

Four possible answers, and the distinction between the last two is where every
interesting case lives:

- **B0** — no benign arm at all.
- **B1** — benign arm in PLAIN form only; it never passes through the transform.
- **B2** — benign content DOES pass through the transform, but is scored for
  CAPABILITY (did the model decode / answer it?), not for refusal.
- **B3** — benign content through the transform, refusal or FPR reported as a rate.

B3 is the cell. B2 is the near miss that matters most, because a paper at B2 built
the corpus, applied the transform, and measured something else on it.

## 2. Method, and its two known limits

**Frame.** The science organ's `literature/llm-security/references.bib` master
(419 entries as of 2026-08-21), screened by `scripts/survey_frame.py`. The keyword
screen is recall-first and decides nothing; it proposes candidates.

⚠️ **The screen's recall is not sufficient on its own, and it proved so on its
first run.** It missed `wei2023jailbroken` — the founding paper of this entire
attack class — because that paper's title and abstract carry no transformation
vocabulary. The frame is therefore *screen results UNION a named canonical seed
set*, and any future extension must add seeds by hand. A keyword screen that
returns nothing is a failed lookup, not an absence.

**Adjudication.** `scripts/survey_adjudicate.py` extracts full text with
`pdftotext`, unwraps the column layout before splitting into sentences (a
line-split search would make strictness a function of column width), and prints
every sentence mentioning a benign corpus, flagging those that also name a
transform. The verdict is then read off the paper's own sentences.

**Positive control, and it is not optional.** An instrument that reports absence
must be shown capable of reporting presence. XSTest — a benchmark made entirely of
benign prompts — returns **72** benign-mentioning sentences. The instrument fires.

**Limit 1: the frame is one curated master, not "the literature".** Every count
below is "of the papers adjudicated", never "of all papers". **Limit 2: PDF text
extraction can miss a table-only result.** A refusal rate living only in a figure
axis would not be caught. Both limits argue for the same caution: the claim is
about what these papers *report*, in text, as a rate.

## 3. ⚠️ The founding paper had the control, and the field dropped it

**Wei, Haghtalab & Steinhardt, "Jailbroken: How Does LLM Safety Training Fail?"
(NeurIPS 2023, arXiv 2307.02483)** — the paper that introduced Base64 and friends
as a safety-training failure mode — **ran a harmless control through every attack**:

> "In the first phase, we test each jailbreak for each model against the curated
> dataset and **an additional harmless control prompt**."

and **observed the effect this paper is about**:

> "(Claude even refuses a harmless control prompt under these roleplay attacks;
> see Appendix D.)"

> "... being UNCLEAR at a high rate and **the harmless control prompts not
> succeeding**."

So the cell is not virgin territory, and any framing claiming it is will be
correctly rejected. What is true is more specific, and all three parts matter:

1. **n = 1.** "An additional harmless control prompt", singular. One prompt cannot
   produce a rate, only an anecdote.
2. **Scored as capability, not refusal.** Appendix D: "The control prompt was
   marked as successful if the model correctly answered the question (with answer
   'unicorn')." That is **B2**. Refusal is one way to fail it, never separated from
   any other.
3. **It lived in an appendix column and a parenthesis**, and **it did not
   propagate**. Not one successor paper in §4 carries it forward.

**This is a better motivation than an empty cell.** The claim is not that nobody
thought of it. It is that the founding paper saw it, recorded it at n=1 in an
appendix, and the literature built on that paper kept the attack and discarded the
control. AS-5 restores the control and turns it into a rate.

## 4. The rows

Population A — input-transformation attacks. Verdicts from full text.

| paper | venue | verdict | evidence |
|---|---|---|---|
| Wei et al., Jailbroken | NeurIPS 2023 | **B2** | harmless control, n=1, scored "answered correctly", Appendix D |
| Yong et al., Low-Resource Languages | 2023 | **B0** | 3 benign mentions, none an eval arm |
| Yuan et al., CipherChat | ICLR 2024 | **B0** | 2 benign mentions in 70k chars: an in-prompt safe demonstration, and a bibliography line |
| Jiang et al., ArtPrompt | ACL 2024 | **B0** | 1 benign mention, no transform co-occurrence |
| Ren et al., CodeAttack | ACL 2024 | **B0** | benign code is an attack INGREDIENT ("we prepend a benign quick sort algorithm ... making it closer to the training distribution"), not an arm |
| Jiang et al., Cipher Characters | NeurIPS 2024 | **B0** | 2 benign mentions, both incidental |
| Handa et al., Novel Ciphers | 2025 | **B2** | **CipherBench: "a benchmark designed to evaluate LLMs' accuracy in decoding encrypted benign text"** |
| He et al., Solving Puzzles | 2025 | **B0** | 2 benign mentions, no transform co-occurrence |
| Jiang et al., CAMO (cross-modal) | 2025 | **B0** | "benign" describes the ATTACK's fragments, not an eval corpus |
| Zhang et al., Mathematical Encoding | 2026 | **B0** | 2 benign mentions, no transform co-occurrence |
| Peng et al., Logic Jailbreak | 2026 | **B0** | sole hit describes token distributions that "appear benign", not an arm |

**Population A: 11 adjudicated, 0 at B3.** Two at B2, nine at B0.

**Handa et al. is the sharpest single exhibit in this survey.** They build
CipherBench (benign text through ten ciphers) and ACE/LACE (harmful queries through
the same ciphers). Both corpora exist; both pass through the transformation; the
benign one is scored for decoding accuracy and the harmful one for ASR. The cell
was one column away and the column was not added.

Population B — over-refusal benchmarks. Here the question inverts: does the benign
corpus ever appear in attack form?

| paper | venue | benign sentences | transform vocabulary in FULL TEXT |
|---|---|---|---|
| Röttger et al., XSTest | NAACL 2024 | 72 | **0** |
| Cui et al., OR-Bench | ICML 2025 | 102 | **0** |

Neither benchmark contains a single occurrence of `encod` / `cipher` / `base64` /
`obfusc` / `ascii art` anywhere in its text. The two sentence-level flags the
screen raised are adjudicated **false positives**: OR-Bench's is about
"transforming" a topic into fictional style, and SCANS's is about the final hidden
state "incorrectly encoding the refusal prediction" — neural encoding, not surface
encoding.

Boundary row — the defense literature, which AS-6 inherits directly:

| paper | verdict | evidence |
|---|---|---|
| Yang et al., DecipherGuard | **B1** | FPR is calibrated on plain content: "a perplexity score is calculated from **a set of safe prompts** ... such that the False Positive Rate ... remains within an acceptable limit, such as 1%" |

⚠️ **AS-6 consequence, do not re-derive:** a defense against encoded attacks that
calibrates its false-positive rate on UNTRANSFORMED benign prompts has not measured
its false-positive rate under the condition it is deployed against. That is the
guard-side form of this paper's finding, and it belongs to AS-6.

## 5. What this licenses, and what it does not

**Licensed.** "Across 11 adjudicated input-transformation attack papers, none
reports a refusal or false-positive rate on benign content passed through its own
transformation; the closest is a single-prompt capability control in the founding
paper. The two flagship over-refusal benchmarks contain no transformation
vocabulary at all."

**NOT licensed.** "No one has ever measured this" — §3 refutes it. "The literature
has never sent benign content through an encoding" — Handa et al. did, and so did
Wei et al. at n=1. Any count stated over "the literature" rather than over the
adjudicated frame.

**Reproduce:**
```
uv run python scripts/survey_frame.py --bib <science>/literature/llm-security/references.bib --out outputs/analysis/survey_frame_20260821.json
uv run python scripts/survey_adjudicate.py <pdf> [...] --out outputs/analysis/survey_adjudication_20260821.json
```
Both keyless, no GPU, no model, seconds. Artifacts in `outputs/analysis/` (gitignored).
