# AS-6 paper check — FINAL mode (2026-08-22)

- **Mode:** final. **Verdict: pass-with-advisories.** Blocking: 0.
- **Venue:** the target venue, named only in the private science venue record
  (`venues/`, live-verified 2026-08-08 and 2026-08-21) and never here. Limits
  taken from that record rather than from memory: 7 pages main content, later
  pages references only, 9 max; double-blind; appendix has no in-PDF home;
  external repository links banned.
- **State keyed:** `paper.pdf` 8 pages, references alone on page 8; `supplement.pdf`
  separate. Any later edit stales this.

⚠️ **TWO PROPERTIES OF THIS RUN ARE DEGRADED, STATED RATHER THAN IMPLIED.** The
skill prescribes a fresh subagent per dimension so the checker is not the author;
the session's standing instruction bars subagents, so **I checked prose I wrote
myself hours earlier** and the author-rubber-stamp risk is live. And the
**cross-family second opinion did not run**, which the skill names as the default
at this stakes level. Both are knowing debts, not oversights.

## Verified

| Check | Result |
|---|---|
| Page fit | content ends p7, References p8 line 1, 8 pages total — **compliant** |
| Debt marks (TODO/TBD/XXX/FIXME) | present only on `%` comment lines, none rendered |
| Unresolved refs (`??` in PDF) | 0 |
| Tables referenced in text | all |
| Build, both kits, both documents | exit 0, 0 errors, 0 undefined, 0 overfull |
| Anonymity | 0 identifying hits; no self-citation; supplement titled "Anonymous submission" |
| Kit parity | pass |
| Claim ledger (artefact half) | pass, this host |
| Claim integrity (clone half) | pass for AS-6; **AS-5 fails 2, other session's paper** |

## Findings acted on in this pass

**1. Self-containment violation (blocking, now fixed).** The venue rule is
explicit: *"moving material out is NOT a length device. Anything a reviewer needs
in order to ACCEPT a claim must stay in the main paper."* The page cut had moved
two whole result subsections out with no trace in the body. Both are now stated
compactly in the body with the detail in the supplement: the two-screen
partition agreement, and the guards' total dissociation on `fullwidth` (85 of 100
against 0). The other moves (Limitations detail, the length-bound construction,
the wrapper factorial table) left their CLAIMS in the body and are compliant as
they stand.

**2. Three provenance gaps (advisory, now fixed).** The paper presented three
established designs as its own. Each now cites its source, and each is doing work
rather than padding:
- the **control floor** is the control-task construction of
  \citet{hewitt2019designing} (EMNLP-IJCNLP 2019, tier A) applied to a
  transferred probe;
- the **circularity argument** against refitting on encoded activations is the
  standard probing critique, \citet{belinkov2022probing} (Computational
  Linguistics 2022, tier A);
- the **wrapper arm** is a format-decorrelation 2x2, the design of
  \citet{devbunova2026formatsensitivity} (ICLR 2026 workshop).

**3. Reference set was thin (advisory, partly fixed).** 16 body citations before
this pass, 21 bib entries now. Still on the low side for the venue. The corpus
holds 478 entries and 31 uncited on-topic ones.

## Open advisories

⛔ **`unknown2026knowingwithoutacting` — "Knowing without Acting: The Disentangled
Geometry of Safety Mechanisms in Large Language Models" (arXiv 2603.05773).** The
title is AS-6's thesis in four words: knowing without acting IS decoded but not
blocked. The corpus marks it CANDIDATE with authors and venue unverified, so it
is **not cited and not dismissed**. This is a scoop check, not a citation task,
and it should run before the paper is defended.

**Further citations available and unadded**, each with a place it would do work:
`zou2023representation` (RepE, reading/steering representations),
`bailey2024obfuscated` (obfuscated activations bypass latent-space defenses —
bears directly on the linear-probe limitation), `zou2024circuitbreakers`,
`bijectionlearning2025` (tunable-complexity encoded attacks),
`qi2024finetuning` / `qi2025shallow` (shallow safety alignment, relevant to the
policy-versus-capability frame).

**Abstract numeral density** remains 6 against the house limit of about 3. Kept
deliberately: four of the six are the 8/7-against-17/23 contrast, which is the
paper's thesis applied to itself.
