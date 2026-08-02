"""Central path definitions.

Import paths from here instead of constructing them; nothing else in the
package should know how deep it sits under the repo root.

Provenance: shaped after `llm_guardrail_security/src/paths.py` (sibling repo),
re-authored for this repo's layout.
"""

from __future__ import annotations

from pathlib import Path

# src/internals_safety/paths.py -> repo root is three levels up.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONF_DIR = PROJECT_ROOT / "conf"
DATA_DIR = PROJECT_ROOT / "data"          # gitignored
OUTPUTS_DIR = PROJECT_ROOT / "outputs"    # gitignored

# Two distinct kinds of artifact, deliberately not nested in each other.
#   activations/ — a cache keyed by (model, prompt set, site). Capture is the
#                  expensive step and is reusable across analyses, so it
#                  outlives any single run.
#   runs/        — one dir per run: results.json (the provenance record; see
#                  .claude/skills/reproducible-run-logging) plus raw outputs.
#                  A run record names which activation cache it read.
ACTIVATIONS_DIR = OUTPUTS_DIR / "activations"
RUNS_DIR = OUTPUTS_DIR / "runs"


def run_dir(phase: str, model_name: str, run_name: str) -> Path:
    """`outputs/runs/<phase>/<model>/<run_name>/` — the run-record layout."""
    return RUNS_DIR / phase / model_name / run_name


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if absent; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
