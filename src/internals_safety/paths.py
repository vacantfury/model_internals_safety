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
ACTIVATIONS_DIR = OUTPUTS_DIR / "activations"


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if absent; return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
