"""The build-completion check — and the reconciliation that keeps it honest.

The point of this file is that the status report cannot quietly go stale. A
hand-maintained "what's done" table drifts the moment someone builds something
and forgets to update it, which is the failure the check exists to prevent, one
level up. So the declared roster is asserted against the tree.
"""

from __future__ import annotations

from internals_safety.completion import (
    CONTROLS,
    NOT_ROSTER,
    ROSTER,
    Item,
    ItemStatus,
    build_status,
    placeholder_knobs,
    python_modules,
    reachable_modules,
    status_of,
)


def test_every_declared_module_actually_exists():
    """Catches a rename or a delete that would otherwise read as 'not built'."""
    present = python_modules()
    for item in ROSTER + CONTROLS:
        for module in item.modules:
            assert module in present, f"{item.key} names {module}, which does not exist"


def test_the_roster_accounts_for_every_measurement_module():
    """The reconciliation. Build a new instrument, forget the manifest, and this
    fails — rather than the status report cheerfully saying the build is done."""
    declared = {
        module.removeprefix("measurements.")
        for item in ROSTER + CONTROLS
        for module in item.modules
        if module.startswith("measurements.")
    } | set(NOT_ROSTER)
    actual = {
        module.removeprefix("measurements.")
        for module in python_modules()
        if module.startswith("measurements.")
    }
    assert actual == declared, (
        f"unaccounted measurement modules: {sorted(actual - declared)}; "
        f"declared but absent: {sorted(declared - actual)}. Add a roster entry, "
        "a control entry, or a NOT_ROSTER reason — never leave one unlisted."
    )


def test_an_item_with_no_modules_is_not_built_rather_than_vacuously_done():
    """I4 and I5 are declared with no modules. `all()` over an empty tuple is
    True, so a naive implementation would report them complete."""
    status = status_of(Item("I9", "imaginary", ()), python_modules(), reachable_modules())
    assert not status.built
    assert status.state == "not-built"


def test_a_declared_incompleteness_can_never_read_as_wired():
    """Regression, 2026-08-06. The first version reported I6 as 'wired' from
    module reachability while its own note said patching was not written — a
    check that can report done when it is not done is worse than no check."""
    module = next(iter(ROSTER[0].modules))
    item = Item("I9", "partly built", (module,), incomplete="half of it is missing")
    status = status_of(item, python_modules(), reachable_modules())
    assert not status.unwired  # the module IS reachable
    assert not status.wired  # ...and it still does not count as done
    assert status.state == "partial"


def _locations() -> list[str]:
    return [where for places in placeholder_knobs().values() for where in places]


def test_placeholder_detection_is_case_sensitive():
    """It matched case-insensitively first and reported twelve knobs, five of
    which were prose about STRING placeholders (`{prompt}`,
    `response_placeholder`). A check that over-reports gets ignored."""
    for location in _locations():
        assert "config.py:93" not in location  # `response_placeholder`
        assert "config.py:147" not in location  # "{prompt} placeholder"


def test_placeholder_knobs_are_reported_with_a_file_and_line():
    for location in _locations():
        path, _, line = location.rpartition(":")
        assert path and line.isdigit()


def test_a_knob_marked_in_BOTH_surfaces_counts_once():
    """⚠️ The same over-report defect as the case-insensitive match, one level
    subtler — and it survived until the owner asked whether the build was done.

    Every tunable is marked twice: once on the live YAML value and once on its
    fail-safe mirror in `config.py`. Counting MARKERS reported 11 where there
    were 6, inflating the headline ~2x. Both places are still listed under the
    knob, because a knob marked in one surface and not the other is a marking
    inconsistency worth seeing — but it is ONE thing needing tuning.
    """
    knobs = placeholder_knobs()
    mirrored = {name: places for name, places in knobs.items() if len(places) > 1}
    assert mirrored, "expected at least one knob marked in both YAML and config.py"
    for name, places in mirrored.items():
        assert len({place.split(":")[0] for place in places}) == len(places), (
            f"{name} is marked twice in the SAME file — that is a duplicate marker, "
            "not a YAML value and its code mirror"
        )
    assert len(knobs) < len(_locations())


def test_every_placeholder_resolves_to_a_config_KEY_not_a_line_number():
    """The fallback key is `file:line`, which is what a marker orphaned from its
    assignment produces. None should exist; if one does, the marker is floating
    in prose and the report cannot say what needs tuning."""
    unresolved = [name for name in placeholder_knobs() if ":" in name]
    assert unresolved == [], f"markers with no key beneath them: {unresolved}"


def test_the_build_is_not_complete_and_says_which_items_are_open():
    """The current answer, asserted so that 'is the build done?' has one home.

    ⚠️ When this test starts failing because `complete` became True, that is the
    build finishing — delete the assertion, do not weaken it.
    """
    status = build_status()
    assert not status.complete
    assert "I4" in " ".join(status.outstanding)


def test_the_orphan_guard_and_the_completion_check_share_one_reachability_rule():
    """Two copies of 'what does wired mean' is how the two start disagreeing.

    Asserted by identity on the imported function rather than by comparing two
    results, so the check fails when the orphan guard grows its own copy — not
    only when the copies happen to diverge on today's tree.
    """
    import test_package_structure

    assert test_package_structure.reachable_modules is reachable_modules
