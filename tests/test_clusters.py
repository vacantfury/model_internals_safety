"""The cluster registry — what stopped being true when xc became a second target.

Until 2026-08-22 every cluster constant in this repo was spelled inline as a
fact about "the cluster", because there was only one. Three were load-bearing
and all three are false on xc: the 8h wall (a literal in `ResourceConfig`'s
validator), the `/scratch/$USER` paths (spelled in `ops/run.sbatch` AND in
`submit.py`, each with a comment claiming it was kept in one place), and the
partition name.

The failure mode this module exists to prevent is specific and quiet. xc has no
`/scratch` at all, so a job that inherited NURC's paths would build its venv and
its HF cache in directories that do not exist, or worse, in ones that do and
that nobody reads. It would re-download 16GB of weights past a populated cache
and look like a slow run rather than a wrong one.

⚠️ This repo is PUBLIC. No test here may assert on a host, address, key, account
name, or the box owner's name, and no such value is in `conf/clusters/`.
Canonical cluster facts live in the devices repo.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from internals_safety.config import (
    list_clusters,
    list_presets,
    load_cluster_config,
    load_preset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "ops" / "run.sbatch"
ENV_PREFIX = "IS_"


class TestTheRegistryIsWellFormed:
    def test_both_clusters_load(self):
        assert set(list_clusters()) >= {"nurc", "xc"}

    @pytest.mark.parametrize("name", list_clusters())
    def test_a_cluster_declares_a_partition_it_actually_runs(self, name):
        cluster = load_cluster_config(name)
        assert cluster.partitions, name
        assert cluster.default_partition in cluster.partitions, (
            f"{name}'s default partition is not in its own partition list"
        )

    @pytest.mark.parametrize("name", list_clusters())
    def test_env_paths_expand_from_nothing_but_user_and_home(self, name):
        """They are expanded by a shell inside an sbatch, before any Python runs.

        A path needing more than `$USER` or `$HOME` has stopped being portable
        across accounts on the same cluster, and it would expand to an empty
        string rather than fail.
        """
        env = load_cluster_config(name).env
        for field, value in env.model_dump().items():
            for variable in re.findall(r"\$\{?(\w+)", value):
                assert variable in {"USER", "HOME"}, (
                    f"{name}.env.{field} uses ${variable}, which the launcher "
                    "does not guarantee"
                )

    @pytest.mark.parametrize("name", list_clusters())
    def test_no_cluster_entry_leaks_an_address_or_an_account(self, name):
        """The public-repo rule, checked rather than remembered.

        Deliberately crude: an IP-shaped literal or an `@`-bearing token in a
        file whose entire job is to describe a remote machine is the shape the
        rule forbids, and a crude check that fires is worth more here than a
        precise one nobody wrote.
        """
        text = (REPO_ROOT / "conf" / "clusters" / f"{name}.yaml").read_text()
        assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", text), f"{name}: IP-shaped literal"
        assert "@" not in text, f"{name}: an address-shaped token"
        assert not re.search(r"\b(ssh|scp)\b", text), f"{name}: a connection recipe"


class TestEveryPresetNamesARealClusterAndPartition:
    @pytest.mark.parametrize("name", list_presets())
    def test_the_cluster_is_registered(self, name):
        cluster_name = load_preset(name).resources.cluster
        assert cluster_name in list_clusters(), (
            f"preset {name} names cluster {cluster_name!r}, which conf/clusters "
            f"does not register ({list_clusters()})"
        )

    @pytest.mark.parametrize("name", list_presets())
    def test_the_partition_exists_on_that_cluster(self, name):
        preset = load_preset(name)
        cluster = load_cluster_config(preset.resources.cluster)
        assert preset.resources.partition in cluster.partitions


class TestTheWallComesFromTheClusterNotFromALiteral:
    def _resources(self, cluster: str, partition: str, time: str):
        from internals_safety.config import ResourceConfig

        return ResourceConfig(cluster=cluster, partition=partition, time=time)

    def test_nurc_still_refuses_a_job_longer_than_its_qos_allows(self):
        with pytest.raises(Exception) as excinfo:
            self._resources("nurc", "gpu", "09:00:00")
        assert "8h" in str(excinfo.value)

    def test_xc_accepts_the_same_ask_because_it_has_no_wall(self):
        resources = self._resources("xc", "main", "09:00:00")
        assert resources.time == "09:00:00"

    def test_a_partition_the_cluster_does_not_have_is_refused(self):
        # `main` does not exist on NURC and `gpu` does not exist on xc; sbatch
        # reports that only after the queue wait.
        with pytest.raises(Exception):
            self._resources("nurc", "main", "01:00:00")
        with pytest.raises(Exception):
            self._resources("xc", "gpu", "01:00:00")

    def test_an_unregistered_cluster_is_refused_and_lists_what_exists(self):
        with pytest.raises(Exception) as excinfo:
            self._resources("aicr", "main", "01:00:00")
        assert "aicr" in str(excinfo.value)


class TestTheLauncherAndTheConfigCannotDRIFT:
    """The one dual-truth risk this design still has, pinned.

    `submit.py` exports `IS_*` from the cluster entry and `ops/run.sbatch`
    requires them by name. Those two lists are the same fact written twice, in
    two languages, and nothing in Python can see the shell. A key added to
    `ClusterEnv` and not to the launcher's required list would simply never
    reach the job; a key removed from the config and left required would make
    every job exit 78.
    """

    def _required_by_launcher(self) -> set[str]:
        match = re.search(r"for required in ([^;]+); do", LAUNCHER.read_text())
        assert match, "the launcher's required-variable loop moved or was renamed"
        return set(match.group(1).split())

    def test_the_launcher_requires_exactly_what_the_config_exports(self):
        from internals_safety.config import ClusterEnv

        exported = {f"{ENV_PREFIX}{field.upper()}" for field in ClusterEnv.model_fields}
        exported.add(f"{ENV_PREFIX}CLUSTER")
        assert self._required_by_launcher() == exported

    def test_the_launcher_has_no_fallback_to_any_clusters_paths(self):
        # Comments stripped FIRST. The launcher explains in prose why it used to
        # hardcode `/scratch/$USER`, and that explanation is the reason the rule
        # survives; a guard that forbids saying the old path also forbids
        # recording why it went, which is how the reasoning gets deleted next
        # time someone makes a test pass.
        text = "\n".join(
            line for line in LAUNCHER.read_text().splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "/scratch/$USER" not in text, (
            "the launcher spells one cluster's paths again. A fallback is how a "
            "job submitted for another cluster comes to write where nobody reads."
        )
        assert "exit 78" in text

    def test_submit_exports_the_env_for_every_registered_cluster(self, tmp_path):
        import subprocess
        import sys

        for name in list_presets():
            preset = load_preset(name)
            cluster = load_cluster_config(preset.resources.cluster)
            # Rendered rather than imported, because the flag string is what
            # sbatch actually receives.
            result = subprocess.run(
                [sys.executable, "scripts/submit.py", name],
                cwd=REPO_ROOT, capture_output=True, text=True,
            )
            assert result.returncode == 0, result.stderr[-400:]
            assert f"{ENV_PREFIX}CLUSTER={cluster.name}" in result.stdout, name
            for field, value in cluster.env.model_dump().items():
                assert f"{ENV_PREFIX}{field.upper()}={value}" in result.stdout, (name, field)
            break  # one preset per cluster is enough; the loop below covers the rest

    @pytest.mark.parametrize("cluster_name", list_clusters())
    def test_each_cluster_has_at_least_one_preset_exercising_it(self, cluster_name):
        """A registry entry no preset uses is a claim nothing checks."""
        users = [n for n in list_presets() if load_preset(n).resources.cluster == cluster_name]
        assert users, f"cluster {cluster_name!r} is registered but no preset runs on it"
