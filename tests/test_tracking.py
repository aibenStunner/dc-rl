"""Tests for optional local TensorBoard and W&B tracking."""
import json
import math

import yaml
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests._harness import run_module_tests
from harl.utils.tracking import create_tracking_writer, normalize_wandb_config
from train_sustaindc import apply_wandb_overrides, run_and_close


class FakeTensorBoardWriter:
    def __init__(self):
        self.scalars = []
        self.scalar_groups = []
        self.exports = []
        self.close_calls = 0

    def add_scalar(self, tag, value, global_step=None):
        self.scalars.append((tag, value, global_step))

    def add_scalars(self, main_tag, values, global_step=None):
        self.scalar_groups.append((main_tag, values, global_step))

    def export_scalars_to_json(self, path):
        self.exports.append(path)

    def close(self):
        self.close_calls += 1


class FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path):
        self.files.append(path)


class FakeWandbRun:
    def __init__(self):
        self.logged = []
        self.finish_calls = 0
        self.id = "fake-run-id"
        self.artifacts = []

    def log(self, values, step=None):
        self.logged.append((values, step))

    def finish(self, **kwargs):
        self.finish_calls += 1

    def log_artifact(self, artifact, aliases):
        self.artifacts.append((artifact, aliases))


class FakeWandb:
    def __init__(self):
        self.init_calls = []
        self.run = FakeWandbRun()

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        return self.run

    Artifact = FakeArtifact


def _args(wandb_config):
    return {
        "main_args": {"algo": "happo", "env": "sustaindc", "exp_name": "unit"},
        "algo_args": {"logger": {"wandb": wandb_config}},
        "env_args": {"reward": {"mode": "team_tariff_targeted_safeguards"}},
    }


def test_disabled_tracking_never_imports_wandb():
    old_wandb = sys.modules.pop("wandb", None)
    try:
        with tempfile.TemporaryDirectory() as directory:
            args = _args({"enabled": False, "mode": "disabled"})
            tensorboard = FakeTensorBoardWriter()
            writer = create_tracking_writer(
                tensorboard_writer=tensorboard,
                run_dir=directory,
                **args,
            )
            writer.add_scalar("train/test", 2.0, 8)
            writer.close()
            assert "wandb" not in sys.modules
            assert tensorboard.scalars == [("train/test", 2.0, 8)]
            assert tensorboard.close_calls == 1
    finally:
        if old_wandb is not None:
            sys.modules["wandb"] = old_wandb


def test_enabled_writer_mirrors_finite_scalars_and_finishes_once():
    fake_wandb = FakeWandb()
    old_wandb = sys.modules.get("wandb")
    sys.modules["wandb"] = fake_wandb
    try:
        with tempfile.TemporaryDirectory() as directory:
            args = _args({
                "enabled": True,
                "mode": "offline",
                "project": "project",
                "entity": None,
                "group": "experiment-1",
                "tags": ["happo"],
                "failure_policy": "error",
            })
            tensorboard = FakeTensorBoardWriter()
            writer = create_tracking_writer(
                tensorboard_writer=tensorboard,
                run_dir=directory,
                **args,
            )
            writer.add_scalar("train/agent0/policy_loss", 1.25, 12)
            writer.add_scalar("invalid", float("nan"), 13)
            writer.close()
            writer.close()
            assert fake_wandb.init_calls[0]["mode"] == "offline"
            assert fake_wandb.run.logged == [
                ({"train/agent0/policy_loss": 1.25}, 12)
            ]
            assert fake_wandb.run.finish_calls == 1
            assert tensorboard.scalars[0] == ("train/agent0/policy_loss", 1.25, 12)
            assert tensorboard.scalars[1][0] == "invalid"
            assert math.isnan(tensorboard.scalars[1][1])
            assert tensorboard.close_calls == 1
    finally:
        if old_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = old_wandb


def test_artifact_upload_uses_existing_files_only():
    fake_wandb = FakeWandb()
    old_wandb = sys.modules.get("wandb")
    sys.modules["wandb"] = fake_wandb
    try:
        with tempfile.TemporaryDirectory() as directory:
            args = _args({
                "enabled": True,
                "mode": "offline",
                "log_artifacts": True,
                "failure_policy": "error",
            })
            writer = create_tracking_writer(
                tensorboard_writer=FakeTensorBoardWriter(),
                run_dir=directory,
                **args,
            )
            config = Path(directory) / "config.json"
            config.write_text("{}", encoding="utf-8")
            writer.log_artifact(
                "unit-checkpoint", "sustaindc-checkpoint",
                [config, Path(directory) / "missing"], aliases=["latest"],
            )
            artifact, aliases = fake_wandb.run.artifacts[0]
            assert artifact.files == [str(config)]
            assert aliases == ["latest"]
            assert config.exists()
            writer.close()
    finally:
        if old_wandb is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = old_wandb


def test_legacy_config_defaults_to_disabled_and_disabled_mode_wins():
    assert normalize_wandb_config({})["enabled"] is False
    config = normalize_wandb_config({
        "wandb": {"enabled": True, "mode": "disabled"},
    })
    assert config["enabled"] is False
    assert config["mode"] == "disabled"


class RaisingRunner:
    def __init__(self, error):
        self.error = error
        self.close_calls = 0

    def run(self):
        raise self.error

    def close(self):
        self.close_calls += 1


def test_explicit_wandb_flags_override_loaded_config():
    algo_args = {"logger": {"wandb": {"enabled": False, "mode": "disabled"}}}
    apply_wandb_overrides(algo_args, {
        "wandb": True,
        "wandb_mode": "offline",
        "wandb_project": "new-project",
        "wandb_entity": "team",
        "wandb_group": "exp-1",
        "wandb_tags": ["tariff", "seed-1"],
        "wandb_artifacts": True,
    })
    assert algo_args["logger"]["wandb"] == {
        "enabled": True,
        "mode": "offline",
        "project": "new-project",
        "entity": "team",
        "group": "exp-1",
        "tags": ["tariff", "seed-1"],
        "log_artifacts": True,
        "artifact_aliases": ["latest"],
        "failure_policy": "local_only",
    }


def test_disabled_wandb_mode_wins_over_enable_flag():
    algo_args = {"logger": {"wandb": {}}}
    apply_wandb_overrides(algo_args, {"wandb": True, "wandb_mode": "disabled"})
    assert algo_args["logger"]["wandb"]["enabled"] is False
    assert algo_args["logger"]["wandb"]["mode"] == "disabled"


def test_every_algorithm_config_has_identical_disabled_wandb_defaults():
    expected = {
        "enabled": False,
        "mode": "disabled",
        "project": "sustaindc-thesis",
        "entity": None,
        "group": None,
        "tags": [],
        "log_artifacts": False,
        "artifact_aliases": ["latest"],
        "failure_policy": "local_only",
    }
    for path in sorted((_REPO_ROOT / "harl/configs/algos_cfgs").glob("*.yaml")):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["logger"]["wandb"] == expected


def test_run_and_close_reraises_and_closes_once():
    runner = RaisingRunner(RuntimeError("train failed"))
    try:
        run_and_close(runner)
    except RuntimeError as exc:
        assert str(exc) == "train failed"
    else:
        raise AssertionError("training exception must be re-raised")
    assert runner.close_calls == 1


def test_local_only_initialization_failure_keeps_tensorboard_writer():
    old_wandb = sys.modules.get("wandb")
    sys.modules.pop("wandb", None)
    try:
        with tempfile.TemporaryDirectory() as directory:
            args = _args({
                "enabled": True,
                "mode": "offline",
                "failure_policy": "local_only",
            })
            tensorboard = FakeTensorBoardWriter()
            writer = create_tracking_writer(
                tensorboard_writer=tensorboard,
                run_dir=directory,
                **args,
            )
            writer.add_scalar("local/metric", 3.0, 4)
            writer.close()
            assert tensorboard.scalars == [("local/metric", 3.0, 4)]
            status = json.loads(
                (Path(directory) / "tracking_status.json").read_text(encoding="utf-8")
            )
            assert status["status"] == "local_only"
            assert "import" in status["warning"].lower()
    finally:
        if old_wandb is not None:
            sys.modules["wandb"] = old_wandb


if __name__ == "__main__":
    sys.exit(1 if run_module_tests(globals()) else 0)
