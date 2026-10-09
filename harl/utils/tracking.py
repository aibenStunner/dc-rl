"""Optional dual TensorBoard and Weights & Biases tracking."""
from __future__ import annotations

import importlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path
from typing import Any


_DEFAULT_WANDB_CONFIG = {
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
_VALID_MODES = frozenset({"disabled", "offline", "online"})
_VALID_FAILURE_POLICIES = frozenset({"local_only", "error"})


def _json_safe(value: Any) -> Any:
    """Return a JSON-compatible representation without importing config tools."""
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        if isinstance(value, Mapping):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [_json_safe(item) for item in value]
        if isinstance(value, list):
            return [_json_safe(item) for item in value]
        return str(value)
    return value


def normalize_wandb_config(
    logger_config: Mapping[str, object] | None,
) -> dict[str, object]:
    """Normalize optional W&B settings while preserving a disabled default."""
    raw_logger = dict(logger_config or {})
    raw_wandb = raw_logger.get("wandb", {})
    if raw_wandb is None:
        raw_wandb = {}
    if not isinstance(raw_wandb, Mapping):
        raise ValueError("logger.wandb must be a mapping")

    config = dict(_DEFAULT_WANDB_CONFIG)
    config.update(raw_wandb)
    mode = config["mode"]
    if mode not in _VALID_MODES:
        raise ValueError(
            "logger.wandb.mode must be one of disabled, offline, or online"
        )
    failure_policy = config["failure_policy"]
    if failure_policy not in _VALID_FAILURE_POLICIES:
        raise ValueError(
            "logger.wandb.failure_policy must be local_only or error"
        )
    if not isinstance(config["tags"], list) or not all(
        isinstance(tag, str) for tag in config["tags"]
    ):
        raise ValueError("logger.wandb.tags must be a list of strings")
    if not isinstance(config["artifact_aliases"], list) or not all(
        isinstance(alias, str) for alias in config["artifact_aliases"]
    ):
        raise ValueError("logger.wandb.artifact_aliases must be a list of strings")

    config["enabled"] = bool(config["enabled"]) and mode != "disabled"
    if not config["enabled"]:
        config["mode"] = "disabled"
    config["log_artifacts"] = bool(config["log_artifacts"])
    return config


def _is_finite_scalar(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


class TensorboardWandbWriter:
    """Mirror finite scalar metrics to an optional W&B run."""

    def __init__(
        self,
        tensorboard_writer: object,
        *,
        run_dir: str | Path,
        config: Mapping[str, object],
        main_args: Mapping[str, object],
        algo_args: Mapping[str, object],
        env_args: Mapping[str, object],
        job_type: str,
    ) -> None:
        self.tensorboard = tensorboard_writer
        self.run_dir = Path(run_dir)
        self.config = dict(config)
        self.run = None
        self._closed = False
        self._nonfinite_metric_count = 0
        self._status = {
            "enabled": self.config["enabled"],
            "requested_mode": self.config["mode"],
            "actual_mode": "disabled",
            "status": "disabled",
            "run_name": self.run_dir.name,
            "job_type": job_type,
            "nonfinite_metric_count": 0,
        }
        self._initialize(main_args, algo_args, env_args, job_type)

    @property
    def status_path(self) -> Path:
        return self.run_dir / "tracking_status.json"

    def _write_status(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._status["nonfinite_metric_count"] = self._nonfinite_metric_count
        self.status_path.write_text(
            json.dumps(_json_safe(self._status), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _initialize(
        self,
        main_args: Mapping[str, object],
        algo_args: Mapping[str, object],
        env_args: Mapping[str, object],
        job_type: str,
    ) -> None:
        if not self.config["enabled"]:
            self._write_status()
            return
        try:
            wandb = importlib.import_module("wandb")
        except Exception as exc:  # noqa: BLE001 - optional dependency boundary
            self._handle_initialization_failure("import", exc)
            return
        try:
            run_config = _json_safe(
                {
                    "main_args": dict(main_args),
                    "algo_args": dict(algo_args),
                    "env_args": dict(env_args),
                    "local_run_dir": str(self.run_dir),
                }
            )
            self.run = wandb.init(
                project=self.config["project"],
                entity=self.config["entity"],
                group=self.config["group"],
                tags=self.config["tags"],
                name=self.run_dir.name,
                dir=str(self.run_dir),
                mode=self.config["mode"],
                config=run_config,
                job_type=job_type,
            )
        except Exception as exc:  # noqa: BLE001 - external SDK boundary
            self._handle_initialization_failure("initialization", exc)
            return
        self._status.update(
            {
                "actual_mode": self.config["mode"],
                "status": "active",
                "wandb_run_id": getattr(self.run, "id", None),
            }
        )
        self._write_status()

    def _handle_initialization_failure(self, stage: str, exc: Exception) -> None:
        message = f"W&B {stage} failed: {type(exc).__name__}: {exc}"
        if self.config["failure_policy"] == "error":
            raise RuntimeError(message) from exc
        self._status.update(
            {
                "actual_mode": "disabled",
                "status": "local_only",
                "warning": message,
            }
        )
        self._write_status()

    def _log(self, metrics: Mapping[str, object], global_step: int | None) -> None:
        if self.run is None:
            return
        finite = {key: float(value) for key, value in metrics.items() if _is_finite_scalar(value)}
        self._nonfinite_metric_count += len(metrics) - len(finite)
        if not finite:
            self._write_status()
            return
        try:
            self.run.log(finite, step=global_step)
        except Exception as exc:  # noqa: BLE001 - tracking must not alter training
            self._status["warning"] = f"W&B metric logging failed: {type(exc).__name__}: {exc}"
            self._status["status"] = "local_only"
            self._write_status()
            self.run = None

    def add_scalar(self, tag: str, scalar_value: object, global_step: int | None = None) -> None:
        self.tensorboard.add_scalar(tag, scalar_value, global_step)
        self._log({tag: scalar_value}, global_step)

    def add_scalars(
        self,
        main_tag: str,
        tag_scalar_dict: Mapping[str, object],
        global_step: int | None = None,
    ) -> None:
        self.tensorboard.add_scalars(main_tag, tag_scalar_dict, global_step)
        self._log(
            {f"{main_tag}/{tag}": value for tag, value in tag_scalar_dict.items()},
            global_step,
        )

    def log_metrics(
        self, metrics: Mapping[str, object], global_step: int | None = None
    ) -> None:
        for tag, value in metrics.items():
            self.tensorboard.add_scalar(tag, value, global_step)
        values = dict(metrics)
        if global_step is not None:
            values["global_env_step"] = global_step
        self._log(values, global_step)

    def export_scalars_to_json(self, path: str) -> None:
        self.tensorboard.export_scalars_to_json(path)

    def log_artifact(
        self,
        name: str,
        type: str,
        paths: Sequence[str | Path],
        aliases: Sequence[str] = (),
    ) -> None:
        if self.run is None or not self.config["log_artifacts"]:
            return
        existing_paths = [Path(path) for path in paths if Path(path).is_file()]
        if not existing_paths:
            return
        try:
            wandb = importlib.import_module("wandb")
            artifact = wandb.Artifact(name=name, type=type)
            for path in existing_paths:
                artifact.add_file(str(path))
            self.run.log_artifact(artifact, aliases=list(aliases))
        except Exception as exc:  # noqa: BLE001 - artifacts are optional
            self._status["artifact_warning"] = (
                f"W&B artifact logging failed: {type(exc).__name__}: {exc}"
            )
            self._write_status()

    def close(self, exit_code: int | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        tensorboard_error = None
        try:
            self.tensorboard.close()
        except Exception as exc:  # noqa: BLE001
            tensorboard_error = exc
        if self.run is not None:
            try:
                kwargs = {} if exit_code is None else {"exit_code": exit_code}
                self.run.finish(**kwargs)
            except Exception as exc:  # noqa: BLE001
                self._status["finish_warning"] = (
                    f"W&B finish failed: {type(exc).__name__}: {exc}"
                )
            finally:
                self.run = None
        self._write_status()
        if tensorboard_error is not None:
            raise tensorboard_error


def create_tracking_writer(
    *,
    tensorboard_writer: object,
    run_dir: str | Path,
    main_args: Mapping[str, object],
    algo_args: Mapping[str, object],
    env_args: Mapping[str, object],
    job_type: str = "train",
) -> TensorboardWandbWriter:
    """Wrap TensorBoard with optional W&B tracking for one local run."""
    logger_config = algo_args.get("logger", {})
    if not isinstance(logger_config, Mapping):
        raise ValueError("algo_args.logger must be a mapping")
    return TensorboardWandbWriter(
        tensorboard_writer,
        run_dir=run_dir,
        config=normalize_wandb_config(logger_config),
        main_args=main_args,
        algo_args=algo_args,
        env_args=env_args,
        job_type=job_type,
    )
