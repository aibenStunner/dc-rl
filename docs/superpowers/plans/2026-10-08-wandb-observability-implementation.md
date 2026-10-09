# SustainDC W&B Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add comprehensive, optional W&B observability for SustainDC Experiment 1 without changing TensorBoard, physical simulation, tariff accounting, or HARL reward delivery.

**Architecture:** Introduce a lazy, dual TensorBoard/W&B writer and a focused, testable facility-metric registry/accumulator. Centralize W&B lifecycle and provenance at writer creation, collect operational metrics from the existing merged SustainDC `info` record, and make rollout versus natural-episode versus evaluation aggregate cadence explicit in names and traces.

**Tech Stack:** Python 3.10, tensorboardX, optional W&B SDK, NumPy, Gymnasium/PettingZoo, vendored HARL, standalone `tests/_harness.py` tests.

**Spec:** `docs/superpowers/specs/2026-10-08-wandb-observability-design.md`

## Global Constraints

- Preserve TensorBoard, local checkpoint/config/CSV output behavior, and existing metric tags.
- W&B is disabled by default and must not be imported in disabled mode.
- W&B modes are exactly `disabled`, `offline`, and `online`; never run login or store credentials in configuration, artifacts, or telemetry.
- W&B configuration lives under `algo_args["logger"]["wandb"]`; missing configuration in older saved runs means disabled.
- Do not change pricing calculations, reward functions, physical dynamics, actor observations, or FP/EP reward routing.
- Read facility telemetry from the existing merged `info` dictionary; do not recompute tariff, reward, PV, battery, or workload values in logging code.
- Count facility tariff cost once per environment step, never once per agent.
- Use `train/rollout/*` for HARL collection windows and `eval/episode/*` only for naturally completed evaluation episodes. Do not label a partial rollout as a 30-day episode.
- Keep raw units separate from reward units: costs in cents, energy in kWh, verified power in kW, and reward values in reward units.
- Do not label cooling-water pump sources as kW in new canonical metric names until their units are separately validated; preserve only `_raw` trace fields.
- Preserve existing uncommitted work. Do not commit, amend, reset, rebase, cherry-pick, push, or otherwise alter Git history.

## Review Focus

1. Disabled and legacy-loaded runs must not import `wandb` or require a W&B package; Task 1 adds mocked-import regression coverage.
2. A failed/interrupting training run must flush local TensorBoard and finish W&B at most once without swallowing the original exception; Task 2 adds lifecycle tests.
3. A shared tariff cost present in all three agent infos must be aggregated once, not tripled; Task 3 tests one-record-per-thread accounting.
4. A 128/1,024-step collection window must be exposed as a rollout rather than an environment episode; Task 4 asserts canonical cadence names and terminal-only episode completion.
5. A trace must preserve both pre-advance billed clock and post-advance state clock; Task 5 verifies full evaluation-export headers/rows and rate/reward fields.

---

## Files and responsibilities

- `requirements-wandb.txt` — optional W&B installation path that leaves base `requirements.txt` free of a hard W&B dependency.
- `harl/utils/tracking.py` — W&B configuration normalization, lazy initialization, dual writer adapter, finite-scalar filtering, local tracking status/provenance, artifact helper.
- `harl/utils/configs_tools.py` — create TensorBoard then tracking adapter centrally; save normalized resolved configuration.
- `train_sustaindc.py` — explicit W&B CLI flags, deterministic override merge, `try/finally` runner cleanup.
- `eval_sustaindc.py` — explicit standalone evaluation tracking opt-in and separate eval-run policy.
- `harl/configs/algos_cfgs/*.yaml` — default-disabled W&B logger configuration for all ten algorithms.
- `sustaindc_env.py` — add already-calculated billing/clock facts to the existing reward/info payload; no tariff behavior change.
- `harl/envs/sustaindc/metrics.py` — metric specifications and pure accumulator for one facility record per environment step.
- `harl/common/base_logger.py` — stable semantic aliases for actor/critic update metrics and per-agent completed-return recording.
- `harl/envs/sustaindc/sustaindc_logger.py` — train rollout and natural evaluation episode collector/emitter using the metric registry.
- `harl/runners/on_policy_base_runner.py` — capture full evaluation trace records, call logger terminal callbacks, publish checkpoint/evaluation artifacts at existing successful seams.
- `harl/runners/off_policy_base_runner.py` — call equivalent logger/artifact lifecycle seams without changing algorithm behavior.
- `tests/test_tracking.py` — isolated writer/config/CLI/lifecycle tests with mocked W&B.
- `tests/test_sustaindc_metrics.py` — pure metric registry reductions, cadence, finite/zero-denominator behavior, and tariff-once tests.
- `tests/test_pricing_integration.py` — real SustainDC billing-clock/info exposure regression tests.
- `tests/test_harl_reward_delivery.py` — extend only as needed for per-agent logger/return exposure.
- `tests/README.md` — add tracker/metric test modules and optional W&B installation/run instructions.

## Task 1: Build the optional dual-writer and provenance boundary

**Files:**
- Create: `requirements-wandb.txt`
- Create: `harl/utils/tracking.py`
- Modify: `harl/utils/configs_tools.py:57-120`
- Create: `tests/test_tracking.py`

**Interfaces:**
- Consumes: TensorBoard `SummaryWriter`, local `run_dir`, resolved `main_args`, `algo_args`, `env_args`, and `logger.wandb` configuration.
- Produces: `create_tracking_writer(...) -> TensorboardWandbWriter`, which supports `add_scalar`, `add_scalars`, `log_metrics`, `export_scalars_to_json`, `log_artifact`, and idempotent `close`; `normalize_wandb_config(...) -> dict`; and `tracking_status.json` under each run directory.

- [ ] **Step 1: Write the failing disabled and enabled writer tests**

Create `tests/test_tracking.py` with an in-memory fake TensorBoard writer and a mocked W&B module injected through `sys.modules`. Add these tests before creating production code:

```python
def test_disabled_tracking_never_imports_wandb(tmp_path, monkeypatch):
    sys.modules.pop("wandb", None)
    writer = create_tracking_writer(
        tensorboard_writer=FakeTensorBoardWriter(),
        run_dir=tmp_path,
        main_args={"algo": "happo", "env": "sustaindc", "exp_name": "unit"},
        algo_args={"logger": {"wandb": {"enabled": False, "mode": "disabled"}}},
        env_args={},
    )
    writer.add_scalar("train/test", 2.0, 8)
    writer.close()
    assert "wandb" not in sys.modules
    assert writer.tensorboard.scalars == [("train/test", 2.0, 8)]
```

```python
def test_enabled_writer_mirrors_finite_scalars_and_finishes_once(tmp_path, monkeypatch):
    fake_wandb = FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    writer = create_tracking_writer(
        tensorboard_writer=FakeTensorBoardWriter(),
        run_dir=tmp_path,
        main_args={"algo": "happo", "env": "sustaindc", "exp_name": "unit"},
        algo_args={"logger": {"wandb": {
            "enabled": True, "mode": "offline", "project": "project",
            "entity": None, "group": "experiment-1", "tags": ["happo"],
            "failure_policy": "error",
        }}},
        env_args={"reward": {"mode": "team_tariff_targeted_safeguards"}},
    )
    writer.add_scalar("train/agent0/policy_loss", 1.25, 12)
    writer.add_scalar("invalid", float("nan"), 13)
    writer.close()
    writer.close()
    assert fake_wandb.init_calls[0]["mode"] == "offline"
    assert fake_wandb.run.logged == [({"train/agent0/policy_loss": 1.25}, 12)]
    assert fake_wandb.run.finish_calls == 1
```

Add tests that a legacy logger config with no `wandb` key normalizes to disabled, `mode="disabled"` wins over `enabled=True`, and local-only initialization failure produces a status file and still returns a TensorBoard-capable writer.

- [ ] **Step 2: Run the new module and verify the expected import failure**

Run:

```bash
PY=~/.pyenv/versions/3.10.14/envs/thesis-dcrl/bin/python
$PY tests/test_tracking.py
```

Expected: the module fails because `harl.utils.tracking` and `create_tracking_writer` do not exist.

- [ ] **Step 3: Add the optional dependency manifest and tracking module**

Create `requirements-wandb.txt`:

```text
-r requirements.txt
wandb==0.19.8
```

Create `harl/utils/tracking.py` with these exact public entry points:

```python
def normalize_wandb_config(logger_config: Mapping[str, object] | None) -> dict[str, object]: ...

def create_tracking_writer(
    *, tensorboard_writer: object, run_dir: str | Path,
    main_args: Mapping[str, object], algo_args: Mapping[str, object],
    env_args: Mapping[str, object], job_type: str = "train",
) -> "TensorboardWandbWriter": ...

class TensorboardWandbWriter:
    def add_scalar(self, tag, scalar_value, global_step=None): ...
    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None): ...
    def log_metrics(self, metrics, global_step): ...
    def export_scalars_to_json(self, path): ...
    def log_artifact(self, name, type, paths, aliases=()): ...
    def close(self, exit_code=None): ...
```

Required behavior:

- Normalize absent/invalid optional keys to the disabled defaults; validate `mode` and `failure_policy` with explicit `ValueError` messages.
- Write `tracking_status.json` containing enabled/requested/actual mode, status, warning/error text, run name, job type, and generated W&B ID when a live run exists.
- Lazy-import `wandb` only when enabled in online/offline mode.
- Call `wandb.init(project=..., entity=..., group=..., tags=..., name=Path(run_dir).name, dir=str(run_dir), mode=..., config=..., job_type=...)` exactly once.
- In `local_only`, catch import/init exceptions, update status, and retain TensorBoard-only operation. In `error`, raise a `RuntimeError` naming import/init failure.
- Always forward original scalar values to TensorBoard. Send only finite real values to W&B, recording omitted non-finite/non-real scalar diagnostics.
- Flatten `add_scalars` into `main_tag/subtag` W&B keys and forward its original call to TensorBoard.
- `log_metrics` filters unsupported values, adds `global_env_step` when supplied, and uses one W&B `log` call.
- `close` closes wrapped TensorBoard once and finishes W&B once; a finish failure must not hide a TensorBoard close exception.
- `log_artifact` is no-op unless W&B is active and `log_artifacts`; uses existing paths only and retains local files after upload failure.

- [ ] **Step 4: Route central writer construction through the adapter**

Change `init_dir` to accept keyword-only `main_args=None`, `algo_args=None`, `job_type="train"`. Keep return tuple/directory layout. After `SummaryWriter(log_path)`, call `create_tracking_writer` only when all resolved config arguments exist; otherwise return the bare TensorBoard writer for external legacy callers.

At both runner constructors, pass:

```python
main_args=args, algo_args=algo_args, job_type="train"
```

while retaining `save_config(args, algo_args, env_args, self.run_dir)` directly after initialization.

- [ ] **Step 5: Run writer tests and existing config tests**

```bash
$PY tests/test_tracking.py
$PY tests/test_team_tariff_reward.py
```

Expected: mocked tracking passes without network/package; reward/config tests remain green.

## Task 2: Add deterministic configuration, CLI overrides, and cleanup lifecycle

**Files:**
- Modify: `train_sustaindc.py:17-105`
- Modify: `eval_sustaindc.py:24-43, 187-192`
- Modify: all `harl/configs/algos_cfgs/*.yaml` logger blocks
- Modify: `harl/runners/on_policy_base_runner.py:70-81, 980-990`
- Modify: `harl/runners/off_policy_base_runner.py:49-64, 702-713`
- Modify: `tests/test_tracking.py`
- Modify: `tests/README.md`

**Interfaces:**
- Consumes: parsed explicit W&B flags and old/new saved configs.
- Produces: resolved `algo_args["logger"]["wandb"]`, train/eval job types, exactly-once cleanup, and default local-only legacy behavior.

- [ ] **Step 1: Write failing configuration precedence and cleanup tests**

Extend `tests/test_tracking.py`:

```python
def test_explicit_wandb_flags_override_loaded_config():
    algo_args = {"logger": {"wandb": {"enabled": False, "mode": "disabled"}}}
    apply_wandb_overrides(algo_args, {
        "wandb": True, "wandb_mode": "offline", "wandb_project": "new-project",
        "wandb_entity": "team", "wandb_group": "exp-1",
        "wandb_tags": ["tariff", "seed-1"], "wandb_artifacts": True,
    })
    assert algo_args["logger"]["wandb"] == {
        "enabled": True, "mode": "offline", "project": "new-project",
        "entity": "team", "group": "exp-1", "tags": ["tariff", "seed-1"],
        "log_artifacts": True, "artifact_aliases": ["latest"],
        "failure_policy": "local_only",
    }
```

```python
def test_run_and_close_reraises_and_closes_once():
    runner = RaisingRunner(RuntimeError("train failed"))
    try:
        run_and_close(runner)
    except RuntimeError as exc:
        assert str(exc) == "train failed"
    else:
        raise AssertionError("training exception must be re-raised")
    assert runner.close_calls == 1
```

Also assert explicit disabled mode wins over `--wandb`.

- [ ] **Step 2: Run tests and verify missing helper failures**

```bash
$PY tests/test_tracking.py
```

Expected: missing `apply_wandb_overrides` and `run_and_close` failures.

- [ ] **Step 3: Add identical defaults to every algorithm YAML**

Under each existing `logger.log_dir`, add:

```yaml
  wandb:
    enabled: false
    mode: disabled
    project: sustaindc-thesis
    entity: null
    group: null
    tags: []
    log_artifacts: false
    artifact_aliases: [latest]
    failure_policy: local_only
```

Apply to `haa2c`, `had3qn`, `haddpg`, `happo`, `hasac`, `hatd3`, `hatrpo`, `maddpg`, `mappo`, and `matd3` YAML files.

- [ ] **Step 4: Implement explicit CLI merge and safe cleanup**

In `train_sustaindc.py`, add options:

```python
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb-mode", choices=("disabled", "offline", "online"))
parser.add_argument("--wandb-project")
parser.add_argument("--wandb-entity")
parser.add_argument("--wandb-group")
parser.add_argument("--wandb-tags", nargs="+")
parser.add_argument("--wandb-artifacts", action="store_true")
```

Implement `apply_wandb_overrides(algo_args, parsed_args)` using `normalize_wandb_config`; it initializes absent mappings, updates only explicit flags, forces enabled for online/offline mode, forces disabled for disabled mode, and lets `--wandb` choose online only when no explicit mode exists.

Implement:

```python
def run_and_close(runner):
    try:
        runner.run()
    finally:
        runner.close()
```

Replace direct `runner.run(); runner.close()` with this helper.

In `eval_sustaindc.py`, force W&B disabled after loading a training config unless a clearly named top-level evaluation opt-in is set. An opt-in evaluation gets `job_type="eval"`, a distinct experiment-name suffix, and no training resume ID. Close the runner with `try/finally` around the standalone loop.

Make both runner `close()` methods idempotent for safe finalization. Preserve normal close behavior and all local files.

- [ ] **Step 5: Document optional installation and modes**

Add to `tests/README.md`:

```text
Optional W&B support: pip install -r requirements-wandb.txt.
Default runs do not require/import W&B. Use --wandb --wandb-mode offline
for local offline tracking, or --wandb --wandb-mode online after configuring
W&B authentication. Never put credentials in YAML.
```

- [ ] **Step 6: Run focused tests**

```bash
$PY tests/test_tracking.py
$PY tests/test_team_tariff_reward.py
```

Expected: all pass without external W&B.

## Task 3: Surface billing facts and create the facility metric registry

**Files:**
- Modify: `sustaindc_env.py:790-860, 976-1005`
- Create: `harl/envs/sustaindc/metrics.py`
- Create: `tests/test_sustaindc_metrics.py`
- Modify: `tests/test_pricing_integration.py`

**Interfaces:**
- Consumes: one merged facility `info` record per environment thread and current `PriceManager` state/observation.
- Produces: billing/clock info keys; `FacilityMetricAccumulator.add(info)`, `snapshot()`, `reset()`, and `aggregate_metric_snapshots(...)`.

- [ ] **Step 1: Write failing billing-info and reduction tests**

Create `tests/test_sustaindc_metrics.py` with hand-computed records:

```python
def test_facility_cost_is_counted_once_per_environment_step():
    accumulator = FacilityMetricAccumulator("train/rollout")
    accumulator.add(_info(total_price_cost_this_step_c=10.0, bat_grid_import_kwh=2.0))
    accumulator.add(_info(total_price_cost_this_step_c=15.0, bat_grid_import_kwh=3.0))
    snapshot = accumulator.snapshot()
    assert snapshot["train/rollout/tariff/cost/total_c"] == 25.0
    assert snapshot["train/rollout/energy/grid_import_kwh"] == 5.0
```

```python
def test_accumulator_preserves_targeted_rewards_and_penalties():
    accumulator = FacilityMetricAccumulator("eval/episode")
    accumulator.add(_info(
        agent_ls_reward=-4.0, agent_dc_reward=-3.0, agent_bat_reward=-2.0,
        shared_tariff_cost_reward=-2.0, ls_overdue_penalty_component=1.0,
        dc_thermal_penalty_component=1.0,
    ))
    snapshot = accumulator.snapshot()
    assert snapshot["eval/episode/reward/agent_ls/sum"] == -4.0
    assert snapshot["eval/episode/reward/agent_dc/sum"] == -3.0
    assert snapshot["eval/episode/reward/agent_bat/sum"] == -2.0
    assert snapshot["eval/episode/reward/mean_across_agents/sum"] == -3.0
```

Also test SoC start/end/min/max/mean, queue mean/max/final/queue-task-steps, battery action counts, thermal exceedance degree-steps, zero-grid ratio omission, nonfinite diagnostic, and aggregate mean/std/min/max across two evaluation snapshots.

Add to `tests/test_pricing_integration.py`:

```python
def test_info_exposes_billed_clock_and_billing_state():
    env = _make_env()
    env.reset()
    env.t_m.day, env.t_m.hour, env.current_hour = 333, 23.75, 23.75
    _, _, _, _, info = env.step(_step_actions(env))
    common = info["__common__"]
    assert (common["billed_day"], common["billed_hour"]) == (333, 23.75)
    assert (common["next_state_day"], common["next_state_hour"]) == (334, 0.0)
    assert common["billing_period_energy_kwh"] >= 0.0
    assert common["billing_running_peak_kw"] >= 0.0
```

- [ ] **Step 2: Run new tests and verify missing module/fields**

```bash
$PY tests/test_sustaindc_metrics.py
$PY tests/test_pricing_integration.py
```

Expected: missing metrics import and missing billing keys.

- [ ] **Step 3: Surface facts without modifying pricing**

Pass `billed_day`, `billed_hour` to `_calculate_reward_params`, then add:

```python
"billed_day": billed_day,
"billed_hour": billed_hour,
"next_state_day": day,
"next_state_hour": hour,
"billing_running_peak_kw": self.price_m.state.running_peak_kw,
"billing_period_energy_kwh": self.price_m.state.billing_period_energy_kwh,
"billing_progress_fraction": self.price_m.get_billing_progress_fraction(),
"selected_tariff": getattr(self.price_m.model, "tariff", None),
```

Do not mutate `PriceManager` or substitute next-state time for billed time.

- [ ] **Step 4: Implement accumulator and registry**

Create `harl/envs/sustaindc/metrics.py`:

```python
@dataclass(frozen=True)
class MetricSpec:
    source_key: str
    canonical_suffix: str
    reduction: str
    required_in_targeted_mode: bool = False

class FacilityMetricAccumulator:
    def __init__(self, namespace, *, targeted_mode=False): ...
    def add(self, info): ...
    def snapshot(self): ...
    def reset(self): ...

def aggregate_metric_snapshots(snapshots, namespace): ...
```

Cover every current scalar merged-info group: all tariff and Experiment 1 reward components; battery/PV/carbon; workload/SLA; cooling/power/water. Map source fields to canonical names specified by the design. Required-targeted fields raise `KeyError` when absent; optional missing values increment `diagnostic/info_missing_key_count` and are omitted. Nonfinite source values increment `diagnostic/nonfinite_metric_count` and are omitted.

Implement derived fields solely from source aggregates:

```text
battery delta = grid import - no-battery energy
PV fractions only with positive PV generation
carbon and water intensity only with positive grid import
PUE = 1 + aggregate HVAC / aggregate ITE only with positive ITE
temperature margin = limit - internal temperature
thermal exceedance degree-steps = max(0, temperature - limit)
constraint steps = sum constraint flag
queue task steps = sum queue depth
```

Pump values may appear only in detailed trace values with `_raw` names, not canonical `_kw` metrics.

- [ ] **Step 5: Run metric/real-env tests**

```bash
$PY tests/test_sustaindc_metrics.py
$PY tests/test_pricing_integration.py
$PY tests/test_team_tariff_reward.py
```

Expected: all pass with pre-advance pricing/reward reconciliation intact.

## Task 4: Integrate rollout/evaluation metric logging and preserve distinct returns

**Files:**
- Modify: `harl/common/base_logger.py:30-197`
- Modify: `harl/envs/sustaindc/sustaindc_logger.py:1-287`
- Modify: `harl/runners/on_policy_base_runner.py:554-715`
- Modify: `harl/runners/off_policy_base_runner.py:523-611`
- Modify: `tests/test_sustaindc_metrics.py`
- Modify: `tests/test_harl_reward_delivery.py`

**Interfaces:**
- Consumes: vector rewards/dones/infos, accumulators, writer `log_metrics`, and ordered agents.
- Produces: update/rollout/evaluation canonical metrics, terminal episode records, and preserved legacy tags.

- [ ] **Step 1: Write failing cadence and per-agent return tests**

Add:

```python
def test_rollout_logging_keeps_targeted_returns_separate():
    logger = _logger_with_fake_writer()
    logger.episode_init(1)
    logger.per_step(_training_data(
        rewards=np.array([[[-4.0], [-3.0], [-2.0]]], dtype=np.float32),
        infos=[[_info(agent_ls_reward=-4.0, agent_dc_reward=-3.0, agent_bat_reward=-2.0)] * 3],
        dones=np.zeros((1, 3, 1), dtype=bool),
    ))
    logger.episode_log([{}, {}, {}], {}, None, None)
    logged = logger.writter.metric_rows[-1][0]
    assert logged["train/rollout/reward/agent_ls/sum"] == -4.0
    assert logged["train/rollout/reward/agent_dc/sum"] == -3.0
    assert logged["train/rollout/reward/agent_bat/sum"] == -2.0
    assert "train/episode/reward/agent_ls/sum" not in logged
```

Add a naturally terminal two-thread evaluation test asserting two `eval/episode` records and one `eval/aggregate` record. Assert retained legacy tag `metrics/Average Net Energy` exists.

Add a BaseLogger test proving heterogeneous completed reward columns yield all four named returns:

```text
train/episode/reward/agent_ls/return
train/episode/reward/agent_dc/return
train/episode/reward/agent_bat/return
train/episode/reward/mean_across_agents/return
```

- [ ] **Step 2: Run tests and verify missing canonical output**

```bash
$PY tests/test_sustaindc_metrics.py
$PY tests/test_harl_reward_delivery.py
```

Expected: current loggers lack rows/tags.

- [ ] **Step 3: Extend `BaseLogger` without changing legacy means**

Keep `train_episode_rewards` and existing `train/average_step_rewards` exactly. Add raw `(threads, agents)` return accumulation and completed episode vectors. When a natural episode ends, retain all targeted returns plus their reporting mean. At log time, emit named per-agent aliases and update/actor/critic semantic aliases under `train/update/*`, retaining old tags.

At eval, keep `metrics/eval_*` behavior. Record separate evaluation return vectors, allowing SustainDC logger to emit true `eval/episode` rows/aggregates. Do not replace three values with the mean.

- [ ] **Step 4: Refactor `SustainDCLogger` around the accumulator**

Maintain:

```python
self.rollout_metrics
self.eval_thread_metrics
self.eval_episode_snapshots
```

`per_step` calls `super()` then adds `infos[thread][0]` once per thread. `episode_log` calls `super()`, emits `train/rollout/*` through `writer.log_metrics` and canonical TensorBoard aliases, retains all current `metrics/*` tags, then resets only rollout state.

`eval_init` makes one accumulator per eval thread. `eval_per_step` adds `eval_infos[thread][0]`. `eval_thread_done` finalizes and logs one `eval/episode` snapshot per naturally completed thread. `eval_log` computes `eval/aggregate` mean/std/min/max, emits it, retains current `eval_metrics/*`, and preserves existing selection score behavior.

- [ ] **Step 5: Wire terminal callbacks with no algorithm changes**

Ensure both runner loops call `eval_thread_done` after final `eval_per_step` and before thread data is reused. Supply an evaluation-event counter to distinguish equal global steps. Do not alter action selection, return computation, checkpoint selection, done detection, or reset behavior.

- [ ] **Step 6: Run focused integration tests**

```bash
$PY tests/test_sustaindc_metrics.py
$PY tests/test_harl_reward_delivery.py
$PY tests/test_pricing_integration.py
```

Expected: distinct return streams, one-time facility tariff accounting, rollout-only partial window labeling, and intact reward delivery.

## Task 5: Create complete evaluation traces and optional artifacts

**Files:**
- Modify: `harl/runners/on_policy_base_runner.py:554-763, 938-990`
- Modify: `harl/runners/off_policy_base_runner.py:523-713`
- Modify: `harl/envs/sustaindc/metrics.py`
- Modify: `tests/test_tracking.py`
- Modify: `tests/test_sustaindc_metrics.py`

**Interfaces:**
- Consumes: evaluation info/actions/dones, run directory, writer artifact API.
- Produces: lossless evaluation CSV traces and optional checkpoint/evaluation W&B artifacts after local persistence.

- [ ] **Step 1: Write failing trace and artifact tests**

```python
def test_trace_row_retains_billing_and_next_state_clocks():
    row = build_trace_row(
        info=_info(billed_day=333, billed_hour=23.75,
                   next_state_day=334, next_state_hour=0.0,
                   total_price_cost_this_step_c=7.5,
                   shared_tariff_cost_reward=-0.075),
        actions=np.array([[1], [2], [0]]),
        run_metadata={"seed": 1, "phase": "eval", "eval_episode": 2,
                      "env_thread": 0, "step_in_episode": 95},
        terminal=True, truncated=True,
    )
    assert row["billed_day"] == 333
    assert row["next_state_hour"] == 0.0
    assert row["reward/shared_tariff_cost_reward"] == -0.075
    assert row["terminal"] is True
```

Test CSV rows retain all optional keys, full precision, environment thread, physical episode ID, and three action columns. In `test_tracking.py`, test `log_artifact` receives only existing files and failed upload preserves those local files/status.

- [ ] **Step 2: Run and confirm helpers do not exist**

```bash
$PY tests/test_sustaindc_metrics.py
$PY tests/test_tracking.py
```

Expected: missing trace/artifact behavior.

- [ ] **Step 3: Implement trace helpers and collect evaluation rows**

Add to `metrics.py`:

```python
def build_trace_row(*, info, actions, run_metadata, terminal, truncated): ...
def write_trace_csv(rows, path): ...
```

Prefix reward/tariff/battery/PV/workload/SLA/thermal/power/water families. Preserve `billed_*` versus `next_state_*`, use `action/ls`, `action/dc`, `action/bat`, JSON-encode arrays, and never round floats.

On-policy evaluation collects one row per environment thread after each `eval_envs.step`, using `eval_infos[thread][0]` plus full action triplet. Track physical episode ID/step per thread. If `dump_eval_metrcs`, write `evaluation_data/eval_event-<event>-trace.csv`. Keep existing `all_agents_episode_<N>.csv` compatibility output and do not replace it with the trace. Add equivalent off-policy SustainDC collection. Render uses the same helper when render dumping is enabled.

- [ ] **Step 4: Publish artifacts after successful local writes**

After existing save completes, call `log_artifact("...", "sustaindc-checkpoint", paths, aliases)` with existing models, config, tracking status, tariff config, summary, and completed traces. Always alias `latest`; add `best` only after existing improved/equal evaluation selection. Publish evaluation trace after successful write as `sustaindc-evaluation`. Artifact failure cannot alter local save/checkpoint selection.

- [ ] **Step 5: Run trace/artifact focused tests**

```bash
$PY tests/test_tracking.py
$PY tests/test_sustaindc_metrics.py
$PY tests/test_pricing_integration.py
```

Expected: two clock domains and Experiment 1 fields survive locally; optional upload does not affect local output.

## Task 6: Document, verify, and smoke-test safely

**Files:**
- Modify: `tests/README.md`
- Modify: `README.md` if it claims TensorBoard is the only monitor
- Modify: `docs/superpowers/specs/2026-10-08-wandb-observability-design.md` only for implementation-path clarification
- Test: tracking and complete existing suite

**Interfaces:**
- Consumes: completed configuration, adapter, registry, runner lifecycle.
- Produces: accurate operator documentation and verified default/optional behavior.

- [ ] **Step 1: Write failing static/default assertions**

```python
def test_every_algorithm_config_has_identical_disabled_wandb_defaults():
    expected = {
        "enabled": False, "mode": "disabled", "project": "sustaindc-thesis",
        "entity": None, "group": None, "tags": [], "log_artifacts": False,
        "artifact_aliases": ["latest"], "failure_policy": "local_only",
    }
    for path in sorted((_REPO_ROOT / "harl/configs/algos_cfgs").glob("*.yaml")):
        assert yaml.safe_load(path.read_text(encoding="utf-8"))["logger"]["wandb"] == expected
```

Also assert documentation names `train/rollout`, `eval/episode`, and the independent-accounting limitation.

- [ ] **Step 2: Run assertion module**

```bash
$PY tests/test_tracking.py
```

Expected: passes after Tasks 1–5; if it already passes because prior task completed the exact behavior, document that expected green result before proceeding.

- [ ] **Step 3: Update docs**

Update `tests/README.md` with tracker/metric test rows, base/optional installation, disabled/offline/online examples, cadence semantics, independent-episode limitation, and taxonomy: update metrics, rollout facility economics, completed evaluation episodes/aggregates, trace artifacts. Update `README.md` only if it says TensorBoard is exclusive.

- [ ] **Step 4: Run focused tests then complete suite**

```bash
$PY tests/test_tracking.py
$PY tests/test_sustaindc_metrics.py
$PY tests/test_team_tariff_reward.py
$PY tests/test_harl_reward_delivery.py
$PY tests/test_battery_efficiency.py
$PY tests/test_pricing_integration.py
$PY tests/test_pricing_manager.py
$PY tests/test_hydro_quebec_pricing.py
$PY tests/test_observation_layout.py
$PY tests/run_all.py
```

Expected: all pass without W&B installed/network.

- [ ] **Step 5: Run disabled and conditional offline smoke tests**

```bash
$PY train_sustaindc.py --algo happo --exp_name wandb-disabled-smoke \
  --n_rollout_threads 1 --n_eval_rollout_threads 1 \
  --num_env_steps 128 --episode_length 128 --use_eval false
```

Then only if `requirements-wandb.txt` was already installed in the approved environment:

```bash
$PY train_sustaindc.py --algo happo --exp_name wandb-offline-smoke \
  --n_rollout_threads 1 --n_eval_rollout_threads 1 \
  --num_env_steps 128 --episode_length 128 --use_eval false \
  --wandb --wandb-mode offline
```

Expected: disabled run imports no W&B and logs normally; offline run creates local W&B data with no network request. These are software smokes, not economic results. If package unavailable, report the offline smoke skipped; do not install without approval.

- [ ] **Step 6: Inspect without committing**

```bash
git status --short
git diff --check
git diff -- requirements-wandb.txt harl train_sustaindc.py eval_sustaindc.py tests docs
```

Expected: intended uncommitted changes only and no whitespace errors.

## Spec coverage review

- Optional/lazy/default-disabled lifecycle: Tasks 1–2.
- Online/offline/disabled policy/config precedence: Task 2.
- Provenance/local status: Task 1.
- Existing TensorBoard preservation/dual writer: Task 1.
- Experiment 1 economics/rewards/operations/optimizer sources: Tasks 3–4.
- Facility tariff-once accounting/raw unit separation: Task 3.
- Rollout vs evaluation cadence and targeted returns: Task 4.
- Billed/next clocks and complete trace: Tasks 3 and 5.
- Opt-in artifact policy/local resilience: Task 5.
- Docs/full-suite/default/offline smoke verification: Task 6.

## Execution note

The user prohibited commits and Git history changes; do not commit while executing this plan. Every new production behavior follows test-first red/green cycles. The offline smoke is conditional on an already-installed optional W&B package and must be reported as skipped rather than installing dependencies without approval.
