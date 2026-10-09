# SustainDC W&B Observability Design

**Status:** Written design awaiting review before implementation planning.

## Purpose

Add comprehensive, optional Weights & Biases (W&B) observability to the active SustainDC simulator while retaining TensorBoard and every local output behavior. The design serves Experiment 1, where the workload, cooling, and battery agents receive distinct targeted returns from a shared tariff component plus agent-specific safeguards.

A tracked run must be scientifically auditable: it must record the exact physical, tariff, reward, and algorithm setup; distinguish tariff costs from reward-scale quantities; reconstruct each agent's return; and show operational outcomes without describing a partial HARL rollout as a completed 30-day billing episode.

## Goals

- Preserve TensorBoard, `config.json`, checkpoints, and existing local result layout.
- Keep W&B disabled by default and lazy-imported.
- Support explicit `disabled`, `offline`, and `online` tracking modes.
- Track all available optimizer, reward, tariff, battery/PV, workload/SLA, thermal/cooling, carbon, water, and evaluation metrics.
- Preserve all three Experiment 1 reward series; a mean across agents is a labelled reporting aggregate, never an actor objective.
- Record reproducible run provenance, evaluation traces, and opt-in versioned artifacts.
- Ensure W&B failure cannot change environment dynamics, reward calculations, checkpoints, or local TensorBoard output.

## Non-goals

- Replacing TensorBoard, local checkpoints, `config.json`, existing CSV exports, or the dashboard.
- Changing tariff calculations, reward values, physical dynamics, observations, or HARL reward delivery.
- Uploading a scalar for every training environment step. Long, detailed traces belong in artifacts.
- Treating independent 30-day episode accounting as a contiguous utility bill.
- Storing W&B credentials in YAML, result files, code, metrics, or artifacts.

## Principles

1. **SustainDC `info` is the source of truth.** The tracker consumes existing environment facts and does not recompute costs, reward components, PV flows, or battery accounting.
2. **Cadence is part of metric identity.** Collection windows, natural environment episodes, and evaluation aggregates must never share an ambiguous namespace.
3. **Facility cost is counted once.** Each vector environment step contributes one complete merged `info` record even though all agent info dictionaries contain it.
4. **Units remain separate.** Tariff costs remain cents, reward terms remain reward units, energies remain kWh, and verified power values remain kW.
5. **No tracker dependency for normal use.** Disabled tracking must never import `wandb`.
6. **No silent scientific loss.** Optional or missing fields are marked absent; required Experiment 1 telemetry is validated instead of silently becoming zero.

## Architecture

### Central dual-writer adapter

Add a focused tracking module under `harl/utils/`. It wraps the existing `tensorboardX.SummaryWriter` and is created at the current sole writer-construction seam, `init_dir(...)` in `harl/utils/configs_tools.py`.

The adapter preserves the writer interface used in the active repository:

```python
add_scalar(tag, scalar_value, global_step=None)
add_scalars(main_tag, tag_scalar_dict, global_step=None)
export_scalars_to_json(path)
close()
```

For a finite numeric scalar, it writes to TensorBoard first and mirrors the same tag/value/step to W&B. This retains all existing TensorBoard tags and captures scalar emissions from base loggers, SustainDC logger, and direct off-policy runner calls without duplicating metric emission across runner families.

The adapter additionally supports a structured scalar batch method for canonical SustainDC metrics. It emits one coherent W&B row for a rollout/evaluation record while retaining equivalent TensorBoard scalar tags. It does not use `sync_tensorboard=True`; explicit mirroring is deterministic and testable without event-file watcher timing.

When disabled, `init_dir` returns the existing TensorBoard writer behavior and imports no W&B code. When enabled, initialization occurs exactly once in the main runner after the local run directory exists. It uses that directory as W&B's local directory, avoiding a repository-root `wandb/` directory. The adapter's `close()` is idempotent and finishes W&B at most once after TensorBoard export/closure.

### W&B configuration and CLI

Each of the ten active algorithm YAML files gets the same default-disabled block beneath the existing `logger` mapping:

```yaml
logger:
  log_dir: "./results"
  wandb:
    enabled: false
    mode: disabled       # disabled | online | offline
    project: sustaindc-thesis
    entity: null
    group: null
    tags: []
    log_artifacts: false
    artifact_aliases: [latest]
    failure_policy: local_only  # local_only | error
```

`enabled: false` or `mode: disabled` means W&B is not imported. `offline` writes W&B offline state beneath the local run directory for a deliberate later `wandb sync`, without attempting a network call. `online` uses the caller's preconfigured W&B credentials and never invokes login.

Add explicit CLI options rather than using the current collision-prone generic leaf override mechanism:

```text
--wandb
--wandb-mode {online,offline,disabled}
--wandb-project PROJECT
--wandb-entity ENTITY
--wandb-group GROUP
--wandb-tags TAG [TAG ...]
--wandb-artifacts
```

After YAML or `--load_config` resolution, the options update only `algo_args["logger"]["wandb"]`. Explicit flags override loaded config; loaded config overrides YAML defaults. `--wandb` enables an otherwise disabled run in online mode. An explicit `--wandb-mode disabled` wins over `--wandb`.

The default `failure_policy: local_only` records a clear warning and tracking-status file, then continues with local TensorBoard/checkpoint/config output on missing package, initialization, or optional artifact failure. `error` raises a precise error before collection starts. Neither policy repeatedly retries a failed initialization.

### Provenance and lifecycle

The existing local directory remains authoritative:

```text
results/<env>/<location>/<algo>/<exp_name>/seed-<seed>-<timestamp>/
```

W&B names the run using the final local directory name, sets `job_type="train"`, and stores a newly generated tracking run ID in local provenance. It does not silently resume a prior training run.

The W&B config is a JSON-safe copy of the same fully resolved object written to `config.json`:

```text
main_args
algo_args
env_args
```

It also includes non-secret derived provenance: local run directory; algorithm/environment/experiment/seed; ordered agent names; state type; natural episode length; collection horizon; evaluation protocol; selected tariff after environment initialization; runtime/device/package versions; Git revision and dirty status when obtainable; and content hashes/source paths for the tariff, weather, carbon, workload, PV metadata, and DC configuration inputs. Missing Git metadata or hashes are explicitly recorded as unavailable, never invented.

`train_sustaindc.py` wraps `runner.run()` in `try/finally` after runner creation, ensuring `runner.close()` flushes local telemetry and invokes W&B finish if a run raises. The original exception is re-raised. Only the main runner creates W&B state; rollout threads never create their own runs.

Standalone `eval_sustaindc.py` defaults W&B to disabled even if a loaded training config enabled it. Explicit evaluation tracking creates a new `job_type="eval"` W&B run, grouped/tagged with source training identity. It never resumes or mutates the original training run.

### Artifacts

Artifact publication is opt-in (`log_artifacts`). A successful checkpoint may publish an immutable `sustaindc-checkpoint` artifact containing the files that exist at publication:

```text
config.json
tracking provenance/status
models/ actor, critic, and value-normalizer files
pricing parameter file
TensorBoard summary export when available
evaluation trace and summaries produced so far
```

A completed evaluation trace publishes a `sustaindc-evaluation` artifact. Artifacts always get `latest`; a `best` alias applies only after the existing evaluation-based checkpoint selection succeeds. Upload failure records status but never deletes or invalidates a local checkpoint.

## Metric contract

### Cadence and axes

Canonical metrics are lower-case, slash-delimited, and include cadence in their namespace:

```text
train/update/*       optimizer/update statistics at the configured log interval
train/rollout/*      one HARL collection window; not necessarily a natural episode
eval/episode/*       one completed natural SustainDC episode
eval/aggregate/*     aggregate of all completed evaluation episodes in one event
```

`global_env_step` is the shared W&B x-axis. Each structured record also contains `runner_update`, collection steps per environment, `eval_event`, and `eval_episode` where applicable. Existing TensorBoard tags remain unchanged and are mirrored. Canonical aliases use agent names, not only numeric agent IDs:

```text
agent_ls -> ls
agent_dc -> dc
agent_bat -> bat
```

The current Experiment 1 natural horizon is 30 days or 2,880 15-minute steps. A 128- or 1,024-step HARL `episode_length` is a collection window unless it actually reaches terminal state. Therefore all training aggregation is `train/rollout/*` unless natural terminal tracking produces an explicit `train/episode/*` record.

### Optimizer and algorithm metrics

Mirror all existing actor/critic scalar tags exactly. Add semantic aliases:

```text
train/update/actor/ls/<actor_info_key>
train/update/actor/dc/<actor_info_key>
train/update/actor/bat/<actor_info_key>
train/update/critic/<critic_info_key>
train/update/global_env_step
train/update/runner_update
train/update/throughput/env_steps_per_second
train/update/wall_time_s
```

The logger preserves every supplied actor/critic key for every supported HARL algorithm. HAPPO normally yields policy loss, entropy, gradient norm, importance ratio, critic value loss, critic gradient norm, and critic-buffer mean reward.

### Reward and tariff metrics

For each rollout and evaluation episode, aggregate each existing source field with stated reductions:

```text
<phase>/reward/shared_tariff_reward/{sum,mean_per_step}
<phase>/reward/agent_ls/{sum,mean_per_step}
<phase>/reward/agent_dc/{sum,mean_per_step}
<phase>/reward/agent_bat/{sum,mean_per_step}
<phase>/reward/mean_across_agents/{sum,mean_per_step}

<phase>/reward/penalty/ls_overdue/{sum,mean_per_step}
<phase>/reward/penalty/ls_dropped/{sum,mean_per_step}
<phase>/reward/penalty/ls_backlog/{sum,mean_per_step}
<phase>/reward/penalty/dc_thermal/{sum,mean_per_step}
<phase>/reward/penalty/dc_constraint/{sum,mean_per_step}
<phase>/reward/penalty/battery_terminal_soc/{sum,mean_per_step}
<phase>/reward/penalty/battery_degradation/{sum,mean_per_step}

<phase>/tariff/cost/energy_c
<phase>/tariff/cost/demand_increment_c
<phase>/tariff/cost/additional_c
<phase>/tariff/cost/total_c
<phase>/tariff/price_c_per_kwh/{mean,min,max}
<phase>/tariff/billing/running_peak_kw
<phase>/tariff/billing/energy_kwh
<phase>/tariff/billing/progress_fraction
```

Here `<phase>` is `train/rollout`, `eval/episode`, or `eval/aggregate` as appropriate. Evaluation aggregates provide mean, standard deviation, minimum, and maximum over completed evaluation episodes.

`reward/mean_across_agents` is a compatibility/reporting aggregate only. It is never called team return and never presented as the objective of any actor under FP targeted-reward training. The current checkpoint-selection aggregate is also emitted as an explicitly named `eval/aggregate/selection_score`, alongside all three agent returns.

Facility costs are summed from one complete `info` record per vector environment step. They are never derived from rewards or multiplied by three.

### Battery, grid, PV, and carbon metrics

Aggregate existing source fields as:

```text
<phase>/energy/grid_import_kwh
<phase>/energy/dc_bus_load_kwh
<phase>/energy/baseline_without_battery_kwh
<phase>/energy/battery_delta_kwh

<phase>/pv/generation_ac_kwh
<phase>/pv/self_consumed_kwh
<phase>/pv/curtailed_kwh
<phase>/pv/self_consumption_fraction
<phase>/pv/curtailment_fraction

<phase>/battery/soc/{start,end,min,max,mean}
<phase>/battery/charge_bus_kwh
<phase>/battery/discharge_bus_kwh
<phase>/battery/charge_cell_kwh
<phase>/battery/discharge_cell_kwh
<phase>/battery/cell_throughput_kwh
<phase>/battery/cumulative_cell_throughput_kwh
<phase>/battery/degradation_cost_c
<phase>/battery/cumulative_degradation_cost_c
<phase>/battery/action/{charge_steps,discharge_steps,idle_steps}
<phase>/battery/terminal_soc_restore_grid_kwh
<phase>/battery/terminal_soc_reference_price_c_per_kwh

<phase>/carbon/grid_intensity
<phase>/carbon/emissions
<phase>/carbon/emissions_per_grid_import_kwh
```

Interval energies/costs are summed; cumulative source quantities are reported as final values; SoC uses start/end/min/max/mean. Fractions have explicit zero-denominator handling: omit/null the derived ratio and emit a diagnostic rather than produce a false zero.

### Workload and SLA metrics

```text
<phase>/workload/original_fraction/{mean,sum}
<phase>/workload/shifted_fraction/{mean,sum}
<phase>/workload/shift_delta_fraction/{mean,sum}
<phase>/workload/tasks/{computed,processed}

<phase>/sla/queue_depth/{mean,max,final}
<phase>/sla/queue_depth_normalized/{mean,max,final}
<phase>/sla/queue_task_steps
<phase>/sla/dropped_tasks
<phase>/sla/overdue_tasks/{sum,max}
<phase>/sla/oldest_task_age_fraction/{mean,max}
<phase>/sla/average_task_age_fraction/mean
<phase>/sla/enforced_steps
```

`queue_task_steps` is an integral of instantaneous queue depth and is never labelled as unique queued tasks. Source values keep their native semantics—e.g. `ls_overdue_penalty` is a per-step count of tasks older than 24 hours and `ls_tasks_dropped` is a per-step count.

### Cooling, facility, and water metrics

```text
<phase>/power/ite_kw/{mean,max,p90}
<phase>/power/hvac_kw/{mean,max,p90}
<phase>/power/cooling_tower_kw/{mean,max,p90}
<phase>/power/chiller_kw/{mean,max,p90}
<phase>/power/facility_kw/{mean,max,p90}

<phase>/thermal/crac_setpoint_c/{mean,min,max}
<phase>/thermal/internal_temperature_c/{mean,max}
<phase>/thermal/ambient_temperature_c/{mean,min,max}
<phase>/thermal/limit_c
<phase>/thermal/temperature_margin_c/{mean,min}
<phase>/thermal/exceedance_degree_c_steps
<phase>/thermal/constraint_violation_steps

<phase>/water/liters
<phase>/water/liters_per_grid_import_kwh
<phase>/efficiency/pue
```

PUE is the ratio of aggregate applicable facility power to aggregate ITE power with a zero-denominator guard and is labelled estimated model PUE. The cooling-water pump fields currently have unverified source units; preserve them as `*_raw` in detailed traces but do not chart them as `_kw` until a dedicated source validation/test establishes the conversion.

### Billing, chronology, and tariff identity

Surface already-calculated billing facts required for reconciliation and trace alignment:

```text
billed_day
billed_hour
next_state_day
next_state_hour
billing_running_peak_kw
billing_period_energy_kwh
billing_progress_fraction
selected_tariff
```

The tariff is priced using the pre-advance billed clock. Post-step `day`/`hour` describe the next decision state, so reports preserve both rather than shifting bill charts by one interval. Rate-specific facts that are available from the model are trace metadata and named with an explicit tariff prefix, such as:

```text
rate_l/day_peak_kw
rate_l/optimization_charge_c
rate_l/winter_peak_kw
```

Tariff identity and rate-card constants belong in provenance/configuration, not high-frequency scalar history.

## Trace tables and exports

Training uses aggregate scalar history; it may write an explicitly configured, bounded sampled trace that always includes terminal steps. Evaluation and rendering write full 15-minute facility traces.

Each trace row has one facility record per step with:

```text
run identity, seed, phase, rollout ID, physical episode ID, environment thread
step-in-episode, global environment step
billed and next-state clocks
all three actions, terminal/truncated state
all current information fields needed by this metric contract
all tariff/reward components and terminal SoC diagnostics
```

The complete trace is saved locally under the run directory and uploaded only as an opt-in evaluation artifact. A W&B Table can display a bounded preview, but the artifact/CSV remains the authoritative complete record. Existing evaluation/render exports are extended rather than replaced with tariff/reward components, terminal SoC fields, detailed SLA signals, thermal constraints, and billed-clock fields.

## SustainDC logger boundary

`SustainDCLogger` becomes the facility metric collector. It reads one fully merged `info` record per environment step, maintains per-thread accumulators, and uses a declarative metric registry instead of scattered `dict.get()` calls. The registry records source key, canonical name, unit, and reduction (`sum`, `mean`, `min`, `max`, `last`, `count`, or percentile).

During training it emits `train/rollout/*` at current logging cadence. During evaluation it completes `eval/episode/*` per naturally terminal thread and emits `eval/aggregate/*` after the event. Existing TensorBoard tags remain intact, including the current legacy metrics. Required Experiment 1 fields cause clear diagnostics when absent in targeted mode; optional missing fields are emitted as an explicit absence/status rather than fabricated zeros.

## Verification requirements

All tests run without network access, a W&B account, or an installed `wandb` module except mocked enabled-path tests.

1. Disabled tracking never imports W&B and preserves TensorBoard writer behavior plus legacy config loading.
2. Enabled mocked tracking initializes once with resolved safe config, correct directory/name/tags/mode/job type, and receives every mirrored scalar with matching value and step.
3. Writer closure and W&B finish occur exactly once; a training exception runs cleanup while re-raising the original error.
4. YAML, loaded configuration, and explicit W&B flags resolve deterministic precedence; a legacy saved config without W&B settings stays disabled.
5. Hand-computed `info` sequences validate every registry reduction, tariff-once accounting, units, terminal values, and zero-denominator behavior.
6. Experiment 1 logs all three agent returns and every safeguard; reporting means do not replace targeted returns.
7. A partial collection logs only `train/rollout`; a naturally completed evaluation episode emits `eval/episode` values.
8. Evaluation trace output includes both clocks, tariff/reward decomposition, terminal SoC diagnostics, and operational/SLA/cooling fields.
9. Successful artifacts contain the intended local files; upload failure preserves local files and scalar tracking.
10. NaN, infinity, arrays, or unsupported values do not enter misleading W&B scalar history or crash an otherwise valid local-only run.

## Dependency policy update

W&B is now installed as a pinned normal dependency in `requirements.txt`:

```text
wandb==0.19.8
```

It remains runtime-disabled by default. The tracking adapter still lazy-imports
it whenever disabled, so normal local TensorBoard runs create no W&B run and
make no W&B network request.

## Implementation boundaries

Expected changes are limited to:

- `harl/utils/configs_tools.py` and a new focused tracking/writer module;
- `train_sustaindc.py` and `eval_sustaindc.py` for explicit tracking policy/lifecycle;
- `harl/common/base_logger.py` for semantic per-agent/algorithm aliases;
- `harl/envs/sustaindc/sustaindc_logger.py` for registry-driven facility aggregation;
- runner lifecycle seams for traces/artifacts;
- `sustaindc_env.py` and pricing output only to surface already-calculated chronology/billing facts;
- all algorithm YAML files for the default-disabled W&B block; and
- isolated tracking, logger, integration, and lifecycle tests.

## Acceptance criteria

A default training command remains TensorBoard/local-only and passes the current suite without W&B installed. An enabled offline mock/smoke run produces a local offline W&B directory, resolved provenance, mirrored existing scalars, canonical Experiment 1 facility metrics, three separate agent return series, and a full evaluation trace/artifact when requested. A dashboard can distinguish tariff cost, a safeguard trade-off, and an optimizer change without relying on an ambiguous averaged reward.
