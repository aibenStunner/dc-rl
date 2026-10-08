# Cooperative Tariff Reward and Billing Design

**Status:** Design approved in conversation; awaiting written-spec review before implementation planning.

## Purpose

This design defines the initial SustainDC reward and billing protocol for coordinated data-centre workload, cooling, battery, and PV control under Hydro-Québec tariffs. Its purpose is to provide an economically defensible cooperative baseline for the thesis ablation programme.

The initial condition is deliberately narrow: every agent receives the shared facility tariff signal, while service, thermal, and episode-boundary safeguards are assigned to the agent with the most direct control over the corresponding violation. Later reward constructs and MARL algorithms are separate, named experiments evaluated under matched physical and billing conditions.

## Scope

### Included

- A shared team reward for the workload, cooling, and battery agents.
- Workload/SLA, thermal/constraint, and terminal-state-of-charge safeguards.
- Independent chronological 30-day billing episodes for training.
- Contiguous pricing carry state and final settlement for headline evaluation.
- Explicit cooperative EP reward semantics in HARL, independent of agent order.
- Component-level logging, reconciliation, and tests.
- A stable reward configuration identifier and coefficient record for every run.

### Excluded from the initial condition

- Battery degradation in the learning reward.
- Synthetic hourly energy-arbitrage prices for Rate M or Rate L.
- Extra PV, charge/discharge, throughput, state-of-charge, or peak bonuses.
- A carbon objective.
- A full rolling 12-month ratchet implementation beyond pricing-model support.
- A new MARL algorithm.

## Economic source of truth

The pricing subsystem is the sole source of facility tariff economics. It prices actual net grid import and maintains tariff-specific billing state. The reward consumes its standardized facts; it does not reimplement price, peak, PV, or battery logic.

For interval `t`, the facility tariff cost in cents is:

```text
C_tariff[t] = C_energy[t] + C_demand[t] + C_additional[t]
```

The canonical source is `total_price_cost_this_step_c`; these component fields remain available for reconciliation:

```text
energy_cost_this_step_c
demand_charge_increment_c
additional_charge_increment_c
total_price_cost_this_step_c
```

Consequently, PV is rewarded only through reduced net import; battery dispatch is rewarded or penalized only by its actual tariff consequence; and peak avoidance is rewarded only through avoided incremental tariff demand charges.

## Initial reward: `team_tariff_targeted_safeguards`

The initial configuration is:

```yaml
reward:
  mode: team_tariff_targeted_safeguards
  scale: 0.01
  include_terminal_soc_settlement: true
  degradation_cost_weight: 0.0
```

`scale` uniformly converts the full objective from cents to a numerically convenient reward scale. It is not an economic parameter. No individual tariff component may be clipped, independently normalized, or flattened before this scale is applied.

Define:

```text
r_tariff[t] = -scale * C_tariff[t]
```

where `scale` is `reward.scale`.

The non-negative safeguard penalties are:

- `P_overdue[t]`: overdue workload;
- `P_dropped[t]`: dropped workload;
- `P_backlog[t]`: excessive queue/deadline backlog;
- `P_thermal[t]`: thermal safety or comfort violation;
- `P_constraint[t]`: cooling-operating constraint violation;
- `P_terminal_soc[t]`: terminal SoC restoration settlement;
- `P_degradation[t]`: battery degradation penalty, initially zero.

The shared facility-economic component is delivered to every agent, while safeguards are assigned to the agent with the most direct control over the associated condition:

```text
ls_reward[t] = r_tariff[t]
               - P_overdue[t]
               - P_dropped[t]
               - P_backlog[t]

dc_reward[t] = r_tariff[t]
               - P_thermal[t]
               - P_constraint[t]

bat_reward[t] = r_tariff[t]
                - P_terminal_soc[t]
                - P_degradation[t]
```

`P_degradation[t]` is identically zero in Experiment 1. The shared tariff component keeps all agents aligned with the facility bill. Targeted safeguards reduce irrelevant reward variance: for example, the battery does not receive a dropped-work penalty that it cannot directly correct, while the load-shifting agent does not receive a battery-boundary penalty that it cannot directly correct.

All components remain logged at facility scope. A future fully shared-safeguard experiment may deliberately give all agents the total penalty sum, but it is not the initial baseline.

### Prohibited additions

The initial reward must not include:

- a second peak bonus or penalty beyond incremental demand tariff charges;
- a synthetic time-of-use differential where the tariff does not provide one;
- a generic reward for battery charge, discharge, throughput, or SoC movement;
- an independent PV self-consumption bonus;
- a carbon term;
- degradation cost when `degradation_cost_weight` is zero;
- a bonus for finishing an episode above the target SoC.

## Terminal state-of-charge settlement

A finite episode can otherwise reward a final battery discharge without recognizing the energy required to restore the starting state. Let `soc_start` be the SoC at the start of a settlement horizon, `soc_end` its terminal SoC, `capacity_kwh` usable capacity, and `eta_charge` bus-to-cell efficiency. Required grid-side restoration energy is:

```text
restore_grid_kwh = max(0, soc_start - soc_end) * capacity_kwh / eta_charge
```

The terminal penalty is:

```text
P_terminal_soc = scale * restore_grid_kwh * terminal_energy_price
```

where `terminal_energy_price` is the standardized terminal marginal energy price from the active pricing model. It must not introduce an invented time-varying price.

Rules:

- Apply once at each independent training or robustness episode end.
- Apply once only at the final end of a contiguous headline evaluation horizon.
- Never apply at an internal carried segment boundary.
- Penalize ending below starting SoC only; do not reward excess terminal SoC.
- Log starting and terminal SoC, restoration energy, reference price, and the penalty separately.

## HARL reward and critic-state contract

The initial system uses the existing environment-provided shared critic state (EP), but reward semantics must remain explicit:

1. SustainDC emits one reward per active agent.
2. Every agent receives the same `r_tariff[t]` facility-economic component.
3. Each agent additionally receives only its own targeted safeguard penalties.
4. HARL must preserve the separate reward columns for any per-agent-return path; it must never silently substitute the first agent's column for all agents.
5. If an algorithm/critic mode requires a scalar team return, its aggregation rule must be configured, documented, logged, and tested rather than inferred from agent order.

The runner must remove the current implicit dependence on `rewards[:, 0]` as the facility objective.

The implementation must determine the exact HARL return/buffer contract supported by the selected out-of-the-box algorithm and expose it clearly. It must not silently average incompatible agent returns, select one agent as the team representative, or claim a cooperative single-return objective when the algorithm is consuming per-agent returns.

The structured `reward:` configuration supersedes ambiguous collaboration controls such as `individual_reward_weight` and `collab_reward`. Agent-targeted safeguards remain intentionally configured by the selected reward mode, rather than through legacy per-agent reward names whose active behavior is unclear.

A future fully shared-safeguard reward, FP critic-state comparison, or alternative MARL algorithm remains a distinct named experiment, not an implicit fallback.

## Billing lifecycle

### Training: independent chronological 30-day episodes

Each training episode is one chronological 30-day horizon selected from valid annual start positions.

1. Reset physical and pricing state independently.
2. Execute the contiguous 30-day period.
3. Accumulate tariff increments under that episode-local billing state.
4. Apply required pricing settlement and terminal SoC settlement once at termination.
5. Do not carry peaks, tiers, or other pricing state into the next sampled episode.

This is a practical training objective with recurrent demand-charge feedback. It is not a reconstructed annual utility bill.

### Headline evaluation: contiguous carried accounting

Headline evaluation is one contiguous chronological horizon. If it is executed or rendered through internal segments, valid pricing carry state must be exported, restored, and preserved across them.

1. Initialize pricing once for the headline horizon.
2. Preserve pricing carry state across internal segments.
3. Maintain physical continuity whenever the harness supports it; a segment cannot become an undeclared independent billing reset.
4. Apply pricing and terminal-SoC settlement once at the final horizon end.
5. Report the result as the modeled tariff ledger for the explicitly stated horizon.

Reports must state the start/end dates, horizon length, tariff scenario, and any pricing-accounting limitations. Claims about calendar or rolling-ratchet billing must not exceed implemented functionality.

### Randomized robustness evaluation

Independent random-start episodes are allowed for sensitivity analysis. They must be labelled as independent-episode economics rather than calendar-bill reconstructions, and each has its own terminal settlement.

## Logging and evaluation contract

The system must keep ledger facts separate from reward values.

### Per-step tariff and reward fields

```text
energy_cost_this_step_c
demand_charge_increment_c
additional_charge_increment_c
total_price_cost_this_step_c
billing_running_peak_kw

shared_facility_cost_c
shared_tariff_cost_reward
ls_overdue_penalty_component
ls_dropped_penalty_component
ls_backlog_penalty_component
dc_thermal_penalty_component
dc_constraint_penalty_component
bat_terminal_soc_penalty_component
bat_degradation_penalty_component
team_reward
```

### Required physical/service outcomes

```text
grid_import_kwh
pv_ac_kwh
pv_self_consumed_kwh
pv_curtailed_kwh
battery bus and cell throughput
battery degradation cost in cents
battery SoC
workload queue, overdue, and dropped-work measures
thermal and operating-constraint measures
```

### Required evaluation aggregates

Every evaluation result must separately report:

1. tariff ledger total and energy/demand/additional subtotals;
2. unscaled reward components and scaled team reward;
3. realized facility peak and relevant billing-state outcomes;
4. PV generation, self-consumption, and curtailment;
5. battery throughput and post-hoc degradation cost;
6. workload service outcomes;
7. thermal and operating outcomes;
8. reward mode and coefficients;
9. tariff/pricing configuration;
10. billing mode and evaluation horizon;
11. algorithm configuration and seed/run identity.

The initial study reports tariff cost and the following post-hoc degradation sensitivity separately:

```text
posthoc_operating_cost = tariff_cost + degradation_cost
```

The latter is not the initial learning objective.

## Reconciliation invariants

Subject only to documented rounding:

```text
sum(total_price_cost_this_step_c) + final pricing settlement
= reported modeled tariff cost
```

```text
ls_reward[t]  = shared_tariff_cost_reward[t]
                - P_overdue[t] - P_dropped[t] - P_backlog[t]

dc_reward[t]  = shared_tariff_cost_reward[t]
                - P_thermal[t] - P_constraint[t]

bat_reward[t] = shared_tariff_cost_reward[t]
                - P_terminal_soc[t] - P_degradation[t]
```

```text
shared_tariff_cost_reward[t]
= -reward.scale * total_price_cost_this_step_c[t]
```

The terminal step includes terminal settlement exactly once. Identical delivery of the tariff component does not cause facility cost to be counted three times in evaluation.

## Required tests

1. Every agent receives the identical shared tariff-cost component for the same pricing-ledger step.
2. Each agent's emitted reward equals its shared tariff component minus only that agent's configured safeguard components, within documented floating-point tolerance.
3. The tariff reward component equals the standardized pricing total times only the configured global sign and scale.
4. Facility reporting counts tariff cost once, not once per agent.
5. Excluded terms—PV bonus, battery-action bonus, synthetic time-of-use reward, carbon term, and zero-weight degradation—do not affect the initial reward.
6. Each workload safeguard responds only to its intended queue, deadline, overdue, or drop condition, and affects the load-shifting reward only.
7. Thermal and cooling-constraint safeguards respond only to their documented violation states and affect the cooling reward only.
8. Terminal-SoC settlement affects the battery reward only, while remaining visible in facility-level reporting.
9. Normal feasible trajectories incur no safeguard penalty.
10. Explicit configuration, including flexible-load settings, reaches each affected subsystem.
11. A terminal SoC deficit yields the expected efficiency-aware restoration penalty; equal or excess SoC yields no bonus.
12. Independent episodes settle terminal SoC once; a contiguous carried evaluation settles it once only at final completion.
13. The selected HARL path preserves agent reward columns or uses an explicit, configured aggregation rule; it never selects a reward by fixed agent index.
14. Reordering the agent list does not change the configured tariff component, per-agent reward contract, or any explicit aggregation result.
15. A regression test prevents return to the implicit `rewards[:, 0]` behavior.
16. Thirty-day training episodes do not inherit previous price/peak/tier state.
17. Contiguous evaluation preserves valid pricing carry state across internal segments.
18. Pricing ledger totals, final settlement, evaluation exports, and reward facts reconcile.
19. Randomized robustness reports are labelled as independent accounting.

## Implementation boundaries

The implementation plan must inspect and specify exact changes in the current tree, expected to include:

- reward construction and component logging in [utils/reward_creator.py](../../utils/reward_creator.py) and SustainDC reward dispatch;
- reward and billing configuration in [harl/configs/envs_cfgs/sustaindc.yaml](../../harl/configs/envs_cfgs/sustaindc.yaml) and associated validation;
- cooperative EP reward validation/insertion in [harl/runners/on_policy_base_runner.py](../../harl/runners/on_policy_base_runner.py);
- pricing carry lifecycle and terminal settlement in SustainDC reset/step/evaluation paths;
- evaluation/render/logger exports;
- dedicated reward, HARL-contract, billing-lifecycle, and reconciliation tests.

It must preserve all existing uncommitted work. No commit, amend, reset, rebase, cherry-pick, push, or other Git-history action is authorized.

## Experimental reproducibility

Every saved run and headline result must record:

```text
reward mode and all coefficients
billing mode and horizon definition
tariff model, tariff parameter file, and selected tariff
battery and PV configuration
algorithm and its configuration
seed(s)
training-episode protocol
evaluation carry/settlement protocol
```

## Reward and algorithm experiments

The current historical reward configuration is not numbered as an experiment because it optimizes a different, mixed objective and does not provide a valid baseline for this tariff/peak-shaving study.

Run the following controlled experiments in order. Each experiment must use the same physical scenario, tariff configuration, chronological billing protocol, seeds, evaluation horizons, and baselines unless the experiment explicitly varies one of those factors.

### Experiment 1 — Shared tariff cost, targeted safeguards

```text
mode: team_tariff_targeted_safeguards
LS reward:      shared tariff cost - workload/SLA safeguards
DC reward:      shared tariff cost - thermal/constraint safeguards
Battery reward: shared tariff cost - terminal-SoC safeguard
Degradation:    report-only; coefficient = 0
Algorithm:      first successful out-of-the-box HARL cooperative configuration
Critic state:   retain the existing EP state representation initially
```

This is the required initial baseline. It tests whether all three controllers can coordinate around one real facility bill without burdening each agent with penalties it cannot directly correct.

### Experiment 2 — Fully shared safeguards

```text
mode: team_tariff_shared_safeguards
Every agent: shared tariff cost - every safeguard penalty
Degradation: report-only; coefficient = 0
```

Compare this against Experiment 1 to test the cooperative-credit-assignment hypothesis: whether making every agent responsible for all facility constraints improves coordination or instead introduces harmful reward variance.

### Experiment 3 — Shared tariff cost only

```text
mode: team_tariff_only
Every agent: shared tariff cost only
```

This measures what the service, thermal, and terminal-state safeguards contribute. It is valid only if evaluation reports violations in full and the run is labelled an unconstrained economic ablation rather than an operationally acceptable policy.

### Experiment 4 — Degradation-aware targeted safeguards

```text
mode: team_tariff_targeted_safeguards_degradation
Same as Experiment 1, plus battery degradation cost with a non-zero, recorded weight
```

This tests whether economically priced cycling changes peak-shaving behavior, tariff savings, and post-hoc total operating cost relative to report-only degradation.

### Experiment 5 — Carbon-secondary targeted safeguards

```text
mode: team_tariff_targeted_safeguards_carbon_secondary
Same as Experiment 1, plus an explicit, separately reported carbon coefficient
```

This is a multi-objective sensitivity study, not a replacement for the tariff objective. It must report cost/emissions trade-offs rather than collapsing them into an unqualified single result.

### Experiment 6 — Critic/reward-delivery and MARL algorithm ablations

After Experiment 1 has trained and evaluated successfully:

```text
- compare supported HARL critic-state/reward-delivery modes under Experiment 1's reward definition;
- compare out-of-the-box algorithms such as HAPPO, MAPPO, HATRPO, and HAA2C only under matched conditions;
- evaluate any new critic, credit-assignment method, or MARL algorithm against Experiment 1 and the strongest matched out-of-the-box result.
```

The exact eligible HARL modes must be established by code inspection and a successful controlled baseline before interpreting them as scientific comparisons.

Each experiment must have a distinct stable name and save all reward coefficients, algorithm settings, and billing/evaluation protocol with its outputs.
