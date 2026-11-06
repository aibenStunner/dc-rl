---
name: scaling_paired_check_summary
description: Verified paired 10-episode results for 1 MW and 10 MW scaled SustainDC (facilities ~998.5 vs 10038.4 kW design; battery 1/2 MWh vs 10/20 MWh); also notes no evaluation trace (dump_eval_metrcs disabled), no external W&B upload, 23/24 test modules pass with one pre-existing tracking failure.
---

# Paired 1 MW / 10 MW verification summary

**Method**: 10 episodes each, id=2026, seed=2026, fixed actions (LS=1, DC=1, bat=charged/discharged by hour), matched start days [68,28,...292], independent billing episodes.

**Verified design calibration** (direct factory measurement):
- 1 MW → 0.9985 MW, −0.149%, 9 racks, 189 servers/rack, 1,701 total, 33.0 °C ambient, 1 MW / 2 MWh battery.
- 10 MW → 10.0384 MW, +0.384%, 79 racks, 200 servers/rack, 15,800 total, 33.0 °C ambient, 10 MW / 20 MWh battery.

**Paired physical outcomes (mean of 10 episodes)**:
- Facility: 415,733 / 4,105,170 kWh (9.87×)
- IT: 306,622 / 2,942,937 kWh (9.57×)
- HVAC: 109,111 / 1,170,273 kWh (10.73×)
- Grid import: 409,491 / 4,046,013 kWh (9.87×)
- Peak demand: 791.8 / 7,834.4 kW (9.89×) — not at nameplate because workload trace ~46% utilization.
- PV: 5,916 / 59,157 kWh (10.00×)
- Cost (modeled tariff): 3,695,676 / 23,550,304 cents (6.37×) — Rate M vs Rate L.
- Battery dispatch (with assigned charge/discharge pattern): rated 1/10 MW reached; discharge only when load permits; 8–14% round-trip loss.

**Key limits observed**:
- No evaluation trace saved; only scalar episode summaries present.
- W&B offline; not uploaded (permission denied; not performed).
- Both runs completed 610 updates / 9,994,240 steps.
- One seed; matched conditions only between these two runs.
- Thermal margin compares mean outlet temperature to max CRAC setpoint — not a validated safe-temperature test.
- Queue/task metrics are model-accounting units; not all downstream metrics are directly comparable.

**Files**: results/happo-1mw-scale-seed101/seed-00102-.../ (if present) and results/happo-10mw-scale-seed102/seed-00102-.../ with config.json, tracking_status.json, progress.txt.
