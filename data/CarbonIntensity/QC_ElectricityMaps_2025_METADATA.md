# Québec hourly direct carbon intensity — 2025

## Simulator file

`QC_NG_&_avgCI.csv` is a simulator-ready, 8,760-row hourly input for
`CI_Manager`. It has the existing required columns:

```text
timestamp,avg_CI
```

`avg_CI` is the Electricity Maps field **Carbon intensity gCO₂eq/kWh
(direct)** for the Québec electricity zone (`CA-QC`), expressed in
gCO₂eq/kWh. The timestamps remain in UTC.

## Why direct intensity is used

This file is the operational control signal for the simulator. The agents make
15-minute scheduling, cooling, and battery decisions, so the relevant
quantity is the hourly direct grid intensity associated with electricity
consumption at that time. It is an average operational intensity, **not** a
marginal emissions factor.

The source's lifecycle-intensity field is intentionally not used as the
control signal. Lifecycle intensity is appropriate for a separate annual/LCA
reporting sensitivity because it includes generating-asset construction and
other non-operational emissions that an hourly controller cannot change.

## Provenance

Raw source snapshot retained beside this file:

```text
QC_ElectricityMaps_2025_hourly_raw.csv
```

Transformation:

1. Validate all records identify Canada / Québec / `CA-QC`.
2. Select only the 2025 `Carbon intensity gCO₂eq/kWh (direct)` column.
3. Convert timestamps from source ISO UTC form to the simulator's existing
   `YYYY-MM-DD HH:MM:SS+00:00` spelling.
4. Preserve all 8,760 non-leap-year hourly values without interpolation,
   aggregation, imputation, or timezone conversion.

Observed direct-intensity summary after validation:

| Statistic | gCO₂eq/kWh |
|---|---:|
| Minimum | 0.00 |
| Maximum | 45.15 |
| Mean | 11.45 |

## Required citation

> Electricity Maps (2025). Canada Québec 2025 Hourly Carbon Intensity Data
> (Version January 27, 2025). Electricity Maps.
> https://www.electricitymaps.com

## Raw 2024 companion

The original 2024 snapshot is retained at the thesis repository root because
it has 8,784 leap-year rows and cannot be loaded by the current manager,
which requires exactly 8,760 hourly values. It is suitable for a future
leap-year support or out-of-sample robustness workflow, not this default
simulator input.

Required citation for that source:

> Electricity Maps (2025). Canada Québec 2024 Hourly Carbon Intensity Data
> (Version January 27, 2025). Electricity Maps.
> https://www.electricitymaps.com
