# Montréal–Trudeau On-Site PV Scenario

## Weather provenance

PV production uses the same EPW file as the facility thermal model:

```text
CAN_QC_Montreal-Trudeau.Intl.AP.716270_TMYx.2011-2025.epw
```

> Climate.OneBuilding (2026). *CAN_QC_Montreal-Trudeau.Intl.AP.716270 TMYx
> Weather File, 2011–2025*. Derived from NCEI Integrated Surface Database and
> ERA5 reanalysis data. Climate.OneBuilding.org.

The weather file is a typical meteorological year, not an observed
chronological year. Its source period is 2011–2025, with representative
months selected from that period.

## PV baseline

| Assumption | Value |
|---|---:|
| System size | 5% of configured data-centre nameplate, adjustable by configuration |
| Array | fixed, monofacial, true south |
| Tilt | 35° |
| Azimuth | 180° |
| DC/AC ratio | 1.20 |
| Module temperature coefficient | −0.0037 /°C |
| Temperature model | pvlib SAPM close-mount glass/glass |
| DC system losses | 14.08% PVWatts aggregate loss assumption |
| Inverter nominal efficiency | 96% |
| Export policy | curtailment; no net-metering credit |

The simulator uses pvlib 0.13.1 with Perez irradiance transposition, SAPM
cell-temperature estimation, PVWatts DC conversion, PVWatts losses, and a
PVWatts inverter model. The hourly EPW PV output is zero-order-held over the
four 15-minute simulator intervals so hourly energy is conserved without
inventing sub-hour cloud variability.

## Electrical boundary

PV AC generation first offsets gross data-centre bus demand and battery
bus-side charging. The simulator reports:

```text
net_bus_kwh = gross_dc_load_kwh + battery_charge_bus_kwh
               - battery_discharge_bus_kwh - pv_ac_kwh
grid_import_kwh = max(0, net_bus_kwh)
pv_curtailed_kwh = max(0, -net_bus_kwh)
```

Surplus PV is curtailed. No export revenue, net-metering credit, or negative
grid import is modeled.

## Configuration

The active baseline configuration is in
`harl/configs/envs_cfgs/sustaindc.yaml`:

```yaml
pv:
  enabled: true
  capacity_fraction_of_datacenter: 0.05
  surface_tilt_deg: 35.0
  surface_azimuth_deg: 180.0
  dc_ac_ratio: 1.20
  gamma_pdc_per_deg_c: -0.0037
  system_losses_fraction: 0.1408
  inverter_efficiency: 0.96
  temperature_model: close_mount_glass_glass
  export_policy: curtail
```

For the current 1 MW data-centre scenario, this corresponds to 50 kWdc of
PV nameplate capacity. The 5% sizing is a roof-constrained scenario
assumption, not a claim about a surveyed facility.

## Deliberate limitations

This phase does not model panel snow coverage or clearing, row/horizon/object
shading, bifacial modules, roof geometry, availability outages, or transformer
losses. The EPW snow-depth field is not treated as panel snow coverage. These
can be added as separately parameterized scenarios when site data justify
them.
