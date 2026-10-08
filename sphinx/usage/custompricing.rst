.. _custompricing_ref:

========================
Custom Pricing Models
========================

SustainDC treats electricity pricing as a configurable, stateful subsystem:

.. code-block:: text

   SustainDC -> PriceManager -> PricingModel -> charges and observations

``PriceManager`` owns shared billing-period state and passes meter readings to
the selected model. A pricing model determines the tariff-specific energy,
demand, and additional charges. Reward functions consume the standardized
charges and do not implement tariff rules.

Configuration
-------------

Every environment configuration must contain one ``pricing`` mapping:

.. code-block:: yaml

   pricing:
     model: hydro_quebec
     config_file: data/Pricing/hydro_quebec_2026.yaml
     options:
       tariff: auto
       demand_floor_kw: 0.0

``model`` is either a built-in name or an explicit Python import path in
``module:attribute`` form. ``config_file`` is absolute or relative to the
repository root. ``options`` contains runtime selections rather than source
tariff figures.

The loader rejects unknown keys, unsupported schema versions, malformed
parameter files, missing files, and legacy top-level pricing keys before the
environment starts.

Built-in models
---------------

=================  ============================================================
Model              Purpose
=================  ============================================================
``hydro_quebec``   Rate M, Rate L, and a clearly labelled proposed data-centre
                   scenario using ``data/Pricing/hydro_quebec_2026.yaml``.
``flat``            Constant energy price.
``time_of_use``     Repeating 24-hour energy-price schedule.
``time_series``     CSV-backed exogenous price series.
=================  ============================================================

Hydro-Québec model
------------------

The supplied ``hydro_quebec_2026.yaml`` records the rate source, effective
date, seasonal boundaries, selection threshold, and tariff figures. The
runtime ``tariff`` option accepts:

.. code-block:: yaml

   options:
     tariff: auto       # Rate M below 5 MW; Rate L at or above 5 MW
     demand_floor_kw: 0.0

``rate_m`` and ``rate_l`` force a specific approved rate. ``new_dc_rate``
selects the proposed large-data-centre scenario explicitly; it is never
automatically selected and its YAML metadata marks it as unapproved and
approximate.

Units
-----

- Energy price: cents per kilowatt-hour (``cents/kWh``)
- Demand charge: cents per kilowatt per billing period (``cents/kW-month``)
- Meter energy: kilowatt-hours (``kWh``)
- Meter demand: kilowatts (``kW``)

Observation and reward contract
-------------------------------

Every pricing model exposes the same three observation values:

1. normalized current energy price;
2. model-defined billing-progress fraction;
3. normalized running demand peak.

The values remain in the same three trailing positions of every agent
observation. Pricing information in the environment ``info`` dictionaries and
reward parameters uses these model-neutral keys:

.. code-block:: text

   energy_cost_this_step_c
   demand_charge_increment_c
   additional_charge_increment_c
   total_price_cost_this_step_c
   norm_price
   price_denorm_c_per_kwh

``default_price_reward`` uses only ``total_price_cost_this_step_c``.

Parameter-file schemas
----------------------

Flat pricing:

.. code-block:: yaml

   schema_version: 1
   model: flat
   metadata:
     name: Example flat rate
     currency: CAD
   energy_price_c_per_kwh: 5.0

Time-of-use pricing:

.. code-block:: yaml

   schema_version: 1
   model: time_of_use
   metadata:
     name: Example daily schedule
     currency: CAD
   hourly_prices_c_per_kwh: [5, 5, 5, 5, 5, 5, 8, 8, 8, 8, 8, 8,
                              6, 6, 6, 6, 9, 9, 9, 9, 6, 6, 5, 5]

Time-series pricing:

.. code-block:: yaml

   schema_version: 1
   model: time_series
   metadata:
     name: Example hourly series
     currency: CAD
   csv_file: data/Pricing/example_prices.csv
   price_column: price_c_per_kwh
   source_interval_minutes: 60
   wrap: true

The time-series model applies zero-order hold when the source interval is an
integer multiple of the simulation timestep. It rejects unsupported ratios
rather than silently interpolating billing prices.

Custom Python plug-ins
----------------------

A custom model can be loaded without modifying the repository:

.. code-block:: yaml

   pricing:
     model: my_package.pricing:CampusTariff
     config_file: data/Pricing/campus_tariff.yaml
     options:
       account: research_site

The class must implement ``PricingModel`` from ``utils.pricing.contracts``:

.. code-block:: python

   class CampusTariff:
       name = "campus_tariff"

       def __init__(self, parameters):
           self.parameters = parameters

       def reset(self, *, context, state, clock, carry_state, options):
           state.model_state["account"] = options["account"]

       def step(self, *, context, state, clock, reading):
           return PricingCharges(
               energy_cost_c=reading.energy_kwh * self.parameters["price_c_per_kwh"],
               demand_cost_increment_c=0.0,
               additional_cost_increment_c=0.0,
           )

       def observe(self, *, context, state, clock):
           return PricingObservation(
               current_price_c_per_kwh=self.parameters["price_c_per_kwh"],
               normalized_price=1.0,
               forecast_normalized=np.ones(context.future_steps, dtype=np.float32),
               billing_progress=0.0,
               normalized_peak=0.0,
           )

       def export_carry_state(self, state):
           return {}

Model state and carry state must contain only JSON-like values: ``None``,
booleans, finite integers/floats, strings, lists, and string-keyed dictionaries.
Keep open handles, locks, clients, and other ephemeral resources on the model
instance rather than in shared pricing state.

Carry state and settlement
--------------------------

``PriceManager`` exports carry state at reset boundaries and restores it into
the next episode. Hydro-Québec carry state includes winter peak demand and the
ending billing peak. Models with a pending final-period charge may implement
``settle_pending_period(state=...)``; SustainDC calls it on episode truncation
before calculating the terminal reward.

Limitations
-----------

The supplied Hydro-Québec model does not model supply-voltage credits,
transformation losses, taxes, account fees, or a full rolling 12-month history
that ages old winter peaks out. The proposed data-centre scenario is pending
approval and its public parameterization is incomplete; use it only as an
explicit scenario, not a current tariff.
