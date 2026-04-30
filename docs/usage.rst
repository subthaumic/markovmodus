Usage
=====

Quick start
-----------

Install ``markovmodus`` in a virtual environment:

.. code-block:: bash

   pip install markovmodus

For local development you can install the editable project with optional tooling:

.. code-block:: bash

   pip install -e .[dev]

Generate a dataset from Python:

.. code-block:: python

   from markovmodus import SimulationParameters, simulate_dataset

   params = SimulationParameters(
       num_states=4,
       num_genes=400,
       num_cells=1000,
       t_final=20.0,
       dt=1.0,
       markers_per_state=120,
       default_transition_rate=0.05,
       baseline_expression=2.0,
       marker_expression=6.0,
       rng_seed=7,
   )

   adata = simulate_dataset(params)
   df = simulate_dataset(params, output="dataframe", save_path="counts.csv")
   print(f"Generated AnnData with {adata.n_obs} cells")

Advanced configuration
----------------------

- Provide explicit static transition rates by passing ``transition_rates`` to :class:`SimulationParameters`.
  The array must be square with a zero diagonal; negative entries are clipped to zero.
  ``transition_matrix`` is still accepted as a deprecated alias.
- Pass a callable ``transition_rates`` when rates should depend on simulation time or the current
  cell population. The callable receives a ``SimulationState`` with ``time``, ``cell_states``,
  ``unspliced`` and ``spliced`` arrays:

  .. code-block:: python

     import numpy as np
     from markovmodus import SimulationParameters, SimulationState

     def rates(state: SimulationState) -> np.ndarray:
         matrix = np.zeros((3, 3), dtype=float)
         matrix[0, 1] = 0.02 if state.time < 10.0 else 0.2
         counts = np.bincount(state.cell_states, minlength=3)
         matrix[1, 2] = 0.01 + 0.001 * counts[1]
         return matrix

     params = SimulationParameters(
         num_states=3,
         num_genes=300,
         num_cells=1000,
         t_final=20.0,
         markers_per_state=100,
         transition_rates=rates,
     )

  The simulator uses a fixed-``dt`` discretized CTMC approximation, not exact Gillespie simulation.
  Dynamic rates are evaluated at the start of each time step and held constant over that step.
- Supply ``state_expression`` to override the automatically generated marker profiles.
- Choose ``marker_reuse_cap=1`` for per-state unique markers or ``marker_reuse_cap=2`` to draw balanced pairwise overlaps. Configurations must satisfy ``num_states * markers_per_state <= 2 * num_genes`` when sharing is enabled.
- Set ``dispersion`` to add per-gene negative-binomial measurement noise after the stochastic simulation.
- Use ``simulate_dataset(..., output="both")`` when you need both an ``AnnData`` object and
  a pandas DataFrame view in one call.
