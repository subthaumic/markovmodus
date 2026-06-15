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

   import numpy as np
   from markovmodus import SimulationParameters, simulate_dataset

   transition_rates = np.full((4, 4), 0.05, dtype=float)
   np.fill_diagonal(transition_rates, 0.0)

   params = SimulationParameters(
       num_states=4,
       transition_rates=transition_rates,

       num_genes=400,
       markers_per_state=120,
       marker_overlap=20,
       baseline_expression=2.0,
       marker_expression=6.0,

       num_cells=1000,
       t_final=20.0,
       dt=1.0,
       rng_seed=7,
   )

   adata = simulate_dataset(params)
   df = simulate_dataset(params, output="dataframe", save_path="counts.csv")
   print(f"Generated AnnData with {adata.n_obs} cells")

``adata.X`` contains spliced counts. The paired unspliced and spliced matrices are also available
as ``adata.layers["unspliced"]`` and ``adata.layers["spliced"]``.

Advanced configuration
----------------------

- Provide explicit static transition rates by passing ``transition_rates`` to :class:`SimulationParameters`.
  The array must be square with a zero diagonal; negative entries are clipped to zero.
- Pass a callable ``transition_rates`` when rates should depend on simulation time or the current
  cell population. The callable receives a ``SimulationState`` with ``time``, ``cell_states``,
  ``unspliced``, ``spliced``, ``birth_parent``, ``birth_time``, ``generation`` and
  ``last_division_time`` arrays:

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
         transition_rates=rates,

         num_genes=300,
         markers_per_state=100,

         num_cells=1000,
         t_final=20.0,
     )

  The simulator uses a fixed-``dt`` discretized CTMC approximation, not exact Gillespie simulation.
  Dynamic rates are evaluated at the start of each time step and held constant over that step.
- During each fixed-``dt`` step, cells sample at most one latent transition. The transition
  probability is ``1 - exp(-total_outgoing_rate * dt)``, and the target state is sampled
  proportional to the outgoing rates for the current latent state.
- Supply ``state_expression`` to provide the full state-by-gene steady-state spliced-count target
  matrix directly. The simulator uses this target to set unspliced production rates via
  ``production_rate = decay_rate * state_expression``.
- ``marker_overlap`` can be a scalar integer, meaning every pair of states connected by a non-zero static transition rate shares that many marker genes, or a dict mapping exact state groups to shared-marker counts, for example ``{(0, 2): 5, (1, 2, 3): 4}``.
  ``marker_reuse_cap`` is the maximum number of states allowed to share one marker gene.
  The simulator checks feasibility against ``markers_per_state`` and ``num_genes`` and raises a clear error when an overlap request cannot be assigned.
- Set ``dispersion`` to add per-gene negative-binomial measurement noise after the stochastic simulation.
- Pass ``proliferation_model`` to simulate cell divisions. The callable receives a
  ``SimulationState`` and returns ``(division_rates, daughter_state_probs)``. Rates may be
  state-level with shape ``(num_states,)`` or per-cell with shape ``(current_num_cells,)``;
  daughter-state probabilities use matching rows and one column per latent state.
  A dividing cell is replaced by one daughter in the existing row and appends a second daughter
  row. Both daughters copy the post-expression-update unspliced/spliced counts and independently
  sample latent states from the same daughter-state probability row. Dividing cells skip ordinary
  latent transitions in that same time step; non-dividing cells still transition normally.
  When proliferation is enabled, ``num_cells`` is the initial population size and outputs may
  contain additional rows. DataFrame and AnnData observations include ``birth_parent``,
  ``birth_time``, ``generation`` and ``last_division_time`` lineage columns.
- Use ``simulate_dataset(..., output="both")`` when you need both an ``AnnData`` object and
  a pandas DataFrame view in one call.
