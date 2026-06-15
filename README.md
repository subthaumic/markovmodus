# markovmodus
[![CI](https://github.com/subthaumic/markovmodus/actions/workflows/ci.yml/badge.svg)](https://github.com/subthaumic/markovmodus/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/markovmodus.svg)](https://pypi.org/project/markovmodus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Markov-modulated splicing simulator for single-cell U/S counts.**

Generate synthetic datasets where a **hidden state graph** (the off-diagonal rates of a continuous-time Markov process) **modulates** unspliced->spliced RNA dynamics.


## Why this exists

Single-cell RNA sequencing captures each cell only once, so trajectory and velocity methods must infer temporal structure from static observations.
Without datasets where the true lineage graph is known, it is hard to validate the assumptions those methods make.

`markovmodus` fills that gap by generating synthetic unspliced/spliced counts with an explicit hidden-state lineage.
Cells hop between phenotypic states according to a continuous-time Markov process, and each state drives its own transcriptional kinetics.

For biologists, think of the hidden states as cellular programs—progenitors, intermediates, terminal fates—with transition rates describing how readily a cell exits one program and commits to another.
Within each program, genes produce pre-mRNA that is spliced and degraded, yielding both nascent and mature counts like those used in RNA velocity analyses.
Because the simulator records the exact state graph, you can stress-test algorithms that aim to recover branching, cyclic, or linear progressions from single-time observations.


## Model

### Latent dynamics (state process)
- States are indexed z = 0, ..., n - 1.
- Users provide an off-diagonal rate matrix ``transition_rates`` with shape n-by-n. Entry ``transition_rates[i, j]`` is the continuous-time rate for a cell in state i to jump to state j, and the diagonal is forced to zero.
- For dynamic transition rates, ``transition_rates`` can be a callable that receives the current ``SimulationState`` and returns either a global n-by-n matrix or a per-cell tensor with shape ``(current_num_cells, n, n)``.
- Internally, the simulator samples at most one latent jump per cell per fixed-``dt`` step with probability ``1 - exp(-sum_j rate[i, j] * dt)`` and chooses the target proportional to the outgoing off-diagonal rates.
- **Directionality** arises from asymmetric off-diagonal rates.

### U/S dynamics (per gene, in state z)
- ``state_expression`` is a state-by-gene matrix of steady-state **spliced-count targets**. The shorthand marker settings generate this matrix automatically.
- For a target spliced level ``S*_z``, the simulator uses global splicing rate ``beta`` and decay rate ``gamma`` with production rate ``alpha_z = gamma * S*_z``:
  - dU/dt = alpha_z - beta * U
  - dS/dt = beta * U - gamma * S
- The implementation uses stochastic fixed-``dt`` updates: unspliced production is Poisson, splicing events are sampled from current unspliced counts and spliced degradation is binomial.
- Initial U/S counts are sampled near the implied steady state, with ``U*_z = gamma * S*_z / beta`` and ``S*_z = state_expression[z]``.
- Final counts can be perturbed by negative-binomial measurement noise by setting ``SimulationParameters.dispersion``.

### Topology encoding via gene sets
- Each state i receives ``markers_per_state`` markers in deterministic gene order.
- ``marker_overlap`` can be a scalar integer, meaning every pair of states connected by a non-zero static transition rate shares that many marker genes, or a dict mapping exact state groups to shared-marker counts, for example ``{(0, 2): 5, (1, 2, 3): 4}``.
- ``marker_reuse_cap`` is the maximum number of states allowed to share one marker gene. Feasibility is checked against ``markers_per_state`` and ``num_genes``; invalid overlap requests raise a clear error.

## Transition Graph Configuration
- Provide explicit static ``transition_rates`` (shape n-by-n, zero diagonal) for arbitrary directed rates.
- Example:
  ```python
  custom = np.full((n, n), 0.05, dtype=float)
  np.fill_diagonal(custom, 0.0)
  params = SimulationParameters(..., transition_rates=custom)
  ```
- For time- or state-dependent rates, pass a callable that receives a ``SimulationState``:
  ```python
  import numpy as np
  from markovmodus import SimulationParameters, SimulationState

  def rates(state: SimulationState) -> np.ndarray:
      custom = np.zeros((3, 3), dtype=float)
      custom[0, 1] = 0.02 if state.time < 10.0 else 0.2

      counts = np.bincount(state.cell_states, minlength=3)
      custom[1, 2] = 0.01 + 0.001 * counts[1]
      return custom

  params = SimulationParameters(..., num_states=3, transition_rates=rates)
  ```
- The latent-state process uses a fixed-``dt`` discretized CTMC approximation rather than exact Gillespie simulation. Dynamic rates are evaluated at the start of each time step and held constant for that step.
- ``SimulationState`` exposes read-only ``time``, ``cell_states``, ``unspliced``, ``spliced``, ``birth_parent``, ``birth_time``, ``generation`` and ``last_division_time`` arrays to dynamic callbacks.

## Proliferation
- Pass ``proliferation_model`` to let cells divide during the simulation. When this is enabled, ``num_cells`` is the initial population size and the final output may contain more rows.
- The callable receives the same ``SimulationState`` view as dynamic transition rates and returns ``(division_rates, daughter_state_probs)``:
  ```python
  def proliferation(state: SimulationState):
      # Per-state division rates and daughter-state probabilities.
      division_rates = np.array([0.08, 0.01, 0.0])
      daughter_state_probs = np.array(
          [
              [0.7, 0.3, 0.0],
              [0.0, 0.8, 0.2],
              [0.0, 0.0, 1.0],
          ]
      )
      return division_rates, daughter_state_probs

  params = SimulationParameters(..., num_states=3, proliferation_model=proliferation)
  ```
- ``division_rates`` may be state-level with shape ``(num_states,)`` or per-cell with shape ``(current_num_cells,)``. ``daughter_state_probs`` should have matching rows and one column per latent state. Probability rows are normalized internally, and rows with positive division rates must have positive probability mass.
- A dividing cell is replaced by one daughter in the existing row and appends a second daughter row. Both daughters copy the post-update unspliced/spliced counts and independently sample latent states from the returned daughter-state probabilities.
- Dividing cells skip ordinary latent transitions in the same time step. Non-dividing cells still follow the usual latent transition dynamics.
- Outputs include ``birth_parent`` and ``birth_time`` row-creation columns, plus ``generation`` and ``last_division_time`` division-history columns.

## Getting Started

- Install from PyPI:
  ```bash
  pip install markovmodus
  ```
- Or, after cloning this repository, install locally:
  ```bash
  pip install .
  ```
- Define your transition model, simulation settings, and run the generator:
  ```python
  import numpy as np
  from markovmodus import SimulationParameters, simulate_dataset

  transition_rates = np.full((5, 5), 0.08, dtype=float)
  np.fill_diagonal(transition_rates, 0.0)

  params = SimulationParameters(
      num_states=5,
      transition_rates=transition_rates,

      num_genes=300,
      markers_per_state=60,
      marker_overlap=10,

      num_cells=2000,
      t_final=30.0,
      dt=1.0,
      rng_seed=42,
  )

  adata = simulate_dataset(params)  # AnnData with spliced counts in X and U/S layers
  ```
- Produce a pandas DataFrame (and optionally persist to CSV):
  ```python
  df = simulate_dataset(params, output="dataframe", save_path="counts.csv")
  ```
- Write an AnnData file for Scanpy workflows:
  ```python
  simulate_dataset(params, save_path="synthetic_counts.h5ad", file_format="h5ad")
  ```
- Request both views when integrating with pipelines:
  ```python
  adata, df = simulate_dataset(params, output="both")
  ```

## Documentation
Start with the [Introduction](https://subthaumic.github.io/markovmodus/introduction.html) for a primer on the biological motivation and simulator design, then dive into the [usage guide](https://subthaumic.github.io/markovmodus/usage.html) and [API reference](https://subthaumic.github.io/markovmodus/api.html).

## Example notebooks
Interactive walkthroughs live in [`notebooks/`](notebooks/). Open them locally in Jupyter or your favourite notebook environment to explore model configuration and downstream analysis patterns.
Start with `quickstart.ipynb` for a compact walkthrough of the core API and output formats.
Then use `dynamic_transition_rates_example.ipynb` for time- and population-dependent transition rates, `proliferation_example.ipynb` for density-limited division and clone growth, and `qad_example.ipynb` for a refined QAD differentiation example with M2-restricted division and D-dependent feedback.


## License

MIT licensed. See `CITATION.cff` for citation details.

## Project resources

- [Changelog](CHANGELOG.md)
- [Documentation](https://subthaumic.github.io/markovmodus/)
- [Contributing guide](CONTRIBUTING.md)
