# markovmodus
[![CI](https://github.com/subthaumic/markovmodus/actions/workflows/ci.yml/badge.svg)](https://github.com/subthaumic/markovmodus/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/markovmodus.svg)](https://pypi.org/project/markovmodus/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Markov-modulated splicing simulator for single-cell U/S counts.**

Generate snapshot datasets where a **hidden state graph** (the support of a continuous-time generator) **modulates** unspliced->spliced RNA dynamics.


## Why this exists

Single-cell RNA sequencing captures each cell only once, so trajectory and velocity methods must infer temporal structure from static snapshots.
Without datasets where the true lineage graph is known, it is hard to validate the assumptions those methods make.

`markovmodus` fills that gap by generating synthetic unspliced/spliced counts with an explicit hidden-state lineage.
Cells hop between phenotypic states according to a continuous-time Markov process, and each state drives its own transcriptional kinetics.

For biologists, think of the hidden states as cellular programs—progenitors, intermediates, terminal fates—with transition rates describing how readily a cell exits one program and commits to another.
Within each program, genes produce pre-mRNA that is spliced and degraded, yielding both nascent and mature counts like those used in RNA velocity analyses.
Because the simulator records the exact state graph, you can stress-test algorithms that aim to recover branching, cyclic, or linear progressions from single snapshots.


## Model

### Latent dynamics (state process)
- States indexed z = 1, ..., n with an **adjacency mask** M (an n-by-n binary matrix) describing the undirected support.
- Generator Q on this support: Q[i, j] > 0 iff M[i, j] = 1, and each row sums to zero via Q[i, i] = -sum_{j != i} Q[i, j].
- **Directionality** arises from asymmetric off-diagonal rates (Q[i, j] != Q[j, i]).

### Emissions (per gene, in state z)
- Linear splicing dynamics with state-specific transcription targets (via the steady-state profile) and global beta (splicing) / gamma (decay):
  - dU/dt = alpha_z - beta * U
  - dS/dt = beta * U - gamma * S
- Snapshots at time t* can be perturbed by negative-binomial noise (enable by setting ``SimulationParameters.dispersion``) so counts remain discrete and overdispersed.

### Topology encoding via gene sets
- Each state i receives ``markers_per_state`` markers. Genes are either unique to one state (reuse cap = 1) or shared by exactly two states (reuse cap = 2).
- When sharing is enabled a balanced random sampler selects how many genes each pair of states shares so overlaps stay comparable while never exceeding two states per gene.
- This yields **overlap-induced continuity** in gene space without introducing higher-order simplices.

## Transition Graph Configuration
- Default behaviour uses a **fully connected graph** with a uniform jump rate set via ``SimulationParameters.default_transition_rate`` (falls back to `0.05` if omitted).
- Provide an explicit ``transition_matrix`` (shape n-by-n, zero diagonal) for arbitrary directed rates; the simulator samples next states proportional to the row's off-diagonal rates.
- Example:
  ```python
  custom = np.full((n, n), 0.05, dtype=float)
  np.fill_diagonal(custom, 0.0)
  params = SimulationParameters(..., transition_matrix=custom)
  ```

## Getting Started

- Install from PyPI:
  ```bash
  pip install markovmodus
  ```
- Or, after cloning this repository, install locally:
  ```bash
  pip install .
  ```
- Define your simulation settings and run the generator:
  ```python
  from markovmodus import SimulationParameters, simulate_dataset

  params = SimulationParameters(
      num_states=5,
      num_genes=300,
      num_cells=2000,
      t_final=30.0,
      dt=1.0,
      markers_per_state=120,
      default_transition_rate=0.08,
      rng_seed=42,
  )

  adata = simulate_dataset(params)  # AnnData with spliced/unspliced layers
  ```
- Produce a pandas DataFrame (and optionally persist to CSV):
  ```python
  df = simulate_dataset(params, output="dataframe", save_path="counts.csv")
  ```
- Write an AnnData file for Scanpy workflows:
  ```python
  simulate_dataset(params, save_path="snapshot.h5ad", file_format="h5ad")
  ```
- Request both views when integrating with pipelines:
  ```python
  adata, df = simulate_dataset(params, output="both")
  ```

## Documentation
Start with the [Introduction](https://subthaumic.github.io/markovmodus/introduction.html) for a primer on the biological motivation and simulator design, then dive into the [usage guide](https://subthaumic.github.io/markovmodus/usage.html) and [API reference](https://subthaumic.github.io/markovmodus/api.html).

## Example notebooks
Interactive walkthroughs live in [`notebooks/`](notebooks/). Open them locally in Jupyter or your favourite notebook environment to explore model configuration and downstream analysis patterns.


## License

MIT licensed. See `CITATION.cff` for citation details.

## Project resources

- [Changelog](CHANGELOG.md)
- [Documentation](https://subthaumic.github.io/markovmodus/)
- [Contributing guide](CONTRIBUTING.md)
