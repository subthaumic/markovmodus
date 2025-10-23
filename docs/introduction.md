# Introduction

`markovmodus` is a simulator for generating synthetic single-cell RNA sequencing
datasets with paired unspliced and spliced counts. It provides a controlled
environment where latent cell-state transitions are governed by a
continuous-time Markov process and directly influence transcriptional dynamics.

## Why build another simulator?

- **Ground-truth topologies** — downstream methods can be benchmarked against
  the exact latent state graph, enabling rigorous evaluation of trajectory
  inference and dynamic modelling tools.
- **Flexible dynamics** — customise the number of states, transition rates and
  state-specific steady-state expression targets to match your experimental scenarios.
- **Accessible outputs** — obtain wide cell-level pandas DataFrames or AnnData objects that
  slot into established single-cell analysis workflows.

## How it works

1. Define the latent state graph, either via a custom transition matrix or by
   relying on the default fully connected graph.
2. Configure transcription and splicing parameters: set global splicing/decay rates,
   and tune per-state steady-state expression profiles (including marker reuse controls).
3. Simulate cells by sampling paths through the latent graph and recording a
   snapshot of unspliced/spliced counts, optionally adding negative-binomial noise via ``dispersion`` for realistic over-dispersion.

The rest of the documentation covers day-to-day usage (`usage.rst`), the full
API reference (`api.rst`) and the changelog (`changelog.md`).
