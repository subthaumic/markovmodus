# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [Unreleased]

- No changes yet.

## [0.3.0] - 2026-06-15

### Added
- Callable ``proliferation_model`` support for true cell division. A dividing cell reuses the
  existing row for daughter 1, appends daughter 2, copies post-update U/S counts to both daughters
  and samples both daughter latent states from the configured fate distribution.
- Division metadata in simulation outputs: ``birth_parent``, ``birth_time``, ``generation`` and
  ``last_division_time``.
- Marker-overlap mappings for explicit shared-marker configurations between state groups.
- AnnData output now stores raw spliced counts in ``X`` and keeps paired raw ``unspliced`` and
  ``spliced`` layers.
- Example notebooks for quickstart, dynamic transition rates, proliferation and QAD differentiation.

### Changed
- ``transition_rates`` is now the public transition-graph API for static matrices and dynamic
  callbacks.
- Marker genes are assigned deterministically in gene-index order, with scalar ``marker_overlap``
  applied to state pairs connected by static non-zero transition rates.

### Removed
- ``transition_matrix`` as a public ``SimulationParameters`` argument. Passing it now raises a
  targeted error pointing users to ``transition_rates``.
- ``default_transition_rate`` from the public API. Omitting ``transition_rates`` still uses an
  internal dense fallback rate for convenience, but users do not configure that rate directly.

## [0.2.0] - 2026-04-30

### Added
- Dynamic ``transition_rates`` support via ``SimulationState`` callbacks.
- ``Simulation`` engine class for stepwise simulation state management.

### Deprecated
- ``transition_matrix`` in favour of ``transition_rates``.


## [0.1.0] - 2025-10-23

### Added
- Markov-modulated simulator with configurable state graphs and expression profiles.
- Continuous integration for multi-version pytest and distribution builds.
- Example notebooks illustrating dense and sparse transition topologies.
- Documentation site with usage guide and API reference.


[unreleased]: https://github.com/subthaumic/markovmodus/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/subthaumic/markovmodus/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/subthaumic/markovmodus/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/subthaumic/markovmodus/releases/tag/v0.1.0
