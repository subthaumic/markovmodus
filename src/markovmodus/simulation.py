from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import SimulationParameters


@dataclass(slots=True)
class SimulationOutput:
    """Container with the result of a simulation run."""

    parameters: SimulationParameters
    transition_matrix: np.ndarray
    state_expression: np.ndarray
    states: np.ndarray
    unspliced: np.ndarray
    spliced: np.ndarray
    timepoint: float

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return unspliced and spliced matrices for convenience."""
        return self.unspliced, self.spliced


def simulate_population(params: SimulationParameters) -> SimulationOutput:
    """
    Run a continuous-time Markov simulation for a population of cells.

    Parameters
    ----------
    params:
        Fully configured :class:`SimulationParameters` instance.

    Returns
    -------
    SimulationOutput
        Final counts and metadata for downstream conversion to tabular or AnnData formats.
    """
    rng = np.random.default_rng(params.rng_seed)

    transition_matrix = params.resolve_transition_matrix()
    state_expression = params.resolve_state_expression(rng)

    num_cells = params.num_cells
    num_genes = params.num_genes

    # Initial states sampled from configured distribution
    current_states = rng.choice(
        params.num_states, size=num_cells, replace=True, p=params.initial_state_distribution
    )

    # Initialise expression near state-specific steady states
    expressions = np.zeros((num_cells, 2 * num_genes), dtype=int)
    for idx in range(num_cells):
        expressions[idx] = _initialise_expression(
            rng,
            steady_state=state_expression[current_states[idx]],
            decay_rate=params.decay_rate,
            splicing_rate=params.splicing_rate,
        )

    t = 0.0
    while t < params.t_final:
        dt = min(params.dt, params.t_final - t)
        t += dt

        # Update gene expression for each cell
        for idx in range(num_cells):
            state = current_states[idx]
            expressions[idx] = _update_expression(
                rng,
                current_expression=expressions[idx],
                steady_state=state_expression[state],
                dt=dt,
                decay_rate=params.decay_rate,
                splicing_rate=params.splicing_rate,
            )

        # Handle latent state transitions
        for idx in range(num_cells):
            state = current_states[idx]
            outgoing_rates = transition_matrix[state]
            total_rate = outgoing_rates.sum()

            if total_rate <= 0.0:
                continue

            transition_prob = 1.0 - np.exp(-total_rate * dt)
            if rng.random() < transition_prob:
                probabilities = outgoing_rates / total_rate
                current_states[idx] = rng.choice(params.num_states, p=probabilities)

    unspliced = expressions[:, 0::2]
    spliced = expressions[:, 1::2]

    if params.dispersion is not None:
        theta = float(params.dispersion)
        if theta <= 0:
            raise ValueError("dispersion must be positive when provided")
        unspliced = _apply_negative_binomial_noise(rng, unspliced, theta)
        spliced = _apply_negative_binomial_noise(rng, spliced, theta)

    return SimulationOutput(
        parameters=params,
        transition_matrix=transition_matrix,
        state_expression=state_expression,
        states=current_states,
        unspliced=unspliced,
        spliced=spliced,
        timepoint=params.t_final,
    )


def _initialise_expression(
    rng: np.random.Generator,
    *,
    steady_state: np.ndarray,
    decay_rate: float,
    splicing_rate: float,
) -> np.ndarray:
    """Sample an initial state near the steady-state counts."""
    steady_state = np.asarray(steady_state, dtype=float)
    steady_state_unspliced = (decay_rate * steady_state) / splicing_rate
    steady_state_spliced = steady_state

    unspliced = rng.poisson(lam=np.maximum(steady_state_unspliced, 0.0))
    spliced = rng.poisson(lam=np.maximum(steady_state_spliced, 0.0))

    expression = np.zeros(2 * steady_state.size, dtype=int)
    expression[0::2] = unspliced
    expression[1::2] = spliced
    return expression


def _update_expression(
    rng: np.random.Generator,
    *,
    current_expression: np.ndarray,
    steady_state: np.ndarray,
    dt: float,
    decay_rate: float,
    splicing_rate: float,
) -> np.ndarray:
    """Advance the unspliced/spliced counts by one time-step."""
    num_genes = steady_state.size
    current_unspliced = current_expression[0::2].astype(float)
    current_spliced = current_expression[1::2].astype(float)

    target_levels = np.asarray(steady_state, dtype=float)

    # Production to reach steady-state around target_levels
    production_rates = target_levels * decay_rate
    new_unspliced = current_unspliced + rng.poisson(lam=production_rates * dt)

    # Splicing events
    splicing_means = np.maximum(current_unspliced, 0.0) * splicing_rate * dt
    splicing_events = rng.poisson(lam=splicing_means)
    splicing_events = np.minimum(splicing_events, new_unspliced).astype(float)

    # Degradation of spliced RNA
    degradation_prob = 1.0 - np.exp(-decay_rate * dt)
    degradation_events = rng.binomial(
        n=np.maximum(current_spliced, 0.0).astype(int),
        p=np.clip(degradation_prob, 0.0, 1.0),
    )

    updated_unspliced = new_unspliced - splicing_events
    updated_spliced = current_spliced - degradation_events + splicing_events

    updated_unspliced = np.maximum(updated_unspliced, 0).astype(int)
    updated_spliced = np.maximum(updated_spliced, 0).astype(int)

    output = np.zeros(2 * num_genes, dtype=int)
    output[0::2] = updated_unspliced
    output[1::2] = updated_spliced
    return output


def _apply_negative_binomial_noise(
    rng: np.random.Generator,
    counts: np.ndarray,
    theta: float,
) -> np.ndarray:
    """Apply per-entry negative-binomial noise around the observed counts."""
    counts = counts.astype(float)
    probabilities = theta / (theta + counts)
    probabilities = np.clip(probabilities, 0.0, 1.0)

    noisy = rng.negative_binomial(theta, probabilities, size=counts.shape)
    return noisy.astype(int)
