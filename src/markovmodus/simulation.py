from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple
import warnings

import numpy as np

from .config import SimulationParameters


@dataclass(slots=True)
class SimulationState:
    """Read-only view of the current population exposed to dynamic rates."""

    time: float
    cell_states: np.ndarray
    unspliced: np.ndarray
    spliced: np.ndarray


@dataclass(slots=True)
class SimulationOutput:
    """Container with the result of a simulation run."""

    parameters: SimulationParameters
    transition_rates: np.ndarray | None
    transition_rate_mode: str
    state_expression: np.ndarray
    states: np.ndarray
    unspliced: np.ndarray
    spliced: np.ndarray
    timepoint: float

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return unspliced and spliced matrices for convenience."""
        return self.unspliced, self.spliced

    @property
    def transition_matrix(self) -> np.ndarray:
        """Deprecated alias for static ``transition_rates``."""
        warnings.warn(
            "SimulationOutput.transition_matrix is deprecated; use transition_rates instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self.transition_rates is None:
            raise ValueError("Dynamic transition rates do not have a static transition_matrix")
        return self.transition_rates


class Simulation:
    """Mutable simulation engine for markovmodus populations."""

    def __init__(self, params: SimulationParameters) -> None:
        self.params = params
        self.rng = np.random.default_rng(params.rng_seed)
        self.time = 0.0

        self._transition_rate_spec = params.resolve_transition_rates()
        self.transition_rate_mode = "dynamic" if callable(self._transition_rate_spec) else "static"

        if callable(self._transition_rate_spec):
            self.static_transition_rates: np.ndarray | None = None
        else:
            self.static_transition_rates = _validate_transition_rates(
                self._transition_rate_spec,
                num_states=params.num_states,
                num_cells=params.num_cells,
            )

        self.state_expression = params.resolve_state_expression(self.rng)
        self.cell_states = self.rng.choice(
            params.num_states,
            size=params.num_cells,
            replace=True,
            p=params.initial_state_distribution,
        )

        self.expressions = np.zeros((params.num_cells, 2 * params.num_genes), dtype=int)
        for idx in range(params.num_cells):
            self.expressions[idx] = _initialise_expression(
                self.rng,
                steady_state=self.state_expression[self.cell_states[idx]],
                decay_rate=params.decay_rate,
                splicing_rate=params.splicing_rate,
            )

        self._output_unspliced: np.ndarray | None = None
        self._output_spliced: np.ndarray | None = None

    @property
    def state(self) -> SimulationState:
        """Return a read-only view of the current simulation state."""
        cell_states = self.cell_states.view()
        unspliced = self.expressions[:, 0::2].view()
        spliced = self.expressions[:, 1::2].view()

        cell_states.setflags(write=False)
        unspliced.setflags(write=False)
        spliced.setflags(write=False)

        return SimulationState(
            time=self.time,
            cell_states=cell_states,
            unspliced=unspliced,
            spliced=spliced,
        )

    def run(self) -> Simulation:
        """Advance the simulation to ``params.t_final`` and return ``self``."""
        while self.time < self.params.t_final:
            dt = min(self.params.dt, self.params.t_final - self.time)
            transition_rates = self._current_transition_rates()

            self._update_expressions(dt)
            self._update_latent_states(dt, transition_rates)
            self.time += dt

        self._finalise_output_arrays()
        return self

    def to_output(self) -> SimulationOutput:
        """Convert the current simulation state into a result container."""
        if self._output_unspliced is None or self._output_spliced is None:
            self._finalise_output_arrays()

        assert self._output_unspliced is not None
        assert self._output_spliced is not None

        transition_rates = (
            None if self.static_transition_rates is None else self.static_transition_rates.copy()
        )
        return SimulationOutput(
            parameters=self.params,
            transition_rates=transition_rates,
            transition_rate_mode=self.transition_rate_mode,
            state_expression=self.state_expression.copy(),
            states=self.cell_states.copy(),
            unspliced=self._output_unspliced.copy(),
            spliced=self._output_spliced.copy(),
            timepoint=self.time,
        )

    def _current_transition_rates(self) -> np.ndarray:
        if self.static_transition_rates is not None:
            return self.static_transition_rates

        assert callable(self._transition_rate_spec)
        return _validate_transition_rates(
            self._transition_rate_spec(self.state),
            num_states=self.params.num_states,
            num_cells=self.params.num_cells,
        )

    def _update_expressions(self, dt: float) -> None:
        for idx in range(self.params.num_cells):
            state = self.cell_states[idx]
            self.expressions[idx] = _update_expression(
                self.rng,
                current_expression=self.expressions[idx],
                steady_state=self.state_expression[state],
                dt=dt,
                decay_rate=self.params.decay_rate,
                splicing_rate=self.params.splicing_rate,
            )

    def _update_latent_states(self, dt: float, transition_rates: np.ndarray) -> None:
        per_cell_rates = transition_rates.ndim == 3

        for idx in range(self.params.num_cells):
            state = self.cell_states[idx]
            outgoing_rates = (
                transition_rates[idx, state] if per_cell_rates else transition_rates[state]
            )
            total_rate = outgoing_rates.sum()

            if total_rate <= 0.0:
                continue

            transition_prob = 1.0 - np.exp(-total_rate * dt)
            if self.rng.random() < transition_prob:
                probabilities = outgoing_rates / total_rate
                self.cell_states[idx] = self.rng.choice(self.params.num_states, p=probabilities)

    def _finalise_output_arrays(self) -> None:
        unspliced = self.expressions[:, 0::2].copy()
        spliced = self.expressions[:, 1::2].copy()

        if self.params.dispersion is not None:
            theta = float(self.params.dispersion)
            if theta <= 0:
                raise ValueError("dispersion must be positive when provided")
            unspliced = _apply_negative_binomial_noise(self.rng, unspliced, theta)
            spliced = _apply_negative_binomial_noise(self.rng, spliced, theta)

        self._output_unspliced = unspliced
        self._output_spliced = spliced


def simulate_population(params: SimulationParameters) -> SimulationOutput:
    """
    Run a discretized continuous-time Markov simulation for a population of cells.

    Parameters
    ----------
    params:
        Fully configured :class:`SimulationParameters` instance.

    Returns
    -------
    SimulationOutput
        Final counts and metadata for downstream conversion to tabular or AnnData formats.
    """
    return Simulation(params).run().to_output()


def _validate_transition_rates(
    rates: np.ndarray,
    *,
    num_states: int,
    num_cells: int,
) -> np.ndarray:
    """Validate and sanitise static, global dynamic, or per-cell transition rates."""
    array = np.asarray(rates, dtype=float)
    valid_shapes = ((num_states, num_states), (num_cells, num_states, num_states))
    if array.shape not in valid_shapes:
        raise ValueError(
            "transition_rates must have shape "
            f"({num_states}, {num_states}) or ({num_cells}, {num_states}, {num_states})"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("transition_rates must contain only finite values")

    cleaned = np.clip(array, a_min=0.0, a_max=None)
    diagonal = np.arange(num_states)
    if cleaned.ndim == 2:
        cleaned[diagonal, diagonal] = 0.0
    else:
        cleaned[:, diagonal, diagonal] = 0.0
    return cleaned


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
