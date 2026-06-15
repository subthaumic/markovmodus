from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .config import SimulationParameters


@dataclass(slots=True)
class SimulationState:
    """
    Read-only start-of-step population view exposed to dynamic callbacks.

    ``transition_rates`` and ``proliferation_model`` callables receive this object before the
    current time step updates U/S counts, latent states or divisions. The arrays cover all cells
    that exist at that time; cells appended by division become visible to callbacks in the next
    time step.
    """

    time: float
    cell_states: np.ndarray
    unspliced: np.ndarray
    spliced: np.ndarray
    birth_parent: np.ndarray
    birth_time: np.ndarray
    generation: np.ndarray
    last_division_time: np.ndarray


@dataclass(slots=True)
class SimulationOutput:
    """
    Final simulation result.

    ``unspliced`` and ``spliced`` contain final raw count matrices. ``birth_parent`` and
    ``birth_time`` describe appended daughter-row creation, while ``generation`` and
    ``last_division_time`` describe division history for both reused and appended daughter rows.
    """

    parameters: SimulationParameters
    transition_rates: np.ndarray | None
    transition_rate_mode: str
    state_expression: np.ndarray
    states: np.ndarray
    unspliced: np.ndarray
    spliced: np.ndarray
    birth_parent: np.ndarray
    birth_time: np.ndarray
    generation: np.ndarray
    last_division_time: np.ndarray
    timepoint: float

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return unspliced and spliced matrices for convenience."""
        return self.unspliced, self.spliced


class Simulation:
    """Mutable engine implementing fixed-step latent, U/S and division dynamics."""

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
            if params.proliferation_model is not None and self.static_transition_rates.ndim == 3:
                raise ValueError(
                    "static per-cell transition_rates are not supported with proliferation_model"
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

        self.birth_parent = np.full(params.num_cells, -1, dtype=int)
        self.birth_time = np.zeros(params.num_cells, dtype=float)
        self.generation = np.zeros(params.num_cells, dtype=int)
        self.last_division_time = np.zeros(params.num_cells, dtype=float)

        self._output_unspliced: np.ndarray | None = None
        self._output_spliced: np.ndarray | None = None

    @property
    def state(self) -> SimulationState:
        """Return a read-only view of the current simulation state."""
        cell_states = self.cell_states.view()
        unspliced = self.expressions[:, 0::2].view()
        spliced = self.expressions[:, 1::2].view()
        birth_parent = self.birth_parent.view()
        birth_time = self.birth_time.view()
        generation = self.generation.view()
        last_division_time = self.last_division_time.view()

        cell_states.setflags(write=False)
        unspliced.setflags(write=False)
        spliced.setflags(write=False)
        birth_parent.setflags(write=False)
        birth_time.setflags(write=False)
        generation.setflags(write=False)
        last_division_time.setflags(write=False)

        return SimulationState(
            time=self.time,
            cell_states=cell_states,
            unspliced=unspliced,
            spliced=spliced,
            birth_parent=birth_parent,
            birth_time=birth_time,
            generation=generation,
            last_division_time=last_division_time,
        )

    def run(self) -> Simulation:
        """
        Advance the simulation to ``params.t_final`` and return ``self``.

        For every step, dynamic transition and proliferation callbacks are evaluated from the
        start-of-step :class:`SimulationState`. Existing cells update U/S counts, division events
        are sampled, non-dividing cells sample ordinary latent transitions and dividing cells are
        replaced by two daughters sampled from the proliferation fate probabilities.
        """
        while self.time < self.params.t_final:
            dt = min(self.params.dt, self.params.t_final - self.time)
            step_state = self.state
            step_cell_count = step_state.cell_states.size
            transition_rates = self._current_transition_rates(step_state)
            proliferation = self._current_proliferation(step_state)

            self._update_expressions(dt, step_cell_count)
            division_mask, daughter_state_probs = self._sample_divisions(
                dt, proliferation, step_cell_count
            )
            self._update_latent_states(
                dt,
                transition_rates,
                step_cell_count,
                transition_mask=~division_mask,
            )
            self._divide_cells(dt, division_mask, daughter_state_probs)
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
            birth_parent=self.birth_parent.copy(),
            birth_time=self.birth_time.copy(),
            generation=self.generation.copy(),
            last_division_time=self.last_division_time.copy(),
            timepoint=self.time,
        )

    def _current_transition_rates(self, state: SimulationState) -> np.ndarray:
        if self.static_transition_rates is not None:
            return self.static_transition_rates

        assert callable(self._transition_rate_spec)
        return _validate_transition_rates(
            self._transition_rate_spec(state),
            num_states=self.params.num_states,
            num_cells=state.cell_states.size,
        )

    def _current_proliferation(
        self, state: SimulationState
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if self.params.proliferation_model is None:
            return None

        return _validate_proliferation_model_output(
            self.params.proliferation_model(state),
            num_states=self.params.num_states,
            cell_states=state.cell_states,
        )

    def _update_expressions(self, dt: float, cell_count: int) -> None:
        for idx in range(cell_count):
            state = self.cell_states[idx]
            self.expressions[idx] = _update_expression(
                self.rng,
                current_expression=self.expressions[idx],
                steady_state=self.state_expression[state],
                dt=dt,
                decay_rate=self.params.decay_rate,
                splicing_rate=self.params.splicing_rate,
            )

    def _update_latent_states(
        self,
        dt: float,
        transition_rates: np.ndarray,
        cell_count: int,
        *,
        transition_mask: np.ndarray | None = None,
    ) -> None:
        per_cell_rates = transition_rates.ndim == 3

        for idx in range(cell_count):
            if transition_mask is not None and not transition_mask[idx]:
                continue

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

    def _sample_divisions(
        self,
        dt: float,
        proliferation: tuple[np.ndarray, np.ndarray] | None,
        cell_count: int,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        division_mask = np.zeros(cell_count, dtype=bool)
        if proliferation is None:
            return division_mask, None

        division_rates, daughter_state_probs = proliferation
        division_probs = 1.0 - np.exp(-division_rates * dt)
        division_mask = self.rng.random(cell_count) < np.clip(division_probs, 0.0, 1.0)
        return division_mask, daughter_state_probs

    def _divide_cells(
        self,
        dt: float,
        division_mask: np.ndarray,
        daughter_state_probs: np.ndarray | None,
    ) -> None:
        parent_indices = np.flatnonzero(division_mask)
        if parent_indices.size == 0:
            return

        assert daughter_state_probs is not None
        daughter_one_states = np.array(
            [
                self.rng.choice(self.params.num_states, p=daughter_state_probs[parent_idx])
                for parent_idx in parent_indices
            ],
            dtype=int,
        )
        daughter_two_states = np.array(
            [
                self.rng.choice(self.params.num_states, p=daughter_state_probs[parent_idx])
                for parent_idx in parent_indices
            ],
            dtype=int,
        )
        daughter_two_expressions = self.expressions[parent_indices].copy()
        daughter_generation = self.generation[parent_indices] + 1
        division_time = self.time + dt

        self.cell_states[parent_indices] = daughter_one_states
        self.generation[parent_indices] = daughter_generation
        self.last_division_time[parent_indices] = division_time

        self.cell_states = np.concatenate([self.cell_states, daughter_two_states])
        self.expressions = np.concatenate([self.expressions, daughter_two_expressions], axis=0)
        self.birth_parent = np.concatenate([self.birth_parent, parent_indices.astype(int)])
        self.birth_time = np.concatenate(
            [
                self.birth_time,
                np.full(parent_indices.size, division_time, dtype=float),
            ]
        )
        self.generation = np.concatenate([self.generation, daughter_generation])
        self.last_division_time = np.concatenate(
            [
                self.last_division_time,
                np.full(parent_indices.size, division_time, dtype=float),
            ]
        )

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
    Run a fixed-step Markov-modulated U/S simulation for a population of cells.

    Parameters
    ----------
    params:
        Fully configured :class:`SimulationParameters` instance.

    Returns
    -------
    SimulationOutput
        Final counts, latent states and lineage metadata for downstream conversion to tabular or
        AnnData formats.
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


def _validate_proliferation_model_output(
    model_output: tuple[np.ndarray, np.ndarray],
    *,
    num_states: int,
    cell_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate division rates and return per-cell rates/probabilities."""
    if not isinstance(model_output, tuple) or len(model_output) != 2:
        raise ValueError("proliferation_model must return (division_rates, daughter_state_probs)")

    raw_division_rates, raw_daughter_state_probs = model_output
    division_rates = np.asarray(raw_division_rates, dtype=float)
    daughter_state_probs = np.asarray(raw_daughter_state_probs, dtype=float)
    cell_count = cell_states.size

    if not np.all(np.isfinite(division_rates)):
        raise ValueError("proliferation_model division_rates must contain only finite values")
    if not np.all(np.isfinite(daughter_state_probs)):
        raise ValueError("proliferation_model daughter_state_probs must contain only finite values")
    if np.any(daughter_state_probs < 0.0):
        raise ValueError("proliferation_model daughter_state_probs must be non-negative")

    if division_rates.shape == (num_states,) and daughter_state_probs.shape == (
        num_states,
        num_states,
    ):
        cleaned_division_rates = np.clip(division_rates, a_min=0.0, a_max=None)
        _validate_active_probability_rows(
            cleaned_division_rates,
            daughter_state_probs,
            row_name="state",
        )
        normalized_probs = _normalize_probability_rows(daughter_state_probs)
        return cleaned_division_rates[cell_states], normalized_probs[cell_states]

    if division_rates.shape == (cell_count,) and daughter_state_probs.shape == (
        cell_count,
        num_states,
    ):
        cleaned_division_rates = np.clip(division_rates, a_min=0.0, a_max=None)
        _validate_active_probability_rows(
            cleaned_division_rates,
            daughter_state_probs,
            row_name="cell",
        )
        return cleaned_division_rates, _normalize_probability_rows(daughter_state_probs)

    raise ValueError(
        "proliferation_model must return division_rates with shape "
        f"({num_states},) or ({cell_count},), and daughter_state_probs with shape "
        f"({num_states}, {num_states}) or ({cell_count}, {num_states})"
    )


def _validate_active_probability_rows(
    division_rates: np.ndarray,
    probabilities: np.ndarray,
    *,
    row_name: str,
) -> None:
    row_sums = probabilities.sum(axis=1)
    active_rows = division_rates > 0.0
    if np.any(row_sums[active_rows] <= 0.0):
        raise ValueError(
            "proliferation_model daughter_state_probs rows must have positive mass "
            f"for every {row_name} with a positive division rate"
        )


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    normalized = probabilities.copy()
    row_sums = normalized.sum(axis=1)
    positive_rows = row_sums > 0.0
    normalized[positive_rows] = normalized[positive_rows] / row_sums[positive_rows, None]
    return normalized


def _initialise_expression(
    rng: np.random.Generator,
    *,
    steady_state: np.ndarray,
    decay_rate: float,
    splicing_rate: float,
) -> np.ndarray:
    """Sample initial U/S counts near the steady-state implied by the spliced target."""
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
    """Advance unspliced/spliced counts by one stochastic time step."""
    num_genes = steady_state.size
    current_unspliced = current_expression[0::2].astype(float)
    current_spliced = current_expression[1::2].astype(float)

    target_levels = np.asarray(steady_state, dtype=float)

    # Production rate alpha = gamma * S* makes target_levels the spliced steady state.
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
