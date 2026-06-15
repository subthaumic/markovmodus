from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import inspect
from typing import Any, Callable, Dict, Optional, Sequence, TypeAlias

import numpy as np

TransitionRateSpec: TypeAlias = np.ndarray | Callable[[Any], np.ndarray]
ProliferationModelSpec: TypeAlias = Callable[[Any], tuple[np.ndarray, np.ndarray]]
MarkerOverlapSpec: TypeAlias = int | Mapping[tuple[int, ...], int]
_INTERNAL_DEFAULT_TRANSITION_RATE = 0.1


class _SimulationParametersMeta(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if "transition_matrix" in kwargs:
            raise TypeError(
                "SimulationParameters no longer accepts transition_matrix. "
                "Use transition_rates instead; it accepts static rate matrices and "
                "dynamic transition-rate callbacks."
            )
        return super().__call__(*args, **kwargs)


def _normalise_probabilities(probs: Optional[Sequence[float]], size: int) -> np.ndarray:
    """Normalise a probability vector or return a uniform distribution."""
    if probs is None:
        return np.full(size, 1.0 / size, dtype=float)

    array = np.asarray(probs, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"initial_state_probs must have shape ({size},)")

    total = array.sum()
    if total <= 0:
        raise ValueError("initial_state_probs must sum to a positive value")

    return array / total


def _ensure_square_matrix(
    matrix: np.ndarray, size: int, *, name: str = "transition_rates"
) -> np.ndarray:
    """Validate that a matrix is square with the requested size."""
    if matrix.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    return matrix


@dataclass(slots=True, kw_only=True)
class SimulationParameters(metaclass=_SimulationParametersMeta):
    """
    Container for configuring a markovmodus simulation.

    Attributes
    ----------
    num_states:
        Number of latent states (nodes) in the hidden Markov graph.
    transition_rates:
        Continuous-time rate matrix or callable returning transition rates for the current
        simulation state. Static matrices must have strictly non-negative off-diagonals and a zero
        diagonal. Callable outputs may be a global matrix with shape ``(num_states, num_states)``
        or a per-cell tensor with shape ``(current_num_cells, num_states, num_states)``. When
        omitted, a dense complete graph with a fixed internal fallback rate is used.
    proliferation_model:
        Optional callable returning division rates and daughter-state probabilities for the current
        simulation state. When a cell divides, its existing row becomes one daughter and a second
        daughter row is appended, so ``num_cells`` is the initial population size. Dividing cells
        skip ordinary latent transitions in that same time step.
    num_genes:
        Number of genes (each contributes an unspliced/spliced pair of counts).
    markers_per_state:
        Number of marker genes to emphasise in each state.
    marker_overlap:
        Either a scalar number of marker genes shared between every pair of states connected by a
        static non-zero transition rate, or a mapping from exact state groups to the number of
        marker genes shared by those states.
    marker_reuse_cap:
        Upper bound on how many distinct states may share the same gene marker.
    baseline_expression:
        Target steady-state spliced count for non-marker genes.
    marker_expression:
        Target steady-state spliced count for marker genes.
    state_expression:
        Optional state-by-gene steady-state spliced-count matrix overriding the automatically
        generated markers.
    splicing_rate:
        Rate at which unspliced RNA converts to spliced RNA.
    decay_rate:
        Decay rate for spliced RNA.
    dispersion:
        Optional negative-binomial dispersion parameter for adding measurement noise.
    gene_ids:
        Optional gene name list. When omitted defaults to ``gene_{i}``.
    initial_state_probs:
        Optional categorical distribution over initial states. Defaults to uniform.
    num_cells:
        Initial number of cells to simulate. The final row count may be larger when
        ``proliferation_model`` is configured.
    t_final:
        Duration of the simulation window.
    dt:
        Fixed time step for stochastic U/S, transition and division updates.
    rng_seed:
        Optional integer seed for reproducible simulations.
    """

    # Latent and population dynamics.
    num_states: int
    transition_rates: Optional[TransitionRateSpec] = None
    proliferation_model: Optional[ProliferationModelSpec] = None

    # State-specific spliced-count targets and U/S dynamics.
    num_genes: int
    markers_per_state: int = 150
    marker_overlap: MarkerOverlapSpec = 0
    marker_reuse_cap: int = 2
    baseline_expression: float = 2.0
    marker_expression: float = 10.0
    state_expression: Optional[np.ndarray] = None
    splicing_rate: float = 0.3
    decay_rate: float = 0.3
    dispersion: Optional[float] = None
    gene_ids: Optional[Sequence[str]] = None

    # Initial condition.
    initial_state_probs: Optional[Sequence[float]] = None

    # Simulation size, time, and reproducibility.
    num_cells: int
    t_final: float
    dt: float = 1.0
    rng_seed: Optional[int] = None

    _resolved_initial_probs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_states <= 0:
            raise ValueError("num_states must be positive")
        if self.num_genes <= 0:
            raise ValueError("num_genes must be positive")
        if self.num_cells <= 0:
            raise ValueError("num_cells must be positive")
        if self.t_final <= 0:
            raise ValueError("t_final must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.markers_per_state <= 0:
            raise ValueError("markers_per_state must be positive")
        if self.marker_reuse_cap <= 0:
            raise ValueError("marker_reuse_cap must be positive")
        if self.splicing_rate <= 0:
            raise ValueError("splicing_rate must be positive")
        if self.decay_rate <= 0:
            raise ValueError("decay_rate must be positive")

        if self.proliferation_model is not None and not callable(self.proliferation_model):
            raise ValueError("proliferation_model must be callable when provided")

        if self.transition_rates is None:
            self.transition_rates = self._default_transition_rates()
        elif not callable(self.transition_rates):
            matrix = _ensure_square_matrix(
                np.asarray(self.transition_rates, dtype=float),
                self.num_states,
                name="transition_rates",
            )
            self.transition_rates = self._sanitise_transition_rates(matrix)

        if self.state_expression is not None:
            self.state_expression = self._validate_state_expression(
                np.asarray(self.state_expression, dtype=float)
            )
        else:
            transition_rates = (
                None if callable(self.transition_rates) else np.asarray(self.transition_rates)
            )
            self.state_expression = _sample_state_expression(
                np.random.default_rng(self.rng_seed),
                num_states=self.num_states,
                num_genes=self.num_genes,
                markers_per_state=self.markers_per_state,
                marker_overlap=self.marker_overlap,
                baseline_expression=self.baseline_expression,
                marker_expression=self.marker_expression,
                reuse_cap=self.marker_reuse_cap,
                transition_rates=transition_rates,
            )

        if self.gene_ids is not None:
            if len(self.gene_ids) != self.num_genes:
                raise ValueError("gene_ids must match num_genes")

        self._resolved_initial_probs = _normalise_probabilities(
            self.initial_state_probs, self.num_states
        )

    def _default_transition_rates(self) -> np.ndarray:
        """Build the internal dense fallback transition-rate matrix."""
        matrix = np.full((self.num_states, self.num_states), _INTERNAL_DEFAULT_TRANSITION_RATE)
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def _sanitise_transition_rates(self, matrix: np.ndarray) -> np.ndarray:
        """Clip negative rates and enforce a zero diagonal."""
        if not np.all(np.isfinite(matrix)):
            raise ValueError("transition_rates must contain only finite values")
        cleaned = np.clip(matrix, a_min=0.0, a_max=None)
        np.fill_diagonal(cleaned, 0.0)
        return cleaned

    def _validate_state_expression(self, expression: np.ndarray) -> np.ndarray:
        """Ensure custom state expression is compatible with the configuration."""
        if expression.shape != (self.num_states, self.num_genes):
            raise ValueError("state_expression must have shape (num_states, num_genes)")
        if np.any(expression < 0):
            raise ValueError("state_expression must be non-negative")
        return expression

    @property
    def gene_names(self) -> np.ndarray:
        """Return gene identifiers as a NumPy array."""
        if self.gene_ids is None:
            return np.array([f"gene_{i}" for i in range(self.num_genes)], dtype=object)
        return np.asarray(self.gene_ids, dtype=object)

    @property
    def initial_state_distribution(self) -> np.ndarray:
        """Return the cached initial state distribution."""
        return self._resolved_initial_probs

    def resolve_transition_rates(self) -> TransitionRateSpec:
        """Return the configured transition rates."""
        if self.transition_rates is not None:
            if callable(self.transition_rates):
                return self.transition_rates
            return self.transition_rates.copy()

        return self._default_transition_rates()

    def resolve_state_expression(self, rng: np.random.Generator) -> np.ndarray:
        """Return a state-by-gene steady-state expression matrix."""
        if self.state_expression is not None:
            return self.state_expression.copy()

        return _sample_state_expression(
            rng,
            num_states=self.num_states,
            num_genes=self.num_genes,
            markers_per_state=self.markers_per_state,
            marker_overlap=self.marker_overlap,
            baseline_expression=self.baseline_expression,
            marker_expression=self.marker_expression,
            reuse_cap=self.marker_reuse_cap,
            transition_rates=(
                None if callable(self.transition_rates) else np.asarray(self.transition_rates)
            ),
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Convert parameters into a JSON-serialisable dictionary."""
        data: Dict[str, Any] = {}

        for item in fields(self):
            key = item.name
            if key == "_resolved_initial_probs":
                continue
            value = getattr(self, key)
            if key in {"transition_rates", "proliferation_model"} and callable(value):
                data[key] = "<callable>"
                continue
            if key == "marker_overlap" and isinstance(value, Mapping):
                data[key] = _marker_overlap_to_metadata(
                    value,
                    num_states=self.num_states,
                    num_genes=self.num_genes,
                    markers_per_state=self.markers_per_state,
                    reuse_cap=self.marker_reuse_cap,
                    transition_rates=(
                        None
                        if callable(self.transition_rates)
                        else np.asarray(self.transition_rates)
                    ),
                )
                continue
            data[key] = value

        # Convert arrays to lists for JSON serialisation
        for key, value in list(data.items()):
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], np.ndarray):
                data[key] = [np.asarray(item).tolist() for item in value]

        return data


SimulationParameters.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
    parameters=list(inspect.signature(SimulationParameters.__init__).parameters.values())[1:]
)


def _sample_state_expression(
    rng: np.random.Generator,
    *,
    num_states: int,
    num_genes: int,
    markers_per_state: int,
    marker_overlap: MarkerOverlapSpec,
    baseline_expression: float,
    marker_expression: float,
    reuse_cap: int,
    transition_rates: np.ndarray | None,
) -> np.ndarray:
    """
    Build a state expression matrix with controlled marker reuse.

    Genes are assigned in deterministic sequential order. Overlap can be specified as a
    scalar transition-edge count or as exact state-group counts.
    """
    del rng

    expression = np.full((num_states, num_genes), baseline_expression, dtype=float)

    if num_states == 0:
        return expression

    groups = _marker_overlap_groups(
        marker_overlap,
        num_states=num_states,
        num_genes=num_genes,
        markers_per_state=markers_per_state,
        reuse_cap=reuse_cap,
        transition_rates=transition_rates,
    )
    shared_slots = _shared_marker_slots(groups, num_states)

    cursor = 0
    for group, count in sorted(groups.items()):
        genes = np.arange(cursor, cursor + count)
        for state in group:
            expression[state, genes] = marker_expression
        cursor += count

    for state in range(num_states):
        unique_needed = markers_per_state - int(shared_slots[state])
        genes = np.arange(cursor, cursor + unique_needed)
        expression[state, genes] = marker_expression
        cursor += unique_needed

    return expression


def _marker_overlap_groups(
    marker_overlap: MarkerOverlapSpec,
    *,
    num_states: int,
    num_genes: int,
    markers_per_state: int,
    reuse_cap: int,
    transition_rates: np.ndarray | None,
) -> dict[tuple[int, ...], int]:
    """Validate marker-overlap configuration and return exact state-group counts."""
    groups: dict[tuple[int, ...], int]

    if isinstance(marker_overlap, bool):
        raise ValueError("marker_overlap must be an integer or a mapping")

    if isinstance(marker_overlap, (int, np.integer)):
        overlap_count = int(marker_overlap)
        if overlap_count < 0:
            raise ValueError("marker_overlap must be non-negative")
        groups = _transition_marker_overlap_groups(
            overlap_count,
            num_states=num_states,
            transition_rates=transition_rates,
        )
    elif isinstance(marker_overlap, Mapping):
        groups = _normalise_marker_overlap_mapping(marker_overlap, num_states=num_states)
    else:
        raise ValueError("marker_overlap must be an integer or a mapping")

    if reuse_cap <= 0:
        raise ValueError("marker_reuse_cap must be positive")
    if markers_per_state > num_genes:
        raise ValueError(
            "Not enough genes to assign distinct markers within each state: "
            f"{markers_per_state} markers requested per state but only {num_genes} genes available."
        )

    for group in groups:
        if len(group) > reuse_cap:
            raise ValueError(
                "marker_overlap group "
                f"{group} uses {len(group)} states but marker_reuse_cap is {reuse_cap}."
            )

    shared_slots = _shared_marker_slots(groups, num_states)
    for state, shared_count in enumerate(shared_slots):
        if shared_count > markers_per_state:
            raise ValueError(
                "marker_overlap assigns "
                f"{int(shared_count)} shared marker slots to state {state}, "
                f"but markers_per_state is {markers_per_state}."
            )

    required_genes = int(sum(groups.values()) + np.sum(markers_per_state - shared_slots))
    if required_genes > num_genes:
        raise ValueError(
            "Not enough genes to assign marker overlaps: "
            f"{required_genes} genes required but only {num_genes} available."
        )

    return groups


def _transition_marker_overlap_groups(
    overlap_count: int,
    *,
    num_states: int,
    transition_rates: np.ndarray | None,
) -> dict[tuple[int, ...], int]:
    if overlap_count == 0:
        return {}
    if transition_rates is None:
        raise ValueError(
            "integer marker_overlap requires static transition_rates. "
            "Use a marker_overlap mapping or provide state_expression for dynamic transition rates."
        )

    support = _transition_support(transition_rates, num_states=num_states)
    return {group: overlap_count for group in support}


def _transition_support(
    transition_rates: np.ndarray,
    *,
    num_states: int,
) -> list[tuple[int, int]]:
    rates = _ensure_square_matrix(
        np.asarray(transition_rates, dtype=float),
        num_states,
        name="transition_rates",
    )
    support = (rates > 0.0) | (rates.T > 0.0)
    np.fill_diagonal(support, False)
    return [
        (source, target)
        for source in range(num_states)
        for target in range(source + 1, num_states)
        if support[source, target]
    ]


def _normalise_marker_overlap_mapping(
    marker_overlap: Mapping[tuple[int, ...], int], *, num_states: int
) -> dict[tuple[int, ...], int]:
    groups: dict[tuple[int, ...], int] = {}

    for raw_group, raw_count in marker_overlap.items():
        if not isinstance(raw_group, tuple):
            raise ValueError("marker_overlap mapping keys must be tuples of state indices")
        if len(raw_group) < 2:
            raise ValueError("marker_overlap groups must contain at least two states")
        if any(
            isinstance(state, bool) or not isinstance(state, (int, np.integer))
            for state in raw_group
        ):
            raise ValueError("marker_overlap group entries must be integer state indices")

        group = tuple(sorted(int(state) for state in raw_group))
        if len(set(group)) != len(group):
            raise ValueError("marker_overlap groups cannot contain duplicate states")
        if group[0] < 0 or group[-1] >= num_states:
            raise ValueError(
                "marker_overlap group "
                f"{raw_group} contains state indices outside [0, {num_states})."
            )
        if group in groups:
            raise ValueError(f"marker_overlap specifies duplicate group {group}.")
        if isinstance(raw_count, bool) or not isinstance(raw_count, (int, np.integer)):
            raise ValueError("marker_overlap counts must be integers")

        count = int(raw_count)
        if count < 0:
            raise ValueError("marker_overlap counts must be non-negative")
        if count > 0:
            groups[group] = count

    return groups


def _shared_marker_slots(groups: Mapping[tuple[int, ...], int], num_states: int) -> np.ndarray:
    shared_slots = np.zeros(num_states, dtype=int)
    for group, count in groups.items():
        for state in group:
            shared_slots[state] += count
    return shared_slots


def _marker_overlap_to_metadata(
    marker_overlap: Mapping[tuple[int, ...], int],
    *,
    num_states: int,
    num_genes: int,
    markers_per_state: int,
    reuse_cap: int,
    transition_rates: np.ndarray | None,
) -> dict[str, int]:
    groups = _marker_overlap_groups(
        marker_overlap,
        num_states=num_states,
        num_genes=num_genes,
        markers_per_state=markers_per_state,
        reuse_cap=reuse_cap,
        transition_rates=transition_rates,
    )
    return {
        ",".join(str(state) for state in group): count for group, count in sorted(groups.items())
    }
