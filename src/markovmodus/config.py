from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np


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


def _ensure_square_matrix(matrix: np.ndarray, size: int) -> np.ndarray:
    """Validate that a matrix is square with the requested size."""
    if matrix.shape != (size, size):
        raise ValueError(f"transition_matrix must have shape ({size}, {size})")
    return matrix


@dataclass(slots=True)
class SimulationParameters:
    """
    Container for configuring a markovmodus simulation.

    Attributes
    ----------
    num_states:
        Number of latent states (nodes) in the hidden Markov graph.
    num_genes:
        Number of genes (each contributes an unspliced/spliced pair of counts).
    num_cells:
        Number of independent cells to simulate.
    t_final:
        Duration of the simulation window.
    dt:
        Integration time-step for the Euler-Maruyama style updates.
    transition_matrix:
        Optional continuous-time rate matrix (strictly non-negative off-diagonals, zero diagonal).
        When omitted a dense complete graph is constructed with ``default_transition_rate``.
    default_transition_rate:
        Rate used for all off-diagonal entries of the default complete graph.
    baseline_expression:
        Target steady-state count for non-marker genes.
    marker_expression:
        Target steady-state count for marker genes.
    markers_per_state:
        Number of marker genes to emphasise in each state.
    marker_reuse_cap:
        Upper bound on how many distinct states may share the same gene marker (1 for unique-only, 2 for pairwise sharing).
    splicing_rate:
        Rate at which unspliced RNA converts to spliced RNA.
    decay_rate:
        Decay rate for spliced RNA.
    dispersion:
        Optional negative-binomial dispersion parameter for adding measurement noise.
    initial_state_probs:
        Optional categorical distribution over initial states. Defaults to uniform.
    gene_ids:
        Optional gene name list. When omitted defaults to ``gene_{i}``.
    state_expression:
        Optional state-by-gene steady-state matrix overriding the automatically generated markers.
    rng_seed:
        Optional integer seed for reproducible simulations.
    """

    num_states: int
    num_genes: int
    num_cells: int
    t_final: float
    dt: float = 1.0
    transition_matrix: Optional[np.ndarray] = None
    default_transition_rate: float = 0.05
    baseline_expression: float = 2.0
    marker_expression: float = 10.0
    markers_per_state: int = 150
    marker_reuse_cap: int = 2
    splicing_rate: float = 0.3
    decay_rate: float = 0.3
    dispersion: Optional[float] = None
    initial_state_probs: Optional[Sequence[float]] = None
    gene_ids: Optional[Sequence[str]] = None
    state_expression: Optional[np.ndarray] = None
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
        if self.default_transition_rate < 0:
            raise ValueError("default_transition_rate must be non-negative")
        if self.splicing_rate <= 0:
            raise ValueError("splicing_rate must be positive")
        if self.decay_rate <= 0:
            raise ValueError("decay_rate must be positive")

        if self.transition_matrix is not None:
            matrix = _ensure_square_matrix(
                np.asarray(self.transition_matrix, dtype=float), self.num_states
            )
            self.transition_matrix = self._sanitise_transition_matrix(matrix)

        if self.state_expression is not None:
            self.state_expression = self._validate_state_expression(
                np.asarray(self.state_expression, dtype=float)
            )

        if self.gene_ids is not None:
            if len(self.gene_ids) != self.num_genes:
                raise ValueError("gene_ids must match num_genes")

        self._resolved_initial_probs = _normalise_probabilities(
            self.initial_state_probs, self.num_states
        )

    def _sanitise_transition_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Clip negative rates and enforce a zero diagonal."""
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

    def resolve_transition_matrix(self) -> np.ndarray:
        """Return a dense transition matrix consistent with the configuration."""
        if self.transition_matrix is not None:
            return self.transition_matrix.copy()

        matrix = np.full(
            (self.num_states, self.num_states), self.default_transition_rate, dtype=float
        )
        np.fill_diagonal(matrix, 0.0)
        return matrix

    def resolve_state_expression(self, rng: np.random.Generator) -> np.ndarray:
        """Return a state-by-gene steady-state expression matrix."""
        if self.state_expression is not None:
            return self.state_expression.copy()

        return _sample_state_expression(
            rng,
            num_states=self.num_states,
            num_genes=self.num_genes,
            markers_per_state=self.markers_per_state,
            baseline_expression=self.baseline_expression,
            marker_expression=self.marker_expression,
            reuse_cap=self.marker_reuse_cap,
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Convert parameters into a JSON-serialisable dictionary."""
        data = asdict(self)
        # Remove cached field
        data.pop("_resolved_initial_probs", None)

        # Convert arrays to lists for JSON serialisation
        for key, value in list(data.items()):
            if isinstance(value, np.ndarray):
                data[key] = value.tolist()
            elif isinstance(value, (list, tuple)) and value and isinstance(value[0], np.ndarray):
                data[key] = [np.asarray(item).tolist() for item in value]

        return data


def _sample_state_expression(
    rng: np.random.Generator,
    *,
    num_states: int,
    num_genes: int,
    markers_per_state: int,
    baseline_expression: float,
    marker_expression: float,
    reuse_cap: int,
) -> np.ndarray:
    """
    Build a state expression matrix with controlled marker reuse.

    The sampler currently supports ``reuse_cap`` values of 1 (unique markers only) or 2
    (pairwise sharing only). For ``reuse_cap`` equal to 2 the number of shared markers is
    drawn randomly while ensuring every state retains a similar share of overlaps and
    no gene is used by more than two states.
    """
    expression = np.full((num_states, num_genes), baseline_expression, dtype=float)

    if num_states == 0:
        return expression

    required_slots = num_states * markers_per_state
    if reuse_cap == 1 or num_states == 1:
        if required_slots > num_genes:
            raise ValueError(
                "Not enough genes to assign unique markers: "
                f"{required_slots} slots requested but only {num_genes} genes available."
            )

        gene_order = rng.permutation(num_genes)
        cursor = 0
        for state in range(num_states):
            selected = gene_order[cursor : cursor + markers_per_state]
            if selected.size < markers_per_state:
                raise ValueError("Ran out of genes while assigning unique markers.")
            expression[state, selected] = marker_expression
            cursor += markers_per_state
        return expression

    if reuse_cap != 2:
        raise ValueError("marker_reuse_cap must be 1 or 2 when sampling state expression.")

    max_assignable_with_cap = 2 * num_genes
    if required_slots > max_assignable_with_cap:
        raise ValueError(
            "Configuration infeasible: num_states * markers_per_state exceeds the "
            "capacity implied by marker_reuse_cap=2."
        )

    def sample_shared_gene_count(min_shared: int, max_shared: int) -> int:
        if max_shared <= 0:
            return 0
        if min_shared >= max_shared:
            return min_shared
        span = max_shared - min_shared
        mean = min_shared + span / 2.0
        std = max(1.0, span / 4.0)
        for _ in range(12):
            draw = int(round(rng.normal(mean, std)))
            if min_shared <= draw <= max_shared:
                return draw
        return int(rng.integers(min_shared, max_shared + 1))

    def allocate_shared_stubs(total_stubs: int) -> np.ndarray:
        counts = np.zeros(num_states, dtype=int)
        if total_stubs == 0:
            return counts

        base = min(markers_per_state, total_stubs // num_states)
        counts[:] = base
        remaining = total_stubs - base * num_states

        if base > 0 and num_states > 1:
            jitter_cap = max(1, base // 3)
            if jitter_cap > 0:
                reductions = rng.integers(0, jitter_cap + 1, size=num_states)
                reductions = np.minimum(reductions, counts)
                counts -= reductions
                remaining += int(reductions.sum())

        while remaining > 0:
            eligible = np.flatnonzero(counts < markers_per_state)
            if eligible.size == 0:
                raise ValueError(
                    "Cannot allocate shared markers without exceeding per-state capacity."
                )
            chosen = int(rng.choice(eligible))
            counts[chosen] += 1
            remaining -= 1
        return counts

    def rebalance_for_pairing(counts: np.ndarray) -> np.ndarray:
        total = int(counts.sum())
        if total == 0:
            return counts

        for _ in range(1024):
            limits = total - counts
            violating = counts > limits
            if not np.any(violating):
                return counts
            idx = int(np.argmax(counts - limits))
            excess = counts[idx] - limits[idx]
            if excess <= 0:
                continue
            counts[idx] -= excess
            total -= excess
            for _ in range(excess):
                candidates = np.flatnonzero(np.arange(num_states) != idx)
                candidates = candidates[counts[candidates] < markers_per_state]
                if candidates.size == 0:
                    raise ValueError("Unable to rebalance shared markers for pairwise assignment.")
                dest = int(rng.choice(candidates))
                counts[dest] += 1
                total += 1

        raise ValueError("Failed to find a pairable shared marker allocation.")

    def build_pair_counts(counts: np.ndarray) -> Dict[tuple[int, int], int]:
        remaining = counts.astype(int).copy()
        total_stubs = int(remaining.sum())
        if total_stubs == 0:
            return {}
        if total_stubs % 2 != 0:
            raise ValueError("Shared marker allocation must sum to an even number.")

        pairs: Dict[tuple[int, int], int] = defaultdict(int)

        while total_stubs > 0:
            active = np.flatnonzero(remaining)
            if active.size < 2:
                raise ValueError(
                    "Unable to pair remaining shared markers without violating constraints."
                )

            active = active[np.argsort(-remaining[active])]
            pivot = int(active[0])
            partner_candidates = active[1:]

            partner_weights = remaining[partner_candidates].astype(float)
            if partner_weights.sum() == 0:
                partner_weights = np.ones_like(partner_weights, dtype=float)

            partner = int(rng.choice(partner_candidates, p=partner_weights / partner_weights.sum()))

            remaining[pivot] -= 1
            remaining[partner] -= 1
            total_stubs -= 2

            if pivot < partner:
                key = (pivot, partner)
            else:
                key = (partner, pivot)
            pairs[key] += 1

        return pairs

    min_shared = max(0, required_slots - num_genes)
    max_shared = required_slots // 2
    shared_genes = sample_shared_gene_count(min_shared, max_shared)
    total_shared_stubs = 2 * shared_genes

    shared_per_state = allocate_shared_stubs(total_shared_stubs)
    shared_per_state = rebalance_for_pairing(shared_per_state)
    pair_counts = build_pair_counts(shared_per_state)

    gene_order = rng.permutation(num_genes)
    cursor = 0

    for (state_a, state_b), count in pair_counts.items():
        if cursor + count > num_genes:
            raise ValueError("Ran out of genes while assigning shared markers.")
        genes = gene_order[cursor : cursor + count]
        expression[state_a, genes] = marker_expression
        expression[state_b, genes] = marker_expression
        cursor += count

    for state in range(num_states):
        shared = shared_per_state[state]
        unique_needed = markers_per_state - shared
        if unique_needed < 0:
            raise ValueError("Shared marker allocation exceeded markers_per_state for a state.")
        if unique_needed == 0:
            continue
        if cursor + unique_needed > num_genes:
            raise ValueError("Ran out of genes while assigning unique markers.")
        genes = gene_order[cursor : cursor + unique_needed]
        expression[state, genes] = marker_expression
        cursor += unique_needed

    return expression
