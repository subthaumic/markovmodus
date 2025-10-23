import numpy as np
import pytest

from markovmodus.config import SimulationParameters


def base_kwargs() -> dict:
    return {
        "num_states": 3,
        "num_genes": 6,
        "num_cells": 10,
        "t_final": 2.0,
        "dt": 0.5,
        "markers_per_state": 2,
        "marker_expression": 5.0,
        "baseline_expression": 1.0,
    }


def test_invalid_positive_parameters():
    with pytest.raises(ValueError):
        SimulationParameters(
            num_states=0, num_genes=5, num_cells=5, t_final=1.0, markers_per_state=2
        )

    with pytest.raises(ValueError):
        kwargs = base_kwargs()
        kwargs["dt"] = 0.0
        SimulationParameters(**kwargs)

    with pytest.raises(ValueError):
        SimulationParameters(**base_kwargs(), default_transition_rate=-0.1)

    with pytest.raises(ValueError):
        SimulationParameters(**base_kwargs(), splicing_rate=0.0)


def test_transition_matrix_shape_validation():
    with pytest.raises(ValueError):
        SimulationParameters(**base_kwargs(), transition_matrix=np.ones((2, 2)))


def test_initial_state_probs_validation():
    with pytest.raises(ValueError):
        SimulationParameters(**base_kwargs(), initial_state_probs=[0.5, 0.5])

    with pytest.raises(ValueError):
        SimulationParameters(**base_kwargs(), initial_state_probs=[0.0, 0.0, 0.0])


def test_gene_id_length_validation():
    with pytest.raises(ValueError):
        SimulationParameters(**base_kwargs(), gene_ids=["g0", "g1"])


def test_state_expression_override_is_respected():
    custom = np.full((3, 6), 4.2)
    params = SimulationParameters(**base_kwargs(), state_expression=custom)

    resolved = params.resolve_state_expression(np.random.default_rng(0))
    assert np.array_equal(resolved, custom)

    # Ensure internal copy is protected.
    resolved[0, 0] = 999.0
    repeat = params.resolve_state_expression(np.random.default_rng(0))
    assert repeat[0, 0] == pytest.approx(4.2)


def test_marker_reuse_cap_is_enforced():
    params = SimulationParameters(
        **base_kwargs(),
        marker_reuse_cap=1,
    )
    rng = np.random.default_rng(123)
    expression = params.resolve_state_expression(rng)

    marker_mask = expression == params.marker_expression
    assert marker_mask.sum(axis=1).tolist() == [2, 2, 2]
    assert np.all(marker_mask.sum(axis=0) <= params.marker_reuse_cap)


def test_pairwise_marker_allocation_is_balanced():
    kwargs = base_kwargs()
    kwargs.update(
        {
            "num_states": 4,
            "num_genes": 40,
            "markers_per_state": 6,
            "marker_reuse_cap": 2,
        }
    )
    params = SimulationParameters(**kwargs)
    rng = np.random.default_rng(321)
    expression = params.resolve_state_expression(rng)

    marker_mask = expression == params.marker_expression
    assert marker_mask.sum(axis=1).tolist() == [6, 6, 6, 6]

    usage = marker_mask.sum(axis=0)
    assert usage.max() <= params.marker_reuse_cap

    shared_gene_mask = usage == 2
    shared_gene_count = int(shared_gene_mask.sum())
    assert shared_gene_count > 0

    shared_per_state = marker_mask[:, shared_gene_mask].sum(axis=1)
    assert shared_per_state.sum() == 2 * shared_gene_count
    assert shared_per_state.max() - shared_per_state.min() <= params.markers_per_state // 2 + 2


def test_pairwise_allocation_requires_sufficient_genes():
    kwargs = base_kwargs()
    kwargs.update(
        {
            "num_genes": 4,
            "markers_per_state": 3,
            "marker_reuse_cap": 1,
        }
    )
    params = SimulationParameters(**kwargs)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        params.resolve_state_expression(rng)


def test_reuse_cap_larger_than_two_not_supported():
    params = SimulationParameters(
        **base_kwargs(),
        marker_reuse_cap=3,
    )
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        params.resolve_state_expression(rng)
