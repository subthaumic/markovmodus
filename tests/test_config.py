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
        SimulationParameters(**base_kwargs(), marker_overlap=-1)

    with pytest.raises(ValueError):
        SimulationParameters(**base_kwargs(), marker_overlap=2)

    with pytest.raises(ValueError):
        SimulationParameters(**base_kwargs(), marker_overlap=1, marker_reuse_cap=1)

    with pytest.raises(ValueError):
        SimulationParameters(**base_kwargs(), splicing_rate=0.0)


def test_default_transition_rate_is_not_a_public_parameter():
    with pytest.raises(TypeError):
        SimulationParameters(**base_kwargs(), default_transition_rate=0.1)  # type: ignore[call-arg]


def test_simulation_parameters_are_keyword_only():
    with pytest.raises(TypeError):
        SimulationParameters(3, 6, 10, 2.0)  # type: ignore[call-arg]


def test_transition_matrix_keyword_points_to_transition_rates():
    with pytest.raises(TypeError, match="Use transition_rates instead"):
        SimulationParameters(**base_kwargs(), transition_matrix=np.ones((3, 3)))  # type: ignore[call-arg]


def test_transition_rates_shape_validation():
    with pytest.raises(ValueError):
        SimulationParameters(**base_kwargs(), transition_rates=np.ones((2, 2)))


def test_transition_rates_finite_validation():
    matrix = np.zeros((3, 3))
    matrix[0, 1] = np.inf

    with pytest.raises(ValueError):
        SimulationParameters(**base_kwargs(), transition_rates=matrix)


def test_internal_default_transition_rates_materialized_on_initialization():
    params = SimulationParameters(**base_kwargs())

    assert isinstance(params.transition_rates, np.ndarray)
    assert params.transition_rates.shape == (3, 3)
    assert np.allclose(np.diag(params.transition_rates), 0.0)
    off_diag = params.transition_rates[~np.eye(3, dtype=bool)]
    assert np.allclose(off_diag, 0.1)
    assert params.to_metadata()["transition_rates"] == params.transition_rates.tolist()


def test_callable_transition_rates_accepted_and_serialised():
    def dynamic_rates(state):
        return np.zeros((3, 3))

    params = SimulationParameters(**base_kwargs(), transition_rates=dynamic_rates)

    assert params.resolve_transition_rates() is dynamic_rates
    assert params.to_metadata()["transition_rates"] == "<callable>"


def test_callable_proliferation_model_serialised():
    def proliferation_model(state):
        return np.zeros(3), np.eye(3)

    params = SimulationParameters(**base_kwargs(), proliferation_model=proliferation_model)

    assert params.proliferation_model is proliferation_model
    assert params.to_metadata()["proliferation_model"] == "<callable>"


def test_marker_overlap_mapping_serialised():
    kwargs = base_kwargs()
    kwargs.update({"num_genes": 7, "marker_overlap": {(1, 0): 1, (1, 2): 1}})
    params = SimulationParameters(**kwargs)

    assert params.to_metadata()["marker_overlap"] == {"0,1": 1, "1,2": 1}


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


def test_marker_shorthand_materializes_state_expression():
    params = SimulationParameters(**base_kwargs())

    assert params.state_expression is not None
    resolved = params.resolve_state_expression(np.random.default_rng(0))
    assert np.array_equal(resolved, params.state_expression)
    assert resolved.shape == (params.num_states, params.num_genes)


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
    assert marker_mask.tolist() == [
        [True, True, False, False, False, False],
        [False, False, True, True, False, False],
        [False, False, False, False, True, True],
    ]


def test_marker_allocation_does_not_depend_on_rng():
    params = SimulationParameters(**base_kwargs(), marker_reuse_cap=1)

    first = params.resolve_state_expression(np.random.default_rng(123))
    second = params.resolve_state_expression(np.random.default_rng(999))

    assert np.array_equal(first, second)


def test_scalar_marker_overlap_uses_static_transition_support():
    transition_rates = np.array(
        [
            [0.0, 0.2, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    kwargs = base_kwargs()
    kwargs.update(
        {
            "num_states": 4,
            "num_genes": 18,
            "markers_per_state": 6,
            "marker_overlap": 2,
            "marker_reuse_cap": 2,
            "transition_rates": transition_rates,
        }
    )
    params = SimulationParameters(**kwargs)
    rng = np.random.default_rng(321)
    expression = params.resolve_state_expression(rng)

    marker_mask = expression == params.marker_expression
    assert marker_mask.sum(axis=1).tolist() == [6, 6, 6, 6]
    assert marker_mask[0, [0, 1, 2, 3, 8, 9]].all()
    assert marker_mask[1, [0, 1, 4, 5, 10, 11]].all()
    assert marker_mask[2, [2, 3, 6, 7, 12, 13]].all()
    assert marker_mask[3, [4, 5, 6, 7, 14, 15]].all()

    usage = marker_mask.sum(axis=0)
    assert usage.max() <= params.marker_reuse_cap
    assert usage.tolist() == [2] * 8 + [1] * 8 + [0] * 2


def test_marker_overlap_mapping_controls_exact_shared_groups():
    params = SimulationParameters(
        num_states=4,
        num_genes=18,
        num_cells=10,
        t_final=2.0,
        markers_per_state=5,
        marker_overlap={(0, 2): 2, (1, 2, 3): 1},
        marker_reuse_cap=3,
        marker_expression=5.0,
        baseline_expression=1.0,
    )

    expression = params.resolve_state_expression(np.random.default_rng(321))
    marker_mask = expression == params.marker_expression

    assert marker_mask.sum(axis=1).tolist() == [5, 5, 5, 5]
    assert marker_mask[0, [0, 1, 3, 4, 5]].all()
    assert marker_mask[1, [2, 6, 7, 8, 9]].all()
    assert marker_mask[2, [0, 1, 2, 10, 11]].all()
    assert marker_mask[3, [2, 12, 13, 14, 15]].all()
    assert marker_mask[:, 0].sum() == 2
    assert marker_mask[:, 2].sum() == 3


def test_marker_overlap_cannot_exceed_per_state_marker_slots():
    kwargs = base_kwargs()
    kwargs.update(
        {
            "num_genes": 10,
            "markers_per_state": 6,
            "marker_overlap": 4,
            "marker_reuse_cap": 2,
        }
    )
    with pytest.raises(ValueError, match="shared marker slots"):
        SimulationParameters(**kwargs)


def test_marker_overlap_mapping_rejects_group_larger_than_reuse_cap():
    with pytest.raises(ValueError, match="marker_reuse_cap"):
        SimulationParameters(
            **base_kwargs(),
            marker_overlap={(0, 1, 2): 1},
            marker_reuse_cap=2,
        )


def test_integer_marker_overlap_requires_static_transition_rates():
    with pytest.raises(ValueError, match="static transition_rates"):
        SimulationParameters(
            **base_kwargs(),
            transition_rates=lambda state: np.zeros((3, 3)),
            marker_overlap=1,
        )


def test_marker_overlap_mapping_requires_sufficient_genes():
    with pytest.raises(ValueError, match="Not enough genes"):
        SimulationParameters(
            num_states=3,
            num_genes=5,
            num_cells=10,
            t_final=2.0,
            markers_per_state=3,
            marker_overlap={(0, 1): 1},
        )


def test_marker_overlap_mapping_rejects_invalid_groups():
    with pytest.raises(ValueError, match="at least two states"):
        SimulationParameters(**base_kwargs(), marker_overlap={(0,): 1})

    with pytest.raises(ValueError, match="outside"):
        SimulationParameters(**base_kwargs(), marker_overlap={(0, 3): 1})

    with pytest.raises(ValueError, match="duplicate states"):
        SimulationParameters(**base_kwargs(), marker_overlap={(0, 0): 1})

    with pytest.raises(ValueError, match="duplicate group"):
        SimulationParameters(**base_kwargs(), marker_overlap={(0, 1): 1, (1, 0): 1})

    with pytest.raises(ValueError, match="counts must be non-negative"):
        SimulationParameters(**base_kwargs(), marker_overlap={(0, 1): -1})


def test_marker_allocation_requires_sufficient_genes():
    kwargs = base_kwargs()
    kwargs.update(
        {
            "num_genes": 4,
            "markers_per_state": 3,
            "marker_reuse_cap": 1,
        }
    )
    with pytest.raises(ValueError, match="Not enough genes"):
        SimulationParameters(**kwargs)


def test_reuse_cap_larger_than_two_supports_larger_overlap_groups():
    kwargs = base_kwargs()
    kwargs.update(
        {
            "num_genes": 7,
            "marker_overlap": {(0, 1, 2): 1},
            "marker_reuse_cap": 3,
        }
    )
    params = SimulationParameters(**kwargs)

    expression = params.resolve_state_expression(np.random.default_rng(0))
    marker_mask = expression == params.marker_expression

    assert marker_mask[:, 0].tolist() == [True, True, True]
    assert marker_mask.sum(axis=1).tolist() == [2, 2, 2]
