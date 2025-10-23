import numpy as np
import pytest

from markovmodus.config import SimulationParameters
from markovmodus.simulation import simulate_population


def test_default_transition_matrix():
    params = SimulationParameters(
        num_states=3,
        num_genes=12,
        num_cells=10,
        t_final=1.0,
        dt=0.5,
        default_transition_rate=0.2,
        markers_per_state=4,
    )
    matrix = params.resolve_transition_matrix()

    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 0.0)
    off_diag = matrix[~np.eye(3, dtype=bool)]
    assert np.allclose(off_diag, 0.2)


def test_custom_transition_matrix_sanitised():
    raw = np.array(
        [
            [0.0, 0.3, -0.5],
            [0.1, 0.0, 0.4],
            [0.2, 0.1, 0.0],
        ]
    )

    params = SimulationParameters(
        num_states=3,
        num_genes=12,
        num_cells=10,
        t_final=1.0,
        dt=0.5,
        transition_matrix=raw,
        markers_per_state=4,
    )

    matrix = params.resolve_transition_matrix()
    assert np.all(matrix >= 0.0)
    assert matrix[0, 2] == pytest.approx(0.0)


def test_initial_state_distribution_normalises():
    params = SimulationParameters(
        num_states=4,
        num_genes=20,
        num_cells=5,
        t_final=1.0,
        dt=0.5,
        initial_state_probs=[1, 1, 2, 0],
        markers_per_state=5,
    )

    probs = params.initial_state_distribution
    assert probs.shape == (4,)
    assert np.isclose(probs.sum(), 1.0)
    assert probs[2] == pytest.approx(0.5)


def test_simulate_population_shapes():
    params = SimulationParameters(
        num_states=3,
        num_genes=15,
        num_cells=12,
        t_final=2.0,
        dt=0.5,
        marker_expression=8.0,
        baseline_expression=1.5,
        markers_per_state=5,
        rng_seed=7,
    )

    output = simulate_population(params)

    assert output.unspliced.shape == (params.num_cells, params.num_genes)
    assert output.spliced.shape == (params.num_cells, params.num_genes)
    assert output.states.shape == (params.num_cells,)
    assert np.all((output.states >= 0) & (output.states < params.num_states))
    assert output.timepoint == pytest.approx(params.t_final)


def test_simulation_reproducible_with_seed():
    params = SimulationParameters(
        num_states=2,
        num_genes=8,
        num_cells=5,
        t_final=1.2,
        dt=0.3,
        markers_per_state=3,
        rng_seed=42,
    )

    first = simulate_population(params)
    second = simulate_population(params)

    assert np.array_equal(first.unspliced, second.unspliced)
    assert np.array_equal(first.spliced, second.spliced)
    assert np.array_equal(first.states, second.states)


def test_simulation_initial_state_distribution_single_state():
    params = SimulationParameters(
        num_states=4,
        num_genes=6,
        num_cells=9,
        t_final=0.5,
        dt=0.5,
        default_transition_rate=0.0,
        markers_per_state=3,
        initial_state_probs=[0.0, 0.0, 1.0, 0.0],
        rng_seed=0,
    )

    output = simulate_population(params)
    assert np.all(output.states == 2)


def test_simulation_dispersion_must_be_positive():
    params = SimulationParameters(
        num_states=2,
        num_genes=5,
        num_cells=4,
        t_final=1.0,
        dt=0.5,
        markers_per_state=2,
        dispersion=0.0,
    )

    with pytest.raises(ValueError):
        simulate_population(params)


def test_simulation_dispersion_applies_noise():
    base_params = SimulationParameters(
        num_states=3,
        num_genes=9,
        num_cells=6,
        t_final=1.5,
        dt=0.5,
        markers_per_state=3,
        rng_seed=11,
    )
    noisy_params = SimulationParameters(
        num_states=3,
        num_genes=9,
        num_cells=6,
        t_final=1.5,
        dt=0.5,
        markers_per_state=3,
        rng_seed=11,
        dispersion=1.0,
    )

    baseline = simulate_population(base_params)
    noisy = simulate_population(noisy_params)

    assert noisy.unspliced.shape == baseline.unspliced.shape
    assert noisy.spliced.shape == baseline.spliced.shape
    assert np.all(noisy.unspliced >= 0)
    assert np.all(noisy.spliced >= 0)
    assert not np.array_equal(noisy.unspliced, baseline.unspliced) or not np.array_equal(
        noisy.spliced, baseline.spliced
    )
