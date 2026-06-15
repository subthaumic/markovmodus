import numpy as np
import pytest

from markovmodus.config import SimulationParameters
from markovmodus.simulation import Simulation, SimulationState, simulate_population


def test_default_transition_rates():
    params = SimulationParameters(
        num_states=3,
        num_genes=12,
        num_cells=10,
        t_final=1.0,
        dt=0.5,
        markers_per_state=4,
    )
    matrix = params.resolve_transition_rates()

    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 0.0)
    off_diag = matrix[~np.eye(3, dtype=bool)]
    assert np.allclose(off_diag, 0.1)


def test_custom_transition_rates_sanitised():
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
        transition_rates=raw,
        markers_per_state=4,
    )

    matrix = params.resolve_transition_rates()
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
    assert output.birth_parent.shape == (params.num_cells,)
    assert output.birth_time.shape == (params.num_cells,)
    assert output.generation.shape == (params.num_cells,)
    assert output.last_division_time.shape == (params.num_cells,)
    assert np.all((output.states >= 0) & (output.states < params.num_states))
    assert np.all(output.birth_parent == -1)
    assert np.all(output.birth_time == 0.0)
    assert np.all(output.generation == 0)
    assert np.all(output.last_division_time == 0.0)
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


def test_simulation_class_runs_to_output():
    params = SimulationParameters(
        num_states=2,
        num_genes=8,
        num_cells=5,
        t_final=1.0,
        dt=0.5,
        markers_per_state=3,
        rng_seed=42,
    )

    output = Simulation(params).run().to_output()

    assert output.timepoint == pytest.approx(params.t_final)
    assert output.transition_rate_mode == "static"
    assert output.transition_rates is not None
    assert not hasattr(output, "transition_matrix")


def test_simulation_initial_state_distribution_single_state():
    params = SimulationParameters(
        num_states=4,
        num_genes=12,
        num_cells=9,
        t_final=0.5,
        dt=0.5,
        transition_rates=np.zeros((4, 4)),
        markers_per_state=3,
        initial_state_probs=[0.0, 0.0, 1.0, 0.0],
        rng_seed=0,
    )

    output = simulate_population(params)
    assert np.all(output.states == 2)


def test_dynamic_transition_rates_receive_simulation_state():
    seen: list[SimulationState] = []

    def dynamic_rates(state: SimulationState) -> np.ndarray:
        seen.append(state)
        return np.zeros((3, 3))

    params = SimulationParameters(
        num_states=3,
        num_genes=6,
        num_cells=4,
        t_final=1.0,
        dt=0.5,
        markers_per_state=2,
        marker_reuse_cap=1,
        transition_rates=dynamic_rates,
        rng_seed=1,
    )

    simulate_population(params)

    assert [state.time for state in seen] == [0.0, 0.5]
    assert seen[0].cell_states.shape == (params.num_cells,)
    assert seen[0].unspliced.shape == (params.num_cells, params.num_genes)
    assert seen[0].spliced.shape == (params.num_cells, params.num_genes)
    assert seen[0].birth_parent.shape == (params.num_cells,)
    assert seen[0].birth_time.shape == (params.num_cells,)
    assert seen[0].generation.shape == (params.num_cells,)
    assert seen[0].last_division_time.shape == (params.num_cells,)
    assert np.all(seen[0].birth_parent == -1)
    assert np.all(seen[0].birth_time == 0.0)
    assert np.all(seen[0].generation == 0)
    assert np.all(seen[0].last_division_time == 0.0)


def test_dynamic_transition_state_arrays_are_read_only():
    checked = False

    def dynamic_rates(state: SimulationState) -> np.ndarray:
        nonlocal checked
        if not checked:
            assert not state.cell_states.flags.writeable
            assert not state.unspliced.flags.writeable
            assert not state.spliced.flags.writeable
            assert not state.birth_parent.flags.writeable
            assert not state.birth_time.flags.writeable
            assert not state.generation.flags.writeable
            assert not state.last_division_time.flags.writeable
            with pytest.raises(ValueError):
                state.cell_states[0] = 1
            with pytest.raises(ValueError):
                state.unspliced[0, 0] = 1
            with pytest.raises(ValueError):
                state.spliced[0, 0] = 1
            with pytest.raises(ValueError):
                state.birth_parent[0] = 1
            with pytest.raises(ValueError):
                state.birth_time[0] = 1.0
            with pytest.raises(ValueError):
                state.generation[0] = 1
            with pytest.raises(ValueError):
                state.last_division_time[0] = 1.0
            checked = True
        return np.zeros((2, 2))

    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=3,
        t_final=0.5,
        dt=0.5,
        markers_per_state=2,
        marker_reuse_cap=1,
        transition_rates=dynamic_rates,
        rng_seed=1,
    )

    simulate_population(params)
    assert checked


def test_time_dependent_transition_rates_drive_transitions():
    def dynamic_rates(state: SimulationState) -> np.ndarray:
        rates = np.zeros((2, 2))
        if state.time >= 1.0:
            rates[0, 1] = 1000.0
        return rates

    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=8,
        t_final=2.0,
        dt=1.0,
        markers_per_state=2,
        initial_state_probs=[1.0, 0.0],
        transition_rates=dynamic_rates,
        rng_seed=5,
    )

    output = simulate_population(params)
    assert np.all(output.states == 1)
    assert output.transition_rate_mode == "dynamic"
    assert output.transition_rates is None
    assert not hasattr(output, "transition_matrix")


def test_state_dependent_transition_rates_can_use_cell_states():
    first_count: np.ndarray | None = None

    def dynamic_rates(state: SimulationState) -> np.ndarray:
        nonlocal first_count
        counts = np.bincount(state.cell_states, minlength=2)
        if first_count is None:
            first_count = counts
        rates = np.zeros((2, 2))
        rates[0, 1] = 1000.0 if counts[0] == state.cell_states.size else 0.0
        return rates

    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=6,
        t_final=1.0,
        dt=1.0,
        markers_per_state=2,
        initial_state_probs=[1.0, 0.0],
        transition_rates=dynamic_rates,
        rng_seed=2,
    )

    output = simulate_population(params)

    assert first_count is not None
    assert first_count.tolist() == [params.num_cells, 0]
    assert np.all(output.states == 1)


def test_per_cell_transition_rate_tensor_affects_only_selected_cells():
    def dynamic_rates(state: SimulationState) -> np.ndarray:
        rates = np.zeros((state.cell_states.size, 2, 2))
        rates[:3, 0, 1] = 1000.0
        return rates

    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=6,
        t_final=1.0,
        dt=1.0,
        markers_per_state=2,
        marker_reuse_cap=1,
        initial_state_probs=[1.0, 0.0],
        transition_rates=dynamic_rates,
        rng_seed=3,
    )

    output = simulate_population(params)

    assert output.states[:3].tolist() == [1, 1, 1]
    assert output.states[3:].tolist() == [0, 0, 0]


@pytest.mark.parametrize(
    "bad_rates",
    [
        np.zeros((2, 3)),
        np.full((2, 2), np.nan),
    ],
)
def test_invalid_dynamic_transition_rates_raise(bad_rates: np.ndarray):
    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=3,
        t_final=1.0,
        dt=1.0,
        markers_per_state=2,
        transition_rates=lambda state: bad_rates,
    )

    with pytest.raises(ValueError):
        simulate_population(params)


def test_state_level_proliferation_divides_cells_with_lineage():
    def proliferation_model(state: SimulationState) -> tuple[np.ndarray, np.ndarray]:
        division_rates = np.array([1000.0, 0.0])
        daughter_state_probs = np.array(
            [
                [0.0, 1.0],
                [1.0, 0.0],
            ]
        )
        return division_rates, daughter_state_probs

    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=4,
        t_final=1.0,
        dt=1.0,
        markers_per_state=2,
        marker_reuse_cap=1,
        initial_state_probs=[1.0, 0.0],
        transition_rates=np.zeros((2, 2)),
        proliferation_model=proliferation_model,
        rng_seed=13,
    )

    output = simulate_population(params)

    assert output.states[: params.num_cells].tolist() == [1, 1, 1, 1]
    assert output.states[params.num_cells :].tolist() == [1, 1, 1, 1]
    assert output.unspliced.shape == (2 * params.num_cells, params.num_genes)
    assert output.spliced.shape == (2 * params.num_cells, params.num_genes)
    np.testing.assert_array_equal(
        output.unspliced[params.num_cells :],
        output.unspliced[: params.num_cells],
    )
    np.testing.assert_array_equal(
        output.spliced[params.num_cells :],
        output.spliced[: params.num_cells],
    )
    np.testing.assert_array_equal(output.birth_parent[: params.num_cells], np.full(4, -1))
    np.testing.assert_array_equal(output.birth_parent[params.num_cells :], np.arange(4))
    np.testing.assert_array_equal(output.birth_time[: params.num_cells], np.zeros(4))
    np.testing.assert_array_equal(output.birth_time[params.num_cells :], np.ones(4))
    np.testing.assert_array_equal(output.generation, np.ones(8, dtype=int))
    np.testing.assert_array_equal(output.last_division_time, np.ones(8))


def test_per_cell_proliferation_affects_only_selected_cells():
    def proliferation_model(state: SimulationState) -> tuple[np.ndarray, np.ndarray]:
        division_rates = np.zeros(state.cell_states.size)
        division_rates[:2] = 1000.0
        daughter_state_probs = np.tile(np.array([0.0, 1.0]), (state.cell_states.size, 1))
        return division_rates, daughter_state_probs

    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=5,
        t_final=1.0,
        dt=1.0,
        markers_per_state=2,
        marker_reuse_cap=1,
        initial_state_probs=[1.0, 0.0],
        transition_rates=np.zeros((2, 2)),
        proliferation_model=proliferation_model,
        rng_seed=17,
    )

    output = simulate_population(params)

    assert output.states[: params.num_cells].tolist() == [1, 1, 0, 0, 0]
    assert output.states[params.num_cells :].tolist() == [1, 1]
    np.testing.assert_array_equal(output.birth_parent[params.num_cells :], np.array([0, 1]))
    np.testing.assert_array_equal(output.birth_time[params.num_cells :], np.ones(2))
    np.testing.assert_array_equal(output.generation, np.array([1, 1, 0, 0, 0, 1, 1]))
    np.testing.assert_array_equal(output.last_division_time, np.array([1, 1, 0, 0, 0, 1, 1]))


def test_dividing_cells_skip_normal_latent_transitions():
    def proliferation_model(state: SimulationState) -> tuple[np.ndarray, np.ndarray]:
        return np.array([1000.0, 0.0]), np.array([[1.0, 0.0], [0.0, 1.0]])

    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=4,
        t_final=1.0,
        dt=1.0,
        markers_per_state=2,
        marker_reuse_cap=1,
        initial_state_probs=[1.0, 0.0],
        transition_rates=np.array([[0.0, 1000.0], [0.0, 0.0]]),
        proliferation_model=proliferation_model,
        rng_seed=23,
    )

    output = simulate_population(params)

    assert output.states.tolist() == [0] * 8


def test_non_dividing_cells_transition_normally_with_proliferation_enabled():
    def proliferation_model(state: SimulationState) -> tuple[np.ndarray, np.ndarray]:
        division_rates = np.zeros(state.cell_states.size)
        division_rates[:2] = 1000.0
        daughter_state_probs = np.tile(np.array([1.0, 0.0]), (state.cell_states.size, 1))
        return division_rates, daughter_state_probs

    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=5,
        t_final=1.0,
        dt=1.0,
        markers_per_state=2,
        marker_reuse_cap=1,
        initial_state_probs=[1.0, 0.0],
        transition_rates=np.array([[0.0, 1000.0], [0.0, 0.0]]),
        proliferation_model=proliferation_model,
        rng_seed=29,
    )

    output = simulate_population(params)

    assert output.states[: params.num_cells].tolist() == [0, 0, 1, 1, 1]
    assert output.states[params.num_cells :].tolist() == [0, 0]


@pytest.mark.parametrize(
    "model_output",
    [
        (np.zeros((2, 2)), np.zeros((2, 2))),
        (np.array([np.nan, 0.0]), np.eye(2)),
        (np.array([1.0, 0.0]), np.zeros((2, 2))),
        (np.array([1.0, 0.0]), np.array([[-1.0, 2.0], [1.0, 0.0]])),
    ],
)
def test_invalid_proliferation_model_outputs_raise(
    model_output: tuple[np.ndarray, np.ndarray],
):
    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=3,
        t_final=1.0,
        dt=1.0,
        markers_per_state=2,
        marker_reuse_cap=1,
        proliferation_model=lambda state: model_output,
    )

    with pytest.raises(ValueError, match="proliferation_model"):
        simulate_population(params)


def test_dynamic_transition_rates_see_expanded_population_after_proliferation():
    seen_sizes: list[int] = []

    def dynamic_rates(state: SimulationState) -> np.ndarray:
        seen_sizes.append(state.cell_states.size)
        return np.zeros((state.cell_states.size, 2, 2))

    def proliferation_model(state: SimulationState) -> tuple[np.ndarray, np.ndarray]:
        if state.time < 1.0:
            division_rates = np.full(state.cell_states.size, 1000.0)
            daughter_state_probs = np.tile(np.array([1.0, 0.0]), (state.cell_states.size, 1))
            return division_rates, daughter_state_probs
        return np.zeros(state.cell_states.size), np.zeros((state.cell_states.size, 2))

    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=3,
        t_final=2.0,
        dt=1.0,
        markers_per_state=2,
        marker_reuse_cap=1,
        initial_state_probs=[1.0, 0.0],
        transition_rates=dynamic_rates,
        proliferation_model=proliferation_model,
        rng_seed=19,
    )

    output = simulate_population(params)

    assert seen_sizes == [3, 6]
    assert output.states.shape == (6,)


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
