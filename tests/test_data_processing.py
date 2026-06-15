from pathlib import Path

import anndata as ad
import numpy as np
import pytest

from markovmodus.config import SimulationParameters
from markovmodus.io import simulation_to_anndata, simulation_to_dataframe, write_output
from markovmodus.simulation import simulate_population


def test_io_conversions_round_trip():
    params = SimulationParameters(
        num_states=2,
        num_genes=5,
        num_cells=3,
        t_final=1.0,
        dt=0.5,
        markers_per_state=3,
        marker_overlap=1,
        rng_seed=99,
    )

    output = simulate_population(params)
    df = simulation_to_dataframe(output)
    adata = simulation_to_anndata(output)

    assert set(df.columns[:3]) == {"cell_id", "time", "state"}
    assert adata.n_obs == params.num_cells
    assert adata.n_vars == params.num_genes
    np.testing.assert_array_equal(adata.X, output.spliced)
    np.testing.assert_array_equal(adata.layers["unspliced"], output.unspliced)
    np.testing.assert_array_equal(adata.layers["spliced"], output.spliced)
    assert adata.uns["transition_rate_mode"] == "static"
    assert "transition_rates" in adata.uns
    assert "transition_matrix" not in adata.uns


def test_write_output_h5ad(tmp_path: Path):
    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=5,
        t_final=1.0,
        dt=0.5,
        markers_per_state=2,
        rng_seed=101,
    )

    output = simulate_population(params)
    path = tmp_path / "synthetic_counts.h5ad"
    write_output(output, path, file_format="h5ad")

    assert path.exists()
    loaded = ad.read_h5ad(path)
    assert loaded.n_obs == params.num_cells
    np.testing.assert_array_equal(loaded.X, output.spliced)
    assert "unspliced" in loaded.layers
    assert "spliced" in loaded.layers


def test_dynamic_transition_rates_metadata():
    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=5,
        t_final=1.0,
        dt=0.5,
        markers_per_state=2,
        transition_rates=lambda state: np.zeros((2, 2)),
        rng_seed=101,
    )

    output = simulate_population(params)
    adata = simulation_to_anndata(output)

    assert adata.uns["transition_rate_mode"] == "dynamic"
    assert "transition_rates" not in adata.uns
    assert "transition_matrix" not in adata.uns


def test_proliferation_lineage_metadata_in_outputs():
    def proliferation_model(state):
        return (
            np.full(state.cell_states.size, 1000.0),
            np.tile(np.array([1.0, 0.0]), (state.cell_states.size, 1)),
        )

    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=3,
        t_final=1.0,
        dt=1.0,
        markers_per_state=2,
        marker_reuse_cap=1,
        initial_state_probs=[1.0, 0.0],
        transition_rates=np.zeros((2, 2)),
        proliferation_model=proliferation_model,
        rng_seed=103,
    )

    output = simulate_population(params)
    df = simulation_to_dataframe(output)
    adata = simulation_to_anndata(output)

    assert df.shape[0] == 2 * params.num_cells
    assert adata.n_obs == 2 * params.num_cells
    lineage_columns = {
        "birth_parent",
        "birth_time",
        "generation",
        "last_division_time",
    }
    assert lineage_columns.issubset(df.columns)
    assert lineage_columns.issubset(adata.obs.columns)
    np.testing.assert_array_equal(df["birth_parent"].to_numpy()[:3], np.full(3, -1))
    np.testing.assert_array_equal(df["birth_parent"].to_numpy()[3:], np.arange(3))
    np.testing.assert_array_equal(adata.obs["birth_parent"].to_numpy()[3:], np.arange(3))
    np.testing.assert_array_equal(df["generation"].to_numpy(), np.ones(6, dtype=int))
    np.testing.assert_array_equal(adata.obs["generation"].to_numpy(), np.ones(6, dtype=int))
    np.testing.assert_array_equal(df["last_division_time"].to_numpy(), np.ones(6))
    np.testing.assert_array_equal(adata.obs["last_division_time"].to_numpy(), np.ones(6))


def test_write_output_invalid_format(tmp_path: Path):
    params = SimulationParameters(
        num_states=2,
        num_genes=4,
        num_cells=5,
        t_final=1.0,
        dt=0.5,
        markers_per_state=2,
        rng_seed=5,
    )
    output = simulate_population(params)

    with pytest.raises(ValueError):
        write_output(output, tmp_path / "data.unknown", file_format="unknown")  # type: ignore[arg-type]
