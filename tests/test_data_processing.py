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
        rng_seed=99,
    )

    output = simulate_population(params)
    df = simulation_to_dataframe(output)
    adata = simulation_to_anndata(output)

    assert set(df.columns[:3]) == {"cell_id", "time", "state"}
    assert adata.n_obs == params.num_cells
    assert adata.n_vars == params.num_genes
    np.testing.assert_array_equal(adata.layers["unspliced"], output.unspliced)
    np.testing.assert_array_equal(adata.layers["spliced"], output.spliced)


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
    path = tmp_path / "snapshot.h5ad"
    write_output(output, path, file_format="h5ad")

    assert path.exists()
    loaded = ad.read_h5ad(path)
    assert loaded.n_obs == params.num_cells
    assert "unspliced" in loaded.layers
    assert "spliced" in loaded.layers


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
