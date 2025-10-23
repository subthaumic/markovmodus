from pathlib import Path

import pandas as pd
import pytest

from markovmodus import SimulationParameters, simulate_dataset


def make_params() -> SimulationParameters:
    return SimulationParameters(
        num_states=3,
        num_genes=10,
        num_cells=20,
        t_final=1.5,
        dt=0.5,
        markers_per_state=4,
        marker_expression=7.5,
        baseline_expression=1.2,
        rng_seed=123,
    )


def test_simulate_dataset_to_dataframe(tmp_path: Path):
    params = make_params()
    csv_path = tmp_path / "synthetic.csv"

    df = simulate_dataset(params, output="dataframe", save_path=csv_path)

    assert isinstance(df, pd.DataFrame)
    assert csv_path.exists()
    assert "state" in df.columns
    assert any(col.startswith("u_") for col in df.columns)
    assert any(col.startswith("s_") for col in df.columns)


def test_simulate_dataset_both_returns_types():
    params = make_params()
    adata, df = simulate_dataset(params, output="both")

    assert hasattr(adata, "layers")
    assert "unspliced" in adata.layers
    assert "spliced" in adata.layers
    assert isinstance(df, pd.DataFrame)
    assert adata.n_obs == len(df)


def test_simulate_dataset_invalid_output():
    params = make_params()
    with pytest.raises(ValueError):
        simulate_dataset(params, output="unsupported")  # type: ignore[arg-type]
