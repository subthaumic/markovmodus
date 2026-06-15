from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import anndata as ad
import numpy as np
import pandas as pd

from .simulation import SimulationOutput


def simulation_to_dataframe(output: SimulationOutput) -> pd.DataFrame:
    """Convert a simulation result into a wide cell-by-feature pandas DataFrame."""
    num_cells, num_genes = output.unspliced.shape
    gene_names = output.parameters.gene_names

    data = {
        "time": np.full(num_cells, output.timepoint, dtype=float),
        "state": output.states.astype(int),
        "birth_parent": output.birth_parent.astype(int),
        "birth_time": output.birth_time.astype(float),
        "generation": output.generation.astype(int),
        "last_division_time": output.last_division_time.astype(float),
    }

    for gene_idx in range(num_genes):
        gene = gene_names[gene_idx]
        data[_column_label("u", gene)] = output.unspliced[:, gene_idx]
        data[_column_label("s", gene)] = output.spliced[:, gene_idx]

    index = pd.Index([f"cell_{i}" for i in range(num_cells)], name="cell_id")
    return pd.DataFrame(data, index=index).reset_index()


def simulation_to_anndata(output: SimulationOutput) -> ad.AnnData:
    """
    Convert a simulation result into an AnnData object.

    ``X`` contains raw spliced counts for single-cell tooling conventions. Raw paired U/S counts
    are also stored in ``layers["unspliced"]`` and ``layers["spliced"]``.
    """
    unspliced, spliced = output.as_arrays()

    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(output.states.size)],
            "state": output.states.astype(int),
            "time": np.full(output.states.size, output.timepoint, dtype=float),
            "birth_parent": output.birth_parent.astype(int),
            "birth_time": output.birth_time.astype(float),
            "generation": output.generation.astype(int),
            "last_division_time": output.last_division_time.astype(float),
        }
    )
    obs = obs.set_index("cell_id")

    gene_names = output.parameters.gene_names
    var = pd.DataFrame({"gene_id": gene_names}).set_index("gene_id")

    adata = ad.AnnData(X=spliced, obs=obs, var=var)
    adata.layers["unspliced"] = unspliced
    adata.layers["spliced"] = spliced

    adata.uns["simulation_params"] = output.parameters.to_metadata()
    adata.uns["transition_rate_mode"] = output.transition_rate_mode
    if output.transition_rates is not None:
        adata.uns["transition_rates"] = output.transition_rates.tolist()
    adata.uns["state_expression"] = output.state_expression.tolist()
    adata.uns["timepoint"] = output.timepoint

    return adata


def write_output(
    output: SimulationOutput,
    path: Path,
    *,
    file_format: Optional[Literal["csv", "parquet", "h5ad"]] = None,
) -> Path:
    """
    Persist a simulation result to disk.

    Parameters
    ----------
    output:
        Simulation result produced by :func:`markovmodus.simulation.simulate_population`.
    path:
        Destination path. The suffix is used to infer the format unless ``file_format`` is provided.
    file_format:
        Optional explicit format override. Supported values: ``csv``, ``parquet``, ``h5ad``.

    Returns
    -------
    Path
        The path that was written (useful when the underlying library adds suffixes).
    """
    path = Path(path)
    fmt = file_format or path.suffix.lstrip(".").lower()

    if fmt == "csv":
        df = simulation_to_dataframe(output)
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        df = simulation_to_dataframe(output)
        df.to_parquet(path, index=False)
    elif fmt == "h5ad":
        adata = simulation_to_anndata(output)
        adata.write_h5ad(path)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")

    return path


def _column_label(prefix: str, gene: str) -> str:
    """Construct a column label from a prefix and gene identifier."""
    return f"{prefix}_{gene}"
