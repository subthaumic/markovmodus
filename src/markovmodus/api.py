from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Tuple, Union

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - optional dependency type hints
    import anndata as ad
    from .config import SimulationParameters

from .io import simulation_to_anndata, simulation_to_dataframe, write_output
from .simulation import simulate_population

ReturnType = Union["ad.AnnData", pd.DataFrame, Tuple["ad.AnnData", pd.DataFrame]]


def simulate_dataset(
    params: SimulationParameters,
    *,
    output: Literal["anndata", "dataframe", "both"] = "anndata",
    save_path: Optional[Union[str, Path]] = None,
    file_format: Optional[Literal["csv", "parquet", "h5ad"]] = None,
) -> ReturnType:
    """
    Run a simulation and return the requested representation.

    Parameters
    ----------
    params:
        Configuration for the simulator.
    output:
        Requested return type. ``"both"`` returns ``(AnnData, DataFrame)``.
    save_path:
        Optional path to persist the generated dataset. The extension selects the storage format unless
        ``file_format`` is provided.
    file_format:
        Optional format specifier overriding the suffix of ``save_path``.
    """
    result = simulate_population(params)

    adata = simulation_to_anndata(result)
    dataframe = simulation_to_dataframe(result)

    if save_path is not None:
        path = Path(save_path)
        write_output(result, path, file_format=file_format)

    if output == "anndata":
        return adata
    if output == "dataframe":
        return dataframe
    if output == "both":
        return adata, dataframe

    raise ValueError(f"Unknown output specification: {output}")


__all__ = ["simulate_dataset"]
