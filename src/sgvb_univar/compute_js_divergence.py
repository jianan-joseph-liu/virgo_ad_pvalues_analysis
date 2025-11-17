"""
Utilities to compute Jensen-Shannon (JS) divergences between posteriors
from multiple inference runs (e.g., SGVB vs Welch).

Typical usage:
    results = {
        "sgvb": load_bilby_result("/path/to/{event}_sgvb_result.json", event),
        "welch": load_bilby_result("/path/to/{event}_welch_result.json", event),
    }
    params = ["chirp_mass", "mass_ratio", "chi_eff"]
    df = compute_js_table(params, results, reference="sgvb")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

import bilby
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

logger = logging.getLogger(__name__)

DEFAULT_BINS = 100


def load_bilby_result(path_template: str | Path, event: str | None = None):
    """
    Read a bilby result file, optionally formatting the path with the event name.

    Parameters
    ----------
    path_template:
        Either a full path to a JSON result file or a format string containing
        ``{event}`` which will be replaced by the provided event name.
    event:
        Event name used to format the path when a template is supplied.
    """
    path = Path(str(path_template).format(event=event))
    logger.info("Loading bilby result: %s", path)
    return bilby.result.read_in_result(filename=str(path))


def load_lvc_posterior(path_template: str | Path, event: str | None = None) -> h5py.File:
    """
    Load an LVC posterior file (HDF5) and return the opened handle.

    Parameters
    ----------
    path_template:
        Either a full path or a path template to the LVC posterior file.
    event:
        Event name used to format the path when a template string is provided.

    Notes
    -----
    Caller is responsible for closing the returned file handle.
    """
    path = Path(str(path_template).format(event=event))
    logger.info("Loading LVC posterior: %s", path)
    return h5py.File(path, "r")


def _parameter_bounds(samples: Iterable[np.ndarray]) -> tuple[float, float]:
    flat = np.concatenate([np.asarray(s, dtype=float).ravel() for s in samples])
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        raise ValueError("No finite samples found for JS divergence computation.")

    low = float(finite.min())
    high = float(finite.max())
    if np.isclose(low, high):
        padding = (abs(low) if low else 1.0) * 0.01
        low -= padding
        high += padding
    return low, high


def _kde_pmf(
    samples: np.ndarray,
    bounds: tuple[float, float],
    bins: int = DEFAULT_BINS,
) -> np.ndarray:
    """
    Build a discrete probability mass function (PMF) from samples with KDE smoothing.
    """
    bin_edges = np.linspace(bounds[0], bounds[1], bins + 1)
    centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    if samples.size >= 2:
        kde = gaussian_kde(samples)
        density = kde(centers)
        density = np.clip(density, a_min=0.0, a_max=None)
    else:
        density = np.zeros_like(centers)

    if not np.any(density) or not np.isfinite(density).all():
        counts, _ = np.histogram(samples, bins=bins, range=bounds, density=False)
        density = counts.astype(float)

    total = np.sum(density)
    if total <= 0.0:
        raise ValueError("Density evaluation produced zero total probability.")

    pmf = density / total
    if not np.isfinite(pmf).all():
        raise ValueError("Encountered invalid PMF values while computing JS divergence.")
    return pmf


def _extract_samples(
    result,
    param: str,
    lvc_dataset: str | None = None,
) -> np.ndarray:
    """
    Extract samples for a parameter from either a bilby result or an LVC HDF5 dataset.
    """
    if isinstance(result, (h5py.Dataset, h5py.Group)):
        samples = np.asarray(result[param])
    elif isinstance(result, h5py.File):
        if lvc_dataset is None:
            raise ValueError(
                "lvc_dataset must be provided when extracting samples from an HDF5 file."
            )
        samples = np.asarray(result[lvc_dataset][param])
    elif hasattr(result, "posterior"):
        samples = np.asarray(result.posterior[param])
    else:
        raise TypeError(
            "Unsupported result type for sample extraction: "
            f"{type(result).__name__}"
        )

    if samples.size == 0:
        raise ValueError(f"No samples found for parameter '{param}'.")
    return samples


def compute_js(
    reference_samples: np.ndarray,
    target_samples: np.ndarray,
    bounds: tuple[float, float],
    bins: int = DEFAULT_BINS,
) -> float:
    """
    Compute the Jensen-Shannon divergence between two sets of samples.
    """
    ref_pmf = _kde_pmf(reference_samples, bounds, bins=bins)
    target_pmf = _kde_pmf(target_samples, bounds, bins=bins)
    return float(jensenshannon(ref_pmf, target_pmf))


def compute_welch_vs_sgvb(
    params: Sequence[str],
    sgvb_result,
    welch_result,
    bins: int = DEFAULT_BINS,
) -> pd.DataFrame:
    """
    Convenience wrapper to compute SGVB-vs-Welch JS divergences.
    """
    results = {"sgvb": sgvb_result, "welch": welch_result}
    return compute_js_table(params, results, reference="sgvb", bins=bins)


def compute_js_table(
    params: Sequence[str],
    results: Mapping[str, object],
    reference: str,
    bins: int = DEFAULT_BINS,
    lvc_dataset: str | None = None,
) -> pd.DataFrame:
    """
    Compute JS divergences for multiple parameters using one result as reference.

    Parameters
    ----------
    params:
        Iterable of parameter names to evaluate.
    results:
        Mapping from a label (e.g., ``"sgvb"``/``"welch"``) to the corresponding
        result object. Supported result types are bilby results or LVC HDF5 datasets.
    reference:
        Key in ``results`` that should be treated as the reference distribution.
    bins:
        Number of bins to use in the KDE-backed histogram.
    lvc_dataset:
        Dataset name used when ``results`` contains an LVC HDF5 handle.
    """
    if reference not in results:
        raise KeyError(f"Reference '{reference}' not found in results.")

    rows: list[MutableMapping[str, float | str]] = []
    for param in params:
        samples = {
            name: _extract_samples(res, param, lvc_dataset=lvc_dataset)
            for name, res in results.items()
        }
        bounds = _parameter_bounds(samples.values())
        ref_samples = samples[reference]

        row: MutableMapping[str, float | str] = {"parameter": param}
        for name, target in samples.items():
            if name == reference:
                continue
            column = f"js_{reference}_vs_{name}"
            row[column] = compute_js(ref_samples, target, bounds, bins=bins)
        rows.append(row)

    return pd.DataFrame(rows)


def save_js_table(df: pd.DataFrame, path: str | Path) -> None:
    """
    Persist JS divergence results to CSV.
    """
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    logger.info("Saved JS divergence table to %s", output)
