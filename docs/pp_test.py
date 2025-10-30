import glob
import os

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tqdm.auto import tqdm

__all__ = ["pp_test"]


def _load_results_and_compute_credible_levels(
    result_regex: str, dim: int = 32
) -> np.ndarray:
    """
    Load posterior samples from NetCDF files, compute credible levels for injected parameters,
    and return an array of credible levels.

    The credible level for a true latent parameter value is computed as the fraction of posterior
    samples that are smaller than the true value. This provides an empirical cumulative
    distribution function (ECDF)-based measure of how well the posterior distribution
    recovers the true injected parameters.

    Parameters:
    result_regex (str): A regex pattern to locate result files.
    dim (int): The number of latent dimensions in the data.

    Returns:
    np.ndarray: A (dim, N) array containing the credible levels for each parameter and each file.
    """
    result_files = glob.glob(result_regex)
    assert (
        len(result_files) > 0
    ), f"No files found with pattern {result_regex}."
    credible_levels = np.zeros((dim, len(result_files)))
    for ri, result_file in tqdm(
        enumerate(result_files), total=len(result_files)
    ):
        try:
            idata = az.from_netcdf(result_file)
            zpost = idata.posterior["z"].stack(sample=("chain", "draw")).values
            ztrue = idata.sample_stats["true_latent"].to_numpy().ravel()
            for i, _ztrue in enumerate(ztrue):
                posterior_samples = zpost[i].flatten()
                credible_levels[i, ri] = np.mean(posterior_samples < _ztrue)
        except Exception as e:
            print(f"Error processing {result_file}: {e}")
            credible_levels[:, ri] = np.nan
    return credible_levels


def _make_pp_plot(
    credible_levels: np.ndarray,
    filename="pp.png",
    confidence_interval=[0.68, 0.95, 0.997],
    title=True,
    color="tab:orange",
    **kwargs,
):
    """
    Generate a probability-probability (P-P) plot comparing empirical credible levels
    to expected uniform behavior.

    The plot is a diagnostic tool for checking whether the posterior credible levels
    are uniformly distributed, which is expected if the inference is well-calibrated.

    Confidence intervals are computed using a binomial cumulative distribution function.
    Each credible level distribution is tested against a uniform distribution using
    the Kolmogorov-Smirnov test.

    Parameters:
    credible_levels (np.ndarray): An array of credible levels from the posterior samples.
    filename (str): The output filename for the P-P plot.
    confidence_interval (list of float): Confidence intervals for shading.
    title (bool): Whether to include a title on the plot.
    confidence_interval_alpha (float): Alpha value for confidence shading.

    Returns:
    Tuple[matplotlib.figure.Figure, dict]: A matplotlib figure and a dictionary of p-values.
    """
    x_values = np.linspace(0, 1, 1001)
    credible_levels = credible_levels[~np.isnan(credible_levels).any(axis=1)]
    dim, N = credible_levels.shape
    assert (
        N > dim
    ), f"Number of results {N} must be greater than the number of dim {dim}."

    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    grays = plt.cm.Greys(np.linspace(0.2, 0.5, len(confidence_interval)))

    # Confidence interval shading
    for i, ci in enumerate(confidence_interval[::-1]):
        edge = (1 - ci) / 2
        lower = scipy.stats.binom.ppf(edge, N, x_values) / N
        upper = scipy.stats.binom.ppf(1 - edge, N, x_values) / N
        ax.fill_between(
            x_values, lower, upper, color=grays[i], lw=0, zorder=10
        )
        # draw line at theses values
        ax.plot(x_values, lower, color=grays[i], lw=1, zorder=10)
        ax.plot(x_values, upper, color=grays[i], lw=1, zorder=10)

    pvalues = []
    for i in range(len(credible_levels)):
        pp = np.array([(credible_levels[i] < x).mean() for x in x_values])
        pvalue = scipy.stats.kstest(credible_levels[i], "uniform").pvalue
        pvalues.append(pvalue)
        print(f"{i}: {pvalue}")
        ax.plot(
            x_values, pp, **kwargs, color=color, alpha=0.3, lw=0.3, zorder=110
        )

    combined_pvalue = scipy.stats.combine_pvalues(pvalues)[1]
    print(f"Combined p-value: {combined_pvalue}")

    if title:
        ax.set_title(f"N = {N}, p-value={combined_pvalue:.4f}")
    ax.set_xlabel("C.I.")
    ax.set_ylabel("Fraction of events in C.I.")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=grays[0], lw=2, label="68%"),
            plt.Line2D([0], [0], color=grays[1], lw=2, label="95%"),
            plt.Line2D([0], [0], color=grays[2], lw=2, label="99.7%"),
            plt.Line2D(
                [0], [0], color=color, lw=2, label=r"$\rm{CDF}_z(\rm{C.I.})$"
            ),
        ],
        loc="upper left",
        frameon=False,
    )
    fig.tight_layout()
    plt.savefig(filename, dpi=500)

    return fig, {"combined_pvalue": combined_pvalue, "pvalues": pvalues}


def pp_test(
    credible_levels_fpath: str,
    plot_fname: str,
    result_regex: str = None,
    include_title: bool = True,
    dim: int = 32,
):
    """
    Perform a P-P test by loading posterior samples, computing credible levels,
    and generating a P-P plot for model validation.

    The function first checks whether the credible levels have already been computed and saved.
    If not, it computes them from the result files. Then, it generates a P-P plot and saves it.

    Parameters:
    result_regex (str): Regex pattern to locate NetCDF result files.
    credible_levels_fpath (str): Path to save or load credible levels (must be .npy).
    plot_fname (str): Path to save the P-P plot (must be .png or .pdf).
    include_title (bool): Whether to include a title in the P-P plot.

    Returns:
    None
    """
    assert credible_levels_fpath.endswith(".npy")
    assert plot_fname.endswith(".png") or plot_fname.endswith(".pdf")

    if not os.path.exists(credible_levels_fpath):
        cred_level = _load_results_and_compute_credible_levels(
            result_regex=result_regex, dim=dim
        )
        np.save(credible_levels_fpath, cred_level)
    cred_level = np.load(credible_levels_fpath)
    _make_pp_plot(cred_level, filename=plot_fname, title=include_title)
