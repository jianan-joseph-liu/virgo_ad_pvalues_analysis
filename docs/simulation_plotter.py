import os
import sys
import glob
import shutil
import numpy as np
import bilby
import corner
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from typing import Dict


def load_results(result_dir: str) -> Dict[str, bilby.gw.result.CBCResult]:
    """
    Load Bilby result files for SGVB and Welch analyses.
    """
    sgvb_files = glob.glob(os.path.join(result_dir, '*_sgvb_result.json'))
    welch_files = glob.glob(os.path.join(result_dir, '*_welch_result.json'))

    if not sgvb_files or not welch_files:
        raise FileNotFoundError(f"Could not find both SGVB and Welch result files in directory. {result_dir}")

    sgvb_result = bilby.result.read_in_result(sgvb_files[0])
    welch_result = bilby.result.read_in_result(welch_files[0])

    return dict(sgvb=sgvb_result, welch=welch_result)


def make_comparison_corner_plot(result_dict, fname):
    result1 = result_dict['sgvb']
    result2 = result_dict['welch']

    # --- Select parameters that appear in both posteriors and injection dict ---
    inj_params = result2.injection_parameters  # Usually both results share same injection
    shared_params = set(result1.posterior.columns) & set(result2.posterior.columns) & set(inj_params.keys())

    # Filter only those that are finite and have variation
    valid_params = []
    for p in shared_params:
        arr1 = np.asarray(result1.posterior[p])
        arr2 = np.asarray(result2.posterior[p])
        if (
            np.all(np.isfinite(arr1)) and np.all(np.isfinite(arr2))
            and np.std(arr1) > 0 and np.std(arr2) > 0
        ):
            valid_params.append(p)

    if not valid_params:
        raise RuntimeError("No valid parameters to plot!")

    valid_params = sorted(valid_params)
    print(f"✅ Plotting {len(valid_params)} parameters with injections:")
    print(valid_params)

    # --- Compute evidences and Bayes factor ---
    ln_Z1 = result1.log_evidence
    ln_Z2 = result2.log_evidence
    Bf_1_vs_2 = ln_Z1 - ln_Z2

    color1 = '#1f77b4'
    color2 = '#2ca02c'

    # --- Corner plot ---
    fig = bilby.result.plot_multiple(
        results=[result1, result2],
        parameters=valid_params,
        priors=True,
        plot_injection=True,
        labels=['SGVB', 'Welch'],
        colour_list=[color1, color2],
        figsize=(8, 8),
        corner_kwargs={
            'hist_kwargs': {'density': True},
            'smooth': 1.0,
            'max_n_ticks': 3,
            'quantiles': [0.05, 0.95]
        }
    )

    # --- Legend ---
    legend_label_1 = f'{result1.label} ($\\ln Z = {ln_Z1:.2f}$)'
    legend_label_2 = f'{result2.label} ($\\ln Z = {ln_Z2:.2f}$)'
    Bf_label = f'Bayes Factor (SGVB/Welch): $\\ln \\mathcal{{B}} = {Bf_1_vs_2:.2f}$'

    patch1 = plt.matplotlib.patches.Patch(color=color1, label=legend_label_1)
    patch2 = plt.matplotlib.patches.Patch(color=color2, label=legend_label_2)
    patch_bf = plt.matplotlib.patches.Patch(color='none', label=Bf_label)
    true_marker = plt.Line2D([0], [0], color='black', marker='x', linestyle='', label='True Injection')

    ax = fig.axes[0]
    ax.legend(
        handles=[patch1, patch2, true_marker, patch_bf],
        loc='upper right',
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        fontsize=10
    )

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    print(f"✅ Saved comparison corner plot to: {fname}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py <output_dir> <result_dir>")
        sys.exit(1)

    outdir = sys.argv[1]
    result_dir = sys.argv[2]

    os.makedirs(outdir, exist_ok=True)
    results = load_results(result_dir)
    make_comparison_corner_plot(results, os.path.join(outdir, "comparison_corner.png"))
