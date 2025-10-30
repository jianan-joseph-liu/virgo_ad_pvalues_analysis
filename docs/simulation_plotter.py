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


def make_comparison_corner_plot(result_dict: Dict[str, bilby.gw.result.CBCResult], fname: str):
    """
    Compare SGVB vs Welch results using a corner plot with Bayes factors and injection truth.
    """
    result1 = result_dict['sgvb']
    result2 = result_dict['welch']

    ln_Z1 = result1.log_evidence
    ln_Z2 = result2.log_evidence
    Bf_1_vs_2 = ln_Z1 - ln_Z2

    parameters_to_plot = list(result1.posterior.keys())

    color1 = '#1f77b4'  # blue
    color2 = '#2ca02c'  # green

    fig = bilby.result.plot_multiple(
        results=[result1, result2],
        parameters=parameters_to_plot,
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

    legend_label_1 = f'{result1.label} ($\\ln Z = {ln_Z1:.2f}$)'
    legend_label_2 = f'{result2.label} ($\\ln Z = {ln_Z2:.2f}$)'
    Bf_label = f'Bayes Factor (SGVB/Welch): $\\ln \\mathcal{{B}} = {Bf_1_vs_2:.2f}$'

    patch1 = plt.matplotlib.patches.Patch(color=color1, label=legend_label_1)
    patch2 = plt.matplotlib.patches.Patch(color=color2, label=legend_label_2)
    patch_bf = plt.matplotlib.patches.Patch(color='none', label=Bf_label)
    true_marker = plt.Line2D([0], [0], color='black', marker='x', linestyle='', label='True Injection')

    ax = fig.axes[0]
    handles = [patch1, patch2, true_marker, patch_bf]
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(fname)
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
