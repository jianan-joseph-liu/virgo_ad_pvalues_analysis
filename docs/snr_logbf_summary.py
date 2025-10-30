#!/usr/bin/env python3
"""
Summarize SNR and log Bayes Factor (SGVB vs Welch) distributions across all seeds.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from bilby.gw.result import CBCResult


# =========================================================
# Helpers
# =========================================================

def _extract_network_snr(result):
    """Compute network SNR = sqrt(sum_i SNR_i^2) from meta_data."""
    snr = np.nan
    try:
        snrs = []
        if (
            hasattr(result, "meta_data")
            and "likelihood" in result.meta_data
            and "interferometers" in result.meta_data["likelihood"]
        ):
            for ifo, info in result.meta_data["likelihood"]["interferometers"].items():
                if isinstance(info, dict) and "optimal_SNR" in info:
                    snrs.append(float(info["optimal_SNR"]))
            if snrs:
                snr = np.sqrt(np.sum(np.square(snrs)))
    except Exception as e:
        print(f"⚠️ Could not extract SNR: {e}")
    return snr


def load_snr_and_bf(result_root):
    """Extract network SNRs (from Welch results) and log Bayes Factors (SGVB vs Welch)."""
    seed_dirs = sorted(
        [d for d in glob.glob(os.path.join(result_root, "seed_*")) if os.path.isdir(d)]
    )
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found under {result_root}")

    snrs, bfs = [], []

    for seed_dir in tqdm(seed_dirs, desc="Extracting SNRs & logBFs", unit="seed"):
        try:
            sgvb_file = glob.glob(os.path.join(seed_dir, "*_sgvb_result.json"))[0]
            welch_file = glob.glob(os.path.join(seed_dir, "*_welch_result.json"))[0]
        except IndexError:
            print(f"⚠️ Skipping {seed_dir}: missing result files")
            continue

        try:
            r_sgvb = CBCResult.from_json(sgvb_file)
            r_welch = CBCResult.from_json(welch_file)

            lnZ_sgvb = r_sgvb.log_evidence
            lnZ_welch = r_welch.log_evidence
            bf = lnZ_sgvb - lnZ_welch
            bfs.append(bf)

            snr = _extract_network_snr(r_welch)
            snrs.append(snr)

        except Exception as e:
            print(f"⚠️ Skipping {seed_dir}: {e}")
            continue

    snrs = np.array(snrs)
    bfs = np.array(bfs)
    print(f"✅ Extracted {len(snrs)} SNRs and {len(bfs)} log Bayes factors")
    return snrs, bfs


# =========================================================
# Plotting
# =========================================================

def plot_snr_logbf(snrs, bfs, filename="snr_logbf_summary.png"):
    """Plot histograms of SNR and log Bayes Factors side-by-side."""
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # SNR histogram
    axs[0].hist(snrs[~np.isnan(snrs)], bins=20, color="#1f77b4", alpha=0.7, edgecolor="black")
    axs[0].set_xlabel("Network SNR")
    axs[0].set_ylabel("Count")
    axs[0].set_title("Distribution of network SNRs")

    # Log Bayes Factor histogram
    axs[1].hist(bfs[~np.isnan(bfs)], bins=20, color="#2ca02c", alpha=0.7, edgecolor="black")
    axs[1].axvline(0, color="black", linestyle="--", lw=1)
    axs[1].set_xlabel(r"$\ln \mathcal{B}_{\mathrm{SGVB/Welch}}$")
    axs[1].set_title("Distribution of log Bayes Factors")

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"✅ saved {filename}")


# =========================================================
# Main
# =========================================================

def main(result_root, out_png="snr_logbf_summary.png"):
    snrs, bfs = load_snr_and_bf(result_root)
    np.save(os.path.join(result_root, "snrs.npy"), snrs)
    np.save(os.path.join(result_root, "log_bayes_factors.npy"), bfs)
    plot_snr_logbf(snrs, bfs, filename=os.path.join(result_root, out_png))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python snr_logbf_summary.py RESULT_ROOT [OUT_PNG]")
        sys.exit(1)

    result_root = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else "snr_logbf_summary.png"
    main(result_root, out_png=out_png)
