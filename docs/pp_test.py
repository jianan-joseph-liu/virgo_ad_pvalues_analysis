#!/usr/bin/env python3
"""
PP plot generator for SGVB and Welch inference results.

For each seed_i directory, reads *_sgvb_result.json and *_welch_result.json,
computes credible levels for injected parameters, and plots both models'
calibration (CDF of credible intervals).
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from tqdm.auto import tqdm
from bilby.gw.result import CBCResult


# =========================================================
# Core: credible level computation
# =========================================================

def compute_credible_levels(result_root, pattern, parameters=None):
    """Compute credible levels (fraction of samples < injected value) across all seeds."""
    seed_dirs = sorted([d for d in glob.glob(os.path.join(result_root, "seed_*")) if os.path.isdir(d)])
    if not seed_dirs:
        raise FileNotFoundError(f"No seed_* directories found under {result_root}")

    result_files = []
    for d in seed_dirs:
        matches = glob.glob(os.path.join(d, pattern))
        if matches:
            result_files.append(matches[0])
    if not result_files:
        raise FileNotFoundError(f"No files found matching pattern {pattern}")

    # Parameters from first file
    first = CBCResult.from_json(result_files[0])
    inj = first.injection_parameters
    post = first.posterior
    if parameters is None:
        parameters = sorted(set(post.columns) & set(inj.keys()))

    credible_levels = np.zeros((len(parameters), len(result_files)))
    for ri, f in enumerate(tqdm(result_files, desc=f"Computing credible levels for {pattern}", unit="result")):
        try:
            res = CBCResult.from_json(f)
            inj = res.injection_parameters
            post = res.posterior
            for pi, p in enumerate(parameters):
                if p not in post.columns or p not in inj:
                    credible_levels[pi, ri] = np.nan
                    continue
                samples = np.asarray(post[p])
                true_val = inj[p]
                credible_levels[pi, ri] = np.mean(samples < true_val)
        except Exception as e:
            print(f"⚠️ Skipped {f}: {e}")
            credible_levels[:, ri] = np.nan
    return np.array(parameters), credible_levels


# =========================================================
# Plotting: overlay SGVB + Welch PP curves
# =========================================================

def make_pp_plot_dual(cred_sgvb, cred_welch, filename="pp_plot.png"):
    """Make P–P plot comparing SGVB (blue) vs Welch (green)."""
    x = np.linspace(0, 1, 1001)

    def plot_one(ax, credible_levels, color, label):
        credible_levels = credible_levels[~np.isnan(credible_levels).any(axis=1)]
        if credible_levels.size == 0:
            print(f"⚠️ No valid data for {label}")
            return
        dim, N = credible_levels.shape
        print(f"{label}: {N} results, {dim} parameters")
        pvals = []
        for i in range(len(credible_levels)):
            pp = np.array([(credible_levels[i] < xx).mean() for xx in x])
            p = scipy.stats.kstest(credible_levels[i], "uniform").pvalue
            pvals.append(p)
            ax.plot(x, pp, color=color, alpha=0.6, lw=0.8, label=None)
        pvals = np.clip(pvals, 1e-300, 1.0)
        combined_p = scipy.stats.combine_pvalues(pvals)[1]
        ax.plot([0, 1], [0, 1], "k--", lw=1, zorder=1)
        print(f"{label}: combined p = {combined_p:.3g}")
        return combined_p

    # --- Plot setup ---
    fig, ax = plt.subplots(figsize=(4, 4))
    grays = plt.cm.Greys(np.linspace(0.2, 0.5, 3))

    # Confidence regions
    for i, ci in enumerate([0.68, 0.95, 0.997][::-1]):
        edge = (1 - ci) / 2
        lower = scipy.stats.binom.ppf(edge, cred_sgvb.shape[1], x) / cred_sgvb.shape[1]
        upper = scipy.stats.binom.ppf(1 - edge, cred_sgvb.shape[1], x) / cred_sgvb.shape[1]
        ax.fill_between(x, lower, upper, color=grays[i], lw=0)
        ax.plot(x, lower, color=grays[i], lw=1)
        ax.plot(x, upper, color=grays[i], lw=1)

    # Plot both datasets
    p_sgvb = plot_one(ax, cred_sgvb, color="#1f77b4", label="SGVB")
    p_welch = plot_one(ax, cred_welch, color="#2ca02c", label="Welch")

    # Final styling
    ax.set_xlabel("Credible interval")
    ax.set_ylabel("Fraction within C.I.")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Posterior calibration (SGVB vs Welch)")
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color="#1f77b4", lw=2, label=f"SGVB (p={p_sgvb:.3f})"),
            plt.Line2D([0], [0], color="#2ca02c", lw=2, label=f"Welch (p={p_welch:.3f})"),
            plt.Line2D([0], [0], color=grays[0], lw=2, label="68%"),
            plt.Line2D([0], [0], color=grays[1], lw=2, label="95%"),
            plt.Line2D([0], [0], color=grays[2], lw=2, label="99.7%"),
        ],
        loc="upper left",
        frameon=False,
        fontsize=10,
    )

    fig.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close(fig)
    print(f"✅ saved {filename}")


# =========================================================
# Main driver
# =========================================================

def main(result_root, out_png="pp_plot_dual.png"):
    print(f"📊 Computing P–P calibration for {result_root}")

    params_sgvb, cred_sgvb = compute_credible_levels(result_root, pattern="*_sgvb_result.json")
    params_welch, cred_welch = compute_credible_levels(result_root, pattern="*_welch_result.json", parameters=params_sgvb)

    # Save numeric arrays for reuse
    np.save(os.path.join(result_root, "credible_levels_sgvb.npy"), cred_sgvb)
    np.save(os.path.join(result_root, "credible_levels_welch.npy"), cred_welch)

    make_pp_plot_dual(cred_sgvb, cred_welch, filename=os.path.join(result_root, out_png))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pp_plotter_dual.py RESULT_ROOT [OUT_PNG]")
        sys.exit(1)

    result_root = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else "pp_plot_dual.png"
    main(result_root, out_png=out_png)
