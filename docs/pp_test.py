#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from tqdm.auto import tqdm
from bilby.gw.result import CBCResult


def compute_credible_levels(result_root, parameters=None, pattern="*_sgvb_result.json"):
    """Compute credible levels (fraction of samples below true value) for each parameter."""
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

    # define parameters from first valid file
    first = CBCResult.from_json(result_files[0])
    inj = first.injection_parameters
    post = first.posterior
    if parameters is None:
        parameters = sorted(set(post.columns) & set(inj.keys()))

    credible_levels = np.zeros((len(parameters), len(result_files)))
    for ri, f in enumerate(tqdm(result_files, desc="Computing credible levels", unit="result")):
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


def make_pp_plot(credible_levels, filename="pp_plot.png", color="#1f77b4"):
    """Generate a P–P plot with binomial confidence regions."""
    x = np.linspace(0, 1, 1001)
    credible_levels = credible_levels[~np.isnan(credible_levels).any(axis=1)]
    dim, N = credible_levels.shape
    if N <= dim:
        print(f"⚠️ Warning: N={N} ≤ dim={dim}")

    fig, ax = plt.subplots(figsize=(4, 4))
    grays = plt.cm.Greys(np.linspace(0.2, 0.5, 3))

    # confidence bands
    for i, ci in enumerate([0.68, 0.95, 0.997][::-1]):
        edge = (1 - ci) / 2
        lower = scipy.stats.binom.ppf(edge, N, x) / N
        upper = scipy.stats.binom.ppf(1 - edge, N, x) / N
        ax.fill_between(x, lower, upper, color=grays[i], lw=0)
        ax.plot(x, lower, color=grays[i], lw=1)
        ax.plot(x, upper, color=grays[i], lw=1)

    # each parameter curve
    pvals = []
    for i in range(len(credible_levels)):
        pp = np.array([(credible_levels[i] < xx).mean() for xx in x])
        p = scipy.stats.kstest(credible_levels[i], "uniform").pvalue
        pvals.append(p)
        ax.plot(x, pp, color=color, alpha=0.3, lw=0.4)

    combined_p = scipy.stats.combine_pvalues(pvals)[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(xlabel="Credible interval", ylabel="Fraction within C.I.", xlim=(0, 1), ylim=(0, 1))
    ax.set_title(f"N={N}, p={combined_p:.3f}")
    ax.legend(
        [
            plt.Line2D([0], [0], color=color, lw=2, label="SGVB"),
            plt.Line2D([0], [0], color=grays[0], lw=2, label="68%"),
            plt.Line2D([0], [0], color=grays[1], lw=2, label="95%"),
            plt.Line2D([0], [0], color=grays[2], lw=2, label="99.7%"),
        ],
        loc="upper left",
        frameon=False,
    )
    fig.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close(fig)
    print(f"✅ saved {filename}")


def main(result_root, out_png="pp_plot.png", pattern="*_sgvb_result.json"):
    params, cred = compute_credible_levels(result_root, pattern=pattern)
    np.save(os.path.join(result_root, "credible_levels.npy"), cred)
    make_pp_plot(cred, filename=os.path.join(result_root, out_png), color="#1f77b4")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pp_plotter.py RESULT_ROOT [OUT_PNG]")
        sys.exit(1)

    result_root = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else "pp_plot.png"
    main(result_root, out_png=out_png)
