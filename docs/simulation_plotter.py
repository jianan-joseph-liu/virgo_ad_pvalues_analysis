#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import corner
import scipy.stats
from tqdm.auto import tqdm
from bilby.gw.result import CBCResult


# =========================================================
# Helper functions
# =========================================================

def load_results(result_dir):
    """Load SGVB and Welch results as CBCResult objects."""
    sgvb_file = glob.glob(os.path.join(result_dir, "*_sgvb_result.json"))[0]
    welch_file = glob.glob(os.path.join(result_dir, "*_welch_result.json"))[0]
    r_sgvb = CBCResult.from_json(sgvb_file)
    r_welch = CBCResult.from_json(welch_file)
    return {"sgvb": r_sgvb, "welch": r_welch}


def _get_valid_params(result1, result2):
    """Return valid overlapping parameter names."""
    inj = result2.injection_parameters
    shared = (
        set(result1.posterior.columns)
        & set(result2.posterior.columns)
        & set(inj.keys())
    )
    valid = []
    for p in sorted(shared):
        a1, a2 = np.asarray(result1.posterior[p]), np.asarray(result2.posterior[p])
        if np.all(np.isfinite(a1)) and np.all(np.isfinite(a2)) and np.std(a1) > 0 and np.std(a2) > 0:
            valid.append(p)
    if not valid:
        raise RuntimeError("No valid parameters to plot.")
    return valid


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
                print(f"Network SNR from interferometers: {snrs} → {snr:.2f}")
    except Exception as e:
        print(f"⚠️ Could not extract network SNR: {e}")
    return snr


# =========================================================
# Corner + Waveform plotting
# =========================================================

def make_comparison_corner_plot(result_dict, fname):
    """Make manual corner plot overlaying SGVB and Welch results."""
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    r1, r2 = result_dict["sgvb"], result_dict["welch"]
    lnZ1, lnZ2 = r1.log_evidence, r2.log_evidence
    bf = lnZ1 - lnZ2
    snr = _extract_network_snr(r2)

    params = _get_valid_params(r1, r2)
    print(f"Plotting params: {params}")

    inj = r2.injection_parameters
    truths = [inj[p] for p in params]

    samples1 = np.column_stack([np.asarray(r1.posterior[p]) for p in params])
    samples2 = np.column_stack([np.asarray(r2.posterior[p]) for p in params])

    # --- main corner overlay ---
    fig = corner.corner(
        samples1,
        labels=params,
        truths=truths,
        color="#1f77b4",
        truth_color="black",
        hist_kwargs={"density": True},
        smooth=1.0,
        max_n_ticks=3,
        quantiles=[0.05, 0.95],
        show_titles=False,
        label_kwargs={"fontsize": 10},
    )

    corner.corner(
        samples2,
        fig=fig,
        labels=params,
        color="#2ca02c",
        hist_kwargs={"density": True},
        smooth=1.0,
        max_n_ticks=3,
        quantiles=[0.05, 0.95],
        show_titles=False,
        label_kwargs={"fontsize": 10},
    )

    # --- legend handles ---
    h_sgvb = mpatches.Patch(color="#1f77b4", label=f"SGVB ($\\ln Z={lnZ1:.2f}$)")
    h_welch = mpatches.Patch(color="#2ca02c", label=f"Welch ($\\ln Z={lnZ2:.2f}$)")
    h_truth = mlines.Line2D([], [], color="black", marker="x", linestyle="", markersize=6, label="Truth")
    handles = [h_sgvb, h_welch, h_truth]

    ax0 = fig.axes[0]
    ax0.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        fontsize=12,
    )

    # --- textbox with Bayes factor + SNR ---
    fig.text(
        0.98, 0.92,
        f"$\\ln\\mathcal{{B}}_{{SGVB/Welch}}={bf:.2f}$\nSNR(network): {snr:.2f}",
        ha="right",
        va="top",
        fontsize=13,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.6, lw=0),
    )

    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ saved {fname}")


def make_waveform_overlays(result_dict, outdir):
    """Save waveform posterior overlays for each IFO."""
    r = result_dict["sgvb"]
    ifos_attr = r.interferometers

    if isinstance(ifos_attr, dict):
        ifos = list(ifos_attr.keys())
    elif isinstance(ifos_attr, list):
        if len(ifos_attr) > 0 and hasattr(ifos_attr[0], "name"):
            ifos = [ifo.name for ifo in ifos_attr]
        else:
            ifos = [str(ifo) for ifo in ifos_attr]
    else:
        raise TypeError(f"Unexpected interferometers type: {type(ifos_attr)}")

    for ifo in ifos:
        try:
            r.plot_interferometer_waveform_posterior(interferometer=ifo, n_samples=500, save=False)
            plt.title(f"{ifo} waveform posterior")
            outpath = os.path.join(outdir, f"waveform_{ifo}.png")
            plt.savefig(outpath, dpi=200)
            plt.close()
            print(f"✅ saved {outpath}")
        except Exception as e:
            print(f"⚠️ Skipped {ifo}: {e}")


# =========================================================
# P–P plot utilities
# =========================================================

def compute_credible_levels(result_dirs, parameters=None):
    """Compute credible levels for SGVB posteriors vs true injections."""
    result_files = []
    for d in result_dirs:
        matches = glob.glob(os.path.join(d, "*_sgvb_result.json"))
        if matches:
            result_files.append(matches[0])
    if not result_files:
        print("❌ No SGVB results found.")
        return None, None

    first = CBCResult.from_json(result_files[0])
    inj = first.injection_parameters
    post = first.posterior
    if parameters is None:
        parameters = sorted(set(post.columns) & set(inj.keys()))

    credible_levels = np.zeros((len(parameters), len(result_files)))
    for ri, f in enumerate(tqdm(result_files, desc="Computing credible levels")):
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
    """Generate the P–P plot."""
    x_values = np.linspace(0, 1, 1001)
    credible_levels = credible_levels[~np.isnan(credible_levels).any(axis=1)]
    dim, N = credible_levels.shape
    if N <= dim:
        print(f"⚠️ Warning: N={N} ≤ dim={dim}, results may be noisy")

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    grays = plt.cm.Greys(np.linspace(0.2, 0.5, 3))

    # confidence regions
    for i, ci in enumerate([0.68, 0.95, 0.997][::-1]):
        edge = (1 - ci) / 2
        lower = scipy.stats.binom.ppf(edge, N, x_values) / N
        upper = scipy.stats.binom.ppf(1 - edge, N, x_values) / N
        ax.fill_between(x_values, lower, upper, color=grays[i], lw=0)
        ax.plot(x_values, lower, color=grays[i], lw=1)
        ax.plot(x_values, upper, color=grays[i], lw=1)

    # individual parameters
    pvalues = []
    for i in range(len(credible_levels)):
        pp = np.array([(credible_levels[i] < x).mean() for x in x_values])
        pvalue = scipy.stats.kstest(credible_levels[i], "uniform").pvalue
        pvalues.append(pvalue)
        ax.plot(x_values, pp, color=color, alpha=0.3, lw=0.4)

    combined_pvalue = scipy.stats.combine_pvalues(pvalues)[1]
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Credible interval")
    ax.set_ylabel("Fraction within C.I.")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"N={N}, p={combined_pvalue:.4f}")
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


# =========================================================
# Batch processor
# =========================================================

def process_all(result_root, outdir=None, save_inside=True):
    seed_dirs = sorted([d for d in glob.glob(os.path.join(result_root, "seed_*")) if os.path.isdir(d)])
    if not seed_dirs:
        print(f"❌ No seed_* directories found under {result_root}")
        return

    print(f"Found {len(seed_dirs)} result directories.")
    for seed_dir in tqdm(seed_dirs, desc="Processing seeds", unit="seed"):
        seed_name = os.path.basename(seed_dir)
        try:
            results = load_results(seed_dir)
        except Exception as e:
            print(f"⚠️ Skipping {seed_name}: could not load results ({e})")
            continue

        try:
            if save_inside:
                plot_path = os.path.join(seed_dir, f"{seed_name}_comparison.png")
                out_waveform = seed_dir
            else:
                os.makedirs(outdir, exist_ok=True)
                plot_path = os.path.join(outdir, f"{seed_name}_comparison.png")
                out_waveform = outdir

            make_comparison_corner_plot(results, plot_path)
            make_waveform_overlays(results, out_waveform)

        except Exception as e:
            print(f"⚠️ Skipping {seed_name}: error during plotting ({e})")
            continue

    # After all seeds → make PP plot
    print("\n📊 Generating combined P–P plot across all seeds...")
    params, credible_levels = compute_credible_levels(seed_dirs)
    if credible_levels is not None:
        make_pp_plot(credible_levels, filename=os.path.join(result_root, "pp_plot.png"), color="#1f77b4")


# =========================================================
# Main entry
# =========================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python simulation_plotter.py RESULT_ROOT [OUTDIR] [--flat]")
        sys.exit(1)

    result_root = sys.argv[1]
    outdir = None
    save_inside = True
    if len(sys.argv) >= 3:
        outdir = sys.argv[2]
    if "--flat" in sys.argv:
        save_inside = False

    process_all(result_root, outdir=outdir, save_inside=save_inside)
