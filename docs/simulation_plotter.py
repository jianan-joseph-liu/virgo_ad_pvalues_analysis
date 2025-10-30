#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import corner
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
# Plotting functions
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

    # Add legend in top-right of corner plot
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

    # Handle dict, list of Interferometer objects, or list of names
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
# Batch driver with TQDM
# =========================================================

def process_all(result_root, outdir=None, save_inside=True):
    """
    Loop over all result directories (e.g. seed_1, seed_2) under result_root.

    Parameters
    ----------
    result_root : str
        Directory containing subfolders with Bilby results.
    outdir : str or None
        If given and save_inside=False, save all plots here.
    save_inside : bool
        If True, saves plots inside each seed directory (default).
    """
    seed_dirs = sorted([d for d in glob.glob(os.path.join(result_root, "seed_*")) if os.path.isdir(d)])
    if not seed_dirs:
        print(f"❌ No seed_* directories found under {result_root}")
        return

    print(f"Found {len(seed_dirs)} result directories.")

    for seed_dir in tqdm(seed_dirs, desc="Processing seeds", unit="seed"):
        seed_name = os.path.basename(seed_dir)
        results = load_results(seed_dir)

        if save_inside:
            plot_path = os.path.join(seed_dir, f"{seed_name}_comparison.png")
            out_waveform = seed_dir
        else:
            os.makedirs(outdir, exist_ok=True)
            plot_path = os.path.join(outdir, f"{seed_name}_comparison.png")
            out_waveform = outdir

        make_comparison_corner_plot(results, plot_path)
        make_waveform_overlays(results, out_waveform)


# =========================================================
# Main entry point
# =========================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python simulation_plotter.py RESULT_ROOT [OUTDIR] [--flat]")
        print("Examples:")
        print("  python simulation_plotter.py out/")
        print("  python simulation_plotter.py out/ plots/ --flat")
        sys.exit(1)

    result_root = sys.argv[1]
    outdir = None
    save_inside = True

    if len(sys.argv) >= 3:
        outdir = sys.argv[2]
    if "--flat" in sys.argv:
        save_inside = False

    process_all(result_root, outdir=outdir, save_inside=save_inside)
