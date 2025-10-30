import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from bilby.gw.result import CBCResult
import corner


def load_results(result_dir):
    """Load SGVB and Welch results as CBCResult objects."""
    sgvb_file = glob.glob(os.path.join(result_dir, "*_sgvb_result.json"))[0]
    welch_file = glob.glob(os.path.join(result_dir, "*_welch_result.json"))[0]

    r_sgvb = CBCResult.from_json(sgvb_file)
    r_welch = CBCResult.from_json(welch_file)

    return {"sgvb": r_sgvb, "welch": r_welch}


def _get_valid_params(result1, result2):
    """
    Pick parameters:
    - in both posteriors
    - present in injection dict
    - finite
    - non-zero variance
    """
    inj = result2.injection_parameters

    shared = (
        set(result1.posterior.columns)
        & set(result2.posterior.columns)
        & set(inj.keys())
    )

    valid = []
    for p in shared:
        a1 = np.asarray(result1.posterior[p])
        a2 = np.asarray(result2.posterior[p])
        if (
            np.all(np.isfinite(a1))
            and np.all(np.isfinite(a2))
            and np.std(a1) > 0
            and np.std(a2) > 0
        ):
            valid.append(p)

    valid = sorted(valid)
    if not valid:
        raise RuntimeError("No valid parameters to plot")

    return valid


def make_comparison_corner_plot(result_dict, fname):
    r1 = result_dict["sgvb"]
    r2 = result_dict["welch"]

    lnZ1 = r1.log_evidence
    lnZ2 = r2.log_evidence
    bf = lnZ1 - lnZ2
    snr = r2.injection_parameters.get("optimal_snr", np.nan)

    params = _get_valid_params(r1, r2)
    print(f"Plotting params: {params}")

    inj = r2.injection_parameters
    truths = [inj[p] for p in params]

    samples1 = np.column_stack([np.asarray(r1.posterior[p]) for p in params])
    samples2 = np.column_stack([np.asarray(r2.posterior[p]) for p in params])

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

    # --- Add floating textbox (instead of legend) ---
    ax0 = fig.axes[0]
    textstr = (
        f"SGVB: $\\ln Z={lnZ1:.2f}$\n"
        f"Welch: $\\ln Z={lnZ2:.2f}$\n"
        f"$\\ln\\mathcal{{B}}_{{\\mathrm{{SGVB/Welch}}}}={bf:.2f}$\n"
        f"Signal SNR: {snr:.2f}"
    )

    # Add textbox to top-right corner
    ax0.text(
        1.02, 1.05, textstr,
        transform=ax0.transAxes,
        fontsize=13,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.0, lw=0),
    )

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"✅ saved {fname}")


def make_waveform_overlays(result_dict, outdir):
    """Save waveform posterior overlays for each IFO."""
    r = result_dict["sgvb"]
    ifos = list(r.interferometers.keys())

    for ifo in ifos:
        r.plot_interferometer_waveform_posterior(
            interferometer=ifo,
            n_samples=500,
            save=False,
        )
        plt.title(f"{ifo} waveform posterior")
        outpath = os.path.join(outdir, f"waveform_{ifo}.png")
        plt.savefig(outpath, dpi=200)
        plt.close()
        print(f"✅ saved {outpath}")


if __name__ == "__main__":
    # Usage: python compare_results.py OUTDIR RESULT_DIR
    outdir = sys.argv[1]
    result_dir = sys.argv[2]

    os.makedirs(outdir, exist_ok=True)

    results = load_results(result_dir)

    make_comparison_corner_plot(results, os.path.join(outdir, "comparison_corner.png"))
    make_waveform_overlays(results, outdir)
