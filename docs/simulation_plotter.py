import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import bilby
from bilby.gw.result import CBCResult


def load_results(result_dir):
    sgvb_file = glob.glob(os.path.join(result_dir, "*_sgvb_result.json"))[0]
    welch_file = glob.glob(os.path.join(result_dir, "*_welch_result.json"))[0]
    r_sgvb = bilby.result.read_in_result(sgvb_file)
    r_welch = bilby.result.read_in_result(welch_file)
    return {"sgvb": r_sgvb, "welch": r_welch}


def _get_valid_params(result1, result2):
    """
    Pick parameters:
    - in both posteriors
    - present in injection dict
    - finite
    - non-zero variance
    """
    inj = result2.true_injection_parameters

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

    params = _get_valid_params(r1, r2)
    print(f"Plotting params: {params}")

    inj = r2.injection_parameters
    truths = [inj[p] for p in params]

    # We'll plot SGVB and Welch as two overlaid corner plots:
    import corner

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

    # Build legend
    lab1 = f"{r1.label} ($\\ln Z={lnZ1:.2f}$)"
    lab2 = f"{r2.label} ($\\ln Z={lnZ2:.2f}$)"
    lab3 = f"$\\ln\\mathcal{{B}}_{{\\mathrm{{SGVB/Welch}}}}={bf:.2f}$"

    handles = [
        plt.matplotlib.patches.Patch(color="#1f77b4", label=lab1),
        plt.matplotlib.patches.Patch(color="#2ca02c", label=lab2),
        plt.Line2D([0], [0], color="black", ls="-", marker="o", markersize=4,
                   markerfacecolor="black", label="truth"),
        plt.matplotlib.patches.Patch(color="none", label=lab3),
    ]

    # "Top right" of the first axis in the grid = use the first axis
    ax0 = fig.axes[0]
    ax0.legend(
        handles=handles,
        loc="upper right",
        frameon=False,   # <-- no frame
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"✅ saved {fname}")


def make_waveform_overlays(result_dict, outdir):
    """
    For each IFO, save waveform posterior overlay.
    Uses Bilby's plot_interferometer_waveform_posterior with save=False.
    """
    r = result_dict["sgvb"]  # choose whichever sampler you trust for waveform posterior
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
    # usage: python compare_results.py OUTDIR RESULT_DIR
    outdir = sys.argv[1]
    result_dir = sys.argv[2]

    os.makedirs(outdir, exist_ok=True)

    results = load_results(result_dir)

    # corner-style PE comparison with truths, legend, Bayes factor
    make_comparison_corner_plot(results, os.path.join(outdir, "comparison_corner.png"))

    # interferometer waveform posterior plots saved alongside
    make_waveform_overlays(results, outdir)
