#!/usr/bin/env python3
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from bilby.gw.result import CBCResult
import bilby


def load_results(result_dir):
    """Load SGVB and Welch results as CBCResult objects."""
    sgvb_file = glob.glob(os.path.join(result_dir, "*_sgvb_result.json"))[0]
    welch_file = glob.glob(os.path.join(result_dir, "*_welch_result.json"))[0]

    r_sgvb = CBCResult.from_json(sgvb_file)
    r_welch = CBCResult.from_json(welch_file)

    return {"sgvb": r_sgvb, "welch": r_welch}


def _get_valid_params(result1, result2):
    """Return parameters in both posteriors & injection dict with finite, non-constant samples."""
    inj = result2.injection_parameters
    shared = (
        set(result1.posterior.columns)
        & set(result2.posterior.columns)
        & set(inj.keys())
    )

    valid = []
    for p in shared:
        a1, a2 = np.asarray(result1.posterior[p]), np.asarray(result2.posterior[p])
        if np.all(np.isfinite(a1)) and np.all(np.isfinite(a2)) and np.std(a1) > 0 and np.std(a2) > 0:
            valid.append(p)

    valid.sort()
    if not valid:
        raise RuntimeError("No valid parameters to plot")
    return valid


def make_comparison_corner_plot(result_dict, fname):
    """Overlay SGVB vs Welch corner plots using Bilby's plot_multiple."""
    r1, r2 = result_dict["sgvb"], result_dict["welch"]
    lnZ1, lnZ2 = r1.log_evidence, r2.log_evidence
    bf = lnZ1 - lnZ2
    snr = r2.injection_parameters.get("optimal_snr", np.nan)

    params = _get_valid_params(r1, r2)
    print(f"Plotting params: {params}")

    inj = r2.injection_parameters
    truths = [inj[p] for p in params]

    # --- Use bilby.result.plot_multiple (no auto-save) ---
    fig = bilby.result.plot_multiple(
        results=[r1, r2],
        parameters=params,
        priors=True,
        evidences=False,
        save=False,  # don't auto-save
        plot_injection=False,  # we'll add truths manually
        labels=["SGVB", "Welch"],
        colours=["#1f77b4", "#2ca02c"],
        corner_labels=params,
        corner_kwargs={
            "smooth": 1.0,
            "max_n_ticks": 3,
            "quantiles": [0.05, 0.95],
            "hist_kwargs": {"density": True},
        },
    )

    # --- Overlay manual truth lines ---
    for ax in fig.axes:
        if ax.get_xlabel() in inj:
            ax.axvline(inj[ax.get_xlabel()], color="black", ls="--", lw=1)
        if ax.get_ylabel() in inj:
            ax.axhline(inj[ax.get_ylabel()], color="black", ls="--", lw=1)

    # --- Add global textbox (figure coords) ---
    text = (
        f"SGVB: $\\ln Z={lnZ1:.2f}$\n"
        f"Welch: $\\ln Z={lnZ2:.2f}$\n"
        f"$\\ln\\mathcal{{B}}_{{SGVB/Welch}}={bf:.2f}$\n"
        f"SNR: {snr:.2f}"
    )
    fig.text(
        0.98, 0.98, text,
        ha="right", va="top",
        fontsize=14,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.6, lw=0),
    )

    plt.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ saved {fname}")


def make_waveform_overlays(result_dict, outdir):
    """Save waveform posterior overlays for each interferometer."""
    r = result_dict["sgvb"]

    # Handle dict or list storage
    if isinstance(r.interferometers, dict):
        ifos = list(r.interferometers.keys())
    elif isinstance(r.interferometers, list):
        ifos = [ifo.name for ifo in r.interferometers]
    else:
        raise TypeError(f"Unexpected interferometers type: {type(r.interferometers)}")

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


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python compare_results.py OUTDIR RESULT_DIR")
        sys.exit(1)

    outdir, result_dir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)

    results = load_results(result_dir)
    make_comparison_corner_plot(results, os.path.join(outdir, "comparison_corner.png"))
    make_waveform_overlays(results, outdir)
