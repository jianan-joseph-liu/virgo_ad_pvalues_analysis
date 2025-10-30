def make_comparison_corner_plot(result_dict, fname):
    """Overlay SGVB vs Welch corner plots using Bilby's plot_multiple."""
    r1, r2 = result_dict["sgvb"], result_dict["welch"]
    lnZ1, lnZ2 = r1.log_evidence, r2.log_evidence
    bf = lnZ1 - lnZ2
    snr = r2.injection_parameters.get("optimal_snr", np.nan)

    params = _get_valid_params(r1, r2)
    print(f"Plotting params: {params}")

    # Bilby stores columns with hyphens instead of underscores
    hyphen_params = [p.replace("_", "-") for p in params]

    inj = r2.injection_parameters
    truths = [inj[p] for p in params]

    fig = bilby.result.plot_multiple(
        results=[r1, r2],
        parameters=hyphen_params,  # use hyphenated keys for posterior
        priors=True,
        evidences=False,
        save=False,
        plot_injection=False,
        labels=["SGVB", "Welch"],
        colours=["#1f77b4", "#2ca02c"],
        corner_labels=params,  # keep nice underscore labels on axes
        corner_kwargs={
            "smooth": 1.0,
            "max_n_ticks": 3,
            "quantiles": [0.05, 0.95],
            "hist_kwargs": {"density": True},
        },
    )

    # --- Overlay manual truth lines ---
    for ax in fig.axes:
        if ax.get_xlabel().replace("-", "_") in inj:
            ax.axvline(inj[ax.get_xlabel().replace("-", "_")], color="black", ls="--", lw=1)
        if ax.get_ylabel().replace("-", "_") in inj:
            ax.axhline(inj[ax.get_ylabel().replace("-", "_")], color="black", ls="--", lw=1)

    # --- Add global textbox ---
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
