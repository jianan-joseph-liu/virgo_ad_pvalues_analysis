"""
Script to help with plotting of bilby results 
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import bilby
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ResultBundle:
    """Container for a Bilby result and useful derived quantities."""

    label: str
    path: Path
    result: bilby.core.result.Result
    optimal_snr: float | None
    matched_filter_snr: complex | None

    @property
    def log_evidence(self) -> float:
        return float(self.result.log_evidence)

    @property
    def abs_matched_filter_snr(self) -> float | None:
        return abs(self.matched_filter_snr) if self.matched_filter_snr is not None else None

    @property
    def log_likelihood_samples(self) -> pd.Series:
        return self.result.posterior["log_likelihood"]

    @property
    def posterior_size(self) -> int:
        return len(self.result.posterior)


def load_result(label: str, path: Path) -> ResultBundle:
    result = bilby.result.read_in_result(str(path))
    interferometers = result.meta_data.get("likelihood", {}).get("interferometers", {})
    optimal_snr = None
    matched_filter_snr = None
    for ifo in interferometers.values():
        optimal_snr = ifo.get("optimal_SNR")
        matched_filter_snr = ifo.get("matched_filter_SNR")
        break
    return ResultBundle(
        label=label,
        path=path,
        result=result,
        optimal_snr=optimal_snr,
        matched_filter_snr=matched_filter_snr,
    )


def format_snr(label: str, bundle: ResultBundle) -> str:
    components = [label]
    if bundle.optimal_snr is not None:
        components.append(f"opt SNR={bundle.optimal_snr:.2f}")
    if bundle.abs_matched_filter_snr is not None:
        components.append(f"|mf SNR|={bundle.abs_matched_filter_snr:.2f}")
    components.append(f"samples={bundle.posterior_size}")
    return " • ".join(components)


def _apply_suptitle(fig: plt.Figure, suffix: str | None) -> None:
    if suffix:
        fig.suptitle(suffix, fontsize=10)


def plot_loglikelihood_trace(bundles: Iterable[ResultBundle], title_suffix: str | None = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for bundle in bundles:
        y = bundle.log_likelihood_samples.to_numpy()
        x = np.arange(bundle.posterior_size)
        ax.plot(x, y, linewidth=1, alpha=0.8, label=format_snr(bundle.label, bundle))
    ax.set_title("Log-likelihood trace")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("log L")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize="small")
    _apply_suptitle(fig, title_suffix)
    return fig


def plot_loglikelihood_histogram(
    bundles: Iterable[ResultBundle], title_suffix: str | None = None
):
    fig, ax = plt.subplots(figsize=(9, 5))
    for bundle in bundles:
        values = bundle.log_likelihood_samples.to_numpy()
        ax.hist(
            values,
            bins=50,
            histtype="step",
            density=True,
            linewidth=1.5,
            label=format_snr(bundle.label, bundle),
        )
    ax.set_title("Log-likelihood distribution")
    ax.set_xlabel("log L")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize="small")
    _apply_suptitle(fig, title_suffix)
    return fig


def plot_loglikelihood_rank(bundles: Iterable[ResultBundle], title_suffix: str | None = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for bundle in bundles:
        values = np.sort(bundle.log_likelihood_samples.to_numpy())
        ranks = np.linspace(0, 1, len(values), endpoint=False)
        ax.plot(ranks, values, linewidth=1.5, label=format_snr(bundle.label, bundle))
    ax.set_title("Log-likelihood quantile comparison")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("log L")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize="small")
    _apply_suptitle(fig, title_suffix)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SGVB and Welch Bilby results via diagnostic plots."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "figures",
        help="Directory to write the generated plots (default: tests/figures).",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices={"png", "pdf", "svg"},
        help="Image format to use when saving plots.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figures interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    base_dir = Path(__file__).parent
    sgvb_bundle = load_result("SGVB", base_dir / "seed_47_sgvb_result.json")
    welch_bundle = load_result("Welch", base_dir / "seed_47_welch_result.json")
    bundles = [sgvb_bundle, welch_bundle]

    log_bf = sgvb_bundle.log_evidence - welch_bundle.log_evidence
    if log_bf > 700:  # avoid overflow in exp
        bf_display = "> ~1e304"
    else:
        bf_display = f"{np.exp(log_bf):.2e}"
    title_suffix = f"log BF (SGVB/Welch) = {log_bf:.2f} • BF ≈ {bf_display}"
    print(
        f"SGVB logZ={sgvb_bundle.log_evidence:.3f}, "
        f"Welch logZ={welch_bundle.log_evidence:.3f}, "
        f"log BF (SGVB/Welch)={log_bf:.3f}, BF≈{bf_display}"
    )

    plot_builders = [
        ("loglikelihood_trace", plot_loglikelihood_trace),
        ("loglikelihood_hist", plot_loglikelihood_histogram),
        ("loglikelihood_rank", plot_loglikelihood_rank),
    ]

    saved_paths = []
    for stem, builder in plot_builders:
        fig = builder(bundles, title_suffix=title_suffix)
        output_path = output_dir / f"{stem}.{args.format}"
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        saved_paths.append(output_path)
        if args.show:
            fig.show()
        plt.close(fig)

    if args.show:
        print("Saved plots:")
        for path in saved_paths:
            print(f" - {path}")


if __name__ == "__main__":
    main()
