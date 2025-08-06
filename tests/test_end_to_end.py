import matplotlib.pyplot as plt
import numpy as np

from sgvb_univar.example_datasets import ARData
from sgvb_univar.example_datasets.lvk_data import LVKData
from sgvb_univar.psd_estimator import PSDEstimator


def test_end_to_end(plot_dir):
    data = ARData(
        order=4,
        duration=4.0,
        fs=2*np.pi
    )
    psd_estimator = PSDEstimator(
        x=np.array(data.ts).reshape(-1, 1),
        N_theta=30,
        nchunks=1,
        fs=data.fs,
    )
    psd_all, pointwise_ci = psd_estimator.run(lr=0.03)

    # plot results
    fig, ax = plt.subplots()
    plot_ci(ax, data.freqs[:-1], pointwise_ci)
    ax.plot(data.freqs, data.psd_theoretical, color='tab:orange', label='True PSD')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/test_end_to_end_psd.png")
    plt.close(fig)


def plot_ci(ax, freqs, pointwise_ci):
    ax.fill_between(
        freqs,
        pointwise_ci[0, :, 0, 0],
        pointwise_ci[1, :, 0, 0],
        color='lightgray',
        alpha=0.5,
        label='95% CI'
    )
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [1/Hz]')


def test_lvk(plot_dir):
    lvk_data = LVKData.load("H1", duration=32*4)
    psd_estimator = PSDEstimator(
        x=np.array(lvk_data.strain.value).reshape(-1, 1),
        N_theta=30,
        nchunks=32,
        fs=lvk_data.fs,
        frange=[20, 2048],
    )
    psd_all, pointwise_ci = psd_estimator.run(lr=0.03)

    # plot results
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_ci(ax, lvk_data.freqs[1:-1], pointwise_ci)
    ax.loglog(lvk_data.freqs, lvk_data.median_psd)
    plt.savefig(f"{plot_dir}/test_lvk_psd.png")
    plt.close(fig)
