#!/usr/bin/env python3
"""Generate Welch and AR_SGVB PSD estimates for GW150914 data."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
from spectrum import aryule, arma2psd
import bilby
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey
from sgvb_univar.psd_estimator import PSDEstimator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = bilby.core.utils.logger

BASE_DIR = Path(__file__).resolve().parent
PSD_DIR = BASE_DIR / "ar_psd_data"
CACHE_DIR = BASE_DIR / "data_cache"

TRIGGER_TIME = 1126259462.4
DETECTORS = ("H1", "L1")
DURATION = 4
POST_TRIGGER_DURATION = 2.0
ROLL_OFF = 0.4
MINIMUM_FREQUENCY = 20
MAXIMUM_FREQUENCY = 896

ANALYSIS_END_TIME = TRIGGER_TIME + POST_TRIGGER_DURATION
ANALYSIS_START_TIME = ANALYSIS_END_TIME - DURATION
PSD_DURATION = 32 * DURATION
PSD_END_TIME = ANALYSIS_START_TIME
PSD_START_TIME = PSD_END_TIME - PSD_DURATION

SGVB_SETTINGS = {
    "N_theta": 1800,
    "nchunks": 32,
    "ntrain_map": 20000,
    "N_samples": 500,
    "degree_fluctuate": 8000,
    "lr": 0.015,
    "max_hyperparm_eval": 1,
    "n_elbo_maximisation_steps": 600,
}


def _time_tag(value: float) -> str:
    return f"{int(round(value * 1000))}"


def _cache_path(detector: str, start: float, end: float) -> Path:
    return CACHE_DIR / f"{detector}_{_time_tag(start)}_{_time_tag(end)}.npz"


def _save_timeseries(ts: TimeSeries, path: Path) -> None:
    np.savez_compressed(
        path,
        values=ts.value,
        epoch=float(ts.epoch.gps),
        dt=float(ts.dt.value),
    )


def _load_timeseries(path: Path) -> TimeSeries:
    with np.load(path, allow_pickle=False) as npz:
        epoch = float(npz["epoch"])
        dt = float(npz["dt"])
        values = np.array(npz["values"])
    return TimeSeries(values, epoch=epoch, dt=dt)


def load_or_fetch_timeseries(detector: str, start: float, end: float) -> TimeSeries:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(detector, start, end)
    if cache_file.exists():
        LOGGER.info("Loading cached data for %s from %s", detector, cache_file)
        return _load_timeseries(cache_file)

    LOGGER.info("Downloading data for %s (%.3f - %.3f)", detector, start, end)
    ts = TimeSeries.fetch_open_data(detector, start, end)
    _save_timeseries(ts, cache_file)
    return ts


def _get_sample_rate(ts: TimeSeries) -> float:
    sample_rate = ts.sample_rate
    try:
        return float(sample_rate.value)
    except AttributeError:  # sample_rate already a float
        return float(sample_rate)


def bandlimit_psd(
    freq: np.ndarray,
    psd: np.ndarray,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (freq >= fmin) & (freq <= fmax)
    return freq[mask], psd[mask]


def estimate_welch_psd(ts: TimeSeries) -> Tuple[np.ndarray, np.ndarray]:
    psd_alpha = 2 * ROLL_OFF / DURATION
    spectrum = ts.psd(
        fftlength=DURATION,
        overlap=2,
        window=("tukey", psd_alpha),
        method="median",
    )
    return np.asarray(spectrum.frequencies.value), np.asarray(spectrum.value)


def window_in_chunks(x: np.ndarray, n_chunks: int, alpha: float) -> np.ndarray:
    seglen = x.size // n_chunks
    tapered = x[: seglen * n_chunks].reshape(n_chunks, seglen)
    window = tukey(seglen, alpha=alpha)
    tapered *= window
    return tapered.reshape(-1)


def whiten_by_ar_psd_rfft_chunks(x_1d, ar_psd, fs, nchunks):
    x_1d = np.asarray(x_1d)

    segN = x_1d.size // nchunks
    X = x_1d.reshape(nchunks, segN)

    out = np.empty_like(X)

    scale = np.sqrt(fs * segN)

    for i in range(nchunks):
        V_f = np.fft.rfft(X[i])[1:-1] / scale
        wh_f = V_f / np.sqrt(ar_psd)
        out[i] = np.fft.irfft(wh_f * scale, n=segN)

    return out.reshape(-1)


def estimate_ar_sgvb_psd(ts: TimeSeries, sample_rate: float) -> Tuple[np.ndarray, np.ndarray]:
    config = dict(SGVB_SETTINGS)
    n_samples_segment = int(round(DURATION * sample_rate))
    tukey_alpha = 2 * ROLL_OFF / DURATION
    window = tukey(n_samples_segment, tukey_alpha)
    ew = np.sqrt(np.mean(window**2))

    ts_value = ts.value.copy()
    data_win = window_in_chunks(ts_value, 1, tukey_alpha)
    chunked_win_data = window_in_chunks(ts_value, config["nchunks"], tukey_alpha)

    # --- AR ---
    ar, variance, _ = aryule(data_win, 350)
    ar_PSD = arma2psd(ar, B=None, rho=variance, T=sample_rate, 
                      NFFT=n_samples_segment, norm=False)[1:n_samples_segment//2]

    # --- whitening---
    import_cleaned_data = whiten_by_ar_psd_rfft_chunks(chunked_win_data, ar_PSD, 
                                                       sample_rate, config["nchunks"])
    import_cleaned_data = import_cleaned_data[:, np.newaxis]

    # --- SGVB ---
    estimator = PSDEstimator(
        x=import_cleaned_data,
        N_theta=config["N_theta"],
        nchunks=config["nchunks"],
        ntrain_map=config["ntrain_map"],
        N_samples=config["N_samples"],
        fs=sample_rate,
        max_hyperparm_eval=config["max_hyperparm_eval"],
        degree_fluctuate=config["degree_fluctuate"],
        n_elbo_maximisation_steps=config["n_elbo_maximisation_steps"],
        frange=[MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY],
    )
    estimator.run(lr=config["lr"])
    freq = np.asarray(estimator.freq)
    psd = np.asarray(estimator.pointwise_ci[1])
    sgvb_psd = np.real(psd[:, 0, 0]) * 2 / (ew**2)

    # --- final PSD ---
    f = np.fft.rfftfreq(n_samples_segment, 1.0 / sample_rate)
    frequency = f[1:-1]
    mask_band = (MINIMUM_FREQUENCY <= frequency) & (frequency <= MAXIMUM_FREQUENCY)
    ar_band = ar_PSD[mask_band]
    final_psd = ar_band * sgvb_psd

    return freq, final_psd


def compute_periodogram(ts: TimeSeries) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a simple periodogram using only the final analysis segment."""
    sample_rate = _get_sample_rate(ts)
    n_samples = int(round(DURATION * sample_rate))
    segment = ts[-n_samples:]

    tukey_alpha = 2 * ROLL_OFF / DURATION
    spectrum = segment.psd(fftlength=DURATION, overlap=0, window=("tukey", tukey_alpha))
    return np.asarray(spectrum.frequencies.value), np.asarray(spectrum.value)


def save_psd(path: Path, freq: np.ndarray, psd: np.ndarray) -> None:
    data = np.column_stack([freq, psd])
    np.savetxt(path, data, fmt="%.18e %.18e")
    LOGGER.info("Saved PSD data to %s", path)


def load_txt_psd(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]


def load_bayeswave_psd(detector: str) -> Tuple[np.ndarray, np.ndarray] | None:
    path = PSD_DIR / f"bayeswave_{detector.lower()}_psd.txt"
    if not path.exists():
        LOGGER.warning("BayesWave PSD not found: %s", path)
        return None
    freq, psd = load_txt_psd(path)
    return bandlimit_psd(freq, psd, MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY)


def plot_psd_comparison(
    detector: str,
    periodogram: Tuple[np.ndarray, np.ndarray],
    welch: Tuple[np.ndarray, np.ndarray],
    ar_sgvb: Tuple[np.ndarray, np.ndarray],
    bayeswave: Tuple[np.ndarray, np.ndarray],
) -> None:
    freq_periodogram, psd_periodogram = periodogram
    freq_welch, psd_welch = welch
    freq_sgvb, psd_ar_sgvb = ar_sgvb
    freq_bw, psd_bw = bayeswave

    plt.figure(figsize=(8, 5))
    plt.loglog(freq_periodogram, psd_periodogram,
           label="Periodogram", color="0.6", alpha=0.5, lw=1)
    plt.loglog(freq_welch, psd_welch,
               label="Welch", color="blue", alpha=0.5, lw=1)    
    plt.loglog(freq_sgvb, psd_ar_sgvb,
               label="AR SGVB", color="red", alpha=1.0, lw=1)    
    plt.loglog(freq_bw, psd_bw,
               label="BayesWave", color="black", alpha=0.5, lw=1)
    
    plt.xlim(MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"PSD [$1/\mathrm{Hz}$]")
    plt.title(f"GW150914 PSD comparison ({detector})")
    plt.legend(loc="best")
    plt.grid(True, which="both", ls=":", lw=0.5)
    plot_path = PSD_DIR / f"{detector.lower()}_psd_comparison.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    LOGGER.info("Saved PSD comparison plot to %s", plot_path)


def main() -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    PSD_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}

    for detector in DETECTORS:
        LOGGER.info("Processing %s", detector)
        ts = load_or_fetch_timeseries(detector, PSD_START_TIME, PSD_END_TIME)
        sample_rate = _get_sample_rate(ts)

        welch_freq, welch_psd = estimate_welch_psd(ts)
        sgvb_freq, ar_sgvb_psd = estimate_ar_sgvb_psd(ts, sample_rate)
        bayeswave = load_bayeswave_psd(detector)
        periodogram = compute_periodogram(ts)

        welch_freq, welch_psd = bandlimit_psd(welch_freq, welch_psd, MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY)
        sgvb_freq, ar_sgvb_psd = bandlimit_psd(sgvb_freq, ar_sgvb_psd, MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY)
        periodogram = (
            bandlimit_psd(periodogram[0], periodogram[1], MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY)
        )

        save_psd(PSD_DIR / f"welch_{detector.lower()}_psd.txt", welch_freq, welch_psd)
        save_psd(PSD_DIR / f"ar_sgvb_{detector.lower()}_psd.txt", sgvb_freq, ar_sgvb_psd)

        plot_psd_comparison(detector, periodogram, (welch_freq, welch_psd),
                           (sgvb_freq, ar_sgvb_psd), bayeswave=bayeswave)

        results[detector] = {
            "welch": (welch_freq, welch_psd),
            "ar_sgvb": (sgvb_freq, ar_sgvb_psd),
            "periodogram": periodogram,
        }

    LOGGER.info("Finished generating PSDs.")
    return results


if __name__ == "__main__":
    main()
