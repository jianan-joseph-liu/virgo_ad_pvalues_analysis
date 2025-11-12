#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on GW150914.

This version caches the data locally after the first download and compares
parameter estimation runs using two PSD estimation strategies:
  1. Welch PSD
  2. SGVB PSD
"""
from pathlib import Path
from typing import Dict, Tuple
import multiprocessing as mp
import os
import bilby
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey
from sgvb_univar.psd_estimator import PSDEstimator
from sgvb_univar.lnz_correction import apply_psd_corrections

logger = bilby.core.utils.logger
outdir = "outdir"
label = "GW150914"

CACHE_DIR = Path("data_cache")
PSD_METHODS = ("welch", "sgvb")
SGVB_SETTINGS = {
    "N_theta": 6000,
    "nchunks": 32,
    "ntrain_map": 10000,
    "N_samples": 500,
    "degree_fluctuate": 8000,
    "lr": 0.0078,
    "max_hyperparm_eval": 1,
    "n_elbo_maximisation_steps": 600,
}

# Analysis configuration
trigger_time = 1126259462.4
detectors = ["H1", "L1"]
maximum_frequency = 1024
minimum_frequency = 20
roll_off = 0.4  # Roll-off duration of Tukey window in seconds
duration = 4  # Analysis segment duration
post_trigger_duration = 2  # Time between trigger time and end of segment
end_time = trigger_time + post_trigger_duration
start_time = end_time - duration

psd_duration = 32 * duration
psd_start_time = start_time - psd_duration
psd_end_time = start_time


def _time_tag(value: float) -> str:
    """Return a filesystem-friendly tag for a GPS time."""
    return f"{int(round(value * 1000))}"


def _cache_path(detector: str, start: float, end: float, kind: str) -> Path:
    return CACHE_DIR / f"{detector}_{_time_tag(start)}_{_time_tag(end)}_{kind}.npz"


def _save_timeseries(ts: TimeSeries, path: Path) -> None:
    """Persist a TimeSeries as compressed numpy arrays."""
    np.savez_compressed(
        path,
        values=ts.value,
        epoch=float(ts.epoch.gps),
        dt=float(ts.dt.value),
    )


def _load_timeseries(path: Path) -> TimeSeries:
    """Load a cached TimeSeries stored with _save_timeseries."""
    with np.load(path, allow_pickle=False) as npz:
        epoch = float(npz["epoch"])
        dt = float(npz["dt"])
        values = np.array(npz["values"])
    return TimeSeries(values, epoch=epoch, dt=dt)


def load_or_fetch_timeseries(detector: str, start: float, end: float, kind: str) -> TimeSeries:
    """Load a detector time series from cache, or download and cache it."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(detector, start, end, kind)
    if cache_file.exists():
        logger.info("Loading cached %s data for %s from %s", kind, detector, cache_file)
        return _load_timeseries(cache_file)

    logger.info("Downloading %s data for ifo %s", kind, detector)
    ts = TimeSeries.fetch_open_data(detector, start, end)
    _save_timeseries(ts, cache_file)
    return ts


def _get_sample_rate(ts: TimeSeries) -> float:
    """Return the sampling frequency for a TimeSeries as a float."""
    sample_rate = ts.sample_rate
    try:
        return float(sample_rate.value)
    except AttributeError:
        return float(sample_rate)


def bandlimit_psd(freq: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """Restrict a PSD to the analysis band."""
    mask = (freq >= fmin) & (freq <= fmax)
    return freq[mask], psd[mask]


def estimate_welch_psd(
    ts: TimeSeries,
    duration_seconds: float,
    roll_off_seconds: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a PSD via the standard Welch median estimator."""
    psd_alpha = 2 * roll_off_seconds / duration_seconds
    spectrum = ts.psd(fftlength=duration_seconds, overlap=2, window=("tukey", psd_alpha), method="median")
    return np.asarray(spectrum.frequencies.value), np.asarray(spectrum.value)


def window_in_chunks(x: np.ndarray, n_chunks: int = 32, alpha: float = 0.4) -> np.ndarray:
    """Split x into n_chunks, apply Tukey window to each chunk, and re-concatenate."""
    N = x.size
    seglen = N // n_chunks
    w = tukey(seglen, alpha=alpha)
    X = x.reshape(n_chunks, seglen) * w
    return X.reshape(N)


def estimate_sgvb_psd(
    ts: TimeSeries,
    duration_seconds: float,
    roll_off_seconds: float,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    sampling_frequency_hz: float,
    settings: Dict[str, float] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate a PSD using the SGVB PSD estimator."""
    config = dict(SGVB_SETTINGS)
    if settings:
        config.update(settings)

    n_samples_segment = int(round(duration_seconds * sampling_frequency_hz))
    tukey_alpha = 2 * roll_off_seconds / duration_seconds
    window = tukey(n_samples_segment, tukey_alpha)
    ew = np.sqrt(np.mean(window**2))

    '''
    # window in chunks
    '''
    data = window_in_chunks(ts.value, config["nchunks"], tukey_alpha)
    data = np.asarray(data).reshape(-1, 1)
    
    estimator = PSDEstimator(
        x=data,
        N_theta=config["N_theta"],
        nchunks=config["nchunks"],
        ntrain_map=config["ntrain_map"],
        N_samples=config["N_samples"],
        fs=sampling_frequency_hz,
        max_hyperparm_eval=config["max_hyperparm_eval"],
        degree_fluctuate=config["degree_fluctuate"],
        n_elbo_maximisation_steps=config["n_elbo_maximisation_steps"],
        frange=[minimum_frequency_hz, maximum_frequency_hz],
    )
    estimator.run(lr=config["lr"])
    freq = np.asarray(estimator.freq)
    psd = np.asarray(estimator.pointwise_ci[1])
    psd = np.real(psd[:, 0, 0]) * 2 / (ew**2)
    return freq, psd


def prepare_data() -> Dict[str, Dict[str, TimeSeries]]:
    """Download or load cached time-domain data for each detector."""
    data_by_detector: Dict[str, Dict[str, TimeSeries]] = {}
    for det in detectors:
        analysis_ts = load_or_fetch_timeseries(det, start_time, end_time, "analysis")
        psd_ts = load_or_fetch_timeseries(det, psd_start_time, psd_end_time, "psd")

        analysis_rate = _get_sample_rate(analysis_ts)
        psd_rate = _get_sample_rate(psd_ts)
        if not np.isclose(analysis_rate, psd_rate):
            raise ValueError(f"Sampling frequency mismatch for {det}: {analysis_rate} vs {psd_rate}")

        data_by_detector[det] = {
            "analysis_ts": analysis_ts,
            "psd_ts": psd_ts,
            "sample_rate": analysis_rate,
        }
    return data_by_detector


def compute_psds(
    data_by_detector: Dict[str, Dict[str, TimeSeries]],
) -> Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Compute the PSD for each detector using all configured methods."""
    psds: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for det, info in data_by_detector.items():
        psd_ts = info["psd_ts"]
        sample_rate = info["sample_rate"]

        welch_freq, welch_psd = estimate_welch_psd(psd_ts, duration, roll_off)
        welch_freq, welch_psd = bandlimit_psd(welch_freq, welch_psd, minimum_frequency, maximum_frequency)

        sgvb_freq, sgvb_psd = estimate_sgvb_psd(
            psd_ts,
            duration_seconds=duration,
            roll_off_seconds=roll_off,
            minimum_frequency_hz=minimum_frequency,
            maximum_frequency_hz=maximum_frequency,
            sampling_frequency_hz=sample_rate,
        )
        sgvb_freq, sgvb_psd = bandlimit_psd(sgvb_freq, sgvb_psd, minimum_frequency, maximum_frequency)

        psds[det] = {
            "welch": (welch_freq, welch_psd),
            "sgvb": (sgvb_freq, sgvb_psd),
        }
    return psds


def build_interferometers(
    data_by_detector: Dict[str, Dict[str, TimeSeries]],
    psds: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
) -> Dict[str, bilby.gw.detector.InterferometerList]:
    """Construct interferometer lists for each PSD method."""
    interferometers_by_method: Dict[str, bilby.gw.detector.InterferometerList] = {
        method: bilby.gw.detector.InterferometerList([]) for method in PSD_METHODS
    }

    for det in detectors:
        info = data_by_detector[det]
        analysis_ts = info["analysis_ts"]
        for method in PSD_METHODS:
            freq, psd_values = psds[det][method]
            ifo = bilby.gw.detector.get_empty_interferometer(det)
            ifo.strain_data.roll_off = roll_off
            ifo.minimum_frequency = minimum_frequency
            ifo.maximum_frequency = min(maximum_frequency, float(freq[-1]))
            ifo.strain_data.set_from_gwpy_timeseries(analysis_ts)
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
                frequency_array=freq,
                psd_array=psd_values,
            )
            interferometers_by_method[method].append(ifo)
    return interferometers_by_method


def run_parameter_estimation(
    interferometers_by_method: Dict[str, bilby.gw.detector.InterferometerList],
) -> Dict[str, bilby.result.Result]:
    """Run bilby parameter estimation for each PSD method."""
    results: Dict[str, bilby.result.Result] = {}

    waveform_generator = bilby.gw.WaveformGenerator(
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": "IMRPhenomPv2",
            "reference_frequency": 50,
        },
    )

    base_priors = bilby.gw.prior.BBHPriorDict(filename="GW150914.prior")
    base_priors["geocent_time"] = bilby.core.prior.Uniform(
        trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
    )

    for method, ifo_list in interferometers_by_method.items():
        priors = base_priors.copy()
        method_label = f"{label}_{method}"
        logger.info("Starting parameter estimation using the %s PSD", method.upper())

        likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
            ifo_list,
            waveform_generator,
            priors=priors,
            time_marginalization=True,
            phase_marginalization=False,
            distance_marginalization=True,
        )

        result_json = Path(outdir) / f"{method_label}_result.json"
        if result_json.exists():
            logger.info("Found existing result at %s; loading instead of rerunning sampler.", result_json)
            result = bilby.result.read_in_result(outdir=outdir, label=method_label)
        else:
            npool = min(mp.cpu_count(), int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
            print("npool = ", npool)
            result = bilby.run_sampler(
                likelihood,
                priors,
                sampler="dynesty",
                outdir=outdir,
                label=method_label,
                nlive=2000,
                sample="rwalk",
                walks=100,
                nact=10,
                check_point_delta_t=10000,
                check_point_plot=True,
                npool=npool,
                conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            )
        results[method] = result
        result.plot_corner()
        
    # compute Bayes factor SGVB vs Welch
    logZ_welch = results["welch"].log_evidence
    logZ_sgvb = results["sgvb"].log_evidence
    logBF = logZ_sgvb - logZ_welch
    BF = np.exp(logBF)

    print(f"logZ (welch) = {logZ_welch:.3f}")
    print(f"logZ (sgvb)  = {logZ_sgvb:.3f}")
    print(f"logBF (sgvb - welch) = {logBF:.3f}, BF = {BF:.3e}")

    print("Welch posterior rows:", len(results["welch"].posterior))
    print("SGVB  posterior rows:", len(results["sgvb"].posterior))    

    logz_corrected = {}
    logz_corrected = apply_psd_corrections(results, interferometers_by_method, PSD_METHODS, outdir)    
    
    return results


def main() -> Dict[str, bilby.result.Result]:
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
    data_by_detector = prepare_data()
    psds = compute_psds(data_by_detector)
    interferometers_by_method = build_interferometers(data_by_detector, psds)

    logger.info("Saving data plots to %s", outdir)
    interferometers_by_method["welch"].plot_data(outdir=outdir, label=f"{label}_welch")
    interferometers_by_method["sgvb"].plot_data(outdir=outdir, label=f"{label}_sgvb")

    return run_parameter_estimation(interferometers_by_method)


if __name__ == "__main__":
    main()
