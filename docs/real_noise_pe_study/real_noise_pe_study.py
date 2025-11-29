"""
Runs a parameter estimation study comparing SGVB and Welch PSD estimates.
"""
import bilby
import corner
from sgvb_univar.psd_estimator import PSDEstimator
from sgvb_univar.lnz_correction import apply_psd_corrections
from sgvb_univar.compute_js_divergence import (
    compute_welch_vs_sgvb,
    save_js_table,
)
import numpy as np
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
import copy
from bilby.gw.detector.psd import PowerSpectralDensity
from matplotlib.lines import Line2D
import multiprocessing as mp
import sys
import os, re, glob
import h5py
from pathlib import Path


duration = 4.0
sampling_frequency = 4096.0
minimum_frequency = 20
maximum_frequency = 896
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency
    ),
)


'''
# get the files that corresponds to the GPS time
'''
data_dir = "/datasets/LIGO/public/gwosc.osgstorage.org/gwdata/O3b/strain.4k/hdf.v1/H1/"
CACHE_ROOT = os.path.join("outdir_pe_study", "cache")


def _ensure_cache_dir():
    os.makedirs(CACHE_ROOT, exist_ok=True)
    return CACHE_ROOT

def _get_data_files_and_gps_times(data_dir: str):
    
    search_str = os.path.join(data_dir, "*/*.hdf5")
    files = glob.glob(search_str)
    if not files:
        raise FileNotFoundError(f"No HDF5 files found at {search_str}")

    result = []
    
    for f in files:
        m = re.search(r"R1-(\d+)-(\d+)\.hdf5", f)
        if m:
            start = int(m.group(1))
            dur   = int(m.group(2))
            stop  = start + dur
            result.append((start, stop, f))

    result.sort(key=lambda x: x[0])
    return result


def load_local_ts(data_dir, t0, t1):
    pieces = []
    for start, stop, path in _get_data_files_and_gps_times(data_dir):
        if stop <= t0:
            continue
        if start >= t1:
            break
        seg_start = max(t0, start)
        seg_stop  = min(t1, stop)
        ts = TimeSeries.read(path, format='hdf5.gwosc', start=seg_start, end=seg_stop)
        pieces.append(ts)

    if not pieces:
        raise ValueError("No files covering this period of time")

    return pieces


'''
# get noise data, start from GPS time = 1256664443.
'''
segment_file = "H1_valid_segments_1256664443-1269361988.txt"
INJECTION_CSV = Path(__file__).resolve().parent / "bbh_injections_snr_10_50_clean.csv"
_injection_catalog = None
INJECTION_COLUMN_MAP = [
    "mass_ratio",
    "chirp_mass",
    "luminosity_distance",
    "dec",
    "ra",
    "theta_jn",
    "psi",
    "phase",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
]


def _load_injection_catalog():
    global _injection_catalog
    if _injection_catalog is None:
        if not INJECTION_CSV.exists():
            raise FileNotFoundError(
                f"Injection catalog not found at {INJECTION_CSV}; cannot sample injections."
            )
        _injection_catalog = np.genfromtxt(
            INJECTION_CSV,
            delimiter=",",
            names=True,
            dtype=None,
            encoding=None,
        )
    return _injection_catalog


def sample_injection_from_catalog(seed: int, h1_snr_threshold: float = 10.0) -> dict:
    catalog = _load_injection_catalog()
    if "H1_snr" not in catalog.dtype.names:
        raise ValueError("Injection catalog missing 'H1_snr' column.")
    mask = catalog["H1_snr"] > h1_snr_threshold
    filtered = catalog[mask]
    if filtered.size == 0:
        raise ValueError(
            f"No injections exceed H1_snr>{h1_snr_threshold} in {INJECTION_CSV}."
        )
    rng = np.random.default_rng(seed)
    row = rng.choice(filtered)
    params = {}
    for key in INJECTION_COLUMN_MAP:
        if key in filtered.dtype.names:
            params[key] = float(row[key])
    return params


def get_noise_segment_from_seed(segment_file, data_dir, seed):
    segments = np.loadtxt(segment_file, dtype=np.int64, skiprows=1)
    t0, t1 = segments[seed]
    pieces = load_local_ts(data_dir, t0, t1)
    y = np.concatenate([p.value for p in pieces])
    return y, t0, t1


def evaluate_stationarity(
    data,
    fs,
    frange=(20.0, 896.0),
    qrange=(4.0, 32.0),
    pad=1.0,
    whiten: bool = True,
):
    ts = TimeSeries(np.asarray(data), dt=1.0/fs)
    ts = ts.highpass(frange[0])
    ts = ts.lowpass(frange[1])
    ts = ts.whiten()
    
    duration = ts.duration.value
    start = float(max(0.0, pad))
    stop  = float(max(start, duration - pad))
    qspec = ts.q_transform(outseg=(start, stop), qrange=qrange, frange=frange)
    
    power_time = np.sum(np.abs(qspec.value)**2, axis=0)
    power_time = power_time / (np.median(power_time))
    stationarity_sigma = float(np.std(power_time))

    return stationarity_sigma


def prepare_interferometers(det_names, sampling_frequency, duration, seed):
    """Create interferometers populated with on-source strain data."""
    ifos = bilby.gw.detector.InterferometerList(det_names)

    cache_dir = _ensure_cache_dir()
    strain_cache_path = os.path.join(cache_dir, f"seed_{seed}_strain.npz")
    if os.path.exists(strain_cache_path):
        cached = np.load(strain_cache_path)
        segment = cached["segment"]
        t0 = int(cached["t0"])
        t1 = int(cached["t1"])
        print(f"Loaded cached strain data from {strain_cache_path}")
    else:
        segment, t0, t1 = get_noise_segment_from_seed(segment_file, data_dir, seed)
        np.savez_compressed(strain_cache_path, segment=segment, t0=t0, t1=t1)
        print(f"Cached strain data to {strain_cache_path}")
    start_time = t1 - duration
    print('Start GPS time:', start_time, 'End GPS time:', t1)
    boundary = int(32*sampling_frequency*duration)
    on_source_data = segment[boundary: ]
    off_source_data = segment[:boundary]
    
    for ifo in ifos:
        ifo.strain_data.set_from_time_domain_strain(
            time_domain_strain=on_source_data,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            duration=duration
        )
        
        ifo.maximum_frequency = maximum_frequency
        ifo.minimum_frequency = minimum_frequency
    return ifos, on_source_data, off_source_data, start_time


def get_fd_data(strain_data: np.ndarray, times: np.ndarray, det: str, roll_off: float, fmin: float, fmax: float):
    """Fixed function with correct variable names and parameters."""
    if fmax is None:
        fmax = sampling_frequency//2
    
    strain_ts = TimeSeries(strain_data, times=times)
    ifo = bilby.gw.detector.get_empty_interferometer(det)
    ifo.strain_data.roll_off = roll_off
    ifo.maximum_frequency = fmax  # Fixed: was f_f
    ifo.minimum_frequency = fmin  # Fixed: was f_i
    ifo.strain_data.set_from_gwpy_timeseries(strain_ts)

    x = ifo.strain_data.frequency_array
    y = ifo.strain_data.frequency_domain_strain
    Ew = np.sqrt(ifo.strain_data.window_factor)

    I = (x >= fmin) & (x <= fmax)  # Fixed: was f_i and f_f
    return x[I], y[I] / Ew


def estimate_welch_psd(
        ts: TimeSeries,
        sampling_frequency: float = 4096,
        duration = 4.0,
        psd_fractional_overlap: float = 0.5,
        post_trigger_duration: float = 2.0,
        psd_length: int = 32,
        psd_maximum_duration: int = 1024,
        psd_method: str = "median",
        psd_start_time: float | None = None,
        minimum_frequency: float | None = 20.0,
        maximum_frequency: float | None = 896.0,
        tukey_roll_off: float = 0.4,
):
    """
    Estimate the PSD of a time series using Welch's method via gwpy.

    Parameters
    ----------
    ts : gwpy.timeseries.TimeSeries
        Input strain or time series data.
    sampling_frequency : float
        Sampling frequency in Hz (default: 4096).
    psd_fractional_overlap : float
        Fractional overlap of Welch segments (default: 0.5).
    post_trigger_duration : float
        Duration (s) of data after trigger (default: 2.0).
    psd_length : int
        Multiplier for segment duration (default: 32).
    psd_maximum_duration : int
        Maximum PSD segment duration (default: 1024 s).
    psd_method : str
        PSD estimator method ("mean", "median", etc.) (default: "median").
    psd_start_time : float or None
        Relative start time (seconds before main segment). If None,
        uses `psd_length * post_trigger_duration` before segment start.
    minimum_frequency : float or None
        Minimum frequency (Hz) to include (default: 20).
    maximum_frequency : float or None
        Maximum frequency (Hz) to include (default: None = Nyquist).
    tukey_roll_off : float
        Roll-off (s) for Tukey window (default: 0.4).

    Returns
    -------
    psd : gwpy.spectrum.Spectrum
        The estimated power spectral density.

    SEE https://git.ligo.org/lscsoft/bilby_pipe/-/blob/master/bilby_pipe/parser.py?ref_type=heads#L334
    and
    https://git.ligo.org/lscsoft/bilby_pipe/-/blob/master/bilby_pipe/data_generation.py?ref_type=heads#L572

    """
    #psd_duration = min(psd_length * duration, psd_maximum_duration)
    if maximum_frequency is None:
        maximum_frequency = sampling_frequency//2

    # Calculate the Tukey alpha parameter
    tukey_alpha = 2 * tukey_roll_off / duration
    overlap = psd_fractional_overlap * duration

    print(
        "Welch PSD settings: fftlength=%s, overlap=%s, method=%s, "
        "tukey_alpha=%s (roll-off=%s)",
        duration, overlap, psd_method, tukey_alpha, tukey_roll_off,
    )

    # Apply Welch PSD estimation
    psd = ts.psd(
        fftlength=duration,
        overlap=overlap,
        window=("tukey", tukey_alpha),
        method=psd_method,
    )

    # Restrict frequency band
    freqs_welch = psd.frequencies.value
    mask = (freqs_welch >= minimum_frequency) & (freqs_welch <= maximum_frequency)
    freqs_welch = freqs_welch[mask]
    psd = psd[mask]

    return freqs_welch, psd


def window_in_chunks(x: np.ndarray, n_chunks: int = 32, alpha: float = 0.4) -> np.ndarray:
    """Split x into n_chunks, apply Tukey window to each chunk, and re-concatenate."""
    N = x.size
    seglen = N // n_chunks
    w = tukey(seglen, alpha=alpha)
    X = x.reshape(n_chunks, seglen) * w
    return X.reshape(N)


def estimate_sgvb_psd(time_series, sampling_frequency, duration=4,
                      minimum_frequency = 20.0, maximum_frequency = 896.0,
                      N_theta=6000, nchunks=32, ntrain_map=10000,
                      N_samples=500, degree_fluctuate=8000, seed=None,
                      tukey_roll_off = 0.4):
    
    '''
    # window in chunks
    '''
    tukey_alpha = 2 * tukey_roll_off / duration
    import_data = window_in_chunks(time_series.value, nchunks, tukey_alpha)
    import_data = import_data.reshape(-1, 1)
    
    N = int(duration * sampling_frequency)        
    w = tukey(N, tukey_alpha)
    Ew = np.sqrt(np.mean(w**2))
    
    if maximum_frequency is None:
        maximum_frequency = sampling_frequency//2
        
    frange = [minimum_frequency, maximum_frequency]    
    psd_est = PSDEstimator(
        x=import_data,
        N_theta=N_theta,
        nchunks=nchunks,
        ntrain_map=ntrain_map,
        N_samples=N_samples,
        fs=sampling_frequency,
        max_hyperparm_eval=1,
        degree_fluctuate=degree_fluctuate,
        n_elbo_maximisation_steps=600,
        frange=frange
    )
    psd_est.run(lr=0.0078)
    freqs = psd_est.freq
    psd = psd_est.pointwise_ci[1]
    psd = psd*2 / Ew**2
    return freqs, np.real(psd[:,0,0])


def run_pe_study(
        det_names=("H1",),
        sampling_frequency_local=sampling_frequency,
        minimum_frequency_local=minimum_frequency,
        maximum_frequency_local=maximum_frequency,
        outdir_="outdir_pe_study",
        sgvb_settings=None,
        seed=0,
):
    bilby.core.utils.random.seed(seed)
    label = f"seed_{seed}"
    outdir = f"outdir_pe_study/seed_{seed}"
    os.makedirs(outdir, exist_ok=True)

    print(">>>> Running PE study with seed =", seed, " <<<<")

    # Setup logger and seed specifically for this run
    bilby.core.utils.setup_logger(outdir=outdir, label=label)
    bilby.core.utils.random.seed(seed)

    ifos, on_source_data, off_source_data, segment_start_time = prepare_interferometers(
        det_names,
        sampling_frequency_local,
        duration=duration,
        seed=seed,
    )

    trigger_offset = 2.0  # seconds after segment start
    trigger_time = segment_start_time + trigger_offset

    inj_prior = bilby.gw.prior.BBHPriorDict()
    injection_params = inj_prior.sample()
    catalog_params = sample_injection_from_catalog(seed)
    injection_params.update(catalog_params)
    injection_params["geocent_time"] = trigger_time
    print("Selected injection from catalog (H1_snr > 10):")
    for key, value in catalog_params.items():
        print(f"  {key}: {value}")

    analysis_prior = bilby.gw.prior.BBHPriorDict()
    analysis_prior["geocent_time"] = bilby.core.prior.Uniform(
        trigger_time - 0.1, trigger_time + 0.1, name="geocent_time"
    )
    analysis_prior.validate_prior(duration, minimum_frequency_local)

    print("Injection parameters:")
    for k, v in injection_params.items():
        print(f"  {k}: {v}")
    print("\nAnalysis priors:")
    for k, v in analysis_prior.items():
        print(f"  {k}: {v}")

    off_sigma = evaluate_stationarity(
        data=off_source_data,
        fs=sampling_frequency_local,
        pad=1.0
    )
    print(f"[stationarity] Off-source segment σ = {off_sigma:.3f}")
    
    on_sigma = evaluate_stationarity(
        data=on_source_data,
        fs=sampling_frequency_local,
        pad=0.25
    )
    print(f"[stationarity] On-source segment σ = {on_sigma:.3f}")
    
    
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_params)

    psd_estimates = {}
    psd_cache_path = os.path.join(_ensure_cache_dir(), f"seed_{seed}_psd.npz")

    if os.path.exists(psd_cache_path):
        cached_psd = np.load(psd_cache_path)
        psd_estimates['welch'] = (
            cached_psd['freqs_welch'],
            cached_psd['welch_psd']
        )
        psd_estimates['sgvb'] = (
            cached_psd['freqs_sgvb'],
            cached_psd['sgvb_psd']
        )
        print(f"Loaded cached PSDs from {psd_cache_path}")
    else:
        # Extract noise-only segment (all data after injection )
        psd_data = TimeSeries(off_source_data, dt=1.0/sampling_frequency)

        # Compute Welch PSD + SGVB PSD
        freqs_welch, welch_psd = estimate_welch_psd(psd_data, sampling_frequency_local)
        freqs_welch = np.asarray(freqs_welch)
        welch_psd = np.asarray(welch_psd)
        freqs_sgvb, sgvb_psd = estimate_sgvb_psd(psd_data, sampling_frequency_local)

        psd_estimates['welch'] = (freqs_welch, welch_psd)
        psd_estimates['sgvb'] = (freqs_sgvb, sgvb_psd)

        np.savez_compressed(
            psd_cache_path,
            freqs_welch=freqs_welch,
            welch_psd=welch_psd,
            freqs_sgvb=freqs_sgvb,
            sgvb_psd=sgvb_psd
        )
        print(f"Cached PSDs to {psd_cache_path}")

    # plot the two PSDs for comparison along with on-source data
    freqs_welch, welch_psd = psd_estimates['welch']
    print("Welch PSD freq:", "length = ", len(freqs_welch), " min =", freqs_welch[0], " max =", freqs_welch[-1],
      " df =", freqs_welch[1]-freqs_welch[0])
    freqs_sgvb,  sgvb_psd  = psd_estimates['sgvb']
    print("SGVB PSD freq:", "length = ", len(freqs_sgvb), " min =", freqs_sgvb[0], " max =", freqs_sgvb[-1],
      " df =", freqs_sgvb[1]-freqs_sgvb[0])
    
    N = len(on_source_data)                   
    times = np.arange(N) / sampling_frequency_local         
    f, on_source_f = get_fd_data(on_source_data, times = times, det='H1', roll_off = 0.4,
                               fmin = minimum_frequency, fmax = maximum_frequency)
    print("on_source_f:", "length = ", len(f), " min =", f[0], " max =", f[-1],
      " df =", f[1]-f[0])
    
    fig = plt.figure(figsize=(7, 5))
    plt.loglog(f, np.abs(on_source_f)**2, alpha=0.3, label="Data", color="lightgray")
    plt.loglog(freqs_welch, welch_psd, alpha=0.7, label="Welch PSD", color="green")
    plt.loglog(freqs_sgvb, sgvb_psd, alpha=1, label="SGVB PSD", color="red")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [strain²/Hz]")
    plt.legend()
    plt.tight_layout()
    outpath = f"{outdir}/SGVB_Welch_PSDs.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    
     
    # collect Original, Welch and SGVB PSDs
    ifos_welch = copy.deepcopy(ifos)
    ifos_sgvb = copy.deepcopy(ifos)
    
    welch_psd_object = PowerSpectralDensity.from_amplitude_spectral_density_array(
        freqs_welch, welch_psd
    )
    sgvb_psd_object = PowerSpectralDensity.from_amplitude_spectral_density_array(
        freqs_sgvb, sgvb_psd
    )
    
    for i in range(len(ifos)):
        ifos_welch[i].power_spectral_density = welch_psd_object
        ifos_sgvb[i].power_spectral_density = sgvb_psd_object
    
    ifos_for_analysis = dict(
        welch=ifos_welch,
        sgvb=ifos_sgvb
    )
    psd_methods = list(ifos_for_analysis.keys())


    # Now we do the analysis twice, once with each PSD
    results = {}
    optimal_snrs = {}
    h_pols = waveform_generator.frequency_domain_strain(injection_params)
    for name, analysis_ifos in ifos_for_analysis.items():
        ifo = analysis_ifos[0]
        h_ifo = ifo.get_detector_response(h_pols, injection_params)
        snr2 = ifo.optimal_snr_squared(h_ifo)
        optimal_snrs[name] = float(np.sqrt(np.real(snr2)))
        
        print("Running analysis with", name, "PSD")
        run_label = f"{label}_{name}"
        result_json = os.path.join(outdir, f"{run_label}_result.json")
        if os.path.exists(result_json):
            print(f"Found existing result at {result_json}; loading instead of rerunning sampler.")
            res = bilby.result.read_in_result(outdir=outdir, label=run_label)
        else:
            likelihood = bilby.gw.GravitationalWaveTransient(
                interferometers=analysis_ifos,
                waveform_generator=waveform_generator,
                time_marginalization=True,
                phase_marginalization=False,
                distance_marginalization=True,
                priors=analysis_prior,
            )
            
            test_params = analysis_prior.sample()
            likelihood.parameters.update(test_params)
            ll = likelihood.log_likelihood()  
            print("Test logL:", ll)

            npool = min(mp.cpu_count(), int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
            print("npool = ", npool)
            res = bilby.run_sampler(
                likelihood=likelihood,
                priors=analysis_prior,
                sampler="dynesty",
                nlive=2000,
                nact=20,
                sample="rwalk",
                npool=npool,
                queue_size=npool,
                injection_parameters=injection_params,
                outdir=outdir,
                label=run_label,
                resume=False,
                conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
            )
        results[name] = res
        
    print(
        f"Optimal SNR (H1): "
        f"Welch = {optimal_snrs['welch']:.2f}, "
        f"SGVB = {optimal_snrs['sgvb']:.2f}"
    )    
        

    # Compute JS divergence between SGVB and Welch posteriors
    param_names = list(results["welch"].search_parameter_keys)
    js_df = compute_welch_vs_sgvb(param_names, results["sgvb"], results["welch"])
    js_path = os.path.join(outdir, f"{label}_js_divergence.csv")
    save_js_table(js_df, js_path)
    print(f"Saved JS divergences to {js_path}")

    print("Welch posterior rows:", len(results["welch"].posterior))
    print("SGVB  posterior rows:", len(results["sgvb"].posterior))

    # --- Save results (compact numeric file) ---

    meta = dict(
        welch_freq=freqs_welch,
        welch_psd=welch_psd,
        sgvb_freq=freqs_sgvb,
        sgvb_median=sgvb_psd,
    )

    
    logz_corrected = apply_psd_corrections(
        results,
        ifos_for_analysis,
        psd_methods,
        outdir,
        duration=duration,
    )

    csv_path = os.path.join(outdir_, "log_evidence_summary.csv")
    header = (
        "seed,"
        "sgvb_log_evidence,sgvb_log_noise_evidence,sgvb_log_bayes_factor, sgvb_optimal_snr,"
        "welch_log_evidence,welch_log_noise_evidence,welch_log_bayes_factor, welch_optimal_snr,"
        "off_source_sigma,on_source_sigma\n"
    )
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as csv_file:
            csv_file.write(header)

    with open(csv_path, "a") as csv_file:
        sgvb = logz_corrected["sgvb"]
        welch = logz_corrected["welch"]
        csv_file.write(
            f"{seed},"
            f"{sgvb['log_evidence']:.2f},{sgvb['log_noise_evidence']:.2f},{sgvb['log_bayes_factor']:.2f},{optimal_snrs['sgvb']:.2f},"
            f"{welch['log_evidence']:.2f},{welch['log_noise_evidence']:.2f},{welch['log_bayes_factor']:.2f},{optimal_snrs['welch']:.2f},"
            f"{off_sigma:.2f},{on_sigma:.2f}\n"
        )
    
    
    return results, meta


if __name__ == '__main__':
    # Run with the default parameters defined at module scope
    args = sys.argv[1:]
    seed = 0
    if len(args) > 0:
        seed = int(args[0])
        
    run_pe_study(seed=seed)
