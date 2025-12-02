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
import os
import multiprocessing as mp
import sys
import h5py

CACHE_ROOT = os.path.join("outdir_simulate_pe_study", "cache")

def _ensure_cache_dir():
    os.makedirs(CACHE_ROOT, exist_ok=True)
    return CACHE_ROOT

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
    ),
)


def prepare_interferometers(det_names, sampling_frequency, duration, start_time):
    """Create an InterferometerList and set strain data from PSDs.

    Returns the InterferometerList.
    """
    ifos = bilby.gw.detector.InterferometerList([])
    for det in det_names:
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.maximum_frequency = maximum_frequency
        ifo.minimum_frequency = minimum_frequency
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=sampling_frequency,
            duration=duration,
            start_time=start_time,
        )
        ifos.append(ifo)
    return ifos


def prepare_analysis_ifos(det_names, data):
    """Create an InterferometerList and inject stain data for PE.

    Returns the InterferometerList.
    """
    ifos = bilby.gw.detector.InterferometerList([])
    for det in det_names:
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.maximum_frequency = maximum_frequency
        ifo.minimum_frequency = minimum_frequency
        ifo.strain_data.set_from_gwpy_timeseries(data)
        ifos.append(ifo)
    return ifos


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
                      N_theta=6500, nchunks=32, ntrain_map=10000,
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
    psd_est.run(lr=0.0075)
    freqs = psd_est.freq
    psd = psd_est.pointwise_ci[1]
    psd = psd*2 / Ew**2
    return freqs, np.real(psd[:,0,0])


def run_pe_study(
        det_names=("H1",),
        sampling_frequency_local=sampling_frequency,
        minimum_frequency_local=minimum_frequency,
        maximum_frequency_local=maximum_frequency,
        outdir_="outdir_simulate_pe_study",
        sgvb_settings=None,
        seed=0,
):
    bilby.core.utils.random.seed(seed)
    label = f"seed_{seed}"
    outdir = f"outdir_simulate_pe_study/seed_{seed}"
    os.makedirs(outdir, exist_ok=True)

    print(">>>> Running PE study with seed =", seed, " <<<<")

    # Setup logger and seed specifically for this run
    bilby.core.utils.setup_logger(outdir=outdir, label=label)
    bilby.core.utils.random.seed(seed)

    inj_prior = bilby.gw.prior.BBHPriorDict()
    injection_params = inj_prior.sample()
    injection_params['geocent_time'] = 2.0

    injections_h5 = os.path.join(outdir_, "injection_parameters.h5")
    os.makedirs(os.path.dirname(injections_h5), exist_ok=True)
    with h5py.File(injections_h5, "a") as hf:
        group_name = f"seed_{seed}"
        if group_name in hf:
            del hf[group_name]
        grp = hf.create_group(group_name)
        for key, value in injection_params.items():
            try:
                grp.attrs[key] = float(value)
            except (TypeError, ValueError):
                grp.attrs[key] = str(value)

    analysis_prior = bilby.gw.prior.BBHPriorDict()
    analysis_prior["geocent_time"] = bilby.core.prior.Uniform(
        2 - 0.1, 2 + 0.1, name="geocent_time"
    )
    analysis_prior.validate_prior(duration, minimum_frequency_local)

    # print out the injection parameters, and analysis priors
    print("Injection parameters:")
    for k, v in injection_params.items():
        print(f"  {k}: {v}")
    print("\nAnalysis priors:")
    for k, v in analysis_prior.items():
        print(f"  {k}: {v}")

    start_time = injection_params["geocent_time"] - 2

    noise_ifos = prepare_interferometers(det_names, sampling_frequency_local,
                                         duration=duration * 33, start_time=start_time)
    cache_dir = _ensure_cache_dir()

    for ifo in noise_ifos:
        noise_strain = ifo.strain_data.time_domain_strain
        cache_path = os.path.join(cache_dir, f"{label}_{ifo.name}_noise.npz")
        np.savez_compressed(
            cache_path,
            strain=noise_strain,
            dt=1/sampling_frequency_local
        )

        off_samples = int(duration * 32 * sampling_frequency_local)
        off_source_data = noise_strain[:off_samples]
        on_source_data = noise_strain[off_samples:]

        off_source = TimeSeries(
            off_source_data,
            dt=1/sampling_frequency_local,
            epoch=0.0,
        )
        on_source = TimeSeries(
            on_source_data,
            dt=1/sampling_frequency_local,
            epoch=0.0
        )
        
    # Compute Welch PSD + SGVB PSD
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
        # Compute Welch PSD + SGVB PSD
        freqs_welch, welch_psd = estimate_welch_psd(off_source, sampling_frequency_local)
        freqs_sgvb, sgvb_psd = estimate_sgvb_psd(off_source, sampling_frequency_local)

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
    freqs_true = noise_ifos[0].power_spectral_density.frequency_array
    true_psd = noise_ifos[0].power_spectral_density.psd_array
    mask = (freqs_true >= minimum_frequency_local) & (freqs_true <= maximum_frequency_local)
    freqs_true = freqs_true[mask]
    true_psd = true_psd[mask]
    
    N = len(on_source_data)                   
    times = np.arange(N) / sampling_frequency_local         
    f, on_source_f = get_fd_data(on_source_data, times = times, det='H1',
                              roll_off = 0.4, fmin = minimum_frequency, fmax = maximum_frequency)

    fig = plt.figure(figsize=(7, 5))
    plt.loglog(f, np.abs(on_source_f)**2, alpha=0.3, label="Data", color = "lightgray")
    plt.loglog(freqs_welch, welch_psd, alpha=0.7, label="Welch PSD", color = "green")
    plt.loglog(freqs_sgvb, sgvb_psd, alpha=1, label="SGVB PSD", color = "red")
    plt.loglog(freqs_true, true_psd, alpha=0.5, label="Original PSD", color = "blue")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [strain²/Hz]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    outpath = f"{outdir}/SGVB_Welch_PSDs.png"
    fig.savefig(outpath, dpi=200)

    # input simulated on source noise into InterferometerList and inject the signal
    ifos_analysis = prepare_analysis_ifos(det_names, on_source)
    ifos_analysis.inject_signal(waveform_generator=waveform_generator, parameters=injection_params)

    # collect Original, Welch and SGVB PSDs
    ifos_welch = copy.deepcopy(ifos_analysis)
    ifos_sgvb = copy.deepcopy(ifos_analysis)
    
    welch_psd_object = PowerSpectralDensity.from_amplitude_spectral_density_array(
        freqs_welch, np.sqrt(welch_psd)
    )
    sgvb_psd_object = PowerSpectralDensity.from_amplitude_spectral_density_array(
        freqs_sgvb, np.sqrt(sgvb_psd)
    )
    
    for i in range(len(ifos_analysis)):
        ifos_welch[i].power_spectral_density = welch_psd_object
        ifos_sgvb[i].power_spectral_density = sgvb_psd_object
    
    ifos_for_analysis = dict(
        welch=ifos_welch,
        sgvb=ifos_sgvb,
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
                jitter_time=True,
                reference_frame="sky",
                time_reference="geocent",
            )

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
        )
    
    
    return results, meta


if __name__ == '__main__':
    # Run with the default parameters defined at module scope
    args = sys.argv[1:]
    seed = 0
    if len(args) > 0:
        seed = int(args[0])
        
    run_pe_study(seed=seed)










