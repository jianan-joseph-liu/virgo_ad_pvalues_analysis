"""
Runs a parameter estimation study comparing SGVB and Welch PSD estimates.
"""
import bilby
import corner
from sgvb_univar.psd_estimator import PSDEstimator
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
import os, re, glob


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

# DONT ANALYSE THESE FOR NOW -- TO SPEED UP
fixed_analysis_params = [
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "psi",
    "ra",
    "dec",
    "geocent_time",
    "phase",
    # to make even faster
    'theta_jn',
    'luminosity_distance'
]


'''
# get the files that corresponds to the GPS time
'''
data_dir = "/datasets/LIGO/public/gwosc.osgstorage.org/gwdata/O3b/strain.4k/hdf.v1/H1/"

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
def get_noise_segment_from_seed(segment_file, data_dir, seed):
    segments = np.loadtxt(segment_file, dtype=np.int64, skiprows=1)
    t0, t1 = segments[seed]
    pieces = load_local_ts(data_dir, t0, t1)
    y = np.concatenate([p.value for p in pieces])
    return y, t0, t1


def prepare_interferometers(det_names, sampling_frequency, duration, seed):
    """Create an InterferometerList and set strain data from PSDs.

    Returns the InterferometerList.
    """
    ifos = bilby.gw.detector.InterferometerList(det_names)
    
    segment, t0, t1 = get_noise_segment_from_seed(segment_file, data_dir, seed)
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
    return ifos, on_source_data, off_source_data


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
    psd_est.run(lr=0.008)
    freqs = psd_est.freq
    psd = psd_est.pointwise_ci[1]
    psd = psd*2 / Ew**2
    return freqs, np.real(psd[:,0,0])


def run_pe_study(
        det_names=("H1",),
        sampling_frequency_local=sampling_frequency,
        minimum_frequency_local=minimum_frequency,
        maximum_frequency_local=maximum_frequency,
        outdir="outdir_real_noise_pe_study",
        sgvb_settings=None,
        seed=0,
):
    bilby.core.utils.random.seed(seed)
    label = f"seed_{seed}"
    outdir = f"outdir_real_noise_pe_study/seed_{seed}"

    print(">>>> Running PE study with seed =", seed, " <<<<")
    ## SETUP INJECTION + PRIORS FOR ANALYSIS

    inj_prior = bilby.gw.prior.BBHPriorDict()
    injection_params = inj_prior.sample()
    injection_params['geocent_time'] = 2.0
    # delta functions for all params not in params_to_sample
    analysis_prior = bilby.gw.prior.BBHPriorDict()
    for key in fixed_analysis_params:
        # Assign a delta-prior by setting the prior to the injected value
        analysis_prior[key] = injection_params[key]
    analysis_prior.validate_prior(duration, minimum_frequency_local)

    # print out the injection parameters, and analysis priors
    print("Injection parameters:")
    for k, v in injection_params.items():
        print(f"  {k}: {v}")
    print("\nAnalysis priors:")
    for k, v in analysis_prior.items():
        print(f"  {k}: {v}")

    # Setup logger and seed specifically for this run
    bilby.core.utils.setup_logger(outdir=outdir, label=label)
    bilby.core.utils.random.seed(seed)

    ifos, on_source_data, off_source_data = prepare_interferometers(det_names, 
                                                                    sampling_frequency_local, 
                                                                    duration=duration, 
                                                                    seed=seed)
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_params)

    psd_estimates = {}

    # Extract noise-only segment (all data after injection )
    for ifo in ifos:
        psd_data = TimeSeries(off_source_data, dt = 1.0/sampling_frequency)

        # Compute Welch PSD + SGVB PSD
        freqs_welch, welch_psd = estimate_welch_psd(psd_data, sampling_frequency_local)
        freqs_sgvb, sgvb_psd = estimate_sgvb_psd(psd_data, sampling_frequency_local)

        # plot the two PSDs for comparison along with on-source data
        psd_estimates['welch'] = (freqs_welch, welch_psd)
        psd_estimates['sgvb'] = (freqs_sgvb, sgvb_psd)
        
        
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
    plt.loglog(f, np.abs(on_source_f)**2, alpha=0.3, label="Data", color = "lightgray")
    plt.loglog(freqs_welch, welch_psd, alpha=0.7, label="Welch PSD", color = "green")
    plt.loglog(freqs_sgvb, sgvb_psd, alpha=1, label="SGVB PSD", color = "red")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [strain²/Hz]")
    plt.legend()
    plt.tight_layout()
    plt.show()
    outpath = f"{outdir}/SGVB_Welch_PSDs.png"
    fig.savefig(outpath, dpi=200)
    
     
    # collect Original, Welch and SGVB PSDs
    ifos_orig = copy.deepcopy(ifos)
    ifos_welch = copy.deepcopy(ifos)
    ifos_sgvb = copy.deepcopy(ifos)
    
    welch_psd_object = PowerSpectralDensity.from_amplitude_spectral_density_array(
        freqs_welch, np.sqrt(welch_psd)
    )
    sgvb_psd_object = PowerSpectralDensity.from_amplitude_spectral_density_array(
        freqs_sgvb, np.sqrt(sgvb_psd)
    )
    
    for i in range(len(ifos_orig)):
        ifos_welch[i].power_spectral_density = welch_psd_object
        ifos_sgvb[i].power_spectral_density = sgvb_psd_object
    
    ifos_for_analysis = dict(
        welch=ifos_welch,
        sgvb=ifos_sgvb,
        original=ifos_orig,
    )


    # Now we do the analysis twice, once with each PSD
    results = {}
    for name, analysis_ifos in ifos_for_analysis.items():
        likelihood = bilby.gw.GravitationalWaveTransient(interferometers=analysis_ifos, waveform_generator=waveform_generator)
        print("Running analysis with", name, "PSD")
        print(likelihood.interferometers[0].power_spectral_density)
        run_label = f"{label}_{name}"
        npool = min(mp.cpu_count(), int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
        print("npool = ", npool)
        res = bilby.run_sampler(
            likelihood=likelihood,
            priors=analysis_prior,
            sampler="dynesty",
            npoints=2000,
            dlogz=0.01,
            checkpoint_interval=1000,
            npool=npool,
            queue_size=npool,
            injection_parameters=injection_params,
            outdir=outdir,
            label=run_label,
            resume=False,
        )
        results[name] = res

    # compute Bayes factor SGVB vs Welch
    logZ_welch = results["welch"].log_evidence
    logZ_sgvb = results["sgvb"].log_evidence
    logZ_original = results["original"].log_evidence
    logBF = logZ_sgvb - logZ_welch
    BF = np.exp(logBF)

    print(f"logZ (welch) = {logZ_welch:.3f}")
    print(f"logZ (sgvb)  = {logZ_sgvb:.3f}")
    print(f"logZ (original)  = {logZ_original:.3f}")
    print(f"logBF (sgvb - welch) = {logBF:.3f}, BF = {BF:.3e}")

    print("Welch posterior rows:", len(results["welch"].posterior))
    print("SGVB  posterior rows:", len(results["sgvb"].posterior))

    # Produce an overlaid corner plot of the sampled posteriors
    post_w = results["welch"].posterior
    post_s = results["sgvb"].posterior
    param_names = list(results["welch"].search_parameter_keys)

    samples_w = post_w[param_names].to_numpy()
    samples_s = post_s[param_names].to_numpy()

    fig = corner.corner(samples_w, labels=param_names, color='C0', show_titles=True,
                        title_fmt='.2f', plot_datapoints=False)
    corner.corner(samples_s, labels=param_names, fig=fig, color='C1', plot_datapoints=False,
                  plot_contours=True, fill_contours=False)
    fig.set_size_inches(10, 10) 
    handles = [Line2D([0],[0], color="C0", lw=2),
               Line2D([0],[0], color="C1", lw=2)]
    labels  = ["Welch", "SGVB"]
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle(f"Overlaid posteriors: SGVB (C1) vs Welch (C0)\nlogBF={logBF:.2f}")
    fig.tight_layout()
    outpath = f"{outdir}/{label}_overlaid_corner.png"
    fig.savefig(outpath, dpi=200)
    print(f"Saved overlaid corner to {outpath}")

    # Make a corner plot for the last result as in the original example.
    outpath2 = f"{outdir}/{label}_sgvb_corner.png"
    results['sgvb'].plot_corner(parameters=param_names,
                                priors=False,
                                save=True,
                                filename=outpath2,
                                dpi=200)

    # --- Save evidences (compact numeric file) ---
    # write: logZ_welch, logZ_sgvb, logZ_original, logBF, BF as one row
    os.makedirs(outdir, exist_ok=True)
    ev = np.array([logZ_welch, logZ_sgvb, logZ_original, logBF, BF], dtype=float)
    header = "logZ_welch logZ_sgvb logZ_original logBF_sgvb_vs_welch BF_sgvb_vs_welch"
    np.savetxt(os.path.join(outdir, f"{label}_evidences.txt"), ev.reshape(1, -1), header=header, fmt="%.6e")

    meta = dict(
        welch_freq=freqs_welch,
        welch_psd=welch_psd,
        sgvb_freq=freqs_sgvb,
        sgvb_median=sgvb_psd,
    )
    
    
    return results, meta


if __name__ == '__main__':
    # Run with the default parameters defined at module scope
    args = sys.argv[1:]
    seed = 0
    if len(args) > 0:
        seed = int(args[0])
        
    run_pe_study(seed=seed)

