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

duration = 4.0
sampling_frequency = 1024.0
minimum_frequency = 20
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
]


def prepare_interferometers(det_names, sampling_frequency, duration, start_time):
    """Create an InterferometerList and set strain data from PSDs.

    Returns the InterferometerList.
    """
    ifos = bilby.gw.detector.InterferometerList(det_names)
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )
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
        maximum_frequency: float | None = None,
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


def estimate_sgvb_psd(time_series, sampling_frequency, duration=4,
                      minimum_frequency = 20.0, maximum_frequency = None,
                      N_theta=6000, nchunks=32, ntrain_map=10000,
                      N_samples=500, degree_fluctuate=8000, seed=None,
                      tukey_roll_off = 0.4):
    
    N = duration * sampling_frequency        
    tukey_alpha = 2 * tukey_roll_off / duration
    w = tukey(N, tukey_alpha)
    Ew = np.sqrt(np.mean(w**2))
    
    if maximum_frequency is None:
        maximum_frequency = sampling_frequency//2
        
    frange = [minimum_frequency, maximum_frequency]    
    x = np.asarray(time_series).reshape(-1, 1)
    psd_est = PSDEstimator(
        x=x,
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
    return freqs, psd


def run_pe_study(
        det_names=("H1",),
        sampling_frequency_local=sampling_frequency,
        minimum_frequency_local=minimum_frequency,
        outdir="outdir_pe_study",
        sgvb_settings=None,
        seed=0,
):
    bilby.core.utils.random.seed(seed)
    label = f"seed_{seed}"

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

    start_time = injection_params["geocent_time"] - 2
    signal_end_time = start_time + duration
    psd_start_time = start_time + duration
    psd_end_time = psd_start_time + duration * 32
    ifos = prepare_interferometers(det_names, sampling_frequency_local, duration=duration, start_time=start_time)
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_params)
    
    noise_ifos = prepare_interferometers(det_names, sampling_frequency_local,
                                         duration=duration * 32, start_time=psd_start_time)

    psd_estimates = {}

    # Extract noise-only segment (all data after injection )
    for ifo in ifos:
        on_source_data = np.asarray(ifo.strain_data.time_domain_strain)#.time_slice(start_time, signal_end_time)
        psd_data = TimeSeries(np.asarray(noise_ifos[0].strain_data.time_domain_strain), 
                              dt = 1.0/sampling_frequency)#.time_slice(signal_end_time, psd_end_time)

        # Compute Welch PSD + SGVB PSD
        freqs_welch, welch_psd = estimate_welch_psd(psd_data, sampling_frequency_local)
        freqs_sgvb, sgvb_psd = estimate_sgvb_psd(psd_data, sampling_frequency_local)

        # TODO: plot the two PSDs for comparison along with on-source data

        psd_estimates['welch'] = (freqs_welch, welch_psd)
        psd_estimates['sgvb'] = (freqs_sgvb, sgvb_psd)
        
        
    # plot the two PSDs for comparison along with on-source data
    freqs_welch, welch_psd = psd_estimates['welch']
    freqs_sgvb,  sgvb_psd  = psd_estimates['sgvb']
    
    N = len(on_source_data)                   
    times = np.arange(N) / sampling_frequency_local         
    f, on_source_f = get_fd_data(on_source_data, times = times, det='H1',
                              roll_off = 0.4, fmin = minimum_frequency, fmax = None)
    
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
    

    # edit IFO to only have the first 4 secnds of data (with signal) -- crop the rest out
    for ifo in ifos:
        ifo.strain_data = ifo.strain_data.time_slice(start_time, signal_end_time)

    # Now we do the analysis twice, once with each PSD
    results = {}
    for psd_name, (freqs, psd_array) in psd_estimates.items():
        for ifo in ifos:
            # TODO: check that the PSD freq bins match the ifo freq bins
            ifo.power_spectral_density_array = np.asarray(psd_array)
        likelihood = bilby.gw.GravitationalWaveTransient(interferometers=ifos, waveform_generator=waveform_generator)
        run_label = f"{label}_{psd_name}"
        res = bilby.run_sampler(
            likelihood=likelihood,
            priors=analysis_prior,
            sampler="dynesty",
            npoints=50,
            injection_parameters=injection_params,
            outdir=outdir,
            label=run_label,
            resume=False,
        )
        results[psd_name] = res

    # compute Bayes factor SGVB vs Welch
    logZ_welch = results["welch"].log_evidence
    logZ_sgvb = results["sgvb"].log_evidence
    logBF = logZ_sgvb - logZ_welch
    BF = np.exp(logBF)

    print(f"logZ (welch) = {logZ_welch:.3f}")
    print(f"logZ (sgvb)  = {logZ_sgvb:.3f}")
    print(f"logBF (sgvb - welch) = {logBF:.3f}, BF = {BF:.3e}")

    # Produce an overlaid corner plot of the sampled posteriors
    post_w = results["welch"].posterior
    post_s = results["sgvb"].posterior
    param_names = [c for c in post_w.columns if c not in ["log_likelihood", "weight"]]

    samples_w = post_w[param_names].to_numpy()
    samples_s = post_s[param_names].to_numpy()

    fig = corner.corner(samples_w, labels=param_names, color='C0', show_titles=True,
                        title_fmt='.2f', plot_datapoints=False)
    corner.corner(samples_s, labels=param_names, fig=fig, color='C1', plot_datapoints=False,
                  plot_contours=True, fill_contours=False)
    fig.suptitle(f"Overlaid posteriors: SGVB (C1) vs Welch (C0)\nlogBF={logBF:.2f}")
    outpath = f"{outdir}/{label}_overlaid_corner.png"
    fig.savefig(outpath, dpi=200)
    print(f"Saved overlaid corner to {outpath}")

    # Make a corner plot for the last result as in the original example.
    results['sgvb'].plot_corner()

    meta = dict(
        welch_freq=freqs_welch,
        welch_psd=welch_psd,
        sgvb_freq=freqs_sgvb,
        sgvb_median=sgvb_psd,
    )
    return results, meta


if __name__ == '__main__':
    # Run with the default parameters defined at module scope
    run_pe_study(seed=0)
