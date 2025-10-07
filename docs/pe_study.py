#!/usr/bin/env python
"""
Tutorial to demonstrate running parameter estimation on a reduced parameter
space for an injected signal.

This example estimates the masses using a uniform prior in both component masses
and distance using a uniform in comoving volume prior on luminosity distance
between luminosity distances of 100Mpc and 5Gpc, the cosmology is Planck15.
"""

import bilby
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
try:
    import corner
except Exception:
    corner = None

# Import SGVB PSD estimator from the package
from sgvb_univar.psd_estimator import PSDEstimator

SEED = 0 # change this via CLI and repeat this 100 times

# Set the duration and sampling frequency of the data segment that we're
# going to inject the signal into
duration = 4.0
sampling_frequency = 2048.0
minimum_frequency = 20

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "fast_tutorial"
bilby.core.utils.setup_logger(outdir=outdir, label=label)

# Set up a random seed for result reproducibility.  This is optional!
bilby.core.utils.random.seed(SEED)

# We are going to inject a binary black hole waveform.  We first establish a
# dictionary of parameters that includes all of the different waveform
# parameters, including masses of the two black holes (mass_1, mass_2),
# spins of both black holes (a, tilt, phi), etc.
injection_parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=2000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

# Fixed arguments passed into the source model
waveform_arguments = dict(
    waveform_approximant="IMRPhenomPv2",
    reference_frequency=50.0,
    minimum_frequency=minimum_frequency,
)

# Create the waveform_generator using a LAL BinaryBlackHole source function
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# Set up interferometers.  In this case we'll use one interferometer (H1)
ifos = bilby.gw.detector.InterferometerList(["H1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - 2,
)

# compute welch_psd here
welch_psd = None
# compute SGVB psd here
# Prepare data for PSDEstimator: shape (n_samples, p) where p=1 for single interferometer
x = time_series.reshape(-1, 1)
# Use lightweight SGVB settings for a quick run by default; user can increase iterations
psd_est = PSDEstimator(
    x,
    N_theta=30,
    nchunks=1,
    ntrain_map=200,       # small for a smoke test; increase for better estimates
    N_samples=100,        # surrogate samples
    fs=sampling_frequency,
    max_hyperparm_eval=5, # speed up hyperopt
    seed=88170235,
)

# run estimator (this will perform hyperparameter search + MAP training)
psd_all, pointwise_ci = psd_est.run()
# pointwise_ci has shape (3, nfreq, p, p) where median is pointwise_ci[1]
sgvb_median = np.real(pointwise_ci[1, :, 0, 0])
sgvb_freq = psd_est.freq
# map to ifo frequency grid
sgvb_psd_array = map_psd_to_ifo(sgvb_freq, sgvb_median, ifo, fill_with_existing=True)

# inject signal after we've computed PSD estimates from the noise-only time series
ifos.inject_signal(
    waveform_generator=waveform_generator, parameters=injection_parameters
)

# Set up a PriorDict, which inherits from dict.
# By default we will sample all terms in the signal models.  However, this will
# take a long time for the calculation, so for this example we will set almost
# all of the priors to be equall to their injected values.  This implies the
# prior is a delta function at the true, injected value.  In reality, the
# sampler implementation is smart enough to not sample any parameter that has
# a delta-function prior.
# The above list does *not* include mass_1, mass_2, theta_jn and luminosity
# distance, which means those are the parameters that will be included in the
# sampler.  If we do nothing, then the default priors get used.
priors = bilby.gw.prior.BBHPriorDict()
for key in [
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
]:
    priors[key] = injection_parameters[key]

# Perform a check that the prior does not extend to a parameter space longer than the data
priors.validate_prior(duration, minimum_frequency)

# run sampler twice: once with Welch PSD, once with SGVB PSD
results = {}
for name, psd_array in [('welch', welch_psd_array), ('sgvb', sgvb_psd_array)]:
    # assign PSD array to all interferometers
    for ifo in ifos:
        # set the array used by bilby internals
        ifo.power_spectral_density_array = np.asarray(psd_array)

    likelihood = bilby.gw.GravitationalWaveTransient(
        interferometers=ifos, waveform_generator=waveform_generator
    )

    # Run sampler. Use a modest npoints by default; increase for production
    run_label = f"{label}_{name}"
    res = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        npoints=50,  # <--- change to e.g. 1000 for full runs
        injection_parameters=injection_parameters,
        outdir=outdir,
        label=run_label,
        resume=False,
    )
    results[name] = res

# compute Bayes factor SGVB vs Welch
logZ_welch = results['welch'].log_evidence
logZ_sgvb = results['sgvb'].log_evidence
logBF = logZ_sgvb - logZ_welch
BF = np.exp(logBF)

print(f"logZ (welch) = {logZ_welch:.3f}")
print(f"logZ (sgvb)  = {logZ_sgvb:.3f}")
print(f"logBF (sgvb - welch) = {logBF:.3f}, BF = {BF:.3e}")

# Produce an overlaid corner plot of the sampled posteriors
# Extract posterior samples as arrays and parameter names
post_w = results['welch'].posterior
post_s = results['sgvb'].posterior

# Choose a set of parameters to plot: use sampler parameters present in the posterior
try:
    param_names = [c for c in post_w.columns if c not in ['log_likelihood', 'weight']]
except Exception:
    # fall back to sample column names
    param_names = [c for c in post_w.keys() if c not in ['log_likelihood', 'weight']]

if corner is None:
    print("corner package not available; saving individual corner plots instead.")
    results['welch'].plot_corner(save=True, outdir=outdir, label=f"{label}_welch_corner")
    results['sgvb'].plot_corner(save=True, outdir=outdir, label=f"{label}_sgvb_corner")
else:
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
